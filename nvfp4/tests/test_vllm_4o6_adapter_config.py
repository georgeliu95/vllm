# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from enum import Enum
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Final

import pytest
import torch

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
ADAPTER_ROOT: Final = REPO_ROOT / "runtime"
sys.path.insert(0, str(ADAPTER_ROOT))

from vllm_adapter.config import AdapterConfigError, load_runtime_config
from vllm_adapter.install import _configure_flashinfer_quantization
from vllm_adapter.linear import (
    make_dynamic_linear_method,
    make_svdq_linear_method,
    svdq_us_needs_output_sharding,
)
from vllm_adapter.moe import (
    _set_moe_scale_loading_metadata,
    make_batched_svdq_moe_kernel,
    make_dynamic_moe_method,
    make_svdq_moe_method,
)
from vllm_adapter.moe_kernel import (
    _expert_ranges,
    _grouped_gemm_mask,
    _initialize_grouped_gemm_dummy_rows,
    grouped_lowrank_masked,
)
from vllm_adapter import quantization


def test_dense_activation_global_scale_is_runtime_dynamic() -> None:
    values = torch.tensor([[-2.0, 1.0], [0.5, -1.5]])

    global_scale, dequant_scale = quantization.dynamic_global_scale(
        values, quant_range=8.0
    )

    torch.testing.assert_close(global_scale, torch.tensor(4.0))
    torch.testing.assert_close(dequant_scale, torch.tensor(0.25))


def test_grouped_activation_global_scale_is_shared_and_ignores_invalid_rows() -> None:
    values = torch.tensor(
        [
            [[-2.0, 1.0], [0.5, -1.5], [100.0, 100.0]],
            [[4.0, -1.0], [200.0, 200.0], [300.0, 300.0]],
        ]
    )
    valid_rows = torch.tensor([2, 1], dtype=torch.int32)

    global_scale, dequant_scale = quantization.dynamic_grouped_global_scale(
        values, valid_rows, quant_range=8.0
    )

    torch.testing.assert_close(global_scale, torch.tensor([2.0, 2.0]))
    torch.testing.assert_close(dequant_scale, torch.tensor([0.5, 0.5]))


def test_grouped_activation_global_scale_is_safe_for_inactive_experts() -> None:
    values = torch.full((2, 3, 4), float("nan"))
    valid_rows = torch.zeros(2, dtype=torch.int32)

    global_scale, dequant_scale = quantization.dynamic_grouped_global_scale(
        values, valid_rows, quant_range=8.0
    )

    torch.testing.assert_close(global_scale, torch.full((2,), 8.0))
    torch.testing.assert_close(dequant_scale, torch.full((2,), 0.125))


def test_grouped_activation_global_scale_chunks_temporary_storage(monkeypatch) -> None:
    monkeypatch.setattr(quantization, "GROUPED_SCALE_TEMP_BYTES", 16)
    values = torch.tensor(
        [
            [[1.0, -2.0], [3.0, -4.0], [100.0, 100.0], [200.0, 200.0]],
            [[5.0, -6.0], [7.0, -8.0], [9.0, -10.0], [300.0, 300.0]],
        ],
        dtype=torch.bfloat16,
    )
    valid_rows = torch.tensor([2, 3], dtype=torch.int32)

    global_scale, dequant_scale = quantization.dynamic_grouped_global_scale(
        values, valid_rows, quant_range=20.0
    )

    torch.testing.assert_close(global_scale, torch.full((2,), 2.0))
    torch.testing.assert_close(dequant_scale, torch.full((2,), 0.5))


def test_dynamic_linear_updates_quantizer_scale_and_gemm_alpha(monkeypatch) -> None:
    linear_module = ModuleType("vllm.model_executor.layers.linear")
    linear_module.register_weight_loader_v2_supported_method = lambda cls: cls
    monkeypatch.setitem(sys.modules, linear_module.__name__, linear_module)
    observed: dict[str, torch.Tensor] = {}

    class BaseLinearMethod:
        def apply(self, layer, x, bias=None):
            del bias
            observed["global_scale"] = layer.input_global_scale_inv.clone()
            observed["alpha"] = layer.alpha.clone()
            return x

    method_class = make_dynamic_linear_method(BaseLinearMethod, quant_range=8.0)
    method = method_class()
    layer = SimpleNamespace(
        input_global_scale_inv=torch.tensor(99.0),
        weight_global_scale=torch.tensor(0.125),
        alpha=torch.tensor(99.0),
    )
    values = torch.tensor([[-2.0, 1.0]])

    method.apply(layer, values)

    torch.testing.assert_close(observed["global_scale"], torch.tensor(4.0))
    torch.testing.assert_close(observed["alpha"], torch.tensor(0.03125))


def test_moe_scale_parameters_keep_loader_layout_metadata(monkeypatch) -> None:
    class ScaleLayout(Enum):
        TENSOR = "tensor"
        CHANNEL = "channel"
        GROUP = "group"
        BLOCK = "block"

    routed_experts = ModuleType("vllm.model_executor.layers.fused_moe.routed_experts")
    routed_experts.FusedMoeWeightScaleSupported = ScaleLayout
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.routed_experts",
        routed_experts,
    )
    layer = SimpleNamespace(
        **{
            name: torch.nn.Parameter(torch.empty(1))
            for name in (
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
                "w13_input_scale",
                "w2_input_scale",
            )
        }
    )

    _set_moe_scale_loading_metadata(layer)

    assert layer.w13_weight_scale.quant_method == "block"
    assert layer.w2_weight_scale.quant_method == "block"
    assert layer.w13_weight_scale_2.quant_method == "tensor"
    assert layer.w2_weight_scale_2.quant_method == "tensor"
    assert layer.w13_input_scale.quant_method == "tensor"
    assert layer.w2_input_scale.quant_method == "tensor"


def test_dynamic_moe_captures_weight_scale_before_backend_fuses_input_scale() -> None:
    class Backend(Enum):
        FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
        FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"

    class BaseExperts:
        def process_weights_after_loading(self, layer) -> None:
            layer.w13_weight_scale_2.mul_(layer.w13_input_scale)
            layer.w2_weight_scale_2.mul_(layer.w2_input_scale)

    class BaseMoeMethod:
        def __init__(self, quant_config, moe_config) -> None:
            self.nvfp4_backend = Backend.FLASHINFER_CUTEDSL

    method_class = make_dynamic_moe_method(
        BaseMoeMethod, BaseExperts, activation_quant_range=2688.0
    )
    experts_class = method_class(None, None).experts_cls
    experts = experts_class()
    layer = SimpleNamespace(
        w13_weight_scale_2=torch.tensor([0.125, 0.25]),
        w2_weight_scale_2=torch.tensor([0.0625, 0.5]),
        w13_input_scale=torch.tensor([2.0, 4.0]),
        w2_input_scale=torch.tensor([8.0, 16.0]),
    )

    experts.process_weights_after_loading(layer)

    torch.testing.assert_close(
        experts.w1_weight_global_scale, torch.tensor([0.125, 0.25])
    )
    torch.testing.assert_close(
        experts.w2_weight_global_scale, torch.tensor([0.0625, 0.5])
    )
    torch.testing.assert_close(layer.w13_weight_scale_2, torch.tensor([0.25, 1.0]))
    torch.testing.assert_close(layer.w2_weight_scale_2, torch.tensor([0.5, 8.0]))


def _write_metadata(checkpoint: Path, *, rank: int = 64) -> None:
    payload = {
        "quantization": {"quant_algo": "NVFP4", "group_size": 16},
        "svdquant": {
            "format": "offline-lowrank-residual-v1",
            "rank": rank,
        },
    }
    (checkpoint / "hf_quant_config.json").write_text(json.dumps(payload))


def test_svdq_contract_accepts_matching_rank(tmp_path, monkeypatch) -> None:
    _write_metadata(tmp_path)
    monkeypatch.setenv("VLLM_NVFP4_4O6_MODE", "4o6-svdq")
    monkeypatch.setenv("VLLM_NVFP4_SVDQ_RANK", "64")

    config = load_runtime_config(tmp_path)

    assert config.svdq_enabled
    assert config.rank == 64


def test_standard_mode_uses_runtime_dynamic_standard_nvfp4_range(
    tmp_path, monkeypatch
) -> None:
    _write_metadata(tmp_path)
    monkeypatch.setenv("VLLM_NVFP4_4O6_MODE", "standard")

    config = load_runtime_config(tmp_path)

    assert not config.four_over_six_enabled
    assert config.activation_quant_range == quantization.STANDARD_NVFP4_QUANT_RANGE


def test_4o6_mode_reserves_e4m3_headroom_for_candidate_four(
    tmp_path, monkeypatch
) -> None:
    # Given 4o6 selects between a 4-level candidate and standard NVFP4 candidate 6.
    _write_metadata(tmp_path)
    monkeypatch.setenv("VLLM_NVFP4_4O6_MODE", "4o6")

    # When the shared Dense/MoE runtime contract is resolved.
    config = load_runtime_config(tmp_path)

    # Then 256 * 1.5 remains representable when candidate 4 expands its scale.
    assert config.activation_quant_range == quantization.FOUR_OVER_SIX_QUANT_RANGE
    assert config.activation_quant_range == 256.0 * 6.0


def test_4o6_flashinfer_config_enables_e4m3_headroom(monkeypatch) -> None:
    # Given a process environment without FourOverSix's E4M3 headroom flag.
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", raising=False)
    config = SimpleNamespace(four_over_six_enabled=True)

    # When the canonical 4o6 runtime config is installed.
    _configure_flashinfer_quantization(config)

    # Then FlashInfer caps the base block scale at 256 before 1.5x expansion.
    assert os.environ["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] == "1"


def test_svdq_contract_rejects_rank_mismatch(tmp_path, monkeypatch) -> None:
    _write_metadata(tmp_path, rank=32)
    monkeypatch.setenv("VLLM_NVFP4_4O6_MODE", "4o6-svdq")
    monkeypatch.setenv("VLLM_NVFP4_SVDQ_RANK", "64")

    with pytest.raises(AdapterConfigError, match="does not match"):
        load_runtime_config(tmp_path)


def test_svdq_linear_method_registers_weight_loader_v2(monkeypatch) -> None:
    registered: list[str] = []
    linear_module = ModuleType("vllm.model_executor.layers.linear")

    def register(method: type) -> type:
        registered.append(method.__name__)
        return method

    linear_module.register_weight_loader_v2_supported_method = register
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor", ModuleType("vllm.model_executor")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers",
        ModuleType("vllm.model_executor.layers"),
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_module)

    class BaseLinearMethod:
        pass

    svdq_method = make_svdq_linear_method(BaseLinearMethod, rank=64)

    assert svdq_method.__name__ in registered


def test_svdq_linear_method_can_load_factors_without_applying_them(
    monkeypatch,
) -> None:
    linear_module = ModuleType("vllm.model_executor.layers.linear")
    linear_module.register_weight_loader_v2_supported_method = lambda method: method
    monkeypatch.setitem(sys.modules, "vllm", ModuleType("vllm"))
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor", ModuleType("vllm.model_executor")
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers",
        ModuleType("vllm.model_executor.layers"),
    )
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_module)

    class BaseLinearMethod:
        def apply(self, layer, x, bias=None):
            del layer, bias
            return x + 1

    method_class = make_svdq_linear_method(
        BaseLinearMethod, rank=64, enable_correction=False
    )
    method = method_class()
    layer = SimpleNamespace(svdquant_vh=None, svdquant_us=None)
    values = torch.zeros(2, 3)

    torch.testing.assert_close(method.apply(layer, values), values + 1)


def test_svdq_us_preserves_fused_projection_shards_at_tp1() -> None:
    assert svdq_us_needs_output_sharding([12288, 12288], 24576, 24576)
    assert svdq_us_needs_output_sharding([4096, 1024, 1024], 6144, 6144)


def test_svdq_moe_forces_batched_experts_after_backend_selection() -> None:
    class Backend(Enum):
        FLASHINFER_CUTEDSL = "FLASHINFER_CUTEDSL"
        FLASHINFER_CUTEDSL_BATCHED = "FLASHINFER_CUTEDSL_BATCHED"

    class SelectedNonBatchedExperts:
        pass

    class BatchedExperts:
        pass

    class BaseMoeMethod:
        def __init__(self, quant_config, moe_config) -> None:
            self.experts_cls = SelectedNonBatchedExperts
            self.nvfp4_backend = Backend.FLASHINFER_CUTEDSL

    method_class = make_svdq_moe_method(BaseMoeMethod, BatchedExperts, rank=64)
    method = method_class(None, None)

    assert issubclass(method.experts_cls, BatchedExperts)
    assert method.nvfp4_backend is Backend.FLASHINFER_CUTEDSL_BATCHED


def test_svdq_moe_builds_batched_prepare_without_deployment_dp(monkeypatch) -> None:
    calls: dict[str, object] = {}

    class Prepare:
        def __init__(self, **kwargs) -> None:
            calls["prepare"] = kwargs
            self.rank = kwargs["rank"]

        def max_num_tokens_per_rank(self) -> int:
            return 2048

        def num_dispatchers(self) -> int:
            return 1

        def prepare(self, **kwargs):
            calls["prepare_call"] = kwargs
            return "prepared"

    class Experts:
        def __init__(self, **kwargs) -> None:
            calls["experts"] = kwargs

    class Kernel:
        def __init__(self, prepare, experts) -> None:
            calls["kernel"] = (prepare, experts)

    batched_module = ModuleType(
        "vllm.model_executor.layers.fused_moe.prepare_finalize.batched"
    )
    batched_module.BatchedPrepareAndFinalize = Prepare
    modular_kernel = ModuleType("vllm.model_executor.layers.fused_moe.modular_kernel")
    modular_kernel.FusedMoEKernel = Kernel
    package = ModuleType("vllm.model_executor.layers.fused_moe")
    package.modular_kernel = modular_kernel
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.prepare_finalize.batched",
        batched_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.fused_moe.modular_kernel",
        modular_kernel,
    )

    moe_config = SimpleNamespace(
        max_num_tokens=2048,
        num_local_experts=128,
        ep_rank=3,
        moe_parallel_config=SimpleNamespace(ep_rank=3),
    )
    kernel = make_batched_svdq_moe_kernel(
        moe_quant_config="quant",
        moe_config=moe_config,
        experts_cls=Experts,
    )

    assert isinstance(kernel, Kernel)
    assert calls["prepare"]["rank"] == 3, (
        "Batched MoE prepare must select this EP rank's global-expert range; "
        f"expected 3, got {calls['prepare']['rank']}."
    )
    assert calls["prepare"] == {
        "max_num_tokens": 2048,
        "num_local_experts": 128,
        "num_dispatchers": 1,
        "rank": 3,
    }
    assert calls["experts"] == {
        "moe_config": moe_config,
        "quant_config": "quant",
        "max_num_tokens": 2048,
        "num_dispatchers": 1,
    }
    prepare, _ = calls["kernel"]
    with pytest.raises(RuntimeError, match="local rank-zero routing"):
        prepare.prepare(
            a1="hidden",
            topk_weights="weights",
            topk_ids="ids",
            num_experts=256,
            expert_map=None,
            apply_router_weight_on_input=False,
            quant_config="nvfp4",
        )


def test_svdq_moe_masked_lowrank_skips_padded_and_empty_expert_rows() -> None:
    values = torch.arange(3 * 9 * 8, dtype=torch.bfloat16).reshape(3, 9, 8)
    us = torch.randn(3, 6, 2, dtype=torch.bfloat16)
    vh = torch.randn(3, 2, 8, dtype=torch.bfloat16)
    masked_m = torch.tensor([9, 0, 2], dtype=torch.int32)

    actual = grouped_lowrank_masked(values, us, vh, masked_m)
    expected = torch.zeros_like(actual)
    expected[0] = (values[0] @ vh[0].T) @ us[0].T
    expected[2, :2] = (values[2, :2] @ vh[2].T) @ us[2].T

    torch.testing.assert_close(actual, expected)


def test_svdq_moe_chunks_experts_below_failing_128_expert_boundary() -> None:
    assert _expert_ranges(64) == ((0, 64),)
    assert _expert_ranges(128) == ((0, 64), (64, 128))
    assert _expert_ranges(129) == ((0, 64), (64, 128), (128, 129))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Uses the fused CUDA dummy-row kernel"
)
def test_grouped_gemm_dummy_rows_are_initialized_before_quantization() -> None:
    values = torch.full((3, 2, 4), 7.0, device="cuda")
    masked_m = torch.tensor([2, 0, 1], dtype=torch.int32, device="cuda")

    grouped_gemm_mask = _initialize_grouped_gemm_dummy_rows(values, masked_m)

    torch.testing.assert_close(
        grouped_gemm_mask, torch.tensor([2, 1, 1], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(values[1, 0], torch.zeros(4, device="cuda"))
    torch.testing.assert_close(values[1, 1], torch.full((4,), 7.0, device="cuda"))
    torch.testing.assert_close(_grouped_gemm_mask(masked_m), grouped_gemm_mask)


@pytest.mark.parametrize("scale_count", [1, 4])
def test_grouped_quantization_preserves_native_mask_and_layout(
    monkeypatch, scale_count
):
    # Given masked experts, including NaN padding, and a native strided output.
    values = torch.full((4, 256, 32), torch.nan, dtype=torch.bfloat16)
    mask = torch.tensor([1, 100, 129, 0], dtype=torch.int32)
    global_scale = torch.ones(scale_count)
    packed = torch.empty((4, 256, 16), dtype=torch.uint8).permute(1, 2, 0)
    scales = torch.empty((4, 2, 32, 4, 4), dtype=torch.uint8).permute(2, 3, 1, 4, 0)
    observed = []

    def native_quantize(actual_values, actual_mask, actual_scale):
        observed.append((actual_values, actual_mask, actual_scale))
        return packed, scales

    native = ModuleType("flashinfer")
    native.scaled_fp4_grouped_quantize = native_quantize
    monkeypatch.setitem(sys.modules, "flashinfer", native)

    # When delegating to the current native grouped quantizer.
    got_packed, got_scales = quantization.quantize_grouped_4o6(
        values, mask, global_scale
    )

    # Then preserve operands and physical output strides without Python repacking.
    assert len(observed) == 1
    assert observed[0][0] is values
    assert observed[0][1] is mask
    assert observed[0][2] is global_scale
    assert got_packed is packed
    assert got_scales is scales
    assert not got_scales.is_contiguous()
