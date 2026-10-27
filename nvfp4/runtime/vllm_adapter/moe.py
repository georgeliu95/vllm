# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM ModelOpt MoE registration and loading for rank-R SVDQ."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
import os
from typing import Any

import torch

from .moe_kernel import run_4o6_svdq_moe
from .moe_capacity import bounded_capacity
from .local_routing import finalize_local, prepare_local


FactorLoader = Callable[..., bool | None]
_MOE_LAYOUT_LOGGED = False
_SM120_FALLBACK_LOGGED = False


def _use_sm120_moe(device: torch.device) -> bool:
    """Token-major SM120 path (sm120_moe.py) unless VLLM_NVFP4_SM120_MOE=0."""
    if os.environ.get("VLLM_NVFP4_SM120_MOE", "1") != "1" or device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device)[0] == 12


def _log_sm120_fallback(x: torch.Tensor, shared_experts: Any, layer: torch.nn.Module) -> None:
    global _SM120_FALLBACK_LOGGED
    if not _SM120_FALLBACK_LOGGED:
        print(f"SM120_MOE_FALLBACK tokens={x.shape[0]} dtype={x.dtype} shared={shared_experts is not None} "
              f"expert_map={layer.expert_map is not None} weighted_input={layer.apply_router_weight_on_input} "
              f"activation={layer.activation}", flush=True)
        _SM120_FALLBACK_LOGGED = True


def _set_moe_scale_loading_metadata(layer: torch.nn.Module) -> None:
    """Attach the scale layouts required by RoutedExperts.weight_loader."""
    from vllm.model_executor.layers.fused_moe.routed_experts import (
        FusedMoeWeightScaleSupported,
    )

    layouts = {
        "w13_weight_scale": FusedMoeWeightScaleSupported.BLOCK.value,
        "w2_weight_scale": FusedMoeWeightScaleSupported.BLOCK.value,
        "w13_weight_scale_2": FusedMoeWeightScaleSupported.TENSOR.value,
        "w2_weight_scale_2": FusedMoeWeightScaleSupported.TENSOR.value,
        "w13_input_scale": FusedMoeWeightScaleSupported.TENSOR.value,
        "w2_input_scale": FusedMoeWeightScaleSupported.TENSOR.value,
    }
    for name, quant_method in layouts.items():
        parameter = getattr(layer, name)
        parameter.quant_method = quant_method


def _layout(name: str, value: torch.Tensor | None) -> str:
    if value is None:
        return f"{name}=None"
    return (
        f"{name}.shape={tuple(value.shape)} "
        f"{name}.stride={value.stride()} {name}.dtype={value.dtype}"
    )


def make_batched_svdq_moe_kernel(
    moe_quant_config: Any,
    moe_config: Any,
    experts_cls: type,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    *,
    backend: Enum | None = None,
    layer: torch.nn.Module | None = None,
) -> Any:
    """Build the local batched-expert kernel independently of deployment DP."""
    del routing_tables, backend, layer
    from vllm.model_executor.layers.fused_moe import modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.prepare_finalize.batched import (
        BatchedPrepareAndFinalize,
    )

    class UnquantizedPrepareConfig:
        quant_dtype = None
        is_quantized = False
        a1_scale = None

    class SvdqBatchedPrepareAndFinalize(BatchedPrepareAndFinalize):
        def prepare(
            self,
            a1: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            num_experts: int,
            expert_map: torch.Tensor | None,
            apply_router_weight_on_input: bool,
            quant_config: Any,
            defer_input_quant: bool = False,
        ) -> Any:
            del quant_config
            if self.rank != 0 or num_experts != self.num_local_experts or defer_input_quant:
                raise RuntimeError("Compact performance adapter requires local rank-zero routing.")
            if apply_router_weight_on_input and topk_ids.shape[1] != 1:
                raise RuntimeError("Input router weighting requires top-k one.")
            prepared, counts = prepare_local(
                a1, topk_weights, topk_ids, num_experts,
                bounded_capacity(a1.shape[0], self.max_num_tokens),
                apply_router_weight_on_input,
            )
            metadata = mk.ExpertTokensMetadata(
                expert_num_tokens=counts, expert_num_tokens_cpu=None,
            )
            return prepared, None, metadata, None, None

        def finalize(
            self,
            output: torch.Tensor,
            fused_expert_output: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            apply_router_weight_on_input: bool,
            weight_and_reduce_impl: mk.TopKWeightAndReduce,
        ) -> None:
            del weight_and_reduce_impl
            finalize_local(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input,
            )

    prepare_finalize = SvdqBatchedPrepareAndFinalize(
        max_num_tokens=moe_config.max_num_tokens,
        num_local_experts=moe_config.num_local_experts,
        num_dispatchers=1,
        rank=moe_config.ep_rank,
    )
    max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
    if max_num_tokens is None:
        raise RuntimeError("Batched SVDQ MoE prepare did not provide a token limit.")
    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        max_num_tokens=max_num_tokens,
        num_dispatchers=prepare_finalize.num_dispatchers(),
    )
    return mk.FusedMoEKernel(prepare_finalize, experts)


def _make_factor_loader(
    layer: torch.nn.Module, local_intermediate: int
) -> FactorLoader:
    original_loader = layer.weight_loader

    def load_factor(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if "svdquant_" not in weight_name:
            return original_loader(
                param,
                loaded_weight,
                weight_name,
                shard_id,
                expert_id,
                return_success,
            )
        local_expert = layer._map_global_expert_id_to_local_expert_id(expert_id)
        if local_expert == -1:
            return False if return_success else None
        target = param.data[local_expert]
        tp_rank = layer.moe_config.tp_rank
        if "w13_svdquant_us" in weight_name:
            local = loaded_weight.narrow(0, tp_rank * local_intermediate, local_intermediate)
            offset = 0 if shard_id == "w1" else local_intermediate
            target[offset : offset + local_intermediate].copy_(local)
        elif "w13_svdquant_vh" in weight_name:
            offset = 0 if shard_id == "w1" else loaded_weight.shape[0]
            target[offset : offset + loaded_weight.shape[0]].copy_(loaded_weight)
        elif "w2_svdquant_us" in weight_name:
            target.copy_(loaded_weight)
        elif "w2_svdquant_vh" in weight_name:
            local = loaded_weight.narrow(1, tp_rank * local_intermediate, local_intermediate)
            target.copy_(local)
        else:
            raise ValueError(f"Unknown SVDQ factor {weight_name}.")
        load_state = getattr(layer, "_svdq_factor_load_state")
        load_state.add((local_expert, weight_name.rsplit(".", 1)[-1], shard_id))
        return True if return_success else None

    return load_factor


def make_dynamic_moe_method(
    base_class: type,
    experts_base: type,
    activation_quant_range: float,
    rank: int | None = None,
    *,
    enable_w13_correction: bool = True,
    enable_w2_correction: bool = True,
) -> type:
    """Build a runtime-dynamic ModelOpt MoE method backed by CuTeDSL."""

    class DynamicExperts(experts_base):
        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            w1_weight_global_scale = layer.w13_weight_scale_2.detach().clone()
            w2_weight_global_scale = layer.w2_weight_scale_2.detach().clone()
            super().process_weights_after_loading(layer)
            self.w1_weight_global_scale = w1_weight_global_scale
            self.w2_weight_global_scale = w2_weight_global_scale
            if rank is not None:
                self.w13_svdquant_us = layer.w13_svdquant_us
                self.w13_svdquant_vh = layer.w13_svdquant_vh
                self.w2_svdquant_us = layer.w2_svdquant_us
                self.w2_svdquant_vh = layer.w2_svdquant_vh
            self.mainloop = None
            self.sm120 = None
            if activation_quant_range == 1536.0 and _use_sm120_moe(layer.w13_weight.device):
                from .sm120_moe import Sm120SvdqMoE
                self.sm120 = Sm120SvdqMoE(
                    layer.w13_weight, self.w1_scale, w1_weight_global_scale,
                    layer.w2_weight, self.w2_scale, w2_weight_global_scale,
                    activation_quant_range,
                    getattr(self, "w13_svdquant_us", None) if enable_w13_correction else None,
                    getattr(self, "w13_svdquant_vh", None) if enable_w13_correction else None,
                    getattr(self, "w2_svdquant_us", None) if enable_w2_correction else None,
                    getattr(self, "w2_svdquant_vh", None) if enable_w2_correction else None,
                )
            if os.environ.get("VLLM_NVFP4_MAINLOOP_FUSION") == "1":
                from .mainloop import MainloopMoE
                if rank not in (None, 64) or not (enable_w13_correction and enable_w2_correction):
                    raise RuntimeError("Mainloop fusion supports the full rank-64 SVDQ recipe.")
                self.mainloop = MainloopMoE(
                    layer.w13_weight, self.w1_scale, w1_weight_global_scale,
                    layer.w2_weight, self.w2_scale, w2_weight_global_scale,
                    activation_quant_range,
                    getattr(self, "w13_svdquant_us", None),
                    getattr(self, "w13_svdquant_vh", None),
                    getattr(self, "w2_svdquant_us", None),
                    getattr(self, "w2_svdquant_vh", None),
                )

        def apply(
            self,
            output: torch.Tensor,
            hidden_states: torch.Tensor,
            w1: torch.Tensor,
            w2: torch.Tensor,
            topk_weights: torch.Tensor,
            topk_ids: torch.Tensor,
            activation: str,
            global_num_experts: int,
            expert_map: torch.Tensor | None,
            a1q_scale: torch.Tensor | None,
            a2_scale: torch.Tensor | None,
            workspace13: torch.Tensor | None,
            workspace2: torch.Tensor | None,
            expert_tokens_meta: Any,
            apply_router_weight_on_input: bool,
        ) -> None:
            global _MOE_LAYOUT_LOGGED
            if (
                os.environ.get("VLLM_NVFP4_DEBUG_MOE_LAYOUT") == "1"
                and not _MOE_LAYOUT_LOGGED
            ):
                print(
                    "SVDQ_MOE_LAYOUT "
                    + " | ".join(
                        (
                            _layout("hidden", hidden_states),
                            _layout("w1", w1),
                            _layout("w1_scale", self.w1_scale),
                            _layout("w2", w2),
                            _layout("w2_scale", self.w2_scale),
                            _layout("workspace13", workspace13),
                            _layout("workspace2", workspace2),
                            _layout("output", output),
                            _layout("w13_us", getattr(self, "w13_svdquant_us", None)),
                            _layout("w13_vh", getattr(self, "w13_svdquant_vh", None)),
                            _layout("w2_us", getattr(self, "w2_svdquant_us", None)),
                            _layout("w2_vh", getattr(self, "w2_svdquant_vh", None)),
                        )
                    ),
                    flush=True,
                )
                _MOE_LAYOUT_LOGGED = True
            if workspace2 is None or expert_tokens_meta is None:
                raise RuntimeError("Dynamic NVFP4 requires batched expert metadata.")
            run_4o6_svdq_moe(
                hidden_states=hidden_states,
                activation_quant_range=activation_quant_range,
                w1=w1,
                w1_blockscale=self.w1_scale,
                w1_weight_global_scale=self.w1_weight_global_scale,
                w2=w2,
                w2_blockscale=self.w2_scale,
                w2_weight_global_scale=self.w2_weight_global_scale,
                masked_m=expert_tokens_meta.expert_num_tokens,
                workspace=workspace2,
                out=output,
                w13_us=(
                    getattr(self, "w13_svdquant_us", None)
                    if enable_w13_correction
                    else None
                ),
                w13_vh=(
                    getattr(self, "w13_svdquant_vh", None)
                    if enable_w13_correction
                    else None
                ),
                w2_us=(
                    getattr(self, "w2_svdquant_us", None)
                    if enable_w2_correction
                    else None
                ),
                w2_vh=(
                    getattr(self, "w2_svdquant_vh", None)
                    if enable_w2_correction
                    else None
                ),
            )

    class DynamicMoeMethod(base_class):
        def __init__(self, quant_config: Any, moe_config: Any) -> None:
            super().__init__(quant_config, moe_config)
            backend_type = type(self.nvfp4_backend)
            self.nvfp4_backend = backend_type["FLASHINFER_CUTEDSL_BATCHED"]
            self.experts_cls = DynamicExperts

        def apply(
            self, layer: torch.nn.Module, x: torch.Tensor,
            topk_weights: torch.Tensor, topk_ids: torch.Tensor,
            shared_experts: torch.nn.Module | None,
            shared_experts_input: torch.Tensor | None,
        ) -> torch.Tensor:
            """Use the qualified token-major fusion before batched dispatch."""
            sm120 = getattr(self.moe_kernel.fused_experts, "sm120", None)
            # Shared experts: the batched prepare/finalize this path replaces is synchronous, so the modular
            # kernel never runs them (it only does in its async-finalize branch); vLLM's runner does.
            if (sm120 is not None and x.shape[0] > 0
                    and not self.moe_kernel.prepare_finalize.supports_async()
                    and layer.expert_map is None
                    and not layer.apply_router_weight_on_input
                    and layer.global_num_experts == sm120.experts
                    and x.dtype == torch.bfloat16
                    and str(getattr(layer.activation, "value", layer.activation)) == "silu"):
                return sm120(x, topk_ids, topk_weights)
            if sm120 is not None:
                _log_sm120_fallback(x, shared_experts, layer)
            mainloop = getattr(self.moe_kernel.fused_experts, "mainloop", None)
            if (mainloop is not None and 0 < x.shape[0] <= 16384
                    and shared_experts is None and layer.expert_map is None
                    and not layer.apply_router_weight_on_input
                    and layer.global_num_experts == mainloop.experts
                    and x.dtype == torch.bfloat16
                    and str(getattr(layer.activation, "value", layer.activation)) == "silu"):
                return mainloop(x.contiguous(), topk_ids.to(torch.int32).contiguous(),
                                topk_weights.float().contiguous())
            return super().apply(layer, x, topk_weights, topk_ids,
                                 shared_experts, shared_experts_input)

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            from vllm.model_executor.layers.quantization import modelopt

            if rank is not None:
                expected = {
                    (expert, factor, shard)
                    for expert in range(layer.moe_config.num_local_experts)
                    for factor, shards in (
                        ("w13_svdquant_us", ("w1", "w3")),
                        ("w13_svdquant_vh", ("w1", "w3")),
                        ("w2_svdquant_us", ("w2",)),
                        ("w2_svdquant_vh", ("w2",)),
                    )
                    for shard in shards
                }
                missing = expected.difference(layer._svdq_factor_load_state)
                if missing:
                    preview = sorted(missing)[:8]
                    raise RuntimeError(
                        "Incomplete MoE SVDQ factor load: "
                        f"missing {len(missing)} entries; first entries: {preview}"
                    )

            original_factory = modelopt.make_nvfp4_moe_kernel
            modelopt.make_nvfp4_moe_kernel = make_batched_svdq_moe_kernel
            try:
                super().process_weights_after_loading(layer)
            finally:
                modelopt.make_nvfp4_moe_kernel = original_factory

        def create_weights(
            self,
            layer: torch.nn.Module,
            num_experts: int,
            hidden_size: int,
            intermediate_size_per_partition: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: object,
        ) -> None:
            super().create_weights(
                layer,
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                params_dtype,
                **extra_weight_attrs,
            )
            _set_moe_scale_loading_metadata(layer)
            if rank is None:
                return
            from vllm.model_executor.utils import set_weight_attrs

            layer._svdq_factor_load_state = set()
            loader = _make_factor_loader(layer, intermediate_size_per_partition)
            shapes = {
                "w13_svdquant_us": (num_experts, 2 * intermediate_size_per_partition, rank),
                "w13_svdquant_vh": (num_experts, 2 * rank, hidden_size),
                "w2_svdquant_us": (num_experts, hidden_size, rank),
                "w2_svdquant_vh": (num_experts, rank, intermediate_size_per_partition),
            }
            for name, shape in shapes.items():
                parameter = torch.nn.Parameter(
                    torch.zeros(shape, dtype=params_dtype), requires_grad=False
                )
                set_weight_attrs(parameter, {"weight_loader": loader})
                layer.register_parameter(name, parameter)

    DynamicMoeMethod.__name__ = (
        "ModelOptNvFp4SvdqFusedMoE"
        if rank is not None
        else "RuntimeDynamicNvFp4FusedMoE"
    )
    DynamicExperts.__name__ = DynamicMoeMethod.__name__ + "Experts"
    DynamicMoeMethod.experts_class = DynamicExperts
    return DynamicMoeMethod


def make_svdq_moe_method(
    base_class: type,
    experts_base: type,
    rank: int,
    *,
    activation_quant_range: float = 1536.0,
    enable_w13_correction: bool = True,
    enable_w2_correction: bool = True,
) -> type:
    """Build the SVDQ specialization of the runtime-dynamic MoE method."""
    return make_dynamic_moe_method(
        base_class,
        experts_base,
        activation_quant_range,
        rank,
        enable_w13_correction=enable_w13_correction,
        enable_w2_correction=enable_w2_correction,
    )


__all__ = [
    "make_batched_svdq_moe_kernel",
    "make_dynamic_moe_method",
    "make_svdq_moe_method",
]
