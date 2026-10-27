# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install the adapter into the pinned vLLM process."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

from . import fused_dynamic
from .config import RuntimeConfig, load_runtime_config

CHECKPOINT_ENV: Final = "VLLM_NVFP4_CHECKPOINT"
DISABLE_DENSE_SVDQ_ENV: Final = "VLLM_NVFP4_DISABLE_DENSE_SVDQ"
DISABLE_MOE_SVDQ_ENV: Final = "VLLM_NVFP4_DISABLE_MOE_SVDQ"
DISABLE_MOE_W13_SVDQ_ENV: Final = "VLLM_NVFP4_DISABLE_MOE_W13_SVDQ"
DISABLE_MOE_W2_SVDQ_ENV: Final = "VLLM_NVFP4_DISABLE_MOE_W2_SVDQ"


def _configure_flashinfer_quantization(config: RuntimeConfig) -> None:
    os.environ["FLASHINFER_NVFP4_4OVER6"] = (
        "1" if config.four_over_six_enabled else "0"
    )
    if config.four_over_six_enabled:
        os.environ["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] = "1"
        os.environ.setdefault("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MSE")
    else:
        os.environ.pop("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", None)
        os.environ.pop("FLASHINFER_NVFP4_4OVER6_ERR_MODE", None)


def _patch_activation_quantizers(config: RuntimeConfig) -> None:
    if not config.four_over_six_enabled:
        return
    import vllm._custom_ops as custom_ops
    import vllm.model_executor.kernels.linear.nvfp4.cutlass as cutlass_linear
    import vllm.model_executor.kernels.linear.nvfp4.flashinfer as flashinfer_linear
    import vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe as batched_moe

    from .quantization import (
        quantize_dense_4o6,
        quantize_grouped_4o6,
    )

    def scaled_fp4_quant(
        input,
        input_global_scale,
        is_sf_swizzled_layout=True,
        backend="none",
        padded_n=None,
    ):
        if padded_n is not None and padded_n != input.shape[-1]:
            return quantize_dense_4o6(
                input,
                input_global_scale,
                padded_n=padded_n,
                swizzled_layout=is_sf_swizzled_layout,
            )
        # Opaque custom op so torch.compile can trace the 4o6 dense path.
        return fused_dynamic.quant_4o6_e4m3(input, input_global_scale, is_sf_swizzled_layout)

    custom_ops.scaled_fp4_quant = scaled_fp4_quant
    cutlass_linear.scaled_fp4_quant = scaled_fp4_quant
    flashinfer_linear.scaled_fp4_quant = scaled_fp4_quant
    batched_moe.scaled_fp4_grouped_quantize = quantize_grouped_4o6


def _patch_dynamic_dense(config: RuntimeConfig) -> None:
    from vllm.model_executor.layers.quantization import modelopt

    from .linear import make_dynamic_linear_method

    linear_method = make_dynamic_linear_method(
        modelopt.ModelOptNvFp4LinearMethod,
        config.activation_quant_range,
    )
    modelopt.ModelOptNvFp4LinearMethod = linear_method
    modelopt.ModelOptNvFp4Config.LinearMethodCls = linear_method


def _patch_dense_svdq(config: RuntimeConfig) -> None:
    if not config.svdq_enabled:
        return
    from vllm.model_executor.layers.quantization import modelopt

    from .linear import make_svdq_linear_method

    linear_method = make_svdq_linear_method(
        modelopt.ModelOptNvFp4LinearMethod,
        config.rank,
        enable_correction=os.environ.get(DISABLE_DENSE_SVDQ_ENV) != "1",
    )
    modelopt.ModelOptNvFp4LinearMethod = linear_method
    modelopt.ModelOptNvFp4Config.LinearMethodCls = linear_method


def _patch_dynamic_moe(config: RuntimeConfig) -> None:
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (
        FlashInferCuteDSLBatchedExperts,
    )
    from vllm.model_executor.layers.quantization import modelopt

    from .moe import make_dynamic_moe_method

    disable_all = os.environ.get(DISABLE_MOE_SVDQ_ENV) == "1"
    moe_method = make_dynamic_moe_method(
        modelopt.ModelOptNvFp4FusedMoE,
        FlashInferCuteDSLBatchedExperts,
        config.activation_quant_range,
        config.rank if config.svdq_enabled else None,
        enable_w13_correction=(
            not disable_all and os.environ.get(DISABLE_MOE_W13_SVDQ_ENV) != "1"
        ),
        enable_w2_correction=(
            not disable_all and os.environ.get(DISABLE_MOE_W2_SVDQ_ENV) != "1"
        ),
    )
    modelopt.ModelOptNvFp4FusedMoE = moe_method
    modelopt.ModelOptNvFp4Config.FusedMoEMethodCls = moe_method


def install(checkpoint: str | Path | None = None) -> RuntimeConfig:
    """Validate the artifact and patch Dense plus MoE NVFP4 seams."""
    checkpoint_value = checkpoint or os.environ.get(CHECKPOINT_ENV)
    if checkpoint_value is None:
        raise ValueError(f"Set {CHECKPOINT_ENV} to the converted checkpoint.")
    config = load_runtime_config(checkpoint_value)
    _configure_flashinfer_quantization(config)
    _patch_activation_quantizers(config)
    _patch_dynamic_dense(config)
    _patch_dense_svdq(config)
    _patch_dynamic_moe(config)
    return config


__all__ = ["install"]
