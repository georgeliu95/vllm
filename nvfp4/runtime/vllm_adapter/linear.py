# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense vLLM Linear support for NVFP4 4o6 plus optional SVDQ."""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

from . import fused_dynamic  # noqa: F401  registers torch.ops.nvfp4_opt.*
from .quantization import FOUR_OVER_SIX_QUANT_RANGE, STANDARD_NVFP4_QUANT_RANGE


def svdq_us_needs_output_sharding(
    output_partition_sizes: list[int], local_output: int, output_size: int
) -> bool:
    """Return whether U/S must accept logical or TP output shards."""
    return len(output_partition_sizes) > 1 or local_output != output_size


def _make_parameter(
    data: torch.Tensor,
    weight_loader: Callable[..., object],
    *,
    input_dim: int | None,
    output_dim: int | None,
) -> torch.nn.Parameter:
    from vllm.model_executor.parameter import BasevLLMParameter, ModelWeightParameter

    if input_dim is None and output_dim is None:
        return BasevLLMParameter(data=data, weight_loader=weight_loader)
    return ModelWeightParameter(
        data=data,
        input_dim=0 if input_dim is None else input_dim,
        output_dim=0 if output_dim is None else output_dim,
        weight_loader=weight_loader,
    )


# torch.ops is reached only from module-level helpers: vLLM cloudpickles the quant config, which pickles
# the local DynamicLinearMethod class by value, and cloudpickle cannot pickle the torch.ops namespace
# that a method body naming ``torch.ops`` would drag in.
def _write_dynamic_scale(layer: torch.nn.Module, x: torch.Tensor, quant_range: float) -> None:
    """One launch writes the inv / alpha the reference computed with dynamic_global_scale +
    multiply + two copy_ (see fused_dynamic.py)."""
    torch.ops.nvfp4_opt.dynamic_scale_(
        x, layer.input_global_scale_inv, layer.alpha, layer.weight_global_scale, quant_range
    )


def _apply_fused_dynamic_cutlass(
    layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None, quant_range: float
) -> torch.Tensor:
    """FlashInferCutlassNvFp4LinearKernel.apply_weights with the dynamic scale and the activation
    quantization (standard or 4o6, chosen by the quant range) done by one kernel (nvfp4_opt::dyn_quant)."""
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_activation_for_cutlass,
        slice_nvfp4_output,
    )
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    output_size = layer.output_size_per_partition
    x_fp4, x_blockscale = torch.ops.nvfp4_opt.dyn_quant(
        x, layer.input_global_scale_inv, layer.alpha, layer.weight_global_scale, quant_range,
        quant_range == FOUR_OVER_SIX_QUANT_RANGE,
    )
    x_fp4 = pad_nvfp4_activation_for_cutlass(x_fp4, getattr(layer, "weights_padding_cols", 0))
    out = flashinfer_scaled_fp4_mm(
        x_fp4, layer.weight, x_blockscale.view(torch.float8_e4m3fn), layer.weight_scale, layer.alpha,
        x.dtype, backend="cutlass",
    )
    out = slice_nvfp4_output(out, output_size)
    if bias is not None:
        out = out + bias
    return out.view(*x.shape[:-1], output_size)


def _fused_svdq_eligible(method: object, layer: torch.nn.Module) -> bool:
    """The SM120 fused SVDQ GEMM (svdq_fused.py) covers 4o6 activations, unpadded FlashInfer-CUTLASS
    weight layouts; it is enabled by pointing VLLM_NVFP4_SVDQ_SO at the built module."""
    return (
        bool(os.environ.get("VLLM_NVFP4_SVDQ_SO"))
        and os.environ.get("FLASHINFER_NVFP4_4OVER6") == "1"
        and type(getattr(method, "kernel", None)).__name__ == "FlashInferCutlassNvFp4LinearKernel"
        and getattr(layer, "weights_padding_cols", 0) == 0
        and layer.weight.shape[0] == layer.output_size_per_partition
    )


def _apply_fused_svdq(layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    from .svdq_fused import svdq_linear_4o6

    return svdq_linear_4o6(layer, x, FOUR_OVER_SIX_QUANT_RANGE)


def make_dynamic_linear_method(base_class: type, quant_range: float) -> type:
    """Build a LinearMethod that recomputes activation global scale per call."""

    fuse_quant = (
        quant_range in (STANDARD_NVFP4_QUANT_RANGE, FOUR_OVER_SIX_QUANT_RANGE)
        and os.environ.get("VLLM_NVFP4_FUSED_DYN_QUANT", "1") == "1"
    )

    class DynamicLinearMethod(base_class):
        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if fuse_quant and type(self.kernel).__name__ == "FlashInferCutlassNvFp4LinearKernel":
                return _apply_fused_dynamic_cutlass(layer, x, bias, quant_range)
            _write_dynamic_scale(layer, x, quant_range)
            return super().apply(layer, x, bias)

    DynamicLinearMethod.__name__ = "RuntimeDynamicNvFp4LinearMethod"
    from vllm.model_executor.layers.linear import (
        register_weight_loader_v2_supported_method,
    )

    return register_weight_loader_v2_supported_method(DynamicLinearMethod)


def make_svdq_linear_method(
    base_class: type, rank: int, *, enable_correction: bool = True
) -> type:
    """Build a pinned-version subclass without modifying the vLLM install."""

    class SvdqLinearMethod(base_class):
        def create_weights(
            self,
            layer: torch.nn.Module,
            input_size_per_partition: int,
            output_partition_sizes: list[int],
            input_size: int,
            output_size: int,
            params_dtype: torch.dtype,
            **extra_weight_attrs: object,
        ) -> None:
            super().create_weights(
                layer,
                input_size_per_partition,
                output_partition_sizes,
                input_size,
                output_size,
                params_dtype,
                **extra_weight_attrs,
            )
            weight_loader = extra_weight_attrs["weight_loader"]
            if not callable(weight_loader):
                raise TypeError("weight_loader must be callable.")
            local_output = sum(output_partition_sizes)
            is_row_parallel = input_size_per_partition != input_size
            needs_output_sharding = svdq_us_needs_output_sharding(
                output_partition_sizes, local_output, output_size
            )
            us = _make_parameter(
                torch.empty(local_output, rank, dtype=params_dtype),
                weight_loader,
                input_dim=None,
                output_dim=(
                    0 if needs_output_sharding else None
                ),
            )
            vh = _make_parameter(
                torch.empty(rank, input_size_per_partition, dtype=params_dtype),
                weight_loader,
                input_dim=1 if is_row_parallel else None,
                output_dim=None,
            )
            layer.register_parameter("svdquant_us", us)
            layer.register_parameter("svdquant_vh", vh)

        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if enable_correction and bias is None and _fused_svdq_eligible(self, layer):
                return _apply_fused_svdq(layer, x)
            quantized = super().apply(layer, x, bias)
            if not enable_correction:
                return quantized
            correction = (x @ layer.svdquant_vh.T) @ layer.svdquant_us.T
            return quantized + correction

    SvdqLinearMethod.__name__ = "ModelOptNvFp4SvdqLinearMethod"
    from vllm.model_executor.layers.linear import (
        register_weight_loader_v2_supported_method,
    )

    return register_weight_loader_v2_supported_method(SvdqLinearMethod)


__all__ = [
    "make_dynamic_linear_method",
    "make_svdq_linear_method",
    "svdq_us_needs_output_sharding",
]
