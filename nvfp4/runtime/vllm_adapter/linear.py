# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense vLLM Linear support for NVFP4 4o6 plus optional SVDQ."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .quantization import dynamic_global_scale


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


def make_dynamic_linear_method(base_class: type, quant_range: float) -> type:
    """Build a LinearMethod that recomputes activation global scale per call."""

    class DynamicLinearMethod(base_class):
        def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            bias: torch.Tensor | None = None,
        ) -> torch.Tensor:
            global_scale, dequant_scale = dynamic_global_scale(x, quant_range)
            dynamic_alpha = dequant_scale * layer.weight_global_scale
            with torch.no_grad():
                layer.input_global_scale_inv.copy_(global_scale)
                layer.alpha.copy_(dynamic_alpha)
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
