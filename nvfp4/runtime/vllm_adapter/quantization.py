# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""4o6 activation quantization with vLLM-compatible NVFP4 layouts."""

from __future__ import annotations

import torch
from torch.nn import functional as torch_functional

NVFP4_BLOCK_SIZE = 16
STANDARD_NVFP4_QUANT_RANGE = 448.0 * 6.0
FOUR_OVER_SIX_QUANT_RANGE = 256.0 * 6.0
GROUPED_SCALE_TEMP_BYTES = 64 * 1024 * 1024


class DynamicScaleError(ValueError):
    """A runtime activation tensor cannot produce valid NVFP4 scales."""


def dynamic_global_scale(
    values: torch.Tensor, quant_range: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return runtime input multiplier and its dequantization reciprocal."""
    if quant_range <= 0.0:
        raise DynamicScaleError("quant_range must be positive.")
    if values.is_cuda and values.is_contiguous() and 0 < values.numel() <= 128 * 12288:
        from .fused_scale import dense_scale
        return dense_scale(values, quant_range)
    amax = values.detach().float().abs().amax().clamp_min(1e-12)
    dequant_scale = amax / quant_range
    return dequant_scale.reciprocal(), dequant_scale


def dynamic_grouped_global_scale(
    values: torch.Tensor,
    valid_rows: torch.Tensor,
    quant_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute one shared runtime global scale over all valid expert rows."""
    if values.ndim != 3 or valid_rows.shape != (values.shape[0],):
        raise DynamicScaleError(
            "Grouped values and valid-row counts have incompatible shapes."
        )
    if quant_range <= 0.0:
        raise DynamicScaleError("quant_range must be positive.")
    if values.is_cuda and values.is_contiguous() and values.shape[1] <= 128:
        from .fused_scale import grouped_scale
        inv, deq, _ = grouped_scale(values, valid_rows, quant_range)
        return inv, deq
    elements_per_row = values.shape[0] * values.shape[2]
    rows_per_chunk = max(
        1,
        GROUPED_SCALE_TEMP_BYTES
        // max(1, elements_per_row * values.element_size()),
    )
    amax = torch.zeros((), dtype=torch.float32, device=values.device)
    for row_start in range(0, values.shape[1], rows_per_chunk):
        row_end = min(values.shape[1], row_start + rows_per_chunk)
        rows = torch.arange(row_start, row_end, device=values.device)
        valid = rows.unsqueeze(0) < valid_rows.unsqueeze(1)
        # The source tensor is already BF16/FP16. Taking abs/max in its source
        # dtype preserves the selected representable value while avoiding a
        # full-tensor FP32 copy (32 GiB at the largest MoE scheduler batch).
        chunk_amax = (
            values[:, row_start:row_end]
            .detach()
            .abs()
            .masked_fill(~valid.unsqueeze(2), 0.0)
            .amax()
            .float()
        )
        amax = torch.maximum(amax, chunk_amax)
    amax = amax.clamp_min(1e-12)
    has_valid_rows = valid_rows.sum() > 0
    safe_amax = torch.where(has_valid_rows, amax, torch.ones_like(amax))
    shared_dequant_scale = safe_amax / quant_range
    dequant_scale = shared_dequant_scale.expand(values.shape[0])
    return dequant_scale.reciprocal(), dequant_scale


def _cute_quantize(
    values: torch.Tensor, global_scale: torch.Tensor, *, sf_layout: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    from flashinfer.quantization.kernels.nvfp4_quantize import (
        nvfp4_quantize_cute_dsl,
    )

    packed, scales = nvfp4_quantize_cute_dsl(
        values, global_scale, sf_layout=sf_layout, enable_pdl=False
    )
    return packed, scales.view(torch.float8_e4m3fn)


def quantize_dense_4o6(
    values: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    padded_n: int | None = None,
    swizzled_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``[M,K]`` activations with adaptive 4-or-6 scaling."""
    original_shape = values.shape
    matrix = values.reshape(-1, original_shape[-1])
    target_n = padded_n if padded_n is not None else matrix.shape[-1]
    if target_n < matrix.shape[-1] or target_n % NVFP4_BLOCK_SIZE:
        raise ValueError("padded_n must cover K and be divisible by 16.")
    if target_n != matrix.shape[-1]:
        padded = matrix.new_zeros((matrix.shape[0], target_n))
        padded[:, : matrix.shape[-1]] = matrix
        matrix = padded
    sf_layout = 0 if swizzled_layout else 2
    quant_scale = (
        global_scale.reshape(1)
        if global_scale.numel() == 1
        else _shared_scale(global_scale)
    )
    return _cute_quantize(
        matrix.contiguous(), quant_scale, sf_layout=sf_layout
    )


def _shared_scale(scales: torch.Tensor) -> torch.Tensor:
    flattened = scales.float().flatten()
    if flattened.numel() == 0:
        raise ValueError("The grouped global scale cannot be empty.")
    if not torch.allclose(flattened, flattened[:1].expand_as(flattened)):
        raise ValueError("4o6 grouped quantization requires a shared global scale.")
    return flattened[:1]


def quantize_grouped_4o6(
    values: torch.Tensor,
    mask: torch.Tensor,
    global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize masked experts with one runtime global scale per expert."""
    experts = values.shape[0]
    if global_scale.numel() not in (1, experts):
        raise DynamicScaleError(
            "Grouped global scale must be scalar or contain one value per expert."
        )
    from flashinfer import scaled_fp4_grouped_quantize

    return scaled_fp4_grouped_quantize(values, mask, global_scale)


def swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """Apply the unfused SwiGLU used before the second expert GEMM."""
    gate, up = gate_up.chunk(2, dim=-1)
    return torch_functional.silu(gate) * up


__all__ = [
    "FOUR_OVER_SIX_QUANT_RANGE",
    "STANDARD_NVFP4_QUANT_RANGE",
    "DynamicScaleError",
    "dynamic_global_scale",
    "dynamic_grouped_global_scale",
    "quantize_dense_4o6",
    "quantize_grouped_4o6",
    "swiglu",
]
