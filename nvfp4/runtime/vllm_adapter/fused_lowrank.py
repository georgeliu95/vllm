# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accumulate masked SVDQ reconstruction on supported GPU architectures."""
from __future__ import annotations

from typing import Final

import torch
from trt_kernels.masked_lowrank import _check_layout, _tile_map
from trt_kernels.runner import Sm100Bf16ContiguousGroupedGemmRunner

LOWRANK_ROWS_PER_CHUNK: Final[int] = 128


def _add_sm120_correction(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
    counts: torch.Tensor, output: torch.Tensor, *, gated: bool,
) -> None:
    """Use native BF16 batched GEMMs with bounded reconstruction storage."""
    from .moe_kernel import grouped_gate_up_lowrank, grouped_lowrank

    capacity = min(values.shape[1], int(counts.amax().item()))
    for start in range(0, capacity, LOWRANK_ROWS_PER_CHUNK):
        end = min(start + LOWRANK_ROWS_PER_CHUNK, capacity)
        chunk = values[:, start:end]
        correction = (
            grouped_gate_up_lowrank(chunk, us, vh)
            if gated else grouped_lowrank(chunk, us, vh)
        )
        positions = torch.arange(start, end, device=values.device)
        active = positions[None, :] < counts[:, None]
        original = output[:, start:end]
        output[:, start:end] = torch.where(
            active[:, :, None], original + correction, original
        )


def add_correction(values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
                   counts: torch.Tensor, output: torch.Tensor, *, gated: bool) -> None:
    """Round reconstruction to BF16, then reduce-add into the main GEMM result.

    Only valid rows are observable. Padding may contain arbitrary values, just
    as in the masked FP4 GEMM; downstream reductions must use the live counts.
    """
    _check_layout(values, us, vh, counts, gated=gated)
    major = torch.cuda.get_device_capability(values.device)[0]
    if 12 <= major < 13:
        _add_sm120_correction(values, us, vh, counts, output, gated=gated)
        return
    experts, capacity, hidden = values.shape
    mapping = _tile_map(experts, capacity, torch.cuda.current_device())
    runner = Sm100Bf16ContiguousGroupedGemmRunner.get_cached(64, torch.cuda.current_device())
    projected = values.new_empty((experts * capacity, vh.shape[1]))
    runner.forward(values.reshape(-1, hidden), vh.to(torch.bfloat16).contiguous(), projected, mapping, None)
    result = output.view(experts * capacity, -1)
    if gated:
        for branch in range(2):
            runner.forward(projected.chunk(2, dim=1)[branch],
                           us.chunk(2, dim=1)[branch].to(torch.bfloat16).contiguous(),
                           result.chunk(2, dim=1)[branch], mapping, None, accumulate=True)
    else:
        runner.forward(projected, us.to(torch.bfloat16).contiguous(), result,
                       mapping, None, accumulate=True)
