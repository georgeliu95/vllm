# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Accumulate SVDQ reconstruction through the retained TRT-LLM TMA epilogue."""
from __future__ import annotations

import torch
from trt_kernels.masked_lowrank import _check_layout, _tile_map
from trt_kernels.runner import Sm100Bf16ContiguousGroupedGemmRunner


def add_correction(values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
                   counts: torch.Tensor, output: torch.Tensor, *, gated: bool) -> None:
    """Round reconstruction to BF16, then reduce-add into the main GEMM result.

    Only valid rows are observable. Padding may contain arbitrary values, just
    as in the masked FP4 GEMM; downstream reductions must use the live counts.
    """
    _check_layout(values, us, vh, counts, gated=gated)
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
