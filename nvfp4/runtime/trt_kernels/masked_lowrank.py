# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration boundary for masked expert-major SVDQ corrections."""

from dataclasses import dataclass
from functools import lru_cache

import torch

from .runner import Sm100Bf16ContiguousGroupedGemmRunner


@dataclass(frozen=True, slots=True)
class LayoutError(ValueError):
    """The serving adapter supplied an unsupported low-rank tensor layout."""

    reason: str

    def __str__(self) -> str:
        return self.reason


@lru_cache(maxsize=128)
def _tile_map(experts: int, tokens: int, device_index: int) -> torch.Tensor:
    """Map each fixed 64-row tile to its padded-layout expert without readback."""
    expert_ids = torch.arange(
        experts, device=torch.device("cuda", device_index), dtype=torch.int32,
    )
    return expert_ids.repeat_interleave(tokens // 64)


def _check_layout(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
    valid_rows: torch.Tensor, *, gated: bool,
) -> None:
    """Check metadata only; valid-row tensor contents stay on the GPU."""
    if values.ndim != 3 or us.ndim != 3 or vh.ndim != 3:
        raise LayoutError("Expected three-dimensional expert-major operands")
    experts, tokens, hidden = values.shape
    branches = 2 if gated else 1
    if not values.is_cuda or values.dtype != torch.bfloat16 or tokens % 64:
        raise LayoutError("Expected CUDA BF16 rows padded to a multiple of 64")
    if us.shape[0] != experts or vh.shape[0] != experts or vh.shape[2] != hidden:
        raise LayoutError("Expert counts or contracted activation width differ")
    if us.shape[2] != 64 or vh.shape[1] != branches * 64:
        raise LayoutError("This candidate requires the validated rank-64 factors")
    if us.shape[1] % branches or valid_rows.shape != (experts,):
        raise LayoutError("Invalid gate/up split or expert valid-row counts")
    if any(tensor.device != values.device for tensor in (us, vh, valid_rows)):
        raise LayoutError("All low-rank operands must share one CUDA device")


def _project_into(
    values: torch.Tensor, weights: torch.Tensor, output: torch.Tensor,
    tile_map: torch.Tensor,
) -> None:
    """Use the original TRT-LLM kernel for one factor projection."""
    runner = Sm100Bf16ContiguousGroupedGemmRunner.get_cached(
        64, torch.cuda.current_device(),
    )
    runner.forward(
        values, weights.to(torch.bfloat16).contiguous(), output, tile_map, None,
    )


def _zero_padding(result: torch.Tensor, valid_rows: torch.Tensor) -> torch.Tensor:
    """Clear all invalid rows, including NaNs propagated from uninitialized padding."""
    rows = torch.arange(result.shape[1], device=result.device)
    invalid = rows[None, :] >= valid_rows[:, None]
    return result.masked_fill_(invalid[:, :, None], 0)


def grouped_lowrank_masked(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """Compute masked (X @ Vh.T) @ US.T without dynamic expert indexing."""
    _check_layout(values, us, vh, valid_rows, gated=False)
    experts, tokens, hidden = values.shape
    mapping = _tile_map(experts, tokens, torch.cuda.current_device())
    projected = values.new_empty((experts * tokens, vh.shape[1]))
    output = values.new_empty((experts * tokens, us.shape[1]))
    _project_into(values.reshape(-1, hidden), vh, projected, mapping)
    _project_into(projected, us, output, mapping)
    return _zero_padding(output.view(experts, tokens, -1), valid_rows)


def grouped_gate_up_lowrank_masked(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor,
    valid_rows: torch.Tensor,
) -> torch.Tensor:
    """Keep independent gate/up factors while removing token-bucket dispatch."""
    _check_layout(values, us, vh, valid_rows, gated=True)
    experts, tokens, hidden = values.shape
    mapping = _tile_map(experts, tokens, torch.cuda.current_device())
    projected = values.new_empty((experts * tokens, vh.shape[1]))
    output = values.new_empty((experts * tokens, us.shape[1]))
    _project_into(values.reshape(-1, hidden), vh, projected, mapping)
    gate_projected, up_projected = projected.chunk(2, dim=1)
    gate_us, up_us = us.chunk(2, dim=1)
    gate_output, up_output = output.chunk(2, dim=1)
    _project_into(gate_projected, gate_us, gate_output, mapping)
    _project_into(up_projected, up_us, up_output, mapping)
    return _zero_padding(output.view(experts, tokens, -1), valid_rows)
