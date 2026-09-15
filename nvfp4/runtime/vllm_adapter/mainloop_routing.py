# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native TRT compact routing and bounded reductions for captured Qwen3 shapes."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _input_amax(X, Ids, Partial, M: tl.constexpr, H: tl.constexpr,
                K: tl.constexpr, Start: tl.constexpr, E: tl.constexpr,
                BM: tl.constexpr, BH: tl.constexpr):
    rows = tl.program_id(1) * BM + tl.arange(0, BM)
    routes = tl.arange(0, K)
    ids = tl.load(Ids + rows[:, None] * K + routes[None, :], rows[:, None] < M, -1)
    selected = tl.sum(((ids >= Start) & (ids < Start + E)).to(tl.int32), 1) > 0
    cols = tl.program_id(0) * BH + tl.arange(0, BH)
    x = tl.load(X + rows[:, None] * H + cols[None, :],
                (rows[:, None] < M) & (cols[None, :] < H) & selected[:, None], 0).to(tl.float32)
    tl.store(Partial + tl.program_id(1) * triton.cdiv(H, BH) + tl.program_id(0),
             tl.max(tl.max(tl.abs(x), 1), 0))


@triton.jit
def _scale(Partial, Live, Inv, One, P: tl.constexpr, Q: tl.constexpr):
    p = tl.arange(0, triton.next_power_of_2(P))
    amax = tl.maximum(tl.max(tl.load(Partial + p, p < P, 0)), 1.e-12)
    valid = tl.load(Live) > 0
    amax = tl.where(valid, amax, 1.)
    tl.store(Inv, tl.div_rn(1., amax * (1. / Q)))
    tl.store(One, 1.)


@dataclass(frozen=True, slots=True)
class Routes:
    groups: torch.Tensor
    limits: torch.Tensor
    permutation: torch.Tensor
    live: torch.Tensor
    inverse_scale: torch.Tensor
    one: torch.Tensor


def prepare(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, start: int,
            experts: int, total_experts: int, quant_range: float) -> Routes:
    """Use donor multi-tile routing and retain the per-64-expert dynamic scale."""
    from trt_routing import compact_sort

    tokens, hidden = x.shape
    groups, limits, permutation, live = compact_sort(ids, weights, total_experts, start, experts)
    inv = torch.empty(1, device=x.device, dtype=torch.float32)
    one = torch.empty_like(inv)
    row_block = min(128, triton.next_power_of_2(tokens))
    columns = triton.cdiv(hidden, 64)
    row_blocks = triton.cdiv(tokens, row_block)
    partial = torch.empty(columns * row_blocks, device=x.device, dtype=torch.float32)
    _input_amax[(columns, row_blocks)](x, ids, partial, tokens, hidden, ids.shape[1],
                                      start, experts, row_block, 64)
    _scale[(1,)](partial, live, inv, one, partial.numel(), quant_range)
    return Routes(groups, limits, permutation, live, inv, one)


@triton.jit
def _compact_amax(X, Limits, Live, Partial, H: tl.constexpr, BH: tl.constexpr):
    tile = tl.program_id(0)
    block = tl.program_id(1)
    row = tile * 128 + tl.arange(0, 128)
    col = block * BH + tl.arange(0, BH)
    live = tl.load(Live)
    limit = tl.load(Limits + tile, tile < live, 0)
    value = tl.load(X + row[:, None] * H + col[None, :],
                    (row[:, None] < limit) & (col[None, :] < H), 0).to(tl.float32)
    tl.store(Partial + tile * triton.cdiv(H, BH) + block,
             tl.max(tl.max(tl.abs(value), 1), 0))


@triton.jit
def _amax_finish(Partial, Result, P: tl.constexpr, Q: tl.constexpr):
    p = tl.arange(0, triton.next_power_of_2(P))
    amax = tl.maximum(tl.max(tl.load(Partial + p, p < P, 0)), 1.e-12)
    tl.store(Result, amax)
    tl.store(Result + 1, tl.div_rn(Q, amax))


def compact_scale(values: torch.Tensor, route: Routes, quant_range: float) -> torch.Tensor:
    """Independent live-row amax for validating the fused publication ABI."""
    rows, hidden = values.shape
    blocks = triton.cdiv(hidden, 64)
    partial = torch.empty(rows // 128 * blocks, device=values.device, dtype=torch.float32)
    result = torch.empty(2, device=values.device, dtype=torch.float32)
    _compact_amax[(rows // 128, blocks)](values, route.limits, route.live, partial, hidden, 64)
    _amax_finish[(1,)](partial, result, partial.numel(), quant_range)
    return result
