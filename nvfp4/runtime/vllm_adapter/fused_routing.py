# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU-only local routing with stable token order and BF16 combine rounding."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@dataclass(frozen=True, slots=True)
class RoutingContractError(ValueError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@triton.jit
def _metadata(IDS, ROWS, MAP, COUNTS, T: tl.constexpr, K: tl.constexpr,
              C: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(0)
    route = tl.arange(0, B)
    ids = tl.load(IDS + route, route < T * K, other=-1)
    selected = ids == expert
    row = tl.cumsum(selected.to(tl.int32)) - 1
    tl.store(ROWS + route, row, selected & (route < T * K))
    tl.store(MAP + expert * C + row, route // K, selected & (row < C))
    tl.store(COUNTS + expert, tl.sum(selected.to(tl.int32)))


@triton.jit
def _dispatch(X, W, MAP, COUNTS, Y, C: tl.constexpr, H: tl.constexpr,
              K: tl.constexpr, WEIGHTED: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(0)
    index = tl.program_id(1) * B + tl.arange(0, B)
    row, col = index // H, index % H
    count = tl.load(COUNTS + expert)
    valid = (row < count) & (row < C)
    token = tl.load(MAP + expert * C + row, valid, other=0)
    value = tl.load(X + token * H + col, valid, other=0)
    if WEIGHTED:
        weight = tl.load(W + token, valid, other=0).to(value.dtype)
        value = (value.to(tl.float32) * weight.to(tl.float32)).to(value.dtype)
        tl.store(X + token * H + col, value, valid)
    tl.store(Y + expert * C * H + index, value, index < C * H)


@triton.jit
def _combine(Y, EXPERT, W, IDS, ROWS, C: tl.constexpr, H: tl.constexpr,
             K: tl.constexpr, WEIGHTED: tl.constexpr, B: tl.constexpr):
    token = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    routes = tl.arange(0, triton.next_power_of_2(K))
    ids = tl.load(IDS + token * K + routes, routes < K, other=2147483647)
    acc = tl.full((B,), 0, tl.float32)
    for _ in range(K):
        expert = tl.min(ids, 0)
        route = tl.min(tl.where(ids == expert, routes, 2147483647), 0)
        row = tl.load(ROWS + token * K + route)
        ptr = EXPERT + (expert * C + row) * H + col
        value = tl.load(ptr, col < H, other=0)
        if not WEIGHTED:
            weight = tl.load(W + token * K + route)
            value = (value.to(tl.float32) * weight).to(value.dtype)
            tl.store(ptr, value, col < H)
        acc = (acc + value.to(tl.float32)).to(value.dtype).to(tl.float32)
        ids = tl.where(routes == route, 2147483647, ids)
    tl.store(Y + token * H + col, acc, col < H)


def metadata(ids: torch.Tensor, experts: int, capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, topk = ids.shape
    rows = torch.empty_like(ids)
    token_map = torch.empty((experts, capacity), device=ids.device, dtype=torch.int32)
    counts = torch.empty(experts, device=ids.device, dtype=torch.int32)
    _metadata[(experts,)](ids, rows, token_map, counts, tokens, topk, capacity,
                           triton.next_power_of_2(tokens * topk))
    return rows, token_map, counts


def prepare_local(values: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor,
                  experts: int, capacity: int, weighted_input: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if weighted_input and ids.shape[1] != 1:
        raise RoutingContractError("Input router weighting requires top-k one.")
    _, token_map, counts = metadata(ids, experts, capacity)
    result = torch.empty((experts, capacity, values.shape[1]), device=values.device, dtype=values.dtype)
    _dispatch[(experts, triton.cdiv(capacity * values.shape[1], 4096))](
        values, weights, token_map, counts, result, capacity, values.shape[1],
        ids.shape[1], weighted_input, 4096)
    return result, counts


def finalize_local(output: torch.Tensor, expert_output: torch.Tensor,
                   weights: torch.Tensor, ids: torch.Tensor, weighted_input: bool) -> None:
    experts, capacity, hidden = expert_output.shape
    rows, _, _ = metadata(ids, experts, capacity)
    _combine[(ids.shape[0], triton.cdiv(hidden, 256))](
        output, expert_output, weights, ids, rows, capacity, hidden,
        ids.shape[1], weighted_input, 256, enable_fp_fusion=False)
