# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Valid-row reductions and activation preparation without host synchronization."""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _partial(X, COUNTS, PART, ACT, C: tl.constexpr, H: tl.constexpr,
             SWIGLU: tl.constexpr, B: tl.constexpr, BLOCKS: tl.constexpr):
    expert, block = tl.program_id(0), tl.program_id(1)
    col = block * B + tl.arange(0, B)
    count = tl.load(COUNTS + expert)
    maximum = tl.full((B,), 0, tl.float32)
    for row in range(count):
        if SWIGLU:
            base = (expert * C + row) * H * 2 + col
            gate = tl.load(X + base, col < H, other=0)
            up = tl.load(X + base + H, col < H, other=0)
            sigmoid_denominator = 1.0 + libdevice.exp(-gate.to(tl.float32))
            silu = libdevice.div_rn(gate.to(tl.float32), sigmoid_denominator).to(gate.dtype)
            value = (silu.to(tl.float32) * up.to(tl.float32)).to(gate.dtype)
            tl.store(ACT + (expert * C + row) * H + col, value, col < H)
        else:
            value = tl.load(X + (expert * C + row) * H + col, col < H, other=0)
        maximum = tl.maximum(maximum, tl.abs(value.to(tl.float32)))
    tl.store(PART + expert * BLOCKS + block, tl.max(maximum, 0))


@triton.jit
def _reduce(PART, COUNTS, INV, DEQ, E: tl.constexpr, P: tl.constexpr,
            Q: tl.constexpr, B: tl.constexpr, BE: tl.constexpr):
    idx = tl.arange(0, B)
    value = tl.load(PART + idx, idx < P, other=0)
    expert = tl.arange(0, BE)
    counts = tl.load(COUNTS + expert, expert < E, other=0)
    amax = tl.maximum(tl.max(value, 0), 1.0e-12)
    amax = tl.where(tl.sum(counts, 0) > 0, amax, 1.0)
    deq = amax * (1.0 / Q)
    inv = libdevice.div_rn(1.0, deq)
    tl.store(DEQ + expert, deq, expert < E)
    tl.store(INV + expert, inv, expert < E)


def grouped_scale(values: torch.Tensor, counts: torch.Tensor, quant_range: float,
                  *, swiglu: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    experts, capacity, width = values.shape
    hidden = width // 2 if swiglu else width
    blocks = triton.cdiv(hidden, 512)
    partial = torch.empty(experts * blocks, device=values.device, dtype=torch.float32)
    activated = torch.empty((experts, capacity, hidden), device=values.device, dtype=values.dtype) if swiglu else values
    inv = torch.empty(experts, device=values.device, dtype=torch.float32)
    deq = torch.empty_like(inv)
    _partial[(experts, blocks)](values, counts, partial, activated, capacity, hidden,
                               swiglu, 512, blocks, enable_fp_fusion=False)
    _reduce[(1,)](partial, counts, inv, deq, experts, experts * blocks, quant_range,
                  triton.next_power_of_2(experts * blocks), triton.next_power_of_2(experts))
    return inv, deq, activated


@triton.jit
def _dummy(X, COUNTS, MASK, C: tl.constexpr, H: tl.constexpr, B: tl.constexpr):
    expert = tl.program_id(0)
    count = tl.load(COUNTS + expert)
    col = tl.arange(0, B)
    tl.store(MASK + expert, tl.maximum(count, 1))
    if count == 0:
        tl.store(X + expert * C * H + col, 0, col < H)


def initialize_dummy_rows(values: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    mask = torch.empty_like(counts)
    _dummy[(values.shape[0],)](values, counts, mask, values.shape[1], values.shape[2],
                              triton.next_power_of_2(values.shape[2]))
    return mask


@triton.jit
def _dense_partial(X, PART, N: tl.constexpr, B: tl.constexpr):
    block = tl.program_id(0)
    index = block * B + tl.arange(0, B)
    value = tl.load(X + index, index < N, other=0).to(tl.float32)
    tl.store(PART + block, tl.max(tl.abs(value), 0))


@triton.jit
def _dense_reduce(PART, INV, DEQ, P: tl.constexpr, Q: tl.constexpr, B: tl.constexpr):
    index = tl.arange(0, B)
    value = tl.load(PART + index, index < P, other=0)
    amax = tl.maximum(tl.max(value, 0), 1.0e-12)
    deq = amax * (1.0 / Q)
    tl.store(DEQ, deq)
    tl.store(INV, libdevice.div_rn(1.0, deq))


def dense_scale(values: torch.Tensor, quant_range: float) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = triton.cdiv(values.numel(), 4096)
    partial = torch.empty(blocks, device=values.device, dtype=torch.float32)
    inv = torch.empty((), device=values.device, dtype=torch.float32)
    deq = torch.empty_like(inv)
    _dense_partial[(blocks,)](values, partial, values.numel(), 4096)
    _dense_reduce[(1,)](partial, inv, deq, blocks, quant_range, triton.next_power_of_2(blocks))
    return inv, deq
