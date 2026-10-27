# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-major 4o6 + SVDQ MoE for SM120, safe under CUDA graph capture.

Replaces, on SM12x, the padded ``[E, capacity, K]`` path of ``moe_kernel.run_4o6_svdq_moe`` (whose SM120
fallback copies all expert weights and synchronizes with the host on every call) while keeping its
arithmetic:

* routes are sorted by expert (one sorting program up to ROUTING_MAX_ROWS routes, else vLLM's
  ``get_cutlass_moe_mm_data``); both expert GEMMs are vLLM's ``cutlass_fp4_group_mm`` (SM120 CUTLASS),
  which skips experts without rows;
* each activation global scale is shared by a 64-expert chunk, as the reference computes one scale per
  ``_MAX_EXPERTS_PER_GROUPED_GEMM`` chunk call, from the same amax and with the same FP32 expressions;
* activations are quantized by FlashInfer's TRT-LLM 4o6 device code (``csrc/moe_sm120.cu``);
* the SwiGLU used for the FC2 scale and the FC2 correction is the reference's Triton expression;
* rank-64 corrections are rounded to BF16 and added to the BF16 GEMM outputs;
* routes are combined per token in ascending expert order with BF16 rounding after every add.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

CHUNK_EXPERTS = 64  # experts per shared activation scale (the reference's grouped-GEMM chunk)
ROUTING_MAX_ROWS = 2048  # one-program routing (sort) up to this many routes
GEMV_ROWS = 4  # rows of one expert per grouped-GEMV tile (csrc kGemvRows)
GEMV_MAX_ROWS = int(os.environ.get("VLLM_NVFP4_SM120_GEMV_MAX_ROWS", "64"))  # routes; above: CUTLASS
OVERLAP_MAX_ROWS = int(os.environ.get("VLLM_NVFP4_SM120_OVERLAP_MAX_ROWS", "2048"))  # side-stream low-rank
_ext = None


def extension():
    """JIT-build (cached in TORCH_EXTENSIONS_DIR) the 4o6 expert quantizer with FlashInfer's flags."""
    global _ext
    if _ext is None:
        import flashinfer
        from torch.utils.cpp_extension import load

        data = Path(flashinfer.__file__).parent / "data"
        csrc = data / "csrc" / "nv_internal"
        # TRT-LLM headers need the half / bfloat16 operators that torch's extension flags switch off.
        flags = ["-DENABLE_BF16", "-DENABLE_FP8", "-DENABLE_FP4", "-U__CUDA_NO_HALF_OPERATORS__",
                 "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                 "-U__CUDA_NO_HALF2_OPERATORS__"]
        _ext = load(
            name="nvfp4_q35_moe_sm120_v4",
            sources=[str(Path(__file__).with_name("csrc") / "moe_sm120.cu")],
            extra_include_paths=[str(csrc), str(csrc / "include"), str(data / "include"),
                                 str(data / "cutlass" / "include"), str(data / "cutlass" / "tools" / "util" / "include"),
                                 str(data / "spdlog" / "include")],
            extra_cflags=["-O3", "-DNDEBUG", *flags],
            extra_cuda_cflags=["-O3", "-DNDEBUG", "-use_fast_math", "-std=c++17",
                               "-gencode=arch=compute_120f,code=sm_120f", *flags],
        )
    return _ext


def _rcp(quant_range: float) -> float:
    return float(torch.tensor(1.0 / quant_range, dtype=torch.float32))


@triton.jit
def _publish_scales(cmax, OFFSETS, AMAX, COUNTER, WGS, GSCALE, ALPHA, E: tl.constexpr, CHUNK: tl.constexpr,
                    CHUNKS: tl.constexpr, RCP_Q, SINGLE: tl.constexpr):
    """Reduce the per-chunk maxima of all programs; the last program writes inv and alpha per expert.

    deq = max(amax, 1e-12) * (1 / Q) and inv = 1 / deq (round to nearest), amax = 1 for chunks without
    rows: the expressions of fused_scale._reduce and of the eager dynamic_grouped_global_scale.
    """
    chunks = tl.arange(0, CHUNKS)
    last = True
    if not SINGLE:
        tl.atomic_max(AMAX + chunks, cmax.to(tl.int32, bitcast=True))
        last = tl.atomic_add(COUNTER, 1) == tl.num_programs(0) - 1
    if last:
        if not SINGLE:
            cmax = tl.atomic_xchg(AMAX + chunks, tl.zeros((CHUNKS,), tl.int32)).to(tl.float32, bitcast=True)
            tl.atomic_xchg(COUNTER, 0)
        start = tl.load(OFFSETS + chunks * CHUNK)
        end = tl.load(OFFSETS + chunks * CHUNK + CHUNK)
        amax = tl.where(end > start, tl.maximum(cmax, 1.0e-12), 1.0)
        deq = amax * RCP_Q
        inv = libdevice.div_rn(1.0, deq)
        expert = tl.arange(0, E)
        mine = (expert // CHUNK)[:, None] == chunks[None, :]
        deq_e = tl.sum(tl.where(mine, deq[None, :], 0.0), 1)
        inv_e = tl.sum(tl.where(mine, inv[None, :], 0.0), 1)
        tl.store(GSCALE + expert, inv_e)
        tl.store(ALPHA + expert, deq_e * tl.load(WGS + expert))


@triton.jit
def _input_scale(X, IDS, OFFSETS, AMAX, COUNTER, WGS, GSCALE, ALPHA, M, H: tl.constexpr,
                 TOPK: tl.constexpr, BK: tl.constexpr, E: tl.constexpr, CHUNK: tl.constexpr,
                 CHUNKS: tl.constexpr, RCP_Q, TOKENS: tl.constexpr, BH: tl.constexpr, SINGLE: tl.constexpr):
    """Per-chunk |x| max over the tokens routed into the chunk; GSCALE / ALPHA per expert."""
    pid = tl.program_id(0)
    col = tl.arange(0, BH)
    route = tl.arange(0, BK)
    chunks = tl.arange(0, CHUNKS)
    cmax = tl.zeros((CHUNKS,), tl.float32)
    for i in range(TOKENS):
        t = pid * TOKENS + i
        if t < M:
            amax = tl.max(tl.abs(tl.load(X + t * H + col, col < H, other=0).to(tl.float32)), 0)
            ids = tl.load(IDS + t * TOPK + route, route < TOPK, other=-CHUNK)
            member = tl.max(((ids // CHUNK)[None, :] == chunks[:, None]).to(tl.int32), 1) > 0
            cmax = tl.where(member, tl.maximum(cmax, amax), cmax)
    _publish_scales(cmax, OFFSETS, AMAX, COUNTER, WGS, GSCALE, ALPHA, E, CHUNK, CHUNKS, RCP_Q, SINGLE)


@triton.jit
def _swiglu_scale(X, ROW_EXPERT, OFFSETS, ACT, AMAX, COUNTER, WGS, GSCALE, ALPHA, R,
                  H: tl.constexpr, E: tl.constexpr, CHUNK: tl.constexpr, CHUNKS: tl.constexpr,
                  ROWS: tl.constexpr, RB: tl.constexpr, BH: tl.constexpr, SINGLE: tl.constexpr):
    """Reference grouped_scale(swiglu=True): activated rows, per-chunk |act| max, GSCALE / ALPHA.

    RB rows at a time as one [RB, BH] block (row by row left a single-program call latency-bound).
    """
    pid = tl.program_id(0)
    col = tl.arange(0, BH)
    chunks = tl.arange(0, CHUNKS)
    cmax = tl.zeros((CHUNKS,), tl.float32)
    for rb in range(0, ROWS, RB):
        r = pid * ROWS + rb + tl.arange(0, RB)
        valid = r < R
        mask = valid[:, None] & (col[None, :] < H)
        base = r[:, None] * H * 2 + col[None, :]
        gate = tl.load(X + base, mask, other=0)
        up = tl.load(X + base + H, mask, other=0)
        sigmoid_denominator = 1.0 + libdevice.exp(-gate.to(tl.float32))
        silu = libdevice.div_rn(gate.to(tl.float32), sigmoid_denominator).to(gate.dtype)
        value = (silu.to(tl.float32) * up.to(tl.float32)).to(gate.dtype)
        tl.store(ACT + r[:, None] * H + col[None, :], value, mask)
        amax = tl.max(tl.abs(value.to(tl.float32)), 1)
        chunk = tl.load(ROW_EXPERT + r, valid, other=-CHUNK) // CHUNK
        cmax = tl.maximum(cmax, tl.max(tl.where(chunk[:, None] == chunks[None, :], amax[:, None], 0.0), 0))
    _publish_scales(cmax, OFFSETS, AMAX, COUNTER, WGS, GSCALE, ALPHA, E, CHUNK, CHUNKS, 1.0 / 1536.0, SINGLE)


@triton.jit
def _routing(IDS, OFFSETS, SF_OFFSETS, SIZES1, SIZES2, TILES, TILES_G, MAP, MAP_G, A_MAP, C_MAP, ROW_EXPERT, R,
             E: tl.constexpr, TOPK: tl.constexpr, H: tl.constexpr, I: tl.constexpr, BM: tl.constexpr,
             GBM: tl.constexpr, BR: tl.constexpr, MAXT, MAXTG, MT: tl.constexpr, MTG: tl.constexpr):
    """get_cutlass_moe_mm_data in one program for R <= BR routes, plus row experts, the row-tile prefixes
    of the low-rank kernels (BM rows) and of the grouped GEMV (GBM rows), and per-tile maps
    MAP[t] = (expert, first row, row end, expert's first row), expert -1 for unused tiles. With the maps a
    tile kernel starts with one load instead of a binary search (eight dependent loads, ~4 us at decode).

    Rows are sorted by (expert, route), so the order is deterministic.
    """
    i = tl.arange(0, BR)
    valid = i < R
    ids = tl.load(IDS + i, valid, other=E)
    key = tl.sort(ids * BR + i)
    expert_of_row = key // BR
    route = key % BR
    ok = expert_of_row < E
    tl.store(A_MAP + i, route // TOPK, ok)
    tl.store(C_MAP + route, i, ok)
    tl.store(ROW_EXPERT + i, expert_of_row, ok)
    e = tl.arange(0, E)
    counts = tl.histogram(tl.where(valid, ids, 0), E) - tl.where(e == 0, BR - R, 0)
    ends = tl.cumsum(counts, 0)
    tl.store(OFFSETS + e, ends - counts)
    tl.store(OFFSETS + E, R)
    padded = (counts + 127) // 128 * 128
    sf_ends = tl.cumsum(padded, 0)
    tl.store(SF_OFFSETS + e, sf_ends - padded)
    tl.store(SF_OFFSETS + E, tl.sum(padded, 0))
    tl.store(SIZES1 + e * 3, counts)
    tl.store(SIZES1 + e * 3 + 1, tl.full((E,), 2 * I, tl.int32))
    tl.store(SIZES1 + e * 3 + 2, tl.full((E,), H, tl.int32))
    tl.store(SIZES2 + e * 3, counts)
    tl.store(SIZES2 + e * 3 + 1, tl.full((E,), H, tl.int32))
    tl.store(SIZES2 + e * 3 + 2, tl.full((E,), I, tl.int32))
    tiles = (counts + BM - 1) // BM
    tl.store(TILES + 1 + e, tl.cumsum(tiles, 0))
    tl.store(TILES, 0)
    tiles = (counts + GBM - 1) // GBM
    tl.store(TILES_G + 1 + e, tl.cumsum(tiles, 0))
    tl.store(TILES_G, 0)
    t = tl.arange(0, MT)
    tl.store(MAP + t * 4, tl.full((MT,), -1, tl.int32), t < MAXT)
    t = tl.arange(0, MTG)
    tl.store(MAP_G + t * 4, tl.full((MTG,), -1, tl.int32), t < MAXTG)
    tl.debug_barrier()
    first = tl.load(OFFSETS + expert_of_row, ok, other=0)
    end = tl.load(OFFSETS + expert_of_row + 1, ok, other=0)
    rank = i - first
    tile = tl.load(TILES + expert_of_row, ok, other=0) + rank // BM
    tl.store(MAP + tile * 4, expert_of_row, ok)
    tl.store(MAP + tile * 4 + 1, first + rank // BM * BM, ok)
    tl.store(MAP + tile * 4 + 2, end, ok)
    tl.store(MAP + tile * 4 + 3, first, ok)
    tile = tl.load(TILES_G + expert_of_row, ok, other=0) + rank // GBM
    tl.store(MAP_G + tile * 4, expert_of_row, ok)
    tl.store(MAP_G + tile * 4 + 1, first + rank // GBM * GBM, ok)
    tl.store(MAP_G + tile * 4 + 2, end, ok)
    tl.store(MAP_G + tile * 4 + 3, first, ok)


@triton.jit
def _tile(TILES, pid, E: tl.constexpr, LOG_E: tl.constexpr):
    """Expert owning row tile ``pid`` (TILES: [E + 1] prefix of per-expert tile counts)."""
    lo = 0
    hi = E
    for _ in range(LOG_E):
        mid = (lo + hi) // 2
        go = tl.load(TILES + mid) <= pid
        lo = tl.where(go, mid, lo)
        hi = tl.where(go, hi, mid)
    return lo


@triton.jit
def _tile_rows(TILES, MAP, OFFSETS, pid, E: tl.constexpr, LOG_E: tl.constexpr, BM: tl.constexpr,
               HAS_MAP: tl.constexpr):
    """(expert, first row, row end) of row tile ``pid``; expert -1 past the last tile."""
    if HAS_MAP:
        e = tl.load(MAP + pid * 4)
        first = tl.load(MAP + pid * 4 + 1)
        last = tl.load(MAP + pid * 4 + 2)
    else:
        e = _tile(TILES, pid, E, LOG_E)
        first = tl.load(OFFSETS + e) + (pid - tl.load(TILES + e)) * BM
        last = tl.load(OFFSETS + e + 1)
        e = tl.where(pid < tl.load(TILES + E), e, -1)
    return e, first, last


@triton.jit
def _tile_prefix(OFFSETS, TILES, E: tl.constexpr, BM: tl.constexpr):
    expert = tl.arange(0, E)
    count = tl.load(OFFSETS + expert + 1) - tl.load(OFFSETS + expert)
    tiles = (count + BM - 1) // BM
    tl.store(TILES + 1 + expert, tl.cumsum(tiles, 0))
    tl.store(TILES, 0)


@triton.jit
def _lowrank_down(X, SRC, VH, TILES, MAP, OFFSETS, P, R, K: tl.constexpr, RANK: tl.constexpr,
                  E: tl.constexpr, LOG_E: tl.constexpr, GATHER: tl.constexpr, SPLIT: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, HAS_MAP: tl.constexpr):
    """P[s, r, :] = X[src(r), K-split s] @ VH[e(r)][:, K-split s]^T: FP32 partial sums, or with one split the
    BF16-rounded projection itself (P's dtype), which halves what the up kernel reads."""
    pid_t = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n = tl.program_id(2)
    e, first, last = _tile_rows(TILES, MAP, OFFSETS, pid_t, E, LOG_E, BM, HAS_MAP)
    if e < 0:
        return
    rows = first + tl.arange(0, BM)
    valid = rows < last
    if GATHER:
        src = tl.load(SRC + rows, valid, other=0)
    else:
        src = rows
    n = pid_n * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    k_len: tl.constexpr = K // SPLIT
    for k0 in range(pid_s * k_len, pid_s * k_len + k_len, BK):
        k = k0 + tl.arange(0, BK)
        a = tl.load(X + src[:, None] * K + k[None, :], valid[:, None], other=0)
        b = tl.load(VH + (e * RANK + n[:, None]) * K + k[None, :])
        acc = tl.dot(a, tl.trans(b), acc)
    tl.store(P + (pid_s * R + rows[:, None]) * RANK + n[None, :], acc.to(P.dtype.element_ty), valid[:, None])


@triton.jit
def _lowrank_up(P, US, TILES, MAP, OFFSETS, OUT, R, N: tl.constexpr, RANK: tl.constexpr, BRANCH_N: tl.constexpr,
                E: tl.constexpr, LOG_E: tl.constexpr, SPLIT: tl.constexpr, BM: tl.constexpr,
                BN: tl.constexpr, HAS_MAP: tl.constexpr, NSUB: tl.constexpr):
    """OUT[r, n] = bf16(OUT[r, n] + bf16(bf16(P[r, branch]) @ US[e(r)][n, :]^T)).

    The projection is rounded to BF16 like the reference's BF16 intermediate; with gated factors the
    output columns of branch b (BRANCH_N wide) use projection columns [64 b, 64 b + 64). A program covers
    NSUB blocks of BN columns inside one branch, so the projection tile is read once for all of them.
    """
    pid_t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e, first, last = _tile_rows(TILES, MAP, OFFSETS, pid_t, E, LOG_E, BM, HAS_MAP)
    if e < 0:
        return
    rows = first + tl.arange(0, BM)
    valid = rows < last
    branch = (pid_n * BN * NSUB) // BRANCH_N
    j = branch * 64 + tl.arange(0, 64)
    if SPLIT == 1:
        p = tl.load(P + (rows[:, None]) * RANK + j[None, :], valid[:, None], other=0).to(tl.bfloat16)
    else:
        acc = tl.zeros((BM, 64), tl.float32)
        for s in range(SPLIT):
            acc += tl.load(P + (s * R + rows[:, None]) * RANK + j[None, :], valid[:, None], other=0)
        p = acc.to(tl.bfloat16)
    for q in range(NSUB):
        n = (pid_n * NSUB + q) * BN + tl.arange(0, BN)
        us = tl.load(US + (e * N + n[:, None]) * 64 + tl.arange(0, 64)[None, :])
        corr = tl.dot(p, tl.trans(us)).to(tl.bfloat16)
        ptr = OUT + rows[:, None] * N + n[None, :]
        out = tl.load(ptr, valid[:, None], other=0)
        tl.store(ptr, (out.to(tl.float32) + corr.to(tl.float32)).to(tl.bfloat16), valid[:, None])


@triton.jit
def _combine(Y, C_MAP, W, IDS, OUT, H: tl.constexpr, K: tl.constexpr, B: tl.constexpr):
    """Reference fused_routing._combine over sorted rows: ascending expert, BF16 after every add."""
    token = tl.program_id(0)
    col = tl.program_id(1) * B + tl.arange(0, B)
    routes = tl.arange(0, triton.next_power_of_2(K))
    ids = tl.load(IDS + token * K + routes, routes < K, other=2147483647)
    acc = tl.full((B,), 0, tl.float32)
    for _ in range(K):
        expert = tl.min(ids, 0)
        route = tl.min(tl.where(ids == expert, routes, 2147483647), 0)
        row = tl.load(C_MAP + token * K + route)
        value = tl.load(Y + row * H + col, col < H, other=0)
        weight = tl.load(W + token * K + route)
        value = (value.to(tl.float32) * weight).to(value.dtype)
        acc = (acc + value.to(tl.float32)).to(value.dtype).to(tl.float32)
        ids = tl.where(routes == route, 2147483647, ids)
    tl.store(OUT + token * H + col, acc, col < H)


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


class Sm120SvdqMoE:
    """Local unique top-k routing, BF16 activations, SiLU-gated experts, rank-64 SVDQ (or none)."""

    def __init__(self, w1: torch.Tensor, w1_scale: torch.Tensor, w1_gs: torch.Tensor,
                 w2: torch.Tensor, w2_scale: torch.Tensor, w2_gs: torch.Tensor, quant_range: float,
                 w13_us: torch.Tensor | None, w13_vh: torch.Tensor | None,
                 w2_us: torch.Tensor | None, w2_vh: torch.Tensor | None) -> None:
        experts, width, packed_hidden = w1.shape
        self.experts = experts
        self.hidden = packed_hidden * 2
        self.inter = width // 2
        if experts % CHUNK_EXPERTS or experts & (experts - 1):
            raise RuntimeError("SM120 MoE expects a power-of-two multiple of 64 experts.")
        if quant_range != 1536.0:
            raise RuntimeError("SM120 MoE implements the 4o6 recipe only.")
        self.quant_range = quant_range
        self.w1 = w1.view(torch.uint8)
        self.w2 = w2.view(torch.uint8)
        self.w1_scale = w1_scale.view(torch.float8_e4m3fn)
        self.w2_scale = w2_scale.view(torch.float8_e4m3fn)
        self.w1_gs = w1_gs.float().contiguous()
        self.w2_gs = w2_gs.float().contiguous()
        self.svdq = w13_us is not None
        if self.svdq:
            if w13_us.shape[-1] != 64 or w2_us.shape[-1] != 64:
                raise RuntimeError("SM120 MoE supports rank-64 SVDQ factors.")
            self.w13_us = w13_us.to(torch.bfloat16).contiguous()   # [E, 2I, 64]
            self.w13_vh = w13_vh.to(torch.bfloat16).contiguous()   # [E, 128, H]
            self.w2_us = w2_us.to(torch.bfloat16).contiguous()     # [E, H, 64]
            self.w2_vh = w2_vh.to(torch.bfloat16).contiguous()     # [E, 64, I]
        dev = w1.device
        self.amax = torch.zeros(2, experts // CHUNK_EXPERTS, dtype=torch.int32, device=dev)
        self.counter = torch.zeros(2, dtype=torch.int32, device=dev)
        self.gscale1 = torch.empty(experts, dtype=torch.float32, device=dev)
        self.alpha1 = torch.empty_like(self.gscale1)
        self.gscale2 = torch.empty_like(self.gscale1)
        self.alpha2 = torch.empty_like(self.gscale1)
        self.log_e = experts.bit_length() - 1
        self._streams: dict[torch.device, torch.cuda.Stream] = {}
        extension()

    def _side_stream(self, dev: torch.device) -> torch.cuda.Stream:
        if dev not in self._streams:
            self._streams[dev] = torch.cuda.Stream(dev)
        return self._streams[dev]

    def __call__(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> torch.Tensor:
        from vllm import _custom_ops as ops

        ext = extension()
        x = x.contiguous()
        m, hidden = x.shape
        topk = topk_ids.shape[1]
        rows = m * topk
        e, inter, dev = self.experts, self.inter, x.device
        ids = topk_ids.to(torch.int32).contiguous()
        offsets = torch.empty(e + 1, dtype=torch.int32, device=dev)
        sf_offsets = torch.empty(e + 1, dtype=torch.int32, device=dev)
        sizes1 = torch.empty(e, 3, dtype=torch.int32, device=dev)
        sizes2 = torch.empty(e, 3, dtype=torch.int32, device=dev)
        a_map = torch.empty(rows, dtype=torch.int32, device=dev)
        c_map = torch.empty(rows, dtype=torch.int32, device=dev)
        row_expert = torch.empty(rows, dtype=torch.int32, device=dev)
        tiles = torch.empty(e + 1, dtype=torch.int32, device=dev)
        tiles_g = torch.empty(e + 1, dtype=torch.int32, device=dev)
        bm = 16 if rows <= 512 else 64
        max_tiles = triton.cdiv(rows, bm) + min(e, rows)
        gemv_tiles = triton.cdiv(rows, GEMV_ROWS) + min(e, rows)
        has_map = rows <= ROUTING_MAX_ROWS
        gemv = has_map and rows <= GEMV_MAX_ROWS
        tile_map = torch.empty(max_tiles, 4, dtype=torch.int32, device=dev)
        gemv_map = torch.empty(gemv_tiles, 4, dtype=torch.int32, device=dev)
        if has_map:
            _routing[(1,)](ids, offsets, sf_offsets, sizes1, sizes2, tiles, tiles_g, tile_map, gemv_map, a_map,
                           c_map, row_expert, rows, e, topk, hidden, inter, bm, GEMV_ROWS,
                           triton.next_power_of_2(rows), max_tiles, gemv_tiles, triton.next_power_of_2(max_tiles),
                           triton.next_power_of_2(gemv_tiles), num_warps=8)
        else:
            ops.get_cutlass_moe_mm_data(ids, offsets, sizes1, sizes2, a_map, c_map, e, inter, hidden,
                                        sf_offsets)
            _tile_prefix[(1,)](offsets, tiles, e, bm)

        chunks = e // CHUNK_EXPERTS
        tokens_per_program = 1 if m <= 64 else 8
        grid = triton.cdiv(m, tokens_per_program)
        if grid <= 8:  # one program covers the tokens: no cross-program reduction
            tokens_per_program, grid = m, 1
        # The low-rank down projections depend only on the routed inputs: at small M every kernel is
        # latency-bound, so they run on a side stream next to the quantize + GEMM chain (joined before use).
        main = torch.cuda.current_stream(dev)
        side = self._side_stream(dev) if self.svdq and rows <= OVERLAP_MAX_ROWS else None
        if self.svdq:
            split = 8 if rows <= 512 else 1
            bn13 = 64 if split > 1 else 128  # one split: the whole rank per program, so x is gathered once
            p13 = torch.empty(split, rows, 128, dtype=torch.float32 if split > 1 else torch.bfloat16, device=dev)
            if side is not None:
                side.wait_stream(main)
            with torch.cuda.stream(side or main):
                _lowrank_down[(max_tiles, split, 128 // bn13)](
                    x, a_map, self.w13_vh, tiles, tile_map, offsets, p13, rows, hidden, 128, e, self.log_e, True,
                    split, bm, bn13, 64, has_map)
        _input_scale[(grid,)](x, ids, offsets, self.amax[0], self.counter[0:1], self.w1_gs, self.gscale1,
                              self.alpha1, m, hidden, topk, triton.next_power_of_2(topk), e, CHUNK_EXPERTS,
                              chunks, _rcp(self.quant_range), tokens_per_program, triton.next_power_of_2(hidden),
                              grid == 1)

        sf_rows = _round_up(rows, 128) + 128 * min(e, rows)
        aq = torch.empty(rows, hidden // 2, dtype=torch.uint8, device=dev)
        aq_sf = torch.empty(sf_rows, _round_up(hidden // 16, 4), dtype=torch.uint8, device=dev)
        ext.experts_quant_4o6(x, a_map, offsets, sf_offsets, self.gscale1, hidden, False, aq, aq_sf, row_expert)

        gate_up = torch.empty(rows, 2 * inter, dtype=x.dtype, device=dev)
        if gemv:
            ext.grouped_gemv_fp4(aq, aq_sf, self.w1, self.w1_scale, self.alpha1, sf_offsets, gemv_map,
                                 gemv_tiles, gate_up)
        else:
            ops.cutlass_fp4_moe_mm(gate_up, aq, self.w1, aq_sf.view(torch.float8_e4m3fn), self.w1_scale,
                                   self.alpha1, sizes1, offsets[:-1], sf_offsets[:-1])

        if self.svdq:
            if side is not None:
                main.wait_stream(side)
            nsub13 = 1 if rows <= 512 else inter // 64  # decode: more programs; prefill: one per branch
            _lowrank_up[(max_tiles, 2 * inter // (64 * nsub13))](
                p13, self.w13_us, tiles, tile_map, offsets, gate_up, rows, 2 * inter, 128, inter, e, self.log_e,
                split, bm, 64, has_map, nsub13)

        activated = torch.empty(rows, inter, dtype=x.dtype, device=dev)
        rows_per_program = 8 if rows <= 4096 else 16
        grid = triton.cdiv(rows, rows_per_program)
        if grid <= 4:
            rows_per_program, grid = _round_up(rows, 8), 1
        _swiglu_scale[(grid,)](
            gate_up, row_expert, offsets, activated, self.amax[1], self.counter[1:2], self.w2_gs,
            self.gscale2, self.alpha2, rows, inter, e, CHUNK_EXPERTS, chunks, rows_per_program, 8,
            triton.next_power_of_2(inter), grid == 1, enable_fp_fusion=False)

        if self.svdq:
            p2 = torch.empty(1, rows, 64, dtype=torch.bfloat16, device=dev)
            if side is not None:
                side.wait_stream(main)
            with torch.cuda.stream(side or main):
                _lowrank_down[(max_tiles, 1, 1)](
                    activated, a_map, self.w2_vh, tiles, tile_map, offsets, p2, rows, inter, 64, e, self.log_e,
                    False, 1, bm, 64, 64, has_map)
        iq = torch.empty(rows, inter // 2, dtype=torch.uint8, device=dev)
        iq_sf = torch.empty(sf_rows, _round_up(inter // 16, 4), dtype=torch.uint8, device=dev)
        ext.experts_quant_4o6(gate_up, None, offsets, sf_offsets, self.gscale2, inter, True, iq, iq_sf, None)

        down = torch.empty(rows, hidden, dtype=x.dtype, device=dev)
        if gemv:
            ext.grouped_gemv_fp4(iq, iq_sf, self.w2, self.w2_scale, self.alpha2, sf_offsets, gemv_map,
                                 gemv_tiles, down)
        else:
            ops.cutlass_fp4_moe_mm(down, iq, self.w2, iq_sf.view(torch.float8_e4m3fn), self.w2_scale,
                                   self.alpha2, sizes2, offsets[:-1], sf_offsets[:-1])
        if self.svdq:
            if side is not None:
                main.wait_stream(side)
            nsub2 = 1 if rows <= 512 else 8
            _lowrank_up[(max_tiles, hidden // (128 * nsub2))](
                p2, self.w2_us, tiles, tile_map, offsets, down, rows, hidden, 64, hidden, e, self.log_e, 1, bm, 128,
                has_map, nsub2)

        out = torch.empty(m, hidden, dtype=x.dtype, device=dev)
        _combine[(m, triton.cdiv(hidden, 256))](down, c_map, topk_weights.contiguous(), ids, out, hidden, topk,
                                               256, enable_fp_fusion=False)
        return out


__all__ = ["CHUNK_EXPERTS", "Sm120SvdqMoE", "extension"]
