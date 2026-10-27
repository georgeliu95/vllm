# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-launch dynamic NVFP4 activation scale, and graph/compile-safe quantizer ops.

``dynamic_scale_`` replaces ``dynamic_global_scale`` + the alpha multiply + two parameter
``copy_`` calls of :class:`RuntimeDynamicNvFp4LinearMethod` with one Triton launch. It writes
bit-identical values: the reference picks between a Triton reduction (``deq = amax * (1/Q)``)
and an eager FP32 chain (``deq = amax / Q``) by tensor size, and so does this kernel.

Every program reduces a grid-strided slice, publishes its maximum with an integer atomic max
(non-negative IEEE floats order like their bit patterns), and the last program to arrive writes
``inv = 1/deq`` and ``alpha = deq * weight_global_scale`` and resets the scratch words, so the
op is safe to replay inside CUDA graphs. ``N`` is a runtime argument: no recompilation per
token count.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# The reference adapter switches to the eager FP32 chain above this many elements.
TRITON_REFERENCE_MAX_NUMEL = 128 * 12288
_BLOCK = 4096
_scratch: dict[torch.device, torch.Tensor] = {}
_num_sms: dict[torch.device, int] = {}


@triton.jit
def _dynamic_scale_kernel(X, N, SCRATCH, INV, ALPHA, WGS, RCP_Q, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, N, nprog * BLOCK):
        index = start + tl.arange(0, BLOCK)
        value = tl.load(X + index, index < N, other=0).to(tl.float32)
        acc = tl.maximum(acc, tl.abs(value))
    local = tl.max(acc, 0)
    tl.atomic_max(SCRATCH, local.to(tl.int32, bitcast=True))
    arrived = tl.atomic_add(SCRATCH + 1, 1)
    if arrived == nprog - 1:
        amax_bits = tl.atomic_xchg(SCRATCH, 0)
        tl.atomic_xchg(SCRATCH + 1, 0)
        amax = tl.maximum(amax_bits.to(tl.float32, bitcast=True), 1.0e-12)
        deq = amax * RCP_Q
        tl.store(INV, libdevice.div_rn(1.0, deq))
        tl.store(ALPHA, deq * tl.load(WGS))


def _launch(x: torch.Tensor, inv: torch.Tensor, alpha: torch.Tensor, wgs: torch.Tensor,
            quant_range: float) -> None:
    dev = x.device
    if dev not in _scratch:
        _scratch[dev] = torch.zeros(2, dtype=torch.int32, device=dev)
        _num_sms[dev] = torch.cuda.get_device_properties(dev).multi_processor_count
    n = x.numel()
    _dynamic_scale_kernel[(max(1, min(triton.cdiv(n, _BLOCK), 2 * _num_sms[dev])),)](
        x, n, _scratch[dev], inv, alpha, wgs, _reciprocal(quant_range, 0 < n <= TRITON_REFERENCE_MAX_NUMEL),
        BLOCK=_BLOCK)


def _reciprocal(quant_range: float, triton_regime: bool) -> float:
    """FP32 1/Q exactly as the reference regime forms it before multiplying by amax.

    Triton regime: the constexpr ``1.0 / Q`` is folded in double, then rounded to FP32.
    Eager regime: ``amax / Q`` with a Python scalar runs as ``amax * (1.0f / float(Q))``.
    """
    if triton_regime:
        return float(torch.tensor(1.0 / quant_range, dtype=torch.float32))
    return float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(quant_range, dtype=torch.float32))


@torch.library.custom_op("nvfp4_opt::dynamic_scale_", mutates_args=("inv", "alpha"))
def dynamic_scale_(x: torch.Tensor, inv: torch.Tensor, alpha: torch.Tensor,
                   weight_global_scale: torch.Tensor, quant_range: float) -> None:
    """Write the runtime input multiplier ``inv`` and GEMM ``alpha`` for activation ``x``."""
    _launch(x.contiguous(), inv, alpha, weight_global_scale, quant_range)


@dynamic_scale_.register_fake
def _(x, inv, alpha, weight_global_scale, quant_range) -> None:
    return None


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


_ext = None
_quant_scratch: dict[torch.device, torch.Tensor] = {}


def dyn_quant_extension():
    """JIT-build (once, cached in TORCH_EXTENSIONS_DIR) the cooperative dynamic-quant kernel."""
    global _ext
    if _ext is None:
        from pathlib import Path

        from torch.utils.cpp_extension import load

        csrc = Path(__file__).with_name("csrc")
        _ext = load(name="nvfp4_opt_dyn_quant_v4",
                    sources=[str(csrc / "dyn_quant.cu"), str(csrc / "lora_down.cu")],
                    extra_cuda_cflags=["-O3", "-gencode=arch=compute_120a,code=sm_120a"])
    return _ext


@torch.library.custom_op("nvfp4_opt::dyn_quant", mutates_args=("inv", "alpha"))
def dyn_quant(x: torch.Tensor, inv: torch.Tensor, alpha: torch.Tensor, weight_global_scale: torch.Tensor,
              quant_range: float, four_over_six: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic scale + NVFP4 quantization in one launch (csrc/dyn_quant.cu).

    Writes ``inv`` / ``alpha`` like ``dynamic_scale_`` and returns the packed e2m1 values ``[M, K/2]`` and
    the swizzled UE4M3 scales that ``scaled_fp4_quant(x, inv, True)`` (standard) or
    ``quantize_dense_4o6(x, inv)`` (``four_over_six``) would.
    """
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    dev = matrix.device
    if dev not in _quant_scratch:
        # one float per cooperative block (the grid is capped at this size)
        _quant_scratch[dev] = torch.zeros(2048, dtype=torch.float32, device=dev)
    n = matrix.numel()
    return tuple(dyn_quant_extension().dyn_quant(
        matrix, inv, alpha, weight_global_scale, _reciprocal(quant_range, 0 < n <= TRITON_REFERENCE_MAX_NUMEL),
        _quant_scratch[dev], four_over_six))


@dyn_quant.register_fake
def _(x, inv, alpha, weight_global_scale, quant_range, four_over_six=False):
    m = x.numel() // x.shape[-1]
    k = x.shape[-1]
    return (x.new_empty((m, k // 2), dtype=torch.uint8),
            x.new_empty((_round_up(m, 128), _round_up(k // 16, 4)), dtype=torch.uint8))


@torch.library.custom_op("nvfp4_opt::quant_4o6", mutates_args=())
def quant_4o6(x: torch.Tensor, global_scale: torch.Tensor, swizzled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """4o6 activation quantization (FlashInfer CuTe-DSL) as an opaque op for torch.compile."""
    from .quantization import quantize_dense_4o6
    packed, scales = quantize_dense_4o6(x, global_scale, swizzled_layout=swizzled)
    return packed, scales.view(torch.uint8)


def quant_4o6_e4m3(x: torch.Tensor, global_scale: torch.Tensor, swizzled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """``scaled_fp4_quant``-shaped result of the 4o6 op (packed e2m1, E4M3-typed scales)."""
    packed, scales = torch.ops.nvfp4_opt.quant_4o6(x.reshape(-1, x.shape[-1]), global_scale, swizzled)
    return packed, scales.view(torch.float8_e4m3fn)


@quant_4o6.register_fake
def _(x, global_scale, swizzled):
    m = x.numel() // x.shape[-1]
    k = x.shape[-1]
    packed = x.new_empty((m, k // 2), dtype=torch.uint8)
    if swizzled:
        scales = x.new_empty((_round_up(m, 128), _round_up(k // 16, 4)), dtype=torch.uint8)
    else:
        scales = x.new_empty((m, k // 16), dtype=torch.uint8)
    return packed, scales
