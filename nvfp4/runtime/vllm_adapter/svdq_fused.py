# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense SVDQ correction fused into the NVFP4 GEMM on SM120 (flashinfer PR #5228 kernel).

The PR module computes ``out = alpha * (a @ bᵀ + d @ l1ᵀ)`` in one CUTLASS kernel, accumulating the rank-r
BF16 LoRA-up in the FP4 GEMM's FP32 accumulator. The reference adapter instead adds
``(x @ vhᵀ) @ usᵀ`` to the BF16 GEMM output with two extra GEMMs and an elementwise add. With ``l1 = us``
and ``d = (x @ vhᵀ) / alpha`` the PR kernel yields the same quantity with one BF16 rounding at the end, so
results are close to but not bit-identical with the reference.

The module is built out of process (scripts/svdq_build.py) and loaded here through tvm_ffi; its path comes
from ``VLLM_NVFP4_SVDQ_SO``. Tactics come from the table written by scripts/svdq_tune.py
(``VLLM_NVFP4_SVDQ_TACTICS``): per (n, k) the best tactic per M bucket; M uses the smallest bucket >= M.
"""

from __future__ import annotations

import bisect
import json
import os

import torch

_module = None
_tactics: dict[tuple[int, int], tuple[list[int], list[int]]] | None = None
_workspace: dict[torch.device, torch.Tensor] = {}


def module():
    global _module
    if _module is None:
        import tvm_ffi

        _module = tvm_ffi.load_module(os.environ["VLLM_NVFP4_SVDQ_SO"])
    return _module


def _tactic_table() -> dict[tuple[int, int], tuple[list[int], list[int]]]:
    global _tactics
    if _tactics is None:
        _tactics = {}
        path = os.environ.get("VLLM_NVFP4_SVDQ_TACTICS")
        if path:
            for key, buckets in json.load(open(path)).items():
                n, k = (int(v) for v in key.split("x"))
                ms = sorted(int(m) for m in buckets)
                _tactics[(n, k)] = (ms, [int(buckets[str(m)]) for m in ms])
    return _tactics


_pdl_early_safe: bool | None = None


def pdl_early_safe() -> bool:
    """Whether the loaded module declares that a preceding kernel may trigger PDL early (every read of data the
    preceding kernel writes happens after griddepcontrol.wait, in all warp roles)."""
    global _pdl_early_safe
    if _pdl_early_safe is None:
        if not os.environ.get("VLLM_NVFP4_SVDQ_SO"):
            _pdl_early_safe = False  # no fused GEMM module configured: nothing to launch early
        else:
            try:
                _pdl_early_safe = bool(module().nvfp4_svdquant_gemm_pdl_early_safe())
            except AttributeError:
                _pdl_early_safe = False
    return _pdl_early_safe


def tactic_for(m: int, n: int, k: int, rank: int) -> int:
    entry = _tactic_table().get((n, k))
    if entry is not None:
        ms, tactics = entry
        return tactics[min(bisect.bisect_left(ms, m), len(ms) - 1)]
    return int(module().nvfp4_svdquant_gemm_fallback_tactic(m, n, k, rank))


def workspace(device: torch.device, nbytes: int) -> torch.Tensor:
    buf = _workspace.get(device)
    if buf is None or buf.numel() < nbytes:
        # 64 MiB covers every tuned tactic for Qwen3-8B shapes up to M=8192 (checked by svdq_tune.py)
        buf = torch.empty(max(nbytes, 64 << 20), dtype=torch.uint8, device=device)
        _workspace[device] = buf
    return buf


@torch.library.custom_op("nvfp4_opt::svdq_gemm", mutates_args=())
def svdq_gemm(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, alpha: torch.Tensor,
              d: torch.Tensor, l1: torch.Tensor) -> torch.Tensor:
    """``alpha * (a @ bᵀ + d @ l1ᵀ)`` in BF16; a/b packed e2m1 uint8, a_sf/b_sf swizzled UE4M3 uint8."""
    m, n, k = a.shape[0], b.shape[0], a.shape[1] * 2
    rank = d.shape[1]
    out = torch.empty((m, n), dtype=torch.bfloat16, device=a.device)
    tactic = tactic_for(m, n, k, rank)
    mod = module()
    ws = workspace(a.device, int(mod.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)))
    mod.nvfp4_svdquant_gemm(a, b, a_sf.reshape(-1), b_sf.reshape(-1), alpha.reshape(1), d, l1, None, out, ws,
                            tactic, True)
    return out


@svdq_gemm.register_fake
def _(a, b, a_sf, b_sf, alpha, d, l1):
    return a.new_empty((a.shape[0], b.shape[0]), dtype=torch.bfloat16)


@torch.library.custom_op("nvfp4_opt::lora_down_scaled", mutates_args=())
def lora_down_scaled(x: torch.Tensor, vh: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """``(x @ vhᵀ) / alpha`` in BF16. Decode-sized inputs take one fused launch (FP32 sums, one rounding;
    csrc/lora_down.cu), larger ones cuBLAS + one division kernel (the same two BF16 roundings as the traced v2
    path, where inductor fused the division)."""
    from .fused_dynamic import dyn_quant_extension

    ext = dyn_quant_extension()
    x = x.contiguous()
    if ext.lora_down_supported(x.shape[0], x.shape[1]):
        return ext.lora_down_scaled(x, vh, alpha)
    d = x @ vh.T
    ext.div_by_alpha_(d, alpha)
    return d


@lora_down_scaled.register_fake
def _(x, vh, alpha):
    return x.new_empty((x.shape[0], vh.shape[0]), dtype=torch.bfloat16)


_lora_scratch_bufs: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _lora_scratch(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-device scratch of dyn_quant_lora's cooperative path: per-block maxima and LoRA partial sums
    (k / 256 slices x m <= 64 rows x rank 64; 1 MiB covers K <= 16384)."""
    bufs = _lora_scratch_bufs.get(device)
    if bufs is None:
        bufs = (torch.zeros(2048, dtype=torch.float32, device=device),
                torch.zeros(64 * 64 * 64, dtype=torch.float32, device=device))
        _lora_scratch_bufs[device] = bufs
    return bufs


@torch.library.custom_op("nvfp4_opt::dyn_quant_lora", mutates_args=("inv", "alpha"))
def dyn_quant_lora(x: torch.Tensor, inv: torch.Tensor, alpha: torch.Tensor, weight_global_scale: torch.Tensor,
                   quant_range: float, vh: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """4o6 ``dyn_quant`` plus ``d = bf16((x @ vhᵀ) / alpha)``. Decode-sized inputs do both in one launch
    (csrc/dyn_quant.cu: the LoRA blocks derive alpha from the same amax with the same expression); otherwise
    ``dyn_quant`` followed by ``lora_down_scaled``."""
    from .fused_dynamic import TRITON_REFERENCE_MAX_NUMEL, _reciprocal, dyn_quant_extension

    ext = dyn_quant_extension()
    matrix = x.reshape(-1, x.shape[-1]).contiguous()
    if vh.shape[0] == 64 and ext.dyn_quant_lora_supported(matrix.shape[0], matrix.shape[1]):
        n = matrix.numel()
        scratch, partials = _lora_scratch(matrix.device)
        q, s, d = ext.dyn_quant_lora(matrix, inv, alpha, weight_global_scale,
                                     _reciprocal(quant_range, 0 < n <= TRITON_REFERENCE_MAX_NUMEL), True, vh,
                                     scratch, partials, pdl_early_safe())
        return q, s, d
    q, s = torch.ops.nvfp4_opt.dyn_quant(matrix, inv, alpha, weight_global_scale, quant_range, True)
    return q, s, torch.ops.nvfp4_opt.lora_down_scaled(matrix, vh, alpha)


@dyn_quant_lora.register_fake
def _(x, inv, alpha, weight_global_scale, quant_range, vh):
    m, k = x.numel() // x.shape[-1], x.shape[-1]
    return (x.new_empty((m, k // 2), dtype=torch.uint8),
            x.new_empty(((m + 127) // 128 * 128, (k // 16 + 3) // 4 * 4), dtype=torch.uint8),
            x.new_empty((m, vh.shape[0]), dtype=torch.bfloat16))


def svdq_linear_4o6(layer: torch.nn.Module, x: torch.Tensor, quant_range: float) -> torch.Tensor:
    """Fused dynamic 4o6 quantization + LoRA-down, then the fused NVFP4/SVDQ GEMM, for one dense layer
    (no bias, no padding)."""
    matrix = x.reshape(-1, x.shape[-1])
    xq, x_sf, d = torch.ops.nvfp4_opt.dyn_quant_lora(
        matrix, layer.input_global_scale_inv, layer.alpha, layer.weight_global_scale, quant_range, layer.svdquant_vh
    )
    out = torch.ops.nvfp4_opt.svdq_gemm(
        xq, layer.weight, x_sf, layer.weight_scale.view(torch.uint8), layer.alpha, d, layer.svdquant_us,
    )
    return out.view(*x.shape[:-1], layer.output_size_per_partition)
