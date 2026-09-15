# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unmodified TRT-LLM grouped GEMM runner with standalone import bindings."""

import os
from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch

from .bf16_contiguous_grouped_gemm import Sm100Bf16ContiguousGroupedGemmKernel
from .utils import make_ptr


def get_sm_version() -> int:
    """Provide the runner's device query without importing TRT-LLM runtime."""
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


class Sm100Bf16ContiguousGroupedGemmRunner:
    """Runner for the BF16 grouped GEMM over the moe_sort permuted layout.

        Deliberately not a ``TunableRunner``: the tactic is chosen on the host
        from the problem shape instead of through ``AutoTuner``.  The autotuner
        profiles candidate tactics with fresh allocations, which is what breaks
        the SVDQuant path under CUDA graph capture, and the launch geometry here
        is already fixed by the padded row count, so there is little left to
        tune.  Compiled kernels are cached on the class, so a shape that has run
        once replays without touching the JIT.
        """
    kernel_class = Sm100Bf16ContiguousGroupedGemmKernel
    kernel_cache = dict()
    SUPPORTED_TILE_SIZES = (64, 128)
    _instance_cache = dict()

    @classmethod
    def get_cached(cls, tile_size: int, device_index: int) -> 'Sm100Bf16ContiguousGroupedGemmRunner':
        """Return a shared runner for ``(device_index, tile_size)``.

            Keyed by device even though the runner holds no device state, so a
            multi-device process can never reuse an instance whose validation
            ran against another device's SM version.
            """
        key = (device_index, tile_size)
        runner = cls._instance_cache.get(key)
        if runner is None:
            runner = cls(tile_size)
            cls._instance_cache[key] = runner
        return runner

    @classmethod
    def select_tactic(cls, n: int, tile_size: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Pick ``(mma_tiler_mn, cluster_shape_mn)`` for one problem.

            Uses the largest N tile that divides N so the epilogue never stores
            a partial tile; falls back to 128 and relies on TMA out-of-bounds
            handling otherwise.  The rank-64 SVDQuant stages land on 64 (the
            down projection) and 256 (the up projection).
            """
        if n <= 64:
            n_tile = 64
        elif n <= 128:
            n_tile = 128
        else:
            n_tile = next((candidate for candidate in (256, 128, 64) if n % candidate == 0), 128)
        return ((tile_size, n_tile), (1, 1))

    def __init__(self, tile_size: int):
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(f'{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}')
        if tile_size not in self.SUPPORTED_TILE_SIZES:
            raise ValueError(f'{self.__class__.kernel_class.__name__} needs one MMA tile per routing tile, so tile_size must be one of {self.SUPPORTED_TILE_SIZES}; got {tile_size}')
        self.tile_size = tile_size
        self.sm_version = sm_version

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, tile_idx_to_group_idx: torch.Tensor, num_non_exiting_tiles: Optional[torch.Tensor], accumulate: bool=False, tile_idx_to_mn_limit: Optional[torch.Tensor]=None, permuted_idx_to_expanded_idx: Optional[torch.Tensor]=None, top_k: int=1, correction_alpha: Optional[torch.Tensor]=None, correction_alpha_numerator: Optional[torch.Tensor]=None, correction_alpha_denominator: Optional[torch.Tensor]=None) -> None:
        """Run ``c[m] = a[m] @ b[group(m)].T`` in place.

            ``accumulate`` is a per-call argument rather than constructor state
            so one cached runner can drive both stages of a low-rank pair, where
            only the second one adds onto an existing buffer.

            Args:
                a: [M, K] bf16 activation. Must be K-contiguous; the M stride
                    may exceed K so a column slice of a wider tensor works.
                b: [L, N, K] bf16 weights, contiguous.
                c: [M, N] bf16 output. Must be N-contiguous; the M stride may
                    exceed N so a column slice of a wider tensor works. Written
                    in place.
                tile_idx_to_group_idx: [M // tile_size] int32 tile-to-expert map.
                num_non_exiting_tiles: optional [1] int32 live tile count.
                accumulate: when true the epilogue adds onto ``c`` instead of
                    overwriting it, and tiles skipped by the valid-tile gate
                    leave their rows of ``c`` untouched.
            """
        input_m, k = a.shape
        m = c.shape[0]
        num_groups, n = (b.shape[0], b.shape[1])
        if m == 0:
            return
        mma_tiler_mn, cluster_shape_mn = self.select_tactic(n, self.tile_size)
        has_valid_tile_gate = num_non_exiting_tiles is not None
        gate_source = num_non_exiting_tiles if has_valid_tile_gate else tile_idx_to_group_idx
        gather = permuted_idx_to_expanded_idx is not None
        assert gather == (tile_idx_to_mn_limit is not None), 'CuTe DSL bf16 gather needs both routing maps'
        mn_limit_source = tile_idx_to_mn_limit if gather else tile_idx_to_group_idx
        permuted_source = permuted_idx_to_expanded_idx if gather else tile_idx_to_group_idx
        has_correction_alpha = correction_alpha is not None
        has_correction_runtime_alpha = correction_alpha_numerator is not None
        if has_correction_runtime_alpha != (correction_alpha_denominator is not None):
            raise ValueError('correction alpha numerator and denominator must be provided together')
        if has_correction_runtime_alpha and (not has_correction_alpha):
            raise ValueError('runtime correction alpha scalars require per-expert alpha')
        if has_correction_alpha:
            if correction_alpha.dtype != torch.float32:
                raise ValueError('correction alpha must be float32')
            if correction_alpha.numel() != num_groups:
                raise ValueError('correction alpha must contain one value per expert')
        if has_correction_runtime_alpha:
            if correction_alpha_numerator.dtype != torch.float32 or correction_alpha_denominator.dtype != torch.float32:
                raise ValueError('runtime correction alpha scalars must be float32')
            if correction_alpha_numerator.numel() != 1 or correction_alpha_denominator.numel() != 1:
                raise ValueError('runtime correction alpha scalars must be scalar tensors')
        alpha_source = correction_alpha if has_correction_alpha else c
        numerator_source = correction_alpha_numerator if has_correction_runtime_alpha else alpha_source
        denominator_source = correction_alpha_denominator if has_correction_runtime_alpha else alpha_source
        a_stride_m = a.stride(0)
        c_stride_m = c.stride(0)
        c_anchor = c.view(1, m, n).permute(1, 2, 0)
        use_tvm_ffi = os.environ.get('TRTLLM_SVDQUANT_BF16_TVM_FFI', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        device_index = torch.cuda.current_device()
        cache_key = (device_index, self.sm_version, self.tile_size, mma_tiler_mn, cluster_shape_mn, has_valid_tile_gate, accumulate, gather, top_k, has_correction_alpha, has_correction_runtime_alpha, 'tvm_ffi' if use_tvm_ffi else 'cuda_python')
        if not use_tvm_ffi:
            a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            c_ptr = make_ptr(cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
            tile_idx_to_group_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem)
            num_non_exiting_tiles_ptr = make_ptr(cutlass.Int32, gate_source.data_ptr(), cute.AddressSpace.gmem)
            tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32, mn_limit_source.data_ptr(), cute.AddressSpace.gmem)
            permuted_idx_to_expanded_idx_ptr = make_ptr(cutlass.Int32, permuted_source.data_ptr(), cute.AddressSpace.gmem)
            correction_alpha_ptr = make_ptr(cutlass.Float32, alpha_source.data_ptr(), cute.AddressSpace.gmem)
            correction_alpha_numerator_ptr = make_ptr(cutlass.Float32, numerator_source.data_ptr(), cute.AddressSpace.gmem)
            correction_alpha_denominator_ptr = make_ptr(cutlass.Float32, denominator_source.data_ptr(), cute.AddressSpace.gmem)
            stream_anchor = cute.runtime.from_dlpack(c_anchor).mark_layout_dynamic(leading_dim=1)
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if cache_key not in self.__class__.kernel_cache:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f'CuteDSL BF16 grouped GEMM must be compiled before CUDA graph capture; run one warm-up call per distinct shape bucket first (missing tactic {cache_key}).')
            if not self.__class__.kernel_class.can_implement(ab_dtype=cutlass.BFloat16, acc_dtype=cutlass.Float32, c_dtype=cutlass.BFloat16, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, tile_size=self.tile_size, m=m, n=n, k=k, num_groups=num_groups):
                raise ValueError(f'{self.__class__.kernel_class.__name__} cannot implement m={m}, n={n}, k={k}, num_groups={num_groups}, tile_size={self.tile_size} with tactic {(mma_tiler_mn, cluster_shape_mn)}.')
            if use_tvm_ffi:
                a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                c_ptr = make_ptr(cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
                tile_idx_to_group_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem)
                num_non_exiting_tiles_ptr = make_ptr(cutlass.Int32, gate_source.data_ptr(), cute.AddressSpace.gmem)
                tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32, mn_limit_source.data_ptr(), cute.AddressSpace.gmem)
                permuted_idx_to_expanded_idx_ptr = make_ptr(cutlass.Int32, permuted_source.data_ptr(), cute.AddressSpace.gmem)
                correction_alpha_ptr = make_ptr(cutlass.Float32, alpha_source.data_ptr(), cute.AddressSpace.gmem)
                correction_alpha_numerator_ptr = make_ptr(cutlass.Float32, numerator_source.data_ptr(), cute.AddressSpace.gmem)
                correction_alpha_denominator_ptr = make_ptr(cutlass.Float32, denominator_source.data_ptr(), cute.AddressSpace.gmem)
                stream_anchor = cute.runtime.from_dlpack(c_anchor).mark_layout_dynamic(leading_dim=1)
                stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            gemm = self.__class__.kernel_class(cutlass.Float32, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, tile_size=self.tile_size, accumulate=accumulate, gather=gather, top_k=top_k)
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
            self.__class__.kernel_cache[cache_key] = cute.compile(gemm.wrapper, a_ptr, b_ptr, c_ptr, tile_idx_to_group_idx_ptr, num_non_exiting_tiles_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, correction_alpha_ptr, correction_alpha_numerator_ptr, correction_alpha_denominator_ptr, stream_anchor, m, input_m, n, k, num_groups, a_stride_m, c_stride_m, tile_size=self.tile_size, has_valid_tile_gate=has_valid_tile_gate, has_correction_alpha=has_correction_alpha, has_correction_runtime_alpha=has_correction_runtime_alpha, max_active_clusters=max_active_clusters, stream=stream, options='--opt-level 2 --enable-tvm-ffi' if use_tvm_ffi else '--opt-level 2')
        compiled_gemm = self.__class__.kernel_cache[cache_key]
        if use_tvm_ffi:
            compiled_gemm(a.data_ptr(), b.data_ptr(), c.data_ptr(), tile_idx_to_group_idx.data_ptr(), gate_source.data_ptr(), mn_limit_source.data_ptr(), permuted_source.data_ptr(), alpha_source.data_ptr(), numerator_source.data_ptr(), denominator_source.data_ptr(), c_anchor, m, input_m, n, k, num_groups, a_stride_m, c_stride_m)
        else:
            compiled_gemm(a_ptr, b_ptr, c_ptr, tile_idx_to_group_idx_ptr, num_non_exiting_tiles_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, correction_alpha_ptr, correction_alpha_numerator_ptr, correction_alpha_denominator_ptr, stream_anchor, m, input_m, n, k, num_groups, a_stride_m, c_stride_m, stream=stream)
