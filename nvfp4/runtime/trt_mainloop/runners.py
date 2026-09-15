# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Donor runners extracted without autotuning; fixed tactics, original forward bodies."""
from __future__ import annotations
import os
from typing import Optional, Tuple, List
import torch
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from .utils import make_ptr
from .activation import ActivationType, is_gated_activation
from .blockscaled_contiguous_gather_grouped_gemm_act_fusion import BlockScaledContiguousGatherGroupedGemmKernel, validate_activation_type
from .blockscaled_contiguous_grouped_gemm_finalize_fusion import Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
from .bf16_contiguous_grouped_gemm import Sm100Bf16ContiguousGroupedGemmKernel

def get_sm_version() -> int:
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor

def _runtime_alpha_ptrs(alpha: torch.Tensor, alpha_numerator: Optional[torch.Tensor], alpha_denominator: Optional[torch.Tensor]):
    """Build the optional runtime-alpha half of the grouped-GEMM ABI.

        The static per-expert ``alpha`` a checkpoint carries was baked with the
        checkpoint's activation scale.  When the activation is re-quantized at
        runtime the epilogue needs ``alpha[expert] * alpha_numerator /
        alpha_denominator`` instead, so both scalars travel with the GEMM and
        are folded in registers rather than by standalone Div/Mul kernels.

        Each is an optional single-element float32 device tensor: pass the
        checkpoint activation scale as ``alpha_numerator`` and the global scale
        the runtime quantizer chose as ``alpha_denominator``.  Absent operands
        contribute ``1.0``; their pointer slot reuses ``alpha`` as an aligned
        placeholder that the compiled kernel never dereferences, and the two
        presence flags are compile-time constants that belong in the kernel
        cache key.

        Returns ``(numerator_ptr, denominator_ptr, has_numerator,
        has_denominator)``.
        """
    ptrs = []
    for scalar in (alpha_numerator, alpha_denominator):
        source = alpha if scalar is None else scalar
        if scalar is not None:
            assert scalar.dtype == torch.float32, f'Runtime alpha scalars must be float32, got {scalar.dtype}'
            assert scalar.numel() == 1, f'Runtime alpha scalars must hold exactly one element, got {scalar.numel()}'
        ptrs.append(make_ptr(cutlass.Float32, source.data_ptr(), cute.AddressSpace.gmem))
    return (ptrs[0], ptrs[1], alpha_numerator is not None, alpha_denominator is not None)

class Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner:
    kernel_class = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel
    kernel_cache = dict()
    tuning_config_cache = dict()
    _instance_cache = dict()

    @classmethod
    def get_cached(cls, device_index: int, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int, tile_size: int, output_dtype: torch.dtype, scaling_vector_size: int, has_correction: bool, has_fused_lora: bool=False) -> 'Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner':
        key = (device_index, num_experts, top_k, num_local_experts, local_expert_offset, tile_size, output_dtype, scaling_vector_size, has_correction, has_fused_lora)
        runner = cls._instance_cache.get(key)
        if runner is None:
            runner = cls(num_experts, top_k, num_local_experts, local_expert_offset, tile_size, output_dtype, scaling_vector_size, has_correction, has_fused_lora)
            cls._instance_cache[key] = runner
        return runner

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int, tile_size: int, output_dtype: torch.dtype, scaling_vector_size: int=16, has_correction: bool=False, has_fused_lora: bool=False):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.tile_size = tile_size
        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size
        self.has_correction = has_correction
        self.has_fused_lora = has_fused_lora
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(f'{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}')
        if self.tile_size not in (128, 256):
            raise ValueError(f'{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}')

    def forward(self, inputs: List[torch.Tensor], tactic: Optional[tuple]) -> torch.Tensor:
        if len(inputs) == 14:
            inputs = [*inputs, None, None]
        a, b, a_sf, b_sf, alpha, c, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, token_final_scales, alpha_numerator, alpha_denominator, correction, lora_down, lora_us = inputs
        assert a.dtype == torch.float4_e2m1fn_x2
        assert a.dim() == 2
        assert b.dtype == torch.float4_e2m1fn_x2
        assert b.dim() == 3
        assert a_sf.dtype == torch.uint8
        assert a_sf.dim() == 1
        assert b_sf.dtype == torch.uint8
        assert b_sf.dim() == 3
        assert alpha.dtype == torch.float32
        assert alpha.dim() == 1
        m, k = (a.size(0), a.size(1) * 2)
        l, n = (b.size(0), b.size(1))
        scale_k = k // self.scaling_vector_size
        assert m % self.tile_size == 0
        assert k % (self.scaling_vector_size * 4) == 0
        assert b.size(2) * 2 == k
        assert a_sf.size(0) == m * scale_k
        assert b_sf.size(0) == l
        assert b_sf.size(1) == n
        assert b_sf.size(2) == scale_k
        assert alpha.size(0) == l
        assert c.dtype == self.output_dtype
        assert c.dim() == 2
        num_tokens = c.size(0)
        assert c.size(1) == n
        num_tiles = m // self.tile_size
        assert tile_idx_to_group_idx.dtype == torch.int32
        assert tile_idx_to_group_idx.size() == (num_tiles,)
        assert tile_idx_to_mn_limit.dtype == torch.int32
        assert tile_idx_to_mn_limit.size() == (num_tiles,)
        assert permuted_idx_to_expanded_idx.dtype == torch.int32
        assert permuted_idx_to_expanded_idx.size() == (m,)
        assert num_non_exiting_tiles.dtype == torch.int32
        assert num_non_exiting_tiles.numel() == 1
        assert token_final_scales.dtype == torch.float32
        assert token_final_scales.dim() == 2
        assert token_final_scales.size() == (num_tokens, self.top_k)
        has_correction = correction is not None
        assert has_correction == self.has_correction
        if has_correction:
            assert correction.dtype == torch.bfloat16
            assert correction.size() == (m, n)
            assert correction.stride(1) == 1
        has_fused_lora = lora_down is not None
        assert has_fused_lora == self.has_fused_lora
        assert (lora_down is None) == (lora_us is None)
        assert not (has_correction and has_fused_lora)
        if has_fused_lora:
            assert lora_down.dtype == torch.bfloat16
            assert lora_down.size() == (m, 64)
            assert lora_down.is_contiguous()
            assert lora_us.dtype == torch.bfloat16
            assert lora_us.size() == (l, n, 64)
            assert lora_us.is_contiguous()
        use_tvm_ffi_fast = os.environ.get('TRTLLM_SVDQUANT_MAIN_TVM_FFI', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        if isinstance(tactic, tuple):
            fast_mma_tiler_mn, fast_cluster_shape_mn, fast_raster_along_m = tactic
        else:
            fast_mma_tiler_mn = (self.tile_size, 128)
            fast_cluster_shape_mn = (self.tile_size // 128, 1)
            fast_raster_along_m = False
        fast_has_alpha_numerator = alpha_numerator is not None
        fast_has_alpha_denominator = alpha_denominator is not None
        fast_cache_key = (self.scaling_vector_size, self.tile_size, fast_mma_tiler_mn, fast_cluster_shape_mn, fast_raster_along_m, fast_has_alpha_numerator, fast_has_alpha_denominator, has_correction, has_fused_lora, 'tvm_ffi')
        fast_compiled_gemm = self.__class__.kernel_cache.get(fast_cache_key) if use_tvm_ffi_fast else None
        if fast_compiled_gemm is not None:
            correction_source = correction if has_correction else c
            lora_down_source = lora_down if has_fused_lora else c
            lora_us_source = lora_us if has_fused_lora else c
            c_stream_anchor_tensor = c.view(num_tokens, n, 1)
            fast_compiled_gemm(a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), c.data_ptr(), alpha.data_ptr(), tile_idx_to_group_idx.data_ptr(), tile_idx_to_mn_limit.data_ptr(), permuted_idx_to_expanded_idx.data_ptr(), num_non_exiting_tiles.data_ptr(), token_final_scales.data_ptr(), alpha_numerator.data_ptr() if fast_has_alpha_numerator else alpha.data_ptr(), alpha_denominator.data_ptr() if fast_has_alpha_denominator else alpha.data_ptr(), correction_source.data_ptr(), lora_down_source.data_ptr(), lora_us_source.data_ptr(), c_stream_anchor_tensor, m, n, k, l, num_tokens, self.top_k)
            return c
        a_ptr = make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        b_ptr = make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
        alpha_numerator_ptr, alpha_denominator_ptr, has_alpha_numerator, has_alpha_denominator = _runtime_alpha_ptrs(alpha, alpha_numerator, alpha_denominator)
        tile_idx_to_group_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem)
        tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem)
        permuted_idx_to_expanded_idx_ptr = make_ptr(cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem)
        num_non_exiting_tiles_ptr = make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem)
        token_final_scales_ptr = make_ptr(cutlass.Float32, token_final_scales.data_ptr(), cute.AddressSpace.gmem)
        c_ptr = make_ptr(cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        correction_ptr = make_ptr(cutlass.BFloat16, correction.data_ptr() if has_correction else c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        lora_down_ptr = make_ptr(cutlass.BFloat16, lora_down.data_ptr() if has_fused_lora else c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        lora_us_ptr = make_ptr(cutlass.BFloat16, lora_us.data_ptr() if has_fused_lora else c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        c_stream_anchor_tensor = c.view(num_tokens, n, 1)
        c_stream_anchor = cute.runtime.from_dlpack(c_stream_anchor_tensor).mark_layout_dynamic(leading_dim=1)
        correction_source = correction if has_correction else c
        lora_down_source = lora_down if has_fused_lora else c
        lora_us_source = lora_us if has_fused_lora else c
        use_tvm_ffi = os.environ.get('TRTLLM_SVDQUANT_MAIN_TVM_FFI', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        if use_tvm_ffi:
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        else:
            torch_stream = torch.cuda.current_stream()
            stream = cuda.CUstream(torch_stream.cuda_stream)
        if isinstance(tactic, tuple):
            mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
        else:
            mma_tiler_mn = (self.tile_size, 128)
            cluster_shape_mn = (self.tile_size // 128, 1)
            raster_along_m = False
        assert mma_tiler_mn[0] == self.tile_size, f'Tactic ({tactic}) is incompatible with tile size ({self.tile_size})'
        cache_key = (self.scaling_vector_size, self.tile_size, mma_tiler_mn, cluster_shape_mn, raster_along_m, has_alpha_numerator, has_alpha_denominator, has_correction)
        cache_key = (*cache_key, has_fused_lora)
        cache_key = (*cache_key, 'tvm_ffi' if use_tvm_ffi else 'cuda_python')
        if cache_key not in self.__class__.kernel_cache:
            gemm = self.__class__.kernel_class(sf_vec_size=self.scaling_vector_size, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, raster_along_m=raster_along_m)
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
            compile_args = [a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr, tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr, token_final_scales_ptr, alpha_numerator_ptr, alpha_denominator_ptr, correction_ptr, lora_down_ptr, lora_us_ptr, c_stream_anchor, m, n, k, l, num_tokens, self.top_k]
            compiled_gemm = cute.compile(gemm.wrapper, *compile_args, tile_size=self.tile_size, scaling_vector_size=self.scaling_vector_size, max_active_clusters=max_active_clusters, stream=stream, has_alpha_numerator=has_alpha_numerator, has_alpha_denominator=has_alpha_denominator, has_correction=has_correction, has_fused_lora=has_fused_lora, options='--opt-level 2 --enable-tvm-ffi' if use_tvm_ffi else '--opt-level 2')
            self.__class__.kernel_cache[cache_key] = compiled_gemm
        else:
            compiled_gemm = self.__class__.kernel_cache[cache_key]
        exec_args = [a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, alpha_ptr, tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr, token_final_scales_ptr, alpha_numerator_ptr, alpha_denominator_ptr, correction_ptr, lora_down_ptr, lora_us_ptr, c_stream_anchor, m, n, k, l, num_tokens, self.top_k]
        if use_tvm_ffi:
            compiled_gemm(a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), c.data_ptr(), alpha.data_ptr(), tile_idx_to_group_idx.data_ptr(), tile_idx_to_mn_limit.data_ptr(), permuted_idx_to_expanded_idx.data_ptr(), num_non_exiting_tiles.data_ptr(), token_final_scales.data_ptr(), alpha_numerator.data_ptr() if has_alpha_numerator else alpha.data_ptr(), alpha_denominator.data_ptr() if has_alpha_denominator else alpha.data_ptr(), correction_source.data_ptr(), lora_down_source.data_ptr(), lora_us_source.data_ptr(), c_stream_anchor_tensor, m, n, k, l, num_tokens, self.top_k)
        else:
            compiled_gemm(*exec_args, stream=stream)
        return c

class Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner:
    kernel_class = BlockScaledContiguousGatherGroupedGemmKernel
    kernel_cache = dict()
    tuning_config_cache = dict()
    amax_workspace_cache = dict()
    max_active_clusters_cache = dict()
    _instance_cache = dict()

    @classmethod
    def get_cached(cls, device_index: int, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int, tile_size: int, scaling_vector_size: int=16, activation_type: ActivationType=ActivationType.Swiglu, output_dtype: torch.dtype=torch.float4_e2m1fn_x2, publish_amax: bool=False, quant_range: float=0.0, has_fused_lora: bool=False) -> 'Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner':
        key = (device_index, num_experts, top_k, num_local_experts, local_expert_offset, tile_size, scaling_vector_size, activation_type, output_dtype, publish_amax, quant_range, has_fused_lora)
        runner = cls._instance_cache.get(key)
        if runner is None:
            runner = cls(num_experts, top_k, num_local_experts, local_expert_offset, tile_size, scaling_vector_size, activation_type, output_dtype, publish_amax, quant_range, has_fused_lora)
            cls._instance_cache[key] = runner
        return runner

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int, local_expert_offset: int, tile_size: int, scaling_vector_size: int=16, activation_type: ActivationType=ActivationType.Swiglu, output_dtype: torch.dtype=torch.float4_e2m1fn_x2, publish_amax: bool=False, quant_range: float=0.0, has_fused_lora: bool=False):
        """Initialize the runner.

            Args:
                activation_type: ``ActivationType`` for the fused epilogue.
                    ``Identity``, ``Swiglu`` (gated), and ``Relu2`` are
                    supported.
            """
        super().__init__()
        self.activation_type = validate_activation_type(activation_type)
        self.is_gated = is_gated_activation(self.activation_type)
        if output_dtype not in (torch.float4_e2m1fn_x2, torch.bfloat16):
            raise ValueError(f'FC13 fused activation output must be NVFP4 or BF16, got {output_dtype}.')
        self.output_dtype = output_dtype
        if publish_amax and output_dtype != torch.bfloat16:
            raise ValueError('amax publication requires the BF16 epilogue')
        if publish_amax and quant_range <= 0.0:
            raise ValueError('amax publication requires a positive quant range')
        self.publish_amax = publish_amax
        self.quant_range = float(quant_range)
        if has_fused_lora and (self.activation_type not in (ActivationType.Identity, ActivationType.Swiglu) or output_dtype != torch.bfloat16):
            raise ValueError('FC13 fused SVDQuant requires Identity or SwiGLU BF16 output')
        self.has_fused_lora = has_fused_lora
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        if tile_size not in [128, 256]:
            raise ValueError(f'Tile size {tile_size} is not supported, it only supports 128 and 256.')
        self.tile_size = tile_size
        self.scaling_vector_size = scaling_vector_size
        if (sm_version := get_sm_version()) not in (100, 103):
            raise ValueError(f'{self.__class__.kernel_class.__name__} supports SM 100 (B200) and SM 103 (B300) only, but got SM {sm_version}')
        if self.tile_size not in (128, 256):
            raise ValueError(f'{self.__class__.kernel_class.__name__} supports tile_size (MMA tile M dimension) 128 and 256 only, but got {self.tile_size}')

    @classmethod
    def _get_max_active_clusters(cls, device: torch.device, cluster_shape_mn: Tuple[int, int]) -> int:
        """Return the occupancy result without a per-forward driver query."""
        cache_key = (device, cluster_shape_mn)
        max_active_clusters = cls.max_active_clusters_cache.get(cache_key)
        if max_active_clusters is None:
            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(cluster_shape_mn[0] * cluster_shape_mn[1])
            cls.max_active_clusters_cache[cache_key] = max_active_clusters
        return max_active_clusters

    def forward(self, inputs: List, tactic: Optional[tuple]) -> torch.Tensor:
        """Forward pass.

            Input layout:
                0: a                               - tensor
                1: b                               - tensor
                2: a_sf                            - tensor
                3: b_sf                            - tensor
                4: alpha                           - tensor
                5: tile_idx_to_group_idx           - tensor
                6: tile_idx_to_mn_limit            - tensor
                7: permuted_idx_to_expanded_idx    - tensor
                8: num_non_exiting_tiles           - tensor
                9: global_sf                       - tensor
               10: alpha_numerator                 - tensor or None
               11: alpha_denominator               - tensor or None
            """
        if len(inputs) == 12:
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf, alpha_numerator, alpha_denominator = inputs
            lora_down = None
            lora_us_lo = None
            lora_us_hi = None
        else:
            assert len(inputs) == 15
            a, b, a_sf, b_sf, alpha, tile_idx_to_group_idx, tile_idx_to_mn_limit, permuted_idx_to_expanded_idx, num_non_exiting_tiles, global_sf, alpha_numerator, alpha_denominator, lora_down, lora_us_lo, lora_us_hi = inputs
        assert a.dtype == torch.float4_e2m1fn_x2
        assert a.dim() == 2
        assert b.dtype == torch.float4_e2m1fn_x2
        assert b.dim() == 3
        assert a_sf.dtype == torch.uint8
        assert a_sf.dim() == 2
        assert b_sf.dtype == torch.uint8
        assert b_sf.dim() == 3
        assert alpha.dtype == torch.float32
        assert alpha.dim() == 1
        has_fused_lora = lora_down is not None
        assert has_fused_lora == self.has_fused_lora
        orig_m, k = (a.size(0), a.size(1) * 2)
        m = permuted_idx_to_expanded_idx.size(0)
        l, n = (b.size(0), b.size(1))
        scale_k = k // self.scaling_vector_size
        interm_size = n // 2 if self.is_gated else n
        assert m % self.tile_size == 0
        assert k % (self.scaling_vector_size * 4) == 0
        if self.is_gated:
            assert n % (self.scaling_vector_size * 4 * 2) == 0
        else:
            assert n % (self.scaling_vector_size * 4) == 0
        assert b.size(2) * 2 == k
        assert a_sf.size(0) == orig_m
        assert a_sf.size(1) == scale_k
        num_tiles = m // self.tile_size
        assert tile_idx_to_group_idx.dtype == torch.int32
        assert tile_idx_to_group_idx.size() == (num_tiles,)
        assert tile_idx_to_mn_limit.dtype == torch.int32
        assert tile_idx_to_mn_limit.size() == (num_tiles,)
        assert permuted_idx_to_expanded_idx.dtype == torch.int32
        assert permuted_idx_to_expanded_idx.size() == (m,)
        assert num_non_exiting_tiles.dtype == torch.int32
        assert num_non_exiting_tiles.numel() == 1
        assert global_sf.dtype == torch.float32
        assert global_sf.numel() == 1
        if has_fused_lora:
            assert lora_down.dtype == torch.bfloat16
            assert lora_down.size() == (m, 128)
            assert lora_down.is_contiguous()
            assert lora_us_lo.dtype == torch.bfloat16
            assert lora_us_hi.dtype == torch.bfloat16
            lora_us_n = n if self.is_gated else n // 2
            assert lora_us_lo.size() == (l, lora_us_n, 64)
            assert lora_us_hi.size() == (l, lora_us_n, 64)
            assert lora_us_lo.is_contiguous()
            assert lora_us_hi.is_contiguous()
        packed_output = self.output_dtype == torch.float4_e2m1fn_x2
        c = torch.empty(m, interm_size // 2 if packed_output else interm_size, dtype=self.output_dtype, device=a.device)
        c_sf_numel = m * interm_size // self.scaling_vector_size if packed_output else 1
        c_sf = torch.empty(c_sf_numel, dtype=a_sf.dtype, device=a_sf.device)
        use_tvm_ffi_fast = os.environ.get('TRTLLM_SVDQUANT_MAIN_TVM_FFI', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        if isinstance(tactic, tuple):
            fast_mma_tiler_mn, fast_cluster_shape_mn, fast_raster_along_m = tactic
        else:
            fast_mma_tiler_mn = (self.tile_size, 128)
            fast_cluster_shape_mn = (self.tile_size // 128, 1)
            fast_raster_along_m = False
        fast_has_alpha_numerator = alpha_numerator is not None
        fast_has_alpha_denominator = alpha_denominator is not None
        fast_cache_key = (self.scaling_vector_size, self.tile_size, self.top_k, fast_mma_tiler_mn, fast_cluster_shape_mn, fast_raster_along_m, self.activation_type, self.output_dtype, fast_has_alpha_numerator, fast_has_alpha_denominator, self.publish_amax, self.quant_range, has_fused_lora, 'tvm_ffi')
        fast_compiled_gemm = self.__class__.kernel_cache.get(fast_cache_key) if use_tvm_ffi_fast else None
        if fast_compiled_gemm is not None:
            torch_stream = torch.cuda.current_stream()
            active_clusters_key = (a.device, fast_cluster_shape_mn)
            max_active_clusters = self.__class__.max_active_clusters_cache.get(active_clusters_key)
            if self.publish_amax:
                if max_active_clusters is None:
                    max_active_clusters = self._get_max_active_clusters(a.device, fast_cluster_shape_mn)
                max_grid_ctas = max_active_clusters * fast_cluster_shape_mn[0] * fast_cluster_shape_mn[1]
                workspace_key = (a.device, int(torch_stream.cuda_stream), max_grid_ctas)
                workspace = self.__class__.amax_workspace_cache.get(workspace_key)
                if workspace is None:
                    fast_compiled_gemm = None
                else:
                    block_max, retirement_count, amax_scale = workspace
            else:
                block_max = alpha
                retirement_count = num_non_exiting_tiles
                amax_scale = alpha
            if fast_compiled_gemm is not None:
                c_stream_anchor_tensor = c.view(m, -1, 1)
                placeholder_address = c.data_ptr()
                lora_down_lo_address = lora_down[:, :64].data_ptr() if has_fused_lora else placeholder_address
                lora_down_hi_address = lora_down[:, 64:].data_ptr() if has_fused_lora else placeholder_address
                lora_us_lo_address = lora_us_lo.data_ptr() if has_fused_lora else placeholder_address
                lora_us_hi_address = lora_us_hi.data_ptr() if has_fused_lora else placeholder_address
                fast_compiled_gemm(a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), c.data_ptr(), c_sf.data_ptr(), alpha.data_ptr(), tile_idx_to_group_idx.data_ptr(), tile_idx_to_mn_limit.data_ptr(), permuted_idx_to_expanded_idx.data_ptr(), num_non_exiting_tiles.data_ptr(), global_sf.data_ptr(), alpha_numerator.data_ptr() if fast_has_alpha_numerator else alpha.data_ptr(), alpha_denominator.data_ptr() if fast_has_alpha_denominator else alpha.data_ptr(), block_max.data_ptr(), retirement_count.data_ptr(), amax_scale.data_ptr(), lora_down_lo_address, lora_down_hi_address, lora_us_lo_address, lora_us_hi_address, c_stream_anchor_tensor, orig_m, m, n, k, l)
                if self.publish_amax:
                    return (c, c_sf, amax_scale)
                return (c, c_sf)
        a_ptr = make_ptr(cutlass.Float4E2M1FN, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        b_ptr = make_ptr(cutlass.Float4E2M1FN, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        a_sf_ptr = make_ptr(cutlass.Float8E4M3FN, a_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        b_sf_ptr = make_ptr(cutlass.Float8E4M3FN, b_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        alpha_ptr = make_ptr(cutlass.Float32, alpha.data_ptr(), cute.AddressSpace.gmem)
        alpha_numerator_ptr, alpha_denominator_ptr, has_alpha_numerator, has_alpha_denominator = _runtime_alpha_ptrs(alpha, alpha_numerator, alpha_denominator)
        c_ptr = make_ptr(cutlass.Float4E2M1FN if packed_output else cutlass.BFloat16, c.data_ptr(), cute.AddressSpace.gmem, assumed_align=32)
        c_stream_anchor_tensor = c.view(m, -1, 1)
        c_stream_anchor = cute.runtime.from_dlpack(c_stream_anchor_tensor).mark_layout_dynamic(leading_dim=1)
        c_sf_ptr = make_ptr(cutlass.Float8E4M3FN, c_sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        tile_idx_to_group_idx_ptr = make_ptr(cutlass.Int32, tile_idx_to_group_idx.data_ptr(), cute.AddressSpace.gmem)
        tile_idx_to_mn_limit_ptr = make_ptr(cutlass.Int32, tile_idx_to_mn_limit.data_ptr(), cute.AddressSpace.gmem)
        permuted_idx_to_expanded_idx_ptr = make_ptr(cutlass.Int32, permuted_idx_to_expanded_idx.data_ptr(), cute.AddressSpace.gmem)
        num_non_exiting_tiles_ptr = make_ptr(cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem)
        global_sf_ptr = make_ptr(cutlass.Float32, global_sf.data_ptr(), cute.AddressSpace.gmem)
        placeholder_ptr = c.data_ptr()
        lora_down_lo_ptr = make_ptr(cutlass.BFloat16, lora_down[:, :64].data_ptr() if has_fused_lora else placeholder_ptr, cute.AddressSpace.gmem, assumed_align=16)
        lora_down_hi_ptr = make_ptr(cutlass.BFloat16, lora_down[:, 64:].data_ptr() if has_fused_lora else placeholder_ptr, cute.AddressSpace.gmem, assumed_align=16)
        lora_us_lo_ptr = make_ptr(cutlass.BFloat16, lora_us_lo.data_ptr() if has_fused_lora else placeholder_ptr, cute.AddressSpace.gmem, assumed_align=16)
        lora_us_hi_ptr = make_ptr(cutlass.BFloat16, lora_us_hi.data_ptr() if has_fused_lora else placeholder_ptr, cute.AddressSpace.gmem, assumed_align=16)
        use_tvm_ffi = os.environ.get('TRTLLM_SVDQUANT_MAIN_TVM_FFI', '1').strip().lower() not in ('0', 'false', 'no', 'off')
        torch_stream = torch.cuda.current_stream()
        if use_tvm_ffi:
            stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        else:
            stream = cuda.CUstream(torch_stream.cuda_stream)
        if isinstance(tactic, tuple):
            mma_tiler_mn, cluster_shape_mn, raster_along_m = tactic
        else:
            mma_tiler_mn = (self.tile_size, 128)
            cluster_shape_mn = (self.tile_size // 128, 1)
            raster_along_m = False
        assert mma_tiler_mn[0] == self.tile_size, f'Tactic ({tactic}) is incompatible with tile size ({self.tile_size})'
        active_clusters_key = (a.device, cluster_shape_mn)
        max_active_clusters = self.__class__.max_active_clusters_cache.get(active_clusters_key)
        if self.publish_amax:
            if max_active_clusters is None:
                max_active_clusters = self._get_max_active_clusters(a.device, cluster_shape_mn)
            max_grid_ctas = max_active_clusters * cluster_shape_mn[0] * cluster_shape_mn[1]
            workspace_key = (a.device, int(torch_stream.cuda_stream), max_grid_ctas)
            workspace = self.__class__.amax_workspace_cache.get(workspace_key)
            if workspace is None:
                workspace = (torch.empty(max_grid_ctas * 128, dtype=torch.float32, device=a.device), torch.zeros(1, dtype=torch.int32, device=a.device), torch.empty(2, dtype=torch.float32, device=a.device))
                self.__class__.amax_workspace_cache[workspace_key] = workspace
            block_max, retirement_count, amax_scale = workspace
        else:
            block_max = alpha
            retirement_count = num_non_exiting_tiles
            amax_scale = alpha
        block_max_ptr = make_ptr(cutlass.Float32, block_max.data_ptr(), cute.AddressSpace.gmem)
        retirement_count_ptr = make_ptr(cutlass.Int32, retirement_count.data_ptr(), cute.AddressSpace.gmem)
        amax_scale_ptr = make_ptr(cutlass.Float32, amax_scale.data_ptr(), cute.AddressSpace.gmem)
        cache_key = (self.scaling_vector_size, self.tile_size, self.top_k, mma_tiler_mn, cluster_shape_mn, raster_along_m, self.activation_type, self.output_dtype, has_alpha_numerator, has_alpha_denominator, self.publish_amax, self.quant_range, has_fused_lora, 'tvm_ffi' if use_tvm_ffi else 'cuda_python')
        if cache_key not in self.__class__.kernel_cache:
            if max_active_clusters is None:
                max_active_clusters = self._get_max_active_clusters(a.device, cluster_shape_mn)
            gemm = self.__class__.kernel_class(sf_vec_size=self.scaling_vector_size, mma_tiler_mn=mma_tiler_mn, cluster_shape_mn=cluster_shape_mn, vectorized_f32=True, topk=self.top_k, raster_along_m=raster_along_m, activation_type=self.activation_type)
            compile_args = [a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr, tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr, global_sf_ptr, alpha_numerator_ptr, alpha_denominator_ptr, block_max_ptr, retirement_count_ptr, amax_scale_ptr, lora_down_lo_ptr, lora_down_hi_ptr, lora_us_lo_ptr, lora_us_hi_ptr, c_stream_anchor, orig_m, m, n, k, l]
            compiled_gemm = cute.compile(gemm.wrapper, *compile_args, tile_size=self.tile_size, scaling_vector_size=self.scaling_vector_size, max_active_clusters=max_active_clusters, stream=stream, activation_type=self.activation_type, has_alpha_numerator=has_alpha_numerator, has_alpha_denominator=has_alpha_denominator, publish_amax=self.publish_amax, quant_range=self.quant_range, has_fused_lora=has_fused_lora, options='--opt-level 2 --enable-tvm-ffi' if use_tvm_ffi else '--opt-level 2')
            self.__class__.kernel_cache[cache_key] = compiled_gemm
        else:
            compiled_gemm = self.__class__.kernel_cache[cache_key]
        exec_args = [a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr, c_sf_ptr, alpha_ptr, tile_idx_to_group_idx_ptr, tile_idx_to_mn_limit_ptr, permuted_idx_to_expanded_idx_ptr, num_non_exiting_tiles_ptr, global_sf_ptr, alpha_numerator_ptr, alpha_denominator_ptr, block_max_ptr, retirement_count_ptr, amax_scale_ptr, lora_down_lo_ptr, lora_down_hi_ptr, lora_us_lo_ptr, lora_us_hi_ptr, c_stream_anchor, orig_m, m, n, k, l]
        if use_tvm_ffi:
            lora_down_lo_address = lora_down[:, :64].data_ptr() if has_fused_lora else c.data_ptr()
            lora_down_hi_address = lora_down[:, 64:].data_ptr() if has_fused_lora else c.data_ptr()
            compiled_gemm(a.data_ptr(), b.data_ptr(), a_sf.data_ptr(), b_sf.data_ptr(), c.data_ptr(), c_sf.data_ptr(), alpha.data_ptr(), tile_idx_to_group_idx.data_ptr(), tile_idx_to_mn_limit.data_ptr(), permuted_idx_to_expanded_idx.data_ptr(), num_non_exiting_tiles.data_ptr(), global_sf.data_ptr(), alpha_numerator.data_ptr() if has_alpha_numerator else alpha.data_ptr(), alpha_denominator.data_ptr() if has_alpha_denominator else alpha.data_ptr(), block_max.data_ptr(), retirement_count.data_ptr(), amax_scale.data_ptr(), lora_down_lo_address, lora_down_hi_address, lora_us_lo.data_ptr() if has_fused_lora else c.data_ptr(), lora_us_hi.data_ptr() if has_fused_lora else c.data_ptr(), c_stream_anchor_tensor, orig_m, m, n, k, l)
        else:
            compiled_gemm(*exec_args, stream=stream)
        if self.publish_amax:
            assert amax_scale is not None
            return (c, c_sf, amax_scale)
        return (c, c_sf)

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
