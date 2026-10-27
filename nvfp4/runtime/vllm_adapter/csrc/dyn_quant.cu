// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Dynamic per-tensor NVFP4 activation quantization in one launch (SM120, BF16 input).
//
//   phase 1  |x| max: small inputs, every block reduces the whole input itself (no inter-block dependency; the
//            same launch can carry the SVDQ LoRA-down blocks, dyn_quant_lora); larger inputs, a cooperative grid
//            whose blocks publish per-block maxima, cross one grid barrier, and each reduce all of them
//   phase 2  deq = max(amax, 1e-12) * rcp_q ; inv = 1 / deq (round-to-nearest) ; alpha = deq * wgs
//   phase 3  per 16-element block, one of two recipes, scales in the 128x4 swizzled layout, padding zeroed:
//            standard  vLLM scaled_fp4_quant / TRT-LLM cvt_warp_fp16_to_fp4 (UE4M3 scale, rcp.approx.ftz,
//                      cvt.rn.satfinite.e2m1x2)
//            4o6       FlashInfer CuTe-DSL 4over6 (fast-math scales, strict MSE between the /6 and /4 scale)
//
// inv / alpha / packed values / scales are bit-identical to the reference adapter's dynamic_global_scale +
// scaled_fp4_quant (standard) or + quantize_dense_4o6 (4o6) when the caller passes the FP32 rcp_q the reference
// used. The per-block maxima are rewritten by every launch, so launches replay inside CUDA graphs.
// Only c10 CUDA headers: ATen/cuda/CUDAContext.h needs cusparse.h, which the vLLM image does not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "lora_row.cuh"

namespace cg = cooperative_groups;

namespace {

constexpr int kThreads = 256;
// Inputs up to this many 16-byte chunks (32 Ki BF16 values: decode batches up to 8 rows at K=4096, 2 rows at
// K=12288) take the redundant-amax kernel: every block reduces the whole input itself with all of its 16-byte
// loads in flight (8 per thread up to 2048 chunks, else 16; 16 costs ~0.2 µs more on the smallest inputs), so no
// block waits for another. Above that, the cooperative grid. results/bench-dyn-quant-v4b.log / -v4c.log
constexpr int64_t kRedundantMaxChunks = 16 * kThreads;
static_assert(kThreads == lora::kThreads, "LoRA blocks share the quantization launch");
// Up to this many chunks the cooperative grid is capped at kSmallBlocksPerSm blocks per SM: fewer, longer-lived
// blocks beat a full-occupancy grid there (M=128, K=12288: 7.9 -> 5.4 µs; M=1024, K=4096: 10.6 -> 8.7 µs).
constexpr int64_t kSmallGridChunks = int64_t{1} << 19;
constexpr int kSmallBlocksPerSm = 2;

__device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
    return b;
}

// 8 floats -> 8 e2m1 nibbles, element 2i in the low nibble of byte i (same operand order as TRT-LLM / vLLM).
__device__ __forceinline__ uint32_t to_e2m1x8(const float2 (&v)[4]) {
    uint32_t out;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
        "mov.b32 %0, {b0, b1, b2, b3};\n"
        "}\n"
        : "=r"(out)
        : "f"(v[0].x), "f"(v[0].y), "f"(v[1].x), "f"(v[1].y), "f"(v[2].x), "f"(v[2].y), "f"(v[3].x), "f"(v[3].y));
    return out;
}

// [mTile, kTile, 32 (outer m), 4 (inner m), 4 (inner k)] swizzle of the UE4M3 scale matrix.
__device__ __forceinline__ int64_t sf_offset(int64_t m, int64_t kb, int64_t num_kb) {
    const int64_t num_k_tiles = (num_kb + 3) / 4;
    return (m / 128) * num_k_tiles * 512 + (kb / 4) * 512 + (m % 32) * 16 + ((m % 128) / 32) * 4 + (kb % 4);
}

// |x| max over this thread's grid-strided chunks (one chunk = 16 bytes = 8 BF16)
__device__ __forceinline__ float chunk_amax(const uint4* __restrict__ xv, int64_t first, int64_t chunks,
                                            int64_t stride) {
    float local = 0.f;
    for (int64_t i = first; i < chunks; i += stride) {
        const uint4 u = xv[i];
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __bfloat1622float2(h[j]);
            local = fmaxf(local, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    return local;
}

// Block-wide max of non-negative values, valid in every thread: each warp reduces the per-warp maxima itself
// with shuffles instead of one thread looping over them and broadcasting through a second barrier. The caller
// must separate two calls with a block barrier (warp_max is reused).
template <int THREADS>
__device__ __forceinline__ float block_max(float local) {
    __shared__ float warp_max[THREADS / 32];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = local;
    __syncthreads();
    float b = (threadIdx.x & 31) < THREADS / 32 ? warp_max[threadIdx.x & 31] : 0.f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) b = fmaxf(b, __shfl_xor_sync(0xffffffffu, b, off));
    return b;
}

// Zero the scale-matrix padding with 16-byte stores: the last, partially filled 128-row tile is one
// contiguous range of the swizzled layout (the whole tensor if K/16 is not a multiple of 4). Valid
// entries inside that range are written afterwards, behind a block or grid barrier.
__device__ __forceinline__ void zero_sf_padding(uint8_t* __restrict__ sf_out, int64_t m, int64_t num_kb,
                                                int64_t m_padded, int64_t first, int64_t stride) {
    const int64_t tile_bytes = (num_kb + 3) / 4 * 512;
    const bool cols_padded = num_kb % 4 != 0;
    const int64_t begin = cols_padded ? 0 : m / 128 * tile_bytes;
    const int64_t end = (!cols_padded && m % 128 == 0) ? begin : m_padded / 128 * tile_bytes;
    uint4* p = reinterpret_cast<uint4*>(sf_out + begin);
    for (int64_t i = first; i < (end - begin) / 16; i += stride) p[i] = make_uint4(0u, 0u, 0u, 0u);
}

// deq = max(amax, 1e-12) * rcp_q ; inv = 1/deq ; the given thread also publishes inv and alpha.
__device__ __forceinline__ float finalize_scale(float amax, float rcp_q, const float* __restrict__ wgs,
                                                float* __restrict__ inv_out, float* __restrict__ alpha_out,
                                                bool publish) {
    const float deq = fmaxf(amax, 1.0e-12f) * rcp_q;
    const float sf_scale = __fdiv_rn(1.0f, deq);
    if (publish) {
        *inv_out = sf_scale;
        *alpha_out = deq * (*wgs);
    }
    return sf_scale;
}

// One 16-byte chunk (8 values) of the standard recipe. The even and odd chunk of a 16-element block sit in
// adjacent lanes of `mask`; `store` false computes (keeping the pair's shuffles convergent) without writing.
__device__ __forceinline__ void quantize_chunk_std(const uint4& u, int64_t i, int64_t chunks_per_row, int64_t num_kb,
                                                   float sf_scale, float rcp_sf_scale, unsigned mask, bool store,
                                                   uint32_t* __restrict__ fp4_out, uint8_t* __restrict__ sf_out) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
    __nv_bfloat162 lmax = __habs2(h[0]);
#pragma unroll
    for (int j = 1; j < 4; ++j) lmax = __hmax2(lmax, __habs2(h[j]));
    lmax = __hmax2(__shfl_xor_sync(mask, lmax, 1), lmax);
    const float vec_max = __bfloat162float(__hmax(lmax.x, lmax.y));

    float sf_value = sf_scale * (vec_max * rcp_approx_ftz(6.0f));
    const __nv_fp8_e4m3 sf8 = __nv_fp8_e4m3(sf_value);
    sf_value = static_cast<float>(sf8);
    const float out_scale = vec_max != 0.f ? rcp_approx_ftz(sf_value * rcp_sf_scale) : 0.0f;

    float2 v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = __bfloat1622float2(h[j]);
        v[j].x *= out_scale;
        v[j].y *= out_scale;
    }
    const uint32_t packed = to_e2m1x8(v);
    if (store) {
        fp4_out[i] = packed;
        if ((i & 1) == 0) {
            const int64_t row = i / chunks_per_row;
            sf_out[sf_offset(row, (i % chunks_per_row) / 2, num_kb)] = sf8.__x;
        }
    }
}

// The per-16-element recipe of vLLM's scaled_fp4_quant. Chunks come in (even, odd) lane pairs that
// share one scale block; strides are even so the active lanes of a warp are an even-length prefix.
__device__ __forceinline__ void quantize_chunks(const uint4* __restrict__ xv, int64_t chunks, int64_t k,
                                                float sf_scale, uint32_t* __restrict__ fp4_out,
                                                uint8_t* __restrict__ sf_out, int64_t first, int64_t stride) {
    const int64_t num_kb = k / 16;
    const int64_t chunks_per_row = k / 8;
    const float rcp_sf_scale = rcp_approx_ftz(sf_scale);
    for (int64_t i = first; i < chunks; i += stride) {
        quantize_chunk_std(xv[i], i, chunks_per_row, num_kb, sf_scale, rcp_sf_scale, __activemask(), true, fp4_out,
                           sf_out);
    }
}

// 8 e2m1 codes (as packed by to_e2m1x8) -> 8 exact float values, element order kept; the inverse conversion
// is exact, so this equals the e2m1 magnitude table with the sign applied.
__device__ __forceinline__ void from_e2m1x8(uint32_t packed, float2 (&v)[4]) {
    uint32_t h[4];
    asm("{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        "mov.b32 {b0, b1, b2, b3}, %4;\n"
        "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
        "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
        "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
        "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
        "}\n"
        : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
        : "r"(packed));
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = __half22float2(*reinterpret_cast<const __half2*>(&h[j]));
}

// Squared errors of 8 elements against their dequantized codes, dq = (code * scale) * gds rounded per step.
__device__ __forceinline__ void squared_errors8(const float2 (&f)[4], uint32_t packed, float scale, float gds,
                                                float (&sq)[8]) {
    float2 q[4];
    from_e2m1x8(packed, q);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float dx = __fsub_rn(__fmul_rn(__fmul_rn(q[j].x, scale), gds), f[j].x);
        const float dy = __fsub_rn(__fmul_rn(__fmul_rn(q[j].y, scale), gds), f[j].y);
        sq[2 * j] = __fmul_rn(dx, dx);
        sq[2 * j + 1] = __fmul_rn(dy, dy);
    }
}

// Strict in-order sum of the 16 squared errors of a block split over a lane pair (even lane: elements 0-7,
// odd lane: 8-15), returned to both lanes. Both lanes run the same instructions: each sums its own 8 terms
// once from 0 (the even lane's result is the partial sum) and once from the partner's first sum (the odd
// lane's result is the total). Every lane of `mask` must call this.
__device__ __forceinline__ float pair_ordered_sum16(unsigned mask, bool odd, const float (&sq)[8]) {
    float partial = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) partial = __fadd_rn(partial, sq[j]);
    float total = __shfl_xor_sync(mask, partial, 1);
#pragma unroll
    for (int j = 0; j < 8; ++j) total = __fadd_rn(total, sq[j]);
    const float other = __shfl_xor_sync(mask, total, 1);
    return odd ? total : other;
}

// 4o6 (FlashInfer CuTe-DSL _nvfp4_4over6_quant_from_values with fast-math scales, strict MSE and row_amax = 0,
// the dense kernels' value). Lane pairs share a 16-element block as in quantize_chunks; /4 wins only if its
// error is strictly smaller. All lanes run the candidate math so the shuffles stay convergent; all-zero
// blocks are written as 0 afterwards.
__device__ __forceinline__ void quantize_chunk_4o6(const uint4& u, int64_t i, int64_t chunks_per_row, int64_t num_kb,
                                                   float global_scale, float gds, unsigned mask, bool store,
                                                   uint32_t* __restrict__ fp4_out, uint8_t* __restrict__ sf_out) {
    const bool odd = (i & 1) != 0;
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&u);
    __nv_bfloat162 lmax = __habs2(h[0]);
#pragma unroll
    for (int j = 1; j < 4; ++j) lmax = __hmax2(lmax, __habs2(h[j]));
    lmax = __hmax2(__shfl_xor_sync(mask, lmax, 1), lmax);
    const float block_max = __bfloat162float(__hmax(lmax.x, lmax.y));
    float2 f[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) f[j] = __bfloat1622float2(h[j]);

    const float sf6_hp = global_scale * (block_max * rcp_approx_ftz(6.0f));
    const float sf4_hp = sf6_hp * 1.5f;
    const __nv_fp8_e4m3 s4 = __nv_fp8_e4m3(sf4_hp);
    const __nv_fp8_e4m3 s6 = __nv_fp8_e4m3(sf6_hp);
    const float scale4 = static_cast<float>(s4);
    const float scale6 = static_cast<float>(s6);
    const float os4 = rcp_approx_ftz(scale4 * gds);
    const float os6 = rcp_approx_ftz(scale6 * gds);
    float2 a[4], c[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        a[j] = make_float2(f[j].x * os4, f[j].y * os4);
        c[j] = make_float2(f[j].x * os6, f[j].y * os6);
    }
    const uint32_t p4 = to_e2m1x8(a);
    const uint32_t p6 = to_e2m1x8(c);
    float sq[8];
    squared_errors8(f, p4, scale4, gds, sq);
    const float err4 = pair_ordered_sum16(mask, odd, sq);
    squared_errors8(f, p6, scale6, gds, sq);
    const float err6 = pair_ordered_sum16(mask, odd, sq);
    const bool use4 = err4 < err6;
    uint32_t packed = use4 ? p4 : p6;
    uint8_t sf_byte = use4 ? s4.__x : s6.__x;
    if (block_max == 0.f) {
        packed = 0u;
        sf_byte = 0;
    }
    if (store) {
        fp4_out[i] = packed;
        if (!odd) {
            const int64_t row = i / chunks_per_row;
            sf_out[sf_offset(row, (i % chunks_per_row) / 2, num_kb)] = sf_byte;
        }
    }
}

__device__ __forceinline__ void quantize_chunks_4o6(const uint4* __restrict__ xv, int64_t chunks, int64_t k,
                                                    float global_scale, uint32_t* __restrict__ fp4_out,
                                                    uint8_t* __restrict__ sf_out, int64_t first, int64_t stride) {
    const int64_t num_kb = k / 16;
    const int64_t chunks_per_row = k / 8;
    const float gds = rcp_approx_ftz(global_scale);
    for (int64_t i = first; i < chunks; i += stride) {
        quantize_chunk_4o6(xv[i], i, chunks_per_row, num_kb, global_scale, gds, __activemask(), true, fp4_out, sf_out);
    }
}

template <bool FOUR_OVER_SIX>
__device__ __forceinline__ void quantize(const uint4* __restrict__ xv, int64_t chunks, int64_t k, float sf_scale,
                                         uint32_t* __restrict__ fp4_out, uint8_t* __restrict__ sf_out,
                                         int64_t first, int64_t stride) {
    if constexpr (FOUR_OVER_SIX) {
        quantize_chunks_4o6(xv, chunks, k, sf_scale, fp4_out, sf_out, first, stride);
    } else {
        quantize_chunks(xv, chunks, k, sf_scale, fp4_out, sf_out, first, stride);
    }
}

// Scale-padding bytes of the swizzled layout, zeroed with 4-byte stores (one row, kb % 4 = 0..3 per word) by
// first/stride. Only bytes that no valid (row < m, kb < num_kb) scale occupies are written, so blocks that zero
// padding and blocks that write scales touch disjoint bytes and need no ordering.
__device__ __forceinline__ void zero_sf_padding_disjoint(uint8_t* __restrict__ sf_out, int64_t m, int64_t num_kb,
                                                         int64_t m_padded, int64_t first, int64_t stride) {
    const int64_t tile_bytes = (num_kb + 3) / 4 * 512;
    if (num_kb % 4 == 0) {
        // Only rows of the last, partial 128-row tile are padding. 16-byte vector v of the tile holds rows
        // j, j + 32, j + 64, j + 96 (j = v % 32, one word each; see sf_offset), so the test needs no division.
        if (m % 128 == 0) return;
        const int m_rel = static_cast<int>(m % 128);
        uint4* tile = reinterpret_cast<uint4*>(sf_out + m / 128 * tile_bytes);
        for (int64_t v = first; v < tile_bytes / 16; v += stride) {
            const int j = static_cast<int>(v & 31);
            if (j >= m_rel) {
                tile[v] = make_uint4(0u, 0u, 0u, 0u);
            } else {
                uint32_t* w = reinterpret_cast<uint32_t*>(&tile[v]);
                for (int t = 1; t < 4; ++t) {
                    if (t * 32 + j >= m_rel) w[t] = 0u;
                }
            }
        }
        return;
    }
    // Padded columns: every tile has padding bytes; recover (row, kb) per word.
    const int64_t end = m_padded / 128 * tile_bytes;
    uint32_t* words = reinterpret_cast<uint32_t*>(sf_out);
    for (int64_t w = first; w < end / 4; w += stride) {
        const int64_t off = w * 4, r = off % 512;
        const int64_t row = off / tile_bytes * 128 + (r % 16) / 4 * 32 + r / 16;
        const int64_t kb0 = off % tile_bytes / 512 * 4;
        if (row >= m) {
            words[w] = 0u;
        } else if (kb0 + 3 >= num_kb) {
            for (int b = 0; b < 4; ++b) {
                if (kb0 + b >= num_kb) sf_out[off + b] = 0;
            }
        }
    }
}

// Small inputs (<= kRedundantMaxChunks chunks), one launch, no inter-block dependency: every block computes the
// global |x| max itself, then blocks [0, quant_blocks) quantize kThreads chunks each and zero their share of the
// scale padding, and (LORA_M > 0) blocks [quant_blocks, quant_blocks + rank) each compute one SVDQ LoRA-down row
// with the same alpha that finalize_scale publishes (identical expression, so identical bits).
template <bool FOUR_OVER_SIX, int LORA_M, int LOADS>
__global__ void __launch_bounds__(kThreads) dyn_quant_redundant_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t m, int64_t k, float rcp_q, const float* __restrict__ wgs,
    float* __restrict__ inv_out, float* __restrict__ alpha_out, uint32_t* __restrict__ fp4_out,
    uint8_t* __restrict__ sf_out, int64_t m_padded, int quant_blocks, const __nv_bfloat16* __restrict__ vh,
    __nv_bfloat16* __restrict__ d_out, int rank, bool pdl_trigger) {
    // PDL: let the dependent GEMM launch now (pdl_trigger) so it can stream its weights while this kernel runs.
    // Only for a GEMM that reads nothing this kernel writes before griddepcontrol.wait in any warp role; CUTLASS
    // epilogues read *alpha_ptr when constructed, so FlashInfer 0.6.14's FP4 GEMM is not such a GEMM.
    if (pdl_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const int64_t chunks = m * k / 8;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    uint4 stage[LOADS];
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
        const int64_t i = threadIdx.x + j * kThreads;
        stage[j] = i < chunks ? xv[i] : make_uint4(0u, 0u, 0u, 0u);
    }
    float local = 0.f;
#pragma unroll
    for (int j = 0; j < LOADS; ++j) {
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&stage[j]);
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float2 f = __bfloat1622float2(h[q]);
            local = fmaxf(local, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
    }
    const float amax = block_max<kThreads>(local);
    if (static_cast<int>(blockIdx.x) < quant_blocks) {
        const int64_t first = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
        zero_sf_padding_disjoint(sf_out, m, k / 16, m_padded, first, static_cast<int64_t>(quant_blocks) * kThreads);
        const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, first == 0);
        const int64_t end = std::min<int64_t>(chunks, static_cast<int64_t>(blockIdx.x + 1) * kThreads);
        quantize<FOUR_OVER_SIX>(xv, end, k, sf_scale, fp4_out, sf_out, first, kThreads);
    } else if constexpr (LORA_M > 0) {
        const float deq = fmaxf(amax, 1.0e-12f) * rcp_q;
        const float alpha = deq * (*wgs);
        lora::row<LORA_M>(x, vh, alpha, d_out, static_cast<int>(m), static_cast<int>(k), rank,
                          static_cast<int>(blockIdx.x) - quant_blocks);
    }
}

// Large inputs: cooperative grid. Each block stores its maximum in its own scratch slot; after the one grid
// barrier every block reduces all slots. No same-address atomics besides the barrier's own, and no reset: the
// slots are fully rewritten by the next launch, which stream order keeps behind this one.
template <bool FOUR_OVER_SIX>
__global__ void __launch_bounds__(kThreads) dyn_quant_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t m, int64_t k, float rcp_q, const float* __restrict__ wgs,
    float* __restrict__ inv_out, float* __restrict__ alpha_out, uint32_t* __restrict__ fp4_out,
    uint8_t* __restrict__ sf_out, int64_t m_padded, float* __restrict__ block_maxima, bool pdl_trigger) {
    // PDL: let the dependent GEMM launch now (pdl_trigger) so it can stream its weights while this kernel runs.
    // Only for a GEMM that reads nothing this kernel writes before griddepcontrol.wait in any warp role; CUTLASS
    // epilogues read *alpha_ptr when constructed, so FlashInfer 0.6.14's FP4 GEMM is not such a GEMM.
    if (pdl_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    cg::grid_group grid = cg::this_grid();
    const int64_t chunks = m * k / 8;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const uint4* xv = reinterpret_cast<const uint4*>(x);

    zero_sf_padding(sf_out, m, k / 16, m_padded, first, stride);
    const float b = block_max<kThreads>(chunk_amax(xv, first, chunks, stride));
    if (threadIdx.x == 0) block_maxima[blockIdx.x] = b;
    grid.sync();

    float partial = 0.f;
    for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += kThreads) partial = fmaxf(partial, __ldcg(&block_maxima[i]));
    const float amax = block_max<kThreads>(partial);
    const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, blockIdx.x == 0 && threadIdx.x == 0);
    quantize<FOUR_OVER_SIX>(xv, chunks, k, sf_scale, fp4_out, sf_out, first, stride);
}

// Mid-size inputs with the SVDQ LoRA-down (rank 64, M <= kCoopLoraMaxM, K a multiple of 256): one cooperative
// launch replaces dyn_quant + cuBLAS (x @ vhᵀ) + the division. Block (g, s) owns rows [16g, 16g + 16) and columns
// [256s, 256s + 256): it stages that x tile (taking |x| max on the way) and vh[:, 256s : 256s + 256] in shared memory
// and writes the partial sums P[s][row][r] over its columns with tensor-core MMAs. After the grid barrier, one thread per output sums
// d[row][r] = bf16(Σ_s P[s][row][r] / alpha) in slice order (deterministic), then the grid quantizes exactly as
// dyn_quant_kernel. alpha is the expression finalize_scale publishes, so the GEMM's alpha and d's agree bit for bit.
constexpr int kCoopRows = 16;
constexpr int kCoopSliceK = 256;
constexpr int kCoopRank = 64;
constexpr int kCoopLoraMaxM = 64;
constexpr int kCoopPitch = kCoopSliceK + 8;  // bf16 per shared row; the extra 16 B keeps 16-byte loads conflict-free
constexpr int kCoopSmemBytes = (kCoopRows + kCoopRank) * kCoopPitch * 2;
static_assert(kThreads / 32 * 8 == kCoopRank && kCoopRows == 16, "warp w: mma tile rows 0..15 x ranks 8w..8w+7");

template <bool FOUR_OVER_SIX>
__global__ void __launch_bounds__(kThreads) dyn_quant_lora_coop_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t m, int64_t k, float rcp_q, const float* __restrict__ wgs,
    float* __restrict__ inv_out, float* __restrict__ alpha_out, uint32_t* __restrict__ fp4_out,
    uint8_t* __restrict__ sf_out, int64_t m_padded, float* __restrict__ block_maxima,
    const __nv_bfloat16* __restrict__ vh, __nv_bfloat16* __restrict__ d_out, float* __restrict__ partials,
    bool pdl_trigger) {
    if (pdl_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // see dyn_quant_kernel
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) unsigned char smem[kCoopSmemBytes];
    auto xs = reinterpret_cast<__nv_bfloat16 (*)[kCoopPitch]>(smem);
    auto ws = reinterpret_cast<__nv_bfloat16 (*)[kCoopPitch]>(smem + kCoopRows * kCoopPitch * 2);
    constexpr int kSliceChunks = kCoopSliceK / 8;

    const int64_t chunks = m * k / 8;
    const int64_t cpr = k / 8;
    const int slices = static_cast<int>(k / kCoopSliceK);
    const int s = static_cast<int>(blockIdx.x) % slices;
    const int64_t row0 = static_cast<int64_t>(blockIdx.x / slices) * kCoopRows;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* wv = reinterpret_cast<const uint4*>(vh);

    zero_sf_padding(sf_out, m, k / 16, m_padded, first, stride);
    // All staging loads are issued before any shared store: the trip counts are compile-time, so the loads of
    // one thread are in flight together instead of each waiting out a DRAM round trip.
    constexpr int kXLoads = kCoopRows * kSliceChunks / kThreads;   // 2
    constexpr int kWLoads = kCoopRank * kSliceChunks / kThreads;   // 8
    static_assert(kCoopRows * kSliceChunks % kThreads == 0 && kCoopRank * kSliceChunks % kThreads == 0);
    uint4 xr[kXLoads], wr[kWLoads];
#pragma unroll
    for (int j = 0; j < kWLoads; ++j) {
        const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
        wr[j] = wv[static_cast<int64_t>(rr) * cpr + static_cast<int64_t>(s) * kSliceChunks + cc];
    }
#pragma unroll
    for (int j = 0; j < kXLoads; ++j) {
        const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
        xr[j] = row0 + rr < m ? xv[(row0 + rr) * cpr + static_cast<int64_t>(s) * kSliceChunks + cc]
                              : make_uint4(0u, 0u, 0u, 0u);
    }
    float local = 0.f;
#pragma unroll
    for (int j = 0; j < kXLoads; ++j) {
        const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
        const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xr[j]);
#pragma unroll
        for (int q2 = 0; q2 < 4; ++q2) {
            const float2 f = __bfloat1622float2(h[q2]);
            local = fmaxf(local, fmaxf(fabsf(f.x), fabsf(f.y)));
        }
        *reinterpret_cast<uint4*>(&xs[rr][cc * 8]) = xr[j];
    }
#pragma unroll
    for (int j = 0; j < kWLoads; ++j) {
        const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
        *reinterpret_cast<uint4*>(&ws[rr][cc * 8]) = wr[j];
    }
    const float b = block_max<kThreads>(local);  // its barrier also publishes the staged tiles
    if (threadIdx.x == 0) block_maxima[blockIdx.x] = b;

    // LoRA partial on tensor cores: warp w computes rows [0, 16) x ranks [8w, 8w + 8) over the 256 staged
    // columns with 16 mma.m16n8k16 (BF16 in, FP32 accumulate; fixed hardware order, so deterministic). A = x tile
    // (row-major), B = vh slice stored [rank][k], i.e. the column-major operand mma.row.col expects.
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float c[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int k0 = 0; k0 < kCoopSliceK; k0 += 16) {
        uint32_t a[4], bb[2];
        const uint32_t a_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&xs[lane % 16][k0 + (lane / 16) * 8]));
        const uint32_t b_addr = static_cast<uint32_t>(
            __cvta_generic_to_shared(&ws[warp * 8 + lane % 8][k0 + ((lane / 8) % 2) * 8]));
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                     : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                     : "=r"(bb[0]), "=r"(bb[1]) : "r"(b_addr));
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, "
                     "{%8, %9}, {%0, %1, %2, %3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
    }
    // accumulator fragment: c0/c1 at row lane/4, c2/c3 at row lane/4 + 8; columns (lane % 4) * 2 + {0, 1}
    const int col = warp * 8 + (lane % 4) * 2;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int64_t row = row0 + lane / 4 + h * 8;
        if (row < m) {
            float2* dst = reinterpret_cast<float2*>(&partials[(static_cast<int64_t>(s) * m + row) * kCoopRank + col]);
            *dst = make_float2(c[2 * h], c[2 * h + 1]);
        }
    }
    grid.sync();

    float partial = 0.f;
    for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += kThreads) partial = fmaxf(partial, __ldcg(&block_maxima[i]));
    const float amax = block_max<kThreads>(partial);
    const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, blockIdx.x == 0 && threadIdx.x == 0);
    const float deq = fmaxf(amax, 1.0e-12f) * rcp_q;
    const float alpha = deq * (*wgs);
    for (int64_t o = first; o < m * kCoopRank; o += stride) {
        // load all slices' partials first (up to 64), then add them in slice order
        float pv[64];
#pragma unroll
        for (int ss = 0; ss < 64; ++ss) pv[ss] = ss < slices ? __ldcg(&partials[static_cast<int64_t>(ss) * m * kCoopRank + o]) : 0.f;
        float sum = 0.f;
#pragma unroll
        for (int ss = 0; ss < 64; ++ss) if (ss < slices) sum += pv[ss];
        d_out[o] = __float2bfloat16_rn(__fdiv_rn(sum, alpha));
    }
    quantize<FOUR_OVER_SIX>(xv, chunks, k, sf_scale, fp4_out, sf_out, first, stride);
}

// Large inputs with the SVDQ LoRA-down (rank 64, K a multiple of 128): one cooperative launch replaces
// dyn_quant + cuBLAS (x @ vhᵀ) + the division, reading x twice instead of three times. Pass 1 takes |x| max over a
// grid-stride sweep (as dyn_quant_kernel). After the grid barrier each block walks TILE_M-row tiles; per tile it
// streams x and vh through shared memory in 128-column slices: each staged x chunk is quantized from registers
// (quantize_chunk_std / _4o6, the exact recipes of quantize()) and the slice feeds tensor-core MMAs that accumulate
// the LoRA-down over all of K in registers (no cross-block reduction; fixed order, deterministic). Finally
// d = bf16(acc / alpha), alpha being the expression finalize_scale publishes.
constexpr int kTileK = 128;
constexpr int kTilePitch = kTileK + 8;  // bf16; the extra 16 B keeps ldmatrix rows conflict-free

template <bool FOUR_OVER_SIX, int TILE_M>
__global__ void __launch_bounds__(kThreads) dyn_quant_lora_tile_kernel(
    const __nv_bfloat16* __restrict__ x, int64_t m, int64_t k, float rcp_q, const float* __restrict__ wgs,
    float* __restrict__ inv_out, float* __restrict__ alpha_out, uint32_t* __restrict__ fp4_out,
    uint8_t* __restrict__ sf_out, int64_t m_padded, float* __restrict__ block_maxima,
    const __nv_bfloat16* __restrict__ vh, __nv_bfloat16* __restrict__ d_out, bool pdl_trigger) {
    if (pdl_trigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");  // see dyn_quant_kernel
    constexpr int MT = TILE_M / 16;                                  // m16 tiles; each warp owns MT n8 tiles
    constexpr int kSliceChunks = kTileK / 8;                          // 16
    constexpr int kXLoads = TILE_M * kSliceChunks / kThreads;         // 1 or 4
    constexpr int kWLoads = kCoopRank * kSliceChunks / kThreads;      // 4
    static_assert(TILE_M * kSliceChunks % kThreads == 0 && MT * 8 <= 8 * MT && 8 % MT == 0);
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) __nv_bfloat16 xs[TILE_M][kTilePitch];
    __shared__ __align__(16) __nv_bfloat16 ws[kCoopRank][kTilePitch];

    const int64_t chunks = m * k / 8;
    const int64_t cpr = k / 8;
    const int64_t num_kb = k / 16;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
    const int64_t first = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    const uint4* wv = reinterpret_cast<const uint4*>(vh);

    zero_sf_padding(sf_out, m, num_kb, m_padded, first, stride);
    const float b = block_max<kThreads>(chunk_amax(xv, first, chunks, stride));
    if (threadIdx.x == 0) block_maxima[blockIdx.x] = b;
    grid.sync();

    float partial = 0.f;
    for (int i = threadIdx.x; i < static_cast<int>(gridDim.x); i += kThreads) partial = fmaxf(partial, __ldcg(&block_maxima[i]));
    const float amax = block_max<kThreads>(partial);
    const float sf_scale = finalize_scale(amax, rcp_q, wgs, inv_out, alpha_out, blockIdx.x == 0 && threadIdx.x == 0);
    const float deq = fmaxf(amax, 1.0e-12f) * rcp_q;
    const float alpha = deq * (*wgs);
    const float rcp_sf_scale = rcp_approx_ftz(sf_scale);  // standard recipe
    const float gds = rcp_approx_ftz(sf_scale);           // 4o6 recipe (global decode scale)

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int mt = warp % MT, nt0 = (warp / MT) * MT;
    const int64_t tiles = (m + TILE_M - 1) / TILE_M;
    const int slices = static_cast<int>(k / kTileK);
    for (int64_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int64_t row0 = tile * TILE_M;
        float c[MT][4];
#pragma unroll
        for (int t = 0; t < MT; ++t) c[t][0] = c[t][1] = c[t][2] = c[t][3] = 0.f;
        for (int sl = 0; sl < slices; ++sl) {
            uint4 xr[kXLoads], wr[kWLoads];
#pragma unroll
            for (int j = 0; j < kWLoads; ++j) {
                const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
                wr[j] = wv[static_cast<int64_t>(rr) * cpr + static_cast<int64_t>(sl) * kSliceChunks + cc];
            }
#pragma unroll
            for (int j = 0; j < kXLoads; ++j) {
                const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
                xr[j] = row0 + rr < m ? xv[(row0 + rr) * cpr + static_cast<int64_t>(sl) * kSliceChunks + cc]
                                      : make_uint4(0u, 0u, 0u, 0u);
            }
            // quantize the staged chunks; chunk i and i + 1 of a row (one 16-element block) sit in adjacent lanes
#pragma unroll
            for (int j = 0; j < kXLoads; ++j) {
                const int i = threadIdx.x + j * kThreads, rr = i / kSliceChunks, cc = i % kSliceChunks;
                const int64_t gi = (row0 + rr) * cpr + static_cast<int64_t>(sl) * kSliceChunks + cc;
                if constexpr (FOUR_OVER_SIX) {
                    quantize_chunk_4o6(xr[j], gi, cpr, num_kb, sf_scale, gds, 0xffffffffu, row0 + rr < m, fp4_out, sf_out);
                } else {
                    quantize_chunk_std(xr[j], gi, cpr, num_kb, sf_scale, rcp_sf_scale, 0xffffffffu, row0 + rr < m,
                                       fp4_out, sf_out);
                }
            }
            __syncthreads();  // the previous slice's MMAs are done reading xs / ws
#pragma unroll
            for (int j = 0; j < kXLoads; ++j) {
                const int i = threadIdx.x + j * kThreads;
                *reinterpret_cast<uint4*>(&xs[i / kSliceChunks][(i % kSliceChunks) * 8]) = xr[j];
            }
#pragma unroll
            for (int j = 0; j < kWLoads; ++j) {
                const int i = threadIdx.x + j * kThreads;
                *reinterpret_cast<uint4*>(&ws[i / kSliceChunks][(i % kSliceChunks) * 8]) = wr[j];
            }
            __syncthreads();
#pragma unroll
            for (int k0 = 0; k0 < kTileK; k0 += 16) {
                uint32_t a[4];
                const uint32_t a_addr = static_cast<uint32_t>(
                    __cvta_generic_to_shared(&xs[mt * 16 + lane % 16][k0 + (lane / 16) * 8]));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                             : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(a_addr));
#pragma unroll
                for (int t = 0; t < MT; ++t) {
                    uint32_t bb[2];
                    const uint32_t b_addr = static_cast<uint32_t>(
                        __cvta_generic_to_shared(&ws[(nt0 + t) * 8 + lane % 8][k0 + ((lane / 8) % 2) * 8]));
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                 : "=r"(bb[0]), "=r"(bb[1]) : "r"(b_addr));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, "
                                 "{%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                 : "+f"(c[t][0]), "+f"(c[t][1]), "+f"(c[t][2]), "+f"(c[t][3])
                                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]));
                }
            }
        }
#pragma unroll
        for (int t = 0; t < MT; ++t) {
            const int col = (nt0 + t) * 8 + (lane % 4) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int64_t row = row0 + mt * 16 + lane / 4 + h * 8;
                if (row < m) {
                    *reinterpret_cast<__nv_bfloat162*>(&d_out[row * kCoopRank + col]) =
                        __floats2bfloat162_rn(__fdiv_rn(c[t][2 * h], alpha), __fdiv_rn(c[t][2 * h + 1], alpha));
                }
            }
        }
        __syncthreads();  // the next tile's first slice overwrites xs / ws
    }
}

int g_grid_limit[2][64] = {{0}};
int g_sms[64] = {0};

}  // namespace

namespace {

struct Launch {
    at::Tensor fp4, sf;
    int64_t m, k, m_padded, chunks;
    const __nv_bfloat16* xp;
    float rq;
    const float* wp;
    float* ip;
    float* ap;
    uint32_t* fp;
    uint8_t* sp;
    cudaStream_t stream;
    bool pdl_trigger;
};

Launch prepare(const at::Tensor& x, at::Tensor& inv, at::Tensor& alpha, const at::Tensor& wgs, double rcp_q,
               bool pdl_trigger) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous(),
                "x must be a contiguous 2-D CUDA BF16 tensor");
    TORCH_CHECK(x.size(1) % 16 == 0, "K must be a multiple of 16");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
    TORCH_CHECK(inv.scalar_type() == at::kFloat && alpha.scalar_type() == at::kFloat && wgs.scalar_type() == at::kFloat);
    Launch l;
    l.m = x.size(0);
    l.k = x.size(1);
    l.m_padded = (l.m + 127) / 128 * 128;
    l.chunks = l.m * l.k / 8;
    auto opts = x.options().dtype(at::kByte);
    l.fp4 = at::empty({l.m, l.k / 2}, opts);
    l.sf = at::empty({l.m_padded, (l.k / 16 + 3) / 4 * 4}, opts);
    l.xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    l.rq = static_cast<float>(rcp_q);
    l.wp = wgs.data_ptr<float>();
    l.ip = inv.data_ptr<float>();
    l.ap = alpha.data_ptr<float>();
    l.fp = reinterpret_cast<uint32_t*>(l.fp4.data_ptr());
    l.sp = l.sf.data_ptr<uint8_t>();
    l.stream = c10::cuda::getCurrentCUDAStream().stream();
    l.pdl_trigger = pdl_trigger;
    return l;
}

template <bool FOUR_OVER_SIX, int LORA_M>
void launch_redundant(const Launch& l, const __nv_bfloat16* vh, __nv_bfloat16* d, int rank) {
    const int quant_blocks = static_cast<int>((l.chunks + kThreads - 1) / kThreads);
    const int grid = quant_blocks + (LORA_M > 0 ? rank : 0);
    if (l.chunks <= 8 * kThreads) {
        dyn_quant_redundant_kernel<FOUR_OVER_SIX, LORA_M, 8><<<grid, kThreads, 0, l.stream>>>(
            l.xp, l.m, l.k, l.rq, l.wp, l.ip, l.ap, l.fp, l.sp, l.m_padded, quant_blocks, vh, d, rank, l.pdl_trigger);
    } else {
        dyn_quant_redundant_kernel<FOUR_OVER_SIX, LORA_M, 16><<<grid, kThreads, 0, l.stream>>>(
            l.xp, l.m, l.k, l.rq, l.wp, l.ip, l.ap, l.fp, l.sp, l.m_padded, quant_blocks, vh, d, rank, l.pdl_trigger);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <int LORA_M>
void launch_redundant(const Launch& l, bool four_over_six, const __nv_bfloat16* vh, __nv_bfloat16* d, int rank) {
    if (four_over_six) {
        launch_redundant<true, LORA_M>(l, vh, d, rank);
    } else {
        launch_redundant<false, LORA_M>(l, vh, d, rank);
    }
}

}  // namespace

std::vector<at::Tensor> dyn_quant(const at::Tensor& x, at::Tensor& inv, at::Tensor& alpha, const at::Tensor& wgs,
                                  double rcp_q, at::Tensor& scratch, bool four_over_six, bool pdl_trigger) {
    TORCH_CHECK(scratch.scalar_type() == at::kFloat && scratch.is_cuda() && scratch.numel() >= 1);
    const c10::cuda::CUDAGuard guard(x.device());
    Launch l = prepare(x, inv, alpha, wgs, rcp_q, pdl_trigger);
    if (l.chunks <= kRedundantMaxChunks) {
        launch_redundant<0>(l, four_over_six, nullptr, nullptr, 0);
        return {l.fp4, l.sf};
    }
    const int dev = x.get_device();
    void* coop_kernel = four_over_six ? reinterpret_cast<void*>(dyn_quant_kernel<true>)
                                      : reinterpret_cast<void*>(dyn_quant_kernel<false>);
    int& grid_limit = g_grid_limit[four_over_six ? 1 : 0][dev];
    if (grid_limit == 0) {
        int per_sm = 0, sms = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, coop_kernel, kThreads, 0));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        grid_limit = per_sm * sms;
        g_sms[dev] = sms;
    }
    const int64_t cap = l.chunks <= kSmallGridChunks ? int64_t{kSmallBlocksPerSm} * g_sms[dev] : grid_limit;
    const int64_t want = (l.chunks + kThreads - 1) / kThreads;
    int grid = static_cast<int>(std::min<int64_t>({want, cap, grid_limit, scratch.numel()}));
    float* scp = scratch.data_ptr<float>();
    const __nv_bfloat16* xp = l.xp;
    float rq = l.rq;
    const float* wp = l.wp;
    float* ip = l.ip;
    float* ap = l.ap;
    uint32_t* fp = l.fp;
    uint8_t* sp = l.sp;
    int64_t mm = l.m, kk = l.k, mp = l.m_padded;
    bool pt = l.pdl_trigger;
    void* args[] = {&xp, &mm, &kk, &rq, &wp, &ip, &ap, &fp, &sp, &mp, &scp, &pt};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(coop_kernel, dim3(grid), dim3(kThreads), args, 0, l.stream));
    return {l.fp4, l.sf};
}

constexpr int kQuantLoraMaxM = 8;

bool redundant_lora_fits(int64_t m, int64_t k) {
    return m >= 1 && m <= kQuantLoraMaxM && k % 16 == 0 && m * k / 8 <= kRedundantMaxChunks &&
           k <= int64_t{lora::kMaxIters} * lora::kThreads * 8;
}

bool coop_lora_fits(int64_t m, int64_t k) { return m >= 1 && m <= kCoopLoraMaxM && k % kCoopSliceK == 0; }

// Off: without software pipelining the per-slice loop is latency-bound and slower than dyn_quant + cuBLAS
// LoRA-down (results/test_lora_down-v7-tm16.log / -tm64.log).
bool tile_lora_fits(int64_t, int64_t) { return false; }

// Rank-64 inputs dyn_quant_lora handles: small ones in the redundant-amax launch, mid-size ones in the cooperative
// K-split launch (dyn_quant_lora_coop_kernel), larger ones in the cooperative row-tile launch
// (dyn_quant_lora_tile_kernel).
bool dyn_quant_lora_supported(int64_t m, int64_t k) {
    return redundant_lora_fits(m, k) || coop_lora_fits(m, k) || tile_lora_fits(m, k);
}

int g_coop_lora_limit[2][64] = {{0}};
int g_tile_lora_limit[2][2][64] = {{{0}}};

// Tile height of dyn_quant_lora_tile_kernel (NVFP4_TILE_M=16|64 overrides, for tuning experiments only).
int tile_rows_for(int64_t m) {
    static const int forced = std::getenv("NVFP4_TILE_M") ? std::atoi(std::getenv("NVFP4_TILE_M")) : 0;
    if (forced == 16 || forced == 64) return forced;
    return 64;
}

template <bool FOUR_OVER_SIX, int TILE_M>
void launch_tile(const Launch& l, const __nv_bfloat16* vp, __nv_bfloat16* dp, at::Tensor& scratch, int dev) {
    void* kernel = reinterpret_cast<void*>(dyn_quant_lora_tile_kernel<FOUR_OVER_SIX, TILE_M>);
    int& limit = g_tile_lora_limit[FOUR_OVER_SIX ? 1 : 0][TILE_M == 64 ? 1 : 0][dev];
    if (limit == 0) {
        int per_sm = 0, sms = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        limit = per_sm * sms;
        g_sms[dev] = sms;
    }
    const int64_t cap = l.chunks <= kSmallGridChunks ? int64_t{kSmallBlocksPerSm} * g_sms[dev] : limit;
    const int64_t want = (l.chunks + kThreads - 1) / kThreads;
    int grid = static_cast<int>(std::min<int64_t>({want, cap, limit, scratch.numel()}));
    const __nv_bfloat16* xp = l.xp;
    float rq = l.rq;
    const float* wp = l.wp;
    float* ip = l.ip;
    float* ap = l.ap;
    uint32_t* fp = l.fp;
    uint8_t* sp = l.sp;
    float* scp = scratch.data_ptr<float>();
    int64_t mm = l.m, kk = l.k, mp = l.m_padded;
    bool pt = l.pdl_trigger;
    void* args[] = {&xp, &mm, &kk, &rq, &wp, &ip, &ap, &fp, &sp, &mp, &scp, &vp, &dp, &pt};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(kThreads), args, 0, l.stream));
}

// dyn_quant plus the SVDQ LoRA-down d = bf16((x @ vhᵀ) / alpha) in the same launch. scratch: >= grid floats of
// per-block maxima; partials: >= (k / 256) * m * 64 floats (cooperative path).
std::vector<at::Tensor> dyn_quant_lora(const at::Tensor& x, at::Tensor& inv, at::Tensor& alpha, const at::Tensor& wgs,
                                       double rcp_q, bool four_over_six, const at::Tensor& vh, at::Tensor& scratch,
                                       at::Tensor& partials, bool pdl_trigger) {
    TORCH_CHECK(dyn_quant_lora_supported(x.size(0), x.size(1)), "unsupported dyn_quant_lora shape");
    TORCH_CHECK(vh.is_cuda() && vh.scalar_type() == at::kBFloat16 && vh.dim() == 2 && vh.is_contiguous() &&
                vh.size(1) == x.size(1) && reinterpret_cast<uintptr_t>(vh.data_ptr()) % 16 == 0);
    const c10::cuda::CUDAGuard guard(x.device());
    Launch l = prepare(x, inv, alpha, wgs, rcp_q, pdl_trigger);
    const int rank = static_cast<int>(vh.size(0));
    at::Tensor d = at::empty({l.m, rank}, x.options());
    const __nv_bfloat16* vp = reinterpret_cast<const __nv_bfloat16*>(vh.data_ptr());
    __nv_bfloat16* dp = reinterpret_cast<__nv_bfloat16*>(d.data_ptr());
    if (redundant_lora_fits(l.m, l.k)) {
        if (l.m <= 1) {
            launch_redundant<1>(l, four_over_six, vp, dp, rank);
        } else if (l.m <= 2) {
            launch_redundant<2>(l, four_over_six, vp, dp, rank);
        } else if (l.m <= 4) {
            launch_redundant<4>(l, four_over_six, vp, dp, rank);
        } else {
            launch_redundant<8>(l, four_over_six, vp, dp, rank);
        }
        return {l.fp4, l.sf, d};
    }
    TORCH_CHECK(rank == kCoopRank, "the cooperative LoRA-down paths are rank ", kCoopRank);
    if (tile_lora_fits(l.m, l.k)) {
        const int dev = x.get_device();
        const int tile_m = tile_rows_for(l.m);
        if (four_over_six) {
            tile_m == 64 ? launch_tile<true, 64>(l, vp, dp, scratch, dev) : launch_tile<true, 16>(l, vp, dp, scratch, dev);
        } else {
            tile_m == 64 ? launch_tile<false, 64>(l, vp, dp, scratch, dev) : launch_tile<false, 16>(l, vp, dp, scratch, dev);
        }
        return {l.fp4, l.sf, d};
    }
    const int slices = static_cast<int>(l.k / kCoopSliceK);
    const int grid = static_cast<int>((l.m + kCoopRows - 1) / kCoopRows) * slices;
    TORCH_CHECK(scratch.scalar_type() == at::kFloat && scratch.numel() >= grid);
    TORCH_CHECK(partials.scalar_type() == at::kFloat && partials.numel() >= int64_t{slices} * l.m * kCoopRank);
    void* kernel = four_over_six ? reinterpret_cast<void*>(dyn_quant_lora_coop_kernel<true>)
                                 : reinterpret_cast<void*>(dyn_quant_lora_coop_kernel<false>);
    const int dev = x.get_device();
    int& limit = g_coop_lora_limit[four_over_six ? 1 : 0][dev];
    if (limit == 0) {
        int per_sm = 0, sms = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0));
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
        limit = per_sm * sms;
    }
    TORCH_CHECK(grid <= limit, "dyn_quant_lora: cooperative grid ", grid, " exceeds ", limit);
    const __nv_bfloat16* xp = l.xp;
    float rq = l.rq;
    const float* wp = l.wp;
    float* ip = l.ip;
    float* ap = l.ap;
    uint32_t* fp = l.fp;
    uint8_t* sp = l.sp;
    float* scp = scratch.data_ptr<float>();
    float* pp = partials.data_ptr<float>();
    int64_t mm = l.m, kk = l.k, mp = l.m_padded;
    bool pt = l.pdl_trigger;
    void* args[] = {&xp, &mm, &kk, &rq, &wp, &ip, &ap, &fp, &sp, &mp, &scp, &vp, &dp, &pp, &pt};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel, dim3(grid), dim3(kThreads), args, 0, l.stream));
    return {l.fp4, l.sf, d};
}

// lora_down.cu
bool lora_down_supported(int64_t m, int64_t k);
at::Tensor lora_down_scaled(const at::Tensor& x, const at::Tensor& vh, const at::Tensor& alpha);
void div_by_alpha_(at::Tensor& d, const at::Tensor& alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, mod) {
    mod.def("dyn_quant", &dyn_quant, "Dynamic per-tensor NVFP4 quantization in one launch");
    mod.def("dyn_quant_lora_supported", &dyn_quant_lora_supported, "Whether dyn_quant_lora handles an [m, k] input");
    mod.def("dyn_quant_lora", &dyn_quant_lora, "dyn_quant plus the SVDQ LoRA-down (x @ vh^T) / alpha, one launch");
    mod.def("lora_down_supported", &lora_down_supported, "Whether lora_down_scaled handles an [m, k] input");
    mod.def("lora_down_scaled", &lora_down_scaled, "bf16((x @ vh^T) / alpha) for small M in one launch");
    mod.def("div_by_alpha_", &div_by_alpha_, "d = bf16(float(d) / alpha), in place");
}
