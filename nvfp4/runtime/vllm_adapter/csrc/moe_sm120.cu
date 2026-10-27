// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// 4o6 NVFP4 quantization of token-major (expert-sorted) MoE rows for the SM120 CUTLASS grouped GEMM.
//
// The reference path quantizes padded [E, capacity, K] buffers with FlashInfer's TRT-LLM kernels
// (scaled_fp4_grouped_quantize and silu_and_mul_scaled_nvfp4_experts_quantize, 4o6 through the
// FLASHINFER_NVFP4_4OVER6* environment). This file calls the same device functions with the same
// template configuration (E4M3 max 256, MSE error, no error fast math, FP4 fast math on) and is built
// with FlashInfer's JIT flags, so every code and scale byte matches the reference. Only the row
// placement differs: rows are the expert-sorted routes of vLLM's get_cutlass_moe_mm_data, and each
// expert's scale block starts at its 128-row-aligned blockscale offset, as cutlass_fp4_group_mm reads it.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "tensorrt_llm/kernels/quantization_utils.cuh"

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

namespace {

using Config4o6 = tc::NVFP44Over6Config<256, tc::NVFP44Over6ErrMode::MSE, false>;
// quantization.cuh's CVT_FP16_TO_FP4_ELTS_PER_THREAD for SM >= 100 with CUDA >= 12.9 (FlashInfer builds SM12x
// as sm_120f, which needs 12.9): one thread per 16-element scale block, so the 4o6 error sums match.
constexpr int kElts = 16;
constexpr int kSfVec = 16;
using Vec = tk::PackedVec<__nv_bfloat16, kElts>;

// The expert whose segment [offsets[e], offsets[e + 1]) holds `row`; empty experts share their
// successor's offset, so the largest e with offsets[e] <= row is the owner.
__device__ __forceinline__ int owner(const int32_t* __restrict__ offsets, int experts, int row) {
    int lo = 0, hi = experts;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (offsets[mid] <= row) lo = mid; else hi = mid;
    }
    return lo;
}

__device__ __forceinline__ void load_vec(Vec& v, const __nv_bfloat16* p) {
    // The 32-byte vector load of TRT-LLM's loadPackedVec.
    reinterpret_cast<uint4*>(&v)[0] = reinterpret_cast<const uint4*>(p)[0];
    reinterpret_cast<uint4*>(&v)[1] = reinterpret_cast<const uint4*>(p)[1];
}

// One thread per 16 output elements of one row. SILU: the row holds [gate | up] of width 2 * cols and
// the quantized value is TRT-LLM's silu_and_mul(gate, up), as in cvt_fp16_to_fp4_expert.
template <bool SILU>
__global__ void __launch_bounds__(256) experts_quant_4o6_kernel(
    int rows, int cols, const __nv_bfloat16* __restrict__ in, int64_t in_stride,
    const int32_t* __restrict__ src_rows, const int32_t* __restrict__ offsets,
    const int32_t* __restrict__ sf_offsets, int experts, const float* __restrict__ gscale,
    uint64_t* __restrict__ out, uint32_t* __restrict__ sf, int32_t* __restrict__ row_expert) {
    const int vecs = cols / kElts;
    const int64_t total = static_cast<int64_t>(rows) * vecs;
    const int sf_words = (cols + kSfVec * 4 - 1) / (kSfVec * 4);  // uint32 words per scale row
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int r = static_cast<int>(i / vecs), c = static_cast<int>(i % vecs);
        const int e = owner(offsets, experts, r);
        if (row_expert != nullptr && c == 0) row_expert[r] = e;
        const int64_t src = src_rows != nullptr ? src_rows[r] : r;
        const __nv_bfloat16* row = in + src * in_stride;
        Vec v;
        load_vec(v, row + c * kElts);
        if constexpr (SILU) {
            Vec u;
            load_vec(u, row + cols + c * kElts);
            tk::silu_and_mul<__nv_bfloat16, kElts>(v, u);
        }
        uint32_t* sf_expert = sf + static_cast<int64_t>(sf_offsets[e]) * sf_words;
        uint8_t* sf_out = tk::cvt_quant_to_fp4_get_sf_out_offset<uint32_t, kSfVec, kSfVec / kElts>(
            r - offsets[e], c, cols, sf_expert);
        out[i] = tk::cvt_warp_fp16_to_fp4<__nv_bfloat16, kSfVec, kElts, false, false, Config4o6>(
            v, gscale[e], sf_out);
    }
}


// ---------------------------------------------------------------------------------------------------------
// Small-M grouped NVFP4 GEMM (decode): out[r, n] = bf16(alpha[e] * sum_k a[r, k] w[e, n, k]) over the
// expert-sorted rows, with the same operands and scale layouts as cutlass_fp4_group_mm. Each 16-element
// block contributes (sum of e2m1 products) * sf_a * sf_w, which is exact in FP32 as in the tensor-core
// MMA; only the FP32 accumulation order over blocks differs from CUTLASS.

constexpr int kGemvRows = 4;      // rows of one expert per CTA (a tile)
constexpr int kGemvThreads = 128;
constexpr int kUnroll = 4;        // weight rows in flight per thread

__device__ __forceinline__ float2 e2m1x2_to_float2(uint32_t byte) {
    float2 r;
    asm volatile(
        "{\n"
        ".reg .b8 b0, b1, b2, b3;\n"
        ".reg .b32 h2;\n"
        ".reg .b16 lo, hi;\n"
        "mov.b32 {b0, b1, b2, b3}, %2;\n"
        "cvt.rn.f16x2.e2m1x2 h2, b0;\n"
        "mov.b32 {lo, hi}, h2;\n"
        "cvt.f32.f16 %0, lo;\n"
        "cvt.f32.f16 %1, hi;\n"
        "}"
        : "=f"(r.x), "=f"(r.y)
        : "r"(byte));
    return r;
}

__device__ __forceinline__ float e4m3_to_float(uint32_t byte) {
    return static_cast<float>(*reinterpret_cast<const __nv_fp8_e4m3*>(&byte));
}

// Sum over 16 elements (8 bytes of each operand) of the e2m1 products: exact in FP32.
__device__ __forceinline__ float block_dot(uint2 a, uint2 w) {
    float sum = 0.f;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint32_t av = half ? a.y : a.x, wv = half ? w.y : w.x;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const float2 x = e2m1x2_to_float2((av >> (8 * b)) & 0xFF);
            const float2 y = e2m1x2_to_float2((wv >> (8 * b)) & 0xFF);
            sum = fmaf(x.x, y.x, sum);
            sum = fmaf(x.y, y.y, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ int64_t sf_offset(int row, int kb, int k_tiles) {
    return static_cast<int64_t>(row / 128) * k_tiles * 512 + (kb / 4) * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 +
           kb % 4;
}

template <int K, int N_PER_CTA>
__global__ void __launch_bounds__(kGemvThreads) grouped_gemv_fp4_kernel(
    const uint8_t* __restrict__ a, const uint8_t* __restrict__ a_sf, const uint8_t* __restrict__ w,
    const uint8_t* __restrict__ w_sf, const float* __restrict__ alpha, const int32_t* __restrict__ sf_offsets,
    const int4* __restrict__ tile_map, int n_total, int n_padded, __nv_bfloat16* __restrict__ out) {
    constexpr int kRowBytes = K / 2;
    constexpr int kTpr = kRowBytes / 32;            // threads per weight row, 32 bytes = 64 elements = 4 blocks each
    static_assert(kTpr >= 1 && kTpr <= 32 && 32 % kTpr == 0, "row split");
    constexpr int kRowsPerWarp = 32 / kTpr;
    constexpr int kWarps = kGemvThreads / 32;
    constexpr int kKTiles = (K / 16 + 3) / 4;
    const int4 tile = tile_map[blockIdx.x];  // (expert, first row, row end, expert's first row)
    const int e = tile.x;
    if (e < 0) return;
    const int row0 = tile.y;
    const int rows = min(kGemvRows, tile.z - row0);
    const int expert_first = tile.w;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int sub = lane % kTpr;                     // this thread's 64-element K slice
    const int kb0 = sub * 4;
    uint4 a_codes[kGemvRows][2];
    float a_scale[kGemvRows][4];
    const uint8_t* a_sf_e = a_sf + static_cast<int64_t>(sf_offsets[e]) * kKTiles * 4;
#pragma unroll
    for (int j = 0; j < kGemvRows; ++j) {
        if (j < rows) {
            const uint4* src = reinterpret_cast<const uint4*>(a + static_cast<int64_t>(row0 + j) * kRowBytes + sub * 32);
            a_codes[j][0] = src[0];
            a_codes[j][1] = src[1];
            const uint32_t sf4 = *reinterpret_cast<const uint32_t*>(a_sf_e + sf_offset(row0 + j - expert_first, kb0, kKTiles));
#pragma unroll
            for (int b = 0; b < 4; ++b) a_scale[j][b] = e4m3_to_float((sf4 >> (8 * b)) & 0xFF);
        }
    }
    const float scale_out = alpha[e];
    const uint8_t* w_e = w + static_cast<int64_t>(e) * n_total * kRowBytes;
    const uint8_t* w_sf_e = w_sf + static_cast<int64_t>(e) * n_padded * kKTiles * 4;
    const int n_base = blockIdx.y * N_PER_CTA;
    const int n_end = min(n_base + N_PER_CTA, n_total);
    constexpr int kStride = kWarps * kRowsPerWarp;  // rows between this thread's consecutive weight rows
    // kUnroll weight rows per thread are loaded before any is used, so each warp keeps kUnroll x 32 x 32 bytes of
    // the weight stream in flight (one row at a time left the decode GEMV latency-bound at ~400 GB/s).
    for (int n0 = n_base + warp * kRowsPerWarp + lane / kTpr; n0 < n_end; n0 += kStride * kUnroll) {
        uint4 wv[kUnroll][2];
        uint32_t wsf[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int n = n0 + u * kStride;
            if (n < n_end) {
                const uint4* wsrc = reinterpret_cast<const uint4*>(w_e + static_cast<int64_t>(n) * kRowBytes + sub * 32);
                wv[u][0] = __ldg(wsrc);
                wv[u][1] = __ldg(wsrc + 1);
                wsf[u] = __ldg(reinterpret_cast<const uint32_t*>(w_sf_e + sf_offset(n, kb0, kKTiles)));
            } else {
                wv[u][0] = wv[u][1] = make_uint4(0, 0, 0, 0);
                wsf[u] = 0;
            }
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const int n = n0 + u * kStride;
            float w_scale[4];
#pragma unroll
            for (int b = 0; b < 4; ++b) w_scale[b] = e4m3_to_float((wsf[u] >> (8 * b)) & 0xFF);
            const uint4 w0 = wv[u][0], w1 = wv[u][1];
            float acc[kGemvRows];
#pragma unroll
            for (int j = 0; j < kGemvRows; ++j) {
                acc[j] = 0.f;
                if (j < rows) {
                    const uint4 x0 = a_codes[j][0], x1 = a_codes[j][1];
                    acc[j] = fmaf(block_dot(make_uint2(x0.x, x0.y), make_uint2(w0.x, w0.y)) * a_scale[j][0], w_scale[0], acc[j]);
                    acc[j] = fmaf(block_dot(make_uint2(x0.z, x0.w), make_uint2(w0.z, w0.w)) * a_scale[j][1], w_scale[1], acc[j]);
                    acc[j] = fmaf(block_dot(make_uint2(x1.x, x1.y), make_uint2(w1.x, w1.y)) * a_scale[j][2], w_scale[2], acc[j]);
                    acc[j] = fmaf(block_dot(make_uint2(x1.z, x1.w), make_uint2(w1.z, w1.w)) * a_scale[j][3], w_scale[3], acc[j]);
                }
            }
#pragma unroll
            for (int j = 0; j < kGemvRows; ++j) {
#pragma unroll
                for (int off = kTpr / 2; off > 0; off /= 2) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], off);
            }
            if (sub == 0 && n < n_end) {
#pragma unroll
                for (int j = 0; j < kGemvRows; ++j)
                    if (j < rows) out[static_cast<int64_t>(row0 + j) * n_total + n] = __float2bfloat16_rn(scale_out * acc[j]);
            }
        }
    }
}

}  // namespace

// in: [*, in_stride] BF16 (in_stride = cols, or 2 * cols with silu); src_rows: [rows] source row of each
// sorted route (nullable: identity); offsets / sf_offsets: [experts + 1]; gscale: [experts] FP32.
// Writes out [rows, cols / 2] (uint8 e2m1x2), sf (swizzled UE4M3, per-expert blocks) and, if given,
// row_expert [rows].
void experts_quant_4o6(torch::Tensor in, c10::optional<torch::Tensor> src_rows, torch::Tensor offsets,
                       torch::Tensor sf_offsets, torch::Tensor gscale, int64_t cols, bool silu,
                       torch::Tensor out, torch::Tensor sf, c10::optional<torch::Tensor> row_expert) {
    TORCH_CHECK(in.scalar_type() == at::kBFloat16 && in.is_contiguous(), "BF16 contiguous input");
    TORCH_CHECK(cols % 64 == 0, "cols must be a multiple of 64");
    TORCH_CHECK(in.size(-1) == (silu ? 2 * cols : cols), "input width");
    const int rows = static_cast<int>(out.size(0));
    const int experts = static_cast<int>(gscale.numel());
    TORCH_CHECK(offsets.numel() == experts + 1 && sf_offsets.numel() >= experts, "offsets");
    TORCH_CHECK(out.size(1) == cols / 2 && out.is_contiguous(), "out shape");
    if (rows == 0) return;
    const c10::cuda::CUDAGuard guard(in.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const int64_t total = static_cast<int64_t>(rows) * (cols / kElts);
    const int threads = 256;
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, in.get_device()));
    const int blocks = static_cast<int>(std::min<int64_t>((total + threads - 1) / threads, 8LL * sms));
    auto launch = [&](auto kernel) {
        kernel<<<blocks, threads, 0, stream>>>(
            rows, static_cast<int>(cols), reinterpret_cast<const __nv_bfloat16*>(in.data_ptr()), in.size(-1),
            src_rows ? src_rows->data_ptr<int32_t>() : nullptr, offsets.data_ptr<int32_t>(),
            sf_offsets.data_ptr<int32_t>(), experts, gscale.data_ptr<float>(),
            reinterpret_cast<uint64_t*>(out.data_ptr()), reinterpret_cast<uint32_t*>(sf.data_ptr()),
            row_expert ? row_expert->data_ptr<int32_t>() : nullptr);
    };
    if (silu) launch(experts_quant_4o6_kernel<true>); else launch(experts_quant_4o6_kernel<false>);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Rows of each expert in tiles of kGemvRows; tile_map [max_tiles, 4] int32 = (expert, first row, row end,
// expert's first row), expert -1 for unused tiles (written by sm120_moe._routing).
// a [rows, K/2] / a_sf: as written by experts_quant_4o6; w [E, N, K/2], w_sf [E, N_pad, K/16 pad] (the
// cutlass_fp4_group_mm operands); out [rows, N] BF16.
void grouped_gemv_fp4(torch::Tensor a, torch::Tensor a_sf, torch::Tensor w, torch::Tensor w_sf, torch::Tensor alpha,
                      torch::Tensor sf_offsets, torch::Tensor tile_map, int64_t max_tiles, torch::Tensor out) {
    const int n_total = static_cast<int>(w.size(1));
    const int k = static_cast<int>(w.size(2)) * 2;
    const int n_padded = static_cast<int>(w_sf.size(1));
    TORCH_CHECK(out.size(1) == n_total && a.size(1) == k / 2, "grouped_gemv_fp4 shapes");
    if (max_tiles == 0) return;
    const c10::cuda::CUDAGuard guard(a.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    auto launch = [&](auto kernel, int n_per_cta) {
        const dim3 grid(static_cast<unsigned>(max_tiles), static_cast<unsigned>((n_total + n_per_cta - 1) / n_per_cta));
        kernel<<<grid, kGemvThreads, 0, stream>>>(
            a.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(a_sf.data_ptr()),
            reinterpret_cast<const uint8_t*>(w.data_ptr()), reinterpret_cast<const uint8_t*>(w_sf.data_ptr()),
            alpha.data_ptr<float>(), sf_offsets.data_ptr<int32_t>(),
            reinterpret_cast<const int4*>(tile_map.data_ptr<int32_t>()), n_total, n_padded,
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
    };
    if (k == 2048) launch(grouped_gemv_fp4_kernel<2048, 16>, 16);
    else if (k == 256) launch(grouped_gemv_fp4_kernel<256, 128>, 128);
    else TORCH_CHECK(false, "grouped_gemv_fp4: unsupported K ", k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("grouped_gemv_fp4", &grouped_gemv_fp4);
    m.def("experts_quant_4o6", &experts_quant_4o6);
}
