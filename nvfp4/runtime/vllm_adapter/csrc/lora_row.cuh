// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// One rank row of the SVDQ LoRA-down, d[i, r] = bf16(fp32(sum_k x[i, k] * vh[r, k]) / alpha) for i < m, computed by
// one 256-thread block. Shared by lora_down.cu (standalone launch) and dyn_quant.cu (merged into the quantization
// launch). The block's threads split K and stage their vh chunks in registers first so all weight loads are in
// flight together; the FP32 sums are reduced in a fixed order (deterministic, no atomics) and rounded once.
#pragma once

#include <cuda_bf16.h>

namespace lora {

constexpr int kThreads = 256;
constexpr int kMaxIters = 8;  // K <= kMaxIters * kThreads * 8 = 16384

__device__ __forceinline__ float dot8(const uint4& a, const uint4& b, float acc) {
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 fa = __bfloat1622float2(ha[j]);
        const float2 fb = __bfloat1622float2(hb[j]);
        acc = fmaf(fa.x, fb.x, acc);
        acc = fmaf(fa.y, fb.y, acc);
    }
    return acc;
}

// Must be called by all kThreads threads of the block (it contains a block barrier).
template <int MAX_M>
__device__ __forceinline__ void row(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ vh,
                                   float alpha, __nv_bfloat16* __restrict__ d, int m, int k, int rank, int r) {
    const int chunks = k / 8;
    const uint4* wrow = reinterpret_cast<const uint4*>(vh + static_cast<int64_t>(r) * k);
    const uint4* xv = reinterpret_cast<const uint4*>(x);
    uint4 w[kMaxIters];
#pragma unroll
    for (int it = 0; it < kMaxIters; ++it) {
        const int c = threadIdx.x + it * kThreads;
        if (c < chunks) w[it] = wrow[c];
    }
    float acc[MAX_M];
#pragma unroll
    for (int i = 0; i < MAX_M; ++i) acc[i] = 0.f;
#pragma unroll
    for (int it = 0; it < kMaxIters; ++it) {
        const int c = threadIdx.x + it * kThreads;
        if (c < chunks) {
#pragma unroll
            for (int i = 0; i < MAX_M; ++i) {
                if (i < m) acc[i] = dot8(xv[static_cast<int64_t>(i) * chunks + c], w[it], acc[i]);
            }
        }
    }
    __shared__ float partial[kThreads / 32][MAX_M];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int i = 0; i < MAX_M; ++i) {
        float v = acc[i];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) partial[warp][i] = v;
    }
    __syncthreads();
    if (threadIdx.x < m) {
        float s = 0.f;
#pragma unroll
        for (int wi = 0; wi < kThreads / 32; ++wi) s += partial[wi][threadIdx.x];
        d[static_cast<int64_t>(threadIdx.x) * rank + r] = __float2bfloat16_rn(__fdiv_rn(s, alpha));
    }
}

}  // namespace lora
