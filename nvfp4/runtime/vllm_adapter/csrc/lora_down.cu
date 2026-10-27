// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// SVDQ LoRA-down for decode-sized inputs: d[m, r] = bf16(fp32(sum_k x[m, k] * vh[r, k]) / alpha), one launch,
// one block per rank row (lora_row.cuh). Replaces cuBLAS (x @ vhᵀ) + the pointwise "/ alpha" for small inputs
// (M*K <= 49152, i.e. decode batches up to 12 rows at K=4096 or 4 rows at K=12288) in the fused SVDQ path when
// the quantization launch cannot carry the LoRA blocks itself (dyn_quant.cu, dyn_quant_lora).

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "lora_row.cuh"

namespace {

constexpr int kLoraMaxM = 16;
// Every block re-reads all of x, so the fused kernel only wins while x is small (graph replay, fused vs cuBLAS +
// division: K=4096 M=8 4.9 vs 7.3 µs, M=16 8.2 vs 7.4; K=12288 M=4 6.3 vs 7.8, M=8 11.1 vs 9.1).
constexpr int64_t kLoraMaxElements = 49152;

template <int MAX_M>
__global__ void __launch_bounds__(lora::kThreads) lora_down_kernel(const __nv_bfloat16* __restrict__ x,
                                                                   const __nv_bfloat16* __restrict__ vh,
                                                                   const float* __restrict__ alpha,
                                                                   __nv_bfloat16* __restrict__ d, int m, int k,
                                                                   int rank) {
    lora::row<MAX_M>(x, vh, *alpha, d, m, k, rank, blockIdx.x);
}

template <int MAX_M>
void launch(const at::Tensor& x, const at::Tensor& vh, const at::Tensor& alpha, at::Tensor& d, cudaStream_t stream) {
    const int m = static_cast<int>(x.size(0)), k = static_cast<int>(x.size(1)), rank = static_cast<int>(vh.size(0));
    lora_down_kernel<MAX_M><<<rank, lora::kThreads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(vh.data_ptr()),
        alpha.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(d.data_ptr()), m, k, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// d = bf16(float(d) / alpha) in place: the division and rounding inductor fused after cuBLAS (x @ vhᵀ) in the
// traced path, as one kernel (an eager "/ alpha" with a 1-element FP32 alpha promotes to FP32 and needs a cast).
__global__ void div_by_alpha_kernel(__nv_bfloat162* __restrict__ d, int64_t pairs, const float* __restrict__ alpha) {
    const float a = *alpha;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < pairs;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const float2 v = __bfloat1622float2(d[i]);
        d[i] = __floats2bfloat162_rn(__fdiv_rn(v.x, a), __fdiv_rn(v.y, a));
    }
}

}  // namespace

void div_by_alpha_(at::Tensor& d, const at::Tensor& alpha) {
    TORCH_CHECK(d.is_cuda() && d.scalar_type() == at::kBFloat16 && d.is_contiguous() && d.numel() % 2 == 0);
    TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == at::kFloat && alpha.numel() == 1);
    const c10::cuda::CUDAGuard guard(d.device());
    const int64_t pairs = d.numel() / 2;
    const int blocks = static_cast<int>(std::min<int64_t>((pairs + 255) / 256, 1024));
    div_by_alpha_kernel<<<std::max(blocks, 1), 256, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
        reinterpret_cast<__nv_bfloat162*>(d.data_ptr()), pairs, alpha.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

bool lora_down_supported(int64_t m, int64_t k) {
    return m >= 1 && m <= kLoraMaxM && m * k <= kLoraMaxElements && k % 8 == 0 &&
           k <= int64_t{lora::kMaxIters} * lora::kThreads * 8;
}

at::Tensor lora_down_scaled(const at::Tensor& x, const at::Tensor& vh, const at::Tensor& alpha) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.is_contiguous());
    TORCH_CHECK(vh.is_cuda() && vh.scalar_type() == at::kBFloat16 && vh.dim() == 2 && vh.is_contiguous());
    TORCH_CHECK(alpha.is_cuda() && alpha.scalar_type() == at::kFloat && alpha.numel() == 1);
    TORCH_CHECK(vh.size(1) == x.size(1) && lora_down_supported(x.size(0), x.size(1)), "unsupported lora_down shape");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(vh.data_ptr()) % 16 == 0);
    const c10::cuda::CUDAGuard guard(x.device());
    at::Tensor d = at::empty({x.size(0), vh.size(0)}, x.options());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const int64_t m = x.size(0);
    if (m <= 1) {
        launch<1>(x, vh, alpha, d, stream);
    } else if (m <= 2) {
        launch<2>(x, vh, alpha, d, stream);
    } else if (m <= 4) {
        launch<4>(x, vh, alpha, d, stream);
    } else if (m <= 8) {
        launch<8>(x, vh, alpha, d, stream);
    } else {
        launch<16>(x, vh, alpha, d, stream);
    }
    return d;
}
