// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/routing/RoutingKernel.h"

std::vector<at::Tensor> compact_sort(at::Tensor const& ids, at::Tensor const& weights,
                                    int64_t num_experts, int64_t start, int64_t local_experts) {
    TORCH_CHECK(ids.is_cuda() && ids.dim() == 2 && ids.is_contiguous() && ids.scalar_type() == at::kInt);
    TORCH_CHECK(weights.device() == ids.device() && weights.sizes() == ids.sizes()
                && weights.is_contiguous() && weights.scalar_type() == at::kFloat);
    TORCH_CHECK(num_experts > 0 && num_experts <= 2048 && start >= 0 && local_experts > 0
                && start + local_experts <= num_experts);
    int64_t const m = ids.size(0), topk = ids.size(1);
    TORCH_CHECK(m > 0 && m <= 16384 && topk > 0 && topk <= 8);
    c10::cuda::CUDAGuard guard(ids.device());
    int64_t const filled = std::min(local_experts, m * topk);
    int64_t const tiles = filled + (m * topk - filled) / 128;
    auto options = ids.options();
    auto groups = at::empty({tiles}, options), limits = at::empty({tiles}, options);
    auto perm = at::empty({tiles * 128}, options), inverse = at::empty_like(ids);
    auto live = at::empty({1}, options), padded = at::empty({1}, options);
    auto histogram = at::empty({std::max(num_experts * 2, int64_t(512))}, options);
    moe::dev::routing::routingCustom::Data data;
    data.mDtypeOutput = batchedGemm::trtllm::gen::Dtype::Fp32;
    data.mDtypeInput = batchedGemm::trtllm::gen::Dtype::Fp32;
    data.mPtrTopKIds = ids.data_ptr<int>();
    data.mPtrTopKWeights = weights.data_ptr();
    data.mPtrExpertCounts = histogram.data_ptr<int>();
    data.mPtrPermutedIdxSize = padded.data_ptr<int>();
    data.mPtrExpandedIdxToPermutedIdx = inverse.data_ptr<int>();
    data.mPtrPermutedIdxToExpandedIdx = perm.data_ptr<int>();
    data.mPtrCtaIdxXyToBatchIdx = groups.data_ptr<int>();
    data.mPtrCtaIdxXyToMnLimit = limits.data_ptr<int>();
    data.mPtrNumNonExitingCtas = live.data_ptr<int>();
    data.mNumTokens = m;
    data.mNumExperts = num_experts;
    data.mTopK = topk;
    data.mPaddingLog2 = 7;
    data.mTileTokensDim = 128;
    data.mLocalExpertsStartIdx = start;
    data.mLocalExpertsStrideLog2 = 0;
    data.mNumLocalExperts = local_experts;
    moe::dev::routing::routingCustom::run(data, at::cuda::getCurrentCUDAStream().stream());
    return {groups, limits, perm, live};
}

TORCH_LIBRARY(project_trt_routing, m) {
    m.def("compact_sort(Tensor ids, Tensor weights, int num_experts, int start, int local_experts) -> Tensor[]");
}
TORCH_LIBRARY_IMPL(project_trt_routing, CUDA, m) {
    m.impl("compact_sort", &compact_sort);
}
