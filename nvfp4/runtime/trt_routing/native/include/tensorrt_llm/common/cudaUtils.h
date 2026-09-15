// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "assert.h"

namespace tensorrt_llm::common {
inline void routingCudaCheck(cudaError_t status) {
    if (status != cudaSuccess) routingError("TRT routing CUDA error: %s", cudaGetErrorString(status));
}
inline int routingDeviceAttribute(cudaDeviceAttr attr) {
    int device = 0, value = 0;
    routingCudaCheck(cudaGetDevice(&device));
    routingCudaCheck(cudaDeviceGetAttribute(&value, attr, device));
    return value;
}
inline int getSMVersion() {
    return 10 * routingDeviceAttribute(cudaDevAttrComputeCapabilityMajor)
        + routingDeviceAttribute(cudaDevAttrComputeCapabilityMinor);
}
inline int getMultiProcessorCount() { return routingDeviceAttribute(cudaDevAttrMultiProcessorCount); }
}
#define TLLM_CUDA_CHECK(expression) tensorrt_llm::common::routingCudaCheck(expression)
