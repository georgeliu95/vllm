/*
 * Copyright (c) 2024-2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Dispatch + instantiation for adaptive 4/6 FP4 quantization and global amax.
 * Derived from tllm_linear_lite/quantize/quantization.cu.
 */

#include "tensorrt_llm/kernels/fp4QuantizeAdaptive.cuh"
#include <float.h>
#include <unordered_map>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// invokeFP4QuantizationEx: dispatch to v0/v1/adaptive kernels

template <typename T, int SF_VEC_SIZE>
void invokeFP4QuantizationEx(int b, int m, int n, T const* input, float const* SFScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream,
    int kernelVersion, int scaleRule)
{
#ifdef ENABLE_FP8
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
    {
        // FP8→FP4 only supports v0; adaptive not supported for FP8 input.
        dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
        int const numBlocksPerSM = std::max(1u, 2048u / block.x);
        int numBlocksForM = layout == QuantizationSFLayout::SWIZZLED ? PadUpFn(m, 128)
            : (layout == QuantizationSFLayout::R8C4 ? PadUpFn(m, 8) : m);
        dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));

        auto* kernel_instance = useUE8M0
            ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, true>
            : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, false>;
        kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOutput), layout);
    }
    else
#endif
    {
        // FP16/BF16→FP4.
        int numBlocksForM = layout == QuantizationSFLayout::SWIZZLED ? PadUpFn(m, 128)
            : (layout == QuantizationSFLayout::R8C4 ? PadUpFn(m, 8) : m);

        cudaLaunchConfig_t config;
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attrs[1];
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
        config.numAttrs = 1;
        config.attrs = attrs;

        if (scaleRule != 0)
        {
            // Adaptive 4/6: 16 elems/thread, per-block MSE/MAE/ABS_MAX selection.
            dim3 block(std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512));
            int const numBlocksPerSM = std::max(1u, 2048u / block.x);
            dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));
            config.gridDim = grid;
            config.blockDim = block;

#define LAUNCH_ADAPTIVE_KERNEL(RULE)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        auto* ki = &opt_quantize_with_block_size_adaptive<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE,     \
            AdaptiveScaleRule::RULE>;                                                                                   \
        cudaLaunchKernelEx(&config, ki, b, m, n, n, input, SFScale, reinterpret_cast<uint32_t*>(output),               \
            reinterpret_cast<uint32_t*>(SFOutput), layout);                                                            \
    } while (0)

            switch (scaleRule)
            {
            case 1: LAUNCH_ADAPTIVE_KERNEL(MSE); break;
            case 2: LAUNCH_ADAPTIVE_KERNEL(MAE); break;
            case 3: LAUNCH_ADAPTIVE_KERNEL(ABS_MAX); break;
            }
#undef LAUNCH_ADAPTIVE_KERNEL
        }
        else if (kernelVersion == 0)
        {
            // v0: 8 elements per thread (same as existing invokeFP4Quantization).
            dim3 block(std::min(int(n / CVT_ELTS_PER_THREAD), 512));
            int const numBlocksPerSM = std::max(1u, 2048u / block.x);
            dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));
            config.gridDim = grid;
            config.blockDim = block;

            auto* kernel_instance = useUE8M0
                ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, true>
                : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
            cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput), layout);
        }
        else
        {
            // v1: 16 elements per thread, 256-bit loads, better ILP.
            dim3 block(std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512));
            int const numBlocksPerSM = std::max(1u, 2048u / block.x);
            dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));
            config.gridDim = grid;
            config.blockDim = block;

            auto* kernel_instance = useUE8M0
                ? &opt_quantize_with_block_size_v1<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, true>
                : &opt_quantize_with_block_size_v1<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
            cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput), layout);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Global Amax: single-kernel last-block reduction

template <typename T>
struct VecTypeImpl_
{
    using type = T;
};

template <>
struct VecTypeImpl_<half>
{
    using type = half2;
};

#ifdef ENABLE_BF16
template <>
struct VecTypeImpl_<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};
#endif

template <typename T>
using VecType_ = typename VecTypeImpl_<T>::type;

template <typename T>
__device__ float getMaxAbs_(float4& vec)
{
    auto absMaxVec = cuda_abs(reinterpret_cast<VecType_<T>*>(&vec)[0]);
    for (int i = 1; i < 4; ++i)
    {
        absMaxVec = cuda_max(absMaxVec, cuda_abs(reinterpret_cast<VecType_<T>*>(&vec)[i]));
    }
    float absMaxVal;
    if constexpr (sizeof(T) == 4)
    {
        absMaxVal = static_cast<float>(absMaxVec);
    }
    else
    {
        absMaxVal = static_cast<float>(cuda_max(absMaxVec.x, absMaxVec.y));
    }
    tensorrt_llm::common::blockReduceMaxV2<float, 1>(&absMaxVal);
    return absMaxVal;
}

template <typename T>
__global__ void computeGlobalAmaxKernel(int m, int n, T const* input, float* blockMaxBuf, float* output,
    int* retirementCount, float quantRange, float eps)
{
    static constexpr int ElemsPerVec = 16 / sizeof(T);

    // Phase 1: per-block row max
    float blockMax = 0.f;
    for (int rowIdx = blockIdx.x; rowIdx < m; rowIdx += gridDim.x)
    {
        float rowMaxAbsVal = 0.f;
        for (int vecIdx = threadIdx.x; vecIdx < n / ElemsPerVec; vecIdx += blockDim.x)
        {
            float4 vec = reinterpret_cast<float4 const*>(input + rowIdx * n)[vecIdx];
            float maxAbsVal = getMaxAbs_<T>(vec);
            rowMaxAbsVal = cuda_max(rowMaxAbsVal, maxAbsVal);
        }
        blockMax = cuda_max(blockMax, rowMaxAbsVal);
    }
    if (threadIdx.x == 0)
    {
        blockMaxBuf[blockIdx.x] = blockMax;
    }

    __threadfence();

    // Retire this block; last block does the final reduction.
    __shared__ bool isLastBlock;
    if (threadIdx.x == 0)
    {
        int ticket = atomicAdd(retirementCount, 1);
        isLastBlock = (ticket == gridDim.x - 1);
    }
    __syncthreads();

    if (isLastBlock)
    {
        float maxVal = 0.f;
        for (int i = threadIdx.x; i < gridDim.x; i += blockDim.x)
        {
            maxVal = cuda_max(maxVal, blockMaxBuf[i]);
        }
        blockReduceMaxV2<float, 1>(&maxVal);
        if (threadIdx.x == 0)
        {
            float amax = fmaxf(maxVal, eps);
            output[0] = amax;
            output[1] = (quantRange > 0.f) ? (quantRange / amax) : amax;
            *retirementCount = 0;
        }
    }
}

template <typename T>
void computeGlobalAmax(int m, int n, T const* input, float* blockMaxBuf, float* output, int* retirementCount,
    float quantRange, float eps, int multiProcessorCount, cudaStream_t stream)
{
    static constexpr int ElemsPerVec = 16 / sizeof(T);
    TLLM_CHECK(n % (ElemsPerVec * 32) == 0);
    dim3 block(std::min(n / ElemsPerVec, 1024));
    dim3 grid(std::min(m, multiProcessorCount * 4));

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    int numAttrs = 0;
    attrs[numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[numAttrs].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    numAttrs++;

    size_t inputBytes = static_cast<size_t>(m) * n * sizeof(T);

    struct CachedL2Info
    {
        int l2CacheSize = -1;
        size_t maxWindowSize = 0;
    };

    static thread_local CachedL2Info cachedL2;
    if (cachedL2.l2CacheSize < 0)
    {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        cachedL2.l2CacheSize = prop.l2CacheSize;
        cachedL2.maxWindowSize = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    }
    size_t maxPolicyBytes = std::min(static_cast<size_t>(cachedL2.l2CacheSize),
        cachedL2.maxWindowSize > 0 ? cachedL2.maxWindowSize : static_cast<size_t>(cachedL2.l2CacheSize));
    if (inputBytes > 0 && inputBytes <= maxPolicyBytes && cachedL2.l2CacheSize > 0)
    {
        cudaAccessPolicyWindow policyWindow = {};
        policyWindow.base_ptr = const_cast<void*>(static_cast<void const*>(input));
        policyWindow.num_bytes = inputBytes;
        policyWindow.hitRatio = 1.0f;
        policyWindow.hitProp = cudaAccessPropertyPersisting;
        policyWindow.missProp = cudaAccessPropertyStreaming;
        attrs[numAttrs].id = cudaLaunchAttributeAccessPolicyWindow;
        attrs[numAttrs].val.accessPolicyWindow = policyWindow;
        numAttrs++;
    }
    config.numAttrs = numAttrs;
    config.attrs = attrs;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(
        &config, computeGlobalAmaxKernel<T>, m, n, input, blockMaxBuf, output, retirementCount, quantRange, eps));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused amax + FP4 quantization launchers.

static bool isSM100Plus()
{
    static thread_local std::unordered_map<int, bool> cache;
    int deviceId = 0;
    cudaGetDevice(&deviceId);
    auto it = cache.find(deviceId);
    if (it != cache.end())
    {
        return it->second;
    }
    int major = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceId);
    bool result = major >= 10;
    cache[deviceId] = result;
    return result;
}

struct OccupancyKey
{
    void const* func;
    int blockSize;
    bool operator==(OccupancyKey const& other) const
    {
        return func == other.func && blockSize == other.blockSize;
    }
};

struct OccupancyKeyHash
{
    size_t operator()(OccupancyKey const& key) const
    {
        return std::hash<void const*>()(key.func) ^ (std::hash<int>()(key.blockSize) << 1);
    }
};

template <typename KernelFunc>
static int cachedMaxActiveBlocks(KernelFunc func, int blockSize)
{
    static thread_local std::unordered_map<OccupancyKey, int, OccupancyKeyHash> cache;
    OccupancyKey key{reinterpret_cast<void const*>(func), blockSize};
    auto it = cache.find(key);
    if (it != cache.end())
    {
        return it->second;
    }
    int maxActive = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActive, func, blockSize, 0);
    cache[key] = maxActive;
    return maxActive;
}

/// Pick the block shape and grid size a persistent v2 prologue kernel runs with.
///
/// Returns false when the kernel cannot cover every row it must, which is the
/// signal for callers to fall back to the unfused launchers.
template <typename KernelFunc>
static bool planFusedPrologueV2Launch(KernelFunc kernelFunc, int numBlocksForM, int n, int multiProcessorCount,
    int testMaxActiveBlocks, dim3& block, dim3& grid)
{
    int defaultBlockSize = std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512);
    block = dim3(defaultBlockSize);
    int maxActiveBlocksPerSM = 0;
    if (testMaxActiveBlocks > 0)
    {
        maxActiveBlocksPerSM = testMaxActiveBlocks;
    }
    else
    {
        int defaultOcc = cachedMaxActiveBlocks(kernelFunc, defaultBlockSize);
        int altBlockSize = 256;
        if (altBlockSize < defaultBlockSize)
        {
            int altOcc = cachedMaxActiveBlocks(kernelFunc, altBlockSize);
            if (altOcc * altBlockSize > defaultOcc * defaultBlockSize)
            {
                block = dim3(altBlockSize);
                maxActiveBlocksPerSM = altOcc;
            }
            else
            {
                maxActiveBlocksPerSM = defaultOcc;
            }
        }
        else
        {
            maxActiveBlocksPerSM = defaultOcc;
        }
    }
    // Callers allocate blockMaxBuf for the same 4 CTA/SM upper bound used by
    // the non-fused quantization launcher.
    int maxTotalBlocks = std::min(maxActiveBlocksPerSM * multiProcessorCount, 4 * multiProcessorCount);
    if (maxTotalBlocks < 1)
    {
        return false;
    }
    if (testMaxActiveBlocks > 0 && maxTotalBlocks < numBlocksForM)
    {
        return false;
    }
    grid = dim3(std::min(numBlocksForM, maxTotalBlocks));
    return true;
}

/// Append an L2 persisting-access window over [base, base + bytes) when the
/// device can hold the whole range; otherwise leave the attribute list alone.
static void addPersistingL2Window(void const* base, size_t bytes, cudaLaunchAttribute* attrs, int& numAttrs)
{
    struct CachedL2Info
    {
        int l2CacheSize = -1;
        size_t maxWindowSize = 0;
    };

    static thread_local CachedL2Info cachedL2;
    if (cachedL2.l2CacheSize < 0)
    {
        int deviceId = 0;
        cudaGetDevice(&deviceId);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        cachedL2.l2CacheSize = prop.l2CacheSize;
        cachedL2.maxWindowSize = static_cast<size_t>(prop.accessPolicyMaxWindowSize);
    }
    size_t maxPolicyBytes = std::min(static_cast<size_t>(cachedL2.l2CacheSize),
        cachedL2.maxWindowSize > 0 ? cachedL2.maxWindowSize : static_cast<size_t>(cachedL2.l2CacheSize));
    if (bytes == 0 || bytes > maxPolicyBytes || cachedL2.l2CacheSize <= 0)
    {
        return;
    }
    cudaAccessPolicyWindow policyWindow = {};
    policyWindow.base_ptr = const_cast<void*>(base);
    policyWindow.num_bytes = bytes;
    policyWindow.hitRatio = 1.0f;
    policyWindow.hitProp = cudaAccessPropertyPersisting;
    policyWindow.missProp = cudaAccessPropertyStreaming;
    attrs[numAttrs].id = cudaLaunchAttributeAccessPolicyWindow;
    attrs[numAttrs].val.accessPolicyWindow = policyWindow;
    numAttrs++;
}

template <typename T, int SF_VEC_SIZE>
bool invokeFusedPrologueQuantization(int m, int n, T const* input, float quantRange, float eps, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream, int testMaxActiveBlocks)
{
    if constexpr (SF_VEC_SIZE != 16)
    {
        return false;
    }
    if (useUE8M0 || !isSM100Plus())
    {
        return false;
    }

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numBlocksForM = isSf128x4Layout ? PadUpFn(m, 128) : (isSf8x4Layout ? PadUpFn(m, 8) : m);
    dim3 block(std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512));
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));

    auto* kernelFunc = &fused_prologue_quantize_v1<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
    int maxActiveBlocksPerSM = testMaxActiveBlocks > 0 ? testMaxActiveBlocks : cachedMaxActiveBlocks(kernelFunc, block.x);
    if (maxActiveBlocksPerSM * multiProcessorCount < static_cast<int>(grid.x))
    {
        return false;
    }

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelFunc, m, n, n, input, quantRange, eps,
        reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput), layout, blockMaxBuf,
        retirementCount, globalScaleOut));
    return true;
}

template <typename T, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
bool invokeFp4QuantizePhase2(int m, int n, T const* input, float const* amaxScale, int64_t* output,
    int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
{
    if constexpr (SF_VEC_SIZE != 16)
    {
        return false;
    }
    if (!isSM100Plus())
    {
        return false;
    }

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numBlocksForM = isSf128x4Layout ? PadUpFn(m, 128) : (isSf8x4Layout ? PadUpFn(m, 8) : m);
    dim3 block(std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512));
    int const numBlocksPerSM = std::max(1u, 2048u / block.x);
    dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;

    auto* kernelFunc = &phase2_only_quantize<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, Rule>;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelFunc, m, n, n, input, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOutput), layout, amaxScale));
    return true;
}

template <typename T, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
bool invokeFusedPrologueQuantizationV2(int m, int n, T const* input, float quantRange, float eps, int64_t* output,
    int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf, int* retirementCount,
    float* globalScaleOut, cudaStream_t stream, int testMaxActiveBlocks)
{
    if constexpr (SF_VEC_SIZE != 16)
    {
        return false;
    }
    if (!isSM100Plus())
    {
        return false;
    }

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numBlocksForM = isSf128x4Layout ? PadUpFn(m, 128) : (isSf8x4Layout ? PadUpFn(m, 8) : m);

    auto* kernelFunc = &fused_prologue_quantize_v2<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, Rule>;
    dim3 block;
    dim3 grid;
    if (!planFusedPrologueV2Launch(kernelFunc, numBlocksForM, n, multiProcessorCount, testMaxActiveBlocks, block, grid))
    {
        return false;
    }

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;

    cudaLaunchAttribute attrs[2];
    int numAttrs = 0;
    attrs[numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[numAttrs].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    numAttrs++;

    // Both phases stream over the input, so it is the range worth pinning.
    addPersistingL2Window(input, static_cast<size_t>(m) * n * sizeof(T), attrs, numAttrs);
    config.numAttrs = numAttrs;
    config.attrs = attrs;

    TLLM_CUDA_CHECK(
        cudaLaunchKernelEx(&config, kernelFunc, m, n, n, input, quantRange, eps, reinterpret_cast<uint32_t*>(output),
            reinterpret_cast<uint32_t*>(SFOutput), layout, blockMaxBuf, retirementCount, globalScaleOut));
    return true;
}

template <typename T, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
bool invokeFusedMoePrologueQuantizationV2(int m, int n, T const* input, int32_t const* tileIdxToMnLimit,
    int32_t const* numNonExitingTiles, int tileSize, float quantRange, float eps, int64_t* output, int32_t* SFOutput,
    QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf, int* retirementCount,
    float* globalScaleOut, cudaStream_t stream, int testMaxActiveBlocks)
{
    if constexpr (SF_VEC_SIZE != 16)
    {
        return false;
    }
    if (!isSM100Plus())
    {
        return false;
    }

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numBlocksForM = isSf128x4Layout ? PadUpFn(m, 128) : (isSf8x4Layout ? PadUpFn(m, 8) : m);
    auto* kernelFunc = &fused_moe_prologue_quantize_v2<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, Rule>;
    dim3 block;
    dim3 grid;
    if (!planFusedPrologueV2Launch(kernelFunc, numBlocksForM, n, multiProcessorCount, testMaxActiveBlocks, block, grid))
    {
        return false;
    }

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    int numAttrs = 0;
    attrs[numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[numAttrs].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    numAttrs++;

    // Both phases stream over the input, so it is the range worth pinning.
    addPersistingL2Window(input, static_cast<size_t>(m) * n * sizeof(T), attrs, numAttrs);
    config.numAttrs = numAttrs;
    config.attrs = attrs;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelFunc, m, n, n, input, tileIdxToMnLimit, numNonExitingTiles,
        tileSize, quantRange, eps, reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOutput), layout,
        blockMaxBuf, retirementCount, globalScaleOut));
    return true;
}

template <typename T, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
bool invokeFusedMoeSwigluPrologueQuantizationV2(int m, int n, T const* preactivation, T* swigluOutput,
    int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int tileSize, float quantRange, float eps,
    int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream, int testMaxActiveBlocks)
{
    if constexpr (SF_VEC_SIZE != 16)
    {
        return false;
    }
    if (!isSM100Plus())
    {
        return false;
    }

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numBlocksForM = isSf128x4Layout ? PadUpFn(m, 128) : (isSf8x4Layout ? PadUpFn(m, 8) : m);
    auto* kernelFunc
        = &fused_moe_swiglu_prologue_quantize_v2<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, Rule>;
    dim3 block;
    dim3 grid;
    if (!planFusedPrologueV2Launch(kernelFunc, numBlocksForM, n, multiProcessorCount, testMaxActiveBlocks, block, grid))
    {
        return false;
    }

    cudaLaunchConfig_t config;
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    int numAttrs = 0;
    attrs[numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[numAttrs].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    numAttrs++;

    // The pre-activation is read once, but the activation phase 1 stores is read
    // straight back by phase 2, so that is the range worth pinning in L2.
    addPersistingL2Window(swigluOutput, static_cast<size_t>(m) * n * sizeof(T), attrs, numAttrs);
    config.numAttrs = numAttrs;
    config.attrs = attrs;

    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernelFunc, m, n, n, preactivation, swigluOutput, tileIdxToMnLimit,
        numNonExitingTiles, tileSize, quantRange, eps, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOutput), layout, blockMaxBuf, retirementCount, globalScaleOut));
    return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dequant FP4 + swizzled SF → BF16/FP16

template <typename OutType>
void invokeDequantNvfp4SwizzledSF(int numRows, int numCols,
    uint8_t const* fp4_packed, uint8_t const* sf_swizzled,
    float const* globalScale, int sfVecSize, OutType* output,
    QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream)
{
    int packedCols = numCols / 2;
    dim3 block(std::min(packedCols, 512));
    dim3 grid(std::min(numRows, multiProcessorCount * 4));

    dequant_nvfp4_swizzled_sf_kernel<OutType><<<grid, block, 0, stream>>>(
        numRows, numCols, fp4_packed, sf_swizzled, globalScale, sfVecSize, layout, output);
}

template void invokeDequantNvfp4SwizzledSF<__nv_bfloat16>(int, int,
    uint8_t const*, uint8_t const*, float const*, int, __nv_bfloat16*, QuantizationSFLayout, int, cudaStream_t);
template void invokeDequantNvfp4SwizzledSF<half>(int, int,
    uint8_t const*, uint8_t const*, float const*, int, half*, QuantizationSFLayout, int, cudaStream_t);

////////////////////////////////////////////////////////////////////////////////////////////////////
// Alpha correction: out[i] = alphaIn[i] * fc2InputScale / dynamicGs[0]

__global__ void scaleAlphaByAmaxKernel(
    float const* __restrict__ alphaIn,
    float const* __restrict__ dynamicGsPtr,
    float fc2InputScale,
    int numExperts,
    float* __restrict__ alphaOut)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < numExperts)
    {
        alphaOut[i] = alphaIn[i] * fc2InputScale / dynamicGsPtr[0];
    }
}

void invokeScaleAlphaByAmax(float const* alphaIn, float const* dynamicGlobalScalePtr,
    float fc2InputScale, int numExperts, float* alphaOut, cudaStream_t stream)
{
    int threads = std::min(numExperts, 256);
    int blocks = (numExperts + threads - 1) / threads;
    scaleAlphaByAmaxKernel<<<blocks, threads, 0, stream>>>(
        alphaIn, dynamicGlobalScalePtr, fc2InputScale, numExperts, alphaOut);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Template instantiations

// half
template void invokeFP4QuantizationEx<half, 16>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream, int kernelVersion, int scaleRule);
template void invokeFP4QuantizationEx<half, 32>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream, int kernelVersion, int scaleRule);
template void computeGlobalAmax<half>(int m, int n, half const* input, float* blockMaxBuf, float* output,
    int* retirementCount, float quantRange, float eps, int multiProcessorCount, cudaStream_t stream);
template bool invokeFusedPrologueQuantization<half, 16>(int m, int n, half const* input, float quantRange,
    float eps, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, float* blockMaxBuf, int* retirementCount, float* globalScaleOut, cudaStream_t stream,
    int testMaxActiveBlocks);
#define INSTANTIATE_PHASE2(T, RULE)                                                                                   \
    template bool invokeFp4QuantizePhase2<T, 16, AdaptiveScaleRule::RULE>(int m, int n, T const* input,               \
        float const* amaxScale, int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout,                       \
        int multiProcessorCount, cudaStream_t stream)
INSTANTIATE_PHASE2(half, NONE);
INSTANTIATE_PHASE2(half, MSE);
#define INSTANTIATE_FUSED_V2(T, RULE)                                                                                 \
    template bool invokeFusedPrologueQuantizationV2<T, 16, AdaptiveScaleRule::RULE>(int m, int n, T const* input,      \
        float quantRange, float eps, int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout,                  \
        int multiProcessorCount, float* blockMaxBuf, int* retirementCount, float* globalScaleOut, cudaStream_t stream, \
        int testMaxActiveBlocks)
INSTANTIATE_FUSED_V2(half, NONE);
INSTANTIATE_FUSED_V2(half, MSE);
INSTANTIATE_FUSED_V2(half, MAE);
INSTANTIATE_FUSED_V2(half, ABS_MAX);
#define INSTANTIATE_FUSED_MOE_V2(T, RULE)                                                                              \
    template bool invokeFusedMoePrologueQuantizationV2<T, 16, AdaptiveScaleRule::RULE>(int m, int n, T const* input,   \
        int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int tileSize, float quantRange, float eps, \
        int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,  \
        int* retirementCount, float* globalScaleOut, cudaStream_t stream, int testMaxActiveBlocks)
INSTANTIATE_FUSED_MOE_V2(half, NONE);
INSTANTIATE_FUSED_MOE_V2(half, MSE);
INSTANTIATE_FUSED_MOE_V2(half, MAE);
INSTANTIATE_FUSED_MOE_V2(half, ABS_MAX);
#define INSTANTIATE_FUSED_MOE_SWIGLU_V2(T, RULE)                                                                       \
    template bool invokeFusedMoeSwigluPrologueQuantizationV2<T, 16, AdaptiveScaleRule::RULE>(int m, int n,             \
        T const* preactivation, T* swigluOutput, int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles,   \
        int tileSize, float quantRange, float eps, int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout,    \
        int multiProcessorCount, float* blockMaxBuf, int* retirementCount, float* globalScaleOut, cudaStream_t stream, \
        int testMaxActiveBlocks)
INSTANTIATE_FUSED_MOE_SWIGLU_V2(half, NONE);
INSTANTIATE_FUSED_MOE_SWIGLU_V2(half, MSE);

#ifdef ENABLE_BF16
template void invokeFP4QuantizationEx<__nv_bfloat16, 16>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream, int kernelVersion, int scaleRule);
template void invokeFP4QuantizationEx<__nv_bfloat16, 32>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream, int kernelVersion, int scaleRule);
template void computeGlobalAmax<__nv_bfloat16>(int m, int n, __nv_bfloat16 const* input, float* blockMaxBuf,
    float* output, int* retirementCount, float quantRange, float eps, int multiProcessorCount, cudaStream_t stream);
template bool invokeFusedPrologueQuantization<__nv_bfloat16, 16>(int m, int n, __nv_bfloat16 const* input,
    float quantRange, float eps, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, float* blockMaxBuf, int* retirementCount, float* globalScaleOut, cudaStream_t stream,
    int testMaxActiveBlocks);
INSTANTIATE_FUSED_V2(__nv_bfloat16, NONE);
INSTANTIATE_FUSED_V2(__nv_bfloat16, MSE);
INSTANTIATE_FUSED_V2(__nv_bfloat16, MAE);
INSTANTIATE_FUSED_V2(__nv_bfloat16, ABS_MAX);
INSTANTIATE_FUSED_MOE_V2(__nv_bfloat16, NONE);
INSTANTIATE_FUSED_MOE_V2(__nv_bfloat16, MSE);
INSTANTIATE_FUSED_MOE_V2(__nv_bfloat16, MAE);
INSTANTIATE_FUSED_MOE_V2(__nv_bfloat16, ABS_MAX);
INSTANTIATE_FUSED_MOE_SWIGLU_V2(__nv_bfloat16, NONE);
INSTANTIATE_FUSED_MOE_SWIGLU_V2(__nv_bfloat16, MSE);
INSTANTIATE_PHASE2(__nv_bfloat16, NONE);
INSTANTIATE_PHASE2(__nv_bfloat16, MSE);
#endif
#undef INSTANTIATE_FUSED_MOE_SWIGLU_V2
#undef INSTANTIATE_FUSED_MOE_V2
#undef INSTANTIATE_FUSED_V2
#undef INSTANTIATE_PHASE2

#ifdef ENABLE_FP8
template void invokeFP4QuantizationEx<__nv_fp8_e4m3, 16>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream, int kernelVersion, int scaleRule);
template void invokeFP4QuantizationEx<__nv_fp8_e4m3, 32>(int b, int m, int n, __nv_fp8_e4m3 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream, int kernelVersion, int scaleRule);
#endif

} // namespace kernels

TRTLLM_NAMESPACE_END
