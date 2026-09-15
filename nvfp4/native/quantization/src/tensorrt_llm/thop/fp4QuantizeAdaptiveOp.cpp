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
 * PyTorch op wrappers for adaptive 4/6 FP4 quantization + global amax.
 *
 * Registers:
 *   torch.ops.trtllm.fp4_quantize_ex(...)   — extended fp4_quantize with kernelVersion + scaleRule
 *   torch.ops.trtllm.calculate_global_amax(...)  — single-kernel runtime amax + global_scale
 *
 * Derived from tllm_linear_lite/quantize/fp4_quantize_op.cu.
 */

// Minimal CHECK macros (avoid thUtils.h which pulls in NvInferRuntime.h)
#ifndef CHECK_TH_CUDA
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, #x " dtype mismatch")
#define CHECK_INPUT(x, st) CHECK_TH_CUDA(x); CHECK_CONTIGUOUS(x); CHECK_TYPE(x, st)
#endif
#include "tensorrt_llm/kernels/fp4QuantizeAdaptive.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <torch/torch.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>
#include <optional>
#include <unordered_map>

namespace torch_ext
{

// FP4 E2M1 packed as uint8 (2 values per byte)
constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte;
// Scale factor dtype: FP8 E4M3 stored as uint8
constexpr auto SF_DTYPE = torch::ScalarType::Byte;

// ---------------------------------------------------------------------------
// fp4_quantize_ex
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor> fp4_quantize_ex(at::Tensor const& self,
    std::optional<at::Tensor> const& globalScale, int64_t sfVecSize, bool sfUseUE8M0, bool isSfSwizzledLayout,
    int64_t kernelVersion, int64_t scaleRule)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    if (sfUseUE8M0)
    {
        TORCH_CHECK(sfVecSize == 32, "sfVecSize can only be 32 when sfUseUE8M0 is true");
    }
    else
    {
        TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfUseUE8M0 is false");
        CHECK_INPUT(globalScale.value(), torch::kFloat32);
        TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16 when sfUseUE8M0 is false");
    }

    float* globalScalePtr = nullptr;
    if (globalScale.has_value())
    {
        globalScalePtr = globalScale->data_ptr<float>();
    }

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;
    at::Tensor valueE2M1 = at::empty(outputShape, self.options().dtype(FLOAT4_E2M1X2));

    int64_t SFSize = isSfSwizzledLayout
        ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);
    at::Tensor scaleFP8SF = at::empty({SFSize}, self.options().dtype(SF_DTYPE));

    thread_local int const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device()).stream();

#define LAUNCH_FP4_QUANTIZE_EX(T, SF_VEC_SIZE)                                                                        \
    tensorrt_llm::kernels::invokeFP4QuantizationEx<T, SF_VEC_SIZE>(1, m, k,                                           \
        reinterpret_cast<T*>(self.data_ptr()), globalScalePtr, reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),        \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), sfUseUE8M0, layout, mMultiProcessorCount, stream,           \
        static_cast<int>(kernelVersion), static_cast<int>(scaleRule))

    if (sfUseUE8M0)
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_EX(half, 32);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_bfloat16, 32);
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_fp8_e4m3, 32);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_ex only supports fp16/bf16/fp8_e4m3 input.");
        }
    }
    else
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_EX(half, 16);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_bfloat16, 16);
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
            LAUNCH_FP4_QUANTIZE_EX(__nv_fp8_e4m3, 16);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_ex only supports fp16/bf16/fp8_e4m3 input.");
        }
    }

#undef LAUNCH_FP4_QUANTIZE_EX

    return {valueE2M1, scaleFP8SF};
}

// ---------------------------------------------------------------------------
// calculate_global_amax
// ---------------------------------------------------------------------------

at::Tensor calculate_global_amax(at::Tensor const& input, double quantRange, double eps)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);

    auto const& inputShape = input.sizes();
    auto const rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input must be >= 2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const n = inputShape[rank - 1];

    static int multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();
    int maxGridX = std::min(static_cast<int>(m), multiProcessorCount * 4);

    // Persistent internal buffers — allocated once, reused across calls.
    static at::Tensor blockMaxBuf;
    static at::Tensor retirementCount;
    static at::Tensor outputBuf;
    static int allocatedGridX = 0;

    if (allocatedGridX < maxGridX)
    {
        auto opts = input.options();
        blockMaxBuf = at::empty({maxGridX}, opts.dtype(torch::kFloat32));
        outputBuf = at::empty({2}, opts.dtype(torch::kFloat32));
        retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
        allocatedGridX = maxGridX;
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

    if (input.scalar_type() == at::ScalarType::Half)
    {
        tensorrt_llm::kernels::computeGlobalAmax<half>(m, n, reinterpret_cast<half const*>(input.data_ptr()),
            blockMaxBuf.data_ptr<float>(), outputBuf.data_ptr<float>(), retirementCount.data_ptr<int>(),
            static_cast<float>(quantRange), static_cast<float>(eps), multiProcessorCount, stream);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        tensorrt_llm::kernels::computeGlobalAmax<__nv_bfloat16>(m, n,
            reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()), blockMaxBuf.data_ptr<float>(),
            outputBuf.data_ptr<float>(), retirementCount.data_ptr<int>(), static_cast<float>(quantRange),
            static_cast<float>(eps), multiProcessorCount, stream);
    }
    else
    {
        TORCH_CHECK(false, "calculate_global_amax only supports fp16/bf16 input.");
    }

    return outputBuf;
}

// ---------------------------------------------------------------------------
// Persistent fused-prologue workspace
// ---------------------------------------------------------------------------

/// Scratch the persistent fused kernels reduce through, kept per (device,
/// stream) so a CUDA-graph capture always replays the same buffers.
struct FusedWorkspace
{
    at::Tensor blockMaxBuf;
    at::Tensor retirementCount;
    int capacity = 0;
};

/// Grid width the fused prologue launchers derive from the output row/column
/// counts. The workspace must hold one float per block of that grid.
inline int fusedPrologueGridX(int numBlocksForM, int64_t numCols, int multiProcessorCount)
{
    int blockX = std::min(static_cast<int>(numCols / 16), 512);
    int numBlocksPerSM = std::max(1, 2048 / blockX);
    return std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM);
}

inline FusedWorkspace& fusedWorkspace(at::Device device, cudaStream_t stream, int gridX)
{
    struct WsKey
    {
        int device;
        uintptr_t stream;

        bool operator==(WsKey const& other) const
        {
            return device == other.device && stream == other.stream;
        }
    };

    struct WsKeyHash
    {
        size_t operator()(WsKey const& key) const
        {
            return std::hash<int>()(key.device) ^ (std::hash<uintptr_t>()(key.stream) << 1);
        }
    };

    static constexpr size_t kMaxWsCacheEntries = 32;
    thread_local std::unordered_map<WsKey, FusedWorkspace, WsKeyHash> ws_cache;
    WsKey wsKey{device.index(), reinterpret_cast<uintptr_t>(stream)};
    if (ws_cache.size() >= kMaxWsCacheEntries && ws_cache.find(wsKey) == ws_cache.end())
    {
        ws_cache.erase(ws_cache.begin());
    }
    auto& ws = ws_cache[wsKey];
    if (ws.capacity < gridX)
    {
        auto opts = at::TensorOptions().device(device).dtype(torch::kFloat32);
        ws.blockMaxBuf = at::empty({gridX}, opts);
        ws.retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
        ws.capacity = gridX;
    }
    return ws;
}

// ---------------------------------------------------------------------------
// fp4_quantize_fused
// ---------------------------------------------------------------------------

std::tuple<at::Tensor, at::Tensor, at::Tensor> fp4_quantize_fused(at::Tensor const& self, int64_t sfVecSize,
    bool sfUseUE8M0, bool isSfSwizzledLayout, int64_t scaleRule, double quantRange, double eps,
    int64_t testMaxActiveBlocks, int64_t forceV2, std::optional<at::Tensor> const& tileIdxToMnLimit,
    std::optional<at::Tensor> const& numNonExitingTiles, int64_t tileSize)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    c10::cuda::CUDAGuard device_guard(self.device());

    TORCH_CHECK(scaleRule >= 0 && scaleRule <= 3,
        "Invalid scaleRule: ", scaleRule, ". Must be 0 (static_6), 1 (MSE), 2 (MAE), or 3 (ABS_MAX).");
    bool configEligible = (sfVecSize == 16) && (!sfUseUE8M0);

    auto const& inputShape = self.sizes();
    auto const rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input must be >= 2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    bool const moeMasked = tileIdxToMnLimit.has_value() || numNonExitingTiles.has_value();
    TORCH_CHECK(tileIdxToMnLimit.has_value() == numNonExitingTiles.has_value(),
        "tileIdxToMnLimit and numNonExitingTiles must be provided together");
    if (moeMasked)
    {
        CHECK_INPUT(tileIdxToMnLimit.value(), torch::kInt32);
        CHECK_INPUT(numNonExitingTiles.value(), torch::kInt32);
        TORCH_CHECK(tileIdxToMnLimit->get_device() == self.get_device()
                && numNonExitingTiles->get_device() == self.get_device(),
            "MoE routing metadata and input must be on the same CUDA device");
        TORCH_CHECK(tileSize > 0 && m % tileSize == 0, "MoE rows must be a multiple of tileSize");
        TORCH_CHECK(
            tileIdxToMnLimit->numel() == m / tileSize, "tileIdxToMnLimit must contain one entry per routing tile");
        TORCH_CHECK(numNonExitingTiles->numel() == 1, "numNonExitingTiles must contain one value");
        TORCH_CHECK((scaleRule == 0 || scaleRule == 1) && sfVecSize == 16 && !sfUseUE8M0 && isSfSwizzledLayout,
            "MoE-masked runtime quantization requires standard or MSE scaling, SF vector 16, E4M3 SF, and "
            "swizzled layout");
    }

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;
    at::Tensor valueE2M1 = at::empty(outputShape, self.options().dtype(FLOAT4_E2M1X2));

    auto const layout = isSfSwizzledLayout ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
                                           : tensorrt_llm::QuantizationSFLayout::LINEAR;

    int64_t SFSize = isSfSwizzledLayout
        ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);
    at::Tensor scaleFP8SF = at::empty({SFSize}, self.options().dtype(SF_DTYPE));
    at::Tensor amaxScale = at::empty({2}, self.options().dtype(torch::kFloat32));

    int mMultiProcessorCount = 0;
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, self.get_device());
    auto stream = at::cuda::getCurrentCUDAStream(self.get_device()).stream();

    int numBlocksForM = isSfSwizzledLayout ? PadUpFn(static_cast<int>(m), 128) : static_cast<int>(m);
    int gridX = fusedPrologueGridX(numBlocksForM, k, mMultiProcessorCount);
    auto& ws = fusedWorkspace(self.device(), stream, gridX);

    bool fusedTaken = false;
    if (configEligible)
    {
#define TRY_FUSED_V1_LAUNCH(T)                                                                                       \
    fusedTaken = tensorrt_llm::kernels::invokeFusedPrologueQuantization<T, 16>(m, k,                                  \
        reinterpret_cast<T const*>(self.data_ptr()), static_cast<float>(quantRange), static_cast<float>(eps),          \
        reinterpret_cast<int64_t*>(valueE2M1.data_ptr()), reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()),           \
        sfUseUE8M0, layout, mMultiProcessorCount, ws.blockMaxBuf.data_ptr<float>(),                                    \
        ws.retirementCount.data_ptr<int>(), amaxScale.data_ptr<float>(), stream, static_cast<int>(testMaxActiveBlocks))

#define TRY_FUSED_V2_LAUNCH(T, RULE)                                                                                   \
    do                                                                                                                 \
    {                                                                                                                  \
        if (moeMasked)                                                                                                 \
        {                                                                                                              \
            fusedTaken = tensorrt_llm::kernels::invokeFusedMoePrologueQuantizationV2<T, 16,                            \
                tensorrt_llm::kernels::AdaptiveScaleRule::RULE>(m, k, reinterpret_cast<T const*>(self.data_ptr()),     \
                tileIdxToMnLimit->data_ptr<int32_t>(), numNonExitingTiles->data_ptr<int32_t>(),                        \
                static_cast<int>(tileSize), static_cast<float>(quantRange), static_cast<float>(eps),                   \
                reinterpret_cast<int64_t*>(valueE2M1.data_ptr()), reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()),   \
                layout, mMultiProcessorCount, ws.blockMaxBuf.data_ptr<float>(), ws.retirementCount.data_ptr<int>(),    \
                amaxScale.data_ptr<float>(), stream, static_cast<int>(testMaxActiveBlocks));                           \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            fusedTaken = tensorrt_llm::kernels::invokeFusedPrologueQuantizationV2<T, 16,                               \
                tensorrt_llm::kernels::AdaptiveScaleRule::RULE>(m, k, reinterpret_cast<T const*>(self.data_ptr()),     \
                static_cast<float>(quantRange), static_cast<float>(eps),                                               \
                reinterpret_cast<int64_t*>(valueE2M1.data_ptr()), reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()),   \
                layout, mMultiProcessorCount, ws.blockMaxBuf.data_ptr<float>(), ws.retirementCount.data_ptr<int>(),    \
                amaxScale.data_ptr<float>(), stream, static_cast<int>(testMaxActiveBlocks));                           \
        }                                                                                                              \
    } while (0)

#define FUSED_DISPATCH(T)                                                                                              \
    do                                                                                                                 \
    {                                                                                                                  \
        if (scaleRule == 0)                                                                                            \
        {                                                                                                              \
            if (forceV2 == 0 && !moeMasked)                                                                            \
            {                                                                                                          \
                TRY_FUSED_V1_LAUNCH(T);                                                                                \
            }                                                                                                          \
            if (!fusedTaken)                                                                                           \
            {                                                                                                          \
                TRY_FUSED_V2_LAUNCH(T, NONE);                                                                          \
            }                                                                                                          \
        }                                                                                                              \
        else if (scaleRule == 1)                                                                                       \
        {                                                                                                              \
            TRY_FUSED_V2_LAUNCH(T, MSE);                                                                               \
        }                                                                                                              \
        else if (scaleRule == 2)                                                                                       \
        {                                                                                                              \
            TRY_FUSED_V2_LAUNCH(T, MAE);                                                                               \
        }                                                                                                              \
        else if (scaleRule == 3)                                                                                       \
        {                                                                                                              \
            TRY_FUSED_V2_LAUNCH(T, ABS_MAX);                                                                           \
        }                                                                                                              \
    } while (0)

        if (self.scalar_type() == at::ScalarType::Half)
        {
            FUSED_DISPATCH(half);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            FUSED_DISPATCH(__nv_bfloat16);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_fused only supports fp16/bf16 input.");
        }

#undef TRY_FUSED_V1_LAUNCH
#undef TRY_FUSED_V2_LAUNCH
#undef FUSED_DISPATCH
    }

    if (!fusedTaken)
    {
        TORCH_CHECK(!moeMasked, "MoE-masked adaptive quantization requires the SM100 persistent fused kernel");
        int fallbackGridX = std::min(static_cast<int>(m), mMultiProcessorCount * 4);
        if (ws.capacity < fallbackGridX)
        {
            auto opts = at::TensorOptions().device(self.device()).dtype(torch::kFloat32);
            ws.blockMaxBuf = at::empty({fallbackGridX}, opts);
            ws.retirementCount = at::zeros({1}, opts.dtype(torch::kInt32));
            ws.capacity = fallbackGridX;
        }

        if (self.scalar_type() == at::ScalarType::Half)
        {
            tensorrt_llm::kernels::computeGlobalAmax<half>(m, k, reinterpret_cast<half const*>(self.data_ptr()),
                ws.blockMaxBuf.data_ptr<float>(), amaxScale.data_ptr<float>(), ws.retirementCount.data_ptr<int>(),
                static_cast<float>(quantRange), static_cast<float>(eps), mMultiProcessorCount, stream);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
            tensorrt_llm::kernels::computeGlobalAmax<__nv_bfloat16>(m, k,
                reinterpret_cast<__nv_bfloat16 const*>(self.data_ptr()), ws.blockMaxBuf.data_ptr<float>(),
                amaxScale.data_ptr<float>(), ws.retirementCount.data_ptr<int>(), static_cast<float>(quantRange),
                static_cast<float>(eps), mMultiProcessorCount, stream);
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize_fused only supports fp16/bf16 input.");
        }

        at::Tensor globalScaleTensor = amaxScale.slice(0, 1, 2);
        auto [packedFallback, sfFallback] = fp4_quantize_ex(
            self, globalScaleTensor, sfVecSize, sfUseUE8M0, isSfSwizzledLayout, 1, scaleRule);
        return {packedFallback, sfFallback, amaxScale};
    }

    return {valueE2M1, scaleFP8SF, amaxScale};
}

// ---------------------------------------------------------------------------
// fp4_quantize_phase2
// ---------------------------------------------------------------------------

/// Opt-in, phase2-only NVFP4 quantization for a caller-precomputed amax/scale.
///
/// Unlike fp4_quantize_fused, this performs no amax reduction, no retirement
/// counter, and no grid-wide barrier: it launches a single non-persistent
/// kernel that reads @p amaxScale[1] as the global scale and quantizes
/// @p input directly, using either the standard rule (scaleRule=0) or the
/// adaptive MSE 4/6 rule (scaleRule=1). Unsupported configurations are
/// rejected with a clear error rather than silently falling back.
std::tuple<at::Tensor, at::Tensor> fp4_quantize_phase2(at::Tensor const& input, at::Tensor const& amaxScale,
    int64_t sfVecSize, bool isSfSwizzledLayout, int64_t scaleRule)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);
    c10::cuda::CUDAGuard device_guard(input.device());

    TORCH_CHECK(sfVecSize == 16, "fp4_quantize_phase2 only supports sfVecSize=16, got ", sfVecSize);
    TORCH_CHECK(isSfSwizzledLayout, "fp4_quantize_phase2 only supports the swizzled SF layout");
    TORCH_CHECK(scaleRule == 0 || scaleRule == 1,
        "fp4_quantize_phase2 only supports the standard rule (0) or the adaptive MSE 4/6 rule (1), got ", scaleRule);
    CHECK_INPUT(amaxScale, torch::kFloat32);
    TORCH_CHECK(amaxScale.numel() == 2, "amaxScale must contain exactly {amax, global_scale}, got ",
        amaxScale.numel(), " elements");
    TORCH_CHECK(amaxScale.get_device() == input.get_device(), "amaxScale and input must be on the same CUDA device");

    auto const& inputShape = input.sizes();
    auto const rank = inputShape.size();
    TORCH_CHECK(rank >= 2, "Input must be >= 2D tensor.");

    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;
    at::Tensor valueE2M1 = at::empty(outputShape, input.options().dtype(FLOAT4_E2M1X2));

    auto const layout = tensorrt_llm::QuantizationSFLayout::SWIZZLED;
    int64_t SFSize = tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize);
    at::Tensor scaleFP8SF = at::empty({SFSize}, input.options().dtype(SF_DTYPE));

    int mMultiProcessorCount = 0;
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, input.get_device());
    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

    bool launched = false;

#define TRY_PHASE2_LAUNCH(T, RULE)                                                                                    \
    launched = tensorrt_llm::kernels::invokeFp4QuantizePhase2<T, 16, tensorrt_llm::kernels::AdaptiveScaleRule::RULE>( \
        static_cast<int>(m), static_cast<int>(k), reinterpret_cast<T const*>(input.data_ptr()),                       \
        amaxScale.data_ptr<float>(), reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), layout, mMultiProcessorCount, stream)

#define PHASE2_DISPATCH(T)                                                                                            \
    do                                                                                                                \
    {                                                                                                                 \
        if (scaleRule == 0)                                                                                           \
        {                                                                                                             \
            TRY_PHASE2_LAUNCH(T, NONE);                                                                                \
        }                                                                                                             \
        else                                                                                                          \
        {                                                                                                             \
            TRY_PHASE2_LAUNCH(T, MSE);                                                                                 \
        }                                                                                                             \
    } while (0)

    if (input.scalar_type() == at::ScalarType::Half)
    {
        PHASE2_DISPATCH(half);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
        PHASE2_DISPATCH(__nv_bfloat16);
    }
    else
    {
        TORCH_CHECK(false, "fp4_quantize_phase2 only supports fp16/bf16 input, got ", input.scalar_type());
    }

#undef PHASE2_DISPATCH
#undef TRY_PHASE2_LAUNCH

    TORCH_CHECK(launched, "fp4_quantize_phase2 requires an SM100+ (Blackwell) GPU");

    return {valueE2M1, scaleFP8SF};
}

// ---------------------------------------------------------------------------
// fp4_swiglu_quantize_fused
// ---------------------------------------------------------------------------

/// SwiGLU + runtime-scaled NVFP4 for the SVDQuant FC13 epilogue.
///
/// Phase 1 of the persistent adaptive kernel now evaluates SwiGLU over the
/// ``[m, 2n]`` pre-activation, so the standalone activation kernel disappears
/// while the ``[m, n]`` BF16 activation the FC2 low-rank correction reads is
/// still materialized -- phase 2 quantizes exactly that tensor.
///
/// Returns {packed FP4, swizzled SF, {amax, quantRange/amax}, BF16 activation}.
std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> fp4_swiglu_quantize_fused(at::Tensor const& preactivation,
    int64_t sfVecSize, bool isSfSwizzledLayout, int64_t scaleRule, double quantRange, double eps,
    int64_t testMaxActiveBlocks, at::Tensor const& tileIdxToMnLimit, at::Tensor const& numNonExitingTiles,
    int64_t tileSize)
{
    CHECK_TH_CUDA(preactivation);
    CHECK_CONTIGUOUS(preactivation);
    c10::cuda::CUDAGuard device_guard(preactivation.device());

    TORCH_CHECK(preactivation.dim() == 2, "SwiGLU pre-activation must be a 2D tensor.");
    TORCH_CHECK(preactivation.size(1) % 2 == 0, "SwiGLU pre-activation must have an even column count.");
    TORCH_CHECK(sfVecSize == 16 && isSfSwizzledLayout,
        "Fused SwiGLU quantization requires SF vector 16 and the swizzled layout");
    TORCH_CHECK(scaleRule == 0 || scaleRule == 1,
        "Fused SwiGLU quantization supports standard (0) or MSE 4/6 (1) scaling, got ", scaleRule);

    int64_t const m = preactivation.size(0);
    int64_t const k = preactivation.size(1) / 2;
    TORCH_CHECK(k % sfVecSize == 0, "SwiGLU output width must be divisible by sfVecSize");

    CHECK_INPUT(tileIdxToMnLimit, torch::kInt32);
    CHECK_INPUT(numNonExitingTiles, torch::kInt32);
    TORCH_CHECK(tileIdxToMnLimit.get_device() == preactivation.get_device()
            && numNonExitingTiles.get_device() == preactivation.get_device(),
        "MoE routing metadata and pre-activation must be on the same CUDA device");
    TORCH_CHECK(tileSize > 0 && m % tileSize == 0, "MoE rows must be a multiple of tileSize");
    TORCH_CHECK(tileIdxToMnLimit.numel() == m / tileSize, "tileIdxToMnLimit must contain one entry per routing tile");
    TORCH_CHECK(numNonExitingTiles.numel() == 1, "numNonExitingTiles must contain one value");

    // Routing padding is left untouched here, exactly as the standalone
    // activation kernel left it; downstream grouped GEMMs mask those rows.
    at::Tensor swigluOut = at::empty({m, k}, preactivation.options());
    at::Tensor valueE2M1 = at::empty({m, k / 2}, preactivation.options().dtype(FLOAT4_E2M1X2));
    at::Tensor scaleFP8SF = at::empty(
        {tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)}, preactivation.options().dtype(SF_DTYPE));
    at::Tensor amaxScale = at::empty({2}, preactivation.options().dtype(torch::kFloat32));

    int mMultiProcessorCount = 0;
    cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, preactivation.get_device());
    auto stream = at::cuda::getCurrentCUDAStream(preactivation.get_device()).stream();
    auto& ws = fusedWorkspace(
        preactivation.device(), stream, fusedPrologueGridX(PadUpFn(static_cast<int>(m), 128), k, mMultiProcessorCount));

    bool fusedTaken = false;

#define TRY_FUSED_SWIGLU_LAUNCH(T, RULE)                                                                               \
    fusedTaken = tensorrt_llm::kernels::invokeFusedMoeSwigluPrologueQuantizationV2<T, 16,                              \
        tensorrt_llm::kernels::AdaptiveScaleRule::RULE>(m, k, reinterpret_cast<T const*>(preactivation.data_ptr()),    \
        reinterpret_cast<T*>(swigluOut.data_ptr()), tileIdxToMnLimit.data_ptr<int32_t>(),                              \
        numNonExitingTiles.data_ptr<int32_t>(), static_cast<int>(tileSize), static_cast<float>(quantRange),            \
        static_cast<float>(eps), reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                     \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()), tensorrt_llm::QuantizationSFLayout::SWIZZLED,               \
        mMultiProcessorCount, ws.blockMaxBuf.data_ptr<float>(), ws.retirementCount.data_ptr<int>(),                    \
        amaxScale.data_ptr<float>(), stream, static_cast<int>(testMaxActiveBlocks))

#define FUSED_SWIGLU_DISPATCH(T)                                                                                       \
    do                                                                                                                 \
    {                                                                                                                  \
        if (scaleRule == 0)                                                                                            \
        {                                                                                                              \
            TRY_FUSED_SWIGLU_LAUNCH(T, NONE);                                                                          \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            TRY_FUSED_SWIGLU_LAUNCH(T, MSE);                                                                           \
        }                                                                                                              \
    } while (0)

    if (preactivation.scalar_type() == at::ScalarType::Half)
    {
        FUSED_SWIGLU_DISPATCH(half);
    }
    else if (preactivation.scalar_type() == at::ScalarType::BFloat16)
    {
        FUSED_SWIGLU_DISPATCH(__nv_bfloat16);
    }
    else
    {
        TORCH_CHECK(false, "fp4_swiglu_quantize_fused only supports fp16/bf16 input.");
    }

#undef FUSED_SWIGLU_DISPATCH
#undef TRY_FUSED_SWIGLU_LAUNCH

    TORCH_CHECK(fusedTaken, "Fused SwiGLU adaptive quantization requires the SM100 persistent fused kernel");

    return {valueE2M1, scaleFP8SF, amaxScale, swigluOut};
}

/// Dequantize FP4 with swizzled SF layout to BF16.
/// Input: packed FP4 [M, interm_size/2] + swizzled SF (flat) + global_scale (scalar)
/// Output: BF16 [M, interm_size]
at::Tensor dequant_nvfp4_swizzled_sf(
    at::Tensor const& fp4_packed,   // [M, packed_cols], dtype uint8 or float4_e2m1fn_x2
    at::Tensor const& sf_swizzled,  // flat 1D uint8
    at::Tensor const& global_scale, // scalar float32
    int64_t sfVecSize)
{
    CHECK_TH_CUDA(fp4_packed);
    CHECK_TH_CUDA(sf_swizzled);
    CHECK_INPUT(global_scale, torch::kFloat32);
    TORCH_CHECK(global_scale.numel() == 1, "global_scale must contain exactly one float");
    TORCH_CHECK(global_scale.get_device() == fp4_packed.get_device(),
        "global_scale and fp4_packed must be on the same CUDA device");

    int64_t m = fp4_packed.size(0);
    int64_t packed_cols = fp4_packed.size(1);
    int64_t interm_size = packed_cols * 2;

    auto output = at::empty({m, interm_size},
        fp4_packed.options().dtype(torch::kBFloat16));

    static int mpc = tensorrt_llm::common::getMultiProcessorCount();
    auto stream = at::cuda::getCurrentCUDAStream(fp4_packed.get_device()).stream();

    tensorrt_llm::kernels::invokeDequantNvfp4SwizzledSF<__nv_bfloat16>(
        m, interm_size,
        fp4_packed.data_ptr<uint8_t>(),
        sf_swizzled.data_ptr<uint8_t>(),
        global_scale.data_ptr<float>(), static_cast<int>(sfVecSize),
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
        tensorrt_llm::QuantizationSFLayout::SWIZZLED,
        mpc, stream);

    return output;
}

} // namespace torch_ext

// ---------------------------------------------------------------------------
// Op registration under trtllm namespace
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("fp4_quantize_ex(Tensor input, Tensor? globalScale, int sfVecSize, "
          "bool sfUseUE8M0=False, bool isSfSwizzledLayout=True, "
          "int kernelVersion=1, int scaleRule=0) -> (Tensor, Tensor)");
    m.def("calculate_global_amax(Tensor input, float quantRange=0.0, float eps=1e-12) -> Tensor");
    m.def(
        "fp4_quantize_fused(Tensor input, int sfVecSize, "
        "bool sfUseUE8M0=False, bool isSfSwizzledLayout=True, "
        "int scaleRule=0, float quantRange=2688.0, float eps=1e-12, "
        "int testMaxActiveBlocks=0, int forceV2=0, "
        "Tensor? tileIdxToMnLimit=None, Tensor? numNonExitingTiles=None, "
        "int tileSize=0) -> (Tensor, Tensor, Tensor)");
    m.def(
        "fp4_swiglu_quantize_fused(Tensor preactivation, int sfVecSize, "
        "bool isSfSwizzledLayout, int scaleRule, float quantRange, float eps, "
        "int testMaxActiveBlocks, Tensor tileIdxToMnLimit, "
        "Tensor numNonExitingTiles, int tileSize) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "fp4_quantize_phase2(Tensor input, Tensor amaxScale, int sfVecSize=16, "
        "bool isSfSwizzledLayout=True, int scaleRule=0) -> (Tensor, Tensor)");
    m.def("dequant_nvfp4_swizzled_sf(Tensor fp4_packed, Tensor sf_swizzled, "
          "Tensor global_scale, int sfVecSize=16) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fp4_quantize_ex", TORCH_FN(torch_ext::fp4_quantize_ex));
    m.impl("calculate_global_amax", TORCH_FN(torch_ext::calculate_global_amax));
    m.impl("fp4_quantize_fused", TORCH_FN(torch_ext::fp4_quantize_fused));
    m.impl("fp4_swiglu_quantize_fused", TORCH_FN(torch_ext::fp4_swiglu_quantize_fused));
    m.impl("fp4_quantize_phase2", TORCH_FN(torch_ext::fp4_quantize_phase2));
    m.impl("dequant_nvfp4_swizzled_sf", TORCH_FN(torch_ext::dequant_nvfp4_swizzled_sf));
}
