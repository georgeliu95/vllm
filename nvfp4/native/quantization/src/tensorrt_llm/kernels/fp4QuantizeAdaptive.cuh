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
 * Adaptive 4/6 (FourOverSix) NVFP4 quantization kernels — device code.
 *
 * New kernels (not in tensorrt_llm/kernels/quantization.cuh):
 *   - opt_quantize_with_block_size_v1: Optimized v1 (16 elts/thread, 256-bit loads)
 *   - opt_quantize_with_block_size_adaptive: Per-block MSE/MAE/ABS_MAX 4/6 selection
 *   - opt_quantize_with_block_size_adaptive_v2: 8 elts/thread variant
 *   - computeGlobalAmaxKernel: Single-kernel last-block reduction for runtime amax
 *
 * Derived from tllm_linear_lite (https://github.com/nvidia/tllm_linear_lite).
 * Depends on base functions from tensorrt_llm/kernels/quantization.cuh:
 *   fp32_vec_to_e2m1, reciprocal_approximate_ftz, PackedVec, cvt_quant_get_sf_out_offset,
 *   quantize_with_block_size (v0 kernel fallback).
 */

#pragma once

// Real TRT-LLM headers (replaces tllm_compat.cuh used in standalone build)
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"

// Base FP4 quantization (v0 kernel, PackedVec, fp32_vec_to_e2m1, etc.)
#include "tensorrt_llm/kernels/quantization.cuh"

// Public API for this extension
#include "tensorrt_llm/kernels/fp4QuantizeAdaptive.h"

#include <float.h>

using namespace tensorrt_llm::common;

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// Atomic max for float (positive values only — safe for amax which is always >= 0).
// Uses IEEE 754 property: for non-negative floats, integer comparison preserves order.
static __device__ __forceinline__ float atomicMaxFloat(float* addr, float value)
{
    return __int_as_float(atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value)));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// v1 Optimization: 16 elements per thread, 256-bit vectorized load, increased ILP

// v1 processes 16 elements per thread (vs 8 in v0).
constexpr int CVT_OPT_ELTS_PER_THREAD = 16;

/*
 * PackedVec_Opt: 32-byte aligned vector holding 16 FP16/BF16 elements (8 vec2 values).
 * Used exclusively by opt_quantize_with_block_size_v1.
 */
template <class Type>
struct __align__(32) PackedVec_Opt
{
    typename TypeConverter<Type>::Type elts[8];
    static_assert(sizeof(elts) == sizeof(Type) * CVT_OPT_ELTS_PER_THREAD,
        "Vector size should match the number of elements per thread.");
};

/*
 * Load 32 bytes (256 bits) from global memory using two consecutive 128-bit loads.
 * The two-instruction sequence guarantees LDG.E.128 in SASS, which is optimal for
 * 32-byte aligned global loads. A single 256-bit PTX load is not available on sm_100.
 */
template <class T>
__device__ __forceinline__ void load_256bit(T* dst, void const* src)
{
    static_assert(sizeof(T) == 32, "load_256bit requires T to be exactly 32 bytes");
    uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(dst);
    char const* src_char = reinterpret_cast<char const*>(src);

    // First 128-bit load (bytes 0-15)
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst_u32[0]), "=r"(dst_u32[1]), "=r"(dst_u32[2]), "=r"(dst_u32[3])
        : "l"(src_char)
    );

    // Second 128-bit load (bytes 16-31)
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst_u32[4]), "=r"(dst_u32[5]), "=r"(dst_u32[6]), "=r"(dst_u32[7])
        : "l"(src_char + 16)
    );
}

/*
 * Optimized (v1) inner implementation: process 16 elements per thread for increased ILP.
 *
 * Key differences from cvt_warp_fp16_to_fp4 (v0, 8 elements/thread):
 * - Input: PackedVec_Opt with 8 half2/bfloat162 (16 elements, 32 bytes)
 * - Output: uint64_t (16 e2m1 values)
 * - SF thread cooperation:
 *   - SF_VEC_SIZE=16: CVT_NUM_THREADS_PER_SF=1 (single thread, no shuffle)
 *   - SF_VEC_SIZE=32: CVT_NUM_THREADS_PER_SF=2 (two threads, one shuffle)
 *
 * More compute per memory load hides L1TEX scoreboard stall (~7 cycles, ~40% of CPI).
 */
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF, typename VecType>
__device__ uint64_t cvt_warp_fp16_to_fp4_impl_opt(VecType& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Get absolute maximum values among the local 16 values (8 vec2 elements).
    auto localMax = cuda_abs(vec.elts[0]);

#pragma unroll
    for (int i = 1; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    // SF thread cooperation:
    // - SF_VEC_SIZE=16, CVT_OPT_ELTS_PER_THREAD=16: CVT_NUM_THREADS_PER_SF=1 (single thread)
    // - SF_VEC_SIZE=32, CVT_OPT_ELTS_PER_THREAD=16: CVT_NUM_THREADS_PER_SF=2 (two threads)
    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_OPT_ELTS_PER_THREAD;
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v1 only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    // Cross-thread max reduction only needed when two threads share one SF slot.
    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    }

    float vecMax = float(cuda_max(localMax.x, localMax.y));

    uint8_t fp8SFVal;
    float outputScale;
    if constexpr (UE8M0_SF)
    {
        __nv_fp8_e8m0 tmp;
        vecMax *= reciprocal_approximate_ftz(6.0f);
        tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);
        fp8SFVal = tmp.__x;
        outputScale = vecMax != 0 ? exp2f_rcp(fp8SFVal) : 0.0f;
    }
    else
    {
        // maximum value of e2m1 = 6.0
        auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        fp8SFVal = tmp.__x;
        SFValue = static_cast<float>(tmp);
        outputScale = vecMax != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
    }

    if (SFout)
    {
        *SFout = fp8SFVal;
    }

    // Convert the input to float: 8 vec2 elements = 16 floats.
    float2 fp2Vals[CVT_OPT_ELTS_PER_THREAD / 2];

#pragma unroll
    for (int i = 0; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        if constexpr (std::is_same_v<Type, half>)
        {
            fp2Vals[i] = __half22float2(vec.elts[i]);
        }
        else
        {
            fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        }
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    // Convert 16 floats to 16 e2m1 values packed in uint64_t.
    uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
    return e2m1Vec;
#else
    return 0;
#endif
}

/*
 * v1 Kernel: Optimized 16 elements per thread for increased ILP (FP16/BF16 to FP4 only).
 *
 * This kernel processes 16 elements per thread instead of 8, doubling the
 * compute work per memory load. This helps hide L1TEX scoreboard stalls
 * by providing more independent operations for the scheduler.
 *
 * Key differences from v0 (quantize_with_block_size):
 * - ELTS_PER_THREAD: 16 instead of 8
 * - PackedVec: 32 bytes instead of 16 bytes (32-byte aligned vectorized load via load_256bit)
 * - Output: uint64_t (16 e2m1 values) instead of uint32_t (8 e2m1 values)
 * - numColThreads: numCols / 16 instead of numCols / 8
 *
 * NOTE: This v1 kernel only supports FP16/BF16 to FP4 quantization.
 * FP8 to FP4 and FP16 to MXFP8 are not supported in v1.
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 4) opt_quantize_with_block_size_v1(
#else
opt_quantize_with_block_size_v1(
#endif
        int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    // v1 only supports FP16/BF16 to FP4 quantization.
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "opt_quantize_with_block_size_v1 only supports FP16_TO_FP4 quantization type");

    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVec = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size is not matched.");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v1 only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numPaddedRowsForSf = isSf128x4Layout ? PadUpFn(numRows, 128) : (isSf8x4Layout ? PadUpFn(numRows, 8) : numRows);
    int numColsForSf = (isSf128x4Layout || isSf8x4Layout) ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    // v1 processes 16 elements per thread, so numColThreads = numCols / 16.
    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    asm volatile("griddepcontrol.wait;");
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
            {
                std::optional<int> optionalBatchIdx = batchIdx;
                std::optional<int> optionalNumRows = numRows;

                // Each thread covers SF_VEC_SIZE/16 = 1 or 2 SF slots.
                auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                    optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

                int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
                int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

                // Zero out padded columns in the output tensor.
                if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
                {
                    // v1 outputs uint64_t (16 e2m1 values).
                    reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
                }

                // Zero out SF for padding rows or padding columns.
                if (rowIdx >= numRows || colIdx >= numColThreads)
                {
                    if (sf_out != nullptr)
                    {
                        sf_out[0] = 0x00;
                    }
                }
                else
                {
                    // Load the input vector using 256-bit vectorized load (two 128-bit loads).
                    // This guarantees LDG.E.128 instructions in SASS. See load_256bit() for details.
                    PackedVec in_vec;
                    load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVec));

                    // Dispatch the v1 quantization implementation (16 elements, uint64_t output).
                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_impl_opt<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FourOverSix Adaptive 4/6 Block Scaling

// AdaptiveScaleRule is defined in fp4QuantizeAdaptive.h

/*
 * Fake-quantize 4 half2 values (8 elements) to E2M1, dequantize back, and
 * accumulate error — all in a single register-efficient pass.
 *
 * Accepts half2 input directly (no float array materialization). Float
 * conversion happens just-in-time inside the loop body, and the converted
 * values are immediately consumed for scaling, error computation, then
 * discarded — keeping register pressure minimal.
 *
 * Returns the packed E2M1 output (uint32_t = 8 nibbles).
 *
 * Processes 8 elements (4 half2) rather than the full 16-element block to cap
 * register pressure. The caller invokes this twice (lo + hi halves) so the
 * second call reuses the same physical regs after the first's die.
 *
 * Quantization uses the f16x2/bf16x2 → e2m1x2 path (PTX ISA 9.1, CUDA 13.1+):
 * scale multiplication stays in half precision (HMUL2, 1 inst per pair) and
 * cvt.rn.satfinite.e2m1x2.f16x2 takes the half2 directly — avoiding the
 * half→float conversion + f32 scale multiply of the original f32 path.
 * Trade-off: fp16 scale has 10-bit mantissa vs fp32's 23-bit; acceptable for
 * the adaptive error-comparison use case (both candidates lose similar precision).
 * Requires: CUDA 13.1+ (PTX ISA 9.1). CUDA 13.0 ptxas will reject this.
 *
 * NOTE on .b8 boundary constraint and why batch mode is optimal:
 * cvt.e2m1x2 outputs .b8 which CANNOT cross asm boundaries via "=r" — ptxas
 * rejects "Arguments mismatch". Two alternatives were tested and rejected:
 * - F-cvt (per-pair + cvt.u32.u8 widen): spill +224%, +8% regression
 * - F2 (all-in-PTX single asm): 11 operands live → spill +224%, +6% regression
 * Batch mode (4 small asm blocks) lets the compiler release/reuse registers
 * between blocks, minimizing spill. See issue doc §8 experiments 14-16.
 */
template <AdaptiveScaleRule Rule, class HalfType>
__device__ __forceinline__ uint32_t fake_quant_e2m1_8(
    HalfType const (&h2)[4], float outputScale, float decodeScale, float& err)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // --- Quant: scale in half precision + f16x2/bf16x2 → e2m1x2 (PTX ISA 9.1) ---
    uint32_t e2m1_packed;
#if defined(B300_TRT_QUANT_FP32)
    float2 scaled[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        float2 value;
        if constexpr (std::is_same_v<HalfType, __half2>)
            value = __half22float2(h2[i]);
        else
            value = __bfloat1622float2(h2[i]);
        scaled[i].x = __fmul_rn(value.x, outputScale);
        scaled[i].y = __fmul_rn(value.y, outputScale);
    }
    e2m1_packed = fp32_vec_to_e2m1(scaled);
#else
    {
        HalfType scale_vec;
        if constexpr (std::is_same_v<HalfType, __half2>)
            scale_vec = __float2half2_rn(outputScale);
        else
            scale_vec = __float2bfloat162_rn(outputScale);

        HalfType sq0 = __hmul2(h2[0], scale_vec);
        HalfType sq1 = __hmul2(h2[1], scale_vec);
        HalfType sq2 = __hmul2(h2[2], scale_vec);
        HalfType sq3 = __hmul2(h2[3], scale_vec);

        uint32_t u0 = reinterpret_cast<uint32_t&>(sq0);
        uint32_t u1 = reinterpret_cast<uint32_t&>(sq1);
        uint32_t u2 = reinterpret_cast<uint32_t&>(sq2);
        uint32_t u3 = reinterpret_cast<uint32_t&>(sq3);

        if constexpr (std::is_same_v<HalfType, __half2>)
        {
            asm(
                "{\n"
                ".reg .b8 byte0, byte1, byte2, byte3;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte0, %1;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte1, %2;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte2, %3;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte3, %4;\n"
                "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                "}"
                : "=r"(e2m1_packed)
                : "r"(u0), "r"(u1), "r"(u2), "r"(u3)
            );
        }
        else
        {
            asm(
                "{\n"
                ".reg .b8 byte0, byte1, byte2, byte3;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte0, %1;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte1, %2;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte2, %3;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte3, %4;\n"
                "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                "}"
                : "=r"(e2m1_packed)
                : "r"(u0), "r"(u1), "r"(u2), "r"(u3)
            );
        }
    }

#endif
    // Per-byte dequant: each DEQUANT_BYTE extracts one .b8 from e2m1_packed and
    // converts back to f16x2. Separate asm blocks (non-volatile) so the compiler
    // can interleave dequant with error computation if profitable.
#define DEQUANT_BYTE(IDX, BSEL)                                       \
    uint32_t dq##IDX;                                                  \
    asm(                                                               \
        "{\n"                                                          \
        ".reg .b8 b0, b1, b2, b3;\n"                                  \
        "mov.b32 {b0, b1, b2, b3}, %1;\n"                             \
        "cvt.rn.f16x2.e2m1x2 %0, " BSEL ";\n"                        \
        "}\n"                                                          \
        : "=r"(dq##IDX) : "r"(e2m1_packed)                            \
    )

#define ERROR_PAIR(IDX)                                                \
    {                                                                  \
        float2 orig;                                                   \
        if constexpr (std::is_same_v<HalfType, __half2>)               \
            orig = __half22float2(h2[IDX]);                            \
        else                                                           \
            orig = __bfloat1622float2(h2[IDX]);                        \
        half2 dq_h = reinterpret_cast<half2&>(dq##IDX);               \
        float2 dq_f = __half22float2(dq_h);                           \
        dq_f.x *= decodeScale;                                        \
        dq_f.y *= decodeScale;                                        \
        float dx = dq_f.x - orig.x;                                   \
        float dy = dq_f.y - orig.y;                                   \
        if constexpr (Rule == AdaptiveScaleRule::MSE)                  \
            err += dx * dx + dy * dy;                                  \
        else if constexpr (Rule == AdaptiveScaleRule::MAE)             \
            err += fabsf(dx) + fabsf(dy);                             \
        else if constexpr (Rule == AdaptiveScaleRule::ABS_MAX)         \
            err = fmaxf(err, fmaxf(fabsf(dx), fabsf(dy)));           \
    }

    DEQUANT_BYTE(0, "b0"); ERROR_PAIR(0);
    DEQUANT_BYTE(1, "b1"); ERROR_PAIR(1);
    DEQUANT_BYTE(2, "b2"); ERROR_PAIR(2);
    DEQUANT_BYTE(3, "b3"); ERROR_PAIR(3);

#undef DEQUANT_BYTE
#undef ERROR_PAIR

    return e2m1_packed;
#else
    return 0;
#endif
}

/*
 * Adaptive 4/6 inner implementation (streaming, 16 elements per thread).
 *
 * Sequentially computes fake-quant + error for r=6 then r=4, selecting the
 * candidate with lower error. Only ~21 R32 registers at peak — no spilling.
 *
 * The r=4 candidate uses scale_expansion_factor=1.5 (sf_4 = sf_6_hp * 1.5),
 * mapping input to E2M1 range [-4, 4] instead of [-6, 6]. This avoids the
 * coarse 4→6 interval (gap=2) at the cost of one fewer quantization level.
 *
 * Both candidates' SF values fit in E4M3 (max 256 for r=6, max 384 for r=4,
 * both < 448). The decoder is agnostic to the r choice — it simply uses
 * x_reconstructed = e2m1_val * sf_e4m3 / SFScaleVal.
 */
template <class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule, typename VecType>
__device__ uint64_t cvt_warp_fp16_to_fp4_adaptive(VecType& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(Rule != AdaptiveScaleRule::NONE, "Use cvt_warp_fp16_to_fp4_impl_opt for non-adaptive");

    // --- Step 1: Compute vecMax (same as v1) ---
    auto localMax = cuda_abs(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_OPT_ELTS_PER_THREAD;
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "adaptive only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    }

    float vecMax = float(cuda_max(localMax.x, localMax.y));

    // --- Step 2: Compute E4M3 SF candidates for r=6 and r=4 ---
    float sf_hp = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 sf_6_e4m3 = __nv_fp8_e4m3(sf_hp);
    __nv_fp8_e4m3 sf_4_e4m3 = __nv_fp8_e4m3(sf_hp * 1.5f);

    float rcp_SFScale = reciprocal_approximate_ftz(SFScaleVal);

    // Zero-copy references: reinterpret vec.elts[0..3] and vec.elts[4..7] as
    // const-ref arrays without copying values into separate registers.
    // Saves ~8 regs vs naive `lo_6[4] = {vec.elts[0], ...}` value copy.
    using HalfVecT = typename TypeConverter<Type>::Type;
    using Arr4Ref = HalfVecT const (&)[4];
    Arr4Ref lo = reinterpret_cast<Arr4Ref>(vec.elts[0]);
    Arr4Ref hi = reinterpret_cast<Arr4Ref>(vec.elts[4]);

    // --- Step 3+4: r=6 fake-quant + error (stored as initial winner) ---
    // Scoped block: sf_f / outputScale / decodeScale die at '}', freeing ~3 regs
    // before r=4 scope allocates its own set — prevents both sets coexisting.
    float best_err = 0.0f;
    uint32_t e2m1_lo, e2m1_hi;
    __nv_fp8_e4m3 sf_best = sf_6_e4m3;
    {
        float sf_f = static_cast<float>(sf_6_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        // Process 16 elements as two batches of 8 to keep register pressure low.
        // fake_quant_e2m1_8 peaks at 8 regs (s0-s7 scaled floats for quant asm input);
        // a hypothetical 16-at-once would need 16 simultaneous regs, blowing the budget.
        // The second call reuses the same physical regs after the first call's s0-s7 die.
        // best_err accumulates across both calls (passed by reference).
        e2m1_lo = fake_quant_e2m1_8<Rule>(lo, outputScale, decodeScale, best_err);
        e2m1_hi = fake_quant_e2m1_8<Rule>(hi, outputScale, decodeScale, best_err);
    }

    // --- Step 5: r=4 fake-quant + error ---
    // Scoped block reuses the same ~3 reg slots freed from r=6 scope above.
    float err_4 = 0.0f;
    uint32_t e2m1_4_lo, e2m1_4_hi;
    {
        float sf_f = static_cast<float>(sf_4_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        e2m1_4_lo = fake_quant_e2m1_8<Rule>(lo, outputScale, decodeScale, err_4);
        e2m1_4_hi = fake_quant_e2m1_8<Rule>(hi, outputScale, decodeScale, err_4);
    }

    // --- Step 6: For SF_VEC_SIZE=32, reduce errors across 2 threads ---
    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        if constexpr (Rule == AdaptiveScaleRule::ABS_MAX)
        {
            best_err = fmaxf(best_err, __shfl_xor_sync(uint32_t(-1), best_err, 1));
            err_4 = fmaxf(err_4, __shfl_xor_sync(uint32_t(-1), err_4, 1));
        }
        else
        {
            best_err += __shfl_xor_sync(uint32_t(-1), best_err, 1);
            err_4 += __shfl_xor_sync(uint32_t(-1), err_4, 1);
        }
    }

    // --- Step 7: Overwrite with r=4 if it produces lower error ---
    // Serialized winner selection: r=6 result already in e2m1_lo/hi, conditionally
    // overwrite with r=4. Avoids keeping separate winner_lo/hi/sf variables.
    if (err_4 < best_err)
    {
        e2m1_lo = e2m1_4_lo;
        e2m1_hi = e2m1_4_hi;
        sf_best = sf_4_e4m3;
    }

    if (SFout)
    {
        *SFout = sf_best.__x;
    }

    // Pack two uint32_t halves into one uint64_t.
    uint64_t e2m1_out;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(e2m1_out) : "r"(e2m1_lo), "r"(e2m1_hi));
    return e2m1_out;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FourOverSix Adaptive Kernel: streaming 4/6 selection with per-thread sequential error comparison

/*
 * Adaptive kernel: same structure as opt_quantize_with_block_size_v1 but calls
 * cvt_warp_fp16_to_fp4_adaptive for per-block 4/6 scale selection.
 *
 * Only supports FP16/BF16→FP4 with E4M3 scale factors (not UE8M0).
 * The caller must pass globalScale = 1536/amax (not 2688/amax).
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // (512, 3): register budget = 65536 / (512*3) = 42 regs/thread.
    // (512, 3): register budget = 65536 / (512*3) = 42 regs/thread.
    // Kernel compiles to 40 regs with 8 STL/LDL spill instructions (acceptable).
    // (512, 4) at 32 regs: 28 STL/LDL (+250%), spill requests +237%, latency +7%.
    // SASS code is nearly identical (21 LDG both); the difference is entirely
    // from local memory spilling at the tighter register budget.
    __launch_bounds__(512, 3) opt_quantize_with_block_size_adaptive(
#else
opt_quantize_with_block_size_adaptive(
#endif
        int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "adaptive kernel only supports FP16_TO_FP4");
    static_assert(Rule != AdaptiveScaleRule::NONE,
        "use opt_quantize_with_block_size_v1 for non-adaptive quantization");

    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVec = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size mismatch.");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "adaptive only supports SF_VEC_SIZE of 16 or 32");

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numPaddedRowsForSf = isSf128x4Layout ? PadUpFn(numRows, 128) : (isSf8x4Layout ? PadUpFn(numRows, 8) : numRows);
    int numColsForSf = (isSf128x4Layout || isSf8x4Layout) ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    asm volatile("griddepcontrol.wait;");
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
            {
                std::optional<int> optionalBatchIdx = batchIdx;
                std::optional<int> optionalNumRows = numRows;

                auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                    optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

                int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
                int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

                if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
                {
                    reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
                }

                if (rowIdx >= numRows || colIdx >= numColThreads)
                {
                    if (sf_out != nullptr)
                    {
                        sf_out[0] = 0x00;
                    }
                }
                else
                {
                    PackedVec in_vec;
                    load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVec));

                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_adaptive<Type, SF_VEC_SIZE, Rule>(in_vec, SFScaleVal, sf_out);
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Method G: 2 threads/SF adaptive kernel (8 elements per thread)
//
// Compared to the v1-based adaptive kernel (16 elts/thread):
// - vec.elts[4] instead of [8] → ~4 fewer regs → less spill
// - Single fake_quant_e2m1_8 call (no lo/hi split)
// - Needs cross-thread shuffle for amax + error reduction
// - Output: uint32 per thread (vs uint64 in v1-based)
// - numColThreads = numCols / 8 (2× more column iterations per block)

/*
 * Inner implementation: 8 elements/thread, 2+ threads cooperate per SF block.
 * Same adaptive logic (r=6 vs r=4 selection) but lower register pressure.
 */
template <class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__device__ uint32_t cvt_warp_fp16_to_fp4_adaptive_v2(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(Rule != AdaptiveScaleRule::NONE, "Use cvt_warp_fp16_to_fp4 for non-adaptive");

    // --- Step 1: amax over 4 half2 (8 elements) ---
    auto localMax = cuda_abs(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    // Cross-thread amax: 2 threads for SF_VEC_SIZE=16, 4 for SF_VEC_SIZE=32.
    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
    static_assert(CVT_NUM_THREADS_PER_SF == 2 || CVT_NUM_THREADS_PER_SF == 4,
        "v2 adaptive requires SF_VEC_SIZE >= 16 with 8 elts/thread");

    localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    if constexpr (CVT_NUM_THREADS_PER_SF == 4)
    {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
    }

    float vecMax = float(cuda_max(localMax.x, localMax.y));

    // --- Step 2: E4M3 SF candidates for r=6 and r=4 ---
    float sf_hp = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 sf_6_e4m3 = __nv_fp8_e4m3(sf_hp);
    __nv_fp8_e4m3 sf_4_e4m3 = __nv_fp8_e4m3(sf_hp * 1.5f);

    float rcp_SFScale = reciprocal_approximate_ftz(SFScaleVal);

    using HalfVecT = typename TypeConverter<Type>::Type;
    using Arr4Ref = HalfVecT const (&)[4];
    Arr4Ref data = reinterpret_cast<Arr4Ref>(vec.elts[0]);

    // --- Step 3: r=6 fake-quant + error (single call, 8 elements) ---
    float best_err = 0.0f;
    uint32_t e2m1;
    __nv_fp8_e4m3 sf_best = sf_6_e4m3;
    {
        float sf_f = static_cast<float>(sf_6_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        e2m1 = fake_quant_e2m1_8<Rule>(data, outputScale, decodeScale, best_err);
    }

    // --- Step 4: r=4 fake-quant + error ---
    float err_4 = 0.0f;
    uint32_t e2m1_4;
    {
        float sf_f = static_cast<float>(sf_4_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        e2m1_4 = fake_quant_e2m1_8<Rule>(data, outputScale, decodeScale, err_4);
    }

    // --- Step 5: Cross-thread error reduction ---
    if constexpr (Rule == AdaptiveScaleRule::ABS_MAX)
    {
        best_err = fmaxf(best_err, __shfl_xor_sync(uint32_t(-1), best_err, 1));
        err_4 = fmaxf(err_4, __shfl_xor_sync(uint32_t(-1), err_4, 1));
        if constexpr (CVT_NUM_THREADS_PER_SF == 4)
        {
            best_err = fmaxf(best_err, __shfl_xor_sync(uint32_t(-1), best_err, 2));
            err_4 = fmaxf(err_4, __shfl_xor_sync(uint32_t(-1), err_4, 2));
        }
    }
    else
    {
        best_err += __shfl_xor_sync(uint32_t(-1), best_err, 1);
        err_4 += __shfl_xor_sync(uint32_t(-1), err_4, 1);
        if constexpr (CVT_NUM_THREADS_PER_SF == 4)
        {
            best_err += __shfl_xor_sync(uint32_t(-1), best_err, 2);
            err_4 += __shfl_xor_sync(uint32_t(-1), err_4, 2);
        }
    }

    // --- Step 6: Winner selection ---
    if (err_4 < best_err)
    {
        e2m1 = e2m1_4;
        sf_best = sf_4_e4m3;
    }

    if (SFout)
    {
        *SFout = sf_best.__x;
    }

    return e2m1;
#else
    return 0;
#endif
}

/*
 * v2 Adaptive kernel: 8 elements/thread, 2D grid (rows × column tiles).
 *
 * blockIdx.x → row, blockIdx.y → column tile. Column dimension is parallelized
 * in the grid instead of serialized in the inner loop. Each block processes
 * exactly 1 column tile per row → no multi-iteration column loop overhead.
 *
 * With SF_VEC_SIZE=16 and 8 elts/thread: 2 threads cooperate per SF block via
 * shuffle. Lower register pressure (~36 regs) than the v1-based 16 elts/thread.
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Same budget as v1's (512,3) = 42 regs. blockDim=384, same concurrent
    // threads/SM as v1 (4 blocks × 384 = 1536). Column parallelism via gridDim.y.
    __launch_bounds__(512, 3) opt_quantize_with_block_size_adaptive_v2(
#else
opt_quantize_with_block_size_adaptive_v2(
#endif
        int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "adaptive v2 kernel only supports FP16_TO_FP4");
    static_assert(Rule != AdaptiveScaleRule::NONE,
        "use opt_quantize_with_block_size_v1 for non-adaptive quantization");

    static constexpr int ELTS_PER_THREAD = CVT_ELTS_PER_THREAD;  // = 8
    using PackedVecT = PackedVec<Type>;  // 16 bytes (4 × half2)
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVecT) == sizeof(Type) * ELTS_PER_THREAD, "Vec size mismatch.");
    static_assert(CVT_NUM_THREADS_PER_SF == 2 || CVT_NUM_THREADS_PER_SF == 4,
        "v2 adaptive requires 2 or 4 threads per SF");

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numPaddedRowsForSf = isSf128x4Layout ? PadUpFn(numRows, 128) : (isSf8x4Layout ? PadUpFn(numRows, 8) : numRows);
    int numColsForSf = (isSf128x4Layout || isSf8x4Layout) ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    // Column index: blockIdx.y selects the column tile, threadIdx.x within it.
    int colIdx = blockIdx.y * blockDim.x + threadIdx.x;

    asm volatile("griddepcontrol.wait;");
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            // Skip threads beyond the column range (from the last tile).
            if (colIdx >= numColThreadsForSf) continue;

            std::optional<int> optionalBatchIdx = batchIdx;
            std::optional<int> optionalNumRows = numRows;

            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

            int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
            int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

            if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
            {
                reinterpret_cast<uint32_t*>(out)[outOffset] = 0u;
            }

            if (rowIdx >= numRows || colIdx >= numColThreads)
            {
                if (sf_out != nullptr)
                {
                    sf_out[0] = 0x00;
                }
            }
            else
            {
                PackedVecT in_vec;
                {
                    uint32_t* dst = reinterpret_cast<uint32_t*>(&in_vec);
                    char const* src = reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVecT);
                    asm volatile(
                        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
                        : "l"(src)
                    );
                }

                reinterpret_cast<uint32_t*>(out)[outOffset]
                    = cvt_warp_fp16_to_fp4_adaptive_v2<Type, SF_VEC_SIZE, Rule>(in_vec, SFScaleVal, sf_out);
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Fused Prologue + Quantization (single kernel: amax reduction -> quantize)
//
// The fused path writes the same packed FP4 tensor and SF layout as the
// regular quantization kernels. Layout handling intentionally includes R8C4
// because TRTLLMGen FC2 uses that layout for the intermediate scale factors.

template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 4) fused_prologue_quantize_v1(
#else
fused_prologue_quantize_v1(
#endif
        int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in, float quantRange, float eps,
        uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout, float* blockMaxBuf, int* retirementCount,
        float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "fused_prologue_quantize_v1 only supports FP16_TO_FP4");

    asm volatile("griddepcontrol.wait;");

    static constexpr int ElemsPerVec = 16 / sizeof(Type);
    float threadMax = 0.f;
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int vecIdx = threadIdx.x; vecIdx < numCols / ElemsPerVec; vecIdx += blockDim.x)
        {
            float4 vec = reinterpret_cast<float4 const*>(in + static_cast<int64_t>(rowIdx) * numCols)[vecIdx];
            auto* elems = reinterpret_cast<Type const*>(&vec);
            for (int e = 0; e < ElemsPerVec; ++e)
            {
                threadMax = fmaxf(threadMax, fabsf(static_cast<float>(elems[e])));
            }
        }
    }
    blockReduceMaxV2<float, 1>(&threadMax);
    if (threadIdx.x == 0)
    {
        blockMaxBuf[blockIdx.x] = threadMax;
    }

    __threadfence();

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
            globalScaleOut[0] = amax;
            globalScaleOut[1] = quantRange / amax;
            __threadfence();
            *retirementCount = 0;
        }
    }

    if (threadIdx.x == 0 && !isLastBlock)
    {
        volatile int* rc = reinterpret_cast<volatile int*>(retirementCount);
        while (*rc != 0)
        {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
            __nanosleep(32);
#endif
        }
    }
    __threadfence();
    __syncthreads();

    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVec = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size mismatch");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v1 only supports SF_VEC_SIZE of 16 or 32");

    float const SFScaleVal = globalScaleOut[1];

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numPaddedRowsForSf = isSf128x4Layout ? PadUpFn(numRows, 128) : (isSf8x4Layout ? PadUpFn(numRows, 8) : numRows);
    int numColsForSf = (isSf128x4Layout || isSf8x4Layout) ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
        {
            std::optional<int> optionalBatchIdx = 0;
            std::optional<int> optionalNumRows = numRows;

            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

            int64_t inOffset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
            int64_t outOffset = static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx;

            if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
            {
                reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
            }

            if (rowIdx >= numRows || colIdx >= numColThreads)
            {
                if (sf_out != nullptr)
                {
                    sf_out[0] = 0x00;
                }
            }
            else
            {
                PackedVec in_vec;
                load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVec));
                reinterpret_cast<uint64_t*>(out)[outOffset]
                    = cvt_warp_fp16_to_fp4_impl_opt<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/*
 * Publish the grid-wide amax and hold every block until it is readable.
 *
 * Each block contributes its own reduced maximum, the last block to retire
 * reduces those partials into {amax, quantRange/amax} and resets the counter,
 * and every other block spins on that reset.  The acquiring fence is placed
 * after a __syncthreads() so it cannot run before the flag has been observed,
 * which is what makes everything phase 1 wrote -- the published scale, and for
 * the SwiGLU-fused variant the activation itself -- visible to phase 2.
 */
__device__ __forceinline__ void fused_publish_amax_and_barrier(
    float threadMax, float quantRange, float eps, float* blockMaxBuf, int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    blockReduceMaxV2<float, 1>(&threadMax);
    if (threadIdx.x == 0)
    {
        blockMaxBuf[blockIdx.x] = threadMax;
    }

    __threadfence();

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
            globalScaleOut[0] = amax;
            globalScaleOut[1] = quantRange / amax;
            __threadfence();
            *retirementCount = 0;
        }
    }

    if (threadIdx.x == 0 && !isLastBlock)
    {
        int volatile* rc = reinterpret_cast<int volatile*>(retirementCount);
        while (*rc != 0)
        {
            __nanosleep(32);
        }
    }
    __syncthreads();
    __threadfence();
    __syncthreads();
#endif
}

template <class Type>
__device__ __noinline__ void fused_phase1_amax_barrier(int32_t numRows, int32_t numCols, Type const* in,
    float quantRange, float eps, float* blockMaxBuf, int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int ElemsPerVec = 16 / sizeof(Type);
    float threadMax = 0.f;
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int vecIdx = threadIdx.x; vecIdx < numCols / ElemsPerVec; vecIdx += blockDim.x)
        {
            float4 vec = reinterpret_cast<float4 const*>(in + static_cast<int64_t>(rowIdx) * numCols)[vecIdx];
            auto* elems = reinterpret_cast<Type const*>(&vec);
            for (int e = 0; e < ElemsPerVec; ++e)
            {
                threadMax = fmaxf(threadMax, fabsf(static_cast<float>(elems[e])));
            }
        }
    }
    fused_publish_amax_and_barrier(threadMax, quantRange, eps, blockMaxBuf, retirementCount, globalScaleOut);
#endif
}

template <class Type>
__device__ __noinline__ void fused_moe_phase1_amax_barrier(int32_t numRows, int32_t numCols, Type const* in,
    int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int32_t tileSize, float quantRange,
    float eps, float* blockMaxBuf, int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int ElemsPerVec = 16 / sizeof(Type);
    float threadMax = 0.f;
    int32_t const liveTiles = numNonExitingTiles[0];
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        int32_t const tileIdx = rowIdx / tileSize;
        if (tileIdx >= liveTiles || rowIdx >= tileIdxToMnLimit[tileIdx])
        {
            continue;
        }
        for (int vecIdx = threadIdx.x; vecIdx < numCols / ElemsPerVec; vecIdx += blockDim.x)
        {
            float4 vec = reinterpret_cast<float4 const*>(in + static_cast<int64_t>(rowIdx) * numCols)[vecIdx];
            auto* elems = reinterpret_cast<Type const*>(&vec);
#pragma unroll
            for (int e = 0; e < ElemsPerVec; ++e)
            {
                threadMax = fmaxf(threadMax, fabsf(static_cast<float>(elems[e])));
            }
        }
    }
    fused_publish_amax_and_barrier(threadMax, quantRange, eps, blockMaxBuf, retirementCount, globalScaleOut);
#endif
}

/*
 * SwiGLU as the standalone MoE activation kernel evaluates it: SiLu(gate) is
 * folded onto the linear half in FP32 and only the product is rounded back to
 * Type.  The fast exponential and reciprocal match the native CuTeDSL SwiGLU
 * epilogue rather than compiling the much longer correctly-rounded expf path.
 */
__device__ __forceinline__ float swiglu_f32(float gate, float linear)
{
    return (gate * reciprocal_approximate_ftz(1.0f + __expf(-gate))) * linear;
}

/*
 * SwiGLU-fused MoE phase 1.
 *
 * Reads the ``[numRows, 2 * numCols]`` FC13 pre-activation -- linear half
 * first, gate half second, matching the de-interleaved CuteDSL FC13 output --
 * writes the ``[numRows, numCols]`` BF16 activation the SVDQuant FC2 low-rank
 * correction consumes, and reduces amax over that activation.  Routing padding
 * is skipped in both roles: it never reaches the reduction, and its rows are
 * left untouched exactly as the standalone activation kernel left them.
 */
template <class Type>
__device__ __noinline__ void fused_moe_swiglu_phase1_amax_barrier(int32_t numRows, int32_t numCols,
    Type const* preactivation, Type* swigluOut, int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles,
    int32_t tileSize, float quantRange, float eps, float* blockMaxBuf, int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int ElemsPerVec = 16 / sizeof(Type);
    int32_t const numVecs = numCols / ElemsPerVec;
    float threadMax = 0.f;
    int32_t const liveTiles = numNonExitingTiles[0];
    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        int32_t const tileIdx = rowIdx / tileSize;
        if (tileIdx >= liveTiles || rowIdx >= tileIdxToMnLimit[tileIdx])
        {
            continue;
        }
        auto const* linearRow
            = reinterpret_cast<float4 const*>(preactivation + static_cast<int64_t>(rowIdx) * 2 * numCols);
        auto const* gateRow = linearRow + numVecs;
        auto* outRow = reinterpret_cast<float4*>(swigluOut + static_cast<int64_t>(rowIdx) * numCols);
        for (int vecIdx = threadIdx.x; vecIdx < numVecs; vecIdx += blockDim.x)
        {
            float4 linearVec = linearRow[vecIdx];
            float4 gateVec = gateRow[vecIdx];
            auto const* linearElems = reinterpret_cast<Type const*>(&linearVec);
            auto const* gateElems = reinterpret_cast<Type const*>(&gateVec);
            float4 outVec;
            auto* outElems = reinterpret_cast<Type*>(&outVec);
#pragma unroll
            for (int e = 0; e < ElemsPerVec; ++e)
            {
                Type const activated = static_cast<Type>(
                    swiglu_f32(static_cast<float>(gateElems[e]), static_cast<float>(linearElems[e])));
                outElems[e] = activated;
                threadMax = fmaxf(threadMax, fabsf(static_cast<float>(activated)));
            }
            outRow[vecIdx] = outVec;
        }
    }
    fused_publish_amax_and_barrier(threadMax, quantRange, eps, blockMaxBuf, retirementCount, globalScaleOut);
#endif
}

template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__device__ __noinline__ void fused_phase2_quantize(int32_t numRows, int32_t numCols, int32_t numPaddedCols,
    Type const* in, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout, float const* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVecT = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVecT) == sizeof(Type) * ELTS_PER_THREAD, "Vec size mismatch");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v2 only supports SF_VEC_SIZE of 16 or 32");

    float const SFScaleVal = globalScaleOut[1];

    bool isSf128x4Layout = layout == QuantizationSFLayout::SWIZZLED;
    bool isSf8x4Layout = layout == QuantizationSFLayout::R8C4;
    int numPaddedRowsForSf = isSf128x4Layout ? PadUpFn(numRows, 128) : (isSf8x4Layout ? PadUpFn(numRows, 8) : numRows);
    int numColsForSf = (isSf128x4Layout || isSf8x4Layout) ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
        {
            std::optional<int> optionalBatchIdx = 0;
            std::optional<int> optionalNumRows = numRows;

            auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

            int64_t inOffset = static_cast<int64_t>(rowIdx) * numColThreads + colIdx;
            int64_t outOffset = static_cast<int64_t>(rowIdx) * numPaddedColThreads + colIdx;

            if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
            {
                reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
            }

            if (rowIdx >= numRows || colIdx >= numColThreads)
            {
                if (sf_out != nullptr)
                {
                    sf_out[0] = 0x00;
                }
            }
            else
            {
                PackedVecT in_vec;
                load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVecT));

                if constexpr (Rule == AdaptiveScaleRule::NONE)
                {
                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_impl_opt<Type, SF_VEC_SIZE, false>(in_vec, SFScaleVal, sf_out);
                }
                else
                {
                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_adaptive<Type, SF_VEC_SIZE, Rule>(in_vec, SFScaleVal, sf_out);
                }
            }
        }
    }
#endif
}

/*
 * Phase2-only NVFP4 quantization.
 *
 * Runs ::fused_phase2_quantize directly against a caller-supplied
 * {amax, global_scale} pair -- no amax reduction, no retirement counter, and
 * no grid-wide barrier precede it. A single, non-persistent launch, so it is
 * safe to capture and replay: replay re-reads both `in` and `amaxScale` at
 * kernel-launch time, exactly like the other non-persistent kernels in this
 * file (e.g. opt_quantize_with_block_size_v1).
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 4) phase2_only_quantize(
#else
phase2_only_quantize(
#endif
        int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in, uint32_t* out, uint32_t* SFout,
        QuantizationSFLayout layout, float const* amaxScale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "phase2_only_quantize only supports FP16_TO_FP4");

    asm volatile("griddepcontrol.wait;");
    fused_phase2_quantize<quantization_type, Type, SF_VEC_SIZE, Rule>(
        numRows, numCols, numPaddedCols, in, out, SFout, layout, amaxScale);
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 3) fused_prologue_quantize_v2(
#else
fused_prologue_quantize_v2(
#endif
        int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in, float quantRange, float eps,
        uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout, float* blockMaxBuf, int* retirementCount,
        float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "fused_prologue_quantize_v2 only supports FP16_TO_FP4");

    asm volatile("griddepcontrol.wait;");
    fused_phase1_amax_barrier<Type>(numRows, numCols, in, quantRange, eps, blockMaxBuf, retirementCount,
        globalScaleOut);
    fused_phase2_quantize<quantization_type, Type, SF_VEC_SIZE, Rule>(numRows, numCols, numPaddedCols, in, out,
        SFout, layout, globalScaleOut);
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 3) fused_moe_prologue_quantize_v2(
#else
fused_moe_prologue_quantize_v2(
#endif
        int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int32_t tileSize, float quantRange,
        float eps, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout, float* blockMaxBuf,
        int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "fused_moe_prologue_quantize_v2 only supports FP16_TO_FP4");

    asm volatile("griddepcontrol.wait;");
    fused_moe_phase1_amax_barrier<Type>(numRows, numCols, in, tileIdxToMnLimit, numNonExitingTiles, tileSize,
        quantRange, eps, blockMaxBuf, retirementCount, globalScaleOut);
    fused_phase2_quantize<quantization_type, Type, SF_VEC_SIZE, Rule>(
        numRows, numCols, numPaddedCols, in, out, SFout, layout, globalScaleOut);
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

/*
 * SwiGLU + adaptive NVFP4 in one persistent launch.
 *
 * Phase 1 evaluates SwiGLU over the ``[numRows, 2 * numCols]`` pre-activation
 * and materializes its ``[numRows, numCols]`` BF16 result, so phase 2 is the
 * unmodified quantizer running over that activation and the SVDQuant FC2
 * low-rank correction still gets the exact tensor it needs.  The grid-wide
 * barrier between the two phases is what makes the hand-off legal.
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Kimi FC13 uses 128 threads for N=2048.  Requiring three resident
    // 512-thread CTAs nevertheless caps the compiler at roughly 42 registers
    // per thread and spills the SwiGLU/quant conversion state to local memory.
    // The persistent launcher caps this kernel at four CTAs per SM, so one
    // 512-thread CTA is the correct worst-case residency contract while the
    // occupancy planner still sizes the runtime grid from the compiled kernel.
    __launch_bounds__(512, 1) fused_moe_swiglu_prologue_quantize_v2(
#else
fused_moe_swiglu_prologue_quantize_v2(
#endif
        int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* preactivation, Type* swigluOut,
        int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int32_t tileSize, float quantRange,
        float eps, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout, float* blockMaxBuf,
        int* retirementCount, float* globalScaleOut)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "fused_moe_swiglu_prologue_quantize_v2 only supports FP16_TO_FP4");

    asm volatile("griddepcontrol.wait;");
    fused_moe_swiglu_phase1_amax_barrier<Type>(numRows, numCols, preactivation, swigluOut, tileIdxToMnLimit,
        numNonExitingTiles, tileSize, quantRange, eps, blockMaxBuf, retirementCount, globalScaleOut);
    fused_phase2_quantize<quantization_type, Type, SF_VEC_SIZE, Rule>(
        numRows, numCols, numPaddedCols, swigluOut, out, SFout, layout, globalScaleOut);
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Dequantize FP4 (E2M1) with swizzled SF to BF16/FP16.
//
// Reads packed FP4 data in row-major layout and SF in 128×4 swizzled layout
// (same as CuteDsl/CUTLASS output format), produces BF16 output.
//
// This avoids the need for a separate de-swizzle pass before dequant.

// E2M1 lookup table (device constant)
__device__ __constant__ float k_e2m1_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

template <typename OutType>
__global__ void dequant_nvfp4_swizzled_sf_kernel(
    int32_t numRows, int32_t numCols,       // numCols = interm_size (unpacked FP4 count)
    uint8_t const* __restrict__ fp4_packed,  // [numRows, numCols/2] uint8
    uint8_t const* __restrict__ sf_swizzled, // flat swizzled SF array
    float const* __restrict__ globalScale,   // scalar global scale on device
    int32_t sfVecSize,                       // 16
    QuantizationSFLayout layout,
    OutType* __restrict__ output)            // [numRows, numCols]
{
    int32_t sfCols = numCols / sfVecSize;
    float const globalScaleRcp = 1.0f / globalScale[0];
    int32_t packedCols = numCols / 2;

    for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
    {
        for (int colIdx = threadIdx.x; colIdx < packedCols; colIdx += blockDim.x)
        {
            // Read packed FP4 byte (2 values)
            uint8_t packed = fp4_packed[static_cast<int64_t>(rowIdx) * packedCols + colIdx];
            float val_lo = k_e2m1_lut[packed & 0x0F];
            float val_hi = k_e2m1_lut[(packed >> 4) & 0x0F];

            // Compute which SF block these two values belong to
            int unpackedCol_lo = colIdx * 2;
            int unpackedCol_hi = colIdx * 2 + 1;
            int sfCol_lo = unpackedCol_lo / sfVecSize;
            int sfCol_hi = unpackedCol_hi / sfVecSize;

            auto readSF = [&](int mIdx, int kIdx) -> float {
                int64_t offset;
                if (layout == QuantizationSFLayout::SWIZZLED)
                {
                    offset = get_sf_out_offset_128x4(std::nullopt, mIdx, kIdx, numRows, sfCols);
                }
                else if (layout == QuantizationSFLayout::R8C4)
                {
                    offset = get_sf_out_offset_8x4(std::nullopt, mIdx, kIdx, numRows, sfCols);
                }
                else
                {
                    offset = static_cast<int64_t>(mIdx) * sfCols + kIdx;
                }

                uint8_t sf_byte = sf_swizzled[offset];
                // Convert E4M3 to float (avoiding NaN: 0x7F and 0xFF)
                if (sf_byte == 0x7F || sf_byte == 0xFF)
                    return 0.0f;
                __nv_fp8_e4m3 sf_e4m3;
                sf_e4m3.__x = sf_byte;
                return static_cast<float>(sf_e4m3);
            };

            float sf_lo = readSF(rowIdx, sfCol_lo);
            float sf_hi = (sfCol_hi == sfCol_lo) ? sf_lo : readSF(rowIdx, sfCol_hi);

            // Dequant: x_orig ≈ fp4_val * sf / global_scale
            float deq_lo = val_lo * sf_lo * globalScaleRcp;
            float deq_hi = val_hi * sf_hi * globalScaleRcp;

            int64_t outIdx = static_cast<int64_t>(rowIdx) * numCols;
            output[outIdx + unpackedCol_lo] = static_cast<OutType>(deq_lo);
            output[outIdx + unpackedCol_hi] = static_cast<OutType>(deq_hi);
        }
    }
}

// computeGlobalAmaxKernel and computeGlobalAmax are in fp4QuantizeAdaptive.cu

} // namespace kernels

TRTLLM_NAMESPACE_END
