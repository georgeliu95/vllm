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
 * Adaptive 4/6 (FourOverSix) NVFP4 quantization kernels.
 *
 * Extends the base FP4 quantization (quantization.h) with:
 *   - kernelVersion=1: Optimized v1 kernel (16 elts/thread, 256-bit loads)
 *   - scaleRule=1/2/3: Adaptive per-block max_e2m1 selection {4, 6} via MSE/MAE/ABS_MAX
 *   - computeGlobalAmax: Single-kernel last-block reduction for runtime amax + global_scale
 *
 * Derived from tllm_linear_lite (https://github.com/nvidia/tllm_linear_lite).
 */

#pragma once

#include "tensorrt_llm/kernels/quantization.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

/// Per-block adaptive scale rule for 4/6 selection.
enum class AdaptiveScaleRule
{
    NONE = 0,    ///< Standard NVFP4 (max_e2m1=6, no selection)
    MSE = 1,     ///< Minimize mean squared error
    MAE = 2,     ///< Minimize mean absolute error
    ABS_MAX = 3, ///< Minimize max absolute error
};

/// Extended FP4 quantization supporting kernelVersion and adaptive scaleRule.
///
/// @param kernelVersion  0 = v0 (8 elts/thread, same as invokeFP4Quantization),
///                       1 = v1 (16 elts/thread, 256-bit loads, better throughput)
/// @param scaleRule      0 = standard NVFP4, 1/2/3 = adaptive 4/6 (MSE/MAE/ABS_MAX)
template <typename T, int SF_VEC_SIZE = 16>
void invokeFP4QuantizationEx(int b, int m, int n, T const* input, float const* globalScale, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream = 0,
    int kernelVersion = 1, int scaleRule = 0);

/// Compute global amax and optional global_scale in a single kernel launch.
///
/// Uses last-block reduction pattern: no zero-init required for blockMaxBuf,
/// self-resetting atomic counter.
///
/// @param blockMaxBuf     Temp buffer of size >= gridDim.x floats (no init needed)
/// @param output          output[0] = max(amax, eps), output[1] = quantRange/amax (or amax if quantRange==0)
/// @param retirementCount Atomic counter, must be 0 on first call (self-resets after each invocation)
/// @param quantRange      If > 0, output[1] = quantRange / max(amax, eps). If 0, output[1] = amax.
/// @param eps             Floor for amax to prevent division by zero
template <typename T>
void computeGlobalAmax(int m, int n, T const* input, float* blockMaxBuf, float* output, int* retirementCount,
    float quantRange, float eps, int multiProcessorCount, cudaStream_t stream = 0);

/// Fused amax + FP4 quantization for static NVFP4.
///
/// Returns false when the fused kernel is not eligible, so callers can fall
/// back to computeGlobalAmax + invokeFP4QuantizationEx.
template <typename T, int SF_VEC_SIZE = 16>
bool invokeFusedPrologueQuantization(int m, int n, T const* input, float quantRange, float eps, int64_t* output,
    int32_t* SFOutput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream = 0, int testMaxActiveBlocks = 0);

/// Fused amax + FP4 quantization for adaptive 4/6 rules.
///
/// Uses the same packed output and SF layouts as invokeFP4QuantizationEx,
/// including LINEAR, R8C4, and 128x4 SWIZZLED.
template <typename T, int SF_VEC_SIZE = 16, AdaptiveScaleRule Rule = AdaptiveScaleRule::MSE>
bool invokeFusedPrologueQuantizationV2(int m, int n, T const* input, float quantRange, float eps, int64_t* output,
    int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream = 0, int testMaxActiveBlocks = 0);

/// Phase2-only NVFP4 quantization for a precomputed amax/global_scale pair.
///
/// Runs exactly the phase-2 quantize step used by ::invokeFusedPrologueQuantizationV2
/// -- no amax reduction, no retirement counter, no grid-wide barrier, and no
/// elementwise scale correction. The kernel is a single, non-persistent launch,
/// so it is safe under CUDA graph capture/replay: replay re-reads both @p input
/// and @p amaxScale at kernel-launch time.
///
/// @param amaxScale  Device pointer to 2 floats: {amax, quantRange / max(amax, eps)}.
///                    Only amaxScale[1] (the global scale) is read.
template <typename T, int SF_VEC_SIZE = 16, AdaptiveScaleRule Rule = AdaptiveScaleRule::NONE>
bool invokeFp4QuantizePhase2(int m, int n, T const* input, float const* amaxScale, int64_t* output,
    int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream = 0);

/// MoE-specialized fused adaptive quantization.
///
/// The amax phase excludes routing padding using ``tileIdxToMnLimit`` and the
/// device-resident live-tile count. Quantization may write padding rows because
/// downstream grouped GEMMs apply the same routing mask.
template <typename T, int SF_VEC_SIZE = 16, AdaptiveScaleRule Rule = AdaptiveScaleRule::MSE>
bool invokeFusedMoePrologueQuantizationV2(int m, int n, T const* input, int32_t const* tileIdxToMnLimit,
    int32_t const* numNonExitingTiles, int tileSize, float quantRange, float eps, int64_t* output,
    int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream = 0, int testMaxActiveBlocks = 0);

/// MoE-specialized fused SwiGLU + adaptive quantization.
///
/// Phase 1 evaluates SwiGLU over the ``[m, 2 * n]`` FC13 pre-activation (linear
/// half first, gate half second) and stores the ``[m, n]`` BF16 activation the
/// SVDQuant FC2 low-rank correction consumes; phase 2 quantizes that stored
/// activation. The amax reduction excludes routing padding, whose rows are left
/// untouched, exactly as the standalone activation kernel left them.
///
/// @param preactivation  [m, 2 * n] FC13 output
/// @param swigluOutput   [m, n] BF16 activation, written by phase 1
template <typename T, int SF_VEC_SIZE = 16, AdaptiveScaleRule Rule = AdaptiveScaleRule::MSE>
bool invokeFusedMoeSwigluPrologueQuantizationV2(int m, int n, T const* preactivation, T* swigluOutput,
    int32_t const* tileIdxToMnLimit, int32_t const* numNonExitingTiles, int tileSize, float quantRange, float eps,
    int64_t* output, int32_t* SFOutput, QuantizationSFLayout layout, int multiProcessorCount, float* blockMaxBuf,
    int* retirementCount, float* globalScaleOut, cudaStream_t stream = 0, int testMaxActiveBlocks = 0);

/// Dequantize FP4 (packed E2M1) with TRTLLMGen/CUTLASS SF layout to BF16.
/// Supports 128x4 swizzled, 8x4 TRTLLMGen, and linear layouts.
/// @param numRows     M dimension
/// @param numCols     Unpacked column count (interm_size, NOT packed)
/// @param fp4_packed  [numRows, numCols/2] uint8 packed FP4
/// @param sf_swizzled Flat swizzled SF array
/// @param globalScale Device pointer to a per-tensor global scale (e.g. fc2_input_scale = 2688/max_amax)
/// @param sfVecSize   Scale factor vector size (16)
/// @param output      [numRows, numCols] bf16/fp16 output
template <typename OutType>
void invokeDequantNvfp4SwizzledSF(int numRows, int numCols,
    uint8_t const* fp4_packed, uint8_t const* sf_swizzled,
    float const* globalScale, int sfVecSize, OutType* output,
    QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream = 0);

/// Elementwise alpha correction for adaptive 4/6 FC2:
///   out[i] = alphaIn[i] * fc2InputScale / dynamicGlobalScale[0]
///
/// @param alphaIn              Per-expert alpha, [numExperts] float
/// @param dynamicGlobalScalePtr Pointer to a single float on device (= quantRange / amax)
/// @param fc2InputScale        Static global scale from calibration
/// @param numExperts           Number of local experts
/// @param alphaOut             [numExperts] float output (may alias alphaIn)
void invokeScaleAlphaByAmax(float const* alphaIn, float const* dynamicGlobalScalePtr,
    float fc2InputScale, int numExperts, float* alphaOut, cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
