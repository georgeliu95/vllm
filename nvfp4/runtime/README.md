# Runtime scope

All runtime packages required by this adapter are included together here.
Use the parent [installation guide](../README.md) for environment selection,
native builds and activation. No experiment directory is imported.

The default path retains the previously tested partial fusion implementation.
Dense standard NVFP4 uses a two-kernel global-scale reduction. Dense 4o6 and
4o6 + SVDQ can use the qualified fused quantizer with `B300_TRT_QUANT=1`.
The partial MoE path keeps grouped FlashInfer GEMMs and fuses bounded local
routing, scaling and low-rank reconstruction at supported shapes.

On SM12x (RTX PRO Blackwell) the 4o6 MoE runs `vllm_adapter/sm120_moe.py` instead of the
padded `[E, capacity, K]` path, unless `VLLM_NVFP4_SM120_MOE=0`. It replaces the grouped-GEMM
fallback that copied every expert's weights and synchronized with the host on each call, so the
model could run only eagerly. The new path is safe under CUDA graph capture:

- Routes are sorted by expert in one program up to 2048 routes, else by vLLM's
  `get_cutlass_moe_mm_data`.
- The expert GEMMs are vLLM's SM120 `cutlass_fp4_group_mm`, or, up to 64 routes, a grouped NVFP4
  GEMV (`csrc/moe_sm120.cu`) with the same operands. Each 16-element block product is exact in
  FP32, so only the accumulation order differs.
- Activations are quantized by FlashInfer's TRT-LLM 4o6 device code with the reference's template
  configuration (E4M3 max 256, MSE). Each 64-expert chunk shares one dynamic global scale, as in
  the reference's per-chunk calls.
- Rank-64 SVDQ corrections are Triton kernels. The projection is rounded to BF16, and the
  correction is added to the BF16 GEMM output with BF16 rounding.
- Routes are combined in ascending expert order with BF16 rounding after every add.

Against the reference path on synthetic TP2 shapes (`E=256, H=2048, I=256`), outputs are
bit-identical at M=1, 200, 1024 and 4096. At M=4..64 the reference's cuBLAS correction GEMMs
differ by at most a few BF16 ulps. Shared experts are left to vLLM's runner, as with the batched
kernel this path replaces.

`VLLM_NVFP4_MAINLOOP_FUSION=1` enables the complete B300 token-major MoE port:

- Original compact TRT routing supports multiple tiles per expert.
- Input quantization uses the standalone native FP32 adaptive quantizer.
- FC13 combines FP4 GEMM, SVDQ reconstruction, SwiGLU and live-row amax.
- FC2 combines FP4 GEMM, SVDQ reconstruction, router weighting and BF16 scatter-add.
- Four low-rank down-projection GEMMs remain per full 128-expert layer call.

The optional path is qualified for BF16 input, local unique top-k routing,
M=1..16384, aligned weights in 64-expert chunks, and no correction or full
rank-64 SVDQ correction. It packs gate/up weights, scales and factors at load
time. Shared experts, EP maps, input router weighting and other activation
contracts are outside the measured full-port scope. Existing fallback dispatch
is retained where implemented; it is not an assertion of distributed support.

The donor scatter-add has nondeterministic BF16 reduction order; fused
correction changes intermediate rounding. Extra packed weight copies remain
resident. Full-checkpoint serving and memory usage need separate qualification.

The FC13 amax publication accounts for all three CTA grid dimensions, including
CuTe 4.5.2 persistent scheduling in Z. `VLLM_MAINLOOP_AMAX_EXTERNAL=1` selects
an independent diagnostic reduction and is not the normal benchmark path.

Donor details are in `trt_mainloop/PROVENANCE.json`,
`trt_routing/native/DONOR.json`, and `trt_kernels/runner.provenance.json`.
The parent source manifest fixes the packaged runtime files.
