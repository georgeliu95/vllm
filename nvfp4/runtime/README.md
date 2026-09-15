# Runtime scope

All runtime packages required by this adapter are included together here.
Use the parent [installation guide](../README.md) for environment selection,
native builds and activation. No experiment directory is imported.

The default path retains the previously tested partial fusion implementation.
Dense standard NVFP4 uses a two-kernel global-scale reduction. Dense 4o6 and
4o6 + SVDQ can use the qualified fused quantizer with `B300_TRT_QUANT=1`.
The partial MoE path keeps grouped FlashInfer GEMMs and fuses bounded local
routing, scaling and low-rank reconstruction at supported shapes.

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
