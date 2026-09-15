# Qwen3 captured-shape regression set

The selected fixture contains **86 dense M/N/K cases and 15 routed MoE layer
cases** from the completed OCI-HSG full-v3 MMLU/GSM8K/MBPP captures. The full
capture remains at `out/qwen3-eval-shapes-oci-20260914` in the parent workspace.
This subset replaces an exhaustive sweep at the user's request.

`data/qwen3_shapes.json` records provenance and the selection rule. Dense cases
cover every projection size in both models, observed M range endpoints, a
frequency-weighted median, and nearest observed M to 16/127/128/129/512/1024/4096.
MoE cases cover decode, prefill and mixed steps, each task's median/maximum,
both sides of the fusion boundary, and the most skewed observed layer for each
selected phase/M. The exact 128 expert counts include idle experts.

The tests reconstruct unique top-8 routes with those counts. Token/expert
correlations, router weights, activations and model weights are **synthetic**;
these are dimension/occupancy regression cases, not checkpoint-output replay.
All four recipes use identical seeds and occupancy. SVDQ means 4o6+SVDQ R64,
including both gate/up branches and the down correction. Router and LM head
remain BF16 for every recipe, so those cases run once under BF16.

Dense numerical checks cover the whole output for finiteness/shape, and an
FP32 reference for 16 input rows and approximately 256 output columns. NVFP4
also checks full-call/prequantized-call equality. MoE checks 16 spread-out token
rows against a grouped FP32 formula with rank-64 factors of scale .02. For
M<=128, the mainloop result additionally agrees with the full batched fallback
within NMSE .005. The prior quantization-error thresholds are preserved:
dense .05, MoE .1, BF16 dense 2e-5 and BF16 MoE .001. These thresholds do not
constitute model accuracy evaluation.

The MoE test calls the maintained `make_dynamic_moe_method(...).apply` dispatch:
Every selected M uses native compact TRT routing and fused mainloops; entering the
legacy batched fallback fails the test. The compact allocation uses the donor
worst-case tile bound, while each GEMM consumes only live tiles. The log field
`capacity` is the old per-expert capacity for baseline reference, not the new allocation. FC13=(1536,2048), FC2=(2048,768), E=128, top-k=8.

Run CPU fixture checks with `python -m pytest tests/test_shape_selection.py`.
GPU tests use the pinned B300 vLLM image and explicit TRT dependency paths;
the reproducible launch snapshot and logs are retained under
`out/b300-eval-shapes-20260915`. Run each recipe in its own process because FP4
quantization flags affect kernel compilation. `B300_BENCHMARK=1` enables
separate eager/CUDA graph timing of the same unit/layer calls after validation.
This reports microseconds per projection/layer; serving throughput and TTFT
still require InferLab.

`test_trt_routing.py` covers multi-tile routing, skew, empty chunks and changing
CUDA graph routes through M=16384. `test_mainloop_scale.py` checks that unrouted
outliers do not affect a shard's input scale. `test_mainloop_math.py` checks
actual packed FC13/FC2 and SVDQ operands independently, exact live-row amax,
poisoned inactive low-rank rows, and changed-input/chunk graph replays. The
complete migration snapshot and logs are in `out/b300-full-trt-migration-20260915`.

## Qwen3-8B on RTX PRO 5000 / SM120

Run `test_qwen3_8b_sm120.py` explicitly on SM120. It selects only the existing
47 Qwen3-8B dense shapes: 40 quantized cases and 7 shared BF16 LM-head cases.
No Qwen3-30B dense or MoE shapes enter this suite. Each recipe runs in a separate
process with `RECIPE=bf16|nvfp4|4o6|svdq`; quantized recipes also require
`SM120_TRT_QUANT_LIBRARY` pointing to the compiled native quantizer.
`SM120_BENCHMARK=1` enables separate eager and CUDA graph timing.

The explicit backend is CUTLASS SM120. All FP4 recipes use the same native
fused amax/quantization API; SVDQ remains 4o6 plus two BF16 rank-64 correction
GEMMs and an add. This suite does not claim SM100 mainloop or serving support.
Packed formulas and changed-input/zero-input graph replays precede timing.
The tested snapshot, build wrapper, logs and tables are retained under
`out/pro5000-qwen3-8b-unit-20260915/`.
