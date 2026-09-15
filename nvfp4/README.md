# NVFP4, 4o6 and 4o6 + SVDQ

This independently maintained `feat/nvfp4-4o6-svdq` branch packages the project's
vLLM adapter, standalone TRT-derived kernels, evaluation tooling and regression
tests. SVDQ means **4o6 + rank-64 SVDQ correction**. Direct NVFP4 and ModelOpt
NVFP4 are one serving-performance recipe.

## Runtime versions

The source branch starts at upstream vLLM `v0.24.0`
(`ee0da84ab9e04ac7610e28580af62c365e898389`). The measured B300 runtime was
NVIDIA vLLM **26.07**, reporting `0.24.0+092c4842.dev`, PyTorch
`2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3 and CuTe DSL 4.5.2. The release tag
and NVIDIA container build are different versions. Use the NVIDIA container
for the previously qualified adapter interface; installing a fresh build of
this source tree has not been qualified end to end.

The adapter is explicitly enabled on Python's import path. It does not replace
the installed vLLM package. Run from outside the checkout root so the source
tree does not shadow the container's installed vLLM. Preserve the container's
FlashInfer package, including `NVFP44Over6Config` and its 4o6 quantizers.
The custom CUDA sources below are included; no experiment directory is needed.

## Build and enable

Inside the compatible container, with this checkout mounted at `/opt/vllm-fork`:

```bash
export NVFP4_PYTHON=python
source /opt/vllm-fork/nvfp4/env.sh
cd /tmp
NVFP4_CUDA_ARCH=103a "$NVFP4_PYTHON" "$NVFP4_ROOT/native/quantization/build.py"
export B300_TRT_QUANT_LIBRARY="$NVFP4_ROOT/native/quantization/build/103a/nvfp4_trt_adaptive.so"
```

The B300 routing library builds on first use, before CUDA graph capture. Set
`VLLM_TRT_ROUTING_LIBRARY` to reuse a locally built library. Build/cache output
is ignored by Git. Native quantization retains the qualified FP32 adaptive
arithmetic, including its scale operation order.

Select a converted checkpoint and recipe:

```bash
export VLLM_NVFP4_CHECKPOINT=/models/Qwen3-8B-4o6
export VLLM_NVFP4_4O6_MODE=4o6  # standard | 4o6 | 4o6-svdq
export VLLM_NVFP4_SVDQ_RANK=64
"$NVFP4_PYTHON" -m vllm.entrypoints.cli.main serve "$VLLM_NVFP4_CHECKPOINT" \
  --dtype bfloat16 --enforce-eager --max-model-len 4096
```

For BF16, unset `VLLM_NVFP4_4O6_MODE`. `standard` selects the project's dynamic
standard NVFP4 adapter. For MoE, select `--enable-expert-parallel
--moe-backend flashinfer_cutedsl` with TP=1/local routing. Distributed expert
parallelism is not qualified by this port.

`B300_TRT_QUANT=1` optionally installs the measured fused dense 4o6 quantizer
and requires `B300_TRT_QUANT_LIBRARY`; use it only for `4o6` / `4o6-svdq`.
`VLLM_NVFP4_MAINLOOP_FUSION=1` selects the full B300 MoE mainloop port. It is
opt-in because the latest full port has layer/UT qualification, not a new
full-checkpoint serving regression. See [runtime scope](runtime/README.md).

## Checkpoint conversion

Use the vLLM converter included in this branch. It exports the
`offline-lowrank-residual-v1` format consumed by this adapter. The companion
TRT-LLM branch retains its original INT4-derived conversion/loading contract.

```bash
"$NVFP4_PYTHON" "$NVFP4_ROOT/converter/convert_4o6_ckpt.py" \
  --input /models/Qwen3-8B --output /models/Qwen3-8B-4o6 \
  --source-format bf16 --all-linear --quant-backend flashinfer \
  --activation-mode 4o6
```

Use `--activation-mode standard` for direct NVFP4. For the final Qwen SVDQ
recipe add `--svdquant-rank 64 --svdquant-decomposition quantization-error`.
Router and LM head stay BF16. Outputs and source checkpoints belong outside Git.

## Tests and evaluation

Install `pytest`, `safetensors` and the evaluation dependencies into the
appropriate runtime environment. The evaluation driver uses lm-eval-harness
0.4.10; keep its dependency resolution separate from vLLM/Torch/CUDA upgrades.

```bash
# CPU-only fixture, dispatch, configuration and evaluation compatibility checks:
bash "$NVFP4_ROOT/run_tests.sh" tests/test_shape_selection.py \
  tests/test_mainloop_dispatch.py tests/test_vllm_4o6_adapter_config.py \
  tests/test_vllm_4o6_eval_compat.py tests/test_vllm_4o6_mbpp_normalizer.py \
  tests/test_vllm_4o6_merge_shards.py tests/test_eval_tenacity_compat.py \
  tests/test_consolidate_final_accuracy.py tests/test_convert_4o6_ckpt_cli.py \
  tests/test_nvfp4_quantization_policy.py

# B300: one process per recipe; native dependencies must be built first.
for recipe in bf16 nvfp4 4o6 svdq; do
  RECIPE=$recipe bash "$NVFP4_ROOT/run_tests.sh" \
    tests/test_captured_dense.py tests/test_captured_moe.py
done

# Small accuracy smoke; omit --limit for a complete task.
bash "$NVFP4_ROOT/evaluation/run.sh" /models/Qwen3-8B-4o6 4o6 \
  gsm8k /results/qwen3-8b-4o6 --limit 8
```

For MoE evaluation add `VLLM_EXTRA_MODEL_ARGS=enable_expert_parallel=True,moe_backend=flashinfer_cutedsl`.
MMLU is 5-shot/batch 64; GSM8K is 5-shot and MBPP is 3-shot, both auto-batched.
`evaluation/analysis/` contains the existing shard merger, MBPP normalizer and
accuracy consolidation tools. MBPP executes generated code; use an isolated
evaluation container. These runs measure accuracy, not serving performance.

Serving throughput, TTFT and TPOT use [InferLab](benchmarks/inferlab/README.md).
Unit timing uses `B300_BENCHMARK=1`, reporting microseconds per layer or GEMM.
These are different measurements; profiling timing is not an unprofiled latency.

For **Qwen3-8B only on SM120**, build with `NVFP4_CUDA_ARCH=120a`, export
`SM120_TRT_QUANT_LIBRARY` to that build's library, then run
`tests/test_qwen3_8b_sm120.py` in separate processes for each `RECIPE`.
`SM120_BENCHMARK=1` enables timing. This suite selects 47 Qwen3-8B shapes and
does not include Qwen3-30B shapes. Its measured environment was vLLM 0.26.0
container, Torch 2.11.0+cu130 and FlashInfer 0.6.14, with CUTLASS SM120 GEMM.
This is standalone unit coverage; it does not establish SM120 serving support.

## Source and evidence

[SOURCES.md](SOURCES.md) records ownership and source provenance. The selected
shape JSON is a curated test fixture. Raw captures, benchmark results, traces,
checkpoints, environment caches and compiled libraries are excluded. Historical
performance results remain attached to their original experiment versions;
this packaging change does not claim a new GPU performance measurement.

For component development, install `ruff` and `pre-commit`, then use
`pre-commit install --config nvfp4/.pre-commit-config.yaml` and
`pre-commit run --config nvfp4/.pre-commit-config.yaml`. These component checks
preserve the measured donor formatting and verify its source hashes. Update
the manifest deliberately when changing runtime/native source, and repeat the
corresponding GPU qualification before attaching performance claims.
