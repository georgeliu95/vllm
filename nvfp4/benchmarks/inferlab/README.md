# InferLab serving performance

Use InferLab 0.15.0 with AIPerf 0.12.0 and its pinned Pixi lock. The supplied
workspace configuration preserves the existing five-model, four-recipe request
matrix, including prime windows. It contains no measurements or local paths.
Use a dedicated checkout/workspace of InferLab; preserve its `pixi.toml`,
`pixi.lock`, adapter SDK and vLLM integration. Install/run the serving side in
the NVIDIA vLLM 26.07 environment described in the parent README. A plain CPU
InferLab controller environment is not a replacement for that CUDA runtime.

```bash
mkdir -p "$INFERLAB_WORKSPACE/.inferlab"
cp "$NVFP4_ROOT/benchmarks/inferlab/workspace.toml" "$INFERLAB_WORKSPACE/.inferlab/workspace.toml"
cp "$NVFP4_ROOT/benchmarks/inferlab/local.example.toml" "$INFERLAB_WORKSPACE/.inferlab/local.toml"
ln -s "$NVFP4_ROOT" "$INFERLAB_WORKSPACE/nvfp4"
```

Edit the local file to choose an available GPU and set checkpoint locators.
It is deliberately untracked. `NVFP4_CHECKPOINT` below must match that recipe's
locator. Set `NVFP4_RUNTIME_PYTHONPATH` if the serving container needs explicit
site-packages / CuTe `.pth` paths; these paths must exist on the serving machine.

```bash
export NVFP4_CHECKPOINT=/models/qwen3-8b-4o6
bash "$NVFP4_ROOT/benchmarks/inferlab/run.sh" "$INFERLAB_WORKSPACE" qwen3-8b 4o6 --dry-run
bash "$NVFP4_ROOT/benchmarks/inferlab/run.sh" "$INFERLAB_WORKSPACE" qwen3-8b 4o6
```

The model IDs are `qwen3-8b`, `qwen3-30b-a3b`, `qwen3-5-27b`,
`qwen3-5-35b-a3b`, and `qwen3-8-27b`. Recipes are `bf16`, `nvfp4`, `4o6`, `svdq`.
The adapter/native feature switches are explicit overrides. Default settings
use the retained partial MoE port; the full port remains opt-in and requires a
fresh serving qualification. The old unoptimized and partially optimized
serving results do not describe the new full port.

The scored populations include input/output tokens 128/128, 2048/128 and
128/1024 at concurrency 1/16. Prime populations are excluded from comparisons.
Use native InferLab records for throughput, TTFT and TPOT. Run paired recipes
on the same GPU, weights, token populations and runtime, with warmup; record
the feature commit and selected native library hash. These are small-sample
measurements, not full MMLU/GSM8K/MBPP accuracy evaluations.

Record outputs outside the repository. Nsys capture is a separate profiling
run; do not substitute profiled request latency for unprofiled performance.
