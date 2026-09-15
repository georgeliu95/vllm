# Source ownership and reproducibility

This directory is the maintained vLLM feature implementation. The previous
workspace-only `vllm-4o6-svdq/runtime` directory and dated experiment snapshots
are historical source records; they are not required at installation/runtime.

| Component | Source and scope |
|---|---|
| `runtime/vllm_adapter` | Latest maintained Qwen3 dense/local-MoE adapter, including the B300 full TRT mainloop port |
| `runtime/trt_mainloop` | TRT-LLM donor kernels plus standalone CuTe 4.5.2 adaptations; per-file donor hashes in `PROVENANCE.json` |
| `runtime/trt_routing` | Original TRT CUDA routing plus independent Torch binding; `native/DONOR.json` identifies the donors |
| `runtime/trt_kernels` | Grouped low-rank implementation retained from the B300 TRT port; runner provenance accompanies the source |
| `runtime/trt_quant` | Qualified FP32 fused dense integration, including the corrected scale arithmetic order |
| `native/quantization` | Standalone FP32 adaptive CUDA operator source used by B300 and SM120 unit runs; architecture is selected at build time |
| `runtime/sitecustomize.py` | Original lm-eval/vLLM compatibility installer, now with an explicit optional fused dense hook |
| `evaluation` | Existing lm-eval compatibility, shard merger, MBPP normalization and accuracy consolidation; portable launcher added |
| `converter` | Recent standalone direct NVFP4 / 4o6 / quantization-error SVDQ converter for vLLM; separated from TRT's legacy INT4 metadata contract |
| `tests/data/qwen3_shapes.json` | Curated selection from OCI-HSG full-v3 task captures, containing dimensions/occupancy rather than requests or checkpoints |

The TRT donor branch tip is `847572282a43cac577f005bc8c5a0a710bb21974` in
`georgeliu95/TensorRT-LLM`. The extracted quantizer includes the previously
qualified FP32 compatibility variant; it is not represented as identical to
an unmodified stock TRT operator. Licenses/copyrights are retained in source.

`SOURCE_SHA256SUMS` fixes the runtime/native source content packaged here.
Check from the repository root with `sha256sum -c nvfp4/SOURCE_SHA256SUMS`.
These are source hashes, not a bundle of generated benchmark results.

## Historical qualification boundaries

- Prior InferLab serving runs qualified the partial grouped low-rank/dense
  quantizer port in the pinned NVIDIA vLLM container.
- Latest full B300 MoE port: local unique top-k, BF16 input, rank 64, 64-expert
  chunks, M through 16384. Layer and graph-replay qualification does not imply
  full-checkpoint memory, EP or serving qualification.
- Qwen3-8B SM120: 47 selected dense shapes, separate CUTLASS SM120 backend,
  standalone layer tests. The SVDQ correction remains two BF16 GEMMs plus add.
- CPU packaging validation checks configuration/dispatch, selected-shape
  invariants, conversion metadata and evaluation tooling. GPU performance
  measurements have not been rerun for this source reorganization.

Record new branch commits with future experiments. Keep raw outputs and
profiles outside the repository and cite their artifact locations separately.
