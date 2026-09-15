#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if (( $# < 4 )); then
    echo 'usage: run.sh CHECKPOINT bf16|nvfp4|4o6|svdq mmlu|gsm8k|mbpp OUTPUT [lm-eval arguments...]' >&2
    exit 2
fi
checkpoint=$(realpath -- "$1")
recipe=$2 task=$3
output=$(realpath -m -- "$4")
shift 4
source "$(dirname -- "$0")/../env.sh"
export PYTHONPATH="$NVFP4_ROOT/evaluation:$PYTHONPATH"
export VLLM_NVFP4_EVAL_COMPAT=1 VLLM_NVFP4_DISABLE_THINKING=1
export VLLM_NVFP4_CHECKPOINT="$checkpoint"
case "$recipe" in
    bf16) unset VLLM_NVFP4_4O6_MODE ;;
    nvfp4) export VLLM_NVFP4_4O6_MODE=standard ;;
    4o6) export VLLM_NVFP4_4O6_MODE=4o6 ;;
    svdq) export VLLM_NVFP4_4O6_MODE=4o6-svdq VLLM_NVFP4_SVDQ_RANK=64 ;;
    *) echo "Unknown recipe: $recipe" >&2; exit 2 ;;
esac
case "$task" in
    mmlu) task_args=(--batch_size "${MMLU_BATCH_SIZE:-64}" --num_fewshot 5) ;;
    gsm8k) task_args=(--batch_size auto --num_fewshot 5 --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS:-2048}") ;;
    mbpp)
        export HF_ALLOW_CODE_EVAL=1
        task_args=(--batch_size auto --num_fewshot 3 --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS:-2048}" --confirm_run_unsafe_code)
        ;;
    *) echo "Unknown task: $task" >&2; exit 2 ;;
esac
model_args="pretrained=$checkpoint,dtype=bfloat16,enforce_eager=True,tensor_parallel_size=${TP_SIZE:-1}"
if [[ -n ${VLLM_EXTRA_MODEL_ARGS:-} ]]; then
    model_args+=",$VLLM_EXTRA_MODEL_ARGS"
fi
# Avoid importing the repository's unbuilt vllm tree over the installed runtime.
cd /tmp
exec "${NVFP4_PYTHON:-python}" -m lm_eval --model vllm --model_args "$model_args" \
    --tasks "$task" --apply_chat_template --fewshot_as_multiturn --log_samples \
    --output_path "$output" "${task_args[@]}" "$@"
