#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
if (( $# < 3 )); then
    echo 'usage: run.sh INFERLAB_WORKSPACE MODEL-ID bf16|nvfp4|4o6|svdq [InferLab arguments...]' >&2
    exit 2
fi
workspace=$(realpath -- "$1")
model=$2 recipe=$3
shift 3
bundle=$(cd -- "$(dirname -- "$0")/../.." && pwd)
: "${NVFP4_CHECKPOINT:?Set NVFP4_CHECKPOINT to the local.toml checkpoint for this recipe}"
case "$recipe" in
    bf16) mode='' ;;
    nvfp4) mode=standard ;;
    4o6) mode=4o6 ;;
    svdq) mode=4o6-svdq ;;
    *) echo "Unknown recipe: $recipe" >&2; exit 2 ;;
esac
# Overrides are TOML literals. Reject delimiters rather than interpolate unsafe literals.
for value in "$bundle" "$NVFP4_CHECKPOINT" "${NVFP4_RUNTIME_PYTHONPATH:-}" "${B300_TRT_QUANT_LIBRARY:-}"; do
    if [[ $value == *\"* || $value == *\\* || $value == *$'\n'* ]]; then
        echo 'Paths must not contain quotes, backslashes or newlines.' >&2
        exit 2
    fi
done
patches=(
    --set "server.settings.extra_env.PYTHONPATH=\"$bundle/runtime${NVFP4_RUNTIME_PYTHONPATH:+:$NVFP4_RUNTIME_PYTHONPATH}\""
    --set "server.settings.extra_env.VLLM_NVFP4_4O6_MODE=\"$mode\""
    --set "server.settings.extra_env.VLLM_NVFP4_CHECKPOINT=\"$NVFP4_CHECKPOINT\""
    --set 'server.settings.extra_env.VLLM_NVFP4_SVDQ_RANK="64"'
    --set 'server.settings.extra_env.VLLM_NVFP4_EVAL_COMPAT=""'
    --set "server.settings.extra_env.B300_TRT_QUANT=\"${B300_TRT_QUANT:-0}\""
    --set "server.settings.extra_env.B300_TRT_QUANT_LIBRARY=\"${B300_TRT_QUANT_LIBRARY:-}\""
    --set "server.settings.extra_env.VLLM_NVFP4_MAINLOOP_FUSION=\"${VLLM_NVFP4_MAINLOOP_FUSION:-0}\""
)
unset PYTHONPATH VLLM_NVFP4_4O6_MODE VLLM_NVFP4_EVAL_COMPAT
exec inferlab recipe run "$model-$recipe" --workspace "$workspace" \
    --local "$workspace/.inferlab/local.toml" "${patches[@]}" "$@"
