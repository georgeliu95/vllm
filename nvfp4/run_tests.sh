#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
unset VLLM_NVFP4_4O6_MODE VLLM_NVFP4_EVAL_COMPAT B300_TRT_QUANT
source "$(dirname -- "$0")/env.sh"
export PYTHONPATH="$NVFP4_ROOT:$NVFP4_ROOT/tests:$PYTHONPATH"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
cd "$NVFP4_ROOT"
exec "${NVFP4_PYTHON:-python}" -m pytest -c pytest.ini --confcutdir=tests "$@"
