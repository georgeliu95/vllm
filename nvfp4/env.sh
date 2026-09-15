#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# Source this file inside the documented runtime environment.
NVFP4_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export NVFP4_ROOT
export PYTHONPATH="$NVFP4_ROOT/runtime${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
