# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone binding of the project's original TRT-LLM CUDA routing kernels."""

from functools import cache
import os
from pathlib import Path

import torch


@cache
def load_library() -> None:
    """Load a prebuilt library, or compile the vendored donor before graph capture."""
    library = os.environ.get("VLLM_TRT_ROUTING_LIBRARY")
    if library:
        torch.ops.load_library(library)
        return
    from torch.utils.cpp_extension import load
    import flashinfer

    root = Path(__file__).parent / "native"
    include = root / "include"
    kernels = include / "tensorrt_llm/kernels/trtllmGenKernels"
    routing = kernels / "blockScaleMoe/routing"
    cutlass = Path(flashinfer.__file__).parent / "data/cutlass/include"
    build = os.environ.get("VLLM_TRT_ROUTING_BUILD")
    if build:
        Path(build).mkdir(parents=True, exist_ok=True)
    load(
        name="project_trt_routing",
        sources=[
            str(root / "binding.cpp"),
            str(routing / "RoutingCustom.cu"),
            str(routing / "RoutingFromTopKIds.cu"),
        ],
        extra_include_paths=[
            str(include),
            str(cutlass),
            str(kernels / "batchedGemm/trtllmGen_bmm_export"),
        ],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-lineinfo",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_103a,code=sm_103a",
        ],
        build_directory=build,
        is_python_module=False,
        verbose=True,
    )


def compact_sort(
    ids: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    start: int,
    local_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return original TRT tile groups, row limits, expanded permutation, and live count."""
    load_library()
    groups, limits, permutation, live = torch.ops.project_trt_routing.compact_sort(
        ids, weights, num_experts, start, local_experts
    )
    return groups, limits, permutation, live
