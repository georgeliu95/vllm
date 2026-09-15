# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Build the qualified FP32 adaptive quantizer in the active CUDA environment.

Run with the runtime's Python: NVFP4_CUDA_ARCH=103a python build.py.
For Qwen3-8B SM120 unit tests, select NVFP4_CUDA_ARCH=120a.
"""

import logging
import os
import site
from pathlib import Path


def main() -> None:
    """Compile unchanged measured CUDA sources for the selected architecture."""
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parent
    source = root / "src/tensorrt_llm"
    architecture = os.environ.get("NVFP4_CUDA_ARCH", "103a")
    if architecture not in {"103a", "120a"}:
        raise ValueError(f"Unsupported NVFP4_CUDA_ARCH: {architecture}")
    build = Path(
        os.environ.get("NVFP4_QUANT_BUILD", str(root / "build" / architecture))
    )
    build.mkdir(parents=True, exist_ok=True)
    wheel_headers = sorted(
        path
        for base in site.getsitepackages()
        for path in Path(base).glob("nvidia/*/include")
        if path.is_dir()
    )
    sources = [
        source / "thop/fp4QuantizeAdaptiveOp.cpp",
        source / "kernels/fp4QuantizeAdaptive.cu",
        source / "common/assert.cpp",
        source / "common/tllmException.cpp",
        source / "common/stringUtils.cpp",
        source / "common/logger.cpp",
        root / "env_binding.cpp",
    ]
    load(
        name="nvfp4_trt_adaptive",
        sources=[str(path) for path in sources],
        extra_include_paths=[str(root / "include"), str(root / "src")]
        + [str(path) for path in wheel_headers],
        extra_cflags=["-O3", "-std=c++17", "-DENABLE_BF16", "-DENABLE_FP8"],
        extra_cuda_cflags=[
            "-DB300_TRT_QUANT_FP32=1",
            "-O3",
            "-std=c++17",
            "-DENABLE_BF16",
            "-DENABLE_FP8",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            f"-gencode=arch=compute_{architecture},code=sm_{architecture}",
            "-lineinfo",
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    logging.getLogger(__name__).warning("Library: %s", build / "nvfp4_trt_adaptive.so")


if __name__ == "__main__":
    main()
