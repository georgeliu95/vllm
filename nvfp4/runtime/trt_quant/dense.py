# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replace runtime scale and activation quantization, preserving the CuTe GEMM."""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Protocol

import torch
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4LinearKernel,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    pad_nvfp4_activation_for_cutlass,
    slice_nvfp4_output,
)
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm


class QuantLayer(Protocol):
    """Processed dense weight layout consumed by the frozen CuTe backend."""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale_inv: torch.Tensor
    alpha: torch.Tensor
    output_size_per_partition: int
    weights_padding_cols: int


class QuantMethod(Protocol):
    """The original quantization method's selected matrix kernel."""

    kernel: FlashInferCuteDslNvFp4LinearKernel


@dataclass(frozen=True, slots=True)
class IntegrationError(RuntimeError):
    """An unsupported serving contract cannot silently fall back."""

    reason: str

    def __str__(self) -> str:
        return self.reason


Apply = Callable[[QuantMethod, QuantLayer, torch.Tensor, torch.Tensor | None], torch.Tensor]


@cache
def load_library() -> None:
    """Load operators only inside a process that actually performs quantization."""
    torch.ops.load_library(os.environ["B300_TRT_QUANT_LIBRARY"])
    logging.getLogger("b300_trt_quant").warning(
        "B300_TRT_QUANT_LIBRARY_LOADED path=%s", os.environ["B300_TRT_QUANT_LIBRARY"],
    )


def fused_output(
    layer: QuantLayer, values: torch.Tensor, bias: torch.Tensor | None,
) -> torch.Tensor:
    """Keep Torch operator references out of the serialized dynamic method closure."""
    load_library()
    matrix = values.reshape(-1, values.shape[-1]).contiguous()
    packed, scales, amax_scale = torch.ops.trtllm.fp4_quantize_fused(
        matrix, 16, False, True, 1, 1536.0, 1e-12, 0, 0,
    )
    scales = scales.view((matrix.shape[0] + 127) // 128 * 128, -1)
    with torch.no_grad():
        layer.input_global_scale_inv.copy_(amax_scale[1])
        layer.alpha.copy_((amax_scale[0] / 1536.0) * layer.weight_global_scale)
    packed = pad_nvfp4_activation_for_cutlass(packed, layer.weights_padding_cols)
    output = flashinfer_scaled_fp4_mm(
        packed, layer.weight, scales.view(torch.float8_e4m3fn),
        layer.weight_scale, layer.alpha, values.dtype, backend="cute-dsl",
    )
    output = slice_nvfp4_output(output, layer.output_size_per_partition)
    if bias is not None:
        output = output + bias
    return output.view(*values.shape[:-1], layer.output_size_per_partition)


def make_apply(reference: Apply) -> Apply:
    """Create a measured replacement with optional real-layer qualification."""
    validated: set[tuple[int, tuple[int, ...]]] = set()
    validate = os.environ.get("B300_TRT_VALIDATE") == "1"
    logger = logging.getLogger("b300_trt_quant")

    def apply(
        method: QuantMethod, layer: QuantLayer, values: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not isinstance(method.kernel, FlashInferCuteDslNvFp4LinearKernel):
            raise IntegrationError("The port requires the frozen FlashInfer CuTe GEMM")
        output = fused_output(layer, values, bias)
        key = (layer.weight.data_ptr(), tuple(values.shape))
        if validate and key not in validated:
            expected = reference(method, layer, values, bias)
            assert torch.isfinite(output).all()
            energy = expected.float().square().mean().clamp_min(1e-20)
            nmse = float(((output.float() - expected.float()).square().mean() / energy).item())
            if nmse > 0.001:
                raise IntegrationError(f"Fused dense output NMSE exceeds 0.001: {nmse}")
            validated.add(key)
            logger.warning("B300_TRT_QUANT_VALIDATED layers_shapes=%d shape=%s nmse=%.9g", len(validated), tuple(values.shape), nmse)
        return output

    return apply


def install() -> None:
    """Patch only the dynamic base method, leaving the SVDQ subclass intact."""
    from vllm.model_executor.layers.quantization import modelopt

    matches = [base for base in modelopt.ModelOptNvFp4LinearMethod.__mro__ if base.__name__ == "RuntimeDynamicNvFp4LinearMethod"]
    if len(matches) != 1:
        raise IntegrationError("Expected exactly one frozen dynamic quantization base")
    base = matches[0]
    base.apply = make_apply(base.apply)
    logging.getLogger("b300_trt_quant").warning("B300_TRT_QUANT_INSTALLED pid=%d validation=%s", os.getpid(), os.environ.get("B300_TRT_VALIDATE", "0"))
