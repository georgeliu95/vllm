"""Explicit SM120 CUTLASS dense path with the project's native FP4 quantizer."""

import os
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from typing import assert_never

import flashinfer
import torch


class Recipe(StrEnum):
    BF16 = "bf16"
    NVFP4 = "nvfp4"
    FOUR = "4o6"
    SVDQ = "svdq"


@cache
def load_quantizer() -> None:
    """Load the explicitly supplied SM120 build without a JIT fallback."""
    torch.ops.load_library(os.environ["SM120_TRT_QUANT_LIBRARY"])


@dataclass(frozen=True, slots=True)
class PackedActivation:
    codes: torch.Tensor
    scales: torch.Tensor
    alpha: torch.Tensor


@dataclass(frozen=True, slots=True)
class DenseLayer:
    """One synthetic layer; low-rank factors follow the existing dense tests."""

    recipe: Recipe
    weight: torch.Tensor
    weight_codes: torch.Tensor | None
    weight_scales: torch.Tensor | None
    weight_dequant: torch.Tensor | None
    us: torch.Tensor | None
    vh: torch.Tensor | None

    @property
    def quant_range(self) -> float:
        match self.recipe:
            case Recipe.BF16 | Recipe.NVFP4:
                return 2688.0
            case Recipe.FOUR | Recipe.SVDQ:
                return 1536.0
            case unreachable:
                assert_never(unreachable)

    @property
    def scale_rule(self) -> int:
        return int(self.recipe in (Recipe.FOUR, Recipe.SVDQ))

    @property
    def backend(self) -> str:
        return (
            "torch_bf16"
            if self.recipe == Recipe.BF16
            else "flashinfer_cutlass_sm120_native_quant"
        )

    @classmethod
    def create(cls, recipe: Recipe, n: int, k: int) -> "DenseLayer":
        """Prepare static weights outside every timed region."""
        torch.manual_seed(1234)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01
        match recipe:
            case Recipe.BF16:
                return cls(recipe, weight, None, None, None, None, None)
            case Recipe.NVFP4 | Recipe.FOUR | Recipe.SVDQ:
                load_quantizer()
                quant_range = 2688.0 if recipe == Recipe.NVFP4 else 1536.0
                dequant = weight.float().abs().amax().clamp_min(1e-12) / quant_range
                codes, scales = torch.ops.trtllm.fp4_quantize_ex(
                    weight,
                    dequant.reciprocal().reshape(1),
                    16,
                    False,
                    True,
                    1,
                    int(recipe != Recipe.NVFP4),
                )
                us = (
                    torch.randn(n, 64, device="cuda", dtype=torch.bfloat16) * 0.01
                    if recipe == Recipe.SVDQ
                    else None
                )
                vh = (
                    torch.randn(64, k, device="cuda", dtype=torch.bfloat16) * 0.01
                    if recipe == Recipe.SVDQ
                    else None
                )
                return cls(
                    recipe, weight, codes, scales.reshape(n, -1), dequant, us, vh
                )
            case unreachable:
                assert_never(unreachable)

    def pack(self, values: torch.Tensor) -> PackedActivation:
        """Use the same fused amax/quantization API for all quantized recipes."""
        assert self.weight_dequant is not None
        codes, scales, amax_scale = torch.ops.trtllm.fp4_quantize_fused(
            values,
            16,
            False,
            True,
            self.scale_rule,
            self.quant_range,
            1e-12,
            0,
            0,
        )
        scales = scales.reshape((values.shape[0] + 127) // 128 * 128, -1)
        alpha = (amax_scale[0] / self.quant_range) * self.weight_dequant
        return PackedActivation(codes, scales, alpha)

    def packed_mm(self, packed: PackedActivation) -> torch.Tensor:
        """Require native SM120 FP4 GEMM; do not select backend='auto'."""
        assert self.weight_codes is not None and self.weight_scales is not None
        return flashinfer.mm_fp4(
            packed.codes,
            self.weight_codes.T,
            packed.scales,
            self.weight_scales.T,
            packed.alpha,
            torch.bfloat16,
            backend="cutlass",
        )

    def correction(self, values: torch.Tensor) -> torch.Tensor:
        """Preserve the existing dense rank-64 two-GEMM correction contract."""
        assert self.vh is not None and self.us is not None
        return (values @ self.vh.T) @ self.us.T

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        match self.recipe:
            case Recipe.BF16:
                return torch.nn.functional.linear(values, self.weight)
            case Recipe.NVFP4 | Recipe.FOUR:
                return self.packed_mm(self.pack(values))
            case Recipe.SVDQ:
                return self.packed_mm(self.pack(values)) + self.correction(values)
            case unreachable:
                assert_never(unreachable)
