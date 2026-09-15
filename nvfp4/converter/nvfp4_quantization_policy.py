# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed policy for standard NVFP4 and adaptive 4o6 checkpoint conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal, assert_never

STANDARD_NVFP4_QUANT_RANGE: Final = 448.0 * 6.0
ADAPTIVE_FP4_QUANT_RANGE: Final = 256.0 * 6.0

ActivationMode = Literal["4o6", "standard"]
FlashInferErrorMode = Literal["MSE", "MAE"]


class QuantizationPolicyError(ValueError):
    """A requested checkpoint quantization policy is unsupported."""


@dataclass(frozen=True, slots=True)
class FlashInferQuantizationPolicy:
    """Resolved FlashInfer weight-quantization behavior."""

    quant_range: float
    four_over_six_enabled: bool
    error_mode: FlashInferErrorMode | None


def parse_adaptive_scale_rule(name: str) -> int:
    """Parse a 4o6 adaptive block-scale selection rule."""
    named_rules: Final = {"mse": 1, "mae": 2, "abs_max": 3, "absmax": 3}
    if name in named_rules:
        return named_rules[name]
    try:
        value = int(name)
    except ValueError as error:
        raise QuantizationPolicyError(
            "--scale-rule must be mse, mae, abs_max, 1, 2, or 3"
        ) from error
    if value not in (1, 2, 3):
        raise QuantizationPolicyError(
            "--scale-rule must be mse, mae, abs_max, 1, 2, or 3"
        )
    return value


def weight_scale_rule(activation_mode: ActivationMode, scale_rule: str) -> int:
    """Resolve the weight rule coupled to the requested activation format."""
    match activation_mode:
        case "standard":
            return 0
        case "4o6":
            return parse_adaptive_scale_rule(scale_rule)
        case unreachable:
            assert_never(unreachable)


def flashinfer_quantization_policy(scale_rule: int) -> FlashInferQuantizationPolicy:
    """Resolve supported standard or adaptive FlashInfer quantization settings."""
    policies: Final = {
        0: FlashInferQuantizationPolicy(
            quant_range=STANDARD_NVFP4_QUANT_RANGE,
            four_over_six_enabled=False,
            error_mode=None,
        ),
        1: FlashInferQuantizationPolicy(
            quant_range=ADAPTIVE_FP4_QUANT_RANGE,
            four_over_six_enabled=True,
            error_mode="MSE",
        ),
        2: FlashInferQuantizationPolicy(
            quant_range=ADAPTIVE_FP4_QUANT_RANGE,
            four_over_six_enabled=True,
            error_mode="MAE",
        ),
    }
    try:
        return policies[scale_rule]
    except KeyError as error:
        raise QuantizationPolicyError(
            "FlashInfer quantization supports standard NVFP4, MSE 4o6, or MAE 4o6."
        ) from error
