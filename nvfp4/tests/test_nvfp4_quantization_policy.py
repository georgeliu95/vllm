# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from converter.nvfp4_quantization_policy import (
    ADAPTIVE_FP4_QUANT_RANGE,
    STANDARD_NVFP4_QUANT_RANGE,
    flashinfer_quantization_policy,
    weight_scale_rule,
)


def test_standard_activation_mode_selects_standard_nvfp4_weight_quantization() -> None:
    # Given a direct BF16 conversion requesting standard NVFP4 activations.
    activation_mode = "standard"

    # When the converter resolves its weight quantization rule and backend policy.
    rule = weight_scale_rule(activation_mode, "mse")
    policy = flashinfer_quantization_policy(rule)

    # Then weights use the standard six-range NVFP4 path, not adaptive 4o6.
    assert rule == 0
    assert policy.quant_range == STANDARD_NVFP4_QUANT_RANGE
    assert policy.four_over_six_enabled is False
    assert policy.error_mode is None


def test_4o6_weight_policy_reserves_e4m3_headroom_for_candidate_four() -> None:
    # Given 4o6 weight conversion with its MSE candidate-selection rule.
    activation_mode = "4o6"

    # When the converter resolves the FlashInfer quantization policy.
    rule = weight_scale_rule(activation_mode, "mse")
    policy = flashinfer_quantization_policy(rule)

    # Then the shared global scale leaves room for candidate 4's 1.5x block scale.
    assert rule == 1
    assert policy.quant_range == ADAPTIVE_FP4_QUANT_RANGE
    assert ADAPTIVE_FP4_QUANT_RANGE == 256.0 * 6.0
    assert ADAPTIVE_FP4_QUANT_RANGE != STANDARD_NVFP4_QUANT_RANGE
    assert policy.four_over_six_enabled is True
    assert policy.error_mode == "MSE"
