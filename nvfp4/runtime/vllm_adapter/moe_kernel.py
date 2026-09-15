# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer masked MoE execution with 4o6 and SVDQ corrections."""

from __future__ import annotations

import torch

from .quantization import (
    dynamic_grouped_global_scale,
    quantize_grouped_4o6,
    swiglu,
)

_PADDED_BMM_MAX_TOKENS = 8
_MAX_EXPERTS_PER_GROUPED_GEMM = 64


def _expert_ranges(experts: int) -> tuple[tuple[int, int], ...]:
    """Partition experts into FlashInfer grouped-GEMM-safe chunks."""
    return tuple(
        (start, min(start + _MAX_EXPERTS_PER_GROUPED_GEMM, experts))
        for start in range(0, experts, _MAX_EXPERTS_PER_GROUPED_GEMM)
    )


def _grouped_gemm_mask(masked_m: torch.Tensor) -> torch.Tensor:
    """Give inactive experts one deterministic zero row for kernel safety."""
    return masked_m.clamp_min(1)


def _initialize_grouped_gemm_dummy_rows(
    values: torch.Tensor, masked_m: torch.Tensor
) -> torch.Tensor:
    """Materialize rows promoted from zero to one by the kernel mask."""
    from .fused_scale import initialize_dummy_rows
    return initialize_dummy_rows(values, masked_m)


def grouped_lowrank(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor
) -> torch.Tensor:
    """Evaluate per-expert ``(X @ Vh.T) @ US.T``."""
    projected = torch.bmm(values, vh.transpose(1, 2).to(values.dtype))
    return torch.bmm(projected, us.transpose(1, 2).to(values.dtype))


def grouped_gate_up_lowrank(
    values: torch.Tensor, us: torch.Tensor, vh: torch.Tensor
) -> torch.Tensor:
    """Evaluate independently factorized gate and up expert projections."""
    if us.shape[1] % 2 != 0 or vh.shape[1] % 2 != 0:
        raise RuntimeError("Gate/up SVDQ factors must have even concatenated axes.")
    gate_us, up_us = us.chunk(2, dim=1)
    projected = torch.bmm(values, vh.transpose(1, 2).to(values.dtype))
    gate_projected, up_projected = projected.chunk(2, dim=2)
    return torch.cat(
        (
            torch.bmm(
                gate_projected.contiguous(),
                gate_us.transpose(1, 2).to(values.dtype),
            ),
            torch.bmm(
                up_projected.contiguous(),
                up_us.transpose(1, 2).to(values.dtype),
            ),
        ),
        dim=-1,
    )


def _token_buckets(max_tokens: int) -> list[tuple[int, int]]:
    """Return power-of-two token-count buckets up to ``max_tokens``."""
    buckets: list[tuple[int, int]] = []
    lower = 0
    upper = min(_PADDED_BMM_MAX_TOKENS, max_tokens)
    while lower < max_tokens:
        buckets.append((lower, upper))
        lower = upper
        upper = min(upper * 2, max_tokens)
    return buckets


def grouped_lowrank_masked(
    values: torch.Tensor,
    us: torch.Tensor,
    vh: torch.Tensor,
    masked_m: torch.Tensor,
) -> torch.Tensor:
    """Evaluate corrections in token buckets with at most about 2x padding."""
    result = values.new_zeros((*values.shape[:2], us.shape[1]))
    for lower, upper in _token_buckets(values.shape[1]):
        selected = (masked_m > lower) & (masked_m <= upper)
        bucket_values = values[selected, :upper]
        correction = grouped_lowrank(bucket_values, us[selected], vh[selected])
        valid = (
            torch.arange(upper, device=values.device)[None, :]
            < masked_m[selected, None]
        )
        result[selected, :upper] = correction * valid[:, :, None]
    return result


def grouped_gate_up_lowrank_masked(
    values: torch.Tensor,
    us: torch.Tensor,
    vh: torch.Tensor,
    masked_m: torch.Tensor,
) -> torch.Tensor:
    """Evaluate fused gate/up corrections for only valid expert rows."""
    if us.shape[1] % 2 != 0 or vh.shape[1] % 2 != 0:
        raise RuntimeError("Gate/up SVDQ factors must have even concatenated axes.")
    result = values.new_zeros((*values.shape[:2], us.shape[1]))
    for lower, upper in _token_buckets(values.shape[1]):
        selected = (masked_m > lower) & (masked_m <= upper)
        bucket_values = values[selected, :upper]
        correction = grouped_gate_up_lowrank(
            bucket_values, us[selected], vh[selected]
        )
        valid = (
            torch.arange(upper, device=values.device)[None, :]
            < masked_m[selected, None]
        )
        result[selected, :upper] = correction * valid[:, :, None]
    return result


def run_4o6_svdq_moe(
    *,
    hidden_states: torch.Tensor,
    activation_quant_range: float,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_weight_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_weight_global_scale: torch.Tensor,
    masked_m: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    w13_us: torch.Tensor | None,
    w13_vh: torch.Tensor | None,
    w2_us: torch.Tensor | None,
    w2_vh: torch.Tensor | None,
) -> None:
    """Run grouped GEMMs in expert chunks and add residual corrections."""
    experts = hidden_states.shape[0]
    for start, end in _expert_ranges(experts):
        _run_4o6_svdq_moe_chunk(
            hidden_states=hidden_states[start:end],
            activation_quant_range=activation_quant_range,
            w1=w1[start:end],
            w1_blockscale=w1_blockscale[start:end],
            w1_weight_global_scale=w1_weight_global_scale[start:end],
            w2=w2[start:end],
            w2_blockscale=w2_blockscale[start:end],
            w2_weight_global_scale=w2_weight_global_scale[start:end],
            masked_m=masked_m[start:end],
            workspace=workspace[start:end],
            out=out[start:end],
            w13_us=None if w13_us is None else w13_us[start:end],
            w13_vh=None if w13_vh is None else w13_vh[start:end],
            w2_us=None if w2_us is None else w2_us[start:end],
            w2_vh=None if w2_vh is None else w2_vh[start:end],
        )


def _run_4o6_svdq_moe_chunk(
    *,
    hidden_states: torch.Tensor,
    activation_quant_range: float,
    w1: torch.Tensor,
    w1_blockscale: torch.Tensor,
    w1_weight_global_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_weight_global_scale: torch.Tensor,
    masked_m: torch.Tensor,
    workspace: torch.Tensor,
    out: torch.Tensor,
    w13_us: torch.Tensor | None,
    w13_vh: torch.Tensor | None,
    w2_us: torch.Tensor | None,
    w2_vh: torch.Tensor | None,
) -> None:
    """Run one expert chunk through both NVFP4 GEMMs and corrections."""
    from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutedsl_batched_moe import (
        get_cute_dtype,
    )
    from vllm.utils.flashinfer import (
        flashinfer_cutedsl_grouped_gemm_nt_masked,
        silu_and_mul_scaled_nvfp4_experts_quantize,
    )

    experts, _, hidden = hidden_states.shape
    intermediate = w2.shape[-1] * 2
    grouped_gemm_mask = _initialize_grouped_gemm_dummy_rows(
        hidden_states, masked_m
    )
    input_global_scale, input_dequant_scale = dynamic_grouped_global_scale(
        hidden_states,
        masked_m,
        activation_quant_range,
    )
    w1_alpha = input_dequant_scale * w1_weight_global_scale
    aq, aq_sf = quantize_grouped_4o6(
        hidden_states, grouped_gemm_mask, input_global_scale
    )
    workspace_logical = workspace.permute(1, 2, 0)
    flashinfer_cutedsl_grouped_gemm_nt_masked(
        (aq, aq_sf),
        (w1.permute(1, 2, 0), w1_blockscale),
        workspace_logical,
        grouped_gemm_mask,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=get_cute_dtype(hidden_states),
        sf_vec_size=16,
        alpha=w1_alpha.view(1, 1, experts),
        alpha_dtype=get_cute_dtype(w1_alpha),
    )

    gate_up = workspace_logical.permute(2, 0, 1)
    if w13_us is not None and w13_vh is not None:
        from .fused_lowrank import add_correction
        add_correction(hidden_states, w13_us, w13_vh, masked_m, gate_up, gated=True)
    from .fused_scale import grouped_scale
    a2_global_scale, a2_dequant_scale, activated = grouped_scale(
        gate_up, masked_m, activation_quant_range, swiglu=True,
    )
    w2_alpha = a2_dequant_scale * w2_weight_global_scale
    diq, diq_sf = silu_and_mul_scaled_nvfp4_experts_quantize(
        gate_up, grouped_gemm_mask, a2_global_scale
    )

    out_logical = out.permute(1, 2, 0)
    flashinfer_cutedsl_grouped_gemm_nt_masked(
        (diq, diq_sf),
        (w2.permute(1, 2, 0), w2_blockscale),
        out_logical,
        grouped_gemm_mask,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype=get_cute_dtype(hidden_states),
        sf_vec_size=16,
        alpha=w2_alpha.view(1, 1, experts),
        alpha_dtype=get_cute_dtype(w2_alpha),
    )
    if w2_us is not None and w2_vh is not None:
        from .fused_lowrank import add_correction
        add_correction(activated, w2_us, w2_vh, masked_m, out, gated=False)


    if hidden != out.shape[-1] or gate_up.shape[-1] != 2 * intermediate:
        raise RuntimeError("Unexpected FlashInfer MoE workspace shape.")


__all__ = [
    "_initialize_grouped_gemm_dummy_rows",
    "_grouped_gemm_mask",
    "grouped_gate_up_lowrank",
    "grouped_gate_up_lowrank_masked",
    "grouped_lowrank",
    "grouped_lowrank_masked",
    "run_4o6_svdq_moe",
]
