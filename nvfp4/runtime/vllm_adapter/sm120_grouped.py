"""Adapt masked NVFP4 buffers to the native SM120 groupwise GEMM ABI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Final, Literal, Mapping, assert_never

import torch


@dataclass(frozen=True, slots=True)
class GroupedLayoutError(ValueError):
    field: Literal["scales", "codes", "counts", "output", "dtype", "alpha"]
    shape: tuple[int, ...]

    def __str__(self) -> str:
        return f"Unsupported grouped NVFP4 {self.field}: {self.shape}"


class ScaleLayout(Enum):
    EXPERT_BLOCKED = 3
    LOGICAL_GROUPED = 6


SCALE_LAYOUTS: Final[Mapping[int, ScaleLayout]] = MappingProxyType(
    {3: ScaleLayout.EXPERT_BLOCKED, 6: ScaleLayout.LOGICAL_GROUPED}
)


def linearize_scales(scales: torch.Tensor) -> torch.Tensor:
    """Undo the 128-by-4 scale swizzle without changing any FP8 bits."""
    if scales.element_size() != 1:
        raise GroupedLayoutError("scales", tuple(scales.shape))
    try:
        layout = SCALE_LAYOUTS[scales.ndim]
    except KeyError as error:
        raise GroupedLayoutError("scales", tuple(scales.shape)) from error
    bits = scales.contiguous().view(torch.uint8)
    match layout:
        case ScaleLayout.EXPERT_BLOCKED:
            experts, rows, columns = bits.shape
            if rows % 128 or columns % 4:
                raise GroupedLayoutError("scales", tuple(scales.shape))
            return bits.reshape(experts, rows // 128, columns // 4, 32, 4, 4).permute(
                0, 1, 4, 3, 2, 5
            ).reshape(experts, rows, columns)
        case ScaleLayout.LOGICAL_GROUPED:
            if bits.shape[0] != 32 or bits.shape[1] != 4 or bits.shape[3] != 4:
                raise GroupedLayoutError("scales", tuple(scales.shape))
            experts = bits.shape[5]
            rows = bits.shape[2] * 128
            columns = bits.shape[4] * 4
            return bits.permute(5, 2, 1, 0, 4, 3).reshape(experts, rows, columns)
        case unreachable:
            assert_never(unreachable)


@dataclass(frozen=True, slots=True)
class GroupwiseOperands:
    codes: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor
    weight_scales: torch.Tensor
    indptr: torch.Tensor
    valid: torch.Tensor
    active: torch.Tensor


def block_scales(scales: torch.Tensor) -> torch.Tensor:
    """Encode CUTLASS's 128-by-4 block-scale storage without changing bytes."""
    experts, rows, columns = scales.shape
    if rows % 128 or columns % 4 or scales.element_size() != 1:
        raise GroupedLayoutError("scales", tuple(scales.shape))
    return scales.reshape(experts, rows // 128, 4, 32, columns // 4, 4).permute(
        0, 1, 4, 3, 2, 5
    ).contiguous().reshape(experts, rows, columns)


def prepare_groupwise(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    counts: torch.Tensor,
) -> GroupwiseOperands:
    """Compact valid rows, pad segments to four, and preserve masked storage."""
    activation_codes, activation_scales = lhs
    weight_codes, weight_scales = rhs
    if activation_codes.ndim != 3 or weight_codes.ndim != 3:
        raise GroupedLayoutError("codes", tuple(activation_codes.shape))
    capacity, packed_inner, experts = activation_codes.shape
    output_width, weight_inner, weight_experts = weight_codes.shape
    if (
        capacity % 4
        or packed_inner % 64
        or output_width % 32
        or weight_inner != packed_inner
        or weight_experts != experts
        or activation_codes.element_size() != 1
        or weight_codes.element_size() != 1
    ):
        raise GroupedLayoutError("codes", tuple(activation_codes.shape))
    if counts.shape != (experts,) or bool(((counts < 1) | (counts > capacity)).any().item()):
        raise GroupedLayoutError("counts", tuple(counts.shape))
    linear_activation = linearize_scales(activation_scales)
    linear_weights = linearize_scales(weight_scales)
    block_columns = packed_inner // 8
    if (
        linear_activation.shape[0] != experts
        or linear_activation.shape[1] < capacity
        or linear_activation.shape[2] < block_columns
        or linear_weights.shape[0] != experts
        or linear_weights.shape[1] < output_width
        or linear_weights.shape[2] < block_columns
    ):
        raise GroupedLayoutError("scales", tuple(activation_scales.shape))
    padded_counts = ((counts.to(torch.int32) + 3) // 4) * 4
    positions = torch.arange(capacity, device=counts.device)[None, :]
    padded = positions < padded_counts[:, None]
    active = positions < counts[:, None]
    valid = active[padded]
    codes = activation_codes.permute(2, 0, 1).contiguous().view(torch.uint8)[padded]
    codes = torch.where(valid[:, None], codes, torch.zeros_like(codes))
    indptr = torch.cat(
        (padded_counts.new_zeros(1), padded_counts.cumsum(0, dtype=torch.int32))
    )
    expert_indices = torch.arange(experts, device=counts.device)
    scale_offsets = ((indptr[:-1] + expert_indices * 127) // 128) * 128
    scale_rows = int((scale_offsets + ((padded_counts + 127) // 128) * 128).amax().item())
    scale_values = linear_activation[:, :capacity, :block_columns][padded]
    scale_values = torch.where(valid[:, None], scale_values, torch.zeros_like(scale_values))
    linear_scales = scale_values.new_zeros(scale_rows, block_columns)
    scale_indices = (scale_offsets[:, None] + positions.expand(experts, -1))[padded]
    linear_scales[scale_indices] = scale_values
    scales = block_scales(linear_scales.unsqueeze(0)).reshape(block_columns, scale_rows).transpose(0, 1)
    return GroupwiseOperands(
        codes=codes,
        weights=weight_codes.permute(2, 0, 1).contiguous().view(torch.uint8),
        scales=scales,
        weight_scales=block_scales(linear_weights[:, :, :block_columns]),
        indptr=indptr,
        valid=valid,
        active=active,
    )


def dispatch_grouped_gemm_nt_masked(
    lhs: tuple[torch.Tensor, torch.Tensor],
    rhs: tuple[torch.Tensor, torch.Tensor],
    out: torch.Tensor,
    masked_m: torch.Tensor,
    *,
    ab_dtype: str,
    sf_dtype: str,
    c_dtype: str,
    sf_vec_size: int,
    alpha: torch.Tensor,
    alpha_dtype: str,
) -> None:
    """Use SM120's native NVFP4 GEMM; leave other architectures unchanged."""
    major = torch.cuda.get_device_capability(lhs[0].device)[0]
    if major < 12 or major >= 13:
        from vllm.utils.flashinfer import flashinfer_cutedsl_grouped_gemm_nt_masked

        flashinfer_cutedsl_grouped_gemm_nt_masked(
            lhs, rhs, out, masked_m, ab_dtype=ab_dtype, sf_dtype=sf_dtype,
            c_dtype=c_dtype, sf_vec_size=sf_vec_size, alpha=alpha, alpha_dtype=alpha_dtype,
        )
        return
    if (
        ab_dtype != "float4_e2m1fn"
        or sf_dtype != "float8_e4m3fn"
        or sf_vec_size != 16
        or alpha_dtype != "float32"
        or out.dtype not in (torch.bfloat16, torch.float16)
    ):
        raise GroupedLayoutError("dtype", tuple(out.shape))
    operands = prepare_groupwise(lhs, rhs, masked_m)
    if out.shape != (lhs[0].shape[0], rhs[0].shape[0], lhs[0].shape[2]):
        raise GroupedLayoutError("output", tuple(out.shape))
    if alpha.numel() != lhs[0].shape[2] or alpha.dtype != torch.float32:
        raise GroupedLayoutError("alpha", tuple(alpha.shape))
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise

    result = group_gemm_nvfp4_nt_groupwise(
        operands.codes, operands.weights, operands.scales, operands.weight_scales,
        operands.indptr, alpha=alpha.reshape(-1).contiguous(), out_dtype=out.dtype,
    )
    out.permute(2, 0, 1)[operands.active] = result[operands.valid]
