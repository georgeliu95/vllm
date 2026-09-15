"""Candidate local expert routing, pending numerical verification."""

from __future__ import annotations

import torch


def routing_rows(ids: torch.Tensor, experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign rows in token order for unique top-k expert IDs on a local rank."""
    flat = ids.reshape(-1).long()
    order = flat.argsort(stable=True)
    sorted_ids = flat[order]
    counts = torch.zeros(experts, device=ids.device, dtype=torch.int32)
    counts.scatter_add_(0, flat, torch.ones_like(flat, dtype=torch.int32))
    offsets = counts.cumsum(0) - counts
    sorted_rows = torch.arange(flat.numel(), device=ids.device) - offsets[sorted_ids]
    rows = torch.empty_like(flat)
    rows.scatter_(0, order, sorted_rows)
    return rows.view_as(ids), counts


def prepare_local(
    values: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    experts: int,
    capacity: int,
    weighted_input: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dispatch unique local top-k routes without per-expert host synchronization."""
    if values.is_cuda and values.is_contiguous() and ids.is_contiguous() and weights.is_contiguous() and ids.numel() <= 1024:
        from .fused_routing import prepare_local as fused_prepare
        return fused_prepare(values, weights, ids, experts, capacity, weighted_input)
    if weighted_input:
        values.mul_(weights.to(values.dtype))
    rows, counts = routing_rows(ids, experts)
    result = values.new_zeros(experts, capacity, values.shape[-1])
    token_ids = torch.arange(values.shape[0], device=values.device)
    result[ids.long(), rows] = values[token_ids[:, None].expand_as(ids)]
    return result, counts


def finalize_local(
    output: torch.Tensor,
    expert_output: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    weighted_input: bool,
) -> None:
    """Preserve reference in-place multiplication and ascending-expert sums."""
    if output.is_cuda and output.is_contiguous() and expert_output.is_contiguous() and ids.is_contiguous() and weights.is_contiguous() and ids.numel() <= 1024:
        from .fused_routing import finalize_local as fused_finalize
        fused_finalize(output, expert_output, weights, ids, weighted_input)
        return
    rows, _ = routing_rows(ids, expert_output.shape[0])
    order = ids.argsort(dim=1)
    sorted_ids = ids.gather(1, order).long()
    sorted_rows = rows.gather(1, order)
    sorted_weights = weights.gather(1, order)
    output.zero_()
    for route in range(ids.shape[1]):
        expert = sorted_ids[:, route]
        row = sorted_rows[:, route]
        values = expert_output[expert, row]
        if not weighted_input:
            values.mul_(sorted_weights[:, route, None])
            expert_output[expert, row] = values
        output.add_(values)
