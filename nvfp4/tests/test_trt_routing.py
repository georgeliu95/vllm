"""Validate original TRT routing across multi-tile, skewed and empty shards."""

import pytest
import torch

from trt_routing import compact_sort


@pytest.mark.parametrize("tokens", [1, 128, 129, 1024, 16384])
@pytest.mark.parametrize("start", [0, 64])
def test_compact_routing_covers_each_local_route_once(tokens: int, start: int) -> None:
    """Every selected route occurs once, inside the correct expert's live tiles."""
    ids = (
        torch.arange(tokens, device="cuda")[:, None] + torch.arange(8, device="cuda")
    ) % 128
    # Keep unique top-k but make the first half maximally skewed.
    ids[: tokens // 2] = torch.arange(8, device="cuda")
    ids = ids.to(torch.int32).contiguous()
    weights = torch.full((tokens, 8), 0.125, device="cuda")
    groups, limits, perm, live = compact_sort(ids, weights, 128, start, 64)
    flat = ids.flatten().cpu()
    groups, limits, perm = groups.cpu(), limits.cpu(), perm.cpu()
    seen = []
    count = int(live.item())
    for tile in range(count):
        end = min(int(limits[tile]), (tile + 1) * 128)
        assert tile * 128 < end
        routes = perm[tile * 128 : end].long()
        assert (flat[routes] == groups[tile] + start).all()
        seen.extend(routes.tolist())
    expected = torch.where((flat >= start) & (flat < start + 64))[0].tolist()
    assert sorted(seen) == expected
    counts = torch.bincount(flat.long(), minlength=128)[start : start + 64]
    assert count == int(((counts + 127) // 128).sum())
    assert perm.numel() <= tokens * 8 + 64 * 128


@pytest.mark.parametrize("tokens", [131, 16384])
def test_empty_chunk_and_graph_replay_changes_routes(tokens: int) -> None:
    """Device-side routing must refresh live tiles and permutation on each replay."""
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    weights = torch.full((tokens, 8), 0.125, device="cuda")
    compact_sort(ids, weights, 128, 64, 64)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        groups, limits, perm, live = compact_sort(ids, weights, 128, 64, 64)
    graph.replay()
    assert live.item() == 0
    ids.add_(64)
    graph.replay()
    tiles_per_expert = (tokens + 127) // 128
    tiles = 8 * tiles_per_expert
    assert live.item() == tiles
    assert groups[:tiles].min() == 0 and groups[:tiles].max() == 7
    assert limits[:tiles].max() == 7 * tiles_per_expert * 128 + tokens
    assert perm[0].item() % 8 == 0
