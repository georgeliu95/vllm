"""Select observed dimensions and reconstruct occupancy-preserving test routes."""

from collections.abc import Mapping
from heapq import heapify, heappop, heappush


def representative_m(counts: Mapping[int, int]) -> tuple[int, ...]:
    """Keep range ends, weighted median, and observed neighbors of dispatch edges."""
    ordered = sorted(counts)
    total = sum(counts.values())
    cumulative = 0
    selected = {ordered[0], ordered[-1]}
    for m in ordered:
        cumulative += counts[m]
        if cumulative * 2 >= total:
            selected.add(m)
            break
    for target in (16, 127, 128, 129, 512, 1024, 4096):
        selected.add(min(ordered, key=lambda m: (abs(m - target), m)))
    return tuple(sorted(selected))


def routes_from_counts(
    counts: tuple[int, ...], topk: int
) -> tuple[tuple[int, ...], ...]:
    """Construct unique top-k routes with captured degrees, using Havel-Hakimi.

    Token/expert identity correlations and router weights are synthetic; the
    per-expert counts, including idle experts and imbalance, remain exact.
    """
    tokens, remainder = divmod(sum(counts), topk)
    assert remainder == 0 and max(counts) <= tokens
    heap = [(-count, expert) for expert, count in enumerate(counts) if count]
    heapify(heap)
    rows: list[tuple[int, ...]] = []
    for _ in range(tokens):
        picked = [heappop(heap) for _ in range(topk)]
        rows.append(tuple(expert for _, expert in picked))
        for negative_count, expert in picked:
            if negative_count < -1:
                heappush(heap, (negative_count + 1, expert))
    assert not heap
    return tuple(rows)
