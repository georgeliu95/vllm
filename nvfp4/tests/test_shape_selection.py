"""Representative coverage must retain real observations and dispatch edges."""

from shape_cases import selection
import json
from pathlib import Path


def test_selection_keeps_observed_edges_and_weighted_middle() -> None:
    """Given sparse observations, choose real rows including both sides of 128."""
    counts = {1: 1, 16: 2, 61: 100, 127: 1, 128: 1, 129: 1, 978: 3, 16384: 90}
    chosen = selection.representative_m(counts)
    assert {1, 61, 127, 128, 129, 16384} <= set(chosen)
    assert set(chosen) <= counts.keys()


def test_routes_preserve_counts_and_unique_topk() -> None:
    """Given skew and idle experts, reconstructed routes keep exact occupancy."""
    counts = (5, 4, 1, 0)
    ids = selection.routes_from_counts(counts, topk=2)
    assert len(ids) == 5
    assert all(len(set(row)) == 2 for row in ids)
    assert tuple(sum(e in row for row in ids) for e in range(4)) == counts


def test_checked_in_cases_cover_models_phases_and_real_occupancy() -> None:
    """The regression fixture must cover both models and the large-M fallback."""
    fixture = json.loads((Path(__file__).parent / "data/qwen3_shapes.json").read_text())
    dense = fixture["dense"]
    assert {c["model"] for c in dense} == {"Qwen3-8B", "Qwen3-30B-A3B"}
    assert {(c["n"], c["k"]) for c in dense} == {
        (128, 2048),
        (2048, 4096),
        (5120, 2048),
        (151936, 2048),
        (4096, 4096),
        (4096, 12288),
        (6144, 4096),
        (24576, 4096),
        (151936, 4096),
    }
    assert {c["phase"] for c in fixture["moe"]} == {"prefill", "decode", "mixed"}
    assert {c["task"] for c in fixture["moe"]} == {"mmlu", "gsm8k", "mbpp"}
    assert any(128 < c["m"] < 140 for c in fixture["moe"])
    for case in fixture["moe"]:
        counts = tuple(case["counts"])
        assert len(counts) == 128 and sum(counts) == 8 * case["m"]
        rows = selection.routes_from_counts(counts, 8)
        observed = [0] * 128
        for row in rows:
            assert len(set(row)) == 8
            for expert in row:
                observed[expert] += 1
        assert tuple(observed) == counts
