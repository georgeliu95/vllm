# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate and merge the four-way lm-eval accuracy shards."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Final

EXPECTED_SAMPLES: Final = {"mmlu": 14_042, "gsm8k": 1_319, "mbpp": 500}


class MergeError(RuntimeError):
    """Raised when persisted shards are incomplete or overlap."""


def merge_weighted_metric(shards: Iterable[tuple[float, int]]) -> tuple[float, int]:
    """Merge shard means using their effective sample counts."""
    values = tuple(shards)
    total = sum(count for _, count in values)
    if total <= 0:
        raise MergeError("cannot merge a metric with no effective samples")
    return sum(value * count for value, count in values) / total, total


def validate_doc_ids(shards: Sequence[Sequence[int]], expected: int) -> None:
    """Require an exact, disjoint cover of ``range(expected)``."""
    flattened = tuple(doc_id for shard in shards for doc_id in shard)
    if len(flattened) != len(set(flattened)):
        raise MergeError("duplicate doc_ids across shards")
    actual = set(flattened)
    wanted = set(range(expected))
    if actual != wanted:
        missing = sorted(wanted - actual)[:8]
        extra = sorted(actual - wanted)[:8]
        raise MergeError(
            f"expected doc_ids 0..{expected - 1}; missing={missing}, extra={extra}"
        )


def _sole_file(root: Path, pattern: str) -> Path:
    matches: tuple[Path, ...] = ()
    # Lustre may transiently return ESTALE while pathlib scans a completed
    # shard tree. Path.glob suppresses that entry-level OSError, which can
    # otherwise make a complete result look randomly incomplete.
    for _ in range(8):
        matches = tuple(root.glob(pattern))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            break
    raise MergeError(f"expected one {pattern!r} under {root}, found {len(matches)}")


def _load_result(shard_root: Path) -> dict[str, object]:
    path = _sole_file(shard_root, "**/results_*.json")
    with path.open(encoding="utf-8") as source:
        result = json.load(source)
    if not isinstance(result, dict):
        raise MergeError(f"result must be an object: {path}")
    return result


def _completed_roots(output_root: Path, prefix: str) -> tuple[Path, ...]:
    """Return labeled raw directories containing exactly one result file."""
    raw_root = output_root / "raw"
    roots: list[Path] = []
    for root in sorted(raw_root.glob(f"{prefix}*")):
        if root.is_dir() and len(tuple(root.glob("**/results_*.json"))) == 1:
            roots.append(root)
    if not roots:
        raise MergeError(f"no completed {prefix!r} result directories under {raw_root}")
    return tuple(roots)


def _load_sample_ids(shard_root: Path, task: str) -> tuple[int, ...]:
    path = _sole_file(shard_root, f"**/samples_{task}_*.jsonl")
    with path.open(encoding="utf-8") as source:
        # lm-eval writes one GSM8K sample record per output filter (strict and
        # flexible), so a document can legitimately appear twice in JSONL.
        return tuple(sorted({int(json.loads(line)["doc_id"]) for line in source}))


def _effective_samples(result: dict[str, object], task: str) -> int:
    n_samples = result.get("n-samples")
    if not isinstance(n_samples, dict) or task not in n_samples:
        raise MergeError(f"missing n-samples entry for {task}")
    entry = n_samples[task]
    if not isinstance(entry, dict) or "effective" not in entry:
        raise MergeError(f"invalid n-samples entry for {task}")
    return int(entry["effective"])


def _metric(result: dict[str, object], task: str, metric: str) -> float:
    results = result.get("results")
    if not isinstance(results, dict) or task not in results:
        raise MergeError(f"missing result entry for {task}")
    entry = results[task]
    if not isinstance(entry, dict) or metric not in entry:
        raise MergeError(f"missing metric {metric} for {task}")
    return float(entry[metric])


def merge_generation_task(output_root: Path, task: str) -> dict[str, float | int]:
    """Merge GSM8K or MBPP shards and validate their selected samples."""
    roots = _completed_roots(output_root, task)
    results = tuple(_load_result(root) for root in roots)
    sample_ids = tuple(_load_sample_ids(root, task) for root in roots)
    validate_doc_ids(sample_ids, EXPECTED_SAMPLES[task])
    metric_names = (
        (
            ("strict_match", "exact_match,strict-match"),
            ("flexible_extract", "exact_match,flexible-extract"),
        )
        if task == "gsm8k"
        else (("pass_at_1", "pass_at_1,none"),)
    )
    merged: dict[str, float | int] = {}
    sample_count = 0
    for output_name, metric_name in metric_names:
        value, sample_count = merge_weighted_metric(
            (
                _metric(result, task, metric_name),
                len(ids),
            )
            for result, ids in zip(results, sample_ids, strict=True)
        )
        merged[output_name] = value
    if sample_count != EXPECTED_SAMPLES[task]:
        raise MergeError(
            f"{task}: expected {EXPECTED_SAMPLES[task]} effective samples, "
            f"found {sample_count}"
        )
    merged["samples"] = sample_count
    return merged


def merge_mmlu(output_root: Path) -> dict[str, float | int]:
    """Merge 57 disjoint MMLU subtasks using effective sample counts."""
    weighted: list[tuple[float, int]] = []
    task_names: set[str] = set()
    sample_total = 0
    for root in _completed_roots(output_root, "mmlu"):
        result = _load_result(root)
        n_samples = result.get("n-samples")
        if not isinstance(n_samples, dict):
            raise MergeError(f"invalid n-samples in MMLU shard {root.name}")
        for task in n_samples:
            if not str(task).startswith("mmlu_"):
                continue
            task_name = str(task)
            if task_name in task_names:
                raise MergeError(f"duplicate MMLU subtask: {task_name}")
            task_names.add(task_name)
            count = _effective_samples(result, task_name)
            ids = _load_sample_ids(root, task_name)
            validate_doc_ids((ids,), count)
            weighted.append((_metric(result, task_name, "acc,none"), count))
            sample_total += count
    if len(task_names) != 57:
        raise MergeError(f"expected 57 MMLU subtasks, found {len(task_names)}")
    score, merged_total = merge_weighted_metric(weighted)
    if merged_total != sample_total or sample_total != EXPECTED_SAMPLES["mmlu"]:
        raise MergeError(
            f"MMLU: expected {EXPECTED_SAMPLES['mmlu']} samples, found {sample_total}"
        )
    return {"acc": score, "samples": sample_total, "subtasks": len(task_names)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("result_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve(strict=True)
    summary = {
        "mmlu": merge_mmlu(output_root),
        "gsm8k": merge_generation_task(output_root, "gsm8k"),
        "mbpp": merge_generation_task(output_root, "mbpp"),
    }
    args.result_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
