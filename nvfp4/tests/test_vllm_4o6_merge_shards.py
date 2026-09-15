# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "evaluation" / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from merge_sharded_lm_eval import (
    MergeError,
    _completed_roots,
    _load_sample_ids,
    merge_weighted_metric,
    validate_doc_ids,
)


def test_completed_roots_ignores_empty_failed_labels(tmp_path: Path) -> None:
    failed = tmp_path / "raw" / "mmlu-shard-0"
    repaired = tmp_path / "raw" / "mmlu-repair-0"
    failed.mkdir(parents=True)
    repaired.mkdir(parents=True)
    (repaired / "results_test.json").write_text("{}", encoding="utf-8")

    assert _completed_roots(tmp_path, "mmlu") == (repaired,)


def test_merge_weighted_metric_uses_effective_sample_counts() -> None:
    shards = ((0.25, 100), (0.75, 300))

    assert merge_weighted_metric(shards) == (0.625, 400)


def test_validate_doc_ids_accepts_disjoint_complete_shards() -> None:
    validate_doc_ids(((0, 4), (1, 5), (2, 6), (3, 7)), expected=8)


def test_validate_doc_ids_rejects_duplicates() -> None:
    with pytest.raises(MergeError, match="duplicate"):
        validate_doc_ids(((0, 1), (1, 2)), expected=3)


def test_validate_doc_ids_rejects_missing_samples() -> None:
    with pytest.raises(MergeError, match="expected doc_ids"):
        validate_doc_ids(((0, 2), (1,)), expected=4)


def test_load_sample_ids_deduplicates_lm_eval_output_filters(
    tmp_path: Path,
) -> None:
    samples = tmp_path / "samples_gsm8k_test.jsonl"
    samples.write_text(
        '{"doc_id": 3, "filter": "strict"}\n'
        '{"doc_id": 3, "filter": "flexible"}\n'
        '{"doc_id": 1, "filter": "strict"}\n',
        encoding="utf-8",
    )

    assert _load_sample_ids(tmp_path, "gsm8k") == (1, 3)
