# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import json
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = REPO_ROOT / "evaluation" / "analysis"
sys.path.insert(0, str(ANALYSIS_ROOT))

from reevaluate_all_mbpp import (
    EvaluationError,
    find_samples,
    load_samples,
    normalize_completion,
    parse_variant_specs,
)


def test_find_samples_accepts_sixteen_shards(tmp_path: Path) -> None:
    raw = tmp_path / "result" / "raw"
    for index in range(16):
        shard = raw / f"mbpp-shard-{index}"
        shard.mkdir(parents=True)
        (shard / f"samples_mbpp_{index}.jsonl").touch()

    paths = find_samples(tmp_path, "result")

    assert len(paths) == 16


def test_find_samples_rejects_missing_samples(tmp_path: Path) -> None:
    (tmp_path / "result" / "raw").mkdir(parents=True)

    with pytest.raises(EvaluationError, match="at least one"):
        find_samples(tmp_path, "result")


def test_normalize_completion_prefers_python_fence() -> None:
    response = "Explanation\n```python\ndef answer():\n    return 42\n```\nDone"

    assert normalize_completion(response) == "def answer():\n    return 42"


def test_normalize_completion_accepts_generic_fence() -> None:
    response = "```\ndef answer():\n    return 42\n```"

    assert normalize_completion(response) == "def answer():\n    return 42"


def test_normalize_completion_preserves_direct_code() -> None:
    response = "\n  def answer():\n      return 42\n"

    assert normalize_completion(response) == "def answer():\n      return 42"


def test_load_samples_merges_and_orders_four_shards(tmp_path: Path) -> None:
    paths = tuple(tmp_path / f"samples-{index}.jsonl" for index in range(4))
    for shard, path in enumerate(paths):
        records = ({"doc_id": index} for index in range(shard, 500, 4))
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    samples = load_samples(paths)

    assert tuple(sample["doc_id"] for sample in samples) == tuple(range(500))


def test_load_samples_rejects_overlapping_shards(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    records = ({"doc_id": index % 499} for index in range(500))
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    with pytest.raises(EvaluationError, match="disjoint"):
        load_samples((path,))


def test_parse_variant_specs_accepts_explicit_result_directory() -> None:
    assert parse_variant_specs(["Qwen3.8-27B|4o6=full/qwen38-4o6"]) == {
        "Qwen3.8-27B|4o6": "full/qwen38-4o6"
    }


@pytest.mark.parametrize(
    "spec",
    ("Qwen3.8-27B|4o6", "Qwen3.8-27B=output", "|4o6="),
)
def test_parse_variant_specs_rejects_invalid_syntax(spec: str) -> None:
    with pytest.raises(EvaluationError, match="MODEL\\|PRECISION=RESULT_DIR"):
        parse_variant_specs([spec])
