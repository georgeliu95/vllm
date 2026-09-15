#!/usr/bin/env python3
"""Build the final accuracy tables from validated merged rerun outputs."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Final


MODEL_METHOD_DIRS: Final[dict[str, dict[str, str | None]]] = {
    "Qwen3.5-35B-A3B": {
        "BF16-direct NVFP4": "qwen35-35b-a3b-direct",
        "4o6": "qwen35-35b-a3b-4o6",
        "4o6+SVDQ": "qwen35-35b-a3b-4o6-svdq-sharded-final",
    },
    "Qwen3.5-27B": {
        "BF16-direct NVFP4": "qwen35-27b-direct",
        "4o6": "qwen35-27b-4o6",
        "4o6+SVDQ": "qwen35-27b-4o6-svdq",
    },
    "Qwen3.8-27B": {
        "BF16-direct NVFP4": "qwen38-27b-direct",
        "4o6": "qwen38-27b-4o6",
        "4o6+SVDQ": "qwen38-27b-4o6-svdq",
    },
    "Qwen3-30B-A3B": {
        "BF16-direct NVFP4": None,
        "4o6": "qwen3-30b-a3b-4o6",
        "4o6+SVDQ": "qwen3-30b-a3b-4o6-svdq-sharded-final",
    },
    "Qwen3-8B": {
        "BF16-direct NVFP4": None,
        "4o6": "qwen3-8b-4o6",
        "4o6+SVDQ": "qwen3-8b-4o6-svdq",
    },
}
METHODS: Final = (
    "BF16",
    "ModelOpt NVFP4",
    "BF16-direct NVFP4",
    "4o6",
    "4o6+SVDQ",
)
EXPECTED_SAMPLES: Final = {"mmlu": 14042, "gsm8k": 1319, "mbpp": 500}


class ConsolidationError(RuntimeError):
    """Raised when final evaluation evidence is missing or inconsistent."""


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ConsolidationError(f"missing {path}") from error
    except json.JSONDecodeError as error:
        raise ConsolidationError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ConsolidationError(f"expected JSON object in {path}")
    return value


def _percentage(value: object, label: str) -> float:
    if not isinstance(value, (int, float)):
        raise ConsolidationError(f"expected numeric {label}, found {value!r}")
    return round(float(value) * 100.0, 4)


def load_rerun(result_dir: Path, model: str, method: str) -> dict[str, float]:
    """Load and validate one merged lm-eval result plus normalized MBPP."""
    merged = _load_object(result_dir / "merged_results.json")
    for task, expected in EXPECTED_SAMPLES.items():
        task_result = merged.get(task)
        if not isinstance(task_result, dict) or task_result.get("samples") != expected:
            raise ConsolidationError(
                f"{model} {method} {task}: expected {expected} samples"
            )

    normalized_path = result_dir / "normalized_mbpp.json"
    try:
        normalized = json.loads(normalized_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ConsolidationError(
            f"invalid normalized MBPP file {normalized_path}"
        ) from error
    if not isinstance(normalized, list) or len(normalized) != 1:
        raise ConsolidationError(f"expected one MBPP summary in {normalized_path}")
    summary = normalized[0]
    if (
        not isinstance(summary, dict)
        or summary.get("samples") != EXPECTED_SAMPLES["mbpp"]
    ):
        raise ConsolidationError(f"invalid MBPP summary in {normalized_path}")
    if summary.get("model") != model or summary.get("precision") != method:
        raise ConsolidationError(
            f"MBPP identity mismatch in {normalized_path}: {summary!r}"
        )

    mmlu = merged["mmlu"]
    gsm8k = merged["gsm8k"]
    mbpp = merged["mbpp"]
    assert isinstance(mmlu, dict) and isinstance(gsm8k, dict) and isinstance(mbpp, dict)
    return {
        "mmlu": _percentage(mmlu.get("acc"), "mmlu.acc"),
        "gsm8k": _percentage(gsm8k.get("flexible_extract"), "gsm8k.flexible_extract"),
        "mbpp_raw": _percentage(mbpp.get("pass_at_1"), "mbpp.pass_at_1"),
        "mbpp_normalized": round(float(summary["normalized_pass"]) / 5.0, 4),
    }


def consolidate(baseline: dict[str, Any], results_root: Path) -> dict[str, Any]:
    """Return the final table object with every requested rerun cell replaced."""
    output = deepcopy(baseline)
    models = output.get("models")
    if not isinstance(models, list):
        raise ConsolidationError("baseline models must be a list")
    by_name = {row.get("model"): row for row in models if isinstance(row, dict)}
    if set(by_name) != set(MODEL_METHOD_DIRS):
        raise ConsolidationError("baseline model set does not match final matrix")

    for model, method_dirs in MODEL_METHOD_DIRS.items():
        row = by_name[model]
        scores = row.get("scores")
        if not isinstance(scores, dict):
            raise ConsolidationError(f"missing scores for {model}")
        for method in tuple(scores):
            if method not in METHODS:
                del scores[method]
        for method, directory in method_dirs.items():
            scores[method] = (
                {"status": "not run"}
                if directory is None
                else load_rerun(results_root / directory, model, method)
            )
    output.pop("invalidated_methods", None)
    output["rerun"] = {
        "date": "2026-09-02",
        "activation_global_scale": "runtime dynamic",
        "svdq_rank": 64,
        "results_root": str(results_root.resolve()),
    }
    return output


def _cell(score: object, metric: str) -> str:
    if not isinstance(score, dict) or score.get("status") == "not run":
        return "not run"
    value = score.get(metric)
    if not isinstance(value, (int, float)):
        raise ConsolidationError(f"missing {metric} in {score!r}")
    return f"{float(value):.4f}"


def render_markdown(table: dict[str, Any]) -> str:
    """Render separate MMLU, GSM8K, and MBPP tables with full model names."""
    models = table["models"]
    lines = [
        "# Final Qwen BF16 / NVFP4 / 4o6 accuracy",
        "",
        "All values are percentages. New-mode checkpoints were converted from BF16; both ",
        "BF16-direct NVFP4 and 4o6 use runtime-dynamic activation global scaling. ",
        "4o6+SVDQ uses rank 64. vLLM is pinned to `0.24.0+092c4842.dev` and ",
        "lm-eval to `0.4.10`.",
        "",
    ]
    for title, metric in (
        ("MMLU (5-shot)", "mmlu"),
        ("GSM8K flexible-extract (5-shot)", "gsm8k"),
    ):
        lines.extend((f"## {title}", "", "| Model | " + " | ".join(METHODS) + " |"))
        lines.append("|---|" + "---:|" * len(METHODS))
        for row in models:
            scores = row["scores"]
            values = [_cell(scores.get(method), metric) for method in METHODS]
            lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
        lines.append("")
    lines.extend(
        (
            "## MBPP (3-shot)",
            "",
            "Each cell is `raw / normalized`; normalized applies deterministic Markdown-fence extraction before the same evaluator.",
            "",
        )
    )
    lines.append("| Model | " + " | ".join(METHODS) + " |")
    lines.append("|---|" + "---:|" * len(METHODS))
    for row in models:
        scores = row["scores"]
        values = []
        for method in METHODS:
            score = scores.get(method)
            raw = _cell(score, "mbpp_raw")
            norm = _cell(score, "mbpp_normalized")
            values.append("not run" if raw == "not run" else f"{raw} / {norm}")
        lines.append(f"| {row['model']} | " + " | ".join(values) + " |")
    lines.extend(("", "## Coverage", "", "| Model | GEMM path |", "|---|---|"))
    for row in models:
        lines.append(f"| {row['model']} | {row['gemm']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("output_markdown", type=Path)
    args = parser.parse_args()
    table = consolidate(
        _load_object(args.baseline), args.results_root.resolve(strict=True)
    )
    args.output_json.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    args.output_markdown.write_text(render_markdown(table), encoding="utf-8")


if __name__ == "__main__":
    main()
