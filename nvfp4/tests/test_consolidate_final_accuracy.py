import json
from pathlib import Path
import sys


EXAMPLE_ROOT = Path(__file__).resolve().parents[1] / "evaluation"
sys.path.insert(0, str(EXAMPLE_ROOT))

from analysis.consolidate_final_accuracy import (  # noqa: E402
    MODEL_METHOD_DIRS,
    ConsolidationError,
    consolidate,
    render_markdown,
)


def _baseline() -> dict[str, object]:
    gemm = {
        "Qwen3.5-35B-A3B": "MoE GEMM",
        "Qwen3.5-27B": "Dense GEMM",
        "Qwen3.8-27B": "Dense GEMM",
        "Qwen3-30B-A3B": "MoE GEMM",
        "Qwen3-8B": "Dense GEMM",
    }
    return {
        "invalidated_methods": {"4o6": "stale"},
        "models": [
            {
                "model": model,
                "gemm": gemm[model],
                "scores": {
                    "BF16": {
                        "mmlu": 1.0,
                        "gsm8k": 2.0,
                        "mbpp_raw": 3.0,
                        "mbpp_normalized": 4.0,
                    },
                    "ModelOpt NVFP4": {
                        "mmlu": 5.0,
                        "gsm8k": 6.0,
                        "mbpp_raw": 7.0,
                        "mbpp_normalized": 8.0,
                    },
                    "obsolete": {"mmlu": 0.0},
                },
            }
            for model in MODEL_METHOD_DIRS
        ],
    }


def _write_result(root: Path, directory: str, model: str, method: str) -> None:
    target = root / directory
    target.mkdir()
    (target / "merged_results.json").write_text(
        json.dumps(
            {
                "mmlu": {"acc": 0.81, "samples": 14042, "subtasks": 57},
                "gsm8k": {
                    "flexible_extract": 0.92,
                    "strict_match": 0.91,
                    "samples": 1319,
                },
                "mbpp": {"pass_at_1": 0.63, "samples": 500},
            }
        )
    )
    (target / "normalized_mbpp.json").write_text(
        json.dumps(
            [
                {
                    "model": model,
                    "precision": method,
                    "samples": 500,
                    "normalized_pass": 370,
                }
            ]
        )
    )


def test_consolidates_requested_matrix_and_renders_separate_tables(
    tmp_path: Path,
) -> None:
    for model, methods in MODEL_METHOD_DIRS.items():
        for method, directory in methods.items():
            if directory is not None:
                _write_result(tmp_path, directory, model, method)

    result = consolidate(_baseline(), tmp_path)
    assert "invalidated_methods" not in result
    qwen35 = result["models"][0]
    assert qwen35["scores"]["BF16-direct NVFP4"] == {
        "mmlu": 81.0,
        "gsm8k": 92.0,
        "mbpp_raw": 63.0,
        "mbpp_normalized": 74.0,
    }
    qwen3_moe = result["models"][3]
    assert qwen3_moe["scores"]["BF16-direct NVFP4"] == {"status": "not run"}
    markdown = render_markdown(result)
    assert "## MMLU (5-shot)" in markdown
    assert "## GSM8K flexible-extract (5-shot)" in markdown
    assert "## MBPP (3-shot)" in markdown
    assert "Qwen3.5-35B-A3B" in markdown
    assert "63.0000 / 74.0000" in markdown


def test_rejects_incomplete_sample_coverage(tmp_path: Path) -> None:
    model = "Qwen3.5-35B-A3B"
    method = "BF16-direct NVFP4"
    directory = MODEL_METHOD_DIRS[model][method]
    assert directory is not None
    _write_result(tmp_path, directory, model, method)
    merged_path = tmp_path / directory / "merged_results.json"
    merged = json.loads(merged_path.read_text())
    merged["mmlu"]["samples"] = 14041
    merged_path.write_text(json.dumps(merged))
    try:
        consolidate(_baseline(), tmp_path)
    except ConsolidationError as error:
        assert "expected 14042 samples" in str(error)
    else:
        raise AssertionError("incomplete MMLU coverage was accepted")
