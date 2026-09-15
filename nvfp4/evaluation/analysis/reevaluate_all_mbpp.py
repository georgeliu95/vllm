# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-evaluate persisted MBPP samples after deterministic fence extraction."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Callable, Final

PYTHON_FENCE: Final = re.compile(r"```python\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
GENERIC_FENCE: Final = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
VARIANTS: Final = {
    "Qwen3.5-35B-A3B|BF16": "qwen35-bf16-vllm-6629415",
    "Qwen3.5-35B-A3B|NVFP4": "qwen35-nvfp4-oci-hsg-6621390",
    "Qwen3.5-35B-A3B|4o6": "qwen35-35b-a3b-4o6-full-sharded",
    "Qwen3.5-35B-A3B|4o6+SVDQ": ("qwen35-35b-a3b-svdq-qerror-full-sharded-final-4096"),
    "Qwen3.5-27B|BF16": "qwen35-27b-bf16-vllm",
    "Qwen3.5-27B|NVFP4": "Qwen3.5-27B-NVFP4-CompressedTensors-vLLM",
    "Qwen3.5-27B|4o6": "qwen35-27b-4o6-full",
    "Qwen3.5-27B|4o6+SVDQ": "qwen35-27b-svdq-qerror-full",
    "Qwen3.8-27B|BF16": "Qwen3.8-27B-BF16-vLLM",
    "Qwen3.8-27B|NVFP4": "Qwen3.8-27B-NVFP4-vLLM",
    "Qwen3.8-27B|4o6": "qwen38-27b-4o6-full",
    "Qwen3.8-27B|4o6+SVDQ": "qwen38-27b-svdq-qerror-full",
    "Qwen3-30B-A3B|BF16": "Qwen3-30B-A3B-BF16-vLLM",
    "Qwen3-30B-A3B|NVFP4": "Qwen3-30B-A3B-NVFP4-vLLM-BF16KV",
    "Qwen3-30B-A3B|4o6": "qwen3-30b-a3b-4o6-full",
    "Qwen3-30B-A3B|4o6+SVDQ": ("qwen3-30b-a3b-svdq-qerror-full-sharded-dummy-row"),
    "Qwen3-8B|BF16": "Qwen3-8B-BF16-vLLM",
    "Qwen3-8B|NVFP4": "Qwen3-8B-NVFP4-vLLM-BF16KV",
    "Qwen3-8B|4o6": "qwen3-8b-4o6-full",
    "Qwen3-8B|4o6+SVDQ": "qwen3-8b-svdq-qerror-full",
}


@dataclass(frozen=True)
class Summary:
    model: str
    precision: str
    samples: int
    fenced: int
    raw_pass: int
    normalized_pass: int
    normalized_failures: dict[str, int]


class EvaluationError(RuntimeError):
    """Raised when persisted evaluator inputs violate the expected contract."""


def normalize_completion(response: str) -> str:
    """Extract the first Python or generic Markdown code block when present."""
    python_match = PYTHON_FENCE.search(response)
    if python_match is not None:
        return python_match.group(1).strip()
    generic_match = GENERIC_FENCE.search(response)
    if generic_match is not None:
        return generic_match.group(1).strip()
    return response.strip()


def find_samples(output_root: Path, result_dir: str) -> tuple[Path, ...]:
    """Resolve one or more sample files; ``load_samples`` validates coverage."""
    matches = tuple((output_root / result_dir / "raw").glob("**/samples_mbpp_*.jsonl"))
    if not matches:
        raise EvaluationError(
            f"expected at least one MBPP sample file under {result_dir}, found 0"
        )
    return tuple(sorted(matches))


def load_samples(paths: tuple[Path, ...]) -> tuple[dict[str, object], ...]:
    """Load the JSONL records at the persisted-results trust boundary."""
    samples: tuple[dict[str, object], ...] = ()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            samples += tuple(json.loads(line) for line in source)
    if len(samples) != 500:
        raise EvaluationError(
            f"expected 500 MBPP samples across {paths}, found {len(samples)}"
        )
    by_doc_id = {int(sample["doc_id"]): sample for sample in samples}
    if len(by_doc_id) != 500 or set(by_doc_id) != set(range(500)):
        raise EvaluationError(
            "MBPP shards do not form a disjoint, complete doc_id cover"
        )
    return tuple(by_doc_id[index] for index in range(500))


def evaluate_variant(
    key: str,
    sample_paths: tuple[Path, ...],
    checker: Callable[[str, float, int, int], dict[str, object]],
) -> Summary:
    """Apply one parsing rule and the pinned code evaluator to one variant."""
    model, precision = key.split("|", maxsplit=1)
    samples = load_samples(sample_paths)
    responses = tuple(str(sample["filtered_resps"][0]) for sample in samples)
    programs = tuple(
        f"{normalize_completion(response)}\n{sample['target']}"
        for response, sample in zip(responses, samples, strict=True)
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = tuple(
            executor.map(
                checker,
                programs,
                (3.0 for _ in samples),
                (int(sample["doc_id"]) for sample in samples),
                (0 for _ in samples),
            )
        )
    failures = Counter(
        str(result["result"]) for result in results if not bool(result["passed"])
    )
    return Summary(
        model=model,
        precision=precision,
        samples=len(samples),
        fenced=sum("```" in response for response in responses),
        raw_pass=sum(float(sample["pass_at_1"]) == 1.0 for sample in samples),
        normalized_pass=sum(bool(result["passed"]) for result in results),
        normalized_failures=dict(failures.most_common()),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("result_json", type=Path)
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="evaluate only this variant key or result-directory name (repeatable)",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="MODEL|PRECISION=RESULT_DIR",
        help="override the built-in variant map (repeatable)",
    )
    return parser.parse_args()


def parse_variant_specs(specs: list[str]) -> dict[str, str]:
    """Parse explicit labels and result directories at the CLI boundary."""
    variants: dict[str, str] = {}
    for spec in specs:
        key, separator, result_dir = spec.partition("=")
        if separator == "" or "|" not in key or not result_dir:
            raise EvaluationError(
                "--variant must use MODEL|PRECISION=RESULT_DIR syntax"
            )
        if key in variants:
            raise EvaluationError(f"duplicate --variant key: {key}")
        variants[key] = result_dir
    return variants


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve(strict=True)
    from code_eval.execute import check_correctness

    requested = set(args.only)
    configured_variants = parse_variant_specs(args.variant) or VARIANTS
    variants = {
        key: result_dir
        for key, result_dir in configured_variants.items()
        if not requested or key in requested or result_dir in requested
    }
    if requested and not variants:
        raise EvaluationError(f"no variants matched --only={sorted(requested)}")
    summaries = tuple(
        evaluate_variant(
            key,
            find_samples(output_root, result_dir),
            check_correctness,
        )
        for key, result_dir in variants.items()
    )
    args.result_json.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
