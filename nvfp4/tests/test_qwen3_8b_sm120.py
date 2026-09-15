"""Qwen3-8B only: packed math, changing-data graph replay, then optional timing."""

import json
import os
from dataclasses import asdict

import pytest
import torch

from shape_cases.timing import measure, measure_graph
from sm120_dense.backend import DenseLayer, Recipe
from sm120_dense.cases import CASES, DenseCase
from sm120_dense.reference import check_output, check_replay


def test_shape_scope() -> None:
    """Guard the user-requested model boundary independently of recipe choice."""
    assert {case.model for case in CASES} == {"Qwen3-8B"}
    assert {(case.n, case.k) for case in CASES} == {
        (4096, 4096),
        (6144, 4096),
        (24576, 4096),
        (4096, 12288),
        (151936, 4096),
    }


@pytest.fixture(scope="module")
def layers() -> dict[tuple[int, int], DenseLayer]:
    """Retain one weight pack per dimension without rebuilding for each M."""
    return {}


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.label)
def test_dense(case: DenseCase, layers: dict[tuple[int, int], DenseLayer]) -> None:
    """Forbid unsupported architecture and validate before timing the full call."""
    recipe = Recipe(os.environ["RECIPE"])
    if not case.quantized and recipe != Recipe.BF16:
        pytest.skip("Qwen3-8B LM head stays BF16 for every recipe")
    assert torch.cuda.get_device_capability() == (12, 0)
    with torch.inference_mode():
        key = (case.n, case.k)
        if key not in layers:
            layers[key] = DenseLayer.create(recipe, case.n, case.k)
        layer = layers[key]
        torch.manual_seed(9000 + case.m)
        values = torch.randn(case.m, case.k, device="cuda", dtype=torch.bfloat16)
        evidence = check_output(layer, values)
        check_replay(layer, values)
        print(
            json.dumps(
                {
                    "recipe": recipe,
                    "model": case.model,
                    "shape_mnk": [case.m, case.n, case.k],
                    "backend": layer.backend,
                    "correctness": evidence,
                }
            ),
            flush=True,
        )
        if os.environ.get("SM120_BENCHMARK") == "1":
            for timing in (
                measure("full_eager", lambda: layer(values), iterations=10),
                measure_graph("full_cuda_graph", lambda: layer(values)),
            ):
                print(
                    json.dumps(
                        {
                            "recipe": recipe,
                            "model": case.model,
                            "shape_mnk": [case.m, case.n, case.k],
                            "backend": layer.backend,
                            "timing": asdict(timing),
                        }
                    ),
                    flush=True,
                )
