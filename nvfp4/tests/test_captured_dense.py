"""Real Qwen3 dimensions; each process selects one quantization recipe."""

import json
import os
from pathlib import Path

import pytest
import torch

from shape_cases.dense import Recipe, run

CASES = json.loads((Path(__file__).parent / "data/qwen3_shapes.json").read_text())[
    "dense"
]
GROUPS = sorted({(row["model"], row["n"], row["k"], row["quantized"]) for row in CASES})


@pytest.mark.parametrize("model,n,k,quantized", GROUPS)
def test_captured_dense(model: str, n: int, k: int, quantized: bool) -> None:
    """Every selected M passes numerical checks before separate timing runs."""
    recipe = Recipe(os.environ["RECIPE"]) if quantized else Recipe.BF16
    if not quantized and os.environ["RECIPE"] != "bf16":
        pytest.skip("Unquantized router/LM head is shared by all recipes")
    ms = tuple(
        sorted(
            {
                row["m"]
                for row in CASES
                if row["model"] == model and row["n"] == n and row["k"] == k
            }
        )
    )
    with torch.inference_mode():
        run(recipe, n, k, ms)
