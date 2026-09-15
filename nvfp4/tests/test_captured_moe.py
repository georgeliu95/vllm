"""Routed Qwen3-30B cases share native occupancy across all four recipes."""

import json
import os
from pathlib import Path

import pytest
import torch

from shape_cases.moe import MoELayer

CASES = json.loads((Path(__file__).parent / "data/qwen3_shapes.json").read_text())[
    "moe"
]


@pytest.fixture(scope="module")
def layer() -> MoELayer:
    with torch.inference_mode():
        return MoELayer(os.environ["RECIPE"])


@pytest.mark.parametrize(
    "index",
    range(len(CASES)),
    ids=[f"{c['task']}-{c['phase']}-M{c['m']}" for c in CASES],
)
def test_captured_moe(layer: MoELayer, index: int) -> None:
    case = CASES[index]
    with torch.inference_mode():
        layer.check(
            tuple(case["counts"]),
            f"{case['task']}/{case['phase']}/step{case['step']}/{case['module']}",
        )
