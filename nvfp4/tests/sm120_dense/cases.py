"""Select only the existing Qwen3-8B dimensions and token counts."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class DenseCase:
    """One already captured dense operation, with model identity preserved."""

    m: int
    n: int
    k: int
    quantized: bool
    model: str = "Qwen3-8B"

    @property
    def label(self) -> str:
        return f"{self.model}-M{self.m}-N{self.n}-K{self.k}"


def selected_cases() -> tuple[DenseCase, ...]:
    """Exclude every MoE case and every dense case belonging to Qwen3-30B."""
    source = Path(__file__).resolve().parents[1] / "data/qwen3_shapes.json"
    rows = json.loads(source.read_text())["dense"]
    cases = tuple(
        DenseCase(int(row["m"]), int(row["n"]), int(row["k"]), bool(row["quantized"]))
        for row in rows
        if row["model"] == "Qwen3-8B"
    )
    assert len(cases) == 47 and len(set(cases)) == 47
    assert sum(case.quantized for case in cases) == 40
    return cases


CASES: Final = selected_cases()
