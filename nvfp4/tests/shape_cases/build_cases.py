"""Build a bounded regression set from the verified full-v3 capture exports."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .selection import representative_m


def build(source: Path, destination: Path) -> None:
    """Select dense dimensions and occupancy records; retain source provenance."""
    groups: dict[tuple[str, str, int, int], Counter[int]] = defaultdict(Counter)
    origins: dict[tuple[str, str, int, int, int], set[str]] = defaultdict(set)
    union = source / "logical-gemm-union.csv.gz"
    with gzip.open(union, "rt") as stream:
        for row in csv.DictReader(stream):
            if row["family"] not in {"dense", "lm_head"}:
                continue
            key = (row["model"], row["family"], int(row["N"]), int(row["K"]))
            m = int(row["M"])
            groups[key][m] += int(row["occurrences"])
            origins[(*key, m)].add(f"{row['recipe']}/{row['task']}/{row['phase']}")
    dense = []
    for (model, family, n, k), counts in sorted(groups.items()):
        for m in representative_m(counts):
            dense.append(
                {
                    "model": model,
                    "family": family,
                    "m": m,
                    "n": n,
                    "k": k,
                    "quantized": family != "lm_head" and n != 128,
                    "origins": sorted(origins[model, family, n, k, m]),
                }
            )
    moe = []
    for task in ("gsm8k", "mbpp", "mmlu"):
        cell = source / "full-v3/Qwen3-30B-A3B/bf16" / task
        validated = json.loads((cell / "validated-shapes.json").read_text())
        selected: set[int] = set()
        for phase, values in validated["phase_M"].items():
            ordered = sorted(map(int, values))
            cumulative = 0
            total = sum(values.values())
            for m in ordered:
                cumulative += values[str(m)]
                if cumulative * 2 >= total:
                    selected.add(m)
                    break
            selected.add(ordered[-1])
            if phase == "decode":
                for target in (1, 128, 129):
                    selected.add(min(ordered, key=lambda m: (abs(m - target), m)))
                above = [m for m in ordered if m > 128]
                if above:
                    selected.add(above[0])
        best = {}
        phases = {}
        with gzip.open(cell / "worker/events.jsonl.gz", "rt") as stream:
            for line in stream:
                event = json.loads(line)
                if event["kind"] == "step":
                    phase = (
                        "mixed"
                        if event["prefill"] and event["decode"]
                        else ("prefill" if event["prefill"] else "decode")
                    )
                    phases[event["step"]] = phase
                if event["kind"] != "routing" or event["M"] not in selected:
                    continue
                key = (phases[event["step"]], event["M"])
                if key not in best or max(event["counts"]) > max(best[key]["counts"]):
                    best[key] = event
        for (phase, m), event in sorted(best.items()):
            moe.append(
                {
                    "task": task,
                    "phase": phase,
                    "m": m,
                    "counts": event["counts"],
                    "step": event["step"],
                    "module": event["module"],
                    "origin": event["origin"],
                    "source": str(
                        (cell / "worker/events.jsonl.gz").relative_to(source)
                    ),
                }
            )
    payload = {
        "schema": 1,
        "source": source.name,
        "union_sha256": hashlib.sha256(union.read_bytes()).hexdigest(),
        "selection": "observed range/median/dispatch neighbors; maximum expert skew per selected phase/M",
        "dense": dense,
        "moe": moe,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {"dense": len(dense), "moe": len(moe), "destination": str(destination)}
        ),
        flush=True,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python -m shape_cases.build_cases CAPTURE_ROOT OUTPUT_JSON"
        )
    build(Path(sys.argv[1]), Path(sys.argv[2]))
