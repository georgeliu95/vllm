# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration boundary for the pinned vLLM adapter."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Final, Literal

from .quantization import FOUR_OVER_SIX_QUANT_RANGE, STANDARD_NVFP4_QUANT_RANGE

MODE_ENV: Final = "VLLM_NVFP4_4O6_MODE"
RANK_ENV: Final = "VLLM_NVFP4_SVDQ_RANK"
SUPPORTED_ARTIFACT: Final = "offline-lowrank-residual-v1"

RuntimeMode = Literal["standard", "4o6", "4o6-svdq"]


class AdapterConfigError(ValueError):
    """The checkpoint and requested runtime mode are incompatible."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Validated runtime contract shared by Dense and MoE paths."""

    mode: RuntimeMode
    rank: int
    checkpoint: Path

    @property
    def svdq_enabled(self) -> bool:
        return self.mode == "4o6-svdq"

    @property
    def four_over_six_enabled(self) -> bool:
        return self.mode != "standard"

    @property
    def activation_quant_range(self) -> float:
        return (
            FOUR_OVER_SIX_QUANT_RANGE
            if self.four_over_six_enabled
            else STANDARD_NVFP4_QUANT_RANGE
        )


def _load_metadata(checkpoint: Path) -> dict[str, object]:
    metadata_path = checkpoint / "hf_quant_config.json"
    try:
        payload = json.loads(metadata_path.read_text())
    except FileNotFoundError as error:
        raise AdapterConfigError(f"Missing {metadata_path}.") from error
    except json.JSONDecodeError as error:
        raise AdapterConfigError(f"Invalid JSON in {metadata_path}.") from error
    if not isinstance(payload, dict):
        raise AdapterConfigError(f"Expected an object in {metadata_path}.")
    return payload


def load_runtime_config(checkpoint: str | Path) -> RuntimeConfig:
    """Parse environment flags and validate checkpoint metadata."""
    checkpoint_path = Path(checkpoint).resolve()
    mode_value = os.environ.get(MODE_ENV, "4o6")
    if mode_value not in ("standard", "4o6", "4o6-svdq"):
        raise AdapterConfigError(f"{MODE_ENV} must be standard, 4o6, or 4o6-svdq.")
    mode: RuntimeMode = mode_value

    rank_text = os.environ.get(RANK_ENV, "64")
    try:
        rank = int(rank_text)
    except ValueError as error:
        raise AdapterConfigError(f"{RANK_ENV} must be an integer.") from error
    if rank <= 0:
        raise AdapterConfigError(f"{RANK_ENV} must be positive.")

    metadata = _load_metadata(checkpoint_path)
    quantization = metadata.get("quantization")
    if not isinstance(quantization, dict) or quantization.get("quant_algo") != "NVFP4":
        raise AdapterConfigError("Checkpoint must use ModelOpt NVFP4 weights.")
    if mode == "4o6-svdq":
        svdq = metadata.get("svdquant")
        if not isinstance(svdq, dict):
            raise AdapterConfigError("SVDQ mode requires svdquant checkpoint metadata.")
        if svdq.get("format") != SUPPORTED_ARTIFACT:
            raise AdapterConfigError("Unsupported SVDQ artifact format.")
        if svdq.get("rank") != rank:
            raise AdapterConfigError(
                f"Runtime rank {rank} does not match checkpoint rank {svdq.get('rank')}."
            )
    return RuntimeConfig(mode=mode, rank=rank, checkpoint=checkpoint_path)
