# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess
import sys
import json
from pathlib import Path
from typing import Final

import torch
import pytest
from safetensors import safe_open

from converter.convert_4o6_ckpt import (
    ShardWriter,
    TensorMeta,
    build_svdquant_metadata,
    decompose_svdquant_weight,
    fused_expert_source,
    input_scale,
    make_groups,
    SourceInfo,
    split_fused_svdquant_factors,
    svdquant_factor_base,
    virtual_fused_expert_bases,
    write_quant_configs,
    import_runtime_deps,
)

import_runtime_deps()

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
CONVERTER_MODULE: Final = (
    REPO_ROOT / "converter" / "convert_4o6_ckpt.py"
)


def test_svdquant_pads_narrow_projection_to_requested_rank() -> None:
    weight = torch.randn(3, 8, dtype=torch.float32)

    us, vh, residual = decompose_svdquant_weight(
        weight, rank=5, us_dtype=torch.float32, device="cpu"
    )

    assert us.shape == (3, 5)
    assert vh.shape == (5, 8)
    assert torch.count_nonzero(us[:, 3:]) == 0
    assert torch.count_nonzero(vh[3:]) == 0
    torch.testing.assert_close(residual + us @ vh, weight)


def test_vllm_quantization_method_is_modelopt(tmp_path) -> None:
    source = SourceInfo(
        fmt="bf16",
        config={"model_type": "qwen3"},
        hf_quant_config=None,
    )

    write_quant_configs(tmp_path, source, ["lm_head"])

    config = json.loads((tmp_path / "config.json").read_text())
    assert config["quantization_config"] == {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "ignore": ["lm_head"],
    }


def test_uncalibrated_4o6_activation_scale_uses_safe_vllm_range() -> None:
    assert input_scale("fc31", "4o6").item() == pytest.approx(1.0 / 64.0)
    assert input_scale("linear", "4o6").item() == pytest.approx(1.0 / 64.0)
    assert input_scale("fc2", "4o6").item() == pytest.approx(1.0 / 16.0)


def test_convert_4o6_ckpt_module_exposes_cli_help() -> None:
    # Given an invocation of the installable converter module.
    command = [sys.executable, str(CONVERTER_MODULE), "--help"]

    # When argparse renders the command help without loading model weights.
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then the CLI exits successfully and advertises the Kimi INT4 input mode.
    assert result.returncode == 0, result.stderr
    assert "--source-format" in result.stdout
    assert "--svdquant-rank" in result.stdout


def test_svdquant_factor_serialization_and_metadata_round_trip(tmp_path) -> None:
    # Given one BF16 low-rank factor pair and the offline artifact contract.
    writer = ShardWriter(tmp_path, max_shard_bytes=1024 * 1024)
    us = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    vh = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    # When the converter's normal shard writer persists and indexes the pair.
    writer.add("model.layers.1.mlp.experts.0.gate_proj.svdquant_us", us)
    writer.add("model.layers.1.mlp.experts.0.gate_proj.svdquant_vh", vh)
    writer.finalize()

    # Then values, dtype, and immutable INT4-derived metadata round-trip.
    shard = tmp_path / "model-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        torch.testing.assert_close(
            handle.get_tensor(
                "model.layers.1.mlp.experts.0.gate_proj.svdquant_us"), us)
        torch.testing.assert_close(
            handle.get_tensor(
                "model.layers.1.mlp.experts.0.gate_proj.svdquant_vh"), vh)
    assert build_svdquant_metadata(64, "bfloat16", "bf16") == {
        "format": "offline-lowrank-residual-v1",
        "rank": 64,
        "factor_dtype": "bfloat16",
        "source_format": "bf16",
        "decomposition": "source-residual",
        "stages": ["dense-linear", "moe-fc13", "moe-fc2"],
        "reference": "source-weight",
    }


def test_svdquant_quantization_error_metadata() -> None:
    metadata = build_svdquant_metadata(
        64, "bfloat16", "bf16", "quantization-error"
    )

    assert metadata["decomposition"] == "quantization-error"
    assert metadata["reference"] == "quantization-error"


def test_svdquant_metadata_preserves_int4_reference() -> None:
    assert build_svdquant_metadata(
        64, "bfloat16", "int4-compressed-tensors"
    ) == {
        "format": "offline-lowrank-residual-v1",
        "rank": 64,
        "factor_dtype": "bfloat16",
        "source_format": "int4-compressed-tensors",
        "stages": ["dense-linear", "moe-fc13", "moe-fc2"],
        "reference": "dequantized-native-int4",
        "decomposition": "source-residual",
    }


def test_dense_fused_factor_names_match_vllm_modules() -> None:
    assert svdquant_factor_base([
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
    ]) == "model.layers.0.self_attn.qkv_proj"
    assert svdquant_factor_base([
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ]) == "model.layers.0.mlp.gate_up_proj"
    assert svdquant_factor_base([
        "model.layers.0.mlp.experts.3.gate_proj",
        "model.layers.0.mlp.experts.3.up_proj",
    ]) is None


def test_qwen35_linear_attention_factors_share_vllm_fused_basis() -> None:
    prefix = "model.language_model.layers.0.linear_attn"
    groups = make_groups([
        f"{prefix}.in_proj_a",
        f"{prefix}.in_proj_qkv",
        f"{prefix}.in_proj_z",
        f"{prefix}.in_proj_b",
    ])

    assert groups == [
        [f"{prefix}.in_proj_b", f"{prefix}.in_proj_a"],
        [f"{prefix}.in_proj_qkv", f"{prefix}.in_proj_z"],
    ]
    assert svdquant_factor_base(groups[0]) == f"{prefix}.in_proj_ba"
    assert svdquant_factor_base(groups[1]) == f"{prefix}.in_proj_qkvz"


def test_fused_svdquant_factors_use_source_names_for_vllm_stacked_loader() -> None:
    group = [
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
    ]
    us = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    vh = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    factors = split_fused_svdquant_factors(group, [2, 4], us, vh)

    assert [base for base, _, _ in factors] == group
    torch.testing.assert_close(factors[0][1], us[:2])
    torch.testing.assert_close(factors[1][1], us[2:])
    torch.testing.assert_close(factors[0][2], vh)
    torch.testing.assert_close(factors[1][2], vh)
    assert factors[0][2].data_ptr() != factors[1][2].data_ptr()


def test_qwen35_fused_experts_expand_to_virtual_vllm_weights() -> None:
    metas = {
        "model.layers.0.mlp.experts.gate_up_proj": TensorMeta(
            dtype="BF16", shape=(2, 1024, 2048), filename="model.safetensors"
        ),
        "model.layers.0.mlp.experts.down_proj": TensorMeta(
            dtype="BF16", shape=(2, 2048, 512), filename="model.safetensors"
        ),
    }

    bases = virtual_fused_expert_bases(metas)

    assert len(bases) == 6
    assert "model.layers.0.mlp.experts.1.up_proj" in bases
    assert fused_expert_source("model.layers.0.mlp.experts.1.up_proj") == (
        "model.layers.0.mlp.experts.gate_up_proj",
        1,
        "up_proj",
    )
