# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
# Run: RECIPE=bf16|nvfp4|4o6|svdq retained-container python dense_breakdown.py
"""Isolate the Qwen3-8B down projection using the retained serving primitives."""

import json
import os
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import assert_never

import torch
from torch.nn import functional as functional

from .timing import measure, measure_graph


class Recipe(StrEnum):
    BF16 = "bf16"
    NVFP4 = "nvfp4"
    FOUR = "4o6"
    SVDQ = "svdq"


@dataclass(frozen=True, slots=True)
class Layer:
    weight: torch.Tensor
    weight_scale: torch.Tensor
    weight_global_scale: torch.Tensor
    input_global_scale_inv: torch.Tensor
    alpha: torch.Tensor
    output_size_per_partition: int
    weights_padding_cols: int


def run(recipe: Recipe, n: int, k: int, ms: tuple[int, ...]) -> None:
    """Compare complete layer calls with separately warmed constituent calls."""
    match recipe:
        case Recipe.BF16 | Recipe.NVFP4:
            quant_range = 2688.0
            os.environ["FLASHINFER_NVFP4_4OVER6"] = "0"
        case Recipe.FOUR | Recipe.SVDQ:
            quant_range = 1536.0
            os.environ.update(
                FLASHINFER_NVFP4_4OVER6="1",
                FLASHINFER_NVFP4_4OVER6_E4M3_USE_256="1",
                FLASHINFER_NVFP4_4OVER6_ERR_MODE="MSE",
            )
        case unreachable:
            assert_never(unreachable)
    from flashinfer.quantization.kernels.nvfp4_quantize import nvfp4_quantize_cute_dsl
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm
    from vllm_adapter.quantization import dynamic_global_scale, quantize_dense_4o6
    from trt_quant.dense import fused_output, load_library

    torch.manual_seed(1234)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.01
    if recipe == Recipe.BF16:
        for m in ms:
            os.environ["B300_SHAPE_LABEL"] = f"dense/{recipe}/M={m}"
            torch.manual_seed(9000 + m)
            values = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            result = functional.linear(values, weight)
            assert result.shape == (m, n) and torch.isfinite(result).all()
            ref = functional.linear(
                values[:16].float(), weight[:: max(1, n // 256)].float()
            )
            got = result[:16, :: max(1, n // 256)].float()
            nmse = float((got - ref).square().sum() / ref.square().sum())
            assert nmse < 2.0e-5, nmse
            print(
                json.dumps(
                    {"recipe": recipe, "shape_mnk": [m, n, k], "reference_nmse": nmse}
                ),
                flush=True,
            )
            if os.environ.get("B300_BENCHMARK") == "1":
                print(
                    json.dumps(
                        {
                            "recipe": recipe,
                            "shape_mnk": [m, n, k],
                            "timing": asdict(
                                measure(
                                    "full", lambda: functional.linear(values, weight)
                                )
                            ),
                        }
                    ),
                    flush=True,
                )
                print(
                    json.dumps(
                        {
                            "recipe": recipe,
                            "shape_mnk": [m, n, k],
                            "timing": asdict(
                                measure_graph(
                                    "full_cuda_graph",
                                    lambda: functional.linear(values, weight),
                                )
                            ),
                        }
                    ),
                    flush=True,
                )
        return
    weight_inv, weight_deq = dynamic_global_scale(weight, quant_range)
    packed_w, scale_w = nvfp4_quantize_cute_dsl(
        weight, weight_inv.reshape(1), sf_layout=0, enable_pdl=False
    )
    layer = Layer(
        packed_w,
        scale_w.view(torch.float8_e4m3fn),
        weight_deq,
        torch.ones((), device="cuda"),
        torch.ones((), device="cuda"),
        n,
        0,
    )
    us = torch.randn(n, 64, device="cuda", dtype=torch.bfloat16) * 0.01
    vh = torch.randn(64, k, device="cuda", dtype=torch.bfloat16) * 0.01
    if recipe in (Recipe.FOUR, Recipe.SVDQ):
        load_library()
    for m in ms:
        os.environ["B300_SHAPE_LABEL"] = f"dense/{recipe}/M={m}"
        torch.manual_seed(9000 + m)
        values = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)

        def scale_update() -> torch.Tensor:
            inv, deq = dynamic_global_scale(values, quant_range)
            layer.input_global_scale_inv.copy_(inv)
            layer.alpha.copy_(deq * layer.weight_global_scale)
            return layer.alpha

        def quantize() -> tuple[torch.Tensor, torch.Tensor]:
            if recipe in (Recipe.FOUR, Recipe.SVDQ):
                return quantize_dense_4o6(values, layer.input_global_scale_inv)
            return scaled_fp4_quant(
                values,
                layer.input_global_scale_inv,
                is_sf_swizzled_layout=True,
                backend="flashinfer-cutedsl",
            )

        scale_update()
        aq, aq_sf = quantize()

        def gemm() -> torch.Tensor:
            return flashinfer_scaled_fp4_mm(
                aq,
                layer.weight,
                aq_sf,
                layer.weight_scale,
                layer.alpha,
                values.dtype,
                backend="cute-dsl",
            )

        def correction() -> torch.Tensor:
            return (values @ vh.T) @ us.T

        def fused_quant() -> torch.Tensor:
            return torch.ops.trtllm.fp4_quantize_fused(
                values, 16, False, True, 1, 1536.0, 1e-12, 0, 0
            )[0]

        def full() -> torch.Tensor:
            match recipe:
                case Recipe.BF16:
                    return functional.linear(values, weight)
                case Recipe.NVFP4:
                    scale_update()
                    codes, scales = quantize()
                    return flashinfer_scaled_fp4_mm(
                        codes,
                        layer.weight,
                        scales,
                        layer.weight_scale,
                        layer.alpha,
                        values.dtype,
                        backend="cute-dsl",
                    )
                case Recipe.FOUR:
                    return fused_output(layer, values, None)
                case Recipe.SVDQ:
                    return fused_output(layer, values, None) + correction()
                case unreachable:
                    assert_never(unreachable)

        actual = full()
        columns = slice(None, None, max(1, n // 256))
        reference = functional.linear(values[:16].float(), weight[columns].float())
        if recipe == Recipe.SVDQ:
            reference = reference + correction()[:16, columns].float()
        nmse = (
            (actual[:16, columns].float() - reference).square().mean()
            / reference.square().mean()
        ).item()
        assert nmse < 0.05, nmse
        print(
            json.dumps(
                {"recipe": recipe, "shape_mnk": [m, n, k], "reference_nmse": nmse}
            ),
            flush=True,
        )
        assert actual.shape == (m, n) and torch.isfinite(actual).all()
        if recipe == Recipe.NVFP4:
            torch.testing.assert_close(actual, gemm(), rtol=0, atol=0)
        print(
            json.dumps(
                {
                    "recipe": recipe,
                    "shape_mnk": [m, n, k],
                    "input": "seeded synthetic BF16",
                    "weight": "seeded synthetic; checkpoint dimensions",
                    "correctness": "finite and output shape; NVFP4 prequantized/full equality",
                }
            ),
            flush=True,
        )
        if os.environ.get("B300_BENCHMARK") != "1":
            continue
        operations = [("full", full)]
        if recipe != Recipe.BF16:
            operations.extend(
                [
                    ("main_gemm_prequantized", gemm),
                    ("dynamic_scale_and_alpha", scale_update),
                    ("unfused_activation_quant_only", lambda: quantize()[0]),
                ]
            )
        if recipe in (Recipe.FOUR, Recipe.SVDQ):
            operations.append(("fused_amax_quant_only", fused_quant))
        if recipe == Recipe.SVDQ:
            operations.append(("rank64_project_reconstruct", correction))
        for name, operation in operations:
            print(
                json.dumps(
                    {
                        "recipe": recipe,
                        "shape_mnk": [m, n, k],
                        "timing": asdict(measure(name, operation)),
                    }
                ),
                flush=True,
            )
        if os.environ.get("B300_COMPONENT_GRAPHS") == "1":
            for name, operation in operations[1:]:
                print(
                    json.dumps(
                        {
                            "recipe": recipe,
                            "shape_mnk": [m, n, k],
                            "timing": asdict(
                                measure_graph(name + "_cuda_graph", operation)
                            ),
                        }
                    ),
                    flush=True,
                )
        print(
            json.dumps(
                {
                    "recipe": recipe,
                    "shape_mnk": [m, n, k],
                    "timing": asdict(measure_graph("full_cuda_graph", full)),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    with torch.inference_mode():
        run(Recipe(os.environ["RECIPE"]), 4096, 12288, (1, 16, 128))
