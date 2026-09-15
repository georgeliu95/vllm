"""Qwen3 routed layer tests using captured occupancy and the serving dispatch."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import asdict
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from .selection import routes_from_counts
from .timing import measure, measure_graph


class MoELayer:
    """Reuse seeded weights across the selected real occupancy cases."""

    def __init__(self, recipe: str) -> None:
        self.recipe = recipe
        four = recipe in ("4o6", "svdq")
        self.quant_range = 1536.0 if four else 2688.0
        os.environ.update(
            FLASHINFER_NVFP4_4OVER6=str(int(four)),
            FLASHINFER_NVFP4_4OVER6_E4M3_USE_256="1",
            FLASHINFER_NVFP4_4OVER6_ERR_MODE="MSE",
        )
        import vllm.model_executor.layers.fused_moe  # noqa: F401 -- register native MoE operators
        from vllm_adapter.mainloop import MainloopMoE
        from vllm_adapter.quantization import dynamic_global_scale
        from flashinfer.quantization.kernels.nvfp4_quantize import (
            nvfp4_quantize_cute_dsl,
        )

        torch.manual_seed(1234)
        self.w1 = (
            torch.randn(128, 1536, 2048, device="cuda", dtype=torch.bfloat16) * 0.01
        )
        self.w2 = (
            torch.randn(128, 2048, 768, device="cuda", dtype=torch.bfloat16) * 0.01
        )
        self.factors: dict[str, torch.Tensor | None] = {
            name: None for name in ("w13_us", "w13_vh", "w2_us", "w2_vh")
        }
        self.mainloop = None
        if recipe == "bf16":
            from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
                convert_moe_weights_to_flashinfer_trtllm_block_layout,
            )

            self.b1, self.b2 = convert_moe_weights_to_flashinfer_trtllm_block_layout(
                {}, self.w1, self.w2, is_gated_act_gemm=True
            )
            return

        def pack(
            weight: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            codes, scales, deqs = [], [], []
            for expert in weight:
                inv, deq = dynamic_global_scale(expert, self.quant_range)
                q, s = nvfp4_quantize_cute_dsl(
                    expert, inv.reshape(1), sf_layout=0, enable_pdl=False
                )
                codes.append(q)
                scales.append(s.view(torch.float8_e4m3fn))
                deqs.append(deq)
            return torch.stack(codes), torch.stack(scales), torch.stack(deqs)

        self.q1, self.s1, self.d1 = pack(self.w1)
        self.q2, self.s2, self.d2 = pack(self.w2)
        if recipe == "svdq":
            for name, shape in (
                ("w13_us", (128, 1536, 64)),
                ("w13_vh", (128, 128, 2048)),
                ("w2_us", (128, 2048, 64)),
                ("w2_vh", (128, 64, 768)),
            ):
                self.factors[name] = (
                    torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.02
                )
        self.mainloop = MainloopMoE(
            self.q1,
            self.s1,
            self.d1,
            self.q2,
            self.s2,
            self.d2,
            self.quant_range,
            **self.factors,
        )

    def reference(
        self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        """FP32 sampled-token formula, grouped by expert, including strong R64 factors."""
        selected = torch.linspace(
            0, x.shape[0] - 1, min(x.shape[0], 16), device="cuda"
        ).long()
        values, routes, gates = x[selected].float(), ids[selected], weights[selected]
        result = torch.zeros_like(values)
        for expert in range(128):
            token, slot = torch.where(routes == expert)
            if token.numel() == 0:
                continue
            inp = values[token]
            gu = F.linear(inp, self.w1[expert].float())
            us, vh = self.factors["w13_us"], self.factors["w13_vh"]
            if us is not None and vh is not None:
                gp, up = F.linear(inp, vh[expert].float()).chunk(2, -1)
                gus, uus = us[expert].float().chunk(2, 0)
                gu += torch.cat((F.linear(gp, gus), F.linear(up, uus)), -1)
            gate, up = gu.chunk(2, -1)
            activated = F.silu(gate) * up
            val = F.linear(activated, self.w2[expert].float())
            us2, vh2 = self.factors["w2_us"], self.factors["w2_vh"]
            if us2 is not None and vh2 is not None:
                val += F.linear(
                    F.linear(activated, vh2[expert].float()), us2[expert].float()
                )
            result.index_add_(0, token, val * gates[token, slot, None])
        return result

    def check(self, counts: tuple[int, ...], label: str) -> None:
        """Verify the dispatched layer and optionally time the same callable."""
        from flashinfer.fused_moe import trtllm_bf16_routed_moe
        from vllm_adapter.local_routing import prepare_local, finalize_local
        from vllm_adapter.moe_capacity import bounded_capacity
        from vllm_adapter.moe_kernel import run_4o6_svdq_moe
        from vllm_adapter.moe import make_dynamic_moe_method

        m = sum(counts) // 8
        cap = bounded_capacity(m, 16384)
        torch.manual_seed(9000 + m)
        x = torch.randn(m, 2048, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor(
            routes_from_counts(counts, 8), device="cuda", dtype=torch.int32
        )
        weights = torch.randn(m, 8, device="cuda").softmax(-1)
        observed = torch.bincount(ids.flatten().long(), minlength=128)
        assert tuple(observed.cpu().tolist()) == counts
        if self.recipe == "bf16":
            packed_ids = (ids << 16) | (
                weights.to(torch.bfloat16).view(torch.int16).to(torch.int32) & 65535
            )

            def full() -> torch.Tensor:
                return trtllm_bf16_routed_moe(
                    packed_ids,
                    x,
                    self.b1,
                    self.b2,
                    128,
                    8,
                    None,
                    None,
                    768,
                    0,
                    128,
                    routing_method_type=4,
                )

            operation: Callable[[], torch.Tensor] = full
            path = "flashinfer_trtllm_bf16"
        else:
            workspace = torch.empty(
                128, cap if m <= 128 else 0, 1536, device="cuda", dtype=torch.bfloat16
            )
            out = torch.empty(
                128, cap if m <= 128 else 0, 2048, device="cuda", dtype=torch.bfloat16
            )
            result = torch.empty_like(x)

            def fallback() -> torch.Tensor:
                values, mask = prepare_local(x, weights, ids, 128, cap, False)
                run_4o6_svdq_moe(
                    hidden_states=values,
                    activation_quant_range=self.quant_range,
                    w1=self.q1,
                    w1_blockscale=self.s1,
                    w1_weight_global_scale=self.d1,
                    w2=self.q2,
                    w2_blockscale=self.s2,
                    w2_weight_global_scale=self.d2,
                    masked_m=mask,
                    workspace=workspace,
                    out=out,
                    **self.factors,
                )
                finalize_local(result, out, weights, ids, False)
                return result

            class Previous:
                def apply(self, *args: torch.Tensor) -> torch.Tensor:
                    raise AssertionError(
                        "Captured shape unexpectedly dispatched the legacy fallback"
                    )

            cls = make_dynamic_moe_method(
                Previous,
                Previous,
                self.quant_range,
                64 if self.recipe == "svdq" else None,
            )
            method = cls.__new__(cls)
            method.moe_kernel = SimpleNamespace(
                fused_experts=SimpleNamespace(mainloop=self.mainloop)
            )
            layer = SimpleNamespace(
                expert_map=None,
                apply_router_weight_on_input=False,
                global_num_experts=128,
                activation="silu",
            )

            def full() -> torch.Tensor:
                return method.apply(layer, x, weights, ids, None, None)

            operation = full
            path = "trt_compact_mainloop_fusion"
            if m <= 128:
                expected = fallback().clone()
                actual = full()
                error = float(
                    (actual.float() - expected.float()).square().sum()
                    / expected.float().square().sum()
                )
                assert error < 0.005, (label, "fused/fallback", error)
        actual = operation().clone()
        assert actual.shape == x.shape and torch.isfinite(actual).all()
        ref = self.reference(x, ids, weights)
        selected = torch.linspace(0, m - 1, min(m, 16), device="cuda").long()
        error = float(
            (actual[selected].float() - ref).square().sum() / ref.square().sum()
        )
        assert error < (0.001 if self.recipe == "bf16" else 0.1), (label, error)
        evidence = {
            "case": label,
            "recipe": self.recipe,
            "M": m,
            "expert_max": max(counts),
            "idle_experts": counts.count(0),
            "capacity": cap,
            "path": path,
            "reference_nmse": error,
        }
        print(json.dumps(evidence), flush=True)
        if os.environ.get("B300_BENCHMARK") == "1":
            for timing in (
                measure("full_eager", operation, iterations=10),
                measure_graph("full_cuda_graph", operation, nmse_limit=1.0e-4),
            ):
                print(json.dumps({**evidence, "timing": asdict(timing)}), flush=True)
