# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRT-LLM FC13/FC2 mainloop fusion over token-major local MoE inputs."""
from __future__ import annotations

from dataclasses import dataclass
import os

import torch

from .mainloop_routing import prepare, compact_scale


def _swizzle_indices(rows: int, columns: int, device: torch.device) -> torch.Tensor:
    """Map linear scale coordinates to NVIDIA's 128-by-4 scale layout."""
    r = torch.arange(rows, device=device)[:, None]
    c = torch.arange(columns, device=device)[None, :]
    return r // 128 * (columns // 4) * 512 + c // 4 * 512 + r % 32 * 16 + r % 128 // 32 * 4 + c % 4


def _interleave(values: torch.Tensor) -> torch.Tensor:
    """Convert vLLM [gate | up] to donor 64-column [up, gate] blocks."""
    experts, width, inner = values.shape
    gate, up = values.chunk(2, 1)
    return torch.stack((up.reshape(experts, width // 128, 64, inner),
                        gate.reshape(experts, width // 128, 64, inner)), 2).reshape_as(values).contiguous()


@dataclass(frozen=True, slots=True)
class Chunk:
    w13: torch.Tensor
    sf13: torch.Tensor
    alpha13: torch.Tensor
    w2: torch.Tensor
    sf2: torch.Tensor
    alpha2: torch.Tensor
    vh13: torch.Tensor | None
    up_us: torch.Tensor | None
    gate_us: torch.Tensor | None
    vh2: torch.Tensor | None
    us2: torch.Tensor | None


class MainloopMoE:
    """Prepare immutable weight layouts once, then execute compact fused GEMMs.

    Scope is local unique top-k routing, BF16 input, M<=16384, and rank 64.
    FC2 uses the donor BF16 scatter-add, whose reduction order is nondeterministic.
    """

    def __init__(self, w1: torch.Tensor, s1: torch.Tensor, d1: torch.Tensor,
                 w2: torch.Tensor, s2: torch.Tensor, d2: torch.Tensor,
                 quant_range: float, w13_us: torch.Tensor | None = None,
                 w13_vh: torch.Tensor | None = None, w2_us: torch.Tensor | None = None,
                 w2_vh: torch.Tensor | None = None) -> None:
        experts, width, packed_hidden = w1.shape
        if width % 128 or packed_hidden % 32 or experts % 64:
            raise RuntimeError("Mainloop MoE requires aligned weights and 64-expert chunks.")
        factors = (w13_us, w13_vh, w2_us, w2_vh)
        if any(v is not None for v in factors) and not all(v is not None for v in factors):
            raise RuntimeError("Mainloop SVDQ requires both FC13 and FC2 factor pairs.")
        from trt_quant.dense import load_library
        load_library()
        self.scale_rule = int(quant_range == 1536.0)
        self.external_amax = os.environ.get("VLLM_MAINLOOP_AMAX_EXTERNAL") == "1"
        self.quant_range = quant_range
        self.experts = experts
        self.hidden = packed_hidden * 2
        self.chunks: list[Chunk] = []
        index = _swizzle_indices(width, self.hidden // 16, w1.device)
        linear_sf = s1.view(torch.uint8).reshape(experts, -1)[:, index]
        interleaved_sf = _interleave(linear_sf)
        packed_sf = torch.empty_like(s1, dtype=torch.uint8)
        packed_sf.reshape(experts, -1)[:, index] = interleaved_sf
        packed_w1 = _interleave(w1.view(torch.uint8)).view(torch.float4_e2m1fn_x2)
        packed_vh = sparse_up = sparse_gate = None
        if w13_us is not None and w13_vh is not None:
            if w13_us.shape[-1] != 64 or w13_vh.shape[1] != 128:
                raise RuntimeError("Mainloop fusion requires rank 64 per FC13 branch.")
            gate_vh, up_vh = w13_vh.chunk(2, 1)
            packed_vh = torch.cat((up_vh, gate_vh), 1).to(torch.bfloat16).contiguous()
            gate, up = w13_us.to(torch.bfloat16).chunk(2, 1)
            up = up.reshape(experts, width // 128, 64, 64)
            gate = gate.reshape_as(up)
            zeros = torch.zeros_like(up)
            sparse_up = torch.stack((up, zeros), 2).reshape(experts, width, 64).contiguous()
            sparse_gate = torch.stack((zeros, gate), 2).reshape(experts, width, 64).contiguous()
        for start in range(0, experts, 64):
            end = start + 64
            self.chunks.append(Chunk(
                packed_w1[start:end], packed_sf[start:end], d1[start:end].contiguous(),
                w2[start:end].view(torch.float4_e2m1fn_x2), s2[start:end].view(torch.uint8), d2[start:end].contiguous(),
                None if packed_vh is None else packed_vh[start:end],
                None if sparse_up is None else sparse_up[start:end],
                None if sparse_gate is None else sparse_gate[start:end],
                None if w2_vh is None else w2_vh[start:end].to(torch.bfloat16).contiguous(),
                None if w2_us is None else w2_us[start:end].to(torch.bfloat16).contiguous(),
            ))

    def __call__(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Fuse correction/SwiGLU/amax in FC13 and correction/scatter-add in FC2."""
        from trt_mainloop.activation import ActivationType
        from trt_mainloop.runners import (
            Sm100Bf16ContiguousGroupedGemmRunner,
            Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
        )
        if not (x.ndim == 2 and 0 < x.shape[0] <= 16384 and x.shape[1] == self.hidden
                and ids.ndim == 2 and ids.shape[0] == x.shape[0]
                and ids.shape[1] in (1, 2, 4, 8) and weights.shape == ids.shape
                and x.dtype == torch.bfloat16 and x.is_contiguous()
                and ids.is_contiguous() and weights.is_contiguous() and ids.dtype == torch.int32
                and weights.dtype == torch.float32):
            raise RuntimeError("Mainloop MoE expects contiguous BF16 M<=16384 and int32/FP32 routes.")
        topk = ids.shape[1]
        output = torch.zeros_like(x)
        device_index = x.device.index
        lowrank = Sm100Bf16ContiguousGroupedGemmRunner.get_cached(128, device_index)
        for chunk_index, chunk in enumerate(self.chunks):
            route = prepare(x, ids, weights, chunk_index * 64, 64, self.experts, self.quant_range)
            rows = route.permutation.numel()
            aq, sf = torch.ops.trtllm.fp4_quantize_ex(
                x, route.inverse_scale, 16, False, False, 1, self.scale_rule)
            sf = sf.view(x.shape[0], self.hidden // 16)
            down13 = None
            if chunk.vh13 is not None:
                down13 = torch.empty((rows, 128), device=x.device, dtype=torch.bfloat16)
                lowrank.forward(x, chunk.vh13, down13, route.groups, route.live,
                                tile_idx_to_mn_limit=route.limits,
                                permuted_idx_to_expanded_idx=route.permutation, top_k=topk,
                                correction_alpha=chunk.alpha13,
                                correction_alpha_numerator=route.one,
                                correction_alpha_denominator=route.inverse_scale)
            fc13 = Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner.get_cached(
                device_index, 64, topk, 64, 0, 128, 16, ActivationType.Swiglu,
                torch.bfloat16, not self.external_amax, self.quant_range, down13 is not None)
            operands = [aq.view(torch.float4_e2m1fn_x2), chunk.w13, sf.view(torch.uint8), chunk.sf13,
                        chunk.alpha13, route.groups, route.limits, route.permutation, route.live,
                        route.one, route.one, route.inverse_scale]
            if down13 is not None:
                operands.extend((down13, chunk.up_us, chunk.gate_us))
            fc13_result = fc13.forward(operands, tactic=((128, 128), (1, 1), False))
            activated = fc13_result[0]
            amax_scale = (compact_scale(activated, route, self.quant_range)
                          if self.external_amax else fc13_result[2])
            inverse2 = amax_scale[1:2]
            q2, sf2 = torch.ops.trtllm.fp4_quantize_phase2(
                activated, amax_scale, 16, True, self.scale_rule)
            down2 = None
            if chunk.vh2 is not None:
                assert down13 is not None
                down2 = down13.view(-1)[:rows * 64].view(rows, 64)
                lowrank.forward(activated, chunk.vh2, down2, route.groups, route.live,
                                correction_alpha=chunk.alpha2,
                                correction_alpha_numerator=route.one,
                                correction_alpha_denominator=inverse2)
            fc2 = Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner.get_cached(
                device_index, 64, topk, 64, 0, 128, torch.bfloat16, 16, False, down2 is not None)
            fc2.forward([q2.view(torch.float4_e2m1fn_x2), chunk.w2, sf2.view(torch.uint8).flatten(),
                         chunk.sf2, chunk.alpha2, output, route.groups, route.limits, route.permutation,
                         route.live, weights, route.one, inverse2, None, down2, chunk.us2],
                        tactic=((128, 128), (1, 1), False))
        return output
