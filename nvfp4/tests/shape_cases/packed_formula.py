"""Independent FP32 formulas over captured packed inputs; untimed validation only."""

import torch
from trt_mainloop.runners import (
    Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner as FC13,
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner as FC2,
)


def decode(q, sf, swizzled):
    rows, half_k = q.shape
    codes = q.view(torch.uint8).long()
    unpacked = torch.stack((codes & 15, codes >> 4), -1).reshape(rows, 2 * half_k)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device=q.device
    )
    if swizzled:
        r = torch.arange(rows, device=q.device)[:, None]
        c = torch.arange(half_k // 8, device=q.device)[None, :]
        index = (
            r // 128 * (half_k // 8) * 128
            + c // 4 * 512
            + r % 32 * 16
            + r % 128 // 32 * 4
            + c % 4
        )
        sf = sf.view(torch.uint8).flatten()[index]
    scales = (
        sf.contiguous().view(torch.float8_e4m3fn).float().reshape(rows, half_k // 8)
    )
    return table[unpacked] * scales.repeat_interleave(16, 1)


def nmse(a, b):
    return float(
        (a.float() - b.float()).square().sum()
        / b.float().square().sum().clamp_min(1.0e-20)
    )


def check(operation):
    from trt_mainloop.runners import Sm100Bf16ContiguousGroupedGemmRunner as Down
    import vllm_adapter.mainloop as mainloop

    old_scale = mainloop.compact_scale
    old_down = Down.forward
    old13, old2 = FC13.forward, FC2.forward
    evidence = []
    last_limits = None

    def check13(self, inputs, tactic):
        nonlocal last_limits
        last_limits = inputs[6]
        result = old13(self, inputs, tactic)
        (
            a,
            b,
            sa,
            sb,
            alpha,
            groups,
            limits,
            perm,
            live,
            global_sf,
            num,
            den,
            *factors,
        ) = inputs
        deqa = decode(a, sa, False)
        expected = []
        actual = []
        for tile in range(int(live)):
            expert = int(groups[tile])
            begin = tile * 128
            end = int(limits[tile])
            tokens = perm[begin:end].long() // self.top_k
            accum = deqa[tokens] @ decode(b[expert], sb[expert], True).T
            if factors:
                down, up_us, gate_us = factors
                accum += down[begin:end, :64].float() @ up_us[expert].float().T
                accum += down[begin:end, 64:].float() @ gate_us[expert].float().T
            accum *= alpha[expert] * num / den
            pairs = accum.reshape(end - begin, -1, 2, 64)
            up = pairs[:, :, 0, :].reshape(end - begin, -1)
            gate = pairs[:, :, 1, :].reshape_as(up)
            ref = (torch.nn.functional.silu(gate) * up).to(torch.bfloat16)
            expected.append(ref)
            actual.append(result[0][begin:end])
        if expected:
            ref = torch.cat(expected)
            got = torch.cat(actual)
            error = nmse(got, ref)
            assert error < 2.0e-5, ("FC13 formula", error)
            amax = float(got.float().abs().max())
            if len(result) == 3 and float(result[2][0]) != amax:
                for cache_key, (
                    block_max,
                    tickets,
                    scale,
                ) in self.amax_workspace_cache.items():
                    print(
                        {
                            "amax_workspace": str(cache_key),
                            "partial_max": float(block_max.max()),
                            "tickets": int(tickets),
                            "scale": scale.tolist(),
                            "expected_amax": amax,
                        },
                        flush=True,
                    )
            assert len(result) == 2 or float(result[2][0]) == amax, (
                "live amax",
                float(result[2][0]),
                amax,
            )
            evidence.append(
                {
                    "stage": "fc13",
                    "nmse": error,
                    "amax_exact": True,
                    "rows": ref.shape[0],
                }
            )
        return result

    def check2(self, inputs, tactic):
        (
            a,
            b,
            sa,
            sb,
            alpha,
            out,
            groups,
            limits,
            perm,
            live,
            weights,
            num,
            den,
            correction,
            down,
            us,
        ) = inputs
        before = out.clone().float()
        result = old2(self, inputs, tactic)
        ref = before
        deqa = decode(a, sa, True)
        for tile in range(int(live)):
            expert = int(groups[tile])
            begin = tile * 128
            end = int(limits[tile])
            expanded = perm[begin:end].long()
            accum = deqa[begin:end] @ decode(b[expert], sb[expert], True).T
            if down is not None:
                accum += down[begin:end].float() @ us[expert].float().T
            accum *= alpha[expert] * num / den
            weighted = (
                (accum * weights.flatten()[expanded, None]).to(torch.bfloat16).float()
            )
            ref.index_add_(0, expanded // self.top_k, weighted)
        error = nmse(result, ref)
        assert error < 1.0e-4, ("FC2 formula", error)
        evidence.append({"stage": "fc2", "nmse": error, "live_tiles": int(live)})
        return result

    def check_down(self, a, b, c, groups, live, **kwargs):
        old_down(self, a, b, c, groups, live, **kwargs)
        den = kwargs["correction_alpha_denominator"]
        num = kwargs["correction_alpha_numerator"]
        alpha = kwargs["correction_alpha"]
        for tile in range(int(live)):
            expert = int(groups[tile])
            begin = tile * 128
            if "permuted_idx_to_expanded_idx" in kwargs:
                end = int(kwargs["tile_idx_to_mn_limit"][tile])
                idx = (
                    kwargs["permuted_idx_to_expanded_idx"][begin:end].long()
                    // kwargs["top_k"]
                )
                values = a[idx]
            else:
                end = int(last_limits[tile])
                values = a[begin:end]
            ref = (
                (values.float() @ b[expert].float().T) * (den / num / alpha[expert])
            ).to(torch.bfloat16)
            error = nmse(c[begin:end], ref)
            evidence.append(
                {
                    "stage": "down",
                    "nmse": error,
                    "tile": tile,
                    "gather": "permuted_idx_to_expanded_idx" in kwargs,
                }
            )
            assert error < 2.0e-5, ("down normalization/gather", error)
        limits = kwargs.get("tile_idx_to_mn_limit", last_limits)
        row = torch.arange(c.shape[0], device=c.device)
        tile = row // 128
        valid = (tile < int(live)) & (row < limits[tile])
        c[~valid] = float("nan")
        evidence.append({"stage": "poison_inactive_down_rows", "live_tiles": int(live)})

    def check_scale(values, route, quant_range):
        result = old_scale(values, route, quant_range)
        expected = max(
            float(values[t * 128 : int(route.limits[t])].float().abs().max())
            for t in range(int(route.live))
        )
        assert float(result[0]) == expected, (
            "independent compact amax",
            float(result[0]),
            expected,
        )
        evidence.append({"stage": "compact_amax", "exact": True, "amax": expected})
        return result

    FC13.forward, FC2.forward, Down.forward = check13, check2, check_down
    mainloop.compact_scale = check_scale
    try:
        operation()
    finally:
        FC13.forward, FC2.forward, Down.forward = old13, old2, old_down
        mainloop.compact_scale = old_scale
    return evidence
