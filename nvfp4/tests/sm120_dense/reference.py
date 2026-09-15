"""Independent packed-FP4 math and changing-data graph replay checks."""

from typing import TypedDict

import torch

from .backend import DenseLayer, Recipe


class Evidence(TypedDict):
    reference_nmse: float
    packed_nmse: float
    sampled_rows: int
    sampled_columns: int


def nmse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Measure error in FP32 with a finite denominator for zero inputs."""
    return float(
        (actual.float() - reference.float()).square().sum()
        / reference.float().square().sum().clamp_min(1e-20)
    )


def decode_rows(
    codes: torch.Tensor, scales: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Decode E2M1 and the 128x4 SF swizzle independently of the GEMM backend."""
    half_k = codes.shape[1]
    selected = codes[rows].long()
    unpacked = torch.stack((selected & 15, selected >> 4), -1).reshape(
        rows.numel(), 2 * half_k
    )
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=codes.device,
    )
    r = rows[:, None]
    c = torch.arange(half_k // 8, device=codes.device)[None, :]
    indices = (
        r // 128 * (half_k // 8) * 128
        + c // 4 * 512
        + r % 32 * 16
        + r % 128 // 32 * 4
        + c % 4
    )
    factors = (
        scales.view(torch.uint8)
        .flatten()[indices]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    return table[unpacked] * factors.repeat_interleave(16, 1)


def check_output(layer: DenseLayer, values: torch.Tensor) -> Evidence:
    """Check first, middle and last token rows, including tile tails."""
    actual = layer(values)
    assert actual.shape == (values.shape[0], layer.weight.shape[0])
    assert torch.isfinite(actual).all()
    m, n = actual.shape
    rows = torch.tensor(
        sorted(set(range(min(m, 16))) | {m // 2, m - 1}), device=values.device
    )
    columns = torch.arange(0, n, max(1, n // 256), device=values.device)
    x = values[rows].float()
    reference = x @ layer.weight[columns].float().T
    got = actual[rows][:, columns].float()
    if layer.recipe == Recipe.SVDQ:
        assert layer.vh is not None and layer.us is not None
        down = (x @ layer.vh.float().T).to(torch.bfloat16)
        correction = (
            (down.float() @ layer.us[columns].float().T).to(torch.bfloat16).float()
        )
        reference += correction
    reference_error = nmse(got, reference)
    assert reference_error < (2e-5 if layer.recipe == Recipe.BF16 else 0.05), (
        reference_error
    )
    packed_error = 0.0
    if layer.recipe != Recipe.BF16:
        assert layer.weight_codes is not None and layer.weight_scales is not None
        packed = layer.pack(values)
        aq = decode_rows(packed.codes, packed.scales, rows)
        wq = decode_rows(layer.weight_codes, layer.weight_scales, columns)
        expected = (aq @ wq.T * packed.alpha).to(torch.bfloat16)
        if layer.recipe == Recipe.SVDQ:
            expected = (expected.float() + correction).to(torch.bfloat16)
        packed_error = nmse(got, expected)
        assert packed_error < 2e-5, packed_error
    return {
        "reference_nmse": reference_error,
        "packed_nmse": packed_error,
        "sampled_rows": rows.numel(),
        "sampled_columns": columns.numel(),
    }


def check_replay(layer: DenseLayer, values: torch.Tensor) -> None:
    """A graph must recompute scales after its input data and amax change."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(values)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = layer(values)
    for zero in (False, True):
        if zero:
            values.zero_()
        else:
            values.mul_(0.125)
            values[-1, -1] = 9.0
        expected = layer(values).clone()
        graph.replay()
        torch.cuda.synchronize()
        assert nmse(output, expected) < 2e-6
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.manual_seed(9000 + values.shape[0])
    values.normal_()
