"""Independent packed-operand formulas for multi-tile TRT fusion."""

import json
import os

import pytest
import torch

from shape_cases.moe import MoELayer
from shape_cases.packed_formula import check


@pytest.fixture(scope="module")
def layer() -> MoELayer:
    with torch.inference_mode():
        return MoELayer(os.environ["RECIPE"])


@pytest.mark.parametrize("tokens", [131, 1024])
def test_multitile_packed_math_and_inactive_row_poison(
    layer: MoELayer, tokens: int
) -> None:
    """Compare actual packed GEMMs, fused corrections and amax, not quantization error."""
    torch.manual_seed(9300 + tokens)
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    # Multiple tiles per hot expert, both chunks active, and many empty experts.
    ids[tokens // 2 :] += 64
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(tokens, 8, device="cuda").softmax(-1)
    assert layer.mainloop is not None
    with torch.inference_mode():
        evidence = check(lambda: layer.mainloop(x, ids, weights))
    print(
        json.dumps({"recipe": layer.recipe, "M": tokens, "packed_formula": evidence}),
        flush=True,
    )


@pytest.mark.parametrize("tokens", [131, 16384])
def test_graph_replay_updates_activations_and_expert_shard(
    layer: MoELayer, tokens: int
) -> None:
    """Replay must not reuse previous routing, global scales, or reduction tickets."""
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    weights = torch.full((tokens, 8), 0.125, device="cuda")
    assert layer.mainloop is not None
    with torch.inference_mode():
        layer.mainloop(x, ids, weights)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = layer.mainloop(x, ids, weights)
        for iteration in range(3):
            x.mul_(0.7)
            ids.copy_((ids + 64) % 128)
            graph.replay()
            expected = layer.mainloop(x, ids, weights)
            error = (
                actual.float() - expected.float()
            ).square().sum() / expected.float().square().sum()
            assert error < 1.0e-4, (iteration, float(error))
