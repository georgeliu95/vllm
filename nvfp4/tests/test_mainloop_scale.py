"""Global input amax must exclude tokens outside each local expert shard."""

import pytest
import torch

from vllm_adapter.mainloop_routing import prepare


@pytest.mark.parametrize("tokens", [129, 1024, 16384])
def test_large_input_scale_masks_unrouted_outliers(tokens: int) -> None:
    torch.manual_seed(tokens)
    x = torch.randn(tokens, 2048, device="cuda", dtype=torch.bfloat16)
    x[-1] = 1000
    ids = torch.arange(8, device="cuda", dtype=torch.int32).repeat(tokens, 1)
    ids[-1] += 64
    weights = torch.full((tokens, 8), 0.125, device="cuda")
    for start, selected in ((0, x[:-1]), (64, x[-1:])):
        route = prepare(x, ids, weights, start, 64, 128, 1536.0)
        expected = 1.0 / (selected.float().abs().max() * (1.0 / 1536.0))
        torch.testing.assert_close(
            route.inverse_scale.squeeze(), expected, rtol=1.0e-6, atol=0
        )
