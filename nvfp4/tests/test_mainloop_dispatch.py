"""Check the vLLM token-major entry and preserved fallback contract."""

from types import SimpleNamespace
import pytest
import torch
from vllm_adapter.moe import make_dynamic_moe_method


class Previous:
    def apply(self, layer, x, weights, ids, shared_experts, shared_input):
        self.used_previous = True
        return x + 7


class Fused:
    experts = 128

    def __call__(self, x, ids, weights):
        assert ids.dtype == torch.int32 and weights.dtype == torch.float32
        return x + 3


@pytest.mark.parametrize(
    "tokens", [1, 16, 61, 127, 128, 129, 201, 512, 1024, 4096, 16384]
)
def test_mainloop_covers_captured_token_counts(tokens):
    cls = make_dynamic_moe_method(Previous, Previous, 1536.0, 64)
    method = object.__new__(cls)
    method.used_previous = False
    method.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(mainloop=Fused()))
    layer = SimpleNamespace(
        expert_map=None,
        apply_router_weight_on_input=False,
        global_num_experts=128,
        activation="silu",
    )
    x = torch.ones(tokens, 2048, dtype=torch.bfloat16)
    ids = torch.zeros(tokens, 8, dtype=torch.int64)
    weights = torch.ones(tokens, 8, dtype=torch.float32) / 8
    got = method.apply(layer, x, weights, ids, None, None)
    torch.testing.assert_close(got, x + 3)
    assert not method.used_previous


@pytest.mark.parametrize("reason", ["weighted_input", "expert_map", "activation"])
def test_unsupported_contract_preserves_previous_path(reason):
    cls = make_dynamic_moe_method(Previous, Previous, 1536.0, 64)
    method = object.__new__(cls)
    method.used_previous = False
    method.moe_kernel = SimpleNamespace(fused_experts=SimpleNamespace(mainloop=Fused()))
    layer = SimpleNamespace(
        expert_map=None,
        apply_router_weight_on_input=False,
        global_num_experts=128,
        activation="silu",
    )
    if reason == "weighted_input":
        layer.apply_router_weight_on_input = True
    if reason == "expert_map":
        layer.expert_map = torch.arange(128)
    if reason == "activation":
        layer.activation = "relu2"
    x = torch.ones(1, 2048, dtype=torch.bfloat16)
    got = method.apply(
        layer, x, torch.ones(1, 8) / 8, torch.zeros(1, 8, dtype=torch.int32), None, None
    )
    torch.testing.assert_close(got, x + 7)
    assert method.used_previous
