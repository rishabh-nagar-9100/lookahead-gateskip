"""P2-04 soft-gated training forward; T-EQ (router disabled -> logits equal the HF model, fp32, atol 1e-5)."""

import math

import pytest
import torch

from lagate.models.gated import LookaheadGatedModel
from lagate.routing.router import RouterConfig

GEN = torch.Generator().manual_seed(0)


def ids(B=2, T=17):
    return torch.randint(0, 128, (B, T), generator=GEN)


def test_t_eq_router_disabled_equals_hf(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig()).eval()
    m.router_enabled = False
    x = ids()
    with torch.no_grad():
        out = m(x)
        ref = tiny_model(x).logits
    torch.testing.assert_close(out.logits, ref, rtol=0, atol=1e-5)
    assert out.g_layer is None and out.sparsity_loss is None


def test_t_eq_router_disabled_with_right_padding(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig()).eval()
    m.router_enabled = False
    x = ids(B=2, T=12)
    mask = torch.ones_like(x)
    mask[1, 8:] = 0
    with torch.no_grad():
        out = m(x, attention_mask=mask)
        ref = tiny_model(x, attention_mask=mask).logits
    torch.testing.assert_close(out.logits[0], ref[0], rtol=0, atol=1e-5)
    torch.testing.assert_close(out.logits[1, :8], ref[1, :8], rtol=0, atol=1e-5)  # real tokens only


def test_saturated_gates_equal_dense(tiny_model):
    """Gates at σ(50) = 1.0 in fp32 give exactly the dense forward."""
    m = LookaheadGatedModel(tiny_model, RouterConfig(bias_init=50.0, head_init_std=0.0)).eval()
    x = ids()
    with torch.no_grad():
        out = m(x)
        ref = tiny_model(x).logits
    torch.testing.assert_close(out.logits, ref, rtol=0, atol=1e-5)
    assert torch.all(out.g_layer == 1.0)


def test_init_gates_open_and_close_to_dense(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig()).eval()
    x = ids()
    with torch.no_grad():
        out = m(x)
        ref = tiny_model(x).logits
    s5 = 1 / (1 + math.exp(-5))
    assert out.g_layer.shape == out.g_mlp.shape == (2, 17, 3)
    assert (out.g_layer - s5).abs().max() < 0.01 and (out.g_mlp - s5).abs().max() < 0.01
    assert (out.logits - ref).abs().max() < 0.5  # dense start, not identical
    assert 0.98 < out.sparsity_loss.item() < 1.0


def test_sparsity_loss_matches_formula(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig(), train_context_length=9.0)
    g_l = torch.full((1, 4, 3), 0.5)
    g_m = torch.full((1, 4, 3), 0.25)
    c_a, c_m = m.costs.sparsity_weights(9.0)
    expect = (c_a * 0.5 + c_m * 0.5 * 0.25) / (c_a + c_m)
    assert m.sparsity_loss(g_l, g_m, None).item() == pytest.approx(expect)
    valid = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    g_l2 = g_l.clone()
    g_l2[:, 2:] = 1.0  # padded positions must not count
    assert m.sparsity_loss(g_l2, g_m, valid).item() == pytest.approx(expect)


def test_gradients_flow_to_router_and_backbone(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig()).train()
    x = ids()
    out = m(x, labels=x, sparsity_weight=0.1)
    out.loss.backward()
    assert all(p.grad is not None for p in m.router_parameters())
    assert tiny_model.model.layers[0].mlp.up_proj.weight.grad is not None
    assert torch.isclose(out.loss, out.ce_loss + 0.1 * out.sparsity_loss)


def test_rejects_left_padding(tiny_model):
    m = LookaheadGatedModel(tiny_model, RouterConfig())
    x = ids(B=1, T=5)
    with pytest.raises(ValueError):
        m(x, attention_mask=torch.tensor([[0, 1, 1, 1, 1]]))
