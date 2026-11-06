"""P2-03 dual-head router: shapes, +5 bias init, dependence on its inputs, config loading."""

import math

import pytest
import torch

from lagate.routing.router import DualHeadRouter, RouterConfig
from lagate.utils import REPO_ROOT, load_config

H = 32


def make_router(**kw):
    torch.manual_seed(0)
    return DualHeadRouter(H, RouterConfig(**kw), rms_eps=1e-6)


def test_shapes_and_param_count():
    r = make_router(bottleneck=8)
    h, mu, attn = torch.randn(2, 5, H), torch.randn(2, 5, H), torch.randn(2, 5, H)
    g_l, z = r.layer_gate(h, mu)
    g_m = r.mlp_gate(z, attn)
    assert g_l.shape == g_m.shape == (2, 5) and z.shape == (2, 5, 8)
    assert ((g_l > 0) & (g_l < 1)).all() and ((g_m > 0) & (g_m < 1)).all()
    # W_1 (2H·b) + w_L (b + 1) + attn RMSNorm (H) + W_a (H·b) + w_M (2b + 1)
    assert sum(p.numel() for p in r.parameters()) == 2 * H * 8 + 9 + H + H * 8 + 17


def test_bias_init_opens_gates():
    r = make_router()  # defaults: bias +5, head std 0.01
    assert torch.all(r.head_l.bias == 5.0) and torch.all(r.head_m.bias == 5.0)
    g_l, z = r.layer_gate(torch.randn(100, H), torch.randn(100, H))
    g_m = r.mlp_gate(z, torch.randn(100, H))
    s5 = 1 / (1 + math.exp(-5))  # 0.9933
    for g in (g_l, g_m):
        assert (g - s5).abs().max() < 5e-3


def test_gates_depend_on_ema_and_attention():
    r = make_router(head_init_std=1.0)
    h = torch.randn(4, H)
    g1, z = r.layer_gate(h, torch.zeros(4, H))
    g2, _ = r.layer_gate(h, torch.randn(4, H))
    assert not torch.allclose(g1, g2)  # μ changes g_L
    assert not torch.allclose(r.mlp_gate(z, torch.randn(4, H)), r.mlp_gate(z, torch.randn(4, H)))  # ΔAttn changes g_M


def test_gradients_reach_all_parameters():
    r = make_router()
    g_l, z = r.layer_gate(torch.randn(3, H), torch.randn(3, H))
    (g_l + r.mlp_gate(z, torch.randn(3, H))).sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())


def test_config_file_loads():
    cfg = RouterConfig.from_dict(load_config(REPO_ROOT / "configs/router/lagate.yaml"))
    assert cfg == RouterConfig()  # file and dataclass defaults agree
    with pytest.raises(ValueError):
        RouterConfig.from_dict({"alhpa": 0.7})
