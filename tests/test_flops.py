"""T-FLOP: the FLOP counter equals a hand calculation for one layer and one token, and equals the
matmul FLOPs PyTorch measures when the adapter actually runs that token."""

import torch
from torch.utils.flop_counter import FlopCounterMode

from lagate.engine.flops import LayerCosts, avg_context_length
from lagate.models.adapter import DecoderAdapter, Dims
from lagate.routing.router import DualHeadRouter, RouterConfig

LLAMA_1B = Dims(num_layers=16, hidden=2048, intermediate=8192, n_heads=32, n_kv_heads=8, head_dim=64, vocab=128256)


def test_t_flop_hand_calculation_tiny():
    # H=64, I=128, 4 heads x 16, 2 KV heads, bottleneck 8, context n=10.
    c = LayerCosts(Dims(3, 64, 128, 4, 2, 16, 128), bottleneck=8)
    qo, kv, attn, mlp = 2 * 64 * 64, 2 * 64 * 32, 2 * 10 * 64, 3 * 64 * 128  # 8192, 4096, 1280, 24576
    rl, rm = 2 * 64 * 8 + 8, 64 * 8 + 16                                        # 1032, 528
    assert c.dense(10) == qo + kv + attn + mlp == 38144
    assert c.tier(1, 10) == 38144 + rl + rm == 39704
    assert c.tier(2, 10) == qo + kv + attn + rl + rm == 15128
    assert c.tier(3, 10) == kv + rl == 5128
    assert c.sparsity_weights(10) == (qo + attn, mlp)


def test_t_flop_llama_1b_matches_spec_estimate():
    """ARCHITECTURE §3: router ≈ 0.39M MACs vs ≈ 60M for the layer's linear maps (≈ 0.65%)."""
    c = LayerCosts(LLAMA_1B, bottleneck=64)
    linear = c.qo + c.kv + c.mlp
    assert linear == 8_388_608 + 2_097_152 + 50_331_648  # 60.8M
    assert abs(c.router - 393_216) < 200                  # 2H·64 + H·64 plus the two heads
    assert 0.0064 < c.router / linear < 0.0066


def test_t_flop_matches_measured_matmuls(tiny_model):
    """Run one new token at context n through the adapter and count matmul FLOPs with PyTorch."""
    ad = DecoderAdapter(tiny_model)
    router = DualHeadRouter(ad.dims.hidden, RouterConfig(bottleneck=8), rms_eps=1e-6)
    costs = LayerCosts(ad.dims, bottleneck=8)
    n, layer = 10, 1
    h_ctx = torch.randn(1, n, ad.dims.hidden)
    pos = torch.arange(n)
    with torch.no_grad():
        cos, sin = ad.rope(h_ctx, pos)
        k_all, v_all = ad.kv(layer, ad.input_norm(layer, h_ctx), cos, sin)
        h = h_ctx[:, -1:]
        c1, s1 = cos[:, -1:], sin[:, -1:]
        with FlopCounterMode(display=False) as fc:
            nrm = ad.input_norm(layer, h)
            ad.kv(layer, nrm, c1, s1)
            g_l, z = router.layer_gate(nrm, torch.zeros_like(nrm))
        tier3 = fc.get_total_flops()
        with FlopCounterMode(display=False) as fc:
            attn = ad.attend(layer, ad.q(layer, nrm, c1, s1), k_all, v_all, pos[-1:], pos)
            router.mlp_gate(z, attn)
            ad.mlp(layer, ad.post_norm(layer, h + attn))
        rest = fc.get_total_flops()
    assert tier3 == 2 * costs.tier(3, n)
    assert tier3 + rest == 2 * costs.tier(1, n)


def test_avg_context_length():
    assert avg_context_length(1) == 1 and avg_context_length(1024) == 512.5
