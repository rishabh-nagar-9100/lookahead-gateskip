"""P2-05 three-tier engine. T-KV: cached decode = uncached recompute under fixed tiers.
T-T3: Tier-3 cache entries equal a direct K,V projection with RoPE at the right position.

T-KV and T-T3 run in float64 so they can use a tolerance (1e-9) far below fp32 rounding: running
row subsets changes matmul shapes, which moves fp32 results by ~1e-5 even when the logic is exact.
The dense-equals-HF checks stay in fp32 at 1e-5 (T-EQ)."""

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from lagate.engine.tiered import DensePolicy, EngineConfig, FixedTierPolicy, TieredEngine
from lagate.models.gated import LookaheadGatedModel
from lagate.routing.router import RouterConfig

P, N_NEW = 9, 7  # prompt length, decoded tokens


def random_tiers(T, L, seed=0):
    return torch.randint(1, 4, (T, L), generator=torch.Generator().manual_seed(seed))


def seq(T, seed=0):
    return torch.randint(0, 128, (1, T), generator=torch.Generator().manual_seed(seed))


def test_dense_engine_equals_hf_prefill_and_generate(tiny_model):
    eng = TieredEngine(tiny_model, DensePolicy())
    x = seq(P)
    with torch.no_grad():
        ref = tiny_model(x).logits
    torch.testing.assert_close(eng.forward_chunk(x, eng.new_cache()), ref, rtol=0, atol=1e-5)
    gen, cache = eng.generate(x, N_NEW)
    hf = tiny_model.generate(x, max_new_tokens=N_NEW, do_sample=False, min_new_tokens=N_NEW)
    assert torch.equal(gen, hf[:, P:])
    assert cache.length == P + N_NEW - 1  # the last generated token is not fed back


TOL64 = dict(rtol=0, atol=1e-9)


def test_t_kv_cached_decode_equals_uncached_recompute(tiny_model):
    tiny_model.double()
    L = tiny_model.config.num_hidden_layers
    x = seq(P + N_NEW, seed=1)
    tiers = random_tiers(P + N_NEW, L, seed=2)
    assert set(tiers.unique().tolist()) == {1, 2, 3}

    eng = TieredEngine(tiny_model, FixedTierPolicy(tiers))
    cache = eng.new_cache()
    cached = [eng.forward_chunk(x[:, :P], cache)]
    for t in range(P, P + N_NEW):
        cached.append(eng.forward_chunk(x[:, t : t + 1], cache, phase="decode"))
    cached = torch.cat(cached, dim=1)

    full = TieredEngine(tiny_model, FixedTierPolicy(tiers)).forward_chunk(x, eng.new_cache())
    torch.testing.assert_close(cached, full, **TOL64)

    # Different tier decisions change the output, so the test is not vacuous.
    other = TieredEngine(tiny_model, FixedTierPolicy(random_tiers(P + N_NEW, L, seed=3))).forward_chunk(x, eng.new_cache())
    assert (other - full).abs().max() > 1e-3


def test_t_kv_lagate_policy_cached_equals_recompute(tiny_model):
    """Same check with the real router deciding (EMA state carried in the cache)."""
    tiny_model.double()
    gated = LookaheadGatedModel(tiny_model, RouterConfig(head_init_std=3.0, bias_init=0.0))
    eng = TieredEngine.from_gated_model(gated, tau_layer=0.5, tau_mlp=0.5)
    x = seq(P + N_NEW, seed=4)
    cache = eng.new_cache()
    cached = [eng.forward_chunk(x[:, :P], cache)]
    for t in range(P, P + N_NEW):
        cached.append(eng.forward_chunk(x[:, t : t + 1], cache, phase="decode"))
    counts = dict(eng.stats.tier_counts)
    full = TieredEngine.from_gated_model(gated, 0.5, 0.5).forward_chunk(x, eng.new_cache())
    torch.testing.assert_close(torch.cat(cached, 1), full, **TOL64)
    assert counts[1] and counts[2] and counts[3]  # all three tiers were exercised


def test_t_t3_tier3_cache_entries_are_exact(tiny_model):
    ad_model = tiny_model.double()
    L = ad_model.config.num_hidden_layers
    T = P + N_NEW
    x = seq(T, seed=5)
    tiers = random_tiers(T, L, seed=6)
    eng = TieredEngine(ad_model, FixedTierPolicy(tiers))
    eng.record_hidden = True
    h_in = {}
    eng.layer_hooks.append(lambda r: h_in.setdefault(r["layer"], []).append(r["h_in"]))
    cache = eng.new_cache()
    eng.forward_chunk(x[:, :P], cache)
    for t in range(P, T):
        eng.forward_chunk(x[:, t : t + 1], cache, phase="decode")
    h_in = {layer: torch.cat(v, 0) for layer, v in h_in.items()}  # (T, H) per layer

    rotary = ad_model.model.rotary_emb
    n_checked = 0
    for layer in range(L):
        attn = ad_model.model.layers[layer].self_attn
        norm = ad_model.model.layers[layer].input_layernorm
        for t in (tiers[:, layer] == 3).nonzero().flatten().tolist():
            h = h_in[layer][t].view(1, 1, -1)
            cos, sin = rotary(h, position_ids=torch.tensor([[t]]))
            n = norm(h)
            k = attn.k_proj(n).view(1, 1, -1, attn.head_dim).transpose(1, 2)
            v = attn.v_proj(n).view(1, 1, -1, attn.head_dim).transpose(1, 2)
            k, _ = apply_rotary_pos_emb(k, k, cos, sin)  # HF's own RoPE, independent of the adapter
            torch.testing.assert_close(cache.keys[layer][:, :, t : t + 1], k, **TOL64)
            torch.testing.assert_close(cache.values[layer][:, :, t : t + 1], v, **TOL64)
            if layer + 1 < L:  # Tier 3 leaves the residual unchanged
                torch.testing.assert_close(h_in[layer + 1][t], h_in[layer][t], rtol=0, atol=0)
            if layer > 0:  # and the entry is not the layer below's entry
                assert not torch.allclose(cache.keys[layer][:, :, t], cache.keys[layer - 1][:, :, t])
            n_checked += 1
    assert n_checked >= 5


def test_skipped_work_is_not_executed(tiny_model):
    """All-Tier-3 tokens never call Q, O or the MLP; FLOP stats match the counter."""
    L = tiny_model.config.num_hidden_layers
    tiers = torch.full((P, L), 3)
    tiers[0] = 1  # the first token is dense
    calls = {"q": 0, "mlp": 0}
    hooks = [tiny_model.model.layers[i].self_attn.q_proj.register_forward_hook(lambda *a: calls.__setitem__("q", calls["q"] + 1)) for i in range(L)]
    hooks += [tiny_model.model.layers[i].mlp.register_forward_hook(lambda *a: calls.__setitem__("mlp", calls["mlp"] + 1)) for i in range(L)]
    eng = TieredEngine(tiny_model, FixedTierPolicy(tiers))
    eng.forward_chunk(seq(P), eng.new_cache())
    for hk in hooks:
        hk.remove()
    assert calls == {"q": L, "mlp": L}  # one call per layer, for the single dense row
    c = eng.costs
    expect = L * (c.tier(1, 1) + sum(c.tier(3, n) for n in range(2, P + 1)))
    assert eng.stats.macs == expect
    assert eng.stats.tier_counts == {1: L, 2: 0, 3: (P - 1) * L, 0: 0}


def test_gate_scaling_option(tiny_model):
    gated = LookaheadGatedModel(tiny_model, RouterConfig())  # gates ≈ 0.993, all tier 1 at τ = 0
    x = seq(P)
    scaled = TieredEngine.from_gated_model(gated, 0.0, 0.0, EngineConfig(gate_scaling=True)).forward_chunk(x, TieredEngine(tiny_model, DensePolicy()).new_cache())
    with torch.no_grad():
        soft = gated(x).logits
    torch.testing.assert_close(scaled, soft, rtol=0, atol=1e-5)  # scaled engine = soft training forward
    plain = TieredEngine.from_gated_model(gated, 0.0, 0.0, EngineConfig(gate_scaling=False)).forward_chunk(x, TieredEngine(tiny_model, DensePolicy()).new_cache())
    with torch.no_grad():
        torch.testing.assert_close(plain, tiny_model(x).logits, rtol=0, atol=1e-5)  # unscaled = dense


def test_engine_config_file_loads():
    from lagate.utils import REPO_ROOT, load_config

    assert EngineConfig.from_dict(load_config(REPO_ROOT / "configs/engine/lagate.yaml")) == EngineConfig()


# ----- P5-05 fast paths ---------------------------------------------------------------------------
def test_cache_growth_beyond_capacity_matches_recompute(tiny_model):
    """Decode past the initial buffer (64 slots): growth keeps every entry (fp64, T-KV tolerance)."""
    tiny_model.double()
    T = 100
    x = seq(T, seed=8)
    tiers = random_tiers(T, tiny_model.config.num_hidden_layers, seed=9)
    eng = TieredEngine(tiny_model, FixedTierPolicy(tiers))
    cache = eng.new_cache()  # no capacity hint -> starts at 64, must grow
    outs = [eng.forward_chunk(x[:, :5], cache)]
    for t in range(5, T):
        outs.append(eng.forward_chunk(x[:, t : t + 1], cache, phase="decode"))
    full = TieredEngine(tiny_model, FixedTierPolicy(tiers)).forward_chunk(x, eng.new_cache())
    torch.testing.assert_close(torch.cat(outs, 1), full, **TOL64)
    assert cache.keys[0].shape[2] == T and cache._k[0].shape[2] >= T


def test_dense_fast_path_equals_general_path(tiny_model):
    L = tiny_model.config.num_hidden_layers
    x = seq(P + N_NEW, seed=10)
    fast = TieredEngine(tiny_model, DensePolicy())
    slow = TieredEngine(tiny_model, FixedTierPolicy(torch.ones(P + N_NEW, L, dtype=torch.long)))
    outs = {}
    for name, eng in (("fast", fast), ("slow", slow)):
        cache = eng.new_cache()
        o = [eng.forward_chunk(x[:, :P], cache)]
        for t in range(P, P + N_NEW):
            o.append(eng.forward_chunk(x[:, t : t + 1], cache, phase="decode"))
        outs[name] = torch.cat(o, 1)
    torch.testing.assert_close(outs["fast"], outs["slow"], rtol=0, atol=1e-5)
    assert fast.stats.tier_counts == slow.stats.tier_counts == {1: (P + N_NEW) * L, 2: 0, 3: 0, 0: 0}
    router = slow.costs.router * (P + N_NEW) * L
    assert slow.stats.macs - fast.stats.macs == router and fast.stats.compute_frac == 1.0


def test_last_only_logits(tiny_model):
    eng = TieredEngine(tiny_model, DensePolicy())
    x = seq(P)
    full = eng.forward_chunk(x, eng.new_cache())
    last = eng.forward_chunk(x, eng.new_cache(), last_only=True)
    assert last.shape[1] == 1
    torch.testing.assert_close(last[:, 0], full[:, -1], rtol=0, atol=1e-5)  # 1-row vs T-row LM-head matmul
    eng.reset_stats()
    assert eng.stats.macs == 0
