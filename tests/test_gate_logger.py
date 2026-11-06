"""P1-02 gate logging hook (+ the E2/E3 pilot statistics built on it)."""

import numpy as np
import torch

from lagate.analysis.gate_logger import GateLogger, load_gate_log
from lagate.analysis.gate_stats import adjacent_gate_correlation, skip_rates
from lagate.engine.tiered import FixedTierPolicy, TieredEngine
from lagate.models.gated import LookaheadGatedModel
from lagate.routing.router import RouterConfig


def test_logs_engine_generation(tiny_model, tmp_path):
    gated = LookaheadGatedModel(tiny_model, RouterConfig(head_init_std=3.0, bias_init=0.0))
    eng = TieredEngine.from_gated_model(gated, tau_layer=0.5, tau_mlp=0.5)
    log = GateLogger().attach(eng)
    L = tiny_model.config.num_hidden_layers
    n_tok = 0
    for doc_id, (p, new) in enumerate([(6, 4), (9, 3)]):
        log.new_sequence(doc_id=doc_id)
        prompt = torch.randint(0, 128, (1, p), generator=torch.Generator().manual_seed(doc_id))
        gen, _ = eng.generate(prompt, new)
        log.set_meta(correct=bool(doc_id))
        n_tok += p + gen.shape[1] - 1  # last generated token is not fed back
    a = log.to_arrays()
    assert a["g_layer"].shape == a["tier"].shape == (n_tok, L)
    assert set(np.unique(a["phase"])) == {0, 1}
    # g_M is NaN exactly where attention did not run; tiers agree with the decisions
    assert np.array_equal(np.isnan(a["g_mlp"]), ~a["run_attn"])
    assert np.array_equal(a["tier"] == 3, ~a["run_attn"] & ~a["run_mlp"])
    assert np.array_equal(a["run_attn"], a["g_layer"] > 0.5)
    # stats agree with the engine's own tier counts
    assert int((a["tier"] == 3).sum()) == eng.stats.tier_counts[3]
    path = log.save(tmp_path / "gates.npz")
    arrays, seqs = load_gate_log(path)
    assert seqs == [{"doc_id": 0, "correct": False}, {"doc_id": 1, "correct": True}]
    assert np.array_equal(arrays["position"][a["seq"] == 1], np.arange(9 + 2))
    log.detach()
    assert eng.layer_hooks == []


def test_logs_fixed_tiers_without_gates(tiny_model):
    tiers = torch.randint(1, 4, (8, 3), generator=torch.Generator().manual_seed(0))
    eng = TieredEngine(tiny_model, FixedTierPolicy(tiers))
    log = GateLogger().attach(eng)
    eng.forward_chunk(torch.randint(0, 128, (1, 8)), eng.new_cache())
    a = log.to_arrays()
    assert np.array_equal(a["tier"], tiers.numpy()) and np.isnan(a["g_layer"]).all()


def test_logs_training_forward_with_padding(tiny_model):
    gated = LookaheadGatedModel(tiny_model, RouterConfig())
    log = GateLogger().attach(gated)
    x = torch.randint(0, 128, (2, 6))
    mask = torch.tensor([[1] * 6, [1] * 4 + [0] * 2])
    with torch.no_grad():
        out = gated(x, attention_mask=mask)
    a = log.to_arrays()
    assert a["g_layer"].shape == (10, 3) and set(a["phase"]) == {2}
    np.testing.assert_allclose(a["g_layer"][:6], out.g_layer[0].numpy())
    np.testing.assert_allclose(a["g_mlp"][6:], out.g_mlp[1, :4].numpy())
    assert a["run_attn"].all() and (a["tier"] == 1).all()


def test_adjacent_correlation_and_skip_rates():
    rng = np.random.default_rng(0)
    T, L = 400, 2
    slow = np.cumsum(rng.normal(size=T)) * 0.1          # layer 0: slowly varying -> high lag-1 r
    noise = rng.normal(size=T)                            # layer 1: white noise -> r ≈ 0
    arrays = {
        "g_layer": np.stack([slow, noise], 1), "seq": np.repeat([0, 1], T // 2),
        "position": np.tile(np.arange(T // 2), 2), "phase": np.ones(T, np.int8),
        "run_attn": np.stack([slow > 0, noise > 0], 1), "run_mlp": np.zeros((T, L), bool),
        "tier": np.where(np.stack([slow > 0, noise > 0], 1), 2, 3).astype(np.int8),
    }
    r = adjacent_gate_correlation(arrays, lags=[1, 8])
    assert r[0, 0] > 0.9 and abs(r[0, 1]) < 0.2 and r[1, 0] < r[0, 0]
    s = skip_rates(arrays)
    np.testing.assert_allclose(s["attn_skip_rate"], 1 - arrays["run_attn"].mean(0))
    np.testing.assert_allclose(s["tier2_rate"] + s["tier3_rate"], 1.0)
