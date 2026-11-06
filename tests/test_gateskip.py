"""P2-09 GateSkip port: hand calculations, cross-checks against the official code (when the pinned
clone is present), calibrated thresholds at bs=1 (D-18), and the engine policy (GateSkip-exactKV)."""

import ast
import math
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lagate.calibration.ecdf import ecdf_threshold, match_budget, skip_fraction
from lagate.engine.tiered import EngineConfig, TieredEngine
from lagate.routing.gateskip import (
    GateSkipConfig, GateSkipGate, GateSkipModel, GateSkipPolicy, calibrate_gateskip, ramp_threshold,
)
from lagate.utils import REPO_ROOT, load_config

OFFICIAL = REPO_ROOT / "baselines/GateSkip/experiment"
needs_official = pytest.mark.skipif(not OFFICIAL.exists(), reason="pinned GateSkip clone not present (git-ignored)")


def _load_official_function(rel_path: str, name: str, ns: dict):
    """Extract one function/method from the official source and exec it standalone."""
    src = (OFFICIAL / rel_path).read_text()
    node = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == name)
    code = textwrap.dedent(ast.get_source_segment(src, node))
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("@"))
    exec(code, ns)
    return ns[name]


# ----- hand calculations ------------------------------------------------------------------------
def test_ramp_threshold_hand_calculation():
    cfg = GateSkipConfig(start_thr_percentile=0.0, end_thr_percentile=0.2, num_increasing_steps=100)
    imp = torch.tensor([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.35])  # sorted: .1 .2 .3 .35 .4 ...
    assert ramp_threshold(imp, 0, cfg) == pytest.approx(0.1)     # pct 0   -> k 0
    assert ramp_threshold(imp, 50, cfg) == pytest.approx(0.2)    # pct 0.1 -> k 1
    assert ramp_threshold(imp, 100, cfg) == pytest.approx(0.3)   # pct 0.2 -> k 2
    assert ramp_threshold(imp, 10_000, cfg) == pytest.approx(0.3)  # capped at end
    assert ramp_threshold(torch.ones(4), 100, cfg) == 0.999      # 1.0 -> 0.999


def test_gate_init_and_value_hand_calculation():
    torch.manual_seed(0)
    g = GateSkipGate(4, GateSkipConfig())
    assert g.proj.weight.shape == (4, 4) and torch.all(g.proj.bias == 5.0)
    assert g.proj.weight.std() < 0.03
    with torch.no_grad():
        g.proj.weight.copy_(torch.eye(4))
        g.proj.bias.zero_()
    x = torch.tensor([[0.0, 1.0, -1.0, 2.0]])
    expect = torch.sigmoid(x)
    torch.testing.assert_close(g(x), expect)
    assert g(x).mean(-1).item() == pytest.approx(expect.mean().item())   # token importance
    assert GateSkipGate(4, GateSkipConfig(gate_dim=1)).proj.out_features == 1


def test_sparsity_loss_and_training_skip(tiny_model):
    cfg = GateSkipConfig(num_increasing_steps=10)
    m = GateSkipModel(tiny_model, cfg).train()
    x = torch.randint(0, 128, (2, 10), generator=torch.Generator().manual_seed(0))
    mask = torch.ones_like(x)
    mask[1, 7:] = 0
    m.global_step = 0
    out = m(x, attention_mask=mask, labels=x)
    # step 0, start 0.0: exactly the lowest-importance valid token of each module is skipped
    for mod in ("attn", "mlp"):
        assert out.skipped[mod].sum(dim=(0, 1)).tolist() == [1, 1, 1]
        assert not out.skipped[mod][1, 7:].any()                  # padding never skipped
        assert torch.all(out.importance[mod][1, 7:] == 1.0)       # padding importance forced to 1
    # L2 loss: mean over the 6 modules of mean(g²); at init g ≈ σ(5), so L ≈ σ(5)²
    assert out.sparsity_loss.item() == pytest.approx(1 / (1 + math.exp(-5)) ** 2, abs=5e-3)
    assert torch.isclose(out.loss, out.ce_loss + 1.0 * out.sparsity_loss)
    # at the end of the ramp, 20% of the 17 valid tokens -> k = int(0.2·17) = 3 -> 4 skipped (≤ threshold)
    m.global_step = 10
    out = m(x, attention_mask=mask)
    assert out.skipped["attn"][..., 0].sum().item() == 4


def test_gates_see_detached_inputs(tiny_model):
    m = GateSkipModel(tiny_model, GateSkipConfig()).train()
    m.apply_train_skipping = False
    x = torch.randint(0, 128, (1, 6))
    m(x).sparsity_loss.backward()  # the L2 loss alone must not reach the backbone (gate input detached)
    assert all(p.grad is not None for p in m.router_parameters())
    assert all(p.grad is None for p in tiny_model.parameters())


def test_saturated_gates_without_skipping_equal_dense(tiny_model):
    m = GateSkipModel(tiny_model, GateSkipConfig(bias_init=50.0, gate_init_std=0.0)).eval()
    m.apply_train_skipping = False
    x = torch.randint(0, 128, (2, 11))
    with torch.no_grad():
        torch.testing.assert_close(m(x).logits, tiny_model(x).logits, rtol=0, atol=1e-5)


def test_config_file_loads():
    assert GateSkipConfig.from_dict(load_config(REPO_ROOT / "configs/router/gateskip.yaml")) == GateSkipConfig()


# ----- cross-checks against the official code ------------------------------------------------------
@needs_official
def test_ramp_threshold_equals_official_property():
    threshold = _load_official_function("models/gating/GatedWrapper.py", "threshold", {"torch": torch})
    cfg = GateSkipConfig(num_increasing_steps=6800)
    gen = torch.Generator().manual_seed(0)
    for step in (0, 1, 1000, 3400, 6800, 9000):
        imp = torch.rand(2, 37, generator=gen)
        valid = torch.rand(2, 37, generator=gen) > 0.2
        fake = SimpleNamespace(
            current_token_importance=imp, current_validity_mask=valid, global_step=step,
            config=SimpleNamespace(increasing_threshold=True, randomly_skip=False, desired_skip_ratio=0.0,
                                   start_thr_percentile=0.0, end_thr_percentile=0.2, num_increasing_steps=6800),
        )
        assert ramp_threshold(imp[valid], step, cfg) == threshold(fake)


@needs_official
def test_gate_init_equals_official():
    init_gate = _load_official_function("models/gating/GateLayer.py", "init_gate", {"torch": torch, "nn": torch.nn, "ModelConfig": object})
    cfg = SimpleNamespace(use_mlp_gate=False, single_number_gates=False, gate_init_mean=0.0, gate_init_std=0.01,
                          gate_init_value=5.0)
    torch.manual_seed(3)
    theirs = init_gate(None, 16, cfg)
    torch.manual_seed(3)
    ours = GateSkipGate(16, GateSkipConfig()).proj
    torch.testing.assert_close(ours.weight, theirs.weight)
    torch.testing.assert_close(ours.bias, theirs.bias)


@needs_official
def test_official_batch_quantile_skips_nothing_at_bs1_decode_but_calibrated_does():
    """D-18 evidence: threshold_finder.py returns 0.0 for a one-token forward; our fixed τ still skips."""
    find_threshold = _load_official_function("utils/threshold_finder.py", "find_threshold", {"torch": torch})
    rng = np.random.default_rng(0)
    calib = rng.uniform(0.95, 1.0, size=5000)                      # importances near σ(5)-ish, as after training
    tau = ecdf_threshold(calib, 0.25)
    decode_imps = torch.tensor(rng.uniform(0.95, 1.0, size=400), dtype=torch.float32)
    official_skips = sum(bool(i <= find_threshold(None, i.reshape(1), 0.25)) for i in decode_imps)
    ours_skips = int((decode_imps <= tau).sum())
    assert official_skips == 0
    assert 0.2 < ours_skips / 400 < 0.3


# ----- calibration --------------------------------------------------------------------------------
def test_ecdf_threshold_and_budget():
    s = np.array([0.5, 0.1, 0.4, 0.2, 0.3, np.nan])
    assert ecdf_threshold(s, 0.0) == -math.inf
    assert ecdf_threshold(s, 0.4) == 0.2 and skip_fraction(s, 0.2) == 0.4
    assert ecdf_threshold(s, 1.0) == 0.5
    with pytest.raises(ValueError):
        ecdf_threshold(s, 1.5)
    p = match_budget(lambda q: 1 - 0.5 * q, target_frac=0.85)       # compute falls linearly with skip fraction
    assert p == pytest.approx(0.3, abs=1e-9)


def test_calibrate_gateskip_per_module():
    rng = np.random.default_rng(1)
    imp = {"attn": rng.uniform(size=(1000, 3)), "mlp": rng.uniform(size=(1000, 3))}
    tau = calibrate_gateskip(imp, 0.15)
    for m in ("attn", "mlp"):
        for l in range(3):
            assert skip_fraction(imp[m][:, l], tau[m][l]) == pytest.approx(0.15)
    only_mlp = calibrate_gateskip(imp, 0.3, skip_module_types=["mlp"])
    assert only_mlp["attn"] == [-math.inf] * 3


# ----- engine (GateSkip-exactKV) -------------------------------------------------------------------
def _engine(model, tau, **cfg):
    return TieredEngine(model.hf_model, GateSkipPolicy(model.gates, tau), EngineConfig(**cfg))


def test_engine_without_skipping_equals_soft_forward(tiny_model):
    m = GateSkipModel(tiny_model, GateSkipConfig(gate_init_std=0.3)).eval()
    m.apply_train_skipping = False
    x = torch.randint(0, 128, (1, 12))
    no_skip = {"attn": [-math.inf] * 3, "mlp": [-math.inf] * 3}
    eng = _engine(m, no_skip)
    with torch.no_grad():
        soft = m(x).logits
    torch.testing.assert_close(eng.forward_chunk(x, eng.new_cache()), soft, rtol=0, atol=1e-5)
    plain = _engine(m, no_skip, gate_scaling=False)
    with torch.no_grad():
        torch.testing.assert_close(plain.forward_chunk(x, plain.new_cache()), tiny_model(x).logits, rtol=0, atol=1e-5)


def test_engine_calibrated_skipping_bs1_decode_and_kv(tiny_model):
    tiny_model.double()
    m = GateSkipModel(tiny_model, GateSkipConfig(gate_init_std=0.3)).eval()
    m.apply_train_skipping = False
    calib = torch.randint(0, 128, (4, 32), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        out = m(calib)
    imp = {k: v.reshape(-1, 3).numpy() for k, v in out.importance.items()}
    tau = calibrate_gateskip(imp, 0.3)

    x = calib[:1]
    eng = _engine(m, tau)
    seen = []
    eng.layer_hooks.append(lambda r: seen.append((r["phase"], r["layer"], r["run_attn"].clone(), r["run_mlp"].clone())))
    cache = eng.new_cache()
    chunks = [eng.forward_chunk(x[:, :20], cache)]
    for t in range(20, 32):
        chunks.append(eng.forward_chunk(x[:, t : t + 1], cache, phase="decode"))
    # layer-0 attention sees the same inputs as calibration: its skip rate equals the calibrated fraction
    a0 = torch.cat([ra for ph, l, ra, rm in seen if l == 0])
    assert 1 - a0.double().mean().item() == pytest.approx(skip_fraction(imp["attn"][:32, 0], tau["attn"][0]))
    decode_skips = sum(int((~ra).sum() + (~rm).sum()) for ph, l, ra, rm in seen if ph == "decode")
    assert decode_skips > 0  # unlike the official batch quantile, bs=1 decode steps do skip
    # T-KV for this policy: cached decode = uncached recompute
    full = _engine(m, tau).forward_chunk(x, eng.new_cache())
    torch.testing.assert_close(torch.cat(chunks, 1), full, rtol=0, atol=1e-9)
    # FLOPs: both vector gates run for every token, 2·H·H MACs per layer
    assert GateSkipPolicy(m.gates, tau).router_macs(eng.costs, True) == 2 * 64 * 64
