"""Pilot statistics on gate logs (P1-03 E2, P1-04 E3). Input: arrays from GateLogger.to_arrays()."""

from __future__ import annotations

import numpy as np


def adjacent_gate_correlation(arrays: dict[str, np.ndarray], lags=range(1, 9), key: str = "g_layer",
                              phase: int | None = None) -> np.ndarray:
    """E3: Pearson r between a token's gate and the gate ``lag`` tokens later in the same sequence,
    per layer. Returns (len(lags), L); NaN where too few pairs or zero variance."""
    g = arrays[key]
    seq, pos = arrays["seq"], arrays["position"]
    keep = np.ones(len(seq), bool) if phase is None else arrays["phase"] == phase
    order = np.lexsort((pos, seq))
    order = order[keep[order]]
    g, seq, pos = g[order], seq[order], pos[order]
    out = np.full((len(lags), g.shape[1]), np.nan)
    for i, lag in enumerate(lags):
        if len(seq) <= lag:
            continue
        same = (seq[lag:] == seq[:-lag]) & (pos[lag:] - pos[:-lag] == lag)
        a, b = g[:-lag][same], g[lag:][same]
        for layer in range(g.shape[1]):
            x, y = a[:, layer], b[:, layer]
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() >= 3 and x[ok].std() > 0 and y[ok].std() > 0:
                out[i, layer] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def skip_rates(arrays: dict[str, np.ndarray], mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """E2: per-layer fraction of tokens (optionally a subset ``mask``) that skipped attention / MLP,
    and the tier histogram."""
    m = np.ones(len(arrays["seq"]), bool) if mask is None else mask
    run_attn, run_mlp, tier = arrays["run_attn"][m], arrays["run_mlp"][m], arrays["tier"][m]
    return {
        "n_tokens": np.array(int(m.sum())),
        "attn_skip_rate": 1 - run_attn.mean(0),
        "mlp_skip_rate": 1 - run_mlp.mean(0),
        **{f"tier{t}_rate": (tier == t).mean(0) for t in (0, 1, 2, 3)},
    }
