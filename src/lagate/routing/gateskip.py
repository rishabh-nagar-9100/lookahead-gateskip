"""P2-09: GateSkip re-implemented in our pipeline (D-16), from the official code at commit 8d56513
(`baselines/GateSkip/experiment/models/gating/`; file:line references below).

Faithful parts (job `GateSkip-separate-vector`, `jobs/job_common.sh:20–33`):
- One gate per module (attention and MLP of every layer): g = σ(W x + b), W: H → H ("vector"
  gate; ``gate_dim: 1`` gives the scalar variant). W ~ N(0, 0.01), b = +5 (GateLayer.py:34–40,
  GatingConfig.py:97–103). x is the module's **input**, detached (GatedWrapper.py:227): for
  attention input_layernorm(h), for the MLP post_attention_layernorm(h').
- Output scaling: module_out ← g ⊙ module_out (GatedWrapper.py:267–268, ``actually_gate``).
- Token importance = mean(g) over the gate dimension (:229); padding and EOS tokens get
  importance 1 and are left out of the threshold (:121–125, DefaultLightningModule.py:124–126).
- Training skip: threshold = the p-th entry of the sorted importances of the valid tokens in the
  current batch, p ramping linearly from ``start`` to ``end`` over ``num_increasing_steps``;
  a threshold of exactly 1.0 becomes 0.999; tokens with importance ≤ threshold get a zero module
  output (:73–98, :238, :270–271). With start = 0 the lowest-importance token is already skipped
  at step 0 (k = 0), as in the original.
- Sparsity loss: mean over modules of mean(g²) over all positions, weight 1.0 (ModelGating.py:59–91).

Documented adaptation (D-18): at **inference** the per-module thresholds are fixed values
calibrated on the calibration split with the same empirical-CDF procedure as our method
(`lagate.calibration.ecdf`), instead of the batch quantile of `threshold_finder.py`, which returns
0.0 for a one-token forward and so skips nothing at bs=1 decode. ``GateSkipPolicy`` applies them in
our engine with real skipping and exact per-layer K,V (GateSkip-exactKV).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from lagate.calibration.ecdf import ecdf_threshold
from lagate.models.adapter import DecoderAdapter
from lagate.models.gated import _is_right_padded

MODULES = ("attn", "mlp")


@dataclass
class GateSkipConfig:
    gate_dim: str | int = "vector"          # "vector" (= H, separate-vector job) or 1 (scalar)
    gate_init_std: float = 0.01
    bias_init: float = 5.0
    sparsity_loss_weight: float = 1.0       # L2 on gate values
    start_thr_percentile: float = 0.0
    end_thr_percentile: float = 0.2
    num_increasing_steps: int = 6800
    skip_module_types: list[str] = field(default_factory=lambda: ["attn", "mlp"])
    invalid_token_ids: list[int] = field(default_factory=list)  # e.g. [pad, eos], importance forced to 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "GateSkipConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown GateSkip config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GateSkipGate(nn.Module):
    def __init__(self, hidden: int, cfg: GateSkipConfig):
        super().__init__()
        out = hidden if cfg.gate_dim == "vector" else int(cfg.gate_dim)
        self.proj = nn.Linear(hidden, out)
        nn.init.normal_(self.proj.weight, mean=0.0, std=cfg.gate_init_std)
        nn.init.constant_(self.proj.bias, cfg.bias_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.proj(x))


def ramp_threshold(importances_valid: torch.Tensor, step: int, cfg: GateSkipConfig) -> float:
    """GatedWrapper.threshold, increasing-threshold branch (GatedWrapper.py:73–98)."""
    s = torch.sort(importances_valid.detach().float().flatten()).values
    frac = step / cfg.num_increasing_steps if step is not None else 1.0
    pct = cfg.start_thr_percentile + min(1.0, frac) * (cfg.end_thr_percentile - cfg.start_thr_percentile)
    k = min(int(pct * len(s)), len(s) - 1)
    thr = s[k].item()
    return thr if thr != 1.0 else 0.999


@dataclass
class GateSkipOutput:
    logits: torch.Tensor
    importance: dict[str, torch.Tensor]   # "attn"/"mlp" -> (B, T, L) mean gate value
    skipped: dict[str, torch.Tensor]      # "attn"/"mlp" -> (B, T, L) bool (training skip mask)
    sparsity_loss: torch.Tensor
    ce_loss: torch.Tensor | None
    loss: torch.Tensor | None


class GateSkipModel(nn.Module):
    """Training-time GateSkip on a HF Llama/Qwen2 model, through our adapter."""

    def __init__(self, hf_model: nn.Module, cfg: GateSkipConfig):
        super().__init__()
        self.adapter = DecoderAdapter(hf_model)
        self.cfg = cfg
        H, L = self.adapter.dims.hidden, self.adapter.dims.num_layers
        self.gates = nn.ModuleList(nn.ModuleDict({m: GateSkipGate(H, cfg) for m in MODULES}) for _ in range(L))
        self.gates.to(next(hf_model.parameters()).dtype)
        self.global_step = 0            # set by the training loop (Lightning's global_step)
        self.apply_train_skipping = True  # False: soft gates only (calibration pass, D-18)
        self.layer_hooks: list[Callable[[dict], None]] = []

    @property
    def hf_model(self) -> nn.Module:
        return self.adapter.model

    def router_parameters(self):
        return self.gates.parameters()

    def _gate(self, layer: int, module: str, x: torch.Tensor, valid: torch.Tensor):
        g = self.gates[layer][module](x.detach())
        imp = torch.where(valid, g.mean(-1), torch.ones((), dtype=g.dtype, device=g.device))
        skip = torch.zeros_like(valid)
        if self.apply_train_skipping and module in self.cfg.skip_module_types and valid.any():
            thr = ramp_threshold(imp[valid], self.global_step, self.cfg)
            skip = imp <= thr
        return g, imp, skip

    def forward(self, input_ids, attention_mask=None, labels=None, sparsity_weight: float | None = None) -> GateSkipOutput:
        ad = self.adapter
        B, T = input_ids.shape
        valid_attn = torch.ones(B, T, dtype=torch.bool, device=input_ids.device) if attention_mask is None else attention_mask.bool()
        if not _is_right_padded(valid_attn):
            raise ValueError("attention_mask must be right-padded")
        valid = valid_attn.clone()
        if self.cfg.invalid_token_ids:
            valid &= ~torch.isin(input_ids, torch.tensor(self.cfg.invalid_token_ids, device=input_ids.device))

        h = ad.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = ad.rope(h, pos)
        imps = {m: [] for m in MODULES}
        skips = {m: [] for m in MODULES}
        sq = []
        for layer in range(ad.dims.num_layers):
            n = ad.input_norm(layer, h)
            g, imp, skip = self._gate(layer, "attn", n, valid)
            k, v = ad.kv(layer, n, cos, sin)
            out = g * ad.attend(layer, ad.q(layer, n, cos, sin), k, v, pos, pos, key_valid=valid_attn)
            h = h + torch.where(skip.unsqueeze(-1), torch.zeros_like(out), out)
            sq.append(g.pow(2).mean())
            imps["attn"].append(imp); skips["attn"].append(skip)

            m_in = ad.post_norm(layer, h)
            g, imp, skip = self._gate(layer, "mlp", m_in, valid)
            out = g * ad.mlp(layer, m_in)
            h = h + torch.where(skip.unsqueeze(-1), torch.zeros_like(out), out)
            sq.append(g.pow(2).mean())
            imps["mlp"].append(imp); skips["mlp"].append(skip)
            for hook in self.layer_hooks:
                hook({"layer": layer, "phase": "train", "positions": pos, "input_ids": input_ids, "valid": valid_attn,
                      "g_layer": imps["attn"][-1], "g_mlp": imp,
                      "run_attn": ~skips["attn"][-1], "run_mlp": ~skip,
                      "tier": None})
        logits = ad.final_logits(h)
        l_s = torch.stack(sq).mean()
        ce = loss = None
        if labels is not None:
            ce = nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                                             labels[:, 1:].reshape(-1), ignore_index=-100)
            w = self.cfg.sparsity_loss_weight if sparsity_weight is None else sparsity_weight
            loss = ce + w * l_s
        return GateSkipOutput(logits, {m: torch.stack(v, -1) for m, v in imps.items()},
                              {m: torch.stack(v, -1) for m, v in skips.items()}, l_s, ce, loss)


# ----- inference: calibrated thresholds + engine policy -------------------------------------------
def calibrate_gateskip(importance: dict[str, np.ndarray], skip_frac: float,
                       skip_module_types=MODULES) -> dict[str, list[float]]:
    """Per-module fixed thresholds from calibration-split importances (D-18).

    ``importance[m]`` is (N_tokens, L) for module m, collected with ``apply_train_skipping=False``
    (soft gates, no skipping). Same per-module skip fraction for every module, as their
    ``desired_skip_ratio`` (EvaluationRunner.py:265–280); modules not in ``skip_module_types`` get −inf.
    """
    out = {}
    for m in MODULES:
        L = importance[m].shape[1]
        out[m] = [ecdf_threshold(importance[m][:, l], skip_frac) if m in skip_module_types else float("-inf")
                  for l in range(L)]
    return out


class GateSkipPolicy:
    """GateSkip-exactKV in our engine: attention and MLP decided independently from their own
    gates on their own inputs; skipped = importance ≤ τ (GatedWrapper.py:238). The MLP gate reads
    post_attention_layernorm(h') for every token, so it may run where attention was skipped
    (engine tier 0)."""

    needs_mlp_input = True
    uses_ema = False
    all_dense = False

    def __init__(self, gates: nn.ModuleList, tau: dict[str, list[float]]):
        self.gates = gates
        self.tau = tau

    def attn(self, layer, n, mu, pos):
        g = self.gates[layer]["attn"](n[0])
        imp = g.mean(-1)
        return imp > self.tau["attn"][layer], {"g_attn_vec": g, "g_layer": imp}

    def mlp(self, layer, ctx, attn_out, run_attn, mlp_in):
        g = self.gates[layer]["mlp"](mlp_in)
        imp = g.mean(-1)
        ctx["g_mlp_vec"], ctx["g_mlp"] = g, imp
        return imp > self.tau["mlp"][layer]

    def attn_scale(self, ctx, rows):
        return ctx["g_attn_vec"][rows]

    def mlp_scale(self, ctx, rows):
        return ctx["g_mlp_vec"][rows]

    def router_macs(self, costs, run_attn):
        """Both gates run for every token: 2 · H · gate_dim MACs."""
        g = self.gates[0]["attn"].proj
        return 2 * g.in_features * g.out_features
