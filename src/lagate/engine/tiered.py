"""P2-05: three-tier inference engine with real skipping and exact Tier-3 K,V (ARCHITECTURE §5).

Batch size 1, greedy. The same ``forward_chunk`` serves prefill (T prompt tokens) and decode (T = 1).
At every layer ℓ, for the chunk's tokens:

    n  = input_layernorm(h)                         all tokens
    μ  = EMA(n) carried in the cache                all tokens (D-17)
    K,V = k_proj/v_proj(n) + RoPE → cache[ℓ]        all tokens, every tier — computed from the token's
                                                    own layer-ℓ state, never copied from another layer
    run_attn = policy                               Tier 1–2 rows: Q, attention, O-proj
    run_mlp  = policy                               Tier 1 rows:   MLP
    Tier 3 rows: h unchanged.

Only the selected rows go through Q/attention/O and the MLP, so skipped work is not executed.
A *policy* makes the decisions, so the same engine runs our router (``LagatePolicy``), dense
(``DensePolicy``), fixed tiers for tests (``FixedTierPolicy``) and, later, the GateSkip port.

Gate scaling (``EngineConfig.gate_scaling``): executed modules are scaled by g_L / g_L·g_M as in
training (GateSkip-style), or added unscaled. This is the ARCHITECTURE §4 open question, decided in
Phase 3; both are supported.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Protocol

import torch

from lagate.engine.flops import LayerCosts
from lagate.models.adapter import DecoderAdapter
from lagate.routing.ema import ema_scan, ema_step
from lagate.routing.router import DualHeadRouter, RouterConfig


@dataclass
class EngineConfig:
    gate_scaling: bool = True   # scale executed modules by their gates (open question §4)
    kv_mode: str = "exact"      # "exact" only for now; "copy_prev" is task P2-10 (D-16)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EngineConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown engine config keys: {sorted(unknown)}")
        cfg = cls(**d)
        if cfg.kv_mode != "exact":
            raise NotImplementedError(f"kv_mode={cfg.kv_mode!r} (copy_prev is task P2-10)")
        return cfg


# ----- cache -----------------------------------------------------------------------------------
@dataclass
class TieredCache:
    """Per-layer K, V (1, kv_heads, n, d) and the EMA state μ (1, H) for one sequence."""

    num_layers: int
    keys: list[torch.Tensor | None] = field(default_factory=list)
    values: list[torch.Tensor | None] = field(default_factory=list)
    mu: list[torch.Tensor | None] = field(default_factory=list)
    length: int = 0

    def __post_init__(self):
        self.keys = [None] * self.num_layers
        self.values = [None] * self.num_layers
        self.mu = [None] * self.num_layers

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.keys[layer] is None:
            self.keys[layer], self.values[layer] = k, v
        else:
            self.keys[layer] = torch.cat([self.keys[layer], k], dim=2)
            self.values[layer] = torch.cat([self.values[layer], v], dim=2)
        return self.keys[layer], self.values[layer]


# ----- policies --------------------------------------------------------------------------------
class Policy(Protocol):
    needs_mlp_input: bool  # True: ``mlp`` receives post_attention_layernorm(h') for every row

    def attn(self, layer: int, n: torch.Tensor, mu: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """-> run_attn (T,) bool, ctx (passed to ``mlp``; may hold gates for scaling/logging).
        ctx["g_layer"] / ctx["g_mlp"] (T,) are logged when present."""

    def mlp(self, layer: int, ctx: dict, attn_out: torch.Tensor, run_attn: torch.Tensor,
            mlp_in: torch.Tensor | None) -> torch.Tensor:
        """-> run_mlp (T,) bool. ``attn_out`` (T_attn, H) holds ΔAttn of the run_attn rows only."""

    def attn_scale(self, ctx: dict, rows: torch.Tensor) -> torch.Tensor | None:
        """Gate scaling for executed attention rows, (R, 1) or (R, H); None = unscaled."""

    def mlp_scale(self, ctx: dict, rows: torch.Tensor) -> torch.Tensor | None: ...

    def router_macs(self, costs: LayerCosts, run_attn: bool) -> float:
        """Router/gate MACs for one token at one layer."""


class _NoScale:
    needs_mlp_input = False

    def attn_scale(self, ctx, rows):
        return None

    def mlp_scale(self, ctx, rows):
        return None


class DensePolicy(_NoScale):
    def attn(self, layer, n, mu, pos):
        return torch.ones(n.shape[1], dtype=torch.bool, device=n.device), {}

    def mlp(self, layer, ctx, attn_out, run_attn, mlp_in):
        return run_attn.clone()

    def router_macs(self, costs, run_attn):
        return 0.0


class FixedTierPolicy(_NoScale):
    """Tiers given per absolute position and layer: ``tiers[pos, layer]`` in {1, 2, 3}. For tests.
    Counted with our router's cost, so FLOPs equal LayerCosts.tier()."""

    def __init__(self, tiers: torch.Tensor):
        self.tiers = tiers

    def attn(self, layer, n, mu, pos):
        return self.tiers[pos, layer] <= 2, {"tier": self.tiers[pos, layer]}

    def mlp(self, layer, ctx, attn_out, run_attn, mlp_in):
        return ctx["tier"] == 1

    def router_macs(self, costs, run_attn):
        return costs.router_layer_head + (costs.router_mlp_head if run_attn else 0)


class LagatePolicy:
    """Our router with calibrated thresholds: Tier 1 if g_L > τ_L and g_M > τ_M; Tier 2 if g_L > τ_L
    and g_M ≤ τ_M; Tier 3 if g_L ≤ τ_L. g_M is computed only for rows where attention ran."""

    needs_mlp_input = False

    def __init__(self, routers: list[DualHeadRouter], tau_layer, tau_mlp):
        self.routers = routers
        L = len(routers)
        self.tau_layer = _per_layer(tau_layer, L)
        self.tau_mlp = _per_layer(tau_mlp, L)

    def attn(self, layer, n, mu, pos):
        g_l, z = self.routers[layer].layer_gate(n[0], mu[0])
        return g_l > self.tau_layer[layer], {"g_layer": g_l, "z": z}

    def mlp(self, layer, ctx, attn_out, run_attn, mlp_in):
        T = run_attn.shape[0]
        g_m = torch.full((T,), float("nan"), dtype=attn_out.dtype, device=attn_out.device)
        if run_attn.any():
            g_m[run_attn] = self.routers[layer].mlp_gate(ctx["z"][run_attn], attn_out)
        ctx["g_mlp"] = g_m
        return run_attn & (g_m > self.tau_mlp[layer])

    def attn_scale(self, ctx, rows):
        return ctx["g_layer"][rows].unsqueeze(-1)

    def mlp_scale(self, ctx, rows):
        return (ctx["g_layer"][rows] * ctx["g_mlp"][rows]).unsqueeze(-1)

    def router_macs(self, costs, run_attn):
        return costs.router_layer_head + (costs.router_mlp_head if run_attn else 0)


def _per_layer(tau, L: int) -> list[float]:
    if isinstance(tau, (int, float)):
        return [float(tau)] * L
    tau = [float(t) for t in tau]
    if len(tau) != L:
        raise ValueError(f"expected {L} thresholds, got {len(tau)}")
    return tau


# ----- engine ----------------------------------------------------------------------------------
@dataclass
class EngineStats:
    macs: float = 0.0          # executed MACs (decoder layers only, see flops.py)
    dense_macs: float = 0.0    # same tokens through the dense model
    tier_counts: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 0: 0})  # 0 = attn skipped, MLP ran

    @property
    def compute_frac(self) -> float:
        return self.macs / self.dense_macs if self.dense_macs else float("nan")


class TieredEngine:
    def __init__(
        self,
        hf_model: torch.nn.Module,
        policy: Policy,
        cfg: EngineConfig | None = None,
        router_cfg: RouterConfig | None = None,
    ):
        self.adapter = DecoderAdapter(hf_model)
        self.policy = policy
        self.cfg = cfg or EngineConfig()
        self.router_cfg = router_cfg or RouterConfig()
        self.costs = LayerCosts(self.adapter.dims, self.router_cfg.bottleneck)
        self.layer_hooks: list[Callable[[dict], None]] = []  # P1-02
        self.record_hidden = False  # add each layer's input h to hook records (tests, analysis)
        self.stats = EngineStats()

    @classmethod
    def from_gated_model(cls, gated, tau_layer, tau_mlp, cfg: EngineConfig | None = None) -> "TieredEngine":
        """Inference engine for a trained LookaheadGatedModel (shares its weights)."""
        policy = LagatePolicy(list(gated.routers), tau_layer, tau_mlp)
        return cls(gated.hf_model, policy, cfg, gated.router_cfg)

    def new_cache(self) -> TieredCache:
        return TieredCache(self.adapter.dims.num_layers)

    @torch.no_grad()
    def forward_chunk(self, input_ids: torch.Tensor, cache: TieredCache, phase: str = "prefill") -> torch.Tensor:
        """Run ``input_ids`` (1, T) after the tokens already in ``cache``. Returns logits (1, T, V)."""
        if input_ids.shape[0] != 1:
            raise ValueError("TieredEngine runs batch size 1")
        ad, pol = self.adapter, self.policy
        T = input_ids.shape[1]
        dev = input_ids.device
        pos = torch.arange(cache.length, cache.length + T, device=dev)
        h = ad.embed(input_ids)
        cos, sin = ad.rope(h, pos)
        ctx_lens = (pos + 1).tolist()
        alpha = self.router_cfg.alpha

        for layer in range(ad.dims.num_layers):
            h_in = h
            n = ad.input_norm(layer, h)
            if T == 1:
                mu = ema_step(n, None if cache.mu[layer] is None else cache.mu[layer][:, None], alpha)
            else:
                mu = ema_scan(n, alpha, mu0=cache.mu[layer], chunk_size=self.router_cfg.ema_chunk_size)
            cache.mu[layer] = mu[:, -1]

            k, v = ad.kv(layer, n, cos, sin)  # exact K,V for every token, every tier
            k_all, v_all = cache.append(layer, k, v)
            k_pos = torch.arange(k_all.shape[2], device=dev)

            run_attn, ctx = pol.attn(layer, n, mu, pos)
            rows = run_attn.nonzero(as_tuple=True)[0]
            attn_rows = h.new_zeros(0, h.shape[-1])
            h = h.clone()
            if rows.numel():
                q = ad.q(layer, n[:, rows], cos[:, rows], sin[:, rows])
                attn_rows = ad.attend(layer, q, k_all, v_all, pos[rows], k_pos)[0]  # (R, H)
                scale = pol.attn_scale(ctx, rows) if self.cfg.gate_scaling else None
                h[0, rows] = h[0, rows] + (attn_rows if scale is None else scale * attn_rows)

            mlp_in = ad.post_norm(layer, h)[0] if pol.needs_mlp_input else None  # (T, H)
            run_mlp = pol.mlp(layer, ctx, attn_rows, run_attn, mlp_in)
            mrows = run_mlp.nonzero(as_tuple=True)[0]
            if mrows.numel():
                x = mlp_in[mrows][None] if mlp_in is not None else ad.post_norm(layer, h[:, mrows])
                out = ad.mlp(layer, x)[0]
                scale = pol.mlp_scale(ctx, mrows) if self.cfg.gate_scaling else None
                h[0, mrows] = h[0, mrows] + (out if scale is None else scale * out)

            self._account(run_attn, run_mlp, ctx_lens)
            if self.layer_hooks:
                rec = {
                    "layer": layer, "phase": phase, "positions": pos, "input_ids": input_ids[0],
                    "g_layer": ctx.get("g_layer"), "g_mlp": ctx.get("g_mlp"),
                    "run_attn": run_attn, "run_mlp": run_mlp, "tier": tiers_from(run_attn, run_mlp),
                }
                if self.record_hidden:
                    rec["h_in"], rec["h_out"] = h_in[0], h[0]
                for hook in self.layer_hooks:
                    hook(rec)

        cache.length += T
        return ad.final_logits(h)

    def _account(self, run_attn: torch.Tensor, run_mlp: torch.Tensor, ctx_lens: list[int]) -> None:
        c, s = self.costs, self.stats
        for a, m, n in zip(run_attn.tolist(), run_mlp.tolist(), ctx_lens):
            s.macs += c.module_macs(a, m, n, router="none") + self.policy.router_macs(c, a)
            s.dense_macs += c.dense(n)
            s.tier_counts[1 if a and m else 2 if a else 0 if m else 3] += 1

    @torch.no_grad()
    def generate(
        self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_ids: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, TieredCache]:
        """Greedy decoding. Returns (generated ids (1, n_new), cache)."""
        cache = self.new_cache()
        logits = self.forward_chunk(prompt_ids, cache, phase="prefill")
        out = []
        for i in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
            if nxt.item() in eos_token_ids or i == max_new_tokens - 1:
                break  # the last token is never fed back (no wasted forward)
            logits = self.forward_chunk(nxt, cache, phase="decode")
        return torch.cat(out, dim=1), cache


def tiers_from(run_attn: torch.Tensor, run_mlp: torch.Tensor) -> torch.Tensor:
    """1 = attention + MLP, 2 = attention only, 3 = neither, 0 = MLP without attention (GateSkip only)."""
    t = torch.full_like(run_attn, 3, dtype=torch.int8)
    t[run_attn & ~run_mlp] = 2
    t[run_attn & run_mlp] = 1
    t[~run_attn & run_mlp] = 0
    return t
