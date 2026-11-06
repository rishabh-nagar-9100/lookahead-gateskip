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
class TieredCache:
    """Per-layer K, V and the EMA state μ (1, H) for one sequence.

    K,V live in preallocated buffers (1, kv_heads, capacity, d) written in place (P5-05), so a decode
    step does not copy the whole cache as ``torch.cat`` would. Capacity doubles when exceeded;
    ``generate`` sizes it to prompt + new tokens up front. ``keys`` / ``values`` give the filled views.
    """

    def __init__(self, num_layers: int, capacity: int = 0):
        self.num_layers = num_layers
        self.capacity = capacity
        self._k: list[torch.Tensor | None] = [None] * num_layers
        self._v: list[torch.Tensor | None] = [None] * num_layers
        self.filled = [0] * num_layers
        self.mu: list[torch.Tensor | None] = [None] * num_layers
        self.length = 0

    def _ensure(self, layer: int, need: int, like: torch.Tensor) -> None:
        buf = self._k[layer]
        cap = 0 if buf is None else buf.shape[2]
        if need <= cap:
            return
        new_cap = max(need, 2 * cap, self.capacity, 64)
        shape = (like.shape[0], like.shape[1], new_cap, like.shape[3])
        k = like.new_empty(shape)
        v = like.new_empty(shape)
        if buf is not None:
            n = self.filled[layer]
            k[:, :, :n] = self._k[layer][:, :, :n]
            v[:, :, :n] = self._v[layer][:, :, :n]
        self._k[layer], self._v[layer] = k, v

    def append(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.filled[layer]
        end = start + k.shape[2]
        self._ensure(layer, end, k)
        self._k[layer][:, :, start:end] = k
        self._v[layer][:, :, start:end] = v
        self.filled[layer] = end
        return self._k[layer][:, :, :end], self._v[layer][:, :, :end]

    @property
    def keys(self) -> list[torch.Tensor | None]:
        return [None if b is None else b[:, :, :n] for b, n in zip(self._k, self.filled)]

    @property
    def values(self) -> list[torch.Tensor | None]:
        return [None if b is None else b[:, :, :n] for b, n in zip(self._v, self.filled)]


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
    uses_ema = False   # skip the EMA update when the policy never reads μ (P5-05)
    all_dense = False  # True: every module always runs; the engine takes its fast path

    def attn_scale(self, ctx, rows):
        return None

    def mlp_scale(self, ctx, rows):
        return None


class DensePolicy(_NoScale):
    all_dense = True

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
    uses_ema = True
    all_dense = False

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
        if attn_out.shape[0]:  # rows where attention ran (engine passes them in ctx["sel"]; no host sync)
            sel = ctx["sel"]
            g_m[sel] = self.routers[layer].mlp_gate(ctx["z"][sel], attn_out)
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
        self._stats = EngineStats()
        # Decisions stay on the device until ``stats`` is read (P5-05: no per-token host sync).
        self._pending: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]] = []

    @classmethod
    def from_gated_model(cls, gated, tau_layer, tau_mlp, cfg: EngineConfig | None = None) -> "TieredEngine":
        """Inference engine for a trained LookaheadGatedModel (shares its weights)."""
        policy = LagatePolicy(list(gated.routers), tau_layer, tau_mlp)
        return cls(gated.hf_model, policy, cfg, gated.router_cfg)

    def new_cache(self, capacity: int = 0) -> TieredCache:
        return TieredCache(self.adapter.dims.num_layers, capacity)

    # ----- FLOP / tier statistics (computed lazily) ------------------------------------------------
    @property
    def stats(self) -> EngineStats:
        if self._pending:
            self._flush_stats()
        return self._stats

    def reset_stats(self) -> None:
        self._pending.clear()
        self._stats = EngineStats()

    def _flush_stats(self) -> None:
        c, s = self.costs, self._stats
        r_on = float(self.policy.router_macs(c, True))
        r_off = float(self.policy.router_macs(c, False))
        for pos, run_attn, run_mlp in self._pending:
            n = (pos.to("cpu", torch.float64) + 1)
            a = torch.ones_like(n, dtype=torch.bool) if run_attn is None else run_attn.cpu()
            m = torch.ones_like(n, dtype=torch.bool) if run_mlp is None else run_mlp.cpu()
            attn = 2 * n * c.dims.n_heads * c.dims.head_dim
            macs = c.kv + a * (c.qo + attn) + m * c.mlp + torch.where(a, r_on, r_off)
            s.macs += float(macs.sum())
            s.dense_macs += float((c.qo + c.kv + attn + c.mlp).sum())
            tiers = tiers_from(a, m)
            for tier in (0, 1, 2, 3):
                s.tier_counts[tier] += int((tiers == tier).sum())
        self._pending.clear()

    # ----- forward --------------------------------------------------------------------------------
    @torch.no_grad()
    def forward_chunk(self, input_ids: torch.Tensor, cache: TieredCache, phase: str = "prefill",
                      last_only: bool = False) -> torch.Tensor:
        """Run ``input_ids`` (1, T) after the tokens already in ``cache``.

        Returns logits (1, T, V), or (1, 1, V) for the last position if ``last_only``."""
        if input_ids.shape[0] != 1:
            raise ValueError("TieredEngine runs batch size 1")
        ad, pol = self.adapter, self.policy
        T = input_ids.shape[1]
        dev = input_ids.device
        start = cache.length
        pos = torch.arange(start, start + T, device=dev)
        k_pos = torch.arange(start + T, device=dev)  # positions of all cached keys after this chunk
        h = ad.embed(input_ids)
        cos, sin = ad.rope(h, pos)
        alpha = self.router_cfg.alpha
        dense = getattr(pol, "all_dense", False)
        uses_ema = getattr(pol, "uses_ema", True)
        decode = T == 1  # the single query sees every cached key: no mask, no row gather

        for layer in range(ad.dims.num_layers):
            h_in = h
            n = ad.input_norm(layer, h)
            mu = None
            if uses_ema:
                if decode:
                    mu = ema_step(n, None if cache.mu[layer] is None else cache.mu[layer][:, None], alpha)
                else:
                    mu = ema_scan(n, alpha, mu0=cache.mu[layer], chunk_size=self.router_cfg.ema_chunk_size)
                cache.mu[layer] = mu[:, -1]

            k, v = ad.kv(layer, n, cos, sin)  # exact K,V for every token, every tier
            k_all, v_all = cache.append(layer, k, v)

            if dense:  # fast path: no decisions, no gathers, no host syncs
                h = h + ad.attend(layer, ad.q(layer, n, cos, sin), k_all, v_all, pos, k_pos, all_visible=decode)
                h = h + ad.mlp(layer, ad.post_norm(layer, h))
                self._pending.append((pos, None, None))
                if self.layer_hooks:
                    ones = torch.ones(T, dtype=torch.bool, device=dev)
                    self._run_hooks(layer, phase, pos, input_ids, {}, ones, ones, h_in, h)
                continue

            run_attn, ctx = pol.attn(layer, n, mu, pos)
            sel = _selection(run_attn, decode)  # slice / index tensor / None (nothing runs)
            ctx["sel"] = sel
            attn_rows = h.new_zeros(0, h.shape[-1])
            if sel is not None:
                q = ad.q(layer, n[:, sel], cos[:, sel], sin[:, sel])
                attn_rows = ad.attend(layer, q, k_all, v_all, pos[sel], k_pos, all_visible=decode)[0]  # (R, H)
                scale = pol.attn_scale(ctx, sel) if self.cfg.gate_scaling else None
                upd = attn_rows if scale is None else scale * attn_rows
                h = _add_rows(h, sel, upd)

            mlp_in = ad.post_norm(layer, h)[0] if pol.needs_mlp_input else None  # (T, H)
            run_mlp = pol.mlp(layer, ctx, attn_rows, run_attn, mlp_in)
            msel = _selection(run_mlp, decode)
            if msel is not None:
                x = mlp_in[msel][None] if mlp_in is not None else ad.post_norm(layer, h[:, msel])
                out = ad.mlp(layer, x)[0]
                scale = pol.mlp_scale(ctx, msel) if self.cfg.gate_scaling else None
                h = _add_rows(h, msel, out if scale is None else scale * out)

            self._pending.append((pos, run_attn, run_mlp))
            if self.layer_hooks:
                self._run_hooks(layer, phase, pos, input_ids, ctx, run_attn, run_mlp, h_in, h)

        cache.length += T
        return ad.final_logits(h[:, -1:] if last_only else h)

    def _run_hooks(self, layer, phase, pos, input_ids, ctx, run_attn, run_mlp, h_in, h_out) -> None:
        rec = {
            "layer": layer, "phase": phase, "positions": pos, "input_ids": input_ids[0],
            "g_layer": ctx.get("g_layer"), "g_mlp": ctx.get("g_mlp"),
            "run_attn": run_attn, "run_mlp": run_mlp, "tier": tiers_from(run_attn, run_mlp),
        }
        if self.record_hidden:
            rec["h_in"], rec["h_out"] = h_in[0], h_out[0]
        for hook in self.layer_hooks:
            hook(rec)

    @torch.no_grad()
    def generate(
        self, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_ids: tuple[int, ...] = ()
    ) -> tuple[torch.Tensor, TieredCache]:
        """Greedy decoding. Returns (generated ids (1, n_new), cache)."""
        cache = self.new_cache(capacity=prompt_ids.shape[1] + max_new_tokens)
        logits = self.forward_chunk(prompt_ids, cache, phase="prefill", last_only=True)
        out = []
        for i in range(max_new_tokens):
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
            if i == max_new_tokens - 1 or (eos_token_ids and nxt.item() in eos_token_ids):
                break  # the last token is never fed back (no wasted forward)
            logits = self.forward_chunk(nxt, cache, phase="decode", last_only=True)
        return torch.cat(out, dim=1), cache


def _selection(mask: torch.Tensor, decode: bool):
    """Rows to run: ``slice(None)`` (all), an index tensor, or None (none).

    A decode step has one row, so a single ``bool()`` (one host sync, needed for real skipping)
    replaces ``nonzero``. Prefill uses ``nonzero``."""
    if decode:
        return slice(None) if bool(mask[0]) else None
    rows = mask.nonzero(as_tuple=True)[0]
    if rows.numel() == 0:
        return None
    return slice(None) if rows.numel() == mask.shape[0] else rows


def _add_rows(h: torch.Tensor, sel, upd: torch.Tensor) -> torch.Tensor:
    if isinstance(sel, slice):
        return h + upd[None]
    h = h.clone()
    h[0, sel] = h[0, sel] + upd
    return h


def tiers_from(run_attn: torch.Tensor, run_mlp: torch.Tensor) -> torch.Tensor:
    """1 = attention + MLP, 2 = attention only, 3 = neither, 0 = MLP without attention (GateSkip only)."""
    t = torch.full_like(run_attn, 3, dtype=torch.int8)
    t[run_attn & ~run_mlp] = 2
    t[run_attn & run_mlp] = 1
    t[~run_attn & run_mlp] = 0
    return t
