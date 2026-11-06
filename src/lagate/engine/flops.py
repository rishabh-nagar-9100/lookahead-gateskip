"""P2-06 FLOP accounting per token and layer (ARCHITECTURE §7). Units: multiply-accumulates (MACs);
FLOPs = 2 × MACs.

Per token at one layer, with n = context length the token attends over (including itself):

    Q,O projections      H·(nh·d) + (nh·d)·H            (= 2H² for Llama/Qwen)
    K,V projections      2·H·(kvh·d)
    attention            2·n·(nh·d)                      (QKᵀ row + weighted sum of V; = 2nH)
    MLP (SwiGLU)         3·H·I
    router, layer head   2H·b + b                        (W_1, w_L)   always computed
    router, MLP head     H·b + 2b                        (W_a, w_M)   only when attention ran

    Tier 1 = all; Tier 2 = all but MLP; Tier 3 = K,V + router layer head.

Not counted: embeddings / LM head (identical for every method), RMSNorm, RoPE, SiLU, the EMA
update (≈3H element-wise ops, not MACs), biases (Qwen's q/k/v biases are H-length adds).
"""

from __future__ import annotations

from dataclasses import dataclass

from lagate.models.adapter import Dims

TIERS = (1, 2, 3)


@dataclass(frozen=True)
class LayerCosts:
    dims: Dims
    bottleneck: int

    @property
    def qo(self) -> int:
        d = self.dims
        return 2 * d.hidden * d.n_heads * d.head_dim

    @property
    def kv(self) -> int:
        d = self.dims
        return 2 * d.hidden * d.n_kv_heads * d.head_dim

    def attn(self, n: float) -> float:
        d = self.dims
        return 2 * n * d.n_heads * d.head_dim

    @property
    def mlp(self) -> int:
        return 3 * self.dims.hidden * self.dims.intermediate

    @property
    def router_layer_head(self) -> int:
        return 2 * self.dims.hidden * self.bottleneck + self.bottleneck

    @property
    def router_mlp_head(self) -> int:
        return self.dims.hidden * self.bottleneck + 2 * self.bottleneck

    @property
    def router(self) -> int:
        return self.router_layer_head + self.router_mlp_head

    def dense(self, n: float) -> float:
        """A dense layer without router (the 100% reference)."""
        return self.qo + self.kv + self.attn(n) + self.mlp

    def tier(self, tier: int, n: float) -> float:
        if tier == 1:
            return self.dense(n) + self.router
        if tier == 2:
            return self.qo + self.kv + self.attn(n) + self.router
        if tier == 3:
            return self.kv + self.router_layer_head
        raise ValueError(f"tier must be 1, 2 or 3, got {tier}")

    def module_macs(self, run_attn: bool, run_mlp: bool, n: float, router: str = "lagate") -> float:
        """MACs for arbitrary module decisions (also covers GateSkip, which can skip attention but
        run the MLP). K,V always run (exact K,V). ``router``: 'lagate', 'none' (dense) or a
        callable cost is added by the caller."""
        macs = self.kv + (self.qo + self.attn(n) if run_attn else 0) + (self.mlp if run_mlp else 0)
        if router == "lagate":
            macs += self.router_layer_head + (self.router_mlp_head if run_attn else 0)
        elif router != "none":
            raise ValueError(router)
        return macs

    # ----- sparsity-loss weights (ARCHITECTURE §4, D-17) -------------------------------------
    def sparsity_weights(self, n_avg: float) -> tuple[float, float]:
        """(c_attn, c_mlp): the costs that g_L and g_L·g_M switch off. K,V always run (Tier 3), so
        c_attn = Q,O + attention at the average training context length."""
        return self.qo + self.attn(n_avg), float(self.mlp)

    def router_overhead(self, n: float) -> float:
        """Full router cost as a fraction of a dense layer at context length n."""
        return self.router / self.dense(n)


def avg_context_length(seq_len: int) -> float:
    """Mean attended length over positions 1..T of one causal sequence: (T + 1) / 2."""
    return (seq_len + 1) / 2
