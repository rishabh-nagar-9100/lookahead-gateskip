"""Dual-head router, one per decoder layer (ARCHITECTURE §3, implementation details in DECISIONS D-17).

    x   = [ RMSNorm(h_t) ∥ μ_t ]          2H
    z   = SiLU(W_1 x)                     W_1: 2H → b (b = bottleneck, default 64), no bias
    g_L = σ(w_L · z + b_L)                layer gate
    a   = W_a · RMSNorm(ΔAttn_t)          W_a: H → b, no bias; only when attention ran
    g_M = σ(w_M · [z ∥ a] + b_M)          MLP gate

RMSNorm(h_t) is the decoder layer's own ``input_layernorm`` output (the same tensor that feeds
Q, K, V, so it costs nothing extra); the caller passes it in. RMSNorm(ΔAttn) is router-owned.
b_L = b_M = ``bias_init`` (+5, gates start ≈ 0.993); w_L, w_M ~ N(0, head_init_std).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
from torch import nn


@dataclass
class RouterConfig:
    alpha: float = 0.7            # EMA decay weight (§2)
    bottleneck: int = 64          # shared code size b
    bias_init: float = 5.0        # b_L, b_M at init
    head_init_std: float = 0.01   # std of w_L, w_M at init
    ema_chunk_size: int = 64      # prefill scan chunk (numerics/speed only, not a modelling choice)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RouterConfig":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown router config keys: {sorted(unknown)}")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RMSNorm(nn.Module):
    """Same definition as Llama/Qwen2 RMSNorm (fp32 statistics)."""

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x32 = x.float()
        x32 = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x32.to(dtype)


class DualHeadRouter(nn.Module):
    def __init__(self, hidden_size: int, cfg: RouterConfig, rms_eps: float):
        super().__init__()
        b = cfg.bottleneck
        self.hidden_size, self.bottleneck = hidden_size, b
        self.w1 = nn.Linear(2 * hidden_size, b, bias=False)
        self.head_l = nn.Linear(b, 1)
        self.attn_norm = RMSNorm(hidden_size, rms_eps)
        self.w_a = nn.Linear(hidden_size, b, bias=False)
        self.head_m = nn.Linear(2 * b, 1)
        self.reset_heads(cfg)

    @torch.no_grad()
    def reset_heads(self, cfg: RouterConfig) -> None:
        for head in (self.head_l, self.head_m):
            nn.init.normal_(head.weight, mean=0.0, std=cfg.head_init_std)
            nn.init.constant_(head.bias, cfg.bias_init)

    def layer_gate(self, h_norm: torch.Tensor, mu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """g_L (shape (...,)) and the shared code z (shape (..., b)) from RMSNorm(h) and μ."""
        z = nn.functional.silu(self.w1(torch.cat([h_norm, mu], dim=-1)))
        return torch.sigmoid(self.head_l(z)).squeeze(-1), z

    def mlp_gate(self, z: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """g_M (shape (...,)) from z and the attention sub-layer output ΔAttn (before gate scaling)."""
        a = self.w_a(self.attn_norm(attn_out))
        return torch.sigmoid(self.head_m(torch.cat([z, a], dim=-1))).squeeze(-1)
