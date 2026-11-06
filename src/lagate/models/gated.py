"""P2-04: soft-gated training forward pass (ARCHITECTURE §4).

At every layer ℓ (n = input_layernorm(h), μ = EMA scan of n over the sequence):

    g_L, z = router.layer_gate(n, μ)
    ΔAttn  = Attn(n)                                   (O-proj output)
    h'     = h  + g_L · ΔAttn
    g_M    = router.mlp_gate(z, ΔAttn)
    h''    = h' + g_L · g_M · MLP(post_attention_layernorm(h'))

    L_S = mean over valid tokens and layers of [c_attn·g_L + c_mlp·g_L·g_M] / (c_attn + c_mlp)

With ``router_enabled=False`` all gates are exactly 1 and no router runs: the forward is the
plain model (test T-EQ). Batches must be right-padded (``attention_mask`` 1 = token, 0 = pad);
padding only follows real tokens, so the causal attention and the EMA never mix it into them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from lagate.engine.flops import LayerCosts, avg_context_length
from lagate.models.adapter import DecoderAdapter
from lagate.routing.ema import ema_scan
from lagate.routing.router import DualHeadRouter, RouterConfig


@dataclass
class GatedOutput:
    logits: torch.Tensor                 # (B, T, V)
    g_layer: torch.Tensor | None         # (B, T, L) g_L; None when the router is disabled
    g_mlp: torch.Tensor | None           # (B, T, L) g_M
    sparsity_loss: torch.Tensor | None   # scalar L_S
    ce_loss: torch.Tensor | None         # scalar, if labels were given
    loss: torch.Tensor | None            # ce_loss + λ·L_S, if labels were given


class LookaheadGatedModel(nn.Module):
    """Wraps a HF Llama/Qwen2 causal LM; adds one DualHeadRouter per layer."""

    def __init__(self, hf_model: nn.Module, router_cfg: RouterConfig, train_context_length: float | None = None):
        super().__init__()
        self.adapter = DecoderAdapter(hf_model)
        self.router_cfg = router_cfg
        dims = self.adapter.dims
        eps = hf_model.config.rms_norm_eps
        self.routers = nn.ModuleList(DualHeadRouter(dims.hidden, router_cfg, eps) for _ in range(dims.num_layers))
        self.routers.to(next(hf_model.parameters()).dtype)
        self.costs = LayerCosts(dims, router_cfg.bottleneck)
        self.train_context_length = train_context_length  # n_avg for c_attn; None = from the batch
        self.router_enabled = True
        self.layer_hooks: list[Callable[[dict], None]] = []  # P1-02 gate logging

    @property
    def hf_model(self) -> nn.Module:
        return self.adapter.model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sparsity_weight: float = 0.0,
    ) -> GatedOutput:
        ad = self.adapter
        B, T = input_ids.shape
        valid = None if attention_mask is None else attention_mask.bool()
        if valid is not None and not _is_right_padded(valid):
            raise ValueError("attention_mask must be right-padded (pads only after the last token)")

        h = ad.embed(input_ids)
        pos = torch.arange(T, device=input_ids.device)
        cos, sin = ad.rope(h, pos)
        g_layers, g_mlps = [], []
        for layer in range(ad.dims.num_layers):
            n = ad.input_norm(layer, h)
            k, v = ad.kv(layer, n, cos, sin)
            attn = ad.attend(layer, ad.q(layer, n, cos, sin), k, v, pos, pos, key_valid=valid)
            if self.router_enabled:
                router = self.routers[layer]
                mu = ema_scan(n, self.router_cfg.alpha, chunk_size=self.router_cfg.ema_chunk_size)
                g_l, z = router.layer_gate(n, mu)
                g_m = router.mlp_gate(z, attn)
                h = h + g_l.unsqueeze(-1) * attn
                h = h + (g_l * g_m).unsqueeze(-1) * ad.mlp(layer, ad.post_norm(layer, h))
                g_layers.append(g_l)
                g_mlps.append(g_m)
            else:
                g_l = g_m = None
                h = h + attn
                h = h + ad.mlp(layer, ad.post_norm(layer, h))
            for hook in self.layer_hooks:
                hook({"layer": layer, "phase": "train", "positions": pos, "input_ids": input_ids,
                      "g_layer": g_l, "g_mlp": g_m, "valid": valid})
        logits = ad.final_logits(h)

        g_layer = g_mlp = l_s = None
        if self.router_enabled:
            g_layer, g_mlp = torch.stack(g_layers, -1), torch.stack(g_mlps, -1)
            l_s = self.sparsity_loss(g_layer, g_mlp, valid)
        ce = loss = None
        if labels is not None:
            ce = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(), labels[:, 1:].reshape(-1), ignore_index=-100
            )
            loss = ce if l_s is None else ce + sparsity_weight * l_s
        return GatedOutput(logits, g_layer, g_mlp, l_s, ce, loss)

    def sparsity_loss(self, g_layer: torch.Tensor, g_mlp: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        """Compute-weighted expected fraction of skippable FLOPs used (ARCHITECTURE §4, D-17)."""
        if self.train_context_length is not None:
            n_avg = self.train_context_length
        else:
            lengths = valid.sum(-1).float() if valid is not None else torch.full((g_layer.shape[0],), float(g_layer.shape[1]))
            n_avg = float(avg_context_length(lengths.mean().item()))
        c_attn, c_mlp = self.costs.sparsity_weights(n_avg)
        used = (c_attn * g_layer + c_mlp * g_layer * g_mlp) / (c_attn + c_mlp)  # (B, T, L)
        if valid is None:
            return used.mean()
        w = valid.unsqueeze(-1).to(used.dtype).expand_as(used)
        return (used * w).sum() / w.sum()

    def router_parameters(self):
        return self.routers.parameters()


def _is_right_padded(valid: torch.Tensor) -> bool:
    # Once a row hits a pad, it must stay padded: the mask is non-increasing along T.
    return bool((valid[:, 1:].int() <= valid[:, :-1].int()).all())
