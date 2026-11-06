"""P2-01: per-layer access to a Hugging Face Llama / Qwen2 causal LM.

Our training forward (P2-04) and inference engine (P2-05) run their own decoder loop over the
model's submodules instead of calling ``LlamaDecoderLayer.forward``, because they need the pieces
separately: RMSNorm, K,V projection with RoPE (all tiers), Q + attention + O-proj (tiers 1–2) and
the MLP (tier 1). Weights stay in the HF modules, so PEFT/LoRA and checkpoint loading work as usual.

Attention is the eager formulation (fp32 softmax, like HF ``eager_attention_forward``). The
causal mask is built from absolute positions, so the same code serves prefill, cached decode and
a subset of query rows (``attend`` takes the queries' positions).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

SUPPORTED = ("llama", "qwen2")


@dataclass(frozen=True)
class Dims:
    num_layers: int
    hidden: int
    intermediate: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int

    @classmethod
    def from_hf_config(cls, c) -> "Dims":
        head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
        return cls(c.num_hidden_layers, c.hidden_size, c.intermediate_size, c.num_attention_heads,
                   c.num_key_value_heads, head_dim, c.vocab_size)


class DecoderAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        cfg = model.config
        if cfg.model_type not in SUPPORTED:
            raise ValueError(f"unsupported model_type {cfg.model_type!r}; supported: {SUPPORTED}")
        if any(t != "full_attention" for t in (getattr(cfg, "layer_types", None) or [])):
            raise ValueError("sliding-window layers are not supported (Qwen2.5 runs without them)")
        self.model = model
        self.config = cfg
        self.dims = Dims.from_hf_config(cfg)
        self.scaling = self.dims.head_dim ** -0.5

    # ----- module access ---------------------------------------------------------------------
    @property
    def layers(self) -> nn.ModuleList:
        return self.model.model.layers

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.model.embed_tokens(input_ids)

    def final_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.model.lm_head(self.model.model.norm(h))

    def rope(self, h: torch.Tensor, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cos, sin for absolute ``positions`` (shape (B, T) or (T,))."""
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        return self.model.model.rotary_emb(h, position_ids=positions)

    def input_norm(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        return self.layers[layer].input_layernorm(h)

    def post_norm(self, layer: int, h: torch.Tensor) -> torch.Tensor:
        return self.layers[layer].post_attention_layernorm(h)

    # ----- attention pieces --------------------------------------------------------------------
    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n·d) -> (B, n, T, d)."""
        B, T, _ = x.shape
        return x.view(B, T, -1, self.dims.head_dim).transpose(1, 2)

    def kv(self, layer: int, n: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """K (with RoPE) and V for normalised hidden states ``n`` (B, T, H) -> (B, kv_heads, T, d)."""
        attn = self.layers[layer].self_attn
        k = self._heads(attn.k_proj(n))
        v = self._heads(attn.v_proj(n))
        cos_, sin_ = cos.unsqueeze(1), sin.unsqueeze(1)
        k = k * cos_ + _rotate_half(k) * sin_
        return k, v

    def q(self, layer: int, n: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        attn = self.layers[layer].self_attn
        q = self._heads(attn.q_proj(n))
        cos_, sin_ = cos.unsqueeze(1), sin.unsqueeze(1)
        return q * cos_ + _rotate_half(q) * sin_

    def attend(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        q_pos: torch.Tensor,
        k_pos: torch.Tensor,
        key_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Causal attention + O-proj. q: (B, nh, Tq, d); k, v: (B, kvh, Tk, d); positions 1-D.

        ``key_valid`` (B, Tk) bool masks padded keys. Returns the attention sub-layer output (B, Tq, H).
        """
        groups = self.dims.n_heads // self.dims.n_kv_heads
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        allowed = k_pos[None, :] <= q_pos[:, None]  # (Tq, Tk)
        allowed = allowed[None, None]
        if key_valid is not None:
            allowed = allowed & key_valid[:, None, None, :]
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        return self.layers[layer].self_attn.o_proj(out)

    def mlp(self, layer: int, x: torch.Tensor) -> torch.Tensor:
        return self.layers[layer].mlp(x)

    # ----- reference dense forward --------------------------------------------------------------
    def dense_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Plain dense forward through the adapter pieces (no router). Must equal HF logits (T-EQ)."""
        h = self.embed(input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)
        cos, sin = self.rope(h, pos)
        for layer in range(self.dims.num_layers):
            n = self.input_norm(layer, h)
            k, v = self.kv(layer, n, cos, sin)
            h = h + self.attend(layer, self.q(layer, n, cos, sin), k, v, pos, pos)
            h = h + self.mlp(layer, self.post_norm(layer, h))
        return self.final_logits(h)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


__all__ = ["DecoderAdapter", "Dims"]
