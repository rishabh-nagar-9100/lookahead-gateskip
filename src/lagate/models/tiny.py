"""Tiny random Llama / Qwen2 models for CPU tests and smoke runs (nothing downloaded)."""

from __future__ import annotations

from typing import Any

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

TINY_DEFAULTS: dict[str, Any] = dict(
    vocab_size=128,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,   # GQA, as both real models
    max_position_embeddings=256,
    initializer_range=0.2,   # larger than default so attention patterns are not uniform
    attn_implementation="eager",
)


def build_tiny(arch: str, seed: int = 0, randomize_norms_and_biases: bool = True, **overrides) -> torch.nn.Module:
    cfg = {**TINY_DEFAULTS, **overrides}
    torch.manual_seed(seed)
    if arch == "llama":
        model = LlamaForCausalLM(LlamaConfig(**cfg))
    elif arch == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**cfg, use_sliding_window=False))
    else:
        raise ValueError(f"unknown tiny arch {arch!r}")
    if randomize_norms_and_biases:  # HF initialises them to 1 / 0; tests should exercise every parameter
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name.endswith("norm.weight"):
                    p.uniform_(0.5, 1.5)
                elif name.endswith(".bias"):
                    p.normal_(0.0, 0.1)
    return model.eval()
