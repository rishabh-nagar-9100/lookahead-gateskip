"""Shared fixtures: tiny random Llama and Qwen2 models (CPU, fp32, nothing downloaded)."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

TINY = dict(
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


def build_tiny(kind: str, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    if kind == "llama":
        model = LlamaForCausalLM(LlamaConfig(**TINY))
    elif kind == "qwen2":
        model = Qwen2ForCausalLM(Qwen2Config(**TINY, use_sliding_window=False))
    else:
        raise ValueError(kind)
    # Random norm weights and biases (HF initialises them to 1 / 0), so tests exercise every parameter.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.uniform_(0.5, 1.5)
            elif name.endswith(".bias"):
                p.normal_(0.0, 0.1)
    return model.eval()


@pytest.fixture(params=["llama", "qwen2"])
def tiny_model(request):
    return build_tiny(request.param)
