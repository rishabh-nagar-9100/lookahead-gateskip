"""P2-01 adapter: our decoder loop over HF submodules reproduces HF logits (base of T-EQ)."""

import pytest
import torch

from lagate.models.adapter import DecoderAdapter


def test_dims(tiny_model):
    d = DecoderAdapter(tiny_model).dims
    assert (d.num_layers, d.hidden, d.intermediate, d.n_heads, d.n_kv_heads, d.head_dim) == (3, 64, 128, 4, 2, 16)


def test_dense_forward_equals_hf(tiny_model):
    ids = torch.randint(0, 128, (2, 23), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = tiny_model(ids).logits
        got = DecoderAdapter(tiny_model).dense_forward(ids)
    torch.testing.assert_close(got, ref, rtol=0, atol=1e-5)


def test_kv_matches_hf_cache(tiny_model):
    """K (with RoPE) and V from the adapter equal what HF writes to its own cache."""
    ids = torch.randint(0, 128, (1, 9), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        out = tiny_model(ids, use_cache=True)
        ad = DecoderAdapter(tiny_model)
        h = ad.embed(ids)
        pos = torch.arange(9)
        cos, sin = ad.rope(h, pos)
        for layer in range(ad.dims.num_layers):
            n = ad.input_norm(layer, h)
            k, v = ad.kv(layer, n, cos, sin)
            torch.testing.assert_close(k, out.past_key_values.layers[layer].keys, rtol=0, atol=1e-5)
            torch.testing.assert_close(v, out.past_key_values.layers[layer].values, rtol=0, atol=1e-5)
            h = h + ad.attend(layer, ad.q(layer, n, cos, sin), k, v, pos, pos)
            h = h + ad.mlp(layer, ad.post_norm(layer, h))


def test_rejects_sliding_window():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, use_sliding_window=True,
                      sliding_window=4, max_window_layers=0)
    with pytest.raises(ValueError):
        DecoderAdapter(Qwen2ForCausalLM(cfg))
