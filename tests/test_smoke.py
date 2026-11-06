"""CPU smoke tests. Nothing here downloads a model or dataset."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pathlib import Path

import pytest
import torch
import yaml
from transformers import LlamaConfig, LlamaForCausalLM

from lagate.utils import create_run_dir, load_config, set_seed

REPO_ROOT = Path(__file__).resolve().parents[1]

VOCAB, HIDDEN, LAYERS, BATCH, SEQ = 128, 64, 2, 2, 8


@pytest.fixture
def tiny_llama() -> LlamaForCausalLM:
    set_seed(0)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(cfg).eval()


def test_tiny_llama_forward_shapes(tiny_llama):
    input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
    with torch.no_grad():
        out = tiny_llama(input_ids=input_ids, output_hidden_states=True)

    assert out.logits.shape == (BATCH, SEQ, VOCAB)
    assert len(out.hidden_states) == LAYERS + 1  # embeddings + one per layer
    for h in out.hidden_states:
        assert h.shape == (BATCH, SEQ, HIDDEN)
    assert torch.isfinite(out.logits).all()


def test_set_seed_is_reproducible():
    set_seed(123)
    a = torch.randn(4)
    set_seed(123)
    b = torch.randn(4)
    assert torch.equal(a, b)


def test_create_run_dir_writes_config_and_meta(tmp_path):
    cfg = {"name": "unit", "x": 1}
    run_dir = create_run_dir(cfg, seed=7, name="unit", results_root=tmp_path)
    assert run_dir.parent == tmp_path and run_dir.name.endswith("_unit_s7")
    assert yaml.safe_load((run_dir / "config.yaml").read_text()) == cfg
    meta = yaml.safe_load((run_dir / "meta.json").read_text())  # JSON is valid YAML
    assert meta["seed"] == 7
    assert {"commit", "dirty"} <= set(meta["git"])
    assert "torch" in meta["packages"]


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs" / "eval").glob("*.yaml")))
def test_eval_configs_follow_protocol(path):
    """Eval configs must encode the fixed GSM8K protocol from docs/EXPERIMENTS.md."""
    cfg = load_config(path)
    assert cfg["name"] == path.stem
    assert cfg["model"]["pretrained"]
    ev = cfg["eval"]
    assert ev["num_fewshot"] == 8
    assert ev["gen_kwargs"]["do_sample"] is False
    assert ev["gen_kwargs"]["max_gen_toks"] == 512
