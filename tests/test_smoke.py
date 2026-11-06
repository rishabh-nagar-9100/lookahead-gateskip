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


# GSM8K protocols from docs/DECISIONS.md: D-11 is primary, D-08 is the secondary dense sanity check.
GSM8K_PROTOCOLS = {
    "D-11": {"tasks": ["gsm8k"], "num_fewshot": 0, "apply_chat_template": False,
             "fewshot_as_multiturn": False, "primary_metric": "exact_match,strict-match"},
    "D-08": {"tasks": ["gsm8k_cot_llama"], "num_fewshot": 8, "apply_chat_template": True,
             "fewshot_as_multiturn": True, "primary_metric": "exact_match,flexible-extract"},
}


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "configs" / "eval").glob("*.yaml")))
def test_eval_configs_follow_protocol(path):
    """Eval configs must encode the GSM8K protocol they declare (docs/EXPERIMENTS.md, DECISIONS)."""
    cfg = load_config(path)
    assert cfg["name"] == path.stem
    assert cfg["model"]["pretrained"]
    assert cfg["protocol"] in GSM8K_PROTOCOLS, f"unknown protocol {cfg.get('protocol')}"
    ev = cfg["eval"]
    for key, expected in GSM8K_PROTOCOLS[cfg["protocol"]].items():
        assert ev[key] == expected, f"{path.name}: eval.{key}={ev[key]!r}, expected {expected!r}"
    assert ev["gen_kwargs"]["do_sample"] is False
    assert ev["gen_kwargs"]["max_gen_toks"] == 512


def _git(repo, *args):
    import subprocess

    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo, check=True, capture_output=True,
    )


def test_git_dirty_ignores_only_untracked_results(tmp_path):
    from lagate.utils import git_info

    _git(tmp_path, "init", "-q")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "init")
    info = git_info(tmp_path)
    assert info["commit"] and info["dirty"] is False

    # A run's own untracked output under results/ does not make the tree dirty.
    (tmp_path / "results" / "run1").mkdir(parents=True)
    (tmp_path / "results" / "run1" / "meta.json").write_text("{}")
    assert git_info(tmp_path)["dirty"] is False

    # An untracked source file does.
    (tmp_path / "src" / "new.py").write_text("y = 2\n")
    assert git_info(tmp_path)["dirty"] is True
    (tmp_path / "src" / "new.py").unlink()

    # So does a modified tracked file.
    (tmp_path / "src" / "a.py").write_text("x = 2\n")
    assert git_info(tmp_path)["dirty"] is True
