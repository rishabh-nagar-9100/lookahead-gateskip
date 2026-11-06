"""Shared training loop (P2-08 machinery): a few steps of every method / fine-tuning mode on tiny models."""

import copy
import math

import pytest
import torch

from lagate.training.data import PAD_ID, collate, make_batches
from lagate.training.loop import run, train
from lagate.utils import REPO_ROOT, load_config


def smoke_cfg(method: str, arch: str = "llama", steps: int = 3) -> dict:
    cfg = copy.deepcopy(load_config(REPO_ROOT / f"configs/train/smoke_{method}_{arch}.yaml"))
    cfg["train"]["steps"] = steps
    cfg["smoke_eval"].update(calibration_batches=1, n_sequences=2, new_tokens=3)
    return cfg


def test_collate_right_pads_and_masks_labels():
    b = collate([[5, 6, 7], [8]])
    assert b.input_ids.tolist() == [[5, 6, 7], [8, PAD_ID, PAD_ID]]
    assert b.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b.labels.tolist() == [[5, 6, 7], [8, -100, -100]]


def test_splits_differ_and_are_reproducible():
    d = {"source": "synthetic", "batch_size": 2, "min_len": 5, "max_len": 9}
    a = next(make_batches(d, 128, 0, "train")).input_ids
    assert torch.equal(a, next(make_batches(d, 128, 0, "train")).input_ids)
    assert not torch.equal(a, next(make_batches(d, 128, 0, "calibration")).input_ids[: a.shape[0], : a.shape[1]])


@pytest.mark.parametrize("method", ["lagate", "gateskip"])
@pytest.mark.parametrize("arch", ["llama", "qwen2"])
def test_run_end_to_end(method, arch, tmp_path):
    m = run(smoke_cfg(method, arch), seed=0, run_dir=tmp_path)
    assert len(m["train_log"]) == 3 and all(math.isfinite(r["loss"]) for r in m["train_log"])
    ev = m["smoke_eval"]["by_skip_frac"]
    assert ev["0.00"]["compute_frac"] > ev["0.35"]["compute_frac"]  # skipping saves compute in the engine
    assert (tmp_path / "metrics.json").exists()


def test_dense_ft_and_router_only_and_lora():
    cfg = smoke_cfg("lagate")
    cfg["method"]["name"] = "dense"
    model, log = train(cfg, seed=0)
    assert "sparsity" not in log[0] and not any(p.requires_grad for p in model.routers.parameters())

    cfg = smoke_cfg("lagate")
    cfg["finetune"] = {"mode": "router_only"}
    model, _ = train(cfg, seed=0)
    assert not any(p.requires_grad for p in model.hf_model.parameters())

    cfg = smoke_cfg("gateskip")
    cfg["finetune"] = {"mode": "lora", "lora_r": 4, "lora_alpha": 8, "lora_target_modules": ["q_proj", "v_proj", "up_proj"]}
    model, log = train(cfg, seed=0)
    trainable = {n for n, p in model.hf_model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert math.isfinite(log[-1]["loss"])
