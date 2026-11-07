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


# ----- checkpoint / resume (Colab sessions end without warning) -----------------------------------
class _Killed(Exception):
    pass


def _killing(batches, n):
    for i, b in enumerate(batches):
        if i == n:
            raise _Killed
        yield b


@pytest.mark.parametrize("method", ["lagate", "gateskip"])
def test_resume_matches_uninterrupted_run(method, tmp_path):
    from lagate.training.loop import setup, trainable_state

    cfg = smoke_cfg(method, steps=6)
    cfg["train"]["grad_accum"] = 2
    cfg["checkpoint"] = {"enabled": True, "root": str(tmp_path / "a"), "every_minutes": 1e9, "at_steps": [3]}
    ref, ref_log = train(copy.deepcopy(cfg), seed=0)

    cfg["checkpoint"]["root"] = str(tmp_path / "b")
    torch.manual_seed(0)
    s = list(setup(cfg, 0, "cpu"))
    s[3] = _killing(s[3], 4 * 2)  # dies during step 4 (after the step-3 checkpoint)
    with pytest.raises(_Killed):
        train(cfg, seed=0, setup_out=tuple(s))
    assert (tmp_path / "b" / f"{cfg['name']}_s0" / "latest.pt").exists()

    torch.manual_seed(0)
    resumed, log = train(cfg, seed=0)  # fresh process state; resumes from step 3
    assert log[-1]["resumed_at_step"] == 3 and len(log) == 6
    for a, b in zip(ref_log, log):
        assert a["loss"] == pytest.approx(b["loss"], rel=1e-6, abs=1e-7)
    ra, rb = trainable_state(ref), trainable_state(resumed)
    assert ra.keys() == rb.keys()
    for k in ra:
        torch.testing.assert_close(ra[k], rb[k], rtol=1e-6, atol=1e-7)


def test_resume_refuses_a_different_config(tmp_path):
    cfg = smoke_cfg("lagate", steps=2)
    cfg["checkpoint"] = {"enabled": True, "root": str(tmp_path)}
    train(copy.deepcopy(cfg), seed=0)
    cfg["train"]["lr"] *= 2
    with pytest.raises(RuntimeError, match="different training config"):
        train(cfg, seed=0)


def test_autocast_bf16_with_fp32_trainables():
    cfg = smoke_cfg("lagate", steps=3)
    cfg["train"].update(precision="autocast", autocast_dtype="bfloat16")
    model, log = train(cfg, seed=0)
    assert all(math.isfinite(r["loss"]) for r in log)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)


def test_export_hf_merges_lora(tmp_path):
    from transformers import AutoModelForCausalLM

    from lagate.training.loop import export_hf

    cfg = smoke_cfg("lagate", steps=2)
    cfg["method"]["name"] = "dense"
    cfg["finetune"] = {"mode": "lora", "lora_r": 4, "lora_alpha": 8, "lora_target_modules": ["q_proj", "v_proj"]}
    model, _ = train(cfg, seed=0)
    ids = torch.tensor([[3, 4, 5, 6, 7]])
    with torch.no_grad():
        before = model(ids).logits
    export_hf(model, tmp_path / "hf")
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path / "hf").eval()
    assert not any("lora" in k for k in reloaded.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(reloaded(ids).logits, before, rtol=1e-4, atol=1e-4)
