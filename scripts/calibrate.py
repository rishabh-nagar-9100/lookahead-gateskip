"""Calibrate fixed thresholds for a trained gated model (ours or the GateSkip port) on the calibration split.

Usage:
    python scripts/calibrate.py --config configs/calibrate/lagate_qwen.yaml --seed 1

The config names the training config (``train_config``); the trained weights are the finished
checkpoint of that run and seed (``<checkpoint root>/<name>_s<seed>/latest.pt``, which must have
reached ``train.steps``). Soft gates are collected on ``n_batches`` calibration batches (D-06
held-out train rows, never test) with no skipping, then per-module empirical-CDF thresholds are
computed for each ``skip_fracs`` value (D-18; the same rule for both methods). Writes
results/<run_id>/{config.yaml, meta.json, metrics.json, thresholds.json}. The thresholds file is
small and committed; the gate arrays themselves are not saved.

This is the per-gate version used by the smoke runs. P3-04's joint (τ_L, τ_M) search for a target
compute budget (``lagate.calibration.ecdf.match_budget``) builds on the same gate arrays.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from dotenv import load_dotenv

from lagate.training.loop import checkpoint_dir, collect_calibration_gates, setup
from lagate.calibration.ecdf import ecdf_threshold
from lagate.routing.gateskip import GateSkipModel, calibrate_gateskip
from lagate.utils import REPO_ROOT, create_run_dir, load_config, resolve_device, resolve_dtype_name, save_json, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True, help="seed of the training run to calibrate")
    args = p.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    tcfg = load_config(REPO_ROOT / cfg["train_config"])
    set_seed(args.seed)
    device = resolve_device(cfg.get("device", tcfg.get("device", "auto")))
    ck_path = checkpoint_dir(tcfg, args.seed) / "latest.pt"
    run_dir = create_run_dir(cfg, args.seed, cfg["name"], extra_meta={"device": device, "checkpoint": str(ck_path)})
    print(f"[calibrate] run dir: {run_dir}  checkpoint: {ck_path}")

    model, _, _, _, chat, _ = setup(tcfg, args.seed, device)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    if ck["step"] < tcfg["train"]["steps"]:
        raise SystemExit(f"checkpoint at step {ck['step']} < {tcfg['train']['steps']}: training is not finished")
    model.load_state_dict(ck["trainable"], strict=False)
    model.eval()

    ac_dtype = getattr(torch, resolve_dtype_name(tcfg["train"].get("autocast_dtype") or tcfg["model"].get("dtype", "float32"), device))
    with torch.autocast(device_type=device.split(":")[0], dtype=ac_dtype,
                        enabled=tcfg["train"].get("precision") == "autocast"):
        gates = collect_calibration_gates(model, tcfg, args.seed, device, n_batches=cfg["n_batches"], chat=chat)
    thresholds = {}
    for frac in cfg["skip_fracs"]:
        if isinstance(model, GateSkipModel):
            thresholds[f"{frac:.2f}"] = calibrate_gateskip(gates, frac, model.cfg.skip_module_types)
        else:
            L = gates["g_layer"].shape[1]
            thresholds[f"{frac:.2f}"] = {
                "tau_layer": [ecdf_threshold(gates["g_layer"][:, l], frac) for l in range(L)],
                "tau_mlp": [ecdf_threshold(gates["g_mlp"][:, l], frac) for l in range(L)],
            }
    stats = {k: {"n_tokens": int(v.shape[0]), "mean_per_layer": np.asarray(v).mean(0).tolist()} for k, v in gates.items()}
    save_json(thresholds, run_dir / "thresholds.json")
    save_json({"method": tcfg["method"]["name"], "train_config": cfg["train_config"], "checkpoint_step": ck["step"],
               "gate_stats": stats, "skip_fracs": cfg["skip_fracs"]}, run_dir / "metrics.json")
    print(f"[calibrate] {sum(s['n_tokens'] for s in stats.values()) // max(1, len(stats))} tokens; "
          f"thresholds for {cfg['skip_fracs']} -> {run_dir / 'thresholds.json'}")


if __name__ == "__main__":
    main()
