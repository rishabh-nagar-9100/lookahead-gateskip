"""Training loop shared by every method (D-16: one pipeline for ours, the GateSkip port and Dense-FT).

    method: lagate    LookaheadGatedModel (EMA + dual-head router), loss CE + λ·L_S
            gateskip  GateSkipModel (official recipe, D-18), loss CE + w·L2(gates)
            dense     LookaheadGatedModel with the router off (Dense-FT reference, D-15)
    finetune: full | lora | router_only   (D-13 is still open, so all three are supported)

After training, ``smoke_eval`` runs the deployment path end to end: a soft-gate calibration pass
on the calibration split, empirical-CDF thresholds for one skip fraction (``lagate.calibration``),
then the bs=1 tiered engine on held-out sequences, reporting continuation accuracy and the
measured compute fraction. That calibration is the simple per-gate version, not P3-04's
(τ_L, τ_M) budget search.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from lagate.calibration.ecdf import ecdf_threshold
from lagate.engine.tiered import DensePolicy, EngineConfig, LagatePolicy, TieredEngine
from lagate.models.gated import LookaheadGatedModel
from lagate.models.tiny import build_tiny
from lagate.routing.gateskip import GateSkipConfig, GateSkipModel, GateSkipPolicy, calibrate_gateskip
from lagate.routing.router import RouterConfig
from lagate.training.data import EOS_ID, PAD_ID, make_batches, synthetic_sequence
from lagate.utils import REPO_ROOT, load_config, save_json

METHODS = ("lagate", "gateskip", "dense")


# ----- construction ------------------------------------------------------------------------------
def build_hf_model(model_cfg: dict, seed: int) -> torch.nn.Module:
    if "tiny" in model_cfg:
        t = dict(model_cfg["tiny"])
        return build_tiny(t.pop("arch"), seed=seed, **t)
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, model_cfg.get("dtype", "float32"))
    return AutoModelForCausalLM.from_pretrained(model_cfg["pretrained"], dtype=dtype)


def apply_finetune_mode(hf_model: torch.nn.Module, ft: dict) -> None:
    mode = ft["mode"]
    if mode == "full":
        hf_model.requires_grad_(True)
    elif mode == "router_only":
        hf_model.requires_grad_(False)
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft.get("lora_dropout", 0.0),
                        target_modules=list(ft["lora_target_modules"]), bias="none")
        get_peft_model(hf_model, lc)  # replaces the Linear modules in place and freezes the base weights
    else:
        raise ValueError(f"unknown finetune mode {mode!r}")


def build_method(cfg: dict, hf_model: torch.nn.Module) -> torch.nn.Module:
    m = cfg["method"]
    name = m["name"]
    if name in ("lagate", "dense"):
        rc = RouterConfig.from_dict({**load_config(REPO_ROOT / m["router_config"]), **m.get("router_overrides", {})})
        model = LookaheadGatedModel(hf_model, rc, train_context_length=m.get("train_context_length"))
        model.router_enabled = name == "lagate"
        if name == "dense":
            model.routers.requires_grad_(False)
        return model
    if name == "gateskip":
        gc = GateSkipConfig.from_dict({**load_config(REPO_ROOT / m["router_config"]), **m.get("router_overrides", {})})
        if m.get("invalid_pad_eos", True):
            gc.invalid_token_ids = sorted(set(gc.invalid_token_ids) | {PAD_ID, EOS_ID})
        return GateSkipModel(hf_model, gc)
    raise ValueError(f"unknown method {name!r}; expected one of {METHODS}")


def lr_lambda(train: dict):
    warm, decay = train.get("warmup_steps", 0), train.get("lr_decay_steps")

    def f(step: int) -> float:
        if warm and step < warm:
            return (step + 1) / warm
        if decay:
            return max(0.0, 1 - (step - warm) / max(1, decay - warm))
        return 1.0

    return f


def build_optimizer(model: torch.nn.Module, train: dict) -> torch.optim.Optimizer:
    router = [p for p in model.router_parameters() if p.requires_grad]
    rid = {id(p) for p in router}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in rid]
    groups = [g for g in ({"params": backbone, "lr": train["lr"]}, {"params": router, "lr": train["router_lr"]}) if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=train.get("weight_decay", 0.0))


# ----- training ------------------------------------------------------------------------------------
def train(cfg: dict, seed: int, device: str = "cpu") -> tuple[torch.nn.Module, list[dict]]:
    hf = build_hf_model(cfg["model"], seed).to(device)
    apply_finetune_mode(hf, cfg["finetune"])
    model = build_method(cfg, hf).to(device).train()
    tr = cfg["train"]
    opt = build_optimizer(model, tr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tr))
    batches = make_batches(cfg["data"], hf.config.vocab_size, seed, "train")
    weight = cfg["method"]["sparsity_weight"]
    log = []
    t0 = time.perf_counter()
    for step in range(tr["steps"]):
        if isinstance(model, GateSkipModel):
            model.global_step = step
        opt.zero_grad(set_to_none=True)
        stats: dict[str, float] = {}
        for _ in range(tr.get("grad_accum", 1)):
            b = next(batches)
            out = model(b.input_ids.to(device), attention_mask=b.attention_mask.to(device),
                        labels=b.labels.to(device), sparsity_weight=weight)
            (out.loss / tr.get("grad_accum", 1)).backward()
            _accumulate(stats, _step_stats(out, b.attention_mask.to(device)))
        trainable = [p for p in model.parameters() if p.requires_grad]
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, tr["max_grad_norm"]).item()
        opt.step()
        sched.step()
        row = {"step": step, **{k: v / tr.get("grad_accum", 1) for k, v in stats.items()},
               "grad_norm": gnorm, "lr": sched.get_last_lr()[0]}
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError(f"non-finite training stats at step {step}: {row}")
        log.append(row)
        if step % tr.get("log_every", 1) == 0 or step == tr["steps"] - 1:
            print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
    log[-1]["train_seconds"] = time.perf_counter() - t0
    return model.eval(), log


def _step_stats(out, mask: torch.Tensor) -> dict[str, float]:
    s = {"loss": out.loss.item(), "ce": out.ce_loss.item()}
    if out.sparsity_loss is not None:
        s["sparsity"] = out.sparsity_loss.item()
    valid = mask.bool()
    if hasattr(out, "g_layer") and out.g_layer is not None:
        s["mean_g_layer"] = out.g_layer[valid].mean().item()
        s["mean_g_mlp"] = out.g_mlp[valid].mean().item()
    if hasattr(out, "skipped"):
        for m in ("attn", "mlp"):
            s[f"train_skip_{m}"] = out.skipped[m][valid].float().mean().item()
            s[f"mean_imp_{m}"] = out.importance[m][valid].mean().item()
    return s


def _accumulate(acc: dict, new: dict) -> None:
    for k, v in new.items():
        acc[k] = acc.get(k, 0.0) + v


# ----- smoke evaluation: calibrate -> bs=1 engine --------------------------------------------------
@torch.no_grad()
def collect_calibration_gates(model, cfg: dict, seed: int, device: str) -> dict[str, np.ndarray]:
    """Soft gates on the calibration split, no skipping. Returns per-gate (N_valid_tokens, L) arrays."""
    batches = make_batches(cfg["data"], model.hf_model.config.vocab_size, seed, "calibration")
    acc: dict[str, list[np.ndarray]] = {}
    if isinstance(model, GateSkipModel):
        model.apply_train_skipping = False
    for _ in range(cfg["smoke_eval"]["calibration_batches"]):
        b = next(batches)
        out = model(b.input_ids.to(device), attention_mask=b.attention_mask.to(device))
        valid = b.attention_mask.bool().to(device)
        if isinstance(model, GateSkipModel):
            gates = {"attn": out.importance["attn"], "mlp": out.importance["mlp"]}
        elif out.g_layer is not None:
            gates = {"g_layer": out.g_layer, "g_mlp": out.g_mlp}
        else:
            gates = {}
        for k, v in gates.items():
            acc.setdefault(k, []).append(v[valid].cpu().numpy())
    if isinstance(model, GateSkipModel):
        model.apply_train_skipping = True
    return {k: np.concatenate(v) for k, v in acc.items()}


def build_engine(model, gates: dict[str, np.ndarray], skip_frac: float, engine_cfg: EngineConfig) -> tuple[TieredEngine, dict]:
    if isinstance(model, GateSkipModel):
        tau = calibrate_gateskip(gates, skip_frac, model.cfg.skip_module_types)
        return TieredEngine(model.hf_model, GateSkipPolicy(model.gates, tau), engine_cfg), tau
    if not model.router_enabled:
        return TieredEngine(model.hf_model, DensePolicy(), engine_cfg), {}
    L = gates["g_layer"].shape[1]
    tau_l = [ecdf_threshold(gates["g_layer"][:, l], skip_frac) for l in range(L)]
    tau_m = [ecdf_threshold(gates["g_mlp"][:, l], skip_frac) for l in range(L)]
    eng = TieredEngine.from_gated_model(model, tau_l, tau_m, engine_cfg)
    return eng, {"tau_layer": tau_l, "tau_mlp": tau_m}


@torch.no_grad()
def smoke_eval(model, cfg: dict, seed: int, device: str = "cpu") -> dict[str, Any]:
    se = cfg["smoke_eval"]
    engine_cfg = EngineConfig.from_dict(load_config(REPO_ROOT / se["engine_config"]))
    gates = collect_calibration_gates(model, cfg, seed, device)
    vocab = model.hf_model.config.vocab_size
    gen = torch.Generator().manual_seed(seed + 3_000_017)
    seqs = [synthetic_sequence(gen, vocab, se["prompt_len"] + se["new_tokens"] + 1)[:-1] for _ in range(se["n_sequences"])]
    results = {}
    for p in [0.0] + list(se["skip_fracs"]):
        eng, tau = build_engine(model, gates, p, engine_cfg)
        correct = total = 0
        for s in seqs:
            prompt = torch.tensor([s[: se["prompt_len"]]], device=device)
            out, _ = eng.generate(prompt, se["new_tokens"])
            target = torch.tensor(s[se["prompt_len"] : se["prompt_len"] + out.shape[1]], device=device)
            correct += int((out[0] == target).sum())
            total += out.shape[1]
        st = eng.stats
        results[f"{p:.2f}"] = {
            "skip_frac": p, "continuation_acc": correct / total, "compute_frac": st.compute_frac,
            "tier_counts": dict(st.tier_counts), "thresholds": tau,
        }
    return {"by_skip_frac": results, "calibration_tokens": int(next(iter(gates.values())).shape[0]) if gates else 0}


def save_trainable(model: torch.nn.Module, path: Path) -> None:
    """Router/gate + trainable backbone weights only (checkpoints/ is git-ignored)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    torch.save({k: v for k, v in model.state_dict().items() if k in names or "routers" in k or "gates" in k}, path)


def run(cfg: dict, seed: int, run_dir: Path, device: str = "cpu") -> dict[str, Any]:
    model, log = train(cfg, seed, device)
    metrics: dict[str, Any] = {"method": cfg["method"]["name"], "finetune": cfg["finetune"]["mode"], "train_log": log}
    first, last = log[0], log[-1]
    metrics["summary"] = {k: {"first": first[k], "last": last[k]} for k in first if k not in ("step", "lr")}
    if "smoke_eval" in cfg:
        metrics["smoke_eval"] = smoke_eval(model, cfg, seed, device)
    if cfg.get("save_checkpoint"):
        ck = REPO_ROOT / "checkpoints" / run_dir.name / "trainable.pt"
        save_trainable(model, ck)
        metrics["checkpoint"] = str(ck.relative_to(REPO_ROOT))
    save_json(metrics, run_dir / "metrics.json")
    return metrics
