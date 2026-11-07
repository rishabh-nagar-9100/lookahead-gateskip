"""Training loop shared by every method (D-16: one pipeline for ours, the GateSkip port and Dense-FT).

    method: lagate    LookaheadGatedModel (EMA + dual-head router), loss CE + λ·L_S
            gateskip  GateSkipModel (official recipe, D-18), loss CE + w·L2(gates)
            dense     LookaheadGatedModel with the router off (Dense-FT reference, D-15)
    finetune: full | lora | router_only   (D-13 is still open, so all three are supported)

Real models (P3): ``data.config`` points at the D-06 chat data (``lagate.training.chat_data``),
``model.dtype: auto`` picks bf16 on L4/A100 (fp16 on a T4, where training is not planned), and
``train.precision: autocast`` keeps trainable weights (LoRA, routers) in fp32 and runs the forward
under autocast. ``checkpoint`` saves the trainable state, optimiser, scheduler, RNG and log every
``every_minutes`` to ``<checkpoint.root>/<name>_s<seed>/latest.pt`` and a restarted run resumes
from it (Colab sessions end without warning). ``export: hf`` writes a merged Hugging Face model
(Dense-FT, for lm-eval) at the end.

After training, ``smoke_eval`` runs the deployment path end to end: a soft-gate calibration pass
on the calibration split, empirical-CDF thresholds for one skip fraction (``lagate.calibration``),
then the bs=1 tiered engine on held-out sequences, reporting continuation accuracy and the
measured compute fraction. That calibration is the simple per-gate version, not P3-04's
(τ_L, τ_M) budget search.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
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
from lagate.utils import REPO_ROOT, load_config, resolve_device, resolve_dtype_name, save_json

METHODS = ("lagate", "gateskip", "dense")


# ----- construction ------------------------------------------------------------------------------
RECIPE_BLOCKS = ("finetune", "data", "train")


def resolve_recipe(cfg: dict) -> dict:
    """D-13 condition 1: one shared fine-tuning recipe for Dense-FT, the GateSkip port and ours.

    A config with ``recipe: <path>`` takes its ``finetune``, ``data`` and ``train`` blocks from that
    file and may not set any of them itself, so the three methods cannot drift apart. Method-specific
    settings (router config, sparsity weight, ``method.router_lr``) live under ``method``. Returns a
    new dict with the blocks filled in and ``recipe_file`` recording the source; resolving twice is a
    no-op.
    """
    if "recipe" not in cfg:
        return cfg
    clash = [b for b in RECIPE_BLOCKS if b in cfg]
    if clash:
        raise ValueError(f"{cfg.get('name')}: a recipe config may not set {clash}; change {cfg['recipe']} instead "
                         "(it applies to every method, D-13)")
    recipe = load_config(REPO_ROOT / cfg["recipe"])
    missing = [b for b in RECIPE_BLOCKS if b not in recipe]
    if missing:
        raise ValueError(f"recipe {cfg['recipe']} lacks {missing}")
    out = {k: v for k, v in cfg.items() if k != "recipe"}
    out.update({b: recipe[b] for b in RECIPE_BLOCKS}, recipe_file=cfg["recipe"])
    return out


def resolve_data_cfg(data: dict) -> dict:
    """``data.config`` (e.g. configs/data/d06_chat.yaml) merged with the run's own data keys."""
    if "config" not in data:
        return data
    return {**load_config(REPO_ROOT / data["config"]), **{k: v for k, v in data.items() if k != "config"}}


def build_hf_model(model_cfg: dict, seed: int, device: str = "cpu") -> torch.nn.Module:
    if "tiny" in model_cfg:
        t = dict(model_cfg["tiny"])
        m = build_tiny(t.pop("arch"), seed=seed, **t)
        if "dtype" in model_cfg:  # e.g. bf16 tiny models for the precision tests
            m = m.to(getattr(torch, resolve_dtype_name(model_cfg["dtype"], device)))
        return m
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, resolve_dtype_name(model_cfg.get("dtype", "float32"), device))
    return AutoModelForCausalLM.from_pretrained(model_cfg["pretrained"], dtype=dtype,
                                                attn_implementation=model_cfg.get("attn_implementation", "sdpa"))


def build_tokenizer(model_cfg: dict):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_cfg.get("tokenizer", model_cfg["pretrained"]))


def apply_finetune_mode(hf_model: torch.nn.Module, ft: dict):
    """Returns the PeftModel for LoRA (needed to merge at export), else None."""
    mode = ft["mode"]
    if mode == "full":
        hf_model.requires_grad_(True)
    elif mode == "router_only":
        hf_model.requires_grad_(False)
    elif mode == "lora":
        from peft import LoraConfig, get_peft_model

        lc = LoraConfig(r=ft["lora_r"], lora_alpha=ft["lora_alpha"], lora_dropout=ft.get("lora_dropout", 0.0),
                        target_modules=list(ft["lora_target_modules"]), bias="none")
        return get_peft_model(hf_model, lc)  # replaces the Linear modules in place and freezes the base weights
    else:
        raise ValueError(f"unknown finetune mode {mode!r}")
    return None


def build_method(cfg: dict, hf_model: torch.nn.Module, special_ids: list[int] | None = None) -> torch.nn.Module:
    """``special_ids``: pad / end-of-turn ids of a real tokenizer (GateSkip forces their importance to 1)."""
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
            extra = set(special_ids) if special_ids is not None else {PAD_ID, EOS_ID}
            gc.invalid_token_ids = sorted(set(gc.invalid_token_ids) | extra)
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


def build_optimizer(model: torch.nn.Module, train: dict, method: dict | None = None) -> torch.optim.Optimizer:
    """Backbone (LoRA / full) at ``train.lr``; router or gate parameters at ``method.router_lr``
    (falls back to ``train.router_lr`` for the older smoke configs)."""
    router = [p for p in model.router_parameters() if p.requires_grad]
    rid = {id(p) for p in router}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in rid]
    router_lr = (method or {}).get("router_lr", train.get("router_lr"))
    if router and router_lr is None:
        raise ValueError("trainable router/gate parameters but no method.router_lr")
    groups = [g for g in ({"params": backbone, "lr": train["lr"]}, {"params": router, "lr": router_lr}) if g["params"]]
    return torch.optim.AdamW(groups, weight_decay=train.get("weight_decay", 0.0))


# ----- checkpoints (resume across Colab sessions) -------------------------------------------------
def checkpoint_dir(cfg: dict, seed: int) -> Path:
    """Stable per (config name, seed), so a restarted run finds its own checkpoint.

    Root: ``checkpoint.root`` (relative to the repo), else $LAGATE_CHECKPOINT_ROOT, else checkpoints/
    (git-ignored; on Colab a symlink to Google Drive).
    """
    root = cfg.get("checkpoint", {}).get("root") or os.environ.get("LAGATE_CHECKPOINT_ROOT") or "checkpoints"
    root_p = Path(root) if Path(root).is_absolute() else REPO_ROOT / root
    return root_p / f"{cfg['name']}_s{seed}"


_NOT_TRAINING_KEYS = ("checkpoint", "smoke_eval", "export", "device", "save_checkpoint")


def config_digest(cfg: dict) -> str:
    """Hash of everything that changes training; resuming under a different config is refused."""
    core = {k: v for k, v in cfg.items() if k not in _NOT_TRAINING_KEYS}
    return hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in names or "routers" in k or "gates" in k}


def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])
    torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(path: Path, model, opt, sched, step: int, log: list[dict], cfg: dict, elapsed: float) -> None:
    """Atomic write (tmp + rename): a session killed mid-save leaves the previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"step": step, "trainable": trainable_state(model), "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "log": log, "rng": _rng_state(),
                "config_digest": config_digest(cfg), "train_seconds": elapsed}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, model, opt, sched, cfg: dict) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["config_digest"] != config_digest(cfg):
        raise RuntimeError(f"{path} was written with a different training config; move it away or rename the run")
    missing = set(ck["trainable"]) - set(model.state_dict())
    if missing:
        raise RuntimeError(f"checkpoint keys not in the model: {sorted(missing)[:5]}")
    model.load_state_dict(ck["trainable"], strict=False)
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    _set_rng_state(ck["rng"])
    return ck


# ----- training ------------------------------------------------------------------------------------
def setup(cfg: dict, seed: int, device: str):
    """Model, optimiser, scheduler, batch iterator, chat dataset (or None) and the PeftModel (or None)."""
    cfg = resolve_recipe(cfg)
    data_cfg = resolve_data_cfg(cfg["data"])
    hf = build_hf_model(cfg["model"], seed, device).to(device)
    chat = None
    if data_cfg["source"] == "chat":
        from lagate.training.chat_data import ChatDataset

        chat = ChatDataset(data_cfg, build_tokenizer(cfg["model"]))
    peft_model = apply_finetune_mode(hf, cfg["finetune"])
    model = build_method(cfg, hf, chat.special_ids if chat else None).to(device).train()
    tr = cfg["train"]
    if tr.get("precision", "fp32") == "autocast":
        for p in model.parameters():  # LoRA / router / full-FT weights in fp32; frozen weights keep the model dtype
            if p.requires_grad:
                p.data = p.data.float()
    # D-13 condition 2: trainable parameters are always stored in fp32 (bf16/fp16 only inside autocast).
    # A bf16 base without `precision: autocast` would train bf16 weights; refuse instead.
    low = sorted({str(p.dtype) for p in model.parameters() if p.requires_grad and p.dtype != torch.float32})
    if low:
        raise ValueError(f"trainable parameters in {low}; set train.precision: autocast (D-13)")
    opt = build_optimizer(model, tr, cfg["method"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(tr))
    batches = make_batches(data_cfg, hf.config.vocab_size, seed, "train", chat=chat)
    return model, opt, sched, batches, chat, peft_model


def _autocast(device: str, tr: dict, model_dtype: torch.dtype):
    if tr.get("precision", "fp32") != "autocast" or model_dtype == torch.float32:
        return torch.autocast(device_type=device.split(":")[0], enabled=False)
    if model_dtype == torch.float16:  # would need loss scaling; T4 sessions are for tests only (D-13, D-23)
        raise RuntimeError("fp16 autocast training is not used in this project: train on an L4/A100 (bf16)")
    return torch.autocast(device_type=device.split(":")[0], dtype=model_dtype)


def train(cfg: dict, seed: int, device: str = "cpu", setup_out=None) -> tuple[torch.nn.Module, list[dict]]:
    cfg = resolve_recipe(cfg)
    model, opt, sched, batches, chat, peft_model = setup_out or setup(cfg, seed, device)
    tr = cfg["train"]
    accum = tr.get("grad_accum", 1)
    weight = cfg["method"]["sparsity_weight"]
    ck_cfg = cfg.get("checkpoint", {})
    ck_path = checkpoint_dir(cfg, seed) / "latest.pt"
    every_s = 60 * ck_cfg.get("every_minutes", 30)
    log: list[dict] = []
    start, prior_s = 0, 0.0
    if ck_cfg.get("enabled", False) and ck_cfg.get("resume", True) and ck_path.exists():
        ck = load_checkpoint(ck_path, model, opt, sched, cfg)
        start, log, prior_s = ck["step"], ck["log"], ck["train_seconds"]
        for _ in range(start * accum):  # same data order as an uninterrupted run
            next(batches)
        print(f"[train] resumed from {ck_path} at step {start}")
    # autocast dtype: train.autocast_dtype, else the model's load dtype (auto -> bf16 on L4/A100)
    model_dtype = getattr(torch, resolve_dtype_name(tr.get("autocast_dtype") or cfg["model"].get("dtype", "float32"), device))
    t0 = time.perf_counter()
    last_save = t0
    for step in range(start, tr["steps"]):
        if isinstance(model, GateSkipModel):
            model.global_step = step
        opt.zero_grad(set_to_none=True)
        stats: dict[str, float] = {}
        for _ in range(accum):
            b = next(batches)
            with _autocast(device, tr, model_dtype):
                out = model(b.input_ids.to(device), attention_mask=b.attention_mask.to(device),
                            labels=b.labels.to(device), sparsity_weight=weight)
            (out.loss / accum).backward()
            _accumulate(stats, _step_stats(out, b.attention_mask.to(device)))
        trainable = [p for p in model.parameters() if p.requires_grad]
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, tr["max_grad_norm"]).item()
        opt.step()
        sched.step()
        row = {"step": step, **{k: v / accum for k, v in stats.items()},
               "grad_norm": gnorm, "lr": sched.get_last_lr()[0]}
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError(f"non-finite training stats at step {step}: {row}")
        log.append(row)
        if step % tr.get("log_every", 1) == 0 or step == tr["steps"] - 1:
            print("[train] " + "  ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        now = time.perf_counter()
        if ck_cfg.get("enabled", False) and (now - last_save >= every_s or step + 1 in ck_cfg.get("at_steps", [])):
            save_checkpoint(ck_path, model, opt, sched, step + 1, log, cfg, prior_s + now - t0)
            last_save = now
            print(f"[train] checkpoint step {step + 1} -> {ck_path}")
    total_s = prior_s + time.perf_counter() - t0
    if ck_cfg.get("enabled", False):
        save_checkpoint(ck_path, model, opt, sched, tr["steps"], log, cfg, total_s)
    if log:
        log[-1]["train_seconds"] = total_s
        log[-1]["resumed_at_step"] = start
    model._peft_ref = peft_model  # plain attribute: not a submodule (see export_hf)
    model._chat = chat
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
def collect_calibration_gates(model, cfg: dict, seed: int, device: str, n_batches: int | None = None,
                              chat=None) -> dict[str, np.ndarray]:
    """Soft gates on the calibration split, no skipping. Returns per-gate (N_valid_tokens, L) arrays."""
    cfg = resolve_recipe(cfg)
    chat = chat if chat is not None else getattr(model, "_chat", None)
    batches = make_batches(resolve_data_cfg(cfg["data"]), model.hf_model.config.vocab_size, seed, "calibration", chat=chat)
    n_batches = n_batches if n_batches is not None else cfg["smoke_eval"]["calibration_batches"]
    acc: dict[str, list[np.ndarray]] = {}
    if isinstance(model, GateSkipModel):
        model.apply_train_skipping = False
    for _ in range(n_batches):
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
    torch.save(trainable_state(model), path)


def export_hf(model, out_dir: Path) -> None:
    """Merged Hugging Face model + tokenizer (Dense-FT, evaluated with lm-eval). LoRA is merged first."""
    peft_model = getattr(model, "_peft_ref", None)
    hf = peft_model.merge_and_unload() if peft_model is not None else model.hf_model
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out_dir)
    chat = getattr(model, "_chat", None)
    if chat is not None:
        chat.tokenizer.save_pretrained(out_dir)


def run(cfg: dict, seed: int, run_dir: Path, device: str = "cpu") -> dict[str, Any]:
    device = resolve_device(device)
    cfg = resolve_recipe(cfg)
    model, log = train(cfg, seed, device)
    metrics: dict[str, Any] = {"method": cfg["method"]["name"], "finetune": cfg["finetune"]["mode"], "train_log": log}
    first, last = log[0], log[-1]
    metrics["summary"] = {k: {"first": first[k], "last": last[k]} for k in first if k not in ("step", "lr")}
    if getattr(model, "_chat", None) is not None:
        metrics["data"] = model._chat.stats()
    if "smoke_eval" in cfg:
        metrics["smoke_eval"] = smoke_eval(model, cfg, seed, device)
    ck_dir = checkpoint_dir(cfg, seed)
    if cfg.get("save_checkpoint"):
        save_trainable(model, ck_dir / "trainable.pt")
        metrics["checkpoint"] = str(ck_dir / "trainable.pt")
    if cfg.get("export") == "hf":
        export_hf(model, ck_dir / "hf")
        metrics["hf_export"] = str(ck_dir / "hf")
    save_json(metrics, run_dir / "metrics.json")
    return metrics
