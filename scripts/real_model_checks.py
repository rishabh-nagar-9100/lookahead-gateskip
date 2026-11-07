"""Real-model checks (no downloads; uses the local HF cache).

    1. Dims from config.json vs the dims our FLOP accounting uses.
    2. T-EQ at full scale: our adapter's dense forward and the dense engine vs HF logits, per dtype.
    3. Dense engine greedy decoding vs HF ``generate`` on GSM8K prompts (chat format, D-15).
    4. P5-01 first pass: bs=1 ms/token, HF generate vs dense engine; warm-up excluded, median of N runs.

Usage:
    python scripts/real_model_checks.py --config configs/checks/qwen_mps.yaml --seed 0
"""

from __future__ import annotations

import argparse
import gc
import os
import statistics
import time

# Offline by default (use the local cache, never download silently). On a fresh GPU machine set
# HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 for the first run (scripts/setup_gpu.sh does this).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch

from lagate.engine.tiered import DensePolicy, EngineConfig, FixedTierPolicy, TieredEngine
from lagate.models.gated import LookaheadGatedModel
from lagate.routing.router import RouterConfig
from lagate.models.adapter import DecoderAdapter, Dims
from lagate.utils import REPO_ROOT, create_run_dir, load_config, resolve_dtype_name, save_json, set_seed


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def load_model(name: str, dtype: str, device: str, attn: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype), attn_implementation=attn)
    return m.to(device).eval()


def build_prompts(cfg: dict, tok) -> list[torch.Tensor]:
    from datasets import load_dataset

    pc = cfg["prompts"]
    ds = load_dataset(pc["dataset"], pc["subset"], split=pc["split"])
    system = load_config(REPO_ROOT / pc["chat_config"])["eval"]["system_instruction"]
    out = []
    for i in pc["doc_ids"]:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": ds[i]["question"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        out.append(ids["input_ids"] if hasattr(ids, "keys") else ids)
    return out


@torch.no_grad()
def teq(model, prompts, device) -> dict:
    ad = DecoderAdapter(model)
    eng = TieredEngine(model, DensePolicy())
    rows = []
    for ids in prompts:
        ids = ids.to(device)
        ref = model(ids).logits.float()
        ours = ad.dense_forward(ids).float()
        engl = eng.forward_chunk(ids, eng.new_cache()).float()
        rows.append({
            "prompt_tokens": ids.shape[1],
            "max_abs_diff_adapter": (ours - ref).abs().max().item(),
            "max_abs_diff_engine": (engl - ref).abs().max().item(),
            "max_abs_logit": ref.abs().max().item(),
            "argmax_agree_frac": (ours.argmax(-1) == ref.argmax(-1)).float().mean().item(),
        })
    return {"per_prompt": rows,
            "max_abs_diff": max(max(r["max_abs_diff_adapter"], r["max_abs_diff_engine"]) for r in rows)}


@torch.no_grad()
def hf_generate(model, ids, n, **overrides):
    return model.generate(ids, max_new_tokens=n, min_new_tokens=n, do_sample=False, pad_token_id=0,
                          **overrides)[:, ids.shape[1]:]


def compare_generate(model, prompts, n, device, overrides: dict) -> dict:
    """Dense engine (pure greedy) vs HF generate with the model's generation_config as is and with
    ``overrides`` (e.g. repetition_penalty 1.0 = pure greedy)."""
    eng = TieredEngine(model, DensePolicy())
    out = {"hf_generation_config": {k: v for k, v in model.generation_config.to_diff_dict().items()
                                    if k in ("repetition_penalty", "temperature", "top_p", "top_k", "do_sample")}}
    for label, kw in (("hf_default_config", {}), ("hf_pure_greedy", overrides)):
        rows = []
        for ids in prompts:
            ids = ids.to(device)
            a = hf_generate(model, ids, n, **kw)[0].tolist()
            b = eng.generate(ids, n)[0][0].tolist()
            div = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            rows.append({"identical": a == b, "first_divergence": div, "n_tokens": len(a)})
        out[label] = rows
    return out


def time_fn(fn, warmup: int, runs: int, device: str, n_tokens: int) -> dict:
    for _ in range(warmup):
        fn(); sync(device)
    ts = []
    for _ in range(runs):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    ms = [1000 * t / n_tokens for t in ts]
    return {"median_ms_per_token": statistics.median(ms), "runs_ms_per_token": ms}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = cfg["device"]
    # 'auto' / 'bfloat16' resolve per GPU: bf16 on L4/A100, fp16 on a T4 (no native bf16); both names are recorded
    teq_dtypes = list(dict.fromkeys(resolve_dtype_name(d, device) for d in cfg["teq_dtypes"]))
    gen_dtype = resolve_dtype_name(cfg["gen_dtype"], device)
    run_dir = create_run_dir(cfg, args.seed, cfg["name"], extra_meta={
        "device": device, "teq_dtypes_resolved": teq_dtypes, "gen_dtype_resolved": gen_dtype})
    from transformers import AutoConfig, AutoTokenizer

    out: dict = {}
    # 1. dims
    hf_dims = Dims.from_hf_config(AutoConfig.from_pretrained(cfg["model"]))
    ours = Dims(**load_config(REPO_ROOT / cfg["expected_dims_config"])["dims"])
    out["dims"] = {"config_json": hf_dims.__dict__, "ours": ours.__dict__, "match": hf_dims == ours}
    print(f"[dims] match={hf_dims == ours} {hf_dims}")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    prompts = build_prompts(cfg, tok)
    out["prompt_tokens"] = [int(x.shape[1]) for x in prompts]

    # 2. T-EQ per dtype
    out["teq"] = {}
    for dt in teq_dtypes:
        model = load_model(cfg["model"], dt, device)
        r = teq(model, prompts, device)
        out["teq"][dt] = r
        print(f"[T-EQ {dt}] max|Δlogit| = {r['max_abs_diff']:.3e}  "
              f"(max|logit| {max(x['max_abs_logit'] for x in r['per_prompt']):.1f}; "
              f"argmax agree {min(x['argmax_agree_frac'] for x in r['per_prompt']):.4f})")
        del model; gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 3 + 4. generation and latency in the deployment dtype
    n = cfg["generate"]["max_new_tokens"]
    lat = cfg["latency"]
    out["latency"] = {"new_tokens": n, "per_prompt": []}
    models = {a: load_model(cfg["model"], gen_dtype, device, attn=a) for a in lat["hf_attn_implementations"][:1]}
    base = models[lat["hf_attn_implementations"][0]]
    overrides = cfg["generate"].get("hf_overrides", {})
    out["generate_match"] = compare_generate(base, prompts, n, device, overrides)
    print(f"[generate] model generation_config: {out['generate_match']['hf_generation_config']}")
    for label in ("hf_default_config", "hf_pure_greedy"):
        for i, r in enumerate(out["generate_match"][label]):
            print(f"[generate] {label} prompt {i}: identical={r['identical']} first_divergence={r['first_divergence']}")
    L = base.config.num_hidden_layers
    engines = {}
    for variant in lat.get("engine_variants", ["dense"]):
        if variant == "dense":
            engines[variant] = TieredEngine(base, DensePolicy())
        elif variant == "general_tier1":
            engines[variant] = TieredEngine(base, FixedTierPolicy(torch.ones(4096, L, dtype=torch.long, device=device)))
        elif variant in ("lagate_noskip", "lagate_noskip_per_layer"):
            gated = LookaheadGatedModel(base, RouterConfig.from_dict(load_config(REPO_ROOT / "configs/router/lagate.yaml"))).to(device)
            mode = "per_layer" if variant.endswith("per_layer") else "per_module"
            engines[variant] = TieredEngine.from_gated_model(gated, float("-inf"), float("-inf"), EngineConfig(sync_mode=mode))
        else:
            raise ValueError(variant)
    for i, ids in enumerate(prompts):
        ids = ids.to(device)
        row = {"prompt_tokens": int(ids.shape[1])}
        for attn in lat["hf_attn_implementations"]:
            base.set_attn_implementation(attn)
            row[f"hf_{attn}"] = time_fn(lambda: hf_generate(base, ids, n, **overrides), lat["warmup_runs"], lat["timed_runs"], device, n)
        base.set_attn_implementation(lat["hf_attn_implementations"][0])
        for variant, eng in engines.items():
            row[f"engine_{variant}"] = time_fn(lambda: eng.generate(ids, n), lat["warmup_runs"], lat["timed_runs"], device, n)
        out["latency"]["per_prompt"].append(row)
        print(f"[latency] prompt {i}: " + "  ".join(f"{k}={v['median_ms_per_token']:.1f} ms/tok"
                                                    for k, v in row.items() if isinstance(v, dict)))
    pp = out["latency"]["per_prompt"]
    for attn in lat["hf_attn_implementations"]:
        hf = statistics.median(r[f"hf_{attn}"]["median_ms_per_token"] for r in pp)
        for variant in engines:
            en = statistics.median(r[f"engine_{variant}"]["median_ms_per_token"] for r in pp)
            out["latency"][f"ratio_engine_{variant}_over_hf_{attn}"] = en / hf
            print(f"[latency] engine_{variant}/HF-{attn} = {en / hf:.2f}x  (engine {en:.1f}, HF {hf:.1f} ms/token, medians over prompts)")
    save_json(out, run_dir / "metrics.json")
    print(f"[real_model_checks] -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
