"""Estimate GPU hours per experiment for a minimal and a full plan (docs/GPU_BUDGET.md).

Usage:
    python scripts/gpu_budget.py --config configs/budget/gpu_budget.yaml --seed 0

Pure arithmetic on the assumptions in the config (no GPU needed). Writes results/<run_id>/metrics.json
and prints Markdown tables. Two eval modes: every accuracy eval through the bs=1 engine, or batched
accuracy evals (lm-eval batching for dense models; for gated models this needs the batched masked
eval mode, task P5-08) with bs=1 kept for the latency study only.
"""

from __future__ import annotations

import argparse

from lagate.utils import create_run_dir, load_config, save_json, set_seed


def eval_hours(cfg: dict, gpu: dict, model: str, gated: bool, kind: str, batched: bool) -> float:
    ev = cfg["eval"]
    ms = gpu["decode_ms_per_token"][model] * (ev["routing_overhead"] if gated and not batched else 1.0)
    speed = ev["batched_speedup"] if batched else 1.0
    gen = lambda items, toks: items * toks * ms / 1000 / 3600 / speed  # noqa: E731
    mc = ev["mc_hours_a100"] / gpu["train_speed"]
    if kind == "gsm8k":
        return gen(ev["gsm8k"]["items"], ev["gsm8k"]["gen_tokens"])
    if kind == "dev":
        return gen(ev["dev_items"], ev["gsm8k"]["gen_tokens"])
    if kind == "suite":
        return gen(ev["gsm8k"]["items"], ev["gsm8k"]["gen_tokens"]) + gen(ev["mbpp"]["items"], ev["mbpp"]["gen_tokens"]) + mc
    raise ValueError(kind)


def row_hours(cfg: dict, gpu_key: str, spec: dict, batched: bool) -> tuple[float, float]:
    gpu = cfg["gpus"][gpu_key]
    tr = cfg["training"]
    models = ["llama", "qwen"] if spec["model"] == "both" else [spec["model"]]
    train = evals = 0.0
    for m in models:
        if spec["method"] == "official":
            if gpu_key != "a100":
                return float("nan"), 0.0  # their recipe needs the 80 GB A100
            train += spec["runs"] * spec["seeds"] * tr["official_gateskip_hours_a100"]
            continue
        n_models = spec["runs"] * spec["seeds"]
        if spec["runs"]:
            per = tr["hours_a100"][m] * tr["method_overhead"][spec["method"]] / gpu["train_speed"]
            train += n_models * (per + (tr["calibration_hours_a100"] / gpu["train_speed"] if spec["method"] != "dense" else 0))
        n_eval_models = max(n_models, spec["seeds"])  # untrained methods: one eval per seed
        gated = spec["method"] not in ("dense", "none")
        for kind, key in (("gsm8k", "gsm8k_budgets"), ("suite", "suite_budgets"), ("dev", "dev")):
            k = spec.get(key, 0)
            if k:
                evals += n_eval_models * k * eval_hours(cfg, gpu, m, gated, kind, batched)
    return train, evals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    out: dict = {}
    for plan, rows in cfg["plans"].items():
        out[plan] = {}
        for gpu_key, gpu in cfg["gpus"].items():
            for batched in (True, False):
                mode = "batched_eval" if batched else "bs1_eval"
                table, tot_t, tot_e = [], 0.0, 0.0
                for name, spec in rows.items():
                    t, e = row_hours(cfg, gpu_key, spec, batched)
                    table.append({"experiment": name, "train_h": t, "eval_h": e})
                    tot_t += 0 if t != t else t  # skip NaN (not runnable on this GPU)
                    tot_e += e
                lat = cfg["latency_study_hours"][plan]
                out[plan][f"{gpu_key}/{mode}"] = {"rows": table, "train_h": tot_t, "eval_h": tot_e,
                                                  "latency_h": lat, "total_h": tot_t + tot_e + lat}
    run_dir = create_run_dir(cfg, args.seed, cfg["name"])
    save_json(out, run_dir / "metrics.json")

    for plan in out:
        print(f"\n### {plan} plan (GPU hours)\n")
        keys = list(out[plan])
        print("| Experiment | " + " | ".join(k.replace("/", " · ") for k in keys) + " |")
        print("|---|" + "---|" * len(keys))
        names = [r["experiment"] for r in out[plan][keys[0]]["rows"]]
        for i, n in enumerate(names):
            cells = []
            for k in keys:
                r = out[plan][k]["rows"][i]
                t = "n/a" if r["train_h"] != r["train_h"] else f"{r['train_h'] + r['eval_h']:.1f}"
                cells.append(t)
            print(f"| {n} | " + " | ".join(cells) + " |")
        print("| E11 latency study | " + " | ".join(f"{out[plan][k]['latency_h']:.1f}" for k in keys) + " |")
        print("| **Total** | " + " | ".join(f"**{out[plan][k]['total_h']:.0f}**" for k in keys) + " |")
        print("| of which training | " + " | ".join(f"{out[plan][k]['train_h']:.0f}" for k in keys) + " |")
    print(f"\n[gpu_budget] -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
