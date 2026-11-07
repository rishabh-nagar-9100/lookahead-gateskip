"""Merge sharded evaluation runs (scripts/eval_dense.py --shard i --num-shards n) into one result.

Usage:
    python scripts/merge_shards.py --config configs/eval/gsm8k_chat_dense_qwen.yaml --seed 0 \
        --runs results/<id>_shard0of4_s0 results/<id>_shard1of4_s0 ...
    python scripts/merge_shards.py --config ... --seed 0 --glob 'results/*_gsm8k_chat_dense_qwen_shard*of4_s0'

Checks before merging (any failure aborts):
  * every run used this config (identical config.yaml except ``eval.limit``, which --limit writes
    into it and which must be the same in every shard),
  * the same code in every shard: identical commits, or commits that differ only in results/ and
    docs/ (result commits pushed from Colab between sessions), and a clean tree
    (``--allow-dirty`` / ``--allow-mixed-commits`` relax this, and the merged meta records it),
  * the same seed, device type, dtype and GPU name,
  * shard indices 0..n-1 each present exactly once, the doc ids disjoint, and their union equal to
    the full selection.
Mean metrics are combined weighted by documents (exact, since every shard reports a mean over its
docs); standard errors are recomputed for the binary metrics as sqrt(p(1-p)/(n-1)) (lm-eval's
formula). Writes results/<run_id>/{config.yaml, meta.json, metrics.json}; the shard run ids are
listed in metrics.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any

import yaml

from lagate.utils import REPO_ROOT, create_run_dir, load_config, same_code, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="the eval config every shard ran with")
    p.add_argument("--seed", type=int, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--runs", nargs="+", help="shard run directories")
    g.add_argument("--glob", help="glob for shard run directories (relative to the repo)")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-mixed-commits", action="store_true")
    return p.parse_args()


def load_run(d: Path) -> dict[str, Any]:
    with open(d / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {"dir": d, "config": cfg, "meta": json.loads((d / "meta.json").read_text()),
            "metrics": json.loads((d / "metrics.json").read_text())}


def _without_limit(cfg: dict) -> dict:
    return {**cfg, "eval": {k: v for k, v in cfg.get("eval", {}).items() if k != "limit"}} if "eval" in cfg else cfg


def check_shards(runs: list[dict], cfg: dict, seed: int, allow_dirty: bool, allow_mixed: bool) -> dict[str, Any]:
    if not runs:
        raise SystemExit("no shard runs given")
    problems = []
    limits = {r["config"].get("eval", {}).get("limit") for r in runs}
    if len(limits) > 1:
        problems.append(f"shards used different --limit values {limits}")
    for r in runs:
        if _without_limit(r["config"]) != _without_limit(cfg):
            problems.append(f"{r['dir'].name}: config differs from {cfg['name']}")
        if r["meta"]["seed"] != seed:
            problems.append(f"{r['dir'].name}: seed {r['meta']['seed']} != {seed}")
        if "shard" not in r["metrics"]:
            problems.append(f"{r['dir'].name}: not a sharded run")
    if problems:
        raise SystemExit("\n".join(problems))
    commits = {r["meta"]["git"]["commit"] for r in runs}
    first = sorted(commits)[0]
    code_differs = [c for c in commits if not same_code(first, c)]
    if code_differs and not allow_mixed:
        raise SystemExit(f"shards ran different code: {sorted(commits)} (--allow-mixed-commits to override)")
    dirty = [r["dir"].name for r in runs if r["meta"]["git"].get("dirty")]
    if dirty and not allow_dirty:
        raise SystemExit(f"shards from a dirty tree: {dirty} (--allow-dirty to override)")
    for key in ("device", "dtype"):
        vals = {r["meta"].get(key) for r in runs}
        if len(vals) > 1:
            raise SystemExit(f"shards differ in {key}: {vals}")
    gpus = {(r["meta"].get("accelerator") or {}).get("gpu_name") for r in runs}
    if len(gpus) > 1:
        raise SystemExit(f"shards ran on different GPUs {gpus}; results are not one measurement")
    shards = [r["metrics"]["shard"] for r in runs]
    n = {s["num_shards"] for s in shards}
    if len(n) != 1:
        raise SystemExit(f"inconsistent num_shards {n}")
    n = n.pop()
    idx = sorted(s["shard"] for s in shards)
    if idx != list(range(n)):
        raise SystemExit(f"need shards 0..{n - 1} exactly once, got {idx}")
    ids = [i for s in sorted(shards, key=lambda s: s["shard"]) for i in s["doc_ids"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("doc ids overlap between shards")
    n_sel = {s["n_selected"] for s in shards}
    if len(n_sel) != 1 or len(ids) != n_sel.pop():
        raise SystemExit("the shards do not cover the full selection")
    return {"num_shards": n, "n_docs": len(ids), "limit": limits.pop(), "commits": sorted(commits),
            "code_identical": not code_differs, "dirty_shards": dirty}


def _weighted(values: list[tuple[float, int]]) -> float:
    total = sum(n for _, n in values)
    return sum(v * n for v, n in values) / total


def merge_metrics(runs: list[dict]) -> dict[str, Any]:
    runs = sorted(runs, key=lambda r: r["metrics"]["shard"]["shard"])
    tasks = list(runs[0]["metrics"]["results"])
    out: dict[str, Any] = {"results": {}, "output_length": {}, "robust_extract": {}, "format": {}, "n_samples": {}}
    for t in tasks:
        res = [r["metrics"]["results"][t] for r in runs]
        ns = [r["metrics"]["n_samples"][t]["effective"] for r in runs]
        merged: dict[str, Any] = {"name": t, "sample_len": sum(ns)}
        for k in res[0]:
            if k.startswith("exact_match,") or (k.startswith(("acc,", "acc_norm,")) and "stderr" not in k):
                p = _weighted([(r[k], n) for r, n in zip(res, ns)])
                merged[k] = p
                se_key = k.replace(",", "_stderr,", 1)
                N = sum(ns)
                merged[se_key] = math.sqrt(p * (1 - p) / (N - 1)) if N > 1 else float("nan")
        out["results"][t] = merged
        out["n_samples"][t] = {"original": runs[0]["metrics"]["n_samples"][t]["original"], "effective": sum(ns)}
        for block in ("output_length", "robust_extract", "format"):
            parts = [r["metrics"].get(block, {}).get(t) for r in runs]
            if not all(parts):
                continue
            nd = [p["n_docs"] for p in parts]
            m = {"n_docs": sum(nd)}
            for k in parts[0]:
                if k == "n_docs":
                    continue
                m[k] = max(p[k] for p in parts) if k.startswith("max_") else _weighted([(p[k], n) for p, n in zip(parts, nd)])
            out[block][t] = m
    out["runtime_seconds_total"] = sum(r["metrics"].get("runtime_seconds", 0.0) for r in runs)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    dirs = [Path(d) for d in args.runs] if args.runs else [Path(p) for p in sorted(glob.glob(str(REPO_ROOT / args.glob)))]
    runs = [load_run(d if d.is_absolute() else REPO_ROOT / d) for d in dirs]
    check = check_shards(runs, cfg, args.seed, args.allow_dirty, args.allow_mixed_commits)
    merged = merge_metrics(runs)
    merged["merged_from"] = [r["dir"].name for r in sorted(runs, key=lambda r: r["metrics"]["shard"]["shard"])]
    merged["merge_checks"] = check
    m0 = runs[0]["meta"]
    run_dir = create_run_dir(cfg, args.seed, f"{cfg['name']}_merged{check['num_shards']}",
                             extra_meta={"device": m0.get("device"), "dtype": m0.get("dtype"),
                                         "shard_accelerator": m0.get("accelerator"), "merge": check})
    save_json(merged, run_dir / "metrics.json")
    for t, res in merged["results"].items():
        shown = {k: round(v, 4) for k, v in res.items() if k.startswith("exact_match,")}
        print(f"[merge_shards] {t} ({res['sample_len']} docs from {check['num_shards']} shards): {shown}")
        if t in merged["robust_extract"]:
            print(f"[merge_shards] {t} robust {merged['robust_extract'][t]}  format {merged['format'].get(t)}")
    print(f"[merge_shards] -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
