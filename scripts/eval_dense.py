"""E0: dense-model GSM8K evaluation through lm-evaluation-harness.

Usage:
    python scripts/eval_dense.py --config configs/eval/gsm8k_dense_llama.yaml --seed 0 --limit 50
    python scripts/eval_dense.py --config ... --seed 0 --shard 2 --num-shards 4   # docs [N/2, 3N/4)

Sharding (Colab sessions are time-limited): the selected documents (all, the first --limit, or
``eval.doc_ids`` such as the D-06 calibration split) are cut into ``num_shards`` contiguous blocks;
each shard is its own run, and ``scripts/merge_shards.py`` combines them. ``model.pretrained`` may
be a local path relative to the repo (e.g. a Dense-FT export under checkpoints/).

Writes to results/<run_id>/: config.yaml, meta.json (git hash, versions), metrics.json.
Raw generations (if eval.log_samples) go to results/raw/<run_id>/, which is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lagate.eval.answer import score_samples
from lagate.utils import REPO_ROOT, create_run_dir, load_config, resolve_device, resolve_dtype_name, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/eval/gsm8k_dense_llama.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems (overrides eval.limit)")
    p.add_argument("--shard", type=int, default=0, help="Shard index i in [0, num_shards)")
    p.add_argument("--num-shards", type=int, default=1)
    return p.parse_args()


def resolve_dtype(model_cfg: dict[str, Any], device: str) -> str:
    """Configured dtype for this device; bf16 on a CUDA GPU without native bf16 (T4) becomes fp16."""
    name = model_cfg["dtype"]
    if name == "auto":
        name = model_cfg["dtype_by_device"][device.split(":")[0]]
    return resolve_dtype_name(name, device)


def resolve_pretrained(name: str) -> str:
    """A path that exists relative to the repo (checkpoints/<run>/hf) is used as a local model."""
    local = REPO_ROOT / name
    return str(local) if local.exists() else name


def shard_bounds(n: int, shard: int, num_shards: int) -> tuple[int, int]:
    if not 0 <= shard < num_shards:
        raise ValueError(f"shard {shard} not in [0, {num_shards})")
    return shard * n // num_shards, (shard + 1) * n // num_shards


def selected_doc_ids(eval_cfg: dict[str, Any], task_manager, task: str) -> list[int]:
    """All test docs, the first ``limit``, or ``eval.doc_ids`` (e.g. the D-06 calibration split)."""
    spec = eval_cfg.get("doc_ids")
    if spec is not None:
        from lagate.training.chat_data import calibration_row_ids, load_raw

        data_cfg = load_config(REPO_ROOT / spec["from_data_config"])
        src = spec["source"]
        n_rows = len(load_raw({**data_cfg, "datasets": {src: data_cfg["datasets"][src]}})[src])
        ids = calibration_row_ids(n_rows, data_cfg, src)
    else:
        from lm_eval.tasks import TaskManager, get_task_dict

        t = get_task_dict([task], task_manager or TaskManager())[task]
        ids = list(range(len(t.eval_docs)))
    if eval_cfg.get("limit") is not None:
        ids = ids[: eval_cfg["limit"]]
    return ids


HASH_ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")  # the strict-match regex of the GSM8K tasks


def format_stats(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Share of responses with a strict '#### N' line (D-15 / P3-08 criterion: >= 90%)."""
    by_doc = {s["doc_id"]: s["resps"][0][0] for s in samples if s.get("resps")}
    if not by_doc:
        return {}
    return {"n_docs": len(by_doc), "frac_hash_answer": sum(bool(HASH_ANSWER.search(t)) for t in by_doc.values()) / len(by_doc)}


def gen_kwargs_to_str(gen_kwargs: dict[str, Any]) -> str:
    """lm-eval parses gen_kwargs as 'k=v,k=v' in every 0.4.x release."""
    return ",".join(f"{k}={v}" for k, v in gen_kwargs.items())


def output_length_stats(
    samples: list[dict[str, Any]], tokenizer, max_gen_toks: int
) -> dict[str, float]:
    """Generated length (raw response, before answer filtering) and how often the token cap was hit.

    lm-eval logs one sample per (doc, filter), so responses are de-duplicated by doc_id.
    Re-tokenising decoded text can differ by a token from the generated ids, so a response
    within 1 token of the cap counts as having hit it.
    """
    by_doc = {s["doc_id"]: s["resps"][0][0] for s in samples if s.get("resps")}
    texts = list(by_doc.values())
    if not texts:
        return {}
    n_tok = [len(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
    hit = sum(n >= max_gen_toks - 1 for n in n_tok)
    return {
        "n_docs": len(texts),
        "avg_output_chars": sum(map(len, texts)) / len(texts),
        "avg_output_tokens": sum(n_tok) / len(n_tok),
        "max_output_tokens": max(n_tok),
        "frac_hit_max_gen_toks": hit / len(texts),
    }


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env")  # HF_TOKEN is read from the environment by huggingface_hub

    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["eval"]["limit"] = args.limit
    set_seed(args.seed)

    model_cfg, eval_cfg = cfg["model"], cfg["eval"]
    device = resolve_device(model_cfg["device"])
    dtype = resolve_dtype(model_cfg, device)
    pretrained = resolve_pretrained(model_cfg["pretrained"])

    import lm_eval
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    # Optional: our own task YAMLs (e.g. configs/eval/tasks/gsm8k_chat_zs.yaml), relative to the repo root.
    include_path = eval_cfg.get("include_path")
    task_manager = TaskManager(include_path=str(REPO_ROOT / include_path)) if include_path else None

    # Document selection: lm-eval's `samples` (explicit doc ids) whenever we shard or pick ids.
    samples_arg, limit_arg, shard_meta = None, eval_cfg["limit"], {}
    if args.num_shards > 1 or eval_cfg.get("doc_ids") is not None:
        if len(eval_cfg["tasks"]) != 1:
            raise ValueError("sharding / doc_ids support one task per config")
        task = eval_cfg["tasks"][0]
        ids = selected_doc_ids(eval_cfg, task_manager, task)
        lo, hi = shard_bounds(len(ids), args.shard, args.num_shards)
        samples_arg, limit_arg = {task: ids[lo:hi]}, None
        shard_meta = {"shard": args.shard, "num_shards": args.num_shards, "n_selected": len(ids),
                      "doc_ids": ids[lo:hi]}

    name = cfg["name"] + (f"_shard{args.shard}of{args.num_shards}" if args.num_shards > 1 else "")
    run_dir = create_run_dir(
        cfg,
        args.seed,
        name,
        extra_meta={"device": device, "dtype": dtype, "dtype_configured": model_cfg["dtype"],
                    "pretrained": pretrained, "limit": eval_cfg["limit"],
                    **{k: v for k, v in shard_meta.items() if k != "doc_ids"}},
    )
    print(f"[eval_dense] run dir: {run_dir}  device={device} dtype={dtype} limit={eval_cfg['limit']} "
          f"shard={args.shard}/{args.num_shards} docs={len(samples_arg[eval_cfg['tasks'][0]]) if samples_arg else 'all'}")

    t0 = time.perf_counter()
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={pretrained},dtype={dtype}",
        tasks=list(eval_cfg["tasks"]),
        num_fewshot=eval_cfg["num_fewshot"],
        batch_size=eval_cfg["batch_size"],
        device=device,
        limit=limit_arg,
        samples=samples_arg,
        gen_kwargs=gen_kwargs_to_str(eval_cfg["gen_kwargs"]),
        apply_chat_template=eval_cfg["apply_chat_template"],
        fewshot_as_multiturn=eval_cfg["fewshot_as_multiturn"],
        system_instruction=eval_cfg.get("system_instruction"),
        task_manager=task_manager,
        log_samples=eval_cfg["log_samples"],
        random_seed=args.seed,
        numpy_random_seed=args.seed,
        torch_random_seed=args.seed,
        fewshot_random_seed=args.seed,
    )
    runtime_s = time.perf_counter() - t0

    metrics: dict[str, Any] = {
        "results": out["results"],
        "n_samples": out.get("n-samples"),
        "runtime_seconds": runtime_s,
        "device": device,
        "dtype": dtype,
        "lm_eval_task_configs": out.get("configs"),
        "lm_eval_versions": out.get("versions"),
        **({"shard": shard_meta} if shard_meta else {}),
    }

    samples = out.get("samples") or {}
    if samples:
        tok = AutoTokenizer.from_pretrained(pretrained)
        raw_dir = REPO_ROOT / "results" / "raw" / run_dir.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        lengths, robust, fmt = {}, {}, {}
        for task, task_samples in samples.items():
            with open(raw_dir / f"samples_{task}.jsonl", "w", encoding="utf-8") as f:
                for s in task_samples:
                    f.write(json.dumps(s, default=str) + "\n")
            lengths[task] = output_length_stats(task_samples, tok, eval_cfg["gen_kwargs"]["max_gen_toks"])
            robust[task] = score_samples(task_samples)  # secondary metric (D-15); strict stays headline
            fmt[task] = format_stats(task_samples)
        metrics["output_length"] = lengths
        metrics["robust_extract"] = robust
        metrics["format"] = fmt
        metrics["raw_samples_dir"] = str(raw_dir.relative_to(REPO_ROOT))

    save_json(metrics, run_dir / "metrics.json")

    primary = eval_cfg.get("primary_metric")
    for task, res in out["results"].items():
        shown = {k: v for k, v in res.items() if "exact_match" in k}
        print(f"[eval_dense] {task}: {shown}")
        if primary in res:
            print(f"[eval_dense] {task} {primary} = {res[primary]:.4f}")
        if task in metrics.get("robust_extract", {}):
            print(f"[eval_dense] {task} robust_extract = {metrics['robust_extract'][task]}")
    print(f"[eval_dense] runtime {runtime_s:.1f}s -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
