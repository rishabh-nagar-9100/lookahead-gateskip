"""E0: dense-model GSM8K evaluation through lm-evaluation-harness.

Usage:
    python scripts/eval_dense.py --config configs/eval/gsm8k_dense_llama.yaml --seed 0 --limit 50

Writes to results/<run_id>/: config.yaml, meta.json (git hash, versions), metrics.json.
Raw generations (if eval.log_samples) go to results/raw/<run_id>/, which is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from lagate.utils import REPO_ROOT, create_run_dir, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML config, e.g. configs/eval/gsm8k_dense_llama.yaml")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N problems (overrides eval.limit)")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(model_cfg: dict[str, Any], device: str) -> str:
    if model_cfg["dtype"] != "auto":
        return model_cfg["dtype"]
    return model_cfg["dtype_by_device"][device.split(":")[0]]


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

    run_dir = create_run_dir(
        cfg,
        args.seed,
        cfg["name"],
        extra_meta={"device": device, "dtype": dtype, "limit": eval_cfg["limit"]},
    )
    print(f"[eval_dense] run dir: {run_dir}  device={device} dtype={dtype} limit={eval_cfg['limit']}")

    import lm_eval
    from transformers import AutoTokenizer

    t0 = time.perf_counter()
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_cfg['pretrained']},dtype={dtype}",
        tasks=list(eval_cfg["tasks"]),
        num_fewshot=eval_cfg["num_fewshot"],
        batch_size=eval_cfg["batch_size"],
        device=device,
        limit=eval_cfg["limit"],
        gen_kwargs=gen_kwargs_to_str(eval_cfg["gen_kwargs"]),
        apply_chat_template=eval_cfg["apply_chat_template"],
        fewshot_as_multiturn=eval_cfg["fewshot_as_multiturn"],
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
    }

    samples = out.get("samples") or {}
    if samples:
        tok = AutoTokenizer.from_pretrained(model_cfg["pretrained"])
        raw_dir = REPO_ROOT / "results" / "raw" / run_dir.name
        raw_dir.mkdir(parents=True, exist_ok=True)
        lengths = {}
        for task, task_samples in samples.items():
            with open(raw_dir / f"samples_{task}.jsonl", "w", encoding="utf-8") as f:
                for s in task_samples:
                    f.write(json.dumps(s, default=str) + "\n")
            lengths[task] = output_length_stats(task_samples, tok, eval_cfg["gen_kwargs"]["max_gen_toks"])
        metrics["output_length"] = lengths
        metrics["raw_samples_dir"] = str(raw_dir.relative_to(REPO_ROOT))

    save_json(metrics, run_dir / "metrics.json")

    primary = eval_cfg.get("primary_metric")
    for task, res in out["results"].items():
        shown = {k: v for k, v in res.items() if "exact_match" in k}
        print(f"[eval_dense] {task}: {shown}")
        if primary in res:
            print(f"[eval_dense] {task} {primary} = {res[primary]:.4f}")
    print(f"[eval_dense] runtime {runtime_s:.1f}s -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
