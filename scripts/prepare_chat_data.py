"""P3-06: build the D-06 training data in chat format for one tokenizer and record statistics.

Usage:
    python scripts/prepare_chat_data.py --config configs/data/prepare_d06_qwen.yaml --seed 0
    python scripts/prepare_chat_data.py --config ... --seed 0 --sources gsm8k    # subset of sources

Writes results/<run_id>/: metrics.json (sizes, length distribution, drops over max_length, label
counts, the calibration row-id digests), calibration_rows.json (the held-out train row ids per
source; P3-08 and threshold calibration use exactly these) and rendered_examples.json (a few
examples as the model sees them, with the learned span marked). The data itself is rebuilt from
the pinned datasets at training time, so nothing large is written. The seed is recorded only; the
split uses data.split_seed.
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from lagate.training.chat_data import ChatDataset, load_raw
from lagate.utils import REPO_ROOT, create_run_dir, load_config, save_json, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sources", nargs="+", default=None, help="override config sources")
    args = p.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    if args.sources:
        cfg["sources"] = args.sources
    set_seed(args.seed)
    data_cfg = load_config(REPO_ROOT / cfg["data_config"])
    data_cfg["datasets"] = {s: data_cfg["datasets"][s] for s in cfg["sources"]}
    data_cfg["calibration_size"] = {s: data_cfg["calibration_size"][s] for s in cfg["sources"]}
    run_dir = create_run_dir(cfg, args.seed, cfg["name"], extra_meta={"sources": cfg["sources"]})

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    ds = ChatDataset(data_cfg, tok, raw=load_raw(data_cfg))
    stats = ds.stats()
    save_json({"tokenizer": cfg["tokenizer"], "sources": cfg["sources"], **stats}, run_dir / "metrics.json")
    save_json(ds.calibration_rows, run_dir / "calibration_rows.json")
    shown = []
    for split, exs in ds.splits.items():
        for src in cfg["sources"]:
            for e in [e for e in exs if e.source == src][: cfg["n_rendered_examples"]]:
                first = next(i for i, l in enumerate(e.labels) if l != -100)
                last = max(i for i, l in enumerate(e.labels) if l != -100)
                shown.append({"split": split, "source": src, "prompt": tok.decode(e.input_ids[:first]),
                              "learned": tok.decode(e.input_ids[first : last + 1]),
                              "after_learned": tok.decode(e.input_ids[last + 1 :])})
    save_json(shown, run_dir / "rendered_examples.json")
    for split in ("train", "calibration"):
        s = stats[split]
        print(f"[prepare] {split}: n={s['n']} {s['by_source']} len mean {s['len_mean']:.0f} p50 {s['len_p50']} "
              f"p99 {s['len_p99']} max {s['len_max']}; learned tokens/example {s['label_tokens_mean']:.0f}")
    print(f"[prepare] dropped over max_length: {stats['dropped_over_max_length']}")
    print(f"[prepare] -> {run_dir}")


if __name__ == "__main__":
    main()
