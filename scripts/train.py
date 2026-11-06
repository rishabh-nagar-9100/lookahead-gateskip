"""Train one method (lagate / gateskip / dense) with the shared loop (lagate.training.loop).

Usage:
    python scripts/train.py --config configs/train/smoke_lagate_llama.yaml --seed 0

Writes results/<run_id>/{config.yaml, meta.json, metrics.json}; optional checkpoint under
checkpoints/<run_id>/ (git-ignored).
"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from lagate.training.loop import run
from lagate.utils import REPO_ROOT, create_run_dir, load_config, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = cfg.get("device", "cpu")
    run_dir = create_run_dir(cfg, args.seed, cfg["name"], extra_meta={"device": device})
    print(f"[train] run dir: {run_dir}")
    m = run(cfg, args.seed, run_dir, device)
    s = m["summary"]
    print(f"[train] ce {s['ce']['first']:.3f} -> {s['ce']['last']:.3f}")
    for k, r in m.get("smoke_eval", {}).get("by_skip_frac", {}).items():
        print(f"[smoke_eval] skip_frac={k}: continuation_acc={r['continuation_acc']:.3f} "
              f"compute_frac={r['compute_frac']:.3f} tiers={r['tier_counts']}")
    print(f"[train] -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
