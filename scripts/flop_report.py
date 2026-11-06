"""P2-06: per-tier MACs per token and layer, and the router overhead, for one model.

Usage:
    python scripts/flop_report.py --config configs/flops/llama.yaml --seed 0

Writes results/<run_id>/metrics.json. Needs no weights (dims come from the config).
"""

from __future__ import annotations

import argparse

from lagate.engine.flops import LayerCosts
from lagate.models.adapter import Dims
from lagate.routing.router import RouterConfig
from lagate.utils import REPO_ROOT, create_run_dir, load_config, save_json, set_seed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    router = RouterConfig.from_dict(load_config(REPO_ROOT / cfg["router_config"]))
    costs = LayerCosts(Dims(**cfg["dims"]), bottleneck=router.bottleneck)

    rows = []
    for n in cfg["context_lengths"]:
        dense = costs.dense(n)
        rows.append({
            "context_length": n,
            "dense_macs": dense,
            **{f"tier{t}_macs": costs.tier(t, n) for t in (1, 2, 3)},
            **{f"tier{t}_frac_of_dense": costs.tier(t, n) / dense for t in (1, 2, 3)},
            "router_macs": costs.router,
            "router_overhead_frac": costs.router_overhead(n),
            "mlp_share_of_dense": costs.mlp / dense,
        })
    run_dir = create_run_dir(cfg, args.seed, cfg["name"])
    save_json({"model": cfg["model"], "bottleneck": router.bottleneck, "per_layer_per_token": rows},
              run_dir / "metrics.json")
    print(f"[flop_report] {cfg['model']}  router = {costs.router:,} MACs/token/layer")
    print(f"{'n':>6} {'dense MACs':>13} {'router %':>9} {'MLP %':>7} {'T1 %':>7} {'T2 %':>7} {'T3 %':>7}")
    for r in rows:
        print(f"{r['context_length']:>6} {r['dense_macs']:>13,.0f} {100 * r['router_overhead_frac']:>9.3f} "
              f"{100 * r['mlp_share_of_dense']:>7.1f} {100 * r['tier1_frac_of_dense']:>7.1f} "
              f"{100 * r['tier2_frac_of_dense']:>7.1f} {100 * r['tier3_frac_of_dense']:>7.1f}")
    print(f"[flop_report] -> {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
