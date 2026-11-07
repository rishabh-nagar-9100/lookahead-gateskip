"""Sharded evaluation: shard bounds (eval_dense.py) and the merge checks / weighting (merge_shards.py)."""

import importlib.util
import math
from pathlib import Path

import pytest

from lagate.utils import REPO_ROOT


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ev, ms = _load("eval_dense"), _load("merge_shards")


@pytest.mark.parametrize("n,k", [(1319, 4), (500, 3), (7, 7), (5, 1)])
def test_shard_bounds_partition(n, k):
    ids = [i for s in range(k) for i in range(*ev.shard_bounds(n, s, k))]
    assert ids == list(range(n))
    sizes = [b - a for a, b in (ev.shard_bounds(n, s, k) for s in range(k))]
    assert max(sizes) - min(sizes) <= 1
    with pytest.raises(ValueError):
        ev.shard_bounds(n, k, k)


def test_format_stats_counts_hash_answers():
    s = [{"doc_id": 0, "resps": [["x\n#### 72"]]}, {"doc_id": 0, "resps": [["x\n#### 72"]]},
         {"doc_id": 1, "resps": [["the answer is 5"]]}]
    assert ev.format_stats(s) == {"n_docs": 2, "frac_hash_answer": 0.5}


def _run(shard, n_shards, ids, acc, cfg, commit="abc", dirty=False, gpu="NVIDIA L4"):
    return {
        "dir": Path(f"r{shard}"),
        "config": cfg,
        "meta": {"seed": 0, "git": {"commit": commit, "dirty": dirty}, "device": "cuda", "dtype": "bfloat16",
                 "accelerator": {"gpu_name": gpu}},
        "metrics": {
            "shard": {"shard": shard, "num_shards": n_shards, "n_selected": 10, "doc_ids": ids},
            "results": {"t": {"exact_match,strict-match": acc, "exact_match_stderr,strict-match": 0.1}},
            "n_samples": {"t": {"original": 1319, "effective": len(ids)}},
            "robust_extract": {"t": {"accuracy": acc, "n_docs": len(ids), "frac_extracted": 1.0}},
            "output_length": {"t": {"n_docs": len(ids), "avg_output_tokens": 100.0 * (shard + 1), "max_output_tokens": 200 + shard}},
            "runtime_seconds": 10.0,
        },
    }


CFG = {"name": "x"}


def test_merge_weights_by_docs_and_recomputes_stderr():
    runs = [_run(0, 2, [0, 1, 2], 1.0, CFG), _run(1, 2, list(range(3, 10)), 0.0, CFG)]
    chk = ms.check_shards(runs, CFG, 0, False, False)
    assert chk["n_docs"] == 10
    m = ms.merge_metrics(runs)
    p = m["results"]["t"]["exact_match,strict-match"]
    assert p == pytest.approx(0.3)
    assert m["results"]["t"]["exact_match_stderr,strict-match"] == pytest.approx(math.sqrt(0.3 * 0.7 / 9))
    assert m["robust_extract"]["t"]["accuracy"] == pytest.approx(0.3)
    assert m["output_length"]["t"]["avg_output_tokens"] == pytest.approx((3 * 100 + 7 * 200) / 10)
    assert m["output_length"]["t"]["max_output_tokens"] == 201


@pytest.mark.parametrize("bad", ["missing", "overlap", "commit", "dirty", "gpu", "config"])
def test_merge_refuses_inconsistent_shards(bad):
    a = _run(0, 2, [0, 1, 2, 3, 4], 0.5, CFG)
    b = _run(1, 2, [5, 6, 7, 8, 9], 0.5, CFG)
    if bad == "missing":
        runs = [a]
    elif bad == "overlap":
        b["metrics"]["shard"]["doc_ids"] = [4, 6, 7, 8, 9]
        runs = [a, b]
    elif bad == "commit":
        runs = [a, _run(1, 2, [5, 6, 7, 8, 9], 0.5, CFG, commit="def")]
    elif bad == "dirty":
        runs = [a, _run(1, 2, [5, 6, 7, 8, 9], 0.5, CFG, dirty=True)]
    elif bad == "gpu":
        runs = [a, _run(1, 2, [5, 6, 7, 8, 9], 0.5, CFG, gpu="Tesla T4")]
    else:
        runs = [a, _run(1, 2, [5, 6, 7, 8, 9], 0.5, {"name": "y"})]
    with pytest.raises(SystemExit):
        ms.check_shards(runs, CFG, 0, False, False)
