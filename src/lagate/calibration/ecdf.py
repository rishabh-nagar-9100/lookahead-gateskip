"""Empirical-CDF thresholds from calibration-split gate values (ARCHITECTURE §6; D-12(b), D-18).

One primitive serves every method: given the gate scores a module produced on the calibration split
(never the test set), return τ such that the fraction of scores ≤ τ equals the desired skip
fraction. Tokens with score ≤ τ are skipped, so at deployment the skip rate on data distributed
like the calibration split is ≈ that fraction, at any batch size (including bs=1 decode).

Our method's (τ_L, τ_M) grid search (P3-04) and the GateSkip port's per-module thresholds both use
``ecdf_threshold``; ``match_budget`` finds the skip fraction that hits a target compute reduction.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def ecdf_threshold(scores: np.ndarray, skip_frac: float) -> float:
    """τ with P̂(score ≤ τ) = skip_frac on the calibration scores (NaNs ignored).

    skip_frac 0 → τ = −inf (skip nothing). Otherwise τ is the k-th smallest score with
    k = round(skip_frac · N), so exactly k calibration scores satisfy score ≤ τ (ties aside).
    """
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    s = s[~np.isnan(s)]
    if not 0.0 <= skip_frac <= 1.0:
        raise ValueError(f"skip_frac must be in [0, 1], got {skip_frac}")
    k = int(round(skip_frac * len(s)))
    if k == 0 or len(s) == 0:
        return float("-inf")
    return float(s[k - 1])


def skip_fraction(scores: np.ndarray, tau: float) -> float:
    s = np.asarray(scores, dtype=np.float64).ravel()
    s = s[~np.isnan(s)]
    return float((s <= tau).mean()) if len(s) else 0.0


def match_budget(compute_frac_at: Callable[[float], float], target_frac: float,
                 lo: float = 0.0, hi: float = 1.0, iters: int = 40) -> float:
    """Bisection on a skip fraction p so that ``compute_frac_at(p)`` (expected compute / dense,
    non-increasing in p) equals ``target_frac`` = 1 − r for a compute reduction r."""
    if compute_frac_at(lo) <= target_frac:
        return lo
    if compute_frac_at(hi) > target_frac:
        raise ValueError(f"budget {target_frac:.3f} unreachable: compute at p={hi} is {compute_frac_at(hi):.3f}")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if compute_frac_at(mid) > target_frac:
            lo = mid
        else:
            hi = mid
    return hi
