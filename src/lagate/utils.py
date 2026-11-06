"""Shared helpers: config loading, seeding and run-directory bookkeeping."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Packages whose versions are recorded with every run.
TRACKED_PACKAGES = (
    "lagate",
    "torch",
    "transformers",
    "accelerate",
    "peft",
    "datasets",
    "lm_eval",
    "numpy",
    "pyyaml",
)


def load_config(path: str | os.PathLike) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(cfg).__name__}")
    return cfg


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU, CUDA, MPS)."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)  # also seeds CUDA and MPS devices
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def git_info(
    repo: Path = REPO_ROOT, ignore_untracked: tuple[str, ...] = ("results/",)
) -> dict[str, Any]:
    """Return the current commit hash and whether the working tree has uncommitted changes.

    Any tracked modification or untracked file marks the tree dirty, except untracked paths
    under ``ignore_untracked`` (by default results/, where a run writes its own outputs).
    """

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")  # paths are relative to the repo root
    if status is None:
        return {"commit": commit, "dirty": None}
    changes = [
        line
        for line in status.splitlines()
        if not (line.startswith("?? ") and line[3:].strip('"').startswith(ignore_untracked))
    ]
    return {"commit": commit, "dirty": bool(changes)}


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str | None]:
    """Installed versions of the tracked packages (None if not installed)."""
    versions: dict[str, str | None] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def make_run_id(name: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{name}_s{seed}"


def create_run_dir(
    cfg: dict[str, Any],
    seed: int,
    name: str,
    results_root: str | os.PathLike | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    """Create results/<run_id>/ and save config.yaml and meta.json into it.

    meta.json holds the run id, seed, git commit, package versions, Python/platform info
    and the command line, so every number can be traced back to code and environment.
    """
    root = Path(results_root) if results_root is not None else REPO_ROOT / "results"
    run_id = make_run_id(name, seed)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    meta = {
        "run_id": run_id,
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "packages": package_versions(),
        "python": sys.version,
        "platform": platform.platform(),
        "argv": sys.argv,
    }
    if extra_meta:
        meta.update(extra_meta)
    save_json(meta, run_dir / "meta.json")
    return run_dir


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """Write JSON; non-serialisable values (dtypes, paths, ...) are stored as strings."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
