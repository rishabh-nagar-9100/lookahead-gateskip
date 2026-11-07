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
        "accelerator": accelerator_info(),
    }
    if extra_meta:
        meta.update(extra_meta)
    save_json(meta, run_dir / "meta.json")
    return run_dir


def resolve_device(requested: str = "auto") -> str:
    """'auto' -> cuda, else mps, else cpu; anything else is returned unchanged."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cuda_has_native_bf16() -> bool:
    """True on Ampere or newer (A100, L4, RTX 30xx/40xx). A T4 (sm_75) is False.

    ``torch.cuda.is_bf16_supported()`` is not used: recent torch releases report True on a T4
    because bf16 can be emulated there, which is slow and not what we want to measure.
    """
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) >= (8, 0)


def resolve_dtype_name(name: str, device: str) -> str:
    """Map a configured dtype name to the one actually used on ``device``.

    'auto' -> bfloat16 on CUDA with native bf16 (L4/A100), float16 on older CUDA (T4) and on MPS,
    float32 on CPU. An explicit 'bfloat16' on a CUDA GPU without native bf16 falls back to float16;
    the caller records both names (meta.json), so the substitution is never silent.
    """
    dev = device.split(":")[0]
    if name == "auto":
        if dev == "cuda":
            return "bfloat16" if cuda_has_native_bf16() else "float16"
        return "float16" if dev == "mps" else "float32"
    if name == "bfloat16" and dev == "cuda" and not cuda_has_native_bf16():
        return "float16"
    return name


def accelerator_info() -> dict[str, Any]:
    """GPU type and capabilities for meta.json (empty fields on CPU / MPS)."""
    info: dict[str, Any] = {"cuda": False}
    try:
        import torch
    except ImportError:
        return info
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(
            cuda=True,
            gpu_name=p.name,
            gpu_count=torch.cuda.device_count(),
            compute_capability=f"{p.major}.{p.minor}",
            total_memory_gb=round(p.total_memory / 2**30, 1),
            native_bf16=cuda_has_native_bf16(),
            cuda_version=torch.version.cuda,
        )
    else:
        info["mps"] = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    info["colab"] = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    return info


def save_json(obj: Any, path: str | os.PathLike) -> None:
    """Write JSON; non-serialisable values (dtypes, paths, ...) are stored as strings."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
