"""Helpers for notebooks/colab_runner.ipynb (standard library only: imported before the venv exists).

Secrets: HF_TOKEN and GITHUB_TOKEN live in Colab Secrets and are only ever placed in the
environment of this kernel and its child processes. Git authenticates through an inline credential
helper passed with ``-c`` on each command; the helper text contains the *name* $GITHUB_TOKEN, which
the shell expands at call time, so the token is never written to .git/config, a remote URL, a file
or a log. Command echo and errors are scrubbed of both tokens just in case.

Persistence: the repo and venv live on the fast, ephemeral /content disk; Drive holds what must
survive a session: HF cache, checkpoints, raw generations and a mirror of results/.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/rishabh-nagar-9100/lookahead-gateskip.git"
SECRET_NAMES = ("HF_TOKEN", "GITHUB_TOKEN")

# Files from results/<run_id>/ that may be committed: small JSON/YAML only. Raw generations
# (results/raw/) and checkpoints are git-ignored and stay on Drive.
COMMITTABLE = {"config.yaml", "meta.json", "metrics.json", "thresholds.json", "calibration_rows.json",
               "rendered_examples.json"}
MAX_COMMIT_BYTES = 2_000_000

_CRED_HELPER = '!f() { echo username=x-access-token; echo "password=${GITHUB_TOKEN}"; }; f'


def scrub(text: str) -> str:
    for name in SECRET_NAMES:
        val = os.environ.get(name)
        if val:
            text = text.replace(val, f"<{name}>")
    return re.sub(r"(gh[pousr]_|github_pat_|hf_)[A-Za-z0-9_]{10,}", r"\1<redacted>", text)


def sh(cmd: str | list[str], cwd: str | Path | None = None, check: bool = True, env: dict | None = None) -> int:
    """Run a command, streaming scrubbed output. Returns the exit code (raises if check)."""
    args = cmd if isinstance(cmd, list) else ["bash", "-lc", cmd]
    shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(a) for a in cmd)
    print(f"$ {scrub(shown)}", flush=True)
    p = subprocess.Popen(args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for line in p.stdout:
        print(scrub(line), end="", flush=True)
    rc = p.wait()
    if check and rc != 0:
        raise RuntimeError(f"command failed ({rc}): {scrub(shown)}")
    return rc


def git_cmd(*args: str) -> list[str]:
    """git with the inline credential helper (needs GITHUB_TOKEN in the environment)."""
    return ["git", "-c", "credential.helper=", "-c", f"credential.helper={_CRED_HELPER}", *args]


def git(*args: str, cwd: str | Path | None = None, check: bool = True) -> int:
    if not os.environ.get("GITHUB_TOKEN"):
        raise RuntimeError("GITHUB_TOKEN is not set (add it to Colab Secrets and run the secrets cell)")
    return sh(git_cmd(*args), cwd=cwd, check=check)


def clone_or_pull(repo_dir: str | Path, branch: str = "main") -> None:
    repo_dir = Path(repo_dir)
    if (repo_dir / ".git").exists():
        git("fetch", "origin", branch, cwd=repo_dir)
        git("checkout", branch, cwd=repo_dir)
        git("pull", "--ff-only", "origin", branch, cwd=repo_dir)
    else:
        git("clone", "--branch", branch, REPO_URL, str(repo_dir))
    url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=repo_dir, capture_output=True, text=True).stdout
    if "@" in url.split("//", 1)[-1].split("/", 1)[0]:
        raise RuntimeError("the origin URL contains credentials; it must be the plain https URL")


def link_to_drive(repo_dir: str | Path, drive_root: str | Path) -> dict[str, str]:
    """results/raw and checkpoints/ in the repo become symlinks to Drive (both are git-ignored)."""
    repo_dir, drive_root = Path(repo_dir), Path(drive_root)
    links = {"results/raw": drive_root / "raw", "checkpoints": drive_root / "checkpoints"}
    for rel, target in links.items():
        target.mkdir(parents=True, exist_ok=True)
        path = repo_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            if Path(os.readlink(path)) == target:
                continue
            path.unlink()
        elif path.exists():
            if any(path.iterdir()):
                raise RuntimeError(f"{path} exists and is not empty; move its contents to {target} first")
            path.rmdir()
        path.symlink_to(target, target_is_directory=True)
    return {k: str(v) for k, v in links.items()}


def gpu_info() -> dict[str, str]:
    """GPU name and the dtype policy: bf16 on Ampere+ (L4, A100), fp16 on a T4 (tests/smoke/speed only)."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                        "--format=csv,noheader"], capture_output=True, text=True)
    if q.returncode != 0:
        return {"gpu": "none"}
    name, mem, driver, cap = [x.strip() for x in q.stdout.splitlines()[0].split(",")]
    major = int(cap.split(".")[0])
    bf16 = major >= 8
    return {"gpu": name, "memory": mem, "driver": driver, "compute_cap": cap,
            "dtype": "bfloat16" if bf16 else "float16", "training_allowed": str(bf16)}


def changed_result_files(repo_dir: str | Path) -> list[str]:
    """Untracked or modified files under results/ that are allowed to be committed."""
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", "results"],
                         cwd=repo_dir, capture_output=True, text=True, check=True).stdout
    return select_committable([line[3:].strip('"') for line in out.splitlines()], repo_dir)


def select_committable(paths: list[str], repo_dir: str | Path) -> list[str]:
    keep = []
    for p in paths:
        parts = Path(p).parts
        if len(parts) != 3 or parts[0] != "results" or parts[1] == "raw" or parts[2] not in COMMITTABLE:
            continue
        f = Path(repo_dir) / p
        if f.is_file() and f.stat().st_size <= MAX_COMMIT_BYTES:
            keep.append(p)
    return sorted(keep)


def mirror_results(repo_dir: str | Path, drive_root: str | Path) -> None:
    """Copy results/<run_id>/ (not raw/) to Drive, so nothing is lost if a push fails."""
    src, dst = Path(repo_dir) / "results", Path(drive_root) / "results_mirror"
    for d in src.iterdir():
        if d.is_dir() and d.name != "raw" and not d.is_symlink():
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)


def commit_and_push(repo_dir: str | Path, message: str, branch: str = "main") -> list[str]:
    """Commit only small result files (config/meta/metrics/thresholds), rebase on origin, push."""
    files = changed_result_files(repo_dir)
    if not files:
        print("nothing to commit")
        return []
    sh(["git", "add", "--", *files], cwd=repo_dir)
    sh(["git", "commit", "-m", message], cwd=repo_dir)
    git("pull", "--rebase", "origin", branch, cwd=repo_dir)
    git("push", "origin", f"HEAD:{branch}", cwd=repo_dir)
    return files


def same_code(repo_dir: str | Path, a: str, b: str) -> bool:
    """Same rule as lagate.utils.same_code: commits differing only in results/ and docs/."""
    if a == b:
        return True
    p = subprocess.run(["git", "diff", "--name-only", a, b], cwd=repo_dir, capture_output=True, text=True)
    return p.returncode == 0 and all(x.startswith(("results/", "docs/")) for x in p.stdout.split())


def completed_shards(repo_dir: str | Path, name: str, num_shards: int, seed: int) -> set[int]:
    """Shard indices that already have a finished run (metrics.json) with the current code."""
    import json

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True).stdout.strip()
    done = set()
    pat = re.compile(rf"_{re.escape(name)}_shard(\d+)of{num_shards}_s{seed}$")
    for d in (Path(repo_dir) / "results").glob(f"*_{name}_shard*of{num_shards}_s{seed}"):
        m = pat.search(d.name)
        if m and (d / "metrics.json").exists():
            commit = json.loads((d / "meta.json").read_text())["git"]["commit"]
            if head and commit and same_code(repo_dir, commit, head):
                done.add(int(m.group(1)))
    return done
