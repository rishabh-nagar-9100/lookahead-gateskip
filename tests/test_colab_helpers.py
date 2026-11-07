"""notebooks/colab_runner.ipynb helpers: token hygiene, what may be committed, Drive links."""

import importlib.util
import os

import pytest

from lagate.utils import REPO_ROOT

spec = importlib.util.spec_from_file_location("colab_helpers", REPO_ROOT / "scripts" / "colab_helpers.py")
ch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ch)

FAKE = "ghp_" + "x" * 36  # not a real token


def test_git_command_never_contains_the_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", FAKE)
    cmd = ch.git_cmd("push", "origin", "HEAD:main")
    assert not any(FAKE in a for a in cmd)
    assert any("${GITHUB_TOKEN}" in a for a in cmd)  # expanded by the shell at call time only


def test_scrub_removes_tokens(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "y" * 30)
    s = ch.scrub(f"url https://x-access-token:{FAKE}@github.com token={os.environ['HF_TOKEN']}")
    assert FAKE not in s and os.environ["HF_TOKEN"] not in s


def test_only_small_result_files_are_committable(tmp_path):
    run = tmp_path / "results" / "20261107-000000_x_s0"
    run.mkdir(parents=True)
    for f in ("config.yaml", "meta.json", "metrics.json", "thresholds.json", "big.npz"):
        (run / f).write_text("{}")
    (run / "huge_metrics.json").write_text("x")
    (tmp_path / "results" / "raw" / "r").mkdir(parents=True)
    (tmp_path / "results" / "raw" / "r" / "metrics.json").write_text("{}")
    big = tmp_path / "results" / "20261107-000001_y_s0"
    big.mkdir()
    (big / "metrics.json").write_bytes(b"0" * (ch.MAX_COMMIT_BYTES + 1))
    paths = [str(p.relative_to(tmp_path)) for p in (tmp_path / "results").rglob("*") if p.is_file()]
    keep = ch.select_committable(paths, tmp_path)
    assert keep == sorted(f"results/20261107-000000_x_s0/{f}" for f in ("config.yaml", "meta.json", "metrics.json", "thresholds.json"))


def test_link_to_drive_creates_symlinks_and_refuses_nonempty(tmp_path):
    repo, drive = tmp_path / "repo", tmp_path / "drive"
    (repo / "results").mkdir(parents=True)
    ch.link_to_drive(repo, drive)
    assert (repo / "checkpoints").is_symlink() and (repo / "results" / "raw").is_symlink()
    ch.link_to_drive(repo, drive)  # idempotent
    (repo / "checkpoints").unlink()
    (repo / "checkpoints").mkdir()
    (repo / "checkpoints" / "x.pt").write_text("w")
    with pytest.raises(RuntimeError, match="not empty"):
        ch.link_to_drive(repo, drive)


def test_same_code_identical_commit():
    head = os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip()
    assert ch.same_code(REPO_ROOT, head, head)
