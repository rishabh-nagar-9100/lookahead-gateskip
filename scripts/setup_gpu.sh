#!/usr/bin/env bash
# Set up and verify LookAhead-GateSkip on a Linux CUDA machine (docs/GPU_RUNBOOK.md).
#
# Usage (from anywhere):
#   bash setup_gpu.sh [stage ...]
# Stages, run in this order when given (default: env test):
#   clone   clone the repo into $LAGATE_DIR (skipped if it exists)
#   env     uv venv (Python 3.11) + CUDA torch + requirements.lock.txt + editable lagate
#   dotenv  create .env from .env.example if missing (you then edit it); checks HF_TOKEN without printing it
#   test    pytest -q (CPU tests; must be green)
#   teq     full-scale real-model checks on CUDA (T-EQ, greedy match, bs=1 latency): Qwen, then Llama if accessible
#   smoke   50-problem GSM8K smoke test (D-08), Llama if accessible, else Qwen
#   all     = clone env dotenv test teq smoke
#
# Environment overrides:
#   LAGATE_REPO   (default https://github.com/rishabh-nagar-9100/lookahead-gateskip.git)
#   LAGATE_DIR    (default ./lookahead-gateskip)
#   TORCH_VERSION (default: the torch pin in requirements.lock.txt)
#   CUDA_TAG      (default cu128; must match a wheel index at download.pytorch.org/whl/<tag>
#                  and be supported by the machine's NVIDIA driver, see `nvidia-smi`)
#   HF_HOME       (put it on a persistent volume on cloud machines; models are ≈ 3 GB each)
#
# Never pass tokens on the command line or write them into tracked files. HF_TOKEN goes in .env
# (git-ignored, chmod 600). For a private GitHub clone use `gh auth login` or a credential helper.
set -euo pipefail

LAGATE_REPO="${LAGATE_REPO:-https://github.com/rishabh-nagar-9100/lookahead-gateskip.git}"
LAGATE_DIR="${LAGATE_DIR:-$PWD/lookahead-gateskip}"
CUDA_TAG="${CUDA_TAG:-cu128}"
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(env test)
[ "${STAGES[0]}" = "all" ] && STAGES=(clone env dotenv test teq smoke)

log() { printf '\n[setup_gpu] %s\n' "$*"; }
die() { printf '\n[setup_gpu] ERROR: %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

stage_clone() {
  if [ -d "$LAGATE_DIR/.git" ]; then log "repo exists at $LAGATE_DIR (git pull to update)"; return; fi
  has git || die "git not found"
  log "cloning $LAGATE_REPO -> $LAGATE_DIR"
  git clone "$LAGATE_REPO" "$LAGATE_DIR"
}

in_repo() { [ -f "$LAGATE_DIR/requirements.lock.txt" ] || die "no repo at $LAGATE_DIR (run the clone stage)"; cd "$LAGATE_DIR"; }

stage_env() {
  in_repo
  has nvidia-smi || die "nvidia-smi not found: no NVIDIA driver on this machine"
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  if ! has uv; then
    log "uv not found; installing it for this user (https://docs.astral.sh/uv/)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi
  local tv="${TORCH_VERSION:-$(grep -E '^torch==' requirements.lock.txt | cut -d= -f3)}"
  [ -n "$tv" ] || die "no torch pin in requirements.lock.txt"
  log "creating .venv (Python 3.11), torch==$tv from the $CUDA_TAG index"
  uv venv .venv --python 3.11
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install "torch==$tv" --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  # Everything else exactly as locked; the lock's torch line is the macOS build, so skip it.
  grep -vE '^torch==' requirements.lock.txt > .venv/requirements.notorch.txt
  uv pip install -r .venv/requirements.notorch.txt
  uv pip install -e . --no-deps
  python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch: check CUDA_TAG vs `nvidia-smi` driver"
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, device: {torch.cuda.get_device_name(0)}, "
      f"bf16: {torch.cuda.is_bf16_supported()}")
PY
  uv pip freeze > .venv/freeze_gpu.txt
  log "environment ready; package list saved to .venv/freeze_gpu.txt (compare with requirements.lock.txt)"
}

activate() { in_repo; [ -f .venv/bin/activate ] || die "no .venv (run the env stage)"; source .venv/bin/activate; }

stage_dotenv() {
  in_repo
  if [ ! -f .env ]; then
    cp .env.example .env
    chmod 600 .env
    log "created .env from .env.example. Edit it now and set HF_TOKEN (needed for Llama), e.g. with: nano .env"
  fi
  chmod 600 .env
  activate
  python - <<'PY'
from dotenv import dotenv_values
tok = (dotenv_values(".env").get("HF_TOKEN") or "").strip()
print("[setup_gpu] HF_TOKEN in .env:", "set" if tok else "EMPTY (Qwen works without it; Llama needs it)")
PY
}

stage_test() {
  activate
  log "pytest (CPU tests)"
  pytest -q -o addopts="" | tail -3
}

llama_access() {
  python - <<'PY'
import os, sys
from dotenv import load_dotenv
load_dotenv(".env")
from huggingface_hub import model_info
try:
    model_info("meta-llama/Llama-3.2-1B-Instruct")
    sys.exit(0)
except Exception as e:  # gated (403) or no token
    print(f"[setup_gpu] Llama not accessible: {type(e).__name__}", file=sys.stderr)
    sys.exit(1)
PY
}

stage_teq() {
  activate
  set -a; [ -f .env ] && source .env; set +a   # HF_TOKEN into the environment (not printed)
  export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0  # first run downloads models / GSM8K into $HF_HOME
  log "real-model checks: Qwen2.5-1.5B (T-EQ fp32 + bf16, greedy match, bs=1 latency)"
  python scripts/real_model_checks.py --config configs/checks/qwen_cuda.yaml --seed 0
  if llama_access; then
    log "real-model checks: Llama-3.2-1B"
    python scripts/real_model_checks.py --config configs/checks/llama_cuda.yaml --seed 0
  else
    log "skipping Llama checks (no gated access yet, P0-01)"
  fi
}

stage_smoke() {
  activate
  set -a; [ -f .env ] && source .env; set +a
  local cfg=configs/eval/gsm8k_dense_qwen.yaml
  if llama_access; then cfg=configs/eval/gsm8k_dense_llama.yaml; fi
  log "50-problem GSM8K smoke test (D-08): $cfg"
  python scripts/eval_dense.py --config "$cfg" --seed 0 --limit 50
}

for s in "${STAGES[@]}"; do
  case "$s" in
    clone) stage_clone ;; env) stage_env ;; dotenv) stage_dotenv ;; test) stage_test ;;
    teq) stage_teq ;; smoke) stage_smoke ;;
    *) die "unknown stage '$s' (clone env dotenv test teq smoke all)" ;;
  esac
done
log "done: ${STAGES[*]}. Register any results/ runs in docs/EXPERIMENTS.md, then commit and push."
