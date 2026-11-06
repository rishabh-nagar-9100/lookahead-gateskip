# Setup

## Accounts and access

| Item | Needed for | Status |
|---|---|---|
| Hugging Face account + read token | downloading models and datasets | ☐ |
| Approved access to `meta-llama/Llama-3.2-1B-Instruct` (gated; accept Meta's licence on the model page) | main model | ☐ |
| `Qwen/Qwen2.5-1.5B-Instruct` | second model (no approval needed) | ☐ |
| GPU with bf16 support and ≥ 24 GB VRAM (e.g. A100, L4, RTX 3090/4090) | training and GPU latency | ☐ |
| Apple Silicon Mac (M1 or newer) | edge (MPS) latency | ☐ |
| Private GitHub repository | code, docs, CI | ☐ (local git repo done; remote not yet created) |
| Weights & Biases account (optional) | run tracking | ☐ |

Never commit tokens. Put them in `.env` (git-ignored) and load with `python-dotenv`.

## Hardware record

| Machine | GPU / chip | VRAM / RAM | OS | Access hours | Notes |
|---|---|---|---|---|---|
| Dev Mac | Apple Silicon (MPS) | | macOS 26 (Darwin 25.6) | always | local dev, MPS latency; fill in chip/RAM |
| GPU machine | TBD (P0-02) | | | | |

## Environment

We use `uv` with a project-local `.venv` (DECISIONS D-09; conda is not installed on the dev Mac).

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt     # on a CUDA machine install the matching torch build first
uv pip install -e . --no-deps
uv pip freeze --exclude-editable > requirements.lock.txt
```

Exact reproduction from the lock file: `uv pip install -r requirements.lock.txt && uv pip install -e . --no-deps`.

Recorded 2026-11-06 on the dev Mac (macOS 26 / Darwin 25.6, Apple Silicon, Python 3.11.15):

| Package | Version |
|---|---|
| torch | 2.15.0 (MPS available) |
| transformers | 5.22.1 |
| peft | 0.21.2 |
| lm-eval | 0.4.13 |
| accelerate | 1.15.0 |
| datasets | 5.1.0 |
| CUDA driver / toolkit | n/a on Mac; fill in for the GPU machine (P0-02) |

## Baseline code

| Baseline | Source | Pinned commit | Runs? |
|---|---|---|---|
| GateSkip | https://github.com/Thiggel/GateSkip | `8d565130266fecc4a2a5374d0ced135f05e4b9f9` | ☐ (cloned; needs CUDA env, see baselines/README.md) |
| D-LLM | https://github.com/Jyk-122/D-LLM | | ☐ |
| SkipGPT | https://github.com/EIT-NLP/SkipGPT | | ☐ |
| FlexiDepth | released by the authors (find repo link in the arXiv page) | | ☐ |
| FFN-skip (DiffSkip-style) | re-implement in `baselines/` if no code | – | ☐ |

## Notes

- Apple MPS has incomplete bf16/op coverage; use fp16 or fp32 there and record which.
- Hugging Face `generate()` adds overhead; latency numbers come from our own decode loop (P5-01).

## GPU machine (Linux + CUDA)

See [GPU_RUNBOOK.md](GPU_RUNBOOK.md) and `scripts/setup_gpu.sh`: the same lock file with a CUDA torch build (`CUDA_TAG`, default cu128). Budget: [GPU_BUDGET.md](GPU_BUDGET.md).
