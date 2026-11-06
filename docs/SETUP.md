# Setup

## Accounts and access

| Item | Needed for | Status |
|---|---|---|
| Hugging Face account + read token | downloading models and datasets | ☐ |
| Approved access to `meta-llama/Llama-3.2-1B-Instruct` (gated; accept Meta's licence on the model page) | main model | ☐ |
| `Qwen/Qwen2.5-1.5B-Instruct` | second model (no approval needed) | ☐ |
| GPU with bf16 support and ≥ 24 GB VRAM (e.g. A100, L4, RTX 3090/4090) | training and GPU latency | ☐ |
| Apple Silicon Mac (M1 or newer) | edge (MPS) latency | ☐ |
| Private GitHub repository | code, docs, CI | ☐ |
| Weights & Biases account (optional) | run tracking | ☐ |

Never commit tokens. Put them in `.env` (git-ignored) and load with `python-dotenv`.

## Hardware record

| Machine | GPU / chip | VRAM / RAM | OS | Access hours | Notes |
|---|---|---|---|---|---|
| | | | | | |

## Environment

```bash
conda create -n lagate python=3.11 -y
conda activate lagate
pip install torch            # pick the CUDA build that matches the driver
pip install transformers accelerate peft datasets lm-eval pyyaml python-dotenv pytest
pip install wandb            # optional
```

After installing, freeze exact versions:

```bash
pip freeze > requirements.lock.txt
python -c "import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)"
```

Record here:

| Package | Version |
|---|---|
| torch | |
| transformers | |
| peft | |
| lm-eval | |
| CUDA driver / toolkit | |

## Baseline code

| Baseline | Source | Pinned commit | Runs? |
|---|---|---|---|
| GateSkip | https://github.com/Thiggel/GateSkip | | ☐ |
| D-LLM | https://github.com/Jyk-122/D-LLM | | ☐ |
| SkipGPT | https://github.com/EIT-NLP/SkipGPT | | ☐ |
| FlexiDepth | released by the authors (find repo link in the arXiv page) | | ☐ |
| FFN-skip (DiffSkip-style) | re-implement in `baselines/` if no code | – | ☐ |

## Notes

- Apple MPS has incomplete bf16/op coverage; use fp16 or fp32 there and record which.
- Hugging Face `generate()` adds overhead; latency numbers come from our own decode loop (P5-01).
