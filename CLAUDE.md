# Project context for AI coding assistants

LookAhead-GateSkip: token-wise dynamic layer skipping for SLMs (Llama-3.2-1B-Instruct, Qwen2.5-1.5B-Instruct), built on GateSkip (https://github.com/Thiggel/GateSkip).

## Read first
- **`docs/STATUS.md` – read this before anything else.** Current status, blockers, next steps and the Colab checklist.
- `docs/ARCHITECTURE.md` – the spec. Code must match it.
- `docs/TASKS.md` – current task IDs and status.
- `docs/DECISIONS.md` – do not reverse an active decision without adding a new entry.

## Core design (short)
- Per-layer cross-token EMA: μ_t = α·RMSNorm(h_t) + (1−α)·μ_(t−1).
- Router: z = SiLU(W_1[RMSNorm(h) ∥ μ]); g_L = σ(w_L·z + b_L); g_M = σ(w_M·[z ∥ W_a·RMSNorm(ΔAttn)] + b_M); biases init +5.
- Tiers: 1 full; 2 attention only; 3 K,V projection only (with RoPE) – never copy K,V across layers.

## Conventions
- Python 3.11, PyTorch, Hugging Face transformers/peft. Package name `lagate` under `src/`.
- Configs in YAML under `configs/`; no hard-coded hyperparameters in scripts.
- Every script takes `--config` and `--seed`, writes to `results/<run_id>/` with the config and git hash.
- Tests in `tests/`; run `pytest -q` before committing. T-EQ, T-KV, T-T3, T-FLOP, T-EMA must stay green.
- Commit messages start with the task ID, e.g. `P2-05: tier-3 K,V path`.
- Never commit tokens, checkpoints or raw generations.

## After finishing a task
Update `docs/TASKS.md` (status), append to `docs/DEV_LOG.md`, and register any run in `docs/EXPERIMENTS.md`. Keep `docs/STATUS.md` current when a phase, blocker or next step changes.
