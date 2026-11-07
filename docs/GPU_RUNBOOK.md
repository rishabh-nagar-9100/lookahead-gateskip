# GPU Runbook (Linux + NVIDIA CUDA)

Goal: from a fresh CUDA machine to a verified setup in about an hour, so Phase 3 can start right away. Everything below is also scripted in `scripts/setup_gpu.sh`. The stages can be run one by one, and the script stops at the first failure.

**Status:** written on macOS. `bash -n` passes, and the `dotenv` and `test` stages have been run locally. The `clone`, `env`, `teq` and `smoke` stages have **not been run on a CUDA machine yet**. Expect to fix small things on first use, and note them in DEV_LOG.md.

## 0. What you need

| Item | Value |
|---|---|
| OS | Linux x86_64 (Ubuntu 22.04/24.04 tested by most providers) |
| GPU | NVIDIA, ≥ 24 GB (RTX 4090 / L4 / A10G / A100 / H100); bf16 support (Ampere or newer) |
| Driver | recent enough for the CUDA wheel you pick. `nvidia-smi` shows the highest CUDA version the driver supports; choose `CUDA_TAG` (e.g. `cu128`, `cu126`) at or below it |
| Disk | ≥ 100 GB: HF cache ≈ 3 GB per model + datasets, checkpoints (LoRA small; full FT ≈ 3–6 GB each), results |
| Accounts | GitHub access to the private repo; Hugging Face token with Llama-3.2 access (P0-01) |
| Tools | git, curl, bash. `uv` is installed by the script if missing |

## 1. Clone

```bash
git clone https://github.com/rishabh-nagar-9100/lookahead-gateskip.git
cd lookahead-gateskip
```

The repo is private, so authenticate in one of these ways. **Never write a token into a file in the repo.**
- `gh auth login` (GitHub CLI, browser or device code). Recommended.
- An HTTPS personal access token, typed at git's password prompt, with `git config --global credential.helper 'cache --timeout=36000'` so it stays in memory only.
- A read-only deploy key (SSH) added to the repo's settings.

`bash scripts/setup_gpu.sh clone` does the clone if you run the script from outside the repo.

## 2. Environment (CUDA torch + the lock file)

```bash
bash scripts/setup_gpu.sh env            # optional: CUDA_TAG=cu126 bash scripts/setup_gpu.sh env
```

What it does:
1. Checks `nvidia-smi` and installs `uv` if needed.
2. Creates `.venv` with Python 3.11 (D-09).
3. Installs `torch==<lock pin>` from `https://download.pytorch.org/whl/$CUDA_TAG`.
4. Installs every other package **exactly as in `requirements.lock.txt`**. The lock's torch line is the macOS build, so it's skipped.
5. Installs `lagate` in editable mode, `--no-deps`.
6. Asserts `torch.cuda.is_available()` and prints the GPU and bf16 support.
7. Saves `uv pip freeze` to `.venv/freeze_gpu.txt`. Diff it against the lock and note any differences in SETUP.md.

If the torch pin has no wheel for your `CUDA_TAG`, pick the closest tag that exists, or override with `TORCH_VERSION=...`, and record that in DECISIONS (it changes the environment).

## 3. `.env` (Hugging Face token)

```bash
bash scripts/setup_gpu.sh dotenv         # creates .env from .env.example (chmod 600) if missing
nano .env                                # set HF_TOKEN=... (needed for Llama; Qwen works without it)
bash scripts/setup_gpu.sh dotenv         # reports "set" / "EMPTY", never prints the token
```

`.env` is git-ignored. Don't `cat` it in shared terminals or logs.

## 4. Unit tests

```bash
bash scripts/setup_gpu.sh test           # pytest; must be all green (141 on 2026-11-07)
```

## 5. T-EQ at full scale + first latency numbers

```bash
export HF_HOME=/path/on/persistent/disk/hf   # cloud: keep the cache on the volume
bash scripts/setup_gpu.sh teq
```

This runs `scripts/real_model_checks.py` with `configs/checks/qwen_cuda.yaml`, then with `configs/checks/llama_cuda.yaml` if your token has Llama access. Each run writes `results/<run_id>/metrics.json`. Expected:
- **Dims** match config.json.
- **T-EQ:** max |Δlogit| ≈ 0 in fp32 and bf16. On MPS it was 0.0 for both. On CUDA a small nonzero value (≤ 1e-3 in bf16) is acceptable; record it.
- **Greedy:** the dense engine is identical to HF pure greedy (repetition_penalty 1.0, D-21) on 3/3 prompts.
- **Latency:** bs=1 ms/token for HF, the dense engine, the decision path and our router in both sync modes. These are the inputs for P5-02, D-22 and `configs/budget/gpu_budget.yaml`.

The first run downloads the models (≈ 3 GB each) and GSM8K: the stage sets `HF_HUB_OFFLINE=0` for itself.

## 6. 50-problem smoke test

```bash
bash scripts/setup_gpu.sh smoke
```

This runs `scripts/eval_dense.py --limit 50` with the D-08 config: Llama if accessible, else Qwen. Compare with the MPS reference: Qwen D-08, pure greedy, 62.0% strict / flexible (run `20261106-182046`). The numbers can move by a few problems between MPS fp16 and CUDA bf16; record the difference.

## 7. After the checks

1. Register every new `results/` run in `docs/EXPERIMENTS.md` (run log) and add a DEV_LOG entry.
2. P0-13: replace the assumptions in `configs/budget/gpu_budget.yaml` with the measured numbers (200 real training steps per method, decode ms/token), re-run `scripts/gpu_budget.py`, and update `docs/GPU_BUDGET.md`.
3. `git add` (results without `results/raw/`; checkpoints stay out of git), commit with task IDs, push.
4. Phase 3 starts with P3-06 (chat-format data), then P3-07 (Dense-FT) and the P3-08 format checkpoint.

All at once: `bash scripts/setup_gpu.sh all` (clone env dotenv test teq smoke). It pauses nowhere, so edit `.env` first if you need Llama.

## 8. Cloud GPU rental

Applies to RunPod, Lambda, Vast.ai, Paperspace, or GCP/AWS/Azure VMs.

1. **Choose an instance.** A 24 GB card (4090 / L4 / A10G) for LoRA and the minimal plan; an A100 80 GB for full fine-tunes and E1-official (GPU_BUDGET.md). Use an image with the NVIDIA driver and CUDA 12.x, Ubuntu 22.04+, and Python available (any version: uv installs 3.11).
2. **Storage.** Attach a persistent volume (≥ 100 GB) and put the repo, `HF_HOME` and `checkpoints/` on it, so a stopped or preempted instance loses nothing. Example: `export HF_HOME=/workspace/hf` and clone into `/workspace/`.
3. **Access.** Use SSH with your own key; never paste tokens into provider web consoles that log input. Use `gh auth login --web` or a deploy key for GitHub. Put `HF_TOKEN` only in `.env` on the volume.
4. **Run.** `tmux new -s lagate`, then `bash scripts/setup_gpu.sh all`. Long training and eval runs go in tmux, so a dropped SSH session doesn't kill them.
5. **Cost control.**
   - Note the hourly rate and stop the instance when idle; most providers bill a stopped instance's volume only.
   - Spot/interruptible instances are fine for evals and resumable training, but not for latency measurements.
   - Run latency (E11) on a dedicated, idle GPU and record the exact GPU model, driver and clock settings.
6. **Get results back.**
   - Commit and push `results/<run_id>/` (small JSON).
   - Copy raw generations and checkpoints with `rsync -avP user@host:/workspace/lookahead-gateskip/results/raw/ results/raw/`. They are git-ignored and must never be committed.
7. **Shut down.** `git status` clean and pushed → stop or delete the instance → delete the volume only after checkpoints are copied somewhere safe.

## 9. Google Colab (D-23)

Everything runs from **`notebooks/colab_runner.ipynb`**. Open it in Colab (File → Open notebook → GitHub, or upload it), pick the runtime type, and run the Setup cells top to bottom at the start of every session.

**GPU roles.**
- **T4** (free, sm_75, no native bf16): fp16. Use it **only** for `pytest`, T-EQ, smoke runs and speed checks. Its latency numbers are setup checks, not E11 results.
- **L4** (paid, 24 GB, bf16): training and every reported evaluation. The training cells refuse to run on a T4.
- **A100:** only needed for an optional full fine-tune reference row (D-13).

**One-time setup.**
1. Colab Secrets (key icon in the left sidebar):
   - `HF_TOKEN`: your read token.
   - `GITHUB_TOKEN`: a **fine-grained** personal access token, limited to this repository, with *Contents: read and write* and an expiry date.
   - Enable "Notebook access" for both. Never paste a token into a cell.
2. Google Drive: the notebook creates `MyDrive/lagate/` with these folders:

   | Folder | Holds | Notes |
   |---|---|---|
   | `hf_cache/` | `HF_HOME` | Models and datasets download once |
   | `checkpoints/` | Training checkpoints | Symlinked from the repo's `checkpoints/`; also `$LAGATE_CHECKPOINT_ROOT` |
   | `raw/` | Raw generations | Symlinked from `results/raw/` |
   | `results_mirror/` | Copy of every `results/<run_id>/` | Safety copy in case a push fails |

**What happens in Setup.**
- **Tokens:** they are read with `google.colab.userdata` into the kernel's environment only. Git authenticates through an inline credential helper that expands `$GITHUB_TOKEN` at call time, so the token never lands in `.git/config`, a remote URL, a file or the cell output (output is also scrubbed).
- **Repo:** cloned to `/content/lookahead-gateskip` (fast, ephemeral), or pulled if it is already there.
- **Environment:** `.venv` is built with `setup_gpu.sh env`, using CUDA torch from `CUDA_TAG` (default `cu128`) plus `requirements.lock.txt`. This takes about 3–5 min per new runtime.
- **GPU detection:** `nvidia-smi` gives the name and compute capability, which decide bf16 (L4/A100) vs fp16 (T4). `lagate.utils.accelerator_info()` writes the GPU into every run's `meta.json`.

**Jobs.** Each cell is one job and takes a config path:

| Job | Script | Notes |
|---|---|---|
| Setup stages | `setup_gpu.sh test`, `teq`, `smoke` | One cell each |
| Smoke test | `eval_dense.py --limit 50` | |
| CUDA speed check | `real_model_checks.py` | HF vs our dense engine vs the routing path, bs=1 |
| Training | `train.py` | Resumes from `latest.pt` on Drive; Dense-FT also exports a merged HF model |
| Calibration | `calibrate.py` | |
| Sharded evaluation | `eval_dense.py --shard i --num-shards n` | Finished shards are skipped, so the cell can be re-run in later sessions |
| Merge | `merge_shards.py` | |
| P3-08 check | — | Reads the merged run and applies the D-15 criteria |

Each job ends with `finish()`. It mirrors `results/` to Drive, then commits and pushes **only** the small files (`config.yaml`, `meta.json`, `metrics.json`, thresholds; each ≤ 2 MB) after `git pull --rebase`.

**Session limits.**
- Training checkpoints every 30 min. After a disconnect, run Setup and then the same training cell; it resumes at the saved step with the same data order.
- Size eval shards so that one finishes well inside a session: 4 shards for the GSM8K test set on an L4.
- All shards of one evaluation must use the same GPU type; the merge checks this.

**Session order (TASKS.md, "Colab session plan").**

| Session | GPU | Jobs |
|---|---|---|
| S1 | T4 | Setup, `test`, `teq` (with the speed check), `smoke` |
| S2 | L4 | Dense-FT + P3-08 |
| S3 | L4 | GateSkip port + pilot E1–E3 |
| S4 | L4 | Ours + calibration + E4 |

**Colab-specific troubleshooting.**
- `torch.cuda.is_available()` is False after `env`: the runtime's driver is older than `CUDA_TAG`. Set `CUDA_TAG = "cu126"` in the settings cell, delete `.venv` (`!rm -rf /content/lookahead-gateskip/.venv`) and re-run.
- "MISSING" for a secret: add it in Secrets and toggle notebook access, then re-run the secrets cell.
- Push rejected: run the manual commit cell, which rebases on `origin/main`. Results files never conflict, because each run has its own directory.
- Drive is slow for the first model load (≈ 3 GB from `hf_cache`). Later loads in the same session are cached by the OS.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available()` is False | `CUDA_TAG` newer than the driver supports: re-run `env` with a lower tag (check `nvidia-smi`) |
| 403 on Llama | Licence not approved yet, or wrong token in `.env` (P0-01). Qwen stages still run |
| NaN logits in fp16 | Expected for eager attention on Qwen2.5 (D-21 notes, P2-01 fix): our adapter uses SDPA; use bf16 on CUDA |
| HF greedy ≠ engine | Check `repetition_penalty` 1.0 (D-21) and that sampling is off |
| Out of memory in a full fine-tune on 24 GB | Use LoRA (D-13) or gradient checkpointing / 8-bit Adam; note it in DECISIONS |
