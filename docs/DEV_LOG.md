# Development Log

Newest entry first. One entry per work session. Keep it short and factual.

## Template

```
### YYYY-MM-DD · <hours> h · <task IDs>
**Done:**
**Results / numbers:** (link run IDs from EXPERIMENTS.md)
**Problems:**
**Next:**
```

---

### 2026-11-06 · D-11 eval configs + smoke run · P0-10
**Done:**
- Added `configs/eval/gsm8k_zs_dense_{llama,qwen}.yaml` (D-11) and tagged every eval config with `protocol: D-08|D-11`.
- The config test now checks each config against its declared protocol (8 tests pass).
- `eval_dense.py` now logs `frac_hit_max_gen_toks`, de-duplicated per document.
- Registered and ran the 50-problem Qwen smoke run (Llama still gated; no new downloads, `HF_HUB_OFFLINE=1`).
**Results / numbers:** `20261106-060855_gsm8k_zs_dense_qwen_s0`: 0.0% strict / 16.0% flexible. Avg output 352 tokens; 40% of outputs hit the 512 cap; 0/50 contain `#### N`. Generation 14:06 (≈17 s/problem). Same 50 problems under D-08: 62.0%.
**Problems:** without a chat template, the untrained Instruct model answers and then runs on (no end-of-turn token). D-11 is therefore only meaningful for models trained on the D-06 format. I proposed D-14: the dense reference under D-11 is the trained model at 0% skip, and untrained dense is reported under D-08. The run took 14.4 min, close to the 15-min limit; slower than D-08 because the outputs are longer.
**Next:** Rishabh to confirm D-14; Llama access (P0-01); GPU (P0-02).

### 2026-11-06 · GateSkip code review · P0-06 fix, P0-07
**Done:**
- `git_info` now marks a run dirty for any change except untracked files under `results/`, with a new test (6 tests pass).
- Cloned GateSkip to `baselines/GateSkip` (git-ignored) and pinned `8d56513` in `baselines/README.md` and SETUP.md.
- Read its README, job scripts, gating, evaluator and data code; notes in LITERATURE.md.
- Decisions: D-06 resolved (GateSkip's `csqa_gsm8k_gen` data), D-08 superseded by D-11 (GateSkip's 0-shot `#### N` GSM8K format, full test set), D-12 added (retrain GateSkip on Instruct, evaluate at bs=1 with fixed calibrated thresholds), D-13 opened (full fine-tune vs LoRA).
**Results / numbers:** no runs. Hub check: no GateSkip checkpoints or datasets released.
**Problems / findings:**
- No checkpoint exists, so reproducing GateSkip needs a GPU. Their recipe: 1× A100 80 GB, full fine-tune, 2 h cap per seed.
- Main 1B runs use **base** Llama-3.2-1B.
- GSM8K eval is hard-coded to the first 100 problems.
- Inference thresholds are batch quantiles, so there is no skipping at bs=1 decode.
- Skipping zeroes outputs after running the module (no real savings in HF eval).
- The skipped-token KV code copies from layer ℓ−1 inside a silent `try/except`.
- Their dependency stack (CUDA-only, Python 3.10, lm-eval ≤ 0.4.8) conflicts with ours.
**Next:** P0-10 (D-11 eval configs + dense smoke re-run); GPU details (P0-02); OK to set up the GateSkip env on the GPU machine.

### 2026-11-06 · E0 smoke run · P0-01, P0-06
**Done:** HF token in `.env` verified (user rishabh9100). Llama-3.2-1B-Instruct still gated (403) → per plan ran the 50-problem smoke test on Qwen2.5-1.5B-Instruct instead. Registered in EXPERIMENTS.md before launch (commit e9e5451).
**Results / numbers:** `20261106-053918_gsm8k_dense_qwen_s0`: GSM8K (first 50 test problems) 62.0% strict-match, 62.0% flexible-extract (31/50, stderr 6.9 pts); avg output 143.3 tokens (max 312, so the 512 cap never hit); generation 10:00 on MPS fp16 (≈12 s/problem); 1042 s total including first-time downloads. Spot-checked misses: genuine reasoning errors, not answer-format failures (46/50 outputs end with "The final answer is").
**Problems:** `meta.json` git `dirty` flag counted the run's own untracked results dir → fixed in `utils.git_info` (`--untracked-files=no`). Full test set on MPS would take ~4.5 h → GPU.
**Next:** wait for Llama approval, then Llama 50-problem smoke; full-set E0 on GPU (P0-02); P0-07 GateSkip clone and check D-08 against its eval.

### 2026-11-06 · Phase 0 scaffolding · P0-03, P0-04, P0-05, P0-06
**Done:** git repo initialised (`main`) with the README layout (`src/lagate/*` subpackages, configs, scripts, tests, results, baselines, paper, notebooks) and `.gitignore`. `pyproject.toml` (package `lagate`, Python 3.11, src layout), `requirements.txt`, `requirements.lock.txt`, `.env.example`. uv `.venv` with Python 3.11.15 (D-09). `src/lagate/utils.py` (config loading, seeding, `results/<run_id>/` with config, git hash, package versions). `scripts/eval_dense.py` + `configs/eval/gsm8k_dense_{llama,qwen}.yaml` (task choice D-08). `tests/test_smoke.py` (tiny random Llama forward pass, utils, config protocol checks). CI workflow `.github/workflows/tests.yml` (CPU only, D-10).
**Results / numbers:** `pytest -q` → 5 passed locally (CPU, no downloads). No E0 numbers yet.
**Problems:** E0 cannot run until Llama access + HF token exist. Doc issues found: SETUP used conda (→ D-09); GSM8K harness variant unspecified (→ D-08); CLAUDE.md lists T-EMA as required but PROJECT_PLAN Phase 2 omits it; ARCHITECTURE §7 says "the MLP share depends on n" but only the attention term depends on n.
**Next:** P0-01/P0-02 (Rishabh), push to GitHub and check CI, E0 50-problem smoke run, P0-07 GateSkip clone (confirm D-08 against their eval).

### 2026-11-06 · Planning
**Done:** Zeroth-review deck v2 finalised; project docs created (plan, tasks, architecture, experiments, decisions, literature, setup, risks, paper plan).
**Next:** P0-01 to P0-04 (access, GPU, repo, environment).
