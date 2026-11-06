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
