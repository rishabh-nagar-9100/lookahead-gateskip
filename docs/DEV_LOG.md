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

### 2026-11-06 · Phase 2 core on CPU · P2-00, P2-02, P2-03 …
**Done:**
- D-16 active, with Rishabh's two additions: line-referenced evidence for the GateSkip claims (LITERATURE.md) and two port variants (exactKV main, copyKV for G4). Added P2-09, P2-10, P4-06 and E1-exactKV, E1-copyKV, E1-official.
- P2-02 EMA memory (`lagate/routing/ema.py`): chunked causal scan + decode step; T-EMA green.
- P2-03 dual-head router (`lagate/routing/router.py`, `configs/router/lagate.yaml`); open spec details recorded as D-17.
**Results / numbers:** none (CPU unit tests only).
**Problems / findings:**
- **Correction to the GateSkip code review entry below:** their K,V copy from ℓ−1 never actually happens. `update_kv_cache` reads the cache from `module_output[1]`, which is the attention-weight tensor; the resulting error or early `None` is swallowed. Shown by running their function verbatim (`scripts/check_gateskip_kvcopy.py`). Their cache is therefore exact, and the G4 evidence has to come from our copyKV run.
**Next:** P2-01 model adapter, P2-06 FLOP counter, P2-04 soft-gated forward.

### 2026-11-06 · D-15 protocol decision + robust extractor · P0-11, P0-12, P3-01
**Done:**
- Recorded **D-15**, which supersedes D-14 and D-11:
  - chat-format training and evaluation for all trained models;
  - original untrained models reported under D-08 only;
  - Dense-FT reference (no router), with E-DFT/E-DFT-Q rows in EXPERIMENTS.md;
  - format checkpoint P3-08 on the first Dense-FT model before any gated training.
- Added `lagate.eval.answer`, a robust secondary extractor handling `#### N`, `\boxed{N}`, "The (final) answer is N", `**N**`, `$`, commas and the last-number fallback. `eval_dense.py` logs it as `robust_extract`.
- Tests: 20 extractor cases, including the five protocol-B outputs strict-match rejected (docs 0, 1, 2, 3, 10), all scored correctly; plus a test that all D-15 configs share one system instruction.
- Config headers now state each protocol's role (D-08 for original models, D-15 main, D-11 superseded).
- Phase 3 reordered: P3-06 chat data conversion → P3-07 Dense-FT → P3-08 checkpoint → gated training.
**Results / numbers:** robust accuracy re-scored post hoc on the three E0 smoke runs (same 50 problems): D-08 64% (strict 62), D-11 48% (strict 0), chat 64% (strict 26). No false positives in a manual check of every case where robust disagreed with strict and flexible. 31 tests pass.
**Problems:** none new. The robust numbers for the three E0 runs come from re-scoring their saved samples at this commit, not from the original runs.
**Next:** waiting on Llama access (P0-01) and GPU (P0-02); then P3-06 data conversion and the P3-07 Dense-FT run.

### 2026-11-06 · Protocol-B check · P0-10, P0-11
**Done:**
- Added the custom lm-eval task `configs/eval/tasks/gsm8k_chat_zs.yaml` (bare question as the user turn, stock gsm8k scoring) and configs `gsm8k_chat_dense_{llama,qwen}.yaml`.
- `eval_dense.py` now supports `include_path` and `system_instruction`; the protocol test covers D-15 (10 tests pass).
- Previewed the rendered prompt, registered the run, and ran it on 50 problems (Qwen; Llama still gated).
**Results / numbers:** `20261106-063044_gsm8k_chat_dense_qwen_s0`: 26.0% strict / 56.0% flexible; 40% of outputs contain `#### N`; 2% hit the 512 cap; avg 219 tokens; 7:04 generation.

| Same 50 problems | Strict | Flexible | `####` | Hit cap | Avg tok |
|---|---|---|---|---|---|
| D-08 8-shot chat | 62.0 | 62.0 | n/a ("The final answer is") | 0% | 143 |
| D-11 0-shot plain | 0.0 | 16.0 | 0% | 40% | 352 |
| B 0-shot chat + instruction | 26.0 | 56.0 | 40% | 2% | 219 |

**Problems:** B fails 2 of 3 criteria. The untrained model ignores the `####` instruction and uses `\boxed{N}`, bold or "Therefore … $18". Flexible-extract is also fragile here: on doc 2 it extracted `$.` although the answer was a boxed `70000`. The decision is not recorded yet; Rishabh to choose a fallback after seeing the failing outputs.
**Next:** decide P0-11, then update the decision, configs and TASKS accordingly.

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
