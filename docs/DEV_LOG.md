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

### 2026-11-07 · GPU prep, budget, sync modes · P0-14, P5-06, D-22
**Done:**
- `docs/GPU_RUNBOOK.md` and `scripts/setup_gpu.sh`: stages clone / env (uv, Python 3.11, CUDA torch at the lock pin, rest of the lock) / dotenv (never prints the token) / test / teq (full-scale real-model checks, Qwen then Llama if accessible) / smoke (50 problems). Cloud rental section. CUDA check configs `configs/checks/{qwen,llama}_cuda.yaml`. Script syntax-checked; dotenv + test stages run locally; CUDA stages untested.
- GPU-hour estimator `scripts/gpu_budget.py` + `configs/budget/gpu_budget.yaml` → `docs/GPU_BUDGET.md` (minimal vs full plan, A100 vs 4090, batched vs bs=1 evals).
- Engine `sync_mode: per_layer`: speculative decode attention, so only the MLP decision syncs. Same outputs and decisions as per_module (test); extra executed MACs counted.
- D-22 lookahead decision mode (proposed) and E12; P5-07 (implement if accepted), P5-08 (batched masked eval, proposed).
**Results / numbers:**
- per_layer on MPS: 62.2 vs 65.1 ms/token (dense 40.8), only ≈ 3 ms saved. Any in-loop sync drains the queue (run `20261106-190740`).
- Budget: minimal ≈ 28 A100-h with batched evals (101 h at bs=1); full ≈ 91 h (459 h). Training is only 15–38 h of it.
**Problems:**
- The "earlier message" with specs for the one-sync change, D-22 and E12 never arrived in this session. They were built from my own R4 proposals (b) and (d); waiting for Rishabh to confirm.
- Budget training times are unmeasured (±2×).
**Next:** P0-13 on the first GPU session; decide P5-08 and D-22 after CUDA numbers.

### 2026-11-07 · GitHub push + CI · P0-03, P0-05
**Done:** remote `origin` = https://github.com/rishabh-nagar-9100/lookahead-gateskip.git; `main` pushed with the stored macOS keychain credentials (no token entered or written anywhere). Before pushing, checked that no `.env`, checkpoints, raw generations or token-like strings appear anywhere in the history.
**Results / numbers:**
- First Actions run (42272366671, `af8074a`): **failed** in `pytest`. The LoRA test imports `peft`, which CI did not install; reproduced locally with peft blocked (1 failed, 137 passed).
- Fix: CI installs peft and pins all test deps to `requirements.lock.txt` (D-10 update).
- Second run (42272751382, `dc0c6b8`): **passed** (job ≈ 60 s).
**Problems:** job logs need authentication (API 403), so the CI test count isn't visible from here. The three official-GateSkip cross-check tests skip in CI, because the clone is git-ignored.
**Next:** Llama access (P0-01), GPU (P0-02), P3-06 chat data.

### 2026-11-06 · D-21 pure greedy · E0 re-run
**Done:** D-21 active. `repetition_penalty: 1.0` added to all six eval configs, a protocol test enforces it, and the three earlier Qwen runs are flagged as penalty-1.1 in EXPERIMENTS.md. Re-ran the Qwen 50-problem D-08 smoke.
**Results / numbers:** `20261106-182046_gsm8k_dense_qwen_s0`:
- 62.0% strict / 62.0% flexible (same as the penalty-1.1 run); robust 68.0%.
- Avg 128 tokens (was 143); 1 cap hit (long algebra, not a loop); 360 s.
- 0/50 outputs identical to the old run, confirming the penalty really was applied before. 10 correctness flips, 5 each way.
- lm-eval's logged generation_kwargs now include repetition_penalty 1.0.
**Problems:** none. At n = 50 the ±6.9-pt stderr hides any accuracy effect of the penalty.
**Next:** check Llama's generation_config when access arrives; push once the URL arrives.

### 2026-11-06 · Engine speed · P5-05
**Done:**
- Engine hot path rewritten:
  - preallocated KV cache, written in place and doubling when full;
  - FLOP/tier stats kept on the device and computed when `stats` is read (no per-token `.tolist()`);
  - a dense fast path with no decisions or gathers;
  - decode attention without host syncs (`all_visible`);
  - one `bool()` per decision at decode instead of `nonzero`;
  - EMA updated only for policies that use it;
  - last-position logits in `generate`.
- New tests: cache growth past capacity (fp64 T-KV), fast path = general path, last-only logits. 138 passed.
- `real_model_checks.py` also times the general decision path and our router with no skipping.
**Results / numbers** (run `20261106-180357_realcheck_qwen_mps_s0`, Qwen2.5-1.5B fp16, MPS):
- T-EQ still 0.0; pure-greedy match 3/3.
- **Dense engine 40.8 vs HF 39.0 ms/token = 1.05×** (was 2.38×).
- Decision path, every token Tier 1: 61.6 (1.58×). Our router, no skipping: 65.5 (1.68×).
**Problems:** routing host syncs (≈ 0.37 ms each, 56 per token) cost more than the router itself. On MPS, skipping only pays off at high skip rates (R4). torch.compile was not tried: not needed for the dense target, and the sync cost is not something it removes.
**Next:** P5-06 (proposed) after CUDA numbers; D-21 decision; push once the URL arrives.

### 2026-11-06 · Gaps, gate-cost check, real-model checks on MPS · P0-07, P2-07, P5-01
**Done:**
- G1–G6 written into PAPER_PLAN.md (Research gaps) and linked from DECISIONS (D-19, D-20). FlexiDepth / D-LLM marked verified (Rishabh checked the papers).
- GateSkip gate cost verified in code (`GateLayer.py:34`, `ModelAdapter.py:72–76, 102–111`): Linear(H, H) per module, run on every token, = 2H² MACs per layer = 13.7% of a dense Llama-1B layer.
  - Their "compute saved" is a token-skip fraction that ignores the gates (`DefaultLightningModule.py:138–142`, `compare_results.py:19–25`).
  - The paper reports parameter overhead only. Its Table 13 prints 4.06% (H = 1024, L = 24), but Llama-3.2-1B dims give 10.8%.
- 300-step diagnostic re-run through `scripts/train.py` and registered (not for reporting; numbers identical to the scratchpad run).
- `scripts/real_model_checks.py` on the cached Qwen2.5-1.5B (MPS, no downloads).
**Results / numbers** (run `20261106-172716_realcheck_qwen_mps_s0`):
- Dims = config.json.
- T-EQ full scale: max abs diff 0.0 in fp32 and fp16.
- Dense engine = HF pure greedy on 3/3 GSM8K prompts × 64 tokens.
- bs=1 fp16 latency: HF 34.9 ms/token vs dense engine 83.1, so **2.38× slower**. Ablation (scratchpad): FLOP accounting ≈ 13 ms/token, row indexing and host syncs ≈ 18 ms/token; a plain dense loop reaches 54.5 vs HF 37.2.
**Problems:**
- **Bug fixed:** our eager attention overflowed fp16 on Qwen2.5 (q·kᵀ = inf at layer 0 → NaN; HF eager does the same). The adapter now uses SDPA, as HF's default does. Tiny fixtures were switched to sdpa so T-EQ compares like with like.
- **Protocol issue (D-21 proposed):** Qwen's generation_config applies repetition_penalty 1.1 even under greedy, and lm-eval doesn't reset it. Our earlier Qwen GSM8K numbers used it.
- HF timings varied 54 → 35 ms/token across sessions (thermal/background?), so treat single-session ratios with care.
**Next:** decide D-21; P5-05 engine speed fixes (proposed) before Phase 3; push once the GitHub URL arrives.

### 2026-11-06 · GateSkip port, training loop, gap revision · P2-09, P2-08, P3-02 (part)
**Done:**
- P2-09 GateSkip port (`lagate/routing/gateskip.py`): faithful training forward (vector gates on detached module inputs, output scaling, padding/EOS importance 1, percentile ramp, L2 loss) and `GateSkipPolicy` for the engine. Fixed per-module thresholds from the calibration-split empirical CDF (`lagate/calibration/ecdf.py`), recorded as a documented adaptation (D-18) with the `threshold_finder.py` evidence. Tests cross-check the ramp threshold and gate init against the official functions, and show their batch quantile skips 0 of 400 decode tokens at bs=1 while calibrated thresholds skip ≈ 25%.
- Engine policies now provide their own gate scaling (vector for GateSkip), MLP input and router cost.
- Shared training loop (`lagate/training/loop.py`, `scripts/train.py`): ours / GateSkip / Dense-FT; full, LoRA, router-only; synthetic data source; end-to-end smoke eval (calibrate → bs=1 engine).
- P2-08: four 20-step smoke runs (both methods × tiny Llama, Qwen2).
- D-19: G4 rests on FlexiDepth vs D-LLM; exact K,V is a correctness property, not a claim against GateSkip. The bs=1 threshold finding joins G6 and becomes contribution C6 (PAPER_PLAN). P2-10 / E1-copyKV are optional.
**Results / numbers:**
- FLOP report re-run: GateSkip's vector gates cost 2H² MACs per layer = 13.7% of a dense Llama-1B layer (10.0% Qwen), vs 0.6% for our router.
- Smoke runs (EXPERIMENTS run log): CE about 6.1 → 5.0 in 20 steps. A 300-step diagnostic (scratchpad, not registered) reached CE 0.89 (ours) and 1.32 (GateSkip), with 84% dense continuation accuracy, so the loop learns. At H=64 the gates are huge relative to a layer, so tiny-model compute numbers mean nothing.
**Problems:** G1–G3, G5 are not in the repo (only in the R0 deck).
**Next:** push to GitHub (waiting on the URL); P3-06 chat data; GPU (P0-02).

### 2026-11-06 · Phase 2 core on CPU · P1-02, P2-00 … P2-07
**Done:**
- D-16 active, with Rishabh's two additions: line-referenced evidence for the GateSkip claims (LITERATURE.md) and two port variants (exactKV main, copyKV for G4). Added P2-09, P2-10, P4-06 and E1-exactKV, E1-copyKV, E1-official.
- P2-02 EMA memory (`lagate/routing/ema.py`): chunked causal scan + decode step; T-EMA green.
- P2-03 dual-head router (`lagate/routing/router.py`, `configs/router/lagate.yaml`); open spec details recorded as D-17.
- P2-01 layer adapter (`lagate/models/adapter.py`): our own decoder loop over HF submodules (norms, K,V+RoPE, Q, position-masked attention, MLP) for Llama and Qwen2; equals HF logits and HF's cached K,V at 1e-5. Tiny random fixtures in `tests/conftest.py`.
- P2-06 FLOP counter (`lagate/engine/flops.py`) + `scripts/flop_report.py`; T-FLOP checks a hand calculation and PyTorch's measured matmul FLOPs.
- P2-04 soft-gated training forward (`lagate/models/gated.py`): h' = h + g_L·ΔAttn, h'' = h' + g_L·g_M·MLP, compute-weighted L_S over valid tokens. T-EQ green: router disabled equals HF logits at 1e-5, with and without right padding.
- P2-05 three-tier engine (`lagate/engine/tiered.py`): bs=1 greedy, its own K,V + EMA cache, exact K,V for every token in every tier, and Q/attention/O and the MLP run only on the rows that need them (checked with call counters). Decisions come from a policy (Lagate, Dense, FixedTier; GateSkip later). T-KV (fixed tiers and the real router) and T-T3 pass in fp64 at 1e-9. The dense engine reproduces HF greedy `generate`.
- P1-02 gate logger (`lagate/analysis/gate_logger.py`): attaches to the engine or the soft forward and saves per-token, per-layer g_L, g_M, run_attn, run_mlp, tier, position, token id, phase and sequence metadata (npz + JSON). Pilot helpers `adjacent_gate_correlation` (E3) and `skip_rates` (E2) in `gate_stats.py`.
- P2-07 on CPU: T-EQ, T-KV, T-T3, T-FLOP and T-EMA all green on tiny random Llama and Qwen2; full suite 107 passed (≈5 s). GPU run pending.
**Results / numbers:** FLOP reports `20261106-155823_flops_llama_s0`, `20261106-155824_flops_qwen_s0`. Router = 0.57–0.64% of a dense layer (Llama), 0.56–0.63% (Qwen). The MLP is 73–88% of a layer, so Tier 2 alone saves most of the compute; Tier 3 costs 2–4%.
**Problems / findings:**
- **Correction to the GateSkip code review entry below:** their K,V copy from ℓ−1 never actually happens. `update_kv_cache` reads the cache from `module_output[1]`, which is the attention-weight tensor; the resulting error or early `None` is swallowed. Shown by running their function verbatim (`scripts/check_gateskip_kvcopy.py`). Their cache is therefore exact, and the G4 evidence has to come from our copyKV run.
**Next:** P2-09 GateSkip port and P2-10 `copy_prev` KV mode (both CPU-only); P2-08 20-step training smoke run (needs a training loop, P3-01/P3-06 data). Real-model checks (Dims vs config.json, T-EQ at 1B scale) wait for Llama access / GPU.

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
