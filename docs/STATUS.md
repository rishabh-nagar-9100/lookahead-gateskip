# Status and resume guide

**Read this first.** It is the one-page entry point to the project. It was last updated on **2026-11-11**, when the project was paused for a week. Details live in the files it links to.

## Where things stand

The code for Phases 0–2 is done and tested on CPU and MPS. The Phase 3 training pipeline is ready (data, shared LoRA recipe, checkpoints with resume, sharded eval, calibration), but **nothing has run on a CUDA GPU yet**. The next step is Colab session S1 on a free T4.

| Phase | Status | Task IDs |
|---|---|---|
| 0 Setup | **Partly done.** Done: P0-02, -03, -04, -05, -10, -11, -12, -14, -15. Open: P0-13 (= Colab S1, then 200-step timing on an L4), P0-08, P0-09. Blocked: P0-01 (Llama licence), P0-06 (Llama E0), P0-07 (running official GateSkip) | P0-01 … P0-15 |
| 1 Pilot study | **Partly done.** P1-02 (gate logger) done. P1-01 and P1-03 to P1-06 need trained models (S3) | P1-01 … P1-06 |
| 2 Core implementation | **Done on CPU/MPS.** P2-00 to P2-06, P2-08 and P2-09 done. P2-07 is green on CPU and MPS; CUDA is pending (S1). P2-10 is optional | P2-00 … P2-10 |
| 3 Training & calibration | **Partly done.** P3-01 and P3-06 done; the P3-02 loop works on CPU. P3-07 Dense-FT, the P3-08 format check, P3-03 λ sweep, P3-04 calibration and P3-05 E4 all need the L4 | P3-01 … P3-08 |
| 4 Evaluation & ablations | **Not started** | P4-01 … P4-06 |
| 5 Latency | **Partly done.** P5-05 done (dense engine 1.05× HF on MPS); P5-06 in progress; P5-01 first pass on MPS; P5-07 and P5-08 proposed only | P5-01 … P5-08 |
| 6 Paper | **Not started** | P6-01 … P6-05 |
| 7 Final | **Not started** | P7-01 … P7-04 |

Live task list: [TASKS.md](TASKS.md).

## What exists now

- **Last code commit before this pause:** `f87f5b8` (D-13 active). The pause commit itself only touches docs; `git log -1` shows the latest one.
- **Tests:** 183 pass on CPU (`pytest -q`). CI is GitHub Actions on CPU and passed on `f87f5b8` (run 42349555697).
- **Package `src/lagate/`:**

  | Module | What it does |
  |---|---|
  | `models/adapter.py` | Per-layer attention, MLP and K,V access for Llama and Qwen2; logits match HF |
  | `models/gated.py` | Soft-gated training forward for our method (CE + λ·L_S) |
  | `routing/ema.py`, `routing/router.py` | Cross-token EMA memory and the dual-head router (g_L, g_M with ΔAttn) |
  | `routing/gateskip.py` | GateSkip port: training forward plus a fixed-threshold engine policy |
  | `engine/tiered.py`, `engine/flops.py` | bs=1 three-tier engine with exact Tier-3 K,V; FLOP counter |
  | `calibration/ecdf.py` | Empirical-CDF thresholds, the same rule for both methods |
  | `training/loop.py` | Shared loop for ours, the GateSkip port and Dense-FT: LoRA, shared recipe, fp32 trainables under autocast, checkpoint and resume, HF export |
  | `training/chat_data.py` | D-06 data in chat format, assistant-only loss, fixed calibration hold-out |
  | `eval/answer.py` | Robust secondary GSM8K extractor |
  | `analysis/gate_logger.py`, `analysis/gate_stats.py` | Gate logging and E2/E3 helpers |
  | `utils.py` | Configs, seeds, run dirs, git and GPU info in `meta.json`, dtype policy |

- **Scripts (`scripts/`):** `train.py`, `calibrate.py`, `eval_dense.py` (with `--shard`/`--num-shards`), `merge_shards.py`, `prepare_chat_data.py`, `real_model_checks.py` (T-EQ, greedy match, bs=1 speed), `flop_report.py`, `gpu_budget.py`, `setup_gpu.sh` (stages `env`, `test`, `teq`, `smoke`), `colab_helpers.py`.
- **Configs (`configs/`):**
  - Shared training recipe: `train/recipes/qwen_lora16.yaml` (D-13).
  - Dense-FT: `train/denseft_qwen.yaml`.
  - Data: `data/d06_chat.yaml`.
  - Evaluation: `eval/gsm8k_*` (D-08, D-11, D-15 chat, Dense-FT test and calibration split).
  - CUDA checks: `checks/qwen_cuda.yaml`, `checks/llama_cuda.yaml`.
  - Budget: `budget/gpu_budget.yaml`.
  - **Still to write:** the gated Qwen training configs (GateSkip port and ours) and `configs/calibrate/<method>_qwen.yaml`.
- **Notebook:** `notebooks/colab_runner.ipynb`. It has Setup cells, one cell for each setup stage, and one config-driven cell per job (smoke, speed, train, calibrate, sharded eval, merge, P3-08 check, manual commit). See [GPU_RUNBOOK.md §9](GPU_RUNBOOK.md#9-google-colab-d-23).

## Active decisions (one line each; full text in [DECISIONS.md](DECISIONS.md))

- **D-01:** build on GateSkip as the base method.
- **D-02:** memory runs across tokens, not across depth (depends on the Phase 1 go/no-go).
- **D-03:** Tier 3 computes its own K,V; K,V are never copied across layers.
- **D-04:** two gates (g_L, g_M), not two thresholds on one score.
- **D-05:** the primary metric is GSM8K accuracy at matched compute against GateSkip.
- **D-06:** training data is GateSkip's `csqa_gsm8k_gen` mix (GSM8K + CommonsenseQA).
- **D-09:** environments use `uv` + `.venv`.
- **D-10:** CI installs CPU-only test dependencies.
- **D-12:** GateSkip is retrained on our model and evaluated at bs=1 with fixed calibrated thresholds; D-12(c) checks the port against the official code.
- **D-13:** LoRA r=16 for the main comparison.
  - One shared recipe (`configs/train/recipes/qwen_lora16.yaml`); configs can't override it.
  - Routers and gates are trained fully at `method.router_lr`.
  - Trainable parameters are fp32 under bf16 autocast; fp16 training is refused.
  - D-12(c) stays separate.
- **D-15:** trained models are trained and evaluated in chat format (0-shot, `#### N`).
  - Untrained models are reported under D-08.
  - The Dense-FT model is the 0%-skip reference.
  - The P3-08 format check must pass before any gated training.
- **D-16:** everything lives in our own `lagate` package; GateSkip is re-implemented there and checked against the official code.
- **D-17:** implementation details the spec leaves open (router, EMA, sparsity costs).
- **D-18:** the GateSkip port uses fixed per-module thresholds from the calibration-split ECDF (a documented adaptation).
- **D-19 / D-20:** research gaps G1–G6, in [PAPER_PLAN.md](PAPER_PLAN.md).
- **D-21:** "greedy" means pure greedy, with `repetition_penalty` 1.0 in every generation config.
- **D-23:** Qwen2.5-1.5B-Instruct is the primary model until Llama is approved; the GPU is Colab.
  - T4 runs in fp16 and is for tests only.
  - L4 runs in bf16 and does training and reported evaluations.
- **D-08:** active only as the protocol for untrained dense models.
  - D-11 and D-14 are superseded by D-15.
- **Proposed or open, don't implement without a go-ahead:**
  - D-22 / P5-07 / E12: lookahead decision mode. Not before Phase 3 training works on the GPU; decide after the first CUDA latency numbers.
  - P5-08: batched masked evaluation. It needs the bs=1 equivalence test, including the left-padding conditions in TASKS.
  - D-07 (gate scaling vs binary gates at inference) is still open.

## Key findings so far

- **GateSkip's thresholds skip nothing at bs=1 decode.** Its inference threshold is a quantile over the current forward pass. One decode token gives a threshold of 0.0, so no module is skipped (0 skips over 400 decode tokens, from their own `find_threshold`). Hence D-12/D-18: fixed calibrated thresholds. ([LITERATURE.md](LITERATURE.md), DECISIONS D-18)
- **GateSkip's K,V copy is a silent no-op.** The intended copy from layer ℓ−1 fails inside a swallowed `try/except` or returns early, so their published numbers use each token's own exact K,V. Their wrapper also runs every module and then zeroes its output, so it saves no real compute. ([LITERATURE.md](LITERATURE.md), evidence for claim 2)
- **Gate cost.** GateSkip's two H×H vector gates cost 8.39M MACs per token and layer, **13.7%** of a dense Llama-1B layer at n = 128 (10.0% for Qwen). Our router costs **≈ 0.6%**. (runs `20261106-162120`, `-162121`)
- **MPS host-sync cost.**
  - The dense engine runs at 40.8 ms/token against HF's 39.0 (1.05×).
  - The routing decision path with no skipping is 61.6 ms/token, and our router + EMA 65.5 (1.68×). That is ≈ 21 ms/token of host syncs (2 × 28 layers × ≈ 0.37 ms) plus ≈ 4 ms of router compute.
  - One sync per layer only gets this to 62.2 ms/token. Reaching dense speed needs ≈ 0 in-loop syncs (D-22), or CUDA, where a sync is much cheaper (not yet measured).
  - Runs `20261106-180357`, `-190740`; risk R4.
- **Qwen smoke results** (50 GSM8K problems, MPS fp16, untrained Qwen2.5-1.5B-Instruct):

  | Protocol | Strict | Flexible | Robust | Note | Run |
  |---|---|---|---|---|---|
  | D-08, pure greedy | 62.0% | 62.0% | 68.0% | 128 tokens on average | `20261106-182046` |
  | D-11 (no chat template) | 0.0% | 16.0% | — | Degenerate | — |
  | Chat protocol, untrained | 26.0% | 56.0% | 64% | Fails the format check, which motivated D-15 and Dense-FT | — |

- **Real-model correctness (Qwen, MPS):** T-EQ max abs diff 0.0 in fp32 and fp16, and the dense engine matches HF pure greedy on 3 of 3 prompts.
- **Training data (P3-06):** 16,211 train examples (6,973 GSM8K + 9,238 CSQA) and 1,000 calibration (500 + 500).
  - Mean 445 tokens per example, about **1.5×** the budget assumption; 7.2M tokens per epoch.
  - One epoch is 1,013 steps at an effective batch of 16.

## Blockers

1. **Llama licence (P0-01):** the token works, but `meta-llama/Llama-3.2-1B-Instruct` still returns 403. Qwen is primary meanwhile (D-23). This blocks P0-06, the E10 replication, and D-12(c) / P0-07, which also need base Llama and an A100.
2. **Colab S1 not yet run:** nothing has run on CUDA. All GPU work waits for S1.

## Next steps, in order

1. **S1 · T4 (free):** run Setup, then `test`, `teq` (includes the bs=1 CUDA speed check: HF vs dense engine vs routing path), `smoke` (50-problem D-08; expect ≈ 62%; T4 is fp16, so small differences are fine). If anything fails, fix the notebook or `setup_gpu.sh` before S2. Register the pushed runs in EXPERIMENTS.md.
2. **S2 · L4:**
   1. Run `teq` again for bf16 numbers.
   2. Time 200 training steps per method (P0-13) and update `configs/budget/gpu_budget.yaml` and GPU_BUDGET.md.
   3. P3-07 Dense-FT (`configs/train/denseft_qwen.yaml`, seed 1).
   4. P3-08: sharded eval of `configs/eval/gsm8k_chat_denseft_qwen_calib.yaml`, merge, then the check cell. **If P3-08 fails, stop and revisit the protocol.**
3. **Before S3, on the Mac:** write the gated Qwen training configs on the shared recipe (`method.router_lr`, λ) and `configs/calibrate/<method>_qwen.yaml`.
4. **S3 · L4:** train the GateSkip port, calibrate it, run E1 at 0/15/25/35% (sharded), and log gates for E2/E3. Then pilot go/no-go (P1-05).
5. **S4 · L4:** train ours (λ from the P3-03 sweep), calibrate τ_L/τ_M (P3-04), and run E4 at 3 budgets.

## Colab checklist (every session)

- [ ] **Secrets** (key icon). The values are never pasted into a cell.
  - `HF_TOKEN`: read token.
  - `GITHUB_TOKEN`: a **fine-grained** PAT for this repo only, with *Contents: read and write* and an expiry date.
  - **Notebook access** turned on for both.
- [ ] **Runtime type:** T4 GPU for S1; **L4** for S2–S4. The training cells refuse to run on a T4.
- [ ] **Cells to run:** all Setup cells top to bottom (settings, Drive + secrets, clone, GPU, env), then only that session's cells:
  - S1: `test`, `teq`, `smoke`.
  - S2: `teq`, training, sharded eval, merge, P3-08 check.
- [ ] **What to bring back:** each job's `finish()` pushes the small result files itself (`config.yaml`, `meta.json`, `metrics.json`; for calibration also `thresholds.json`).
  - Next time, tell Claude Code the session number and paste any failing cell's output.
  - Raw generations and checkpoints stay on Drive (`MyDrive/lagate/`); `results_mirror/` holds a safety copy.
- [ ] **If a job stopped before `finish`,** run the manual commit cell.
- [ ] **Disconnect when done:** Runtime → Disconnect and delete runtime. An idle runtime still spends units.

## Budget notes

- **Colab budget:** 200 compute units per month. Prefer the **L4** (bf16, 24 GB, cheaper per hour than an A100); use the T4 only for setup checks.
- **Training is longer than assumed:** the data averages 445 tokens per example, about 1.5× the budget's assumption. Expect training time to scale up by about that much until it is measured.
- **Measure real usage in P0-13:** time 200 steps per method on the L4 and note the units spent per hour (shown in Colab's resources panel). Then update `configs/budget/gpu_budget.yaml` and [GPU_BUDGET.md](GPU_BUDGET.md).
- **Eval cost:** without P5-08 (batched eval), bs=1 evaluation of gated models dominates the GPU estimate (≈ 4× more). P5-08 is proposed but not approved.

## Open questions and known risks

Full register: [RISKS.md](RISKS.md). The ones that matter now:

- **R4, no bs=1 speedup:** host syncs dominate on MPS; the CUDA numbers come from S1.
- **R5, GPU time:** only 200 units a month, and the data is 1.5× longer than budgeted.
- **R10, Dense-FT fails the P3-08 format check.**
- **R11, Llama licence never arrives:** the paper would be Qwen-only plus another replication model.
- **Open questions:**
  - Does P3-08 pass?
  - Is the CUDA sync cost small enough that D-22 is unnecessary?
  - Which λ values to sweep (P3-03)?
  - Should P5-08 be approved, given the GPU budget?
  - D-07 is still open.

## How to resume

Open a new Claude Code session in `lookahead-gateskip/` and give it this as the first message:

> Read `docs/STATUS.md` first, then `CLAUDE.md`, `docs/TASKS.md` and the active entries in `docs/DECISIONS.md`. We're resuming after a week's pause. Run `git pull` and `pytest -q` and tell me if anything changed or fails. Then: I have [not yet run / just run] Colab session S1 [paste any failing cell output here]. If S1 results were pushed, register them in EXPERIMENTS.md and DEV_LOG.md, fix any first-use problems in the notebook or `setup_gpu.sh`, and prepare S2. Keep the project rules: ask before downloads, installs, deletes or runs over 15 minutes; never write tokens into files; commit messages start with the task ID.
