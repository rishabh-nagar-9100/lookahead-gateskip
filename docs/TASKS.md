# Task Tracker

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked (say why).
Each task has an ID; reference it in commits (e.g. `P1-03: add gate correlation script`).

## Phase 0 – Setup
- [!] P0-01 Request Hugging Face access to meta-llama/Llama-3.2-1B-Instruct; create read token — token created and working (user rishabh9100); Llama licence approval still pending (403 GatedRepoError on 2026-11-06)
- [!] P0-02 Confirm GPU access (model, VRAM, hours/week); record in SETUP.md — waiting on Rishabh / department
- [~] P0-03 Create private GitHub repo with the README layout; add .gitignore (results/raw, checkpoints, .env) — local repo, layout and .gitignore done; GitHub remote still to be created and pushed by Rishabh
- [x] P0-04 Create Python 3.11 env; install pinned requirements; record versions in SETUP.md (uv, D-09; `requirements.lock.txt`)
- [~] P0-05 CI: GitHub Actions running pytest on CPU with a tiny random model — workflow + tests written, pytest green locally; first GitHub run pending the push (P0-03)
- [!] P0-06 E0: dense Llama-3.2-1B-Instruct on GSM8K (50-problem smoke test, then full test set) — pipeline verified with a 50-problem Qwen smoke run (62.0%, run `20261106-053918_gsm8k_dense_qwen_s0`); Llama runs blocked on gated-access approval (P0-01: token works, repo returns 403), full set ideally on GPU (P0-02)
- [~] P0-07 Clone GateSkip, pin commit hash in baselines/README.md, run its example. Cloned and pinned at `8d56513`, code read, notes in LITERATURE.md, D-06/D-11/D-12/D-13 recorded. **Running it is blocked:**
  - no released checkpoint, so it must be trained;
  - its stack needs CUDA (torch 2.8 cu12, flash-attn, vLLM) and Python 3.10 in a separate env (P0-02 GPU);
  - its main model is base `meta-llama/Llama-3.2-1B`, also gated (P0-01);
  - installing its dependencies is waiting on Rishabh's OK.
- [x] P0-10 Add D-11 eval configs (lm-eval `gsm8k`, 0-shot, no chat template, greedy, 512 tokens) and re-run the dense 50-problem smoke test with them. Configs `gsm8k_zs_dense_{llama,qwen}.yaml` + protocol test; run `20261106-060855_gsm8k_zs_dense_qwen_s0` gave 0.0% strict / 16.0% flexible (degenerate for an untrained Instruct model, D-14 proposed)
- [x] P0-11 Choose the dense-reference protocol. D-15: chat-format training and evaluation, untrained models under D-08, Dense-FT reference. Protocol B was checked in run `20261106-063044_gsm8k_chat_dense_qwen_s0` (failed for the untrained model, which motivated D-15)
- [x] P0-12 Robust secondary GSM8K extractor (`lagate.eval.answer`, D-15) handling `#### N`, `\boxed{N}`, `**N**`, `$N`, commas and "The (final) answer is N"; logged by `eval_dense.py` as `robust_extract`; tests include the five failing protocol-B outputs
- [ ] P0-08 Confirm review dates with guide; fill them into PROJECT_PLAN.md
- [ ] P0-09 Optional: Weights & Biases project for run tracking

## Phase 1 – Pilot study
- [ ] P1-01 E1: GateSkip on Llama-3.2-1B, GSM8K at 0/15/25/35% savings (use released checkpoint if available, else train with their config)
- [ ] P1-02 Hook to log per-token, per-layer gate values and skip decisions during generation
- [ ] P1-03 E2: error-token analysis script + token tagging (digits, operators, connectives, boilerplate)
- [ ] P1-04 E3: adjacent-token gate correlation per layer (k = 1…8), plots
- [ ] P1-05 Write pilot findings section in DEV_LOG; record go/no-go in DECISIONS.md
- [ ] P1-06 R1 slides: pilot figures + implementation status

## Phase 2 – Core implementation
- [x] P2-00 Fork vs own package: D-16 active (own `lagate` package; GateSkip ported in two KV variants; evidence in LITERATURE.md)
- [x] P2-01 Model wrapper exposing per-layer attention / MLP / K,V projections for Llama and Qwen — `lagate/models/adapter.py` (Llama + Qwen2, no sliding window); logits = HF at 1e-5 on tiny configs
- [x] P2-02 EMA memory module (prefill scan + decode recurrence) with unit tests — `lagate/routing/ema.py`, T-EMA in `tests/test_ema.py`
- [x] P2-03 Dual-head router module (shared bottleneck, g_L, g_M with ΔAttn input), bias init — `lagate/routing/router.py`, `configs/router/lagate.yaml`, details D-17
- [x] P2-04 Soft-gated training forward pass — `lagate/models/gated.py` (LookaheadGatedModel, CE + λ·L_S, right padding); T-EQ in `tests/test_gated_forward.py`
- [x] P2-05 Three-tier inference engine with exact Tier-3 K,V — `lagate/engine/tiered.py` (bs=1, own KV+EMA cache, policies: Lagate/Dense/FixedTier), `configs/engine/lagate.yaml`; T-KV, T-T3 in `tests/test_engine.py`
- [x] P2-06 FLOP counter (per token, per tier) + router-overhead report — `lagate/engine/flops.py`, `scripts/flop_report.py`; router 0.57–0.64% of a dense layer (Llama), 0.56–0.63% (Qwen)
- [ ] P2-07 Tests T-EQ, T-KV, T-T3, T-FLOP green on CPU and GPU
- [ ] P2-08 Training smoke run (20 steps)
- [ ] P2-09 GateSkip port `lagate/routing/gateskip.py` (vector gate on module input, L2 loss averaged over modules, percentile ramp, gate scaling at inference) with a hand-calculation unit test (D-16)
- [ ] P2-10 Engine option `kv_mode: exact | copy_prev`. `copy_prev` copies layer ℓ−1's cache entry into layer ℓ for skipped tokens (GateSkip's intended behaviour), for E1-copyKV (D-16)

## Phase 3 – Training & calibration
- [x] P3-01 Choose training data (match GateSkip's setup for fairness); log in DECISIONS.md. D-06 (GateSkip's `csqa_gsm8k_gen` mix), chat format per D-15
- [ ] P3-06 Convert the D-06 data to chat format (D-15). Each example: system instruction, question as the user turn, assistant turn `"{steps}\n#### {answer}"`; loss on assistant tokens only; use each model's chat template.
  - GSM8K uses the exact `system_instruction` from `configs/eval/gsm8k_chat_dense_*.yaml`. Read it from there, don't copy it; a test should check the two match.
  - CommonsenseQA gets a matching instruction asking for a final `#### <letter>` line.
  - Pin the dataset revisions, and add a unit test on a few rendered examples for Llama and Qwen.
- [ ] P3-07 Dense-FT reference training (D-15, E-DFT): same chat data, recipe, steps and seeds as the main comparison (D-13 fine-tuning mode), no router or gates. Llama-3.2-1B-Instruct first, then Qwen2.5-1.5B-Instruct (E-DFT-Q).
- [ ] P3-08 **Verification checkpoint (D-15):** on the first Dense-FT model (E-DFT seed 1), calibration split: ≥ 90% `#### N`, ≤ 2% cap hits, |strict − flexible| ≤ 4 pts; also log robust accuracy. **If it fails, revisit the protocol before training any gated model (P3-02 onward, P4-01 GateSkip).**
- [ ] P3-02 Training loop: LoRA r=16 + routers, compute-weighted sparsity loss, budget schedule (gated training starts only after P3-08 passes)
- [ ] P3-03 λ sweep (3–4 values) on Llama
- [ ] P3-04 Calibration script: τ_L, τ_M from empirical CDF for 15/25/35%
- [ ] P3-05 E4: main GSM8K result, 3 seeds

## Phase 4 – Evaluation & ablations
- [ ] P4-01 Baselines: GateSkip, FlexiDepth, D-LLM, FFN-skip (DiffSkip-style), random skip, at matched compute
- [ ] P4-02 E5 full suite: GSM8K, ARC-Challenge, PIQA, MBPP
- [ ] P4-03 E6 EMA off · E7 α sweep · E8 previous-token gate · E9 no ΔAttn / two tiers
- [ ] P4-04 E10 Qwen2.5-1.5B replication
- [ ] P4-05 R2 slides
- [ ] P4-06 D-12(c): our GateSkip port vs official code (base Llama-3.2-1B, native format, 0% and 15%), on the GPU (D-16)

## Phase 5 – Latency
- [ ] P5-01 Batch-size-1 decode harness (warm-up, CUDA sync, median of N runs)
- [ ] P5-02 E11 CUDA latency for all methods; torch.compile / CUDA graphs variants
- [ ] P5-03 E11 Apple MPS latency
- [ ] P5-04 Output-length and time-per-problem analysis

## Phase 6 – Paper
- [ ] P6-01 Figures and tables from results/ via scripts (no manual numbers)
- [ ] P6-02 Draft all sections (PAPER_PLAN.md)
- [ ] P6-03 Final novelty check on arXiv (last 3 months)
- [ ] P6-04 Guide review → arXiv preprint → journal submission
- [ ] P6-05 R3 slides

## Phase 7 – Final
- [ ] P7-01 Demo app (dense vs ours, live tier heat-map)
- [ ] P7-02 Reproducibility pass from a clean clone
- [ ] P7-03 Department report in required format
- [ ] P7-04 Final presentation

## Blocked / waiting
- P0-01 Llama licence approval (token done) — waiting on Meta/Hugging Face
- P0-02 GPU model / VRAM / hours — Rishabh, department
- P0-03 GitHub remote — Rishabh (create private repo, push `main`)
- P0-07 run GateSkip: needs GPU (P0-02), base Llama-3.2-1B access (P0-01), OK to install its deps
- P0-06 Llama E0 runs — blocked on P0-01; full set ideally on GPU (≈12 s/problem on MPS → ~4.5 h for 1,319)
