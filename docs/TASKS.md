# Task Tracker

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked (say why).
Each task has an ID; reference it in commits (e.g. `P1-03: add gate correlation script`).

## Phase 0 – Setup
- [!] P0-01 Request Hugging Face access to meta-llama/Llama-3.2-1B-Instruct; create read token — token created and working (user rishabh9100); Llama licence approval still pending (403 GatedRepoError on 2026-11-06)
- [!] P0-02 Confirm GPU access (model, VRAM, hours/week); record in SETUP.md — waiting on Rishabh / department
- [x] P0-03 GitHub repo with the README layout and .gitignore (results/raw, checkpoints, .env): https://github.com/rishabh-nagar-9100/lookahead-gateskip, `main` pushed 2026-11-07 (keychain credentials, no token in any file)
- [x] P0-04 Create Python 3.11 env; install pinned requirements; record versions in SETUP.md (uv, D-09; `requirements.lock.txt`)
- [x] P0-05 CI: GitHub Actions running pytest on CPU with tiny random models. First run failed (peft not installed); fixed with pinned deps. Run 42272751382 on `dc0c6b8` passed
- [!] P0-06 E0: dense Llama-3.2-1B-Instruct on GSM8K (50-problem smoke test, then full test set) — pipeline verified with a 50-problem Qwen smoke run (62.0%, run `20261106-182046_gsm8k_dense_qwen_s0`, pure greedy (D-21)); Llama runs blocked on gated-access approval (P0-01: token works, repo returns 403), full set ideally on GPU (P0-02)
- [~] P0-07 Clone GateSkip, pin commit hash in baselines/README.md, run its example. Cloned and pinned at `8d56513`, code read, notes in LITERATURE.md, D-06/D-11/D-12/D-13 recorded. **Running it is blocked:**
  - no released checkpoint, so it must be trained;
  - its stack needs CUDA (torch 2.8 cu12, flash-attn, vLLM) and Python 3.10 in a separate env (P0-02 GPU);
  - its main model is base `meta-llama/Llama-3.2-1B`, also gated (P0-01);
  - installing its dependencies is waiting on Rishabh's OK.
- [x] P0-10 Add D-11 eval configs (lm-eval `gsm8k`, 0-shot, no chat template, greedy, 512 tokens) and re-run the dense 50-problem smoke test with them. Configs `gsm8k_zs_dense_{llama,qwen}.yaml` + protocol test; run `20261106-060855_gsm8k_zs_dense_qwen_s0` gave 0.0% strict / 16.0% flexible (degenerate for an untrained Instruct model, D-14 proposed)
- [x] P0-11 Choose the dense-reference protocol. D-15: chat-format training and evaluation, untrained models under D-08, Dense-FT reference. Protocol B was checked in run `20261106-063044_gsm8k_chat_dense_qwen_s0` (failed for the untrained model, which motivated D-15)
- [x] P0-12 Robust secondary GSM8K extractor (`lagate.eval.answer`, D-15) handling `#### N`, `\boxed{N}`, `**N**`, `$N`, commas and "The (final) answer is N"; logged by `eval_dense.py` as `robust_extract`; tests include the five failing protocol-B outputs
- [x] P0-14 GPU prep: `docs/GPU_RUNBOOK.md` (incl. cloud rental), `scripts/setup_gpu.sh` (clone, env from lock with CUDA torch, .env, pytest, full-scale T-EQ, 50-problem smoke), CUDA check configs, GPU-hour estimate `docs/GPU_BUDGET.md`. CUDA stages not yet run on a CUDA machine
- [ ] P0-13 First GPU session: run `scripts/setup_gpu.sh all`, time 200 real training steps per method, then update `configs/budget/gpu_budget.yaml` and `docs/GPU_BUDGET.md` with measured numbers
- [ ] P0-08 Confirm review dates with guide; fill them into PROJECT_PLAN.md
- [ ] P0-09 Optional: Weights & Biases project for run tracking

## Phase 1 – Pilot study
- [ ] P1-01 E1: GateSkip on Llama-3.2-1B, GSM8K at 0/15/25/35% savings. No released checkpoint (P0-07), so train our port with their recipe (D-16, needs P2-09 + GPU)
- [x] P1-02 Hook to log per-token, per-layer gate values and skip decisions during generation — `lagate/analysis/gate_logger.py` (GateLogger on engine or soft forward → npz + sequence meta); E2/E3 helpers in `lagate/analysis/gate_stats.py`
- [ ] P1-03 E2: error-token analysis script + token tagging (digits, operators, connectives, boilerplate). Skip-rate helper `gate_stats.skip_rates` ready
- [ ] P1-04 E3: adjacent-token gate correlation per layer (k = 1…8), plots. Correlation helper `gate_stats.adjacent_gate_correlation` ready
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
- [ ] P2-07 Tests T-EQ, T-KV, T-T3, T-FLOP, T-EMA green on CPU and GPU — **CPU: all green** on tiny random Llama + Qwen2 (132 tests; T-KV/T-T3 in fp64, D-17). **MPS, real Qwen2.5-1.5B:** dims = config.json, T-EQ max abs diff 0.0 in fp32 and fp16, dense engine = HF pure greedy on 3 GSM8K prompts (run `20261106-172716_realcheck_qwen_mps_s0`). Llama (access) and CUDA (P0-02) pending
- [x] P2-08 Training smoke run (20 steps) — `lagate/training/loop.py` + `scripts/train.py`, 4 runs (ours, GateSkip × tiny Llama, Qwen2), see EXPERIMENTS run log
- [x] P2-09 GateSkip port `lagate/routing/gateskip.py` (vector gate on module input, L2 loss averaged over modules, percentile ramp, gate scaling at inference) with a hand-calculation unit test (D-16) — `lagate/routing/gateskip.py` (GateSkipModel training forward, GateSkipPolicy for the engine), `lagate/calibration/ecdf.py`, `configs/router/gateskip.yaml`; calibrated fixed thresholds (D-18); tests incl. official-code cross-checks
- [ ] P2-10 *(optional, low priority)* Engine option `kv_mode: copy_prev`: copy layer ℓ−1's cache entry into layer ℓ for skipped tokens. GateSkip's copy never runs (LITERATURE.md, claim 2), so this would only test an idea no published method uses. Do it only if time allows after Phase 4 (E1-copyKV)

## Phase 3 – Training & calibration
- [x] P3-01 Choose training data (match GateSkip's setup for fairness); log in DECISIONS.md. D-06 (GateSkip's `csqa_gsm8k_gen` mix), chat format per D-15
- [ ] P3-06 Convert the D-06 data to chat format (D-15). Each example: system instruction, question as the user turn, assistant turn `"{steps}\n#### {answer}"`; loss on assistant tokens only; use each model's chat template.
  - GSM8K uses the exact `system_instruction` from `configs/eval/gsm8k_chat_dense_*.yaml`. Read it from there, don't copy it; a test should check the two match.
  - CommonsenseQA gets a matching instruction asking for a final `#### <letter>` line.
  - Pin the dataset revisions, and add a unit test on a few rendered examples for Llama and Qwen.
- [ ] P3-07 Dense-FT reference training (D-15, E-DFT): same chat data, recipe, steps and seeds as the main comparison (D-13 fine-tuning mode), no router or gates. Llama-3.2-1B-Instruct first, then Qwen2.5-1.5B-Instruct (E-DFT-Q).
- [ ] P3-08 **Verification checkpoint (D-15):** on the first Dense-FT model (E-DFT seed 1), calibration split: ≥ 90% `#### N`, ≤ 2% cap hits, |strict − flexible| ≤ 4 pts; also log robust accuracy. **If it fails, revisit the protocol before training any gated model (P3-02 onward, P4-01 GateSkip).**
- [ ] P3-02 Training loop: LoRA r=16 + routers, compute-weighted sparsity loss, budget schedule (gated training starts only after P3-08 passes)
  - Done on CPU (P2-08): shared loop `lagate.training.loop` for ours / GateSkip port / Dense-FT, with full, LoRA and router-only modes and the compute-weighted L_S. Still to do: the real data source (P3-06), the budget schedule, and a LoRA r=16 config once D-13 is decided.
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
- [ ] P5-01 Batch-size-1 decode harness (warm-up, CUDA sync, median of N runs) — first pass in `scripts/real_model_checks.py` (MPS, HF vs engine variants, warm-up + median of 5); CUDA sync and E11 harness still to do
- [x] P5-05 Engine speed (dense mode): preallocated KV cache, lazy on-device FLOP stats, dense fast path, sync-free decode attention, EMA only when used, last-position logits. **Dense engine 1.05× HF on MPS** (40.8 vs 39.0 ms/token, was 2.38×); torch.compile not needed for this target
- [~] P5-06 Cut routing host syncs. Done: `sync_mode: per_layer` (speculative decode attention, 1 sync per layer), with tests. Measured on MPS: 65.1 → 62.2 ms/token (dense 40.8), so it is not enough. Next: CUDA numbers (P0-02), then decide D-22 (lookahead decision mode, E12)
- [ ] P5-07 *(if D-22 accepted; not before Phase 3 training works on the GPU)* Implement `decision: lookahead` as an optional mode (current mode stays the main method): router inputs from token t−1's state, MLP-head options (a) ΔAttn_(t−1) / (b) no ΔAttn / (c) hybrid, shifted training forward, engine path with no in-loop syncs. Tests: shifted-input equivalence (training forward vs engine), T-KV, T-T3. Then E12 on CUDA and MPS
- [ ] P5-08 *(proposed; biggest GPU-hour saving)* Batched masked accuracy evaluation for gated models with fixed thresholds; bs=1 stays for latency (E11). Cuts the estimate ≈ 4× (GPU_BUDGET.md). **Not to be used for any reported number until its equivalence test passes:**
  - Same tier decisions as the bs=1 engine, per token and layer, for every sequence in a mixed-length batch.
  - Same outputs: logits within tolerance, and greedy tokens equal (allow differences only at documented float near-ties, and report them).
  - Same FLOP counts and tier counts.
  - **Left padding:** pad tokens never update the EMA state (μ stays μ_0 = 0 until a row's first real token), never enter attention as keys, never get gate decisions or FLOP/tier counts, and don't shift positions (RoPE positions start at 0 at each row's first real token).
  - Tests: one row vs the same row alone in the batch; rows of different lengths; an all-pad prefix longer than the EMA chunk; at least one budget where all three tiers occur.
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
- ~~P0-03 GitHub remote~~ done 2026-11-07
- P0-07 run GateSkip: needs GPU (P0-02), base Llama-3.2-1B access (P0-01), OK to install its deps
- P0-06 Llama E0 runs — blocked on P0-01; full set ideally on GPU (≈12 s/problem on MPS → ~4.5 h for 1,319)
