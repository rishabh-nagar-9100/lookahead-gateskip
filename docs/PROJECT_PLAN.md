# Project Plan

Timeline: November 2026 – April 2027. Review points follow the department calendar (dates to be confirmed): R0 (Nov), R1 (Dec), R2 (Jan), R3 (Mar), final (Apr).

## Definition of done

The project is complete when all of these hold:

1. A working, tested implementation of LookAhead-GateSkip runs on Llama-3.2-1B-Instruct and Qwen2.5-1.5B-Instruct.
2. Results compare it with dense, GateSkip and at least two other baselines at matched compute budgets (15%, 25%, 35%) on GSM8K, ARC-Challenge, PIQA and MBPP.
3. Ablations isolate each component: EMA memory, ΔAttn input to the MLP head, number of tiers.
4. Batch-size-1 latency, end-to-end time per problem and output length are measured on at least one CUDA GPU and Apple MPS.
5. Code, configs and seeds reproduce every number in the paper from a clean clone.
6. An arXiv preprint is posted, a journal manuscript is submitted, and the final report and demo are delivered.

## Phases

### Phase 0 – Setup (Nov, week 1–2)
**Goal:** everyone and everything can run models.
- Get accounts and access listed in `SETUP.md` (Hugging Face Llama 3.2 approval, GPU, repo).
- Create the repo with the layout from `README.md`; add CI that runs `pytest` on CPU.
- Run dense Llama-3.2-1B-Instruct on 50 GSM8K problems with lm-evaluation-harness; record accuracy and runtime.
- Clone GateSkip, pin the commit, and run its README example.

**Exit criteria:** dense GSM8K number logged in `EXPERIMENTS.md` (E0); GateSkip runs end to end.

### Phase 1 – Pilot study (Nov week 3 – mid Dec) → **R1 evidence**
**Goal:** prove (or disprove) the motivation before building anything big.
- E1: Train or load GateSkip on Llama-3.2-1B; evaluate GSM8K at 0/15/25/35% compute savings.
- E2: Error-token analysis. For problems that are correct dense but wrong under GateSkip, log which tokens skipped which modules. Tag tokens: digits, operators (`=`, `+`, `×`), connectives ("Therefore", "so"), boilerplate.
- E3: Temporal correlation. Compute the correlation of gate values between token t and t−k (k = 1…8) per layer.

**Go/No-go gate (end of Phase 1):**
- **Go** if E2 shows skip-induced errors are over-represented on reasoning tokens, **or** E3 shows strong adjacent-token correlation (e.g. Pearson r > 0.5 in most layers).
- **Pivot** if neither holds: keep the three-tier engine and latency study as the main contribution, and reframe the EMA as an ablation (see RISKS R1).

### Phase 2 – Core implementation (Dec)
- EMA memory module (parallel scan for prefill, recurrent update for decoding).
- Shared-bottleneck dual-head router (layer head g_L, MLP head g_M with ΔAttn input).
- Three-tier inference engine with exact K,V for Tier 3.
- Tests (must pass before any training):
  - T-EQ: router disabled → logits identical to the Hugging Face model (tolerance 1e-5 in fp32).
  - T-KV: cached decoding gives the same logits as uncached recompute under the same tier decisions.
  - T-T3: a Tier-3 token's K,V at layer ℓ equal W_K·norm(h), W_V·norm(h) with RoPE at its position.
  - T-FLOP: the FLOP counter matches a hand calculation for one layer.

**Exit criteria:** all tests green; a 20-step training smoke run decreases loss.

### Phase 3 – Training and calibration (late Dec – Jan)
- Train router + rank-16 LoRA with L = L_CE + λ·L_S (compute-weighted sparsity, see ARCHITECTURE).
- Budget schedule 1.0 → 0.7; sweep λ.
- Calibrate τ_L, τ_M on a held-out split to hit 15/25/35% compute reduction.
- Convert the training data to chat format and train a Dense-FT reference (no router) with the same recipe (D-15).
- Repeat for Qwen2.5-1.5B once Llama results are stable.

**Exit criteria:** E4 (main result on Llama, GSM8K) logged with 3 seeds.

### Phase 4 – Evaluation and ablations (Jan – Feb) → **R2 evidence**
- E5: full benchmark suite vs baselines at matched compute.
- E6–E9: ablations (EMA off; α ∈ {0.5, 0.7, 0.9}; previous-token-gate baseline; no ΔAttn; two tiers vs three).
- E10: Qwen2.5-1.5B replication.

### Phase 5 – Latency (Feb)
- E11: batch-size-1 decode benchmark (ms/token, time per problem, output length, peak memory) on CUDA and MPS.
- Try `torch.compile` and CUDA graphs; report the honest best configuration for every method.

### Phase 6 – Paper and preprint (Feb – Mar) → **R3**
- Write the manuscript following `PAPER_PLAN.md`; post to arXiv; submit to the chosen journal.

### Phase 7 – Final report and demo (Mar – Apr)
- Department-format thesis/report, demo app (side-by-side dense vs LookAhead-GateSkip generation with a live tier heat-map), final presentation.

## Review deliverables

| Review | Month | Show |
|---|---|---|
| R0 (Zeroth) | Nov | Problem, literature, gaps, design, plan (current deck) |
| R1 | Dec | Pilot results E1–E3, go/no-go decision, working EMA + router code with tests |
| R2 | Jan | First trained model, main GSM8K result vs GateSkip, ablation plan |
| R3 | Mar | Full results, latency study, draft paper / preprint |
| Final | Apr | Report, demo, submitted manuscript |

## Weekly rhythm

- Monday: pick tasks from `TASKS.md`, move to *In progress*.
- Every session: append to `DEV_LOG.md`; register runs in `EXPERIMENTS.md` **before** launching them.
- Friday: update `RISKS.md`, send the guide a 5-line status note (done / next / blockers).
