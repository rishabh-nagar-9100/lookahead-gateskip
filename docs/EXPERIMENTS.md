# Experiment Registry

Rules:
1. Register a run here **before** launching it (ID, goal, config path, commit hash).
2. Never report a number that is not traceable to a row here and a file in `results/`.
3. Test sets are used only for final evaluation; tune on validation/calibration splits.

## Fixed evaluation protocol

| Benchmark | Split | Setting | Metric |
|---|---|---|---|
| GSM8K (trained models) | test (1,319) | `gsm8k_chat_zs`: 0-shot, chat template, system instruction asking for `#### N` (same as training), greedy, max 512 new tokens (D-15) | exact match: strict (`#### N`) headline; robust extractor (`lagate.eval.answer`), flexible and cap-hit rate logged |
| GSM8K (original untrained models only) | test (1,319) | `gsm8k_cot_llama` 8-shot chat CoT, greedy, max 512 (D-08) | exact match (flexible headline) |
| ARC-Challenge | test | 25-shot (harness default) | acc_norm |
| PIQA | validation | 0-shot | acc |
| MBPP | test | harness default, greedy | pass@1 |

- Use lm-evaluation-harness with a pinned version; record it in SETUP.md.
- Seeds: 3 per configuration for trained models (report mean ± std).
- Compute budgets: 0%, 15%, 25%, 35% FLOP reduction, measured with the same FLOP counter for every method.
- Always log: accuracy (GSM8K: strict headline + robust + flexible), realised FLOP reduction, average output length, cap-hit rate, tier distribution.
- Order of trained runs (D-15): E-DFT first. Its format check (P3-08) must pass before any gated model (E1, E4, …) is trained.

## Planned experiments

| ID | Goal | Model | Notes | Status |
|---|---|---|---|---|
| E0 | Dense baseline | Llama-3.2-1B-Instruct (+ Qwen2.5-1.5B-Instruct) | GSM8K first, then all four. Llama gated access pending → 50-problem smoke run done on Qwen first | in progress |
| E1 | GateSkip baseline at 0/15/25/35% | Llama-3.2-1B | Our port (D-16), trained once with their recipe in our pipeline. The two rows below are engine settings on the **same weights** | todo |
| E1-exactKV | GateSkip, main baseline | Llama-3.2-1B | Our engine: real skipping, exact per-layer K,V for skipped tokens, calibrated bs=1 thresholds. Differences from E4 come only from routing. Matches what the official code actually computes | todo |
| E1-copyKV | GateSkip, cost of copying K,V (gap G4) | Llama-3.2-1B | Same weights and thresholds as E1-exactKV, `kv_mode: copy_prev` (layer ℓ−1's entry copied into layer ℓ for skipped tokens, GateSkip's intended design). Report Δ accuracy vs E1-exactKV at each budget | todo |
| E1-official | D-12(c) reproduction check | Llama-3.2-1B (base) | Official code at the pinned commit, native format, 0% and 15%. Our port must match within seed noise (P4-06) | todo |
| E2 | Error-token analysis | Llama-3.2-1B + GateSkip | tag tokens; compare skip rates on failed vs passed problems | todo |
| E3 | Adjacent-token gate correlation | Llama-3.2-1B + GateSkip | Pearson r for lags 1–8, per layer | todo |
| E-DFT | Dense-FT reference (0% compute) | Llama-3.2-1B-Instruct | Same chat data, recipe, steps and seeds as E1/E4, no router or gates (D-15). GSM8K main protocol, 3 seeds. Seed-1 model gates the format check P3-08 | todo |
| E4 | Main result | Llama-3.2-1B, ours | GSM8K, 3 seeds, 3 budgets | todo |
| E5 | Full suite vs baselines | Llama-3.2-1B | GateSkip, FlexiDepth, D-LLM, FFN-skip, random | todo |
| E6 | Ablation: EMA off | Llama-3.2-1B | router input = h only | todo |
| E7 | Ablation: α | Llama-3.2-1B | 0.5 / 0.7 / 0.9 | todo |
| E8 | Ablation: previous-token gate | Llama-3.2-1B | replace μ_t with g_(t−1) as extra input | todo |
| E9 | Ablation: no ΔAttn; two tiers | Llama-3.2-1B | isolates the MLP head and Tier 2 | todo |
| E10 | Replication | Qwen2.5-1.5B | E4 + E5 subset | todo |
| E-DFT-Q | Dense-FT reference (0% compute) | Qwen2.5-1.5B-Instruct | As E-DFT, for E10 | todo |
| E11 | Latency | both | bs=1; CUDA + MPS; ms/token, time/problem, length, memory | todo |

## Run log

| Run ID | Exp | Date | Commit | Config | Seed | Budget | GSM8K | ARC-C | PIQA | MBPP | FLOP red. | Avg len | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20261106-053918_gsm8k_dense_qwen_s0 | E0 | 2026-11-06 | e9e5451 | configs/eval/gsm8k_dense_qwen.yaml | 0 | 0% | 62.0% strict / 62.0% flexible (31/50, ±6.9); robust 64.0% (post hoc) | | | | 0% | 143 tok (max 312) | Qwen2.5-1.5B-Instruct, smoke --limit 50, MPS fp16, gsm8k_cot_llama (D-08). Generation 10:00 (≈12 s/problem); total 1042 s incl. first-time model+dataset download. meta.json says dirty=True only because of the untracked run dir (bug fixed after this run); tracked tree was clean. Llama access pending. |
| 20261106-060855_gsm8k_zs_dense_qwen_s0 | E0 | 2026-11-06 | ab21c85 | configs/eval/gsm8k_zs_dense_qwen.yaml | 0 | 0% | 0.0% strict / 16.0% flexible (8/50, ±5.2); robust 48.0% (post hoc) | | | | 0% | 352 tok (max 512; 40% hit cap) | D-11 protocol, untrained Qwen2.5-1.5B-Instruct, MPS fp16, --limit 50. Degenerate: 0/50 outputs contain `#### N`; the model answers ("The answer is N.") and then runs on into unrelated text because no chat template means no end-of-turn token. Flexible-extract takes the last number in that run-on text, so it is not a meaningful score. Generation 14:06 (≈17 s/problem), 866 s total; clean tree. See D-14. |
| 20261106-063044_gsm8k_chat_dense_qwen_s0 | E0 | 2026-11-06 | 1e3d413 | configs/eval/gsm8k_chat_dense_qwen.yaml | 0 | 0% | 26.0% strict / 56.0% flexible (±6.3 / ±7.1); robust 64.0% (post hoc) | | | | 0% | 219 tok (max 512; 2% hit cap) | Protocol-B check, untrained Qwen2.5-1.5B-Instruct, MPS fp16, --limit 50. **FAIL**: only 20/50 (40%) contain `#### N` (needed ≥90%); strict−flexible gap 30 pts (needed ≤4); cap hit 2% (needed ≤2%, passes). The model ignores the system instruction and uses its own habits (`\boxed{N}`, **bold**, "Therefore … $18"). Generation 7:04 (≈8.5 s/problem). Single pre-set instruction, not tuned. |

## Results summary (update after each experiment)

**E0 smoke (2026-11-06), 50 GSM8K test problems, not a reportable number:** Qwen2.5-1.5B-Instruct dense, 8-shot CoT (chat), greedy, MPS fp16 → 62.0% (strict = flexible), avg output 143 tokens, ≈12 s/problem. Llama-3.2-1B pending gated access. Full-set E0 to run on GPU.

**E0 smoke under D-11 (2026-11-06), same 50 problems:** untrained Qwen2.5-1.5B-Instruct 0-shot without chat template → 0.0% strict / 16.0% flexible, 40% of outputs hit the 512-token cap. The protocol only makes sense for models fine-tuned on the `#### N` format; dense-untrained numbers come from D-08 (D-14).

**Protocol-B check (2026-11-06), same 50 problems:** chat format, 0-shot, system instruction asking for `#### N` → 26.0% strict / 56.0% flexible, 40% of outputs use `#### N`, 2% hit the cap, avg 219 tokens. Fails the agreed criteria (≥90% `####`, ≤4-pt gap) for an untrained model. Robust extractor on the same samples: D-08 64%, D-11 48%, chat 64% (so the failure is format, not reasoning). Decision D-15: chat format for training and evaluating trained models; untrained models reported under D-08; Dense-FT reference (E-DFT); format check on the first Dense-FT model before gated training.
