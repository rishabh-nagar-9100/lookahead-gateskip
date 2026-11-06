# Experiment Registry

Rules:
1. Register a run here **before** launching it (ID, goal, config path, commit hash).
2. Never report a number that is not traceable to a row here and a file in `results/`.
3. Test sets are used only for final evaluation; tune on validation/calibration splits.

## Fixed evaluation protocol

| Benchmark | Split | Setting | Metric |
|---|---|---|---|
| GSM8K | test (1,319) | 8-shot CoT, greedy, max 512 new tokens | exact match |
| ARC-Challenge | test | 25-shot (harness default) | acc_norm |
| PIQA | validation | 0-shot | acc |
| MBPP | test | harness default, greedy | pass@1 |

- Use lm-evaluation-harness with a pinned version; record it in SETUP.md.
- Seeds: 3 per configuration for trained models (report mean ± std).
- Compute budgets: 0%, 15%, 25%, 35% FLOP reduction, measured with the same FLOP counter for every method.
- Always log: accuracy, realised FLOP reduction, average output length, tier distribution.

## Planned experiments

| ID | Goal | Model | Notes | Status |
|---|---|---|---|---|
| E0 | Dense baseline | Llama-3.2-1B-Instruct | GSM8K first, then all four | todo |
| E1 | GateSkip baseline at 0/15/25/35% | Llama-3.2-1B | official code, pinned commit | todo |
| E2 | Error-token analysis | Llama-3.2-1B + GateSkip | tag tokens; compare skip rates on failed vs passed problems | todo |
| E3 | Adjacent-token gate correlation | Llama-3.2-1B + GateSkip | Pearson r for lags 1–8, per layer | todo |
| E4 | Main result | Llama-3.2-1B, ours | GSM8K, 3 seeds, 3 budgets | todo |
| E5 | Full suite vs baselines | Llama-3.2-1B | GateSkip, FlexiDepth, D-LLM, FFN-skip, random | todo |
| E6 | Ablation: EMA off | Llama-3.2-1B | router input = h only | todo |
| E7 | Ablation: α | Llama-3.2-1B | 0.5 / 0.7 / 0.9 | todo |
| E8 | Ablation: previous-token gate | Llama-3.2-1B | replace μ_t with g_(t−1) as extra input | todo |
| E9 | Ablation: no ΔAttn; two tiers | Llama-3.2-1B | isolates the MLP head and Tier 2 | todo |
| E10 | Replication | Qwen2.5-1.5B | E4 + E5 subset | todo |
| E11 | Latency | both | bs=1; CUDA + MPS; ms/token, time/problem, length, memory | todo |

## Run log

| Run ID | Exp | Date | Commit | Config | Seed | Budget | GSM8K | ARC-C | PIQA | MBPP | FLOP red. | Avg len | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | | | | | |

## Results summary (update after each experiment)

_(paste the latest summary tables here; keep raw outputs in results/)_
