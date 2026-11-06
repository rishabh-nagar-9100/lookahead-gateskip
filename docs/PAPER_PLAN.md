# Paper Plan

Working title: *LookAhead-GateSkip: Cross-Token Routing Memory for Dynamic Layer Skipping in Small Language Models*

## Claims (each must map to an experiment)

| Claim | Evidence |
|---|---|
| C1 Skip-induced reasoning errors concentrate on specific token types, and gate decisions are temporally correlated | E2, E3 |
| C2 Cross-token memory improves accuracy over GateSkip at matched compute | E4, E5, E6, E8 |
| C3 The ΔAttn-conditioned MLP gate and Tier 2 add further gains | E9 |
| C4 Gains hold on a second model family | E10 |
| C5 Savings translate into batch-size-1 latency (or an honest analysis of why not) | E11 |
| C6 Batch-size-1 deployable thresholds: GateSkip's batch-quantile thresholds skip nothing during bs=1 decoding (code evidence, D-18). Fixed empirical-CDF thresholds calibrated once make token-level skipping work at bs=1 for both methods, and compute is measured with real skipping (G6) | D-18 test, E1-exactKV, E4, E11 |

## Research gaps (from the R0 deck; only G4 and G6 are recorded here so far, see D-19)

- **G4 K,V of skipped tokens.** Token-level layer-skipping methods differ here. FlexiDepth still computes K,V for skipped tokens, keeping the cache complete. D-LLM evicts them, so later tokens lose those entries. Our Tier 3 computes exact per-layer K,V at 2–4% of a layer, and E5 compares against both. Exact K,V is a correctness property of our engine, **not** a claim against GateSkip: GateSkip's intended copy from ℓ−1 never executes, so its cache is exact too (LITERATURE.md). Verify the FlexiDepth/D-LLM wording against their papers before writing.
- **G6 Batch-size-1 deployment.** Reported savings of batch-relative routers do not carry over to bs=1. GateSkip's per-pass quantile threshold is 0.0 for a one-token decode step, so nothing is skipped (`threshold_finder.py:29–36`). We use fixed empirical-CDF thresholds from a calibration split for every method (D-18) and measure compute and latency at bs=1 (contribution C6, E11).
- G1–G3, G5: copy the wording from the R0 deck.

## Outline

1. Introduction – SLMs on edge; adaptive depth; reasoning failure; contributions (C1–C6)
2. Related work – early exit, static pruning, token routing (GateSkip, FlexiDepth, SkipGPT, D-LLM), module skipping (FFN-SkipLLM, DiffSkip), routing memory (HeRo), latency-aware skipping (LiteStage, CLaSp)
3. Pilot analysis – C1
4. Method – EMA memory, dual-head router, three tiers with exact KV (a correctness property), training loss, bs=1 calibration (C6)
5. Experimental setup – models, data, budgets, baselines, FLOP accounting, hardware
6. Results – main table, Pareto curves, ablations, Qwen replication
7. Latency study – bs=1 CUDA and MPS, output length
8. Limitations – model scale, batch size 1 only, benchmarks
9. Conclusion

## Figures and tables

- Fig 1 Overview diagram (from the deck's architecture slide)
- Fig 2 Pilot: skip rate by token type on failed vs passed problems
- Fig 3 Adjacent-token gate correlation vs lag, per layer
- Fig 4 Accuracy vs FLOP reduction (Pareto) on GSM8K for all methods
- Fig 5 Tier distribution heat-map on an example CoT answer
- Table 1 Main results (4 benchmarks × 3 budgets × methods)
- Table 2 Ablations
- Table 3 Latency (ms/token, time per problem, output length, memory)

## Target venues (confirm with guide; verify indexing on Clarivate Master Journal List)

- Journals: Neurocomputing; Knowledge-Based Systems; Engineering Applications of Artificial Intelligence; Expert Systems with Applications; IEEE Access.
- Conference alternative: ACL Rolling Review (ACL / EMNLP Findings).
- Post the arXiv preprint before or at submission.
