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

## Outline

1. Introduction – SLMs on edge; adaptive depth; reasoning failure; contributions (C1–C5)
2. Related work – early exit, static pruning, token routing (GateSkip, FlexiDepth, SkipGPT, D-LLM), module skipping (FFN-SkipLLM, DiffSkip), routing memory (HeRo), latency-aware skipping (LiteStage, CLaSp)
3. Pilot analysis – C1
4. Method – EMA memory, dual-head router, three tiers with exact KV, training loss, calibration
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
