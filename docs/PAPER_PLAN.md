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

## Research gaps

Referenced from DECISIONS.md (D-19, D-20). Each gap maps to the claims above.

- **G1 No cross-token routing memory.** Routers decide from the current hidden state only (GateSkip, FlexiDepth). HeRo adds memory across depth only. Nothing carries routing evidence across consecutive tokens, although CLaSp reports that adjacent tokens skip similar layers. → C1 (E3), C2
- **G2 Reasoning collapse under skipping.** Adaptive-compute methods often fail on generative reasoning, and which token types cause skip-induced errors has not been measured for SLMs. → C1 (E2)
- **G3 No unified tiered policy.** Attention-preserving FFN skipping exists (FFN-SkipLLM, DiffSkip), but no router combines the attention update with temporal context to choose full layer / attention only / layer skip. → C3 (E9)
- **G4 K,V of skipped tokens.** Token-level layer-skipping methods differ here. FlexiDepth keeps the full KV cache for skipped tokens; D-LLM evicts K,V for skipped layers, so later tokens lose those entries (both verified against the papers, LITERATURE.md). Our Tier 3 computes exact per-layer K,V at 2–4% of a layer, and E5 compares against both. Exact K,V is a correctness property of our engine, **not** a claim against GateSkip: GateSkip's intended copy from ℓ−1 never executes, so its cache is exact too (LITERATURE.md). → E5
- **G5 Router overhead in SLMs.** Gates are a larger share of each layer in 1–1.5B models. GateSkip's vector gates (Linear(H, H) per module, run for every token) cost 2H² MACs per layer: 13.7% of a dense Llama-3.2-1B layer at n = 128 (10.0% for Qwen2.5-1.5B), vs 0.6% for our router (verified in code; FLOP report `20261106-162120_flops_llama_s0`). The GateSkip paper reports parameter overhead only and does not say gate FLOPs are subtracted from its token-skip savings; its Table 13 also lists 4% params where the real Llama-1B dims give 10.8% (LITERATURE.md, "Gate cost"). → matched-compute accounting in C2, FLOP report
- **G6 Batch-size-1 deployment.** Reported savings of batch-relative routers do not carry over to bs=1. GateSkip's per-pass quantile threshold is 0.0 for a one-token decode step, so nothing is skipped (`threshold_finder.py:29–36`). We use fixed empirical-CDF thresholds from a calibration split for every method (D-18), and measure compute and latency at bs=1. → C5, C6 (E11)

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
