# Risk Register

Review weekly. Likelihood / impact: L(ow), M(edium), H(igh).

| ID | Risk | L | I | Mitigation / fallback | Trigger to act |
|---|---|---|---|---|---|
| R1 | Pilot shows no adjacent-token correlation and no error clustering | M | H | Pivot: main contribution becomes attention-conditioned three-tier execution + honest latency study; EMA becomes an ablation | E2 and E3 both negative |
| R2 | EMA gives no gain over GateSkip at matched compute | M | H | Try learnable α, gate-EMA instead of hidden-state EMA, previous-token gate; report a careful negative result in ablations | E4 within noise of E1 |
| R3 | A new paper publishes the same idea | M | H | Monthly novelty watch (LITERATURE.md); post arXiv preprint early (Phase 6 start) | Overlapping arXiv paper found |
| R4 | No wall-clock speedup at bs=1 (kernel launch / Python overhead dominates) | H | M | Custom decode loop, torch.compile, CUDA graphs, skip whole modules (no masking); report FLOPs and latency honestly. **Measured on MPS (run 20261106-180357):** dense engine 1.05× HF after P5-05, but the decision path costs ≈ 21 ms/token (2 host syncs × 28 layers) and our router + EMA ≈ 4 ms/token, so with no skipping we are 1.68× HF. At ≈ 39 ms/token dense, that overhead is only recovered at large skip rates. Mitigations to evaluate (P5-06): (a) measure on CUDA, where a sync costs ~10–20 µs; (b) one sync per layer instead of two (read g_L and g_M decisions together where possible); (c) CUDA graphs per tier pattern; (d) a variant that decides token t's tiers from the previous token's state (EMA/μ_(t−1)), so all of a step's decisions are known up front: 1 or 0 syncs per token. This is a method change and needs a DECISIONS entry | E11 speedup < 1.05× |
| R5 | Not enough GPU time | M | H | Start with Llama-3.2-1B only; short runs; cloud credits; reduce seeds to 2 for ablations | Training queue > 1 week |
| R6 | Baseline code does not run or does not support our models | M | M | Re-implement minimal versions; document differences | P4-01 blocked > 3 days |
| R7 | Train/inference mismatch from soft gates | M | M | Straight-through hard-gate phase; calibrate on soft vs hard outputs | Validation accuracy drop > 2 points after binarising |
| R8 | Skipping lengthens outputs, cancelling latency gains | M | M | Report output length; add length-aware calibration if needed | Avg length +10% vs dense |
| R9 | Predatory "fast-track" journal | L | H | Check the journal on Clarivate Master Journal List; discuss with guide | Before submission |
