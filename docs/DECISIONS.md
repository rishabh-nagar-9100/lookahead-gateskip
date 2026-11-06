# Decision Log

Format: ID · date · decision · alternatives considered · reason · status (active / superseded by D-xx).

### D-01 · 2026-11 · Build on GateSkip as the base method
- **Alternatives:** FlexiDepth, SkipGPT, D-LLM.
- **Reason:** official code is available, it is evaluated on Llama-3.2-1B, and it reports results on generative reasoning, which is our target.
- **Status:** active.

### D-02 · 2026-11 · Memory across tokens, not across depth
- **Alternatives:** depth-wise memory like HeRo (arXiv:2609.08189).
- **Reason:** HeRo already covers depth-wise memory; cross-token memory is the open axis, and CLaSp reports that adjacent tokens skip similar layers.
- **Status:** active, conditional on the Phase 1 go/no-go result.

### D-03 · 2026-11 · Tier 3 computes its own K,V (no cross-layer copying)
- **Alternatives:** copy K,V from layer ℓ−1 (wrong projection space); evict as in D-LLM (drops tokens from attention).
- **Reason:** keeps an exact cache for about 3–4% of layer cost.
- **Status:** active.

### D-04 · 2026-11 · Two gates (g_L, g_M) instead of two thresholds on one score
- **Alternatives:** dual quantile thresholds τ1, τ2 on a single mean gate (original deck design).
- **Reason:** a separate MLP decision can use the attention output (ΔAttn), which DiffSkip shows is a strong signal; tiers become a principled 2-gate policy.
- **Status:** active.

### D-05 · 2026-11 · Primary target metric: accuracy at matched compute vs GateSkip on GSM8K
- **Alternatives:** fixed "<1.5% drop at 25–35%" target.
- **Reason:** GateSkip already performs well on instruction-tuned models; a relative Pareto improvement is the defensible claim.
- **Status:** active.

### D-06 · open · Training data for router + LoRA
- **Options:** reuse GateSkip's training setup (preferred for a fair comparison); add a math-heavy SFT subset.
- **Decide in:** P3-01.

### D-07 · open · Gate scaling vs binary gates at inference
- **Decide in:** Phase 3 after comparing both on the validation split.
