# Technical Specification

Keep this file in sync with the code. Any change here needs an entry in `DECISIONS.md`.

## 1. Notation

| Symbol | Meaning |
|---|---|
| L | number of decoder layers (Llama-3.2-1B: 16; Qwen2.5-1.5B: 28) |
| H | hidden size |
| h_t^ℓ | residual-stream input to layer ℓ for token t |
| μ_t^ℓ | cross-token EMA state at layer ℓ |
| ΔAttn_t^ℓ | attention sub-layer output added to the residual for token t |
| g_L, g_M | layer gate and MLP gate, each in (0, 1) |
| τ_L, τ_M | calibrated thresholds for g_L and g_M |

Verify H, MLP width and KV heads from each model's `config.json` (task P2-01) and fill in the table:

| Model | Layers | H | MLP width | Q heads | KV heads |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct | 16 | 2048 | 8192 | 32 | 8 |
| Qwen2.5-1.5B-Instruct | 28 | 1536 | 8960 | 12 | 2 |

## 2. Cross-token EMA memory

For every layer ℓ independently:

```
μ_t^ℓ = α · RMSNorm(h_t^ℓ) + (1 − α) · μ_(t−1)^ℓ,     μ_0^ℓ = 0
```

- Prefill: computed for all prompt positions with a causal scan (cumulative weighted sum).
- Decoding: one recurrent update per token; state stored per layer alongside the KV cache.
- α is a hyperparameter (default 0.7; ablate 0.5 / 0.9). Optional later: learnable α per layer.
- Cost: about 3H element-wise ops per token per layer (negligible).
- The EMA uses the normalised hidden state so its scale is comparable across layers.

## 3. Dual-head router (per layer)

```
x   = [ RMSNorm(h_t) ∥ μ_t ]                 # 2H
z   = SiLU(W_1 x)                            # W_1: 2H → 64 (shared code)
g_L = σ(w_L · z + b_L)                       # layer gate, scalar
a   = W_a · RMSNorm(ΔAttn_t)                 # W_a: H → 64, only if attention ran
g_M = σ(w_M · [z ∥ a] + b_M)                 # MLP gate, scalar
```

- Init b_L = b_M = +5 (gates start open, about 0.993) so training starts from the dense model.
- Router cost per token and layer ≈ 2H·64 + H·64 multiply-accumulates. Llama-3.2-1B: about 0.39M vs about 60M for the layer's linear maps (≈ 0.65%). Report the measured value from the FLOP counter.

## 4. Training forward pass (soft gates)

```
h'  = h  + g_L · Attn(norm(h))
h'' = h' + g_L · g_M · MLP(norm(h'))
```

Compute-weighted sparsity loss (expected fraction of layer FLOPs used):

```
L_S = mean over tokens and layers of [ c_attn · g_L + c_mlp · g_L · g_M ] / (c_attn + c_mlp)
L   = L_CE + λ · L_S
```

c_attn and c_mlp are per-layer FLOP costs from the FLOP counter (attention includes a context-length term; use the average training context length).

Open question (decide in Phase 3, log in DECISIONS): keep gate scaling at inference for executed modules (GateSkip-style) or binarise. Add a short straight-through hard-gate phase at the end of training if the train/inference mismatch hurts accuracy.

## 5. Inference: three tiers

| Tier | Condition | Operations at layer ℓ | Residual |
|---|---|---|---|
| 1 | g_L > τ_L, g_M > τ_M | Q,K,V, attention, O-proj, MLP | h + Attn + MLP |
| 2 | g_L > τ_L, g_M ≤ τ_M | Q,K,V, attention, O-proj | h + Attn |
| 3 | g_L ≤ τ_L | K,V projections only (with RoPE), written to cache | h unchanged |

- g_M is only computed when attention ran (tiers 1–2).
- **Tier 3 never copies K,V from another layer.** Layer-ℓ keys and values are computed from the token's own unchanged hidden state, so every later query attends to a correctly-projected cache entry.
- Batch size 1 is the target setting; the engine may assume one sequence at a time.

## 6. Threshold calibration

1. Run the trained model with soft gates on a held-out calibration split (never the test set).
2. Collect g_L and g_M per token and layer.
3. For a target compute reduction r, search (τ_L, τ_M) on a grid so the expected FLOP use equals (1 − r) of dense, preferring the pair with the lowest calibration loss.
4. Store thresholds in `configs/thresholds/<model>_<budget>.yaml`.

## 7. FLOP accounting (per token, per layer)

Count multiply-accumulates (MACs); report FLOPs = 2 × MACs.

- Q,O projections: 2·H·H
- K,V projections: 2·H·(kv_heads · head_dim)
- Attention scores + weighted sum: 2·n·H (n = current context length)
- MLP (SwiGLU): 3·H·MLP_width
- Router: see §3

Tier 1 = all; Tier 2 = all except MLP; Tier 3 = K,V projections + router. The MLP share depends on n; report it at the actual average context length of each benchmark.

## 8. Correctness tests (must stay green)

| Test | Check |
|---|---|
| T-EQ | Router disabled → logits equal the Hugging Face model (fp32, atol 1e-5) |
| T-KV | Cached decode = uncached recompute under fixed tier decisions |
| T-T3 | Tier-3 cache entries equal direct K,V projection with RoPE at the right position |
| T-FLOP | FLOP counter equals a hand calculation for one layer and one token |
| T-EMA | Scan output equals step-by-step recurrence |
