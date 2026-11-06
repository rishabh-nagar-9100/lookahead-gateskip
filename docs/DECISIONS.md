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

### D-08 · 2026-11-06 · GSM8K harness task: `gsm8k_cot_llama` with chat template, flexible-extract as headline
- **Context:** EXPERIMENTS.md fixes "8-shot CoT, greedy, max 512 new tokens, exact match" but not the harness task, prompt format or answer filter.
- **Decision:** lm-eval task `gsm8k_cot_llama` (fixed 8 exemplars, "The final answer is N" format) with `apply_chat_template=true`, `fewshot_as_multiturn=true`, `do_sample=false`, `max_gen_toks=512` (the task default has no limit, so HF would cap at 256). Same task for Llama and Qwen. Both `strict-match` and `flexible-extract` are logged; `flexible-extract` is the headline number.
- **Alternatives:** `gsm8k_cot` (plain-text "Q:/A:" 8-shot, no chat template); `gsm8k` 5-shot.
- **Reason:** both models are instruction-tuned and the task mirrors the 8-shot CoT format Meta uses for Llama 3 GSM8K evaluation; chat-formatted prompts avoid penalising instruct models on format.
- **Status:** provisional — check against GateSkip's evaluation setup in P0-07 and switch (new entry) if they differ, so E0 and E1 are comparable.

### D-09 · 2026-11-06 · `uv` + `.venv` instead of conda
- **Alternatives:** conda (as written in the original SETUP.md).
- **Reason:** conda is not installed on the dev Mac; `uv` is, and it provides Python 3.11 and fast, lockable installs. Exact versions are frozen in `requirements.lock.txt`.
- **Status:** active.

### D-10 · 2026-11-06 · CI installs only CPU test dependencies
- **Alternatives:** install the full `requirements.txt` in CI.
- **Reason:** the full set pulls CUDA torch and lm-eval (slow, large) although the CPU tests import only torch, transformers, pyyaml and python-dotenv. CI uses the CPU-only torch wheel and `pip install -e . --no-deps`. If a test later needs another package, add it to the workflow.
- **Status:** active.
