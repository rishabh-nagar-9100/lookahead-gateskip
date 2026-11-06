# Literature Notes

Priority: ★★★ must read fully and compare against · ★★ read method + results · ★ skim.

| # | Paper | Link | ★ | What we take from it |
|---|---|---|---|---|
| 1 | GateSkip (Laitenberger et al., ICLR 2026) – base | https://arxiv.org/abs/2510.13876 · code https://github.com/Thiggel/GateSkip | ★★★ | Residual gates, sparsity loss, evaluation setup, main baseline |
| 2 | HeRo (Lin, Wan, Wang, 2026) – closest work | https://arxiv.org/abs/2609.08189 | ★★★ | Depth-wise router memory; our differentiation (cross-token memory) |
| 3 | DiffSkip (Luo, Wang, Yan, Findings ACL 2025) | ACL Anthology 2025.findings-acl.377 | ★★★ | ΔAttn as MLP-skip signal; bottleneck router design |
| 4 | FlexiDepth (Luo, Wang, Yan, COLM 2025) | https://arxiv.org/abs/2503.23798 | ★★★ | Full KV cache for skipped tokens; baseline; no wall-clock speedup |
| 5 | SkipGPT (Zhao et al., 2025) | https://arxiv.org/abs/2506.04179 · code https://github.com/EIT-NLP/SkipGPT | ★★ | Decoupled attention/MLP routers; global budget |
| 6 | D-LLM (Jiang et al., NeurIPS 2024) | NeurIPS 2024 proceedings · code https://github.com/Jyk-122/D-LLM | ★★ | KV eviction for skipped layers; baseline |
| 7 | CLaSp (Chen et al., ACL 2025) | https://arxiv.org/abs/2505.24196 | ★★ | Adjacent tokens skip similar layers (motivation for E3) |
| 8 | LiteStage (Kang, Song, Kim, 2025) | https://arxiv.org/abs/2510.14211 | ★★ | Skipping lengthens outputs → measure output length |
| 9 | Dr.LLM (Heakl et al., ICLR 2026) | ICLR 2026 | ★★ | Skip/execute/repeat routers; sequence scope |
| 10 | PoLar (Li, Li, Zhou, ICML 2026) | https://arxiv.org/abs/2606.06574 | ★★ | Layer programs for math reasoning |
| 11 | FFN-SkipLLM (Jaiswal et al., EMNLP 2024) | https://arxiv.org/abs/2404.03865 | ★★ | FFN skipping keeps KV intact |
| 12 | AdaSkip (He et al., AAAI 2025) | https://arxiv.org/abs/2501.02336 | ★ | Sub-layer importance; long context |
| 13 | FiRST (Jain et al., Findings EMNLP 2025) | Findings EMNLP 2025 | ★ | Prompt-level routing |
| 14 | Mixture-of-Depths (Raposo et al., 2024) | https://arxiv.org/abs/2404.02258 | ★ | Top-k token routing |
| 15 | Dynamic layer selection (Glavas et al., 2024) | https://arxiv.org/abs/2410.20022 | ★★ | Hidden-state controllers are weak routing signals |

## Reading notes

Add one block per paper as you read it:

```
### <Paper>
- Problem:
- Method (2–3 lines):
- Key numbers:
- Code available? commit used:
- How it differs from ours:
- Should it be a baseline? Why / why not:
```

### GateSkip (Laitenberger et al., ICLR 2026): code reading, 2026-11-06
Source: https://github.com/Thiggel/GateSkip at commit `8d565130266fecc4a2a5374d0ced135f05e4b9f9` (2025-11-27), cloned to `baselines/GateSkip/`, which git ignores. Everything below comes from the **code and job scripts, not the paper text**. Check the paper before quoting any of it as "what the paper did".

- **Problem:** token-level dynamic skipping of attention/MLP modules in decoder LLMs without destabilising training.
- **Method:** a sigmoid gate on each module's residual branch. The gate scales the module output, and an L2 sparsity loss pushes gates towards 0. Tokens whose mean gate is below a percentile threshold get their module output zeroed. During training the skip budget ramps from 0% to 20%.
- **Released models / checkpoints:** **none.** There are no models or datasets under the Thiggel Hub account, a Hub search for "gateskip" finds nothing, and the repo has no download links. Checkpoints are written locally (`$CHECKPOINT_ROOT/<experiment>_<seed>.pt`) and job files refer to unreleased `.ckpt` names. **There is no trained Llama-3.2-1B GateSkip checkpoint to download: E1 needs training.**
- **Models in the job files:** main experiments use **`meta-llama/Llama-3.2-1B` (base, not Instruct)**. Scaling runs use Llama-3.2-3B, Llama-3.2-3B-Instruct, Llama-3.1-8B and Gemma-2-2B. There is no 1B-Instruct run.
- **Training setup** (`jobs/job_common.sh`, variant `GateSkip-separate-vector`, the per-module vector gate):
  - **Data:** `csqa_gsm8k_gen` is the CommonsenseQA train split with Nemotron-70B reasoning traces (`multi-domain-reasoning/commonsense_qa`, ≈9.7k) plus the GSM8K train split (`openai/gsm8k`, 7,473), shuffled. Prompt format is `"Question: {q}\nAnswer:"`, the target is `" {steps}\n#### {answer}"` + EOS, and samples longer than the max length are dropped.
  - **Optimisation:** full fine-tuning of backbone + gates (`finetune_mode=full`, not LoRA). lr 3e-6, warm-up 1,000 steps, LR decay over 6,800 steps, batch size 1 with no gradient accumulation, sequence length 1,024, 1 epoch, bf16.
  - **Gates and sparsity:** sparsity loss weight 1.0 (L2). The skip percentile ramps 0→0.2 over 6,800 steps. Gate init: weights N(0, 0.01), bias +5 (the same +5 init as our ARCHITECTURE §3).
  - **Hardware and time:** wall-clock cap `--max-hours 2`. Slurm `a100` partition, 1× A100 80 GB (`-C a100_80`), 23 h job limit covering train + eval. Seeds 1, 2, 3.
  - **Steps actually completed:** ≈17k samples at batch size 1 is ≈17k optimiser steps for one epoch, but the 2 h cap may stop training earlier. I couldn't verify which from the code.
  - **Gate parameter count:** each vector gate is `Linear(2048, 2048)`, i.e. 4.2M params × 32 modules ≈ 134M (≈ 11% of the 1.24B backbone).
- **GSM8K evaluation** (`experiment/model_evaluator/ModelEvaluator.py`, `EvaluationRunner.py`):
  - **Task and prompt:** stock lm-eval `gsm8k` task (requires lm-eval ≤ 0.4.8). **0-shot** (`--num-fewshot 0`), prompt `"Question: …\nAnswer:"`, the same format as their training data. No chat template; `add_bos_token=True`.
  - **Decoding:** greedy. `max_new_tokens=1024` (also `max_length=1024`), stopping at `"Question:"`, `</s>` or `<|im_end|>`.
  - **Metric:** `exact_match`, with strict-match on `#### N` and flexible-extract on the last number.
  - **⚠ Only 100 problems:** `simple_evaluate(..., limit=100)` is hard-coded in both the HF and vLLM evaluators. The job's `--limit 0` and the runner's `limit` argument never reach lm-eval. This means GSM8K accuracy per budget and seed is measured on the first 100 test problems, not all 1,319.
  - **Batching and budgets:** eval batch size 128. Budgets are swept over {0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60}% in a single run.
- **Where gates are computed and applied** (`experiment/models/gating/`):
  - `ModelGating.py` wraps each decoder layer's `self_attn` and `mlp` in a `GatedWrapper` with its own `GateLayer`.
  - `GateLayer.forward`: `g = sigmoid(Linear(h))`, where `h` is the module's **input** hidden state, detached. The gate is a d-dim vector per token (or a scalar with `--single-number-gates`). Token importance is `g.mean(-1)`.
  - `GatedWrapper.forward`:
    1. compute `g`;
    2. skip mask = importance ≤ threshold;
    3. **run the module in full anyway**;
    4. output = `g * module_out` (gate scaling kept at inference);
    5. skipped tokens' output is replaced with zeros.

    Compute is only saved in principle (or via the optional Triton kernel `vllm_kernel.py`, used for FLOP/time profiling). Plain HF evaluation saves no time.
  - **Threshold** (`threshold_finder.py`): at inference it is the `desired_skip_ratio` quantile of importances over all valid tokens **in the current forward pass**, so it is relative to the batch. When the pass holds a single token, it returns 0.0 and **nothing is skipped**. So at batch size 1, decode steps never skip: only the prompt (prefill) or batched decoding does.
  - **KV cache for skipped tokens** (`update_kv_cache`): skipped tokens' K,V are overwritten with **layer ℓ−1's** cache entries. This is the cross-layer copy our D-03 rejects. The code looks up `kwargs["past_key_value"]` and uses tuple indexing inside a bare `try/except: pass`. With current transformers Cache objects (the kwarg is now `past_key_values`), this most likely does nothing silently, so the cache stays exact because the module always runs. Not verified by running.
- **Software:**
  - Python 3.10 inside an Apptainer image (`pytorch.def`), although `environment.yml` says 3.11. Their stack: torch 2.8, CUDA 12, lm-eval ≤ 0.4.8, datasets < 3.0, PyTorch Lightning, DeepSpeed (optional; ZeRO-3 + CPU offload when enabled), flash-attn 2.8.3 (Linux x86_64 cp310 wheel only), vLLM, W&B.
  - Not installable on macOS/MPS as pinned.
  - It conflicts with our env: we have lm-eval 0.4.13, datasets 5.1 and transformers 5.22, so it needs a separate env.
  - The README's job paths (`jobs/cot/llama1b_*.job`) are out of date. The real files are `jobs/<Variant>/{cot,loglikelihood}.job`.
- **How it differs from ours:**
  - Their gate sees only the current token's input. Ours sees the cross-token EMA μ_t, and the MLP head also sees ΔAttn.
  - Ours makes one layer decision plus one MLP decision with three tiers, instead of two independent module gates.
  - Our Tier 3 computes exact per-layer K,V instead of copying from ℓ−1.
  - We use calibrated fixed thresholds that work at batch size 1, instead of batch quantiles.
  - We train LoRA + router; they do full fine-tuning.
  - We target Instruct models; their main 1B results use the base model.
- **Baseline?** Yes, the main one (D-01). Reproducing it needs a GPU (see D-06, D-12). To be comparable at batch size 1, it must be evaluated with fixed calibrated thresholds as well as its native batch-quantile mode.

## Novelty watch

Re-run these searches monthly and before submission (arXiv, Google Scholar, Semantic Scholar):
"layer skipping router memory", "token-wise layer skipping reasoning", "temporal routing layer skip", "history-aware routing LLM", "attention-only MLP skip router".
Log anything new that overlaps here with date found.
