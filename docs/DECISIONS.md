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

### D-06 · 2026-11-06 · Training data for router + LoRA: GateSkip's `csqa_gsm8k_gen` mix
- **Options:** reuse GateSkip's training setup (preferred for a fair comparison); add a math-heavy SFT subset.
- **Decision:** train both GateSkip (E1) and ours on GateSkip's mix: the CommonsenseQA train split with Nemotron-70B reasoning traces (`multi-domain-reasoning/commonsense_qa`) plus the GSM8K train split (`openai/gsm8k`). Format is `"Question: {q}\nAnswer:"` → `" {steps}\n#### {answer}"` + EOS, max length 1,024, overlong samples dropped, 1 epoch. Pin both dataset revisions in the config when P3-01 is implemented.
- **Reason:** identical data removes data as a confound in the GateSkip comparison. A math-heavy extra subset stays an optional later ablation and is never used for the main comparison.
- **Source:** GateSkip code at `8d56513`, `experiment/datasets/DatasetConfigurator.py` + `HFReasoningDataset.py` (notes in LITERATURE.md).
- **Still open (D-13):** full fine-tuning (GateSkip) vs LoRA r=16 (ours).
- **Status:** active.

### D-07 · open · Gate scaling vs binary gates at inference
- **Note (2026-11-06):** GateSkip keeps gate scaling at inference (`output = g * module_out`; skipped tokens get zero), so it is GateSkip-style scaling.
- **Decide in:** Phase 3 after comparing both on the validation split.

### D-08 · 2026-11-06 · GSM8K harness task: `gsm8k_cot_llama` with chat template, flexible-extract as headline
- **Context:** EXPERIMENTS.md fixes "8-shot CoT, greedy, max 512 new tokens, exact match" but not the harness task, prompt format or answer filter.
- **Decision:** lm-eval task `gsm8k_cot_llama` (fixed 8 exemplars, "The final answer is N" format) with `apply_chat_template=true`, `fewshot_as_multiturn=true`, `do_sample=false`, `max_gen_toks=512` (the task default has no limit, so HF would cap at 256). Same task for Llama and Qwen. Both `strict-match` and `flexible-extract` are logged; `flexible-extract` is the headline number.
- **Alternatives:** `gsm8k_cot` (plain-text "Q:/A:" 8-shot, no chat template); `gsm8k` 5-shot.
- **Reason:** both models are instruction-tuned and the task mirrors the 8-shot CoT format Meta uses for Llama 3 GSM8K evaluation; chat-formatted prompts avoid penalising instruct models on format.
- **Status:** superseded for trained-model comparisons, first by D-11 and then by D-15 (2026-11-06). **Active as the protocol for the original (untrained) dense models** under D-15 (`gsm8k_cot_llama`, E0 smoke run `20261106-053918_gsm8k_dense_qwen_s0`).

### D-09 · 2026-11-06 · `uv` + `.venv` instead of conda
- **Alternatives:** conda (as written in the original SETUP.md).
- **Reason:** conda is not installed on the dev Mac; `uv` is, and it provides Python 3.11 and fast, lockable installs. Exact versions are frozen in `requirements.lock.txt`.
- **Status:** active.

### D-10 · 2026-11-06 · CI installs only CPU test dependencies
- **Alternatives:** install the full `requirements.txt` in CI.
- **Reason:** the full set pulls CUDA torch and lm-eval (slow, large) although the CPU tests import only torch, transformers, pyyaml and python-dotenv. CI uses the CPU-only torch wheel and `pip install -e . --no-deps`. If a test later needs another package, add it to the workflow.
- **Status:** active.

### D-11 · 2026-11-06 · Primary GSM8K protocol = GateSkip's format (0-shot "Question/Answer … #### N"), full test set
- **Context:** GateSkip evaluates with the stock lm-eval `gsm8k` task, 0-shot, prompt `"Question: …\nAnswer:"`, no chat template, greedy, 1,024 max new tokens, `exact_match`. That format matches its training data (D-06). D-08 used a different format (8-shot chat, "The final answer is N"). Every trained model we compare (GateSkip, ours, baselines) learns the `#### N` format during training, so D-08 would test them on a prompt format they weren't trained on.
- **Decision:** for all E1–E10 GSM8K numbers (dense included), use lm-eval `gsm8k` with `num_fewshot=0`, no chat template, greedy, **max 512 new tokens** (EXPERIMENTS.md; log how often the cap is hit and raise to 1,024 if > 1%), and **all 1,319 test problems**. GateSkip's code hard-codes `limit=100`. Report strict-match (`#### N`) as the headline and log flexible-extract too.
- **Alternatives:** keep D-08 (chat 8-shot) for everything; copy GateSkip exactly, including `limit=100` and 1,024 tokens.
- **Reason:**
  - It matches how the base method was trained and evaluated.
  - The full test set gives a ±1.3-point standard error instead of ±4–5 on 100 problems.
  - 512 tokens follows our fixed protocol; E0 outputs average 143 tokens (max 312).
- **Consequences:**
  - The eval configs need a D-11 variant: task P0-10.
  - The dense Instruct model may score lower 0-shot in this format than with D-08. That's expected and gets reported as the dense reference under the same protocol.
- **Status:** superseded by D-15 (2026-11-06). The plain-text format stays only for the GateSkip reproduction check, D-12(c), which runs in GateSkip's own code. Kept: full 1,319-problem test set, 512-token cap with cap-hit logging, strict-match headline.

### D-12 · 2026-11-06 · GateSkip baseline: retrain on our model; evaluate at bs=1 with fixed calibrated thresholds
- **Context:**
  - GateSkip has **no released checkpoints**, so it has to be trained.
  - Its main 1B runs use **base** Llama-3.2-1B; we target **Instruct**.
  - Its inference threshold is a quantile over the current forward pass. At bs=1 decode (one token per pass) it returns 0.0 and nothing is skipped.
  - Its skipping zeroes module outputs after running the module, so plain HF evaluation saves no compute.
- **Decision:**
  - (a) Train GateSkip with its official recipe (job `GateSkip-separate-vector`, our D-06 data) on **Llama-3.2-1B-Instruct**, and later on Qwen2.5-1.5B-Instruct.
  - (b) Evaluate it with the same calibration procedure as ours (ARCHITECTURE §6): fixed thresholds per budget on the calibration split, applied at bs=1, compute counted with our FLOP counter, latency measured with real module skipping in our engine.
  - (c) Also report one **reproduction check**: GateSkip on base Llama-3.2-1B in its native batch-quantile mode on GSM8K at 0% and 15%, against the paper's numbers. This confirms our setup reproduces theirs.
- **Alternatives:** compare against their numbers as published (different model, 100 problems, batch-relative thresholds); use their native mode only.
- **Reason:** a fair, deployable bs=1 comparison at matched compute (D-05). Their native mode can't produce bs=1 decode savings.
- **Status:** active. Needs a GPU (P0-02).

### D-13 · open · Fine-tuning mode: full fine-tune (GateSkip recipe) vs LoRA r=16 (our plan)
- **Context:** GateSkip trains the whole backbone (lr 3e-6) plus ≈134M gate params on 1× A100 80 GB. We planned LoRA r=16 + routers.
- **Options:**
  - (a) Each method uses its own recipe (GateSkip full fine-tune, ours LoRA). Faithful, but this mixes up method and fine-tuning capacity.
  - (b) Both use LoRA r=16. Fair, cheaper, and fits a 24 GB GPU, but GateSkip is then weaker than the original.
  - (c) Both use full fine-tuning. Fair and strong, but needs ≥ 40–80 GB VRAM.
- **Lean:** (b) for the main comparison plus (a) for GateSkip as a reference row. The final choice depends on the GPU (P0-02).
- **Decide in:** P3-01, once GPU details are known.

### D-14 · 2026-11-06 · superseded by D-15 · Dense reference under D-11 = fine-tuned model at 0% skip; untrained dense is reported under D-08 only
- **Evidence:** run `20261106-060855_gsm8k_zs_dense_qwen_s0`. Untrained Qwen2.5-1.5B-Instruct under D-11 scored 0.0% strict and 16.0% flexible on 50 problems; 0/50 outputs used `#### N`, and 40% ran to the 512-token cap. The same model scored 62.0% on the same 50 problems under D-08. Without the chat template the Instruct model never emits an end-of-turn token, so it runs on after answering.
- **What is wrong in D-11:** its "dense included" consequence. D-11 is right for trained models, which learn `#### N` + EOS from the D-06 data, but useless for an untouched Instruct model. GateSkip's "0% skip" point is likewise the fine-tuned gated model with no skipping, not the original pretrained model.
- **Proposed change:**
  - (a) Under D-11, the 0%-budget dense reference for each method is the model **after training** with all modules executed. For ours that's LoRA + routers with gates forced open; for GateSkip it's the fine-tuned model at skip ratio 0.
  - (b) The untouched dense Instruct model is reported under D-08 as "original model" context (E0).
  - (c) E0's D-11 row stays as evidence, but isn't used as a baseline.
- **Alternatives:**
  - Apply the chat template under D-11 too. That changes GateSkip's format, so trained models would then be evaluated on a format they weren't trained on.
  - Add `####` few-shot exemplars. That mixes in-context learning into a 0-shot protocol.
- **Status:** superseded by D-15 (2026-11-06). Its idea of a trained 0%-skip reference is kept in D-15.
- **Update (2026-11-06):** the alternative "chat format everywhere" (protocol B) was tested on the untrained Qwen model (run `20261106-063044_gsm8k_chat_dense_qwen_s0`) and **failed** the agreed criteria: 40% `#### N` outputs (needed ≥ 90%) and a 30-point strict/flexible gap (needed ≤ 4). The 2% cap rate passed. It is not adopted. The choice between A + D and other fallbacks is pending.

### D-15 · 2026-11-06 · Chat-format training and evaluation; original models under D-08; Dense-FT reference
- **Replaces:** D-14, and D-11 as the main comparison protocol.
- **Decision:**
  1. **Training format.** Convert the D-06 data (CommonsenseQA + GSM8K train) to chat messages for every trained method: GateSkip retrained under D-12, ours, Dense-FT and other baselines. Each example is a system instruction, then the question as the user turn, then the assistant turn `"{steps}\n#### {answer}"`. Training uses each model's own chat template, with the loss on assistant tokens only. GSM8K uses exactly the `system_instruction` from `configs/eval/gsm8k_chat_dense_*.yaml`. CommonsenseQA uses a matching instruction that asks for a final `#### <letter>` line (defined in P3-06).
  2. **Main GSM8K protocol for trained models:** task `gsm8k_chat_zs` (`configs/eval/tasks/gsm8k_chat_zs.yaml`). 0-shot, chat template, the same system instruction as training, greedy, max 512 new tokens, all 1,319 test problems. Strict-match (`#### N`) is the headline. Always logged alongside it: our robust extractor (item 6), flexible-extract and the cap-hit rate.
  3. **Original (untrained) dense models** are reported under D-08 only (8-shot chat CoT), as context. They are never scored under the main protocol, because they don't reliably produce `#### N` (see the evidence below).
  4. **Dense-FT reference (option D).** Fine-tune the dense model on the same chat-format data, with the same recipe, steps and seeds as the main comparison (D-13 fine-tuning mode), but with **no router or gates**. Evaluate it under the main protocol. This is the 0%-compute reference that separates the effect of fine-tuning from the cost of skipping. Each method's own 0%-skip point (gates forced open) is reported next to it.
  5. **GateSkip reproduction check, D-12(c):** unchanged, on base Llama-3.2-1B in GateSkip's native plain-text format and code.
  6. **Robust secondary extractor** (`lagate.eval.answer`, computed by `eval_dense.py` as `robust_extract`). Markers are tried in priority order:
     - `#### N` (first occurrence);
     - `\boxed{N}` (first);
     - "The final answer is N" / "The answer is N" (first, so run-on text is ignored);
     - `**N**` (last bold number);
     - otherwise the last number in the text.

     `$`, thousands separators, `%` and trailing periods are removed, and the result is compared numerically with the gold answer. It is reported next to strict-match and **never replaces it as the headline**. It separates format failures from reasoning failures. Unit tests (`tests/test_answer_extractor.py`) use the five protocol-B outputs that strict-match rejected (docs 0, 1, 2, 3, 10); all five are scored correctly.
- **Evidence:** E0 runs on the same 50 test problems, untrained Qwen2.5-1.5B-Instruct:

  | Protocol | Strict | Flexible | `#### N` used | Hit cap | Avg tokens |
  |---|---|---|---|---|---|
  | D-08 | 62.0% | 62.0% | n/a | 0% | 143 |
  | D-11 | 0.0% | 16.0% | 0% | 40% | 352 |
  | Chat + instruction | 26.0% | 56.0% | 40% | 2% | 219 |

  Robust extractor (re-scored post hoc on the same samples): D-08 64%, D-11 48%, chat 64%. The chat format fixes the run-on outputs, and its reasoning accuracy matches D-08 (64% vs 64% robust). Only the untrained model's `####` compliance failed, and training on the format addresses that.
- **Reason:** one protocol for every trained model, with stopping at the chat end-of-turn token. That keeps output length and latency realistic (R8, E11), matches how Instruct models are deployed (and the demo), and still compares methods on identical data. Dense-FT answers the reviewer question "is the gain from fine-tuning or from routing?".
- **Alternatives:**
  - A + D: GateSkip's plain-text format with a Dense-FT reference. Rejected because stopping would depend on learned EOS, which risks D-11-style run-on outputs.
  - Chat format with the untrained model scored under it: failed the agreed check.
  - Extra stop strings: model-specific, and they can truncate reasoning.
- **Verification checkpoint (P3-08):** after the **first Dense-FT model** is trained on the GPU, rerun the 3-criteria check on it: ≥ 90% of outputs contain `#### N`, ≤ 2% hit the 512-token cap, |strict − flexible| ≤ 4 points. Run it on the calibration split (held-out train data, never test). **If it fails, revisit the protocol before training any gated model** (GateSkip or ours).
- **Status:** active.

### D-16 · 2026-11-06 · active · Implement in our own `lagate` package; GateSkip is re-implemented there as a baseline and checked against the official code
- **Question:** build our method (a) inside a fork of GateSkip's codebase, or (b) in our own `lagate` package, with GateSkip only as a baseline?

| Criterion | (a) Fork GateSkip | (b) Own `lagate` package |
|---|---|---|
| Sharing the train/eval pipeline | Reuses their Lightning loop, data module and lm-eval wrapper as is. | We write a training loop and data module (needed in Phase 3 anyway). GateSkip's method is re-implemented in the **same** loop: per-module vector gate `σ(W h)`, L2 gate loss averaged over modules, percentile ramp 0→0.2, same lr, steps and seeds. That's ≈ 100 lines (`GateLayer` + `GatedWrapper` + `compute_gate_loss`). |
| Fairness | Shared code by construction, but it also shares their known problems (below), which would have to be patched for both methods. | Same loop, data (D-06/D-15), eval protocol (D-15), calibration (§6), FLOP counter and bs=1 engine for every method. The re-implementation is checked against the official code via D-12(c): base Llama-3.2-1B, native format, 0% and 15%, which must match within seed noise. Reviewers see both the faithful official number and the matched comparison. |
| Execution model | Their wrapper **runs every module and then zeroes its output**. A K,V copy from ℓ−1 is intended but is a silent no-op (exception swallowed or early return; see LITERATURE.md evidence, claim 2). Thresholds are batch quantiles, so nothing is skipped at bs=1 decode, although prefill does skip. Our Tier 3 (exact K,V only), real module skipping, the ΔAttn-conditioned MLP head and the cross-token EMA state would all need re-plumbing of their wrapper and Hugging Face internals. | Written to ARCHITECTURE.md from the start: real skipping, exact Tier-3 K,V, EMA state kept next to the KV cache, our own decode loop for latency (P5-01). |
| Environment | lm-eval ≤ 0.4.8, datasets < 3.0, torch 2.8 + CUDA, flash-attn cp310 Linux wheel, vLLM, Python 3.10 container. No macOS/MPS development and no CPU CI without stripping deps. Conflicts with D-09/D-10 and our tested lm-eval 0.4.13 custom task (`gsm8k_chat_zs`). | Our env as is (Python 3.11, transformers 5.22, lm-eval 0.4.13). CPU CI and MPS latency work. The official GateSkip runs **only** for D-12(c), in its own Apptainer/conda env on the GPU machine, never imported by our code. |
| Fit with D-15 | Their data pipeline is plain text (`Question:/Answer:`). It would have to be rewritten for chat format, loss on assistant tokens only, and our system instruction. Their eval would need the `limit=100` fix and our task and extractor. | D-15's chat conversion (P3-06), eval task and robust extractor are already in our package. |
| Effort | Low to start, then growing: patch their wrapper, KV, thresholds, data and eval, keep their transformers-era code working, and still build our own engine for latency. | Higher up front (training loop + data module + GateSkip port ≈ 1–2 weeks), but Phase 2/3 needs most of it anyway. No upstream drift; tests (T-EQ, T-KV, T-T3, T-FLOP, T-EMA) cover our code directly. |
| Main risk | Hidden coupling to their bugs; two codebases mixed in one repo. | "Your GateSkip is weaker than theirs." Mitigated by the D-12(c) check and by reporting the official number. |

- **Recommendation:** **(b)**. Our method's core (real skipping, exact Tier-3 K,V, bs=1 thresholds, chat-format D-15 pipeline) conflicts with GateSkip's execution model and environment. Reusing their code would save little and import known issues. A same-pipeline GateSkip port plus the official-code check is fairer and easier to defend.
- **Consequences if accepted:**
  - D-12(a) becomes "train **our GateSkip port** with their recipe in our pipeline". The official code is used for the D-12(c) reproduction check only.
  - New tasks: P2-09 GateSkip port in `lagate/routing/gateskip.py` (vector gate, L2 loss, percentile ramp, gate scaling at inference per D-07 note), with a unit test against a hand calculation. P4-06 D-12(c) port-vs-official check on the GPU.
  - `baselines/GateSkip` stays a read-only reference clone.
- **Status:** **active** (approved by Rishabh 2026-11-06), with two additions:
  1. **Evidence:** each implementation claim is backed by file and line references at the pinned commit in LITERATURE.md (GateSkip, "Evidence for the implementation claims"). That check corrected claim 2: the K,V copy is intended but never happens, so their published numbers use each token's own exact K,V.
  2. **Two port variants with the same trained weights.** The variant is an engine setting (`kv_mode`), not a separate training run:
     - **GateSkip-exactKV:** our engine with real skipping and exact per-layer K,V for skipped tokens. **This is the main GateSkip baseline**, so differences from our method come only from routing. It also matches what their code actually computed.
     - **GateSkip-copyKV:** for skipped tokens, layer ℓ's cache entry is copied from layer ℓ−1's entry (their *intended* `update_kv_cache`). This measures what the copy costs, as evidence for gap G4. Because of the correction above, this is a measurement of the copy idea, not a reproduction of their numbers.
  - Tasks: P2-09 GateSkip port, P2-10 `kv_mode: copy_prev` option in the engine, P4-06 D-12(c) check. Experiments: E1-exactKV (main) and E1-copyKV in EXPERIMENTS.md.

### D-17 · 2026-11-06 · active · Implementation details the spec leaves open (router, EMA, sparsity costs)
ARCHITECTURE.md does not pin these down; the choices below are what the code does. None of them changes the method.
- **RMSNorm(h_t)** (router input and EMA input) is the decoder layer's own `input_layernorm` output: the same tensor that feeds Q, K, V. It has a learned per-channel weight, costs nothing extra, and is available in every tier (Tier 3 needs it for K,V anyway).
- **RMSNorm(ΔAttn)** is a router-owned RMSNorm (weight initialised to 1).
- **ΔAttn** is the attention sub-layer output after O-proj and **before** gate scaling; at inference this is exactly what is added to the residual.
- **Linear maps:** W_1 and W_a have no bias. The heads w_L, w_M have the +5 bias; their weights start at N(0, 0.01), as GateSkip's gates (`configs/router/lagate.yaml`).
- **EMA scan:** chunked closed form (chunk 64) carrying the state between chunks; equal to the recurrence to 1e-12 in fp64 (T-EMA). The chunk length is a numerics/speed setting, not a hyperparameter.
- **The EMA is updated for every token at every layer, whatever its tier**, because RMSNorm(h_t) is always computed.
- **Sparsity-loss costs (§4):** K,V run in every tier, so they are not in the loss. c_attn = Q,O + attention at the average training context length ((T+1)/2 for a sequence of length T); c_mlp = MLP. L_S is therefore the expected fraction of the *skippable* FLOPs that are used.
- **FLOP accounting (§7)** counts the linear maps and attention matmuls only; it leaves out embeddings, LM head, norms, RoPE, element-wise ops, the EMA update and biases. The counter agrees with `torch.utils.flop_counter` on the tiny models (T-FLOP).
- **T-KV / T-T3 tolerance:** run in float64 at atol 1e-9. In fp32, running row subsets (only tier-1/2 rows through Q and the MLP) changes matmul shapes and moves logits by about 1e-5 even when the logic is exact, so an fp32 check at 1e-5 would be flaky. T-EQ (router off, dense = HF) stays fp32 at 1e-5 as specified.
- **Engine "tier 0"** (MLP ran, attention skipped) is never produced by our policy. It exists only for the GateSkip port, whose attention and MLP gates are independent (P2-09).
