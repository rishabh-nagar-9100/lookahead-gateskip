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

## Novelty watch

Re-run these searches monthly and before submission (arXiv, Google Scholar, Semantic Scholar):
"layer skipping router memory", "token-wise layer skipping reasoning", "temporal routing layer skip", "history-aware routing LLM", "attention-only MLP skip router".
Log anything new that overlaps here with date found.
