# Baselines

Clones live in this folder and are git-ignored; only this file is tracked. Re-create a clone at the pinned commit with:

```bash
git clone https://github.com/Thiggel/GateSkip baselines/GateSkip
git -C baselines/GateSkip checkout 8d565130266fecc4a2a5374d0ced135f05e4b9f9
```

| Baseline | Source | Pinned commit | Cloned | Runs? | Notes |
|---|---|---|---|---|---|
| GateSkip | https://github.com/Thiggel/GateSkip | `8d565130266fecc4a2a5374d0ced135f05e4b9f9` (2025-11-27) | 2026-11-06 | ☐ not run yet | No released checkpoints, so it must be trained. Needs CUDA (torch 2.8, flash-attn, vLLM), Python 3.10 and its own env. Code notes in docs/LITERATURE.md. |
| D-LLM | https://github.com/Jyk-122/D-LLM | | | ☐ | |
| SkipGPT | https://github.com/EIT-NLP/SkipGPT | | | ☐ | |
| FlexiDepth | authors' release (see arXiv page) | | | ☐ | |
| FFN-skip (DiffSkip-style) | re-implement here | – | | ☐ | |
