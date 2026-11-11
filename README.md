# LookAhead-GateSkip

> **Start here: [docs/STATUS.md](docs/STATUS.md)** — current status, next steps, Colab checklist and how to resume.

Context-aware dynamic layer skipping for reasoning in Small Language Models (SLMs).
Major Project (21CSP401L), SRM Institute of Science and Technology.

- **Student:** Rishabh Nagar (RA2311033010085)
- **Guide:** Dr. Mitul Sudhirkumar Nagar, Department of Computational Intelligence
- **Base paper:** GateSkip, "What Layers When: Learning to Skip Compute in LLMs with Residual Gates" (ICLR 2026), https://arxiv.org/abs/2510.13876, code: https://github.com/Thiggel/GateSkip

## One-paragraph summary

LookAhead-GateSkip adds a **cross-token temporal routing memory** to GateSkip-style residual gates. A per-layer EMA state μ_t summarises hidden states of recently generated tokens. A shared-bottleneck router with two heads produces a layer gate g_L and an MLP gate g_M. The MLP head also reads the attention update ΔAttn. Each token at each layer runs in one of three tiers:

| Tier | Condition | What runs |
|---|---|---|
| 1 | g_L > τ_L and g_M > τ_M | Attention + MLP (full layer) |
| 2 | g_L > τ_L and g_M ≤ τ_M | Attention only (MLP skipped) |
| 3 | g_L ≤ τ_L | Only K,V projections (exact KV cache); rest skipped |

The goal is a better accuracy–compute trade-off than GateSkip on GSM8K at matched compute, with real batch-size-1 latency gains on Llama-3.2-1B and Qwen2.5-1.5B.

## Where to find things

| File | Purpose | Update when |
|---|---|---|
| `docs/STATUS.md` | One-page status and resume guide (read first) | At every pause or phase change |
| `docs/PROJECT_PLAN.md` | Phases, milestones, review deliverables, go/no-go gates | Phase boundaries; scope changes |
| `docs/TASKS.md` | Task checklist with IDs and status | Every work session |
| `docs/ARCHITECTURE.md` | Technical spec: math, shapes, tiers, KV handling, FLOPs | Any design change (and log it in DECISIONS) |
| `docs/EXPERIMENTS.md` | Experiment registry, evaluation protocol, results tables | Before and after every run |
| `docs/DECISIONS.md` | Why each design or scope choice was made | Whenever a choice is made |
| `docs/DEV_LOG.md` | Dated diary of what was done, problems, next steps | End of every session |
| `docs/LITERATURE.md` | Papers, links, what to take from each | New paper found |
| `docs/SETUP.md` | Accounts, hardware, environment, install steps | Environment changes |
| `docs/RISKS.md` | Risk register with mitigations | Weekly |
| `docs/PAPER_PLAN.md` | Paper outline, figures, target venues | From Phase 4 onward |
| `CLAUDE.md` | Compact project context for AI coding assistants | When structure or conventions change |

## Planned repository layout

```
lookahead-gateskip/
├── README.md
├── CLAUDE.md
├── docs/                      # all tracking docs (this folder)
├── configs/                   # YAML configs: model, router, training, eval, latency
├── src/lagate/
│   ├── models/                # wrappers for Llama / Qwen decoder layers
│   ├── routing/               # EMA memory, dual-head router, gate losses
│   ├── engine/                # three-tier inference engine, KV-cache handling
│   ├── training/              # LoRA + router training loop, budget schedule
│   ├── calibration/           # empirical-CDF threshold calibration
│   ├── eval/                  # GSM8K / ARC / PIQA / MBPP evaluation wrappers
│   ├── latency/               # batch-size-1 timing harness (CUDA, MPS)
│   └── analysis/              # pilot study: error-token analysis, gate correlation
├── scripts/                   # entry points: train.py, calibrate.py, evaluate.py, bench.py
├── tests/                     # equivalence + KV-cache correctness tests
├── notebooks/                 # exploration only; never the source of truth
├── results/                   # summarised results (CSV/JSON); raw outputs git-ignored
├── baselines/                 # pinned clones / notes for GateSkip, FlexiDepth, D-LLM, SkipGPT
└── paper/                     # manuscript, figures, bib
```

## Status

See [docs/STATUS.md](docs/STATUS.md) for the summary and `docs/TASKS.md` for live task status. Phases 0–2 are done on CPU/MPS; Phase 3 (training on Colab) is next, starting with Colab session S1. Paused 2026-11-11 for a week.
