"""P1-02: log per-token, per-layer gate values and skip decisions during generation.

Attach to anything with a ``layer_hooks`` list: the inference engine (``TieredEngine``, prefill +
decode) or the soft-gated training model (``LookaheadGatedModel``, teacher-forced). The GateSkip port
(P2-09) runs in the same engine, so its gates are logged the same way: ``g_layer`` / ``g_mlp`` then
hold its attention-gate / MLP-gate importances.

    logger = GateLogger().attach(engine)
    for doc in problems:
        logger.new_sequence(doc_id=doc.id)
        engine.generate(...)
        logger.set_meta(correct=...)          # after scoring
    logger.save("results/<run_id>/gates.npz")

Arrays (N = logged tokens, L = layers): token_id, position, seq, phase (0 prefill, 1 decode,
2 train) of shape (N,); g_layer, g_mlp (float32, NaN where not computed); run_attn, run_mlp (bool);
tier (int8: 1/2/3, 0 = MLP without attention) of shape (N, L). Per-sequence metadata goes in a JSON
side file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

PHASES = {"prefill": 0, "decode": 1, "train": 2}


class GateLogger:
    def __init__(self):
        self._chunks: list[dict[str, Any]] = []  # one per forward chunk, filled layer by layer
        self._seq = -1
        self.sequences: list[dict[str, Any]] = []
        self._targets: list[Any] = []

    # ----- wiring ----------------------------------------------------------------------------
    def attach(self, target) -> "GateLogger":
        target.layer_hooks.append(self)
        self._targets.append(target)
        return self

    def detach(self) -> None:
        for t in self._targets:
            if self in t.layer_hooks:
                t.layer_hooks.remove(self)
        self._targets.clear()

    def new_sequence(self, **meta) -> int:
        self._seq += 1
        self.sequences.append(dict(meta))
        return self._seq

    def set_meta(self, **meta) -> None:
        self.sequences[self._seq].update(meta)

    # ----- hook --------------------------------------------------------------------------------
    @torch.no_grad()
    def __call__(self, rec: dict) -> None:
        if self._seq < 0:
            self.new_sequence()
        layer = rec["layer"]
        if layer == 0:
            self._chunks.append({"rec": rec, "seq": self._seq, "layers": []})
        chunk = self._chunks[-1]
        if len(chunk["layers"]) != layer:
            raise RuntimeError(f"layer records out of order: got layer {layer} after {len(chunk['layers'])}")
        chunk["layers"].append({k: _cpu(rec.get(k)) for k in ("g_layer", "g_mlp", "run_attn", "run_mlp", "tier")})

    # ----- output ------------------------------------------------------------------------------
    def to_arrays(self) -> dict[str, np.ndarray]:
        parts: dict[str, list[np.ndarray]] = {k: [] for k in
                                              ("token_id", "position", "seq", "phase", "g_layer", "g_mlp", "run_attn", "run_mlp", "tier")}
        for chunk in self._chunks:
            rec, layers = chunk["rec"], chunk["layers"]
            ids, pos = _cpu(rec["input_ids"]), _cpu(rec["positions"])
            valid = _cpu(rec.get("valid"))
            if ids.dim() == 1:  # engine: (T,)
                ids, seqs = ids[None], [chunk["seq"]]
            else:               # training model: (B, T), one logged sequence per row
                seqs = [chunk["seq"] + b for b in range(ids.shape[0])]
            B, T = ids.shape
            keep = torch.ones(B, T, dtype=torch.bool) if valid is None else valid.bool()
            stacked = {k: _stack_layers([lay[k] for lay in layers], B, T, k) for k in ("g_layer", "g_mlp", "run_attn", "run_mlp", "tier")}
            for b in range(B):
                m = keep[b]
                parts["token_id"].append(ids[b][m].numpy())
                parts["position"].append(pos.expand(B, T)[b][m].numpy() if pos.dim() == 1 else pos[b][m].numpy())
                parts["seq"].append(np.full(int(m.sum()), seqs[b]))
                parts["phase"].append(np.full(int(m.sum()), PHASES[rec["phase"]], dtype=np.int8))
                for k, v in stacked.items():
                    parts[k].append(v[b][m].numpy())
        if not parts["token_id"]:
            return {}
        return {k: np.concatenate(v) for k, v in parts.items()}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_arrays())
        path.with_suffix(".sequences.json").write_text(json.dumps(self.sequences, indent=1, default=str))
        return path


def load_gate_log(path: str | Path) -> tuple[dict[str, np.ndarray], list[dict]]:
    path = Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return arrays, json.loads(path.with_suffix(".sequences.json").read_text())


def _cpu(x):
    return x.detach().cpu() if isinstance(x, torch.Tensor) else x


def _stack_layers(per_layer: list, B: int, T: int, key: str) -> torch.Tensor:
    """List over layers of (T,) / (B, T) / None -> (B, T, L). None means the source does not decide
    (soft training forward, or the router is off): gates NaN, every module ran (tier 1)."""
    cols = []
    for v in per_layer:
        if v is None:
            fill = float("nan") if key.startswith("g_") else (1 if key == "tier" else True)
            dtype = torch.float32 if key.startswith("g_") else (torch.int8 if key == "tier" else torch.bool)
            v = torch.full((B, T), fill, dtype=dtype)
        else:
            v = v.reshape(B, T) if v.dim() == 2 else v.reshape(1, T).expand(B, T)
            if key.startswith("g_"):
                v = v.float()
        cols.append(v)
    return torch.stack(cols, dim=-1)
