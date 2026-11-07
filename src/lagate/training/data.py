"""Training data behind one ``make_batches`` interface: ``source: synthetic`` (P2-08 smoke runs) and
``source: chat`` (the D-06 data in chat format, P3-06, ``lagate.training.chat_data``).

Synthetic sequences: random start a and step d, token_t = (a + t·d) mod V (tokens 0 and 1 are
reserved as pad and EOS). The next token is predictable from the previous two, so a tiny model's
loss drops within a few steps, which is what a smoke run needs to show. Lengths vary, so batches
are right-padded and the padding / loss-mask paths are exercised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import torch

PAD_ID, EOS_ID, FIRST_TOKEN = 0, 1, 2


@dataclass
class Batch:
    input_ids: torch.Tensor       # (B, T)
    attention_mask: torch.Tensor  # (B, T) 1 = token
    labels: torch.Tensor          # (B, T), -100 = no loss


def synthetic_sequence(gen: torch.Generator, vocab: int, length: int) -> list[int]:
    span = vocab - FIRST_TOKEN
    a = int(torch.randint(0, span, (1,), generator=gen))
    d = int(torch.randint(1, 8, (1,), generator=gen))
    return [FIRST_TOKEN + (a + t * d) % span for t in range(length - 1)] + [EOS_ID]


def collate(seqs: list[list[int]]) -> Batch:
    T = max(map(len, seqs))
    ids = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
        mask[i, : len(s)] = 1
    labels = ids.masked_fill(mask == 0, -100)
    return Batch(ids, mask, labels)


def make_batches(data_cfg: dict, vocab: int, seed: int, split: str = "train", chat=None) -> Iterator[Batch]:
    """``chat``: a ``ChatDataset`` (required for ``source: chat``)."""
    if data_cfg["source"] == "chat":
        if chat is None:
            raise ValueError("source: chat needs a ChatDataset (built from the model's tokenizer)")
        yield from chat.batches(split, data_cfg["batch_size"], seed)
        return
    if data_cfg["source"] != "synthetic":
        raise ValueError(f"unknown data source {data_cfg['source']!r}")
    offset = {"train": 0, "calibration": 1_000_003, "eval": 2_000_003}[split]
    gen = torch.Generator().manual_seed(seed + offset)
    lo, hi = data_cfg["min_len"], data_cfg["max_len"]
    while True:
        lens = torch.randint(lo, hi + 1, (data_cfg["batch_size"],), generator=gen).tolist()
        yield collate([synthetic_sequence(gen, vocab, n) for n in lens])
