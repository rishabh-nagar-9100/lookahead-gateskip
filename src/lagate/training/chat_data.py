"""D-06 training data in chat format (D-15, P3-06).

Sources (D-06, GateSkip's ``csqa_gsm8k_gen`` mix, ``HFReasoningDataset`` at 8d56513):
  gsm8k  ``openai/gsm8k`` (main, train split). Question = ``question``; ``answer`` is
         "{steps}\\n#### {N}".
  csqa   ``multi-domain-reasoning/commonsense_qa`` (train split). Question = ``question`` plus one
         line per choice "A: ...", steps = ``reasoning_nemotron_70B`` without the <reasoning> tags,
         answer = ``answerKey`` (a letter).

Each example becomes three chat messages (D-15 item 1):
  system     the source's instruction; for GSM8K it is read verbatim from the eval config
             (``configs/eval/gsm8k_chat_dense_*.yaml``), so training and evaluation can never drift
  user       the question
  assistant  "{steps}\\n#### {answer}"

Differences from GateSkip's plain-text targets, all deliberate and recorded in DECISIONS (D-23):
  * steps are whitespace-stripped, so the GSM8K target is the dataset's own answer text
    ("...\\n#### N"); GateSkip's ``f" {steps}\\n#### {answer}"`` gives a blank line before "####"
    because its split keeps the trailing newline.
  * GSM8K calculator annotations ("<<48/2=24>>") are kept by default, as in GateSkip
    (``strip_calculator_annotations`` can remove them for an ablation).

Tokenisation renders the conversation with the model's own chat template (text first, then the
tokenizer with ``add_special_tokens=False``, since the template already contains any BOS). The
loss covers the assistant content and the first end-of-turn token after it (``<|im_end|>`` for
Qwen, ``<|eot_id|>`` for Llama), so the model learns to stop; the template's trailing newline is
masked because generation stops at the end-of-turn token. The prompt part must be an exact token
prefix of the full conversation; a template where it isn't raises instead of mislabelling.

Splits: per source, a fixed number of train-split examples are held out as the **calibration**
split (thresholds, P3-08 checks; never the test set). The hold-out uses ``split_seed`` from the
data config, not the run seed, so every method and seed sees the same split. Training order is
shuffled per epoch with the run seed. Examples longer than ``max_length`` tokens are dropped
(as GateSkip does), and the count is reported.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from lagate.utils import REPO_ROOT, load_config

SOURCES = ("gsm8k", "csqa")
LETTERS = "ABCDE"
_CALC = re.compile(r"<<[^>]*>>")


# ----- raw rows -> examples -----------------------------------------------------------------------
def gsm8k_example(row: dict, strip_calculator_annotations: bool = False) -> dict[str, str]:
    steps, sep, answer = row["answer"].rpartition("#### ")
    if not sep:
        raise ValueError(f"GSM8K answer without '#### ': {row['answer'][-80:]!r}")
    steps = steps.strip()
    if strip_calculator_annotations:
        steps = _CALC.sub("", steps)
    return {"source": "gsm8k", "question": row["question"].strip(), "steps": steps, "answer": answer.strip()}


def csqa_example(row: dict) -> dict[str, str]:
    choices = row["choices"]["text"]
    q = row["question"].strip() + "\n" + "\n".join(f"{LETTERS[i]}: {c}" for i, c in enumerate(choices))
    steps = row["reasoning_nemotron_70B"].replace("<reasoning>", "").replace("</reasoning>", "").strip()
    return {"source": "csqa", "question": q, "steps": steps, "answer": row["answerKey"].strip()}


def to_messages(ex: dict[str, str], instructions: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": instructions[ex["source"]]},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": f"{ex['steps']}\n#### {ex['answer']}"},
    ]


def system_instructions(data_cfg: dict) -> dict[str, str]:
    """Per-source system instruction. ``{from_eval_config: path}`` reads ``eval.system_instruction``."""
    out = {}
    for src, spec in data_cfg["system_instruction"].items():
        if isinstance(spec, dict):
            out[src] = load_config(REPO_ROOT / spec["from_eval_config"])["eval"]["system_instruction"]
        else:
            out[src] = spec
    return out


# ----- tokenisation -------------------------------------------------------------------------------
@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]   # -100 = no loss
    source: str


def tokenize_chat(tokenizer, messages: list[dict], end_of_turn_ids: set[int]) -> TokenizedExample:
    prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full[: len(prompt)] != prompt:
        raise ValueError("chat template: the generation prompt is not a token prefix of the full conversation")
    stop = next((i for i in range(len(prompt), len(full)) if full[i] in end_of_turn_ids), None)
    if stop is None:
        raise ValueError("no end-of-turn token after the assistant content (check data.end_of_turn_tokens)")
    labels = [-100] * len(prompt) + full[len(prompt) : stop + 1] + [-100] * (len(full) - stop - 1)
    return TokenizedExample(full, labels, messages[0].get("source", ""))


def end_of_turn_ids(tokenizer, tokens: list[str]) -> set[int]:
    vocab = tokenizer.get_vocab()
    ids = {vocab[t] for t in tokens if t in vocab}
    if not ids:
        raise ValueError(f"none of {tokens} is in the tokenizer vocabulary")
    return ids


# ----- loading + splitting ------------------------------------------------------------------------
def load_raw(data_cfg: dict) -> dict[str, list[dict]]:
    """Train-split rows per source, from the pinned dataset revisions (downloads on first use)."""
    from datasets import load_dataset

    raw = {}
    for src, spec in data_cfg["datasets"].items():
        ds = load_dataset(spec["path"], spec.get("name"), split=spec.get("split", "train"), revision=spec.get("revision"))
        raw[src] = [dict(r) for r in ds]
    return raw


def split_indices(n: int, n_calibration: int, split_seed: int) -> tuple[list[int], list[int]]:
    """(train, calibration) row indices; the calibration ids are sorted for readability."""
    if n_calibration >= n:
        raise ValueError(f"calibration size {n_calibration} >= dataset size {n}")
    order = list(range(n))
    random.Random(split_seed).shuffle(order)
    return sorted(order[n_calibration:]), sorted(order[:n_calibration])


def build_examples(raw: dict[str, list[dict]], data_cfg: dict) -> dict[str, list[dict[str, str]]]:
    """{'train': [...], 'calibration': [...]} examples, each with its source row index (``row``)."""
    out: dict[str, list[dict[str, str]]] = {"train": [], "calibration": []}
    for src, rows in raw.items():
        tr, cal = split_indices(len(rows), data_cfg["calibration_size"][src], data_cfg["split_seed"])
        for split, idx in (("train", tr), ("calibration", cal)):
            for i in idx:
                ex = (gsm8k_example(rows[i], data_cfg.get("strip_calculator_annotations", False))
                      if src == "gsm8k" else csqa_example(rows[i]))
                out[split].append({**ex, "row": i})
    return out


def calibration_row_ids(raw_len: int, data_cfg: dict, source: str) -> list[int]:
    return split_indices(raw_len, data_cfg["calibration_size"][source], data_cfg["split_seed"])[1]


def ids_digest(ids: list[int]) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:16]


class ChatDataset:
    """Tokenised train / calibration examples for one tokenizer."""

    def __init__(self, data_cfg: dict, tokenizer, raw: dict[str, list[dict]] | None = None):
        self.cfg = data_cfg
        self.tokenizer = tokenizer
        raw = raw if raw is not None else load_raw(data_cfg)
        self.raw_sizes = {k: len(v) for k, v in raw.items()}
        instr = system_instructions(data_cfg)
        eot = end_of_turn_ids(tokenizer, list(data_cfg["end_of_turn_tokens"]))
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.special_ids = sorted({self.pad_id, *eot})
        self.splits: dict[str, list[TokenizedExample]] = {}
        self.dropped: dict[str, dict[str, int]] = {}
        self.calibration_rows = {}
        for split, exs in build_examples(raw, data_cfg).items():
            kept, dropped = [], {}
            for ex in exs:
                t = tokenize_chat(tokenizer, to_messages(ex, instr), eot)
                t.source = ex["source"]
                if len(t.input_ids) > data_cfg["max_length"]:
                    dropped[ex["source"]] = dropped.get(ex["source"], 0) + 1
                    continue
                kept.append(t)
            self.splits[split], self.dropped[split] = kept, dropped
            if split == "calibration":
                self.calibration_rows = {s: sorted(e["row"] for e in exs if e["source"] == s) for s in raw}

    def stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {"raw_sizes": self.raw_sizes, "dropped_over_max_length": self.dropped,
                               "calibration_row_digest": {s: ids_digest(v) for s, v in self.calibration_rows.items()}}
        for split, exs in self.splits.items():
            lens = sorted(len(e.input_ids) for e in exs)
            n_lab = [sum(l != -100 for l in e.labels) for e in exs]
            out[split] = {
                "n": len(exs),
                "by_source": {s: sum(e.source == s for e in exs) for s in self.raw_sizes},
                "tokens_total": sum(lens),
                "len_mean": sum(lens) / max(1, len(lens)),
                "len_p50": lens[len(lens) // 2] if lens else 0,
                "len_p99": lens[int(0.99 * (len(lens) - 1))] if lens else 0,
                "len_max": lens[-1] if lens else 0,
                "label_tokens_mean": sum(n_lab) / max(1, len(n_lab)),
            }
        return out

    def batches(self, split: str, batch_size: int, seed: int) -> Iterator["Batch"]:
        """Train: reshuffled every epoch (seed + epoch). Calibration: fixed order, cycling."""
        exs = self.splits[split]
        epoch = 0
        while True:
            order = list(range(len(exs)))
            if split == "train":
                random.Random(seed * 1_000_003 + epoch).shuffle(order)
            for i in range(0, len(order) - batch_size + 1, batch_size):
                yield collate_chat([exs[j] for j in order[i : i + batch_size]], self.pad_id)
            epoch += 1


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def collate_chat(exs: list[TokenizedExample], pad_id: int) -> Batch:
    """Right padding (LookaheadGatedModel requires it); pads get label -100 and mask 0."""
    T = max(len(e.input_ids) for e in exs)
    ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
    labels = torch.full((len(exs), T), -100, dtype=torch.long)
    mask = torch.zeros((len(exs), T), dtype=torch.long)
    for i, e in enumerate(exs):
        n = len(e.input_ids)
        ids[i, :n] = torch.tensor(e.input_ids)
        labels[i, :n] = torch.tensor(e.labels)
        mask[i, :n] = 1
    return Batch(ids, mask, labels)
