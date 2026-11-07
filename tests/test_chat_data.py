"""P3-06: D-06 data in chat format (lagate.training.chat_data).

Most tests use a small stub tokenizer with a Qwen-style template, so they run in CI without
downloads. The last test uses the real Qwen2.5 tokenizer when it is in the local HF cache.
"""

from __future__ import annotations

import re

import pytest
import torch

from lagate.training.chat_data import (
    ChatDataset,
    build_examples,
    collate_chat,
    csqa_example,
    gsm8k_example,
    split_indices,
    system_instructions,
    to_messages,
    tokenize_chat,
)
from lagate.utils import REPO_ROOT, load_config

DATA_CFG = load_config(REPO_ROOT / "configs/data/d06_chat.yaml")

GSM_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
}
CSQA_ROW = {
    "question": "Where would you find a jellyfish?",
    "choices": {"label": list("ABCDE"), "text": ["desert", "ocean", "attic", "library", "oven"]},
    "answerKey": "B",
    "reasoning_nemotron_70B": "<reasoning>Jellyfish live in salt water.\nThe ocean is salt water.</reasoning>",
}


class StubTokenizer:
    """Qwen-style chat template; special tokens are single ids, other text is one id per character."""

    SPECIAL = ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]

    def __init__(self):
        self.vocab = {t: i for i, t in enumerate(self.SPECIAL)}
        self.pad_token_id = self.vocab["<|endoftext|>"]
        self.eos_token_id = self.vocab["<|im_end|>"]
        self._split = re.compile("(" + "|".join(map(re.escape, self.SPECIAL)) + ")")

    def get_vocab(self):
        return dict(self.vocab)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        return s + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for part in self._split.split(text):
            if part in self.vocab:
                ids.append(self.vocab[part])
            else:
                ids.extend(1000 + ord(c) for c in part)
        return {"input_ids": ids}

    def decode(self, ids):
        inv = {v: k for k, v in self.vocab.items()}
        return "".join(inv.get(i, chr(i - 1000) if i >= 1000 else "?") for i in ids)


def test_gsm8k_example_matches_dataset_answer():
    ex = gsm8k_example(GSM_ROW)
    assert ex["answer"] == "72"
    assert f"{ex['steps']}\n#### {ex['answer']}" == GSM_ROW["answer"]  # single newline before ####
    assert "<<48/2=24>>" in ex["steps"]  # kept, as in GateSkip
    assert "<<" not in gsm8k_example(GSM_ROW, strip_calculator_annotations=True)["steps"]


def test_csqa_example_format():
    ex = csqa_example(CSQA_ROW)
    assert ex["question"].splitlines() == ["Where would you find a jellyfish?", "A: desert", "B: ocean", "C: attic", "D: library", "E: oven"]
    assert ex["steps"] == "Jellyfish live in salt water.\nThe ocean is salt water."
    assert ex["answer"] == "B"


def test_gsm8k_system_instruction_is_the_eval_instruction_verbatim():
    instr = system_instructions(DATA_CFG)
    for model in ("qwen", "llama"):
        ev = load_config(REPO_ROOT / f"configs/eval/gsm8k_chat_dense_{model}.yaml")["eval"]["system_instruction"]
        assert instr["gsm8k"] == ev
    assert "####" in instr["csqa"] and "letter" in instr["csqa"]


def test_messages_roles_and_assistant_target():
    msgs = to_messages(gsm8k_example(GSM_ROW), system_instructions(DATA_CFG))
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert msgs[1]["content"] == GSM_ROW["question"]
    assert msgs[2]["content"] == GSM_ROW["answer"]


def test_labels_cover_assistant_content_and_end_of_turn_only():
    tok = StubTokenizer()
    msgs = to_messages(gsm8k_example(GSM_ROW), system_instructions(DATA_CFG))
    t = tokenize_chat(tok, msgs, {tok.eos_token_id})
    assert len(t.labels) == len(t.input_ids)
    learned = [i for i, l in zip(t.input_ids, t.labels) if l != -100]
    assert tok.decode(learned) == GSM_ROW["answer"] + "<|im_end|>"
    # everything before is the generation prompt, the trailing template newline is masked
    first = next(i for i, l in enumerate(t.labels) if l != -100)
    assert tok.decode(t.input_ids[:first]).endswith("<|im_start|>assistant\n")
    assert t.labels[-1] == -100 and tok.decode(t.input_ids[-1:]) == "\n"


def test_template_without_prefix_property_raises():
    class Bad(StubTokenizer):
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            s = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return ("X" + s) if add_generation_prompt else s

    tok = Bad()
    with pytest.raises(ValueError, match="prefix"):
        tokenize_chat(tok, to_messages(gsm8k_example(GSM_ROW), system_instructions(DATA_CFG)), {tok.eos_token_id})


def test_split_is_fixed_disjoint_and_independent_of_run_seed():
    tr, cal = split_indices(100, 10, 1234)
    assert len(cal) == 10 and not set(tr) & set(cal) and sorted(tr + cal) == list(range(100))
    assert split_indices(100, 10, 1234) == (tr, cal)
    assert split_indices(100, 10, 99)[1] != cal


def _raw(n_gsm=30, n_csqa=20):
    gsm = [{"question": f"q{i}", "answer": f"step {i}\n#### {i}"} for i in range(n_gsm)]
    cs = [{**CSQA_ROW, "question": f"c{i}"} for i in range(n_csqa)]
    return {"gsm8k": gsm, "csqa": cs}


def test_dataset_splits_lengths_and_batches():
    cfg = {**DATA_CFG, "calibration_size": {"gsm8k": 5, "csqa": 4}, "batch_size": 4}
    ds = ChatDataset(cfg, StubTokenizer(), raw=_raw())
    st = ds.stats()
    assert st["calibration"]["by_source"] == {"gsm8k": 5, "csqa": 4}
    assert st["train"]["by_source"] == {"gsm8k": 25, "csqa": 16}
    rows = build_examples(_raw(), cfg)
    assert {r["row"] for r in rows["calibration"] if r["source"] == "gsm8k"} == set(ds.calibration_rows["gsm8k"])
    b = next(ds.batches("train", 4, seed=0))
    assert b.input_ids.shape[0] == 4
    assert (b.labels[b.attention_mask == 0] == -100).all()
    assert ((b.labels != -100).sum(1) > 0).all()
    # train order depends on the run seed, calibration order does not
    a0 = next(ds.batches("train", 4, seed=0)).input_ids
    a1 = next(ds.batches("train", 4, seed=1)).input_ids
    assert a0.shape != a1.shape or not torch.equal(a0, a1)
    assert torch.equal(next(ds.batches("calibration", 4, 0)).input_ids, next(ds.batches("calibration", 4, 1)).input_ids)


def test_overlong_examples_dropped_and_counted():
    raw = _raw()
    raw["gsm8k"][0] = {"question": "x" * 3000, "answer": "s\n#### 1"}
    cfg = {**DATA_CFG, "calibration_size": {"gsm8k": 1, "csqa": 1}, "max_length": 1024}
    ds = ChatDataset(cfg, StubTokenizer(), raw=raw)
    total = sum(sum(d.values()) for d in ds.dropped.values())
    assert total == 1
    assert all(len(e.input_ids) <= 1024 for s in ds.splits.values() for e in s)


def test_collate_right_pads():
    tok = StubTokenizer()
    instr = system_instructions(DATA_CFG)
    a = tokenize_chat(tok, to_messages(gsm8k_example(GSM_ROW), instr), {tok.eos_token_id})
    b = tokenize_chat(tok, to_messages(csqa_example(CSQA_ROW), instr), {tok.eos_token_id})
    batch = collate_chat([a, b], tok.pad_token_id)
    n = min(len(a.input_ids), len(b.input_ids))
    short = 0 if len(a.input_ids) == n else 1
    assert batch.attention_mask[short, :n].all() and not batch.attention_mask[short, n:].any()
    assert (batch.input_ids[short, n:] == tok.pad_token_id).all()


def _cached_qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", local_files_only=True)
    except Exception:
        return None


@pytest.mark.skipif(_cached_qwen_tokenizer() is None, reason="Qwen2.5 tokenizer not in the local HF cache")
def test_real_qwen_template_labels():
    tok = _cached_qwen_tokenizer()
    from lagate.training.chat_data import end_of_turn_ids

    eot = end_of_turn_ids(tok, DATA_CFG["end_of_turn_tokens"])
    assert eot == {tok.convert_tokens_to_ids("<|im_end|>")}
    msgs = to_messages(gsm8k_example(GSM_ROW), system_instructions(DATA_CFG))
    t = tokenize_chat(tok, msgs, eot)
    learned = [i for i, l in zip(t.input_ids, t.labels) if l != -100]
    assert tok.decode(learned) == GSM_ROW["answer"] + "<|im_end|>"
    prompt = tok.decode(t.input_ids[: next(i for i, l in enumerate(t.labels) if l != -100)])
    assert prompt.startswith("<|im_start|>system\n" + system_instructions(DATA_CFG)["gsm8k"] + "<|im_end|>")
    assert prompt.endswith("<|im_start|>user\n" + GSM_ROW["question"] + "<|im_end|>\n<|im_start|>assistant\n")
