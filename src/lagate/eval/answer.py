"""Robust GSM8K answer extraction: the secondary metric next to lm-eval's strict-match (DECISIONS D-15).

Strict-match (``#### N``) stays the headline. This extractor also accepts the answer styles
instruction-tuned models use on their own, so format failures can be separated from reasoning
failures. Markers are tried in priority order; the first marker that occurs wins:

1. ``#### N``                        first occurrence (same as lm-eval strict-match)
2. ``\\boxed{N}``                    first occurrence
3. ``The final answer is N`` / ``The answer is N``   first occurrence (run-on text after
   the first answer is ignored)
4. ``**N**`` (bold number)           last occurrence
5. fallback: the last number in the text

Numbers may carry ``$``, thousands separators, a trailing period or a ``%`` sign; all are removed.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUM = r"-?\$?\s?\d[\d,]*(?:\.\d+)?"

_MARKERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"####\s*({_NUM})"), "first"),
    (re.compile(rf"\\boxed\{{\s*(?:\\text\{{)?\s*({_NUM})"), "first"),
    (re.compile(rf"(?:final\s+)?answer\s+is\s*:?\s*\$?\\?\(?\s*({_NUM})", re.IGNORECASE), "first"),
    (re.compile(rf"\*\*\s*({_NUM})\s*\*\*"), "last"),
)
_ANY_NUMBER = re.compile(_NUM)


def normalize_number(text: str) -> str | None:
    """'$1,234.50' -> '1234.5'; '18.' -> '18'; returns None if it is not a number."""
    cleaned = text.replace("$", "").replace(",", "").replace("%", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def extract_answer(text: str) -> str | None:
    """Return the normalised final answer in a model response, or None if there is no number."""
    for pattern, which in _MARKERS:
        matches = pattern.findall(text)
        if matches:
            return normalize_number(matches[0] if which == "first" else matches[-1])
    numbers = _ANY_NUMBER.findall(text)
    return normalize_number(numbers[-1]) if numbers else None


def gsm8k_target(answer_field: str) -> str | None:
    """Gold answer from a GSM8K ``answer`` field ('... #### 18')."""
    return normalize_number(answer_field.split("####")[-1])


def is_correct(response: str, answer_field: str) -> bool:
    pred = extract_answer(response)
    return pred is not None and pred == gsm8k_target(answer_field)


def score_samples(samples: list[dict]) -> dict[str, float]:
    """Robust accuracy over lm-eval logged samples (one per doc; duplicates per filter are merged)."""
    by_doc = {s["doc_id"]: s for s in samples if s.get("resps")}
    if not by_doc:
        return {}
    responses = [(s["resps"][0][0], s["doc"]["answer"]) for s in by_doc.values()]
    n_correct = sum(is_correct(r, a) for r, a in responses)
    n_extracted = sum(extract_answer(r) is not None for r, _ in responses)
    return {
        "accuracy": n_correct / len(responses),
        "n_docs": len(responses),
        "frac_extracted": n_extracted / len(responses),
    }
