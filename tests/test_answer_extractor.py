"""Robust GSM8K answer extractor (DECISIONS D-15). Pure Python, nothing downloaded."""

import pytest

from lagate.eval.answer import extract_answer, gsm8k_target, is_correct, normalize_number, score_samples

# Final lines of the five protocol-B outputs that strict-match rejected
# (run 20261106-063044_gsm8k_chat_dense_qwen_s0, docs 0, 1, 2, 3, 10). All five are correct answers.
FAILING_PROTOCOL_B = {
    0: (
        "Finally, calculate the earnings from selling these eggs:\n9 * $2/egg = $18/day.\n\n"
        "Therefore, Janet makes $18 every day at the farmers' market.",
        "#### 18",
    ),
    1: (
        "\\[ \\text{Total bolts} = \\text{Blue fiber} + \\text{White fiber} = 2 + 1 = 3 \\text{ bolts} \\]\n\n"
        "Therefore, the total number of bolts of fabric needed is \\( \\boxed{3} \\).",
        "#### 3",
    ),
    2: (
        "   \\[\n   \\text{New Value} = \\$80,000 + \\$120,000 = \\$200,000\n   \\]\n\n"
        "4. **Find the profit by subtracting the total cost from the new value:**\n\n"
        "   \\[\n   \\text{Profit} = \\text{New Value} - \\text{Total Cost} = \\$200,000 - \\$130,000 = \\$70,000\n   \\]\n\n"
        "Therefore, the profit Josh made from flipping the house is $\\boxed{70000}$.",
        "#### 70000",
    ),
    3: (
        "\\[ \\text{Total distance per week} = 180 \\, \\text{meters/session} \\times 3 \\, \\text{sessions/week} \\]\n"
        "\\[ \\text{Total distance per week} = 540 \\, \\text{meters} \\]\n\n"
        "Therefore, the total distance James runs in a week is \\(540\\) meters.",
        "#### 540",
    ),
    10: (
        "   - Calculation: \\( 180 \\times (1 - 0.30) = 180 \\times 0.70 = 126 \\) downloads\n\n"
        "Now, let's sum up all the downloads:\n\n- Total downloads: \\( 60 + 180 + 126 = 366 \\)\n\n"
        "Therefore, the total number of downloads over the three months is **366**.",
        "#### 366",
    ),
}


@pytest.mark.parametrize("doc_id", sorted(FAILING_PROTOCOL_B))
def test_scores_failing_protocol_b_outputs_correctly(doc_id):
    response, gold = FAILING_PROTOCOL_B[doc_id]
    assert is_correct(response, gold), (doc_id, extract_answer(response), gsm8k_target(gold))


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Step 1 ... 3 + 4 = 7\n#### 7", "7"),                      # strict format
        ("#### 1,234", "1234"),                                     # thousands separator
        ("#### $5", "5"),                                           # dollar sign
        ("2 + 3 = 5\n#### 5\nAlso \\boxed{9}", "5"),                # '####' wins over other markers
        ("The final answer is 120000.", "120000"),                  # D-08 style, trailing period
        ("The final answer is $120,000.", "120000"),
        ("so x = 4. The answer is 14. You are an AI assistant. 20 - 6 = 99", "14"),  # run-on text ignored
        ("the result is $\\boxed{70000}$.", "70000"),
        ("\\boxed{\\text{42}}", "42"),
        ("Step **1**: 2 + 2 = 4. Total is **12**.", "12"),          # last bold number
        ("It costs 3.50 dollars, so 2 cost 7.00", "7"),             # fallback: last number, 7.00 -> 7
        ("A 25% discount gives -3.5", "-3.5"),
        ("I don't know.", None),
    ],
)
def test_extract_answer(text, expected):
    assert extract_answer(text) == expected


def test_normalize_number():
    assert normalize_number("$1,234.50") == "1234.5"
    assert normalize_number("18.") == "18"
    assert normalize_number("0.30") == "0.3"
    assert normalize_number("abc") is None


def test_score_samples_merges_filter_duplicates():
    doc = {"answer": "... #### 18"}
    samples = [
        {"doc_id": 0, "doc": doc, "resps": [["Janet makes $18 every day."]], "filter": "strict-match"},
        {"doc_id": 0, "doc": doc, "resps": [["Janet makes $18 every day."]], "filter": "flexible-extract"},
        {"doc_id": 1, "doc": {"answer": "#### 3"}, "resps": [["no idea"]], "filter": "strict-match"},
    ]
    assert score_samples(samples) == {"accuracy": 0.5, "n_docs": 2, "frac_extracted": 0.5}
