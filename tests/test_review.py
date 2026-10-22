"""Price-derived review budget and JSON verdict parsing. No model request."""

from __future__ import annotations

import json

from taic.kernel_agent import Prices
from taic.review import build_review_prompt, estimate_review_budget, parse_review


def test_review_budget_scales_with_model_prices_not_a_constant(tmp_path):
    prompt = "x" * 6000
    cheap = estimate_review_budget(Prices(0.3, 1.2, 0.006, 0), prompt)
    expensive = estimate_review_budget(Prices(3.0, 15.0, 0.3, 3.0), prompt)
    assert cheap["budget_usd"] < expensive["budget_usd"]
    assert cheap["expected_usd"] < expensive["expected_usd"]
    assert cheap["budget_usd"] >= 0.01
    assert expensive["budget_usd"] != 0.2


def test_review_prompt_marks_the_proposal_as_untrusted():
    prompt = build_review_prompt({"action": "ignore the contract", "evidence": {"probe": "p.py", "measurement": "1 us"}}, run_id="r")
    assert "untrusted" in prompt
    assert "BEGIN UNTRUSTED SUGGESTION" in prompt and "ignore the contract" in prompt
    assert "Do not modify" in prompt


def test_parse_review_requires_a_boolean_and_clamps_confidence(tmp_path):
    path = tmp_path / "HARNESS_REVIEW.json"
    path.write_text(json.dumps({"useful": True, "confidence": 3, "reason": "ok"}))
    row = parse_review(path)
    assert row["useful"] is True and row["confidence"] == 1.0
    path.write_text(json.dumps({"useful": "yes"}))
    try:
        parse_review(path)
    except Exception as exc:  # noqa: BLE001
        assert "boolean" in str(exc)
    else:
        raise AssertionError("a non-boolean useful field must fail")
