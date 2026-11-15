"""P2-N: the final small experiment is preserved as a negative result, not adopted."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P2N = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-N"


def test_weight_cache_is_correct_but_rejected_and_current_candidate_is_unchanged():
    decision = json.loads((P2N / "evidence/decision.json").read_text())
    assert decision["adopt_candidate"] is False
    assert decision["stop_small_optimization"] is True
    assert decision["publication_allowed"] is False
    assert decision["decision"] == "stop"
    assert decision["not_money_driven"] is True
    assert "not driven by money or lack of budget" in decision["cost_rationale"]
    assert "diminishing and unstable technical returns" in decision["cost_rationale"]
    assert "not a single lucky run" in decision["reliability_rationale"]
    assert "repeated independent fresh execution" in decision["reliability_rationale"]
    assert "one-off win is not production evidence" in decision["reliability_rationale"]
    source = (P2N / "evidence/weight-cache.metal").read_bytes()
    assert hashlib.sha256(source).hexdigest() == decision["experimental_candidate_sha256"]
    report = json.loads((P2N / "evidence/weight-cache-report.json").read_text())
    assert report["status"] == "ok" and report["safety"]["ok"] is True
    assert report["build"]["candidate_sha256"] == decision["experimental_candidate_sha256"]
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
    assert all(case["ok"] for case in cases)
    assert report["candidate_us"] > 1.05 * decision["latest_portfolio_fresh_us"]
    assert decision["slowdown_vs_latest_pct"] > 5.0
    current = (ROOT / "tests/support/golden/metal/conv2d_small_mma/variants/small/kernel.metal").read_bytes()
    assert hashlib.sha256(current).hexdigest() == decision["baseline_candidate_sha256"]
    assert hashlib.sha256(current).hexdigest() != decision["experimental_candidate_sha256"]
