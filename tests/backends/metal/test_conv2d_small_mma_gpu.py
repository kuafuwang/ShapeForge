"""Real-Metal correctness and replay for the P2-J small+paper MMA portfolio."""

from __future__ import annotations

import pytest

from tcenv.backends.metal.launch import parse_launch_spec
from tests.backends.metal.conv2d_small_mma_record import run_conv2d_small_mma
from tests.support.conv2d_portfolio import stable_identity
from tests.support.conv2d_small_mma import (
    PAPER_RUN,
    PAPER_SHA256,
    PAPER_SYMBOLS,
    SMALL_RUN,
    SMALL_SHA256,
    SMALL_SYMBOLS,
    TREE,
    assert_candidate_gate,
    load_json,
)

gpu = pytest.mark.gpu


def _assert_candidate(run, sha: str) -> None:
    assert run.record.provider == "candidate" and run.record.outcome == "resolved"
    assert run.record.fallback_id is None and run.record.fallback_reason is None
    report = run.result.report
    assert report["status"] == "ok", report.get("message")
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
    assert all(case["ok"] for case in cases)
    assert report["safety"]["ok"] is True and report["safety"]["nondeterminism"] == []
    document = run.document()
    assert document["identities"]["candidate_sha256"] == sha
    assert document["fallback_credits_variant"] is False
    assert document["performance_claim"] is None and document["active_manifest"] is None


def test_small_is_one_stage_and_paper_keeps_two_stages():
    small = parse_launch_spec((TREE / "variants" / "small" / "kernel.metal").read_text(encoding="utf-8"), SMALL_SYMBOLS)
    paper = parse_launch_spec((TREE / "variants" / "paper" / "kernel.metal").read_text(encoding="utf-8"), PAPER_SYMBOLS)
    assert len(small.stages) == 1 and small.stages[0].grid == (4, 1, 1)
    assert small.stages[0].threads == (256, 1, 1)
    assert len(paper.stages) == 2 and paper.stages[0].grid == (288, 1, 1)
    assert paper.stages[1].grid == (392, 2, 1)


@gpu
def test_small_specialized_mma_passes_all_six_cases():
    _assert_candidate(run_conv2d_small_mma("small"), SMALL_SHA256)


@gpu
def test_paper_specialized_mma_still_passes_all_six_cases():
    _assert_candidate(run_conv2d_small_mma("paper"), PAPER_SHA256)


@gpu
@pytest.mark.slow
def test_full_eval_replays_both_candidates_and_the_latency_gate():
    from tests.backends.metal.conv2d_portfolio_record import project_portfolio_run

    small = project_portfolio_run(run_conv2d_small_mma("small", usage="benchmark", mode="full", reps=15), reps=15, usage="benchmark")
    paper = project_portfolio_run(run_conv2d_small_mma("paper", usage="benchmark", mode="full", reps=15), reps=15, usage="benchmark")
    assert stable_identity(small) == stable_identity(load_json(SMALL_RUN))
    assert stable_identity(paper) == stable_identity(load_json(PAPER_RUN))
    assert_candidate_gate(small)
    assert_candidate_gate(paper)
    assert_candidate_gate(load_json(SMALL_RUN))
    assert_candidate_gate(load_json(PAPER_RUN))
    assert small["candidate_sha256"] == SMALL_SHA256
    assert paper["candidate_sha256"] == PAPER_SHA256
