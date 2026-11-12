"""Real-Metal regression for the confirmed F(4x4) MMA scratch consumer."""

from __future__ import annotations

import hashlib

import pytest

from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from tcenv.target import load_target

from .conv2d_mma_consumer import CONFIRMED_SHA256, RECOMPUTE_SHA256, metal_source, metal_source_recompute

pytestmark = pytest.mark.gpu


def _locator():
    found = next(
        item
        for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
        if item.op_name == "conv2d"
    )
    return locator_for_instance(found, "fp16", "paper")


def _source() -> str:
    locator = _locator()
    symbols = {str(name): int(value) for name, value in locator.load().instance.symbols.items()}
    source = metal_source(symbols)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == CONFIRMED_SHA256
    assert len(source.encode("utf-8")) < 262_144
    return source


def _recompute_source() -> str:
    locator = _locator()
    symbols = {str(name): int(value) for name, value in locator.load().instance.symbols.items()}
    source = metal_source_recompute(symbols)
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() == RECOMPUTE_SHA256
    assert len(source.encode("utf-8")) < 262_144
    return source


def _why(report: dict) -> str:
    return f"status={report.get('status')!r} message={report.get('message')!r} safety={report.get('safety')!r}"


def test_f4_mma_scratch_passes_every_case_including_ieee():
    """The fast path is independent of poison; all non-finite taps reach direct fp32."""
    report = evaluate_candidate(
        _locator(), _source(), target=load_target("apple-m4-max"), mode="quick", reps=1, seed=7
    ).result.report
    assert report["status"] == "ok", _why(report)
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
    assert all(case["ok"] for case in cases)
    assert report["safety"]["ok"] is True
    assert report["safety"]["nondeterminism"] == []


def test_f4_mma_recompute_control_passes_the_same_six_cases():
    """The A/B control changes only where U comes from, not correctness semantics."""
    report = evaluate_candidate(
        _locator(), _recompute_source(), target=load_target("apple-m4-max"), mode="quick", reps=1, seed=7
    ).result.report
    assert report["status"] == "ok", _why(report)
    cases = report["correctness"]["cases"]
    assert [case["case"] for case in cases] == ["normal", "wide_range", "large", "tiny", "outlier", "ieee"]
    assert all(case["ok"] for case in cases)
    assert report["safety"]["ok"] is True


@pytest.mark.slow
def test_f4_mma_scratch_formal_eval_has_a_real_latency_margin():
    """A full eval regression gate, not a substitute for the archived fresh confirmation."""
    report = evaluate_candidate(
        _locator(), _source(), target=load_target("apple-m4-max"), mode="full", reps=15, seed=7
    ).result.report
    assert report["status"] == "ok", _why(report)
    assert report["benchmark"]["noisy"] is False
    assert report["speedups"]["torch_mps"] > 1.10
    assert report["speedups"]["mlx"] > 1.10
    control = evaluate_candidate(
        _locator(), _recompute_source(), target=load_target("apple-m4-max"), mode="full", reps=15, seed=7
    ).result.report
    assert control["status"] == "ok", _why(control)
    assert control["benchmark"]["noisy"] is False
    assert control["candidate_us"] > 1.5 * report["candidate_us"]
