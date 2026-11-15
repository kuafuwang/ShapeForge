"""Scratch bundle checks.  Two nodes never launch a device evaluation.

``test_registered_kda_scratch_rows_are_four_wide`` only checks ``scratch_texture_layout``.
``test_a_single_read_write_poison_buffer_is_still_refused`` is ``preflight_submission``.
The other three call ``evaluate_candidate``: texture ping-pong, a host program that returns 1,
and the P2-Q device-pointer ping-pong.  All five stay gpu-marked because this module imports
the Metal backend; that mark is not five device runs.
"""

from __future__ import annotations

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.sanitize import GUARD_BYTES
from tcenv.backends.metal.scratch_texture import scratch_texture_layout
from tcenv.backends.metal.search_candidate import evaluate_candidate, preflight_submission
from tcenv.taskcontract import load_task_contract
from tcenv.target import load_target

from .bundle_scratch_seq import SCRATCH_ROW, nonzero_host_source, texture_source
from ...support.seq_handoff import scratch_sources, pingpong_source

pytestmark = pytest.mark.gpu

TARGET = load_target("apple-m4-max")
CASES = ("normal", "wide_range", "outlier")


def _why(report: dict) -> str:
    parts = [f"status={report.get('status')!r}"]
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:800]}")
    violations = (report.get("safety") or {}).get("violations") or []
    if violations:
        parts.append("safety: " + "; ".join(str(item) for item in violations[:4]))
    abi = (report.get("abi") or {}).get("violations") or []
    if abi:
        parts.append("abi: " + "; ".join(str(item) for item in abi[:2]))
    cases = (report.get("correctness") or {}).get("cases") or []
    bad = [item for item in cases if not item.get("ok")]
    if bad:
        parts.append("cases: " + "; ".join(f"{item.get('case')}: {item.get('message')}" for item in bad[:2]))
    return " | ".join(parts)


def _evaluate(source: str, *, cases: tuple[str, ...] | None = None):
    return evaluate_candidate(
        bind_task("kda", "fp16", "small").locator,
        source,
        target=TARGET,
        mode="quick",
        reps=1,
        cases=cases,
        track_id="msl-bundle",
    )


def test_registered_kda_scratch_rows_are_four_wide():
    """The kernel hardcodes the width ``scratch_texture_layout`` returns for every kda state."""
    for size in ("single", "small", "tail", "paper"):
        symbols = load_task_contract("tasks/kda", dtype="fp16", size=size).instance.symbols
        count = int(symbols["H"]) * int(symbols["D"]) * int(symbols["D"])
        width, _height, _row_bytes = scratch_texture_layout(count, guard_bytes=GUARD_BYTES)
        assert width == SCRATCH_ROW, (size, count, width)


def test_a_single_read_write_poison_buffer_is_still_refused():
    outcome = preflight_submission(
        bind_task("kda", "fp16", "small").contract,
        scratch_sources()["uninitialized_poison"],
        target=TARGET,
        track_id="msl-bundle",
    )
    assert not outcome.ok
    assert "first used as read_write" in outcome.rejection
    assert "write-only" in outcome.rejection
    assert "different resolved geometry" not in outcome.rejection


def test_texture_pingpong_executes_every_declared_case():
    """Legal poison ping-pong: host returns 0, all three cases match, guards and coverage stay clean."""
    run = _evaluate(texture_source())
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    assert [case["case"] for case in report["correctness"]["cases"]] == list(CASES)
    assert all(case["ok"] for case in report["correctness"]["cases"]), _why(report)
    assert report["safety"]["ok"] is True, _why(report)
    assert (report["safety"].get("violations") or []) == []
    assert (report["safety"].get("nondeterminism") or []) == []
    assert [item["kernel"] for item in run.evaluation["dispatch"]] == ["kda_piece0", "kda_piece1", "kda_piece2"]
    host_runs = run.evaluation["host_runs"]
    assert host_runs
    assert all(int(item["host_status"]) == 0 for item in host_runs)


def test_a_nonzero_host_program_is_incorrect_and_is_not_an_identity_mismatch():
    run = _evaluate(nonzero_host_source(), cases=("normal",))
    report = run.result.report
    detail = _why(report)
    assert report["status"] == "incorrect", detail
    assert "host_tiling returned 1" in detail
    assert "host program failed" in detail
    assert "different resolved geometry" not in detail
    assert any(int(item["host_status"]) == 1 for item in run.evaluation["host_runs"])


def test_device_pointer_pingpong_fails_reflection_after_identity_matches():
    """The P2-Q candidate now gets past the worker identity check and stops on readwrite reflection."""
    run = _evaluate(pingpong_source(), cases=("normal",))
    report = run.result.report
    detail = _why(report)
    assert report["status"] == "abi_error", detail
    assert "different resolved geometry" not in detail
    assert "reflects readwrite" in detail
    assert "device pointer reflects readwrite" in detail
