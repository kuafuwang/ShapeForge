"""Soundness tests for the real-VM container acceptance report (no Docker)."""

from __future__ import annotations

import json

from tests.support.npu_sim import container_sim_runner as runner


def _passing(report: runner.Report) -> None:
    for name in runner.REQUIRED_CHECKS:
        report.passed(name, "ok")


def test_report_cannot_pass_with_a_recorded_blocker(tmp_path):
    report = runner.Report(tmp_path)
    _passing(report)
    report.blockers.append("a check raised after recording its pass")
    document = report.write(cleanup={"final": True})
    assert document["status"] != "passed"
    assert document["status"] == "blocked"


def test_report_requires_every_required_check_and_a_final_cleanup(tmp_path):
    report = runner.Report(tmp_path)
    for name in runner.REQUIRED_CHECKS[:-1]:
        report.passed(name, "ok")
    document = report.write(cleanup={"final": True})
    assert document["status"] != "passed"
    assert document["status"] == "blocked"
    assert runner.REQUIRED_CHECKS[-1] in document["not_run"]

    report = runner.Report(tmp_path / "second")
    _passing(report)
    document = report.write(cleanup={"final": False})
    assert document["status"] == "failed"

    report = runner.Report(tmp_path / "third")
    _passing(report)
    report.failed(runner.REQUIRED_CHECKS[0], "the fault never fired")
    document = report.write(cleanup={"final": True})
    assert document["status"] == "failed"


def test_report_is_never_a_publication_or_hardware_claim(tmp_path):
    report = runner.Report(tmp_path)
    _passing(report)
    document = report.write(cleanup={"final": True})
    assert document["status"] == "passed"
    assert document["simulated"] is True
    assert document["publication_allowed"] is False
    assert document["hardware_performance_claims"] is False
    written = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert written == document


def test_report_cannot_pass_with_a_forbidden_artifact(tmp_path):
    report = runner.Report(tmp_path)
    _passing(report)
    document = report.write(
        cleanup={"final": True}, extra={"no_run_config_json": False},
    )
    assert document["status"] == "failed"


def test_report_cannot_pass_after_a_failed_evidence_command(tmp_path):
    report = runner.Report(tmp_path)
    _passing(report)
    report.blockers.append("evidence command 'posture-inspect' failed with rc 1")
    document = report.write(
        cleanup={"final": True},
        extra={"no_run_config_json": True, "no_confirmed_or_catalog_written": True},
    )
    assert document["status"] == "blocked"


def test_a_failed_check_cannot_be_overwritten_as_passed(tmp_path):
    report = runner.Report(tmp_path)
    report.failed(runner.REQUIRED_CHECKS[0], "the fault really fired")
    report.passed(runner.REQUIRED_CHECKS[0], "later optimism")
    for name in runner.REQUIRED_CHECKS[1:]:
        report.passed(name, "ok")
    document = report.write(cleanup={"final": True})
    assert document["status"] == "failed"
    assert report.checks[runner.REQUIRED_CHECKS[0]]["status"] == "failed"


def test_a_not_run_check_cannot_be_improved_to_passed(tmp_path):
    report = runner.Report(tmp_path)
    report.skip(runner.REQUIRED_CHECKS[0], "the fault did not fire")
    report.passed(runner.REQUIRED_CHECKS[0], "later optimism")
    for name in runner.REQUIRED_CHECKS[1:]:
        report.passed(name, "ok")
    document = report.write(cleanup={"final": True})
    assert document["status"] != "passed"
    assert report.checks[runner.REQUIRED_CHECKS[0]]["status"] == "not_run"


def test_ps_failure_is_an_error_not_a_visible_process(monkeypatch):
    def failing_run(argv, *, timeout=60.0):
        import subprocess
        return subprocess.CompletedProcess(argv, 1, "", "ps: Operation not permitted")

    monkeypatch.setattr(runner, "_run", failing_run)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="ps failed"):
        runner._ps_tokens("taic-token")
