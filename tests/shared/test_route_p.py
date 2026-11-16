"""G2 route P: local premise activation matches only the outer candidate report."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from taic import harness, trust
from taic.evidence_level import FENCED_HOST_TOOL
from taic.harness_trace import read_traces


def _row(**overrides):
    row = {
        "status": "ok", "candidate_us": 1.0, "speedup": 1.2,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "kernel_sha256": "abc", "build_kind": "native", "measurement_kind": "measured",
    }
    row.update(overrides)
    return row


def _outer(**overrides):
    row = _row(trusted_source="outer candidate report")
    row.update(overrides)
    return row


def _check(work: Path, report=None):
    return harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="abc", measurement="1.0 us", trusted_report=report,
    )


def test_session_template_states_the_route_p_local_boundary():
    assert "ordinary local mode, only the outer candidate report can match" in harness.TEMPLATE
    assert (
        "``.selfcheck/self_checks.jsonl`` and ``self_checks.jsonl`` are model-writable mirrors"
        in harness.TEMPLATE
    )
    assert "``fenced_host_tool`` evidence, not outer ``host_process`` evidence" in harness.TEMPLATE


def test_local_workspace_mirrors_do_not_match_but_the_outer_report_does(monkeypatch, tmp_path):
    work = tmp_path / "agent_work" / "r1.p0"
    (work / ".selfcheck").mkdir(parents=True)
    payload = json.dumps(_row()) + "\n"
    (work / ".selfcheck" / "self_checks.jsonl").write_text(payload)
    (work / "self_checks.jsonl").write_text(payload)
    monkeypatch.setattr(
        "tcenv.selfcheck.authority_log",
        lambda _directory: tmp_path / "host-only" / "self_checks.jsonl",
    )
    ok, why = _check(work)
    assert ok is False and "no matching" in why
    ok, why = _check(work, _outer())
    assert ok is True and why == "outer candidate report"


@pytest.mark.parametrize(
    "change",
    [
        {"target_id": "other"},
        {"protocol_version": "other"},
        {"kernel_sha256": "other"},
        {"candidate_us": 9.0, "speedup": 9.0},
        {"build_kind": "source"},
        {"measurement_kind": "simulated"},
        {"simulated": True},
        {"status": "incorrect"},
    ],
)
def test_outer_report_binding_rules_stay_closed(tmp_path, change):
    work = tmp_path / "work"
    work.mkdir()
    ok, _why = _check(work, _outer(**change))
    assert ok is False


def test_host_bridge_authority_row_is_a_distinct_fenced_mode(monkeypatch, tmp_path):
    work = tmp_path / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    authority = tmp_path / "host" / "self_checks.jsonl"
    authority.parent.mkdir()
    authority.write_text(json.dumps(_row()) + "\n")
    monkeypatch.setattr("tcenv.selfcheck.authority_log", lambda _directory: authority)
    ok, why = _check(work)
    assert ok is True and why.startswith("self_checks.jsonl")

    (work / "probe.py").write_text("print(1)\n")
    doc = {
        "surface": "premise",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先复测这个边界再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "1.0 us"},
    }
    block = "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"
    row = harness.record_proposals(
        tmp_path / "run", record={
            "seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek",
            "model": "deepseek-flash", "effort": "high", "run_id": "run",
        }, parsed={"sections": {"对系统的建议": block}},
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work,
        started=time.time() - 2, kernel_sha256="abc", protocol_version=trust.current_protocol_version(),
    )[0]
    assert row["trusted_record"] is True
    assert row["evidence_level"] == FENCED_HOST_TOOL


def test_settle_consumes_the_persisted_route_p_gate_instead_of_recomputing_it(monkeypatch, tmp_path):
    run = tmp_path / "run"
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    authority = tmp_path / "host" / "self_checks.jsonl"
    monkeypatch.setattr("tcenv.selfcheck.authority_log", lambda _directory: authority)
    doc = {
        "surface": "premise",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先复测这个边界再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "1.0 us"},
    }
    block = "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"
    row = harness.record_proposals(
        run, record={
            "seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek",
            "model": "deepseek-flash", "effort": "high", "run_id": run.name,
        }, parsed={"sections": {"对系统的建议": block}},
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work,
        started=time.time() - 2, kernel_sha256="abc", protocol_version=trust.current_protocol_version(),
    )[0]
    assert row["trusted_record"] is False
    # A later bridge row can be read directly, but settle does not rewrite the old proposal's bit.
    authority.parent.mkdir()
    authority.write_text(json.dumps(_row()) + "\n")
    assert _check(work)[0] is True
    root = tmp_path / "targets"
    trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high",
        scopes=["overlay:premise"], root=root,
    )
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=True,
        shadow=True, trust_root=root,
    )
    assert result["shadow"] == []
    trace = read_traces(run)[-1]
    assert trace["route"] == "P"
    assert trace["gates"][3]["status"] == "fail"
    assert "no matching" in trace["gates"][3]["reason"]


def test_route_m_paired_document_is_not_automatically_route_p_report(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    paired = {
        "schema": "taic.paired_comparison.1", "applied": True, "host_evidence": True,
        "status": "ok", "kernel_sha256": "abc", "trusted_source": "host paired comparison",
        "sha_a": "1" * 64, "sha_b": "2" * 64, "candidate_us": 1.0,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "build_kind": "native", "measurement_kind": "measured",
    }
    ok, _why = _check(work, paired)
    assert ok is False
    sha_only = dict(paired)
    sha_only.pop("schema")
    sha_only.pop("host_evidence")
    ok, _why = _check(work, sha_only)
    assert ok is False
    mislabeled = dict(paired)
    mislabeled.pop("schema")
    mislabeled["trusted_source"] = "outer candidate report"
    ok, _why = _check(work, mislabeled)
    assert ok is False
