"""G4 evidence levels are host-classified and simulation stays separate."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from taic import harness, trust
from taic.evidence_level import (
    FENCED_HOST_TOOL, HOST_PROCESS, HUMAN_CONFIRMED, MODEL_CLAIM, MODEL_SCRIPT,
    SIMULATED, grade_evidence,
)
from taic.overlay_route_m import judge, load_cases
from taic.paired_compare import append_host_evidence
from taic.selfcheck_bridge import SelfCheckServer


def test_five_runner_levels_and_explicit_human_confirmation():
    expected = {
        "model_statement": MODEL_CLAIM,
        "model_probe": MODEL_SCRIPT,
        "host_bridge_self_check": FENCED_HOST_TOOL,
        "outer_candidate_report": HOST_PROCESS,
    }
    for source, level in expected.items():
        grade = grade_evidence(source)
        assert grade["level"] == level
        assert grade["runner_level"] == level
        assert grade["permission_effect"] == "evidence_label_only"
    denied = grade_evidence("human_confirmation", confirmation={"confirmed_by": "person", "evidence_ref": "run/report"})
    assert denied["level"] == MODEL_CLAIM and denied["promotion_allowed"] is False
    confirmed = grade_evidence(
        "human_confirmation", host_verified=True,
        confirmation={"confirmed_by": "person", "evidence_ref": "run/report"},
    )
    assert confirmed["level"] == HUMAN_CONFIRMED
    assert confirmed["rank"] == 4 and confirmed["promotion_allowed"] is True
    workspace = grade_evidence("workspace_self_check")
    assert workspace["level"] == MODEL_SCRIPT
    assert workspace["level"] not in {FENCED_HOST_TOOL, HOST_PROCESS}


def test_simulation_is_separate_for_every_runner_and_cannot_be_cleared():
    for source in ("model_statement", "model_probe", "host_bridge_self_check", "outer_candidate_report", "host_paired_comparison"):
        grade = grade_evidence(source, record={"measurement_kind": "simulated"}, simulated=False)
        assert grade["level"] == SIMULATED
        assert grade["runner_level"] in {MODEL_CLAIM, MODEL_SCRIPT, FENCED_HOST_TOOL, HOST_PROCESS}
        assert grade["rank"] is None and grade["promotion_allowed"] is False
    human = grade_evidence(
        "human_confirmation", host_verified=True,
        confirmation={"confirmed_by": "person", "evidence_ref": "sim/report"},
        record={"simulated": True},
    )
    assert human["level"] == SIMULATED and human["runner_level"] == HUMAN_CONFIRMED
    assert human["promotion_allowed"] is False


def _premise_row(run: Path, *, simulated: bool = False, trusted_report=None, self_level=HOST_PROCESS):
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    doc = {
        "surface": "premise",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先复测这个线程组边界再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "1.0 us", "evidence_level": self_level},
        "rationale": "test",
    }
    block = "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"
    record = {
        "seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek",
        "model": "deepseek-flash", "effort": "high", "run_id": run.name,
        "simulated": simulated, "measurement_kind": "simulated" if simulated else "measured",
        "confirmed_by": "person", "evidence_level": self_level,
    }
    return harness.record_proposals(
        run, record=record, parsed={"sections": {"对系统的建议": block}},
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work,
        started=time.time() - 2, kernel_sha256="abc",
        protocol_version=trust.current_protocol_version(), trusted_report=trusted_report,
    )[0]


def _outer_report():
    return {
        "status": "ok", "candidate_us": 1.0, "speedup": 1.2,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "kernel_sha256": "abc", "build_kind": "native", "measurement_kind": "measured",
        "trusted_source": "outer candidate report",
    }


def test_model_fields_do_not_self_promote_and_outer_report_is_host_process(tmp_path):
    ordinary = _premise_row(tmp_path / "ordinary", self_level=HUMAN_CONFIRMED)
    assert ordinary["trusted_record"] is False
    assert ordinary["evidence_level"] == MODEL_SCRIPT
    assert ordinary["evidence_provenance"]["source"] == "model_probe"
    assert ordinary["evidence_provenance"]["promotion_allowed"] is False

    host = _premise_row(tmp_path / "host", trusted_report=_outer_report())
    assert host["trusted_record"] is True
    assert host["evidence_level"] == HOST_PROCESS
    assert host["evidence_provenance"]["source"] == "outer_candidate_report"
    summary = harness.board(tmp_path / "host")["evidence_levels"]
    assert summary["counts"][HOST_PROCESS] == 1
    assert summary["simulated_promotion_allowed"] is False


def test_only_a_matched_outer_report_marks_host_process_in_a_mixed_batch(tmp_path):
    run = tmp_path / "mixed"
    work = run / "agent_work" / "r1.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    scope = {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}
    documents = [
        {
            "surface": "tip", "scope": scope, "action": "先复测线程组边界再选 tile。",
            "evidence": {"probe": "probe.py", "measurement": "64 threads"},
        },
        {
            "surface": "premise", "scope": scope, "action": "未匹配的数字不能成为宿主证据。",
            "evidence": {"probe": "probe.py", "measurement": "99.0 us"},
        },
        {
            "surface": "premise", "scope": scope, "action": "匹配的数字才可成为宿主证据。",
            "evidence": {"probe": "probe.py", "measurement": "1.0 us"},
        },
        {
            "surface": "premise", "action": "缺 scope 的无效行。",
            "evidence": {"probe": "probe.py", "measurement": "1.0 us"},
        },
    ]
    sections = "\n".join(
        "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"
        for doc in documents
    )
    rows = harness.record_proposals(
        run,
        record={
            "seq": 1, "label": "r1.p0", "committee": "first", "provider": "deepseek",
            "model": "deepseek-flash", "effort": "high", "run_id": run.name,
            "measurement_kind": "measured",
        },
        parsed={"sections": {"对系统的建议": sections}},
        target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work,
        started=time.time() - 2, kernel_sha256="abc",
        protocol_version=trust.current_protocol_version(), trusted_report=_outer_report(),
    )
    assert len(rows) == 4
    tip = next(row for row in rows if row.get("proposal", {}).get("surface") == "tip")
    mismatched = next(row for row in rows if row.get("proposal", {}).get("action") == "未匹配的数字不能成为宿主证据。")
    matched = next(row for row in rows if row.get("proposal", {}).get("action") == "匹配的数字才可成为宿主证据。")
    invalid = next(row for row in rows if row["status"] == "invalid")
    assert tip["trusted_record"] is False and tip["evidence_level"] == MODEL_SCRIPT
    assert mismatched["trusted_record"] is False and mismatched["evidence_level"] == MODEL_SCRIPT
    assert mismatched["evidence_provenance"]["source"] == "model_probe"
    assert matched["trusted_record"] is True and matched["evidence_level"] == HOST_PROCESS
    assert matched["evidence_provenance"]["source"] == "outer_candidate_report"
    assert invalid["evidence_level"] != HOST_PROCESS
    assert harness.board(run)["evidence_levels"]["counts"][HOST_PROCESS] == 1


def test_host_premise_overlay_preserves_host_process_level(tmp_path):
    run = tmp_path / "host-overlay"
    proposal = _premise_row(run, trusted_report=_outer_report())
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
    assert len(result["shadow"]) == 1
    event = result["shadow"][0]
    assert proposal["evidence_level"] == HOST_PROCESS
    assert event["evidence_level"] == HOST_PROCESS
    assert event["evidence_provenance"]["source"] == "outer_candidate_report"


def test_simulated_run_stays_simulated_even_with_an_outer_report(tmp_path):
    run = tmp_path / "sim"
    row = _premise_row(run, simulated=True, trusted_report=_outer_report())
    assert row["trusted_record"] is False
    assert row["evidence_level"] == SIMULATED
    assert row["evidence_provenance"]["runner_level"] == MODEL_SCRIPT
    assert row["evidence_provenance"]["rank"] is None
    assert row["evidence_provenance"]["promotion_allowed"] is False
    root = tmp_path / "targets"
    trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high",
        scopes=["overlay:premise"], root=root,
    )
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=True,
        shadow=False, trust_root=root,
    )
    assert result["applied"] == [] and result["shadow"] == []
    assert not (run / harness.OVERLAY_FILE).exists()
    assert harness.overlay_text(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(),
    ) == ""


def test_simulated_outer_report_cannot_fall_back_to_a_measured_bridge_row(monkeypatch, tmp_path):
    authority_row = _outer_report()
    authority_row.pop("trusted_source")
    authority = tmp_path / "authority.jsonl"
    authority.write_text(json.dumps(authority_row) + "\n")
    monkeypatch.setattr("tcenv.selfcheck.authority_log", lambda _directory: authority)
    run = tmp_path / "sim-report"
    report = _outer_report()
    report["measurement_kind"] = "simulated"
    row = _premise_row(run, simulated=False, trusted_report=report)
    assert row["trusted_record"] is False
    assert row["evidence_level"] == SIMULATED
    assert row["evidence_provenance"]["runner_level"] == MODEL_SCRIPT
    assert row["evidence_provenance"]["rank"] is None
    assert row["evidence_provenance"]["promotion_allowed"] is False
    root = tmp_path / "targets"
    trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high",
        scopes=["overlay:premise"], root=root,
    )
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(), enabled=True,
        shadow=False, trust_root=root,
    )
    assert result["applied"] == [] and result["shadow"] == []
    assert harness.overlay_text(
        run, target_id="apple-m4-max", task="conv2d",
        protocol_version=trust.current_protocol_version(),
    ) == ""


def test_bridge_authority_row_names_fenced_tool_and_simulation(monkeypatch, tmp_path):
    captured = []

    class FakeReport:
        def to_json(self):
            return {
                "status": "ok", "elapsed_s": 0.1, "candidate_us": 1.0, "speedup": 1.2,
                "benchmark": {"candidate": {"cv": 0.01}}, "build": {"build_kind": "native"},
            }

    monkeypatch.setattr("taic.selfcheck_bridge.EvalReport", FakeReport)
    monkeypatch.setattr("tcenv.selfcheck.append", lambda _directory, row: captured.append(row))
    for kind, level in (("measured", FENCED_HOST_TOOL), ("simulated", SIMULATED)):
        bridge = object.__new__(SelfCheckServer)
        bridge._usage = {"count": 0, "gpu_s": 0.0, "hang": False}
        bridge._save_usage = lambda: None
        bridge.config = SimpleNamespace(self_check_dir=tmp_path)
        request = SimpleNamespace(
            mode="timed", task="conv2d", dtype="fp16", size="small", target_id="apple-m4-max",
            source="kernel void k() {}", request_id="req", request_digest="digest",
        )
        result = SimpleNamespace(
            report=FakeReport(), receipt={"receipt_id": "receipt"},
            provenance=SimpleNamespace(measurement_kind=kind),
        )
        bridge._record_usage(request, result)
    assert captured[0]["evidence_level"] == FENCED_HOST_TOOL
    assert captured[0]["evidence_provenance"]["source"] == "host_bridge_self_check"
    assert captured[1]["evidence_level"] == SIMULATED
    assert captured[1]["evidence_provenance"]["runner_level"] == FENCED_HOST_TOOL


def test_host_paired_writer_overrides_self_level_and_marks_simulation(monkeypatch, tmp_path):
    path = tmp_path / "host.jsonl"
    monkeypatch.setattr("taic.paired_compare.host_evidence_path", lambda _directory: path)
    for suffix, kind, level in (("a", "measured", HOST_PROCESS), ("b", "simulated", SIMULATED)):
        append_host_evidence(tmp_path, {
            "applied": True, "sha_a": "1" * 64, "sha_b": "2" * 64,
            "measurement_kind": kind, "evidence_level": MODEL_CLAIM,
        })
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["evidence_level"] for row in rows] == [HOST_PROCESS, SIMULATED]
    assert rows[1]["evidence_provenance"]["runner_level"] == HOST_PROCESS
    assert rows[1]["evidence_provenance"]["promotion_allowed"] is False


def _case(qualified: bool, *, simulated=False, nomination=False):
    return {
        "evidence_id": "host:abc" if qualified else "nomination:abc",
        "evidence_role": "nomination" if nomination else "host_paired_comparison",
        "evidence_path": "path", "session_label": "r1.p0", "author": None,
        "qualified": qualified, "qualify_reason": "model nomination is not host evidence",
        "sha_pair": ["1" * 64, "2" * 64], "conclusion": "too_noisy" if qualified else "",
        "direction": "x", "task": "conv2d", "chip": "apple-m4-max", "backend": "metal",
        "simulated": simulated, "attribution": {"kind": "unknown", "only_a": [], "only_b": []},
        "model_claim_raw": "model claim" if nomination else "", "model_conclusion": "",
    }


def test_route_m_trace_levels_separate_host_nomination_and_simulation():
    host = judge(
        _case(True), target_id="apple-m4-max", task="conv2d", protocol_version="4",
        trust_root=None, prompt_sha1=None, backend="metal", run_name="run",
    )
    assert host["evidence_level"] == HOST_PROCESS
    nomination = judge(
        _case(False, nomination=True), target_id="apple-m4-max", task="conv2d", protocol_version="4",
        trust_root=None, prompt_sha1=None, backend="metal", run_name="run",
    )
    assert nomination["evidence_level"] == MODEL_SCRIPT
    simulated = judge(
        _case(True, simulated=True), target_id="apple-m4-max", task="conv2d", protocol_version="4",
        trust_root=None, prompt_sha1=None, backend="metal", run_name="run",
    )
    assert simulated["evidence_level"] == SIMULATED
    assert simulated["evidence_provenance"]["runner_level"] == HOST_PROCESS


def test_workspace_nomination_cannot_claim_fenced_tool_and_inherits_simulation(tmp_path):
    for simulated in (False, True):
        run = tmp_path / ("nom-sim" if simulated else "nom-measured")
        work = run / "agent_work" / "r1.p0" / ".selfcheck"
        work.mkdir(parents=True)
        (work / "paired_comparisons.jsonl").write_text(json.dumps({
            "sha_a": "1" * 64, "sha_b": "2" * 64, "claim": "model nomination",
            "evidence_level": HOST_PROCESS, "host_evidence": True,
        }) + "\n")
        harness._append_jsonl(run / "feedback.jsonl", {
            "label": "r1.p0", "seq": 1, "provider": "deepseek", "model": "deepseek-flash",
            "effort": "high", "simulated": simulated,
            "measurement_kind": "simulated" if simulated else "measured",
        })
        case = next(row for row in load_cases(run) if row.get("evidence_role") == "nomination")
        trace = judge(
            case, target_id="apple-m4-max", task="conv2d", protocol_version="4",
            trust_root=None, prompt_sha1=None, backend="metal", run_name=run.name,
        )
        assert trace["evidence_level"] == (SIMULATED if simulated else MODEL_SCRIPT)
        assert trace["evidence_provenance"]["runner_level"] == MODEL_SCRIPT
        assert trace["evidence_provenance"]["promotion_allowed"] is False
