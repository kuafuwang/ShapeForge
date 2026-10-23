"""Structured proposals, two-source quorum, shadow/active overlay, and P2 patches. Offline only."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

from taic import harness, trust


def _doc(**overrides):
    doc = {
        "surface": "tip",
        "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"},
        "action": "先读 device.md 的线程组上限再选 tile。",
        "evidence": {"probe": "probe.py", "measurement": "64 threads"},
        "rationale": "test",
    }
    doc.update(overrides)
    return doc


def _block(doc) -> str:
    return "```harness-proposal\n" + json.dumps(doc, ensure_ascii=False) + "\n```\n"


def _session(root: Path, *, seq: int, provider: str, model: str, committee: str, doc, started: float) -> Path:
    work = root / "agent_work" / f"r{seq}.p0"
    work.mkdir(parents=True)
    (work / "probe.py").write_text("print(1)\n")
    return work


def _record(seq: int, provider: str, model: str, committee: str) -> dict:
    return {
        "seq": seq, "label": f"r{seq}.p0", "committee": committee,
        "provider": provider, "model": model, "effort": "high", "run_id": "run-1",
    }


def _record_proposal(run: Path, work: Path, record: dict, doc, started: float, *, kernel_sha256: str = ""):
    parsed = {"sections": {"对系统的建议": _block(doc)}}
    return harness.record_proposals(
        run, record=record, parsed=parsed, target_id="apple-m4-max", task="conv2d",
        dtype="fp16", work=work, started=started, kernel_sha256=kernel_sha256,
        protocol_version=trust.current_protocol_version(),
    )


def _trusted_row(work: Path, *, kernel_sha256: str = "abc", candidate_us: float = 1.0) -> None:
    folder = work / ".selfcheck"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "self_checks.jsonl").write_text(json.dumps({
        "status": "ok", "candidate_us": candidate_us, "speedup": 1.0, "elapsed_s": 0.1,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "kernel_sha256": kernel_sha256,
    }) + "\n")


def test_a_structured_tip_needs_two_trusted_committees_and_shadow_does_not_inject(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    a = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    b = _session(run, seq=2, provider="moonshot", model="kimi-k3", committee="second", doc=_doc(), started=started)
    (b / "probe.py").write_text("print(2)\n")
    first = _record_proposal(run, a, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)
    second = _record_proposal(run, b, _record(2, "moonshot", "kimi-k3", "second"), _doc(), started)
    assert first[0]["status"] == "valid" and second[0]["proposal_key"] == first[0]["proposal_key"]
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=tmp_path / "targets")
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=tmp_path / "targets")
    shadow = harness.settle(
        run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
        enabled=True, shadow=True, trust_root=tmp_path / "targets",
    )
    assert len(shadow["shadow"]) == 1
    assert harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()) == ""
    active = harness.settle(
        run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
        enabled=True, shadow=False, trust_root=tmp_path / "targets",
    )
    assert len(active["applied"]) == 1
    text = harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())
    assert "先读 device.md" in text and "未验证" in text
    # Another target or protocol cannot see this overlay.
    assert harness.overlay_text(run, target_id="other-target", task="conv2d", protocol_version=trust.current_protocol_version()) == ""
    assert harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version="2") == ""


def test_quorum_uses_only_qualifying_model_rows_for_the_overlay_text(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    a = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    b = _session(run, seq=2, provider="moonshot", model="kimi-k3", committee="second", doc=_doc(), started=started)
    (b / "probe.py").write_text("print(2)\n")
    first = _record_proposal(run, a, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)[0]
    _record_proposal(run, b, _record(2, "moonshot", "kimi-k3", "second"), _doc(), started)
    proposals = run / "harness_proposals.jsonl"
    original = proposals.read_text()
    converted = {**first, "seq": 0, "origin": "conversion", "provider": "", "model": "", "committee": "",
                 "proposal": {**first["proposal"], "action": "CONVERSION WORDING"}}
    proposals.write_text(json.dumps(converted, ensure_ascii=False) + "\n" + original)
    root = tmp_path / "targets"
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=root)
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=root)
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
                            enabled=True, shadow=False, trust_root=root)
    assert result["applied"][0]["text"] == first["proposal"]["action"]
    assert result["applied"][0]["source_seq"] == first["seq"]


def test_one_model_or_an_untrusted_second_model_does_not_reach_quorum(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    a = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    b = _session(run, seq=2, provider="moonshot", model="kimi-k3", committee="second", doc=_doc(), started=started)
    (b / "probe.py").write_text("print(2)\n")
    _record_proposal(run, a, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)
    _record_proposal(run, b, _record(2, "moonshot", "kimi-k3", "second"), _doc(), started)
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=tmp_path / "targets")
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True, trust_root=tmp_path / "targets")
    assert result["shadow"] == []
    # A single model, repeated in two sessions, is still one source.
    run2 = tmp_path / "run2"
    a2 = _session(run2, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    b2 = _session(run2, seq=2, provider="deepseek", model="deepseek-flash", committee="second", doc=_doc(), started=started)
    _record_proposal(run2, a2, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)
    _record_proposal(run2, b2, _record(2, "deepseek", "deepseek-flash", "second"), _doc(), started)
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=tmp_path / "targets")
    assert harness.settle(run2, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True, trust_root=tmp_path / "targets")["shadow"] == []


def test_a_premise_needs_a_fresh_local_probe_and_does_not_need_a_vote(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(surface="premise"), started=started)
    _trusted_row(work)
    premise = _doc(surface="premise", evidence={"probe": "probe.py", "measurement": "1.0 us"})
    row = _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), premise, started, kernel_sha256="abc")
    assert row[0]["status"] == "valid" and row[0]["trusted_record"] is True
    # A premise still needs a person to delegate the premise surface.
    assert harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True, trust_root=tmp_path / "targets")["shadow"] == []
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", scopes=["overlay:premise"], root=tmp_path / "targets")
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True, trust_root=tmp_path / "targets")
    assert len(result["shadow"]) == 1
    # With the trusted self-check row and active trust, the premise may now be active.
    active = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=tmp_path / "targets")
    assert active["applied"] and "先读 device.md" in harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())


def test_probe_escape_stale_probe_and_wrong_scope_are_invalid(tmp_path):
    run = tmp_path / "run"
    started = time.time()
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    parsed = {"sections": {"对系统的建议": _block(_doc(scope={"target_id": "other", "task": "conv2d"}))}}
    rows = harness.record_proposals(run, record=_record(1, "deepseek", "deepseek-flash", "first"), parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=started)
    assert rows[0]["status"] == "invalid" and "target" in rows[0]["reason"]
    parsed = {"sections": {"对系统的建议": _block(_doc(evidence={"probe": "../../escape.py", "measurement": "x"}))}}
    rows = harness.record_proposals(run, record=_record(2, "deepseek", "deepseek-flash", "first"), parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=started)
    assert rows[0]["status"] == "invalid" and "escapes" in rows[0]["reason"]
    old = work / "old.py"
    old.write_text("print(1)\n")
    old.touch()
    parsed = {"sections": {"对系统的建议": _block(_doc(evidence={"probe": "old.py", "measurement": "x"}))}}
    rows = harness.record_proposals(run, record=_record(3, "deepseek", "deepseek-flash", "first"), parsed=parsed, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=work, started=time.time() + 10)
    assert rows[0]["status"] == "invalid" and "predates" in rows[0]["reason"]


def test_a_control_disable_hides_an_active_overlay_and_a_p2_patch_is_generated(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(surface="premise"), started=started)
    _trusted_row(work)
    premise = _doc(surface="premise", evidence={"probe": "probe.py", "measurement": "1.0 us"})
    _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), premise, started, kernel_sha256="abc")
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", scopes=["overlay:premise"], root=tmp_path / "targets")
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=tmp_path / "targets")
    assert result["applied"]
    assert "先读 device.md" in harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())
    harness.set_overlay_control(run, disabled=True, by="alice", reason="pilot pause")
    assert harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version()) == ""
    root = tmp_path / "repo"
    (root / "targets" / "apple-m4-max").mkdir(parents=True)
    (root / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n")
    key = result["applied"][0]["proposal_key"]
    meta = harness.write_patch(run, target_id="apple-m4-max", proposal_key=key, root=root)
    patch = Path(meta["patch"]).read_text()
    assert "targets/apple-m4-max/experience.md" in patch and "+- [harness" in patch


def test_a_p2_patch_passes_git_apply_check_and_changes_only_the_target_file(tmp_path):
    import subprocess

    run = tmp_path / "run"
    started = time.time() - 1
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(surface="premise"), started=started)
    _trusted_row(work)
    premise = _doc(surface="premise", evidence={"probe": "probe.py", "measurement": "1.0 us"})
    _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), premise, started, kernel_sha256="abc")
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", scopes=["overlay:premise"], root=tmp_path / "targets")
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=False, trust_root=tmp_path / "targets")
    key = result["applied"][0]["proposal_key"]
    repo = tmp_path / "repo"
    (repo / "targets" / "apple-m4-max").mkdir(parents=True)
    (repo / "targets" / "apple-m4-max" / "experience.md").write_text("# Experience\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    checked = harness.apply_patch(run, target_id="apple-m4-max", proposal_key=key, root=repo, check_only=True)
    assert checked["check"] is True and checked["applied"] is False
    assert "harness" not in (repo / "targets" / "apple-m4-max" / "experience.md").read_text()
    applied = harness.apply_patch(run, target_id="apple-m4-max", proposal_key=key, root=repo)
    assert applied["applied"] is True and applied["change_ref"] == "" and applied["commit_pending"] is True
    text = (repo / "targets" / "apple-m4-max" / "experience.md").read_text()
    assert "先读 device.md" in text
    decision = json.loads((run / "harness_decisions.jsonl").read_text().splitlines()[-1])
    assert decision["event"] == "promotion" and decision["status"] == "local"
    assert decision["change_ref"] == "" and decision["local_patch_sha256"] == applied["sha256"]
    committed = harness.commit_local_promotions(run, message="promote local harness tip", by="alice", root=repo)
    assert committed["change_ref"].startswith("commit:")
    assert committed["paths"] == ["targets/apple-m4-max/experience.md"]
    assert "先读 device.md" in (repo / "targets/apple-m4-max" / "experience.md").read_text()
    assert harness._read_jsonl(run / "harness_decisions.jsonl")[-1]["event"] == "promotion_commit"


def test_the_run_tests_gate_points_at_the_migrated_paths_and_refuses_a_failure(tmp_path, monkeypatch):
    """The promotion gate used the pre-migration flat paths: pytest exited before running a test,
    so every ``apply_patch(run_tests=True)`` failed. These assertions are the regression guard."""
    from taic import trust

    assert harness.OFFLINE_TESTS
    for relative in harness.OFFLINE_TESTS:
        assert (trust.REPO_ROOT / relative).is_file(), relative

    class Done:
        def __init__(self, code: int) -> None:
            self.returncode, self.stdout, self.stderr = code, "", ""

    seen: list[list[str]] = []

    def run_ok(argv, **kw):
        if "-m" in argv and "pytest" in argv:
            seen.append(list(argv))
        return Done(0)

    monkeypatch.setattr(harness, "write_patch", lambda *a, **k: {"path": "targets/apple-m4-max/experience.md", "patch": "/tmp/p.patch", "sha256": "s"})
    monkeypatch.setattr(harness.subprocess, "run", run_ok)
    out = harness.apply_patch(tmp_path / "run", target_id="apple-m4-max", proposal_key="k", root=tmp_path, run_tests=True)
    assert out["applied"] is True
    assert seen and seen[0][:3] == [sys.executable, "-m", "pytest"] and seen[0][4:] == list(harness.OFFLINE_TESTS)

    def run_fail(argv, **kw):
        return Done(1 if ("-m" in argv and "pytest" in argv) else 0)

    monkeypatch.setattr(harness.subprocess, "run", run_fail)
    with pytest.raises(harness.ProposalError, match="offline tests failed"):
        harness.apply_patch(tmp_path / "run2", target_id="apple-m4-max", proposal_key="k", root=tmp_path, run_tests=True)


def test_two_models_with_the_same_probe_bytes_are_one_source(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    a = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    b = _session(run, seq=2, provider="moonshot", model="kimi-k3", committee="second", doc=_doc(), started=started)
    # _session writes the same probe bytes; neither model has an independent measurement.
    _record_proposal(run, a, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)
    _record_proposal(run, b, _record(2, "moonshot", "kimi-k3", "second"), _doc(), started)
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high", root=tmp_path / "targets")
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=tmp_path / "targets")
    result = harness.settle(
        run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(),
        enabled=True, shadow=True, trust_root=tmp_path / "targets",
    )
    assert result["shadow"] == []


def test_a_shell_like_action_is_invalid_and_never_reaches_the_overlay(tmp_path):
    run = tmp_path / "run"
    started = time.time() - 1
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(action="先跑 probe.py; 然后 rm -rf runs"), started=started)
    rows = _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), _doc(action="先跑 probe.py; 然后 rm -rf runs"), started)
    assert rows[0]["status"] == "invalid"


def test_a_premise_needs_a_matching_harness_record_not_just_a_probe(tmp_path):
    work = tmp_path / "work"
    (work / ".selfcheck").mkdir(parents=True)
    (work / ".selfcheck" / "self_checks.jsonl").write_text(json.dumps({
        "status": "ok", "candidate_us": 463.46, "speedup": 1.2,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "kernel_sha256": "abc",
    }) + "\n")
    ok, why = harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="abc", measurement="463.46 us",
    )
    assert ok and why == "self_checks.jsonl candidate_us"
    assert not harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="other", measurement="463.46 us",
    )[0]
    assert not harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="abc", measurement="15.09 TFLOP/s",
    )[0]


def test_an_outer_candidate_report_can_be_the_trusted_record_without_a_selfcheck(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    report = {
        "status": "ok", "candidate_us": 12.5, "speedup": 1.04,
        "target_id": "apple-m4-max", "protocol_version": trust.current_protocol_version(),
        "kernel_sha256": "abc", "trusted_source": "outer candidate report",
    }
    ok, why = harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="abc", measurement="12.5 us", trusted_report=report,
    )
    assert ok and why == "outer candidate report"
    assert not harness.trusted_premise_record(
        work, target_id="apple-m4-max", protocol_version=trust.current_protocol_version(),
        kernel_sha256="abc", measurement="20.0 us", trusted_report=report,
    )[0]


def test_a_useful_independent_review_activates_a_single_tip_for_this_run(tmp_path):
    import json as _json

    run = tmp_path / "run"
    started = time.time() - 1
    work = _session(run, seq=1, provider="deepseek", model="deepseek-flash", committee="first", doc=_doc(), started=started)
    _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), _doc(), started)
    trust.grant("apple-m4-max", "conv2d", "moonshot", "kimi-k3", effort="high", root=tmp_path / "targets")
    (run / "harness_reviews.jsonl").write_text(_json.dumps({
        "schema_version": 1, "event": "review", "proposal_key": _json.loads((run / "harness_proposals.jsonl").read_text().splitlines()[0])["proposal_key"],
        "target_id": "apple-m4-max", "task": "conv2d", "useful": True, "confidence": 0.8,
        "protocol_version": trust.current_protocol_version(),
        "reviewer": {"provider": "moonshot", "model": "kimi-k3", "effort": "high"},
    }) + "\n")
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True, trust_root=tmp_path / "targets")
    assert len(result["applied"]) == 1 and result["shadow"] == []
    assert result["applied"][0]["reviewed"] is True
    assert "先读 device.md" in harness.overlay_text(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version())
