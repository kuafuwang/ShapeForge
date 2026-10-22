"""Structured proposals, two-source quorum, shadow/active overlay, and P2 patches. Offline only."""

from __future__ import annotations

import json
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


def _record_proposal(run: Path, work: Path, record: dict, doc, started: float):
    parsed = {"sections": {"对系统的建议": _block(doc)}}
    return harness.record_proposals(
        run, record=record, parsed=parsed, target_id="apple-m4-max", task="conv2d",
        dtype="fp16", work=work, started=started,
    )


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
    row = _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), _doc(surface="premise"), started)
    assert row[0]["status"] == "valid"
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=True)
    assert len(result["shadow"]) == 1


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
    _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), _doc(surface="premise"), started)
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=False)
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
    _record_proposal(run, work, _record(1, "deepseek", "deepseek-flash", "first"), _doc(surface="premise"), started)
    result = harness.settle(run, target_id="apple-m4-max", task="conv2d", protocol_version=trust.current_protocol_version(), enabled=True, shadow=False)
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
    assert applied["applied"] is True and applied["change_ref"].startswith("patch:")
    text = (repo / "targets" / "apple-m4-max" / "experience.md").read_text()
    assert "先读 device.md" in text
    assert (run / "harness_decisions.jsonl").read_text().count('"promotion"') == 1


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
