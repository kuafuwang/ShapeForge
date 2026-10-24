"""Proposal-specific evidence and acceptance-to-implementation contracts."""

from __future__ import annotations

import json
import subprocess
import time

import pytest

from taic import acceptance, code_agent, harness


def proposal(kind="tool"):
    evidence = {"probe": "", "measurement": "", "reproduction": "The report omits the needed field.", "expected": "A host-written result contains that field.", "acceptance_checks": ["test result contains the field"], "allowed_paths": ["tcenv/report.py"], "effect_metric_id": "first_eligible_cost_usd"}
    return {"proposal_key": "0123456789abcdef", "run_id": "run", "seq": 1, "claim_digest": "d", "proposal": {"surface": "tip", "kind": kind, "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "Add a diagnostic tool.", "evidence": evidence, "rationale": "Make the limitation measurable."}}


def test_a_tool_tip_needs_reproduction_and_expected_result_not_a_probe(tmp_path):
    doc = proposal()["proposal"]
    doc["scope"] = {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}
    normalized = harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())
    assert normalized["kind"] == "tool" and not normalized["probe_path"]
    doc["evidence"] = {**doc["evidence"], "reproduction": ""}
    with pytest.raises(harness.ProposalError, match="reproduction"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_a_premise_still_needs_a_probe_and_measurement(tmp_path):
    doc = {"surface": "premise", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "先验证再选择。", "evidence": {"reproduction": "a", "expected": "b"}}
    with pytest.raises(harness.ProposalError, match="probe"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_a_tip_cannot_change_evaluation_or_budget(tmp_path):
    doc = proposal()["proposal"]
    doc["evidence"]["expected"] = "Change correctness tolerance so the candidate passes."
    with pytest.raises(harness.ProposalError, match="evaluation"):
        harness.validate_proposal(doc, target_id="apple-m4-max", task="conv2d", dtype="fp16", work=tmp_path, started=time.time())


def test_review_recommended_test_reaches_the_implementation_prompt():
    review = {"useful": True, "confidence": 0.8, "risk": "wrong metric", "recommended_test": "compare the paired metric", "review_sha256": "a" * 64}
    spec = acceptance.make_spec(proposal(), review)
    assert acceptance.read_spec(spec) == spec
    prompt = code_agent.build_prompt(proposal()["proposal"], run_id="run", acceptance_spec=spec)
    assert "compare the paired metric" in prompt and "acceptance" in prompt
    report = {"acceptance": [{"check": "test result contains the field", "passed": True}, {"check": "compare the paired metric", "passed": True}]}
    implementation = acceptance.implementation_report(report, spec, ["tcenv/report.py"], "b" * 64)
    assert implementation["accepted"] is True
    implementation = acceptance.implementation_report(report, spec, ["tcenv/evaluate.py"], "b" * 64)
    assert implementation["accepted"] is False and implementation["outside_paths"] == ["tcenv/evaluate.py"]


def test_acceptance_digest_catches_changed_checks():
    spec = acceptance.make_spec(proposal())
    changed = {**spec, "acceptance_checks": ["a much weaker test"]}
    with pytest.raises(acceptance.AcceptanceError, match="digest"):
        acceptance.read_spec(changed)


def test_a_board_keeps_review_implementation_and_effect_separate(tmp_path):
    row = {"schema_version": 1, "event": "proposal", "status": "valid", **proposal()}
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps(row) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": row["proposal_key"], "useful": True, "cost_usd": 0.1}) + "\n")
    (tmp_path / "harness_decisions.jsonl").write_text(json.dumps({"event": "code_promotion", "proposal_key": row["proposal_key"], "status": "local"}) + "\n")
    milestones = harness.board(tmp_path)["acceptance_specs"][0]["milestones"]
    assert milestones == {"reviewed": True, "implemented": False, "experiment_validated": False, "local": True, "committed": False}


def test_verify_code_records_host_acceptance_before_apply(tmp_path, monkeypatch):
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    spec["host_checks"] = [{"check_id": "reporting.offline_contract", "description": "offline report check"}]
    spec["spec_sha256"] = acceptance.digest({key: value for key, value in spec.items() if key != "spec_sha256"})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {"acceptance": [{"check": "test result contains the field", "passed": True}]}}, acceptance_spec=spec)
    class Runner:
        isolated = True

        def evidence(self):
            return {"kind": "fake-isolated", "network": "none", "read_only_root": True}

        def run(self, _repo, _patch, _key, *, check_items=None, base_commit=None):
            del base_commit
            return [{"check_id": item["check_id"], "passed": True, "exit_code": 0, "output_excerpt": "offline fake", "reason": ""} for item in (check_items or [])]

    code_agent.review_code_patch(tmp_path, key, by="alice")
    verified = code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice", runner=Runner())
    assert verified["host_verified"] is True
    # A later static-only invocation must not wipe host-verified evidence.
    assert code_agent.verify_code_patch(tmp_path, key)["host_verified"] is True
    with pytest.raises(code_agent.CodeAgentError, match="no effect validation"):
        code_agent.apply_code_patch(tmp_path, proposal_key=key, require_effect_validation=True, approved_by="alice")


def test_static_verification_does_not_execute_patch_code(tmp_path, monkeypatch):
    """@brief 默认静态验收只检查证据，不运行补丁中的 import/pytest。"""
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    spec["host_checks"] = [{"check_id": "reporting.offline_contract", "description": "offline report check"}]
    spec["spec_sha256"] = acceptance.digest({key: value for key, value in spec.items() if key != "spec_sha256"})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {"acceptance": [{"check": "test result contains the field", "passed": True}]}}, acceptance_spec=spec)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("static verification executed patch code")

    monkeypatch.setattr(code_agent, "_test_patch_in_worktree", forbidden)
    verified = code_agent.verify_code_patch(tmp_path, key)
    assert verified["static_verified"] is True and verified["host_verified"] is False and verified["accepted"] is False


def test_legacy_text_requires_named_approval_and_real_registered_checks(tmp_path, monkeypatch):
    key = "0123456789abcdef"
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {}}, acceptance_spec=spec)
    with pytest.raises(code_agent.CodeAgentError, match="legacy acceptance"):
        code_agent.verify_code_patch(tmp_path, key, legacy_text="old human check", check_ids=["reporting.offline_contract"])
    class Runner:
        isolated = True

        def evidence(self):
            return {"kind": "fake-isolated", "network": "none", "read_only_root": True}

        def run(self, _repo, _patch, _key, *, check_items=None, base_commit=None):
            del base_commit
            return [{"check_id": item["check_id"], "passed": True, "exit_code": 0, "output_excerpt": "ran", "reason": ""} for item in (check_items or [])]

    # The first static upgrade writes the legacy-derived spec. The human then
    # reviews that exact spec before dynamic execution.
    code_agent.verify_code_patch(tmp_path, key, approved_by="alice", legacy_text="old human check", check_ids=["reporting.offline_contract"])
    code_agent.review_code_patch(tmp_path, key, by="alice")
    verified = code_agent.verify_code_patch(tmp_path, key, approved_by="alice", legacy_text="old human check", check_ids=["reporting.offline_contract"], execute=True, runner=Runner())
    assert verified["legacy_upgrade"]["approved_by"] == "alice"
    assert verified["legacy_upgrade"]["original_text"] == "old human check"
    assert verified["host_checks"][0]["output_excerpt"] == "ran"
    assert verified["risk_tier"] == "paired_required"


def test_risk_is_recomputed_from_the_patch_not_the_model_label(tmp_path):
    spec = acceptance.make_spec({"proposal_key": "0123456789abcdef", "proposal": {"surface": "tip", "kind": "tool", "action": "offline tool", "scope": {}, "evidence": {"reproduction": "r", "expected": "e", "allowed_paths": ["tcenv/report.py"], "risk_tier": "offline_only", "host_checks": [{"check_id": "reporting.offline_contract"}]}}})
    patch = "diff --git a/tcenv/evaluate.py b/tcenv/evaluate.py\n--- a/tcenv/evaluate.py\n+++ b/tcenv/evaluate.py\n@@ -0,0 +1 @@\n+change\n"
    assert acceptance.classify_change(spec, patch)["risk_tier"] == "human_gated"


def _write_reviewable_patch(tmp_path, key="0123456789abcdef"):
    row = proposal()
    spec = acceptance.make_spec(row, {"useful": True, "cost_usd": 0.1})
    spec["host_checks"] = [{"check_id": "reporting.offline_contract", "description": "offline report check"}]
    spec["spec_sha256"] = acceptance.digest({k: v for k, v in spec.items() if k != "spec_sha256"})
    (tmp_path / "harness_proposals.jsonl").write_text(json.dumps({"status": "valid", **row}) + "\n")
    (tmp_path / "harness_reviews.jsonl").write_text(json.dumps({"event": "review", "proposal_key": key, "useful": True, "cost_usd": 0.1}) + "\n")
    patch = "diff --git a/tcenv/report.py b/tcenv/report.py\nnew file mode 100644\n--- /dev/null\n+++ b/tcenv/report.py\n@@ -0,0 +1 @@\n+x\n"
    code_agent.write_patch(tmp_path, key, {"patch": patch, "report": {"acceptance": [{"check": "test result contains the field", "passed": True}]}}, acceptance_spec=spec)
    return key, spec, patch


def test_dynamic_verification_requires_a_review_and_an_explicit_runner(tmp_path):
    """@brief execute 无人工 review 或无隔离 runner 时必须拒绝，不能退回本地执行。"""
    key, _spec, _patch = _write_reviewable_patch(tmp_path)
    with pytest.raises(code_agent.CodeAgentError, match="human review"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice")
    code_agent.review_code_patch(tmp_path, key, by="alice")
    with pytest.raises(code_agent.CodeAgentError, match="isolated patch test runner"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice")


def test_review_binding_is_invalidated_when_the_patch_changes(tmp_path):
    """@brief 人工 review 绑定 patch hash；补丁变了，旧 review 不能再用。"""
    key, _spec, patch = _write_reviewable_patch(tmp_path)
    code_agent.review_code_patch(tmp_path, key, by="alice")
    path = tmp_path / code_agent.PATCH_DIR / f"{key}.patch"
    path.write_text(patch + "extra")
    with pytest.raises(code_agent.CodeAgentError, match="changed"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice")


def test_static_verification_is_explicit_and_never_requires_a_review(tmp_path):
    """@brief --static 只检查证据，不需要 review，也不执行 runner。"""
    key, _spec, _patch = _write_reviewable_patch(tmp_path)
    record = code_agent.verify_code_patch(tmp_path, key, static=True)
    assert record["static_verified"] is True
    assert record["host_verified"] is False


def test_dynamic_verification_rejects_empty_and_non_boolean_runner_results(tmp_path):
    """@brief 空结果或字符串形式的 false 都不能变成 host_verified。"""
    key, _spec, _patch = _write_reviewable_patch(tmp_path)
    code_agent.review_code_patch(tmp_path, key, by="alice")

    class EmptyRunner:
        isolated = True
        def evidence(self):
            return {"kind": "fake-isolated", "network": "none", "read_only_root": True}

        def run(self, *args, **kwargs):
            return []

    with pytest.raises(code_agent.CodeAgentError, match="no acceptance checks"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice", runner=EmptyRunner())

    class FalseRunner:
        isolated = True
        def evidence(self):
            return {"kind": "fake-isolated", "network": "none", "read_only_root": True}

        def run(self, *args, **kwargs):
            return [{"check_id": "reporting.offline_contract", "passed": "false"}]

    with pytest.raises(code_agent.CodeAgentError, match="acceptance checks failed"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice", runner=FalseRunner())


def test_dynamic_verification_requires_an_isolated_runner_not_a_generic_command(tmp_path):
    """@brief 只标 runner 名称不够，未声明隔离后端的 runner 必须拒绝。"""
    key, _spec, _patch = _write_reviewable_patch(tmp_path)
    code_agent.review_code_patch(tmp_path, key, by="alice")

    class LocalRunner:
        def run(self, *args, **kwargs):
            return [{"check_id": "reporting.offline_contract", "passed": True}]

    with pytest.raises(code_agent.CodeAgentError, match="not an isolated"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice", runner=LocalRunner())


def test_review_record_cannot_be_reassigned_to_another_proposal(tmp_path):
    """@brief review 文件中的 proposal_key 必须与当前验收提案一致。"""
    key, _spec, _patch = _write_reviewable_patch(tmp_path)
    code_agent.review_code_patch(tmp_path, key, by="alice")
    path = tmp_path / code_agent.PATCH_DIR / f"{key}{code_agent.DIFF_REVIEW_SUFFIX}"
    record = json.loads(path.read_text())
    record["proposal_key"] = "fedcba9876543210"
    path.write_text(json.dumps(record))
    with pytest.raises(code_agent.CodeAgentError, match="different proposal"):
        code_agent.verify_code_patch(tmp_path, key, execute=True, approved_by="alice", runner=object())


def test_docker_runner_constructs_a_restricted_container_without_starting_one(tmp_path, monkeypatch):
    """@brief Docker runner 的 argv 必须禁网、只读根 FS，并只挂临时树和受信 harness。"""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "taic").mkdir()
    (repo / "taic" / "isolated_patch_worker.py").write_text("# trusted test worker\n")
    (repo / "a.txt").write_text("before\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    patch = tmp_path / "candidate.patch"
    patch.write_text("diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-before\n+after\n")
    seen = {}

    class Proc:
        returncode = 0
        def communicate(self, timeout=None):
            return b"[]", b""

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return Proc()

    monkeypatch.setattr(code_agent.DockerPatchTestRunner, "_image_identity", lambda _self: "sha256:test-image")
    monkeypatch.setattr(code_agent, "popen_group", fake_popen)
    monkeypatch.setattr(code_agent, "kill_group", lambda _proc: None)
    monkeypatch.setattr(code_agent, "forget", lambda _proc: None)
    monkeypatch.setattr(code_agent.DockerPatchTestRunner, "_remove_container", lambda *_args: None)
    runner = code_agent.DockerPatchTestRunner("example.invalid/taic-test:1")
    results = runner.run(repo, patch, "key", check_items=[{"check_id": "reporting.offline_contract"}], base_commit="HEAD")
    assert results == []
    argv = seen["argv"]
    assert argv[1:3] == ["run", "--rm"]
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert "--read-only" in argv and "--cap-drop" in argv
    assert "--privileged" not in argv
    mounts = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--mount"]
    assert any("dst=/work" in mount for mount in mounts)
    assert any("dst=/opt/taic" in mount for mount in mounts)
    assert not any("dst=/work,ro" in mount for mount in mounts)
    assert not any("dst=/opt/taic,readonly" in mount and "/repo," in mount for mount in mounts)
    assert not any("dst=/opt/taic" in mount and "src=" + str(repo) + "," in mount for mount in mounts)
    assert seen["kwargs"]["env"].keys() == {"PATH", "HOME"}
    evidence = runner.evidence()
    assert evidence["harness_base_commit"] == "HEAD"
    assert evidence["image_id"] == "sha256:test-image"
    assert evidence["network"] == "none" and evidence["read_only_root"] is True
