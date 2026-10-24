"""Controlled harness experiment plans bind change, configuration, seeds and effect review."""

from __future__ import annotations

import copy

import pytest

from taic import acceptance
from taic.harness_experiment import ExperimentError, authorize_plan, config_template, digest, make_plan, read_plan, record_effect_validation, summarize_result


def _plan():
    spec = acceptance.make_spec({"proposal_key": "0123456789abcdef", "proposal": {"surface": "tip", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "Add a diagnostic tool.", "evidence": {"reproduction": "failure", "expected": "fixed", "effect_metric_id": "first_eligible_cost_usd"}}})
    raw = {"schema_version": 2, "protocol_version": "3", "task": "conv2d", "dtype": "fp16", "size": "paper", "target": {"id": "apple-m4-max", "backend": "metal"}, "llm": {"reasoning_effort": "high"}, "search": {"budget_usd": 1.25}, "committees": [{"name": "first", "members": [{"provider": "p1", "model": "m1"}]}, {"name": "second", "members": [{"provider": "p2", "model": "m2"}]}]}
    template = config_template(raw, budget_usd=1.25)
    return make_plan(spec, change_sha256="a" * 64, protocol_version="3", config_sha256=digest(template), seeds=[7, 11], budget_usd=1.25, config_template_document=template)


def _result(plan):
    arms = {}
    for arm, cost in (("control", 1.0), ("treatment", 0.75)):
        arms[arm] = {
            "config_sha256": plan["config_sha256"], "protocol_version": plan["protocol_version"],
            "seeds": plan["seeds"], "repetitions": 1, "change_sha256": plan[arm]["change_sha256"], "spent_usd": 1.0,
            "first_eligible_cost_usd": {"samples": [cost], "unit": "usd", "run_digests": ["c" * 64]},
            "safety_regressions": {"samples": [0], "unit": "count", "run_digests": ["c" * 64]},
            "correctness_regressions": {"samples": [0], "unit": "count", "run_digests": ["c" * 64]},
        }
    return {"schema_version": 1, "plan": plan, **arms}


def test_dry_run_needs_two_explicit_permissions_and_an_approver():
    plan = _plan()
    assert plan["status"] == "dry_run" and read_plan(plan) == plan
    with pytest.raises(ExperimentError, match="allow-spend"):
        authorize_plan(plan, allow_model_spend=True, allow_gpu=False, approved_by="alice")
    with pytest.raises(ExperimentError, match="named"):
        authorize_plan(plan, allow_model_spend=True, allow_gpu=True, approved_by="cli")
    authorized = authorize_plan(plan, allow_model_spend=True, allow_gpu=True, approved_by="alice")
    assert authorized["experiment_id"] != plan["experiment_id"] and authorized["status"] == "authorized"


def test_a_plan_rejects_tampered_change_and_missing_scope():
    plan = _plan()
    changed = copy.deepcopy(plan)
    changed["change_sha256"] = "d" * 64
    with pytest.raises(ExperimentError, match="experiment id"):
        read_plan(changed)
    with pytest.raises(ExperimentError, match="canonical run_config"):
        make_plan(plan["acceptance_spec"], change_sha256="a" * 64, protocol_version="3", config_sha256="b" * 64, seeds=[1], budget_usd=1, config_template_document={})


def test_result_requires_matching_arms_and_declared_metric():
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan)
    summary = summarize_result(result)
    assert summary["effect_validated"] is False
    changed = copy.deepcopy(result)
    changed["treatment"]["config_sha256"] = "e" * 64
    with pytest.raises(ExperimentError, match="arm differs"):
        summarize_result(changed)
    changed = copy.deepcopy(result)
    del changed["control"]["first_eligible_cost_usd"]
    with pytest.raises(ExperimentError, match="both arms"):
        summarize_result(changed)


def test_a_person_records_effect_validation_without_self_approval(tmp_path):
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan)
    for arm in ("control", "treatment"):
        result[arm]["first_eligible_cost_usd"]["samples"] = [1.0, 1.0, 1.0] if arm == "control" else [0.5, 0.5, 0.5]
    with pytest.raises(ExperimentError, match="reason"):
        record_effect_validation(tmp_path, result, validated=True, approved_by="alice", reason="")
    with pytest.raises(ExperimentError, match="named --by"):
        record_effect_validation(tmp_path, result, validated=True, approved_by="cli", reason="review")
    row = record_effect_validation(tmp_path, result, validated=True, approved_by="bob", reason="paired metrics and no guardrail regression")
    assert row["validated"] is True and row["approved_by"] == "bob"
    assert row["spent_usd"] == 2.0
    with pytest.raises(ExperimentError, match="sole effect reviewer"):
        record_effect_validation(tmp_path, result, validated=True, approved_by="alice", reason="self approval is not independent")


def test_config_template_keeps_shared_conditions_and_drops_run_identity():
    raw = {"schema_version": 2, "created_at": 1, "git_dirty": True, "run_seed": 7, "seed": 9, "task": "conv2d", "dtype": "fp16", "size": "paper", "target": {"id": "apple-m4-max"}, "committees": [{}, {}], "search": {"budget_usd": 99, "eval_cache": True, "harness_change_sha256": "a" * 64}}
    template = config_template(raw, budget_usd=1.25)
    assert template["search"] == {"budget_usd": 1.25, "eval_cache": False}
    assert "created_at" not in template and "run_seed" not in template and "git_dirty" not in template


def test_collect_arm_reads_trace_and_rejects_an_unbound_run(tmp_path):
    from taic.harness_experiment import arm_config, collect_arm
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    config = arm_config(plan, "control", 7)
    config["search"].update({"harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": "control", "harness_change_sha256": ""})
    result = {"schema_version": 1, "protocol_version": "3", "run_seed": 7, "spent_usd": 1.0, "harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": "control", "harness_change_sha256": "", "trace": [{"idx": 1, "status": "ok", "cost_usd": 0.4, "t_s": 8.0}]}
    (tmp_path / "run_config.json").write_text(__import__("json").dumps(config))
    (tmp_path / "result.json").write_text(__import__("json").dumps(result))
    (tmp_path / "feedback.jsonl").write_text(__import__("json").dumps({"candidate_status": "ok", "candidate_us": 1.0, "cost_usd": 1.0}) + "\n")
    arm = collect_arm(plan, tmp_path, "control")
    assert arm["first_eligible_cost_usd"]["samples"] == [0.4]
    assert arm["first_eligible_elapsed_s"]["samples"] == [8.0]
    result["harness_change_sha256"] = "f" * 64
    (tmp_path / "result.json").write_text(__import__("json").dumps(result))
    with pytest.raises(ExperimentError, match="expected harness change"):
        collect_arm(plan, tmp_path, "control")


def test_collect_arm_rejects_a_missing_cost_ledger(tmp_path):
    """@brief 有 trace 或支出时，缺失 feedback 账本不能绕过费用对账。"""
    from taic.harness_experiment import arm_config, collect_arm

    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    config = arm_config(plan, "control", 7)
    config["search"].update({"harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": "control", "harness_change_sha256": ""})
    result = {"schema_version": 1, "protocol_version": "3", "run_seed": 7, "spent_usd": 1.0, "harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": "control", "harness_change_sha256": "", "trace": [{"idx": 1, "status": "ok", "cost_usd": 0.4, "t_s": 8.0}]}
    (tmp_path / "run_config.json").write_text(__import__("json").dumps(config))
    (tmp_path / "result.json").write_text(__import__("json").dumps(result))
    with pytest.raises(ExperimentError, match="complete per-session cost ledger"):
        collect_arm(plan, tmp_path, "control")


def test_latest_effect_verdict_controls_milestones(tmp_path):
    """@brief 同一实验后续 validated=false 必须撤销旧的 true，并绑定结果摘要。"""
    from taic.harness_experiment import milestones

    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan)
    result["control"]["first_eligible_cost_usd"]["samples"] = [1.0, 1.0, 1.0]
    result["treatment"]["first_eligible_cost_usd"]["samples"] = [0.5, 0.5, 0.5]
    record_effect_validation(tmp_path, result, validated=True, approved_by="bob", reason="paired evidence")
    row = record_effect_validation(tmp_path, result, validated=False, approved_by="bob", reason="later review rejected it")
    assert row["result_sha256"] == digest(result)
    decisions = [__import__("json").loads(line) for line in (tmp_path / "harness_decisions.jsonl").read_text().splitlines()]
    assert milestones(decisions, plan["proposal_key"])["experiment_validated"] is False
