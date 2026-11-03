"""Controlled harness experiment plans bind change, configuration, seeds and effect review."""

from __future__ import annotations

import copy
import json
import math

import pytest

from taic import acceptance
from taic.harness_experiment import (
    ExperimentError,
    authorize_plan,
    collect_arm,
    config_template,
    digest,
    make_plan,
    paired_effect,
    plan_cells,
    read_plan,
    record_effect_validation,
    summarize_result,
)


def _plan(*, seeds=(7, 11, 13), repetitions=1):
    spec = acceptance.make_spec({"proposal_key": "0123456789abcdef", "proposal": {"surface": "tip", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "Add a diagnostic tool.", "evidence": {"reproduction": "failure", "expected": "fixed", "effect_metric_id": "first_eligible_cost_usd"}}})
    raw = {"schema_version": 2, "protocol_version": "3", "task": "conv2d", "dtype": "fp16", "size": "paper", "target": {"id": "apple-m4-max", "backend": "metal"}, "llm": {"reasoning_effort": "high"}, "search": {"budget_usd": 1.25}, "committees": [{"name": "first", "members": [{"provider": "p1", "model": "m1"}]}, {"name": "second", "members": [{"provider": "p2", "model": "m2"}]}]}
    template = config_template(raw, budget_usd=1.25)
    return make_plan(spec, change_sha256="a" * 64, protocol_version="3", config_sha256=digest(template), seeds=list(seeds), budget_usd=1.25, repetitions=repetitions, config_template_document=template)


def _cell_records(plan, arm, digest_value, cells=None):
    items = plan_cells(plan) if cells is None else cells
    return [{"seed": item["seed"], "repetition": item["repetition"], "pair_index": item["pair_index"], "order": item["order"], "arm": arm, "run_digest": digest_value} for item in items]


def _measurement(plan, arm, value, *, unit="usd", digest_value=None, cells=None):
    """One sample per planned cell, each carrying its frozen pair identity."""
    digest_value = digest_value or ("c" * 64 if arm == "control" else "d" * 64)
    cells = _cell_records(plan, arm, digest_value) if cells is None else cells
    values = list(value) if isinstance(value, (list, tuple)) else [value] * len(cells)
    return {"samples": values, "unit": unit, "run_digests": [digest_value], "cells": cells}


def _result(plan, *, control=1.0, treatment=0.75):
    arms = {}
    for arm, cost in (("control", control), ("treatment", treatment)):
        arms[arm] = {
            "config_sha256": plan["config_sha256"], "protocol_version": plan["protocol_version"],
            "seeds": plan["seeds"], "repetitions": plan["repetitions"], "change_sha256": plan[arm]["change_sha256"], "spent_usd": 1.0,
            "first_eligible_cost_usd": _measurement(plan, arm, cost),
            "safety_regressions": _measurement(plan, arm, 0, unit="count"),
            "correctness_regressions": _measurement(plan, arm, 0, unit="count"),
        }
    return {"schema_version": 1, "plan": plan, **arms}


def _reseal(document):
    """Recompute the id a tampered plan would carry if it had been written by its author."""
    out = copy.deepcopy(document)
    out["experiment_id"] = digest({key: value for key, value in out.items() if key != "experiment_id"})[:24]
    return out


def _arm_config(plan, arm, seed):
    from taic.harness_experiment import arm_config

    config = arm_config(plan, arm, seed)
    config["search"].update({"harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": arm, "harness_change_sha256": plan[arm]["change_sha256"] or ""})
    return config


def _requests(sessions, total):
    rows, running = [], 0.0
    for index, cost in enumerate(sessions):
        running += cost
        rows.append({"label": f"s{index}", "cost_usd": cost, "total_usd": running})
    gap = round(total - running, 9)
    if gap > 0:
        rows.append({"label": "review:0123456789abcdef", "cost_usd": gap, "total_usd": total})
    return rows


def _write_arm(tmp_path, plan, *, arm="control", seed=7, spent=1.05, sessions=(0.6, 0.4), ledger_total=None, transcript=True, ledger=True, requests=None, feedback="default", extra_transcript=None, trace_cost=0.4):
    """A run directory shaped like a real one: billed transcript, ledger, feedback without cost_usd."""
    (tmp_path / "run_config.json").write_text(json.dumps(_arm_config(plan, arm, seed)))
    result = {"schema_version": 1, "protocol_version": "3", "run_seed": seed, "spent_usd": spent, "harness_experiment_id": plan["experiment_id"], "harness_experiment_arm": arm, "harness_change_sha256": plan[arm]["change_sha256"] or "", "trace": [{"idx": 1, "status": "ok", "cost_usd": trace_cost, "t_s": 8.0}]}
    (tmp_path / "result.json").write_text(json.dumps(result))
    if transcript:
        rows = [{"event": "agent", "label": f"s{index}", "provider": "deepseek" if index == 0 else "moonshot", "model": "deepseek-flash" if index == 0 else "kimi-k3", "committee": "first" if index == 0 else "second", "cost_usd": cost, "cost_incomplete": False, "outcome": "evaluated" if index == 0 else "agent_failed", "tokens": {"input": 10, "output": 20, "cache_read": 30}} for index, cost in enumerate(sessions)]
        if extra_transcript:
            rows.extend(extra_transcript)
        (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    if ledger:
        total = spent if ledger_total is None else ledger_total
        rows = _requests(sessions, total) if requests is None else requests
        (tmp_path / "ledger.json").write_text(json.dumps({"schema_version": 1, "summary": {"total_usd": total, "budget_usd": 2.0, "requests": len(rows)}, "requests": rows}))
    if feedback == "default":
        feedback = [{"format": "agent_feedback", "seq": 1, "label": "s0", "outcome": "evaluated"}, {"format": "agent_feedback", "seq": 2, "label": "s1", "outcome": "agent_failed"}]
    if feedback:
        (tmp_path / "feedback.jsonl").write_text("".join(json.dumps(row) + "\n" for row in feedback))
    return tmp_path


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
    result = _result(plan, control=1.0, treatment=0.5)
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


def test_collect_arm_reads_transcript_and_ledger_not_feedback_cost(tmp_path):
    """@brief 真实形状的 feedback（没有 cost_usd）不再是账单；美元来自 transcript 明细加 ledger 汇总。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan)
    arm = collect_arm(plan, tmp_path, "control")
    assert arm["first_eligible_cost_usd"]["samples"] == [0.4]
    assert arm["first_eligible_elapsed_s"]["samples"] == [8.0]
    assert arm["spent_usd"] == 1.05
    assert arm["session_cost_usd"] == 1.0
    assert arm["non_session_cost_usd"] == 0.05
    assert [(row["provider"], row["cost_usd"], row["sessions"]) for row in arm["model_totals"]] == [("deepseek", 0.6, 1), ("moonshot", 0.4, 1)]
    assert arm["invalid_feedback_rate"]["samples"] == [0.5]
    assert arm["first_eligible_cost_usd"]["cells"] == [{"seed": 7, "repetition": 0, "pair_index": 0, "order": 0, "arm": "control", "run_digest": arm["run_digests"][0]}]


def test_collect_arm_rejects_an_unbound_run(tmp_path):
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan)
    result = json.loads((tmp_path / "result.json").read_text())
    result["harness_change_sha256"] = "f" * 64
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ExperimentError, match="expected harness change"):
        collect_arm(plan, tmp_path, "control")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda tmp: (tmp / "transcript.jsonl").unlink(), "billed session record"),
        (lambda tmp: (tmp / "ledger.json").unlink(), "complete per-session cost ledger"),
        (lambda tmp: (tmp / "ledger.json").write_text(json.dumps({"schema_version": 1, "summary": {"total_usd": 1.05}, "requests": [{"label": "s0", "cost_usd": 0.6}, {"label": "s1", "cost_usd": 0.4}]})), "requests do not add up"),
        (lambda tmp: _rewrite_ledger(tmp, 1.5), "does not reconcile"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": None, "cost_incomplete": True}]), "finite, complete cost"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": None}]), "finite, complete cost"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": 0.2, "cost_incomplete": True}]), "finite, complete cost"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": 1.2, "cost_incomplete": False}]), "ledger bills a session the transcript does not record"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": 0.6, "cost_incomplete": False}]), "ledger bills a session the transcript does not record"),
        (lambda tmp: _rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": 0.6, "cost_incomplete": False}, {"event": "agent", "label": "s1", "cost_usd": 0.4, "cost_incomplete": False}, {"event": "agent", "label": "extra", "cost_usd": 0.0, "cost_incomplete": False}]), "transcript records a session the ledger does not bill"),
        (lambda tmp: (_rewrite_transcript(tmp, [{"event": "agent", "label": "s0", "cost_usd": 2.0, "cost_incomplete": False}]), _drop_requests(tmp)), "bills more than the run ledger"),
    ],
)
def test_collect_arm_fails_closed_without_a_reconciled_cost_ledger(tmp_path, mutate, message):
    """@brief transcript/ledger 缺失或对不上时必须报错，不得用 0 或默认值补齐。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan)
    mutate(tmp_path)
    with pytest.raises(ExperimentError, match=message):
        collect_arm(plan, tmp_path, "control")


def _rewrite_ledger(tmp_path, total):
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    ledger["summary"]["total_usd"] = total
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))


def _drop_requests(tmp_path):
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    ledger.pop("requests", None)
    (tmp_path / "ledger.json").write_text(json.dumps(ledger))


def _rewrite_transcript(tmp_path, rows):
    (tmp_path / "transcript.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_collect_arm_rejects_missing_request_rows_instead_of_inventing_review_cost(tmp_path):
    """@brief requests 缺失时不能把无凭据的差额记成非会话 review 成本。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan, sessions=(0.6,))
    _drop_requests(tmp_path)
    with pytest.raises(ExperimentError, match="no request records"):
        collect_arm(plan, tmp_path, "control")


def test_collect_arm_rejects_an_empty_transcript_even_at_zero_spend(tmp_path):
    """@brief 没有计费会话就是账本不完整，即使 spent=0 也不放行。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan, spent=0.0, sessions=(), trace_cost=0.0)
    with pytest.raises(ExperimentError, match="no billed session"):
        collect_arm(plan, tmp_path, "control")


def test_collect_arm_accepts_the_orchestrators_known_zero_session(tmp_path):
    """@brief ``cost_incomplete=False`` 且无费用是记账方声明的“零账单”，不是用 0 盖住缺失。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan, spent=0.0, sessions=(0.0,), trace_cost=0.0)
    _rewrite_transcript(tmp_path, [{"event": "agent", "label": "s0", "cost_usd": None, "cost_incomplete": False}])
    arm = collect_arm(plan, tmp_path, "control")
    assert arm["spent_usd"] == 0.0 and arm["session_cost_usd"] == 0.0 and arm["non_session_cost_usd"] == 0.0


def test_collect_arm_needs_the_cell_identity_when_a_seed_repeats(tmp_path):
    """@brief repetitions>1 时同一个 seed 有多个 cell，不靠调用顺序猜是哪一次。"""
    plan = authorize_plan(_plan(seeds=(7, 11, 13, 7, 11, 13), repetitions=2), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    _write_arm(tmp_path, plan, seed=7)
    with pytest.raises(ExperimentError, match="needs pair_index and repetition"):
        collect_arm(plan, tmp_path, "control")
    arm = collect_arm(plan, tmp_path, "control", pair_index=0, repetition=1, order=3)
    assert arm["first_eligible_cost_usd"]["cells"][0]["repetition"] == 1
    with pytest.raises(ExperimentError, match="not one of the plan's frozen"):
        collect_arm(plan, tmp_path, "control", pair_index=1, repetition=0)


def test_repetitions_keep_one_pair_count_and_reject_undeclared_repeats():
    """@brief ``len(seeds) == pair_count * repetitions``；重复只来自事前声明的 repetition。"""
    plan = _plan(seeds=(7, 11, 13, 7, 11, 13), repetitions=2)
    assert plan["pair_count"] == 3 and read_plan(plan) == plan
    assert [cell["pair_index"] for cell in plan_cells(plan)] == [0, 1, 2, 0, 1, 2]
    assert [cell["repetition"] for cell in plan_cells(plan)] == [0, 0, 0, 1, 1, 1]
    with pytest.raises(ExperimentError, match="one entry per pair and repetition"):
        _plan(seeds=(7, 11, 13, 7), repetitions=2)
    with pytest.raises(ExperimentError, match="repeats more often"):
        _plan(seeds=(7, 7, 7, 7), repetitions=2)
    with pytest.raises(ExperimentError, match="repeats more often"):
        _plan(seeds=(7, 7), repetitions=1)
    short = copy.deepcopy(plan)
    short["seeds"] = short["seeds"][:-1]
    with pytest.raises(ExperimentError, match="one entry per pair and repetition|pair count"):
        read_plan(_reseal(short))
    wrong = copy.deepcopy(plan)
    wrong["pair_count"] = 6
    with pytest.raises(ExperimentError, match="pair count"):
        read_plan(_reseal(wrong))


def test_v1_upgrade_recomputes_pair_count_from_repetitions():
    """@brief v1 升级不能再把 pair_count 写成 len(seeds)，否则 repetitions>1 会被自己的校验拒掉。"""
    plan = _plan()
    legacy = {key: copy.deepcopy(plan[key]) for key in ("proposal_key", "spec_sha256", "acceptance_spec", "change_sha256", "protocol_version", "config_sha256", "config_template", "scope", "control", "treatment", "guardrail_metrics", "rollback", "authorization", "status", "effect_metric", "effect_metric_id")}
    legacy.update({"schema_version": 1, "seeds": [7, 11, 13, 7, 11, 13], "repetitions": 2, "budget_per_arm_usd": 1.25, "max_total_budget_usd": 15.0})
    upgraded = read_plan(legacy)
    assert upgraded["schema_version"] == 2 and upgraded["pair_count"] == 3 and upgraded["repetitions"] == 2
    assert read_plan(upgraded) == upgraded


def test_paired_effect_joins_cells_by_identity_not_position():
    """@brief 只打乱一臂、每格数值不同；只有按身份配对才能得到 log(0.5) 的退化区间。

    control 按 pair 的值是 1/2/4/8，treatment 是 0.5/1/2/4，按身份配对的比值全是 0.5。
    位置 zip 会得到 log(0.125)、log(1)、log(0.25)、log(2) 混合，区间必然不退化。
    """
    plan = authorize_plan(_plan(seeds=(7, 11, 13, 17)), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    control = _measurement(plan, "control", [1.0, 2.0, 4.0, 8.0])
    treatment = _measurement(plan, "treatment", [0.5, 1.0, 2.0, 4.0])
    order = [2, 0, 3, 1]
    control["samples"] = [control["samples"][index] for index in order]
    control["cells"] = [control["cells"][index] for index in order]
    effect = paired_effect({"plan": plan, "control": {"first_eligible_cost_usd": control}, "treatment": {"first_eligible_cost_usd": treatment}})
    expected = math.log(0.5)
    assert effect["pairs"] == 4 and effect["pair_count"] == 4 and effect["missing_cells"] == 0
    assert effect["point_estimate"] == pytest.approx(expected)
    assert effect["ci95"] == [pytest.approx(expected), pytest.approx(expected)]
    # Same data in plan order must give the same identity result; order is not identity.
    effect_ordered = paired_effect({"plan": plan, "control": {"first_eligible_cost_usd": _measurement(plan, "control", [1.0, 2.0, 4.0, 8.0])}, "treatment": {"first_eligible_cost_usd": _measurement(plan, "treatment", [0.5, 1.0, 2.0, 4.0])}})
    assert effect_ordered["ci95"] == effect["ci95"] and effect_ordered["point_estimate"] == pytest.approx(expected)


def test_real_collector_output_reaches_a_summary(tmp_path):
    """@brief 真实 ``collect_arm`` 输出（control 记录为空 change digest）必须能进 ``summarize_result``。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    control_dir = tmp_path / "control"
    treatment_dir = tmp_path / "treatment"
    control_dir.mkdir()
    treatment_dir.mkdir()
    _write_arm(control_dir, plan, arm="control", seed=7)
    _write_arm(treatment_dir, plan, arm="treatment", seed=7)
    # The run records "no change" as an empty string; the plan spells it as None.
    assert json.loads((control_dir / "result.json").read_text())["harness_change_sha256"] == ""
    control = collect_arm(plan, control_dir, "control")
    treatment = collect_arm(plan, treatment_dir, "treatment")
    assert control["change_sha256"] is None
    assert treatment["change_sha256"] == plan["treatment"]["change_sha256"]
    summary = summarize_result({"schema_version": 1, "plan": plan, "control": control, "treatment": treatment})
    assert summary["effect"]["pairs"] == 1 and summary["effect"]["status"] == "inconclusive"


def test_a_missing_cell_never_becomes_a_favorable_pair():
    """@brief 少一颗 cell 时不能靠位置滑动得到“有利”结果；默认 max_missing_fraction 为 0。"""
    plan = authorize_plan(_plan(seeds=(7, 11, 13, 17)), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan, control=1.0, treatment=0.5)
    holed = copy.deepcopy(result)
    measurement = holed["treatment"]["first_eligible_cost_usd"]
    measurement["samples"].pop(0)
    measurement["cells"].pop(0)
    effect = paired_effect(holed)
    assert effect["pairs"] == 3 and effect["missing_cells"] == 1
    assert effect["status"] == "inconclusive" and effect["reason"] == "too many missing or invalid pairs"


@pytest.mark.parametrize("defect", ["duplicate", "wrong_arm", "wrong_digest", "undeclared_identity", "no_cells"])
def test_a_defective_cell_is_an_explicit_error(defect):
    """@brief 重复、错 arm、错 digest、无法匹配的 cell 都是显式错误，不是可静默丢弃的样本。"""
    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan)
    measurement = result["control"]["first_eligible_cost_usd"]
    if defect == "duplicate":
        measurement["cells"][1] = copy.deepcopy(measurement["cells"][0])
        message = "repeats the same cell"
    elif defect == "wrong_arm":
        measurement["cells"][0]["arm"] = "treatment"
        message = "belongs to"
    elif defect == "wrong_digest":
        measurement["cells"][0]["run_digest"] = "e" * 64
        message = "digest is not declared"
    elif defect == "undeclared_identity":
        measurement["cells"][0]["seed"] = 999
        message = "not one of the plan's frozen"
    else:
        measurement.pop("cells")
        message = "one cell record per sample"
    with pytest.raises(ExperimentError, match=message):
        paired_effect(result)
    with pytest.raises(ExperimentError, match=message):
        summarize_result(result)


def test_repetitions_do_not_add_independent_pairs():
    """@brief 合法 repetition 只降低噪声，不把独立对数从 3 抬到 6。"""
    plan = authorize_plan(_plan(seeds=(7, 11, 13, 7, 11, 13), repetitions=2), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan, control=1.0, treatment=0.5)
    effect = summarize_result(result)["effect"]
    assert effect["repetitions"] == 2 and effect["pair_count"] == 3
    assert effect["pairs"] == 3 and effect["status"] == "supports_effect"
    assert len(result["control"]["first_eligible_cost_usd"]["samples"]) == 6


def test_latest_effect_verdict_controls_milestones(tmp_path):
    """@brief 同一实验后续 validated=false 必须撤销旧的 true，并绑定结果摘要。"""
    from taic.harness_experiment import milestones

    plan = authorize_plan(_plan(), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    result = _result(plan, control=1.0, treatment=0.5)
    record_effect_validation(tmp_path, result, validated=True, approved_by="bob", reason="paired evidence")
    row = record_effect_validation(tmp_path, result, validated=False, approved_by="bob", reason="later review rejected it")
    assert row["result_sha256"] == digest(result)
    decisions = [json.loads(line) for line in (tmp_path / "harness_decisions.jsonl").read_text().splitlines()]
    assert milestones(decisions, plan["proposal_key"])["experiment_validated"] is False
