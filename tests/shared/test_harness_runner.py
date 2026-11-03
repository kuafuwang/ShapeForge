"""Harness experiment materialization and mock execution; no model or GPU."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.harness_runner import build_cells, execute_plan
from types import SimpleNamespace
from taic import acceptance
from taic.harness_experiment import ExperimentError, authorize_plan, config_template, digest, evaluation_rules, make_plan, materialize_arm, paired_effect, plan_cells, read_plan, toml_document


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "taic").mkdir()
    (repo / "taic" / "__init__.py").write_text("")
    (repo / "experiments").mkdir()
    (repo / "experiments" / "__init__.py").write_text("")
    (repo / "a.txt").write_text("before\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    return repo


PATCH = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-before\n+after\n"


def _plan(repo: Path, *, seeds=(3, 5, 7), repetitions=1):
    row = {"proposal_key": "0123456789abcdef", "proposal": {"surface": "tip", "scope": {"target_id": "apple-m4-max", "task": "conv2d", "dtype": "fp16"}, "action": "Add a tool.", "evidence": {"reproduction": "r", "expected": "e"}}}
    spec = acceptance.make_spec(row)
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    config = {"schema_version": 2, "protocol_version": "3", "task": "conv2d", "dtype": "fp16", "size": "paper", "target": {"id": "apple-m4-max", "backend": "metal", "backend_version": "1"}, "llm": {"reasoning_effort": "high"}, "search": {"budget_usd": 1.0, "experience_index": None}, "committees": [{"name": "first", "members": [{"provider": "p1", "model": "m1", "reasoning_effort": "high"}]}, {"name": "second", "members": [{"provider": "p2", "model": "m2", "reasoning_effort": "low"}]}]}
    template = config_template(config, budget_usd=1.0, base_commit=base)
    return make_plan(spec, change_sha256=__import__("hashlib").sha256(PATCH.encode()).hexdigest(), protocol_version="3", config_sha256=digest(template), seeds=list(seeds), budget_usd=1.0, repetitions=repetitions, config_template_document=template, base_commit=base)


def _fake_collect(plan, run_dir, arm, *, pair_index=None, repetition=None, order=None):
    """A stand-in for the host collector that keeps the cell identity the runner passed.

    ``change_sha256`` mirrors what the real ``collect_arm`` returns after normalising the
    run's empty control digest to the plan's ``None``; the real collector path is covered
    by ``test_harness_experiment.test_real_collector_output_reaches_a_summary``.
    """
    identity = next(cell for cell in plan_cells(plan) if cell["pair_index"] == pair_index and cell["repetition"] == repetition and cell["order"] == order)
    digest_value = "c" * 64 if arm == "control" else "d" * 64
    cell = {"seed": identity["seed"], "repetition": identity["repetition"], "pair_index": identity["pair_index"], "order": identity["order"], "arm": arm, "run_digest": digest_value}

    def measured(value, unit):
        return {"samples": [value], "unit": unit, "run_digests": [digest_value], "cells": [cell]}

    return {
        "config_sha256": plan["config_sha256"], "protocol_version": "3", "seeds": plan["seeds"], "repetitions": plan["repetitions"],
        "change_sha256": plan[arm]["change_sha256"], "spent_usd": 1.0, "session_cost_usd": 1.0, "non_session_cost_usd": 0.0, "model_totals": [],
        "first_eligible_cost_usd": measured(1.0 if arm == "control" else 0.5, "usd"),
        "safety_regressions": measured(0, "count"), "correctness_regressions": measured(0, "count"), "agent_failures": measured(0, "count"),
    }


def test_experiment_result_path_cannot_escape_runs(tmp_path):
    """@brief source.run_id 来自提案，不能把实验事件写到 runs/ 之外。"""
    from experiments.harness_runner import _decisions_dir

    repo = _repo(tmp_path)
    with pytest.raises(ExperimentError, match="source run"):
        _decisions_dir({"acceptance_spec": {"source": {"run_id": "../../tmp"}}}, repo / "runs" / "exp", repo)
    assert _decisions_dir({"acceptance_spec": {"source": {"run_id": "apple-m4-max/run"}}}, repo / "runs" / "exp", repo) == (repo / "runs" / "apple-m4-max" / "run").resolve()


def test_raw_config_keeps_committees_and_no_secrets(tmp_path):
    repo = _repo(tmp_path)
    plan = _plan(repo)
    cell = materialize_arm(plan, "treatment", 3)
    assert cell["raw"]["llm"] == {"reasoning_effort": "high"}
    assert [c["members"][0]["model"] for c in cell["raw"]["committees"]] == ["m1", "m2"]
    assert "api_key" not in cell["raw"]["llm"] and "base_url" not in cell["raw"]["llm"]
    toml = toml_document(cell["raw"])
    assert "[[committees.members]]" in toml
    assert "experience_index" not in toml and "None" not in toml
    assert cell["run_args"]["target_id"] == "apple-m4-max" and cell["run_args"]["seed"] == 3


def test_dry_run_materializes_alternating_arms_from_one_base(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    plan = _plan(repo)
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)
    cells = build_cells(plan, out_dir=out, repo=repo)
    assert [cell["arm"] for cell in cells] == ["control", "treatment", "treatment", "control", "control", "treatment"]
    assert (out / "worktrees" / cells[0]["cell"] / "a.txt").read_text() == "before\n"
    assert (out / "worktrees" / cells[1]["cell"] / "a.txt").read_text() == "after\n"
    assert all(cell["base_commit"] == plan["base_commit"] for cell in cells)
    assert build_cells(plan, out_dir=out, repo=repo) == cells


def test_repeated_seeds_get_frozen_pair_identity_not_array_positions(tmp_path, monkeypatch):
    """@brief ``repetitions=2`` 时每个 cell 的身份来自计划，cell 目录不互相覆盖。"""
    repo = _repo(tmp_path)
    plan = _plan(repo, seeds=(3, 5, 7, 3, 5, 7), repetitions=2)
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)
    cells = build_cells(plan, out_dir=out, repo=repo)
    assert len(cells) == 12 and len({cell["cell"] for cell in cells}) == 12
    assert [(cell["seed"], cell["pair_index"], cell["repetition"]) for cell in cells[::2]] == [
        (3, 0, 0), (5, 1, 0), (7, 2, 0), (3, 0, 1), (5, 1, 1), (7, 2, 1),
    ]
    assert [cell["order"] for cell in cells[::2]] == [0, 1, 2, 3, 4, 5]


def test_live_execution_requires_separate_authorization(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(ExperimentError, match="allow-spend"):
        execute_plan(_plan(repo), out_dir=tmp_path / "exp", repo=repo, runner=lambda *args, **kwargs: None)


def test_paired_statistics_require_three_pairs_and_frozen_effect_direction(tmp_path):
    repo = _repo(tmp_path)
    plan = _plan(repo)

    def measurement(arm, value):
        digest_value = "c" * 64 if arm == "control" else "d" * 64
        cells = [{"seed": cell["seed"], "repetition": cell["repetition"], "pair_index": cell["pair_index"], "order": cell["order"], "arm": arm, "run_digest": digest_value} for cell in plan_cells(plan)]
        values = [value] * len(cells) if not isinstance(value, list) else value
        return {"samples": values, "unit": "usd", "run_digests": [digest_value], "cells": cells[:len(values)]}

    small = {"plan": plan, "control": {"first_eligible_cost_usd": measurement("control", [1.0, 1.0])}, "treatment": {"first_eligible_cost_usd": measurement("treatment", [0.5, 0.5])}}
    assert paired_effect(small)["status"] == "inconclusive"
    large = {"plan": plan, "control": {"first_eligible_cost_usd": measurement("control", 1.0), "safety_regressions": measurement("control", 0), "correctness_regressions": measurement("control", 0)}, "treatment": {"first_eligible_cost_usd": measurement("treatment", 0.5), "safety_regressions": measurement("treatment", 0), "correctness_regressions": measurement("treatment", 0)}}
    assert paired_effect(large)["status"] == "supports_effect"
    large["treatment"]["first_eligible_cost_usd"] = measurement("treatment", 1.5)
    assert paired_effect(large)["status"] == "inconclusive"


def test_mock_execution_uses_each_cell_and_keeps_paired_results(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    plan = authorize_plan(_plan(repo), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)
    monkeypatch.setattr("experiments.harness_runner.collect_arm", _fake_collect)
    calls = []
    def runner(task, dtype, size, run_dir, **kwargs):
        calls.append((task, kwargs["target_id"], kwargs["seed"], kwargs["worktree"]))
        return SimpleNamespace(spent_usd=1.0)
    result = execute_plan(plan, out_dir=out, repo=repo, allow_spend=True, allow_gpu=True, approved_by="alice", runner=runner)
    assert len(calls) == 6 and all(call[0] == "conv2d" for call in calls)
    assert {call[2] for call in calls} == {3, 5, 7}
    assert result["summary"]["effect"]["pairs"] == 3
    assert result["result"]["control"]["change_sha256"] is None
    assert result["result"]["treatment"]["change_sha256"] == plan["treatment"]["change_sha256"]
    assert (out / "result.json").is_file()
    events = [json.loads(line) for line in (out / "harness_decisions.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "experiment_result"
    assert events[-1]["spent_usd"] == 6.0


def test_mock_execution_collapses_repetitions_into_three_pairs(tmp_path, monkeypatch):
    """@brief 六个合法 repetition 仍只形成三对独立对照。"""
    repo = _repo(tmp_path)
    plan = authorize_plan(_plan(repo, seeds=(3, 5, 7, 3, 5, 7), repetitions=2), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)
    monkeypatch.setattr("experiments.harness_runner.collect_arm", _fake_collect)
    result = execute_plan(plan, out_dir=out, repo=repo, allow_spend=True, allow_gpu=True, approved_by="alice", runner=lambda *a, **k: SimpleNamespace(spent_usd=1.0))
    assert len(result["cells"]) == 12 and all(cell["status"] == "completed" for cell in result["cells"])
    effect = result["summary"]["effect"]
    assert effect["pair_count"] == 3 and effect["repetitions"] == 2 and effect["pairs"] == 3
    assert len(result["result"]["control"]["first_eligible_cost_usd"]["samples"]) == 6
    assert len({(cell["pair_index"], cell["seed"], cell["repetition"]) for cell in result["result"]["control"]["first_eligible_cost_usd"]["cells"]}) == 6


def test_budget_refusal_keeps_every_cell_unstarted(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    plan = authorize_plan(_plan(repo), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)
    monkeypatch.setattr("experiments.common.experiment_commitment", lambda _path: 100.0)
    calls = []
    result = execute_plan(plan, out_dir=out, repo=repo, allow_spend=True, allow_gpu=True, approved_by="alice", runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "inconclusive" and calls == []
    assert [row["status"] for row in result["cells"]] == ["over_cap"] * 6


def test_a_failed_cell_can_resume_without_losing_its_previous_attempt(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    plan = authorize_plan(_plan(repo), allow_model_spend=True, allow_gpu=True, approved_by="alice")
    out = repo / "runs" / "exp"
    out.mkdir(parents=True)
    (out / "treatment.patch").write_text(PATCH)
    monkeypatch.setattr("experiments.harness_runner.verify_cell_import", lambda path: None)

    def fail(*_args, **_kwargs):
        raise RuntimeError("cell failed")

    first = execute_plan(plan, out_dir=out, repo=repo, allow_spend=True, allow_gpu=True, approved_by="alice", runner=fail)
    assert first["status"] == "inconclusive" and all(row["status"] == "failed" for row in first["cells"])

    monkeypatch.setattr("experiments.harness_runner.collect_arm", _fake_collect)
    second = execute_plan(plan, out_dir=out, repo=repo, allow_spend=True, allow_gpu=True, approved_by="alice", runner=lambda *a, **k: SimpleNamespace(spent_usd=1.0))
    execution = json.loads((out / "execution.json").read_text())
    assert second["status"] == "supports_effect"
    assert all(row["status"] == "completed" and row["previous_attempts"] for row in execution["cells"])
