"""The experiment cap counts what is on disk: a live run reserves its current budget, a finished run counts what it spent."""

from __future__ import annotations

import json

import pytest

from experiments.common import cap_allows, experiment_commitment, fits_cap, run_commitment, write_cap


def _state(directory, *, spent, budget, pid, script=False):
    directory.mkdir(parents=True, exist_ok=True)
    doc = {"script": "run_suite", "phase": "run"} if script else {"spent_usd": spent, "budget_usd": budget, "pid": pid, "phase": "llm"}
    (directory / "state.json").write_text(json.dumps(doc))


def test_a_finished_run_counts_what_it_spent_and_a_job_file_counts_nothing(tmp_path):
    _state(tmp_path, spent=0, budget=0, pid=1, script=True)
    run = tmp_path / "relu_fp16"
    _state(run, spent=2.5, budget=15, pid=999999)
    (run / "result.json").write_text(json.dumps({"spent_usd": 2.5, "budget_usd": 15, "stop_reason": "converged"}))
    assert run_commitment(run) == pytest.approx(2.5)
    assert experiment_commitment(tmp_path) == pytest.approx(2.5)


def test_a_live_run_reserves_its_budget_including_a_top_up(tmp_path, monkeypatch):
    import taic.control as control

    monkeypatch.setattr(control, "pid_alive", lambda pid: pid == 7)
    run = tmp_path / "trial_3"
    _state(run, spent=1.2, budget=15, pid=7)
    (run / "control.json").write_text(json.dumps({"budget_usd": 30}))
    (run / "resume.lock").write_text(json.dumps({"pid": 7}))
    assert run_commitment(run) == pytest.approx(30)


def test_a_dead_run_without_a_result_counts_only_what_it_spent(tmp_path, monkeypatch):
    import taic.control as control

    monkeypatch.setattr(control, "pid_alive", lambda pid: False)
    run = tmp_path / "trial_2"
    _state(run, spent=4.2, budget=15, pid=1)
    assert run_commitment(run) == pytest.approx(4.2)


def test_a_top_up_past_the_cap_is_refused_and_one_inside_it_is_allowed(tmp_path, monkeypatch):
    import taic.control as control

    monkeypatch.setattr(control, "pid_alive", lambda pid: pid == 7)
    write_cap(tmp_path, 20)
    live = tmp_path / "conv2d" / "trial_3"
    _state(live, spent=1.0, budget=15, pid=7)
    done = tmp_path / "conv2d" / "trial_1"
    _state(done, spent=3.0, budget=3, pid=1)
    (done / "result.json").write_text(json.dumps({"spent_usd": 3.0, "budget_usd": 3}))
    assert experiment_commitment(tmp_path) == pytest.approx(18)  # 3 spent plus the live $15 reservation
    ok, why = cap_allows(live, 30)
    assert not ok and "20.00" in why  # raising the live run to $30 reserves 33
    ok, why = cap_allows(live, 16)
    assert ok and why == ""  # 3 + 16 = 19, still under 20
    assert fits_cap(experiment_commitment(tmp_path), 15, 20) is False
