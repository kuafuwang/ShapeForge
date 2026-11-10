"""Sweep ranking with an injected eval. No GPU and no model."""

from __future__ import annotations

from pathlib import Path

import pytest

from taic.kernel_agent import SessionBrief, _sweep_command, render_contract
from tcenv.cli import main
from tcenv.selfcheck import MAX_CHECKS, append, earlier_hang
from tcenv.bench import approximate_speedup_se
from tcenv.sweep import (
    SweepError, SweepObservation, SweepResult, candidate_files, format_table, observation_from_report, run_sweep,
)


def _obs(name, status="ok", median=10.0, cv=0.01, speedup=1.0, elapsed=0.1):
    return SweepObservation(name, status, median, cv, speedup, elapsed)


def _with_se(row, cv, reps=3):
    row.cv = cv
    row.speedup_se = approximate_speedup_se(cv, reps, cv, reps)
    return row


def test_rank_orders_by_speedup_and_marks_a_noisy_pair(tmp_path):
    """@brief 名次按加速比。噪声用加速比标准误；缺标准误的行标未知，不和别人结成噪声对。"""
    close = approximate_speedup_se(0.05, 3, 0.05, 3)
    names = ["c.metal", "a.metal", "b.metal", "d.metal", "bad.metal"]
    table = {
        "c.metal": _with_se(_obs("c.metal", median=10.0, speedup=2.0), 0.01),
        "a.metal": _with_se(_obs("a.metal", median=10.2, speedup=1.11), 0.05),
        "b.metal": _with_se(_obs("b.metal", median=10.05, speedup=1.10), 0.05),
        "d.metal": _obs("d.metal", median=9.0, cv=None, speedup=1.5),
        "bad.metal": _obs("bad.metal", status="incorrect", median=None, cv=None, speedup=None),
    }
    calls = []

    def evaluate(path: Path):
        calls.append(path.name)
        return table[path.name]

    result = run_sweep([tmp_path / name for name in names], evaluate, directory=tmp_path, noise_z=2.0, min_improvement=1.005)
    assert [row.name for row in result.rows] == ["c.metal", "d.metal", "a.metal", "b.metal", "bad.metal"]
    assert ("a.metal", "b.metal") in result.noise_pairs
    assert ("a.metal", "c.metal") not in result.noise_pairs
    assert ("c.metal", "d.metal") not in result.noise_pairs
    text = format_table(result)
    d_line = next(line for line in text.splitlines() if line.startswith("d.metal\t"))
    assert d_line.endswith("\tunknown")
    assert close == pytest.approx(table["a.metal"].speedup_se)
    assert calls == names


def test_more_than_sixteen_files_are_refused_before_eval(tmp_path):
    """@brief 超过 16 个文件时一份都不评。"""
    paths = []
    for i in range(17):
        path = tmp_path / f"k{i:02d}.metal"
        path.write_text("kernel void k() {}\n")
        paths.append(path)
    called = []
    with pytest.raises(SweepError, match="16"):
        run_sweep(paths, lambda path: called.append(path) or _obs(path.name), directory=tmp_path)
    assert called == []
    hidden = tmp_path / ".hidden.metal"
    hidden.write_text("nope\n")
    (tmp_path / "NOTES.md").write_text("notes\n")
    assert ".hidden.metal" not in {path.name for path in candidate_files(tmp_path)}
    assert "NOTES.md" not in {path.name for path in candidate_files(tmp_path)}


def test_a_timeout_stops_the_sweep_and_a_later_call_is_refused(tmp_path):
    """@brief timeout 记下之后不再评后面的文件；同一目录再开一次会因 earlier hang 被拒绝。"""
    for name in ("a.metal", "b.metal", "c.metal"):
        (tmp_path / name).write_text("kernel void k() {}\n")
    calls = []

    def evaluate(path: Path):
        calls.append(path.name)
        if path.name == "b.metal":
            return _obs(path.name, status="timeout", median=None, cv=None, speedup=None)
        return _obs(path.name, speedup=1.0)

    result = run_sweep(candidate_files(tmp_path), evaluate, directory=tmp_path)
    assert calls == ["a.metal", "b.metal"]
    assert result.stopped == "timeout"
    with pytest.raises(SweepError, match="timeout"):
        run_sweep(candidate_files(tmp_path), lambda path: calls.append("again"), directory=tmp_path)


def test_twenty_existing_lines_block_the_first_eval(tmp_path):
    """@brief 目录里已经有 20 行计时记录时，第一份文件也不评。"""
    path = tmp_path / "a.metal"
    path.write_text("kernel void k() {}\n")
    for _ in range(MAX_CHECKS):
        append(tmp_path, {"status": "ok", "elapsed_s": 0.01, "candidate_us": 1.0, "cv": 0.01, "speedup": 1.0})
    called = []
    with pytest.raises(SweepError, match="20"):
        run_sweep([path], lambda item: called.append(item), directory=tmp_path)
    assert called == []


def test_cli_rejects_seventeen_files_without_evaluating(tmp_path):
    """@brief ``sweep-v2`` 在绑定评测之前拒绝 17 个文件，退出码 2。"""
    for i in range(17):
        (tmp_path / f"k{i:02d}.metal").write_text("kernel void k() {}\n")
    assert main(["sweep-v2", "gemm", str(tmp_path)]) == 2


def test_the_session_quota_is_not_the_candidate_directory(tmp_path):
    """@brief 会话 ``.selfcheck`` 已有 20 行时，另一目录里的候选一份都不评。"""
    cand = tmp_path / "cands"
    cand.mkdir()
    (cand / "a.metal").write_text("kernel void k() {}\n")
    quota = tmp_path / ".selfcheck"
    quota.mkdir()
    for _ in range(MAX_CHECKS):
        append(quota, {"status": "ok", "elapsed_s": 0.01, "candidate_us": 1.0, "cv": 0.01, "speedup": 1.0})
    called = []
    with pytest.raises(SweepError, match="20"):
        run_sweep(candidate_files(cand), lambda path: called.append(path.name) or _obs(path.name), directory=quota)
    assert called == []
    assert earlier_hang(cand) is None


def test_a_timeout_is_recorded_on_the_shared_hang_lock(tmp_path):
    """@brief sweep 记下的 timeout 落在配额目录。之后同一把锁拒绝计时，候选目录本身没有挂死记录。"""
    cand = tmp_path / "cands"
    cand.mkdir()
    for name in ("a.metal", "b.metal"):
        (cand / name).write_text("kernel void k() {}\n")
    quota = tmp_path / ".selfcheck"
    quota.mkdir()
    calls = []

    def evaluate(path: Path):
        calls.append(path.name)
        return _obs(path.name, status="timeout", median=None, cv=None, speedup=None)

    result = run_sweep(candidate_files(cand), evaluate, directory=quota)
    assert calls == ["a.metal"]
    assert result.stopped == "timeout"
    assert earlier_hang(quota) is not None
    assert earlier_hang(cand) is None
    with pytest.raises(SweepError, match="timeout"):
        run_sweep(candidate_files(cand), lambda path: calls.append("again"), directory=quota)
    assert calls == ["a.metal"]


def test_cli_shares_the_session_quota_and_stops_with_exit_one(tmp_path, monkeypatch):
    """@brief 满配额在绑定评测之前退出 1。timeout 或 crash 打出表之后也是 1。"""
    cand = tmp_path / "cands"
    cand.mkdir()
    (cand / "a.metal").write_text("kernel void k() {}\n")
    quota = tmp_path / ".selfcheck"
    quota.mkdir()
    for _ in range(MAX_CHECKS):
        append(quota, {"status": "ok", "elapsed_s": 0.01, "candidate_us": 1.0, "cv": 0.01, "speedup": 1.0})
    assert main(["sweep-v2", "gemm", str(cand), "--self-check-dir", str(quota), "--reps", "7"]) == 1
    assert not (cand / "self_checks.jsonl").exists()

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "a.metal").write_text("kernel void k() {}\n")

    def stopped(status):
        def run(paths, evaluate, **kwargs):
            del paths, evaluate, kwargs
            return SweepResult([_obs("a.metal", status=status, median=None, cv=None, speedup=None)], [], status)
        return run

    monkeypatch.setattr("tcenv.sweep.run_sweep", stopped("timeout"))
    assert main(["sweep-v2", "gemm", str(fresh), "--self-check-dir", str(tmp_path / "quota-timeout")]) == 1
    monkeypatch.setattr("tcenv.sweep.run_sweep", stopped("crash"))
    assert main(["sweep-v2", "gemm", str(fresh), "--self-check-dir", str(tmp_path / "quota-crash")]) == 1


def test_the_sweep_command_uses_the_session_selfcheck_and_reps(tmp_path):
    """@brief 任务书里的 sweep 命令带本场 ``.selfcheck`` 和本场 ``--reps``，候选目录仍是占位符。"""
    work = tmp_path / "session"
    brief = SessionBrief(
        repo=tmp_path, task_name="gemm", dtype="fp16", size="small", kernel_name="gemm_kernel",
        triton_source="", description="gemm", contract_text="grid 1,1,1", abi_stub="kernel void gemm_kernel() {}",
        rtol=1e-3, atol_rel=1e-3, reps=7, work_dir=str(work), sweep_tool=True,
    )
    command = _sweep_command(brief)
    quota = str((work / ".selfcheck").resolve())
    assert "--self-check-dir" in command and quota in command
    assert "--reps" in command and "7" in command
    assert "--noise-z" in command and "--min-improvement" in command
    assert "<候选目录>" in command and quota not in command.split("<候选目录>")[0]
    text = render_contract(brief)
    assert quota in text and "--reps" in text and "7" in text
    assert "--noise-z" in text
    assert "本场 `.selfcheck`" in text


def test_missing_speedup_statistics_stay_unknown():
    """@brief 报告里 cv 和重复次数齐全才有标准误。缺任何一项时标准误是 None。"""

    class Complete:
        status = "ok"
        speedup = 1.1
        candidate_us = 10.0
        benchmark = {
            "candidate": {"median_us": 10.0, "cv": 0.05, "reps": 3},
            "baselines": {"mlx": {"median_us": 20.0, "cv": 0.05, "reps": 3}},
            "best_baseline": "mlx",
        }

    class Missing:
        status = "ok"
        speedup = 1.1
        candidate_us = 10.0
        benchmark = {"candidate": {"median_us": 10.0, "cv": 0.05}}

    found = observation_from_report("a.metal", Complete(), 0.2)
    assert found.speedup_se == pytest.approx(approximate_speedup_se(0.05, 3, 0.05, 3))
    assert observation_from_report("b.metal", Missing(), 0.2).speedup_se is None


def test_quota_on_the_next_file_keeps_the_finished_ranking(tmp_path):
    """@brief 第 20 行之后的下一份触发配额时，已经评完的那份仍返回并打出部分名次。"""
    for _ in range(MAX_CHECKS - 1):
        append(tmp_path, {"status": "ok", "elapsed_s": 0.01, "candidate_us": 1.0, "cv": 0.01, "speedup": 1.0})
    first = tmp_path / "a.metal"
    second = tmp_path / "b.metal"
    first.write_text("kernel void k() {}\n")
    second.write_text("kernel void k() {}\n")
    calls = []

    def evaluate(path: Path):
        calls.append(path.name)
        return _with_se(_obs(path.name, speedup=1.2), 0.01)

    result = run_sweep([first, second], evaluate, directory=tmp_path)
    assert calls == ["a.metal"]
    assert result.stopped == "quota"
    assert [row.name for row in result.rows] == ["a.metal"]
    assert "# stopped after quota" in format_table(result)


def test_cli_prints_the_partial_ranking_when_quota_stops_mid_sweep(tmp_path, monkeypatch, capsys):
    """@brief 配额中途停下时，命令先打出已完成的表，再以 1 退出。"""
    cand = tmp_path / "cands"
    cand.mkdir()
    (cand / "a.metal").write_text("kernel void k() {}\n")

    def partial(paths, evaluate, **kwargs):
        del paths, evaluate, kwargs
        return SweepResult([_obs("a.metal", speedup=1.2)], [], "quota")

    monkeypatch.setattr("tcenv.sweep.run_sweep", partial)
    assert main(["sweep-v2", "gemm", str(cand)]) == 1
    printed = capsys.readouterr().out
    assert "a.metal" in printed
    assert "# stopped after quota" in printed


def test_a_host_bridge_session_does_not_offer_sweep_v2(tmp_path):
    """@brief 远端会话的计时走 host bridge。任务书拒绝 sweep-v2，不写出那条本地命令。"""
    work = tmp_path / "session"
    brief = SessionBrief(
        repo=tmp_path, task_name="gemm", dtype="fp16", size="small", kernel_name="gemm_kernel",
        triton_source="", description="gemm", contract_text="grid 1,1,1", abi_stub="kernel void gemm_kernel() {}",
        rtol=1e-3, atol_rel=1e-3, reps=7, work_dir=str(work), sweep_tool=True,
        selfcheck_socket=str(tmp_path / "bridge.sock"), selfcheck_token_file=str(tmp_path / "token"),
    )
    text = render_contract(brief)
    assert "tcenv.cli sweep-v2" not in text
    assert "taic.selfcheck_bridge" in text
    assert "host bridge" in text
    assert "不提供" in text
