"""The self-check budget: a workspace may time a kernel a bounded number of times. No GPU, no model calls.

Each `--timed` runs 15 interleaved repetitions, about 0.2-1 s of GPU, so the cap is what keeps an agent from
turning the timing loop into its training set. The numbers are a budget, not a performance target.
"""

from __future__ import annotations

import json

import pytest

from tcenv.selfcheck import LOG_NAME, MAX_CHECKS, MAX_GPU_S, append, authority_log, budget_reason, earlier_hang


def _log(directory, rows) -> None:
    """@brief 把给定的记录行按 JSON Lines 写入目录下的自测日志。

    @param directory 写入目标目录，需已存在。
    @param rows 要序列化的记录字典列表。
    """
    (directory / LOG_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_append_creates_a_missing_directory(tmp_path):
    """@brief 验证向不存在的会话检查目录追加记录时会自动建目录并可被读取。

    @param tmp_path pytest 提供的临时目录。
    """
    missing = tmp_path / "session_checks" / "r1.p0"
    append(missing, {"status": "ok", "elapsed_s": 0.2})
    assert (missing / LOG_NAME).is_file()
    assert earlier_hang(missing) is None


def test_a_fresh_workspace_may_time(tmp_path):
    """@brief 验证全新工作区没有任何预算限制，也没有挂起记录。

    @param tmp_path pytest 提供的临时目录。
    """
    assert budget_reason(tmp_path) is None and earlier_hang(tmp_path) is None


def test_the_check_count_is_capped(tmp_path):
    """@brief 验证检查次数达到 MAX_CHECKS 后预算被锁，并在原因里给出上限与交还内核的提示。

    @param tmp_path pytest 提供的临时目录。
    """
    _log(tmp_path, [{"status": "ok", "elapsed_s": 1.0}] * (MAX_CHECKS - 1))
    assert budget_reason(tmp_path) is None
    _log(tmp_path, [{"status": "ok", "elapsed_s": 1.0}] * MAX_CHECKS)
    reason = budget_reason(tmp_path)
    assert reason is not None and str(MAX_CHECKS) in reason and "hand the kernel back" in reason


def test_spent_gpu_seconds_are_capped_even_under_the_count(tmp_path):
    """@brief 验证即使检查次数未达上限，累计 GPU 秒数超限也会锁住预算并在原因里报出秒数上限。

    @param tmp_path pytest 提供的临时目录。
    """
    per = MAX_GPU_S / 4 + 1.0  # 4 checks, over the seconds budget
    _log(tmp_path, [{"status": "ok", "elapsed_s": per}] * 4)
    reason = budget_reason(tmp_path)
    assert reason is not None and f"{MAX_GPU_S:.0f}s" in reason


def test_a_hang_is_reported_before_the_budget(tmp_path):
    """@brief 验证超时（挂起）记录会被单独报出，而单次挂起不会立刻锁住预算。

    @param tmp_path pytest 提供的临时目录。
    """
    _log(tmp_path, [{"status": "timeout", "elapsed_s": 2.0}])
    assert "timeout" in (earlier_hang(tmp_path) or "")
    assert budget_reason(tmp_path) is None


def test_damaged_lines_do_not_unlock_the_budget(tmp_path):
    """@brief 验证日志里的损坏行不会重置计数，已达上限时预算仍被锁住。

    @param tmp_path pytest 提供的临时目录。
    """
    (tmp_path / LOG_NAME).write_text("not json\n" + "".join(json.dumps({"status": "ok", "elapsed_s": 1.0}) + "\n" for _ in range(MAX_CHECKS)))
    assert budget_reason(tmp_path) is not None


def test_mirror_failure_cannot_erase_the_host_record(tmp_path):
    """@brief agent-writable mirror 打开失败时，已发生的 timeout 仍先写入 host。"""
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / LOG_NAME).mkdir()
    append(tmp_path, {"status": "timeout", "elapsed_s": 1.0})
    assert "timeout" in (earlier_hang(tmp_path) or "")
    rows = [json.loads(line) for line in authority_log(tmp_path).read_text().splitlines()]
    assert rows == [{"status": "timeout", "elapsed_s": 1.0}]


def test_symlinked_or_nondirectory_mirror_cannot_redirect_authority(tmp_path):
    """@brief .selfcheck 被换成 symlink 或普通文件后，host key 与 timeout 证据仍不变。"""
    checks = tmp_path / ".selfcheck"
    append(checks, {"status": "timeout", "elapsed_s": 1.0})
    authority = authority_log(checks)
    mirror_target = tmp_path / "elsewhere"
    mirror_target.mkdir()
    checks.rename(mirror_target / "old")
    checks.symlink_to(mirror_target, target_is_directory=True)
    assert authority_log(checks) == authority
    assert "timeout" in (earlier_hang(checks) or "")
    append(checks, {"status": "ok", "elapsed_s": 2.0})
    assert not (mirror_target / LOG_NAME).exists()

    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    append(blocked, {"status": "timeout", "elapsed_s": 2.0})
    assert "timeout" in (earlier_hang(blocked) or "")


def test_append_keeps_the_workspace_row_when_host_cache_is_unwritable(tmp_path, monkeypatch):
    """@brief 围栏内 --timed 不能因为 host cache 不可写而丢掉已完成的结果。"""
    from tcenv import selfcheck

    host_file = tmp_path / "host-cache-is-a-file"
    host_file.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(selfcheck, "_authority_root", lambda: host_file)
    checks = tmp_path / ".selfcheck"
    append(checks, {"status": "ok", "elapsed_s": 1.0})
    assert (checks / LOG_NAME).is_file()
    assert "ok" in (checks / LOG_NAME).read_text(encoding="utf-8")


def test_host_record_survives_workspace_log_tampering(tmp_path):
    """@brief agent 清空 workspace 日志也不能重置 host 侧自测配额或 hang 锁。"""
    for _ in range(MAX_CHECKS):
        append(tmp_path, {"status": "ok", "elapsed_s": 1.0})
    append(tmp_path, {"status": "timeout", "elapsed_s": 1.0})
    authority = authority_log(tmp_path)
    assert authority.is_file() and authority.stat().st_mode & 0o077 == 0
    _log(tmp_path, [])
    assert budget_reason(tmp_path) is not None
    assert "timeout" in (earlier_hang(tmp_path) or "")


def test_invalid_gpu_time_fails_closed(tmp_path):
    """@brief NaN/Infinity/负数的 GPU 秒数都不能被当作未超预算。"""
    _log(tmp_path, [{"status": "ok", "elapsed_s": float("nan")}])
    assert budget_reason(tmp_path) is not None
    _log(tmp_path, [{"status": "ok", "elapsed_s": float("inf")}])
    assert budget_reason(tmp_path) is not None
    _log(tmp_path, [{"status": "ok", "elapsed_s": -1.0}])
    assert budget_reason(tmp_path) is not None


def test_environment_cannot_redirect_host_quota_into_workspace(tmp_path, monkeypatch):
    """@brief 被测进程设置 TCENV_SELFCHECK_HOME 也不能把权威记录重定向到工作区。"""
    monkeypatch.setenv("TCENV_SELFCHECK_HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    for _ in range(MAX_CHECKS):
        append(tmp_path, {"status": "ok", "elapsed_s": 1.0})
    authority = authority_log(tmp_path)
    assert not authority.is_relative_to(tmp_path)
    _log(tmp_path, [])
    assert budget_reason(tmp_path) is not None
