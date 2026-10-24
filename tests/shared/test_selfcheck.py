"""The self-check budget: a workspace may time a kernel a bounded number of times. No GPU, no model calls.

Each `--timed` runs 15 interleaved repetitions, about 0.2-1 s of GPU, so the cap is what keeps an agent from
turning the timing loop into its training set. The numbers are a budget, not a performance target.
"""

from __future__ import annotations

import json

from tcenv.selfcheck import LOG_NAME, MAX_CHECKS, MAX_GPU_S, append, budget_reason, earlier_hang


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
