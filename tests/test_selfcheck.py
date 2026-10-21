"""The self-check budget: a workspace may time a kernel a bounded number of times. No GPU, no model calls.

Each `--timed` runs 15 interleaved repetitions, about 0.2-1 s of GPU, so the cap is what keeps an agent from
turning the timing loop into its training set. The numbers are a budget, not a performance target.
"""

from __future__ import annotations

import json

from tcenv.selfcheck import LOG_NAME, MAX_CHECKS, MAX_GPU_S, budget_reason, earlier_hang


def _log(directory, rows) -> None:
    (directory / LOG_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_a_fresh_workspace_may_time(tmp_path):
    assert budget_reason(tmp_path) is None and earlier_hang(tmp_path) is None


def test_the_check_count_is_capped(tmp_path):
    _log(tmp_path, [{"status": "ok", "elapsed_s": 1.0}] * (MAX_CHECKS - 1))
    assert budget_reason(tmp_path) is None
    _log(tmp_path, [{"status": "ok", "elapsed_s": 1.0}] * MAX_CHECKS)
    reason = budget_reason(tmp_path)
    assert reason is not None and str(MAX_CHECKS) in reason and "hand the kernel back" in reason


def test_spent_gpu_seconds_are_capped_even_under_the_count(tmp_path):
    per = MAX_GPU_S / 4 + 1.0  # 4 checks, over the seconds budget
    _log(tmp_path, [{"status": "ok", "elapsed_s": per}] * 4)
    reason = budget_reason(tmp_path)
    assert reason is not None and f"{MAX_GPU_S:.0f}s" in reason


def test_a_hang_is_reported_before_the_budget(tmp_path):
    _log(tmp_path, [{"status": "timeout", "elapsed_s": 2.0}])
    assert "timeout" in (earlier_hang(tmp_path) or "")
    assert budget_reason(tmp_path) is None


def test_damaged_lines_do_not_unlock_the_budget(tmp_path):
    (tmp_path / LOG_NAME).write_text("not json\n" + "".join(json.dumps({"status": "ok", "elapsed_s": 1.0}) + "\n" for _ in range(MAX_CHECKS)))
    assert budget_reason(tmp_path) is not None
