"""The batch-size probe and the interleaved timing loop of ``tcenv.bench``. No GPU."""

from __future__ import annotations

import pytest

from tcenv.bench import choose_iters, interleaved


def test_choose_iters_ignores_a_one_time_warmup_cost():
    """@brief 验证批量探测会剔除一次性 JIT 编译开销，不把它当成单次启动耗时。

    第一次调用耗时 1 秒（编译），之后每次约 100 微秒；目标 15 毫秒时应选 150 次。
    """
    calls = {"n": 0}

    def run_k(k: int) -> float:
        calls["n"] += 1
        return (1.0 if calls["n"] == 1 else 100e-6) * k  # first call: 1 s of compilation, afterwards 100 us per launch

    assert choose_iters(run_k, target_s=0.015, kmin=4) == 150


def test_choose_iters_still_uses_single_launches_for_slow_kernels():
    """@brief 验证内核很慢时只按 kmin 下的单次启动取样，不会凑出长时间批量。

    每次启动固定 0.3 秒，探测应返回 kmin=4 且单次请求的 k 始终为 1。
    """
    calls = []

    def run_k(k: int) -> float:
        calls.append(k)
        return 0.3 * k

    assert choose_iters(run_k) == 4 and max(calls) == 1  # kmin buffer sets, never a long probe batch


def test_interleaved_reports_every_runner():
    """@brief 验证交替计时会为每个 runner 分别给出中位耗时统计。

    两个 runner 的耗时分别为 1 毫秒和 2 毫秒，各自的中位数应与之相符。
    """
    st = interleaved({"a": lambda k: 1e-3 * k, "b": lambda k: 2e-3 * k}, reps=4, warmup_reps=1)
    assert st["a"].median_s == pytest.approx(1e-3) and st["b"].median_s == pytest.approx(2e-3)
