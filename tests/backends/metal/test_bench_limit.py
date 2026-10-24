"""A slow candidate must not be turned into a hang by the batch size. No GPU, no model calls.

Every timed batch runs inside one command buffer with one heartbeat, and the parent kills the worker when that
heartbeat is older than ``tcenv.evaluate.LAUNCH_DEADLINE_S``. ``choose_iters`` therefore has to keep
``k * per_launch`` inside the watchdog, including on the slow path where it wants to cover every rotated buffer set.
"""

from __future__ import annotations

from tcenv.backends.metal.baselines import RING_MAX
from tcenv.bench import BATCH_WALL_BUDGET_S, choose_iters
from tcenv.evaluate import LAUNCH_DEADLINE_S


def test_the_batch_budget_keeps_a_margin_under_the_watchdog():
    """@brief 验证批量墙钟预算比看门狗留出至少 20% 余量，正常路径的上界（旋转组数 x 0.1 s）也留在看门狗之内。
    """
    assert BATCH_WALL_BUDGET_S <= 0.8 * LAUNCH_DEADLINE_S  # 至少 20% 余量给编码/提交/心跳写入
    # 正常路径（per <= 0.1 s）的上界是 kmin × 0.1 s，kmin 最大就是旋转组数；它同样要留在看门狗里。
    assert RING_MAX * 0.1 < LAUNCH_DEADLINE_S


def test_a_slow_launch_never_asks_for_a_batch_that_outlives_the_watchdog():
    """@brief 验证单次启动很慢时 choose_iters 给出的批量既不超过 kmin，也不会活过看门狗的时限。
    """
    for per in (0.5, 0.25, 0.11, 0.1, 0.05):
        for kmin in (1, 4, RING_MAX):
            k = choose_iters(lambda n, per=per: per * n, kmin=kmin)
            assert 1 <= k <= kmin
            assert k * per <= BATCH_WALL_BUDGET_S * 1.05, (per, kmin, k)
            assert k * per < LAUNCH_DEADLINE_S


def test_a_slow_kernel_still_rotates_through_the_buffer_sets_when_it_can():
    """@brief 验证慢内核在预算放得下时仍轮流用完全部缓冲区组，放不下才减少次数（0.5 s 一次、6 组只取 3 次）。
    """
    # 0.3 s 一次、4 组：一批 4 次共 1.2 s，仍在预算内，所以不该退化到 1。
    assert choose_iters(lambda n: 0.3 * n, kmin=4) == 4
    # 0.2 s 一次、6 组：一批 6 次 1.2 s，也放得下。
    assert choose_iters(lambda n: 0.2 * n, kmin=6) == 6
    # 0.5 s 一次、6 组：一批 6 次要 3 秒，越界；1.6 秒的预算只放得下 3 次（1.5 秒）。
    assert choose_iters(lambda n: 0.5 * n, kmin=6) == 3


def test_a_single_repetition_is_marked_noisy_not_infinitely_precise():
    """@brief 验证只测一次时变异系数被标成有噪声而不是 0，单个样本不能因此晋级种子。

    过去 n=1 时 cv=0.0，噪声闸门退化成只看 min_improvement。
    """
    from tcenv.bench import interleaved

    stats = interleaved({"candidate": lambda k: 0.01 * k}, reps=1, warmup_reps=0, target_s=0.001, kmin=1)
    assert stats["candidate"].reps == 1
    assert stats["candidate"].cv >= 1.0
