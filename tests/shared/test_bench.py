"""The batch-size probe and the interleaved timing loop of ``tcenv.bench``. No GPU."""

from __future__ import annotations

import pytest

from tcenv.bench import choose_iters, current_bench_call, interleaved, note_candidate_gpu


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


def test_interleaved_calibrates_an_aliased_aa_runner_once(monkeypatch):
    """@brief 同一 callable 的两个名字必须共用批量 k，避免固定提交开销伪造 A/A 偏差。"""
    calibrations = []

    def fake_choose(run_k, target_s, kmin):
        calibrations.append(run_k)
        # 旧路径会对同一 callable 返回 3 和 7；共享校准后第二个返回值不再被使用。
        return 3 if len(calibrations) == 1 else 7

    monkeypatch.setattr("tcenv.bench.choose_iters", fake_choose)

    def run_k(k):
        # 每次调用包含固定命令缓冲开销。只有相同 k 才能比较同一个每迭代物理量。
        return 1e-3 * k + 2e-3

    stats = interleaved(
        {"candidate": run_k, "candidate_copy_control": run_k},
        reps=4, warmup_reps=0, seed=7,
    )
    assert len(calibrations) == 1
    assert stats["candidate"].iters == stats["candidate_copy_control"].iters == 3
    assert stats["candidate"].samples_s == stats["candidate_copy_control"].samples_s
    assert stats["candidate"].median_s / stats["candidate_copy_control"].median_s == pytest.approx(1.0)


def test_interleaved_reports_every_runner():
    """@brief 验证交替计时会为每个 runner 分别给出中位耗时统计。

    两个 runner 的耗时分别为 1 毫秒和 2 毫秒，各自的中位数应与之相符。
    """
    st = interleaved({"a": lambda k: 1e-3 * k, "b": lambda k: 2e-3 * k}, reps=4, warmup_reps=1)
    assert st["a"].median_s == pytest.approx(1e-3) and st["b"].median_s == pytest.approx(2e-3)


def test_interleaved_marks_calibration_warmup_and_counted_samples():
    """@brief 校准、预热和计入统计的轮次都带阶段标记，对照臂不会冒充候选。"""
    seen = []

    def run_k(k):
        call = current_bench_call()
        seen.append(None if call is None else (call.arm, call.phase, int(k)))
        return 1e-4 * k

    stats = interleaved(
        {"candidate": run_k, "candidate_copy_control": run_k},
        reps=3, warmup_reps=2, seed=1, kmin=1, target_s=0.0004,
    )
    phases = [item[1] for item in seen if item is not None]
    assert "calibration" in phases
    assert "warmup" in phases
    assert phases.count("sample") >= 3
    counted = [item for item in seen if item is not None and item[1] == "sample"]
    assert {item[0] for item in counted} == {"candidate", "candidate_copy_control"}
    assert sum(1 for item in counted if item[0] == "candidate") == 3
    assert len(stats["candidate"].samples_s) == 3
    assert current_bench_call() is None


def test_note_candidate_gpu_drops_calibration_warmup_other_arms_and_zeros():
    """@brief 旧实现会把校准、预热、对照臂和 0 写进中位数；现在这些都不留下。"""
    samples: list[float] = []

    def run_k(k):
        call = current_bench_call()
        if call is None or call.phase == "calibration":
            note_candidate_gpu(samples, 1000.0)
        elif call.phase == "warmup":
            note_candidate_gpu(samples, 2000.0)
        elif call.arm != "candidate":
            note_candidate_gpu(samples, 3000.0)
        else:
            note_candidate_gpu(samples, 3e-6)
            note_candidate_gpu(samples, 0.0)
        return 1e-4 * k

    interleaved(
        {"candidate": run_k, "mlx": lambda k: 2e-4 * k},
        reps=4, warmup_reps=2, seed=3, kmin=1, target_s=0.0004,
    )
    assert samples == pytest.approx([3e-6] * 4)
    assert 0.0 not in samples
    assert 1000.0 not in samples and 2000.0 not in samples
