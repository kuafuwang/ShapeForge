"""CPU synthesis for the candidate GPU-median window.

The old collector appended every launch, then kept the last ``candidate.reps``
entries. That tail is the end of the last batch on the Host path, a mix of the
control arm when ``--control`` is on, and — on v1 — the calibration and warmup
calls. These tests fail on that collector.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from tcenv.backends.metal.runtime import RunTiming
from tcenv.backends.metal.v2_runtime import BENCH_CONTROL, MetalV2Runtime, V2Benchmark, _control_document
from tcenv.bench import BENCH_ATTEMPTS, EXTENDED_ATTEMPTS, current_bench_call, interleaved
from tcenv.contract_errors import UnsupportedContractCapability
from tcenv.evaluate import _gpu_median_us


def _runtime() -> MetalV2Runtime:
    runtime = object.__new__(MetalV2Runtime)
    runtime._attempt_gpu = []
    runtime._gpu_collect = False
    runtime._candidate_gpu_only = False
    runtime._selected_gpu_per_iter_s = ()
    runtime._bench_base_attempts = None
    runtime._bench_extended_attempts = None
    return runtime


def _host_style(runtime, *, poison: float = 80.0):
    """One GPU reading per launch, the way the Host path calls ``run_once``."""

    def cand(k: int) -> float:
        call = current_bench_call()
        counted = call is not None and call.arm == "candidate" and call.phase == "sample"
        for index in range(int(k)):
            runtime._record_attempt_gpu((0.001 * (index + 1)) if counted else poison)
        return 0.2 * k

    return cand


def test_host_style_attempt_keeps_every_counted_iteration_not_the_tail():
    """@brief k 次 launch 都要留下。旧切片只剩最后一批的最后 reps 次。"""
    runtime = _runtime()
    _noise, stats, attempts = runtime._measure(
        _host_style(runtime), {}, seed=1, reps=3, rotation=4,
    )
    selected = runtime._selected_gpu_per_iter_s
    assert attempts == 1
    assert selected.count(0.001) == 3
    assert len(selected) == 3 * 4
    assert all(value < 1.0 for value in selected)
    assert stats["candidate"].reps == 3
    assert _gpu_median_us(SimpleNamespace(gpu_per_iter_s=selected)) == pytest.approx(0.0025 * 1e6)


def test_control_keeps_the_candidate_gpu_samples_and_wall_clock_ratios():
    """@brief 打开对照不改变候选 GPU 样本；对照文档仍是各臂墙钟，并标明 timed_by。"""
    alone = _runtime()
    _noise, alone_stats, _attempts = alone._measure(
        _host_style(alone), {}, seed=5, reps=3, rotation=4, control=False,
    )
    both = _runtime()
    _noise, stats, _attempts = both._measure(
        _host_style(both), {}, seed=5, reps=3, rotation=4, control=True,
    )
    assert both._selected_gpu_per_iter_s == alone._selected_gpu_per_iter_s
    assert 80.0 not in both._selected_gpu_per_iter_s
    assert len(both._selected_gpu_per_iter_s) == len(alone._selected_gpu_per_iter_s) == 12
    document = _control_document(stats, seed=5)
    assert document is not None
    assert document["timed_by"] == "wall_clock"
    assert document["median_candidate_s"] == pytest.approx(0.2)
    assert document["median_control_s"] == pytest.approx(0.2)
    assert _control_document(alone_stats, seed=5) is None
    gpu_median = _gpu_median_us(SimpleNamespace(gpu_per_iter_s=both._selected_gpu_per_iter_s))
    assert gpu_median != pytest.approx(document["median_candidate_s"] * 1e6)


def test_missing_gpu_samples_are_omitted_instead_of_zero():
    """@brief 0 不补进样本。整段都缺失时中位数保持 None。"""
    runtime = _runtime()

    def cand(k: int) -> float:
        for index in range(int(k)):
            runtime._record_attempt_gpu(0.0 if index == 0 else 0.002)
        return 0.2 * k

    runtime._measure(cand, {}, seed=1, reps=3, rotation=4)
    selected = runtime._selected_gpu_per_iter_s
    assert 0.0 not in selected
    assert len(selected) == 3 * 3
    assert all(value == pytest.approx(0.002) for value in selected)

    empty = _runtime()

    def nothing(k: int) -> float:
        for _ in range(int(k)):
            empty._record_attempt_gpu(0.0)
        return 0.2 * k

    empty._measure(nothing, {}, seed=1, reps=3, rotation=4)
    assert empty._selected_gpu_per_iter_s == ()
    assert _gpu_median_us(SimpleNamespace(gpu_per_iter_s=())) is None


def test_collection_outside_the_candidate_window_still_keeps_positive_samples():
    """@brief ab / paired-compare 不打开候选过滤，校准阶段的正数样本仍保留。"""
    from tcenv.bench import BenchCall, _BENCH_CALL

    runtime = _runtime()
    runtime._gpu_collect = True
    token = _BENCH_CALL.set(BenchCall("kernel.metal", "calibration"))
    try:
        runtime._record_attempt_gpu(4e-6)
        runtime._record_attempt_gpu(0.0)
        assert runtime._attempt_gpu == [4e-6]
        runtime._candidate_gpu_only = True
        runtime._record_attempt_gpu(4e-6)
        assert runtime._attempt_gpu == [4e-6]
    finally:
        _BENCH_CALL.reset(token)
    runtime._record_attempt_gpu(5e-6)
    assert runtime._attempt_gpu == [4e-6]


def test_prewarmup_runs_once_per_callable_and_does_not_reorder_samples():
    """@brief 同一 callable 的前置预热只跑一次；计入统计的墙钟顺序不变。"""
    runtime = _runtime()
    direct = {"candidate": [], "mlx": []}
    counted: list[float] = []

    def cand(k: int) -> float:
        call = current_bench_call()
        if call is None:
            direct["candidate"].append(int(k))
        wall = 0.0005 * k
        if call is not None and call.arm == "candidate" and call.phase == "sample":
            counted.append(wall / k)
        return wall

    def mlx(k: int) -> float:
        if current_bench_call() is None:
            direct["mlx"].append(int(k))
        return 0.0005 * k

    _noise, stats, attempts = runtime._measure(
        cand, {"mlx": mlx}, seed=2, reps=4, rotation=8, control=True,
    )
    assert direct["candidate"] == direct["mlx"] == [1, 30]
    assert attempts == 1
    assert runtime._bench_extended_attempts is None
    assert runtime._bench_base_attempts is None
    assert stats["candidate"].samples_s == tuple(counted)
    assert stats["candidate"].median_s == pytest.approx(0.0005)
    assert len(stats[BENCH_CONTROL].samples_s) == 4


def test_a_quiet_extended_attempt_reports_how_many_runs_actually_happened():
    """@brief 第 4 次才安静时，报告写 4，不再被截成 3。选中的是这次安静墙钟。"""
    runtime = _runtime()
    seen = {"attempt": 0, "in_cal": False, "cand_samples": 0}

    def cand(k: int) -> float:
        call = current_bench_call()
        if call is not None and call.phase == "calibration" and call.arm == "candidate":
            if not seen["in_cal"]:
                seen["attempt"] += 1
                seen["in_cal"] = True
        elif call is None or call.phase != "calibration":
            seen["in_cal"] = False
        if seen["attempt"] >= 4:
            return 0.02 * k
        if call is not None and call.arm == "candidate" and call.phase == "sample":
            seen["cand_samples"] += 1
            scale = 0.01 if seen["cand_samples"] % 2 else 0.07
            return scale * k
        return 0.02 * k

    _noise, stats, attempts = runtime._measure(cand, {}, seed=1, reps=4, rotation=8)
    assert attempts == 4
    assert runtime._bench_base_attempts == BENCH_ATTEMPTS
    assert runtime._bench_extended_attempts == 1
    assert stats["candidate"].median_s == pytest.approx(0.02)
    bench = V2Benchmark(
        stats=stats["candidate"],
        baselines={},
        rotation_sets=8,
        dispatch=[],
        host=None,
        guard_violations=[],
        noise_cv=0.0,
        attempts=attempts,
        base_attempts=runtime._bench_base_attempts,
        extended_attempts=runtime._bench_extended_attempts,
    )
    document = bench.document()
    assert document["attempts"] == 4
    assert document["base_attempts"] == 3
    assert document["extended_attempts"] == 1


def test_noisy_extended_attempts_keep_the_base_selection_and_the_real_count():
    """@brief 追加 9 次仍然吵时，墙钟仍在前 3 次里选，但次数写成 12。"""
    runtime = _runtime()
    seen = {"attempt": 0, "in_cal": False, "cand_samples": 0}

    def cand(k: int) -> float:
        call = current_bench_call()
        if call is not None and call.phase == "calibration" and call.arm == "candidate":
            if not seen["in_cal"]:
                seen["attempt"] += 1
                seen["in_cal"] = True
        elif call is None or call.phase != "calibration":
            seen["in_cal"] = False
        if call is not None and call.arm == "candidate" and call.phase == "sample":
            seen["cand_samples"] += 1
            if seen["attempt"] <= BENCH_ATTEMPTS:
                scale = 0.01 if seen["cand_samples"] % 2 else 0.07
            else:
                scale = 0.05 if seen["cand_samples"] % 2 else 0.15
            return scale * k
        return 0.02 * k

    _noise, stats, attempts = runtime._measure(cand, {}, seed=1, reps=4, rotation=8)
    assert attempts == BENCH_ATTEMPTS + EXTENDED_ATTEMPTS
    assert runtime._bench_base_attempts == BENCH_ATTEMPTS
    assert runtime._bench_extended_attempts == 9
    assert stats["candidate"].median_s == pytest.approx(0.04)


def test_v1_gpu_samples_drop_calibration_and_warmup_markers():
    """@brief v1 的 cand_run 以前从校准就开始记。现在只留计入统计的 3 微秒。"""
    from tcenv.backends.metal.backend import MetalBackend

    class Ctx:
        def run(self, kernel, arg_sets, grid, threads, iters=1, dispatch="concurrent"):
            call = current_bench_call()
            counted = call is not None and call.arm == "candidate" and call.phase == "sample"
            gpu = (3e-6 if counted else 9.0) * int(iters)
            return RunTiming(wall_s=1e-4 * int(iters), gpu_s=gpu, iters=int(iters))

    class One:
        def args(self):
            return []

    harness = MetalBackend().prepare_bench(
        Ctx(),
        [object()],
        SimpleNamespace(grid=(1, 1, 1), threads=(1, 1, 1)),
        object(),
        spec=None,
        data0=None,
        reps=4,
        measure_serial=False,
        sets=[One()],
        base_runners={},
        selected=["candidate"],
    )
    first = len(harness.gpu_samples)
    interleaved(
        {"candidate": harness.cand_run, "mlx": lambda k: 2e-4 * k},
        reps=4, warmup_reps=2, seed=3, kmin=1, target_s=0.0004,
    )
    kept = harness.gpu_samples[first:]
    assert kept == pytest.approx([3e-6] * 4)
    assert 9.0 not in kept


def test_the_honest_attempt_count_survives_the_track_explore_gate():
    """@brief 报告如实写 4 次时，track-explore 不再把它当多批次淘汰；裸数字与错的拆分仍拒绝。"""
    from taic.track_explore import TrackExploreError, _validate_confirmation_attempts

    # 旧的 "1 <= report_attempts <= BENCH_ATTEMPTS" 会把这次合法的安静确认判成
    # confirmation_batch_mismatch，理由是假的。
    _validate_confirmation_attempts(
        {"confirmation_attempts": 2},
        {"benchmark": {"attempts": BENCH_ATTEMPTS + 1, "base_attempts": BENCH_ATTEMPTS, "extended_attempts": 1}},
        1,
    )
    for report in (
        {"benchmark": {"attempts": BENCH_ATTEMPTS + 1}},  # 超过基础次数却没报拆分
        {"benchmark": {"attempts": BENCH_ATTEMPTS + 1, "base_attempts": BENCH_ATTEMPTS, "extended_attempts": 2}},
        {"benchmark": {"attempts": BENCH_ATTEMPTS + 1, "base_attempts": BENCH_ATTEMPTS, "extended_attempts": 0}},
        {"benchmark": {"attempts": BENCH_ATTEMPTS + EXTENDED_ATTEMPTS + 1, "extended_attempts": EXTENDED_ATTEMPTS + 1}},
    ):
        with pytest.raises(TrackExploreError, match="confirmation_batch_mismatch"):
            _validate_confirmation_attempts({"confirmation_attempts": 2}, report, 1)
    with pytest.raises(TrackExploreError, match="confirmation_batch_mismatch"):
        # 确认重试本身的次数上限没有变。
        _validate_confirmation_attempts(
            {"confirmation_attempts": BENCH_ATTEMPTS + 1},
            {"benchmark": {"attempts": BENCH_ATTEMPTS + 1, "base_attempts": BENCH_ATTEMPTS, "extended_attempts": 1}},
            1,
        )


def test_tilelang_benchmark_forwards_control_and_rejects_an_unsupported_runtime():
    """@brief TileLang 把 control 交给共享 runtime；真不支持时是 typed 错误，不是 TypeError。"""
    from tcenv.backends.metal.tilelang_runtime import TileLangMetalExecutor

    signature = inspect.signature(TileLangMetalExecutor.benchmark)
    assert "control" in signature.parameters
    assert signature.parameters["control"].default is False

    seen = {}

    class Runtime:
        abi = SimpleNamespace(tensors=())
        workspace_probe = {"w": "00"}
        benchmark_diagnostics = {}
        contract = SimpleNamespace(instance=SimpleNamespace(is_capacity=False))

        def benchmark(self, *, seed, reps, discipline, baseline_runners=None, control=False):
            seen["control"] = control
            return SimpleNamespace(
                guard_violations=[],
                rotation_sets=2,
                attempts=4,
                noise_cv=0.01,
                baselines={},
                base_attempts=3,
                extended_attempts=1,
                control={"timed_by": "wall_clock", "rounds": 2},
            )

    executor = TileLangMetalExecutor.__new__(TileLangMetalExecutor)
    executor.evidence = {}
    TileLangMetalExecutor.benchmark(
        executor, SimpleNamespace(), Runtime(), seed=7, reps=4, control=True,
    )
    assert seen["control"] is True
    assert executor.evidence["benchmark_control"]["timed_by"] == "wall_clock"
    assert executor.evidence["benchmark_protocol"]["attempts"] == 4
    assert executor.evidence["benchmark_protocol"]["base_attempts"] == 3
    assert executor.evidence["benchmark_protocol"]["extended_attempts"] == 1

    class NoControl:
        abi = SimpleNamespace(tensors=())

        def benchmark(self, *, seed, reps, discipline, baseline_runners=None):
            seen["bare"] = True
            return SimpleNamespace()

    bare = TileLangMetalExecutor.__new__(TileLangMetalExecutor)
    bare.evidence = None
    TileLangMetalExecutor.benchmark(
        bare, SimpleNamespace(), NoControl(), seed=1, reps=1, control=False,
    )
    assert seen["bare"] is True
    with pytest.raises(UnsupportedContractCapability, match="same-window control") as raised:
        TileLangMetalExecutor.benchmark(
            bare, SimpleNamespace(), NoControl(), seed=1, reps=1, control=True,
        )
    assert raised.value.code == "unsupported_contract_capability"
