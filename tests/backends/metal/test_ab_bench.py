"""N-way interleaved timing. CPU only: runners are plain callables, no device is opened."""

from __future__ import annotations

import pytest

from taic.paired_compare import PairedCompareError
from tcenv.backends.metal.ab_bench import MAX_FILES, SCHEMA, compare_many


def _runner(per_k_s: float, *, clock: str | None = "gpu_timestamp"):
    """一颗假 runner：``run_k(k)`` 返回整批秒数。"""
    calls: list[int] = []

    def run_k(k: int) -> float:
        calls.append(int(k))
        return per_k_s * int(k)

    if clock is not None:
        run_k.clock_of = lambda: clock
    run_k.calls = calls  # type: ignore[attr-defined]
    return run_k


def test_rows_carry_a_paired_ratio_against_the_first_file():
    """@brief 第一份文件是对照；其余行给逐轮配对比的中位数，不是两个中位数相除。"""
    document = compare_many(
        {"base.metal": _runner(3.00e-6), "fast.metal": _runner(2.94e-6)},
        seed=7, reps=6, warmup_reps=2, kmin=4,
    )
    assert document["schema"] == SCHEMA
    assert document["control"] == "base.metal"
    base, fast = document["rows"]
    assert base["label"] == "base.metal" and base["paired_ratio_to_first"] == 1.0
    assert fast["median_us"] == pytest.approx(2.94, rel=1e-6)
    # base 每次比 fast 慢 3/2.94 倍 -> 配对比约 1.0204。
    assert fast["paired_ratio_to_first"] == pytest.approx(3.00 / 2.94, rel=1e-9)


def test_the_document_says_which_clock_was_used_and_refuses_to_promote():
    """@brief 只有两侧都报 GPU 时间戳时才写 gpu_timestamp；任何时候都不叫发布证据。"""
    gpu = compare_many(
        {"a": _runner(3e-6, clock="gpu_timestamp"), "b": _runner(3e-6, clock="gpu_timestamp")},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert gpu["timed_by"] == "gpu_timestamp"
    assert gpu["readonly"] is True and gpu["promotable"] is False
    assert "not publication evidence" in gpu["note"]

    mixed = compare_many(
        {"a": _runner(3e-6, clock="gpu_timestamp"), "b": _runner(3e-6, clock="wall_clock")},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert mixed["timed_by"] == "wall_clock"


def test_a_runner_without_a_clock_reader_counts_as_wall_clock():
    """@brief 读不出时钟就按墙钟标注，不能默认成 GPU 时间戳。"""
    document = compare_many(
        {"a": _runner(3e-6, clock=None), "b": _runner(3e-6, clock=None)},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert document["timed_by"] == "wall_clock"


def test_warmup_rounds_are_discarded_and_every_file_is_measured_the_same_number_of_times():
    """@brief 丢弃预热之后每颗文件留下的样本数相同，配对才有下标可言。"""
    left = _runner(3e-6)
    right = _runner(3e-6)
    document = compare_many({"a": left, "b": right}, reps=5, warmup_reps=3, kmin=4)
    for row in document["rows"]:
        assert row["reps"] == 5
        assert row["iters"] >= 4
    assert len(left.calls) == len(right.calls)


def test_it_refuses_one_file():
    """@brief 排序至少要两份文件。"""
    with pytest.raises(PairedCompareError):
        compare_many({"only.metal": _runner(3e-6)})


def test_it_refuses_more_files_than_the_cap():
    """@brief 上限是 16 份；再多就不是一段窗口里能排清楚的事了。"""
    runners = {f"v{index}.metal": _runner(3e-6) for index in range(MAX_FILES + 1)}
    with pytest.raises(PairedCompareError):
        compare_many(runners)


def test_bench_paths_refuses_duplicates_and_missing_files(tmp_path):
    """@brief 同一份文件算两次、文件不存在、重名，都在父进程就拒绝。"""
    from tcenv.backends.metal.ab_bench import bench_paths

    one = tmp_path / "one.metal"
    one.write_text("kernel void one() {}\n", encoding="utf-8")
    with pytest.raises(PairedCompareError):
        bench_paths([one], task="conv2d", dtype="fp16", size="small")
    with pytest.raises(PairedCompareError):
        bench_paths([one, one], task="conv2d", dtype="fp16", size="small")
    with pytest.raises(PairedCompareError):
        bench_paths([one, tmp_path / "missing.metal"], task="conv2d", dtype="fp16", size="small")

    other = tmp_path / "sub" / "one.metal"
    other.parent.mkdir()
    other.write_text("kernel void one() {}\n", encoding="utf-8")
    with pytest.raises(PairedCompareError):
        bench_paths([one, other], task="conv2d", dtype="fp16", size="small")


def test_the_cli_refuses_a_non_metal_backend_before_binding_anything(capsys):
    """@brief 非 Metal target 看不见这条命令的能力：它明确拒绝，不建 worker。"""
    from tcenv.cli import main

    code = main([
        "ab", "a.metal", "b.metal", "--task", "conv2d", "--dtype", "fp16",
        "--size", "small", "--backend", "qingwei",
    ])
    assert code == 2
    assert "Metal-only" in capsys.readouterr().err
