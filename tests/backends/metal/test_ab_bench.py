"""N-way interleaved timing. CPU only: runners are plain callables, no device is opened."""

from __future__ import annotations

from pathlib import Path

import pytest

from taic.paired_compare import PairedCompareError
from tcenv.backends.metal.ab_bench import MAX_FILES, SCHEMA, compare_many


def _ok(**extra) -> dict:
    """一份通过 quick 正确性的结论。"""
    verdict = {"status": "ok", "ok": True, "worst_scaled_err": 0.01, "allow": 1.0, "message": ""}
    verdict.update(extra)
    return verdict


def _incorrect(**extra) -> dict:
    verdict = {"status": "incorrect", "ok": False, "worst_scaled_err": 3.4, "allow": 1.0, "message": "case 'normal'"}
    verdict.update(extra)
    return verdict


def _runner(per_k_s: float, *, clock: str | None = "gpu_timestamp", history=None):
    """一颗假 runner：``run_k(k)`` 返回整批秒数。

    ``history`` 是逐批 ``(phase, clock, scope)``；不给时只能靠 ``clock_of``（旧路径）。
    """
    calls: list[int] = []

    def run_k(k: int) -> float:
        calls.append(int(k))
        return per_k_s * int(k)

    if clock is not None:
        run_k.clock_of = lambda: clock
    if history is not None:
        run_k.clock_history = list(history)
    run_k.calls = calls  # type: ignore[attr-defined]
    return run_k


def test_rows_carry_a_paired_ratio_against_the_first_file():
    """@brief 第一份文件是对照；其余行给逐轮配对比的中位数，不是两个中位数相除。"""
    document = compare_many(
        {"base.metal": _runner(3.00e-6), "fast.metal": _runner(2.94e-6)},
        verdicts={"base.metal": _ok(), "fast.metal": _ok()},
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
        verdicts={"a": _ok(), "b": _ok()},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert gpu["timed_by"] == "gpu_timestamp"
    assert gpu["readonly"] is True and gpu["promotable"] is False
    assert "not publication evidence" in gpu["note"]

    mixed = compare_many(
        {"a": _runner(3e-6, clock="gpu_timestamp"), "b": _runner(3e-6, clock="wall_clock")},
        verdicts={"a": _ok(), "b": _ok()},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert mixed["timed_by"] == "wall_clock"


def test_a_runner_without_a_clock_reader_counts_as_wall_clock():
    """@brief 读不出时钟就按墙钟标注，不能默认成 GPU 时间戳。"""
    document = compare_many(
        {"a": _runner(3e-6, clock=None), "b": _runner(3e-6, clock=None)},
        verdicts={"a": _ok(), "b": _ok()},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert document["timed_by"] == "wall_clock"
    assert document["measurement_scope"] == "unspecified"


def test_a_missing_gpu_timestamp_is_not_hidden_by_a_later_gpu_batch():
    """@brief 缺失 -> 恢复 与 恢复 -> 缺失 都要整份标墙钟；不能用最后一批的时钟。"""
    gpu, wall = "gpu_timestamp", "wall_clock"
    for sequence in (
        [wall, gpu],
        [gpu, wall],
        [wall, gpu, gpu],
        [gpu, wall, gpu],
        [gpu, gpu, wall],
    ):
        document = compare_many(
            {
                "a": _runner(3e-6, history=[("sample", clock, "fused_command_buffer_per_iteration") for clock in sequence]),
                "b": _runner(3e-6, history=[("sample", clock, "fused_command_buffer_per_iteration") for clock in sequence]),
            },
            verdicts={"a": _ok(), "b": _ok()},
            reps=3, warmup_reps=1, kmin=4,
        )
        assert document["timed_by"] == "wall_clock", sequence
        assert {row["timed_by"] for row in document["rows"]} == {"wall_clock"}, sequence
    homogeneous = compare_many(
        {
            "a": _runner(3e-6, history=[("sample", gpu, "fused_command_buffer_per_iteration")] * 3),
            "b": _runner(3e-6, history=[("sample", gpu, "fused_command_buffer_per_iteration")] * 3),
        },
        verdicts={"a": _ok(), "b": _ok()},
        reps=3, warmup_reps=1, kmin=4,
    )
    assert homogeneous["timed_by"] == "gpu_timestamp"
    assert homogeneous["measurement_scope"] == "fused_command_buffer_per_iteration"
    # 校准与预热批次不是样本，墙钟预热不能把整份文档拉成墙钟。
    warm_only = [("warmup", "wall_clock", None), ("sample", "gpu_timestamp", "fused_command_buffer_per_iteration")]
    quiet = compare_many(
        {"a": _runner(3e-6, history=warm_only), "b": _runner(3e-6, history=warm_only)},
        verdicts={"a": _ok(), "b": _ok()},
        reps=3, warmup_reps=1, kmin=4,
    )
    assert quiet["timed_by"] == "gpu_timestamp"


def _clock_sequence_runner(per_k_s: float, sample_clocks: list[str]):
    """旧式 runner：只有 ``clock_of``，每次真实调用之后才更新。"""
    from tcenv.bench import current_bench_call

    state = {"clock": "gpu_timestamp", "sample": 0}

    def run_k(k: int) -> float:
        call = current_bench_call()
        if call is not None and call.phase == "sample":
            index = state["sample"]
            state["clock"] = sample_clocks[min(index, len(sample_clocks) - 1)]
            state["sample"] = index + 1
        else:
            state["clock"] = "gpu_timestamp"
        return per_k_s * int(k)

    run_k.clock_of = lambda: state["clock"]  # type: ignore[attr-defined]
    run_k.state = state  # type: ignore[attr-defined]
    return run_k


def test_compare_many_records_the_clock_of_each_call_not_the_last_one():
    """@brief 旧式 runner 的动态调用：中间退回墙钟、最后恢复 GPU 不能洗白整份文档。"""
    recovered = _clock_sequence_runner(3e-6, ["wall_clock", "gpu_timestamp", "gpu_timestamp"])
    document = compare_many(
        {"a": recovered, "b": _clock_sequence_runner(3e-6, ["gpu_timestamp"] * 3)},
        verdicts={"a": _ok(), "b": _ok()},
        reps=3, warmup_reps=1, kmin=4,
    )
    # 每一行只由自己的样本决定：a 有一批退回墙钟，文档才是 wall_clock。
    assert recovered.state["clock"] == "gpu_timestamp"
    assert document["timed_by"] == "wall_clock"
    rows = {row["label"]: row for row in document["rows"]}
    assert rows["a"]["timed_by"] == "wall_clock"
    assert rows["b"]["timed_by"] == "gpu_timestamp"

    missing_last = _clock_sequence_runner(3e-6, ["gpu_timestamp", "gpu_timestamp", "wall_clock"])
    again = compare_many(
        {"a": missing_last, "b": _clock_sequence_runner(3e-6, ["gpu_timestamp"] * 3)},
        verdicts={"a": _ok(), "b": _ok()},
        reps=3, warmup_reps=1, kmin=4,
    )
    assert missing_last.state["clock"] == "wall_clock"
    assert again["timed_by"] == "wall_clock"


def test_ab_takes_a_batch_wide_enough_to_cover_the_rotation_sets():
    """@brief 评测的 kmin 是轮转组数；ab 不能用更小的批量只碰到前几组。"""
    left = _runner(3e-6)
    right = _runner(2e-6)
    left.rotation_sets = 7
    right.rotation_sets = 7
    document = compare_many(
        {"a": left, "b": right}, verdicts={"a": _ok(), "b": _ok()},
        reps=2, warmup_reps=1, kmin=2,
    )
    assert document["kmin"] == 7
    assert all(row["iters"] >= 7 for row in document["rows"])


def test_no_counted_sample_is_not_reported_as_a_gpu_document():
    """@brief 一条样本都没有时 timed_by 是 null，不是 gpu_timestamp。"""
    document = compare_many(
        {"a": _runner(3e-6, history=[]), "b": _runner(2e-6, history=[])},
        verdicts={"a": _ok(), "b": _ok()},
        reps=3, warmup_reps=1, kmin=4,
    )
    assert document["timed_by"] is None
    assert document["measurement_scope"] is None
    assert all(row["timed_by"] is None for row in document["rows"])
    assert document["readonly"] is True and document["promotable"] is False


def test_rows_carry_their_own_digest_and_path():
    """@brief 每一行自带文件身份，不靠与全部输入列表的下标对齐。"""
    files = {
        "base.metal": {"path": "/tmp/base.metal", "sha256": "a" * 64},
        "fast.metal": {"path": "/tmp/fast.metal", "sha256": "b" * 64},
    }
    document = compare_many(
        {"base.metal": _runner(3e-6), "fast.metal": _runner(2e-6)},
        verdicts={"base.metal": _ok(), "fast.metal": _ok()},
        files=files, reps=3, warmup_reps=1, kmin=4,
    )
    rows = {row["label"]: row for row in document["rows"]}
    assert rows["base.metal"]["sha256"] == "a" * 64
    assert rows["base.metal"]["path"] == "/tmp/base.metal"
    assert rows["fast.metal"]["sha256"] == "b" * 64
    assert rows["fast.metal"]["path"] == "/tmp/fast.metal"


def test_warmup_rounds_are_discarded_and_every_file_is_measured_the_same_number_of_times():
    """@brief 丢弃预热之后每颗文件留下的样本数相同，配对才有下标可言。"""
    left = _runner(3e-6)
    right = _runner(3e-6)
    document = compare_many(
        {"a": left, "b": right}, verdicts={"a": _ok(), "b": _ok()},
        reps=5, warmup_reps=3, kmin=4,
    )
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
        compare_many(runners, verdicts={name: _ok() for name in runners})


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


def test_an_incorrect_ablation_is_still_timed_but_kept_out_of_the_ranking():
    """@brief 故意算错的消融仍要被计时（那是它存在的理由），但不参与「哪版更快」。"""
    document = compare_many(
        {
            "submitted.metal": _runner(3.00e-6),
            "cut_loop.metal": _runner(2.10e-6),
            "v_tuned.metal": _runner(2.94e-6),
        },
        verdicts={
            "submitted.metal": _ok(),
            "cut_loop.metal": _incorrect(),
            "v_tuned.metal": _ok(),
        },
        seed=7, reps=4, warmup_reps=1, kmin=4,
    )
    rows = {row["label"]: row for row in document["rows"]}
    # 三份都被计时，包括算错的那份。
    assert set(rows) == {"submitted.metal", "cut_loop.metal", "v_tuned.metal"}
    assert rows["cut_loop.metal"]["median_us"] == pytest.approx(2.10, rel=1e-6)
    # 但排序只收通过正确性的两份，而且算错的那份被列出来说明原因。
    assert document["ranking"]["order"] == ["v_tuned.metal", "submitted.metal"]
    assert [item["label"] for item in document["ranking"]["excluded"]] == ["cut_loop.metal"]
    assert rows["cut_loop.metal"]["correct"] is False
    assert rows["cut_loop.metal"]["status"] == "incorrect"
    assert rows["cut_loop.metal"]["worst_scaled_err"] == pytest.approx(3.4)


def test_a_run_whose_only_alternative_is_incorrect_reports_no_ranking():
    """@brief 可排序的行不足两行时不给顺序，但仍把计时结果交回去。"""
    document = compare_many(
        {"a": _runner(3.0e-6), "b": _runner(2.0e-6)},
        verdicts={"a": _ok(), "b": _incorrect()},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert document["ranking"]["order"] == []
    assert "fewer than two" in document["ranking"]["reason"]
    assert len(document["rows"]) == 2


def test_the_control_must_pass_correctness():
    """@brief 第一份文件是对照；它算错时整次调用拒绝，因为比值没有意义。"""
    with pytest.raises(PairedCompareError):
        compare_many(
            {"broken.metal": _runner(3e-6), "good.metal": _runner(2e-6)},
            verdicts={"broken.metal": _incorrect(), "good.metal": _ok()},
        )


def test_an_unchecked_file_cannot_serve_as_the_control():
    """@brief 没跑正确性检查就不能当对照，不能默认它是对的。"""
    with pytest.raises(PairedCompareError):
        compare_many({"a": _runner(3e-6), "b": _runner(2e-6)})


def test_only_ok_and_incorrect_are_timable():
    """@brief 不安全、被拒、运行期错误都不进计时循环。"""
    from tcenv.backends.metal.ab_bench import TIMABLE

    assert TIMABLE == {"ok", "incorrect"}
    for status in ("unsafe", "rejected", "runtime_error", "compile_error", ""):
        assert status not in TIMABLE


def test_the_worker_leaves_unsafe_and_rejected_files_out_of_the_timing(monkeypatch, tmp_path):
    """@brief 不安全或编译被拒的候选不进装配，也不会在计时循环里被重放。"""
    import tcenv.backends.metal.ab_bench as module
    import tcenv.backends.metal.paired_compare as pair

    labels = ["good.metal", "wrong.metal", "unsafe.metal"]
    body = {
        "good.metal": "kernel void good() {}\n",
        "wrong.metal": "kernel void wrong() {}\n",
        "unsafe.metal": "kernel void unsafe() {}\n",
    }
    for name, text in body.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    verdicts = {
        "good.metal": _ok(),
        "wrong.metal": _incorrect(),
        "unsafe.metal": {"status": "unsafe", "ok": False, "message": "buffer bound past its length"},
    }
    monkeypatch.setattr(module, "_quick_verdict", lambda src, **_kwargs: verdicts[_which(src)])
    monkeypatch.setattr(module, "get_context", None, raising=False)

    assembled: list[list[str]] = []

    def fake_assemble(paths, **_kwargs):
        assembled.append([Path(item).name for item in paths])
        return [object() for _ in paths]

    monkeypatch.setattr(pair, "assemble_v2_group", fake_assemble)
    monkeypatch.setattr(pair, "runner_for_prepared", lambda _prepared: (lambda k: 1e-6 * int(k)))
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: object())
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lambda *_a, **_k: _NullLock())
    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", lambda *_a, **_k: None)

    request = {
        "payload": {
            "task": "conv2d", "dtype": "fp16", "size": "small", "seed": 7, "reps": 2,
            "paths": [str(tmp_path / name) for name in labels],
            "labels": labels,
            "sha256": [_sha(tmp_path / name) for name in labels],
            "track_id": "msl-single", "target_id": "apple-m4-max",
        },
        "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
    }
    document = module._worker_main(request)["ab"]
    # 不安全的那份没有被送进计时装配。
    assert assembled == [["good.metal", "wrong.metal"]]
    assert [item["file"] for item in document["rejected"]] == ["unsafe.metal"]
    assert [row["label"] for row in document["rows"]] == ["good.metal", "wrong.metal"]
    # 被拒的那份带着自己的摘要和路径，不会因为被过滤就把时间安到别的文件上。
    rejected = document["rejected"][0]
    assert rejected["sha256"] == _sha(tmp_path / "unsafe.metal")
    assert rejected["path"] == str(tmp_path / "unsafe.metal")
    assert rejected["status"] == "unsafe"
    # 每一行绑定自己的摘要；汇总列表与 files 按输入顺序逐项对应。
    rows = {row["label"]: row for row in document["rows"]}
    assert rows["good.metal"]["sha256"] == _sha(tmp_path / "good.metal")
    assert rows["wrong.metal"]["sha256"] == _sha(tmp_path / "wrong.metal")
    assert [item["label"] for item in document["files"]] == labels
    assert [item["sha256"] for item in document["files"]] == document["sha256"]
    assert [item["timed"] for item in document["files"]] == [True, True, False]
    # 只剩一份通过的候选，所以没有排序，但两份都被计时了。
    assert document["ranking"]["order"] == []
    assert [item["label"] for item in document["ranking"]["excluded"]] == ["wrong.metal"]


def test_fewer_than_two_timable_files_is_not_a_measurement(monkeypatch, tmp_path):
    """@brief 不足两份可计时文件时不记 timed_by=gpu_timestamp，也不把零样本写成一次测量。"""
    import tcenv.backends.metal.ab_bench as module

    labels = ["good.metal", "unsafe_one.metal", "rejected_two.metal"]
    for name in labels:
        (tmp_path / name).write_text(f"kernel void {name.split('.')[0]}() {{}}\n", encoding="utf-8")
    verdicts = {
        "good.metal": _ok(),
        "unsafe_one.metal": {"status": "unsafe", "ok": False, "message": "buffer bound past its length"},
        "rejected_two.metal": {"status": "rejected", "ok": False, "message": "candidate did not compile"},
    }
    monkeypatch.setattr(module, "_quick_verdict", lambda src, **_kwargs: verdicts[_which(src)])
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lambda *_a, **_k: _NullLock())
    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", lambda *_a, **_k: None)

    def refuse_context():
        raise AssertionError("no timable pair means no device is opened")

    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", refuse_context)
    request = {
        "payload": {
            "task": "conv2d", "dtype": "fp16", "size": "small", "seed": 7, "reps": 2,
            "paths": [str(tmp_path / name) for name in labels],
            "labels": labels,
            "sha256": [_sha(tmp_path / name) for name in labels],
            "track_id": "msl-single", "target_id": "apple-m4-max",
        },
        "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
    }
    envelope = module._worker_main(request)
    document = envelope["ab"]
    assert envelope["report"]["status"] == "no_measurement"
    assert document["status"] == "no_measurement"
    assert document["timed_by"] is None
    assert document["measurement_scope"] is None
    assert document["elapsed_s"] is None
    assert document["rows"] == []
    assert document["readonly"] is True and document["promotable"] is False
    assert "nothing was timed" in document["reason"]
    assert "not a measurement" in document["reason"]
    assert [item["file"] for item in document["rejected"]] == ["unsafe_one.metal", "rejected_two.metal"]
    assert [item["sha256"] for item in document["rejected"]] == [
        _sha(tmp_path / "unsafe_one.metal"), _sha(tmp_path / "rejected_two.metal"),
    ]
    assert [item["timed"] for item in document["files"]] == [False, False, False]
    assert all(item["timed"] is False for item in document["files"])
    assert document["ranking"]["order"] == []
    assert "fewer than two timable files" in document["ranking"]["reason"]


def test_the_worker_keeps_identity_when_the_first_file_is_rejected(monkeypatch, tmp_path):
    """@brief 首份被拒时，旧实现按过滤后的下标对齐 sha256，会把时间安到另一份文件上。"""
    import tcenv.backends.metal.ab_bench as module
    import tcenv.backends.metal.paired_compare as pair

    labels = ["first.metal", "second.metal", "third.metal"]
    for name in labels:
        (tmp_path / name).write_text(f"kernel void {name.split('.')[0]}() {{}}\n", encoding="utf-8")
    digests = {name: _sha(tmp_path / name) for name in labels}
    verdicts = {
        "first.metal": {"status": "unsafe", "ok": False, "message": "buffer bound past its length"},
        "second.metal": _ok(),
        "third.metal": _incorrect(),
    }
    monkeypatch.setattr(module, "_quick_verdict", lambda src, **_kwargs: verdicts[_which(src)])
    monkeypatch.setattr(pair, "assemble_v2_group", lambda paths, **_kwargs: [object() for _ in paths])
    monkeypatch.setattr(pair, "runner_for_prepared", lambda _prepared: (lambda k: 1e-6 * int(k)))
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: object())
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lambda *_a, **_k: _NullLock())
    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", lambda *_a, **_k: None)
    request = {
        "payload": {
            "task": "conv2d", "dtype": "fp16", "size": "small", "seed": 7, "reps": 2,
            "paths": [str(tmp_path / name) for name in labels],
            "labels": labels,
            "sha256": [digests[name] for name in labels],
            "track_id": "msl-single", "target_id": "apple-m4-max",
        },
        "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
    }
    document = module._worker_main(request)["ab"]
    # 对照是第一个可计时的文件，不是第一个输入文件。
    assert document["control"] == "second.metal"
    rows = {row["label"]: row for row in document["rows"]}
    assert set(rows) == {"second.metal", "third.metal"}
    for name, row in rows.items():
        # 每一行带自己的摘要。旧实现把 sha256[0]（首份）与第二份的时间列在一起。
        assert row["sha256"] == digests[name]
        assert row["path"] == str(tmp_path / name)
    rejected = document["rejected"][0]
    assert (rejected["file"], rejected["sha256"]) == ("first.metal", digests["first.metal"])
    # 汇总身份仍按输入顺序，timed 标出哪几份真的进了计时。
    assert [item["sha256"] for item in document["files"]] == [digests[name] for name in labels]
    assert document["sha256"] == [digests[name] for name in labels]
    assert [item["timed"] for item in document["files"]] == [False, True, True]
    assert [item["label"] for item in document["ranking"]["excluded"]] == ["third.metal"]


def _worker_with_a_rejected_middle(monkeypatch, tmp_path, status: str, message: str):
    """三份输入、中间那份被拒：逐行固定 path/sha256/label 与 rejected 身份。"""
    import tcenv.backends.metal.ab_bench as module
    import tcenv.backends.metal.paired_compare as pair

    labels = ["first.metal", "middle.metal", "last.metal"]
    for name in labels:
        (tmp_path / name).write_text(f"kernel void {name.split('.')[0]}() {{}}\n", encoding="utf-8")
    digests = {name: _sha(tmp_path / name) for name in labels}
    verdicts = {
        "first.metal": _ok(),
        "middle.metal": {"status": status, "ok": False, "message": message},
        "last.metal": _incorrect(),
    }
    monkeypatch.setattr(module, "_quick_verdict", lambda src, **_kwargs: verdicts[_which(src)])
    assembled: list[list[str]] = []

    def fake_assemble(paths, **_kwargs):
        assembled.append([Path(item).name for item in paths])
        return [object() for _ in paths]

    monkeypatch.setattr(pair, "assemble_v2_group", fake_assemble)
    monkeypatch.setattr(pair, "runner_for_prepared", lambda _prepared: (lambda k: 1e-6 * int(k)))
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: object())
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lambda *_a, **_k: _NullLock())
    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", lambda *_a, **_k: None)
    request = {
        "payload": {
            "task": "conv2d", "dtype": "fp16", "size": "small", "seed": 7, "reps": 2,
            "paths": [str(tmp_path / name) for name in labels],
            "labels": labels,
            "sha256": [digests[name] for name in labels],
            "track_id": "msl-single", "target_id": "apple-m4-max",
        },
        "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
    }
    document = module._worker_main(request)["ab"]
    # 中间那份不进装配，但前后两份仍按自己的身份入行。
    assert assembled == [["first.metal", "last.metal"]]
    assert document["control"] == "first.metal"
    rows = document["rows"]
    assert [row["label"] for row in rows] == ["first.metal", "last.metal"]
    for name, row in zip(("first.metal", "last.metal"), rows, strict=True):
        assert row["label"] == name
        assert row["path"] == str(tmp_path / name)
        assert row["sha256"] == digests[name]
    rejected = document["rejected"]
    assert len(rejected) == 1
    assert rejected[0]["file"] == rejected[0]["label"] == "middle.metal"
    assert rejected[0]["path"] == str(tmp_path / "middle.metal")
    assert rejected[0]["sha256"] == digests["middle.metal"]
    assert rejected[0]["status"] == status
    assert rejected[0]["reason"] == message
    # 汇总身份按输入顺序：中间那份留在原位，timed 标出前后两份。
    assert [item["label"] for item in document["files"]] == labels
    assert [item["path"] for item in document["files"]] == [str(tmp_path / name) for name in labels]
    assert [item["sha256"] for item in document["files"]] == [digests[name] for name in labels]
    assert [item["timed"] for item in document["files"]] == [True, False, True]
    assert document["sha256"] == [digests[name] for name in labels]
    assert [item["label"] for item in document["ranking"]["excluded"]] == ["last.metal"]
    return document


def test_a_rejected_middle_file_keeps_its_own_row_identity(monkeypatch, tmp_path):
    """@brief 中间不安全：前后两份仍按自己的 path/sha256/label 入行，拒绝项带自己的身份。"""
    _worker_with_a_rejected_middle(monkeypatch, tmp_path, "unsafe", "buffer bound past its length")


def test_a_rejected_middle_file_that_did_not_compile_keeps_the_rows_aligned(monkeypatch, tmp_path):
    """@brief 中间编译被拒：过滤后的 rows 仍逐行对齐 path/sha256/label。"""
    _worker_with_a_rejected_middle(monkeypatch, tmp_path, "rejected", "candidate did not compile")


def test_a_middle_file_rejected_by_a_runtime_error_keeps_the_rows_aligned(monkeypatch, tmp_path):
    """@brief 中间运行期错误：只有前后两份计时，汇总仍按三份输入的顺序。"""
    _worker_with_a_rejected_middle(monkeypatch, tmp_path, "runtime_error", "the kernel exceeded its watchdog deadline")


def test_zero_repetitions_are_refused(tmp_path):
    """@brief 零轮不是一次测量：没有样本就不能写一份 gpu_timestamp 文档。"""
    with pytest.raises(PairedCompareError, match="zero samples"):
        compare_many(
            {"a": _runner(3e-6), "b": _runner(2e-6)},
            verdicts={"a": _ok(), "b": _ok()},
            reps=0, warmup_reps=0, kmin=4,
        )


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _which(src: str) -> str:
    """按源码里的内核名认出这是哪份文件。"""
    import re

    found = re.search(r"void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", src)
    if found:
        return f"{found.group(1)}.metal"
    raise AssertionError(f"unexpected source: {src[:40]}")


class _NullLock:
    def __enter__(self):
        return None

    def __exit__(self, *_exc):
        return False


def _breaking_runner():
    """Quick 已通过，但第一次 ``run_k`` 之后 guard 被写穿。"""
    calls = {"n": 0}

    def run_k(k: int) -> float:
        calls["n"] += 1
        run_k.broken = True
        return 3e-6 * int(k)

    run_k.broken = False
    run_k.clock_of = lambda: "gpu_timestamp"
    run_k.calls = calls

    def safety():
        if run_k.broken:
            return ["buffer `y`: write past the end of its allocation"]
        return []

    run_k.safety_violations = safety
    return run_k


def test_entry_still_rejects_unsafe_and_rejected_before_timing():
    """@brief 进循环之前，unsafe / rejected 仍然不在可计时集合里。"""
    from tcenv.backends.metal.ab_bench import TIMABLE

    assert "unsafe" not in TIMABLE
    assert "rejected" not in TIMABLE
    assert TIMABLE == {"ok", "incorrect"}


def test_a_guard_break_after_timing_marks_the_whole_ab_unsafe_and_stops():
    """@brief 计时后发现 guard 破坏：整份结果 unsafe，不排名，也不再启动。"""
    bad = _breaking_runner()
    other_calls = {"n": 0}

    def other(k: int) -> float:
        other_calls["n"] += 1
        return 2e-6 * int(k)

    other.clock_of = lambda: "gpu_timestamp"
    other.safety_violations = lambda: []
    document = compare_many(
        {"bad.metal": bad, "good.metal": other},
        verdicts={"bad.metal": _ok(), "good.metal": _ok()},
        reps=4, warmup_reps=1, kmin=4,
    )
    assert document["status"] == "unsafe"
    assert document["promotable"] is False
    assert document["ranking"]["order"] == []
    assert document["rows"] == []
    assert any("write past the end" in item for item in document["safety"])
    assert "not ranked" in document["note"]
    # The breaking batch already ran once. The next calibration call must not.
    assert bad.calls["n"] == 1
    assert other_calls["n"] == 0


def test_the_worker_marks_unsafe_when_replay_breaks_a_guard(monkeypatch, tmp_path):
    """@brief worker 信封跟着 ab 文档标 unsafe，不把这次读数写成 ok。"""
    import tcenv.backends.metal.ab_bench as module
    import tcenv.backends.metal.paired_compare as pair

    labels = ["bad.metal", "good.metal"]
    for name in labels:
        (tmp_path / name).write_text(f"kernel void {name.split('.')[0]}() {{}}\n", encoding="utf-8")
    monkeypatch.setattr(module, "_quick_verdict", lambda _src, **_kwargs: _ok())
    monkeypatch.setattr("tcenv.backends.metal.runtime.get_context", lambda: object())
    monkeypatch.setattr("tcenv.gpulock.gpu_lock", lambda *_a, **_k: _NullLock())
    monkeypatch.setattr("tcenv.v2_worker._install_heartbeat", lambda *_a, **_k: None)
    monkeypatch.setattr(pair, "assemble_v2_group", lambda paths, **_kwargs: [object() for _ in paths])

    made = {"n": 0}

    def fake_prepared(_prepared):
        made["n"] += 1
        if made["n"] == 1:
            return _breaking_runner()
        quiet = _runner(2e-6)
        quiet.safety_violations = lambda: []
        return quiet

    monkeypatch.setattr(pair, "runner_for_prepared", fake_prepared)
    request = {
        "payload": {
            "task": "conv2d", "dtype": "fp16", "size": "small", "seed": 7, "reps": 2,
            "paths": [str(tmp_path / name) for name in labels],
            "labels": labels,
            "sha256": [_sha(tmp_path / name) for name in labels],
            "track_id": "msl-single", "target_id": "apple-m4-max",
        },
        "out": str(tmp_path / "out.json"),
        "heartbeat": str(tmp_path / "heartbeat"),
    }
    envelope = module._worker_main(request)
    document = envelope["ab"]
    assert envelope["report"]["status"] == "unsafe"
    assert document["status"] == "unsafe"
    assert document["ranking"]["order"] == []
    assert any("write past the end" in item for item in document["safety"])


def test_a_bound_runner_checks_the_argsets_it_actually_launched():
    """@brief 生产 runner 在计时后用 benchmark 终检读实际用过的轮转组，检查本身不再启动。"""
    from types import SimpleNamespace

    from tcenv.backends.metal.paired_compare import _BoundV2, runner_for_prepared

    class _Arg:
        def __init__(self) -> None:
            self.bad = False

        def load_initial(self, _inputs) -> None:
            return None

        def apply_fill_plan(self, _plan, **_kwargs) -> None:
            return None

        def stage_args(self, _binding):
            return ()

        def workspace_fills_declared_zero(self, exclude):
            return ()

        def guard_violations(self):
            if self.bad:
                return ["write past the end of its allocation"]
            return []

        def safety_violations(self):
            if self.bad:
                return ["buffer `y`: write past the end of its allocation"]
            return []

    arg = _Arg()
    stage = SimpleNamespace(grid=(1, 1, 1), threads=(1, 1, 1))
    tensor = SimpleNamespace(
        name="x", shape=(4,), strides=(1,), dtype="fp16", space="global", access="read",
    )

    class _Runtime:
        def __init__(self) -> None:
            self.argset = arg
            self.rotation_sets = 1
            self.geometry = SimpleNamespace(active_stages=(stage,), workspace_uses=())
            self.bindings = ("buf",)
            self.abi = SimpleNamespace(tensors=(tensor,))
            self.host_plan = None
            # Host Program 的评测逐 launch，所以这条路径仍然是 run_once。
            self.host_program = SimpleNamespace(written_workspaces=frozenset())
            self.host_inside_timing = False
            self._attempt_gpu: list[float] = []
            self._gpu_collect = False
            self.launches = 0

        def new_argset(self):
            return arg

        def _stage_sets(self):
            return [], ((1, 1, 1),), ((1, 1, 1),), None

        def run_once(self, **_kwargs):
            self.launches += 1
            self.argset.bad = True
            self._attempt_gpu.append(1e-6)

    runtime = _Runtime()
    bound = _BoundV2(
        runtime, task="conv2d", dtype="fp16", size="small",
        contract_digest="abc", inputs={"x": b"\x00\x00"},
    )
    runner = runner_for_prepared(bound)
    assert runner(1) > 0
    assert runtime.launches == 1
    found = runner.safety_violations()
    assert any("write past the end" in item for item in found)
    assert runner.safety_violations() == found
    assert runtime.launches == 1


