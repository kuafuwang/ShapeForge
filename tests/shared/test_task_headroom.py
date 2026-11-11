"""Roofline headroom triage over the real v2 tasks and a synthetic runs tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.task_headroom import _fastest_baseline, _positive, collect_headroom, main

ROOT = Path(__file__).resolve().parents[2]


def _write_report(runs: Path, op: str, size: str, baseline_us: float) -> None:
    folder = runs / op / size
    folder.mkdir(parents=True)
    document = {
        "schema_version": 4,
        "task": op,
        "dtype": "fp16",
        "size": size,
        "target_id": "apple-m4-max",
        "status": "ok",
        "baseline_us": {"torch_mps": baseline_us},
    }
    (folder / "report.json").write_text(json.dumps(document), encoding="utf-8")


def _row(rows, op: str, size: str):
    found = [row for row in rows if row.op_name == op and row.size == size]
    assert len(found) == 1, (op, size, [(row.op_name, row.size) for row in rows])
    return found[0]


def test_real_tasks_are_triaged_from_calibration_and_synthetic_baselines(tmp_path, capsys):
    """@brief gemm small、conv2d、relu、融合题和 varlen L2 落在计划规定的档。"""
    empty = tmp_path / "empty-runs"
    empty.mkdir()
    preview = collect_headroom(target_id="apple-m4-max", dtype="fp16", runs_root=empty, tasks_root=ROOT / "tasks")
    gemm = _row(preview, "gemm", "small")
    assert gemm.lower_bound_us == pytest.approx(1.14, abs=0.03)
    assert gemm.tier == "launch_bound_tiny"
    conv = _row(preview, "conv2d", "paper")
    relu = _row(preview, "relu", "paper")
    fused = _row(preview, "gemm_silu", "paper")
    assert conv.lower_bound_us is not None and conv.lower_bound_us >= 3.0
    assert relu.bound == "bandwidth" and relu.lower_bound_us is not None and relu.lower_bound_us >= 3.0
    assert fused.lower_bound_us is not None and fused.lower_bound_us >= 3.0
    capacity = _row(preview, "varlen_kda", "capacity")
    assert capacity.variable_length_kind == "l2"
    assert capacity.flops >= 0 and capacity.min_bytes >= 0
    assert all(not row.op_name.startswith("sem_") for row in preview)

    runs = tmp_path / "runs"
    _write_report(runs, "conv2d", "paper", conv.lower_bound_us * 0.5)
    _write_report(runs, "relu", "paper", relu.lower_bound_us * 1.03)
    _write_report(runs, "gemm_silu", "paper", fused.lower_bound_us * 2.2)
    assert main([
        "--target", "apple-m4-max",
        "--dtype", "fp16",
        "--runs-root", str(runs),
        "--tasks-root", str(ROOT / "tasks"),
        "--json",
    ]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_key = {(item["op_name"], item["size"]): item for item in payload}
    assert by_key[("gemm", "small")]["tier"] == "launch_bound_tiny"
    assert by_key[("conv2d", "paper")]["tier"] == "algorithm_shift_needed"
    assert by_key[("conv2d", "paper")]["headroom_vs_bound"] == pytest.approx(0.5)
    assert by_key[("relu", "paper")]["tier"] == "saturated_bandwidth"
    assert by_key[("gemm_silu", "paper")]["tier"] == "high_headroom"
    assert by_key[("varlen_kda", "capacity")]["variable_length_kind"] == "l2"
    assert all(not item["op_name"].startswith("sem_") for item in payload)

    assert main([
        "--target", "apple-m4-max",
        "--dtype", "fp16",
        "--runs-root", str(runs),
        "--tasks-root", str(ROOT / "tasks"),
    ]) == 0
    table = capsys.readouterr().out
    assert "| op | size |" in table
    assert "launch_bound_tiny" in table
    assert "high_headroom" in table

    assert main([
        "--target", "apple-m4-max",
        "--dtype", "fp16",
        "--runs-root", str(empty),
        "--tasks-root", str(ROOT / "tasks"),
        "--include-semantic",
        "--json",
    ]) == 0
    opened = json.loads(capsys.readouterr().out)
    assert any(item["op_name"].startswith("sem_") for item in opened)


def test_positive_rejects_non_finite_and_non_positive():
    """@brief inf、NaN、负数和 0 都不是可用的基线或加速比。"""
    assert _positive(float("inf")) is None
    assert _positive(float("-inf")) is None
    assert _positive(float("nan")) is None
    assert _positive(0) is None
    assert _positive(-2.0) is None
    assert _positive(True) is None
    assert _positive("1.5") is None
    assert _positive(1.25) == 1.25
    assert _fastest_baseline({"torch_mps": float("inf"), "zero": 0}) is None
    assert _fastest_baseline({"torch_mps": float("inf"), "mlx": 5.0}) == 5.0


def test_confirmed_speedup_comes_only_from_gated_result_json(tmp_path):
    """@brief 报告只提供 schema-4 基线；确认加速比必须来自过门的 result.json。"""
    runs = tmp_path / "runs"
    digest = "ab" * 32

    def write(relative: str, name: str, document: dict) -> None:
        folder = runs / relative
        folder.mkdir(parents=True)
        (folder / name).write_text(json.dumps(document, allow_nan=True), encoding="utf-8")

    write("schema4-slow", "report.json", {
        "schema_version": 4,
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "baseline_us": {"torch_mps": 30.0, "ignored_inf": float("inf"), "ignored_zero": 0},
        "confirmed_speedup": 50.0,
        "speedup": 40.0,
        "best_speedup": 45.0,
    })
    write("schema4-fast", "report.json", {
        "schema_version": 4,
        "op_name": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target": {"id": "apple-m4-max"},
        "baseline_us": {"mlx": 12.0},
    })
    write("old-schema", "report.json", {
        "schema_version": 3,
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "baseline_us": {"torch_mps": 0.5},
        "confirmed_speedup": 9.0,
    })
    write("wrong-target", "report.json", {
        "schema_version": 4,
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "other-chip",
        "baseline_us": {"torch_mps": 1.0},
    })
    write("wrong-size", "report.json", {
        "schema_version": 4,
        "task": "relu",
        "dtype": "fp16",
        "size": "small",
        "target_id": "apple-m4-max",
        "baseline_us": {"torch_mps": 0.25},
    })
    write("best-only", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": digest,
        "best_speedup": 99.0,
        "baseline_us": {"torch_mps": 0.1},
    })
    write("inf-confirmed", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": digest,
        "confirmed_speedup": float("inf"),
        "best_speedup": 80.0,
    })
    write("bad-status", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "timing_uncertain",
        "confirmation_source_sha256": digest,
        "confirmed_speedup": 7.0,
        "best_speedup": 70.0,
    })
    write("bad-digest", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": "AB" * 32,
        "confirmed_speedup": 6.0,
        "best_speedup": 60.0,
    })
    write("wrong-dtype", "result.json", {
        "task": "relu",
        "dtype": "fp32",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": digest,
        "confirmed_speedup": 5.0,
    })

    smuggled = collect_headroom(
        target_id="apple-m4-max", dtype="fp16", runs_root=runs, tasks_root=ROOT / "tasks",
    )
    relu = _row(smuggled, "relu", "paper")
    assert relu.best_baseline_us == 12.0
    assert relu.best_confirmed_speedup is None

    write("nested/candidate", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": digest,
        "confirmed_speedup": 1.4,
        "best_speedup": 9.9,
    })
    write("faster-confirm", "result.json", {
        "task": "relu",
        "dtype": "fp16",
        "size": "paper",
        "target_id": "apple-m4-max",
        "confirmed_status": "ok",
        "confirmation_source_sha256": digest,
        "confirmed_speedup": 2.5,
        "best_speedup": 1.1,
    })
    confirmed = collect_headroom(
        target_id="apple-m4-max", dtype="fp16", runs_root=runs, tasks_root=ROOT / "tasks",
    )
    relu = _row(confirmed, "relu", "paper")
    assert relu.best_baseline_us == 12.0
    assert relu.best_confirmed_speedup == 2.5


def test_high_headroom_sorts_ahead_of_saturated_rows(tmp_path):
    """@brief 排序把高余量放在带宽饱和之前。"""
    runs = tmp_path / "runs"
    preview = collect_headroom(target_id="apple-m4-max", dtype="fp16", runs_root=tmp_path / "missing", tasks_root=ROOT / "tasks")
    relu = _row(preview, "relu", "paper")
    fused = _row(preview, "gemm_silu", "paper")
    _write_report(runs, "relu", "paper", relu.lower_bound_us * 1.03)
    _write_report(runs, "gemm_silu", "paper", fused.lower_bound_us * 2.2)
    rows = collect_headroom(target_id="apple-m4-max", dtype="fp16", runs_root=runs, tasks_root=ROOT / "tasks")
    names = [row.tier for row in rows]
    assert names.index("high_headroom") < names.index("saturated_bandwidth")
