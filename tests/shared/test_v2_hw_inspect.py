"""CPU-only probes and diagnostics for the v2 hardware inspector."""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

_EIGHT = """
#include <metal_stdlib>
using namespace metal;
kernel void gemm_kernel(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[8];
}
"""

_THIRTY_TWO = """
#include <metal_stdlib>
using namespace metal;
kernel void gemm_kernel(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[32];
}
"""


def _threadgroup(count: int) -> str:
    return f"""
#include <metal_stdlib>
using namespace metal;
kernel void attention_kernel(uint3 gid [[thread_position_in_grid]]) {{
    threadgroup half tile[{count}];
}}
"""


def test_probe_hwfeatures_analyze_source_is_gpu_free():
    """@brief 纯文本分析能读出矩阵、threadgroup 字节和 barrier，且不装入 Metal。"""
    from taic.hwfeatures import analyze_source

    before = set(sys.modules)
    source = """
#include <metal_stdlib>
using namespace metal;
kernel void gemm_kernel(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[32];
    threadgroup half tile[8192];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"""
    evidence = analyze_source(source, consts={"M": 2048.0, "N": 2048.0, "K": 2048.0})
    added = set(sys.modules) - before
    assert "Metal" not in added
    assert "matrix" in evidence
    assert evidence["threadgroup"]["declared_bytes"] == 8192 * 2
    assert evidence["sync"]["threadgroup_barrier_sites"] >= 1


def test_probe_calibration_json_is_readable_without_metal():
    """@brief 标定 JSON 直接可读，读的过程不新增 Metal 模块。"""
    before = set(sys.modules)
    document = json.loads((ROOT / "targets/apple-m4-max/calibration.json").read_text(encoding="utf-8"))
    assert isinstance(document["peak_gbps"], float) and document["peak_gbps"] > 0
    assert isinstance(document["peak_tflops"]["fp16"], float) and document["peak_tflops"]["fp16"] > 0
    added = set(sys.modules) - before
    assert "Metal" not in added


def test_probe_all_v2_tasks_expose_work_accounting():
    """@brief 每道已发现的 v2 题都能装配出非负整数工作量。"""
    from tcenv.taskcontract.locator import discover_standard_tasks

    found = discover_standard_tasks(ROOT / "tasks", code_root=ROOT)
    assert found
    for item in found:
        accounting = item.load().instance.work_accounting
        assert accounting.flops >= 0 and accounting.min_bytes >= 0
        assert accounting.flops == int(accounting.flops)
        assert accounting.min_bytes == int(accounting.min_bytes)


def test_v2_hw_inspect_source_does_not_import_a_backend():
    """@brief 探针模块的 AST 里没有 ``tcenv.backends`` 导入。"""
    tree = ast.parse((ROOT / "taic/v2_hw_inspect.py").read_text(encoding="utf-8"))
    offenders = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        if any(name == "tcenv.backends" or name.startswith("tcenv.backends.") for name in names):
            offenders.append(node.lineno)
    assert offenders == []


def test_thirty_two_fragments_warn_and_eight_do_not():
    """@brief 32 槽触发溢出警告，8 槽不触发。"""
    from taic.v2_hw_inspect import inspect_v2_candidate

    wide = inspect_v2_candidate("gemm", _THIRTY_TWO, dtype="fp16", size="small")
    narrow = inspect_v2_candidate("gemm", _EIGHT, dtype="fp16", size="small")
    assert wide.delivery == "msl-single"
    assert wide.stages[0].largest_fragment_array == 32
    assert any("12 to 26" in warning for warning in wide.stages[0].warnings)
    assert narrow.stages[0].largest_fragment_array == 8
    assert not any("spill" in warning.lower() or "690" in warning for warning in narrow.stages[0].warnings)


_TWO_STAGE = """
#include <metal_stdlib>
using namespace metal;

void load_narrow() {
    simdgroup_float8x8 extra[2];
}

kernel void stage_narrow(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[8];
    load_narrow();
}

kernel void stage_wide(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[32];
}

/* taic-launch
launch stage_narrow grid=1,1,1 threads=1,1,1
launch stage_wide grid=1,1,1 threads=1,1,1
*/
"""

_NARROW_FILE = """
#include <metal_stdlib>
using namespace metal;

void load_narrow() {
    simdgroup_float8x8 extra[2];
}

kernel void stage_narrow(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[8];
    load_narrow();
}

kernel void unused_wide(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[32];
}
"""

_WIDE_FILE = """
#include <metal_stdlib>
using namespace metal;

kernel void stage_wide(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[32];
}

kernel void kda_piece0_extra(uint3 gid [[thread_position_in_grid]]) {
    simdgroup_float8x8 acc[9];
}
"""


def _bundle(files: list[tuple[str, str]], kernels: list[str]) -> str:
    from tcenv.track.envelope import encode_envelope
    from tcenv.track.plane import CandidateArtifact, CandidateEntry

    geometry = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [
            {
                "kernel": name,
                "grid": [1, 1, 1],
                "threads": [1, 1, 1],
                "threadgroup_bytes": 0,
                "workspace": [],
                "when": None,
                "derived_scalars": [],
            }
            for name in kernels
        ],
    }
    entries = [
        CandidateEntry(path=path, role="device_kernel", media_type="text/x-metal", data=source.encode("utf-8"))
        for path, source in files
    ]
    entries.append(
        CandidateEntry(
            path="host/execution_geometry.json",
            role="schedule",
            media_type="application/json",
            data=(json.dumps(geometry) + "\n").encode("utf-8"),
        )
    )
    return encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=tuple(entries)))


def test_fragment_pressure_stays_inside_the_launched_kernel_and_its_helpers():
    """@brief 同文件的另一阶段不会把槽位加进来；本阶段调用到的 helper 会计入。"""
    from taic.v2_hw_inspect import inspect_v2_candidate

    report = inspect_v2_candidate("gemm", _TWO_STAGE, dtype="fp16", size="small")
    assert [stage.kernel_name for stage in report.stages] == ["stage_narrow", "stage_wide"]
    narrow, wide = report.stages
    assert narrow.fragment_slots == 10
    assert narrow.largest_fragment_array == 8
    assert not any("12 to 26" in warning for warning in narrow.warnings)
    assert wide.fragment_slots == 32
    assert wide.largest_fragment_array == 32
    assert any("12 to 26" in warning for warning in wide.warnings)


def test_bundle_device_entries_match_exact_kernel_declarations():
    """@brief 多文件 bundle 按完整 ``kernel void`` 名字选源码，缺声明或重名都拒绝。"""
    from taic.v2_hw_inspect import inspect_v2_candidate

    report = inspect_v2_candidate(
        "gemm",
        _bundle(
            [("device/narrow.metal", _NARROW_FILE), ("device/wide.metal", _WIDE_FILE)],
            ["stage_narrow", "stage_wide"],
        ),
        dtype="fp16",
        size="small",
    )
    assert report.delivery == "msl-bundle"
    narrow, wide = report.stages
    assert narrow.source_path == "device/narrow.metal"
    assert narrow.fragment_slots == 10
    assert narrow.largest_fragment_array == 8
    assert not any("12 to 26" in warning for warning in narrow.warnings)
    assert wide.source_path == "device/wide.metal"
    assert wide.fragment_slots == 32
    assert wide.largest_fragment_array == 32
    assert any("12 to 26" in warning for warning in wide.warnings)

    prefix = [
        ("device/first.metal", "kernel void other_kernel(uint3 gid [[thread_position_in_grid]]) { simdgroup_float8x8 acc[4]; }\n"),
        ("device/extra.metal", "kernel void kda_piece0_extra(uint3 gid [[thread_position_in_grid]]) { simdgroup_float8x8 acc[8]; }\n"),
    ]
    with pytest.raises(ValueError, match=r"no device entry declares `kernel void kda_piece0`"):
        inspect_v2_candidate("gemm", _bundle(prefix, ["kda_piece0"]), dtype="fp16", size="small")

    duplicate = [
        ("device/a.metal", "kernel void stage_narrow(uint3 gid [[thread_position_in_grid]]) { simdgroup_float8x8 acc[8]; }\n"),
        ("device/b.metal", "kernel void stage_narrow(uint3 gid [[thread_position_in_grid]]) { simdgroup_float8x8 acc[32]; }\n"),
    ]
    with pytest.raises(ValueError, match="more than one device entry") as caught:
        inspect_v2_candidate("gemm", _bundle(duplicate, ["stage_narrow"]), dtype="fp16", size="small")
    message = str(caught.value)
    assert "device/a.metal" in message and "device/b.metal" in message


def test_threadgroup_resident_bound_drops_above_half_the_limit(monkeypatch):
    """@brief 9216 B 的静态驻留上界至少为 3；17408 B 降为 1 并带驻留警告。"""
    from taic.v2_hw_inspect import inspect_v2_candidate

    monkeypatch.delenv("TAIC_BIND_TASK_ROUTE", raising=False)
    base = inspect_v2_candidate("flash_attention", _threadgroup(4608), dtype="fp16", size="paper")
    padded = inspect_v2_candidate("flash_attention", _threadgroup(8704), dtype="fp16", size="paper")
    assert base.stages[0].static_threadgroup_bytes == 9216
    assert base.stages[0].estimated_resident_threadgroups >= 3
    assert not any("12-14%" in warning for warning in base.stages[0].warnings)
    assert padded.stages[0].static_threadgroup_bytes == 17408
    assert padded.stages[0].estimated_resident_threadgroups == 1
    assert any("12-14%" in warning and "static upper bound" in warning for warning in padded.stages[0].warnings)


def test_kda_bundle_reports_three_piece_stages():
    """@brief bundle envelope 解开 3 个 piece，并求值 grid/threads。"""
    from taic.v2_hw_inspect import inspect_v2_candidate
    from tests.support.kda_fixed_bundle import bundle_source

    report = inspect_v2_candidate("kda", bundle_source(), dtype="fp16", size="small")
    assert report.delivery == "msl-bundle"
    assert [stage.kernel_name for stage in report.stages] == ["kda_piece0", "kda_piece1", "kda_piece2"]
    assert [stage.grid for stage in report.stages] == [(2, 1, 1), (2, 1, 1), (2, 1, 1)]
    assert [stage.threads for stage in report.stages] == [(1, 1, 1), (1, 1, 1), (1, 1, 1)]
    assert "静态上界" in report.basis_note


def test_saturated_schema4_suppresses_hints_and_old_reports_drop_measurements():
    """@brief 95% roofline 不再列缺失项；旧 schema 的脏数字不进入诊断。"""
    from taic.v2_hw_inspect import inspect_v2_candidate

    source = """
#include <metal_stdlib>
using namespace metal;
kernel void gemm_kernel(uint3 gid [[thread_position_in_grid]]) {
}
"""
    plain = inspect_v2_candidate("gemm", source, dtype="fp16", size="small")
    assert plain.missing_hints
    current = {
        "schema_version": 4,
        "status": "ok",
        "candidate_us": 10.0,
        "roofline": {
            "bound": "compute",
            "time_over_lower_bound": 1.0 / 0.95,
            "achieved_tflops": 14.0,
            "compute_fraction": 0.95,
            "peak_tflops_measured": 14.7,
        },
    }
    saturated = inspect_v2_candidate("gemm", source, dtype="fp16", size="small", report=current)
    assert saturated.missing_hints == ()
    assert "suppressed" in saturated.performance_line
    stale = {
        "schema_version": 3,
        "status": "ok",
        "candidate_us": 1.0,
        "roofline": {"bound": "compute", "time_over_lower_bound": 1.01, "achieved_tflops": 987654.321},
    }
    ignored = inspect_v2_candidate("gemm", source, dtype="fp16", size="small", report=stale)
    rendered = ignored.format_text() + json.dumps(ignored.document())
    assert "987654" not in rendered
    assert ignored.missing_hints
    assert ignored.air_compiled is False


def test_air_compiler_can_be_injected_and_a_missing_toolchain_degrades(monkeypatch, tmp_path):
    """@brief 假编译器覆盖 --air；工具链缺失时源码诊断仍返回。"""
    from taic.v2_hw_inspect import inspect_v2_candidate, main

    seen: dict[str, str] = {}

    def compiled(source: str, **kwargs):
        seen["source"] = source
        return {"ok": True, "std": "metal4.0"}

    report = inspect_v2_candidate("gemm", _EIGHT, dtype="fp16", size="small", compile_air_evidence=True, air_compiler=compiled)
    assert report.air_compiled is True
    assert "constant constexpr uint M" in seen["source"]

    def missing_toolchain(source: str, **kwargs):
        return {"ok": False, "skipped": True, "error": "Metal toolchain not available (`xcrun -sdk macosx --find metal` failed)"}

    degraded = inspect_v2_candidate(
        "gemm", _EIGHT, dtype="fp16", size="small", compile_air_evidence=True, air_compiler=missing_toolchain
    )
    assert degraded.air_compiled is False
    assert degraded.stages[0].largest_fragment_array == 8

    path = tmp_path / "kernel.metal"
    path.write_text(_EIGHT, encoding="utf-8")

    def cli_compiler(source: str, **kwargs):
        seen["cli"] = source
        return {"ok": False, "skipped": True, "error": "no xcrun"}

    monkeypatch.setattr("taic.hwfeatures.compile_air", cli_compiler)
    assert main(["gemm", str(path), "--dtype", "fp16", "--size", "small", "--air"]) == 0
    assert "kernel void gemm_kernel" in seen["cli"]
    assert main(["gemm", str(tmp_path / "missing.metal")]) == 2


def test_cli_json_prints_the_report(tmp_path, capsys):
    """@brief JSON 输出退出 0，并带上 schema id。"""
    from taic.v2_hw_inspect import main

    path = tmp_path / "kernel.metal"
    path.write_text(_EIGHT, encoding="utf-8")
    assert main(["gemm", str(path), "--dtype", "fp16", "--size", "small", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_id"] == "taic.v2.hw-inspect.1"
    assert document["delivery"] == "msl-single"
    assert document["air_compiled"] is False
