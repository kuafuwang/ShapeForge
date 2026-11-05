"""V4 前置诊断：Dyn-0 的空 TG / 空 stage 成本（真实 Metal，非性能发布）。

这个文件只回答一个问题：Dyn-0 的静态 grid 与静态 stage dispatch 中，空槽位和不干活的
stage 到底值不值得引入 Dyn-1 的动态几何。它不改变 candidate/geometry，也不宣称 speedup。

使用现有的 V3 host+device fixture：四阶段分别是 empty、piece0/1/2；grid 是
``num_seqs*H*MAX_SPLIT``。mixed 档有 6 条链、每条最多 3 个 piece，因此静态 piece grid 为
18，而实际 item 数通常少于 18；piece1/2 对不属于它的 item 立即退出。

每次 full benchmark 都与登记的 torch_mps baseline 交错，使用相同的 TimingDiscipline；
记录的是真实测量分布，不把一次 noisy 结果写成 Dyn-1 的冻结阈值。
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target

from ...support.hostprogram_fixtures import kda_bundle_source
from ...support.npu_sim.varlen_worklist import plan_pieces

pytestmark = pytest.mark.gpu

TARGET = load_target("apple-m4-max")


def _measure(cases: tuple[str, ...], reps: int) -> dict:
    """@brief 在真实 Metal 上测一组 case 的 Dyn-0 几何与 host/device/E2E 分解。

    @param cases L1 case 名（canonical / max_handoff_stress 等）。
    @param reps 计时重复次数（不是基准尝试数）。
    @return 可 JSON 序列化的诊断记录。
    """
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    run = evaluate_candidate(
        locator,
        kda_bundle_source(),
        target=TARGET,
        mode="full",
        reps=reps,
        cases=cases,
        track_id="msl-bundle",
    )
    report = run.result.report
    assert report["status"] == "ok", report.get("message")
    benchmark = report["benchmark"]
    timed = [item for item in run.evaluation["host_runs"] if item.get("timing_mode") == "timed"]
    assert timed, run.evaluation["host_runs"]
    last = timed[-1]
    protocol = run.evaluation["benchmark_protocol"]
    dispatch = run.evaluation["dispatch"]
    piece_dispatch = [item for item in dispatch if item["kernel"].startswith("varlen_piece")]
    assert piece_dispatch
    return {
        "cases": list(cases),
        "candidate_median_us": benchmark["candidate"]["median_us"],
        "baseline_median_us": {
            name: value["median_us"] for name, value in benchmark.get("baselines", {}).items()
        },
        "noise_cv": benchmark.get("noise_cv"),
        "attempts": benchmark.get("attempts"),
        "host_tiling_wall_ns": last.get("host_tiling_wall_ns"),
        "device_kernel_ns": last.get("device_kernel_ns"),
        "end_to_end_wall_ns": last.get("end_to_end_wall_ns"),
        "timing_mode": last.get("timing_mode"),
        "digest_mode": last.get("digest_mode"),
        "host_native_calls": protocol["host_native_calls"],
        "host_audit_calls": protocol["host_audit_calls"],
        "command_buffers": protocol["command_buffers"],
        "dispatch": dispatch,
    }


def test_dyn0_idle_geometry_diagnostic():
    """@brief 给出 Dyn-0 空槽位的可审计几何与真实测量，不预先决定 Dyn-1。

    mixed 档的静态 grid 是 18 个 piece 槽位。canonical 的 6 条链总共 16 个 item：两条
    len=4 的链被均衡切成 2/1/1，四条 len=8/2 的链分别切成 3/3/2 与 1/1，因此空槽位是 2；
    max_handoff_stress 的 6 条链总共 10 个 item：两条长链各 3 个 piece，四条单 token 链各 1 个，
    空槽位是 8。piece1/piece2 对不属于自己的 item 直接返回，这部分是 Dyn-1 可能省掉的调度成本，
    但实际占比必须由 Metal 测量决定。
    """
    canonical_items = plan_pieces((4, 8, 2), 2, 3)
    stress_items = plan_pieces((12, 1, 1), 2, 3)
    assert len(canonical_items) == 16
    assert len(stress_items) == 10
    grid = 3 * 2 * 3
    assert grid == 18

    # 多次独立 full run，避免把一次机器抖动当成 Dyn-1 的必然收益。
    observations = []
    for _ in range(3):
        observations.append(_measure(("canonical",), reps=15))
        observations.append(_measure(("max_handoff_stress",), reps=15))

    record = {
        "schema_id": "shapeforge.varlen.dyn0-idle-diagnostic.1",
        "target": TARGET.id,
        "grid": grid,
        "cases": {
            "canonical": {"items": len(canonical_items), "idle_slots": grid - len(canonical_items)},
            "max_handoff_stress": {"items": len(stress_items), "idle_slots": grid - len(stress_items)},
        },
        "observations": observations,
        "median_candidate_us_by_case": {
            case: statistics.median(
                item["candidate_median_us"] for item in observations if item["cases"] == [case]
            )
            for case in ("canonical", "max_handoff_stress")
        },
        "interpretation": "diagnostic only; no Dyn-1 decision or speedup is published",
    }
    output = Path("/tmp/varlen_dyn0_idle_diagnostic.json")
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))
    assert output.is_file()
    assert all(item["command_buffers"] == item["host_native_calls"] for item in observations)
    assert all(item["host_audit_calls"] == 1 for item in observations)
    assert all(item["device_kernel_ns"] is not None for item in observations)
