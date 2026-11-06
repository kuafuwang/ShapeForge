"""V5 L2 的真实 Metal continuous-batching 验收（需显式 GPU 授权）。

Host Program 按本次 resolved total/count 生成 work list，device 按列表执行；一个 build 内的
多个 resolved batches 不得重编译，HostRun/launch/benchmark 证据必须能逐批重放。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.target import load_target
from tests.support.hostprogram_fixtures import (
    CAPACITY_PLANNER_FORGES_RESOLVED,
    CAPACITY_PLANNER_SOURCE,
    kda_bundle_source,
)

pytestmark = pytest.mark.gpu

TASK_DIR = Path(__file__).resolve().parents[3] / "tasks" / "varlen_kda"
BASE_SOURCE = (TASK_DIR / "known_good.metal").read_text(encoding="utf-8")


def _geometry() -> dict:
    return {
        "schema_version": 1,
        "source": "candidate",
        "stages": [
            {
                "kernel": "varlen_kda",
                "grid": ["num_seqs*H", 1, 1],
                "threads": [1, 1, 1],
                "threadgroup_bytes": 0,
                "workspace": [
                    {"name": "work_items", "zero": False},
                    {"name": "item_counts", "zero": False},
                ],
                "when": None,
                "derived_scalars": [],
            }
        ],
    }


def _candidate(planner: str = CAPACITY_PLANNER_SOURCE) -> str:
    source = BASE_SOURCE.replace(
        "/* taic-launch\nlaunch varlen_kda grid=num_seqs*H,1,1 threads=1,1,1\n*/",
        "/* taic-launch\nmath safe\n"
        "workspace work_items num_seqs*H*3*2*4\n"
        "workspace item_counts num_seqs*H*3*4\n"
        "launch varlen_kda grid=num_seqs*H,1,1 threads=1,1,1 "
        "workspace=work_items,item_counts\n*/",
    )
    source = source.replace(
        "                       device float* final_state [[buffer(11)]],",
        "                       device float* final_state [[buffer(11)]],\n"
        "                       device uint* work_items [[buffer(12)]],\n"
        "                       device int* item_counts [[buffer(13)]],",
    )
    source = source.replace(
        "    int begin = cu_seqlens[seq];\n    int end = cu_seqlens[seq + 1];",
        "    if (int(seq) >= item_counts[0]) { return; }\n"
        "    uint word0 = work_items[seq * 2];\n"
        "    uint chain = work_items[seq * 2 + 1];\n"
        "    if (chain != seq) { return; }\n"
        "    int begin = int(word0 & 0xFFFFu);\n"
        "    int end = begin + int(word0 >> 16);",
    )
    return kda_bundle_source(
        planner_source=planner,
        device_source=source,
        geometry=_geometry(),
    )


def _evaluate(source: str, *, mode: str = "quick"):
    binding = bind_task("varlen_kda", "fp16", "capacity")
    return evaluate_candidate(
        binding.locator,
        source,
        target=load_target("apple-m4-max"),
        mode=mode,
        reps=3,
        track_id="msl-bundle",
    )


def _why(report: dict) -> str:
    return " | ".join(
        str(case.get("message") or "")
        for case in (report.get("correctness") or {}).get("cases", [])
        if not case.get("ok")
    ) or str(report.get("message") or "")


def test_one_build_executes_every_resolved_batch_with_ordered_host_evidence():
    run = _evaluate(_candidate())
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    cases = report["correctness"]["cases"]
    assert len(cases) == 9
    assert all(case["ok"] for case in cases)
    shapes = [case["resolved_shape"] for case in cases]
    assert {(shape["total_tokens"], shape["num_seqs"]) for shape in shapes} == {
        (16, 4), (1, 1), (9, 3), (3, 2)
    }

    host_runs = run.evaluation["host_runs"]
    assert len(host_runs) == len(cases)
    assert [item["resolved_shape"] for item in host_runs] == shapes
    assert [item["evaluation_input_digest"] for item in host_runs] == [
        case["input_sha256"] for case in cases
    ]
    assert len({item["native_digest"] for item in host_runs}) == 1
    for item, shape in zip(host_runs, shapes):
        assert item["resolved_scalars"] == {
            "total_tokens": shape["total_tokens"],
            "num_seqs": shape["num_seqs"],
        }
        writes = {span["name"] for span in item["workspace_writes"]}
        assert writes == {"work_items", "item_counts"}
        assert item["native_module_digest"]

def test_benchmark_times_the_declared_continuous_batching_mixture():
    run = _evaluate(_candidate(), mode="full")
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    launches = run.evaluation["launch_runs"]
    cases = report["correctness"]["cases"]
    for index, case in enumerate(cases):
        trials = launches[index * 3 : (index + 1) * 3]
        assert len(trials) == 3
        assert all(item["resolved_shape"] == case["resolved_shape"] for item in trials)
        assert all(item["evaluation_input_digest"] == case["input_sha256"] for item in trials)
        assert all(item["case_ordinal"] == index for item in trials)
    batches = run.evaluation["benchmark_batches"]
    assert len(batches) >= 9
    shapes = [item["resolved_shape"] for item in batches]
    assert len({(shape["total_tokens"], shape["num_seqs"]) for shape in shapes}) == 4
    declared = list(bind_task("varlen_kda", "fp16", "capacity").contract.instance.case_domain.cases)
    assert [item["case"] for item in batches[: len(declared)]] == declared
    assert any(shape["full_capacity"] for shape in shapes)
    assert all(item["provider"] == "candidate" for item in batches)
    assert run.evaluation["benchmark_scope"] == "continuous_batching_mixture"
    assert run.evaluation["speedup_policy"]["status"] == "withheld"
    timed = next(item for item in run.evaluation["host_runs"] if item.get("timing_mode") == "timed")
    assert timed["native_wall_ns"] > 0
    assert timed["end_to_end_wall_ns"] >= timed["host_tiling_wall_ns"]
    protocol = run.evaluation["benchmark_protocol"]
    assert protocol["host_native_calls"] == protocol["command_buffers"]
    assert protocol["stage_args_rebuilds"] == 0
    assert protocol["read_only_restores"] == 0


def test_a_candidate_that_rewrites_a_resolved_scalar_is_incorrect_with_evidence():
    run = _evaluate(_candidate(CAPACITY_PLANNER_FORGES_RESOLVED))
    report = run.result.report
    assert report["status"] == "incorrect", _why(report)
    failure = run.evaluation["host_runs"][-1]
    assert failure["schema_id"] == "tcenv.hostprogram.run.2"
    assert failure["typed_error"] == "host_program_invalid"
    assert failure["resolved_shape"]["total_tokens"] >= 1
    assert failure["evaluation_input_digest"]
    assert failure["workspace_writes"] == []
