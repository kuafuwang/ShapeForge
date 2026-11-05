"""V3: a real host+device candidate on Metal — restricted C99 planner + Dyn-0 piece stages.

The candidate is the ``kda_planner.c`` Host Program (canonical mode) plus the unchanged V2 piece
kernels; only the device plan stage is replaced.  This file proves *mechanism*, not KDA
performance:

* the host planner writes ``work_items``/``item_counts``/``handoff_flags`` and the device stages
  consume them; all nine L1 cases pass on the real GPU;
* the evidence records the module/assembled/native digests, the actual read intervals, the valid
  spans, the initialization policy, the derived lifecycle and the host/device/E2E split;
* envelope refusals happen before execution (host-written vs device-zero conflict, a workspace no
  possible stage binds, dangerous source);
* a non-zero return or an out-of-capacity span is an ``incorrect`` candidate, never a partial
  write or an evaluator crash;
* input-dependent host work cannot be declared outside the timed window.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import evaluate_candidate
from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator
from tcenv.contract_errors import TaskContractError
from tcenv.remote.artifact_bundle import MAX_NATIVE_DESCRIPTOR_BYTES
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT
from tcenv.taskcontract.runner import evaluate_task
from tcenv.target import load_target
from tcenv.track import CandidateLifecycle, HostStep, PrepareDeclaration, candidate_geometry
from tcenv.track.envelope import decode_envelope

from ...support.hostprogram_fixtures import (
    HOST_WRITTEN,
    PLANNER_SOURCE,
    host_planned_geometry_document,
    kda_bundle_source,
)

pytestmark = pytest.mark.gpu

TARGET = load_target("apple-m4-max")


def _evaluate(source: str, *, mode: str = "quick", cases: tuple[str, ...] | None = None):
    """@brief 用真实 msl-bundle 交付路径评测 host+device fixture。"""
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    return evaluate_candidate(
        locator,
        source,
        target=TARGET,
        mode=mode,
        reps=3,
        cases=cases,
        track_id="msl-bundle",
    )


def _mutate(source: str, needle: str, replacement: str) -> str:
    assert source.count(needle) == 1, needle
    return source.replace(needle, replacement)


def _why(report: dict) -> str:
    parts = [f"status={report.get('status')!r}"]
    if report.get("stage"):
        parts.append(f"stage={report['stage']!r}")
    if report.get("message"):
        parts.append(str(report["message"]))
    errors = (report.get("compile") or {}).get("errors")
    if errors:
        parts.append(f"compile: {errors[:600]}")
    violations = (report.get("safety") or {}).get("violations") or []
    if violations:
        parts.append("safety: " + "; ".join(str(item) for item in violations[:3]))
    return " | ".join(parts)


def _case_message(report: dict) -> str:
    cases = (report.get("correctness") or {}).get("cases") or []
    return " | ".join(str(case.get("message") or "") for case in cases if not case.get("ok"))


# --- the host+device delivery ---------------------------------------------------------------


def test_the_c99_planner_drives_the_dyn0_piece_stages_on_every_l1_case():
    """@brief 只跑正确性：真实 native 规划 + 真实 Metal 分段/传递/终态。"""
    run = _evaluate(kda_bundle_source())
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    assert report["correctness"]["ok"] is True
    cases = report["correctness"]["cases"]
    assert len(cases) == 9
    assert all(case["ok"] for case in cases)

    evaluation = run.evaluation
    assert evaluation["host_tiling"] is None  # the linear representation is not used
    program = evaluation["host_program"]
    assert program["entry_path"] == "host/tiling.c"
    assert program["used_workspaces"] == list(HOST_WRITTEN)
    assert program["execution_backend"] == "native-c99"
    assert program["memory_bytes"] > 0
    # The module digest is bound into the native descriptor, and therefore into native_digest.
    descriptor = (((report.get("build") or {}).get("native") or {}).get("descriptor")) or {}
    assert descriptor["host_program"]["native_sha256"] == program["native_sha256"]
    assert descriptor["host_program"]["assembled_host_sha256"] == program["assembled_host_sha256"]
    descriptor_bytes = len(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    assert descriptor_bytes <= MAX_NATIVE_DESCRIPTOR_BYTES
    # V2 fixture B is 7,209 B; the native-c99 identity/layout block brings the closed descriptor
    # to 7,505 B without adding bindings or dylib bytes.  Freeze the measured size.
    assert descriptor_bytes == 7505

    runs = evaluation["host_runs"]
    assert runs
    last = runs[-1]
    assert last["schema_id"] == "tcenv.hostprogram.run.2"
    assert last["native_module_digest"] == program["native_sha256"]
    assert last["input_dependent"] is True
    assert last["preprocesses_inputs"] is True
    assert last["inside_timing"] is True
    assert last["simulated"] is False
    assert last["publication_allowed"] is False
    assert last["geometry_digest"] == evaluation["geometry_identity"]
    assert last["launch_geometry_digest"] is None
    reads = {item["name"]: item for item in last["input_reads"]}
    assert reads["cu_seqlens"]["count"] == 4
    assert reads["cu_seqlens"]["sha256"]
    writes = {item["name"]: item for item in last["workspace_writes"]}
    assert set(writes) == set(HOST_WRITTEN)
    assert writes["work_items"]["offset"] == 0
    # The recorded host run is the last L1 case (`fresh_distribution`), not canonical; both are
    # two words per item and stay inside the declared capacity.
    assert writes["work_items"]["count"] % 2 == 0
    assert 0 < writes["work_items"]["count"] <= 36
    assert writes["item_counts"]["count"] == 1
    assert writes["handoff_flags"]["count"] == 12
    assert last["unwritten_policy"]["work_items"] == "host"
    assert last["unwritten_policy"]["handoff_states"] == "poison-sentinel"
    # The host-written blocks are not in the device-zero set, so the timed launch needs no fill
    # command buffer before the single stage command buffer.
    zeroed = {name for name, zero in evaluation["workspace_zero"].items() if zero}
    assert zeroed.isdisjoint(HOST_WRITTEN)

    assert [item["kernel"] for item in evaluation["dispatch"]] == [
        "varlen_empty",
        "varlen_piece0",
        "varlen_piece1",
        "varlen_piece2",
    ]


def test_the_timed_host_run_is_one_native_call_and_one_command_buffer_with_a_host_device_split():
    """@brief full 计时：host/device/E2E 分开记录，发布仍只看 E2E；这里只验证证据形状。"""
    run = _evaluate(kda_bundle_source(), mode="full")
    report = run.result.report
    assert report["status"] == "ok", _why(report)
    assert report["benchmark"]["candidate"]["median_us"] > 0
    runs = run.evaluation["host_runs"]
    timed = [item for item in runs if item.get("timing_mode") == "timed"]
    assert timed, runs
    last = timed[-1]
    assert last["digest_mode"] == "audit"
    assert last["host_tiling_wall_ns"] > 0
    assert last["end_to_end_wall_ns"] >= last["host_tiling_wall_ns"]
    assert last["device_kernel_ns"] is not None and last["device_kernel_ns"] > 0
    # The timed evidence never carries per-iteration digests: the audit run does.
    assert last["input_reads"] and last["workspace_writes"]

    protocol = run.evaluation["benchmark_protocol"]
    assert protocol["host_native_calls"] > 0
    # Exactly one audit launch (digests); every other launch was timed and did not hash.
    assert protocol["host_audit_calls"] == 1
    assert protocol["host_audit_calls"] < protocol["host_native_calls"]
    # One native call -> one committed command buffer; no fill buffer and no rebuilt stage args.
    assert protocol["command_buffers"] == protocol["host_native_calls"]
    assert protocol["stage_args_rebuilds"] == 0
    assert protocol["read_only_restores"] == 0


# --- envelope refusals ----------------------------------------------------------------------


def test_a_host_written_workspace_may_not_also_be_device_zeroed():
    """@brief host 写与 device_zero 同块冲突在 envelope/构建前拒绝。"""
    document = host_planned_geometry_document()
    for stage in document["stages"]:
        for workspace in stage["workspace"]:
            if workspace["name"] == "work_items":
                workspace["zero"] = True
    with pytest.raises(TaskContractError, match="device-zero"):
        _evaluate(kda_bundle_source(geometry=document), cases=("canonical",))


def test_a_workspace_no_possible_stage_binds_is_refused():
    """@brief 引用没有 stage 绑定的 workspace（如 token_stats）会被拒绝，而不是分配大表。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->item_counts[0] = (int32_t)item;",
        "workspaces->item_counts[0] = (int32_t)item; workspaces->token_stats[0] = 0.0f;",
    )
    with pytest.raises(TaskContractError, match="possible stage binds"):
        _evaluate(kda_bundle_source(planner_source=source), cases=("canonical",))


def test_a_dangerous_include_is_refused_before_any_compiler_runs():
    """@brief 投影期静态预检：host include 不进入 Clang。"""
    source = "#include <stdio.h>\n" + PLANNER_SOURCE
    with pytest.raises(TaskContractError, match="#include|preflight|host program"):
        _evaluate(kda_bundle_source(planner_source=source), cases=("canonical",))


# --- candidate runtime failures -------------------------------------------------------------


def test_a_nonzero_host_return_is_incorrect_not_a_runtime_error():
    """@brief 返回非 0 → incorrect，且没有任何 span 被提交。"""
    source = _mutate(
        PLANNER_SOURCE,
        "return sf_plan_canonical(inputs, shapes, workspaces);",
        "return SF_HOST_REFUSED;",
    )
    run = _evaluate(kda_bundle_source(planner_source=source), cases=("canonical",))
    report = run.result.report
    assert report["status"] == "incorrect", _why(report)
    assert "host program failed" in _case_message(report)
    assert "host_program_invalid" in _case_message(report)
    failure = run.evaluation["host_runs"][-1]
    assert failure["schema_id"] == "tcenv.hostprogram.run.1"
    assert failure["host_status"] != 0
    assert failure["typed_error"] == "host_program_invalid"
    assert failure["workspace_writes"] == []
    assert failure["simulated"] is False and failure["publication_allowed"] is False


def test_a_host_span_beyond_the_declared_capacity_is_incorrect():
    """@brief span 越界 → incorrect，不是 evaluator crash。"""
    source = _mutate(
        PLANNER_SOURCE,
        "workspaces->work_items_count = item_count * 2u;",
        "workspaces->work_items_count = SF_WS_WORK_ITEMS_CAPACITY_ELEMS + 2u;",
    )
    run = _evaluate(kda_bundle_source(planner_source=source), cases=("canonical",))
    report = run.result.report
    assert report["status"] == "incorrect", _why(report)
    assert "capacity" in _case_message(report)


def test_a_host_plan_whose_pieces_do_not_cover_the_segment_is_incorrect():
    """@brief host 写出的 work list 与真实分段不符 → 正确性失败（不是被 device 兜住）。"""
    source = _mutate(
        PLANNER_SOURCE,
        "uint32_t base = len / pieces;\n        uint32_t rem = len % pieces;",
        "uint32_t base = 1u;\n        uint32_t rem = 0u;",
    )
    run = _evaluate(kda_bundle_source(planner_source=source), cases=("skewed",))
    report = run.result.report
    # A plan that does not cover the segment leaves output unwritten; the poison/zero trials then
    # disagree, which is refused as `unsafe`.  Either way it is never accepted as correct.
    assert report["status"] in ("incorrect", "unsafe"), _why(report)


# --- lifecycle ------------------------------------------------------------------------------


def _timing_off_lifecycle_factory():
    locator = bind_task("varlen_kda", "fp16", "mixed").locator
    contract = load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed")

    def factory(chosen):
        return CandidateLifecycle(
            track_id="msl-bundle",
            geometry=chosen,
            host_steps=(
                HostStep(
                    PrepareDeclaration(name="host_tiling", input_dependent=False, inside_timing=False),
                    role="host_tiling",
                ),
                HostStep(
                    PrepareDeclaration(name="schedule", input_dependent=False, inside_timing=False),
                    role="schedule",
                ),
            ),
            candidate_sha256=decode_envelope(kda_bundle_source()).candidate_sha256,
            target_id=TARGET.id,
            instance_key=contract.fingerprint(),
        )

    return locator, factory


def test_input_dependent_host_work_cannot_be_declared_outside_the_timed_window():
    """@brief lifecycle 谎称 input-independent/计时外 → build 拒绝，报告 compile_error。"""
    locator, factory = _timing_off_lifecycle_factory()
    candidate = decode_envelope(kda_bundle_source())
    geometry = candidate_geometry(candidate)
    result = evaluate_task(
        locator,
        track_id="msl-bundle",
        candidate=candidate,
        builder=MetalTrackBuilder(target_id=TARGET.id),
        evaluator=MetalTrackEvaluator(target_id=TARGET.id, mode="quick", cases=("canonical",)),
        geometry_resolver=lambda _candidate: geometry,
        lifecycle_factory=factory,
        target_id=TARGET.id,
    )
    report = result.report
    assert report["status"] == "compile_error", _why(report)
    assert "inside_timing" in str((report.get("compile") or {}).get("errors")), report.get("compile")


def test_the_statement_msl_example_compiles():
    """@brief 题面的最小 device 示例可被标准 Metal 编译器编译（不执行 device）。"""
    import subprocess
    import tempfile

    from tcenv.hostprogram.contract import host_contract_spec, minimal_device_example
    from tcenv.taskcontract import load_task_contract
    from tcenv.track.abi import derive_tensor_memory_contract

    contract = load_task_contract(REPO_ROOT / "tasks" / "varlen_kda", dtype="fp16", size="mixed")
    abi = derive_tensor_memory_contract(contract.instance, target=TARGET)
    spec = host_contract_spec(instance=contract.instance, tensors=abi.tensors)
    source = minimal_device_example(spec)
    assert "kernel void example" in source
    with tempfile.TemporaryDirectory(prefix="sf-msl-example-") as tmp:
        path = Path(tmp) / "example.metal"
        out = Path(tmp) / "example.air"
        path.write_text(source, encoding="utf-8")
        done = subprocess.run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(out)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    assert done.returncode == 0, done.stderr[-2000:]
