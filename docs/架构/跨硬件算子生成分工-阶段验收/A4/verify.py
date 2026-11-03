#!/usr/bin/env python3
"""A4 acceptance verifier: real Mac slices for msl-single/msl-bundle and v2 evaluate_task.

The verifier itself starts no model.  Its GPU phase intentionally opens the Mac device and runs
the fixed real-slice/negative tests in ``tests/backends/metal/test_track_a4.py``; that is the A4 evidence,
not a simulated substitute.  It refuses to claim A4 if the evidence files are missing, if the
two tracks are not available with A4 evidence, if a recorded dispatch differs from the launch
block, if the bounded worker selected a different Track/registry/rendering identity than the
control host froze, if the control host compiled the candidate itself, or if the three slices
did not produce distinct candidate/geometry/build identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).resolve().parent / "manifest.sha256"
A4_BASE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"

DELIVERABLES: tuple[str, ...] = (
    "tcenv/backends/metal/track_adapter.py",
    "tcenv/backends/metal/v2_runtime.py",
    "tcenv/backends/metal/stage_binding.py",
    "tcenv/backends/metal/runtime.py",
    "tcenv/backends/metal/sanitize.py",
    "tcenv/backends/metal/screen.py",
    "tcenv/backends/metal/source_builder.py",
    "tcenv/backends/metal/backend.py",
    "tcenv/evaluate.py",
    "tcenv/v2_worker.py",
    "tcenv/track/plane.py",
    "tcenv/taskcontract/runner.py",
    "tcenv/remote/execution_geometry.py",
    "backends/metal/tracks.json",
    "tasks/sigmoid_shift/opspec.json",
    "tasks/sigmoid_shift/reference.py",
    "tests/backends/metal/test_track_a4.py",
    "tests/backends/metal/test_track_registry.py",
    "tests/backends/metal/test_stage_binding.py",
    "tests/backends/metal/test_build_profiles.py",
    "tests/support/report_build.py",
    "tests/support/v2_tasks.py",
    "tests/support/v2_mapping.py",
    "tests/support/fake_v2_worker_stuck.py",
    "tests/support/fake_v2_worker_exclusive.py",
    "tests/support/fake_v2_worker_stuck_compile.py",
    "tests/support/fake_v2_worker_drifting_track.py",
    "tests/support/fake_v2_worker_drifting_build.py",
    "tests/shared/test_evaluate_task.py",
    "tests/shared/test_track_protocol.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_gpulock.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/verify.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/single.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/bundle.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/newop.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/fusion.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/derived.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/readwrite.json",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence/multistage.json",
)

FIXED_CPU_NODE_IDS: tuple[str, ...] = (
    "tests/backends/metal/test_track_a4.py::test_a4_msl_single_refuses_to_invent_a_geometry_without_a_launch_block",
    "tests/backends/metal/test_track_a4.py::test_a4_a_fallback_or_dispatch_deviation_is_a_mismatch",
    "tests/backends/metal/test_track_a4.py::test_a4_workspace_size_mismatch_is_refused_before_allocation",
    "tests/backends/metal/test_track_a4.py::test_a4_host_tiling_plan_is_closed_and_required_not_a_renamed_kernel",
    "tests/backends/metal/test_track_a4.py::test_a4_v2_host_path_include_is_refused_before_the_compiler_is_called",
    "tests/backends/metal/test_track_a4.py::test_a4_external_deadline_kills_a_stuck_v2_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_external_deadline_kills_a_stuck_compile_in_the_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_a_timed_out_bounded_worker_has_no_fabricated_build",
    "tests/backends/metal/test_track_a4.py::test_a4_a_crashed_bounded_worker_has_no_fabricated_build",
    "tests/backends/metal/test_track_a4.py::test_a4_the_report_status_and_build_shape_must_stay_in_the_closed_table",
    "tests/backends/metal/test_track_a4.py::test_a4_an_evaluator_build_must_carry_exactly_the_candidate_and_frozen_geometry",
    "tests/backends/metal/test_track_a4.py::test_a4_a_single_source_build_must_carry_the_canonical_source_manifest",
    "tests/backends/metal/test_track_a4.py::test_a4_a_partial_build_must_still_be_a_native_build",
    "tests/backends/metal/test_track_a4.py::test_a4_frozen_track_selection_never_reads_the_default_registry",
    "tests/backends/metal/test_track_a4.py::test_a4_frozen_track_drift_is_refused_before_any_build",
    "tests/backends/metal/test_track_registry.py",
    "tests/shared/test_evaluate_task.py",
    "tests/shared/test_task_discovery.py",
    "tests/shared/test_track_geometry.py",
    "tests/shared/test_track_protocol.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_track_abi.py",
    "tests/shared/test_track_bundle.py",
    "tests/shared/test_track_lifecycle.py",
    "tests/shared/test_geometry_artifact.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_cross_hardware_boundary.py",
    "tests/shared/test_gpulock.py",
    "tests/backends/metal/test_stage_binding.py",
    "tests/backends/metal/test_build_profiles.py",
)

GPU_NODE_IDS: tuple[str, ...] = (
    "tests/backends/metal/test_track_a4.py::test_a4_msl_single_rmsnorm_real_build_abi_correctness_and_timing",
    "tests/backends/metal/test_track_a4.py::test_a4_msl_bundle_real_host_tiling_and_device_stage",
    "tests/backends/metal/test_track_a4.py::test_a4_two_active_device_stages_chain_output_and_reflect_each_stage",
    "tests/backends/metal/test_track_a4.py::test_a4_new_operator_is_discovered_and_evaluated_without_a_central_list",
    "tests/backends/metal/test_track_a4.py::test_a4_fusion_geometry_comparison_replay_and_evidence",
    "tests/backends/metal/test_track_a4.py::test_a4_derived_scalar_is_set_by_name_and_inactive_when_stage_is_not_dispatched",
    "tests/backends/metal/test_track_a4.py::test_a4_read_write_initial_values_are_restored_on_every_trial",
    "tests/backends/metal/test_track_a4.py::test_a4_actual_runtime_dispatch_cannot_satisfy_fallback_or_a_changed_grid",
    "tests/backends/metal/test_track_a4.py::test_a4_real_runtime_refuses_a_fallback_geometry_against_candidate_selection",
    "tests/backends/metal/test_track_a4.py::test_a4_incorrect_single_is_reported_incorrect_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_incorrect_bundle_is_reported_incorrect_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_unwritten_output_is_reported_unsafe_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_benchmark_checks_guards_after_reps_and_catches_a_second_launch_oob",
    "tests/backends/metal/test_track_a4.py::test_a4_real_evaluator_refuses_a_drifted_full_build",
    "tests/backends/metal/test_track_a4.py::test_a4_fill_workspaces_clears_a_non_word_aligned_span",
    "tests/backends/metal/test_track_a4.py::test_a4_run_pipeline_fill_clears_a_non_word_aligned_span",
    "tests/backends/metal/test_track_a4.py::test_a4_a_bundle_that_ships_host_work_cannot_skip_its_host_step",
    "tests/backends/metal/test_track_a4.py::test_a4_declared_zero_workspace_is_cleared_inside_timing_and_host_tiling_restores_it",
    "tests/backends/metal/test_track_a4.py::test_a4_real_stage_refuses_a_workspace_it_did_not_declare",
    "tests/backends/metal/test_track_a4.py::test_a4_the_worker_uses_the_control_host_frozen_track_not_the_default_registry",
    "tests/backends/metal/test_track_a4.py::test_a4_parent_never_compiles_the_candidate_before_the_bounded_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_track_drift_is_refused",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_build_that_disagrees_with_the_canonical_geometry_is_refused",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_compile_error_is_a_partial_build_not_a_fabricated_full_one",
)

NEGATIVE_NODE_IDS: tuple[str, ...] = (
    "tests/backends/metal/test_track_a4.py::test_a4_msl_single_refuses_to_invent_a_geometry_without_a_launch_block",
    "tests/backends/metal/test_track_a4.py::test_a4_a_fallback_or_dispatch_deviation_is_a_mismatch",
    "tests/backends/metal/test_track_a4.py::test_a4_actual_runtime_dispatch_cannot_satisfy_fallback_or_a_changed_grid",
    "tests/backends/metal/test_track_a4.py::test_a4_real_runtime_refuses_a_fallback_geometry_against_candidate_selection",
    "tests/backends/metal/test_track_a4.py::test_a4_incorrect_single_is_reported_incorrect_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_incorrect_bundle_is_reported_incorrect_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_unwritten_output_is_reported_unsafe_with_the_attached_build",
    "tests/backends/metal/test_track_a4.py::test_a4_benchmark_checks_guards_after_reps_and_catches_a_second_launch_oob",
    "tests/backends/metal/test_track_a4.py::test_a4_real_evaluator_refuses_a_drifted_full_build",
    "tests/backends/metal/test_track_a4.py::test_a4_fill_workspaces_clears_a_non_word_aligned_span",
    "tests/backends/metal/test_track_a4.py::test_a4_run_pipeline_fill_clears_a_non_word_aligned_span",
    "tests/backends/metal/test_track_a4.py::test_a4_workspace_size_mismatch_is_refused_before_allocation",
    "tests/backends/metal/test_track_a4.py::test_a4_host_tiling_plan_is_closed_and_required_not_a_renamed_kernel",
    "tests/backends/metal/test_track_a4.py::test_a4_a_bundle_that_ships_host_work_cannot_skip_its_host_step",
    "tests/backends/metal/test_track_a4.py::test_a4_real_stage_refuses_a_workspace_it_did_not_declare",
    "tests/backends/metal/test_track_a4.py::test_a4_v2_host_path_include_is_refused_before_the_compiler_is_called",
    "tests/backends/metal/test_track_a4.py::test_a4_external_deadline_kills_a_stuck_v2_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_external_deadline_kills_a_stuck_compile_in_the_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_a_timed_out_bounded_worker_has_no_fabricated_build",
    "tests/backends/metal/test_track_a4.py::test_a4_a_crashed_bounded_worker_has_no_fabricated_build",
    "tests/backends/metal/test_track_a4.py::test_a4_the_report_status_and_build_shape_must_stay_in_the_closed_table",
    "tests/backends/metal/test_track_a4.py::test_a4_an_evaluator_build_must_carry_exactly_the_candidate_and_frozen_geometry",
    "tests/backends/metal/test_track_a4.py::test_a4_a_single_source_build_must_carry_the_canonical_source_manifest",
    "tests/backends/metal/test_track_a4.py::test_a4_a_partial_build_must_still_be_a_native_build",
    "tests/backends/metal/test_track_a4.py::test_a4_frozen_track_drift_is_refused_before_any_build",
    "tests/backends/metal/test_track_a4.py::test_a4_parent_never_compiles_the_candidate_before_the_bounded_worker",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_track_drift_is_refused",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_build_that_disagrees_with_the_canonical_geometry_is_refused",
    "tests/backends/metal/test_track_a4.py::test_a4_a_worker_compile_error_is_a_partial_build_not_a_fabricated_full_one",
    "tests/backends/metal/test_track_registry.py::test_the_metal_registry_document_is_closed_and_op_agnostic",
    "tests/shared/test_evaluate_task.py::test_track_selection_refuses_missing_unknown_and_interface_only_tracks",
    "tests/shared/test_evaluate_task.py::test_evaluate_task_refuses_a_bundle_without_a_lifecycle_or_a_drifting_report",
    "tests/shared/test_evaluate_task.py::test_a_single_source_candidate_uses_the_injected_launch_parser",
    "tests/shared/test_geometry_artifact.py::test_attaching_a_geometry_from_another_candidate_or_twice_is_refused",
    "tests/shared/test_geometry_artifact.py::test_a_tampered_descriptor_or_geometry_entry_fails_cross_audit",
    "tests/shared/test_track_geometry.py::test_the_fallback_is_independent_and_enters_the_execution_identity",
    "tests/shared/test_track_geometry.py::test_dispatch_parity_catches_a_worker_that_drifted_from_the_recorded_geometry",
)

FROZEN_SURFACE: tuple[str, ...] = (
    "proto",
    "tcenv/remote/v1",
    "tcenv/remote/artifact_bundle.py",
    "tcenv/remote/contracts.py",
    "tcenv/remote/grpc_client.py",
    "tcenv/remote/grpc_contract.py",
    "tcenv/remote/grpc_service.py",
    "tcenv/remote/task_pack.py",
    "tcenv/task.py",
    "tcenv/contract.py",
    "tcenv/build_projection.py",
    "tcenv/report.py",
    "tcenv/schema.py",
    "tcenv/verify.py",
    "tcenv/backends/metal/lowering.py",
    "tcenv/track/schema/track.v1.schema.json",
    "tcenv/track/schema/tracks.v1.schema.json",
    "tcenv/opspec/schema/opspec.v1.schema.json",
    "targets",
    "backends/metal/backend.toml",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_op_desc_summary.py",
    "docs/架构/跨硬件算子生成分工-G0",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3",
)

REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|america|nvidia)\b", re.IGNORECASE
)


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。"""
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    """@brief 运行 git 并返回 stdout。"""
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def write_manifest() -> None:
    """@brief 重写 A4 manifest.sha256。"""
    lines = [f"{sha256(ROOT / name)}  {name}" for name in DELIVERABLES]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    """@brief 校验 manifest 的摘要与文件集合完全一致。"""
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"bad digest: {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen[name] = digest
    expected = set(DELIVERABLES)
    require(set(seen) == expected, f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}")
    for name, digest in seen.items():
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def verify_frozen_surface() -> None:
    """@brief 六 RPC/field/HMAC/BlobChannel、schema 4、v1 合同与目标面未动；A4 只新增任务与 track 状态。"""
    base = _git("rev-parse", "--verify", f"{A4_BASE_HEAD}^{{commit}}").strip()
    require(base == A4_BASE_HEAD, f"A4 base commit is not available: {A4_BASE_HEAD}")
    changed = _git("diff", "--name-only", A4_BASE_HEAD, "--", *FROZEN_SURFACE).splitlines()
    require(changed == [], "frozen surface changed since the A4 base commit: " + ", ".join(changed))
    # A4 adds one production v2 task.  It must not rewrite any existing (v1) task.
    task_diff = _git("diff", "--name-only", A4_BASE_HEAD, "--", "tasks").splitlines()
    require(task_diff == [], "A4 modified an existing task: " + ", ".join(task_diff))
    task_status = _git("status", "--porcelain", "--", "tasks").splitlines()
    new_task = [line for line in task_status if line.startswith("?? tasks/sigmoid_shift/")]
    require(len(new_task) == len(task_status), "A4 changed a tracked task or added an unexpected task: " + ", ".join(task_status))
    for name in (
        "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
        "docs/架构/跨硬件算子生成分工-阶段验收/A2.md",
        "docs/架构/跨硬件算子生成分工-阶段验收/A3.md",
        "docs/架构/跨硬件算子生成分工-阶段验收/A3/verify.py",
    ):
        require((ROOT / name).is_file(), f"historical acceptance evidence is missing: {name}")
    print(f"OK: frozen surface untouched since {A4_BASE_HEAD[:12]} ({len(FROZEN_SURFACE)} paths; task diff empty)")


def verify_tracks_and_task() -> None:
    """@brief 两个 track 真实可用；新算子只有标准合同 + reference，且不在 v1 中央清单。"""
    sys.path.insert(0, str(ROOT))
    from tasks import get_task
    from tcenv.taskcontract import discover_standard_tasks, load_task_contract
    from tcenv.track import load_track_registry

    registry = load_track_registry()
    require([track.track_id for track in registry.available()] == ["msl-single", "msl-bundle"], "the two Metal tracks are not both available")
    for track in registry.tracks:
        require(track.status == "available", f"{track.track_id} is not available")
        require("tests/backends/metal/test_track_a4.py" in track.evidence, f"{track.track_id} has no A4 evidence")
    require(registry.get("msl-bundle").delivery == "candidate_bundle", "msl-bundle delivery changed")
    require(registry.get("msl-single").delivery == "single_source", "msl-single delivery changed")
    import pytest

    with pytest.raises(KeyError):
        get_task("sigmoid_shift", "fp16", "small")
    found = [locator for locator in discover_standard_tasks(ROOT / "tasks", code_root=ROOT) if locator.op_name == "sigmoid_shift"]
    require(len(found) == 1, f"new operator discovery found {len(found)} locators")
    contract = load_task_contract(ROOT / "tasks/sigmoid_shift")
    require(contract.op_name == "sigmoid_shift" and contract.instance.symbols["N"] > 0, "new operator contract is not v2")
    require(contract.correctness_only is True, "new operator is not correctness-only without baselines")
    print("OK: both tracks available with A4 evidence; sigmoid_shift is discovered, v2 and correctness-only")


def verify_adapter_wiring() -> None:
    """@brief A4 的真实适配器/评测注入是生产路径，不是测试替身。"""
    adapter = (ROOT / "tcenv/backends/metal/track_adapter.py").read_text(encoding="utf-8")
    for needle in (
        "class MetalTrackBuilder",
        "class MetalTrackEvaluator",
        "class MetalV2Executor",
        "MetalSourceBuilder(",
        "_evaluate_in_process(",
        "execution=executor",
        "msl_single_geometry_resolver",
        "def prepare(self, request: CandidateBuildRequest)",
        "def owns_build(self)",
        "self.evaluated_build = dict(build)",
        "worker_module",
        "metal_compile_failure_report",
    ):
        require(needle in adapter, f"A4 adapter is missing {needle}")
    worker = (ROOT / "tcenv/v2_worker.py").read_text(encoding="utf-8")
    for needle in (
        "frozen_track",
        "frozen_document",
        "frozen_track=resolved[\"frozen_track\"]",
        "\"build\": result.build",
        "\"presentation_sha256\": result.presentation_sha256",
        "compile_deadline_s",
        "with released():",
    ):
        require(needle in worker, f"the v2 worker is missing {needle}")
    runner = (ROOT / "tcenv/taskcontract/runner.py").read_text(encoding="utf-8")
    for needle in (
        "FROZEN_TRACK_SCHEMA_ID",
        "verify_evaluator_build",
        "owns_build = bool(getattr(evaluator, \"owns_build\", False))",
        "prepare(request)",
        "evaluated_build",
        "the bounded worker's TrackSpec/registry/rendering identity drifted",
        "may not also name a registry path",
    ):
        require(needle in runner, f"evaluate_task is missing {needle}")
    require("frozen=frozen_track" in runner, "evaluate_task does not pass the frozen track to selection")
    v2 = (ROOT / "tcenv/backends/metal/v2_runtime.py").read_text(encoding="utf-8")
    for needle in ("class MetalV2Runtime", "HostTilingPlan", "bind_pipeline_arguments", "correctness_launch_plan", "workspace_size_problems", "read_write", "DispatchRecord", "guard_violations", "reset_guards=False"):
        require(needle in v2, f"v2 runtime is missing {needle}")
    require("nbytes // 4" not in v2, "v2 fill still truncates a workspace span to whole words")
    for needle in ("build_outcome=request.build", "_build_identity_view(build) != _build_identity_view(expected)", "project_full(request.build"):
        require(needle in adapter, f"A4 adapter is missing {needle}")
    runtime = (ROOT / "tcenv/backends/metal/runtime.py").read_text(encoding="utf-8")
    require("dispatch_recorder" in runtime, "the Metal runtime has no actual dispatch recorder")
    require("fill_workspaces" in runtime, "the runtime cannot complete a workspace fill before Host Tiling")
    sanitize = (ROOT / "tcenv/backends/metal/sanitize.py").read_text(encoding="utf-8")
    require("declared_zero_only" in sanitize, "legacy ArgSet still exposes every workspace as a timed fill")
    backend = (ROOT / "tcenv/backends/metal/backend.py").read_text(encoding="utf-8")
    require("workspace_fills(declared_zero_only=True)" in backend, "v1 pipeline still clears non-zero workspaces in timing")
    require("dispatchThreadgroups_threadsPerThreadgroup_" in runtime, "the recorder is not on the real dispatch path")
    evaluate = (ROOT / "tcenv/evaluate.py").read_text(encoding="utf-8")
    require("task_obj: Any | None = None" in evaluate, "evaluate has no v2 task injection")
    require("build_identity: dict[str, str] | None = None" in evaluate, "evaluate has no build identity injection")
    require("get_task(task_name, dtype, size) if task_obj is None else task_obj" in evaluate, "evaluate still requires a v1 task name")
    tracks = json.loads((ROOT / "backends/metal/tracks.json").read_text(encoding="utf-8"))
    for track in tracks["tracks"]:
        require(track["status"] == "available", f"{track['track_id']} is not available")
        require(not {"op_name", "shape", "shapes", "tile", "grid", "threads", "lowering_profile"} & set(track), f"{track['track_id']} has per-op tiling metadata")
    print("OK: MetalTrackBuilder/Evaluator use the real compiler/evaluator, tracks are op-agnostic")


def verify_boundary() -> None:
    """@brief A4 生产面不携带被拒绝平台名。"""
    offenders: list[str] = []
    for path in (ROOT / "tcenv/backends/metal/track_adapter.py", ROOT / "tasks/sigmoid_shift/opspec.json"):
        for match in REJECTED_PLATFORM.finditer(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    require(offenders == [], "\n".join(offenders))
    print("OK: no rejected platform name in the A4 production surface")


def verify_negative_nodes() -> None:
    """@brief 固定负例节点仍然存在。"""
    for node in NEGATIVE_NODE_IDS:
        path, _, test = node.partition("::")
        test = test.split("[", 1)[0]
        source = (ROOT / path).read_text(encoding="utf-8")
        require(f"def {test}(" in source, f"negative node id is gone: {node}")
    print(f"OK: {len(NEGATIVE_NODE_IDS)} stable negative node ids present")


def self_test() -> None:
    """@brief 可独立重放的 A4 CPU 负例（不需要 GPU）。"""
    sys.path.insert(0, str(ROOT))
    from tcenv.backends.metal.track_adapter import combined_candidate_source, msl_single_geometry_resolver
    from tcenv.contract_errors import GeometryError
    from tcenv.taskcontract import discover_standard_tasks, load_task_contract
    from tcenv.track import (
        CandidateArtifact,
        CandidateEntry,
        ExecutionGeometry,
        candidate_geometry,
        dispatch_parity,
        geometry_pair,
        select_geometry,
        selection_parity,
    )
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tests.support.v2_tasks import rmsnorm_workspace_opspec, write_task

    checks: list[str] = []

    def rejected(name: str, exc_type, action) -> None:
        try:
            action()
        except exc_type:
            checks.append(name)
            return
        except Exception as exc:  # noqa: BLE001
            raise AssertionError(f"{name}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
        raise AssertionError(f"{name}: expected {exc_type.__name__}, nothing was raised")

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary) / "tasks"
        write_task(root, op_name="rmsnorm_v2", opspec=rmsnorm_workspace_opspec("rmsnorm_v2"))
        locator = discover_standard_tasks(root, code_root=ROOT)[0]
        contract = locator.load()
        bare = CandidateArtifact(track_id="msl-single", entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void x() {}\n"),))
        rejected("msl-single without a launch block", GeometryError, lambda: msl_single_geometry_resolver(contract)(bare))
        source = b"/* taic-launch\nlaunch x grid=2,1,1 threads=64,1,1\n*/\n"
        candidate = CandidateArtifact(track_id="msl-single", entries=(CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=source),))
        resolved = msl_single_geometry_resolver(contract)(candidate)
        require(resolved.stages[0].threads == (64, 1, 1), "the msl-single resolver changed threads")
        require(combined_candidate_source(candidate).startswith("/* taic-launch"), "single source was rewritten")
        geometry = candidate_geometry(
            CandidateArtifact(
                track_id="msl-bundle",
                entries=(
                    CandidateEntry(path="device/k.metal", role="device_kernel", media_type="text/plain", data=b"// d\n"),
                    CandidateEntry(path="host/t.metal", role="host_tiling", media_type="text/plain", data=b"// t\n"),
                    CandidateEntry(
                        path="host/execution_geometry.json",
                        role="schedule",
                        media_type="application/json",
                        data=json.dumps({"schema_version": 1, "source": "candidate", "stages": [{"kernel": "k", "grid": [2, 1, 1], "threads": [64, 1, 1]}]}).encode(),
                    ),
                ),
            )
        )
        abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
        pair = geometry_pair(geometry, None, instance=contract.instance, contract=abi, limits=abi_limits_from_target(load_target(DEFAULT_TARGET_ID)))
        chosen = select_geometry(pair)
        good = [("k", (2, 1, 1), (64, 1, 1), 0, ())]
        require(dispatch_parity(chosen.geometry, good) == [], "a matching dispatch was rejected")
        require(dispatch_parity(chosen.geometry, [("k", (4, 1, 1), (64, 1, 1), 0, ())]) != [], "a wrong grid passed")
        require(selection_parity(chosen, provider="candidate", observed=good) == [], "a matching provider was rejected")
        require(selection_parity(chosen, provider="fallback", observed=good) != [], "a silent fallback passed")
        found = [item for item in discover_standard_tasks(ROOT / "tasks", code_root=ROOT) if item.op_name == "sigmoid_shift"]
        require(len(found) == 1, "sigmoid_shift is not discovered from the production tasks root")
        require(load_task_contract(ROOT / "tasks/sigmoid_shift").instance.symbols["N"] == 65536, "sigmoid_shift N changed")
        checks.append("geometry resolver refuses a missing launch; dispatch/fallback parity is observable")
        checks.append("the new operator is discovered from tasks/ without a central list")

    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def run_cpu_pytest() -> None:
    """@brief 运行固定 A4 CPU 节点。"""
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", *FIXED_CPU_NODE_IDS]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
    require(result.returncode == 0, f"fixed A4 CPU pytest failed:\n{tail}\n{result.stderr[-4000:]}")
    print(f"OK: fixed A4 CPU pytest\n{tail}")


def run_gpu_pytest(evidence_dir: Path) -> None:
    """@brief 运行固定真实 Mac 切片/负例，并把 canonical 几何/dispatch/digest 写到证据目录。"""
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", *GPU_NODE_IDS]
    import os

    environment = os.environ.copy()
    environment["A4_EVIDENCE_DIR"] = str(evidence_dir)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
    require(result.returncode == 0, f"real A4 GPU slices failed:\n{tail}\n{result.stderr[-6000:]}")
    print(f"OK: real A4 GPU slices\n{tail}")


def verify_evidence(directory: Path) -> None:
    """@brief 校验真实切片/负例归档，而不是只看固定 pytest 返回码。"""
    expected = {"single.json", "bundle.json", "newop.json", "fusion.json", "derived.json", "readwrite.json", "multistage.json"}
    present = {path.name for path in directory.glob("*.json")}
    require(expected <= present, f"missing A4 evidence files: {sorted(expected - present)}")
    single = json.loads((directory / "single.json").read_text())
    bundle = json.loads((directory / "bundle.json").read_text())
    newop = json.loads((directory / "newop.json").read_text())
    fusion = json.loads((directory / "fusion.json").read_text())
    derived = json.loads((directory / "derived.json").read_text())
    readwrite = json.loads((directory / "readwrite.json").read_text())
    multistage = json.loads((directory / "multistage.json").read_text())
    for name, record in (("single", single), ("bundle", bundle), ("newop", newop)):
        require(record["status"] == "ok", f"{name} slice did not pass")
        require(record.get("abi_ok", True) is True, f"{name} slice ABI failed")
        require(record.get("correctness_ok", True) is True, f"{name} slice correctness failed")
        for key in ("candidate_sha256", "geometry_identity", "bundle_digest", "artifact_set_digest", "native_digest"):
            require(isinstance(record.get(key), str) and len(record[key]) == 64, f"{name}.{key} is not a digest")
    require(single["dispatch"] == [["rmsnorm_single", [64, 1, 1], [256, 1, 1], 0, ["scratch"]]], "single dispatch changed")
    from tcenv.identity import fingerprint
    from tcenv.track import load_track_registry

    shipped = load_track_registry()
    require(single["track"]["track_id"] == "msl-single", "single slice did not run the frozen msl-single Track")
    require(single["track"]["track_version"] == shipped.get("msl-single").track_version, "single slice Track version is not the shipped registration")
    require(single["track"]["track_digest"] == fingerprint(shipped.get("msl-single").document()), "single slice TrackSpec digest is not the shipped registration")
    require(single["track"]["registry_digest"] == fingerprint(shipped.document()), "single slice registry digest is not the shipped registry")
    require(single["worker"]["track_digest"] == single["track"]["track_digest"], "the worker reported a TrackSpec different from the control host's frozen one")
    require(single["worker"]["registry_digest"] == single["track"]["registry_digest"], "the worker selected the track from a different registry")
    require(single["worker"]["presentation_sha256"] == single["presentation_sha256"], "the worker rendered a different presentation identity")
    require(single["worker"]["track_version"] == single["track"]["track_version"], "the worker Track version drifted from the frozen selection")
    require(single["provider"] == "candidate" and bundle["provider"] == "candidate" and newop["provider"] == "candidate", "a real slice did not run the candidate provider")
    require(derived["provider"] == "candidate" and readwrite["provider"] == "candidate", "a protocol slice did not record the candidate provider")
    require([stage[0] for stage in bundle["dispatch"]] == ["rmsnorm_bundle"], "bundle did not dispatch its device stage")
    require(bundle["host_runs"], "bundle evidence has no Host Tiling execution")
    require(bundle["host_runs"][0]["writes"] == 128, "Host Tiling did not write the row-id resources")
    require(bundle["workspace_bytes"]["scratch"] == 64 * 512 * 2, "bundle workspace was not ABI-sized")
    require(bundle["workspace_fills"] == [], "a zero:false workspace was cleared in timing")
    probe = bytes.fromhex(bundle["workspace_probe"]["scratch"])
    require(struct.unpack("<8e", probe[:16]) == tuple(float(i) for i in range(8)), "Host Tiling row resources were not present after the benchmark")
    require(bundle["canonical_bundle"]["lifecycle"] is not None, "bundle evidence has no lifecycle")
    require(bundle["geometry_artifact_sha256"] != single["geometry_artifact_sha256"], "two slices share one geometry artifact")
    require(newop["effective_correctness_only"] is True, "new operator is not correctness-only")
    require(newop.get("canonical_bundle") and newop.get("geometry_artifact_sha256"), "new operator evidence has no canonical geometry")
    require(derived["bindings"][0]["arguments"][-1]["name"] == "BLOCKS" and derived["bindings"][0]["arguments"][-1].get("value") == 2, "Derived Scalar was not set by name")
    require(derived["inactive_kernels"] == [], "an inactive when stage was dispatched")
    require(any(item["access"] == "read_write" for item in readwrite["bindings"][0]["arguments"]), "read_write was collapsed to legacy input/output")
    require(fusion["replay_equal"] is True, "fusion replay changed a stable digest")
    require([item[0] for item in multistage["dispatch"]] == ["stage_one", "stage_two"], "multi-stage dispatch order changed")
    require([item["kernel"] for item in multistage["bindings"]] == ["stage_one", "stage_two"], "multi-stage reflection binding order changed")
    require(len(multistage["canonical_bundle"]["geometry"]["candidate_resolution"]["stages"]) == 2, "multi-stage canonical geometry lost a stage")
    require(multistage["replay_equal"] is True, "multi-stage replay changed a stable digest")
    a, b = fusion["candidates"]["128"], fusion["candidates"]["256"]
    require(a["candidate_sha256"] != b["candidate_sha256"], "fusion candidates share an identity")
    require(a["geometry_identity"] != b["geometry_identity"], "fusion geometries share an identity")
    require(a["artifact_set_digest"] != b["artifact_set_digest"], "fusion builds share an artifact set")
    require(a["dispatch"][0][2] == [128, 1, 1] and b["dispatch"][0][2] == [256, 1, 1], "fusion actual dispatch is not the recorded threads")
    require(a.get("canonical_bundle") and b.get("canonical_bundle"), "fusion evidence has no canonical geometry")
    require(a["bundle_digest"] != b["bundle_digest"], "fusion bundles share a digest")
    archived_dir = ROOT / "docs/架构/跨硬件算子生成分工-阶段验收/A4/evidence"
    for name in sorted(expected):
        live = json.loads((directory / name).read_text())
        archived = json.loads((archived_dir / name).read_text())
        require(live == archived, f"live {name} differs from the archived A4 evidence (toolchain/device/digest drift?)")
    print("OK: A4 evidence (single/bundle/newop/fusion, canonical geometry, dispatch, digests, replay; live == archived)")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="static/CPU only; not an A4 acceptance")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_frozen_surface()
    verify_tracks_and_task()
    verify_adapter_wiring()
    verify_boundary()
    verify_negative_nodes()
    self_test()
    if not args.skip_pytest:
        run_cpu_pytest()
        if not args.skip_gpu:
            with tempfile.TemporaryDirectory(prefix="a4-evidence-") as temporary:
                directory = Path(temporary)
                run_gpu_pytest(directory)
                verify_evidence(directory)
    if args.skip_gpu or args.skip_pytest:
        print("WARNING: a skipped phase is diagnostic only; A4 is not accepted")
    else:
        print("OK: A4 acceptance")


if __name__ == "__main__":
    main()
