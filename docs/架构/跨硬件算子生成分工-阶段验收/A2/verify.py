#!/usr/bin/env python3
"""A2 acceptance verifier: candidate delivery, typed geometry and the Metal ABI gaps.

Default run:

* checks ``manifest.sha256`` over the complete A2 deliverable file set;
* proves the A2 frozen surface (schema 4 / RPC / HMAC / BlobChannel / v1 task surface / G0+A1
  evidence) did not change relative to the A2 base commit, and that the one intentional change to
  an A1-frozen file (``source_builder.py``) is exactly the ``build_profiles.json`` switch;
* scans the A2 production surface for rejected hardware platform names;
* checks that every stable negative node id still exists;
* replays independent self-test negatives for the ABI deriver, geometry, per-stage binding,
  the read_write/workspace/timing protocol, the hardware-neutral lifecycle and the canonical
  bundle;
* runs the fixed A2 + A1-regression pytest set.

``--write-manifest`` regenerates ``manifest.sha256`` after an intentional A2 edit.  The verifier
never calls a model, never builds a kernel and never touches a GPU/NPU.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = Path(__file__).resolve().parent / "manifest.sha256"

#: The complete A2 deliverable surface.  The plan file is a living document; the G0 corpus and
#: the A1 acceptance record are independently frozen and are checked as part of the frozen surface.
DELIVERABLES: tuple[str, ...] = (
    "tcenv/contract_errors.py",
    "tcenv/opspec/v2.py",
    "tcenv/taskcontract/errors.py",
    "tcenv/taskcontract/loader.py",
    "tcenv/track/__init__.py",
    "tcenv/track/abi.py",
    "tcenv/track/bundle.py",
    "tcenv/track/expr.py",
    "tcenv/track/geometry.py",
    "tcenv/track/lifecycle.py",
    "tcenv/track/protocol.py",
    "tcenv/backends/metal/source_builder.py",
    "tcenv/backends/metal/stage_binding.py",
    "backends/metal/build_profiles.json",
    "tests/support/v2_tasks.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_track_abi.py",
    "tests/shared/test_track_bundle.py",
    "tests/shared/test_track_geometry.py",
    "tests/shared/test_track_lifecycle.py",
    "tests/shared/test_track_protocol.py",
    "tests/shared/test_cross_hardware_boundary_a2.py",
    "tests/backends/metal/test_stage_binding.py",
    "tests/backends/metal/test_build_profiles.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2/verify.py",
)

#: Fixed pytest entry points: the A2 files plus the A1/A1-regression files that must stay green.
FIXED_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_track_abi.py",
    "tests/shared/test_track_bundle.py",
    "tests/shared/test_track_geometry.py",
    "tests/shared/test_track_lifecycle.py",
    "tests/shared/test_track_protocol.py",
    "tests/shared/test_cross_hardware_boundary_a2.py",
    "tests/backends/metal/test_stage_binding.py",
    "tests/backends/metal/test_build_profiles.py",
    "tests/backends/metal/test_diagnostic_builds.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_opspec_v2.py",
    "tests/shared/test_track_plane.py",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_op_desc_summary.py",
    "tests/shared/test_cross_hardware_boundary.py",
    "tests/backends/metal/test_track_registry.py",
)

#: Stable negative node ids that must exist and run (they are part of the fixed set above).
NEGATIVE_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_track_abi.py::test_the_slot_budget_buffer_length_and_capabilities_come_from_the_target",
    "tests/shared/test_track_abi.py::test_the_abi_never_accepts_a_symbolic_or_negative_dimension",
    "tests/shared/test_track_abi.py::test_a_strided_tensor_is_measured_by_the_span_it_addresses",
    "tests/shared/test_track_abi.py::test_the_deriver_has_no_per_op_or_tiling_input",
    "tests/shared/test_track_geometry.py::test_symbolic_geometry_evaluates_exactly_and_text_agrees_with_json",
    "tests/shared/test_track_geometry.py::test_when_gates_stages_and_a_zero_active_stage_geometry_is_refused",
    "tests/shared/test_track_geometry.py::test_derived_scalars_are_named_by_the_candidate_and_depend_only_on_symbols_and_attrs",
    "tests/shared/test_track_geometry.py::test_workspace_uses_must_be_declared_and_have_one_clearing_policy",
    "tests/shared/test_track_geometry.py::test_grid_threads_and_threadgroup_memory_must_fit_the_target",
    "tests/shared/test_track_geometry.py::test_the_fallback_is_independent_and_enters_the_execution_identity",
    "tests/shared/test_track_geometry.py::test_dispatch_parity_catches_a_worker_that_drifted_from_the_recorded_geometry",
    "tests/shared/test_track_geometry.py::test_the_geometry_plane_has_no_style_gate_or_sdk_feature_whitelist",
    "tests/shared/test_track_protocol.py::test_read_write_is_restored_on_every_trial_and_never_poisoned",
    "tests/shared/test_track_protocol.py::test_a_pure_write_output_is_poisoned_zeroed_and_poisoned",
    "tests/shared/test_track_protocol.py::test_workspace_clearing_only_touches_declared_zero_regions",
    "tests/shared/test_track_protocol.py::test_benchmark_repetitions_check_only_guards_bounds_and_hardware_state",
    "tests/shared/test_track_protocol.py::test_input_dependent_prepare_must_be_inside_the_timed_window",
    "tests/shared/test_track_protocol.py::test_the_protocol_has_no_per_repetition_value_diff_and_no_hidden_fill",
    "tests/shared/test_track_bundle.py::test_a_geometry_change_moves_the_subject_digest_but_not_the_candidate_identity",
    "tests/shared/test_track_bundle.py::test_the_bundle_refuses_a_fallback_as_candidate_and_an_unknown_schema",
    "tests/shared/test_track_bundle.py::test_the_bundle_binds_the_prepare_run_lifecycle",
    "tests/shared/test_track_lifecycle.py::test_the_lifecycle_splits_host_prepare_from_device_run",
    "tests/shared/test_track_lifecycle.py::test_a_reusable_step_gets_a_computed_cache_identity_and_a_hand_written_one_is_refused",
    "tests/shared/test_track_lifecycle.py::test_input_dependent_prepare_must_be_timed_inside_the_lifecycle",
    "tests/shared/test_track_lifecycle.py::test_the_lifecycle_and_the_candidate_must_agree_on_host_work",
    "tests/backends/metal/test_stage_binding.py::test_access_permission_and_contiguity_are_enforced",
    "tests/backends/metal/test_stage_binding.py::test_derived_scalar_width_and_read_only_are_enforced",
    "tests/backends/metal/test_stage_binding.py::test_a_pipeline_needs_every_result_written_by_an_active_stage",
    "tests/backends/metal/test_build_profiles.py::test_the_registered_profiles_come_from_the_json_document",
    "tests/backends/metal/test_build_profiles.py::test_the_profile_schema_is_closed_and_refuses_unknown_values",
    "tests/shared/test_task_contract_v2.py::test_a_workspace_tensor_enters_the_a2_abi_and_stays_out_of_the_reference_contract",
    "tests/shared/test_cross_hardware_boundary_a2.py::test_a2_production_mentions_no_rejected_hardware_platform",
    "tests/shared/test_cross_hardware_boundary_a2.py::test_the_v2_delivery_plane_has_no_loop_or_goto_style_review",
)

#: The checkout A2 started from (A1's committed acceptance head).
A2_BASE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"

#: Surfaces whose identity/protocol/serialisation must not change in A2.  The one intentional
#: change (``source_builder.py`` → ``build_profiles.json``) is checked separately and substantively.
FROZEN_SURFACE: tuple[str, ...] = (
    "proto",
    "tcenv/remote/v1",
    "tcenv/remote/artifact_bundle.py",
    "tcenv/remote/contracts.py",
    "tcenv/remote/grpc_client.py",
    "tcenv/remote/grpc_contract.py",
    "tcenv/remote/grpc_service.py",
    "tcenv/remote/task_pack.py",
    "tcenv/evaluation.py",
    "tcenv/task.py",
    "tcenv/contract.py",
    "tcenv/build_projection.py",
    "tcenv/report.py",
    "tcenv/schema.py",
    "tcenv/verify.py",
    "tcenv/backends/metal/lowering.py",
    "tcenv/track/schema/track.v1.schema.json",
    "tcenv/opspec/schema/opspec.v1.schema.json",
    "tasks",
    "targets",
    "taic",
    "backends/metal/backend.toml",
    "backends/metal/tracks.json",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_v2_g0_reference_outputs.py",
    "tests/shared/test_op_desc_summary.py",
    "docs/架构/跨硬件算子生成分工-G0",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1",
)

SOURCE_BUILDER = "tcenv/backends/metal/source_builder.py"
BASE_REGISTRY_START = "#: Registered **diagnostic** build profiles"
BASE_REGISTRY_END = "}\n"
CURRENT_REGISTRY_START = "#: The one authoritative registry"
CURRENT_REGISTRY_END = "DIAGNOSTIC_BUILD_PROFILES: dict[str, dict[str, str]] = load_build_profiles()\n"
PROFILE_ONLY_IMPORTS = ("import json\n", "from pathlib import Path\n")

REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|america|nvidia)\b", re.IGNORECASE
)


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。

    @param condition 条件。
    @param message 失败说明。
    @raises AssertionError 条件不成立。
    """
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。

    @param path 文件。
    @return 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    """@brief 重写 A2 manifest.sha256（覆盖固定交付面）。"""
    lines = [f"{sha256(ROOT / name)}  {name}" for name in DELIVERABLES]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    """@brief 校验 manifest 的摘要与文件集合完全一致。

    @raises AssertionError 缺失/多余/摘要不符。
    """
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"bad digest: {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen[name] = digest
    expected = set(DELIVERABLES)
    require(
        set(seen) == expected,
        f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}",
    )
    for name, digest in seen.items():
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def _git(*args: str) -> str:
    """@brief 运行 git 并返回 stdout。

    @param args git 参数。
    @return stdout 文本。
    @raises AssertionError git 失败。
    """
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def verify_frozen_surface() -> None:
    """@brief 证明 schema 4/RPC/HMAC/BlobChannel、v1 题面与 G0/A1 证据都没有被 A2 改动。

    @raises AssertionError 冻结面出现 diff。
    """
    base = _git("rev-parse", "--verify", f"{A2_BASE_HEAD}^{{commit}}").strip()
    require(base == A2_BASE_HEAD, f"A2 base commit is not available: {A2_BASE_HEAD}")
    changed = _git("diff", "--name-only", A2_BASE_HEAD, "--", *FROZEN_SURFACE).splitlines()
    require(changed == [], "frozen surface changed since the A2 base commit: " + ", ".join(changed))
    print(f"OK: frozen surface untouched since {A2_BASE_HEAD[:12]} ({len(FROZEN_SURFACE)} paths)")


def _strip_profile_registry(text: str, start_marker: str, end_marker: str, *, drop_imports=()) -> str:
    """@brief 去掉 build-profile 注册块（以及只属于它的 import）后的源码。

    @param text 源码。
    @param start_marker 注册块首行前缀。
    @param end_marker 注册块末行前缀。
    @param drop_imports 需要同时去掉的 import 行。
    @return 去掉注册块后的源码。
    @raises AssertionError 标记缺失或重复。
    """
    lines = text.splitlines(keepends=True)
    starts = [index for index, line in enumerate(lines) if line.startswith(start_marker)]
    require(len(starts) == 1, f"the build-profile block marker appears {len(starts)} times: {start_marker!r}")
    start = starts[0]
    ends = [index for index in range(start, len(lines)) if lines[index].startswith(end_marker)]
    require(bool(ends), f"the build-profile block end marker is missing: {end_marker!r}")
    end = ends[0]
    removed = set(range(start, end + 1))
    kept = [line for index, line in enumerate(lines) if index not in removed and line not in drop_imports]
    return "".join(kept)


def verify_source_builder_change() -> None:
    """@brief ``source_builder.py`` 的唯一改动就是 profile 表迁移到 JSON 读取器。

    @raises AssertionError 出现注册块之外的构建语义改动。
    """
    base_text = _git("show", f"{A2_BASE_HEAD}:{SOURCE_BUILDER}")
    current_text = (ROOT / SOURCE_BUILDER).read_text(encoding="utf-8")
    base_rest = _strip_profile_registry(base_text, BASE_REGISTRY_START, BASE_REGISTRY_END)
    current_rest = _strip_profile_registry(
        current_text, CURRENT_REGISTRY_START, CURRENT_REGISTRY_END, drop_imports=PROFILE_ONLY_IMPORTS
    )
    require(
        base_rest == current_rest,
        "source_builder.py changed outside the build-profile registry block; A2 may only move the "
        "diagnostic profile table to backends/metal/build_profiles.json",
    )
    required = (
        'BUILD_PROFILES_PATH = Path(__file__).resolve().parents[3] / "backends" / "metal" / "build_profiles.json"',
        "def load_build_profiles(",
        "DIAGNOSTIC_BUILD_PROFILES: dict[str, dict[str, str]] = load_build_profiles()",
    )
    for needle in required:
        require(needle in current_text, f"the single build-profile reader is missing: {needle}")
    for name in ("metal.raw-source-probe", "metal.raw-source-calibration", "metal.internal-fill"):
        require(name not in current_text, f"the profile {name} is still a second (Python) copy")
    print("OK: source_builder.py changed only in the build-profile registry switch")


def _a2_production_sources() -> dict[str, str]:
    """@brief A2 生产面的文本：新文件全文 + 已有文件的新增行。

    A2 修改了 ``source_builder.py``（profile 注册块）等既有文件；它们的旧文本可能已经提到被拒绝
    平台（例如“不是 NPU artifact”这类否定句），那不是 A2 引入的知识。边界因此只看新文件全文与
    既有文件相对 A2 起点的**新增行**。

    @return 文件相对路径到待扫描文本。
    """
    tracked = set(_git("ls-files", "--", *DELIVERABLES).splitlines())
    sources: dict[str, str] = {}
    for name in DELIVERABLES:
        if not name.startswith(("tcenv/", "backends/")):
            continue
        path = ROOT / name
        if path.suffix not in {".py", ".json"}:
            continue
        if name in tracked:
            diff = _git("diff", "--unified=0", A2_BASE_HEAD, "--", name)
            added = [line[1:] for line in diff.splitlines() if line.startswith("+") and not line.startswith("+++")]
            sources[name] = "\n".join(added)
        else:
            sources[name] = path.read_text(encoding="utf-8")
    return sources


def verify_a2_boundary() -> None:
    """@brief A2 生产代码/配置不携带被拒绝平台的名称或专用知识。

    @raises AssertionError 出现禁止词。
    """
    offenders: list[str] = []
    for name, text in _a2_production_sources().items():
        for match in REJECTED_PLATFORM.finditer(text):
            offenders.append(f"{name}: {match.group(0)}")
    require(offenders == [], "\n".join(offenders))
    print("OK: no rejected hardware platform in the A2 production surface")


def verify_negative_nodes() -> None:
    """@brief 确认固定负例节点仍然存在（不被删除/改名）。

    @raises AssertionError 节点不存在。
    """
    for node in NEGATIVE_NODE_IDS:
        path, _, test = node.partition("::")
        test = test.split("[", 1)[0]
        source = (ROOT / path).read_text(encoding="utf-8")
        require(f"def {test}(" in source, f"negative node id is gone: {node}")
    print(f"OK: {len(NEGATIVE_NODE_IDS)} stable negative node ids present")


def _no_device_limit_deriver():
    """@brief 构造一个缺少 ABI 上限的目标替身。

    @return 目标替身。
    """
    from types import SimpleNamespace

    return SimpleNamespace(id="no-limits", backend="metal", raw_limits={"counter_sample_us": 0})


def self_test() -> None:
    """@brief 可独立重放的 A2 稳定负例（不依赖 pytest 收集）。

    @raises AssertionError 任一负例未被拒绝或正向行为不成立。
    """
    sys.path.insert(0, str(ROOT))

    from tcenv.contract_errors import GeometryError, TaskContractError
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.taskcontract import load_task_contract
    from tcenv.track import (
        CORRECTNESS_TRIALS,
        GUARD_PATTERN,
        POISON,
        RESTORE_INITIAL,
        UNTOUCHED,
        ZERO,
        CandidateArtifact,
        CandidateEntry,
        CandidateLifecycle,
        CanonicalBundle,
        ExecutionGeometry,
        GeometryStage,
        HostStep,
        PrepareDeclaration,
        TargetAbiDeriver,
        TargetAbiLimits,
        TimingDiscipline,
        benchmark_launch_plan,
        cache_key,
        check_prepare_timing,
        correctness_launch_plan,
        derive_tensor_memory_contract,
        dispatch_parity,
        geometry_pair,
        resolve_geometry,
        select_geometry,
        selection_parity,
        unwritten_check_applies,
        workspace_fills,
    )
    from tcenv.backends.metal.source_builder import DIAGNOSTIC_BUILD_PROFILES, load_build_profiles
    from tcenv.backends.metal.stage_binding import ReflectedArgument, bind_pipeline_arguments, bind_stage_arguments
    from tests.support.v2_tasks import (
        rmsnorm_opspec,
        rmsnorm_read_write_opspec,
        rmsnorm_workspace_opspec,
        write_task,
    )

    checks: list[str] = []
    limits = TargetAbiLimits(
        max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768
    )

    def rejected(name: str, exc_type, action) -> None:
        try:
            action()
        except exc_type:
            checks.append(name)
            return
        except Exception as exc:  # noqa: BLE001 - report the wrong exception type
            raise AssertionError(f"{name}: expected {exc_type.__name__}, got {type(exc).__name__}: {exc}") from exc
        raise AssertionError(f"{name}: expected {exc_type.__name__}, nothing was raised")

    with tempfile.TemporaryDirectory() as temporary:
        base = Path(temporary)

        def task_dir(name: str = "rmsnorm", **kwargs):
            return write_task(base / name, **kwargs)

        plain = load_task_contract(task_dir("plain"))
        abi = derive_tensor_memory_contract(plain.instance, target=load_target(DEFAULT_TARGET_ID))
        require(abi.target_id == DEFAULT_TARGET_ID and abi.backend_id == "metal", "the ABI lost its target identity")
        require(
            not any(word in repr(abi.document()).lower() for word in ("tile", "grid", "threads", "fallback")),
            "the ABI carries a geometry/tiling field",
        )
        checks.append("TargetAbiDeriver projects ABI/memory/numerics only")

        rejected(
            "target without registered ABI limits",
            TaskContractError,
            lambda: TargetAbiDeriver.for_target(_no_device_limit_deriver()),
        )
        rejected(
            "slot budget from the target",
            TaskContractError,
            lambda: TargetAbiDeriver(
                backend_id="metal",
                target_id=DEFAULT_TARGET_ID,
                limits=TargetAbiLimits(max_buffer_index=1, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768),
            ).derive(plain.instance),
        )

        workspace_task = load_task_contract(task_dir("ws", op_name="rmsnorm_ws", opspec=rmsnorm_workspace_opspec()))
        workspace_abi = derive_tensor_memory_contract(workspace_task.instance, target=load_target(DEFAULT_TARGET_ID))
        require(workspace_abi.tensor("scratch").space == "workspace", "the workspace tensor is not an ABI argument")
        for name in ("scratch",):
            require(name not in workspace_task.instance.reference_inputs, "workspace entered the reference inputs")
            require(name not in workspace_task.instance.reference_outputs, "workspace entered the reference results")
        checks.append("workspace is an ABI argument, not a reference tensor")

        rw_task = load_task_contract(task_dir("rw", op_name="rmsnorm_rw", opspec=rmsnorm_read_write_opspec()))
        rw_abi = derive_tensor_memory_contract(rw_task.instance, target=load_target(DEFAULT_TARGET_ID))
        rw_geometry = resolve_geometry(
            ExecutionGeometry(
                source="candidate",
                stages=(
                    GeometryStage(
                        kernel="rw_kernel", grid=({"ceil_div": ["M*H", 256]}, 1, 1), threads=(256, 1, 1)
                    ),
                ),
            ),
            instance=rw_task.instance,
            contract=rw_abi,
            limits=limits,
        )
        for trial in CORRECTNESS_TRIALS:
            plan = correctness_launch_plan(rw_abi, rw_geometry, trial)
            require(plan.guard_pattern == GUARD_PATTERN == 0xCD, "the correctness guard pattern moved")
            for name in rw_task.instance.read_write_tensors:
                require(plan.action(name) == RESTORE_INITIAL, f"read_write {name} is not restored on {trial}")
                require(plan.action(name) != POISON, f"read_write {name} was poisoned on {trial}")
                require(unwritten_check_applies(rw_abi, name) is False, f"read_write {name} is counted unwritten")
        checks.append("read_write is restored on out1/out2/out3 and never poisoned")

        geometry = ExecutionGeometry(
            source="candidate",
            stages=(
                GeometryStage(kernel="fill", grid=({"ceil_div": ["M*H", 256]}, 1, 1), threads=(256, 1, 1)),
                GeometryStage(kernel="reduce", grid=(1, 1, 1), threads=(64, 1, 1), when=("gt", "M", 10**9)),
            ),
        )
        resolved = resolve_geometry(geometry, instance=plain.instance, contract=abi, limits=limits)
        require([stage.kernel for stage in resolved.active_stages] == ["fill"], "when did not gate the stage")
        rejected(
            "zero grid component",
            GeometryError,
            lambda: resolve_geometry(
                ExecutionGeometry(
                    source="candidate", stages=(GeometryStage(kernel="k", grid=(0, 1, 1), threads=(32, 1, 1)),)
                ),
                instance=plain.instance,
                contract=abi,
                limits=limits,
            ),
        )
        rejected(
            "threads above the target limit",
            GeometryError,
            lambda: resolve_geometry(
                ExecutionGeometry(
                    source="candidate", stages=(GeometryStage(kernel="k", grid=(1, 1, 1), threads=(2048, 1, 1)),)
                ),
                instance=plain.instance,
                contract=abi,
                limits=limits,
            ),
        )
        rejected(
            "zero active stage",
            GeometryError,
            lambda: resolve_geometry(
                ExecutionGeometry(
                    source="candidate",
                    stages=(GeometryStage(kernel="k", grid=(1, 1, 1), threads=(32, 1, 1), when=("gt", "M", 10**9)),),
                ),
                instance=plain.instance,
                contract=abi,
                limits=limits,
            ),
        )
        checks.append("symbolic/when geometry is bounded by the target")

        derived = GeometryStage(
            kernel="k",
            grid=("BLOCKS", 1, 1),
            threads=(256, 1, 1),
            derived_scalars=(("ELEMENTS", "M*H"), ("BLOCKS", {"ceil_div": ["M*H", 256]})),
        )
        resolved_derived = resolve_geometry(
            ExecutionGeometry(source="candidate", stages=(derived,)), instance=plain.instance, contract=abi, limits=limits
        )
        expected_blocks = (plain.instance.symbols["M"] * plain.instance.symbols["H"]) // 256
        require(resolved_derived.stages[0].grid[0] == expected_blocks, "a derived scalar cannot size the launch")
        require(dict(resolved_derived.stages[0].derived_scalars)["BLOCKS"] == expected_blocks, "derived scalar moved")
        rejected(
            "derived scalar depending on a tensor/other scalar",
            GeometryError,
            lambda: resolve_geometry(
                ExecutionGeometry(
                    source="candidate",
                    stages=(
                        GeometryStage(
                            kernel="k",
                            grid=(1, 1, 1),
                            threads=(32, 1, 1),
                            derived_scalars=(("A", "M"), ("B", "A")),
                        ),
                    ),
                ),
                instance=plain.instance,
                contract=abi,
                limits=limits,
            ),
        )
        checks.append("derived scalars are by-name and depend only on symbols/attrs")

        fallback = ExecutionGeometry(
            source="fallback", stages=(GeometryStage(kernel="fb", grid=(1, 1, 1), threads=(32, 1, 1)),)
        )
        pair = geometry_pair(geometry, fallback, instance=plain.instance, contract=abi, limits=limits)
        selection = select_geometry(pair)
        observed = [(stage.kernel, stage.grid, stage.threads, 0, ()) for stage in selection.geometry.active_stages]
        require(dispatch_parity(selection.geometry, observed) == [], "parity rejects the recorded dispatch")
        require(selection_parity(selection, provider="candidate", observed=observed) == [], "selection parity rejects")
        require(bool(selection_parity(selection, provider="fallback", observed=observed)), "fallback use is not a mismatch")
        rejected(
            "fallback as candidate",
            GeometryError,
            lambda: geometry_pair(fallback, instance=plain.instance, contract=abi, limits=limits),
        )
        broken = ExecutionGeometry(
            source="candidate", stages=(GeometryStage(kernel="k", grid=(1, 1, 1), threads=(4096, 1, 1)),)
        )
        broken_pair = geometry_pair(broken, fallback, instance=plain.instance, contract=abi, limits=limits)
        rejected("silent fallback", GeometryError, lambda: select_geometry(broken_pair))
        require(select_geometry(broken_pair, allow_fallback=True).used_fallback is True, "explicit fallback refused")
        checks.append("the independent fallback is explicit and enters the identity")

        binding = bind_stage_arguments(
            resolved.active_stages[0],
            contract=abi,
            reflection=(
                ReflectedArgument(index=0, name="x", kind="buffer", access="read", scalar_type="half", elem_size=2),
                ReflectedArgument(index=1, name="w", kind="buffer", access="read", scalar_type="half", elem_size=2),
                ReflectedArgument(index=2, name="y", kind="buffer", access="write", scalar_type="half", elem_size=2),
            ),
            limits=limits,
        )
        require([argument.name for argument in binding.arguments] == ["x", "w", "y"], "per-stage binding by name")
        pipeline = bind_pipeline_arguments(
            resolved,
            contract=abi,
            reflection={
                "fill": (
                    ReflectedArgument(index=0, name="x", kind="buffer", access="read", scalar_type="half", elem_size=2),
                    ReflectedArgument(index=1, name="w", kind="buffer", access="read", scalar_type="half", elem_size=2),
                    ReflectedArgument(index=2, name="y", kind="buffer", access="write", scalar_type="half", elem_size=2),
                )
            },
            limits=limits,
        )
        require(len(pipeline) == 1, "an inactive stage was bound")
        rejected(
            "unwritten result tensor",
            GeometryError,
            lambda: bind_pipeline_arguments(
                resolved,
                contract=abi,
                reflection={
                    "fill": (
                        ReflectedArgument(index=0, name="x", kind="buffer", access="read", scalar_type="half", elem_size=2),
                        ReflectedArgument(index=1, name="w", kind="buffer", access="read", scalar_type="half", elem_size=2),
                    )
                },
                limits=limits,
            ),
        )
        checks.append("MSL reflection binds per stage and every result is written")

        zero_ws = resolve_geometry(
            ExecutionGeometry(
                source="candidate",
                stages=(
                    GeometryStage(
                        kernel="k",
                        grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                        threads=(256, 1, 1),
                        workspace=({"name": "scratch", "zero": True},),
                    ),
                ),
            ),
            instance=workspace_task.instance,
            contract=workspace_abi,
            limits=limits,
        )
        require([use.name for use in workspace_fills(workspace_abi, zero_ws, phase="benchmark")] == ["scratch"], "declared-zero workspace is not cleared")
        undeclared_ws = resolve_geometry(
            ExecutionGeometry(
                source="candidate",
                stages=(
                    GeometryStage(
                        kernel="k",
                        grid=({"ceil_div": ["M*H", 256]}, 1, 1),
                        threads=(256, 1, 1),
                        workspace=("scratch",),
                    ),
                ),
            ),
            instance=workspace_task.instance,
            contract=workspace_abi,
            limits=limits,
        )
        require(workspace_fills(workspace_abi, undeclared_ws, phase="benchmark") == (), "undeclared workspace was cleared")
        fresh_plan = benchmark_launch_plan(workspace_abi, undeclared_ws, discipline=TimingDiscipline.fresh_rotation())
        require(fresh_plan.action("scratch") == UNTOUCHED, "an undeclared workspace gets a timed clear")
        correctness_plan = correctness_launch_plan(workspace_abi, undeclared_ws, "out2")
        require(correctness_plan.action("scratch") == ZERO, "correctness out2 does not zero the workspace")
        checks.append("workspace clearing only touches declared-zero regions")

        fresh = TimingDiscipline.fresh_rotation(2)
        require(fresh.post_rep_checks() == ("guard_bands", "out_of_bounds", "hardware_state"), "post-rep checks widened")
        stateful = TimingDiscipline.stateful("registered accumulation state")
        require(
            benchmark_launch_plan(rw_abi, rw_geometry, discipline=fresh).action("y") == RESTORE_INITIAL,
            "fresh rotation does not restore the read_write initial value",
        )
        require(
            benchmark_launch_plan(rw_abi, rw_geometry, discipline=stateful).action("y") == UNTOUCHED,
            "stateful timing claims fresh inputs",
        )
        rejected(
            "fresh+stateful mixed timing",
            GeometryError,
            lambda: TimingDiscipline.resolve(fresh_rotation=True, stateful_declaration="mixed"),
        )
        rejected(
            "unregistered timing discipline",
            TaskContractError,
            lambda: TimingDiscipline.resolve(fresh_rotation=False),
        )
        checks.append("fresh-rotation and stateful timing never mix")

        require(
            len(
                check_prepare_timing(
                    [
                        PrepareDeclaration(name="tiling", input_dependent=False),
                        PrepareDeclaration(name="bind", input_dependent=True, inside_timing=True),
                        PrepareDeclaration(name="compile", input_dependent=False, cached=True, cache_identity="scope"),
                    ]
                )
            )
            == 3,
            "a legal prepare declaration was rejected",
        )
        rejected(
            "input-dependent prepare outside timing",
            GeometryError,
            lambda: check_prepare_timing([PrepareDeclaration(name="hidden", input_dependent=True)]),
        )
        rejected(
            "cached prepare without identity",
            GeometryError,
            lambda: check_prepare_timing([PrepareDeclaration(name="compile", input_dependent=False, cached=True)]),
        )
        rejected(
            "Host Tiling computes the reference",
            GeometryError,
            lambda: check_prepare_timing([PrepareDeclaration(name="oracle", input_dependent=False, computes_reference=True)]),
        )
        checks.append("input-dependent prepare is always inside the timed window")

        host_steps = (
            HostStep(PrepareDeclaration(name="compile", input_dependent=False), role="harness"),
            HostStep(PrepareDeclaration(name="tiling", input_dependent=False), role="host_tiling"),
        )
        bundle_candidate = CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="kernel.metal", role="device_kernel", media_type="text/plain", data=b"kernel void k(){}\n"),
                CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// tiling\n"),
            ),
        )
        lifecycle = CandidateLifecycle(
            track_id="msl-bundle",
            geometry=selection.geometry,
            host_steps=host_steps,
            candidate_sha256=bundle_candidate.candidate_sha256,
            target_id=DEFAULT_TARGET_ID,
            instance_key=plain.fingerprint(),
        )
        require(lifecycle.validate_for(bundle_candidate) == "accepted", "the lifecycle/candidate pair is refused")
        require(lifecycle.document()["run"][0]["kernel"] == "fill", "the run phase is not the active geometry")
        cached_step = _lifecycle_with_cached(base, plain)
        require(
            cached_step.document()["prepare"][0]["cache_identity"] == cache_key(cached_step.cache_scope, "compile"),
            "the lifecycle does not compute the cache identity",
        )
        rejected(
            "hand-written cache identity",
            GeometryError,
            lambda: HostStep(
                PrepareDeclaration(name="compile", input_dependent=False, cached=True, cache_identity="mine"),
                role="harness",
            ),
        )
        rejected(
            "host tiling declared but not executed",
            GeometryError,
            lambda: CandidateLifecycle(
                track_id="msl-bundle",
                geometry=selection.geometry,
                host_steps=(HostStep(PrepareDeclaration(name="compile", input_dependent=False), role="harness"),),
                candidate_sha256=bundle_candidate.candidate_sha256,
                target_id=DEFAULT_TARGET_ID,
                instance_key=plain.fingerprint(),
            ).validate_for(bundle_candidate),
        )
        checks.append("the prepare/run lifecycle is host/device split and identity-bound")

        bundle = CanonicalBundle(candidate=bundle_candidate, pair=pair)
        require(bundle.candidate_sha256 == bundle_candidate.candidate_sha256, "the bundle changed the candidate identity")
        require(bundle.document()["lifecycle"] is None, "a lifecycle appeared without being registered")
        with_lifecycle = CanonicalBundle(candidate=bundle_candidate, pair=pair, lifecycle=lifecycle)
        require(with_lifecycle.bundle_digest != bundle.bundle_digest, "the lifecycle did not enter the subject digest")
        require(with_lifecycle.geometry_identity != "", "the bundle has no geometry identity")
        checks.append("the canonical bundle separates submission and build-subject identity")

        require(DIAGNOSTIC_BUILD_PROFILES["metal.internal-fill"]["assemble"] == "raw_source", "the JSON reader lost a profile")
        require(load_build_profiles() == DIAGNOSTIC_BUILD_PROFILES, "the build-profile reader is not authoritative")
        rejected(
            "unknown build profile",
            ValueError,
            lambda: __import__("tcenv.backends.metal.source_builder", fromlist=["MetalSourceBuilder"]).MetalSourceBuilder(
                backend=object(), diagnostic_profile="metal.not-registered"
            ),
        )
        checks.append("build_profiles.json is the single diagnostic-profile reader")

        for source in (
            b"kernel void k(device half *x) { int i = 0; loop: if (i++ < 4) goto loop; }\n",
            b"kernel void k(device half *x) { while (true) { break; } }\n",
        ):
            candidate = CandidateArtifact(
                track_id="msl-single",
                entries=(CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=source),),
            )
            from tcenv.track import load_track_registry

            require(candidate.validate_for(load_track_registry().get("msl-single")) == "accepted", "a goto/loop candidate is refused")
        checks.append("no v2 loop/goto style review")

    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def _lifecycle_with_cached(base: Path, contract) -> "object":
    """@brief 构造一个带缓存 prepare 步骤的生命周期（self-test helper）。

    @param base 临时根目录。
    @param contract 任务合同。
    @return 生命周期。
    """
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.track import (
        CandidateLifecycle,
        ExecutionGeometry,
        GeometryStage,
        HostStep,
        PrepareDeclaration,
        TargetAbiLimits,
        derive_tensor_memory_contract,
        resolve_geometry,
    )

    limits = TargetAbiLimits(max_buffer_index=30, max_threads_per_threadgroup=1024, max_threadgroup_memory_bytes=32768)
    abi = derive_tensor_memory_contract(contract.instance, target=load_target(DEFAULT_TARGET_ID))
    geometry = resolve_geometry(
        ExecutionGeometry(source="candidate", stages=(GeometryStage(kernel="k", grid=(1, 1, 1), threads=(32, 1, 1)),)),
        instance=contract.instance,
        contract=abi,
        limits=limits,
    )
    return CandidateLifecycle(
        track_id="msl-bundle",
        geometry=geometry,
        host_steps=(HostStep(PrepareDeclaration(name="compile", input_dependent=False, cached=True), role="harness"),),
        candidate_sha256="f" * 64,
        target_id=DEFAULT_TARGET_ID,
        instance_key=contract.fingerprint(),
    )


def run_pytest() -> None:
    """@brief 运行固定 A2 pytest 节点。

    @raises AssertionError 任何用例失败。
    """
    command = [sys.executable, "-m", "pytest", "-q", "-n", "0", *FIXED_NODE_IDS]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
    require(result.returncode == 0, f"fixed A2 pytest failed:\n{tail}\n{result.stderr[-4000:]}")
    print(f"OK: fixed A2 pytest\n{tail}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_frozen_surface()
    verify_source_builder_change()
    verify_a2_boundary()
    verify_negative_nodes()
    self_test()
    if not args.skip_pytest:
        run_pytest()
    print("OK: A2 acceptance")


if __name__ == "__main__":
    main()
