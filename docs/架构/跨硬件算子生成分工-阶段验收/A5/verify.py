#!/usr/bin/env python3
"""A5 acceptance verifier: the 16 G0 tasks on the v2 standard entry, with a real Mac gate.

The verifier starts no model and no search.  It checks the *result* of the migration:

* every migrated directory is a v2 standard entry and the retired v1 assembly is gone in the
  same change (``task.py``/``track.json``/``kernel_triton.py``/``kernel_triton_msl.py``/
  ``naive.metal``/``__init__.py``), while ``preserved_v1_out_of_scope`` still assembles on v1;
* the v1 registry refuses a v2 task **by version** (no fallback, no second registry) and an
  invalid/over-long task name is a lookup miss, not ``ENAMETOOLONG``;
* the frozen G0 corpus comparison covers all 16 tasks / 82 combinations (inputs, reference
  outputs, error scales, cases/tolerances/attrs/work, baseline identity and pack closure);
* six real ``evaluate_task`` slices run on the Mac with the G0 candidate body/launch; the live
  candidate/geometry/build/dispatch identity equals the archived A5 evidence;
* the paper **and** small performance gates are <= 1.10 both when recomputed from the three
  archived fresh rounds and when a fourth fresh v2 round is measured here; a missing registered
  baseline is refused, never folded or skipped;
* the five remote regressions (mTLS round trip, Go conformance, three frozen Go vectors) and the
  observe long-task-name regression stay fixed; and the frozen cross-language vectors are
  reproduced from their own frozen identity instead of a retired task assembly;
* the G0/A1-A4 acceptance evidence and verifiers are byte-unchanged.

A skipped phase is diagnostic only: A5 is not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A5 = Path(__file__).resolve().parent
MANIFEST = A5 / "manifest.sha256"
EVIDENCE = A5 / "evidence"
A5_BASE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"

TASKS: tuple[str, ...] = (
    "sigmoid",
    "relu",
    "silu",
    "swiglu",
    "rmsnorm",
    "softmax",
    "reduction_sum",
    "gemm",
    "gemm_silu",
    "sem_gemm_neg",
    "sem_relu_gelu_comment",
    "sem_gelu_named_relu",
    "gen_relu",
    "gen_sigmoid",
    "gen_silu",
    "gen_swiglu",
)
CORRECTNESS_ONLY: tuple[str, ...] = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu")
PRESERVED_V1: tuple[str, ...] = (
    "rope",
    "gemv",
    "conv2d",
    "flash_attention",
    "lion",
    "bitdelta_gemm",
    "mamba2_chunk_scan",
    "sem_attention_odd_scale",
)
RETIRED_FILES: tuple[str, ...] = (
    "task.py",
    "track.json",
    "kernel_triton.py",
    "kernel_triton_msl.py",
    "naive.metal",
    "__init__.py",
)
STANDARD_FILES: tuple[str, ...] = ("opspec.json", "reference.py", "reference.json", "display.json")

A5_TEST = "tests/backends/metal/test_task_migration_a5.py"
A4_TEST = "tests/backends/metal/test_track_a4.py"

DELIVERABLES: tuple[str, ...] = (
    *(f"tasks/{task}/{name}" for task in TASKS for name in STANDARD_FILES),
    *(f"tasks/{task}/{name}" for task in TASKS if task not in CORRECTNESS_ONLY for name in ("baselines.json", "baselines.py")),
    "tasks/__init__.py",
    "tests/support/v1_tasks.py",
    "tests/support/pack_identity.py",
    "tests/support/remote_contract.py",
    "tests/support/job_engine.py",
    A5_TEST,
    "tests/shared/test_remote_grpc_tls.py",
    "tests/shared/test_remote_go_conformance.py",
    "tests/shared/test_remote_blob_go_conformance.py",
    "tests/shared/test_remote_npu_bundle_go_conformance.py",
    "tests/shared/test_remote_npu_build_go_conformance.py",
    "tests/shared/test_v1_identity_g0.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/g0_compare.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/migrate.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/perf.py",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/verify.py",
    *(f"docs/架构/跨硬件算子生成分工-阶段验收/A5/evidence/perf_{phase}_{size}{suffix}.json"
      for phase in ("v1", "v2") for size in ("paper", "small") for suffix in ("", "_r2", "_r3")),
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/evidence/perf_gate.json",
    *(f"docs/架构/跨硬件算子生成分工-阶段验收/A5/evidence/slices/{task}.json"
      for task in ("sigmoid", "rmsnorm", "gemm", "reduction_sum", "gen_swiglu", "sem_gelu_named_relu")),
)

#: Historical acceptance evidence must survive A5 untouched.  These digests pin the documents and
#: the verifiers; the per-stage manifests are not re-run here because A5 intentionally changes the
#: task directories they described (each stage's own acceptance remains the document of record).
HISTORICAL_DIGESTS: dict[str, str] = {
    "docs/架构/跨硬件算子生成分工-阶段验收/G0.md": "cce9fb17f87828d0c6176e1d5c06b801d50e6c1690acadf6631cd04bdde1b883",
    "docs/架构/跨硬件算子生成分工-G0/verify.py": "6890a312b0c57a40fc54d7778cdbd974789c72ccef46d5109f73d224baa3b76f",
    "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json": "057c2bff81d355d9ff4f17ba3e5a4df5092471e97fe4af75f0d325f234be4131",
    "docs/架构/跨硬件算子生成分工-G0/corpus/manifest.sha256": "0d034b17aa2764a9799908247fece8f7611911ebfcaf7487e2bcecfee3dbff21",
    "docs/架构/跨硬件算子生成分工-G0/corpus/index.json": "ee50e82c81b8495a2cca772ff2cc7902ae852a2afcb797514ee6f14ba92c0266",
    "docs/架构/跨硬件算子生成分工-G0/performance-contract.json": "19a239b6f9edf7fb5975eba3c7ae37c35d9b82075148145acc16302d8355d818",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1.md": "b2dfde7687888b6d3fae4c4b41d020d1161e5aa4efc53dc52ae677534d6a3195",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1/verify.py": "1dfb8d44bea0a1d7cf83f6a13439dd648137de181d58427b5706e9d0b05521e9",
    "docs/架构/跨硬件算子生成分工-阶段验收/A1/manifest.sha256": "cc2e444765472cd426a606fdf98fff7f792538a65dc14a37af6d04f253fce0d9",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2.md": "43622ab11574a9ec701be61dd27f78b18ab44b61be1430b6a9b638c46f9cb293",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2/verify.py": "f37b7d530261b037c11452ae30cb3cd6c1cd0e67603ce0d650157da120448f21",
    "docs/架构/跨硬件算子生成分工-阶段验收/A2/manifest.sha256": "654b50f9956ee9fafe1cfd6c69cdb645f1e6c5a04eb82d45c8662a1e0311d42d",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3.md": "19d00e6f0026e6ae9c7a0a341955d21fa25a1b58c4dcafbdcab35639567f56b8",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3/verify.py": "f9fcef86930126d92c5b07beb137352e58fc306e56e443925bb7d2ac833abc58",
    "docs/架构/跨硬件算子生成分工-阶段验收/A3/manifest.sha256": "0f06cce33b64efad90c47b7f9a8ed87237907313d6c43e242e02e26e7f6aa355",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4.md": "ca26ca3e0320a8b2f35760ef02b91262c5eb1ab2c3f23bf33738c427a9604b20",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/verify.py": "389398e9bdb54bf132fedfdf06795b6e2e028f02a9affc8aa68c984f61dd3259",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/manifest.sha256": "4690a55e5fa8cd835c1ae364935864858ee96ecc36e6747efe148ef05b9b3bcc",
}

FIXED_CPU_NODE_IDS: tuple[str, ...] = (
    f"{A5_TEST}::test_a5_g0_combo_matches_the_frozen_corpus",
    f"{A5_TEST}::test_a5_covers_exactly_the_frozen_inventory",
    f"{A5_TEST}::test_a5_the_shipped_standard_dirs_are_v2_and_carry_no_v1_assembly",
    f"{A5_TEST}::test_a5_the_out_of_scope_v1_tasks_keep_their_v1_assembly",
    f"{A5_TEST}::test_a5_the_v1_registry_refuses_a_v2_task_instead_of_falling_back",
    f"{A5_TEST}::test_a5_a_mixed_directory_is_an_assembly_error",
    f"{A5_TEST}::test_a5_a_v2_task_without_a_reference_is_refused",
    f"{A5_TEST}::test_a5_a_v2_task_refuses_a_launch_variant",
    f"{A5_TEST}::test_a5_the_reference_boundary_holds_for_the_migrated_tasks",
    f"{A5_TEST}::test_a5_the_sampler_block_is_identical_in_every_migrated_reference",
    f"{A5_TEST}::test_a5_the_frozen_candidate_geometry_is_the_g0_contract_not_a_fallback",
    f"{A5_TEST}::test_a5_the_baseline_identity_is_the_frozen_g0_registration",
    f"{A5_TEST}::test_a5_the_performance_gate_is_recomputed_not_trusted",
    f"{A5_TEST}::test_a5_a_missing_registered_baseline_is_an_explicit_evidence_error",
    f"{A5_TEST}::test_a5_an_invalid_task_name_is_a_lookup_miss_not_a_filesystem_error",
    f"{A5_TEST}::test_a5_the_frozen_cross_language_vectors_do_not_depend_on_a_live_task",
    "tests/shared/test_v1_identity_g0.py",
    "tests/shared/test_task_contract_v2.py",
    "tests/shared/test_v2_mapping.py",
    "tests/shared/test_opspec.py",
    "tests/shared/test_opspec_v2.py",
    "tests/shared/test_task_fingerprint.py",
    "tests/backends/metal/test_contracts.py::test_the_v1_gemm_launch_variants_are_retired_and_v2_gemm_is_the_live_path",
)

#: The five remote failures and the observe long-name failure must stay fixed.
REGRESSION_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_remote_grpc_tls.py::test_mtls_round_trip_through_the_grpc_client",
    "tests/shared/test_remote_go_conformance.py::test_go_client_completes_conformance_retry_renewal_rotation_and_revocation",
    "tests/shared/test_remote_blob_go_conformance.py::test_the_frozen_blob_vector_matches_the_production_documents",
    "tests/shared/test_remote_npu_bundle_go_conformance.py::test_the_frozen_bundle_vector_matches_the_production_documents",
    "tests/shared/test_remote_npu_build_go_conformance.py::test_the_frozen_build_vector_matches_the_production_documents",
    "tests/shared/test_observe_analysis.py::test_the_panels_keep_the_byte_budget_with_long_text_and_many_epochs",
)

GPU_NODE_IDS: tuple[str, ...] = (
    f"{A5_TEST}::test_a5_gpu_sigmoid_elementwise_real_slice",
    f"{A5_TEST}::test_a5_gpu_rmsnorm_two_inputs_and_an_attr_real_slice",
    f"{A5_TEST}::test_a5_gpu_gemm_tiled_real_slice",
    f"{A5_TEST}::test_a5_gpu_reduction_sum_module_error_scale_real_slice",
    f"{A5_TEST}::test_a5_gpu_gen_swiglu_two_inputs_real_slice",
    f"{A5_TEST}::test_a5_gpu_sem_gelu_named_relu_is_correctness_only_real_slice",
    f"{A5_TEST}::test_a5_gpu_hand_written_simdgroup_gemm_candidate",
    # Frozen A2/A4 invariants the A5 performance protocol relies on: read_write restore, declared
    # zero-workspace fill inside timing, and the post-reps guard check.
    f"{A4_TEST}::test_a4_read_write_initial_values_are_restored_on_every_trial",
    f"{A4_TEST}::test_a4_declared_zero_workspace_is_cleared_inside_timing_and_host_tiling_restores_it",
    f"{A4_TEST}::test_a4_benchmark_checks_guards_after_reps_and_catches_a_second_launch_oob",
    # The live==archived record check needs A5_EVIDENCE_DIR; it must run in this phase, not skip.
    f"{A5_TEST}::test_a5_the_evidence_directory_matches_the_archived_records",
)

NEGATIVE_NODE_IDS: tuple[str, ...] = (
    f"{A5_TEST}::test_a5_the_v1_registry_refuses_a_v2_task_instead_of_falling_back",
    f"{A5_TEST}::test_a5_a_mixed_directory_is_an_assembly_error",
    f"{A5_TEST}::test_a5_a_v2_task_without_a_reference_is_refused",
    f"{A5_TEST}::test_a5_a_v2_task_refuses_a_launch_variant",
    f"{A5_TEST}::test_a5_the_baseline_identity_is_the_frozen_g0_registration",
    f"{A5_TEST}::test_a5_a_missing_registered_baseline_is_an_explicit_evidence_error",
    f"{A5_TEST}::test_a5_an_invalid_task_name_is_a_lookup_miss_not_a_filesystem_error",
    f"{A5_TEST}::test_a5_the_frozen_cross_language_vectors_do_not_depend_on_a_live_task",
    f"{A5_TEST}::test_a5_the_frozen_candidate_geometry_is_the_g0_contract_not_a_fallback",
    "tests/shared/test_task_fingerprint.py::test_a_candidate_launch_override_does_not_change_the_task_fingerprint",
    "tests/backends/metal/test_contracts.py::test_the_v1_gemm_launch_variants_are_retired_and_v2_gemm_is_the_live_path",
)

SLICE_TASKS: tuple[str, ...] = ("sigmoid", "rmsnorm", "gemm", "reduction_sum", "gen_swiglu", "sem_gelu_named_relu")


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。"""
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_a5(name: str):
    """@brief 载入 A5 验收工具模块（与固定测试用的是同一份脚本）。"""
    sys.path.insert(0, str(ROOT))
    path = A5 / f"{name}.py"
    module_name = f"_a5_verify_{name}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def _pytest(command: list[str], *, env: dict[str, str] | None = None, where: str) -> str:
    import os

    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(ROOT))
    environment.update(env or {})
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"{where} failed:\n{output[-6000:]}")
    summary = [line for line in output.splitlines() if re.search(r"\bpassed\b", line)]
    require(summary, f"{where}: no pytest summary line")
    require(" skipped" not in summary[-1] and "xfailed" not in summary[-1], f"{where} skipped a fixed node: {summary[-1]}")
    return summary[-1]


# ---------------------------------------------------------------------------------------------
# Manifest and historical evidence
# ---------------------------------------------------------------------------------------------


def write_manifest() -> None:
    """@brief 重写 A5 manifest.sha256。"""
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
    require(
        set(seen) == expected,
        f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}",
    )
    for name, digest in seen.items():
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def verify_historical() -> None:
    """@brief G0/A1-A4 的验收文档、verifier 与 manifest 逐字节未改。"""
    for name, digest in HISTORICAL_DIGESTS.items():
        path = ROOT / name
        require(path.is_file(), f"historical evidence is missing: {name}")
        require(sha256(path) == digest, f"historical evidence changed: {name}")
    # The A5 baseline commit must still be the commit this stage started from.
    base = _git("rev-parse", "--verify", f"{A5_BASE_HEAD}^{{commit}}").strip()
    require(base == A5_BASE_HEAD, f"A5 base commit is unavailable: {A5_BASE_HEAD}")
    print(f"OK: historical G0/A1-A4 evidence untouched ({len(HISTORICAL_DIGESTS)} files; base {A5_BASE_HEAD[:12]})")


# ---------------------------------------------------------------------------------------------
# Atomic migration, strict dispatch, and the v1 fixture boundary
# ---------------------------------------------------------------------------------------------


def verify_atomic_migration() -> None:
    """@brief 16 个目录只有 v2 标准 entry；旧 v1 装配文件在同一变更里消失。"""
    sys.path.insert(0, str(ROOT))
    from tcenv.taskcontract import inspect_standard_entry

    for task in TASKS:
        directory = ROOT / "tasks" / task
        entry = inspect_standard_entry(directory)
        require(entry.is_v2, f"{task}: not a v2 standard entry")
        require(entry.files[:2] == ("opspec.json", "reference.py"), f"{task}: standard files are {entry.files}")
        for name in RETIRED_FILES:
            require(not (directory / name).exists(), f"{task}: retired v1 file still exists: {name}")
    # Preserved v1 tasks keep their v1 assembly and are not part of this change.
    for task in PRESERVED_V1:
        directory = ROOT / "tasks" / task
        entry = inspect_standard_entry(directory)
        require(not entry.is_v2 and entry.mode == "v1", f"{task}: preserved v1 task changed mode")
        for name in ("task.py", "track.json", "kernel_triton.py"):
            require((directory / name).is_file(), f"{task}: preserved v1 assembly lost {name}")
    changed = _git("status", "--porcelain", "--", *(f"tasks/{task}" for task in PRESERVED_V1)).splitlines()
    require(changed == [], "A5 modified a preserved v1 task: " + ", ".join(changed))
    print(f"OK: 16 v2-only directories, {len(RETIRED_FILES)} retired markers gone, {len(PRESERVED_V1)} v1 tasks untouched")


def verify_strict_dispatch() -> None:
    """@brief v1 registry 按 spec_version 拒绝 v2（无 fallback）；非法题名不是文件系统错误。"""
    sys.path.insert(0, str(ROOT))
    import pytest

    from tasks import get_task
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import discover_standard_tasks

    for task in TASKS:
        for spelling in (task, f"{task}@128x64x32t256"):
            with pytest.raises(TaskContractError) as error:
                get_task(spelling, "fp16", "small")
            require("v2 standard entry" in str(error.value), f"{spelling}: wrong refusal: {error.value}")
    for name in ("task-" + "x" * 494, "../escape", "/etc/passwd", ".."):
        try:
            get_task(name, "fp16", "small")
        except (KeyError, TaskContractError):
            continue
        raise AssertionError(f"invalid task name {name!r} was assembled")
    discovered = {
        locator.op_name for locator in discover_standard_tasks(ROOT / "tasks", code_root=ROOT)
        if locator.op_name in TASKS
    }
    require(discovered == set(TASKS), f"discovery found {sorted(discovered)} != the 16 migrated tasks")
    print(f"OK: strict dispatch (v1 refuses 16 v2 names and @variant; discovery finds {len(discovered)})")


def verify_v1_fixture_boundary() -> None:
    """@brief 测试夹具不复活迁移题：唯一 v1 题名来自冻结边界，指纹校验仍然生效。"""
    sys.path.insert(0, str(ROOT))
    from tests.support.job_engine import TASK_FINGERPRINT, identity
    from tests.support.v1_tasks import V1_TASK, V1_TASKS, v1_task

    boundary = json.loads((ROOT / "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json").read_text(encoding="utf-8"))
    preserved = tuple(name for name in boundary["preserved_v1_out_of_scope"] if not name.startswith("sem_int "))
    require(tuple(V1_TASKS) == preserved, "tests/support/v1_tasks.py is a second list, not the frozen boundary")
    require(V1_TASK in preserved, "the shared v1 fixture task is not a preserved v1 task")
    task = v1_task()
    require(task.name == V1_TASK, "the v1 fixture did not assemble a preserved v1 task")
    service = identity()
    require(service.accepts_task(V1_TASK, "fp16", "small", TASK_FINGERPRINT), "the fixture identity rejects its own task")
    # The fingerprint is bound to one (task, dtype, size): a migrated name is still refused.
    for task_name in ("relu", "gemm", "sem_gelu_named_relu", "gen_swiglu"):
        require(
            not service.accepts_task(task_name, "fp16", "small", TASK_FINGERPRINT),
            f"the fixture fingerprint silently accepts {task_name}",
        )
    print(f"OK: v1 fixture boundary ({V1_TASK!r} from the frozen boundary; fingerprint still binds one task)")


def verify_frozen_vectors() -> None:
    """@brief R4/R6 冻结向量由自带身份重建，不依赖已退役的题装配。"""
    sys.path.insert(0, str(ROOT))
    from tests.support import remote_contract as rc

    require(json.loads(json.dumps(rc.build_blob_golden())) == rc.load_blob_golden(), "blob vector drifted")
    for name, builder in (
        ("npu_bundle_v4.json", rc.build_bundle_golden),
        ("npu_build_v5.json", rc.build_build_golden),
    ):
        frozen = json.loads((rc.SUPPORT / "golden" / "remote" / name).read_text(encoding="utf-8"))
        require(json.loads(json.dumps(builder())) == frozen, f"{name} drifted")
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / "evaluator_v1.json"
        shutil.copy(rc.GOLDEN, target)
        rc.write_evaluator_golden(target)
        require(json.loads(target.read_text(encoding="utf-8")) == rc.load_evaluator_golden(), "evaluator vector drifted")
    print("OK: 4 frozen cross-language vectors reproduce without a live task registry")


# ---------------------------------------------------------------------------------------------
# G0 comparison and performance gate
# ---------------------------------------------------------------------------------------------


def verify_g0_corpus() -> None:
    """@brief 16 题/82 组合的语义面对照（字节级由固定 pytest 节点重放）。"""
    sys.path.insert(0, str(ROOT))
    from tcenv.taskcontract import load_task_contract

    g0 = load_a5("g0_compare")
    document = g0.index()
    combos = g0.combos(document)
    require(len(combos) == 82, f"the G0 corpus has {len(combos)} combos, not 82")
    require({combo["task"] for combo in combos} == set(TASKS), "the G0 corpus task set is not the 16 frozen tasks")
    correctness_only: set[str] = set()
    for combo in combos:
        contract = load_task_contract(g0.task_dir(combo["task"]), dtype=combo["dtype"], size=combo["size"])
        checks = g0.compare_semantics(contract, combo)
        for required in ("cases", "tolerances", "baselines", "pack_closure"):
            require(required in checks, f"{g0.combo_id(combo)}: the G0 comparison did not check {required}")
        if not contract.baselines:
            correctness_only.add(combo["task"])
    require(
        correctness_only == set(CORRECTNESS_ONLY),
        f"correctness-only set {sorted(correctness_only)} != the three tasks without G0 baselines",
    )
    print(f"OK: G0 corpus ({len(combos)} combos, 16 tasks): semantics/cases/tolerances/work/baselines/pack")


def verify_performance(fresh: bool) -> None:
    """@brief 三轮归档 + 一轮现场 v2：paper/small 都 <=1.10，且身份/口径与归档一致。"""
    perf = load_a5("perf")
    archived_gate = json.loads((EVIDENCE / "perf_gate.json").read_text(encoding="utf-8"))
    require(archived_gate["approved_max_ratio"] == 1.10, "the archived gate does not use the approved 1.10")
    require([item["size"] for item in archived_gate["gate_instances"]] == ["paper", "small"], "the gate is not paper+small")
    require(len(archived_gate["comparable_tasks"]) == 13, "the gate does not cover 13 comparable tasks")
    require(archived_gate["correctness_only_tasks"] == list(CORRECTNESS_ONLY), "correctness-only tasks drifted")
    require(archived_gate["gate_pass"] is True, "the archived paper gate failed")
    require(archived_gate["small_geomean"] <= 1.10 and archived_gate["gate_geomean"] <= 1.10, "an archived gate is above 1.10")
    for instance in archived_gate["gate_instances"]:
        require(instance["rounds"] >= 3, f"{instance['size']}: fewer than three fresh rounds")
        require(instance["pass"] is True and instance["round_pass"] is True, f"{instance['size']}: a round failed the gate")
        require(len(instance["ratios"]) == 13, f"{instance['size']}: ratios are not 13")
        require(set(instance["measured_baselines"]) == set(archived_gate["comparable_tasks"]), f"{instance['size']}: baselines missing")
        for task, baselines in instance["measured_baselines"].items():
            require(set(baselines) == {"mlx", "torch_mps"}, f"{task}/{instance['size']}: baselines {sorted(baselines)}")
            for value in baselines.values():
                require(value["v1_us"] > 0 and value["v2_us"] > 0, f"{task}/{instance['size']}: non-positive baseline")

    # ``summarize_gate`` records evidence paths relative to the repository root, so the scratch
    # tree must live under it.
    with tempfile.TemporaryDirectory(prefix=".a5-perf-", dir=ROOT) as temporary:
        root = Path(temporary)
        archived = root / "archived"
        archived.mkdir()
        for name in DELIVERABLES:
            path = ROOT / name
            if path.parent == EVIDENCE and path.suffix == ".json":
                shutil.copy(path, archived / path.name)
        # 1. Recompute the three archived rounds from the records; the stored gate must be exactly
        #    what the current code derives (a rewritten perf_gate.json is caught here).
        perf.EVIDENCE = archived
        recomputed = perf.summarize_gate()
        for key in ("approved_max_ratio", "gate_pass", "comparable_tasks", "correctness_only_tasks", "measurement"):
            require(recomputed[key] == archived_gate[key], f"archived perf_gate.{key} does not recompute")
        for key in ("gate_geomean", "small_geomean"):
            require(abs(recomputed[key] - archived_gate[key]) < 1e-12, f"archived perf_gate.{key} does not recompute")
        for key in ("gate_ratios", "small_ratios"):
            require(recomputed[key] == archived_gate[key], f"archived perf_gate.{key} does not recompute")
        require(recomputed["gate_pass"] is True, "the recomputed archived gate failed")
        print(f"OK: archived 3-round gate recomputed (paper {recomputed['gate_geomean']:.6f}, small {recomputed['small_geomean']:.6f})")

        if not fresh:
            print("WARNING: the fresh v2 re-measurement was skipped; A5 is not accepted")
            return

        # 2. One fresh v2 round on both instances, folded into the same identity/protocol checks.
        fresh_dir = root / "fresh"
        shutil.copytree(archived, fresh_dir)
        perf.EVIDENCE = fresh_dir
        document = perf.index()
        from tcenv.gpulock import gpu_lock

        with gpu_lock("A5 verify fresh v2 re-measurement", exclusive=False):
            for size in ("paper", "small"):
                perf.run_v2(document, perf.evidence_file("v2", size, 4), only=(), size=size)
        # ``summarize_gate`` folds whole rounds and therefore needs a v1 file for every round; the
        # v1 assembly no longer exists, so a v2-only fourth round cannot be folded there.  The fresh
        # check uses the *same* approved metric on the archived three-round v1 median: the fresh v2
        # median over the v1 median, per instance, for the same 13 comparable tasks.
        import statistics

        comparable = set(recomputed["comparable_tasks"])
        fresh_ratios: dict[str, dict[str, float]] = {}
        fresh_geomeans: dict[str, float] = {}
        fresh_records: dict[str, dict[str, dict]] = {}
        v1_median: dict[str, dict[str, float]] = {}
        for size in ("paper", "small"):
            rounds = [
                {
                    record["task"]: record for record in json.loads(
                        (EVIDENCE / f"perf_v1_{size}{'' if index == 1 else f'_r{index}'}.json").read_text()
                    )["records"]
                }
                for index in (1, 2, 3)
            ]
            v1_median[size] = {
                task: statistics.median(round_records[task]["median_us"] for round_records in rounds)
                for task in TASKS
            }
            fresh = json.loads((fresh_dir / f"perf_v2_{size}_r4.json").read_text(encoding="utf-8"))
            require(fresh["phase"] == "v2" and fresh["size"] == size, f"fresh {size}: wrong phase/size")
            fresh_records[size] = {record["task"]: record for record in fresh["records"]}
            require(set(fresh_records[size]) == set(TASKS), f"fresh {size}: not the 16 tasks")
            for task, record in fresh_records[size].items():
                perf.validate_record(record, phase="v2", size=size, where=f"fresh-{size}-{task}")
            fresh_ratios[size] = {
                task: fresh_records[size][task]["median_us"] / v1_median[size][task] for task in comparable
            }
            fresh_geomeans[size] = math.exp(
                sum(math.log(value) for value in fresh_ratios[size].values()) / len(fresh_ratios[size])
            )
            require(
                fresh_geomeans[size] <= 1.10,
                f"fresh {size} v2/v1 median geomean {fresh_geomeans[size]:.6f} > 1.10",
            )
            print(f"OK: fresh {size} v2/v1 median geomean {fresh_geomeans[size]:.6f} <= 1.10 ({len(fresh_ratios[size])} tasks)")

        # 3. The fresh measurements must be the same task/candidate/launch/baseline/protocol.
        archived_v2 = json.loads((EVIDENCE / "perf_v2_paper.json").read_text(encoding="utf-8"))["source"]
        for size in ("paper", "small"):
            fresh = json.loads((fresh_dir / f"perf_v2_{size}_r4.json").read_text(encoding="utf-8"))
            require(fresh["source"]["head"] == archived_v2["head"], f"fresh {size}: different repository head")
            require(
                fresh["source"]["protocol_digest"] == archived_v2["protocol_digest"],
                f"fresh {size}: the timing protocol (bench/evaluate) changed",
            )
            print(
                f"  fresh {size} provenance: head={fresh['source']['head'][:12]} "
                f"protocol={fresh['source']['protocol_digest'][:12]} tasks={fresh['source']['tasks_digest'][:12]}"
            )
            old = {record["task"]: record for record in json.loads((EVIDENCE / f"perf_v2_{size}.json").read_text())["records"]}
            for task, record in fresh_records[size].items():
                for key in (
                    "candidate_sha256", "g0_body_sha256", "g0_opspec_digest", "g0_kernel_name",
                    "g0_grid", "g0_threads", "baseline_implementation_digest", "target_id",
                    "device_name", "gpu_family", "benchmark_rotation_sets",
                    "dispatch", "provider", "effective_correctness_only", "baseline_registrations",
                ):
                    require(record[key] == old[task][key], f"fresh {size}/{task}: {key} drifted")
                require(record["provider"] == "candidate", f"fresh {size}/{task}: not the candidate provider")
                require((record.get("track") or {}).get("track_id") == "msl-single", f"fresh {size}/{task}: wrong track")
                require(
                    {name: value["implementation_id"] for name, value in record["baselines"].items()}
                    == {name: value["implementation_id"] for name, value in old[task]["baselines"].items()},
                    f"fresh {size}/{task}: baseline implementations drifted",
                )
        print("OK: fresh v2 identity/rotation/baseline/protocol/body/launch match the archived rounds")

    # 4. A missing registered baseline is an explicit refusal, never a folded/skipped row.
    record = json.loads((EVIDENCE / "perf_v2_paper.json").read_text(encoding="utf-8"))["records"]
    record = next(item for item in record if item["baselines"])
    perf.validate_record(record, phase="v2", size="paper", where="verify-positive")
    broken = json.loads(json.dumps(record))
    broken["baselines"].pop("mlx")
    try:
        perf.validate_record(broken, phase="v2", size="paper", where="verify-negative")
    except perf.EvidenceError as error:
        require("mlx" in str(error) and "missing" in str(error), f"unexpected refusal: {error}")
    else:
        raise AssertionError("a missing registered baseline was accepted")
    print("OK: missing baseline is refused by name (mlx)")


# ---------------------------------------------------------------------------------------------
# Real GPU slices (canonical geometry/dispatch/build identity)
# ---------------------------------------------------------------------------------------------


def verify_slice_evidence(directory: Path) -> None:
    """@brief 六条真实切片的 live 记录与归档逐字段一致，且 dispatch/几何来自 G0 契约。"""
    archived = EVIDENCE / "slices"
    present = {path.name for path in archived.glob("*.json")}
    require(present == {f"{task}.json" for task in SLICE_TASKS}, f"archived slices differ: {sorted(present)}")
    g0 = load_a5("g0_compare")
    combos = {
        combo["task"]: combo for combo in g0.combos(g0.index())
        if combo["dtype"] == "fp16" and combo["size"] == "small" and combo.get("variant") is None
    }
    for task in SLICE_TASKS:
        live = json.loads((directory / f"{task}.json").read_text(encoding="utf-8"))
        expected = json.loads((archived / f"{task}.json").read_text(encoding="utf-8"))
        for key in (
            "candidate_sha256", "geometry_identity", "bundle_digest", "artifact_set_digest",
            "native_digest", "presentation_sha256",
        ):
            require(isinstance(live.get(key), str) and len(live[key]) == 64, f"{task}.{key} is not a digest")
            require(live[key] == expected[key], f"{task}.{key} differs from the archived slice")
        require(live["dispatch"] == expected["dispatch"], f"{task}: dispatch differs from the archived slice")
        contract = combos[task]["contract"]
        require(
            live["dispatch"]
            == [[contract["kernel_name"], list(map(int, contract["grid"])), list(map(int, contract["threads"])), 0, []]],
            f"{task}: real dispatch {live['dispatch']} != the G0 kernel/grid/threads",
        )
        require(live["provider"] == "candidate", f"{task}: the slice did not run the candidate provider")
        require(live["status"] == "ok" and live["abi_ok"] is True and live["correctness_ok"] is True, f"{task}: slice failed")
        require(live["median_us"] > 0, f"{task}: no real timing")
        require(live["effective_correctness_only"] is (task in CORRECTNESS_ONLY), f"{task}: correctness-only boundary")
        require(live["track"]["track_id"] == "msl-single", f"{task}: wrong track")
        require(0.2 <= live["median_us"] / expected["median_us"] <= 5.0, f"{task}: timing regime changed")
    print(f"OK: {len(SLICE_TASKS)} real slices: G0 dispatch, candidate/build/geometry identity, live == archived")


# ---------------------------------------------------------------------------------------------
# Pytest phases and negatives
# ---------------------------------------------------------------------------------------------


def verify_negative_nodes() -> None:
    """@brief 固定负例节点仍然存在。"""
    for node in NEGATIVE_NODE_IDS:
        path, _, test = node.partition("::")
        test = test.split("[", 1)[0]
        source = (ROOT / path).read_text(encoding="utf-8")
        require(f"def {test}(" in source, f"negative node id is gone: {node}")
    print(f"OK: {len(NEGATIVE_NODE_IDS)} stable negative node ids present")


def run_cpu_pytest() -> None:
    """@brief 运行固定 A5 CPU 节点（含 82 组合的字节级 G0 对照）。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *FIXED_CPU_NODE_IDS],
        where="fixed A5 CPU pytest",
    )
    print(f"OK: fixed A5 CPU pytest\n{summary}")


def run_regression_pytest() -> None:
    """@brief 运行五条 remote 回归与 observe 长题名回归（任何 skip 都不算通过）。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *REGRESSION_NODE_IDS],
        where="fixed A5 remote/observe regressions",
    )
    require("6 passed" in summary, f"expected 6 regression nodes, got: {summary}")
    print(f"OK: fixed A5 remote/observe regressions\n{summary}")


def run_gpu_pytest(evidence_dir: Path) -> None:
    """@brief 运行六条真实 Mac 切片与三条冻结计时不变量，并归档 live 身份。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", *GPU_NODE_IDS],
        env={"A5_EVIDENCE_DIR": str(evidence_dir)},
        where="real A5 GPU slices",
    )
    print(f"OK: real A5 GPU slices\n{summary}")


def self_test() -> None:
    """@brief 可独立重放的 A5 负例（不需要 GPU）。"""
    sys.path.insert(0, str(ROOT))
    import pytest as _pytest

    from tasks import get_task
    from tcenv.contract_errors import AssemblyError, TaskContractError, UnsupportedContractCapability
    from tcenv.taskcontract import inspect_standard_entry, load_task_contract

    checks: list[str] = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for marker in ("track.json", "task.py", "kernel_triton.py", "naive.metal"):
            directory = root / marker.replace(".", "_")
            directory.mkdir()
            for name in ("opspec.json", "reference.py"):
                shutil.copy(ROOT / "tasks/sigmoid" / name, directory / name)
            (directory / marker).write_text("{}\n" if marker.endswith(".json") else "# retired\n")
            try:
                inspect_standard_entry(directory)
            except AssemblyError as error:
                require("mixed v1/v2" in str(error), f"{marker}: wrong mixed error: {error}")
            else:
                raise AssertionError(f"{marker}: a mixed v1/v2 directory was accepted")
        checks.append("a leftover v1 marker in a v2 directory is an assembly error")
        missing = root / "no_reference"
        missing.mkdir()
        shutil.copy(ROOT / "tasks/sigmoid/opspec.json", missing / "opspec.json")
        try:
            inspect_standard_entry(missing)
        except AssemblyError as error:
            require("requires reference.py" in str(error), f"wrong missing-reference error: {error}")
        else:
            raise AssertionError("a v2 task without reference.py was accepted")
        checks.append("a v2 task without reference.py is refused (no v1 fallback)")

    with _pytest.raises(UnsupportedContractCapability):
        load_task_contract(ROOT / "tasks/gemm", dtype="fp16", size="small", variant="128x64x32t256")
    checks.append("a retired launch variant is not a second contract")

    with _pytest.raises(TaskContractError):
        get_task("relu", "fp16", "small")
    with _pytest.raises(KeyError):
        get_task("task-" + "x" * 494, "fp16", "small")
    checks.append("v1 refuses a migrated name; an over-long name is a lookup miss, not ENAMETOOLONG")

    long_task = "task-" + "x" * 494
    sys.path.insert(0, str(ROOT))
    from experiments.common import synthetic_confirmation_documents

    result, report = synthetic_confirmation_documents(
        task=long_task, dtype="fp16", size="small", target_id="apple-m4-max",
        speedup=1.0, label="r1.p0", kernel_sha256="0" * 64, values=[1.0], protocol_version="3",
    )
    require(result["task"] == long_task and report["status"] == "ok", "the observe long-name fixture cannot build")
    checks.append("the observe long-task-name fixture builds a synthetic identity instead of stat-ing a path")

    perf = load_a5("perf")
    record = json.loads((EVIDENCE / "perf_v2_paper.json").read_text(encoding="utf-8"))["records"][0]
    perf.validate_record(record, phase="v2", size="paper", where="self-test")
    broken = json.loads(json.dumps(record))
    broken["baselines"].pop(next(iter(broken["baselines"])))
    try:
        perf.validate_record(broken, phase="v2", size="paper", where="self-test-negative")
    except perf.EvidenceError:
        pass
    else:
        raise AssertionError("a missing baseline was folded into the gate")
    checks.append("a missing registered baseline is an explicit evidence error")

    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="static/CPU only; not an A5 acceptance")
    parser.add_argument("--skip-fresh-perf", action="store_true", help="diagnostic only; not an A5 acceptance")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    verify_manifest()
    verify_historical()
    verify_atomic_migration()
    verify_strict_dispatch()
    verify_v1_fixture_boundary()
    verify_frozen_vectors()
    verify_g0_corpus()
    verify_negative_nodes()
    self_test()
    if not args.skip_pytest:
        run_cpu_pytest()
        run_regression_pytest()
        if not args.skip_gpu:
            with tempfile.TemporaryDirectory(prefix="a5-evidence-") as temporary:
                directory = Path(temporary)
                run_gpu_pytest(directory)
                verify_slice_evidence(directory)
    verify_performance(fresh=not args.skip_fresh_perf)
    if args.skip_gpu or args.skip_pytest or args.skip_fresh_perf:
        print("WARNING: a skipped phase is diagnostic only; A5 is not accepted")
    else:
        print("OK: A5 acceptance")


if __name__ == "__main__":
    main()
