#!/usr/bin/env python3
"""A6 acceptance verifier: the migrated v1 implementation and the duplicated layer are gone.

The verifier starts no model and no search.  It checks the *result* of the retirement:

* the retired implementation is really absent: the 16 migrated names are not registered by the
  v1 registry (a typed ``spec_version`` refusal, no fallback), ``tasks/generated/`` and the
  helpers that only served the deleted assembly are gone, and the 9 migrated Metal
  ``LoweringProfile`` entries / 2 lowerers / GEMM variant factory are not registered — while
  every *preserved* profile is byte-identical to the pre-A6 file (proved by re-registering the
  ``git show HEAD`` version; a changed profile cannot be registered twice);
* the preserved v1 path still assembles and keeps its evaluation boundary: the 8
  ``preserved_v1_out_of_scope`` tasks, the dynamic ``sem_int`` family, ``flash_attention@…``,
  the recorded Mac phase-0 contract/prompt anchors, and the untouched task directories;
* the 16 migrated standard directories are still one v2 entry each with no v1 marker, no
  per-task profile and no launch fact; the frozen G0 corpus still has 82 combinations and the
  A5 byte-level comparison still passes;
* each configuration class from the G0 inventory has one reader, each upstream-map
  responsibility one local authority, and the Track registry stays task-agnostic;
* both live paths run on the real Mac: a v2 ``evaluate_task`` candidate with no v1 profile
  read at all, and a preserved v1 task with real timing; the live records equal the archived
  A6 records;
* the A5 evidence is untouched: its manifest still matches, its v1 pre-migration measurements
  keep their digests, and A6's fresh ``paper``/``small`` v2 round is measured here (not read
  from A5) and folded against the archived v1 median, with the same 1.10 gate.

A skipped phase is diagnostic only: A6 is not accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
A6 = Path(__file__).resolve().parent
MANIFEST = A6 / "manifest.sha256"
EVIDENCE = A6 / "evidence"

A5 = ROOT / "docs/架构/跨硬件算子生成分工-阶段验收/A5"
A5_MANIFEST = A5 / "manifest.sha256"
G0 = ROOT / "docs/架构/跨硬件算子生成分工-G0"
BASE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"

MIGRATED: tuple[str, ...] = (
    "sigmoid", "relu", "silu", "swiglu", "rmsnorm", "softmax", "reduction_sum", "gemm",
    "gemm_silu", "sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu",
    "gen_relu", "gen_sigmoid", "gen_silu", "gen_swiglu",
)
PRESERVED: tuple[str, ...] = (
    "rope", "gemv", "conv2d", "flash_attention", "lion", "bitdelta_gemm",
    "mamba2_chunk_scan", "sem_attention_odd_scale",
)
RETIRED_FILES: tuple[str, ...] = (
    "task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py",
)
RETIRED_PROFILES: tuple[str, ...] = (
    "metal-gemm", "metal-gemm-silu", "metal-reduction-sum", "metal-relu", "metal-rmsnorm",
    "metal-sigmoid", "metal-silu", "metal-softmax", "metal-swiglu",
)
RETIRED_LOWERERS: tuple[str, ...] = ("metal.elementwise", "metal.gemm")
RETIRED_VARIANT_FACTORY = "metal.gemm.variant"
DELETED_PATHS: tuple[str, ...] = (
    "tasks/generated", "tasks/_pointwise.py", "tasks/_matmul.py",
)
CORRECTNESS_ONLY: tuple[str, ...] = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu")

A6_TEST = "tests/shared/test_a6_retirement.py"
A6_METAL_TEST = "tests/backends/metal/test_a6_metal_config.py"
A6_GPU_TEST = "tests/backends/metal/test_a6_retirement_gpu.py"
A5_TEST = "tests/backends/metal/test_task_migration_a5.py"
A4_TEST = "tests/backends/metal/test_track_a4.py"

DELIVERABLES: tuple[str, ...] = (
    "tasks/__init__.py",
    "tasks/semantic.py",
    "tcenv/backends/metal/lowering.py",
    "tests/support/hw_golden.py",
    "tests/support/hang_candidate.py",
    "tests/support/orchestrator.py",
    "tests/shared/test_task_equivalence.py",
    "tests/shared/test_lowering_registry.py",
    "tests/shared/test_track_binding.py",
    "tests/backends/metal/test_generated_tasks.py",
    "tests/backends/metal/test_baselines.py",
    "tests/backends/metal/test_build_path_gpu.py",
    "tests/backends/metal/test_dispatch.py",
    "tests/backends/metal/test_ieee.py",
    "tests/backends/metal/test_orchestrator_gpu.py",
    "tests/backends/metal/test_source_builder.py",
    "tests/backends/metal/test_tcenv.py",
    "tests/backends/metal/test_track_a4.py",
    "tests/backends/metal/test_xctrace.py",
    A6_TEST,
    A6_METAL_TEST,
    A6_GPU_TEST,
    "docs/架构/跨硬件算子生成分工-阶段验收/A6.md",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/verify.py",
    *(f"docs/架构/跨硬件算子生成分工-阶段验收/A6/evidence/{name}" for name in (
        "retirement_inventory.json",
        "gpu_v2_sigmoid.json",
        "gpu_v1_rope.json",
        "perf_v2_paper_a6.json",
        "perf_v2_small_a6.json",
        "perf_gate_a6.json",
    )),
)

#: Historical acceptance documents/verifiers/manifests must survive A6 byte-for-byte.  The
#: G0/A1-A4 entries are copied verbatim from A5/verify.py (a pin taken *before* A6 started, so
#: not a digest A6 could have written); the A5 entries are the digests A5/manifest.sha256 lists
#: for its own records.  A6 never re-runs an old stage verifier: their task directories were
#: migrated in A5, so the record of those stages is the document, not a re-run.
HISTORICAL_DIGESTS: dict[str, str] = {
    "docs/架构/跨硬件算子生成分工-阶段验收/G0.md": "cce9fb17f87828d0c6176e1d5c06b801d50e6c1690acadf6631cd04bdde1b883",
    "docs/架构/跨硬件算子生成分工-G0/verify.py": "6890a312b0c57a40fc54d7778cdbd974789c72ccef46d5109f73d224baa3b76f",
    "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json": "057c2bff81d355d9ff4f17ba3e5a4df5092471e97fe4af75f0d325f234be4131",
    "docs/架构/跨硬件算子生成分工-G0/corpus/manifest.sha256": "0d034b17aa2764a9799908247fece8f7611911ebfcaf7487e2bcecfee3dbff21",
    "docs/架构/跨硬件算子生成分工-G0/corpus/index.json": "ee50e82c81b8495a2cca772ff2cc7902ae852a2afcb797514ee6f14ba92c0266",
    "docs/架构/跨硬件算子生成分工-G0/performance-contract.json": "19a239b6f9edf7fb5975eba3c7ae37c35d9b82075148145acc16302d8355d818",
    "docs/架构/跨硬件算子生成分工-G0/upstream-map.json": "014a652c7793a9a30f469e81fbd0610f3ba6e10314d494c0c1a0674af623cbab",
    "docs/架构/跨硬件算子生成分工-G0/config-inventory.json": "da5f47a32a25b8a9a9233527cdac26d923a8f241b8535f1832cb947acde47461",
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
    "docs/架构/跨硬件算子生成分工-阶段验收/A5.md": "f08e239c86c44a92b69f3e16e5d32d355ff871be8b94bbb8372ee59e037f172d",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/verify.py": "146c7d6e3e707244d46a49929625a7848d9f64cd0a8257d494dd7664e44e7b85",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/manifest.sha256": "b0668b8786952d1171fa51ae8d90fa0d50678c9ec87d484578a9a128a146d406",
}

#: Fixed A6 CPU nodes: the retirement boundary, the preserved path, the one-reader rule.
FIXED_CPU_NODE_IDS: tuple[str, ...] = (
    f"{A6_TEST}::test_a6_the_v1_registry_registers_exactly_the_preserved_tasks",
    f"{A6_TEST}::test_a6_the_generated_family_parser_and_its_helpers_are_gone",
    f"{A6_TEST}::test_a6_the_migrated_lowering_entries_are_gone_and_preserved_ones_are_unchanged",
    f"{A6_TEST}::test_a6_the_preserved_v1_tasks_still_assemble_with_their_evaluation_identity",
    f"{A6_TEST}::test_a6_the_migrated_standard_dirs_are_v2_only_and_carry_no_launch_facts",
    f"{A6_TEST}::test_a6_the_track_registry_stays_task_agnostic",
    f"{A6_TEST}::test_a6_each_configuration_class_has_one_reader",
    f"{A6_TEST}::test_a6_the_upstream_map_assigns_each_responsibility_one_local_authority",
    f"{A6_TEST}::test_a6_the_historical_evidence_is_still_pinned_by_its_own_manifest",
    f"{A6_METAL_TEST}::test_a6_the_build_profiles_have_exactly_one_loader_and_no_python_fallback",
    f"{A6_METAL_TEST}::test_a6_the_track_registry_file_has_exactly_one_reader",
    f"{A6_METAL_TEST}::test_a6_the_numeric_verdict_is_one_implementation_shared_by_both_paths",
    f"{A6_METAL_TEST}::test_a6_the_candidate_lifecycle_and_geometry_have_one_seam",
    f"{A6_METAL_TEST}::test_a6_the_preserved_lowering_module_exports_only_names_that_exist",
    f"{A6_METAL_TEST}::test_a6_the_opt_in_hang_candidate_uses_the_preserved_lion_abi",
    f"{A6_TEST}::test_a6_a_noisy_measurement_is_uncertain_and_never_a_successful_confirmation",
)

#: The rejection path the end-to-end success node must not accept: a noisy confirmation batch is
#: ``timing_uncertain`` in the one publication predicate (an independent negative, no GPU).
NOISE_NEGATIVE_NODE_IDS: tuple[str, ...] = (
    "tests/shared/test_r5_policy.py::test_an_explicitly_unstable_batch_stays_uncertain",
    "tests/shared/test_r5_policy.py::test_a_cached_confirm_report_is_uncertain",
    "tests/shared/test_policy.py::test_confirmation_keeps_the_search_best_inside_the_input_batch_floor",
    "tests/shared/test_orchestrator.py::test_confirmation_reports_the_median_of_independent_batches",
    "tests/shared/test_observe_publications.py::test_publication_reason_covers_every_branch",
    "tests/backends/metal/test_orchestrator_gpu.py::test_real_environment_end_to_end_with_scripted_llm",
)

#: The A5 acceptance nodes A6 re-runs to prove the v2 inventory/G0 comparison still holds.
V2_INVENTORY_NODE_IDS: tuple[str, ...] = (
    # The parameterised G0 comparison: one node per frozen combination, each calling
    # ``A5/g0_compare.compare_combo`` (inputs, reference, error scale, cases/tolerances/attrs/work,
    # baseline identity and pack closure, byte-exact).  Without it the phase only re-ran the
    # structural inventory nodes and never replayed the 82-combination byte comparison.
    f"{A5_TEST}::test_a5_g0_combo_matches_the_frozen_corpus",
    f"{A5_TEST}::test_a5_covers_exactly_the_frozen_inventory",
    f"{A5_TEST}::test_a5_the_shipped_standard_dirs_are_v2_and_carry_no_v1_assembly",
    f"{A5_TEST}::test_a5_the_out_of_scope_v1_tasks_keep_their_v1_assembly",
    f"{A5_TEST}::test_a5_the_v1_registry_refuses_a_v2_task_instead_of_falling_back",
    f"{A5_TEST}::test_a5_the_reference_boundary_holds_for_the_migrated_tasks",
    f"{A5_TEST}::test_a5_the_frozen_candidate_geometry_is_the_g0_contract_not_a_fallback",
    f"{A5_TEST}::test_a5_the_baseline_identity_is_the_frozen_g0_registration",
    f"{A5_TEST}::test_a5_a_v2_task_refuses_a_launch_variant",
    f"{A5_TEST}::test_a5_a_missing_registered_baseline_is_an_explicit_evidence_error",
    f"{A5_TEST}::test_a5_an_invalid_task_name_is_a_lookup_miss_not_a_filesystem_error",
)

#: The G0 corpus the phase must replay: 82 frozen combinations over the 16 migrated tasks.
G0_COMBINATIONS = 82
G0_MIGRATED_TASKS = MIGRATED
#: 10 structural nodes + one node per frozen combination.  Pinned so a silently dropped node id or
#: a lost parametrisation fails the phase instead of shrinking the coverage.
V2_INVENTORY_EXPECTED_PASSED = len(V2_INVENTORY_NODE_IDS) - 1 + G0_COMBINATIONS

#: Preserved-v1 anchors: the R0/Mac identity record, the registry rule, the dynamic families
#: and the assembly-structure rule.
PRESERVED_NODE_IDS: tuple[str, ...] = (
    "tests/backends/metal/test_backend_phase0.py",
    "tests/shared/test_lowering_registry.py",
    "tests/shared/test_track_binding.py",
    "tests/shared/test_sem_int_family.py",
    "tests/shared/test_variant_family.py",
    "tests/shared/test_task_assembly_structure.py",
    "tests/shared/test_task_equivalence.py",
)

#: The A5 real-GPU acceptance nodes: the live v2 slices and the A4 frozen timing invariants.
A5_GPU_NODE_IDS: tuple[str, ...] = (
    f"{A5_TEST}::test_a5_gpu_sigmoid_elementwise_real_slice",
    f"{A5_TEST}::test_a5_gpu_rmsnorm_two_inputs_and_an_attr_real_slice",
    f"{A5_TEST}::test_a5_gpu_gemm_tiled_real_slice",
    f"{A5_TEST}::test_a5_gpu_reduction_sum_module_error_scale_real_slice",
    f"{A5_TEST}::test_a5_gpu_gen_swiglu_two_inputs_real_slice",
    f"{A5_TEST}::test_a5_gpu_sem_gelu_named_relu_is_correctness_only_real_slice",
    f"{A5_TEST}::test_a5_gpu_hand_written_simdgroup_gemm_candidate",
    f"{A4_TEST}::test_a4_read_write_initial_values_are_restored_on_every_trial",
    f"{A4_TEST}::test_a4_declared_zero_workspace_is_cleared_inside_timing_and_host_tiling_restores_it",
    f"{A4_TEST}::test_a4_benchmark_checks_guards_after_reps_and_catches_a_second_launch_oob",
)
A6_GPU_NODE_IDS: tuple[str, ...] = (
    f"{A6_GPU_TEST}::test_a6_gpu_the_v2_path_runs_without_any_v1_lowering_profile",
    f"{A6_GPU_TEST}::test_a6_gpu_a_preserved_v1_task_still_evaluates_with_real_timing",
)


def require(condition: bool, message: str) -> None:
    """@brief 断言一个验收条件。"""
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    """@brief 文件字节 sha256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_of_text(text: str) -> str:
    """@brief 字符串 utf-8 字节 sha256。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pytest(
    command: list[str], *, env: dict[str, str] | None = None, where: str, expect_passed: int | None = None,
) -> str:
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
    if expect_passed is not None:
        # The exact count is part of the contract: a parameterised node that silently loses cases
        # (or a dropped node id) must fail here, not shrink the coverage quietly.
        require(f"{expect_passed} passed" in summary[-1], f"{where}: expected {expect_passed} passed, got {summary[-1]}")
    return summary[-1]


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def load_perf():
    """@brief 载入 A5 的测量工具（工具，不是证据；A6 的证据由本脚本现场生成）。"""
    path = A5 / "perf.py"
    name = "_a6_verify_perf"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# Manifest and historical evidence
# ---------------------------------------------------------------------------------------------


def write_manifest() -> None:
    """@brief 重写 A6 manifest.sha256。"""
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


def verify_historical() -> None:
    """@brief G0/A1-A5 的验收文档、verifier、manifest 与 A5 证据逐字节未改。"""
    for name, digest in HISTORICAL_DIGESTS.items():
        path = ROOT / name
        require(path.is_file(), f"historical evidence is missing: {name}")
        require(sha256(path) == digest, f"historical evidence changed: {name}")
    # A5's own manifest still matches every *evidence* entry it covers: A6 wrote no A5 record.
    # (A5's manifest also lists code deliverables; A6 changes some of those on purpose — see
    # A6.md "Manifest staleness".  The evidence entries are the immutable part.)
    lines = A5_MANIFEST.read_text(encoding="utf-8").splitlines()
    require(lines, "the A5 manifest is empty")
    evidence_prefix = "docs/架构/跨硬件算子生成分工-阶段验收/A5/"
    pinned = 0
    for line in lines:
        digest, name = line.split("  ", 1)
        if not name.startswith(evidence_prefix):
            continue
        pinned += 1
        path = ROOT / name
        require(path.is_file(), f"A5 evidence is missing: {name}")
        require(sha256(path) == digest, f"A5 evidence changed: {name}")
    require(pinned >= 20, f"the A5 manifest pins only {pinned} evidence files")
    # The A5 v1 pre-migration performance records are the frozen comparison basis, not A6 input
    # that was overwritten: they are still present, unchanged, and named v1.
    for size in ("paper", "small"):
        for suffix in ("", "_r2", "_r3"):
            path = A5 / "evidence" / f"perf_v1_{size}{suffix}.json"
            document = json.loads(path.read_text(encoding="utf-8"))
            require(document["phase"] == "v1" and document["size"] == size, f"{path.name}: wrong phase")
            require(len(document["records"]) == 16, f"{path.name}: not 16 records")
    require(_git("rev-parse", "--verify", f"{BASE_HEAD}^{{commit}}").strip() == BASE_HEAD, "the baseline commit is gone")
    changed = _git("status", "--porcelain", "--untracked-files=no", "--",
                   "docs/架构/跨硬件算子生成分工-阶段验收/A5",
                   "docs/架构/跨硬件算子生成分工-阶段验收/A4",
                   "docs/架构/跨硬件算子生成分工-阶段验收/A3",
                   "docs/架构/跨硬件算子生成分工-阶段验收/A2",
                   "docs/架构/跨硬件算子生成分工-阶段验收/A1",
                   "docs/架构/跨硬件算子生成分工-G0")
    require(changed.splitlines() == [], "A6 modified a historical acceptance record: " + changed)
    print(f"OK: historical G0/A1-A5 records untouched ({len(HISTORICAL_DIGESTS)} pinned; A5 evidence {pinned} files)")


# ---------------------------------------------------------------------------------------------
# Retirement, preserved v1, and the v2 inventory
# ---------------------------------------------------------------------------------------------


def verify_retirement() -> None:
    """@brief 旧实现确实退出：删除路径缺席、名字条目消失、迁移 profile 未登记。"""
    sys.path.insert(0, str(ROOT))
    from tasks import get_task, list_tasks
    from tasks.semantic import FIXED_NAMES
    from tcenv import lowering
    from tcenv.contract_errors import TaskContractError

    for relative in DELETED_PATHS:
        require(not (ROOT / relative).exists(), f"the retired implementation is still there: {relative}")
    import tasks as task_package

    require(not hasattr(task_package, "GENERATED_NAMES"), "the generated-family name list is back")
    require(not hasattr(task_package, "SEMANTIC_NAMES"), "the semantic name list is back")
    require(set(task_package.TASK_NAMES + task_package.PHASE2_TASK_NAMES + FIXED_NAMES) == set(PRESERVED),
            "the v1 registry does not register exactly the preserved tasks")
    require(list_tasks() == ("rope", "gemv", "conv2d"), f"list_tasks drifted: {list_tasks()}")
    for task in MIGRATED:
        for spelling in (task, f"{task}@128x64x32t256"):
            try:
                get_task(spelling, "fp16", "small")
            except TaskContractError as error:
                require("v2 standard entry" in str(error), f"{spelling}: unexpected refusal: {error}")
            else:
                raise AssertionError(f"the v1 registry assembled the migrated name {spelling!r}")
    names = {profile.name for profile in lowering.registered_profiles("metal")}
    require(names == {
        "metal-rope", "metal-gemv", "metal-lion", "metal-conv2d", "metal-flash-attention",
        "metal-bitdelta-gemm", "metal-mamba2-chunk-scan", "metal-sem-int",
    }, f"registered Metal profiles drifted: {sorted(names)}")
    for profile in RETIRED_PROFILES:
        try:
            lowering.profile_for("metal", profile)
        except lowering.LoweringError:
            continue
        raise AssertionError(f"the retired profile {profile!r} is still registered")
    for lowerer_id in RETIRED_LOWERERS:
        try:
            lowering.lowerer_version(lowerer_id)
        except lowering.LoweringError:
            continue
        raise AssertionError(f"the retired lowerer {lowerer_id!r} is still registered")
    try:
        lowering.variant_factory(RETIRED_VARIANT_FACTORY)
    except lowering.LoweringError:
        pass
    else:
        raise AssertionError("the retired GEMM variant factory is still registered")
    # A deleted symbol must not survive in the module's public surface: ``__all__`` drives
    # ``from ... import *``, so a stale entry is a public import regression even though the
    # registry no longer has the profile.
    import tcenv.backends.metal.lowering as metal_lowering

    retired_exports = {
        "ELEMENTWISE_LOWERER_ID", "ELEMENTWISE_LOWERER_VERSION", "DEFAULT_GEMM_VARIANT",
        "GEMM_PROFILE", "GEMM_SILU_PROFILE", "RELU_PROFILE", "SIGMOID_PROFILE", "SILU_PROFILE",
        "SWIGLU_PROFILE", "REDUCTION_SUM_PROFILE", "RMSNORM_PROFILE", "RMSNORM_PROFILE_NAME",
        "SOFTMAX_PROFILE",
    }
    exported = set(metal_lowering.__all__)
    require(not (exported & retired_exports), f"retired names are still exported: {sorted(exported & retired_exports)}")
    missing = sorted(name for name in exported if not hasattr(metal_lowering, name))
    require(missing == [], f"__all__ lists names that do not exist: {missing}")
    namespace: dict[str, object] = {}
    exec("from tcenv.backends.metal.lowering import *", namespace)  # noqa: S102 - fixed module
    require(exported <= set(namespace), f"import * is missing {sorted(exported - set(namespace))}")
    # The one dangerous opt-in GPU test still has a candidate of the *preserved* ABI: it must be
    # able to time out, not fail to compile with a retired relu-shaped body.
    from tasks import get_task as _get_task

    from tests.support.hang_candidate import lion_hang_findings, lion_hang_source

    lion_naive = _get_task("lion", "fp16", "small").naive_msl
    findings = lion_hang_findings(lion_naive, lion_hang_source(lion_naive))
    require(findings == [], f"the opt-in hang candidate does not match the lion ABI: {findings}")
    print(f"OK: retirement ({len(MIGRATED)} names refused, {len(RETIRED_PROFILES)} profiles / "
          f"{len(RETIRED_LOWERERS)} lowerers / 1 factory gone, {len(DELETED_PATHS)} paths deleted; "
          f"{len(exported)} live exports, import * clean, hang candidate on the lion ABI)")


def verify_preserved_v1() -> None:
    """@brief 保留的 v1 路径仍可装配、@variant/sem_int 仍在、目录未被改动。"""
    sys.path.insert(0, str(ROOT))
    from tasks import get_task
    from tcenv.task import task_fingerprint

    fingerprints: dict[str, str] = {}
    for task in PRESERVED:
        built = get_task(task, "fp16", "small")
        require(built.contract.kernel_name and built.contract.args, f"{task}: empty contract")
        require(built.triton_source and built.naive_msl and built.cases, f"{task}: missing prompt/known-good/cases")
        require(built.specialization is not None and built.specialization.lowering_profile_digest, f"{task}: no specialization")
        require(built.evaluation_implementations["task_pack"]["digest"], f"{task}: no pack identity")
        fingerprints[task] = task_fingerprint(built)
    require(len(set(fingerprints.values())) == len(PRESERVED), "two preserved tasks share a fingerprint")
    require(get_task("sem_int_5", "fp16", "small").contract.kernel_name == "sem_int_kernel", "sem_int family broke")
    default = get_task("flash_attention", "fp16", "paper")
    variant = get_task("flash_attention@64x32t256", "fp16", "paper")
    require(default.contract.threads == (128, 1, 1) and variant.contract.threads == (256, 1, 1), "the @variant family changed")
    require(variant.specialization.lowering_profile_digest != default.specialization.lowering_profile_digest, "the variant reused the default profile")
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", *(f"tasks/{task}" for task in PRESERVED))
    require(changed.splitlines() == [], "A6 modified a preserved v1 task: " + changed)
    # The recorded Mac phase-0 contract/prompt anchors are still reproduced by the live v1 task.
    import hashlib

    from taic import prompts

    document = json.loads((ROOT / "docs/架构/backend-phase0-baseline.json").read_text(encoding="utf-8"))
    anchored = get_task(document["task"], document["dtype"], document["size"])
    require(hashlib.sha256(anchored.contract.describe().encode()).hexdigest() == document["contract_description_sha256"],
            "the v1 contract description drifted from the phase-0 record")
    prompt = prompts.system_prompt(anchored, use_tools=False, target_id=document["target_id"])
    require(hashlib.sha256(prompt.encode()).hexdigest() == document["system_prompt_sha256"],
            "the v1 system prompt drifted from the phase-0 record")
    print(f"OK: preserved v1 ({len(PRESERVED)} tasks assemble, @variant + sem_int intact, phase-0 anchors reproduce)")


def verify_v2_inventory() -> None:
    """@brief 16 个标准目录仍是唯一 v2 entry；G0 冻结语料仍是 82 组合 / 16 题。"""
    sys.path.insert(0, str(ROOT))
    from tcenv.opspec.model import load_opspec
    from tcenv.taskcontract import discover_standard_tasks, inspect_standard_entry

    for task in MIGRATED:
        directory = ROOT / "tasks" / task
        entry = inspect_standard_entry(directory)
        require(entry.is_v2 and entry.files[:2] == ("opspec.json", "reference.py"), f"{task}: not a v2 standard entry")
        for name in RETIRED_FILES:
            require(not (directory / name).exists(), f"{task}: retired v1 file is back: {name}")
        document = load_opspec(directory / "opspec.json").document
        require(document["spec_version"] == 2 and document["op_name"] == task, f"{task}: wrong document")
    discovered = {locator.op_name for locator in discover_standard_tasks(ROOT / "tasks", code_root=ROOT)}
    require(set(MIGRATED) <= discovered, f"discovery misses migrated tasks: {sorted(set(MIGRATED) - discovered)}")
    index = json.loads((G0 / "corpus" / "index.json").read_text(encoding="utf-8"))
    combos = index["combos"]
    require(len(combos) == G0_COMBINATIONS, f"the frozen G0 corpus has {len(combos)} combinations")
    # The byte comparison is replayed by the parameterised node; the verifier refuses to run the
    # phase if that node is not pinned, and refuses a pass count other than 10 + 82 (see
    # ``run_v2_inventory_pytest``).
    combo_node = f"{A5_TEST}::test_a5_g0_combo_matches_the_frozen_corpus"
    require(combo_node in V2_INVENTORY_NODE_IDS, "the 82-combination G0 node is not part of the fixed inventory phase")
    require(V2_INVENTORY_EXPECTED_PASSED == 92, f"the inventory phase must pin 92 nodes, not {V2_INVENTORY_EXPECTED_PASSED}")
    require({combo["task"] for combo in combos} == set(MIGRATED), "the frozen G0 corpus task set drifted")
    correctness_only = {combo["task"] for combo in combos if not combo["baselines"]}
    require(correctness_only == set(CORRECTNESS_ONLY), f"correctness-only set drifted: {sorted(correctness_only)}")
    print(f"OK: v2 inventory ({len(MIGRATED)} standard dirs, G0 {len(combos)} combinations, "
          f"{len(correctness_only)} correctness-only; byte comparison pinned as one node per combination, "
          f"{V2_INVENTORY_EXPECTED_PASSED} expected passes)")


def verify_retirement_inventory(record: bool = False) -> dict:
    """@brief 交叉核对 A6 的删除/保留清单（与被删实现一一对应）；``record`` 才写入证据。

    The archive is immutable during a verification run: a mismatch is a failure, not a silent
    rewrite.  ``--record-evidence`` regenerates it and the caller then re-runs ``--write-manifest``.
    """
    sys.path.insert(0, str(ROOT))
    from tcenv import lowering

    live = {profile.name: profile.to_dict() for profile in lowering.registered_profiles("metal")}
    inventory = {
        "schema_version": 1,
        "base_head": BASE_HEAD,
        "deleted_paths": list(DELETED_PATHS),
        "retired_profiles": list(RETIRED_PROFILES),
        "retired_lowerers": list(RETIRED_LOWERERS),
        "retired_variant_factory": RETIRED_VARIANT_FACTORY,
        "retired_names": list(MIGRATED),
        "preserved_tasks": list(PRESERVED),
        "preserved_profiles": sorted(live),
        "preserved_profile_digests": {
            name: lowering.lowering_profile_digest(lowering.profile_for("metal", name)) for name in live
        },
        "reason": (
            "A5 moved the 16 G0 tasks to the v2 standard entry; A6 removes the v1 implementation those "
            "tasks no longer have (name entries, generated-family parser and helpers, per-task "
            "LoweringProfile/lowerer/variant factory) while the preserved v1 tasks keep theirs."
        ),
    }
    path = EVIDENCE / "retirement_inventory.json"
    if record or not path.exists():
        path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"OK: recorded retirement inventory ({len(live)} preserved profiles, {len(RETIRED_PROFILES)} retired)")
        return inventory
    archived = json.loads(path.read_text(encoding="utf-8"))
    for key, value in inventory.items():
        require(archived.get(key) == value, f"retirement_inventory.{key} drifted from the recorded evidence")
    print(f"OK: retirement inventory matches the record ({len(live)} preserved profiles, {len(RETIRED_PROFILES)} retired)")
    return inventory


# ---------------------------------------------------------------------------------------------
# Evidence isolation and the fresh v2 performance gate
# ---------------------------------------------------------------------------------------------


def verify_evidence_isolation(fresh: dict[str, dict]) -> None:
    """@brief A6 的 fresh 记录是本次现场测量，不是 A5 归档 v2 记录的拷贝。"""
    perf = load_perf()
    for size, records in fresh.items():
        archived = {
            record["task"]: record
            for record in json.loads((A5 / "evidence" / f"perf_v2_{size}.json").read_text(encoding="utf-8"))["records"]
        }
        for task, record in records.items():
            perf.validate_record(record, phase="v2", size=size, where=f"a6-fresh-{size}-{task}")
            old = archived[task]
            for key in (
                "candidate_sha256", "g0_body_sha256", "g0_opspec_digest", "g0_kernel_name",
                "g0_grid", "g0_threads", "baseline_implementation_digest", "target_id",
                "device_name", "gpu_family", "benchmark_rotation_sets", "dispatch", "provider",
                "effective_correctness_only", "baseline_registrations",
            ):
                require(record[key] == old[key], f"fresh {size}/{task}: {key} differs from the A5 archived v2 record")
            require(record["provider"] == "candidate", f"fresh {size}/{task}: not the candidate provider")
            require((record.get("track") or {}).get("track_id") == "msl-single", f"fresh {size}/{task}: wrong track")
    print("OK: A6 fresh v2 records carry the A5 tree identity and are measured by this run")


def verify_performance(fresh_enabled: bool, record: bool = False) -> dict:
    """@brief 现场测一轮 fresh v2 paper+small，折叠 A5 的 v1 中位数并按 1.10 门限判定。

    The measurement is written to a scratch tree under the repository (never over the pinned
    ``A6/evidence`` files); ``record`` copies it in, after which ``--write-manifest`` is needed.
    """
    perf = load_perf()
    archived_gate = json.loads((A5 / "evidence" / "perf_gate.json").read_text(encoding="utf-8"))
    require(archived_gate["approved_max_ratio"] == 1.10, "the approved gate is not 1.10")
    require(archived_gate["gate_pass"] is True and archived_gate["small_geomean"] <= 1.10, "the A5 archived gate was rewritten")
    require(len(archived_gate["comparable_tasks"]) == 13, "the A5 gate does not cover 13 comparable tasks")

    if not fresh_enabled:
        print("WARNING: the fresh v2 re-measurement was skipped; A6 is not accepted")
        return {}

    document = perf.index()
    require(len(document["combos"]) == 82, "the G0 index is not 82 combos")
    from tcenv.gpulock import gpu_lock

    scratch = Path(tempfile.mkdtemp(prefix=".a6-perf-", dir=ROOT))
    try:
        with gpu_lock("A6 verify fresh v2 re-measurement", exclusive=False):
            for size in ("paper", "small"):
                perf.run_v2(document, scratch / f"perf_v2_{size}_a6.json", only=(), size=size)
        fresh, gate = _fold_fresh(perf, scratch, archived_gate)
        if record:
            for size in ("paper", "small"):
                shutil.copy(scratch / f"perf_v2_{size}_a6.json", EVIDENCE)
            (EVIDENCE / "perf_gate_a6.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print("OK: recorded a fresh v2 round; re-run --write-manifest")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return fresh


def _fold_fresh(perf, scratch: Path, archived_gate: dict) -> tuple[dict, dict]:
    """@brief 把 scratch 里的 fresh 记录折叠成门限结果（不写证据）。"""
    fresh: dict[str, dict] = {}
    comparable = set(archived_gate["comparable_tasks"])
    medians: dict[str, dict[str, float]] = {}
    ratios: dict[str, dict[str, float]] = {}
    for size in ("paper", "small"):
        rounds = [
            {
                record["task"]: record
                for record in json.loads(
                    (A5 / "evidence" / f"perf_v1_{size}{'' if index == 1 else f'_r{index}'}.json").read_text(encoding="utf-8")
                )["records"]
            }
            for index in (1, 2, 3)
        ]
        v1_median = {task: statistics.median(round_records[task]["median_us"] for round_records in rounds) for task in MIGRATED}
        measured = json.loads((scratch / f"perf_v2_{size}_a6.json").read_text(encoding="utf-8"))
        require(measured["phase"] == "v2" and measured["size"] == size, f"fresh {size}: wrong phase/size")
        fresh[size] = {record["task"]: record for record in measured["records"]}
        require(set(fresh[size]) == set(MIGRATED), f"fresh {size}: not the 16 tasks")
        medians[size] = v1_median
        ratios[size] = {task: fresh[size][task]["median_us"] / v1_median[task] for task in comparable}

    verify_evidence_isolation(fresh)
    gate = {
        "schema_version": 1,
        "approved_max_ratio": 1.10,
        "fresh_phase": "v2",
        "comparison_basis": "A5 archived three-round v1 median (read-only, unchanged)",
        "source": {},
        "gate_instances": [],
    }
    for size in ("paper", "small"):
        geomean = math.exp(sum(math.log(value) for value in ratios[size].values()) / len(ratios[size]))
        worst_task = max(ratios[size], key=lambda task: ratios[size][task])
        gate["gate_instances"].append({
            "size": size,
            "comparable_tasks": sorted(ratios[size]),
            "fresh_median_us": {task: fresh[size][task]["median_us"] for task in MIGRATED},
            "v1_median_us": medians[size],
            "ratios": ratios[size],
            "geomean": geomean,
            "worst": {"task": worst_task, "ratio": ratios[size][worst_task]},
            "pass": geomean <= 1.10,
        })
        require(geomean <= 1.10, f"fresh {size} v2/v1 median geomean {geomean:.6f} > 1.10")
        gate["source"][size] = json.loads((scratch / f"perf_v2_{size}_a6.json").read_text(encoding="utf-8"))["source"]
        print(f"OK: fresh {size} v2/v1 median geomean {geomean:.6f} <= 1.10 ({len(ratios[size])} tasks; worst {worst_task} {ratios[size][worst_task]:.3f})")
    gate["gate_pass"] = all(instance["pass"] for instance in gate["gate_instances"])
    gate["correctness_only_tasks"] = list(CORRECTNESS_ONLY)
    verify_archived_gate(gate)
    return fresh, gate


def verify_archived_gate(fresh_gate: dict) -> None:
    """@brief 归档 A6 门与现场门同口径：同来源、同 13 题、通过、逐题比值在 2 倍以内。"""
    path = EVIDENCE / "perf_gate_a6.json"
    require(path.is_file(), "the A6 gate evidence is missing")
    archived = json.loads(path.read_text(encoding="utf-8"))
    require(archived["approved_max_ratio"] == 1.10, "the archived A6 gate is not 1.10")
    require(archived["gate_pass"] is True, "the archived A6 gate failed")
    require(archived["comparison_basis"] == fresh_gate["comparison_basis"], "the gate comparison basis drifted")
    require(archived["correctness_only_tasks"] == fresh_gate["correctness_only_tasks"], "correctness-only tasks drifted")
    for fresh_instance, old_instance in zip(fresh_gate["gate_instances"], archived["gate_instances"]):
        size = fresh_instance["size"]
        require(old_instance["size"] == size, "the archived gate instances are out of order")
        require(old_instance["comparable_tasks"] == fresh_instance["comparable_tasks"], f"{size}: comparable tasks drifted")
        require(old_instance["geomean"] <= 1.10 and old_instance["pass"] is True, f"{size}: the archived gate is above 1.10")
        require(set(old_instance["ratios"]) == set(fresh_instance["ratios"]), f"{size}: archived ratios are not the 13 tasks")
        for task, ratio in fresh_instance["ratios"].items():
            old = old_instance["ratios"][task]
            require(0.5 <= old / ratio <= 2.0, f"{size}/{task}: archived ratio {old:.3f} vs fresh {ratio:.3f} (not the same tree)")
        print(f"OK: archived {size} gate {old_instance['geomean']:.6f} vs fresh {fresh_instance['geomean']:.6f} "
              f"(same 13 tasks, both <= 1.10)")


def verify_slice_evidence() -> None:
    """@brief A6 GPU 记录是本阶段现场产生的真实派发/身份/计时证据。"""
    expected = {
        "gpu_v2_sigmoid.json": ("sigmoid", "msl-single"),
        "gpu_v1_rope.json": ("rope", "v1"),
    }
    for name, (task, track) in expected.items():
        path = EVIDENCE / name
        document = json.loads(path.read_text(encoding="utf-8"))
        require(document["task"] == task, f"{name}: wrong task")
        require(document["track"] == track, f"{name}: wrong track")
        require(document["status"] == "ok", f"{name}: the live run failed")
        require(document["correctness_ok"] is True, f"{name}: correctness failed")
        require(document["device_name"].startswith("Apple "), f"{name}: not a real Apple GPU")
        require(document["dispatch"], f"{name}: no real dispatch recorded")
        for record in document["dispatch"]:
            require(record["kernel"] and record["grid"] and record["threads"], f"{name}: incomplete dispatch")
        require(isinstance(document["median_us"], (int, float)) and document["median_us"] > 0, f"{name}: no real timing")
        require(len(document["identity_sha256"]) == 64, f"{name}: no identity")
        identity = {key: document[key] for key in sorted(document) if key not in ("identity_sha256", "median_us", "elapsed_s")}
        require(sha256_of_text(json.dumps(identity, sort_keys=True)) == document["identity_sha256"],
                f"{name}: the identity digest does not cover the record")
        if track == "msl-single":
            require(document["provider"] == "candidate", f"{name}: not the candidate provider")
            require(document["geometry_identity"] and document["bundle_digest"], f"{name}: no v2 identity")
            require(document["dispatch"][0]["threads"] == [256, 1, 1], f"{name}: candidate geometry changed")
        else:
            require(document["lowering_profile_digest"], f"{name}: no v1 profile identity")
    print("OK: A6 GPU records carry real dispatch/identity/timing")


# ---------------------------------------------------------------------------------------------
# Pytest phases
# ---------------------------------------------------------------------------------------------


def run_fixed_cpu() -> None:
    """@brief 运行 A6 固定 CPU 节点（retirement / preserved / one reader）。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *FIXED_CPU_NODE_IDS],
        where="fixed A6 CPU pytest",
    )
    print(f"OK: fixed A6 CPU pytest\n{summary}")


def run_noise_negatives() -> None:
    """@brief 独立噪声负例：噪声测量只能得到 timing_uncertain，端到端成功节点硬断言 ok。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *NOISE_NEGATIVE_NODE_IDS],
        where="noise/timing_uncertain negatives",
    )
    print(f"OK: confirmation negatives (timing_uncertain/batches_disagree stay rejections; the e2e node requires a hard ok)\n{summary}")


def run_v2_inventory_pytest() -> None:
    """@brief 重跑 A5 的分派/库存节点 + 82 组合的 G0 字节对照（证明 v2 16 题/82 组合仍有效）。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *V2_INVENTORY_NODE_IDS],
        where="fixed A5 inventory pytest",
        expect_passed=V2_INVENTORY_EXPECTED_PASSED,
    )
    print(f"OK: A5 v2 inventory + G0 pytest ({V2_INVENTORY_EXPECTED_PASSED} nodes, "
          f"{len(G0_MIGRATED_TASKS)} tasks / {G0_COMBINATIONS} combinations)\n{summary}")


def run_preserved_pytest() -> None:
    """@brief 重跑 preserved v1 的锚点：标识、registry 规则、动态族、装配结构。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *PRESERVED_NODE_IDS],
        where="preserved v1 pytest",
    )
    print(f"OK: preserved v1 pytest\n{summary}")


def verify_live_gpu_matches_archived(directory: Path) -> None:
    """@brief 现场 GPU 记录与归档逐字段一致（身份/派发/设备），计时只要求为正。"""
    for name in ("gpu_v2_sigmoid.json", "gpu_v1_rope.json"):
        live_path = directory / name
        require(live_path.is_file(), f"the GPU phase wrote no {name}")
        live = json.loads(live_path.read_text(encoding="utf-8"))
        archived = json.loads((EVIDENCE / name).read_text(encoding="utf-8"))
        for key in ("task", "track", "dtype", "size", "status", "provider", "correctness_ok",
                    "candidate_sha256", "bundle_digest", "geometry_identity", "native_digest",
                    "device_name", "gpu_family", "dispatch"):
            require(live[key] == archived[key], f"live {name}.{key} differs from the archived record")
        require(live["median_us"] > 0, f"live {name}: no timing")
    print("OK: live GPU records equal the archived A6 records")


def run_gpu_pytest(evidence_dir: Path) -> None:
    """@brief 运行 A6 的两条真实路径 + A5 的真实 v2 切片/A4 不变量。"""
    summary = _pytest(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *(A6_GPU_NODE_IDS + A5_GPU_NODE_IDS)],
        env={"A6_EVIDENCE_DIR": str(evidence_dir), "A5_EVIDENCE_DIR": str(evidence_dir / "a5")},
        where="real A6/A5 GPU slices",
    )
    print(f"OK: real A6/A5 GPU slices\n{summary}")


def self_test() -> None:
    """@brief 可独立重放的 A6 负例（不需要 GPU）。"""
    sys.path.insert(0, str(ROOT))
    from tasks import get_task
    from tcenv import lowering
    from tcenv.contract_errors import TaskContractError

    checks: list[str] = []
    for relative in DELETED_PATHS:
        try:
            importlib.import_module(relative.replace("/", ".").removesuffix(".py"))
        except ModuleNotFoundError:
            checks.append(f"the deleted module {relative} cannot be imported")
            break
    try:
        lowering.profile_for("metal", "metal-sigmoid")
    except lowering.LoweringError:
        checks.append("the retired sigmoid profile is not registered")
    else:
        raise AssertionError("the retired sigmoid profile is still registered")
    try:
        get_task("relu", "fp16", "small")
    except TaskContractError as error:
        require("v2 standard entry" in str(error), f"wrong refusal: {error}")
        checks.append("the v1 registry refuses a migrated name by spec_version")
    else:
        raise AssertionError("the v1 registry assembled a migrated name")
    task = get_task("rope", "fp16", "small")
    require(task.specialization is not None and task.naive_msl, "the preserved task did not assemble")
    checks.append("the preserved v1 task still assembles with its specialization")
    # The A5 v1 performance records are the comparison basis: a copied v2 record is not evidence.
    perf = load_perf()
    record = json.loads((A5 / "evidence" / "perf_v2_paper.json").read_text(encoding="utf-8"))["records"][0]
    copied = json.loads(json.dumps(record))
    copied["phase"] = "v2"
    try:
        perf.validate_record({**copied, "baselines": {}}, phase="v2", size="paper", where="a6-self-test")
    except perf.EvidenceError:
        checks.append("a v2 record without its measured baselines is refused")
    else:
        raise AssertionError("a v2 record without baselines was accepted")
    print(f"OK: self-test negatives ({len(checks)})")
    for name in checks:
        print(f"  - {name}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true", help="static/CPU only; not an A6 acceptance")
    parser.add_argument("--skip-fresh-perf", action="store_true", help="diagnostic only; not an A6 acceptance")
    parser.add_argument("--record-evidence", action="store_true", help="rewrite A6 evidence; then re-run --write-manifest")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verify_manifest()
    verify_historical()
    verify_retirement()
    verify_preserved_v1()
    verify_v2_inventory()
    verify_retirement_inventory(record=args.record_evidence)
    self_test()
    if not args.skip_pytest:
        run_fixed_cpu()
        run_noise_negatives()
        run_v2_inventory_pytest()
        run_preserved_pytest()
        if not args.skip_gpu:
            with tempfile.TemporaryDirectory(prefix="a6-evidence-") as temporary:
                directory = Path(temporary)
                run_gpu_pytest(directory)
                if args.record_evidence:
                    for name in ("gpu_v2_sigmoid.json", "gpu_v1_rope.json"):
                        require((directory / name).is_file(), f"the GPU phase wrote no {name}")
                        shutil.copy(directory / name, EVIDENCE / name)
                    print("OK: recorded the live GPU slices; re-run --write-manifest")
                else:
                    verify_live_gpu_matches_archived(directory)
    verify_performance(fresh_enabled=not args.skip_fresh_perf, record=args.record_evidence)
    verify_slice_evidence()
    if args.skip_gpu or args.skip_pytest or args.skip_fresh_perf:
        print("WARNING: a skipped phase is diagnostic only; A6 is not accepted")
    else:
        print("OK: A6 acceptance")


if __name__ == "__main__":
    main()
