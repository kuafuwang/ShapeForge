#!/usr/bin/env python3
"""P1 acceptance verifier: the real Mac shape-portfolio vertical slice.

P1 starts no model, no search, no ``taic run/resume`` and no paid experiment.  It proves the
*vertical slice* on the real device:

* the stage-start baseline (``acceptance-baseline.json`` + ``start-worktree.sha256``) is the real
  P0-completed worktree P1 started from: full HEAD, full worktree manifest, contract-surface
  SHA-256 and the Track digest, cross-checked against P0's independently pinned record — the
  current HEAD is never substituted for the start;
* the complete P1 manifest (dispatch runner, fixtures, CPU/GPU nodes, evidence, documents);
* the fixed CPU nodes (composition, every required negative, no build before the preflight, and the
  multi-stage expressiveness node) pass with an exact count and no skip;
* the archived evidence is not "checked against itself": every expected value is rebuilt from the fixed
  slice (named instance -> request, fixed plan/portfolio -> dispatch replay, raw fixed candidate sources
  + ``DomainAuthority`` -> selected projection, the request's own ABI -> geometry pair, the projected
  candidate + request contract -> Host Tiling plan), and the schema-4 build record is audited by
  ``verify_evaluator_build`` against that rebuilt canonical bundle.  A forger who edits several fields and
  recomputes every digest derived from them (artifact set, native, report, result, composition, and even
  the pair/bundle formats) therefore still disagrees with the reconstruction, and the build provenance,
  toolchain version, native descriptor, report conclusions, timing protocol and stage timeline are pinned
  to production/target facts rather than to sibling archive fields, and the measured evidence is checked
  by its production formulas (BenchStats from the samples, attempt noise, attempt bound, roofline from
  task+median+calibration, the fixed reference output and tolerance-normalized worst error), the
  compile log is reproduced byte for byte from the builder's template, and the host block and the
  refused requests are rebuilt from the recording host and the fixed slice; 189 default negatives
  cover it (147 coordinated plus four fully rebalanced measurement attacks);
* the fixed GPU nodes all execute (none skipped) and the archived evidence carries a real device,
  a schema-4 report, the actual dispatch, selection/dispatch parity, the projected
  candidate/geometry/artifact identity, the original geometry entry path + content SHA, and
  ``provider=fallback`` without variant credit;
* a live re-run of the same GPU nodes equals the archived evidence on every stable field.  The
  excluded set is explicit and small: timing fields, and the three digests that hash the schema-4
  report (``report_digest``, ``evaluation_identity``, ``run_identity``).  Those three are never left
  unchecked — they are part of the reconstructed sidecar each archive must equal, so a live run and
  the archive agree on them only after both were rebuilt from the fixed slice;
* P0's acceptance surface is re-verified to the extent this stage can honestly claim: the P0
  manifest (allowing only the declared ``dispatch.py`` extension), the P0 historical records, the
  36 fixed nodes and the 10 safe-regression files at an exact 70 passed.  P0's **14 contract
  self-checks are not part of this gate**: their baseline sub-check stages today's tree and demands
  P0's epoch baseline byte-for-byte, which the declared extension contradicts by construction, and
  P1 does not rewrite P0's verifier.  The A4 geometry anchors / A6 preserved-v1 anchors (exact 55)
  must not regress.

CPU and GPU are separate phases on purpose: ``--phase cpu`` / ``--phase gpu`` are diagnostic and
exit 2.  Only the default (both) run can report success, so a machine without a GPU can never be
mistaken for a passing P1.  Nothing is written unless ``--record-evidence`` / ``--write-manifest``
is passed explicitly.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[4]
P1 = Path(__file__).resolve().parent
MANIFEST = P1 / "manifest.sha256"
BASELINE_JSON = P1 / "acceptance-baseline.json"
BASELINE_MANIFEST = P1 / "acceptance-baseline.sha256"
START_MANIFEST = P1 / "start-worktree.sha256"
BASELINE_GENERATOR = P1 / "baseline_epoch.py"
EVIDENCE = P1 / "evidence"

RUNNER = "tcenv/taskcontract/portfolio_runner.py"
FIXTURE = "tests/support/shape_portfolio_metal.py"
CPU_TEST = "tests/shared/test_shape_portfolio_runner.py"
GPU_TEST = "tests/backends/metal/test_shape_portfolio_gpu.py"
P1_DOC = "docs/架构/shape-portfolio-阶段验收/P1.md"
PLAN = "docs/架构/shape-portfolio实施计划.md"
P1_TEST = "docs/架构/shape-portfolio-阶段验收/P1/verify.py"
P0_TEST = "docs/架构/shape-portfolio-阶段验收/P0/verify.py"
P0_BASELINE = "docs/架构/shape-portfolio-阶段验收/P0/acceptance-baseline.json"

#: Evidence files one recorded run writes; each is produced by exactly one fixed GPU node.
EVIDENCE_FILES: tuple[str, ...] = (
    "threshold.json",
    "boundaries.json",
    "single_source.json",
    "single_source_overflow.json",
    "out_of_domain.json",
    "fallback.json",
)

#: The baseline record's own bytes are pinned here: refreshing the baseline to hide drift is an
#: edit to this verifier (and to ``manifest.sha256``), never a silent re-baseline.
ACCEPTANCE_BASELINE_SHA256 = "481e9000a7438aec498358ebbb8ed2982186d9d676899cbaa6d53b502922fdb0"
ACCEPTANCE_BASELINE_EPOCH = "p1-real-slice-baseline-1"
ACCEPTANCE_BASELINE_HEAD = "4a93ed2a4934fe526eedfcf4577de5dbd366e24a"
ACCEPTANCE_BASELINE_FILE_COUNT = 79
#: sha256 over the newline-joined, sorted covered path list: dropping a covered file from the
#: surface changes this even when every remaining digest still matches.
ACCEPTANCE_BASELINE_SURFACE_SHA256 = "6cca6b8367d690e0639afded617e9a51a4712b2a8050f2e0d562bea1cad83500"
#: The only P1-start file P1 may change: the opt-in ``source_preflight`` extension in P0's dispatch
#: resolver.  Nothing else may drift, and this file is re-pinned by ``P1/manifest.sha256``.
P1_START_DRIFT_ALLOWED = ("tcenv/portfolio/dispatch.py",)

FIXED_CPU_NODE_IDS: tuple[str, ...] = (
    f"{CPU_TEST}::test_p1_dispatch_selects_exactly_one_provider_and_only_that_provider_is_built",
    f"{CPU_TEST}::test_p1_threshold_minus_one_threshold_and_plus_one_route_to_the_declared_buckets",
    f"{CPU_TEST}::test_p1_the_coverage_hole_runs_the_generic_fallback_and_the_odd_tail_stays_specialized",
    f"{CPU_TEST}::test_p1_the_second_boundary_keeps_the_other_single_source_geometry",
    f"{CPU_TEST}::test_p1_a_semantic_domain_outside_request_never_falls_back_and_never_builds",
    f"{CPU_TEST}::test_p1_a_resource_overflow_diverts_only_when_the_fallback_declares_the_reason",
    f"{CPU_TEST}::test_p1_a_fallback_resource_failure_never_chains_into_a_run",
    f"{CPU_TEST}::test_p1_a_single_source_launch_overflow_is_preflighted_and_only_diverts_when_declared",
    f"{CPU_TEST}::test_p1_two_active_stages_with_identical_signatures_are_both_executed",
    f"{CPU_TEST}::test_p1_provider_and_identity_swaps_are_refused",
    f"{CPU_TEST}::test_p1_a_stale_artifact_or_mislabelled_evaluation_is_refused",
    f"{CPU_TEST}::test_p1_a_wrong_grid_or_a_missing_host_run_in_the_execution_record_is_refused",
    f"{CPU_TEST}::test_p1_a_missing_or_illegal_host_tiling_and_a_missing_launch_are_refused_before_a_build",
    f"{CPU_TEST}::test_p1_the_lifecycle_binds_the_child_candidate_and_its_geometry",
    f"{CPU_TEST}::test_p1_the_two_variants_and_the_fallback_have_distinct_projected_identities",
    f"{CPU_TEST}::test_p1_the_composition_record_is_not_the_runtime_dispatch_record",
    f"{CPU_TEST}::test_p1_an_msl_single_projection_without_a_launch_adapter_is_refused_before_the_build",
    f"{CPU_TEST}::test_p1_a_build_free_evaluation_report_is_accepted_only_without_a_runtime_record",
)
FIXED_CPU_EXPECTED_PASSED = len(FIXED_CPU_NODE_IDS)

THRESHOLD_NODE = f"{GPU_TEST}::test_p1_gpu_threshold_variant_a_bundle_runs_host_tiling_and_a_real_dispatch"
BOUNDARIES_NODE = f"{GPU_TEST}::test_p1_gpu_threshold_minus_one_and_plus_one_switch_buckets"
SINGLE_NODE = f"{GPU_TEST}::test_p1_gpu_odd_tail_and_second_boundary_stay_in_the_single_source_variant"
OVERFLOW_NODE = f"{GPU_TEST}::test_p1_gpu_a_single_source_launch_overflow_runs_only_the_declared_fallback"
DOMAIN_NODE = f"{GPU_TEST}::test_p1_gpu_an_out_of_domain_request_is_refused_without_fallback_or_device_work"
FALLBACK_NODE = f"{GPU_TEST}::test_p1_gpu_coverage_hole_runs_the_generic_fallback_without_variant_credit"
FIXED_GPU_NODE_IDS: tuple[str, ...] = (
    THRESHOLD_NODE,
    BOUNDARIES_NODE,
    SINGLE_NODE,
    OVERFLOW_NODE,
    DOMAIN_NODE,
    FALLBACK_NODE,
)
FIXED_GPU_EXPECTED_PASSED = len(FIXED_GPU_NODE_IDS)

#: A4 geometry anchors P1 re-runs on the real device: read_write restore, declared-zero workspace
#: + Host Tiling inside timing, real dispatch vs fallback/changed grid, and the runtime refusing a
#: GeometryPair fallback against the selected candidate geometry.
A4_GEOMETRY_ANCHOR_NODE_IDS: tuple[str, ...] = (
    "tests/backends/metal/test_track_a4.py::test_a4_read_write_initial_values_are_restored_on_every_trial",
    "tests/backends/metal/test_track_a4.py::test_a4_declared_zero_workspace_is_cleared_inside_timing_and_host_tiling_restores_it",
    "tests/backends/metal/test_track_a4.py::test_a4_actual_runtime_dispatch_cannot_satisfy_fallback_or_a_changed_grid",
    "tests/backends/metal/test_track_a4.py::test_a4_real_runtime_refuses_a_fallback_geometry_against_candidate_selection",
)

#: A6 preserved-v1 anchors (the same file list A6's verifier re-runs).
A6_PRESERVED_FILES: tuple[str, ...] = (
    "tests/backends/metal/test_backend_phase0.py",
    "tests/shared/test_lowering_registry.py",
    "tests/shared/test_track_binding.py",
    "tests/shared/test_sem_int_family.py",
    "tests/shared/test_variant_family.py",
    "tests/shared/test_task_assembly_structure.py",
    "tests/shared/test_task_equivalence.py",
)

#: A4-A6 acceptance records P1 must leave byte-identical (the same nine P0 pins).  P0's own four
#: files are enforced by the P1 stage-start baseline and re-checked by P0's verifier below.
HISTORICAL_DIGESTS: dict[str, str] = {
    "docs/架构/跨硬件算子生成分工-阶段验收/A4.md": "ca26ca3e0320a8b2f35760ef02b91262c5eb1ab2c3f23bf33738c427a9604b20",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/verify.py": "389398e9bdb54bf132fedfdf06795b6e2e028f02a9affc8aa68c984f61dd3259",
    "docs/架构/跨硬件算子生成分工-阶段验收/A4/manifest.sha256": "4690a55e5fa8cd835c1ae364935864858ee96ecc36e6747efe148ef05b9b3bcc",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5.md": "f08e239c86c44a92b69f3e16e5d32d355ff871be8b94bbb8372ee59e037f172d",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/verify.py": "146c7d6e3e707244d46a49929625a7848d9f64cd0a8257d494dd7664e44e7b85",
    "docs/架构/跨硬件算子生成分工-阶段验收/A5/manifest.sha256": "b0668b8786952d1171fa51ae8d90fa0d50678c9ec87d484578a9a128a146d406",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6.md": "f22dfed7d9ba20b060bd7f2d44dbd90303e22c47d684d3b1faabe4fc951a4bb8",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/verify.py": "05a38ee168dcd633dd7e8841c4452547db07fc3123148189950f62e1fd1b9ba6",
    "docs/架构/跨硬件算子生成分工-阶段验收/A6/manifest.sha256": "ff8e5d9e136e9aa2e311732b28f5d35d17f8651213a3eabf8a51b7c7b96251f5",
}
HISTORICAL_EVIDENCE_ROOT = "docs/架构/跨硬件算子生成分工-阶段验收"
HISTORICAL_EVIDENCE_STAGES: tuple[str, ...] = ("A4", "A5", "A6")
TRACKS_PATH = "backends/metal/tracks.json"
TRACKS_SHA256 = "00c1f53c0a51f2a910c490519362e52330fabe9f68e4240b5b33c9490e567f93"
REGISTRY_DIGEST = "9d568eec90bab265cdc5d498a7f31baeb0f1406240617e028f9a5da4dd6044cb"
TRACK_DIGESTS = {
    "msl-single": "7d19a788cb2df0b25816b9f4200d9676f8d7eec317d2411ef0a09c1b8c703af8",
    "msl-bundle": "31b10ee5b33bc6d418106cb85154f6d14d04e08fb92d5382462dbe212610b9ba",
}
P0_FIXED_EXPECTED_PASSED = 36
#: P0's safe-regression files must stay at exactly this many passing nodes (a deleted or weakened
#: node that still leaves the file green is otherwise invisible), and their bytes are pinned below.
P0_REGRESSION_EXPECTED_PASSED = 70
#: A6's preserved-v1 anchors must stay at exactly this many passing nodes.
A6_PRESERVED_EXPECTED_PASSED = 55
#: Byte pins for the P0/A6 anchor test files.  They are not part of the P1 start surface (they are
#: pre-existing regression code), so their integrity is enforced here: deleting or weakening a node
#: changes a file, which fails before the count gate is even reached.
ANCHOR_FILE_DIGESTS: dict[str, str] = {
    "tests/backends/metal/test_backend_phase0.py": "296b5f66ecb2bbf31b469a39054367ed264b3d13dcc7f0926a236cad36e9067d",
    "tests/backends/metal/test_launch.py": "2a3e7fe1f09808ec197c9822efc96564d89a8ca944cb3b0727f17b24ad3eb76b",
    "tests/shared/test_cross_hardware_boundary_a2.py": "6c1692d04b8c01ec5acf89fd20619ef4bc891942f12f7783600b8f1faef3a9e1",
    "tests/shared/test_evaluate_task.py": "984fdf97a23015b1a74e4818c270685ec7bc71515c5b49cfeeca27308176d6df",
    "tests/shared/test_lowering_registry.py": "e605f1d9a1e2077004a63755ee18b13e478baaf333356ca8953b24d30ca45450",
    "tests/shared/test_sem_int_family.py": "07fc5ad0775dd75604ad233fac730c57b7286a62fd7f4bfa14cc4b93ad93bbf7",
    "tests/shared/test_structure.py": "501ec33c0b79c2a22c68eee0251091857d812f8cc7a43469df2f859b02976e08",
    "tests/shared/test_task_assembly_structure.py": "818bd3f23d61b35b1b7ba44c00292ee68807d1118f34c17ee9081736ad571a8d",
    "tests/shared/test_task_equivalence.py": "9376754013e4f0a7af55795bb274fc381d202cccdd86a1584b8c4a9effe06902",
    "tests/shared/test_track_abi.py": "96433d137f55d1d2d1e5c48ea6188a45a1e6140f40a40e86ea89abf92adf6f33",
    "tests/shared/test_track_binding.py": "6b125ba7485b8b81e279426e32607d569bf42df151f67113b353b80a4aff7953",
    "tests/shared/test_track_bundle.py": "45c8f9f7a64d1c5ce6e4043a171a1d6175d5ccf630515b57bd2bef67b4c72be1",
    "tests/shared/test_track_delivery.py": "cba46f62d2abe25506eaaff20a9f1cc447ac6f4bb5097ee5141fc1c9bb2d254b",
    "tests/shared/test_track_geometry.py": "01d104633df933e121652a5f34fbbfef3f71348480795bf3062c1c409c3f2770",
    "tests/shared/test_variant_family.py": "accb8bde162a3d6dcfa5cc9267de3b5807914d73612b0c58274b4b541642fcbc",
}
FORBIDDEN_RUNNER_TOKENS = (
    "DispatchManifest",
    "active_portfolio_pointer",
    "portfolio_run.json",
    "ValidationPlan",
    "PortfolioConfirmation",
)
#: The document-level ``slice`` each evidence file must carry (its own identity, not a free label).
EXPECTED_SLICES: dict[str, str] = {
    "threshold.json": "bundle-host-tiling",
    "boundaries.json": "threshold-minus-one-and-plus-one",
    "single_source.json": "odd-tail-and-second-boundary",
    "single_source_overflow.json": "single-source-launch-overflow",
    "out_of_domain.json": "semantic-domain-outside-requests",
    "fallback.json": "fallback-hole",
}

#: The fixed request point every evidence record must describe: dtype, usage and evaluator seed are
#: properties of the fixed slice (the GPU node), never fields read back from the archive.
P1_DTYPE = "fp16"
P1_USAGE = "golden"
FIXED_SEED = 7
#: Apple9 (M4 Max) SIMD width: the compiled pipeline's ``threadExecutionWidth`` is a device fact of
#: the pinned target, not something a report may choose.
EXPECTED_SIMD_WIDTH = 32
#: Per-case seed stride of the v2 runtime (``prepare_v2_case(contract, case, seed + 7919 * (index + 1))``).
CASE_SEED_STRIDE = 7919
#: The Metal source builder's registered native executable kind.
EXPECTED_NATIVE_KIND = "compute-pipeline"
#: The Metal builder's compiler identity (``descriptor.compiler``).
EXPECTED_COMPILER = "apple-metal-msl"
#: The v2 runtime correctness-coverage statement (production literal of ``evaluate.py``).
EXPECTED_COVERAGE_SCOPE = "real Metal reflection/read_write/workspace path"
#: The pinned epoch's MSL version.  It is a *toolchain* fact: ``MetalContext.language_version()``
#: probes the highest MSL the compiler accepts (4.0 first) on this machine, and the full default run
#: requires the real probe to equal this constant (``verify_toolchain_probe``), so it is never taken
#: from the report or from another archive field.
EXPECTED_MSL_VERSION = "4.0"
#: The fixed GPU slice's timing repetitions.  Every fixed node runs ``MetalTrackEvaluator(reps=3)``
#: (the default in ``tests/backends/metal/test_shape_portfolio_gpu.py``, threshold passes 3
#: explicitly), so the repetition count is a slice fact, not something the report may choose.
EXPECTED_REPS = 3
#: ``tcenv.bench.choose_iters`` bounds the batch size, and the bound is a protocol fact rather than
#: a free number: ``kmax`` (4000) caps it, a batch must stay inside ``BATCH_WALL_BUDGET_S`` (1.6 s),
#: and a kernel slower than 0.1 s per launch caps the batch at ``kmin`` instead of growing it.
#: ``interleaved`` records per-launch samples (``runner(k) / k``), so ``iters * min_s`` is the batch's
#: wall clock and the three rules are checkable from the report.  They are cross-checked against the
#: production function by ``verify_bench_constants`` so a copied constant cannot drift.
BENCH_MAX_ITERS = 4000
BENCH_SLOW_LAUNCH_S = 0.1
BENCH_TARGET_S = 0.015
BENCH_BATCH_WALL_BUDGET_S = 1.6
#: Closed key sets of the report's compile/benchmark blocks and of the build provenance.  A block has
#: exactly these fields; an unlisted field would be a claim no check covers.
COMPILE_BLOCK_KEYS = frozenset({"ok", "compile_s", "pipeline_s", "language_version", "math_mode", "launch"})
BENCHMARK_BLOCK_KEYS = frozenset(
    {
        "candidate",
        "candidate_gpu_us_median",
        "baselines",
        "candidate_serial_us",
        "measure_serial",
        "dispatch",
        "rotation_sets",
        "noise_cv",
        "attempts",
        "noisy",
        "gpu_lock_wait_s",
        "protocol",
    }
)
BENCHMARK_STATS_KEYS = frozenset({"cv", "iters", "mean_s", "median_s", "median_us", "min_s", "reps", "samples_s"})
PROVENANCE_KEYS = frozenset(
    {
        "schema_id",
        "builder_id",
        "builder_version",
        "build_host",
        "build_profile",
        "toolchain",
        "compile_profile",
        "source_sha256",
        "artifact_set_digest",
    }
)
COMPILE_PROFILE_KEYS = frozenset({"language", "math", "device", "gpu_family", "unified_memory"})
TOOLCHAIN_KEYS = frozenset({"id", "version", "sdk_version"})
TIMELINE_ENTRY_KEYS = frozenset({"stage", "status", "duration_s"})
#: The schema-4 benchmark protocol facts of the v2 runtime (production literals of ``evaluate.py``).
EXPECTED_BENCHMARK_PROTOCOL = (
    "v2 real runtime, interleaved with the registered baselines; read_write restored per trial, "
    "only declared-zero workspace cleared in timing"
)
EXPECTED_BENCHMARK_DISPATCH = "v2 runtime recorder, concurrent encoder as MLX"
#: The harness stage timeline: every stage the fixed path enters, in order, with status ``ok``.
EXPECTED_TIMELINE_STAGES = {
    "quick": ("device_check", "screen", "compile", "abi", "correctness"),
    "full": ("device_check", "screen", "compile", "abi", "correctness", "bench"),
}
#: The evaluator's honest formal-verification statement for MSL/AIR (``report.verification``).  It is
#: pinned so the block cannot be rewritten into a verification claim while still reading "unsupported".
EXPECTED_VERIFICATION_MESSAGE = (
    "no formal equivalence checker exists for Metal Shading Language / AIR; eligibility rests on safety "
    "checks and randomized differential tests, which can find bugs but cannot prove their absence"
)

EXPECTED_EVIDENCE = (
    # (file, node, expected runs [(key, size, provider_label, fallback_reason, evaluator mode)], kind)
    ("threshold.json", THRESHOLD_NODE, (("", "threshold", "variant-a-bundle", None, "full"),), "standard"),
    (
        "boundaries.json",
        BOUNDARIES_NODE,
        (("below", "below", "variant-b-single", None, "quick"),
         ("above", "above", "variant-a-bundle", None, "quick")),
        "standard",
    ),
    (
        "single_source.json",
        SINGLE_NODE,
        (
            ("runs.tail", "tail", "variant-b-single", None, "quick"),
            ("runs.boundary", "boundary", "variant-b-single", None, "quick"),
            ("runs.large", "large", "variant-b-single", None, "quick"),
        ),
        "standard",
    ),
    (
        "single_source_overflow.json",
        OVERFLOW_NODE,
        (("run", "below", "fallback", "resource_unavailable", "full"),),
        "over_limit",
    ),
    ("out_of_domain.json", DOMAIN_NODE, (), "standard"),
    ("fallback.json", FALLBACK_NODE, (("", "hole", "fallback", "outside_candidate_coverage", "full"),), "standard"),
)


#: Closed key set of one archived successful run record (exactly what the fixed GPU node writes).
EVIDENCE_RUN_KEYS = frozenset(
    {
        "schema_id",
        "node",
        "slice",
        "size",
        "n",
        "instance_key",
        "provider_label",
        "record",
        "projection",
        "evaluation",
        "report",
        "run_identity",
        "compiled_candidate_sha256",
        "built_entry_paths",
        "device",
        "host",
    }
)
#: Closed key set of each evidence file: a group wrapper (``runs``/``run``/``below``+``above``), the
#: optional refusal block, and nothing else.  A run record that is the document itself also carries
#: ``target_id``.  An unlisted field would be an evidence claim no check covers, so it is refused.
EVIDENCE_DOCUMENT_KEYS: dict[str, frozenset[str]] = {
    "threshold.json": EVIDENCE_RUN_KEYS | {"target_id"},
    "boundaries.json": frozenset({"schema_id", "host", "target_id", "node", "slice", "below", "above"}),
    "single_source.json": frozenset({"schema_id", "host", "target_id", "node", "slice", "runs"}),
    "single_source_overflow.json": frozenset(
        {"schema_id", "host", "target_id", "node", "slice", "run", "refused"}
    ),
    "out_of_domain.json": frozenset({"schema_id", "host", "target_id", "node", "slice", "refused"}),
    "fallback.json": EVIDENCE_RUN_KEYS | {"target_id"},
}
#: Closed key set of a Metal native descriptor.  A multi-kernel build adds ``kernel_names`` and
#: ``kernels``; this fixed slice compiles exactly one kernel, so it must carry neither.
DESCRIPTOR_KEYS = frozenset(
    {
        "assembled_sha256",
        "backend",
        "bindings",
        "compiler",
        "device",
        "execution_geometry",
        "execution_width",
        "gpu_family",
        "header_lines",
        "kernel_name",
        "language_version",
        "math",
        "max_total_threads",
        "static_tg_bytes",
    }
)
#: Closed key sets of the schema-4 report blocks that carry the run's own conclusions.
ABI_BLOCK_KEYS = frozenset({"ok", "violations"})
SAFETY_BLOCK_KEYS = frozenset({"ok", "violations", "nondeterminism", "validation"})
CORRECTNESS_BLOCK_KEYS = frozenset({"ok", "cases", "tolerance", "baseline_errors", "coverage"})
CORRECTNESS_CASE_KEYS = frozenset(
    {
        "allow",
        "baseline_scaled_err",
        "case",
        "input_sha256",
        "message",
        "n_bad",
        "n_elems",
        "nan_count",
        "ok",
        "safety",
        "scaled_err",
        "seed",
        "unwritten",
        "worst",
    }
)
CORRECTNESS_WORST_KEYS = frozenset({"got", "index", "output", "scaled_err", "want"})
COVERAGE_KEYS = frozenset({"kind", "scope"})
TOLERANCE_KEYS = frozenset({"rtol", "atol_rel", "exact"})
VERIFICATION_KEYS = frozenset({"status", "counterexample", "message"})
#: Closed key set of one run record's host identity block (``platform`` facts of the recording host).
HOST_BLOCK_KEYS = frozenset({"chip", "machine", "system"})
#: The run-group keys of each grouped evidence file (a renamed or duplicated group is refused).
EVIDENCE_GROUP_KEYS: dict[str, frozenset[str]] = {
    "single_source.json": frozenset({"tail", "boundary", "large"}),
    "boundaries.json": frozenset({"below", "above"}),
}


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

    @param path 文件路径。
    @return 十六进制摘要。
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: Any) -> str:
    """@brief 仓库同一实现的 canonical 摘要。

    @param value 文档。
    @return 十六进制 sha256。
    """
    from tcenv.identity import fingerprint as _fingerprint

    return _fingerprint(value)


def _strict_json_equal(left: Any, right: Any) -> bool:
    """@brief 递归比较 JSON 值，标量类型也必须相同。

    Python's ordinary equality intentionally makes ``True == 1`` and ``False == 0.0``.  Evidence
    types are part of the claim, so archive reconstruction and stable live comparison cannot use
    that coercion: objects/arrays recurse and every scalar requires identical Python JSON types.
    """
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_strict_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
            return False
        return all(_strict_json_equal(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


_PASSED_COUNT = re.compile(r"(\d+) passed\b")


def _passed_count(summary: str, *, where: str) -> int:
    """@brief 从 pytest summary 行解析整数 passed 数（整数相等，不做子串匹配）。

    @param summary pytest 最后一行摘要。
    @param where 诊断位置。
    @return passed 数。
    """
    matches = _PASSED_COUNT.findall(summary)
    require(len(matches) == 1, f"{where}: expected exactly one '<n> passed' count in {summary!r}")
    return int(matches[0])


def _pytest(
    command: list[str],
    *,
    where: str,
    expect_passed: int | None = None,
    env: Mapping[str, str] | None = None,
    allow_deselected: bool = False,
) -> str:
    """@brief 运行一次 pytest 并校验无失败/无跳过/精确通过数。

    @param command 命令。
    @param where 诊断位置。
    @param expect_passed 精确通过数。
    @param env 额外环境变量。
    @param allow_deselected 是否允许 deselect（文件级回归）。
    @return pytest 摘要行。
    """
    environment = os.environ.copy()
    environment.setdefault("PYTHONPATH", str(ROOT))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.update(dict(env or {}))
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=environment)
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"{where} failed:\n{output[-8000:]}")
    summary = [line for line in output.splitlines() if re.search(r"\b(passed|failed|error)\b", line)]
    require(summary, f"{where}: no pytest summary line")
    last = summary[-1]
    require(" skipped" not in last and "xfailed" not in last, f"{where} skipped a fixed node: {last}")
    require(" no tests ran" not in last, f"{where} collected nothing: {last}")
    if not allow_deselected:
        require(" deselected" not in last, f"{where} deselected a selected node: {last}")
    if expect_passed is not None:
        passed = _passed_count(last, where=where)
        require(passed == expect_passed, f"{where}: expected exactly {expect_passed} passed, got {passed} ({last})")
    return last


def _module(path: Path, name: str) -> object:
    """@brief 由路径加载一个模块。

    @param path 文件路径。
    @param name 模块名。
    @return 模块。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"{path}: cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------


def deliverables() -> tuple[str, ...]:
    """@brief P1 交付清单（固定文件 + 六个证据文件）。

    @return 路径元组。
    """
    evidence = tuple(str((EVIDENCE / name).relative_to(ROOT)) for name in EVIDENCE_FILES)
    return (
        RUNNER,
        # The single P0 file P1 extends (opt-in source-launch preflight), re-pinned here because the
        # stage-start baseline records its P1-start bytes.
        "tcenv/portfolio/dispatch.py",
        FIXTURE,
        CPU_TEST,
        GPU_TEST,
        P1_DOC,
        PLAN,
        str(BASELINE_GENERATOR.relative_to(ROOT)),
        str(BASELINE_JSON.relative_to(ROOT)),
        str(BASELINE_MANIFEST.relative_to(ROOT)),
        str(START_MANIFEST.relative_to(ROOT)),
        P1_TEST,
        *evidence,
    )


def write_manifest() -> None:
    """@brief 重写 P1 manifest.sha256（要求证据已录制）。

    @raises SystemExit 证据文件缺失。
    """
    missing = [name for name in EVIDENCE_FILES if not (EVIDENCE / name).is_file()]
    if missing:
        raise SystemExit(f"evidence is missing ({missing}); run `--record-evidence` first, then `--write-manifest`")
    lines = [f"{sha256(ROOT / name)}  {name}" for name in deliverables()]
    MANIFEST.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(lines)} files)")


def verify_manifest() -> None:
    """@brief 校验 manifest 的摘要与文件集合完全一致。

    @raises AssertionError 缺失/多余/漂移。
    """
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    seen: dict[str, str] = {}
    for line in lines:
        digest, name = line.split("  ", 1)
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"bad digest: {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen[name] = digest
    expected = set(deliverables())
    require(
        set(seen) == expected,
        f"manifest file set differs: missing={sorted(expected - set(seen))} extra={sorted(set(seen) - expected)}",
    )
    for name, digest in seen.items():
        path = ROOT / name
        require(path.is_file(), f"manifest file is missing: {name}")
        require(sha256(path) == digest, f"manifest mismatch: {name}")
    require(P1_TEST in seen, "the verifier itself is not pinned by the manifest")
    print(f"OK: manifest.sha256 ({len(seen)} files)")


# ---------------------------------------------------------------------------------------------
# Stage-start baseline
# ---------------------------------------------------------------------------------------------


def _parse_sha256_manifest(path: Path) -> dict[str, str]:
    """@brief 解析 ``<sha>  <path>`` 清单（忽略 ``#`` 说明头）。

    @param path 清单路径。
    @return 路径 → 摘要。
    """
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record or record.startswith("#"):
            continue
        parts = record.split("  ", 1)
        require(len(parts) == 2, f"{path}: malformed manifest line {record!r}")
        digest, name = parts
        require(bool(re.fullmatch(r"[0-9a-f]{64}", digest)), f"{path}: bad digest for {name}")
        require(name not in entries, f"{path}: duplicate entry {name}")
        entries[name] = digest
    return entries


def _git_commit_exists(head: str) -> bool | None:
    """@brief 记录中的 HEAD 是否解析到一个提交（git 不可用时返回 None）。

    @param head 提交 id。
    @return True/False/None。
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
    except OSError:  # pragma: no cover - git is present in this worktree
        return None
    return result.returncode == 0 and result.stdout.strip() == head


def verify_acceptance_baseline() -> None:
    """@brief P1 起点基线：真实 HEAD、完整工作树 manifest、合同面与 Track 摘要未漂移。

    @raises AssertionError 记录被重生、标注不对、或合同面漂移。
    """
    require(BASELINE_JSON.is_file(), "the P1 stage-start baseline is missing")
    require(
        sha256(BASELINE_JSON) == ACCEPTANCE_BASELINE_SHA256,
        "the P1 stage-start baseline was regenerated (its bytes differ from the pinned digest); only a new epoch "
        "plus a full re-acceptance may change that pin",
    )
    record = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    require(record.get("schema_id") == "tcenv.p1.acceptance-baseline.1", "unexpected baseline schema_id")
    require(record.get("epoch") == ACCEPTANCE_BASELINE_EPOCH, "the baseline epoch name changed")
    require(record.get("baseline_kind") == "acceptance-stage-start", "the record is not a stage-start baseline")
    require(record.get("stage") == "P1", "the record is not a P1 baseline")
    require(record.get("is_p1_start_baseline") is True, "the record does not claim to be the real P1 start")
    note = str(record.get("note") or "").lower()
    require("real start of the p1" in note, "the record does not state that it is the real P1 start")
    require("never be used to hide code drift" in note, "the record does not forbid hiding drift")
    head = record.get("head")
    require(isinstance(head, str) and bool(re.fullmatch(r"[0-9a-f]{40}", head)), "the baseline HEAD is not a commit")
    require(head == ACCEPTANCE_BASELINE_HEAD, "the baseline records a different HEAD than the pinned one")
    require(_git_commit_exists(head) is not False, "the recorded HEAD does not resolve to a commit")
    status = record.get("git_status") or {}
    entries = status.get("entries")
    codes = status.get("codes") or {}
    require(isinstance(entries, int) and entries > 0, "the baseline records no git status entries")
    require(isinstance(codes, dict) and sum(codes.values()) == entries, "the baseline git status counts do not add up")
    require(
        bool(re.fullmatch(r"[0-9a-f]{64}", str(status.get("porcelain_z_sha256") or ""))),
        "the baseline records no git status digest",
    )
    tracks = record.get("tracks") or {}
    require(tracks.get("path") == TRACKS_PATH, "the baseline does not pin backends/metal/tracks.json")
    require(tracks.get("sha256") == TRACKS_SHA256, "the baseline tracks.json digest is not the frozen one")
    require(sha256(ROOT / TRACKS_PATH) == tracks["sha256"], "backends/metal/tracks.json drifted from the baseline")
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    require(
        fingerprint(registry.document()) == tracks.get("registry_digest") == REGISTRY_DIGEST,
        "the Track registry digest drifted from the baseline",
    )
    digests = {track.track_id: fingerprint(track.document()) for track in registry.tracks}
    require(digests == tracks.get("track_digests"), f"a Track digest drifted from the baseline: {digests}")
    require(digests == TRACK_DIGESTS, "a Track digest drifted from the frozen A4-A6 value")

    generator = _module(BASELINE_GENERATOR, "p1_baseline_epoch")
    surface = tuple(generator.CONTRACT_SURFACE)
    files = record.get("files")
    require(isinstance(files, dict) and bool(files), "the baseline records no file digests")
    require(
        len(files) == ACCEPTANCE_BASELINE_FILE_COUNT,
        f"the baseline covers {len(files)} files, expected {ACCEPTANCE_BASELINE_FILE_COUNT}",
    )
    require(tuple(sorted(files)) == tuple(sorted(surface)), "the baseline file set differs from the generator surface")
    require(
        hashlib.sha256("\n".join(sorted(files)).encode()).hexdigest() == ACCEPTANCE_BASELINE_SURFACE_SHA256,
        "the baseline covers a different path list than the pinned surface",
    )
    groups = record["contract_surface"]["groups"]
    require(set(groups) == set(generator.GROUPS), f"baseline groups differ: {sorted(groups)}")
    for name, paths in groups.items():
        require(bool(paths), f"baseline group {name} is empty")
        require(len(set(paths)) == len(paths), f"baseline group {name} repeats a path")
        require(set(paths) <= set(files), f"baseline group {name} lists an uncovered file")
    require(
        set().union(*(set(paths) for paths in groups.values())) == set(files) - {TRACKS_PATH},
        "the baseline groups do not add up to the covered file set",
    )
    require(TRACKS_PATH in files, "the baseline does not cover backends/metal/tracks.json")
    # P1 extends exactly one P0 file (the opt-in source-launch preflight in ``dispatch.py``).  Every
    # other covered file must still be byte-identical to the P1 start; the extension is re-pinned by
    # this stage's manifest and must keep P0's frozen dispatch nodes green.
    drifted = sorted(name for name, digest in files.items() if sha256(ROOT / name) != digest)
    require(
        set(drifted) <= set(P1_START_DRIFT_ALLOWED),
        f"the contract surface drifted from the P1 stage start: {sorted(set(drifted) - set(P1_START_DRIFT_ALLOWED))}",
    )
    for name in drifted:
        require(name in deliverables(), f"{name}: the allowed drift is not re-pinned by this stage's manifest")
    if drifted:
        print(f"OK: only {drifted} changed since the P1 start (declared P1 extension, re-pinned here)")
    require(RUNNER not in files, "the baseline claims to cover the P1 runner, which did not exist at the P1 start")
    require(FIXTURE not in files, "the baseline claims to cover the P1 fixture, which did not exist at the P1 start")
    for name in files:
        require((ROOT / name).is_file(), f"baseline-covered file is missing: {name}")
    manifest = _parse_sha256_manifest(BASELINE_MANIFEST)
    require(manifest == files, f"the baseline manifest disagrees with the record: {sorted(set(manifest) ^ set(files))}")
    require(sha256(BASELINE_MANIFEST) == record["contract_surface"]["manifest_sha256"], "the baseline manifest changed")

    worktree = record.get("worktree") or {}
    require(START_MANIFEST.is_file(), "the full start-worktree manifest is missing")
    require(sha256(START_MANIFEST) == worktree.get("manifest_sha256"), "the full start-worktree manifest changed")
    start_entries = _parse_sha256_manifest(START_MANIFEST)
    require(len(start_entries) == worktree.get("files"), "the full worktree manifest lost entries")
    mismatched = sorted(name for name, digest in files.items() if start_entries.get(name) != digest)
    require(not mismatched, f"the full start-worktree manifest disagrees with the contract surface: {mismatched}")
    require(
        set(worktree.get("excluded_dirs") or ()) >= {"runs", "data", ".venvs"},
        "the full worktree manifest does not name its excluded generated trees",
    )

    # Independent corroboration: P0's record is pinned by P0/verify.py, and every file shared with
    # it must still carry the P0-pinned bytes.  A baseline invented today cannot pass this.
    p0_module = _module(ROOT / P0_TEST, "p0_verify")
    p0_baseline = ROOT / P0_BASELINE
    require(sha256(p0_baseline) == p0_module.ACCEPTANCE_BASELINE_SHA256, "the P0 acceptance record was regenerated")
    p0_files = json.loads(p0_baseline.read_text(encoding="utf-8"))["files"]
    shared = sorted(set(files) & set(p0_files))
    require(len(shared) >= 44, f"only {len(shared)} files are shared with P0's record; the baseline is not corroborated")
    drift = [name for name in shared if files[name] != p0_files[name]]
    require(not drift, f"the P1 baseline disagrees with the P0-pinned bytes: {drift}")
    cross = record.get("p0_cross_check") or {}
    require(cross.get("record_sha256") == sha256(p0_baseline), "the record's P0 cross-check names a different P0 record")
    require(cross.get("mismatches") == [], "the record reports P0 cross-check mismatches")
    print(
        f"OK: P1 stage-start baseline {ACCEPTANCE_BASELINE_EPOCH} ({len(files)} files, {len(start_entries)} worktree files), "
        f"HEAD {ACCEPTANCE_BASELINE_HEAD}"
    )
    print(
        f"OK: contract surface + tracks.json + registry/Track digests unchanged since the P1 start; {len(shared)} files "
        "reproduce the P0-pinned bytes"
    )


# ---------------------------------------------------------------------------------------------
# Historical records
# ---------------------------------------------------------------------------------------------


def verify_historical() -> None:
    """@brief A4-A6/P0 记录、冻结 evidence 与 Track/registry digest 未变。

    @raises AssertionError 任一历史记录漂移。
    """
    for name, digest in HISTORICAL_DIGESTS.items():
        path = ROOT / name
        require(path.is_file(), f"historical record is missing: {name}")
        require(sha256(path) == digest, f"historical record changed: {name}")
    # P0's own verifier re-checks its manifest, its acceptance-epoch baseline and the historical
    # evidence; P0's four files are also part of the enforced P1 stage-start surface above.
    verify_p0_manifest()
    _module(ROOT / P0_TEST, "p0_verify").verify_historical()
    checked = 0
    for stage in HISTORICAL_EVIDENCE_STAGES:
        manifest = ROOT / HISTORICAL_EVIDENCE_ROOT / stage / "manifest.sha256"
        require(manifest.is_file(), f"the {stage} manifest is missing: {manifest}")
        prefix = f"{HISTORICAL_EVIDENCE_ROOT}/{stage}/evidence/"
        for line in manifest.read_text(encoding="utf-8").splitlines():
            record = line.strip()
            if not record:
                continue
            parts = record.split("  ", 1)
            require(len(parts) == 2, f"the {stage} manifest has a malformed line: {record!r}")
            digest, name = parts
            if not name.startswith(prefix):
                continue
            require((ROOT / name).is_file(), f"historical evidence is missing: {name}")
            require(sha256(ROOT / name) == digest, f"historical evidence changed: {name}")
            checked += 1
    require(checked > 0, "no historical evidence files were checked")
    require(sha256(ROOT / TRACKS_PATH) == TRACKS_SHA256, "backends/metal/tracks.json changed")
    from tcenv.track.registry import load_track_registry

    registry = load_track_registry()
    require(fingerprint(registry.document()) == REGISTRY_DIGEST, "the Track registry digest changed")
    require({track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}, "a Track was added/removed")
    for track in registry.tracks:
        require(fingerprint(track.document()) == TRACK_DIGESTS[track.track_id], f"track {track.track_id} digest changed")
    print(
        f"OK: A4-A6 records untouched ({checked} frozen historical evidence files) and the "
        "P0 manifest/history/Track digests unchanged (P0's 14 contract self-checks are outside this gate)"
    )


# ---------------------------------------------------------------------------------------------
# Static composition boundary
# ---------------------------------------------------------------------------------------------


def verify_anchor_files() -> None:
    """@brief P0 safe-regression 与 A6 preserved-v1 锚点文件的字节必须未变。

    The exact pass counts below only bind if the nodes they count still exist; pinning the files
    makes "delete a node, keep the rest green" a hard failure instead of a silent weakening.

    @raises AssertionError 任一锚点测试文件被改动或删除。
    """
    missing = [name for name in ANCHOR_FILE_DIGESTS if not (ROOT / name).is_file()]
    require(not missing, f"an acceptance anchor test file is missing: {missing}")
    drifted = sorted(name for name, digest in ANCHOR_FILE_DIGESTS.items() if sha256(ROOT / name) != digest)
    require(not drifted, f"an acceptance anchor test file changed: {drifted}")
    print(f"OK: {len(ANCHOR_FILE_DIGESTS)} P0/A6 anchor test files unchanged (byte-pinned)")


def verify_baseline_tool() -> None:
    """@brief ``baseline_epoch.py`` 的独立复核必须与 P1 声明的唯一漂移规则一致。

    The tool is a second implementation of the stage-start comparison (it does not import the
    verifier).  On the current tree it must report exactly the declared ``dispatch.py`` extension
    and exit 0 — anything else means the two implementations disagree about the allowed drift.

    @raises AssertionError 工具退出码非 0，或报告的漂移不是那一个文件。
    """
    result = subprocess.run(
        [sys.executable, str(BASELINE_GENERATOR)], cwd=ROOT, capture_output=True, text=True
    )
    output = (result.stdout or "") + (result.stderr or "")
    require(result.returncode == 0, f"baseline_epoch.py refused the current tree:\n{output[-4000:]}")
    declared = [name for name in P1_START_DRIFT_ALLOWED if f"declared P1 extension" in output and name in output]
    require(
        len(declared) == len(P1_START_DRIFT_ALLOWED),
        f"baseline_epoch.py did not report the declared drift {list(P1_START_DRIFT_ALLOWED)}\n{output[-2000:]}",
    )
    print("OK: baseline_epoch.py independently reproduces the stage-start baseline (declared drift only)")


def verify_count_gates() -> None:
    """@brief 精确通过数门的负例：错误数量必须让门失败。

    @raises AssertionError 计数门接受了错误数量。
    """
    with tempfile.TemporaryDirectory(prefix="p1-count-gate-") as temporary:
        path = Path(temporary) / "test_count_gate.py"
        path.write_text("def test_one():\n    assert True\n", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            str(path),
        ]
        _pytest(command, where="count gate (correct)", expect_passed=1)
        try:
            _pytest(command, where="count gate (wrong)", expect_passed=2)
        except AssertionError:
            pass
        else:  # pragma: no cover - the gate would be broken
            raise AssertionError("the exact-count gate accepted a wrong pass count")
    print("OK: exact pass-count gate (a wrong count is refused)")


def verify_runner_static() -> None:
    """@brief 组合模块的静态边界：注入适配器、无 backend import、无激活/清单概念、记录类型不同。

    @raises AssertionError 违反边界。
    """
    tree = ast.parse((ROOT / RUNNER).read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.append(node.module)
    require(
        not any(name == "tcenv.backends" or name.startswith("tcenv.backends.") for name in imported),
        f"{RUNNER} imports a backend at module level; the adapters must be injected: {imported}",
    )
    require(not any(name.startswith("taic") for name in imported), f"{RUNNER} imports the search/CLI layer: {imported}")
    # Names, attributes and string literals: prose may *describe* the forbidden concepts, code may
    # not create them.
    identifiers: set[str] = set()
    literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.alias):
            identifiers.add(node.asname or node.name.split(".")[-1])
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            literals.append(node.value)
    require(
        not (identifiers & set(FORBIDDEN_RUNNER_TOKENS)),
        f"{RUNNER} builds a forbidden active concept: {sorted(identifiers & set(FORBIDDEN_RUNNER_TOKENS))}",
    )
    require(
        not any(
            token in literal
            for token in FORBIDDEN_RUNNER_TOKENS
            for literal in literals
            if literal.strip().endswith(".json")
        ),
        f"{RUNNER} writes a forbidden sidecar name",
    )
    text = (ROOT / RUNNER).read_text(encoding="utf-8")
    require("fallback_geometry=None" in text, "the runner must pass fallback_geometry=None to evaluate_task")
    require("resolve_geometry(" in text, "the runner must resolve the selected geometry on the CPU")
    require(
        "geometry_resolver_factory" in text,
        "the runner must take a per-instance resolver factory, so single-source geometry cannot be built for another size",
    )
    fixtures = ast.parse((ROOT / FIXTURE).read_text(encoding="utf-8"))
    fixture_imports: list[str] = []
    for node in ast.walk(fixtures):
        if isinstance(node, ast.Import):
            fixture_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            fixture_imports.append(node.module)
    require(
        not any(name.startswith("tcenv.backends") for name in fixture_imports),
        f"{FIXTURE} imports a backend; the fixture only holds literal candidate documents: {fixture_imports}",
    )
    from tcenv.backends.metal.v2_runtime import DispatchRecord as RuntimeDispatchRecord
    from tcenv.portfolio.dispatch import DISPATCH_RECORD_SCHEMA_ID, PortfolioDispatchRecord

    require(PortfolioDispatchRecord is not RuntimeDispatchRecord, "the composition record is the runtime dispatch record")
    require(PortfolioDispatchRecord.__name__ != RuntimeDispatchRecord.__name__, "the two records share a name")
    require(
        DISPATCH_RECORD_SCHEMA_ID == "tcenv.portfolio.dispatch-record.1",
        "the composition record schema id changed",
    )
    print("OK: runner/fixture boundary (injected adapters, no backend import, no active manifest, distinct record type)")


# ---------------------------------------------------------------------------------------------
# CPU phases
# ---------------------------------------------------------------------------------------------


class FixtureContext:
    """@brief 重建的 P1 有限切片上下文（题、权威实例、域计划、portfolio、dispatch 解析器）。"""

    def __init__(
        self, contract, instances, plan, portfolio, resolver, profile, limits, locator, factory, contracts, keeper
    ) -> None:
        """@brief 保存上下文。

        @param contract 题合同。
        @param instances 权威实例。
        @param plan 域计划。
        @param portfolio 已解析 portfolio。
        @param resolver exact dispatch 解析器（source-aware 预检已开启）。
        @param profile 目标 profile。
        @param limits ABI 限制。
        @param locator 默认实例 locator（用于派生请求实例的合同）。
        @param factory ``msl-single`` launch 解析器工厂（真实 Metal 实现）。
        @param contracts ``(dtype, size)`` → 该实例的合同。
        @param keeper 持有题目录的 ``TemporaryDirectory``（合同/渲染身份需要题目录仍在）。
        """
        self.contract = contract
        self.instances = instances
        self.plan = plan
        self.portfolio = portfolio
        self.resolver = resolver
        self.profile = profile
        self.limits = limits
        self.locator = locator
        self.factory = factory
        self.contracts = dict(contracts)
        self._keeper = keeper

    def contract_for(self, dtype: str, size: str):
        """@brief 请求实例的合同（生命周期/cache scope 身份的来源）。

        @param dtype dtype。
        @param size size。
        @return TaskContract。
        """
        key = (dtype, size)
        contract = self.contracts.get(key)
        require(contract is not None, f"the fixture has no contract for {key}")
        return contract

    def _authority(self):
        """@brief 投影权威。

        @return DomainAuthority。
        """
        from tcenv.portfolio.domain import DomainAuthority

        return DomainAuthority(
            plan=self.plan,
            spec=self.contract.spec,
            target=self.profile,
            instances=self.instances,
            contract=self.contract,
        )

    def instance_for(self, request):
        """@brief 由请求的 dtype/size 重建实例与 ABI（不做任何编译）。

        @param request dispatch 请求。
        @return (instance, abi)。
        """
        from tcenv.opspec.v2 import instantiate_v2
        from tcenv.track.abi import derive_tensor_memory_contract

        instance = instantiate_v2(self.contract.spec, request.dtype, request.size)
        return instance, derive_tensor_memory_contract(instance, target=self.profile)

    def with_portfolio(self, portfolio) -> "FixtureContext":
        """@brief 同一域上的另一份 portfolio（例如超限 variant 的负例）上下文。

        @param portfolio 已解析 portfolio。
        @return 新上下文。
        """
        from tcenv.portfolio.dispatch import ExactDispatchResolver
        from tests.support.shape_portfolio_metal import evidence_all_valid

        resolver = ExactDispatchResolver(
            domain_plan=self.plan,
            portfolio=portfolio,
            evidence=evidence_all_valid(portfolio),
            spec=self.contract.spec,
            target=self.profile,
            instances=self.instances,
            source_preflight=True,
        )
        return FixtureContext(
            self.contract,
            self.instances,
            self.plan,
            portfolio,
            resolver,
            self.profile,
            self.limits,
            self.locator,
            self.factory,
            self.contracts,
            self._keeper,
        )

    def project(self, provider):
        """@brief 投影一个命名 provider（纯 CPU）。

        @param provider variant 或 fallback。
        @return CandidateProjection。
        """
        from tcenv.portfolio.candidate import project_provider

        return project_provider(self.portfolio, provider, authority=self._authority())

    def resolve(self, request):
        """@brief 用同一解析器重放一个请求（CPU 预检，不编译）。

        The hook is provider-aware, exactly like the runner's: a provider that freezes its geometry
        is preflighted with that geometry, and a provider that owns its launch in the source is
        preflighted with the geometry the **real Metal launch resolver** derives for this
        instance.  A replay therefore reproduces the archived preflight status (``ok`` for both
        Tracks, or ``resource_unavailable``) instead of trusting what the evidence claims.

        @param request dispatch 请求。
        @return 组合记录。
        """
        from tcenv.portfolio.dispatch import preflight_geometry

        known = {(item.dtype, item.size) for item in self.instances}
        if (request.dtype, request.size) not in known:
            # An out-of-domain point has no authoritative instance, so it has no preflight hook
            # either: dispatch must refuse it from the finite domain alone (never a fallback).
            return self.resolver.resolve(request)
        instance, abi = self.instance_for(request)
        contract = self.contract_for(request.dtype, request.size)
        launch = self.factory(contract)
        providers = {variant.variant_key: variant for variant in self.portfolio.variants}
        if self.portfolio.fallback is not None:
            providers[self.portfolio.fallback.fallback_id] = self.portfolio.fallback

        def hook(geometry, *, provider=None):
            """@brief 重放该 provider 的 CPU 预检。

            @param geometry 声明几何或 ``None``。
            @param provider provider 稳定键。
            @return GeometryPreflight。
            """
            chosen = geometry
            if chosen is None:
                target_provider = providers[str(provider)]
                projection = self.project(target_provider)
                if projection.track_id == "msl-bundle":
                    from tcenv.track.delivery import candidate_geometry

                    chosen = candidate_geometry(projection.candidate)
                else:
                    chosen = launch(projection.candidate)
            from tcenv.taskcontract.portfolio_runner import PREFLIGHT_WHERE

            return preflight_geometry(
                chosen, instance=instance, contract=abi, limits=self.limits, where=PREFLIGHT_WHERE
            )

        return self.resolver.resolve(request, preflight=hook)

    def label(self, record) -> str:
        """@brief 记录命中的 provider 标签。

        @param record 组合记录。
        @return ``variant_key`` 或 ``fallback``。
        """
        if record.provider == "fallback":
            return "fallback"
        return self.portfolio.variant_for_provider(record.variant_id).variant_key

    def projection(self, record):
        """@brief 投影选中者（纯 CPU）。

        @param record 组合记录。
        @return CandidateProjection。
        """
        from tcenv.portfolio.candidate import project_provider
        from tcenv.portfolio.domain import DomainAuthority

        provider = (
            self.portfolio.fallback
            if record.provider == "fallback"
            else self.portfolio.variant_for_provider(record.variant_id)
        )
        authority = DomainAuthority(
            plan=self.plan,
            spec=self.contract.spec,
            target=self.profile,
            instances=self.instances,
            contract=self.contract,
        )
        return project_provider(self.portfolio, provider, authority=authority)


def fixture_context(where: Path | None = None) -> FixtureContext:
    """@brief 载入固定 P1 题并重建 domain plan / portfolio / resolver（无编译）。

    The task directory is kept alive by the returned context: recomputing the request locator's
    deployment pack digest, the frozen presentation identity and the schema-4 evaluation
    fingerprint all read that pack again.

    @param where 忽略（保留参数以兼容调用方）；任务目录由上下文自行持有。
    @return 上下文。
    """
    del where
    sys.path.insert(0, str(ROOT))
    from tcenv.portfolio.candidate import parse_portfolio
    from tcenv.portfolio.dispatch import ExactDispatchResolver
    from tcenv.portfolio.domain import normalize_semantic_domain_plan
    from tcenv.target import DEFAULT_TARGET_ID, load_target
    from tcenv.taskcontract import discover_standard_tasks
    from tcenv.track import REPO_ROOT
    from tcenv.track.abi import abi_limits_from_target
    from tests.support.shape_portfolio_metal import (
        evidence_all_valid,
        load_p1_task,
        p1_instances,
        parent_entries,
        portfolio_document,
    )

    from tcenv.backends.metal.track_adapter import msl_single_geometry_resolver

    profile = load_target(DEFAULT_TARGET_ID)
    limits = abi_limits_from_target(profile)
    keeper = tempfile.TemporaryDirectory(prefix="p1-fixture-")
    root = Path(keeper.name) / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=profile)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    resolver = ExactDispatchResolver(
        domain_plan=plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=contract.spec,
        target=profile,
        instances=instances,
        source_preflight=True,
    )
    from tcenv.taskcontract.portfolio_runner import locator_for_instance

    contracts = {
        (point.dtype, point.size): locator_for_instance(locator, point.dtype, point.size).load()
        for point in plan.instances
    }
    return FixtureContext(
        contract,
        instances,
        plan,
        portfolio,
        resolver,
        profile,
        limits,
        locator,
        msl_single_geometry_resolver,
        contracts,
        keeper,
    )


#: P0's manifest pins two things P1 must legitimately change: the *living* stage plan (every stage
#: updates its own status) and ``tcenv/portfolio/dispatch.py`` (P1's opt-in ``source_preflight``
#: extension).  Both are re-pinned by this stage's manifest; every other P0 file must be
#: byte-identical.  There is no third entry, and the extension keeps P0's own frozen nodes green.
P0_DRIFT_ALLOWED = ("docs/架构/shape-portfolio实施计划.md", "tcenv/portfolio/dispatch.py")


def verify_p0_manifest() -> None:
    """@brief P0 manifest：除"活文档"计划外，P0 的每个文件摘要都不变。

    @raises AssertionError 计划以外的文件漂移，或计划本身未被 P1 manifest 重新钉住。
    """
    p0_module = _module(ROOT / P0_TEST, "p0_verify")
    p0_manifest_path = ROOT / "docs/架构/shape-portfolio-阶段验收/P0/manifest.sha256"
    entries = _parse_sha256_manifest(p0_manifest_path)
    require(set(entries) == set(p0_module.DELIVERABLES), "the P0 manifest file set changed")
    drifted = sorted(name for name, digest in entries.items() if sha256(ROOT / name) != digest)
    require(
        set(drifted) <= set(P0_DRIFT_ALLOWED),
        f"P0 manifest files changed outside the living plan: {sorted(set(drifted) - set(P0_DRIFT_ALLOWED))}",
    )
    for name in drifted:
        require(name in P0_DRIFT_ALLOWED, f"{name}: not a declared P1 drift from P0's manifest")
        require(name in deliverables(), "the drifted file is not re-pinned by this stage's manifest")
    print(
        f"OK: P0 manifest ({len(entries)} files; only the living plan and the declared dispatch.py extension moved, "
        "and P1's manifest pins their new bytes)"
        if drifted
        else f"OK: P0 manifest ({len(entries)} files, unchanged)"
    )


def verify_p0_epoch_extension() -> None:
    """@brief P0 的 epoch 合同面仍是封闭集，P1 只允许**一个**具名扩展：自己的 runner 测试。

    P0 的 epoch 规则把 ``tests/shared/test_shape_portfolio_*.py`` 当封闭集，因此新增文件必然会被它
    判为漂移。P1 不修改 P0 的验收器或记录来回避这一点，而是把这次具名扩展算清：覆盖集一个不少、
    新增集恰好只有本阶段声明的那个文件。

    @raises AssertionError P0 测试文件丢失或新增了未声明的文件。
    """
    p0_module = _module(ROOT / P0_TEST, "p0_verify")
    require(sha256(ROOT / P0_BASELINE) == p0_module.ACCEPTANCE_BASELINE_SHA256, "the P0 acceptance record changed")
    covered = {
        name
        for name in json.loads((ROOT / P0_BASELINE).read_text(encoding="utf-8"))["files"]
        if name.startswith("tests/shared/test_shape_portfolio_")
    }
    present = {
        f"tests/shared/{path.name}" for path in (ROOT / "tests/shared").glob("test_shape_portfolio_*.py")
    }
    require(not (covered - present), f"a P0 fixed test file disappeared: {sorted(covered - present)}")
    added = present - covered
    require(
        added == {CPU_TEST},
        f"the P0 closed test set may only gain this stage's declared runner test, gained {sorted(added)}",
    )
    print(f"OK: P0 epoch surface is intact and gained exactly {CPU_TEST} (declared P1 extension)")


def verify_fixture_routing(tmp_path: Path) -> None:
    """@brief 不编译任何 kernel 地重放 7 个有限实例的 dispatch，核对 fixture 的路由与分轨。

    @param tmp_path 临时目录。
    """
    from tcenv.opspec.v2 import instantiate_v2
    from tcenv.portfolio.dispatch import DispatchRequest
    from tests.support.shape_portfolio_metal import EXPECTED_PROVIDER, P1_SIZES

    context = fixture_context()
    for size in P1_SIZES:
        instance = instantiate_v2(context.contract.spec, "fp16", size)
        point = context.plan.find("fp16", size, "golden")
        require(point is not None, f"{size} is not a domain point")
        request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="golden")
        record = context.resolve(request)
        label = context.label(record)
        require(
            label == EXPECTED_PROVIDER[size],
            f"{size}: the fixture now routes to {label!r}, but the record says {EXPECTED_PROVIDER[size]!r}",
        )
        projection = context.projection(record)
        expected_track = "msl-bundle" if size in ("threshold", "above") else "msl-single"
        require(projection.track_id == expected_track, f"{size}: expected a {expected_track} projection")
        # Both Tracks are now preflighted on the CPU before any build: the bundle with its frozen
        # geometry entry, and the single source with the launch geometry the real Metal resolver
        # derives for this instance.
        require(record.preflight.status == "ok", f"{size}: the selected geometry was not preflighted")
        require(record.preflight.resolved is not None, f"{size}: the preflight carries no resolved geometry")
    print(f"OK: fixture routing replayed for {len(P1_SIZES)} finite instances (no kernel compiled)")


def run_p0_regression(tmp_path: Path) -> None:
    """@brief 重跑 P0 的验收：manifest（含声明漂移）、历史记录、固定 CPU 节点与 safe regression。

    P0's *contract self-test* (``verify_contracts``) stages today's tree and requires P0's epoch
    baseline to verify against it byte for byte; P1's declared ``dispatch.py`` extension makes that
    single assertion fail by construction (its own negative cases reuse the same file).  P1 does not
    patch or weaken P0's verifier: it records the consequence here and compensates with P0's 36
    frozen nodes (13 of them dispatch), P0's manifest/historical checks and P1's own 79-file
    surface check (which allows exactly that one file).

    @param tmp_path 临时目录（P0 的契约自检不使用它，保留签名以便复用）。
    """
    module = _module(ROOT / P0_TEST, "p0_verify")
    verify_p0_manifest()
    module.verify_historical()
    require(int(module.FIXED_EXPECTED_PASSED) == P0_FIXED_EXPECTED_PASSED, "P0's fixed node count changed")
    _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *module.FIXED_NODE_IDS,
        ],
        where="P0 fixed CPU pytest",
        expect_passed=module.FIXED_EXPECTED_PASSED,
    )
    regression_summary = _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *module.REGRESSION_FILES,
        ],
        where="P0 safe regression",
        expect_passed=P0_REGRESSION_EXPECTED_PASSED,
        allow_deselected=True,
    )
    print(f"OK: P0 re-run ({P0_FIXED_EXPECTED_PASSED} fixed nodes + {len(module.REGRESSION_FILES)} regression files)")
    print(regression_summary)


def run_a6_preserved() -> None:
    """@brief 重跑 A6 preserved-v1 锚点（身份、registry 规则、动态族、装配结构）。"""
    summary = _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *A6_PRESERVED_FILES,
        ],
        where="A6 preserved-v1 pytest",
        expect_passed=A6_PRESERVED_EXPECTED_PASSED,
        allow_deselected=True,
    )
    print(f"OK: A6 preserved-v1 anchors ({len(A6_PRESERVED_FILES)} files)\n{summary}")


def run_fixed_cpu() -> None:
    """@brief P1 固定 CPU 节点（精确数量、无 skip/xfail）。"""
    summary = _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "not gpu and not llm and not slow",
            *FIXED_CPU_NODE_IDS,
        ],
        where="fixed P1 CPU pytest",
        expect_passed=FIXED_CPU_EXPECTED_PASSED,
    )
    print(f"OK: fixed P1 CPU pytest ({FIXED_CPU_EXPECTED_PASSED} nodes)\n{summary}")


# ---------------------------------------------------------------------------------------------
# GPU evidence
# ---------------------------------------------------------------------------------------------

_VOLATILE_REPORT_KEYS = frozenset(
    {
        "elapsed_s",
        "timeline",
        "candidate_us",
        "baseline_us",
        "speedup",
        "speedups",
        "roofline",
    }
)
_STABLE_COMPILE_KEYS = ("ok", "language_version", "math_mode", "launch")
#: Digests that are computed *over* the schema-4 report, and therefore over its timings.  They can
#: never be equal across two live runs, so the live/archived comparison excludes exactly these
#: three; ``_reconstruct_run`` assigns the report/result identities and ``_check_one_evidence``
#: requires the archived composition identity to equal ``rebuilt.run_identity``.
_VOLATILE_EVALUATION_KEYS = frozenset({"report_digest", "evaluation_identity"})


@lru_cache(maxsize=4)
def _expected_device_warnings(target_id: str) -> tuple[str, ...]:
    """@brief 用一次生产 Metal probe 重算目标警告，不读取归档文本。"""
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.target import load_target

    backend = MetalBackend()
    profile = load_target(target_id)
    return tuple(backend.limit_notes(profile, backend.context().info()))


def _stable_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """@brief schema-4 报告中跨运行稳定的部分（耗时字段；``compile`` 只保留结果性字段）。

    @param report 报告。
    @return 稳定投影。
    """
    stable = {key: value for key, value in report.items() if key not in _VOLATILE_REPORT_KEYS}
    compile_ = report.get("compile") or {}
    stable["compile"] = {key: compile_.get(key) for key in _STABLE_COMPILE_KEYS}
    stable["speedup"] = report.get("speedup")
    stable["speedups"] = report.get("speedups")
    # The benchmark block is *not* volatile: only its measurements are.  Its protocol facts (how the
    # candidate was dispatched and measured, and whether serial measurement ran) must be identical
    # across runs, otherwise a report could claim a different timing discipline every time.
    stable["benchmark"] = _stable_benchmark(report.get("benchmark"))
    # Host work is a real execution whose *wall time* is as volatile as the kernel's; its plan,
    # writes and verdict are not.
    build = stable.get("build")
    if isinstance(build, Mapping):
        build = dict(build)
        evidence = build.get("evidence")
        if isinstance(evidence, list):
            build["evidence"] = [_stable_host_step(step) for step in evidence]
        elif isinstance(evidence, Mapping):
            build["evidence"] = _stable_host_step(evidence)
        stable["build"] = build
    return stable


#: Benchmark fields that describe *how* a measurement was taken (stable) rather than its numbers.
_STABLE_BENCHMARK_KEYS = ("protocol", "dispatch", "measure_serial", "rotation_sets")


def _stable_benchmark(benchmark: Any) -> dict[str, Any]:
    """@brief 计时块的协议事实（保留 dispatch/read_write/workspace/serial 语义，去掉测量数值）。

    @param benchmark 报告里的 ``benchmark`` 块。
    @return 稳定投影。
    """
    if not isinstance(benchmark, Mapping):
        return {"present": False}
    projected: dict[str, Any] = {"present": True}
    for key in _STABLE_BENCHMARK_KEYS:
        projected[key] = benchmark.get(key)
    projected["baseline_names"] = sorted((benchmark.get("baselines") or {}))
    projected["serial_measured"] = benchmark.get("candidate_serial_us") is not None
    projected["gpu_median_measured"] = benchmark.get("candidate_gpu_us_median") is not None
    projected["attempts_positive"] = int(benchmark.get("attempts") or 0) >= 1
    return projected


def _stable_host_step(step: Any) -> Any:
    """@brief 一次 host 执行中与耗时无关的部分（保留计划摘要、写入量与结论）。

    @param step build.evidence 中的一步。
    @return 稳定投影。
    """
    if not isinstance(step, Mapping):
        return step
    projected = {key: value for key, value in step.items() if key != "duration_ms"}
    log = projected.get("log")
    if isinstance(log, Mapping):
        projected["log"] = {key: value for key, value in log.items() if key != "sha256"}
    return projected


def _stable_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 一条证据记录中跨运行稳定的部分。

    @param document 证据记录。
    @return 稳定投影。
    """
    evaluation = {
        key: value for key, value in dict(document["evaluation"]).items() if key not in _VOLATILE_EVALUATION_KEYS
    }
    return {
        "schema_id": document.get("schema_id"),
        "node": document.get("node"),
        "slice": document.get("slice"),
        "provider_label": document.get("provider_label"),
        "size": document.get("size"),
        "n": document.get("n"),
        "instance_key": document.get("instance_key"),
        "record": document.get("record"),
        "projection": document.get("projection"),
        "evaluation": evaluation,
        # ``run_identity`` hashes the evaluation document (which carries the report): it is one of
        # the three timing-derived digests, recomputed per archive instead of compared across runs.
        "report": _stable_report(document.get("report") or {}),
        "compiled_candidate_sha256": document.get("compiled_candidate_sha256"),
        "built_entry_paths": document.get("built_entry_paths"),
        "device": document.get("device"),
    }


def _evidence_runs(document: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """@brief 取证据文件包含的逐次运行记录（单条或分组）。

    @param document 证据文件内容。
    @return 记录列表（按文档键排序）。
    """
    if "runs" in document:
        return [document["runs"][key] for key in sorted(document["runs"])]
    if "run" in document:
        return [document["run"]]
    if "below" in document and "above" in document:
        return [document["below"], document["above"]]
    if "record" in document and "evaluation" in document:
        # A single-run document (one node, one run).
        return [document]
    # A document that only archives refusals has no successful run to compare.
    return []


def _source_of(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """@brief 按 ``a.b`` 取证据中的运行记录（``""`` 表示文档本身）。

    @param document 证据文件内容。
    @param key 键路径。
    @return 运行记录。
    """
    value: Any = document
    for part in [piece for piece in key.split(".") if piece]:
        value = value[part]
    require(isinstance(value, Mapping), f"evidence key {key!r} is not a run record")
    return value


def _rebuild_result_document(document: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 攻击者的自洽模型：由（可能已被改写的）归档证据重建结果身份文档。

    This is the *attacker's* identity step, reproduced so that
    :func:`_rebalance_forgery` can make a tampered archive internally consistent before the
    verifier is asked to reject it.  The verifier never derives an expected value from here: every
    expected identity comes from :func:`_reconstruct_run`, which reads only the fixed fixture and
    the production contract functions.

    @param document 证据记录（可能已被改写）。
    @param report 归档的 schema-4 报告。
    @return 结果文档。
    """
    from tcenv.taskcontract.runner import EVALUATION_PLAN_SCHEMA_ID, EVALUATION_PLAN_SCHEMA_VERSION

    evaluation = document["evaluation"]
    return {
        "schema_id": EVALUATION_PLAN_SCHEMA_ID,
        "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
        "locator": evaluation["locator"],
        "track": evaluation["track"],
        "candidate_sha256": evaluation["compiled_candidate_sha256"],
        "task_pack_digest": evaluation["task_pack_digest"],
        "effective_correctness_only": bool(evaluation["effective_correctness_only"]),
        "geometry_identity": evaluation["geometry_identity"],
        "bundle_digest": evaluation["bundle_digest"],
        "presentation_sha256": evaluation["presentation_sha256"],
        "artifact_set_digest": evaluation["artifact_set_digest"],
        "native_digest": evaluation["native_digest"],
        "report": dict(report),
    }


def _rebuild_run_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 攻击者的自洽模型：由归档证据重建组合身份文档（``run_identity`` 的重算步骤）。

    Like :func:`_rebuild_result_document` this exists only to model a thorough forger; the
    verifier's own expectation is :func:`_reconstruct_run`'s.

    @param document 证据记录（可能已被改写）。
    @return 组合文档。
    """
    from tcenv.taskcontract.portfolio_runner import PORTFOLIO_RUN_SCHEMA_ID, PORTFOLIO_RUN_VERSION

    record = document["record"]
    projection = document["projection"]
    evaluation = document["evaluation"]
    binding = projection["binding"]
    identities = {
        "domain_plan_digest": record["domain_plan_digest"],
        "portfolio_digest": record["portfolio_digest"],
        "variant_id": record["variant_id"],
        "fallback_id": record["fallback_id"],
        "provider": record["provider"],
        "fallback_reason": record["fallback_reason"],
        "dispatch_record_digest": fingerprint(record),
        "projection_binding_digest": fingerprint(binding),
        "candidate_sha256": projection["candidate_sha256"],
        "parent_portfolio_sha256": binding["parent_portfolio_sha256"],
        "geometry_origin_path": binding["geometry_origin_path"],
        "geometry_origin_sha256": binding["geometry_origin_sha256"],
        "geometry_identity": evaluation["geometry_identity"],
        "bundle_digest": evaluation["bundle_digest"],
        "artifact_set_digest": evaluation["artifact_set_digest"],
        "native_digest": evaluation["native_digest"],
        "evaluation_identity": evaluation["evaluation_identity"],
    }
    return {
        "schema_id": PORTFOLIO_RUN_SCHEMA_ID,
        "run_version": PORTFOLIO_RUN_VERSION,
        "record": record,
        "projection": projection,
        "evaluation": evaluation,
        "identities": identities,
        "active_manifest": None,
        "activation_generation": None,
        "performance_claim": None,
        "fallback_credits_variant": False,
    }


def _split_parameters(text: str) -> list[str]:
    """@brief 按顶层逗号切分 C/MSL 参数表（``()`` 与 ``[]`` 内不切）。

    @param text 参数表正文（不含外层括号）。
    @return 参数片段列表。
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    if "".join(current).strip():
        parts.append("".join(current))
    return [part.strip() for part in parts]


def _expected_bindings(source_text: str, kernel_name: str, *, where: str) -> list[dict[str, Any]]:
    """@brief 由**固定候选源**的 kernel 签名重建反射绑定行（不经编译）。

    The native descriptor's ``bindings`` are Metal's reflection of the compiled kernel signature:
    parameter name, ``[[buffer(i)]]`` slot, pointee element type, and read/readwrite from the
    ``const`` qualifier.  All of that is present in the fixed source bytes, so the expected rows are
    derived here and compared instead of being read back from the archive.

    @param source_text 固定候选的完整源文本。
    @param kernel_name 实际编译的 kernel 名（由生产的 kernel contract 给出）。
    @param where 诊断位置。
    @return 期望的绑定行（按签名顺序）。
    @raises AssertionError 固定源不再声明该 kernel，或其签名无法机械解析。
    """
    from tcenv.dtypes import DTYPES

    match = re.search(rf"kernel\s+void\s+{re.escape(kernel_name)}\s*\(", source_text)
    require(match is not None, f"{where}: the fixed source declares no kernel {kernel_name!r}")
    position = match.end()
    depth = 1
    end = position
    while end < len(source_text) and depth:
        if source_text[end] == "(":
            depth += 1
        elif source_text[end] == ")":
            depth -= 1
        end += 1
    require(depth == 0, f"{where}: the fixed kernel signature of {kernel_name!r} is unbalanced")
    parameters = source_text[position : end - 1]
    body = source_text[end:]
    scalars = {dtype.msl: dtype for dtype in DTYPES.values()}
    rows: list[dict[str, Any]] = []
    for chunk in _split_parameters(parameters):
        slot = re.search(r"\[\[\s*buffer\s*\(\s*(\d+)\s*\)\s*\]\]", chunk)
        if slot is None:
            continue
        name_match = re.search(r"([A-Za-z_]\w*)\s*\[\[\s*buffer", chunk)
        require(name_match is not None, f"{where}: a [[buffer]] parameter has no name: {chunk!r}")
        name = name_match.group(1)
        declaration = chunk[: name_match.start()]
        scalar = None
        for token in re.findall(r"[A-Za-z_]\w*", declaration):
            if token in scalars:
                scalar = scalars[token]
        require(scalar is not None, f"{where}: buffer parameter {name!r} has no MSL scalar element type")
        require(
            re.search(rf"\b{re.escape(name)}\b", body) is not None,
            f"{where}: buffer parameter {name!r} is never used by the kernel body",
        )
        rows.append(
            {
                "name": name,
                "index": int(slot.group(1)),
                "kind": "buffer",
                "dtype_code": int(scalar.mtl_code),
                "access": "read" if "const" in declaration else "readwrite",
                "elem_size": int(scalar.nbytes),
                "used": True,
                "tg_bytes": None,
            }
        )
    require(rows, f"{where}: the fixed kernel {kernel_name!r} declares no buffer parameter")
    require(
        [row["index"] for row in rows] == list(range(len(rows))),
        f"{where}: the fixed kernel's buffer slots are not 0..n-1 in signature order",
    )
    return rows


def verify_bench_constants() -> None:
    """@brief 计时批量的 kmax/target/慢核阈值/watchdog 必须仍与生产一致。

    ``_check_benchmark`` refuses batch sizes outside ``[kmin, kmax]`` and outside the per-batch wall
    budget, using copies of ``tcenv.bench``'s numbers.  This re-reads the production function's own
    signature, module constant and the AST of its ``if per > 0.1`` branch, so a change there fails
    here instead of silently widening the accepted range.

    @raises AssertionError 验收器钉住的常量与生产函数不一致。
    """
    import inspect

    from tcenv import bench

    parameters = inspect.signature(bench.choose_iters).parameters
    require(
        int(parameters["kmax"].default) == BENCH_MAX_ITERS,
        f"tcenv.bench.choose_iters caps the batch at {parameters['kmax'].default!r}, not {BENCH_MAX_ITERS}",
    )
    require(
        float(parameters["target_s"].default) == BENCH_TARGET_S,
        f"tcenv.bench.choose_iters targets {parameters['target_s'].default!r} s, not the pinned {BENCH_TARGET_S} s",
    )
    require(
        float(bench.BATCH_WALL_BUDGET_S) == BENCH_BATCH_WALL_BUDGET_S,
        f"tcenv.bench.BATCH_WALL_BUDGET_S is {bench.BATCH_WALL_BUDGET_S!r}, not the pinned "
        f"{BENCH_BATCH_WALL_BUDGET_S} s",
    )
    slow_cuts: list[float] = []
    for node in ast.walk(ast.parse(inspect.getsource(bench.choose_iters))):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if (
            isinstance(test.left, ast.Name)
            and test.left.id == "per"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Gt)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and isinstance(test.comparators[0].value, (int, float))
            and not isinstance(test.comparators[0].value, bool)
        ):
            slow_cuts.append(float(test.comparators[0].value))
    require(
        slow_cuts == [BENCH_SLOW_LAUNCH_S],
        f"tcenv.bench.choose_iters slow-launch cuts are {slow_cuts}, not the pinned [{BENCH_SLOW_LAUNCH_S}]",
    )
    print(
        f"OK: the pinned timing constants are production's (kmax={BENCH_MAX_ITERS}, "
        f"target {BENCH_TARGET_S} s, batch wall budget {BENCH_BATCH_WALL_BUDGET_S} s, "
        f"slow-launch cut {BENCH_SLOW_LAUNCH_S} s)"
    )


def _check_report_blocks(
    report: Mapping[str, Any],
    *,
    contract: Any,
    instance: Any,
    mode: str,
    size: str,
    output_tensors: Mapping[str, tuple[tuple[int, ...], str]],
) -> None:
    """@brief ABI/correctness/safety 的内部证据必须**支持**它们自己的 ``ok``。

    The report's ``ok`` flags are conclusions; this checks the evidence that is supposed to imply
    them instead of trusting the flag: ABI violations must be empty, safety's ``ok`` must be exactly
    ``not violations and not nondeterminism``, and correctness's ``ok`` must be exactly
    ``all(case.ok)`` with every case carrying no bad element, no NaN, no unwritten output and no
    per-case safety hit.  The cases, per-case seeds, element counts and tolerances are the fixed
    node's (from the request instance and the evaluator seed), not the archive's.

    @param report schema-4 报告。
    @param instance 请求实例（tolerances / case domain / 元素数）。
    @param mode 固定节点的评测模式。
    @param size 诊断用 size。
    @param contract 请求实例的任务合同（重建参考输出）。
    @param output_tensors ABI 中可写张量的名字 → (形状, dtype)（worst 只能指向它们，且索引必须按 rank 落在形状内）。
    @raises AssertionError 内部证据与 ``ok`` 矛盾，或与固定节点不一致。
    """
    import torch

    from tcenv.backends.metal.testing import _floor, scaled_error
    from tcenv.backends.metal.v2_runtime import prepare_v2_case
    from tests.support.shape_portfolio_metal import P1_SIZES

    verification = report.get("verification") or {}
    require(set(verification) == VERIFICATION_KEYS, f"{size}: the verification block is not the fixed shape")
    require(
        verification.get("status") == "unsupported",
        f"{size}: the report claims a formal verification result this backend cannot produce",
    )
    require(
        verification.get("counterexample") is None,
        f"{size}: an unsupported verification must not carry a counterexample",
    )
    require(
        verification.get("message") == EXPECTED_VERIFICATION_MESSAGE,
        f"{size}: the verification statement is not the harness's 'no checker exists' text",
    )
    abi = report.get("abi") or {}
    require(set(abi) == ABI_BLOCK_KEYS, f"{size}: the ABI block is not the fixed shape")
    require(abi.get("ok") is True, f"{size}: ABI did not pass")
    require(abi.get("violations") == [], f"{size}: a passing ABI block carries violations")
    safety = report.get("safety") or {}
    require(set(safety) == SAFETY_BLOCK_KEYS, f"{size}: the safety block is not the fixed shape")
    violations = safety.get("violations")
    nondeterminism = safety.get("nondeterminism")
    require(isinstance(violations, list) and violations == [], f"{size}: safety.violations is {violations!r}")
    require(isinstance(nondeterminism, list) and nondeterminism == [], f"{size}: safety.nondeterminism is {nondeterminism!r}")
    require(safety.get("validation") == [], f"{size}: safety.validation is not empty")
    require(
        safety.get("ok") is (not violations and not nondeterminism),
        f"{size}: safety.ok is not implied by its own violations/nondeterminism",
    )
    correctness = report.get("correctness") or {}
    require(set(correctness) == CORRECTNESS_BLOCK_KEYS, f"{size}: the correctness block is not the fixed shape")
    cases = correctness.get("cases")
    require(isinstance(cases, list) and bool(cases), f"{size}: correctness carries no case results")
    names = [str(case.get("case")) for case in cases]
    require(
        names == [str(case) for case in instance.case_domain.cases],
        f"{size}: the correctness cases are {names}, the fixed node runs {list(instance.case_domain.cases)}",
    )
    elements = int(P1_SIZES[size]["N"])
    for index, case in enumerate(cases):
        require(set(case) == CORRECTNESS_CASE_KEYS, f"{size}: correctness case {index} is not the fixed shape")
        seed = case.get("seed")
        require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"{size}: case {index} seed is not an integer seed: {seed!r}",
        )
        require(
            seed == FIXED_SEED + 7919 * (index + 1),
            f"{size}: case {index} ran with seed {seed!r}, not the fixed seed's derivation",
        )
        n_elems = case.get("n_elems")
        require(
            isinstance(n_elems, int) and not isinstance(n_elems, bool),
            f"{size}: case {index} n_elems is not an integer element count: {n_elems!r}",
        )
        require(n_elems == elements, f"{size}: case {index} checked {n_elems} elements")
        allow = case.get("allow")
        require(
            isinstance(allow, (int, float)) and not isinstance(allow, bool) and math.isfinite(float(allow)),
            f"{size}: case {index} allow is not a finite number: {allow!r}",
        )
        require(float(allow) == 1.0, f"{size}: case {index} does not use the tolerance scale")
        n_bad = case.get("n_bad")
        require(
            isinstance(n_bad, int) and not isinstance(n_bad, bool),
            f"{size}: case {index} n_bad is not an integer bad-element count: {n_bad!r}",
        )
        require(
            case.get("ok") is (n_bad == 0),
            f"{size}: case {index} reports ok={case.get('ok')!r} with n_bad={n_bad!r}",
        )
        require(case.get("ok") is True, f"{size}: case {index} did not pass")
        require(n_bad == 0, f"{size}: case {index} has {n_bad} bad elements")
        nan_count = case.get("nan_count")
        require(
            isinstance(nan_count, int) and not isinstance(nan_count, bool),
            f"{size}: case {index} nan_count is not an integer NaN count: {nan_count!r}",
        )
        require(nan_count == 0, f"{size}: case {index} produced NaN")
        unwritten = case.get("unwritten")
        require(
            isinstance(unwritten, int) and not isinstance(unwritten, bool),
            f"{size}: case {index} unwritten is not an integer unwritten-element count: {unwritten!r}",
        )
        require(unwritten == 0, f"{size}: case {index} left a pure-write output poisoned")
        require(case.get("safety") == [], f"{size}: case {index} carries a safety violation")
        require(case.get("message") == "", f"{size}: a passing case carries a failure message")
        case_scaled_err = case.get("scaled_err")
        require(
            isinstance(case_scaled_err, (int, float))
            and not isinstance(case_scaled_err, bool)
            and math.isfinite(float(case_scaled_err)),
            f"{size}: case {index} scaled_err is not a finite number: {case_scaled_err!r}",
        )
        require(
            float(case_scaled_err) <= float(allow),
            f"{size}: case {index} reports a worst scaled error above its own tolerance",
        )
        require(
            float(case_scaled_err) == float((case.get("worst") or {}).get("scaled_err") or 0.0),
            f"{size}: case {index} has two different worst errors in one case result",
        )
        require(
            case.get("baseline_scaled_err") is None,
            f"{size}: case {index} claims a v1 baseline comparison the v2 runtime never ran",
        )
        require(
            case.get("input_sha256") == "",
            f"{size}: case {index} claims a legacy input-replay digest the v2 runtime never computed",
        )
        worst = case.get("worst") or {}
        require(set(worst) == CORRECTNESS_WORST_KEYS, f"{size}: case {index} worst block is not the fixed shape")
        worst_scaled_err = worst.get("scaled_err")
        require(
            isinstance(worst_scaled_err, (int, float))
            and not isinstance(worst_scaled_err, bool)
            and math.isfinite(float(worst_scaled_err)),
            f"{size}: case {index} worst.scaled_err is not a finite number: {worst_scaled_err!r}",
        )
        require(
            str(worst.get("output")) in output_tensors,
            f"{size}: case {index} blames {worst.get('output')!r} for the worst element",
        )
        position = worst.get("index")
        shape, dtype = output_tensors[str(worst.get("output"))]
        require(
            isinstance(position, list) and len(position) == len(shape),
            f"{size}: case {index} worst index {position!r} is not a rank-{len(shape)} index inside {shape}",
        )
        require(
            all(isinstance(value, int) and not isinstance(value, bool) for value in position),
            f"{size}: case {index} worst index {position!r} is not an integer worst-element index",
        )
        require(
            all(0 <= value < int(limit) for value, limit in zip(position, shape)),
            f"{size}: case {index} worst index {position!r} is not a rank-{len(shape)} index inside {shape}",
        )
        # Every fixed candidate in this slice computes y=-x with the same fp16 rounding as the
        # reference.  Therefore every error is zero and the production `_check_v2_case` selects
        # the first element of the first writable output (argmax over an all-zero error tensor).
        # A different got value, even one inside tolerance, is not raw evidence of this run.
        require(
            position == [0] * len(shape),
            f"{size}: case {index} worst index {position!r} is not the fixed exact candidate's first element",
        )
        # The reference output for this fixed case/seed, and the production scaled-error formula at
        # the claimed worst element: a rewritten ``worst.got``/``want`` cannot keep a zero error.
        prepared = prepare_v2_case(contract, str(case["case"]), seed)
        flat = 0
        for value, limit in zip(position, shape):
            flat = flat * int(limit) + value
        reference = float(prepared.ref[str(worst["output"])].reshape(-1)[flat])
        want = worst.get("want")
        got = worst.get("got")
        require(
            isinstance(want, (int, float)) and not isinstance(want, bool) and math.isfinite(float(want)),
            f"{size}: case {index} worst.want is not a finite number: {want!r}",
        )
        require(
            isinstance(got, (int, float)) and not isinstance(got, bool) and math.isfinite(float(got)),
            f"{size}: case {index} worst.got is not a finite number: {got!r}",
        )
        require(
            float(want) == reference,
            f"{size}: case {index} worst.want is {want!r}, the fixed reference is {reference!r}",
        )
        require(
            float(got) == reference,
            f"{size}: case {index} worst.got is {got!r}, the fixed exact candidate output is {reference!r}",
        )
        scale_tensor = prepared.scale[str(worst["output"])].reshape(-1)
        scale_value = float(scale_tensor[0] if scale_tensor.numel() == 1 else scale_tensor[flat])
        error = float(
            scaled_error(
                torch.tensor([float(got)], dtype=torch.float64),
                torch.tensor([reference], dtype=torch.float64),
                torch.tensor([scale_value], dtype=torch.float64),
                float(instance.tolerances.rtol),
                float(instance.tolerances.atol_rel),
                bool(instance.tolerances.exact),
                _floor(str(dtype)),
            )[0]
        )
        require(
            abs(error - float(worst_scaled_err)) <= 1e-9 * max(1.0, abs(error)),
            f"{size}: case {index} worst error {worst_scaled_err!r} is not the tolerance-normalized error of "
            f"github/want ({error!r})",
        )
        require(
            float(worst_scaled_err) <= float(allow),
            f"{size}: case {index} worst element is outside its own tolerance",
        )
    require(
        correctness.get("ok") is all(case.get("ok") is True for case in cases),
        f"{size}: correctness.ok is not implied by its own cases",
    )
    require(correctness.get("ok") is True, f"{size}: correctness did not pass")
    require(correctness.get("baseline_errors") == [], f"{size}: correctness reports baseline errors")
    coverage = correctness.get("coverage") or {}
    require(set(coverage) == COVERAGE_KEYS, f"{size}: the correctness coverage block is not the fixed shape")
    require(
        coverage.get("scope") == EXPECTED_COVERAGE_SCOPE,
        f"{size}: the correctness coverage statement is not the v2 runtime's path text",
    )
    require(
        coverage.get("kind") == "v2-runtime",
        f"{size}: the correctness coverage is not the v2 runtime path",
    )
    tolerances = instance.tolerances
    require(set(correctness.get("tolerance") or {}) == TOLERANCE_KEYS, f"{size}: the tolerance block is not the fixed shape")
    require(
        _strict_json_equal(
            correctness.get("tolerance"),
            {"rtol": float(tolerances.rtol), "atol_rel": float(tolerances.atol_rel), "exact": bool(tolerances.exact)},
        ),
        f"{size}: the report's tolerance is not the request instance's",
    )


def _check_timeline(report: Mapping[str, Any], *, mode: str, size: str) -> None:
    """@brief 阶段时间线必须是固定路径的 (stage, status) 序列，只排除时长。

    ``report.timeline`` is not a free-form diagnostic: it says which stages the evaluator entered and
    how each ended.  Its numbers are volatile, but the sequence and verdicts are not, so they are
    compared against the harness path for this mode and every entry is a closed ``stage/status/
    duration_s`` row.

    @param report schema-4 报告。
    @param mode 固定节点的评测模式。
    @param size 诊断用 size。
    @raises AssertionError 阶段序列、状态或条目形状不符。
    """
    timeline = report.get("timeline")
    require(isinstance(timeline, list) and bool(timeline), f"{size}: the report has no stage timeline")
    expected = EXPECTED_TIMELINE_STAGES[mode]
    observed = [(str(entry.get("stage")), str(entry.get("status"))) for entry in timeline]
    require(
        observed == [(stage, "ok") for stage in expected],
        f"{size}: the stage timeline is {observed}, the fixed {mode} path is {[(stage, 'ok') for stage in expected]}",
    )
    for index, entry in enumerate(timeline):
        require(set(entry) == TIMELINE_ENTRY_KEYS, f"{size}: timeline entry {index} is not the fixed shape")
        duration = entry.get("duration_s")
        require(
            isinstance(duration, (int, float)) and not isinstance(duration, bool) and float(duration) >= 0.0,
            f"{size}: timeline entry {index} carries no non-negative duration",
        )


def _check_build_log(
    report: Mapping[str, Any],
    *,
    size: str,
    kernel_name: str,
    math: str,
    language: str,
    assembled: str,
    header_lines: int,
    source_text: str,
    single_source_entry: tuple[str, str] | None,
    descriptor: Mapping[str, Any],
) -> None:
    """@brief 按生产 `MetalSourceBuilder` 的模板逐字节重建编译日志。

    The builder renders one deterministic three-line log and never versions it, so the archived
    ``build.evidence.log`` is reproducible byte for byte *before* A3 attaches the execution geometry
    and before the multi-file adapter rewrites the native entry.  Its pipeline token is therefore the
    digest of the builder's own ``NativeExecutable``: the pre-attach descriptor plus the entry the
    production ``SourceBundle`` used (the candidate file for a single-source candidate, ``kernel.tx``
    with the combined-source digest otherwise).  ``build.evidence.duration_ms`` is the same template's
    rounding of the two measured durations.

    @param report schema-4 报告。
    @param size 诊断用 size。
    @param kernel_name 已校验的 kernel 名。
    @param math 已校验的 math 模式。
    @param language 已校验的 MSL 版本。
    @param assembled 契约头 + 固定候选源码的完整编译单元。
    @param header_lines 契约头行数。
    @param source_text 固定候选的合并源码（``combined_candidate_source``）。
    @param single_source_entry 单源候选的 (path, sha256)；多文件候选为 ``None``。
    @param descriptor 已校验的原生描述符（post-attach，重建时去掉 ``execution_geometry``）。
    @raises AssertionError 日志字节、摘要或 duration_ms 不是该模板的输出。
    """
    from tcenv.remote.artifact_bundle import NativeExecutable

    pre_descriptor = {key: value for key, value in descriptor.items() if key != "execution_geometry"}
    if single_source_entry is None:
        entry_path, entry_sha256 = "kernel.tx", hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    else:
        entry_path, entry_sha256 = single_source_entry
    pipeline = NativeExecutable(
        backend_id=str(descriptor.get("backend") or ""),
        kind=EXPECTED_NATIVE_KIND,
        entry_path=entry_path,
        entry_sha256=entry_sha256,
        descriptor=pre_descriptor,
    )
    compile_block = report.get("compile") or {}
    expected_log = (
        f"metal: kernel={kernel_name} compile_s={float(compile_block['compile_s']):.6f} "
        f"pipeline_s={float(compile_block['pipeline_s']):.6f} language={language} "
        f"math={math}\n"
        f"metal: assembled bytes={len(assembled)} header_lines={header_lines} "
        f"sha256={pre_descriptor['assembled_sha256']}\n"
        f"metal: pipeline={pipeline.digest} bindings={len(pre_descriptor['bindings'])} "
        f"max_threads={pre_descriptor['max_total_threads']} "
        f"thread_execution_width={pre_descriptor['execution_width']} "
        f"tg_bytes={pre_descriptor['static_tg_bytes']}\n"
    ).encode("utf-8")
    log = ((report.get("build") or {}).get("evidence") or {}).get("log") or {}
    require(set(log) == {"sha256", "size_bytes"}, f"{size}: the build log block is not the fixed shape")
    log_size = log.get("size_bytes")
    require(
        isinstance(log_size, int) and not isinstance(log_size, bool),
        f"{size}: build log size_bytes is not an integer byte count: {log_size!r}",
    )
    require(
        log_size == len(expected_log),
        f"{size}: the build log claims {log_size!r} bytes; the production template for this build is "
        f"{len(expected_log)}",
    )
    expected_sha256 = hashlib.sha256(expected_log).hexdigest()
    require(
        str(log.get("sha256")) == expected_sha256,
        f"{size}: the build log digest is {log.get('sha256')!r}; the production template's bytes hash to {expected_sha256}",
    )
    expected_duration = max(0, int(round((float(compile_block["compile_s"]) + float(compile_block["pipeline_s"])) * 1000)))
    require(
        _strict_json_equal(((report.get("build") or {}).get("evidence") or {}).get("duration_ms"), expected_duration),
        f"{size}: build.evidence.duration_ms is not the build's rounded compile+pipeline time",
    )


def verify_toolchain_probe(context: FixtureContext) -> None:
    """@brief 用生产的 Metal 探测核对本机工具链/设备事实（真实 GPU 阶段）。

    The verifier pins the epoch's toolchain (``EXPECTED_MSL_VERSION``), device name, GPU family,
    unified memory and thread cap as constants so the archive is never its own witness.  This runs
    the production probe on the real device and requires those constants to match it, so the pinned
    facts are re-established from the machine every full run.

    @param context 重建的有限切片上下文（提供目标档案）。
    @raises AssertionError 本机探测与钉住的工具链/设备事实不一致。
    """
    from tcenv.backends.metal.backend import MetalBackend

    info = MetalBackend().context().info()
    require(
        info.get("name") == context.profile.calibration_device_name,
        f"the probed device {info.get('name')!r} is not the target's {context.profile.calibration_device_name!r}",
    )
    require(
        info.get("gpu_family") == context.profile.match_family,
        f"the probed GPU family {info.get('gpu_family')!r} is not the target's {context.profile.match_family!r}",
    )
    require(info.get("unified_memory") is True, "the probed device does not report unified memory")
    require(
        str(info.get("language_version")) == EXPECTED_MSL_VERSION,
        f"the probed MSL version {info.get('language_version')!r} is not the pinned {EXPECTED_MSL_VERSION!r}",
    )
    require(
        int(info.get("max_threads_per_threadgroup", 0))
        == int(context.profile.raw_limits["max_threads_per_threadgroup"]),
        "the probed thread-per-threadgroup cap is not the target's",
    )
    print(
        "OK: production Metal probe confirms the pinned toolchain/device facts "
        f"(MSL {EXPECTED_MSL_VERSION}, {context.profile.calibration_device_name}, {context.profile.match_family})"
    )


class ReconstructedRun:
    """@brief 一次固定运行从**固定输入**独立重建出的完整期望。

    Every field is produced by the fixed slice (named instance, fixed candidate sources,
    ``DomainAuthority``, the selected projection, the request contract) and the production
    functions that describe it (``project_provider``, ``geometry_pair``/``select_geometry``,
    ``CanonicalBundle``, ``_lifecycle_factory``, ``HostTilingPlan``, ``verify_evaluator_build``,
    ``select_track``, the TrackRenderer, ``_evaluation_fingerprint``).  No field is copied from
    the archive under review, so a forger who edits a field and recomputes every digest derived
    from it still disagrees with this object.
    """

    def __init__(
        self,
        *,
        request: Any,
        record: Any,
        projection: Any,
        candidate_sha256: str,
        pair: Any,
        resolved: Any,
        bundle_digest: str,
        lifecycle: Mapping[str, Any] | None,
        host_plan: Any,
        host_runs: list[dict[str, Any]],
        rotation_sets: int,
        task: Any,
        calibration: Mapping[str, Any],
        workspace_bytes: dict[str, int],
        workspace_zero: dict[str, bool],
        artifact_set_digest: str,
        native_digest: str | None,
        artifact_entries: tuple[Mapping[str, Any], ...],
        built_entry_paths: list[str],
        evaluation: dict[str, Any],
        result_document: dict[str, Any],
        run_document: dict[str, Any],
        run_identity: str,
        launch_digest: str,
        report_fields: dict[str, Any],
        build_fields: dict[str, Any],
    ) -> None:
        """@brief 保存重建结果。

        @param request 固定请求。
        @param record 重放的 dispatch 记录。
        @param projection 固定 provider 的投影。
        @param candidate_sha256 子候选身份。
        @param pair 重建的几何配对。
        @param resolved 请求实例上的已求值几何。
        @param bundle_digest canonical bundle 摘要。
        @param lifecycle 生命周期文档或 ``None``。
        @param host_plan 由投影候选重新解析出的 Host Tiling 计划。
        @param host_runs 期望的逐次 host 执行记录。
        @param rotation_sets 由 ABI 重建的计时轮转组数。
        @param task 该请求实例的 v2 评测任务（roofline 重建用）。
        @param calibration 目标档案的标定数据（roofline 重建用）。
        @param workspace_bytes 由 ABI 派生的 workspace 字节数。
        @param workspace_zero 由几何派生的清零策略。
        @param artifact_set_digest 已审计构建产物集合摘要。
        @param native_digest 已审计原生可执行文件摘要。
        @param artifact_entries 已审计的精确 artifact entries。
        @param built_entry_paths 已审计的构建路径（排序）。
        @param evaluation 期望的 sidecar 文档。
        @param result_document 期望的评测结果身份文档。
        @param run_document 期望的组合身份文档。
        @param run_identity 期望的组合身份。
        @param launch_digest 固定候选源在该实例上的 launch 块摘要（编译期实际派遣）。
        @param report_fields 由固定切片派生的报告字段期望。
        @param build_fields 由固定切片派生的 ``report.build`` 任务身份期望。
        """
        self.request = request
        self.record = record
        self.projection = projection
        self.candidate_sha256 = candidate_sha256
        self.pair = pair
        self.resolved = resolved
        self.bundle_digest = bundle_digest
        self.lifecycle = lifecycle
        self.host_plan = host_plan
        self.host_runs = host_runs
        self.rotation_sets = rotation_sets
        self.task = task
        self.calibration = calibration
        self.workspace_bytes = workspace_bytes
        self.workspace_zero = workspace_zero
        self.artifact_set_digest = artifact_set_digest
        self.native_digest = native_digest
        self.artifact_entries = artifact_entries
        self.built_entry_paths = built_entry_paths
        self.evaluation = evaluation
        self.result_document = result_document
        self.run_document = run_document
        self.run_identity = run_identity
        self.launch_digest = launch_digest
        self.report_fields = report_fields
        self.build_fields = build_fields


def _expected_request(context: FixtureContext, *, size: str):
    """@brief 由固定切片独立构造请求（fp16 / 该 size / golden），不读归档请求。

    @param context 重建的有限切片上下文。
    @param size 固定命名 size。
    @return ``(DispatchRequest, SemanticInstanceV2)``。
    """
    from tcenv.opspec.v2 import instantiate_v2
    from tcenv.portfolio.dispatch import DispatchRequest

    instance = instantiate_v2(context.contract.spec, P1_DTYPE, size)
    point = context.plan.find(P1_DTYPE, size, P1_USAGE)
    require(point is not None, f"{size}: the fixed semantic domain has no {P1_DTYPE}/{size}/{P1_USAGE} point")
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=P1_USAGE)
    return request, instance


def _reconstruct_run(
    context: FixtureContext, *, size: str, label: str, reason: str | None, mode: str, report: Mapping[str, Any]
) -> ReconstructedRun:
    """@brief 从固定候选/DomainAuthority/选中投影/请求合同独立重建一次运行的期望。

    This is the verifier's *only* source of truth for the archived sidecar.  The request is built
    from the named instance (never parsed from the archive), dispatch is replayed with the fixed
    plan/portfolio/evidence, the provider is projected from the fixed parent entries, the geometry
    pair and canonical bundle are re-derived and resolved against the request's own ABI, the Host
    Tiling plan is re-parsed from the projected candidate, and the schema-4 build record is audited
    by :func:`tcenv.remote.execution_geometry.verify_evaluator_build` against that bundle.

    The single archive input is the schema-4 ``report``: it is the measured object itself, and its
    content is gated by the closed reader plus the independently derived ``report_fields`` (task
    instance, contract digests, evaluation fingerprint, baseline digests, mode, seed, device).

    @param context 重建的有限切片上下文（超限/窄 portfolio 时调用方传入对应上下文）。
    @param size 固定命名 size。
    @param label 期望的 provider 标签（``variant_key`` 或 ``fallback``）。
    @param reason 期望的 fallback 原因（候选运行为 ``None``）。
    @param mode 该固定节点的评测模式（``quick``/``full``，来自固定节点而非归档）。
    @param report 归档的 schema-4 报告（被测对象）。
    @return 重建结果。
    @raises AssertionError 固定切片自身不再产生这条运行，或报告与该切片无关。
    """
    from types import SimpleNamespace

    from tcenv.backends.metal.launch import launch_math, parse_launch_spec
    from tcenv.backends.metal.track_adapter import (
        combined_candidate_source,
        launch_constants,
        metal_contract,
        v2_evaluation_task,
    )
    from tcenv.backends.metal.v2_runtime import HostTilingPlan
    from tcenv.dtypes import get_dtype
    from tcenv.evaluate import (
        _evaluation_fingerprint,
        baseline_environment_digest,
        baseline_implementation_digest,
    )
    from tcenv.portfolio.candidate import BUNDLE_TRACK_ID, project_provider
    from tcenv.remote.execution_geometry import verify_evaluator_build
    from tcenv.taskcontract.portfolio_runner import (
        PORTFOLIO_RUN_SCHEMA_ID,
        PORTFOLIO_RUN_VERSION,
        _lifecycle_factory,
        _stage_signature,
        locator_for_instance,
    )
    from tcenv.taskcontract.runner import (
        EVALUATION_PLAN_SCHEMA_ID,
        EVALUATION_PLAN_SCHEMA_VERSION,
        select_track,
    )
    from tcenv.track.abi import derive_tensor_memory_contract, tensor_span_bytes
    from tcenv.track.bundle import CanonicalBundle
    from tcenv.track.delivery import candidate_geometry
    from tcenv.track.geometry import geometry_pair, select_geometry
    from tcenv.track.plane import TrackRenderer
    from tcenv.schema import PROTOCOL_VERSION
    from tcenv.target import calibration_file
    from tcenv.track.registry import load_track_registry
    from tests.support.shape_portfolio_metal import P1_SIZES

    request, instance = _expected_request(context, size=size)
    record = context.resolve(request)
    require(
        context.label(record) == label,
        f"{size}: the fixed plan/portfolio select {context.label(record)!r} for this request, not {label!r}",
    )
    expected_provider = "fallback" if label == "fallback" else "candidate"
    require(record.provider == expected_provider, f"{size}: the fixed slice does not select a {expected_provider}")
    require(record.fallback_reason == reason, f"{size}: the fixed slice has no declared reason {reason!r}")
    provider = (
        context.portfolio.fallback
        if record.provider == "fallback"
        else context.portfolio.variant_for_provider(record.variant_id)
    )
    require(provider is not None, f"{size}: the fixed portfolio does not contain the selected provider")

    contract = context.contract_for(P1_DTYPE, size)
    locator = locator_for_instance(context.locator, P1_DTYPE, size)
    abi = derive_tensor_memory_contract(instance, target=context.profile)
    # 1. The selected projection, rebuilt from the fixed parent sources + DomainAuthority.
    projection = project_provider(context.portfolio, provider, authority=context._authority())
    require(projection.provider == record.provider, f"{size}: the projection renames the dispatch provider")
    if projection.track_id == BUNDLE_TRACK_ID:
        execution = candidate_geometry(projection.candidate)
    else:
        launch = context.factory(contract)
        require(launch is not None, f"{size}: an msl-single candidate needs the backend launch resolver")
        execution = launch(projection.candidate)
    # 2. Geometry identity: the request's own instance/ABI resolves the projected execution geometry.
    pair = geometry_pair(execution, None, instance=instance, contract=abi, limits=context.limits)
    resolved = select_geometry(pair).geometry
    require(pair.candidate_error is None, f"{size}: the fixed slice's candidate geometry no longer resolves")
    require(
        resolved.document() == pair.candidate.document(),
        f"{size}: the selected geometry is not the pair's resolved candidate",
    )
    # 3. The lifecycle and the canonical bundle (identity of the build subject).
    lifecycle_factory = _lifecycle_factory(
        projection=projection, resolved=resolved, contract=contract, target_id=context.profile.id
    )
    lifecycle = None if lifecycle_factory is None else lifecycle_factory(resolved)
    bundle = CanonicalBundle(candidate=projection.candidate, pair=pair, lifecycle=lifecycle)
    # 4. The schema-4 build record, audited against the rebuilt bundle and the task identity.  This
    #    is the complete artifact check: exact entries (candidate entries + candidate manifest +
    #    frozen execution geometry), exactly one SOURCE carrying the candidate identity, the native
    #    entry/descriptor audit, the assembled digest and the artifact-set digest binding.
    build = report.get("build")
    require(isinstance(build, Mapping), f"{size}: the report carries no build record")
    audited = verify_evaluator_build(
        build,
        canonical_bundle=bundle,
        identity={"opspec_digest": contract.spec.digest, "lowering_profile_digest": contract.fingerprint()},
    )
    require(audited.shape == "full", f"{size}: a real slice must be a full build, not {audited.shape!r}")
    require(audited.native_digest is not None, f"{size}: a real slice must publish a native executable")
    # The compilation unit and the native descriptor: the contract header + the fixed candidate
    # source, assembled by the production function, decide the assembled digest, the compiled kernel
    # name, the header height and the math mode.  A build that names another translation unit, kernel
    # or device is refused even when its own native/artifact digests were recomputed to match.
    source_text = combined_candidate_source(projection.candidate)
    kernel_contract = metal_contract(
        task_contract=contract, abi=abi, geometry=resolved, target_id=context.profile.id
    )
    assembled, header_lines = kernel_contract.assemble(source_text)
    assembled_sha256 = hashlib.sha256(assembled.encode("utf-8")).hexdigest()
    require(
        audited.document["assembled_sha256"] == assembled_sha256,
        f"{size}: the build assembled different bytes than the fixed candidate over this contract header",
    )
    native = audited.document["native"]
    from tcenv.remote.artifact_bundle import NATIVE_EXECUTABLE_SCHEMA_ID

    require(native.get("schema_id") == NATIVE_EXECUTABLE_SCHEMA_ID, f"{size}: the native schema drifted")
    kernels = [str(stage.kernel) for stage in resolved.active_stages]
    multi_kernel = len(set(kernels)) > 1
    require(
        set(native) == frozenset({"schema_id", "backend_id", "kind", "entry_path", "entry_sha256", "descriptor"}),
        f"{size}: the native executable record is not the fixed shape",
    )
    require(native.get("backend_id") == context.profile.backend, f"{size}: the native executable declares another backend")
    require(native.get("kind") == EXPECTED_NATIVE_KIND, f"{size}: the native executable kind is {native.get('kind')!r}")
    descriptor = native["descriptor"]
    require(
        set(descriptor) == DESCRIPTOR_KEYS | ({"kernel_names", "kernels"} if multi_kernel else frozenset()),
        f"{size}: the native descriptor is not the fixed shape (missing "
        f"{sorted((DESCRIPTOR_KEYS | ({"kernel_names", "kernels"} if multi_kernel else frozenset())) - set(descriptor))}, "
        f"unexpected {sorted(set(descriptor) - (DESCRIPTOR_KEYS | ({"kernel_names", "kernels"} if multi_kernel else frozenset())))})",
    )
    if multi_kernel:
        require(
            descriptor.get("kernel_names") == list(dict.fromkeys(kernels)),
            f"{size}: the descriptor's compiled kernel list is not the frozen geometry's",
        )
    require(
        descriptor.get("kernel_name") == kernel_contract.kernel_name,
        f"{size}: the native executable names kernel {descriptor.get('kernel_name')!r}, not the compiled one",
    )
    archived_header_lines = descriptor.get("header_lines")
    require(
        isinstance(archived_header_lines, int) and not isinstance(archived_header_lines, bool),
        f"{size}: descriptor.header_lines is not an integer header-line count: {archived_header_lines!r}",
    )
    require(archived_header_lines == header_lines, f"{size}: the contract header height drifted")
    require(
        descriptor.get("math") == (launch_math(source_text) or "safe"),
        f"{size}: the compiled math mode is not the fixed source's declaration",
    )
    require(
        descriptor.get("device") == context.profile.calibration_device_name,
        f"{size}: the native executable was compiled for another device",
    )
    require(
        descriptor.get("gpu_family") == context.profile.match_family,
        f"{size}: the native executable carries another GPU family",
    )
    require(descriptor.get("backend") == context.profile.backend, f"{size}: the descriptor names another backend")
    require(descriptor.get("compiler") == EXPECTED_COMPILER, f"{size}: the descriptor names another compiler")
    require(
        str(descriptor.get("language_version") or "") == EXPECTED_MSL_VERSION,
        f"{size}: the descriptor's MSL version is {descriptor.get('language_version')!r}, not the pinned toolchain's "
        f"{EXPECTED_MSL_VERSION!r}",
    )
    require(
        str((report.get("compile") or {}).get("language_version") or "") == EXPECTED_MSL_VERSION,
        f"{size}: the compile block's MSL version is not the pinned toolchain's",
    )
    max_total_threads = descriptor.get("max_total_threads")
    require(
        isinstance(max_total_threads, int) and not isinstance(max_total_threads, bool),
        f"{size}: descriptor.max_total_threads is not an integer thread limit: {max_total_threads!r}",
    )
    require(
        max_total_threads == int(context.profile.raw_limits["max_threads_per_threadgroup"]),
        f"{size}: the pipeline's max threads per threadgroup is not the target's",
    )
    execution_width = descriptor.get("execution_width")
    require(
        isinstance(execution_width, int) and not isinstance(execution_width, bool),
        f"{size}: descriptor.execution_width is not an integer SIMD width: {execution_width!r}",
    )
    require(
        execution_width == EXPECTED_SIMD_WIDTH,
        f"{size}: the pipeline's SIMD width is {execution_width!r}, not the target's",
    )
    require(
        re.search(r"\bthreadgroup\s+\w+\s+\w+\s*\[", source_text) is None,
        f"{size}: the fixed source declares static threadgroup memory, so its descriptor must be checked differently",
    )
    static_tg_bytes = descriptor.get("static_tg_bytes")
    require(
        isinstance(static_tg_bytes, int) and not isinstance(static_tg_bytes, bool),
        f"{size}: descriptor.static_tg_bytes is not an integer threadgroup-memory byte count: {static_tg_bytes!r}",
    )
    require(static_tg_bytes == 0, f"{size}: the kernel claims static threadgroup memory it does not declare")
    require(
        _strict_json_equal(
            descriptor.get("bindings"), _expected_bindings(source_text, kernel_contract.kernel_name, where=size)
        ),
        f"{size}: the native descriptor's reflected bindings are not this kernel signature's",
    )
    require(
        audited.document["evidence"]["loaded"] == [],
        f"{size}: a control-host build claims to have loaded an artifact set it did not produce",
    )
    expected_math = launch_math(source_text) or "safe"
    # The build provenance is the *registered* builder/target identity, rebuilt from production
    # constants + this target profile + the recording host's macOS release — never from the archive.
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.source_builder import METAL_BUILDER_ID, METAL_PROFILE_NAME, METAL_TOOLCHAIN_ID

    provenance = audited.document["provenance"]
    require(set(provenance) == PROVENANCE_KEYS, f"{size}: the build provenance is not the fixed shape")
    macos = platform.mac_ver()[0]
    require(bool(macos), f"{size}: the recording host does not report its macOS release")
    expected_provenance = {
        "builder_id": METAL_BUILDER_ID,
        "builder_version": str(MetalBackend.version),
        "build_host": "control_host",
        "build_profile": METAL_PROFILE_NAME,
        "toolchain": {
            "id": METAL_TOOLCHAIN_ID,
            "version": EXPECTED_MSL_VERSION,
            "sdk_version": f"macos-{macos}",
        },
        "compile_profile": {
            "language": "msl",
            "math": expected_math,
            "device": context.profile.calibration_device_name,
            "gpu_family": context.profile.match_family,
            "unified_memory": True,
        },
    }
    for field, expected in expected_provenance.items():
        require(
            provenance.get(field) == expected,
            f"{size}: build.provenance.{field} is {provenance.get(field)!r}, the registered builder's is {expected!r}",
        )
    require(set(provenance["toolchain"]) == TOOLCHAIN_KEYS, f"{size}: the toolchain block is not the fixed shape")
    require(
        set(provenance["compile_profile"]) == COMPILE_PROFILE_KEYS,
        f"{size}: the compile profile is not the fixed shape",
    )
    # 5. Host Tiling: the plan re-parsed from the projected candidate *and* the contract, with the
    #    exact bytes it writes (N * 4 for this fixture's fp32 scratch).
    host_plan = HostTilingPlan.from_candidate(projection.candidate, contract)
    host_runs: list[dict[str, Any]] = []
    if host_plan is not None:
        plan_sha256 = hashlib.sha256(
            json.dumps([write.document() for write in host_plan.writes], sort_keys=True).encode()
        ).hexdigest()
        writes = sum(write.count * get_dtype(write.dtype).nbytes for write in host_plan.writes)
        require(
            writes == int(P1_SIZES[size]["N"]) * 4,
            f"{size}: the fixed Host Tiling plan writes {writes} bytes, not N*4 for this fp32 scratch",
        )
        step = {
            "path": host_plan.path,
            "source_sha256": host_plan.source_sha256,
            "plan_sha256": plan_sha256,
            "writes": writes,
        }
        # A quick node runs the plan once (correctness); a full node runs it again inside the
        # measured window.  Both are real executions and both must match the plan byte for byte.
        host_runs = [dict(step) for _ in range(2 if mode == "full" else 1)]
    # 6. Workspace spans and fill policy from the ABI/geometry, not from the archive.
    workspace_bytes: dict[str, int] = {}
    for tensor in abi.tensors:
        if tensor.space != "workspace":
            continue
        workspace_bytes[tensor.name] = tensor_span_bytes(
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.strides),
            get_dtype(tensor.dtype).nbytes,
        )
    workspace_zero = {
        str(use["name"]): bool(use["zero"])
        for stage in resolved.document()["stages"]
        for use in stage["workspace"]
    }
    # 7. Track selection, rendering and the schema-4 identity of this request instance.
    track = select_track(projection.track_id).document()
    registry = load_track_registry()
    require(fingerprint(registry.document()) == track["registry_digest"], f"{size}: the Track registry drifted")
    registered = registry.get(track["track_id"], version=int(track["track_version"]))
    require(fingerprint(registered.document()) == track["track_digest"], f"{size}: the Track digest drifted")
    presentation = TrackRenderer.for_track(registered).render(
        backend_id=context.profile.backend,
        instance=instance,
        reference_source=contract.reference.source,
        contract=abi,
        display=contract.display,
        track=registered,
    )
    presentation_document = (
        presentation.as_document()
        if hasattr(presentation, "as_document")
        else {
            "track_id": presentation.track_id,
            "track_version": presentation.track_version,
            "statement": presentation.statement,
            "reference_source": presentation.reference_source,
            "abi_document": presentation.abi_document,
        }
    )
    presentation_sha256 = fingerprint(presentation_document)
    # The target's pinned calibration asset (the same source ``get_calibration`` reads) is the input
    # the report's roofline must be a function of; it is a repo file, not a device probe.
    calibration_path, calibration_source = calibration_file(context.profile.id)
    require(
        calibration_path.is_file(),
        f"{size}: the target's calibration asset is missing ({calibration_source})",
    )
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    require(
        calibration.get("device") == context.profile.calibration_device_name,
        f"{size}: the calibration asset belongs to {calibration.get('device')!r}",
    )
    # The launch block the compiler actually consumed: re-parsed from the fixed candidate source
    # for this instance (grid/threads/threadgroup/workspace bytes), so a report cannot claim a
    # different dispatch discipline than the source declares.
    launch_spec = parse_launch_spec(source_text, launch_constants(instance))
    require(launch_spec is not None, f"{size}: the fixed candidate source carries no taic-launch block")
    launch_digest = launch_spec.digest()
    task = v2_evaluation_task(
        SimpleNamespace(
            contract=contract,
            abi=abi,
            target_id=context.profile.id,
            locator=locator,
            selection=SimpleNamespace(geometry=resolved),
        )
    )
    report_fields = {
        "schema_version": 4,
        "status": "ok",
        "task": contract.op_name,
        "dtype": P1_DTYPE,
        "size": size,
        "mode": mode,
        "seed": FIXED_SEED,
        "target_id": context.profile.id,
        "backend": context.profile.backend,
        "backend_version": context.profile.backend_version,
        "protocol_version": PROTOCOL_VERSION,
        "device_name": context.profile.calibration_device_name,
        "gpu_family": context.profile.match_family,
        "stage": "bench" if mode == "full" else "correctness",
        "message": (
            "compiled, ABI ok, safe, correct and measured"
            if mode == "full"
            else "compiled, ABI ok, safe and correct"
        ),
        "cache_hit": False,
        "compiler_baseline": {},
        "eligible": True,
        "pipeline": {"kind": "v2", "stages": len(resolved.active_stages)},
        "evaluation_fingerprint": _evaluation_fingerprint(task, context.profile.backend, context.profile.id),
        "baseline_implementation_digest": baseline_implementation_digest(task, context.profile.backend),
        "baseline_environment_digest": baseline_environment_digest(
            task, context.profile.backend, context.profile.id
        ),
    }
    _check_timeline(report, mode=mode, size=size)
    _check_report_blocks(
        report,
        contract=contract,
        instance=instance,
        mode=mode,
        size=size,
        output_tensors={
            tensor.name: (tuple(int(value) for value in tensor.shape), str(tensor.dtype))
            for tensor in abi.tensors
            if tensor.space != "workspace" and tensor.access in ("write", "read_write")
        },
    )
    #: The task identity the build record must carry (``report.build``, not the report top level).
    build_fields = {"opspec_digest": contract.spec.digest, "lowering_profile_digest": contract.fingerprint()}
    candidate_entries = getattr(projection.candidate, "entries", ()) or ()
    _check_build_log(
        report,
        size=size,
        kernel_name=kernel_contract.kernel_name,
        math=launch_math(source_text) or "safe",
        language=EXPECTED_MSL_VERSION,
        assembled=assembled,
        header_lines=header_lines,
        source_text=source_text,
        single_source_entry=(
            (str(candidate_entries[0].path), str(candidate_entries[0].sha256))
            if getattr(projection.candidate, "single_source", False) and candidate_entries
            else None
        ),
        descriptor=descriptor,
    )
    candidate_sha256 = projection.candidate.candidate_sha256
    evaluation: dict[str, Any] = {
        "provider": record.provider,
        "runtime_provider": "candidate",
        "execution_published": True,
        "worker_status": None,
        "fallback_reason": reason,
        "fallback_geometry": None,
        "compiled_candidate_sha256": candidate_sha256,
        "expected_geometry_identity": pair.identity,
        "expected_bundle_digest": bundle.bundle_digest,
        "geometry_pair": pair.document(),
        "geometry_identity": pair.identity,
        "bundle_digest": bundle.bundle_digest,
        "native_digest_expected": audited.native_digest,
        "artifact_set_digest": audited.artifact_set_digest,
        "native_digest": audited.native_digest,
        "presentation_sha256": presentation_sha256,
        "effective_correctness_only": bool(contract.correctness_only),
        "task_pack_digest": locator.deployment_pack_digest,
        "locator": locator.document(),
        "track": track,
        "report_status": report_fields["status"],
        "report_digest": fingerprint(dict(report)),
        "report_schema_version": report_fields["schema_version"],
        "report_mode": report_fields["mode"],
        "seed": report_fields["seed"],
        "target_id": report_fields["target_id"],
        "backend": report_fields["backend"],
        "backend_version": report_fields["backend_version"],
        "device_name": report_fields["device_name"],
        "gpu_family": report_fields["gpu_family"],
        "evaluation_fingerprint": report_fields["evaluation_fingerprint"],
        "geometry_origin_path": projection.binding.geometry_origin_path,
        "geometry_origin_sha256": projection.binding.geometry_origin_sha256,
        "geometry_path": projection.binding.geometry_path,
        "projected_geometry_sha256": (
            None
            if projection.binding.geometry_path is None
            else projection.binding.geometry_origin_sha256
        ),
        "geometry": resolved.document(),
        "expected_dispatch": [_stage_signature(stage) for stage in resolved.active_stages],
        "dispatch": [_stage_signature(stage) for stage in resolved.active_stages],
        "dispatch_stages": [int(stage.index) for stage in resolved.active_stages],
        "expected_dispatch_stages": [int(stage.index) for stage in resolved.active_stages],
        "host_tiling": None if host_plan is None else host_plan.document(),
        "host_runs": host_runs,
        "workspace_bytes": workspace_bytes,
        "workspace_zero": workspace_zero,
        "lifecycle": None if lifecycle is None else lifecycle.document(),
        "instance_key": request.instance_key,
        "credits_variant": record.provider != "fallback",
    }
    result_document = {
        "schema_id": EVALUATION_PLAN_SCHEMA_ID,
        "schema_version": EVALUATION_PLAN_SCHEMA_VERSION,
        "locator": locator.document(),
        "track": track,
        "candidate_sha256": candidate_sha256,
        "task_pack_digest": locator.deployment_pack_digest,
        "effective_correctness_only": bool(contract.correctness_only),
        "geometry_identity": pair.identity,
        "bundle_digest": bundle.bundle_digest,
        "presentation_sha256": presentation_sha256,
        "artifact_set_digest": audited.artifact_set_digest,
        "native_digest": audited.native_digest,
        "report": dict(report),
    }
    evaluation["evaluation_identity"] = fingerprint(result_document)
    identities = {
        "domain_plan_digest": record.domain_plan_digest,
        "portfolio_digest": record.portfolio_digest,
        "variant_id": record.variant_id,
        "fallback_id": record.fallback_id,
        "provider": record.provider,
        "fallback_reason": record.fallback_reason,
        "dispatch_record_digest": fingerprint(record.document()),
        "projection_binding_digest": projection.binding.identity,
        "candidate_sha256": candidate_sha256,
        "parent_portfolio_sha256": projection.binding.parent_portfolio_sha256,
        "geometry_origin_path": projection.binding.geometry_origin_path,
        "geometry_origin_sha256": projection.binding.geometry_origin_sha256,
        "geometry_identity": pair.identity,
        "bundle_digest": bundle.bundle_digest,
        "artifact_set_digest": audited.artifact_set_digest,
        "native_digest": audited.native_digest,
        "evaluation_identity": evaluation["evaluation_identity"],
    }
    run_document = {
        "schema_id": PORTFOLIO_RUN_SCHEMA_ID,
        "run_version": PORTFOLIO_RUN_VERSION,
        "record": record.document(),
        "projection": projection.document(),
        "evaluation": evaluation,
        "identities": identities,
        "active_manifest": None,
        "activation_generation": None,
        "performance_claim": None,
        "fallback_credits_variant": False,
    }
    return ReconstructedRun(
        request=request,
        record=record,
        projection=projection,
        candidate_sha256=candidate_sha256,
        pair=pair,
        resolved=resolved,
        bundle_digest=bundle.bundle_digest,
        lifecycle=None if lifecycle is None else lifecycle.document(),
        host_plan=host_plan,
        host_runs=host_runs,
        rotation_sets=_expected_rotation(abi),
        task=task,
        calibration=calibration,
        workspace_bytes=workspace_bytes,
        workspace_zero=workspace_zero,
        artifact_set_digest=audited.artifact_set_digest,
        native_digest=audited.native_digest,
        artifact_entries=audited.entries,
        built_entry_paths=sorted(str(entry["path"]) for entry in audited.entries),
        evaluation=evaluation,
        result_document=result_document,
        run_document=run_document,
        run_identity=fingerprint(run_document),
        launch_digest=launch_digest,
        report_fields=report_fields,
        build_fields=build_fields,
    )


def _rebuilt_bench_stats(candidate: Mapping[str, Any], *, size: str) -> Any:
    """@brief 由原始样本按生产的 ``interleaved`` 公式重建候选统计量。

    ``tcenv.bench.interleaved`` derives ``median_s``/``mean_s``/``min_s``/``cv`` from the recorded
    samples; only the chosen batch size (``iters``) is a measurement parameter and cannot be
    recomputed.  Rebuilding the statistics from the samples turns a contradictory ``candidate``
    block into a refusal instead of a self-consistent claim.

    @param candidate 报告的 ``benchmark.candidate`` 块。
    @param size 诊断用 size。
    @return 生产的 ``BenchStats``。
    @raises AssertionError 样本形状不合法。
    """
    import statistics

    from tcenv.bench import BenchStats

    samples = candidate.get("samples_s")
    require(
        isinstance(samples, list) and bool(samples),
        f"{size}: the candidate measurement block carries no samples",
    )
    values = []
    for index, sample in enumerate(samples):
        require(
            isinstance(sample, (int, float)) and not isinstance(sample, bool) and float(sample) > 0.0,
            f"{size}: candidate sample {index} is not a positive duration",
        )
        values.append(float(sample))
    iters = candidate.get("iters")
    require(
        isinstance(iters, int) and not isinstance(iters, bool),
        f"{size}: candidate.iters is not an integer batch size: {iters!r}",
    )
    reps = candidate.get("reps")
    require(
        isinstance(reps, int) and not isinstance(reps, bool),
        f"{size}: candidate.reps is not an integer repetition count: {reps!r}",
    )
    mean = statistics.fmean(values)
    cv = (statistics.pstdev(values) / mean) if (mean > 0 and len(values) > 1) else 1.0
    return BenchStats(
        median_s=statistics.median(values),
        mean_s=mean,
        min_s=min(values),
        cv=cv,
        reps=len(values),
        iters=iters,
        samples_s=tuple(values),
    )


def _expected_rotation(abi: Any) -> int:
    """@brief 由请求 ABI 重建 v2 计时协议的轮转组数（生产常量与同一公式）。

    The v2 evaluator sizes the rotation ring from the ABI's total buffer bytes so the working set
    exceeds the caches.  That number is a function of the request instance, not a free report field.

    @param abi 请求实例的 ``TensorMemoryContract``。
    @return 轮转组数。
    """
    from tcenv.backends.metal.baselines import RING_MAX, RING_TARGET_BYTES
    from tcenv.dtypes import get_dtype
    from tcenv.track.abi import tensor_span_bytes

    total = 0
    for tensor in abi.tensors:
        total += tensor_span_bytes(
            tuple(int(value) for value in tensor.shape),
            tuple(int(value) for value in tensor.strides),
            get_dtype(tensor.dtype).nbytes,
        )
    return max(1, min(RING_MAX, -(-RING_TARGET_BYTES // max(total, 1))))


def _check_benchmark(
    report: Mapping[str, Any],
    *,
    mode: str,
    size: str,
    correctness_only: bool,
    rotation_sets: int,
    task: Any,
    calibration: Mapping[str, Any],
) -> None:
    """@brief 计时/派遣协议与"只发布 correctness"合同的语义（模式固定为节点属性）。

    @param report schema-4 报告。
    @param mode 期望模式（来自固定节点）。
    @param size 诊断用 size。
    @param correctness_only 请求实例的合同是否只发布 correctness。
    @param rotation_sets 由 ABI 重建的轮转组数。
    @param task 该请求实例的 v2 评测任务（roofline 的输入）。
    @param calibration 目标档案的标定数据（roofline 的输入）。
    @raises AssertionError 协议或信用语义不符。
    """
    from tcenv.backends.metal.profile import roofline
    from tcenv.bench import BENCH_ATTEMPTS, NOISY_CV, attempt_noise

    benchmark = report.get("benchmark") or {}
    require(isinstance(benchmark, Mapping), f"{size}: the benchmark block is not an object")
    if mode == "full":
        require(bool(benchmark), f"{size}: a full run has no benchmark block, so its timing discipline is unverifiable")
        require(
            set(benchmark) == BENCHMARK_BLOCK_KEYS,
            f"{size}: the benchmark block is not the fixed shape (missing "
            f"{sorted(BENCHMARK_BLOCK_KEYS - set(benchmark))}, unexpected {sorted(set(benchmark) - BENCHMARK_BLOCK_KEYS)})",
        )
        baselines = benchmark.get("baselines")
        require(
            isinstance(baselines, Mapping),
            f"{size}: benchmark.baselines is not a JSON object: {baselines!r}",
        )
        require(benchmark.get("measure_serial") is False, f"{size}: the report claims a serial measurement")
        require(
            benchmark.get("protocol") == EXPECTED_BENCHMARK_PROTOCOL,
            f"{size}: the benchmark protocol is not the v2 runtime's protocol text",
        )
        require(
            benchmark.get("dispatch") == EXPECTED_BENCHMARK_DISPATCH,
            f"{size}: the benchmark dispatch is not the v2 runtime recorder's dispatch text",
        )
        archived_rotation = benchmark.get("rotation_sets")
        require(
            isinstance(archived_rotation, int) and not isinstance(archived_rotation, bool),
            f"{size}: benchmark.rotation_sets is not an integer rotation count: {archived_rotation!r}",
        )
        require(
            archived_rotation == rotation_sets,
            f"{size}: the rotation sets are {archived_rotation!r}, not the ABI's {rotation_sets}",
        )
        attempts = benchmark.get("attempts")
        require(
            isinstance(attempts, int) and not isinstance(attempts, bool),
            f"{size}: benchmark.attempts is not an integer attempt count: {attempts!r}",
        )
        require(
            1 <= attempts <= BENCH_ATTEMPTS,
            f"{size}: the report claims {attempts!r} measurement attempts; the harness allows "
            f"1..{BENCH_ATTEMPTS}",
        )
        noise = benchmark.get("noise_cv")
        require(
            isinstance(noise, (int, float))
            and not isinstance(noise, bool)
            and math.isfinite(float(noise))
            and float(noise) >= 0.0,
            f"{size}: the noise verdict is missing",
        )
        require(
            benchmark.get("noisy") is (float(noise) > float(NOISY_CV)),
            f"{size}: the noisy verdict is not implied by its own noise_cv",
        )
        # ``measure_interleaved`` keeps the first quiet attempt (which may follow noisy attempts)
        # and only exhausts the attempts when every one of them is noisy: a noisy report therefore
        # cannot have stopped early.  A quiet report can legitimately have retried after noise.
        require(
            benchmark.get("noisy") is not True or attempts == BENCH_ATTEMPTS,
            f"{size}: a noisy attempt kept after {attempts!r} of {BENCH_ATTEMPTS} attempts is not "
            f"the harness's attempt selection",
        )
        lock_wait = benchmark.get("gpu_lock_wait_s")
        require(
            isinstance(lock_wait, (int, float))
            and not isinstance(lock_wait, bool)
            and math.isfinite(float(lock_wait))
            and float(lock_wait) >= 0.0,
            f"{size}: benchmark.gpu_lock_wait_s is not a finite non-negative duration: {lock_wait!r}",
        )
        require(float(lock_wait) == 0.0, f"{size}: the report claims an unmeasured GPU lock wait")
        require(benchmark.get("candidate_serial_us") is None, f"{size}: a serial candidate time was not measured")
        require(
            benchmark.get("candidate_gpu_us_median") is None,
            f"{size}: a GPU-only candidate time was not measured",
        )
        candidate = benchmark.get("candidate") or {}
        require(set(candidate) == BENCHMARK_STATS_KEYS, f"{size}: the candidate measurement block is not the fixed shape")
        stats = _rebuilt_bench_stats(candidate, size=size)
        # The batch size is a measurement-protocol claim, so it is bounded by the rules production
        # itself obeys (``interleaved`` -> ``choose_iters``), not merely by ``>= 1``: samples are
        # per-launch, so ``iters * min_s`` is one batch's wall clock, and a batch above the budget or
        # a fast kernel batched below its rotation ring is a measurement the harness cannot produce.
        iters = candidate["iters"]
        require(
            1 <= iters <= BENCH_MAX_ITERS,
            f"{size}: the candidate reports a batch of {iters} launches; choose_iters caps it at {BENCH_MAX_ITERS}",
        )
        require(
            float(stats.min_s) * iters <= BENCH_BATCH_WALL_BUDGET_S,
            f"{size}: one batch of {iters} launches at {float(stats.min_s):.3g} s each exceeds the "
            f"{BENCH_BATCH_WALL_BUDGET_S} s watchdog budget choose_iters enforces",
        )
        if float(stats.min_s) <= BENCH_SLOW_LAUNCH_S:
            require(
                iters >= rotation_sets,
                f"{size}: a {float(stats.min_s):.3g} s launch is fast enough for the growing path, so its batch of "
                f"{iters} cannot be below the ABI's {rotation_sets} rotation groups",
            )
        else:
            require(
                iters <= rotation_sets,
                f"{size}: a {float(stats.min_s):.3g} s launch takes the slow path, whose batch is capped at the "
                f"rotation groups ({rotation_sets}); {iters} launches cannot come from it",
            )
        require(
            len(stats.samples_s) == EXPECTED_REPS,
            f"{size}: the candidate reports {len(stats.samples_s)} timing repetitions; this fixed node measures "
            f"{EXPECTED_REPS}",
        )
        require(
            _strict_json_equal(candidate, json.loads(json.dumps(stats.to_json()))),
            f"{size}: the candidate statistics are not the interleaved formula over their own samples "
            f"(median/mean/min/cv/reps/median_us)",
        )
        median_us = float(stats.median_us)
        require(median_us > 0, f"{size}: the full run measured no kernel time")
        require(
            _strict_json_equal(report.get("candidate_us"), float(median_us)),
            f"{size}: report.candidate_us is not the candidate median the benchmark block measured",
        )
        # With no registered baselines the attempt's noise is the candidate's own cv, by the one
        # production definition of attempt noise.
        expected_noise = attempt_noise({"candidate": stats})
        require(
            _strict_json_equal(noise, float(expected_noise)),
            f"{size}: benchmark.noise_cv is {noise!r}, the attempt's candidate cv is {expected_noise!r}",
        )
        # The roofline is a pure function of this task, the measured median and the target's
        # calibration asset — it is a derived conclusion, not a free-form measurement blob.
        expected_roofline = roofline(task, stats.median_s, dict(calibration))
        require(
            _strict_json_equal(report.get("roofline"), expected_roofline),
            f"{size}: the roofline block is not this task/median/calibration's roofline",
        )
    else:
        require(not benchmark, f"{size}: a quick run carries a benchmark block it never measured")
        require(report.get("candidate_us") is None, f"{size}: a quick run reports a candidate time it never measured")
        require(report.get("roofline") in (None, {}), f"{size}: a quick run reports a roofline it never measured")
    if correctness_only:
        # The fixed OpSpec registers no baseline, so the contract publishes correctness only: a report
        # that claims a speedup (or times a baseline) would credit timing that was never measured.
        require(not (benchmark.get("baselines") or {}), f"{size}: a correctness-only contract timed a baseline")
        require(report.get("speedup") is None, f"{size}: a correctness-only contract reported a speedup")
        speedups = report.get("speedups")
        require(isinstance(speedups, Mapping), f"{size}: report.speedups is not a JSON object: {speedups!r}")
        require(not speedups, f"{size}: a correctness-only contract reported speedups")
        baseline_us = report.get("baseline_us")
        require(isinstance(baseline_us, Mapping), f"{size}: report.baseline_us is not a JSON object: {baseline_us!r}")
        require(not baseline_us, f"{size}: a correctness-only contract reported baselines")


def _check_one_evidence(
    document: Mapping[str, Any],
    *,
    node: str,
    size: str,
    label: str,
    mode: str,
    context: FixtureContext,
    reason: str | None,
    document_level: bool = False,
) -> dict[str, Any]:
    """@brief 一条真实切片证据：结构检查 + 从固定输入独立重建后逐字段比对。

    The order is fixed: the archive's own closed facts (schema, node, slice, host) first, then the
    schema-4 report's independent gates, then the full reconstruction (:func:`_reconstruct_run`),
    and only then the archived projection/record/evaluation/run identity are compared to it.  A
    coordinated forgery that recomputes every digest derived from its edits therefore still has to
    match bytes and identities it cannot compute (the fixed candidate sources, the request's ABI,
    the frozen Host Tiling plan and the canonical bundle).

    @param document 证据记录。
    @param node 该记录必须来自的固定 GPU 节点。
    @param size 期望的 size 键。
    @param label 期望的 provider 标签。
    @param mode 该固定节点的评测模式。
    @param context 重建的有限切片上下文（用于独立重放）。
    @param reason 期望的 fallback 原因（候选运行为 ``None``）。
    @param document_level 该记录本身是否就是证据文档（此时它额外带 ``target_id``）。
    @return 该记录的身份集合（候选/几何/artifact 身份与请求实例键）。
    """
    from tcenv.evaluate import EvalReport
    from tests.support.shape_portfolio_metal import P1_SIZES

    expected_keys = EVIDENCE_RUN_KEYS | ({"target_id"} if document_level else frozenset())
    require(
        set(document) == expected_keys,
        f"{size}: the run record is not the fixed evidence shape (missing "
        f"{sorted(expected_keys - set(document))}, unexpected {sorted(set(document) - expected_keys)})",
    )
    require(document.get("schema_id") == "tcenv.portfolio.p1-evidence.1", f"{size}: unknown evidence schema")
    require(document.get("node") == node, f"{size}: the evidence was written by {document.get('node')!r}, not {node!r}")
    require(document.get("size") == size, f"{size}: the evidence is for {document.get('size')!r}")
    require(document.get("provider_label") == label, f"{size}: provider label {document.get('provider_label')!r}")
    _check_host_block(document.get("host"), where=size)
    report = document.get("report") or {}
    require(isinstance(report, Mapping), f"{size}: the evidence carries no schema-4 report")
    parsed = EvalReport.from_json(json.loads(json.dumps(report)))  # closed reader: unknown/missing fields are refused
    require(parsed.status == "ok", f"{size}: the report status is {parsed.status!r}")
    elapsed = report.get("elapsed_s")
    require(
        isinstance(elapsed, (int, float))
        and not isinstance(elapsed, bool)
        and math.isfinite(float(elapsed))
        and float(elapsed) >= 0.0,
        f"{size}: report.elapsed_s is not a finite non-negative duration: {elapsed!r}",
    )
    expected_warnings = list(_expected_device_warnings(context.profile.id))
    require(
        report.get("device_warnings") == expected_warnings,
        f"{size}: report.device_warnings is {report.get('device_warnings')!r}, the production target probe gives {expected_warnings!r}",
    )

    # The whole expectation, rebuilt from the fixed slice (see ReconstructedRun).
    rebuilt = _reconstruct_run(context, size=size, label=label, reason=reason, mode=mode, report=report)
    # 1. The report must be this fixed node's report, not a report for another instance/mode/seed.
    for field, expected in rebuilt.report_fields.items():
        require(
            _strict_json_equal(report.get(field), expected),
            f"{size}: report.{field} is not the fixed slice's exact JSON value: "
            f"{report.get(field)!r} != {expected!r}",
        )
    for field, expected in rebuilt.build_fields.items():
        require(
            _strict_json_equal(report["build"].get(field), expected),
            f"{size}: report.build.{field} is not the fixed slice's exact JSON value: "
            f"{report['build'].get(field)!r} != {expected!r}",
        )
    device = document.get("device") or {}
    expected_device = {
        "device_name": rebuilt.report_fields["device_name"],
        "gpu_family": rebuilt.report_fields["gpu_family"],
        "target_id": rebuilt.report_fields["target_id"],
        "backend": rebuilt.report_fields["backend"],
        "backend_version": rebuilt.report_fields["backend_version"],
        "protocol_version": rebuilt.report_fields["protocol_version"],
        "report_schema_version": rebuilt.report_fields["schema_version"],
    }
    require(_strict_json_equal(device, expected_device), f"{size}: the device block is not this report's device identity")
    require("M4 Max" in str(device["device_name"]), f"{size}: no real Apple M4 Max device in the evidence")
    compile_block = report.get("compile") or {}
    require(
        isinstance(compile_block, Mapping) and set(compile_block) == COMPILE_BLOCK_KEYS,
        f"{size}: the compile block is not the fixed shape",
    )
    for timing in ("compile_s", "pipeline_s"):
        require(
            isinstance(compile_block.get(timing), (int, float))
            and not isinstance(compile_block.get(timing), bool)
            and math.isfinite(float(compile_block[timing]))
            and float(compile_block[timing]) >= 0.0,
            f"{size}: compile.{timing} is not a non-negative duration",
        )
    require(compile_block.get("ok") is True, f"{size}: the compile block is not ok")
    require(
        compile_block.get("launch") == rebuilt.launch_digest,
        f"{size}: the compiled launch block is not the fixed candidate source's launch for this instance",
    )
    require(compile_block.get("math_mode") == "safe", f"{size}: the compiled math mode is not the source's 'safe'")
    _check_benchmark(
        report,
        mode=mode,
        size=size,
        correctness_only=bool(rebuilt.evaluation["effective_correctness_only"]),
        rotation_sets=rebuilt.rotation_sets,
        task=rebuilt.task,
        calibration=rebuilt.calibration,
    )
    # 2. The request/dispatch record: an independent replay of the fixed plan, portfolio and evidence.
    require(
        _strict_json_equal(document["record"], rebuilt.record.document()),
        f"{size}: the archived dispatch record cannot be reproduced by the current plan/portfolio",
    )
    require(
        document["instance_key"] == rebuilt.request.instance_key,
        f"{size}: the evidence instance key is not the fixed request's",
    )
    evidence_n = document.get("n")
    require(
        isinstance(evidence_n, int) and not isinstance(evidence_n, bool),
        f"{size}: document.n is not an integer evidence N: {evidence_n!r}",
    )
    require(evidence_n == int(P1_SIZES[size]["N"]), f"{size}: the evidence N is not the fixture's")
    require(int(rebuilt.request.symbols["N"]) == evidence_n, f"{size}: the request's N is not the evidence N")
    # 3. The sidecar semantics, field for field, from the reconstruction.
    require(
        _strict_json_equal(document["projection"], rebuilt.projection.document()),
        f"{size}: the archived projection is not the projection of the fixed provider",
    )
    if not _strict_json_equal(document["evaluation"], rebuilt.evaluation):
        differing = sorted(
            key for key in set(document["evaluation"]) | set(rebuilt.evaluation)
            if not _strict_json_equal(document["evaluation"].get(key), rebuilt.evaluation.get(key))
        )
        raise AssertionError(f"{size}: the archived evaluation differs from the reconstruction in {differing}")
    require(document["compiled_candidate_sha256"] == rebuilt.candidate_sha256, f"{size}: another candidate was compiled")
    require(
        _strict_json_equal(document["built_entry_paths"], rebuilt.built_entry_paths),
        f"{size}: the built artifact paths are not the audited build's exact entries",
    )
    require(
        document["run_identity"] == rebuilt.run_identity,
        f"{size}: the run identity is not the digest of this run's reconstructed composition document",
    )
    # 4. Identity triples the caller aggregates across runs.
    require(
        rebuilt.candidate_sha256 != rebuilt.projection.binding.parent_portfolio_sha256,
        f"{size}: the child reuses the parent portfolio SHA",
    )
    if label == "fallback":
        require(report.get("speedup") is None, f"{size}: a fallback run reported a speedup")
    return {
        "label": label,
        "candidate_sha256": rebuilt.candidate_sha256,
        "geometry_identity": rebuilt.pair.identity,
        "artifact_set_digest": rebuilt.artifact_set_digest,
        "instance_key": rebuilt.request.instance_key,
    }


#: Closed key set of one archived refusal (everything a reader needs to replay it, nothing else).
REFUSAL_KEYS = frozenset(
    {
        "schema_id",
        "slice",
        "request",
        "record",
        "portfolio_digest",
        "domain_plan_digest",
        "built_entry_paths",
        "projection",
        "result",
        "evaluation",
        "builder_calls",
        "evaluator_calls",
    }
)


def _host_facts() -> dict[str, str]:
    """@brief 本录制主机的生产 ``platform`` 事实（文档级与 run 级共用）。

    ``tests/backends/metal/test_shape_portfolio_gpu.py`` writes ``HOST`` from
    ``platform.processor()/machine()/system()``; the verifier reproduces those calls instead of
    accepting a prefix, so a host block that names another machine cannot be archived.

    @return chip/machine/system 三元组。
    """
    return {"chip": platform.processor(), "machine": platform.machine(), "system": platform.system()}


def _check_host_block(host: Any, *, where: str) -> None:
    """@brief host 块必须是固定形状，且逐字段等于本机的生产事实。

    @param host 证据里的 host 块。
    @param where 诊断位置。
    @raises AssertionError 形状或取值不符。
    """
    require(isinstance(host, Mapping) and set(host) == HOST_BLOCK_KEYS, f"{where}: the host block is not the fixed shape")
    facts = _host_facts()
    for key, expected in facts.items():
        require(
            str(host.get(key)) == expected,
            f"{where}: the host {key} is {host.get(key)!r}, this recording host's is {expected!r}",
        )
    require(facts["system"] == "Darwin", f"{where}: the evidence was not produced on a Mac")


def _expected_out_of_domain_request(context: FixtureContext, *, instance_key: str) -> dict[str, Any]:
    """@brief 独立重建一条语义域外请求（threshold 实例只改 dtype 或 size）。

    The fixed GPU node builds both out-of-domain requests from the *authoritative* threshold instance
    and changes exactly one field (``dtype`` for the undeclared dtype, ``size`` for the undeclared
    size), so the verifier can rebuild them instead of trusting the archived request document.

    @param context 标准切片上下文。
    @param instance_key 期望的实例键（``fp32/threshold/golden`` 或 ``fp16/not_a_declared_size/golden``）。
    @return 规范化后的请求文档。
    """
    from tcenv.portfolio.dispatch import DispatchRequest

    base, _instance = _expected_request(context, size="threshold")
    document = dict(base.document())
    if str(instance_key).startswith("fp32/"):
        document["dtype"] = "fp32"
    else:
        document["size"] = "not_a_declared_size"
    document["instance_key"] = str(instance_key)
    return DispatchRequest.parse(document).document()


def _check_refusal(
    entry: Mapping[str, Any],
    *,
    context: FixtureContext,
    slice_name: str,
    reason: str | None,
    where: str,
    expected_request: Mapping[str, Any],
    allowed_extra: frozenset[str] = frozenset(),
) -> None:
    """@brief 一条归档拒绝的闭集校验 + 独立重放（未声明原因 / 语义域外）。

    A refusal is evidence too: it must say which request it refused, under which portfolio, and
    what it refused to do.  The archived record is replayed through the resolver, so a hand-edited
    outcome/provider or a hand-written projection cannot survive.

    @param entry 归档的拒绝记录。
    @param context 该拒绝所属的上下文（narrow/超限 portfolio 或标准 portfolio）。
    @param slice_name 期望的 slice 名。
    @param reason ``"declared_not_registered"`` 或 ``None``（语义域外）。
    @param where 诊断位置。
    @param expected_request 从固定切片独立重建的请求文档（拒绝的 request/record 都必须等于它）。
    @param allowed_extra 该记录允许的额外键（例如 ``instance_key``）。
    @raises AssertionError 结构、身份或语义不符。
    """
    from tcenv.portfolio.dispatch import DispatchRequest

    expected = REFUSAL_KEYS | set(allowed_extra)
    missing = sorted(expected - set(entry))
    extra = sorted(set(entry) - expected)
    require(not missing, f"{where}: the refusal is missing {missing}")
    require(not extra, f"{where}: the refusal carries unexpected fields {extra}")
    require(entry["schema_id"] == "tcenv.portfolio.p1-evidence.1", f"{where}: unknown evidence schema")
    require(entry["slice"] == slice_name, f"{where}: the refusal is {entry['slice']!r}")
    require(entry["built_entry_paths"] == [], f"{where}: a refusal built artifacts")
    for key in ("projection", "result", "evaluation"):
        require(entry[key] is None, f"{where}: a refusal carries a {key}")
    require(entry["builder_calls"] == 0, f"{where}: a refusal called the builder")
    require(entry["evaluator_calls"] == 0, f"{where}: a refusal called the evaluator")
    require(entry["portfolio_digest"] == context.portfolio.digest, f"{where}: the refusal names another portfolio")
    require(entry["domain_plan_digest"] == context.plan.digest, f"{where}: the refusal names another domain plan")
    require(
        _strict_json_equal(dict(entry["request"]), dict(expected_request)),
        f"{where}: the refused request is not the fixed slice's request (op/symbols/attrs/digests are the slice's)",
    )
    request = DispatchRequest.parse(expected_request)
    record = entry["record"]
    require(record.get("schema_id") == "tcenv.portfolio.dispatch-record.1", f"{where}: the record schema drifted")
    require(
        _strict_json_equal(dict(record.get("request") or {}), dict(expected_request)),
        f"{where}: the dispatch record refused a different request than the archived refusal",
    )
    replayed = context.resolve(request)
    require(_strict_json_equal(replayed.document(), record), f"{where}: the refusal cannot be replayed by the current resolver")
    if reason == "declared_not_registered":
        require(record["outcome"] == "variant_unavailable", f"{where}: the outcome is {record['outcome']!r}")
        require(record["provider"] is None, f"{where}: a refusal selected a provider")
        require(record["fallback_reason"] is None, f"{where}: a refusal claims a fallback reason")
        require(
            record["preflight"]["status"] == "resource_unavailable",
            f"{where}: the refusal did not record the resource preflight",
        )
        require(
            "does not declare resource_unavailable" in (record["detail"] or ""),
            f"{where}: the refusal does not say the fallback never registered the reason",
        )
    else:
        require(record["outcome"] == "unsupported_contract_capability", f"{where}: the outcome is {record['outcome']!r}")
        require(record["provider"] is None, f"{where}: an out-of-domain refusal selected a provider")
        require(record["fallback_reason"] is None, f"{where}: an out-of-domain refusal fell back")
        require(record["preflight"]["status"] == "not_required", f"{where}: an out-of-domain refusal preflighted")
        if "instance_key" in entry:
            require(str(entry["instance_key"]) in (record["detail"] or ""), f"{where}: the detail names another point")


def _narrow_over_limit_context(context: FixtureContext) -> FixtureContext:
    """@brief 超限 variant + 未声明 ``resource_unavailable`` 的 fallback 上下文（拒绝负例）。

    @param context 标准上下文。
    @return narrow portfolio 的上下文。
    """
    from tcenv.portfolio.candidate import parse_portfolio
    from tests.support.shape_portfolio_metal import over_limit_entries, portfolio_document

    document = portfolio_document(context.plan.digest)
    document["fallback"]["reasons"] = ["outside_candidate_coverage"]
    portfolio = parse_portfolio(over_limit_entries(document), domain_plan=context.plan)
    return context.with_portfolio(portfolio)


def _over_limit_context(context: FixtureContext) -> FixtureContext:
    """@brief 超限 variant 负例所用的上下文（variant-b 的源内 launch 超过线程上限）。

    @param context 标准上下文。
    @return 超限 portfolio 的上下文。
    """
    from tcenv.portfolio.candidate import parse_portfolio
    from tests.support.shape_portfolio_metal import over_limit_entries, portfolio_document

    portfolio = parse_portfolio(over_limit_entries(portfolio_document(context.plan.digest)), domain_plan=context.plan)
    return context.with_portfolio(portfolio)


def verify_evidence(directory: Path, context: FixtureContext) -> dict[str, Any]:
    """@brief 结构/身份/dispatch/provider 检查一组证据文件，并返回身份汇总。

    @param directory 证据目录。
    @param context 重建的有限切片上下文。
    @return 汇总（每个 provider 的子候选 SHA、几何/artifact 身份集合、覆盖的节点）。
    """
    require(directory.is_dir(), f"{directory}: the evidence directory is missing")
    present = {path.name for path in directory.glob("*.json")}
    require(present == set(EVIDENCE_FILES), f"evidence file set differs: {sorted(present)}")
    labels: dict[str, set[str]] = {}
    geometries: set[str] = set()
    artifacts: set[str] = set()
    subjects: set[tuple[str, str]] = set()
    nodes: list[str] = []
    refusals = 0
    over_limit = _over_limit_context(context)
    narrow_over_limit = _narrow_over_limit_context(context)
    for name, node, runs, kind in EXPECTED_EVIDENCE:
        document = json.loads((directory / name).read_text(encoding="utf-8"))
        run_context = over_limit if kind == "over_limit" else context
        expected_keys = EVIDENCE_DOCUMENT_KEYS[name]
        require(
            set(document) == expected_keys,
            f"{name}: the evidence document is not the fixed shape (missing "
            f"{sorted(expected_keys - set(document))}, unexpected {sorted(set(document) - expected_keys)})",
        )
        if name in EVIDENCE_GROUP_KEYS:
            require(
                set(document["runs"]) == EVIDENCE_GROUP_KEYS[name]
                if "runs" in document
                else set(key for key, *_ in runs) == EVIDENCE_GROUP_KEYS[name],
                f"{name}: the run groups are not the fixed nodes' groups",
            )
        require(document.get("schema_id") == "tcenv.portfolio.p1-evidence.1", f"{name}: unknown evidence schema")
        require(document.get("node") == node, f"{name}: the evidence was not written by {node!r}")
        require(document.get("slice") == EXPECTED_SLICES[name], f"{name}: unexpected slice {document.get('slice')!r}")
        require(document.get("target_id") == "apple-m4-max", f"{name}: the evidence names another target")
        _check_host_block(document.get("host"), where=name)
        nodes.append(node)
        for key, size, label, reason, mode in runs:
            run_document = _source_of(document, key)
            # The per-run identity sets are aggregated from the **reconstruction**, not from the
            # archive: a forged sidecar cannot contribute a geometry/artifact/subject identity.
            entry = _check_one_evidence(
                run_document,
                node=node,
                size=size,
                label=label,
                mode=mode,
                context=run_context,
                reason=reason,
                document_level=(key == ""),
            )
            labels.setdefault(label, set()).add(entry["candidate_sha256"])
            geometries.add(entry["geometry_identity"])
            artifacts.add(entry["artifact_set_digest"])
            subjects.add((entry["instance_key"], label))
        if name == "single_source_overflow.json":
            require("refused" in document, f"{name}: the declared-but-unregistered refusal is not archived")
            _check_refusal(
                document["refused"],
                context=narrow_over_limit,
                slice_name="single-source-launch-overflow-refused",
                reason="declared_not_registered",
                where=f"{name}:refused",
                expected_request=_expected_request(narrow_over_limit, size="below")[0].document(),
            )
            refusals += 1
        elif name == "out_of_domain.json":
            require("refused" in document, f"{name}: no out-of-domain refusal is archived")
            cases = document["refused"]
            require(isinstance(cases, list) and len(cases) == 2, f"{name}: expected two out-of-domain refusals")
            keys = sorted(str(case.get("instance_key")) for case in cases)
            require(keys == ["fp16/not_a_declared_size/golden", "fp32/threshold/golden"], f"{name}: unexpected points {keys}")
            for index, case in enumerate(cases):
                _check_refusal(
                    case,
                    context=context,
                    slice_name="semantic-domain-outside-request",
                    reason=None,
                    where=f"{name}:refused[{index}]",
                    expected_request=_expected_out_of_domain_request(
                        context, instance_key=str(case.get("instance_key") or "")
                    ),
                    allowed_extra=frozenset({"instance_key"}),
                )
            refusals += len(cases)
    require(set(labels) == {"variant-a-bundle", "variant-b-single", "fallback"}, "a provider is missing from the evidence")
    require(all(len(shas) == 1 for shas in labels.values()), "one provider changed its candidate SHA between runs")
    require(len({next(iter(shas)) for shas in labels.values()}) == 3, "two providers share a candidate SHA")
    # Seven unique geometries: the size-agnostic generic fallback resolves to the same one at N=63
    # and N=100, and each of the five per-element runs owns its own instance geometry.
    require(len(geometries) == 7, f"expected 7 instance-resolved geometries, got {len(geometries)}")
    require(len(artifacts) == 7, f"expected 7 artifact sets, got {len(artifacts)}")
    # Eight (instance, provider) subjects: N=63 is resolved twice, once by the specialised variant
    # and once by the generic fallback after its resource divert.
    require(len(subjects) == 8, f"expected 8 instance/provider subjects, got {len(subjects)}")
    require(sorted(nodes) == sorted(FIXED_GPU_NODE_IDS), "the fixed GPU nodes are not all represented")
    print(
        f"OK: evidence ({len(EVIDENCE_FILES)} files, {len(FIXED_GPU_NODE_IDS)} fixed nodes): real M4 Max device, "
        f"schema-4 ok reports, dispatch parity, 3 provider identities, 7 geometries/artifact sets, 8 subjects, "
        f"{refusals} replayed refusals"
    )
    return {"labels": labels, "geometries": geometries, "artifacts": artifacts, "nodes": nodes, "refusals": refusals}


class _Append:
    """@brief 负例变异：把一项追加到列表（"增加一个未选中的 artifact"）。"""

    def __init__(self, item: Any) -> None:
        """@brief 保存要追加的项。

        @param item 追加值。
        """
        self.item = item


def _rebalance_forgery(node: Any) -> None:
    """@brief 攻击者的自洽步骤：把它能重算的每个派生摘要都重算一遍。

    Models the *strongest* forger the verifier must survive.  After editing any field it recomputes
    the artifact-set digest from the edited manifest with the production function, the native digest
    from the edited native descriptor, and the report/result/composition identities from the edited
    documents, so the archive is once again internally consistent and no "some digest differs"
    shortcut can reject it.  The point of the negative group is that these recomputations still
    cannot produce the fixed candidate bytes, the request ABI, the frozen Host Tiling plan, the
    canonical bundle or the exact artifact entries the verifier rebuilds independently.

    @param node 证据文档（或其子树）。
    """
    from tcenv.remote.artifact_bundle import BUNDLE_SCHEMA_ID, NativeExecutable, artifact_set_digest

    if isinstance(node, Mapping):
        if isinstance(node.get("evaluation"), Mapping) and isinstance(node.get("report"), Mapping):
            report = node["report"]
            evaluation = node["evaluation"]
            build = report.get("build")
            artifacts = build.get("artifacts") if isinstance(build, Mapping) else None
            if isinstance(artifacts, Mapping) and isinstance(artifacts.get("entries"), list):
                digest = artifact_set_digest(BUNDLE_SCHEMA_ID, artifacts["entries"])
                artifacts["artifact_set_digest"] = digest
                provenance = build.get("provenance")
                if isinstance(provenance, Mapping):
                    provenance["artifact_set_digest"] = digest
                evaluation["artifact_set_digest"] = digest
                evidence_block = build.get("evidence")
                if isinstance(evidence_block, Mapping):
                    # A thorough forger also keeps the evidence block consistent with the manifest,
                    # so the archive survives the shape validator and only the independent rebuild
                    # can reject it.
                    evidence_block["artifacts"] = json.loads(json.dumps(artifacts))
                    if isinstance(provenance, Mapping):
                        evidence_block["provenance"] = json.loads(json.dumps(provenance))
                node["built_entry_paths"] = sorted(str(entry["path"]) for entry in artifacts["entries"])
            native = build.get("native") if isinstance(build, Mapping) else None
            if isinstance(native, Mapping):
                try:
                    native_digest = NativeExecutable(
                        backend_id=str(native["backend_id"]),
                        kind=str(native["kind"]),
                        entry_path=str(native["entry_path"]),
                        entry_sha256=str(native["entry_sha256"]),
                        descriptor=dict(native["descriptor"]),
                    ).digest
                except Exception:  # noqa: BLE001 - the production type refuses this archive outright
                    native_digest = None
                if native_digest is not None:
                    build["native_digest"] = native_digest
                    evaluation["native_digest"] = native_digest
                    evaluation["native_digest_expected"] = native_digest
            evaluation["report_digest"] = fingerprint(report)
            evaluation["evaluation_identity"] = fingerprint(_rebuild_result_document(node, report))
            node["run_identity"] = fingerprint(_rebuild_run_document(node))
        for value in list(node.values()):
            _rebalance_forgery(value)
    elif isinstance(node, list):
        for value in node:
            _rebalance_forgery(value)


def _benchmark_forgery(mutations: Mapping[str, Any]) -> Any:
    """@brief 构造一个"把样本派生的量全部重算"之后的测量伪造。

    The strongest timing attack: change the samples (and/or the attempt count), then make every
    derived quantity agree again — the benchmark statistics by the production ``interleaved``
    formulas, ``noise_cv`` by ``attempt_noise``, ``noisy`` from that noise, ``candidate_us`` from the
    median, ``roofline`` from the task/median/calibration — and let the caller rebalance every digest
    afterwards.  Only the slice's own facts (its repetition count, the attempt rule, the exact log
    bytes) can refuse it.

    @param mutations ``benchmark`` 段的直接改写（``candidate`` 内为字段改写，其余为整键替换）。
    @return 一个 ``(document, *, task, calibration)`` 动作。
    """

    def apply(document: dict[str, Any], *, task: Any, calibration: Mapping[str, Any]) -> None:
        """@brief 先按攻击者的意图改测量，再把所有派生量配平。

        @param document 伪造中的证据文档。
        @param task 该请求实例的 v2 评测任务。
        @param calibration 目标档案的标定数据。
        """
        from tcenv.backends.metal.profile import roofline
        from tcenv.bench import NOISY_CV, attempt_noise

        report = document["report"]
        benchmark = report["benchmark"]
        benchmark["candidate"].update(mutations.get("candidate") or {})
        for key, value in mutations.items():
            if key != "candidate":
                benchmark[key] = value
        stats = _rebuilt_bench_stats(benchmark["candidate"], size="forgery")
        benchmark["candidate"] = json.loads(json.dumps(stats.to_json()))
        noise = float(attempt_noise({"candidate": stats}))
        benchmark["noise_cv"] = noise
        benchmark["noisy"] = noise > float(NOISY_CV)
        report["candidate_us"] = float(stats.median_us)
        report["roofline"] = roofline(task, stats.median_s, dict(calibration))

    return apply


def verify_forgery_negatives() -> None:
    """@brief 伪造证据的 verifier 负例：改过的副本必须被 ``verify_evidence`` 拒绝。

    Three groups run by default:

    * ``tamperings`` — one field each, no digest repair (a reader who trusts a posted digest);
    * ``coordinated`` — several fields changed *and* every digest the archive derives from them
      recomputed by :func:`_rebalance_forgery` (build artifact set, native, report, result,
      composition), which is the only interesting attack: the archive stays self-consistent.
      Production canonical validation / the closed reader rejects some cases first; the remaining
      cases require independent reconstruction.  The group covers the build artifact set
      (frozen geometry entry + its native audit, an extra unselected artifact, SOURCE, projected
      geometry, native entry, task identity), Host Tiling (executed plan/content/writes/path/shas,
      the plan document, a dropped run), and the sidecar semantics (projection identity, provider,
      fallback reason, report status/schema/mode, geometry path/origin sha, effective correctness
      only, workspace span, lifecycle instance binding, geometry pair, dispatch signature);
    * the archived refusals, replayed through the resolver.

    @raises AssertionError 某个伪造被接受（或负例本身没能改动证据）。
    """
    import copy
    import shutil

    def path_to(node: Any, path: str) -> tuple[Any, Any]:
        """@brief 走到 ``a.b[2].c`` 的父对象与末键。

        @param node 文档根。
        @param path 点分路径（整数段视为列表下标）。
        @return ``(parent, leaf)``。
        """
        parts: list[Any] = []
        for segment in path.split("."):
            parts.append(int(segment) if segment.isdigit() else segment)
        parent: Any = node
        for key in parts[:-1]:
            parent = parent[key]
        return parent, parts[-1]

    zero = "0" * 64
    tamperings: list[tuple[str, str, Any]] = [
        ("threshold.json", "evaluation.evaluation_identity", zero),
        ("threshold.json", "evaluation.report_digest", zero),
        ("threshold.json", "evaluation.bundle_digest", zero),
        ("threshold.json", "evaluation.expected_bundle_digest", zero),
        ("threshold.json", "evaluation.native_digest", zero),
        ("threshold.json", "evaluation.native_digest_expected", zero),
        ("threshold.json", "run_identity", zero),
        ("threshold.json", "evaluation.lifecycle.instance_key", zero),
        ("threshold.json", "evaluation.lifecycle.cache_scope.instance_key", zero),
        ("threshold.json", "projection.candidate_sha256", zero),
        ("threshold.json", "report.build.native.descriptor", {"backend": "metal"}),
        ("single_source.json", "runs.tail.evaluation.evaluation_identity", zero),
        ("single_source.json", "runs.large.evaluation.geometry_identity", zero),
        ("boundaries.json", "above.record.detail", "the projection changed the execution geometry"),
        ("fallback.json", "evaluation.credits_variant", True),
        ("fallback.json", "evaluation.fallback_reason", "not_a_declared_reason"),
        ("single_source_overflow.json", "run.record.fallback_reason", "outside_candidate_coverage"),
        # The reviewer's falsification: rewrite the archived refusal so it looks like a run.
        ("single_source_overflow.json", "refused.record.outcome", "fallback"),
        ("single_source_overflow.json", "refused.record.provider", "fallback"),
        ("single_source_overflow.json", "refused.projection", "forged"),
        ("single_source_overflow.json", "refused.result", "forged"),
        ("single_source_overflow.json", "refused.evaluation", "forged"),
        ("single_source_overflow.json", "refused.builder_calls", 1),
        ("single_source_overflow.json", "refused.evaluator_calls", 1),
        ("single_source_overflow.json", "refused.built_entry_paths", ["device/variant_b.metal"]),
        ("single_source_overflow.json", "refused.portfolio_digest", zero),
        ("single_source_overflow.json", "refused.record.preflight.status", "ok"),
        # ... and the out-of-domain refusals.
        ("out_of_domain.json", "refused.0.record.outcome", "fallback"),
        ("out_of_domain.json", "refused.1.projection", "forged"),
        ("out_of_domain.json", "refused.0.record.provider", "candidate"),
        ("out_of_domain.json", "refused.0.request.size", "above"),
        ("out_of_domain.json", "refused.1.instance_key", "fp32/threshold/golden"),
        # A field nobody checks would be an evidence claim outside this gate: the records are closed.
        ("threshold.json", "forged_document_field", "P1 is accepted"),
        ("boundaries.json", "below.forged_run_field", "accepted"),
        ("single_source.json", "runs.forged_group", {"size": "tail"}),
        # The document-level identity of a refusal-only file is checked too.
        ("out_of_domain.json", "slice", "fallback-hole"),
        ("single_source_overflow.json", "slice", "bundle-host-tiling"),
        ("out_of_domain.json", "host", {"chip": "arm", "machine": "x86_64", "system": "Linux"}),
    ]
    #: Index of each build entry in the archived artifact manifest (path-sorted, as the builder
    #: writes it): 0 candidate manifest (SOURCE), 1 device source, 2 frozen execution geometry,
    #: 3 projected geometry entry, 4 Host Tiling plan.
    ENTRIES = "report.build.artifacts.entries"
    coordinated: list[tuple[str, tuple[tuple[str, Any], ...]]] = [
        # -- 1. build evidence: every digest derived from the edit is recomputed ------------------
        (
            # The frozen geometry entry and the native audit are edited *together*, so the native
            # digest still matches its own descriptor: only the canonical bundle rebuild catches it.
            "threshold.json",
            (
                (f"{ENTRIES}.2.sha256", zero),
                ("report.build.native.descriptor.execution_geometry.sha256", zero),
            ),
        ),
        (
            # An artifact nobody selected: an extra payload in the artifact set (digest rebalanced).
            "threshold.json",
            ((f"{ENTRIES}", _Append({
                "path": "payload/extra.bin",
                "role": "AUXILIARY",
                "media_type": "application/octet-stream",
                "size_bytes": 1,
                "sha256": zero,
            })),),
        ),
        (
            # The SOURCE artifact is retyped (bytes unchanged, artifact identity not).
            "threshold.json",
            ((f"{ENTRIES}.0.media_type", "text/plain"),),
        ),
        (
            # The SOURCE artifact is renamed.
            "threshold.json",
            ((f"{ENTRIES}.0.path", "candidate_manifest_2.json"),),
        ),
        (
            # The *projected* geometry entry no longer carries the parent's declared geometry bytes.
            "threshold.json",
            ((f"{ENTRIES}.3.sha256", zero),),
        ),
        (
            # The native executable names a kernel that was never in this artifact set.
            "threshold.json",
            (
                ("report.build.native.entry_path", "device/generic.metal"),
                ("report.build.native.entry_sha256", zero),
            ),
        ),
        (
            # The native entry digest is not the listed artifact's digest.
            "threshold.json",
            (("report.build.native.entry_sha256", zero),),
        ),
        (
            # The build claims a different compilation unit, at both ends of the record (shape-valid:
            # the assembled digest agrees with its own native descriptor).
            "threshold.json",
            (
                ("report.build.assembled_sha256", zero),
                ("report.build.native.descriptor.assembled_sha256", zero),
            ),
        ),
        (
            # The native executable names another kernel of this slice.
            "threshold.json",
            (("report.build.native.descriptor.kernel_name", "generic"),),
        ),
        (
            # The compiled math mode is not the source's declaration.
            "threshold.json",
            (("report.build.native.descriptor.math", "fast"),),
        ),
        (
            # The contract header height is not the assembled header's.
            "threshold.json",
            (("report.build.native.descriptor.header_lines", 0),),
        ),
        (
            # The native executable was compiled for another device.
            "threshold.json",
            (("report.build.native.descriptor.device", "Apple M9"),),
        ),
        (
            # The task identity the build claims is another task's.
            "threshold.json",
            (("report.build.opspec_digest", zero),),
        ),
        (
            "threshold.json",
            (("report.build.lowering_profile_digest", zero),),
        ),
        (
            # The build claims another candidate's provenance while the artifact set stays put.
            "threshold.json",
            (
                ("report.build.candidate_sha256", zero),
                ("report.build.provenance.source_sha256", zero),
            ),
        ),
        # -- 2. Host Tiling: the plan, its bytes, its content and every executed run ---------------
        (
            # One byte written instead of N*4.
            "threshold.json",
            (("evaluation.host_runs.0.writes", 1), ("evaluation.host_runs.1.writes", 1)),
        ),
        (
            # Wrong write count (63 elements instead of the instance's 64).
            "threshold.json",
            (("evaluation.host_tiling.writes.0.count", 63),),
        ),
        (
            "threshold.json",
            (
                ("evaluation.host_runs.0.writes", 252),
                ("evaluation.host_runs.1.writes", 252),
            ),
        ),
        (
            # Zeroed source/plan digests of the executed plan.
            "threshold.json",
            (
                ("evaluation.host_runs.0.source_sha256", zero),
                ("evaluation.host_runs.0.plan_sha256", zero),
                ("evaluation.host_runs.1.source_sha256", zero),
                ("evaluation.host_runs.1.plan_sha256", zero),
            ),
        ),
        (
            # A plan document that was never parsed from the candidate.
            "threshold.json",
            (("evaluation.host_tiling", {"forged": True}),),
        ),
        (
            "threshold.json",
            (("evaluation.host_runs.0.path", "host/forged_tiling.json"),),
        ),
        (
            # A dropped execution: the full run really ran the plan twice.
            "threshold.json",
            (("evaluation.host_runs", "drop"),),
        ),
        (
            # Host work claimed by a provider that ships none.
            "fallback.json",
            (("evaluation.host_tiling", {"schema_id": "tcenv.metal.host-tiling.1"}),),
        ),
        # -- 3. sidecar semantics: projection/report/geometry/contract/instance -------------------
        (
            "threshold.json",
            (("projection.identity", zero),),
        ),
        (
            # A fallback reason that is not a declared reason of this portfolio.
            "fallback.json",
            (
                ("evaluation.fallback_reason", "not_a_declared_reason"),
                ("record.fallback_reason", "not_a_declared_reason"),
            ),
        ),
        (
            "threshold.json",
            (("evaluation.report_status", "timeout"),),
        ),
        (
            "threshold.json",
            (("evaluation.report_schema_version", 3), ("report.schema_version", 3)),
        ),
        (
            "threshold.json",
            (("evaluation.report_mode", "quick"), ("report.mode", "quick")),
        ),
        (
            "threshold.json",
            (("evaluation.seed", 0), ("report.seed", 0)),
        ),
        (
            # A geometry path that was never projected.
            "threshold.json",
            (("evaluation.geometry_path", "host/forged_geometry.json"),),
        ),
        (
            # The original geometry content SHA is not the parent's declared geometry bytes.
            "threshold.json",
            (
                ("evaluation.geometry_origin_sha256", zero),
                ("evaluation.projected_geometry_sha256", zero),
            ),
        ),
        (
            # The contract publishes correctness only; claiming otherwise (and the timing credit that
            # comes with it) is not this contract's result.
            "threshold.json",
            (
                ("evaluation.effective_correctness_only", False),
                ("report.speedup", 2.5),
                ("report.speedups", {"baseline": 2.5}),
                ("report.baseline_us", {"baseline": 1.0}),
            ),
        ),
        (
            # `effective_correctness_only=true` where the contract demands timing: a fixture with no
            # baseline cannot claim a correctness-only run, so this is the field's other direction.
            "threshold.json",
            (("evaluation.effective_correctness_only", not True),),
        ),
        (
            # Provider mismatch: the sidecar and the projection rename the fallback to a candidate.
            "fallback.json",
            (
                ("evaluation.provider", "candidate"),
                ("record.provider", "candidate"),
                ("projection.provider", "candidate"),
            ),
        ),
        (
            # A workspace span that is not the ABI's.
            "threshold.json",
            (("evaluation.workspace_bytes", {"scratch": 4}),),
        ),
        (
            "threshold.json",
            (("evaluation.workspace_zero", {"scratch": True}),),
        ),
        (
            # Lifecycle bound to another instance (both the binding and the cache scope).
            "threshold.json",
            (
                ("evaluation.lifecycle.instance_key", zero),
                ("evaluation.lifecycle.cache_scope.instance_key", zero),
            ),
        ),
        (
            "threshold.json",
            (("evaluation.lifecycle.candidate_sha256", zero),),
        ),
        (
            # A geometry pair whose resolved grid is not this instance's.
            "threshold.json",
            (
                ("evaluation.geometry_pair.candidate_resolution.stages.0.grid", [1, 1, 1]),
                ("evaluation.geometry.stages.0.grid", [1, 1, 1]),
                ("evaluation.dispatch.0.grid", [1, 1, 1]),
                ("evaluation.expected_dispatch.0.grid", [1, 1, 1]),
            ),
        ),
        (
            # A dispatch that never ran, in a report that claims it did.
            "single_source.json",
            (
                ("runs.large.evaluation.dispatch", []),
                ("runs.large.evaluation.dispatch_stages", []),
            ),
        ),
        (
            # The task instance the report names is another size.
            "boundaries.json",
            (("above.report.size", "below"),),
        ),
        (
            # The compiled launch block claims another dispatch discipline.
            "threshold.json",
            (("report.compile.launch", "math=safe\nlaunch variant_a grid=1,1,1 threads=1,1,1"),),
        ),
        (
            # Device identity claimed by the evidence but not by this report.
            "threshold.json",
            (("device.gpu_family", "Apple8"),),
        ),
        # -- 4. native executable / build evidence / report conclusions ---------------------------
        (
            # Another backend's executable.
            "threshold.json",
            (("report.build.native.backend_id", "cuda"),),
        ),
        (
            # Reflected bindings that no kernel of this slice declared: another dtype/name/access.
            "threshold.json",
            (
                (
                    "report.build.native.descriptor.bindings",
                    [
                        {
                            "name": "a",
                            "index": 0,
                            "kind": "buffer",
                            "dtype_code": 3,
                            "access": "readwrite",
                            "elem_size": 4,
                            "used": True,
                            "tg_bytes": None,
                        }
                    ],
                ),
            ),
        ),
        (
            # The same rows in another order (signature order is the reflection's identity).
            "threshold.json",
            (("report.build.native.descriptor.bindings", "swap"),),
        ),
        (
            # One binding too many.
            "threshold.json",
            (("report.build.native.descriptor.bindings", "duplicate"),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.execution_width", 1),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.max_total_threads", 1),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.static_tg_bytes", 999),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.backend", "cuda"),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.compiler", "clang"),),
        ),
        (
            "threshold.json",
            (("report.build.native.descriptor.language_version", "9.9"),),
        ),
        (
            # A compiled-kernel list the frozen geometry does not have.
            "threshold.json",
            (("report.build.native.descriptor.kernel_names", ["variant_a", "variant_b"]),),
        ),
        (
            # A descriptor field this build never produces.
            "threshold.json",
            (("report.build.native.descriptor.forged", 1),),
        ),
        (
            # A control-host build did not load an artifact set it produced in this process.
            "threshold.json",
            (
                (
                    "report.build.evidence.loaded",
                    [{"path": "payload/unselected.so", "sha256": zero}],
                ),
            ),
        ),
        (
            "threshold.json",
            (
                (
                    "report.build.evidence.loaded",
                    [{"path": "device/variant_a.metal", "sha256": zero}],
                ),
            ),
        ),
        (
            # A passing case rewritten to a failing one while the block still says ok.
            "threshold.json",
            (
                ("report.correctness.cases.0.ok", False),
                ("report.correctness.cases.0.n_bad", 64),
                ("report.correctness.cases.0.worst.got", -999),
            ),
        ),
        (
            # Case evidence that does not belong to this instance/seed.
            "threshold.json",
            (
                ("report.correctness.cases.0.seed", 8),
                ("report.correctness.cases.0.n_elems", 1),
            ),
        ),
        (
            # A duplicated case is not the fixed node's case list.
            "threshold.json",
            (("report.correctness.cases", "duplicate"),),
        ),
        (
            # `correctness.ok` no longer implied by its own cases.
            "threshold.json",
            (("report.correctness.ok", False),),
        ),
        (
            # Safety: a violation while the block claims ok.
            "threshold.json",
            (("report.safety.violations", ["forged unsafe execution"]),),
        ),
        (
            "threshold.json",
            (("report.safety.nondeterminism", ["forged nondeterminism"]),),
        ),
        (
            "threshold.json",
            (("report.safety.ok", False),),
        ),
        (
            # ABI: a violation while the block claims ok.
            "threshold.json",
            (("report.abi.violations", ["forged ABI violation"]),),
        ),
        (
            "threshold.json",
            (("report.abi", {"ok": True, "violations": [], "forged": 1}),),
        ),
        (
            # The correctness conclusion, tolerance and coverage are the contract's, not free text.
            "threshold.json",
            (("report.correctness.tolerance.rtol", 0.5),),
        ),
        (
            "threshold.json",
            (("report.correctness.coverage.kind", "v1-legacy"),),
        ),
        (
            "threshold.json",
            (("report.correctness.cases.0.forged", 1),),
        ),
        (
            "threshold.json",
            (("report.correctness.cases.0.worst.forged", 1),),
        ),
        (
            # A formal-verification result this backend cannot produce.
            "threshold.json",
            (("report.verification.status", "verified"),),
        ),
        (
            "threshold.json",
            (("report.verification.counterexample", {"case": "normal"}),),
        ),
        (
            # A "verified" statement inside the block that still says unsupported.
            "threshold.json",
            (("report.verification.message", "formally verified: equivalence proven by an SMT solver"),),
        ),
        (
            # Run facts that are the fixed node's: a cached build, a baseline, another summary text.
            "threshold.json",
            (("report.cache_hit", True),),
        ),
        (
            "threshold.json",
            (("report.compiler_baseline", {"baseline": 1}),),
        ),
        (
            "threshold.json",
            (("report.message", "compiled, ABI ok, safe and correct"),),
        ),
        (
            # A host identity field nobody recorded, and another platform claim.
            "threshold.json",
            (("host.forged", 1),),
        ),
        (
            "threshold.json",
            (("host.chip", "x86_64"),),
        ),
        (
            # A baseline timed by a contract that registers none.
            "threshold.json",
            (("report.benchmark.baselines", {"naive": {"median_us": 1.0}}),),
        ),
        (
            # A passing case whose reported error is above its own tolerance, a v1 baseline
            # comparison the v2 runtime never ran, and a legacy input digest it never computed.
            "threshold.json",
            (
                ("report.correctness.cases.0.scaled_err", 2.0),
                ("report.correctness.cases.0.baseline_scaled_err", 0.5),
                ("report.correctness.cases.0.input_sha256", "forged-input-digest"),
            ),
        ),
        (
            # A worst-element index outside the output tensor.
            "threshold.json",
            (("report.correctness.cases.0.worst.index", [1000000]),),
        ),
        (
            # Safety "validation" evidence nobody produced.
            "threshold.json",
            (("report.safety.validation", ["forged validation"]),),
        ),
        # -- 5. build provenance / toolchain / timing protocol / timeline --------------------------
        (
            # The MSL version is a toolchain fact: changing both places it appears must not help.
            "threshold.json",
            (
                ("report.build.native.descriptor.language_version", "9.9"),
                ("report.compile.language_version", "9.9"),
            ),
        ),
        (
            # The registered builder identity, not a request field.
            "threshold.json",
            (("report.build.provenance.builder_id", "attacker-controlled-builder"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.build_profile", "attacker-profile"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.build_host", "evaluator"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.toolchain.id", "attacker-toolchain"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.toolchain.version", "9.9"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.toolchain.sdk_version", "linux-1"),),
        ),
        (
            # The compile profile is the target device's, not free text.
            "threshold.json",
            (("report.build.provenance.compile_profile.device", "Apple M9"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.compile_profile.math", "fast"),),
        ),
        (
            "threshold.json",
            (("report.build.provenance.compile_profile.forged", 1),),
        ),
        (
            # A coverage statement that claims a universal proof.
            "threshold.json",
            (("report.correctness.coverage.scope", "formally proven for every input"),),
        ),
        (
            # A rank-2 index for a rank-1 output.
            "threshold.json",
            (("report.correctness.cases.0.worst.index", [0, 0]),),
        ),
        (
            # Two different "worst" errors in one case result.
            "threshold.json",
            (
                ("report.correctness.cases.0.scaled_err", 0.25),
                ("report.correctness.cases.0.worst.scaled_err", 0.75),
            ),
        ),
        (
            # Negation that keeps the keywords is still not the v2 protocol text.
            "threshold.json",
            (("report.benchmark.protocol", "read_write restored workspace timing: attacker did none of this"),),
        ),
        (
            "threshold.json",
            (("report.benchmark.dispatch", "v2 runtime recorder not used; concurrent encoder not used"),),
        ),
        (
            # The rotation policy is the ABI's, not any positive number.
            "threshold.json",
            (("report.benchmark.rotation_sets", 1),),
        ),
        (
            # A timeline that skips stages the path really entered.
            "threshold.json",
            (("report.timeline", [{"stage": "compile", "status": "ok", "duration_s": 0.001}]),),
        ),
        (
            "threshold.json",
            (
                (
                    "report.timeline",
                    [
                        {"stage": "device_check", "status": "ok", "duration_s": 0.001},
                        {"stage": "bench", "status": "ok", "duration_s": 0.001},
                    ],
                ),
            ),
        ),
        (
            # Compile/benchmark blocks are closed and semantic, not "timing plus anything".
            "threshold.json",
            (("report.compile.forged", 1),),
        ),
        (
            # A zero-length "compile log" cannot be the log of a real compile.
            "threshold.json",
            (("report.build.evidence.log", {"sha256": zero, "size_bytes": 0}),),
        ),
        (
            "threshold.json",
            (("report.build.evidence.log.size_bytes", 4096),),
        ),
        # -- 6. measured-evidence formulas: BenchStats, noise/attempts, roofline, worst element -----
        (
            # The samples are the measurement; median/mean/min/cv must be the interleaved formula
            # over exactly these samples.
            "threshold.json",
            (("report.benchmark.candidate.samples_s", [1.0, 2.0, 3.0]),),
        ),
        (
            "threshold.json",
            (("report.benchmark.candidate.median_s", 1.0),),
        ),
        (
            "threshold.json",
            (
                ("report.benchmark.candidate.mean_s", 1.0),
                ("report.benchmark.candidate.min_s", 2.0),
            ),
        ),
        (
            "threshold.json",
            (("report.benchmark.candidate.samples_s", [4.18e-06, 4.15e-06, 4.09e-06, 4.0e-06]),),
        ),
        (
            # With no baselines the attempt noise is the candidate's own cv (one production
            # definition), and the noisy verdict follows from it.
            "threshold.json",
            (
                ("report.benchmark.candidate.cv", 100.0),
                ("report.benchmark.noise_cv", 0.0),
                ("report.benchmark.noisy", False),
            ),
        ),
        (
            "threshold.json",
            (("report.benchmark.noise_cv", 0.0),),
        ),
        (
            # The harness attempts a measurement at most BENCH_ATTEMPTS times.
            "threshold.json",
            (("report.benchmark.attempts", 999),),
        ),
        (
            # The roofline is rebuilt from this task, the measured median and the target's
            # calibration asset — a derived conclusion with a closed key set.
            "threshold.json",
            (
                ("report.roofline.bound", "compute"),
                ("report.roofline.forged", {"claimed": True}),
            ),
        ),
        (
            "threshold.json",
            (("report.roofline.achieved_gbps", 1.0),),
        ),
        (
            # Device warnings are production target/probe facts, not free-form volatile diagnostics.
            "threshold.json",
            (("report.device_warnings", ["attacker-controlled: target limits were not checked"]),),
        ),
        (
            # Even volatile wall-clock fields must be finite non-negative numbers.
            "threshold.json",
            (("report.elapsed_s", -1.0),),
        ),
        (
            # The worst element must be the fixed case/seed's reference value, and its error must be
            # the production tolerance-normalized formula applied to the reported got/want.
            "threshold.json",
            (
                ("report.correctness.cases.0.worst.got", 123456.0),
                ("report.correctness.cases.0.worst.want", 123456.0),
            ),
        ),
        (
            "threshold.json",
            (("report.correctness.cases.0.worst.got", 123456.0),),
        ),
        # -- 7. compile log bytes, host identity, refusal request identity -------------------------
        (
            # The log is a deterministic rendering of this build, so its digest is recomputable.
            "threshold.json",
            (("report.build.evidence.log.sha256", zero),),
        ),
        (
            "threshold.json",
            (("report.build.evidence.duration_ms", 1),),
        ),
        (
            # The host block names this recording host exactly, at run level ...
            "threshold.json",
            (("host.machine", "arm64-attacker-controlled"),),
        ),
        (
            "threshold.json",
            (("host.chip", "arm-attacker"),),
        ),
        (
            # ... and at document level (including a refusal-only document).
            "out_of_domain.json",
            (("host.machine", "arm64-attacker-controlled"),),
        ),
        (
            "out_of_domain.json",
            (("host.chip", "arm-attacker"),),
        ),
        (
            "single_source_overflow.json",
            (("host.machine", "x86_64"),),
        ),
        (
            # The refused request is the fixed slice's request, not a document the archive chose:
            # op/symbols/attrs and both digests are rewritten in both copies.
            "out_of_domain.json",
            (
                ("refused.0.request.op_name", "attacker_controlled_operation"),
                ("refused.0.request.symbols", {"N": 999}),
                ("refused.0.request.attrs", {"forged": 1}),
                ("refused.0.request.semantic_instance_digest", zero),
                ("refused.0.request.target_abi_digest", zero),
                ("refused.0.record.request.op_name", "attacker_controlled_operation"),
                ("refused.0.record.request.symbols", {"N": 999}),
                ("refused.0.record.request.attrs", {"forged": 1}),
                ("refused.0.record.request.semantic_instance_digest", zero),
                ("refused.0.record.request.target_abi_digest", zero),
            ),
        ),
        (
            "out_of_domain.json",
            (
                ("refused.1.request.dtype", "fp16"),
                ("refused.1.request.size", "threshold"),
                ("refused.1.request.symbols", {"N": 64}),
                ("refused.1.record.request.dtype", "fp16"),
                ("refused.1.record.request.size", "threshold"),
            ),
        ),
        (
            "single_source_overflow.json",
            (
                ("refused.request.symbols", {"N": 999}),
                ("refused.record.request.symbols", {"N": 999}),
            ),
        ),
        (
            # The batch size is a protocol claim bounded by ``choose_iters``: a billion launches in one
            # batch is not a measurement this harness can produce, and every digest is rebalanced.
            "threshold.json",
            (
                ("report.benchmark.candidate.iters", 10**9),
            ),
        ),
        (
            # The same rule from the other side: this kernel's 4 µs launch takes the growing path, whose
            # batch cannot be smaller than the ABI's 16 rotation groups.
            "threshold.json",
            (
                ("report.benchmark.candidate.iters", 1),
            ),
        ),
        (
            # A small, in-tolerance lie about the actual candidate output.  The attacker recomputes
            # the production scaled error and every digest, so formula-only validation would pass.
            "threshold.json",
            (
                ("report.correctness.cases.0.worst.got", 1.7001953125),
                ("report.correctness.cases.0.worst.scaled_err", 0.2189957600423968),
                ("report.correctness.cases.0.scaled_err", 0.2189957600423968),
            ),
        ),
    ]
    functional: list[tuple[str, str, Any]] = [
        # -- 8. measured-evidence formulas, fully rebalanced by the attacker ----------------------
        (
            # A noisy attempt cannot have stopped after one try (``measure_interleaved`` only keeps a
            # noisy attempt after exhausting BENCH_ATTEMPTS).  Every derived quantity (stats, noise,
            # noisy, candidate_us, roofline, all digests) is recomputed to agree with the samples.
            "threshold.json",
            "noisy attempt kept after 1 try",
            _benchmark_forgery(
                {"candidate": {"samples_s": [1e-6, 2e-6, 3e-6]}, "attempts": 1},
            ),
        ),
        (
            # The fixed node measures three repetitions; a fourth quiet sample contradicts the slice.
            "threshold.json",
            "four repetitions on a three-repetition node",
            _benchmark_forgery(
                {"candidate": {"samples_s": [4.0e-6, 4.1e-6, 4.2e-6, 4.3e-6]}},
            ),
        ),
    ]
    reasoned_coordinated: list[tuple[str, str, Any, str]] = [
        (
            "threshold.json",
            "report.benchmark.baselines",
            [],
            "benchmark.baselines is not a JSON object",
        ),
        (
            "threshold.json",
            "report.speedups",
            [],
            "report.speedups is not a JSON object",
        ),
        (
            "threshold.json",
            "report.baseline_us",
            [],
            "report.baseline_us is not a JSON object",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.allow",
            True,
            "case 0 allow is not a finite number",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.scaled_err",
            False,
            "case 0 scaled_err is not a finite number",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.worst.scaled_err",
            False,
            "worst.scaled_err is not a finite number",
        ),
        (
            "threshold.json",
            "report.pipeline.stages",
            True,
            "report.pipeline is not the fixed slice's exact JSON value",
        ),
        (
            "threshold.json",
            "report.build.native.descriptor.static_tg_bytes",
            False,
            "static_tg_bytes is not an integer threadgroup-memory byte count",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.n_bad",
            -0.5,
            "not an integer bad-element count",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.nan_count",
            0.5,
            "not an integer NaN count",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.unwritten",
            0.5,
            "not an integer unwritten-element count",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.n_elems",
            64.5,
            "not an integer element count",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.seed",
            7926.9,
            "not an integer seed",
        ),
        (
            "threshold.json",
            "report.correctness.cases.0.worst.index",
            [False],
            "not an integer worst-element index",
        ),
        (
            "threshold.json",
            "n",
            "64",
            "not an integer evidence N",
        ),
        (
            "threshold.json",
            "report.benchmark.rotation_sets",
            16.5,
            "not an integer rotation count",
        ),
        (
            "threshold.json",
            "report.benchmark.candidate.iters",
            "integral-float",
            "candidate.iters is not an integer batch size",
        ),
        (
            "threshold.json",
            "report.benchmark.candidate.reps",
            "integral-float",
            "candidate.reps is not an integer repetition count",
        ),
    ]
    reasoned_functional: list[tuple[str, str, Any, str]] = [
        (
            # The attempt count is a number of loop iterations, not any value int() can truncate.
            "threshold.json",
            "fractional attempt count",
            _benchmark_forgery({"attempts": 1.5}),
            "benchmark.attempts is not an integer attempt count",
        ),
        (
            # bool is an int subclass in Python: without a JSON-type check False compares equal to
            # the required 0.0 and disappears from the stable timing projection.
            "threshold.json",
            "boolean GPU-lock wait",
            _benchmark_forgery({"gpu_lock_wait_s": False}),
            "benchmark.gpu_lock_wait_s is not a finite non-negative duration",
        ),
    ]
    total = (
        len(tamperings) + len(coordinated) + len(functional) + len(reasoned_functional) + len(reasoned_coordinated)
    )
    rejected = 0
    context = fixture_context()
    baseline = _reconstruct_run(
        context,
        size="threshold",
        label="variant-a-bundle",
        reason=None,
        mode="full",
        report=json.loads((EVIDENCE / "threshold.json").read_text(encoding="utf-8"))["report"],
    )
    for name, path, value in tamperings:
        with tempfile.TemporaryDirectory(prefix="p1-forgery-") as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(EVIDENCE, directory)
            target = directory / name
            document = json.loads(target.read_text(encoding="utf-8"))
            parent, leaf = path_to(document, path)
            require(parent.get(leaf) != value if isinstance(parent, Mapping) else parent[leaf] != value,
                    f"the forgery negative cannot change {name}:{path}")
            parent[leaf] = value
            target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
            try:
                verify_evidence(directory, context)
            except Exception:  # noqa: BLE001 - any refusal (schema error or assertion) is the point
                rejected += 1
            else:
                raise AssertionError(f"verify_evidence accepted a forged evidence document: {name}:{path}")
    for name, mutations in coordinated:
        with tempfile.TemporaryDirectory(prefix="p1-forgery-") as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(EVIDENCE, directory)
            target = directory / name
            document = json.loads(target.read_text(encoding="utf-8"))
            before = json.dumps(document, sort_keys=True)
            for path, value in mutations:
                parent, leaf = path_to(document, path)
                if value == "swap":
                    require(isinstance(parent[leaf], list) and len(parent[leaf]) > 1,
                            f"the forgery negative cannot swap {name}:{path}")
                    parent[leaf] = [parent[leaf][1], parent[leaf][0], *parent[leaf][2:]]
                elif value == "duplicate":
                    require(isinstance(parent[leaf], list) and bool(parent[leaf]),
                            f"the forgery negative cannot duplicate {name}:{path}")
                    parent[leaf] = [copy.deepcopy(parent[leaf][0]) for _ in range(2)]
                elif isinstance(value, _Append):
                    require(isinstance(parent[leaf], list), f"the forgery negative cannot append to {name}:{path}")
                    parent[leaf].append(value.item)
                elif value == "drop":
                    require(isinstance(parent[leaf], list) and len(parent[leaf]) > 1,
                            f"the forgery negative cannot drop from {name}:{path}")
                    parent[leaf] = parent[leaf][:-1]
                else:
                    parent[leaf] = value
            # A negative that changed nothing is not a negative (and would silently pass).
            require(
                before != json.dumps(document, sort_keys=True),
                f"the coordinated forgery negative cannot change {name}: {[path for path, _ in mutations]}",
            )
            # The forger's next step: make every derived digest agree with the edited archive again,
            # for every run group the document contains (single run, below/above, runs.*).
            _rebalance_forgery(document)
            target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
            # Stable live comparison is a second, independent gate for fields that are stable by
            # definition.  Warnings cannot hide in the volatile projection, and the fixed
            # candidate's actual output cannot change either.
            stable_attack_paths = {"report.device_warnings", "report.correctness.cases.0.worst.got"}
            if any(path in stable_attack_paths for path, _ in mutations):
                try:
                    compare_live_archived(EVIDENCE, directory)
                except Exception:  # noqa: BLE001 - stable comparison must reject this attack too
                    pass
                else:
                    raise AssertionError(
                        f"stable live comparison accepted a coordinated forgery: {name}: "
                        f"{[path for path, _ in mutations]}"
                    )
            try:
                verify_evidence(directory, context)
            except Exception:  # noqa: BLE001 - any refusal is the point
                rejected += 1
            else:
                raise AssertionError(
                    "verify_evidence accepted a coordinated forgery: "
                    f"{name}:{[path for path, _ in mutations]}"
                )
    for name, path, forgery in functional:
        with tempfile.TemporaryDirectory(prefix="p1-forgery-") as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(EVIDENCE, directory)
            target = directory / name
            document = json.loads(target.read_text(encoding="utf-8"))
            before = json.dumps(document, sort_keys=True)
            forgery(document, task=baseline.task, calibration=baseline.calibration)
            require(
                before != json.dumps(document, sort_keys=True),
                f"the functional forgery negative cannot change {name}:{path}",
            )
            _rebalance_forgery(document)
            target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
            try:
                verify_evidence(directory, context)
            except Exception:  # noqa: BLE001 - any refusal is the point
                rejected += 1
            else:
                raise AssertionError(f"verify_evidence accepted a rebalanced measurement forgery: {name}:{path}")
    for name, path, value, expected_reason in reasoned_coordinated:
        with tempfile.TemporaryDirectory(prefix="p1-forgery-") as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(EVIDENCE, directory)
            target = directory / name
            document = json.loads(target.read_text(encoding="utf-8"))
            before = json.dumps(document, sort_keys=True)
            parent, leaf = path_to(document, path)
            if value == "integral-float":
                require(
                    isinstance(parent[leaf], int) and not isinstance(parent[leaf], bool),
                    f"the integral-float forgery needs an integer at {name}:{path}",
                )
                parent[leaf] = float(parent[leaf])
            else:
                parent[leaf] = value
            require(
                before != json.dumps(document, sort_keys=True),
                f"the reasoned coordinated forgery cannot change {name}:{path}",
            )
            _rebalance_forgery(document)
            target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
            try:
                verify_evidence(directory, context)
            except AssertionError as exc:
                require(
                    expected_reason in str(exc),
                    f"{name}:{path} was rejected for the wrong reason: {exc}; expected {expected_reason!r}",
                )
            except Exception as exc:
                raise AssertionError(
                    f"{name}:{path} raised {type(exc).__name__} instead of the required semantic refusal"
                ) from exc
            else:
                raise AssertionError(f"verify_evidence accepted a reasoned coordinated forgery: {name}:{path}")
            rejected += 1
    for name, path, forgery, expected_reason in reasoned_functional:
        with tempfile.TemporaryDirectory(prefix="p1-forgery-") as temporary:
            directory = Path(temporary) / "evidence"
            shutil.copytree(EVIDENCE, directory)
            target = directory / name
            document = json.loads(target.read_text(encoding="utf-8"))
            before = json.dumps(document, sort_keys=True)
            forgery(document, task=baseline.task, calibration=baseline.calibration)
            require(
                before != json.dumps(document, sort_keys=True),
                f"the reasoned forgery negative cannot change {name}:{path}",
            )
            _rebalance_forgery(document)
            target.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
            try:
                verify_evidence(directory, context)
            except AssertionError as exc:
                require(
                    expected_reason in str(exc),
                    f"{name}:{path} was rejected for the wrong reason: {exc}; expected {expected_reason!r}",
                )
            except Exception as exc:
                raise AssertionError(
                    f"{name}:{path} raised {type(exc).__name__} instead of the required semantic refusal"
                ) from exc
            else:
                raise AssertionError(f"verify_evidence accepted a reasoned measurement forgery: {name}:{path}")
            rejected += 1
    require(rejected == total, f"only {rejected}/{total} tampered copies were rejected")
    print(f"OK: forged-evidence negatives ({rejected}/{total} tampered copies rejected, every coordinated forgery rebalanced)")


def compare_live_archived(live: Path, archived: Path) -> None:
    """@brief 实时重跑与归档证据逐字段相等（排除耗时字段与三个由报告派生的摘要）。

    The three excluded digests (``report_digest``/``evaluation_identity``/``run_identity``) hash the
    report, timings included, so they cannot be equal across runs by construction; each archive's
    value is instead rebuilt by :func:`_reconstruct_run`; :func:`_check_one_evidence` compares
    the archived composition identity with ``rebuilt.run_identity``.

    @param live 实时证据目录。
    @param archived 归档目录。
    """
    require(live.is_dir(), f"{live}: the live evidence directory is missing")
    for name in EVIDENCE_FILES:
        archived_document = json.loads((archived / name).read_text(encoding="utf-8"))
        live_document = json.loads((live / name).read_text(encoding="utf-8"))
        archived_runs = _evidence_runs(archived_document)
        live_runs = _evidence_runs(live_document)
        for key in ("schema_id", "node", "slice", "target_id"):
            require(
                _strict_json_equal(archived_document.get(key), live_document.get(key)),
                f"{name}: {key} drifted between the archive and the live run",
            )
        require(
            _strict_json_equal(archived_document.get("host") or {}, live_document.get("host") or {}),
            f"{name}: the host identity drifted between the archive and the live run",
        )
        require(len(archived_runs) == len(live_runs), f"{name}: live/archived run count differs")
        for index, (archived_run, live_run) in enumerate(zip(archived_runs, live_runs)):
            left = _stable_evidence(archived_run)
            right = _stable_evidence(live_run)
            require(
                _strict_json_equal(left, right),
                f"{name}[{index}]: the live run is not the archived run; differing keys: "
                f"{sorted(key for key in set(left) | set(right) if not _strict_json_equal(left.get(key), right.get(key)))}",
            )
        if archived_runs:
            require(
                _strict_json_equal(archived_runs[0].get("host") or {}, live_runs[0].get("host") or {}),
                f"{name}: the host identity drifted between the archive and the live run",
            )
        # Refusals are timing-free evidence and are compared in full, not folded into the stable
        # projection above (they carry a request + a dispatch record and nothing measured).  A
        # refusal-only document has no run at all, so this is also its only live comparison.
        require(
            _strict_json_equal(archived_document.get("refused"), live_document.get("refused")),
            f"{name}: the archived refusal is not the refusal this run produced",
        )
        if not archived_runs:
            require("refused" in archived_document, f"{name}: a document with no run and no refusal")
    print(
        "OK: live == archived on every stable field of the GPU evidence (timing fields and the three "
        "report-derived digests, which are rebuilt from the fixed slice per archive)"
    )


def run_fixed_gpu(evidence_dir: Path) -> str:
    """@brief 运行 P1 固定 GPU 节点并把实时证据写到给定目录。

    @param evidence_dir 实时证据目录。
    @return pytest 摘要行。
    """
    return _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "gpu",
            *FIXED_GPU_NODE_IDS,
        ],
        where="fixed P1 GPU pytest",
        expect_passed=FIXED_GPU_EXPECTED_PASSED,
        env={"SHAPE_PORTFOLIO_P1_EVIDENCE_DIR": str(evidence_dir)},
    )


def run_a4_anchors() -> str:
    """@brief 重跑 A4 几何锚点（真实设备）。

    @return pytest 摘要行。
    """
    return _pytest(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-n",
            "0",
            "-p",
            "no:cacheprovider",
            "-m",
            "gpu",
            *A4_GEOMETRY_ANCHOR_NODE_IDS,
        ],
        where="A4 geometry anchors",
        expect_passed=len(A4_GEOMETRY_ANCHOR_NODE_IDS),
    )


def record_evidence() -> None:
    """@brief 运行固定 GPU 节点并把证据写入 P1/evidence（显式动作）。"""
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        path = EVIDENCE / name
        if path.exists():
            path.unlink()
    summary = run_fixed_gpu(EVIDENCE)
    missing = [name for name in EVIDENCE_FILES if not (EVIDENCE / name).is_file()]
    if missing:
        raise SystemExit(f"the GPU nodes did not write the expected evidence: {missing}")
    with tempfile.TemporaryDirectory(prefix="p1-record-") as temporary:
        verify_evidence(EVIDENCE, fixture_context())
    print(f"OK: recorded {len(EVIDENCE_FILES)} evidence files\n{summary}")


def main() -> None:
    """@brief 验收入口。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true", help="rewrite manifest.sha256 (after --record-evidence)")
    parser.add_argument("--record-evidence", action="store_true", help="run the GPU nodes and archive their evidence")
    parser.add_argument(
        "--phase",
        choices=("all", "cpu", "gpu"),
        default="all",
        help="cpu/gpu are diagnostic partial phases; only the default run can accept P1",
    )
    args = parser.parse_args()
    if args.record_evidence:
        record_evidence()
        return
    if args.write_manifest:
        write_manifest()
        return
    partial = args.phase != "all"
    verify_manifest()
    verify_acceptance_baseline()
    verify_historical()
    verify_runner_static()
    verify_anchor_files()
    verify_baseline_tool()
    verify_count_gates()
    verify_bench_constants()
    verify_p0_manifest()
    verify_p0_epoch_extension()
    if args.phase in ("all", "cpu"):
        with tempfile.TemporaryDirectory(prefix="p1-fixture-") as temporary:
            verify_fixture_routing(Path(temporary))
        with tempfile.TemporaryDirectory(prefix="p1-p0-") as temporary:
            run_p0_regression(Path(temporary))
        run_a6_preserved()
        run_fixed_cpu()
    if args.phase in ("all", "gpu"):
        # Evidence forgery checks include the production device-warning probe; keep them out of the
        # CPU-only diagnostic phase so that phase never opens a Metal device.
        verify_forgery_negatives()
        with tempfile.TemporaryDirectory(prefix="p1-context-") as temporary:
            context = fixture_context()
        verify_evidence(EVIDENCE, context)
        # The archive is never its own witness for the toolchain/device facts: probe the real device
        # and require the verifier's pinned constants to match it.
        verify_toolchain_probe(context)
        with tempfile.TemporaryDirectory(prefix="p1-live-") as temporary:
            live = Path(temporary)
            summary = run_fixed_gpu(live)
            verify_evidence(live, context)
            compare_live_archived(live, EVIDENCE)
            print(f"OK: fixed P1 GPU pytest ({FIXED_GPU_EXPECTED_PASSED} nodes, no skip)\n{summary}")
        anchor_summary = run_a4_anchors()
        print(f"OK: A4 geometry anchors ({len(A4_GEOMETRY_ANCHOR_NODE_IDS)} nodes)\n{anchor_summary}")
        print(
            "OK: platform "
            + json.dumps(
                {
                    "machine": platform.machine(),
                    "system": platform.system(),
                    "chip": platform.processor(),
                    "python": platform.python_version(),
                },
                sort_keys=True,
            )
        )
    if partial:
        print(
            f"WARNING: --phase {args.phase} is a diagnostic partial phase; both phases are required, so P1 is NOT "
            "accepted by this run"
        )
        raise SystemExit(2)
    print(
        f"OK: P1 acceptance ({FIXED_CPU_EXPECTED_PASSED} fixed CPU nodes + {FIXED_GPU_EXPECTED_PASSED} fixed GPU nodes, "
        "live==archived evidence, P0/A4/A6 anchors)"
    )


if __name__ == "__main__":
    main()
