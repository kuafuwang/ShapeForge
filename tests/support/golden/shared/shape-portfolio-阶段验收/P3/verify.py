#!/usr/bin/env python3
"""P3-A/B/C acceptance: performance model + two finite-domain tasks + P2 anchors + stage baseline."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

from tcenv.identity import is_hex64
from tcenv.portfolio_evidence.projection import strict_json_equal
from tcenv.portfolio_performance.performance import (
    MAX_REGRESSION_RATIO,
    PerformanceReport,
    ShardMeasurement,
)
from tcenv.track import REPO_ROOT
from tests.support.boundaries import _AcceptanceRoot, rebind_acceptance_root
from tests.support.shape_portfolio_metal import EXPECTED_PROVIDER, P1_OP, P1_SIZES

ROOT = _AcceptanceRoot(REPO_ROOT)
P3 = Path(__file__).resolve().parent
BASELINE = P3 / "acceptance-baseline.json"
MANIFEST = P3 / "manifest.sha256"
PLAN = "docs/架构/shape-portfolio实施计划.md"
PRODUCTION = "tcenv/portfolio_performance/performance.py"
PACKAGE_INIT = "tcenv/portfolio_performance/__init__.py"
METAL_PERFORMANCE = "tcenv/backends/metal/portfolio_performance.py"
METAL_BASELINES = "tcenv/backends/metal/baselines.py"
METAL_V2_RUNTIME = "tcenv/backends/metal/v2_runtime.py"
TEST = "tests/shared/test_shape_portfolio_performance.py"
GPU_TEST = "tests/backends/metal/test_shape_portfolio_performance_gpu.py"
RUNTIME = "tcenv/backends/metal/runtime.py"
RUNTIME_TEST = "tests/backends/metal/test_launch.py"
NATIVE_PROD = "tcenv/backends/metal/native_batch.py"
NATIVE_SOURCE = "tcenv/backends/metal/native_batch.m"
BACKEND = "tcenv/backends/metal/backend.py"
ADAPTER = "tcenv/backends/metal/track_adapter.py"
NATIVE_TEST = "tests/backends/metal/test_native_batch.py"
STEEL_FIXTURE = "tests/support/golden/metal/fixtures/gemm_small_steel.metal"
A_DOC = "docs/架构/shape-portfolio-阶段验收/P3/A.md"
B_DOC = "docs/架构/shape-portfolio-阶段验收/P3/B.md"
C_DOC = "docs/架构/shape-portfolio-阶段验收/P3/C.md"
VERIFIER = "docs/架构/shape-portfolio-阶段验收/P3/verify.py"
BASELINE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/acceptance-baseline.json"
START_NAME = "docs/架构/shape-portfolio-阶段验收/P3/start-worktree.sha256"
POLICY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/performance-policy.json"
GEMM_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/gemm-performance.json"
PAPER_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/paper-performance.json"
GEMM_DOMAIN_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/gemm-domain-performance.json"
POINTWISE_DOMAIN_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/pointwise-domain-performance.json"
DIAGNOSTIC_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/small-timing-diagnostic.json"
KERNEL_ONLY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/kernel-only-feasibility.json"
NATIVE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/native-batch-feasibility.json"
CONSULT_NAME = "docs/架构/shape-portfolio-阶段验收/P3/small-gemm-performance-consultation.md"
DISASSEMBLY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/mlx-small-disassembly-analysis.md"
DELIVERABLES = (
    PRODUCTION,
    PACKAGE_INIT,
    METAL_PERFORMANCE,
    METAL_BASELINES,
    METAL_V2_RUNTIME,
    TEST,
    GPU_TEST,
    RUNTIME,
    RUNTIME_TEST,
    NATIVE_PROD,
    NATIVE_SOURCE,
    BACKEND,
    ADAPTER,
    NATIVE_TEST,
    STEEL_FIXTURE,
    A_DOC,
    B_DOC,
    C_DOC,
    VERIFIER,
    BASELINE_NAME,
    START_NAME,
    POLICY_NAME,
    GEMM_EVIDENCE_NAME,
    PAPER_EVIDENCE_NAME,
    GEMM_DOMAIN_EVIDENCE_NAME,
    POINTWISE_DOMAIN_EVIDENCE_NAME,
    DIAGNOSTIC_NAME,
    KERNEL_ONLY_NAME,
    NATIVE_NAME,
    CONSULT_NAME,
    DISASSEMBLY_NAME,
    PLAN,
)

#: Known post-4a93ed2 evolution from varlen V5/V6, native64 Host Program, P2-F re-record, and
#: moving acceptance fixtures out of docs/.  The frozen baseline JSON and start-worktree.sha256
#: are never rewritten; any drift outside this exact set fails.
KNOWN_POST_BASELINE_DRIFT = frozenset(
    {
        PLAN,
        "docs/架构/shape-portfolio-阶段验收/P2/evidence/active_portfolio_pointer.json",
        "docs/架构/shape-portfolio-阶段验收/P2/evidence/dispatch-manifest.json",
        "docs/架构/shape-portfolio-阶段验收/P2/evidence/tune-confirmations.json",
        "docs/架构/shape-portfolio-阶段验收/P2/evidence/unseen-confirmations.json",
        "docs/架构/shape-portfolio-阶段验收/P2/golden.json",
        "docs/架构/shape-portfolio-阶段验收/P2/verify.py",
        "taic/portfolio_confirmation.py",
        "taic/portfolio_manifest.py",
        "tcenv/backends/metal/launch.py",
        "tcenv/backends/metal/portfolio_confirmation.py",
        "tcenv/portfolio/candidate.py",
        "tcenv/portfolio/domain.py",
        "tcenv/taskcontract/portfolio_runner.py",
        "tests/backends/metal/test_shape_portfolio_confirmation.py",
        "tests/backends/metal/test_shape_portfolio_manifest.py",
        "tests/backends/metal/test_shape_portfolio_resume.py",
        "tests/shared/test_shape_portfolio_candidate.py",
        "tests/shared/test_shape_portfolio_runner.py",
        "tests/support/README.md",
    }
)

SHARD_DIGEST_KEYS = (
    "candidate_sha256",
    "geometry_identity",
    "bundle_digest",
    "artifact_set_digest",
    "native_digest",
    "evaluation_fingerprint",
    "task_pack_digest",
)

STABLE_SHARD_KEYS = (
    "instance_key",
    "usage",
    "provider",
    "variant_id",
    "variant_key",
    "fallback_id",
    "fallback_reason",
    "credits_variant",
    "track_id",
    "target_id",
    "device_name",
    "gpu_family",
    "candidate_sha256",
    "geometry_identity",
    "bundle_digest",
    "artifact_set_digest",
    "native_digest",
    "evaluation_fingerprint",
    "task_pack_digest",
    "dispatch",
    "expected_dispatch",
    "dispatch_ok",
    "correctness_ok",
    "best_baseline",
    "baselines_measured",
    "rotation_sets",
    "timing_uncertain",
    "unknown",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text(
        "\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n",
        encoding="utf-8",
    )
    print(f"OK: wrote {MANIFEST.relative_to(REPO_ROOT)} ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P3 manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P3 manifest mismatch: {name}")
    print(f"OK: P3 manifest ({len(entries)} files)")


def verify_baseline() -> None:
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p3-performance-baseline-1", "wrong P3 baseline epoch")
    require(document["head"] == "4a93ed2a4934fe526eedfcf4577de5dbd366e24a", "P3 baseline HEAD drifted")
    require(document["p2_status_at_start"] == "completed", "P3 did not start from completed P2")
    surface = document["contract_surface"]["digests"]
    require(len(surface) == document["contract_surface"]["files"] == 71, "P3 baseline contract surface count drifted")
    drifted = sorted(name for name, digest in surface.items() if sha256(ROOT / name) != digest)
    unexpected = sorted(set(drifted) - KNOWN_POST_BASELINE_DRIFT)
    require(not unexpected, f"P3 baseline unexpected drift outside P2-F/migration set: {unexpected}")
    require(
        sha256(P3 / "start-worktree.sha256") == document["full_worktree"]["manifest_sha256"],
        "P3 full start manifest changed",
    )
    require(document["full_worktree"]["files"] == 2128, "P3 full start file count drifted")
    print(
        f"OK: P3 stage-start baseline intact (71 contract files, 2128 worktree files; "
        f"{len(drifted)} accounted post-4a93ed2 evolutions, 0 unexpected)"
    )


def verify_performance_policy() -> None:
    policy = json.loads((P3 / "evidence/performance-policy.json").read_text(encoding="utf-8"))
    expected = {
        "schema_id": "tcenv.portfolio.performance-policy.1",
        "policy_version": 1,
        "frozen_at_stage": "P3-A before the second GEMM task",
        "max_regression_latency_ratio": 1.1,
        "latency_ratio_definition": "candidate_us / baseline_us (lower is faster)",
        "speedup_definition": "baseline_us / candidate_us (higher is faster)",
        "baseline_selection": "fastest registered baseline by median_us",
        "specialized_hit_aggregation": "median of per-shard latency ratios",
        "effective_end_to_end_aggregation": "sum(candidate_us) / sum(baseline_us)",
        "geomean_role": "descriptive_only",
        "required_usage": "benchmark",
        "fallback_credit": "effective_end_to_end_only",
        "no_specialized_variant_publishable": False,
        "fresh_timing_required_for_performance_claims": True,
        "notes": "This file is a frozen policy input, not a generated performance result. Changes require a new P3 epoch and full re-acceptance.",
    }
    require(policy == expected, "P3 performance-policy.json drifted; a new epoch and full re-acceptance are required")
    require(
        MAX_REGRESSION_RATIO == policy["max_regression_latency_ratio"],
        "production MAX_REGRESSION_RATIO differs from the frozen policy",
    )
    print("OK: frozen performance-policy.json (1.10, two books, no geomean-only gate)")


def verify_package_boundary() -> None:
    package = ROOT / "tcenv/portfolio_performance"
    sources = sorted(package.glob("*.py"))
    require([path.name for path in sources] == ["__init__.py", "performance.py"], "P3 performance package is not closed")
    forbidden = ("PatternResolver", "LoweringRuleRegistry", "StagePartitioner", "TileConnector", "LoweringProfile")
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            require(token not in text, f"{path.name}: forbidden architecture token {token}")
        for line in text.splitlines():
            stripped = line.strip()
            require(
                not stripped.startswith(("import tcenv.backends", "from tcenv.backends", "import taic", "from taic")),
                f"{path.name}: backend/search import",
            )
    print("OK: P3 performance package boundary (2 closed files, no backend/search/forbidden architecture)")


def pytest(arguments: list[str], *, where: str, expected: int, workers: str = "0") -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", workers, "-p", "no:cacheprovider", *arguments],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"{where} failed:\n{output}")
    require(f"{expected} passed" in output, f"{where}: expected exactly {expected} passed:\n{output}")
    require(
        " skipped" not in output and "xfail" not in output and "deselected" not in output,
        f"{where}: skip/xfail/deselect:\n{output}",
    )
    print(f"OK: {where} ({expected} passed)")


def run_performance_gpu(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", GPU_TEST],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "SHAPE_PORTFOLIO_P3_EVIDENCE_DIR": str(directory)},
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"P3-B/C real Mac performance nodes failed:\n{output}")
    require(
        "4 passed" in output and " skipped" not in output and "deselected" not in output,
        f"P3-B/C node count/skip drift:\n{output}",
    )
    print("OK: P3-B/C real Mac performance nodes (4 passed)")


def _shard_measurement(shard: Mapping[str, Any]) -> ShardMeasurement:
    require(shard.get("usage") == "benchmark", f"{shard.get('instance_key')}: non-benchmark shard in performance evidence")
    require(shard.get("correctness_ok") is True, f"{shard.get('instance_key')}: correctness_ok is not True")
    require(shard.get("dispatch_ok") is True, f"{shard.get('instance_key')}: dispatch_ok is not True")
    require(shard.get("timing_uncertain") is False, f"{shard.get('instance_key')}: timing_uncertain is not False")
    require(shard.get("unknown") is False, f"{shard.get('instance_key')}: unknown is not False")
    require(
        shard.get("dispatch") and shard.get("dispatch") == shard.get("expected_dispatch"),
        f"{shard.get('instance_key')}: observed dispatch does not match expected_dispatch",
    )
    require(shard.get("best_baseline") == "mlx", f"{shard.get('instance_key')}: fastest baseline is not mlx")
    require(
        shard.get("baselines_measured") == ["mlx", "torch_mps"],
        f"{shard.get('instance_key')}: registered baselines were not both measured",
    )
    require(
        isinstance(shard.get("device_name"), str) and "Apple" in str(shard.get("device_name")),
        f"{shard.get('instance_key')}: missing real Apple GPU device_name",
    )
    for key in SHARD_DIGEST_KEYS:
        require(is_hex64(shard.get(key)), f"{shard.get('instance_key')}.{key} is not a 64-char hex digest")
    provider = shard.get("provider")
    if provider == "candidate":
        require(is_hex64(shard.get("variant_id")), f"{shard.get('instance_key')}: missing variant_id")
        require(
            isinstance(shard.get("variant_key"), str) and bool(shard.get("variant_key")),
            f"{shard.get('instance_key')}: missing variant_key",
        )
        require(shard.get("fallback_id") is None, f"{shard.get('instance_key')}: candidate shard has fallback_id")
        require(shard.get("credits_variant") is True, f"{shard.get('instance_key')}: candidate shard lost credits_variant")
    else:
        require(provider == "fallback", f"{shard.get('instance_key')}: invalid provider {provider!r}")
        require(shard.get("variant_id") is None and shard.get("variant_key") is None, "fallback shard claims variant_id")
        require(is_hex64(shard.get("fallback_id")), f"{shard.get('instance_key')}: missing fallback_id")
        require(
            shard.get("fallback_reason") == "outside_candidate_coverage",
            f"{shard.get('instance_key')}: unexpected fallback_reason",
        )
        require(shard.get("credits_variant") is False, f"{shard.get('instance_key')}: fallback credited as variant")
    measurement = ShardMeasurement(
        instance_key=str(shard["instance_key"]),
        usage=str(shard["usage"]),
        provider=str(shard["provider"]),
        candidate_us=float(shard["candidate_us"]),
        baseline_us=float(shard["baseline_us"]),
        timing_uncertain=bool(shard["timing_uncertain"]),
        correctness_ok=bool(shard["correctness_ok"]),
        dispatch_ok=bool(shard["dispatch_ok"]),
        unknown=bool(shard["unknown"]),
    )
    require(
        abs(float(shard["latency_ratio"]) - float(measurement.latency_ratio)) <= 1e-12,
        f"{measurement.instance_key}: latency_ratio does not match candidate_us / baseline_us",
    )
    require(
        measurement.latency_ratio <= MAX_REGRESSION_RATIO,
        f"{measurement.instance_key}: latency_ratio {measurement.latency_ratio} exceeds {MAX_REGRESSION_RATIO}",
    )
    return measurement


def _verify_domain_report(
    document: Mapping[str, Any],
    *,
    schema_id: str,
    task: str,
    expected_keys: set[str],
    expected_fallback_count: int,
) -> dict[str, dict[str, Any]]:
    require(document.get("schema_id") == schema_id, f"wrong schema_id in {task} performance report")
    require(document.get("policy") == "performance-policy.json", f"{task}: policy reference drifted")
    require(document.get("task") == task, f"{task}: task name drifted")
    require(is_hex64(document.get("domain_plan_digest")), f"{task}: invalid domain_plan_digest")
    shards = list(document.get("measurements") or [])
    require(len(shards) == len(expected_keys), f"{task}: shard count {len(shards)} != {len(expected_keys)}")
    measurements = [_shard_measurement(shard) for shard in shards]
    recomputed = PerformanceReport.evaluate(measurements, expected_benchmark_keys=expected_keys)
    require(recomputed.publishable is True and not recomputed.failure_reasons, f"{task}: recomputed report not publishable")
    require(document.get("publishable") is True and not document.get("failure_reasons"), f"{task}: archived report not publishable")
    for field in (
        "specialized_hit_latency_ratio",
        "effective_end_to_end_latency_ratio",
        "specialized_geomean_latency_ratio",
        "effective_geomean_latency_ratio",
        "specialized_hit_speedup",
        "effective_end_to_end_speedup",
        "fallback_time_fraction",
        "fallback_count_fraction",
    ):
        got = float(document[field])
        want = float(getattr(recomputed, field))
        require(abs(got - want) <= 1e-12, f"{task}.{field}: archived {got} != recomputed {want}")
    actual_fallbacks = sum(1 for m in measurements if m.provider == "fallback")
    require(actual_fallbacks == expected_fallback_count, f"{task}: fallback count {actual_fallbacks} != {expected_fallback_count}")
    return {str(shard["instance_key"]): dict(shard) for shard in shards}


def verify_performance_evidence(evidence_dir: Path | None = None) -> dict[str, dict[str, Any]]:
    directory = (P3 / "evidence") if evidence_dir is None else evidence_dir
    small = json.loads((directory / "gemm-performance.json").read_text(encoding="utf-8"))
    small_shards = _verify_domain_report(
        small,
        schema_id="tcenv.portfolio.p3-gemm-performance.1",
        task="gemm",
        expected_keys={"fp16/small/benchmark"},
        expected_fallback_count=0,
    )
    require(small_shards["fp16/small/benchmark"]["variant_key"] == "gemm-small", "small GEMM wrong variant_key")

    paper = json.loads((directory / "paper-performance.json").read_text(encoding="utf-8"))
    require(paper.get("schema_id") == "tcenv.portfolio.p3-gemm-paper-performance.1", "paper GEMM wrong schema_id")
    paper_m = _shard_measurement(paper)
    require(paper.get("variant_key") == "gemm-paper", "paper GEMM wrong variant_key")
    require(paper.get("publishable") is True and not paper.get("failure_reasons"), "paper GEMM not publishable")
    require(paper_m.latency_ratio <= MAX_REGRESSION_RATIO, "paper GEMM latency ratio > 1.10")

    gemm_domain = json.loads((directory / "gemm-domain-performance.json").read_text(encoding="utf-8"))
    gemm_shards = _verify_domain_report(
        gemm_domain,
        schema_id="tcenv.portfolio.p3-gemm-domain-performance.1",
        task="gemm",
        expected_keys={"fp16/small/benchmark", "fp16/paper/benchmark"},
        expected_fallback_count=0,
    )
    require(
        strict_json_equal(gemm_shards["fp16/small/benchmark"], small_shards["fp16/small/benchmark"]),
        "gemm-domain small shard disagrees with gemm-performance.json",
    )

    pointwise_domain = json.loads((directory / "pointwise-domain-performance.json").read_text(encoding="utf-8"))
    pointwise_keys = {f"fp16/{size}/benchmark" for size in P1_SIZES}
    pointwise_shards = _verify_domain_report(
        pointwise_domain,
        schema_id="tcenv.portfolio.p3-pointwise-domain-performance.1",
        task=P1_OP,
        expected_keys=pointwise_keys,
        expected_fallback_count=1,
    )
    for size, expected in EXPECTED_PROVIDER.items():
        shard = pointwise_shards[f"fp16/{size}/benchmark"]
        if expected == "fallback":
            require(shard["provider"] == "fallback" and shard["credits_variant"] is False, f"{size}: expected fallback")
        else:
            require(
                shard["provider"] == "candidate" and shard["variant_key"] == expected,
                f"{size}: expected {expected}, got {shard['variant_key']!r}",
            )

    diagnostic = json.loads((P3 / "evidence/small-timing-diagnostic.json").read_text(encoding="utf-8"))
    require(
        diagnostic["diagnostic_only"] is True and diagnostic["publication_allowed"] is False,
        "timing diagnostic was promoted to publication evidence",
    )
    require(diagnostic["candidate_gpu_per_iter_median_us"] > 0, "small diagnostic has no real GPU timestamp")
    require(
        diagnostic["candidate_wall_per_iter_median_us"] > diagnostic["candidate_gpu_per_iter_median_us"],
        "small wall/GPU decomposition is inconsistent",
    )
    require(
        diagnostic["wall_minus_gpu_median_us"] > 0 and diagnostic["capture_count"] > 0,
        "small diagnostic has no overhead samples",
    )
    kernel_only = json.loads((P3 / "evidence/kernel-only-feasibility.json").read_text(encoding="utf-8"))
    require(
        kernel_only["diagnostic_only"] is True and kernel_only["publication_allowed"] is False,
        "kernel-only feasibility was promoted to publication",
    )
    require(
        kernel_only["baseline_gpu_only_available"] is False and kernel_only["kernel_only_gate_allowed"] is False,
        "kernel-only gate is enabled without fastest-baseline GPU timing",
    )
    require(
        kernel_only["baseline_gpu_interval_diagnostic_available"] is True,
        "successful xctrace diagnostic resolution was lost",
    )
    require(kernel_only["required_before_kernel_only_gate"], "kernel-only prerequisites were removed")
    native = json.loads((P3 / "evidence/native-batch-feasibility.json").read_text(encoding="utf-8"))
    require(
        native["diagnostic_only"] is True and native["publication_allowed"] is False,
        "native prototype was promoted to publication",
    )
    require(
        native["native_batch_is_production"] is True and native["native_batch_performance_gate_passed"] is False,
        "native production status or gate result drifted",
    )
    require(
        native["official_full_run"]["latest_production_evidence"]["publishable"] is False,
        "official full-run evidence was incorrectly published",
    )
    for name in ("golden", "tile32", "direct_epilogue"):
        row = native["native_batch"][name]
        require(row["latency_ratio"] > 1.10, f"native {name} prototype is incorrectly marked as passing")
        require(0 < row["wall_minus_gpu_us"] < 0.5, f"native {name} path decomposition drifted")
    print(
        "OK: P3-B/C performance evidence (two tasks, 9 benchmark shards <=1.10, two books + fallback accounting, "
        "full device/dispatch/identity binding; diagnostics stay unpublished)"
    )
    return {**{f"gemm:{k}": v for k, v in gemm_shards.items()}, **{f"pointwise:{k}": v for k, v in pointwise_shards.items()}}


def _stable_shard_view(shard: Mapping[str, Any]) -> dict[str, Any]:
    return {key: shard.get(key) for key in STABLE_SHARD_KEYS}


def verify_live_vs_archived() -> None:
    archived = verify_performance_evidence(P3 / "evidence")
    with tempfile.TemporaryDirectory(prefix="p3-c-live-") as tmp:
        live_dir = Path(tmp)
        run_performance_gpu(live_dir)
        live = verify_performance_evidence(live_dir)
    require(set(live) == set(archived), "live and archived shard sets differ")
    for key in sorted(archived):
        require(
            strict_json_equal(_stable_shard_view(live[key]), _stable_shard_view(archived[key])),
            f"{key}: live and archived stable shard identities differ",
        )
    print(f"OK: live == archived stable performance identities across all {len(archived)} shards (2 tasks)")


def verify_p2_anchors() -> None:
    p2_path = ROOT / "docs/架构/shape-portfolio-阶段验收/P2/verify.py"
    spec = importlib.util.spec_from_file_location("p2_verify_anchor", p2_path)
    assert spec is not None and spec.loader is not None
    p2_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p2_mod)
    rebind_acceptance_root(p2_mod)
    p2_mod.verify_manifest()
    p2_mod.verify_package_boundary()
    with tempfile.TemporaryDirectory(prefix="p3-p2-anchor-") as tmp:
        locator, freeze, plan = p2_mod.fixture(Path(tmp))
        p2_mod.pytest_nodes(list(p2_mod.P0_P1_NODES), where="P0/P1 fixed CPU anchors", expected=58)
        p2_mod.pytest_nodes(["tests/shared/test_structure.py"], where="repository structure", expected=6)
        p2_mod.pytest_nodes(["tests/shared/test_shape_portfolio_evidence.py"], where="P2-0 anchor", expected=1)
        p2_mod.pytest_nodes([p2_mod.TEST], where="P2-A fixed CPU nodes", expected=9)
        pytest([p2_mod.CPU_CONFIRMATION_TEST], where="P2-B pack/archive negative nodes", expected=11, workers="auto")
        pytest([p2_mod.MANIFEST_TEST], where="P2-C DispatchManifest CPU nodes", expected=16, workers="auto")
        pytest([p2_mod.RESUME_TEST], where="P2-D checkpoint/resume CPU nodes", expected=10, workers="auto")
        p2_mod.verify_confirmation_suite(p2_mod.P2 / "evidence", locator, freeze, plan)
        p2_mod.verify_activation_evidence(locator, freeze, plan)
        p2_mod.verify_checkpoint_lifecycle(locator, freeze, plan)
    print("OK: P0/P1/P2-0/P2 acceptance anchors (111 CPU nodes + 14 archived confirmations + active manifest + lifecycle)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--record-evidence", action="store_true")
    parser.add_argument("--live-compare", action="store_true")
    args = parser.parse_args()
    if args.write_manifest:
        write_manifest()
        return
    if args.record_evidence:
        run_performance_gpu(P3 / "evidence")
        verify_performance_evidence()
        return
    verify_manifest()
    verify_baseline()
    verify_performance_policy()
    verify_package_boundary()
    verify_p2_anchors()
    pytest([TEST], where="P3-A performance CPU nodes", expected=20)
    pytest([RUNTIME_TEST], where="P3-B runtime encoding CPU nodes", expected=18)
    pytest([NATIVE_TEST], where="P3-B native/Python parity nodes", expected=5)
    verify_performance_evidence()
    if args.live_compare:
        verify_live_vs_archived()
    print(
        "OK: P3-A/B/C acceptance (20 performance CPU + 18 runtime + 5 native parity + 111 P0-P2 anchors + "
        "2 tasks / 9 real-Mac benchmark shards <=1.10)"
    )


if __name__ == "__main__":
    main()
