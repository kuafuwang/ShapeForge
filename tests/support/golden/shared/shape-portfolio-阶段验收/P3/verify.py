#!/usr/bin/env python3
"""P3-A acceptance: performance model + P2 anchors + real stage baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from tcenv.portfolio_performance.performance import MAX_REGRESSION_RATIO

ROOT = Path(__file__).resolve().parents[4]
P3 = Path(__file__).resolve().parent
BASELINE = P3 / "acceptance-baseline.json"
MANIFEST = P3 / "manifest.sha256"
PLAN = "docs/架构/shape-portfolio实施计划.md"
PRODUCTION = "tcenv/portfolio_performance/performance.py"
PACKAGE_INIT = "tcenv/portfolio_performance/__init__.py"
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
VERIFIER = "docs/架构/shape-portfolio-阶段验收/P3/verify.py"
BASELINE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/acceptance-baseline.json"
START_NAME = "docs/架构/shape-portfolio-阶段验收/P3/start-worktree.sha256"
POLICY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/performance-policy.json"
GEMM_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/gemm-performance.json"
PAPER_EVIDENCE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/paper-performance.json"
DIAGNOSTIC_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/small-timing-diagnostic.json"
KERNEL_ONLY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/kernel-only-feasibility.json"
NATIVE_NAME = "docs/架构/shape-portfolio-阶段验收/P3/evidence/native-batch-feasibility.json"
CONSULT_NAME = "docs/架构/shape-portfolio-阶段验收/P3/small-gemm-performance-consultation.md"
DISASSEMBLY_NAME = "docs/架构/shape-portfolio-阶段验收/P3/mlx-small-disassembly-analysis.md"
DELIVERABLES = (PRODUCTION, PACKAGE_INIT, TEST, GPU_TEST, RUNTIME, RUNTIME_TEST, NATIVE_PROD, NATIVE_SOURCE, BACKEND, ADAPTER, NATIVE_TEST, STEEL_FIXTURE, A_DOC, B_DOC, VERIFIER, BASELINE_NAME, START_NAME, POLICY_NAME, GEMM_EVIDENCE_NAME, PAPER_EVIDENCE_NAME, DIAGNOSTIC_NAME, KERNEL_ONLY_NAME, NATIVE_NAME, CONSULT_NAME, DISASSEMBLY_NAME, PLAN)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest() -> None:
    MANIFEST.write_text("\n".join(f"{sha256(ROOT / name)}  {name}" for name in sorted(DELIVERABLES)) + "\n", encoding="utf-8")
    print(f"OK: wrote {MANIFEST.relative_to(ROOT)} ({len(DELIVERABLES)} files)")


def verify_manifest() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    require(set(entries) == set(DELIVERABLES), "P3-A manifest file set drifted")
    for name, digest in entries.items():
        require(sha256(ROOT / name) == digest, f"P3-A manifest mismatch: {name}")
    print(f"OK: P3-A manifest ({len(entries)} files)")


def verify_baseline() -> None:
    document = json.loads(BASELINE.read_text(encoding="utf-8"))
    require(document["epoch"] == "p3-performance-baseline-1", "wrong P3 baseline epoch")
    require(document["head"] == "4a93ed2a4934fe526eedfcf4577de5dbd366e24a", "P3 baseline HEAD drifted")
    require(document["p2_status_at_start"] == "completed", "P3 did not start from completed P2")
    surface = document["contract_surface"]["digests"]
    require(len(surface) == document["contract_surface"]["files"] == 71, "P3 baseline contract surface count drifted")
    drifted = sorted(name for name, digest in surface.items() if sha256(ROOT / name) != digest)
    require(set(drifted) <= {PLAN}, f"P3 baseline drift outside the living plan: {drifted}")
    require(sha256(P3 / "start-worktree.sha256") == document["full_worktree"]["manifest_sha256"], "P3 full start manifest changed")
    require(document["full_worktree"]["files"] == 2128, "P3 full start file count drifted")
    print(f"OK: real P3 stage-start baseline (71 contract files, 2128 worktree files; drift={drifted})")


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
    require(MAX_REGRESSION_RATIO == policy["max_regression_latency_ratio"], "production MAX_REGRESSION_RATIO differs from the frozen policy")
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
            require(not stripped.startswith(("import tcenv.backends", "from tcenv.backends", "import taic", "from taic")), f"{path.name}: backend/search import")
    print("OK: P3 performance package boundary (2 closed files, no backend/search/forbidden architecture)")


def pytest(arguments: list[str], *, where: str, expected: int) -> None:
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", *arguments], cwd=ROOT, text=True, capture_output=True)
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"{where} failed:\n{output}")
    require(f"{expected} passed" in output, f"{where}: expected exactly {expected} passed:\n{output}")
    require(" skipped" not in output and "xfail" not in output and "deselected" not in output, f"{where}: skip/xfail/deselect:\n{output}")
    print(f"OK: {where} ({expected} passed)")


def run_performance_gpu(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-n", "0", "-p", "no:cacheprovider", GPU_TEST],
        cwd=ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "SHAPE_PORTFOLIO_P3_EVIDENCE_DIR": str(directory)},
    )
    output = result.stdout + result.stderr
    require(result.returncode == 0, f"P3-B real Mac performance nodes failed:\n{output}")
    require("2 passed" in output and " skipped" not in output and "deselected" not in output, f"P3-B node count/skip drift:\n{output}")
    print("OK: P3-B real Mac performance nodes (2 passed)")


def verify_performance_evidence() -> None:
    small = json.loads((P3 / "evidence/gemm-performance.json").read_text(encoding="utf-8"))
    require(small["publishable"] is False, "small GEMM regression was incorrectly published")
    require(small["failure_reasons"], "small GEMM has no recorded regression reason")
    require(
        any("ratio>1.10" in reason or "timing_uncertain" in reason for reason in small["failure_reasons"]),
        "small GEMM regression reason is not the measured latency/noise gate",
    )
    paper = json.loads((P3 / "evidence/paper-performance.json").read_text(encoding="utf-8"))
    require(paper["publishable"] is True and paper["latency_ratio"] <= 1.10, "paper GEMM positive performance evidence is invalid")
    require(paper["provider"] == "candidate" and not paper["failure_reasons"], "paper GEMM must be a clean candidate measurement")
    diagnostic = json.loads((P3 / "evidence/small-timing-diagnostic.json").read_text(encoding="utf-8"))
    require(diagnostic["diagnostic_only"] is True and diagnostic["publication_allowed"] is False, "timing diagnostic was promoted to publication evidence")
    require(diagnostic["candidate_gpu_per_iter_median_us"] > 0, "small diagnostic has no real GPU timestamp")
    require(diagnostic["candidate_wall_per_iter_median_us"] > diagnostic["candidate_gpu_per_iter_median_us"], "small wall/GPU decomposition is inconsistent")
    require(diagnostic["wall_minus_gpu_median_us"] > 0 and diagnostic["capture_count"] > 0, "small diagnostic has no overhead samples")
    kernel_only = json.loads((P3 / "evidence/kernel-only-feasibility.json").read_text(encoding="utf-8"))
    require(kernel_only["diagnostic_only"] is True and kernel_only["publication_allowed"] is False, "kernel-only feasibility was promoted to publication")
    require(kernel_only["baseline_gpu_only_available"] is False and kernel_only["kernel_only_gate_allowed"] is False, "kernel-only gate is enabled without fastest-baseline GPU timing")
    require(kernel_only["baseline_gpu_interval_diagnostic_available"] is True, "successful xctrace diagnostic resolution was lost")
    require(kernel_only["required_before_kernel_only_gate"], "kernel-only prerequisites were removed")
    native = json.loads((P3 / "evidence/native-batch-feasibility.json").read_text(encoding="utf-8"))
    require(native["diagnostic_only"] is True and native["publication_allowed"] is False, "native prototype was promoted to publication")
    require(native["native_batch_is_production"] is True and native["native_batch_performance_gate_passed"] is False, "native production status or gate result drifted")
    require(native["official_full_run"]["latest_production_evidence"]["publishable"] is False, "official full-run evidence was incorrectly published")
    for name in ("golden", "tile32", "direct_epilogue"):
        row = native["native_batch"][name]
        require(row["latency_ratio"] > 1.10, f"native {name} prototype is incorrectly marked as passing")
        require(0 < row["wall_minus_gpu_us"] < 0.5, f"native {name} path decomposition drifted")
    print("OK: P3-B blocked evidence is honest (paper passes <=1.10; small ratio/noise gate blocks publication)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--record-evidence", action="store_true")
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
    result = subprocess.run([sys.executable, "-u", "docs/架构/shape-portfolio-阶段验收/P2/verify.py"], cwd=ROOT, text=True)
    require(result.returncode == 0, "P2 acceptance anchors failed")
    pytest([TEST], where="P3-A performance CPU nodes", expected=20)
    pytest([RUNTIME_TEST], where="P3-B runtime encoding CPU nodes", expected=18)
    pytest([NATIVE_TEST], where="P3-B native/Python parity nodes", expected=2)
    verify_performance_evidence()
    print("OK: P3-A acceptance (20 performance nodes + P2 anchors); P3-B runtime 18 + native parity 2, remains blocked (no publication)")


if __name__ == "__main__":
    main()
