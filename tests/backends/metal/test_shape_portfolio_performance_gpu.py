"""P3-B real-Mac performance slices: a second tiled GEMM task."""

from __future__ import annotations

import json
import os

import pytest

from tcenv.backends.metal.portfolio_performance import measure_shard
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_performance.performance import PerformanceReport
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tcenv.track.plane import CandidateEntry
from tests.backends.metal.test_task_migration_a5 import _simdgroup_gemm_source
from tests.support.shape_portfolio_metal import evidence_all_valid

gpu = pytest.mark.gpu


def _golden_gemm_source(contract, size: str) -> str:
    instance = instantiate_v2(contract.spec, "fp16", size)
    m, n, k = (int(instance.symbols[name]) for name in ("M", "N", "K"))
    fixture = "gemm_small_steel.metal" if size == "small" else "gemm_paper_best.metal"
    body = (REPO_ROOT / "tests/support/golden/metal/fixtures" / fixture).read_text(encoding="utf-8")
    thread_shape = "32,2,2" if size == "small" else "128,1,1"
    return (
        "constant constexpr uint BLOCK_M = 64;\n"
        "constant constexpr uint BLOCK_N = 64;\n"
        "constant constexpr uint BLOCK_K = 32;\n"
        "/* taic-launch\nmath fast\n"
        f"launch gemm_kernel grid={n // 64},{m // 64},1 threads={thread_shape}\n*/\n"
        + body
    )


def _gemm_portfolio(contract, plan):
    small = CandidateEntry(
        path="device/gemm_small.metal",
        role="device_kernel",
        media_type="text/plain",
        data=_golden_gemm_source(contract, "small").encode(),
    )
    paper = CandidateEntry(
        path="device/gemm_paper.metal",
        role="device_kernel",
        media_type="text/plain",
        data=_golden_gemm_source(contract, "paper").encode(),
    )
    generic = CandidateEntry(
        path="device/gemm_generic.metal",
        role="device_kernel",
        media_type="text/plain",
        data=_simdgroup_gemm_source(contract, "64x64x32t256").encode(),
    )
    small_guard = ["and", ["eq", "M", 256], ["and", ["eq", "N", 256], ["eq", "K", 128]]]
    paper_guard = ["and", ["eq", "M", 2048], ["and", ["eq", "N", 2048], ["eq", "K", 2048]]]
    from tcenv.portfolio.domain import guard_digest

    document = {
        "schema_id": "tcenv.candidate-portfolio.1",
        "domain_plan_digest": plan.digest,
        "planner": {"id": "p3-gemm-planner", "version": 1},
        "coverage_plan": {"buckets": [
            {"bucket_key": "small", "guard_digest": guard_digest(small_guard, where="small")},
            {"bucket_key": "paper", "guard_digest": guard_digest(paper_guard, where="paper")},
        ]},
        "variants": [
            {
                "variant_key": "gemm-small",
                "bucket_key": "small",
                "guard": {"expr": small_guard, "digest": guard_digest(small_guard, where="gemm-small")},
                "entries": [small.path],
                "geometry_entry": None,
            },
            {
                "variant_key": "gemm-paper",
                "bucket_key": "paper",
                "guard": {"expr": paper_guard, "digest": guard_digest(paper_guard, where="gemm-paper")},
                "entries": [paper.path],
                "geometry_entry": None,
            },
        ],
        "fallback": {
            "kind": "registered_implementation",
            "implementation_id": "gemm.generic",
            "implementation_version": 1,
            "entries": [generic.path],
            "geometry_entry": None,
            "reasons": ["outside_candidate_coverage", "resource_unavailable"],
        },
    }
    from tests.support.shape_portfolio_metal import manifest_entry, parent_entries

    return parse_portfolio(
        parent_entries(
            document,
            extra_sources={
                small.path: (small.role, small.media_type, small.data),
                paper.path: (paper.role, paper.media_type, paper.data),
                generic.path: (generic.role, generic.media_type, generic.data),
            },
        ),
        domain_plan=plan,
    )


def _fixture(tmp_path):
    root = tmp_path / "tasks"
    task_dir = REPO_ROOT / "tasks/gemm"
    import shutil

    shutil.copytree(task_dir, root / task_dir.name)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    target = load_target(DEFAULT_TARGET_ID)
    instances = tuple(instantiate_v2(contract.spec, "fp16", size) for size in ("small", "paper"))
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(plan=plan, spec=contract.spec, target=target, instances=instances, contract=contract)
    portfolio = _gemm_portfolio(contract, plan)
    return locator, plan, portfolio


def _request(contract, plan, size):
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = plan.find("fp16", size, "benchmark")
    assert point is not None
    return DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="benchmark")


@gpu
def test_p3_b_gemm_small_performance_gate_reports_the_actual_regression(tmp_path):
    locator, plan, portfolio = _fixture(tmp_path)
    contract = locator.load()
    measurements = [
        measure_shard(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(contract, plan, "small"),
            evidence=evidence_all_valid(portfolio),
        )
    ]
    report = PerformanceReport.evaluate(
        measurements,
        expected_benchmark_keys={measurement.instance_key for measurement in measurements},
    )
    directory = os.environ.get("SHAPE_PORTFOLIO_P3_EVIDENCE_DIR")
    if directory:
        from pathlib import Path

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_id": "tcenv.portfolio.p3-gemm-performance.1",
            "policy": "performance-policy.json",
            "measurements": [
                {
                    "instance_key": measurement.instance_key,
                    "provider": measurement.provider,
                    "candidate_us": measurement.candidate_us,
                    "baseline_us": measurement.baseline_us,
                    "latency_ratio": measurement.latency_ratio,
                }
                for measurement in measurements
            ],
            "specialized_hit_latency_ratio": report.specialized_hit_latency_ratio,
            "effective_end_to_end_latency_ratio": report.effective_end_to_end_latency_ratio,
            "specialized_hit_speedup": report.specialized_hit_speedup,
            "effective_end_to_end_speedup": report.effective_end_to_end_speedup,
            "publishable": report.publishable,
            "failure_reasons": list(report.failure_reasons),
        }
        (target / "gemm-performance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert report.publishable is False, report.failure_reasons
    assert any("ratio>1.10" in reason or "timing_uncertain" in reason for reason in report.failure_reasons)
    assert measurements[0].latency_ratio > 1.10 or measurements[0].timing_uncertain
    assert all(measurement.provider == "candidate" for measurement in measurements)
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None


@gpu
def test_p3_b_gemm_paper_performance_gate_passes(tmp_path):
    locator, plan, portfolio = _fixture(tmp_path)
    contract = locator.load()
    measurement = measure_shard(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=_request(contract, plan, "paper"),
        evidence=evidence_all_valid(portfolio),
    )
    report = PerformanceReport.evaluate(
        [measurement], expected_benchmark_keys={measurement.instance_key}
    )
    assert measurement.correctness_ok and measurement.dispatch_ok
    assert measurement.provider == "candidate"
    assert measurement.latency_ratio <= 1.10, measurement
    assert report.publishable is True, report.failure_reasons
    directory = os.environ.get("SHAPE_PORTFOLIO_P3_EVIDENCE_DIR")
    if directory:
        from pathlib import Path

        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_id": "tcenv.portfolio.p3-gemm-paper-performance.1",
            "instance_key": measurement.instance_key,
            "provider": measurement.provider,
            "candidate_us": measurement.candidate_us,
            "baseline_us": measurement.baseline_us,
            "latency_ratio": measurement.latency_ratio,
            "publishable": report.publishable,
            "failure_reasons": list(report.failure_reasons),
        }
        (target / "paper-performance.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
