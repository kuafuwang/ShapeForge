"""P3-B/C real-Mac performance slices: two tasks, two books, identity binding, and a regression negative."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from tcenv.backends.metal.portfolio_performance import measure_shard, measure_shard_with_evidence
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio.domain import DomainAuthority, guard_digest, normalize_semantic_domain_plan
from tcenv.portfolio_performance.performance import PerformanceReport, ShardMeasurement
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tcenv.track.plane import CandidateEntry
from tests.backends.metal.test_task_migration_a5 import _simdgroup_gemm_source
from tests.support.shape_portfolio_metal import (
    EXPECTED_PROVIDER,
    P1_OP,
    P1_SIZES,
    evidence_all_valid,
    opspec,
    p1_instances,
    parent_entries,
    portfolio_document,
)
from tests.support.v2_tasks import SEM_REFERENCE, write_task

gpu = pytest.mark.gpu

P1_BASELINES_JSON = (
    json.dumps(
        {
            "schema_version": 1,
            "baselines": [
                {
                    "name": "mlx",
                    "framework": "mlx",
                    "implementation_id": "p1.baseline.mlx",
                    "implementation_version": 1,
                    "params": {"op": "neg"},
                },
                {
                    "name": "torch_mps",
                    "framework": "torch",
                    "implementation_id": "p1.baseline.torch_mps",
                    "implementation_version": 1,
                    "params": {"op": "neg"},
                },
            ],
        },
        indent=2,
        sort_keys=True,
    )
    + "\n"
)

P1_BASELINES_PY = """from __future__ import annotations


def mlx(x):
    return -x


def torch_mps(x):
    return -x


BASELINES = {
    "mlx": mlx,
    "torch_mps": torch_mps,
}
"""

#: Single-threadgroup variant B for the finite domain (1<=N<=1024) so N=1024 does not launch
#: 1024 one-warp threadgroups when measured against MLX.
P3_DEVICE_B = b"""\
#include <metal_stdlib>
using namespace metal;
kernel void variant_b(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  if (i < N) {
    y[i] = half(-float(x[i]));
  }
}
/* taic-launch
math safe
launch variant_b grid=1,1,1 threads=1024,1,1
*/
"""


def _golden_gemm_source(contract, size: str) -> str:
    instance = instantiate_v2(contract.spec, "fp16", size)
    m, n, _k = (int(instance.symbols[name]) for name in ("M", "N", "K"))
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


def _naive_slow_gemm_source() -> str:
    return (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void gemm_kernel(device const half* A [[buffer(0)]],\n"
        "                        device const half* B [[buffer(1)]],\n"
        "                        device half* C [[buffer(2)]],\n"
        "                        uint gid [[thread_position_in_grid]]) {\n"
        "  if (gid > 0) return;\n"
        "  for (uint m = 0; m < M; ++m) {\n"
        "    for (uint n = 0; n < N; ++n) {\n"
        "      float acc = 0.0f;\n"
        "      for (uint k = 0; k < K; ++k) {\n"
        "        acc += float(A[m * K + k]) * float(B[k * N + n]);\n"
        "      }\n"
        "      C[m * N + n] = half(acc);\n"
        "    }\n"
        "  }\n"
        "}\n"
        "/* taic-launch\nmath fast\n"
        "launch gemm_kernel grid=1,1,1 threads=1,1,1\n*/\n"
    )


def _gemm_portfolio(contract, plan, *, small_source: str | None = None):
    small = CandidateEntry(
        path="device/gemm_small.metal",
        role="device_kernel",
        media_type="text/plain",
        data=(small_source if small_source is not None else _golden_gemm_source(contract, "small")).encode(),
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
    document = {
        "schema_id": "tcenv.candidate-portfolio.1",
        "domain_plan_digest": plan.digest,
        "planner": {"id": "p3-gemm-planner", "version": 1},
        "coverage_plan": {
            "buckets": [
                {"bucket_key": "small", "guard_digest": guard_digest(small_guard, where="small")},
                {"bucket_key": "paper", "guard_digest": guard_digest(paper_guard, where="paper")},
            ]
        },
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
    _authority = DomainAuthority(plan=plan, spec=contract.spec, target=target, instances=instances, contract=contract)
    portfolio = _gemm_portfolio(contract, plan)
    return locator, plan, portfolio


def _pointwise_fixture(tmp_path):
    root = tmp_path / "tasks"
    write_task(
        root,
        op_name=P1_OP,
        opspec=opspec(P1_OP, workspace=True),
        reference=SEM_REFERENCE,
        extra={"baselines.json": P1_BASELINES_JSON, "baselines.py": P1_BASELINES_PY},
    )
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    target = load_target(DEFAULT_TARGET_ID)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    portfolio = parse_portfolio(
        parent_entries(
            portfolio_document(plan.digest),
            extra_sources={"device/variant_b.metal": ("device_kernel", "text/plain", P3_DEVICE_B)},
        ),
        domain_plan=plan,
    )
    return locator, plan, portfolio


def _request(contract, plan, size):
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = plan.find("fp16", size, "benchmark")
    assert point is not None
    return DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="benchmark")


def _report_document(
    *,
    schema_id: str,
    task: str,
    domain_plan_digest: str,
    report: PerformanceReport,
    shards: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_id": schema_id,
        "policy": "performance-policy.json",
        "task": task,
        "domain_plan_digest": domain_plan_digest,
        "measurements": shards,
        "specialized_hit_latency_ratio": report.specialized_hit_latency_ratio,
        "effective_end_to_end_latency_ratio": report.effective_end_to_end_latency_ratio,
        "specialized_geomean_latency_ratio": report.specialized_geomean_latency_ratio,
        "effective_geomean_latency_ratio": report.effective_geomean_latency_ratio,
        "specialized_hit_speedup": report.specialized_hit_speedup,
        "effective_end_to_end_speedup": report.effective_end_to_end_speedup,
        "fallback_time_fraction": report.fallback_time_fraction,
        "fallback_count_fraction": report.fallback_count_fraction,
        "publishable": report.publishable,
        "failure_reasons": list(report.failure_reasons),
    }


def _assert_shard_identity(shard: dict[str, Any], *, expected_provider: str) -> None:
    assert shard["provider"] == expected_provider
    assert shard["credits_variant"] is (expected_provider == "candidate")
    assert shard["correctness_ok"] is True
    assert shard["dispatch_ok"] is True
    assert shard["timing_uncertain"] is False
    assert shard["unknown"] is False
    assert shard["dispatch"] and shard["dispatch"] == shard["expected_dispatch"]
    assert shard["best_baseline"] == "mlx"
    assert shard["baselines_measured"] == ["mlx", "torch_mps"]
    assert shard["device_name"] and "Apple" in shard["device_name"]
    for digest_key in (
        "candidate_sha256",
        "geometry_identity",
        "bundle_digest",
        "artifact_set_digest",
        "native_digest",
        "evaluation_fingerprint",
        "task_pack_digest",
    ):
        value = shard.get(digest_key)
        assert isinstance(value, str) and len(value) == 64, (shard["instance_key"], digest_key, value)


@gpu
def test_p3_b_gemm_small_performance_gate_passes(tmp_path):
    locator, plan, portfolio = _fixture(tmp_path)
    contract = locator.load()
    measurement, shard_doc = measure_shard_with_evidence(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=_request(contract, plan, "small"),
        evidence=evidence_all_valid(portfolio),
    )
    report = PerformanceReport.evaluate(
        [measurement],
        expected_benchmark_keys={measurement.instance_key},
    )
    _assert_shard_identity(shard_doc, expected_provider="candidate")
    assert shard_doc["variant_key"] == "gemm-small"
    assert measurement.latency_ratio <= 1.10, measurement
    assert report.publishable is True, report.failure_reasons
    assert report.specialized_hit_speedup is not None
    assert report.effective_end_to_end_speedup is not None
    directory = os.environ.get("SHAPE_PORTFOLIO_P3_EVIDENCE_DIR")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = _report_document(
            schema_id="tcenv.portfolio.p3-gemm-performance.1",
            task="gemm",
            domain_plan_digest=plan.digest,
            report=report,
            shards=[shard_doc],
        )
        (target / "gemm-performance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@gpu
def test_p3_b_gemm_paper_performance_gate_passes(tmp_path):
    locator, plan, portfolio = _fixture(tmp_path)
    contract = locator.load()
    measurement, shard_doc = measure_shard_with_evidence(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=_request(contract, plan, "paper"),
        evidence=evidence_all_valid(portfolio),
    )
    report = PerformanceReport.evaluate(
        [measurement], expected_benchmark_keys={measurement.instance_key}
    )
    _assert_shard_identity(shard_doc, expected_provider="candidate")
    assert shard_doc["variant_key"] == "gemm-paper"
    assert measurement.latency_ratio <= 1.10, measurement
    assert report.publishable is True, report.failure_reasons
    directory = os.environ.get("SHAPE_PORTFOLIO_P3_EVIDENCE_DIR")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_id": "tcenv.portfolio.p3-gemm-paper-performance.1",
            "policy": "performance-policy.json",
            "task": "gemm",
            "domain_plan_digest": plan.digest,
            **shard_doc,
            "specialized_hit_latency_ratio": report.specialized_hit_latency_ratio,
            "effective_end_to_end_latency_ratio": report.effective_end_to_end_latency_ratio,
            "specialized_hit_speedup": report.specialized_hit_speedup,
            "effective_end_to_end_speedup": report.effective_end_to_end_speedup,
            "publishable": report.publishable,
            "failure_reasons": list(report.failure_reasons),
        }
        (target / "paper-performance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        small_path = target / "gemm-performance.json"
        if small_path.exists():
            small_payload = json.loads(small_path.read_text(encoding="utf-8"))
            small_shard = small_payload["measurements"][0]
            small_m = ShardMeasurement(
                instance_key=small_shard["instance_key"],
                usage=small_shard["usage"],
                provider=small_shard["provider"],
                candidate_us=small_shard["candidate_us"],
                baseline_us=small_shard["baseline_us"],
                timing_uncertain=small_shard["timing_uncertain"],
                correctness_ok=small_shard["correctness_ok"],
                dispatch_ok=small_shard["dispatch_ok"],
                unknown=small_shard["unknown"],
            )
            domain_report = PerformanceReport.evaluate(
                [small_m, measurement],
                expected_benchmark_keys={small_m.instance_key, measurement.instance_key},
            )
            assert domain_report.publishable is True, domain_report.failure_reasons
            combined = _report_document(
                schema_id="tcenv.portfolio.p3-gemm-domain-performance.1",
                task="gemm",
                domain_plan_digest=plan.digest,
                report=domain_report,
                shards=[small_shard, shard_doc],
            )
            (target / "gemm-domain-performance.json").write_text(
                json.dumps(combined, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )


@gpu
def test_p3_c_pointwise_finite_domain_two_books_and_fallback_accounting(tmp_path):
    locator, plan, portfolio = _pointwise_fixture(tmp_path)
    contract = locator.load()
    measurements: list[ShardMeasurement] = []
    shard_docs: list[dict[str, Any]] = []
    for size in P1_SIZES:
        measurement, shard_doc = measure_shard_with_evidence(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(contract, plan, size),
            evidence=evidence_all_valid(portfolio),
        )
        expected = EXPECTED_PROVIDER[size]
        expected_provider = "fallback" if expected == "fallback" else "candidate"
        _assert_shard_identity(shard_doc, expected_provider=expected_provider)
        if expected_provider == "candidate":
            assert shard_doc["variant_key"] == expected
        else:
            assert shard_doc["fallback_reason"] == "outside_candidate_coverage"
        assert measurement.latency_ratio <= 1.10, (size, measurement)
        measurements.append(measurement)
        shard_docs.append(shard_doc)
    expected_keys = {f"fp16/{size}/benchmark" for size in P1_SIZES}
    report = PerformanceReport.evaluate(measurements, expected_benchmark_keys=expected_keys)
    assert report.publishable is True, report.failure_reasons
    assert report.fallback_count_fraction == pytest.approx(1.0 / 7.0)
    assert report.fallback_time_fraction is not None and 0.0 < report.fallback_time_fraction < 1.0
    assert report.specialized_hit_speedup is not None and report.specialized_hit_speedup >= 1.0 / 1.10
    assert report.effective_end_to_end_speedup is not None and report.effective_end_to_end_speedup >= 1.0 / 1.10
    directory = os.environ.get("SHAPE_PORTFOLIO_P3_EVIDENCE_DIR")
    if directory:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        payload = _report_document(
            schema_id="tcenv.portfolio.p3-pointwise-domain-performance.1",
            task=P1_OP,
            domain_plan_digest=plan.digest,
            report=report,
            shards=shard_docs,
        )
        (target / "pointwise-domain-performance.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


@gpu
def test_p3_c_real_gpu_local_regression_blocks_publication_despite_fast_peer(tmp_path):
    locator, plan, _ = _fixture(tmp_path)
    contract = locator.load()
    slow_portfolio = _gemm_portfolio(contract, plan, small_source=_naive_slow_gemm_source())
    slow_small = measure_shard(
        locator=locator,
        domain_plan=plan,
        portfolio=slow_portfolio,
        request=_request(contract, plan, "small"),
        evidence=evidence_all_valid(slow_portfolio),
    )
    assert slow_small.correctness_ok and slow_small.dispatch_ok
    assert slow_small.provider == "candidate"
    assert slow_small.latency_ratio is not None and slow_small.latency_ratio > 1.10
    report = PerformanceReport.evaluate(
        [slow_small],
        expected_benchmark_keys={slow_small.instance_key},
    )
    assert report.publishable is False
    assert "fp16/small/benchmark:specialized_ratio>1.10" in report.failure_reasons
    assert "fp16/small/benchmark:effective_ratio>1.10" in report.failure_reasons
    assert report.specialized_hit_speedup is None
    assert report.effective_end_to_end_speedup is None
