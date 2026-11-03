"""Shape portfolio P1: the real Mac vertical slice (GPU).

One finite elementwise task, two specialised variants with different implementations *and*
geometries, and one independent generic fallback.  Each node runs the real composition seam
(``run_portfolio_task``) with the real Metal adapters — ``MetalTrackBuilder``,
``MetalTrackEvaluator`` (a bounded worker process that owns the first real compile) and
``msl_single_geometry_resolver`` — so the evidence carries a real device, a real schema-4 report,
the actual dispatch the runtime executed, and the projected candidate/geometry/artifact
identities.

The nodes cover the dispatch boundaries at N=64 and N=128, threshold−1/threshold/threshold+1, the
odd tail (N=7), the candidate coverage hole (N=100 → the declared generic fallback) and the far
side of the second boundary (N=1024).  A fallback run is real evidence for *the request*; its
correctness and timing are recorded with ``provider=fallback`` and are never credited to a
generated variant.

Set ``SHAPE_PORTFOLIO_P1_EVIDENCE_DIR`` to archive one JSON record per node (the P1 verifier uses
it for ``--record-evidence`` and re-runs the same nodes to compare live against archived).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import pytest

from tcenv.backends.metal.track_adapter import (
    MetalTrackBuilder,
    MetalTrackEvaluator,
    msl_single_geometry_resolver,
)
from tcenv.backends.metal.v2_runtime import DispatchRecord as RuntimeDispatchRecord
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest, PortfolioDispatchRecord
import tcenv.taskcontract.portfolio_runner as portfolio_runner
from tcenv.portfolio.domain import normalize_semantic_domain_plan
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import run_portfolio_task
from tcenv.track import REPO_ROOT

from ...support.shape_portfolio_metal import (
    P1_SIZES,
    evidence_all_valid,
    load_p1_task,
    p1_instances,
    over_limit_entries,
    parent_entries,
    portfolio_document,
)

gpu = pytest.mark.gpu

EVIDENCE_SCHEMA_ID = "tcenv.portfolio.p1-evidence.1"
PROFILE = load_target(DEFAULT_TARGET_ID)
HOST = {"chip": platform.processor(), "machine": platform.machine(), "system": platform.system()}


def _record(name: str, payload: dict) -> None:
    """@brief 归档一条真实切片记录（仅当 verifier 设置了证据目录）。"""
    directory = os.environ.get("SHAPE_PORTFOLIO_P1_EVIDENCE_DIR")
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    # Grouped records carry the same host/设备 identity as the runs they group.
    payload = {"schema_id": EVIDENCE_SCHEMA_ID, "host": HOST, "target_id": DEFAULT_TARGET_ID, **payload}
    (target / name).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _fixture(tmp_path: Path):
    """@brief 固定题 + 权威实例 + 域计划 + 已解析 portfolio + locator。"""
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    return locator, contract, instances, plan, portfolio


def _run(
    plan,
    portfolio,
    locator,
    contract,
    size: str,
    *,
    mode: str = "quick",
    reps: int = 3,
    request=None,
    builder=None,
    evaluator=None,
):
    """@brief 用真实 Metal 适配器跑一次 portfolio 运行（可选注入计数代理）。"""
    body = request
    if body is None:
        instance = instantiate_v2(contract.spec, "fp16", size)
        point = plan.find("fp16", size, "golden")
        assert point is not None
        body = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="golden")
    return run_portfolio_task(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=body,
        evidence=evidence_all_valid(portfolio),
        builder=builder or MetalTrackBuilder(),
        evaluator=evaluator or MetalTrackEvaluator(mode=mode, seed=7, reps=reps, cases=("normal",)),
        geometry_resolver_factory=msl_single_geometry_resolver,
    )


class _CountingBuilder:
    """@brief 真实 ``MetalTrackBuilder`` 的计数代理（证明某条路径零构建）。"""

    build_kind = "native"

    def __init__(self, inner=None) -> None:
        """@brief 保存被代理的真实 builder。

        @param inner 真实 builder（默认 `MetalTrackBuilder`）。
        """
        self.inner = inner or MetalTrackBuilder()
        self.calls = 0

    def prepare(self, request):
        """@brief 计数并转调 ``prepare``。

        @param request 构建请求。
        """
        self.calls += 1
        return self.inner.prepare(request)

    def build(self, request):
        """@brief 计数并转调 ``build``。

        @param request 构建请求。
        @return 构建结果。
        """
        self.calls += 1
        return self.inner.build(request)


class _CountingEvaluator:
    """@brief 真实 ``MetalTrackEvaluator`` 的计数代理（证明某条路径零执行）。"""

    def __init__(self, inner) -> None:
        """@brief 保存被代理的真实 evaluator。

        @param inner 真实 evaluator。
        """
        self.inner = inner
        self.calls = 0

    def evaluate(self, request):
        """@brief 计数并转调 ``evaluate``。

        @param request 评测请求。
        @return schema-4 报告。
        """
        self.calls += 1
        return self.inner.evaluate(request)


def _label(portfolio, run) -> str:
    """@brief 命中的 provider 标签（variant_key 或 ``fallback``）。"""
    if run.record.provider == "candidate":
        return portfolio.variant_for_provider(run.record.variant_id).variant_key
    return "fallback"


def _evidence(node: str, slice_name: str, size: str, run, portfolio) -> dict:
    """@brief 一条真实切片的完整证据（dispatch + 投影 + 评测 + schema-4 报告）。"""
    report = run.result.report
    build = report.get("build") or {}
    entries = (build.get("artifacts") or {}).get("entries") or []
    return {
        "schema_id": EVIDENCE_SCHEMA_ID,
        "node": node,
        "slice": slice_name,
        "size": size,
        "n": int(P1_SIZES[size]["N"]),
        "instance_key": run.evaluation["instance_key"],
        "provider_label": _label(portfolio, run),
        "record": run.record.document(),
        "projection": run.projection.document(),
        "evaluation": dict(run.evaluation),
        "report": report,
        "run_identity": run.identity,
        "compiled_candidate_sha256": build.get("candidate_sha256"),
        "built_entry_paths": sorted(str(entry.get("path")) for entry in entries),
        "device": {
            "device_name": report.get("device_name"),
            "gpu_family": report.get("gpu_family"),
            "target_id": report.get("target_id"),
            "backend": report.get("backend"),
            "backend_version": report.get("backend_version"),
            "protocol_version": report.get("protocol_version"),
            "report_schema_version": report.get("schema_version"),
        },
        "host": HOST,
    }


def _assert_real_device(run) -> None:
    """@brief 真实设备与 schema-4 报告（不是模拟/离线记录）。"""
    report = run.result.report
    assert report.get("schema_version") == 4
    assert report.get("status") == "ok", report.get("message")
    assert report.get("abi", {}).get("ok") is True
    assert report.get("correctness", {}).get("ok") is True
    assert report.get("safety", {}).get("ok") is True
    assert report.get("target_id") == DEFAULT_TARGET_ID
    assert report.get("backend") == "metal"
    assert "M4 Max" in str(report.get("device_name"))
    assert report.get("gpu_family") == "Apple9"
    assert report.get("mode") in ("quick", "full")


def _assert_projection_matches(run, portfolio, size: str) -> None:
    """@brief candidate/geometry/artifact 身份与投影一致，且不是 parent portfolio 身份。"""
    projection = run.projection
    binding = projection.binding
    report = run.result.report
    build = report["build"]
    assert build["shape"] == "full"
    assert build["candidate_sha256"] == projection.candidate.candidate_sha256
    assert build["provenance"]["source_sha256"] == projection.candidate.candidate_sha256
    assert build["provenance"]["artifact_set_digest"] == run.result.artifact_set_digest
    assert run.result.artifact_set_digest != projection.candidate.candidate_sha256
    assert projection.candidate.candidate_sha256 != binding.parent_portfolio_sha256
    assert run.result.candidate_sha256 == projection.candidate.candidate_sha256
    assert run.result.geometry_identity == run.evaluation["expected_geometry_identity"]
    assert run.evaluation["compiled_candidate_sha256"] == projection.candidate.candidate_sha256
    # The original geometry entry path and content SHA are recorded, not the mapped path alone.
    assert run.evaluation["geometry_origin_path"] == binding.geometry_origin_path
    assert run.evaluation["geometry_origin_sha256"] == binding.geometry_origin_sha256
    if binding.geometry_origin_path is not None:
        parent_entry = portfolio.entry(binding.geometry_origin_path)
        assert hashlib.sha256(parent_entry.data).hexdigest() == binding.geometry_origin_sha256
        projected = next(entry for entry in projection.candidate.entries if entry.path == binding.geometry_path)
        assert projected.data == parent_entry.data
        paths = {entry["path"] for entry in build["artifacts"]["entries"]}
        assert binding.geometry_path in paths
    assert run.evaluation["dispatch"] == run.evaluation["expected_dispatch"]


def _assert_only_the_selected_candidate_was_built(run) -> None:
    """@brief 只编译最终选中的投影候选：构建产物只含它的 entries（+ 冻结几何/清单）。"""
    projection = run.projection
    paths = {entry["path"] for entry in run.result.report["build"]["artifacts"]["entries"]}
    expected = {entry.path for entry in projection.candidate.entries}
    assert expected <= paths
    assert run.result.report["build"]["candidate_sha256"] == projection.candidate.candidate_sha256
    # No other provider's source file can appear in this artifact set.
    for other in ("device/variant_a.metal", "device/variant_b.metal", "device/generic.metal"):
        if other not in expected:
            assert other not in paths


@gpu
def test_p1_gpu_threshold_variant_a_bundle_runs_host_tiling_and_a_real_dispatch(tmp_path, request):
    """@brief N=64：msl-bundle variant 真实 Host Tiling、workspace、回读与计时；身份与投影一致。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    run = _run(plan, portfolio, locator, contract, "threshold", mode="full", reps=3)
    assert PortfolioDispatchRecord is not RuntimeDispatchRecord
    assert run.record.outcome == "resolved"
    assert run.record.provider == "candidate"
    assert _label(portfolio, run) == "variant-a-bundle"
    assert run.projection.track_id == "msl-bundle"
    assert run.record.preflight.status == "ok"
    _assert_real_device(run)
    _assert_projection_matches(run, portfolio, "threshold")
    _assert_only_the_selected_candidate_was_built(run)
    evaluation = run.evaluation
    dispatch = evaluation["dispatch"]
    assert len(dispatch) == 1
    assert dispatch[0]["kernel"] == "variant_a"
    assert dispatch[0]["grid"] == [64, 1, 1]
    assert dispatch[0]["threads"] == [1, 1, 1]
    assert dispatch[0]["workspace"] == ["scratch"]
    assert evaluation["runtime_provider"] == "candidate"
    assert evaluation["fallback_geometry"] is None
    # Candidate Host Tiling really ran, and it wrote the workspace the kernel read.
    assert evaluation["host_tiling"] is not None
    assert evaluation["host_runs"], "the Host Tiling plan was never executed"
    assert evaluation["host_runs"][-1]["path"] == "host/host_tiling_a.json"
    assert evaluation["host_runs"][-1]["source_sha256"] == portfolio.entry("host/host_tiling_a.json").sha256
    assert evaluation["host_runs"][-1]["writes"] == 64 * 4
    assert evaluation["workspace_bytes"] == {"scratch": 64 * 4}
    assert evaluation["workspace_zero"] == {"scratch": False}
    assert evaluation["lifecycle"]["prepare"][0]["role"] == "host_tiling"
    assert evaluation["lifecycle"]["prepare"][0]["inside_timing"] is False
    # The existing timing boundary produced a real measurement; P1 publishes no speedup.
    benchmark = run.result.report["benchmark"]
    assert benchmark["candidate"]["median_us"] > 0
    assert benchmark["attempts"] >= 1
    assert run.result.report["speedup"] is None and run.result.report["speedups"] == {}
    assert evaluation["credits_variant"] is True
    _record(
        "threshold.json",
        _evidence(request.node.nodeid, "bundle-host-tiling", "threshold", run, portfolio),
    )


@gpu
def test_p1_gpu_threshold_minus_one_and_plus_one_switch_buckets(tmp_path, request):
    """@brief threshold−1 (N=63) 走单源 variant；threshold+1 (N=65) 走 bundle；grid 随实例解析。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    below = _run(plan, portfolio, locator, contract, "below")
    above = _run(plan, portfolio, locator, contract, "above")
    assert _label(portfolio, below) == "variant-b-single"
    assert _label(portfolio, above) == "variant-a-bundle"
    for run, size, kernel, grid, threads, workspace in (
        (below, "below", "variant_b", 63, 32, []),
        (above, "above", "variant_a", 65, 1, ["scratch"]),
    ):
        _assert_real_device(run)
        _assert_projection_matches(run, portfolio, size)
        _assert_only_the_selected_candidate_was_built(run)
        dispatch = run.evaluation["dispatch"]
        assert len(dispatch) == 1
        assert dispatch[0]["kernel"] == kernel
        assert dispatch[0]["grid"] == [grid, 1, 1]
        assert dispatch[0]["threads"] == [threads, 1, 1]
        assert dispatch[0]["workspace"] == workspace
        assert run.evaluation["runtime_provider"] == "candidate"
    # Only the bundle instance declares/executes Host Tiling; the single-source one owns its launch.
    assert below.evaluation["host_tiling"] is None and below.evaluation["host_runs"] == []
    assert above.evaluation["host_tiling"] is not None and above.evaluation["host_runs"]
    assert above.evaluation["workspace_bytes"] == {"scratch": 65 * 4}
    assert below.projection.candidate.candidate_sha256 != above.projection.candidate.candidate_sha256
    assert below.result.geometry_identity != above.result.geometry_identity
    _record(
        "boundaries.json",
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "node": request.node.nodeid,
            "slice": "threshold-minus-one-and-plus-one",
            "below": _evidence(request.node.nodeid, "below", "below", below, portfolio),
            "above": _evidence(request.node.nodeid, "above", "above", above, portfolio),
        },
    )


@gpu
def test_p1_gpu_odd_tail_and_second_boundary_stay_in_the_single_source_variant(tmp_path, request):
    """@brief 奇数 tail (N=7) 与第二个边界/远侧 (N=128/1024) 由同一单源 variant 真实执行。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    runs = {}
    for size, grid in (("tail", 7), ("boundary", 128), ("large", 1024)):
        run = _run(plan, portfolio, locator, contract, size)
        assert _label(portfolio, run) == "variant-b-single"
        assert run.projection.track_id == "msl-single"
        _assert_real_device(run)
        _assert_projection_matches(run, portfolio, size)
        _assert_only_the_selected_candidate_was_built(run)
        dispatch = run.evaluation["dispatch"]
        assert dispatch[0]["kernel"] == "variant_b"
        assert dispatch[0]["grid"] == [grid, 1, 1]
        assert dispatch[0]["threads"] == [32, 1, 1]
        assert dispatch[0]["workspace"] == []
        assert run.evaluation["host_tiling"] is None
        runs[size] = run
    # Same candidate bytes, three instance-resolved geometries.
    assert len({run.projection.candidate.candidate_sha256 for run in runs.values()}) == 1
    assert len({run.result.geometry_identity for run in runs.values()}) == 3
    _record(
        "single_source.json",
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "node": request.node.nodeid,
            "slice": "odd-tail-and-second-boundary",
            "runs": {
                size: _evidence(
                    request.node.nodeid,
                    size,
                    size,
                    run,
                    portfolio,
                )
                for size, run in runs.items()
            },
        },
    )


@gpu
def test_p1_gpu_a_single_source_launch_overflow_runs_only_the_declared_fallback(tmp_path, request, monkeypatch):
    """@brief 源内 launch 真的超限：真实解析器+limits 在构建前分流，只编译执行 generic fallback。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    over_portfolio = parse_portfolio(over_limit_entries(portfolio_document(plan.digest)), domain_plan=plan)
    evaluated: list[str] = []
    real_evaluate = portfolio_runner.evaluate_task

    def evaluate_spy(*args, **kwargs):
        evaluated.append(kwargs["candidate"].candidate_sha256)
        return real_evaluate(*args, **kwargs)

    # Count the composition seam itself on this real boundary, not only the Metal adapters beneath
    # it: zero adapter requests does not by itself prove ``evaluate_task`` was never entered.
    monkeypatch.setattr(portfolio_runner, "evaluate_task", evaluate_spy)
    assert portfolio_runner.evaluate_task is evaluate_spy
    # N=63 is variant-b's bucket; its launch now exceeds max_total_threads, and the fallback
    # declares resource_unavailable, so the generic fallback must be the only thing built.
    run = _run(plan, over_portfolio, locator, contract, "below", mode="full", reps=3)
    assert run.record.outcome == "fallback"
    assert run.record.fallback_reason == "resource_unavailable"
    assert run.record.variant_id == over_portfolio.variants[1].variant_id
    assert "exceeds the target limits" in (run.record.detail or "")
    assert run.record.preflight.status == "ok"
    assert run.projection.provider == "fallback"
    assert run.evaluation["credits_variant"] is False
    # Exactly one entry into evaluate_task, and it got the final fallback candidate.
    assert evaluated == [run.projection.candidate.candidate_sha256], evaluated
    _assert_real_device(run)
    _assert_projection_matches(run, over_portfolio, "below")
    _assert_only_the_selected_candidate_was_built(run)
    dispatch = run.evaluation["dispatch"]
    assert dispatch[0]["kernel"] == "generic"
    assert dispatch[0]["threads"] == [1024, 1, 1]
    assert run.evaluation["host_tiling"] is None
    # The over-limit variant was never compiled: its source is absent from the artifact set.
    assert "device/variant_b.metal" not in {
        entry["path"] for entry in run.result.report["build"]["artifacts"]["entries"]
    }

    # The same overflow without the declared reason fails before any build: no final projection is
    # returned, nothing is built or executed, and evaluate_task is never entered, so no GPU work
    # happens for a fallback that did not register the reason.
    narrow = portfolio_document(plan.digest)
    narrow["fallback"]["reasons"] = ["outside_candidate_coverage"]
    narrow_portfolio = parse_portfolio(over_limit_entries(narrow), domain_plan=plan)
    counting_builder, counting_evaluator = _CountingBuilder(), _CountingEvaluator(
        MetalTrackEvaluator(mode="quick", seed=7, reps=3, cases=("normal",))
    )
    refused = _run(
        plan,
        narrow_portfolio,
        locator,
        contract,
        "below",
        builder=counting_builder,
        evaluator=counting_evaluator,
    )
    assert refused.record.outcome == "variant_unavailable"
    assert refused.record.provider is None
    # A variant that cannot run and whose fallback never registered the reason is *not* a fallback
    # record: the outcome is variant_unavailable with no provider and no reason.
    assert refused.record.fallback_reason is None
    assert refused.record.variant_id == narrow_portfolio.variants[1].variant_id
    assert refused.record.preflight.status == "resource_unavailable"
    assert "does not declare resource_unavailable" in (refused.record.detail or "")
    assert refused.projection is None and refused.result is None and refused.evaluation is None
    # Zero device work: the declared fallback may not be borrowed, and the over-limit variant is
    # never compiled or executed.
    assert counting_builder.calls == 0 and counting_evaluator.calls == 0
    # The boundary itself: the refused request never reached the evaluate_task seam, so the count
    # is still exactly the one call the declared run made.
    declared_sha = run.projection.candidate.candidate_sha256
    assert evaluated == [declared_sha], (
        f"the refused run entered evaluate_task: the seam saw {evaluated}, expected only [{declared_sha}]"
    )
    _record(
        "single_source_overflow.json",
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "node": request.node.nodeid,
            "slice": "single-source-launch-overflow",
            "run": _evidence(request.node.nodeid, "below", "below", run, over_portfolio),
            # The declared-but-not-registered negative, archived in full so it can be replayed and
            # compared field by field (nothing here is timing-dependent).
            "refused": {
                "schema_id": EVIDENCE_SCHEMA_ID,
                "slice": "single-source-launch-overflow-refused",
                "request": refused.record.request,
                "record": refused.record.document(),
                "portfolio_digest": narrow_portfolio.digest,
                "domain_plan_digest": plan.digest,
                "built_entry_paths": [],
                "projection": None,
                "result": None,
                "evaluation": None,
                "builder_calls": counting_builder.calls,
                "evaluator_calls": counting_evaluator.calls,
            },
        },
    )


@gpu
def test_p1_gpu_an_out_of_domain_request_is_refused_without_fallback_or_device_work(tmp_path, request, monkeypatch):
    """@brief 语义域外（fp32/未命名 size）真实设备拒绝：unsupported_contract_capability，零构建执行。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    threshold = plan.find("fp16", "threshold", "golden")
    assert threshold is not None
    instance = instantiate_v2(contract.spec, "fp16", "threshold")
    base = DispatchRequest.from_instance(
        instance, target_abi_digest=threshold.target_abi_digest, usage="golden"
    ).document()

    # Two out-of-domain requests: a dtype the finite domain never names, and a size it never names.
    # Both are coherent documents (they parse), so the refusal must come from the *domain*, not from
    # a malformed request: the fallback is a correctness provider inside the domain, never an
    # escape hatch for illegal semantics.
    cases = []
    fp32 = dict(base)
    fp32["dtype"] = "fp32"
    fp32["instance_key"] = "fp32/threshold/golden"
    cases.append(("fp32/threshold/golden", fp32))
    unknown = dict(base)
    unknown["size"] = "not_a_declared_size"
    unknown["instance_key"] = "fp16/not_a_declared_size/golden"
    cases.append(("fp16/not_a_declared_size/golden", unknown))

    evaluated: list[str] = []
    real_evaluate = portfolio_runner.evaluate_task

    def evaluate_spy(*args, **kwargs):
        evaluated.append(kwargs["candidate"].candidate_sha256)
        return real_evaluate(*args, **kwargs)

    # The composition seam itself, on the real boundary: an out-of-domain request must not reach it.
    monkeypatch.setattr(portfolio_runner, "evaluate_task", evaluate_spy)
    assert portfolio_runner.evaluate_task is evaluate_spy

    refused = []
    for key, body in cases:
        builder, evaluator = _CountingBuilder(), _CountingEvaluator(
            MetalTrackEvaluator(mode="quick", seed=7, reps=3, cases=("normal",))
        )
        run = _run(
            plan,
            portfolio,
            locator,
            contract,
            "threshold",
            request=body,
            builder=builder,
            evaluator=evaluator,
        )
        assert run.record.outcome == "unsupported_contract_capability", key
        assert run.record.provider is None and run.record.fallback_reason is None
        assert run.record.preflight.status == "not_required"
        assert key in (run.record.detail or "")
        assert run.projection is None and run.result is None and run.evaluation is None
        assert builder.calls == 0 and evaluator.calls == 0
        assert evaluated == [], f"{key}: evaluate_task was entered for an out-of-domain request"
        refused.append(
            {
                "schema_id": EVIDENCE_SCHEMA_ID,
                "slice": "semantic-domain-outside-request",
                "instance_key": key,
                "request": body,
                "record": run.record.document(),
                "portfolio_digest": portfolio.digest,
                "domain_plan_digest": plan.digest,
                "built_entry_paths": [],
                "projection": None,
                "result": None,
                "evaluation": None,
                "builder_calls": builder.calls,
                "evaluator_calls": evaluator.calls,
            }
        )
    assert len(refused) == 2
    _record(
        "out_of_domain.json",
        {
            "schema_id": EVIDENCE_SCHEMA_ID,
            "node": request.node.nodeid,
            "slice": "semantic-domain-outside-requests",
            "refused": refused,
        },
    )


@gpu
def test_p1_gpu_coverage_hole_runs_the_generic_fallback_without_variant_credit(tmp_path, request):
    """@brief 覆盖空洞 (N=100)：声明 fallback 真执行；provider=fallback，不计入 variant 记功。"""
    locator, contract, _instances, plan, portfolio = _fixture(tmp_path)
    run = _run(plan, portfolio, locator, contract, "hole", mode="full", reps=3)
    assert run.record.outcome == "fallback"
    assert run.record.provider == "fallback"
    assert run.record.fallback_reason == "outside_candidate_coverage"
    assert run.record.variant_id is None
    assert run.projection.provider == "fallback"
    assert run.projection.identity == portfolio.fallback_id
    assert run.projection.track_id == "msl-single"
    assert run.evaluation["credits_variant"] is False
    assert run.evaluation["fallback_reason"] == "outside_candidate_coverage"
    # The projected fallback is an ordinary candidate artifact, not a §A2 GeometryPair fallback.
    assert run.evaluation["runtime_provider"] == "candidate"
    assert run.evaluation["fallback_geometry"] is None
    _assert_real_device(run)
    _assert_projection_matches(run, portfolio, "hole")
    _assert_only_the_selected_candidate_was_built(run)
    report = run.result.report
    dispatch = run.evaluation["dispatch"]
    assert dispatch[0]["kernel"] == "generic"
    assert dispatch[0]["grid"] == [1, 1, 1]
    assert dispatch[0]["threads"] == [1024, 1, 1]
    assert dispatch[0]["workspace"] == []
    assert run.evaluation["host_tiling"] is None
    # The fallback's own timing exists, and it is recorded as the fallback's, not the variant's.
    assert report["benchmark"]["candidate"]["median_us"] > 0
    assert report["speedup"] is None
    assert run.result.candidate_sha256 not in {
        run.projection.binding.parent_portfolio_sha256,
        portfolio.variant_ids[0],
        portfolio.variant_ids[1],
    }
    _record(
        "fallback.json",
        _evidence(request.node.nodeid, "fallback-hole", "hole", run, portfolio),
    )
