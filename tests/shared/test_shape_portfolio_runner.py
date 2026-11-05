"""Shape portfolio P1: the runner seam — parse → dispatch → CPU preflight → project → evaluate.

These nodes are backend-free: the builder, evaluator and the ``msl-single`` launch resolver are
injected adapters, exactly as in production (``evaluate_task`` takes them as arguments and the
Metal implementations live in ``tcenv.backends.metal``).  What is under test is the *composition*:

* a request is parsed and bound to its authoritative instance, dispatch selects exactly one
  provider, and only that provider is projected and built;
* the selected geometry is resolved on the CPU (``resolve_geometry`` with the real instance/ABI/
  limits) **before** ``evaluate_task``, so a resource overflow never surfaces from inside the
  evaluator and a fallback only happens when it declared ``resource_unavailable``;
* the evaluated geometry identity is the one the runner computed, the runtime dispatch matches it
  field by field, and the runtime's ``provider`` stays ``candidate`` (the domain fallback is an
  ordinary projected artifact, never a §A2 ``GeometryPair`` fallback);
* identities cannot be swapped: parent portfolio SHA is not a child SHA, a fallback is not a
  candidate, a stale artifact or a mislabelled evaluation is refused, and the composition record
  is not the runtime dispatch record.

The real Metal slice (real device, compiler, dispatch and schema-4 report) is
``tests/backends/metal/test_shape_portfolio_gpu.py``.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import tcenv.taskcontract.portfolio_runner as portfolio_runner
from tcenv.build_projection import project_full
from tcenv.contract_errors import TaskContractError
from tcenv.evaluate import EvalReport
from tcenv.identity import fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import EvaluationBinding, parse_portfolio, project_provider
from tcenv.portfolio.dispatch import DISPATCH_RECORD_SCHEMA_ID, DispatchRequest, PortfolioDispatchRecord
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProfile,
    BuildProvenance,
)
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, NativeExecutable
from tcenv.schema import PROTOCOL_VERSION
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import (
    PORTFOLIO_RUN_SCHEMA_ID,
    PortfolioRunResult,
    run_portfolio_task,
)
from tcenv.track import ExecutionGeometry, GeometryStage, REPO_ROOT
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from tcenv.track.plane import CandidateArtifact

from ..support.shape_portfolio_metal import (
    EXPECTED_PROVIDER,
    P1_SIZES,
    bundle_fallback_document,
    evidence_all_valid,
    geometry_bytes,
    geometry_document,
    load_p1_task,
    p1_instances,
    parent_entries,
    portfolio_document,
)

PROFILE = load_target(DEFAULT_TARGET_ID)
BUILD_PROFILE = BuildProfile(
    build_profile="metal-msl-control-host-v1",
    builder_id="metal-source-builder",
    builder_version="1",
    build_host="control_host",
    toolchain_id="apple-metal",
    toolchain_version="unknown",
    sdk_version="unknown",
    compile_profile={"language": "msl", "math": "safe", "device": "test", "gpu_family": "test"},
)
#: The injected ``msl-single`` launch adapter used by these CPU nodes.  It mirrors the fixture's
#: two single-source launches (kernel + threads); the *real* parser is the Metal adapter run by
#: the GPU node, so this stub can never stand in for a production launch decision.
SINGLE_THREADS = {"variant_b": 32, "generic": 1024}


class RecordingBuilder:
    """@brief 记录并结构化地“构建”被选中候选（AUXILIARY entries + native）。"""

    def __init__(self, *, wrong_source: bool = False) -> None:
        self.requests: list = []
        self.wrong_source = wrong_source

    def build(self, request) -> BuildOutcome:
        self.requests.append(request)
        candidate = request.candidate
        if candidate.single_source:
            entry = candidate.entries[0]
            entries = (
                BundleEntry(
                    path=entry.path,
                    role="SOURCE",
                    media_type=entry.media_type,
                    size_bytes=len(entry.data),
                    sha256=entry.sha256,
                    data=bytes(entry.data),
                ),
            )
            native_path = entry.path
        else:
            entries = tuple(
                BundleEntry(
                    path=entry.path,
                    role="AUXILIARY",
                    media_type=entry.media_type,
                    size_bytes=len(entry.data),
                    sha256=entry.sha256,
                    data=bytes(entry.data),
                )
                for entry in candidate.entries
            )
            native_path = next(entry for entry in candidate.entries if entry.role == "device_kernel").path
        native = NativeExecutable(
            backend_id="metal",
            kind="compute-pipeline",
            entry_path=native_path,
            entry_sha256=next(entry.sha256 for entry in entries if entry.path == native_path),
            descriptor={"backend": "metal", "kernel_path": native_path},
            handle=object(),
        )
        bundle = ArtifactBundle(entries=entries, native=native)
        provenance = BuildProvenance(
            profile=BUILD_PROFILE,
            source_sha256=("0" * 64) if self.wrong_source else candidate.candidate_sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        evidence = BuildEvidence(
            mode="prebuilt",
            status="ok",
            provenance=provenance,
            artifacts=tuple(BuildArtifact.from_entry(entry) for entry in bundle.entries),
            placement="control_host",
            native=native,
        )
        return BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)


def _workspace_bytes(abi) -> dict[str, int]:
    """@brief ABI workspace 的字节跨度（与真实 runtime 的证据字段同义）。"""
    from tcenv.dtypes import get_dtype
    from tcenv.track.abi import tensor_span_bytes

    return {
        tensor.name: tensor_span_bytes(
            tuple(int(v) for v in tensor.shape), tuple(int(v) for v in tensor.strides), get_dtype(tensor.dtype).nbytes
        )
        for tensor in abi.tensors
        if tensor.space == "workspace"
    }


class RecordingEvaluator:
    """@brief 返回合法 schema-4 报告 + 真实形状的 ``last_execution``（CPU 结构替身）。"""

    def __init__(self, *, status: str = "ok", no_execution: bool = False) -> None:
        self.status = status
        self.no_execution = no_execution
        self.requests: list = []
        self.last_execution: dict | None = None

    def evaluate(self, request) -> dict:
        self.requests.append(request)
        build = (
            project_full(
                request.build,
                build_kind="native",
                identity={"opspec_digest": "a" * 64, "lowering_profile_digest": "b" * 64},
            )
            if self.status == "ok"
            else None
        )
        locator = request.build_request.locator
        report = EvalReport(
            task=locator.op_name,
            dtype=locator.dtype,
            size=locator.size,
            mode="full",
            status=self.status,
            stage="bench" if self.status == "ok" else self.status,
            seed=0,
            build=build,
            abi={"ok": True, "violations": []},
            correctness={"ok": True},
            safety={"ok": True, "violations": []},
            protocol_version=PROTOCOL_VERSION,
            target_id=DEFAULT_TARGET_ID,
            backend="metal",
            backend_version="1",
            device_name="Apple M4 Max",
            gpu_family="Apple9",
        )
        candidate = request.build_request.candidate
        geometry = request.geometry
        host_entry = next((entry for entry in candidate.entries if entry.role == "host_tiling"), None)
        workspaces = _workspace_bytes(request.build_request.abi)
        self.last_execution = None if self.no_execution else {
            "provider": "candidate",
            "geometry_identity": request.canonical_bundle.geometry_identity,
            "bindings": [],
            "workspace_bytes": workspaces,
            "workspace_zero": {use.name: bool(use.zero) for use in geometry.workspace_uses},
            "workspace_fills": [use.name for use in geometry.workspace_uses if use.zero],
            "host_tiling": None if host_entry is None else json.loads(host_entry.data.decode("utf-8")),
            "dispatch": [
                {
                    "stage": int(stage.index),
                    "kernel": stage.kernel,
                    "grid": [int(value) for value in stage.grid],
                    "threads": [int(value) for value in stage.threads],
                    "threadgroup_bytes": int(stage.threadgroup_bytes),
                    "workspace": [str(use.name) for use in stage.workspace],
                    "provider": "candidate",
                    "sequence": index,
                }
                for index, stage in enumerate(geometry.active_stages)
            ],
            "host_runs": []
            if host_entry is None
            else [
                {
                    "path": host_entry.path,
                    "source_sha256": host_entry.sha256,
                    "plan_sha256": hashlib.sha256(host_entry.data).hexdigest(),
                    "writes": int(workspaces.get("scratch", 0)),
                }
            ],
        }
        return report.to_json()


def single_source_resolver(candidate: CandidateArtifact) -> ExecutionGeometry:
    """@brief 测试注入的 ``msl-single`` 几何解析器（不是生产 launch 解析器）。

    The production resolver parses the candidate's ``/* taic-launch */`` block in the Metal
    adapter.  This stub only maps the fixture's two single-source kernels to their declared
    geometry so the composition seam can be exercised on the CPU; the real block, its workspace
    cross-check and its dispatch are exercised by the GPU node.

    @param candidate 投影后的单源候选。
    @return 执行几何。
    """
    text = candidate.entries[0].data.decode("utf-8")
    kernel = next(name for name in SINGLE_THREADS if f"kernel void {name}(" in text)
    return ExecutionGeometry(
        source="candidate",
        stages=(
            GeometryStage(
                kernel=kernel,
                grid=("N", 1, 1),
                threads=(SINGLE_THREADS[kernel], 1, 1),
                workspace=(),
            ),
        ),
    )


def single_source_resolver_with_limit(
    *, kernel: str, threads: int
) -> "callable":
    """@brief 注入一个按 kernel 声明超限 threads 的 ``msl-single`` 解析器（CPU 替身）。

    The production resolver parses the candidate's ``/* taic-launch */`` block; these CPU nodes
    substitute the same *geometry* (one stage, ``grid=N``, the declared thread count) so the
    resource preflight can be exercised without a backend.  The real launch block and the real
    limits are exercised by the GPU nodes.

    @param kernel 声明超限 threads 的 kernel 名。
    @param threads 该 kernel 的线程数。
    @return resolver 工厂。
    """

    def factory(_contract):
        def resolve(candidate: CandidateArtifact) -> ExecutionGeometry:
            text = candidate.entries[0].data.decode("utf-8")
            name = next(item for item in SINGLE_THREADS if f"kernel void {item}(" in text)
            count = threads if name == kernel else SINGLE_THREADS[name]
            return ExecutionGeometry(
                source="candidate",
                stages=(GeometryStage(kernel=name, grid=("N", 1, 1), threads=(count, 1, 1), workspace=()),),
            )

        return resolve

    return factory


def single_source_resolver_repeating(candidate: CandidateArtifact, *, stages: int = 2) -> ExecutionGeometry:
    """@brief 同一 kernel/几何重复 ``stages`` 次的单源执行几何（合法多阶段候选）。

    Two active stages with an *identical* signature are legal: the candidate reads the same buffers
    twice with the same launch shape.  The composition seam must keep both stages and accept the
    dispatch, not fold them into one.

    @param candidate 投影后的单源候选。
    @param stages 阶段数。
    @return 执行几何。
    """
    text = candidate.entries[0].data.decode("utf-8")
    kernel = next(name for name in SINGLE_THREADS if f"kernel void {name}(" in text)
    return ExecutionGeometry(
        source="candidate",
        stages=tuple(
            GeometryStage(
                kernel=kernel,
                grid=("N", 1, 1),
                threads=(SINGLE_THREADS[kernel], 1, 1),
                workspace=(),
            )
            for _ in range(stages)
        ),
    )


def _setup(tmp_path: Path, *, document=None, entry_kwargs=None):
    """@brief 固定题 + 权威实例 + 域计划 + 已解析 portfolio + locator。

    @param tmp_path 临时目录。
    @param document envelope 覆盖。
    @param entry_kwargs ``parent_entries`` 参数。
    @return ``(contract, instances, plan, portfolio, locator)``。
    """
    root = tmp_path / "tasks"
    load_p1_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    contract = locator.load()
    instances = p1_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE)
    body = document if document is not None else portfolio_document(plan.digest)
    portfolio = parse_portfolio(parent_entries(body, **(entry_kwargs or {})), domain_plan=plan)
    return contract, instances, plan, portfolio, locator


def _authority(plan, contract, instances) -> DomainAuthority:
    """@brief 该 portfolio 的权威来源（投影用）。"""
    return DomainAuthority(plan=plan, spec=contract.spec, target=PROFILE, instances=instances, contract=contract)


def _projection_shas(plan, contract, instances, portfolio) -> dict[str, str]:
    """@brief 每个 provider 投影后的子候选 SHA（用于“只构建被选中者”的断言）。"""
    authority = _authority(plan, contract, instances)
    shas: dict[str, str] = {}
    for variant in portfolio.variants:
        shas[variant.variant_key] = project_provider(portfolio, variant, authority=authority).candidate.candidate_sha256
    if portfolio.fallback is not None and portfolio.fallback.kind != "refuse":
        shas["fallback"] = project_provider(
            portfolio, portfolio.fallback, authority=authority
        ).candidate.candidate_sha256
    return shas


def _request(plan, contract, size: str, usage: str = "golden") -> DispatchRequest:
    """@brief 由真实实例构造请求（摘要由实例计算，测试不能伪造）。"""
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = plan.find("fp16", size, usage)
    assert point is not None
    return DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)


def _run(
    plan,
    portfolio,
    locator,
    contract,
    size: str,
    *,
    builder=None,
    evaluator=None,
    resolver=None,
    usage: str = "golden",
    request=None,
) -> PortfolioRunResult:
    """@brief 用固定适配器跑一次 portfolio 运行。"""
    builder = builder or RecordingBuilder()
    evaluator = evaluator or RecordingEvaluator()
    return run_portfolio_task(
        locator=locator,
        domain_plan=plan,
        portfolio=portfolio,
        request=request if request is not None else _request(plan, contract, size, usage),
        evidence=evidence_all_valid(portfolio),
        builder=builder,
        evaluator=evaluator,
        geometry_resolver_factory=resolver or (lambda _contract: single_source_resolver),
    )


def _label(portfolio, record) -> str:
    """@brief 记录对应的 provider 标签（variant_key 或 ``fallback``）。"""
    if record.provider == "candidate":
        return portfolio.variant_for_provider(record.variant_id).variant_key
    return "fallback"


def test_p1_dispatch_selects_exactly_one_provider_and_only_that_provider_is_built(tmp_path):
    """@brief 解析→exactly-one dispatch→只投影/构建被选中者；记录、投影、评测同一身份。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(plan, portfolio, locator, contract, "threshold", builder=builder, evaluator=evaluator)
    assert run.record.outcome == "resolved"
    assert run.record.provider == "candidate"
    assert run.projection is not None and run.result is not None
    assert run.projection.identity == run.record.variant_id
    assert run.projection.track_id == "msl-bundle"
    assert len(builder.requests) == 1 and len(evaluator.requests) == 1
    assert builder.requests[0].candidate.candidate_sha256 == run.projection.candidate.candidate_sha256
    assert run.result.candidate_sha256 == run.projection.candidate.candidate_sha256
    assert run.result.geometry_identity == run.evaluation["expected_geometry_identity"]
    assert run.evaluation["runtime_provider"] == "candidate"
    assert run.evaluation["dispatch"] == run.evaluation["expected_dispatch"]
    document = run.document()
    assert document["schema_id"] == PORTFOLIO_RUN_SCHEMA_ID
    assert document["active_manifest"] is None and document["activation_generation"] is None
    assert document["performance_claim"] is None and document["fallback_credits_variant"] is False
    assert document["identities"]["candidate_sha256"] != document["identities"]["parent_portfolio_sha256"]
    assert run.identity == fingerprint(document)


def test_p1_threshold_minus_one_threshold_and_plus_one_route_to_the_declared_buckets(tmp_path):
    """@brief threshold−1/threshold/threshold+1：每点恰好一个 provider，且只有它被构建。"""
    contract, instances, plan, portfolio, locator = _setup(tmp_path)
    expected_shas = _projection_shas(plan, contract, instances, portfolio)
    for size in ("below", "threshold", "above"):
        builder, evaluator = RecordingBuilder(), RecordingEvaluator()
        run = _run(plan, portfolio, locator, contract, size, builder=builder, evaluator=evaluator)
        label = _label(portfolio, run.record)
        assert label == EXPECTED_PROVIDER[size], size
        assert len(builder.requests) == 1 and len(evaluator.requests) == 1
        built = builder.requests[0].candidate.candidate_sha256
        assert built == run.projection.candidate.candidate_sha256 == expected_shas[label]
        assert built != expected_shas["fallback"]
        assert run.evaluation["compiled_candidate_sha256"] == built
        assert run.projection.binding.geometry_origin_path == (
            "host/geometry_a.json" if size in ("threshold", "above") else None
        )


def test_p1_the_coverage_hole_runs_the_generic_fallback_and_the_odd_tail_stays_specialized(tmp_path):
    """@brief 覆盖空洞→outside_candidate_coverage 的 fallback；奇数 tail 仍由单源 variant 命中。"""
    contract, instances, plan, portfolio, locator = _setup(tmp_path)
    expected_shas = _projection_shas(plan, contract, instances, portfolio)
    run = _run(plan, portfolio, locator, contract, "tail")
    assert _label(portfolio, run.record) == "variant-b-single"
    assert run.projection.track_id == "msl-single"
    assert run.evaluation["dispatch"][0]["grid"] == [int(P1_SIZES["tail"]["N"]), 1, 1]
    for size in ("hole",):
        builder, evaluator = RecordingBuilder(), RecordingEvaluator()
        run = _run(plan, portfolio, locator, contract, size, builder=builder, evaluator=evaluator)
        assert run.record.outcome == "fallback"
        assert run.record.provider == "fallback"
        assert run.record.fallback_reason == "outside_candidate_coverage"
        assert run.record.variant_id is None
        assert run.projection.provider == "fallback"
        assert run.projection.identity == portfolio.fallback_id
        assert run.projection.track_id == "msl-single"
        assert len(builder.requests) == 1
        assert builder.requests[0].candidate.candidate_sha256 == expected_shas["fallback"]
        assert run.evaluation["credits_variant"] is False
        assert run.document()["fallback_credits_variant"] is False
        # A domain fallback is an ordinary candidate artifact: the §A2 geometry provider is candidate.
        assert run.evaluation["runtime_provider"] == "candidate"
        assert run.evaluation["fallback_geometry"] is None
        assert run.evaluation["fallback_reason"] == "outside_candidate_coverage"
        assert run.evaluation["provider"] == "fallback"


def test_p1_the_second_boundary_keeps_the_other_single_source_geometry(tmp_path):
    """@brief 第二个边界（N=128）与其远侧（N=1024）都由单源 variant 命中，几何与 bundle 不同。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    runs = {}
    for size in ("boundary", "large"):
        run = _run(plan, portfolio, locator, contract, size)
        assert _label(portfolio, run.record) == EXPECTED_PROVIDER[size]
        assert run.projection.track_id == "msl-single"
        assert run.projection.binding.geometry_origin_path is None
        assert run.evaluation["host_tiling"] is None and run.evaluation["host_runs"] == []
        assert run.evaluation["dispatch"][0]["threads"] == [32, 1, 1]
        assert run.evaluation["dispatch"][0]["grid"] == [int(P1_SIZES[size]["N"]), 1, 1]
        runs[size] = run
    assert runs["boundary"].result.geometry_identity != runs["large"].result.geometry_identity
    assert (
        runs["boundary"].projection.candidate.candidate_sha256
        == runs["large"].projection.candidate.candidate_sha256
    )


def test_p1_a_semantic_domain_outside_request_never_falls_back_and_never_builds(tmp_path):
    """@brief 语义域外/伪造请求：semantic_unresolved 或 unsupported_contract_capability，永不 fallback。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    document = _request(plan, contract, "threshold").document()

    forged = copy.deepcopy(document)
    forged["semantic_instance_digest"] = "f" * 64
    outside = copy.deepcopy(document)
    outside["dtype"] = "fp32"
    outside["instance_key"] = "fp32/threshold/golden"
    unknown = copy.deepcopy(document)
    unknown["size"] = "not_a_declared_size"
    unknown["instance_key"] = "fp16/not_a_declared_size/golden"
    junk = {"op_name": "x"}
    for body, expected in (
        (forged, "semantic_unresolved"),
        (outside, "unsupported_contract_capability"),
        (unknown, "unsupported_contract_capability"),
        (junk, "semantic_unresolved"),
    ):
        builder, evaluator = RecordingBuilder(), RecordingEvaluator()
        run = _run(
            plan, portfolio, locator, contract, "threshold", builder=builder, evaluator=evaluator, request=body
        )
        assert run.record.outcome == expected, (body, run.record.outcome)
        assert run.record.provider is None
        assert run.record.fallback_reason is None
        assert run.projection is None and run.result is None and run.evaluation is None
        assert builder.requests == [] and evaluator.requests == []
        assert run.document()["identities"]["candidate_sha256"] is None


def test_p1_a_resource_overflow_diverts_only_when_the_fallback_declares_the_reason(tmp_path, monkeypatch):
    """@brief 资源超限：只有声明 resource_unavailable 才切 fallback；两种情形都在构建前分流。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    over = geometry_bytes(geometry_document("variant_a", threads=2048))
    over_source = {"host/geometry_a.json": ("host_tiling", "application/json", over)}
    declared_portfolio = parse_portfolio(
        parent_entries(portfolio_document(plan.digest), extra_sources=over_source), domain_plan=plan
    )
    projected: list[str] = []
    built: list[str] = []
    real_project = portfolio_runner.project_provider
    real_evaluate = portfolio_runner.evaluate_task

    def project_spy(target, provider, *, authority, track_registry=None):
        projected.append(getattr(provider, "variant_key", None) or getattr(provider, "implementation_id", None))
        return real_project(target, provider, authority=authority, track_registry=track_registry)

    def evaluate_spy(*args, **kwargs):
        built.append(kwargs["candidate"].candidate_sha256)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(portfolio_runner, "project_provider", project_spy)
    monkeypatch.setattr(portfolio_runner, "evaluate_task", evaluate_spy)
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(plan, declared_portfolio, locator, contract, "threshold", builder=builder, evaluator=evaluator)
    assert run.record.outcome == "fallback"
    assert run.record.fallback_reason == "resource_unavailable"
    assert run.record.variant_id == declared_portfolio.variants[0].variant_id
    assert "exceeds the target limits" in (run.record.detail or "")
    # The record's preflight is the *final* provider's: the single-source fallback owns its launch,
    # so dispatch resolved that launch geometry for this instance and preflighted it too.
    assert run.record.preflight.status == "ok"
    assert run.record.preflight.resolved is not None
    # Only the declared fallback is ever projected (once to derive its preflight geometry, once for
    # the run) and built; the over-limit variant is neither projected nor built.
    assert set(projected) == {"portfolio.p1.generic"}, projected
    assert len(projected) <= 2
    assert len(built) == 1 and built[0] == run.projection.candidate.candidate_sha256
    assert len(builder.requests) == 1

    # Without the declared reason the variant fails before any projection/build/evaluate_task.
    narrow = portfolio_document(plan.digest)
    narrow["fallback"]["reasons"] = ["outside_candidate_coverage"]
    narrow_portfolio = parse_portfolio(parent_entries(narrow, extra_sources=over_source), domain_plan=plan)
    projected.clear()
    built.clear()
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(plan, narrow_portfolio, locator, contract, "threshold", builder=builder, evaluator=evaluator)
    assert run.record.outcome == "variant_unavailable"
    assert run.record.provider is None
    assert run.record.preflight.status == "resource_unavailable"
    assert "does not declare resource_unavailable" in (run.record.detail or "")
    assert run.projection is None and run.result is None and run.evaluation is None
    assert projected == [] and built == []
    assert builder.requests == [] and evaluator.requests == []


def test_p1_a_fallback_resource_failure_never_chains_into_a_run(tmp_path, monkeypatch):
    """@brief fallback 自身资源不足：variant_unavailable，且不构建任何 provider。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    over = geometry_bytes(geometry_document("generic", threads=2048))
    portfolio = parse_portfolio(
        parent_entries(
            bundle_fallback_document(plan.digest),
            extra_sources={"host/geometry_generic.json": ("host_tiling", "application/json", over)},
        ),
        domain_plan=plan,
    )
    evaluated: list[str] = []
    real_evaluate = portfolio_runner.evaluate_task

    def evaluate_spy(*args, **kwargs):
        evaluated.append(kwargs["candidate"].candidate_sha256)
        return real_evaluate(*args, **kwargs)

    # Count the *composition seam* itself, not only the injected adapters: an adapter that is never
    # handed a request does not prove that ``evaluate_task`` was never entered.
    monkeypatch.setattr(portfolio_runner, "evaluate_task", evaluate_spy)
    # Non-vacuity: the composition seam the runner calls is the spied one (otherwise "no calls"
    # would be indistinguishable from spying on the wrong name).
    assert portfolio_runner.evaluate_task is evaluate_spy
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(plan, portfolio, locator, contract, "hole", builder=builder, evaluator=evaluator)
    assert run.record.outcome == "variant_unavailable"
    assert run.record.provider is None
    assert run.record.fallback_reason == "outside_candidate_coverage"
    assert run.record.preflight.status == "resource_unavailable"
    assert "no chained fallback" in (run.record.detail or "")
    assert run.projection is None and run.result is None
    assert builder.requests == [] and evaluator.requests == []
    assert evaluated == [], "evaluate_task was entered even though the preflight refused the run"


def test_p1_a_single_source_launch_overflow_is_preflighted_and_only_diverts_when_declared(tmp_path, monkeypatch):
    """@brief 源内 launch 几何同样先做纯 CPU 资源预检：声明才切 fallback，且永不构建超限者。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    variant_b = next(variant for variant in portfolio.variants if variant.variant_key == "variant-b-single")
    evaluated: list[str] = []
    real_evaluate = portfolio_runner.evaluate_task

    def evaluate_spy(*args, **kwargs):
        evaluated.append(kwargs["candidate"].candidate_sha256)
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", evaluate_spy)
    assert portfolio_runner.evaluate_task is evaluate_spy

    # (a) The fallback declares resource_unavailable: only the generic fallback is built/executed.
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(
        plan,
        portfolio,
        locator,
        contract,
        "below",
        builder=builder,
        evaluator=evaluator,
        resolver=single_source_resolver_with_limit(kernel="variant_b", threads=2048),
    )
    assert run.record.outcome == "fallback"
    assert run.record.fallback_reason == "resource_unavailable"
    assert run.record.variant_id == variant_b.variant_id
    assert "exceeds the target limits" in (run.record.detail or "")
    # The dispatch-time preflight is the *final* provider's: the generic fallback's own launch.
    assert run.record.preflight.status == "ok" and run.record.preflight.resolved is not None
    generic_sha = run.projection.candidate.candidate_sha256
    assert [request.candidate.candidate_sha256 for request in builder.requests] == [generic_sha]
    assert len(evaluator.requests) == 1
    assert run.result.report["build"]["candidate_sha256"] == generic_sha
    # Exactly one entry into evaluate_task, and it received the final fallback candidate — never the
    # over-limit single source.
    assert evaluated == [generic_sha], evaluated

    # (b) The fallback does not declare the reason: the variant fails before any build.
    evaluated.clear()
    narrow = copy.deepcopy(portfolio_document(plan.digest))
    narrow["fallback"]["reasons"] = ["outside_candidate_coverage"]
    narrow_portfolio = parse_portfolio(parent_entries(narrow), domain_plan=plan)
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(
        plan,
        narrow_portfolio,
        locator,
        contract,
        "below",
        builder=builder,
        evaluator=evaluator,
        resolver=single_source_resolver_with_limit(kernel="variant_b", threads=2048),
    )
    assert run.record.outcome == "variant_unavailable"
    assert run.record.provider is None
    assert run.record.preflight.status == "resource_unavailable"
    assert "does not declare resource_unavailable" in (run.record.detail or "")
    assert run.projection is None and run.result is None
    assert builder.requests == [] and evaluator.requests == []
    assert evaluated == [], "evaluate_task was entered for a variant the preflight refused"

    # (c) The single-source fallback is itself over the limits: no chained fallback, no build.
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(
        plan,
        portfolio,
        locator,
        contract,
        "hole",
        builder=builder,
        evaluator=evaluator,
        resolver=single_source_resolver_with_limit(kernel="generic", threads=2048),
    )
    assert run.record.outcome == "variant_unavailable"
    assert run.record.provider is None
    assert run.record.preflight.status == "resource_unavailable"
    assert "no chained fallback" in (run.record.detail or "")
    assert run.projection is None and run.result is None
    assert builder.requests == [] and evaluator.requests == []
    assert evaluated == [], "evaluate_task was entered for an over-limit single-source fallback"


def test_p1_two_active_stages_with_identical_signatures_are_both_executed(tmp_path):
    """@brief 两个签名完全相同的 active stage 不得被折叠：两条 dispatch 都必须保留并通过。

    这保护候选表达力：是否合并阶段是候选自己的决定，harness 只按 stage 身份逐条核对。
    """
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    run = _run(
        plan,
        portfolio,
        locator,
        contract,
        "below",
        builder=builder,
        evaluator=evaluator,
        resolver=lambda _contract: single_source_resolver_repeating,
    )
    assert run.record.outcome == "resolved"
    assert run.record.provider == "candidate"
    dispatch = run.evaluation["dispatch"]
    assert len(dispatch) == 2, dispatch
    assert dispatch[0] == dispatch[1], dispatch
    assert run.evaluation["expected_dispatch"] == dispatch
    # The stage indices are what distinguish the two identical signatures (not a folded identity).
    assert run.evaluation["dispatch_stages"] == [0, 1]
    assert run.evaluation["expected_dispatch_stages"] == [0, 1]
    # The evaluated geometry keeps both stages too (the identity is over the pair document).
    assert len(run.evaluation["geometry"]["stages"]) == 2
    assert len(builder.requests) == 1 and len(evaluator.requests) == 1

    # An *extra* dispatch entry for a stage the frozen geometry does not have is still refused.
    class _ExtraDispatch(RecordingEvaluator):
        """@brief 多记一条不存在阶段的 evaluator（负例）。"""

        def evaluate(self, request):
            """@brief 在真实形状记录上追加一条 stage 2。

            @param request 评测请求。
            @return schema-4 报告。
            """
            report = super().evaluate(request)
            extra = dict(self.last_execution["dispatch"][0])
            extra["stage"] = 2
            self.last_execution["dispatch"].append(extra)
            return report

    builder = RecordingBuilder()
    with pytest.raises(TaskContractError, match="the runtime dispatched stages"):
        _run(
            plan,
            portfolio,
            locator,
            contract,
            "below",
            builder=builder,
            evaluator=_ExtraDispatch(),
            resolver=lambda _contract: single_source_resolver_repeating,
        )
    # The refused run stopped after the build: the harness never claims it executed.
    assert len(builder.requests) == 1


def test_p1_provider_and_identity_swaps_are_refused(tmp_path):
    """@brief 伪造 provider、fallback 冒充 candidate、parent/variant 身份串用一律拒绝。"""
    contract, instances, plan, portfolio, locator = _setup(tmp_path)
    threshold = _run(plan, portfolio, locator, contract, "threshold")
    hole = _run(plan, portfolio, locator, contract, "hole")
    # A resolved record paired with the fallback projection is a provider swap.
    with pytest.raises(TaskContractError, match="provider 'candidate'"):
        replace(hole, record=threshold.record)
    # A fallback record paired with the variant projection is the same swap in the other direction.
    with pytest.raises(TaskContractError, match="provider 'fallback'"):
        replace(threshold, record=hole.record)
    # A forged evidence key that is neither a variant nor the fallback is refused at construction.
    forged = dict(evidence_all_valid(portfolio))
    forged["c" * 64] = "validated"
    with pytest.raises(TaskContractError, match="not a variant or fallback"):
        run_portfolio_task(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(plan, contract, "threshold"),
            evidence=forged,
            builder=RecordingBuilder(),
            evaluator=RecordingEvaluator(),
            geometry_resolver_factory=lambda _contract: single_source_resolver,
        )
    # The parent portfolio SHA is never a child candidate SHA.
    with pytest.raises(TaskContractError, match="parent portfolio identity"):
        EvaluationBinding(
            provider="candidate",
            variant_id=portfolio.variants[0].variant_id,
            fallback_id=None,
            domain_plan_digest=portfolio.domain_plan_digest,
            portfolio_digest=portfolio.digest,
            parent_portfolio_sha256=threshold.projection.candidate.candidate_sha256,
            candidate_sha256=threshold.projection.candidate.candidate_sha256,
            track_id="msl-bundle",
            geometry_origin_path="host/geometry_a.json",
            geometry_origin_sha256=threshold.projection.binding.geometry_origin_sha256,
            geometry_path="host/execution_geometry.json",
            entries=threshold.projection.binding.entries,
        )
    # A provider from another portfolio cannot be projected.
    foreign = portfolio_document(plan.digest, planner={"id": "another-planner", "version": 3})
    other = parse_portfolio(parent_entries(foreign), domain_plan=plan)
    with pytest.raises(TaskContractError, match="does not belong to this portfolio"):
        project_provider(other, portfolio.variants[0], authority=_authority(plan, contract, instances))


def test_p1_a_stale_artifact_or_mislabelled_evaluation_is_refused(tmp_path, monkeypatch):
    """@brief 失效 artifact：评测必须绑定被选中的投影候选、它的几何和它的 Track。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    real = portfolio_runner.evaluate_task

    def wrong_candidate(*args, **kwargs):
        return replace(real(*args, **kwargs), candidate_sha256=portfolio.parent_sha256)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", wrong_candidate)
    with pytest.raises(TaskContractError, match="different candidate identity"):
        _run(plan, portfolio, locator, contract, "threshold")
    monkeypatch.setattr(portfolio_runner, "evaluate_task", real)

    def wrong_geometry(*args, **kwargs):
        return replace(real(*args, **kwargs), geometry_identity="e" * 64)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", wrong_geometry)
    with pytest.raises(TaskContractError, match="geometry identity is not the geometry"):
        _run(plan, portfolio, locator, contract, "threshold")
    monkeypatch.setattr(portfolio_runner, "evaluate_task", real)

    def wrong_track(*args, **kwargs):
        result = real(*args, **kwargs)
        track = dict(result.track)
        track["track_id"] = "msl-single" if result.track["track_id"] == "msl-bundle" else "msl-bundle"
        return replace(result, track=track)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", wrong_track)
    with pytest.raises(TaskContractError, match="different Track"):
        _run(plan, portfolio, locator, contract, "threshold")


def test_p1_a_wrong_grid_or_a_missing_host_run_in_the_execution_record_is_refused(tmp_path, monkeypatch):
    """@brief 错误 grid / Host Tiling 未执行 / §A2 fallback provider：逐字段拒绝。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    real = portfolio_runner.evaluate_task

    def wrong_grid(*args, **kwargs):
        result = real(*args, **kwargs)
        execution = dict(result.execution)
        dispatch = [dict(item) for item in execution["dispatch"]]
        dispatch[0]["grid"] = [int(dispatch[0]["grid"][0]) + 1, 1, 1]
        execution["dispatch"] = dispatch
        return replace(result, execution=execution)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", wrong_grid)
    with pytest.raises(TaskContractError, match="does not match the frozen geometry"):
        _run(plan, portfolio, locator, contract, "threshold")
    monkeypatch.setattr(portfolio_runner, "evaluate_task", real)

    def no_host_run(*args, **kwargs):
        result = real(*args, **kwargs)
        execution = dict(result.execution)
        execution["host_runs"] = []
        return replace(result, execution=execution)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", no_host_run)
    with pytest.raises(TaskContractError, match="never executed its Host Tiling plan"):
        _run(plan, portfolio, locator, contract, "threshold")
    monkeypatch.setattr(portfolio_runner, "evaluate_task", real)

    def fallback_provider(*args, **kwargs):
        result = real(*args, **kwargs)
        execution = dict(result.execution)
        execution["provider"] = "fallback"
        return replace(result, execution=execution)

    monkeypatch.setattr(portfolio_runner, "evaluate_task", fallback_provider)
    with pytest.raises(TaskContractError, match="geometry source 'fallback'"):
        _run(plan, portfolio, locator, contract, "threshold")


def test_p1_a_missing_or_illegal_host_tiling_and_a_missing_launch_are_refused_before_a_build(tmp_path):
    """@brief 缺/非法 Host Tiling 与缺 taic-launch 在投影期拒绝；不会进入 evaluate_task。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    document = portfolio_document(plan.digest)
    # 1. The declared Host Tiling entry is missing from the parent bundle.
    with pytest.raises(TaskContractError, match="does not exist in the parent bundle"):
        parse_portfolio(parent_entries(document, drop=("host/host_tiling_a.json",)), domain_plan=plan)
    # 2. The variant stops delivering Host Tiling while still declaring a geometry.
    no_host = copy.deepcopy(document)
    no_host["variants"][0]["entries"] = ["device/variant_a.metal"]
    with pytest.raises(TaskContractError, match="exactly one host_tiling entry"):
        _run(plan, parse_portfolio(parent_entries(no_host), domain_plan=plan), locator, contract, "threshold")
    # 3. The Host Tiling entry is not a legal closed plan.
    illegal = parse_portfolio(
        parent_entries(
            document,
            extra_sources={"host/host_tiling_a.json": ("host_tiling", "application/json", b"// not a plan\n")},
        ),
        domain_plan=plan,
    )
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    with pytest.raises(TaskContractError, match="not JSON|host tiling"):
        _run(plan, illegal, locator, contract, "threshold", builder=builder, evaluator=evaluator)
    assert builder.requests == [] and evaluator.requests == []
    # 4. The single-source variant lost its taic-launch block.
    no_launch = b"kernel void variant_b(device const half* x, device half* y) { y[0] = x[0]; }\n"
    broken = parse_portfolio(
        parent_entries(document, extra_sources={"device/variant_b.metal": ("device_kernel", "text/plain", no_launch)}),
        domain_plan=plan,
    )
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    with pytest.raises(TaskContractError, match="taic-launch"):
        _run(plan, broken, locator, contract, "below", builder=builder, evaluator=evaluator)
    assert builder.requests == [] and evaluator.requests == []


def test_p1_the_lifecycle_binds_the_child_candidate_and_its_geometry(tmp_path):
    """@brief 生命周期用子候选 SHA + **请求实例** 的合同身份，不用 parent portfolio 或默认实例身份。"""
    from tcenv.taskcontract.portfolio_runner import locator_for_instance

    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    runs = {size: _run(plan, portfolio, locator, contract, size) for size in ("threshold", "above")}
    for size, run in runs.items():
        lifecycle = run.evaluation["lifecycle"]
        assert lifecycle is not None, size
        assert lifecycle["candidate_sha256"] == run.projection.candidate.candidate_sha256
        assert lifecycle["candidate_sha256"] != portfolio.parent_sha256
        assert lifecycle["track_id"] == "msl-bundle"
        assert lifecycle["geometry"] == run.evaluation["geometry"]
        assert [step["role"] for step in lifecycle["prepare"]] == ["host_tiling", "schedule"]
        assert all(step["inside_timing"] is False for step in lifecycle["prepare"])
        assert all(step["input_dependent"] is False for step in lifecycle["prepare"])
        # The lifecycle and its cache scope must name the instance this request resolved, not the
        # locator's default one: that is what the evaluator's cache keys on.
        requested = locator_for_instance(locator, "fp16", size).load()
        assert lifecycle["instance_key"] == requested.fingerprint(), size
        assert lifecycle["cache_scope"]["instance_key"] == requested.fingerprint(), size
        assert lifecycle["cache_scope"]["candidate_sha256"] == run.projection.candidate.candidate_sha256
        assert lifecycle["cache_scope"]["target_id"] == DEFAULT_TARGET_ID
    # N=64 and N=65 are different instances, so their lifecycle identities must differ even though
    # both run the same bundle variant.
    assert (
        runs["threshold"].evaluation["lifecycle"]["instance_key"]
        != runs["above"].evaluation["lifecycle"]["instance_key"]
    )
    assert runs["threshold"].projection.candidate.candidate_sha256 == runs["above"].projection.candidate.candidate_sha256
    # A single-source provider owns no host work, so it has no lifecycle at all.
    assert _run(plan, portfolio, locator, contract, "below").evaluation["lifecycle"] is None


def test_p1_the_two_variants_and_the_fallback_have_distinct_projected_identities(tmp_path):
    """@brief candidate/geometry/artifact 身份互不相同，并与投影一致（原 geometry 路径+内容 SHA）。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    bundle = _run(plan, portfolio, locator, contract, "threshold")
    single = _run(plan, portfolio, locator, contract, "below")
    fallback = _run(plan, portfolio, locator, contract, "hole")
    shas = {
        bundle.projection.candidate.candidate_sha256,
        single.projection.candidate.candidate_sha256,
        fallback.projection.candidate.candidate_sha256,
    }
    assert len(shas) == 3
    assert (
        len({bundle.result.geometry_identity, single.result.geometry_identity, fallback.result.geometry_identity}) == 3
    )
    assert len({bundle.result.artifact_set_digest, single.result.artifact_set_digest, fallback.result.artifact_set_digest}) == 3
    for run in (bundle, single, fallback):
        assert run.result.artifact_set_digest != run.projection.candidate.candidate_sha256
        assert run.evaluation["geometry_origin_path"] == run.projection.binding.geometry_origin_path
        assert run.evaluation["geometry_origin_sha256"] == run.projection.binding.geometry_origin_sha256
        assert run.evaluation["projected_geometry_sha256"] == run.projection.binding.geometry_origin_sha256
        assert run.projection.candidate.candidate_sha256 != portfolio.parent_sha256
    # The bundle's declared geometry is the parent's original path and content, not a copy.
    parent_entry = portfolio.entry("host/geometry_a.json")
    assert bundle.evaluation["geometry_origin_path"] == "host/geometry_a.json"
    assert bundle.evaluation["geometry_origin_sha256"] == hashlib.sha256(parent_entry.data).hexdigest()
    projected = next(
        entry for entry in bundle.projection.candidate.entries if entry.path == "host/execution_geometry.json"
    )
    assert projected.data == parent_entry.data
    # The single-source variant has no declared geometry but still binds a candidate identity.
    assert single.evaluation["geometry_origin_path"] is None
    assert single.evaluation["projected_geometry_sha256"] is None
    assert single.result.candidate_sha256 == single.projection.candidate.candidate_sha256


def test_p1_the_composition_record_is_not_the_runtime_dispatch_record(tmp_path):
    """@brief PortfolioDispatchRecord 与单 stage runtime 记录不重名、不混用；记录无性能/激活字段。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    run = _run(plan, portfolio, locator, contract, "threshold")
    assert type(run.record).__name__ == "PortfolioDispatchRecord"
    assert type(run.record).__module__ == "tcenv.portfolio.dispatch"
    assert isinstance(run.record, PortfolioDispatchRecord)
    assert DISPATCH_RECORD_SCHEMA_ID == "tcenv.portfolio.dispatch-record.1"
    record_document = run.record.document()
    for forbidden in ("grid", "threads", "median_us", "speedup", "active", "activation_generation"):
        assert forbidden not in record_document
    # The runtime dispatch is a different type with a different vocabulary (grid/threads/provider).
    assert "grid" in run.evaluation["dispatch"][0]
    assert run.evaluation["provider"] == "candidate"
    assert run.evaluation["runtime_provider"] == "candidate"
    assert run.document()["record"]["schema_id"] == DISPATCH_RECORD_SCHEMA_ID


def test_p1_an_msl_single_projection_without_a_launch_adapter_is_refused_before_the_build(tmp_path):
    """@brief 单源投影没有注入 launch 解析器时在构建前拒绝（不猜 grid/threads）。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    builder, evaluator = RecordingBuilder(), RecordingEvaluator()
    with pytest.raises(TaskContractError, match="needs the backend launch resolver"):
        run_portfolio_task(
            locator=locator,
            domain_plan=plan,
            portfolio=portfolio,
            request=_request(plan, contract, "below"),
            evidence=evidence_all_valid(portfolio),
            builder=builder,
            evaluator=evaluator,
            geometry_resolver_factory=None,
        )
    assert builder.requests == [] and evaluator.requests == []


def test_p1_a_build_free_evaluation_report_is_accepted_only_without_a_runtime_record(tmp_path):
    """@brief worker timeout/crash 类 build-free 报告允许无 runtime 记录，但不得伪造成正常报告。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    run = _run(plan, portfolio, locator, contract, "below", evaluator=RecordingEvaluator(status="timeout", no_execution=True))
    assert run.evaluation["report_status"] == "timeout"
    assert run.evaluation["execution_published"] is False
    assert run.evaluation["runtime_provider"] is None
    assert run.result.report["build"] is None
    assert run.document()["identities"]["artifact_set_digest"] == run.result.artifact_set_digest
    # A normal status without a runtime record is refused: real evidence must say what ran.
    with pytest.raises(TaskContractError, match="no execution record"):
        _run(plan, portfolio, locator, contract, "below", evaluator=RecordingEvaluator(no_execution=True))


class _EmptyDispatchEvaluator(RecordingEvaluator):
    """@brief 返回 incorrect 且 dispatch=[] 的评测器，用于反例（不伪造候选证据）。"""

    def __init__(self, *, host_program=None, host_failure=None, **kwargs) -> None:
        # Build a full schema-4 record first (an incorrect report still needs the build it ran),
        # then change only the verdict and drop the dispatch evidence.
        super().__init__(status="ok", **kwargs)
        self._host_program = host_program
        self._host_failure = host_failure

    def evaluate(self, request) -> dict:
        report = super().evaluate(request)
        report["status"] = "incorrect"
        report["stage"] = "correct"
        report["eligible"] = False
        assert self.last_execution is not None
        self.last_execution["dispatch"] = []
        if self._host_program is not None:
            self.last_execution["host_program"] = dict(self._host_program)
        if self._host_failure is not None:
            self.last_execution["host_runs"] = [dict(self._host_failure)]
        return report


def test_p1_an_msl_single_empty_dispatch_without_host_evidence_is_a_parity_mismatch(tmp_path):
    """@brief msl-single 不能靠 ``incorrect`` + 空 dispatch 绕过静态 parity。

    ``pre_dispatch_failure`` may only exempt a candidate that really ships a C Host Program and
    has a matching typed HostProgramFailed document.  An msl-single candidate has neither, so the
    empty dispatch must still fail the frozen-geometry check.
    """
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    evaluator = _EmptyDispatchEvaluator()
    with pytest.raises(TaskContractError, match="does not match the frozen geometry"):
        _run(plan, portfolio, locator, contract, "below", evaluator=evaluator)
    assert evaluator.requests, "the evaluator must have run; this is not a pre-build refusal"


def test_p1_a_linear_bundle_cannot_claim_a_host_failure_to_skip_dispatch(tmp_path):
    """@brief linear JSON bundle 没有 C Host Program，同样不得空 dispatch。"""
    contract, _instances, plan, portfolio, locator = _setup(tmp_path)
    failure = {
        "typed_error": "host_program_invalid",
        "host_status": -1,
        "workspace_writes": [],
        "input_dependent": True,
        "preprocesses_inputs": True,
        "inside_timing": True,
    }
    evaluator = _EmptyDispatchEvaluator(host_failure=failure)
    with pytest.raises(TaskContractError, match="does not match the frozen geometry"):
        _run(plan, portfolio, locator, contract, "threshold", evaluator=evaluator)
    assert evaluator.requests
