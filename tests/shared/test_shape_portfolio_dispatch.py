"""Shape portfolio P0: offline exact dispatch (CPU).

The resolver order is fixed and each step has its own outcome:

1. unparseable request → ``semantic_unresolved``;
2. request outside the finite semantic domain → ``unsupported_contract_capability`` (never a
   fallback: a fallback is a correctness provider inside the domain, not an escape hatch);
3. undecidable guard/evidence → ``dispatch_unknown`` (a ``rem(N,0)`` guard is unknown, never
   false);
4. multiple matches → ``dispatch_overlap`` (no priority, no ranking);
5. exactly one match → the candidate variant, after the runner-owned CPU geometry preflight;
6. zero matches → a declared, applicable, validated fallback or ``dispatch_unmatched``.

Fallback reasons are derived from the execution stage, never self-reported; a resource overflow
diverts before ``evaluate_task``; a fallback's own failure never chains.  The record is a
composition decision (:class:`PortfolioDispatchRecord`), not a runtime dispatch record, an active
manifest or a performance claim.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import TaskContractError
from tcenv.identity import fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.opspec.validator import SchemaFailure, validate
from tcenv.portfolio.candidate import FALLBACK_REASONS, parse_portfolio
from tcenv.portfolio.dispatch import (
    DISPATCH_RECORD_SCHEMA_ID,
    DISPATCH_RECORD_SCHEMA_PATH,
    DispatchRequest,
    ExactDispatchResolver,
    GeometryPreflight,
    PortfolioDispatchRecord,
    preflight_geometry,
)
from tcenv.portfolio.domain import (
    guard_digest,
    instance_key,
    normalize_semantic_domain_plan,
    parse_semantic_domain_plan,
)
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from tcenv.track.delivery import parse_candidate_geometry

from ..support.shape_portfolio import (
    bundle_fallback_document,
    evidence_all_valid,
    geometry_bytes,
    geometry_document,
    load_portfolio_task,
    parent_entries,
    portfolio_document,
    portfolio_instances,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_target(DEFAULT_TARGET_ID)
LIMITS = abi_limits_from_target(PROFILE)
SCHEMA = json.loads(DISPATCH_RECORD_SCHEMA_PATH.read_text(encoding="utf-8"))


def _setup(tmp_path, *, document=None, entry_kwargs=None):
    """@brief 固定题 + 域计划 + 已解析 portfolio。

    @param tmp_path 临时目录。
    @param document envelope 覆盖。
    @param entry_kwargs ``parent_entries`` 参数。
    @return ``(contract, instances, plan, portfolio)``。
    """
    contract = load_portfolio_task(tmp_path)
    instances = portfolio_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE)
    body = document if document is not None else portfolio_document(plan.digest)
    portfolio = parse_portfolio(parent_entries(body, **(entry_kwargs or {})), domain_plan=plan)
    return contract, instances, plan, portfolio


def _request(contract, plan, size, usage="golden"):
    """@brief 由真实实例构造请求 + 它的 ABI。

    @param contract 任务合同。
    @param plan 域计划。
    @param size size 档位名。
    @param usage 用途。
    @return ``(request, instance, abi)``。
    """
    instance = instantiate_v2(contract.spec, "fp16", size)
    abi = derive_tensor_memory_contract(instance, target=PROFILE)
    point = plan.find("fp16", size, usage)
    assert point is not None
    return (
        DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage),
        instance,
        abi,
    )


def _hook(instance, abi):
    """@brief 该请求的 CPU 几何预检钩子。

    @param instance 语义实例。
    @param abi ABI 合同。
    @return 预检函数。
    """
    return lambda geometry: preflight_geometry(geometry, instance=instance, contract=abi, limits=LIMITS)


def _resolver(contract, plan, instances, portfolio, *, evidence=None, preflight=None):
    """@brief 构造 resolver（默认全部 provider 已验证）。

    @param plan 域计划。
    @param instances 权威语义实例（每个 (dtype,size) 点一个）。
    @param portfolio portfolio。
    @param evidence 证据覆盖。
    @param preflight 预检钩子。
    @return resolver。
    """
    return ExactDispatchResolver(
        domain_plan=plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio) if evidence is None else evidence,
        spec=contract.spec,
        target=PROFILE,
        instances=instances,
        preflight=preflight,
    )


def _variant(document, key):
    """@brief 按诊断名取 variant 文档。

    @param document envelope。
    @param key 诊断名。
    @return variant 文档。
    """
    for item in document["variants"]:
        if item["variant_key"] == key:
            return item
    raise AssertionError(f"no variant {key!r}")


def _set_guard(document, key, bucket_index, expr):
    """@brief 一致地替换一个 variant 的 guard 与它所属 bucket 的摘要。

    @param document envelope。
    @param key variant 诊断名。
    @param bucket_index bucket 下标。
    @param expr 新 guard 表达式。
    """
    _variant(document, key)["guard"] = {"expr": expr, "digest": guard_digest(expr, where=key)}
    document["coverage_plan"]["buckets"][bucket_index]["guard_digest"] = guard_digest(expr, where="bucket")


def _expect_schema(record: PortfolioDispatchRecord):
    """@brief 记录必须通过登记 schema 的闭集校验。

    @param record 决策记录。
    """
    document = record.document()
    try:
        validate(document, SCHEMA)
    except SchemaFailure as exc:  # pragma: no cover - assertion helper
        raise AssertionError(f"dispatch record violates its registered schema: {exc}") from exc


def test_p0_unparseable_requests_are_semantic_unresolved(tmp_path):
    """@brief 闭集解析：每类字段错误独立拒绝，合法 document 可重放。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    resolver = _resolver(contract, plan, instances, portfolio)
    good = _request(contract, plan, "threshold")[0].document()
    # A legitimate replay document is closed *and* complete: parse(document()) is the request.
    assert DispatchRequest.parse(good) == _request(contract, plan, "threshold")[0]
    assert DispatchRequest.parse(good).document() == good
    cases = (
        ("not an object", "not-an-object", "must be an object"),
        ("missing symbols", {key: value for key, value in good.items() if key != "symbols"}, "expected exactly"),
        ("missing instance_key", {key: value for key, value in good.items() if key != "instance_key"}, "expected exactly"),
        ("unknown field", {**good, "evidence": "validated"}, "expected exactly"),
        ("non-string op_name", {**good, "op_name": 123}, "DispatchRequest.op_name must be a non-empty string"),
        ("non-string dtype", {**good, "dtype": ["fp16"]}, "DispatchRequest.dtype must be a non-empty string"),
        ("boolean symbol", {**good, "symbols": {"N": True}}, "DispatchRequest.symbols.N"),
        ("float attr", {**good, "attrs": {"EPS": 1.0}}, "DispatchRequest.attrs.EPS"),
        ("usage typo", {**good, "usage": "benchmark_but_typo"}, "DispatchRequest.usage"),
        ("forged instance_key", {**good, "instance_key": "fp16/large/golden"}, "is not the key derived"),
        ("symbol/attr collision", {**good, "attrs": {"N": 1}}, "both a symbol and an attr"),
        ("malformed digest", {**good, "semantic_instance_digest": "nope"}, "64-character lowercase hex"),
    )
    for label, document, fragment in cases:
        record = resolver.resolve(document)
        assert record.outcome == "semantic_unresolved", label
        assert record.provider is None and record.variant_id is None and record.fallback_id is None, label
        assert record.request is None, label
        assert record.guards == (), label
        assert fragment in (record.detail or ""), (label, record.detail)
        _expect_schema(record)
    # A request the registered record schema cannot carry is refused at parse time
    # (semantic_unresolved), never recorded as an illegal $.request.
    for field, value in (
        ("op_name", "o" * 129),
        ("dtype", "d" * 33),
        ("size", "s" * 65),
    ):
        too_long = {**good, field: value}
        too_long["instance_key"] = instance_key(too_long["dtype"], too_long["size"], too_long["usage"])
        record = resolver.resolve(too_long)
        assert record.outcome == "semantic_unresolved", (field, record.document())
        assert record.request is None, field
        assert "longer than" in (record.detail or ""), (field, record.detail)
        _expect_schema(record)
    # A request-controlled symbol name can make a refusal detail long; the record clamps it to
    # the schema's detail cap instead of failing $ .detail.
    huge = {**good, "symbols": {"K" * 5000: 1}}
    record = resolver.resolve(huge)
    assert record.outcome == "semantic_unresolved"
    assert record.request is not None and record.request["symbols"] == {"K" * 5000: 1}
    assert len(record.detail or "") <= 4000
    _expect_schema(record)
    # The parse-level refusals are independently reproducible (no mask from another field error).
    for document, fragment in (
        ({**good, "op_name": 123}, "non-empty string"),
        ({**good, "symbols": {"N": True}}, "must be an integer"),
        ({**good, "usage": "benchmark_but_typo"}, "usage must be one of"),
        ({**good, "instance_key": "fp16/large/golden"}, "derived"),
        ({**good, "attrs": {"N": 1}}, "both a symbol and an attr"),
    ):
        with pytest.raises(TaskContractError, match=fragment):
            DispatchRequest.parse(document)


def test_p0_a_request_outside_the_finite_domain_never_falls_back(tmp_path):
    """@brief 域外 unsupported；与权威实例矛盾 semantic_unresolved；两者都不 fallback。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    resolver = _resolver(contract, plan, instances, portfolio)
    good, _, _ = _request(contract, plan, "threshold")
    document = good.document()
    # Not a named point of the finite domain.
    outside = DispatchRequest(
        op_name=good.op_name,
        dtype="fp16",
        size="not_a_named_size",
        usage="golden",
        symbols={"N": 96},
        attrs={},
        semantic_instance_digest=good.semantic_instance_digest,
        target_abi_digest=good.target_abi_digest,
    )
    record = resolver.resolve(outside)
    assert record.outcome == "unsupported_contract_capability"
    assert record.provider is None and record.fallback_id is None
    assert "not a named point" in (record.detail or "")
    _expect_schema(record)
    # A named point whose declared identity contradicts the authoritative instance cannot be
    # resolved to that instance: it is semantic_unresolved and never reaches a provider/fallback.
    contradictions = (
        ("op_name", {**document, "op_name": "completely_other_op"}),
        ("symbols", {**document, "symbols": {"N": 1024}}),
        ("attrs", {**document, "attrs": {"EPS": 1}}),
        ("semantic_instance_digest", {**document, "semantic_instance_digest": "a" * 64}),
        ("target_abi_digest", {**document, "target_abi_digest": "b" * 64}),
    )
    for field, request in contradictions:
        record = resolver.resolve(request)
        assert record.outcome == "semantic_unresolved", (field, record.document())
        assert record.provider is None and record.variant_id is None and record.fallback_id is None, field
        assert "contradicts the authoritative semantic instance" in (record.detail or ""), field
        assert field in (record.detail or ""), (field, record.detail)
        _expect_schema(record)
    # Even with validated fallback evidence, neither an out-of-domain point nor a
    # contradictory request can trigger a fallback.
    assert portfolio.fallback is not None
    assert evidence_all_valid(portfolio)[portfolio.fallback_id] == "validated"
    assert record.fallback_reason is None


def test_p0_a_request_document_round_trips_and_binds_to_the_authoritative_instance(tmp_path):
    """@brief 合法 document 的 parse/resolve 往返一致；resolver 必须持有权威实例。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    resolver = _resolver(contract, plan, instances, portfolio)
    for size in ("below", "threshold", "tail", "large"):
        request, instance, abi = _request(contract, plan, size)
        document = request.document()
        reparsed = DispatchRequest.parse(document)
        assert reparsed == request, size
        assert reparsed.document() == document, size
        assert reparsed.instance_key == f"fp16/{size}/golden", size
        direct = resolver.resolve(request, preflight=_hook(instance, abi))
        replay = resolver.resolve(document, preflight=_hook(instance, abi))
        assert direct.outcome == replay.outcome, size
        assert direct.document() == replay.document(), size
        assert direct.identity == replay.identity, size
        assert direct.request == document, size
    # Dispatch is bound to the authoritative instances, not to a request's self-declared identity.
    with pytest.raises(TaskContractError, match="authoritative SemanticInstance"):
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=portfolio,
            evidence=evidence_all_valid(portfolio),
            spec=contract.spec,
            target=PROFILE,
            instances=(),
        )
    duplicate = next(instance for instance in instances if instance.size == "large")
    with pytest.raises(TaskContractError, match="repeats the authoritative instance|exactly the domain plan"):
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=portfolio,
            evidence=evidence_all_valid(portfolio),
            spec=contract.spec,
            target=PROFILE,
            instances=tuple(item for item in instances if item.size != "threshold") + (duplicate,),
        )
    wrong_instance = tuple(item for item in instances if item.size != "large")
    with pytest.raises(TaskContractError, match="exactly the domain plan"):
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=portfolio,
            evidence=evidence_all_valid(portfolio),
            spec=contract.spec,
            target=PROFILE,
            instances=wrong_instance,
        )


def test_p0_an_undecidable_guard_or_evidence_is_dispatch_unknown(tmp_path):
    """@brief 未知字段/除零 guard/未验证证据都是 dispatch_unknown，不是 false 或成功。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    resolver = _resolver(contract, plan, instances, portfolio)
    # A guard that references a field the request does not declare.
    unknown_field = portfolio_document(plan.digest)
    _set_guard(unknown_field, "variant-c-tail", 2, ["eq", "K", 3])
    unknown_portfolio = parse_portfolio(parent_entries(unknown_field), domain_plan=plan)
    request, _, _ = _request(contract, plan, "tail")
    record = _resolver(contract, plan, instances, unknown_portfolio).resolve(request)
    assert record.outcome == "dispatch_unknown"
    assert record.provider is None
    assert any(guard.result == "unknown" for guard in record.guards)
    assert "not decidable" in (record.detail or "")
    _expect_schema(record)
    # A guard that divides by zero is an evaluation failure, not a false branch.
    divide_by_zero = portfolio_document(plan.digest)
    _set_guard(divide_by_zero, "variant-c-tail", 2, ["eq", ["trunc_div", "N", 0], 1])
    zero_portfolio = parse_portfolio(parent_entries(divide_by_zero), domain_plan=plan)
    record = _resolver(contract, plan, instances, zero_portfolio).resolve(request)
    assert record.outcome == "dispatch_unknown"
    assert any("divisor is zero" in (guard.detail or "") for guard in record.guards)
    # The matched variant's evidence is not validated (explicitly unknown or absent).
    threshold, _, _ = _request(contract, plan, "threshold")
    variant_a = portfolio.variants[0].variant_id
    for evidence in (
        {**evidence_all_valid(portfolio), variant_a: "unknown"},
        {key: value for key, value in evidence_all_valid(portfolio).items() if key != variant_a},
    ):
        record = _resolver(contract, plan, instances, portfolio, evidence=evidence).resolve(threshold)
        assert record.outcome == "dispatch_unknown"
        assert record.provider is None
        assert "no validated evidence" in (record.detail or "")
        _expect_schema(record)


def test_p0_threshold_odd_tail_and_large_points_dispatch_exactly_one_variant(tmp_path):
    """@brief threshold−1/0/+1、奇数 tail、large 的 exactly-one 选择与 guard 记录。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    resolver = _resolver(contract, plan, instances, portfolio)
    cases = (
        ("below", "golden", "fallback", None, "outside_candidate_coverage"),
        ("threshold", "golden", "resolved", "variant-a-bundle", None),
        ("above", "golden", "resolved", "variant-a-bundle", None),
        ("tail", "golden", "resolved", "variant-c-tail", None),
        ("large", "golden", "resolved", "variant-b-single", None),
        ("threshold", "benchmark", "resolved", "variant-a-bundle", None),
    )
    keys = {variant.variant_key: variant for variant in portfolio.variants}
    for size, usage, outcome, provider_key, reason in cases:
        request, instance, abi = _request(contract, plan, size, usage)
        record = resolver.resolve(request, preflight=_hook(instance, abi))
        assert record.outcome == outcome, (size, usage, record.document())
        assert record.fallback_reason == reason
        assert len(record.guards) == len(portfolio.variants)
        if provider_key is None:
            assert record.provider == "fallback"
            assert record.variant_id is None
            assert record.fallback_id == portfolio.fallback_id
        else:
            chosen = keys[provider_key]
            assert record.provider == "candidate"
            assert record.variant_id == chosen.variant_id
            assert record.fallback_id is None
            guards = {guard.variant_id: guard for guard in record.guards}
            assert guards[chosen.variant_id].result == "true"
            assert guards[chosen.variant_id].guard_digest == chosen.guard.digest
        _expect_schema(record)
    # The threshold window is exactly [64, 128): 63 is out, 64 and 65 are in.
    for size, expected in (("below", False), ("threshold", True), ("above", True)):
        request, _, _ = _request(contract, plan, size)
        assert request.names["N"] in (63, 64, 65)
        hit = keys["variant-a-bundle"].guard.evaluate(request.names)
        assert hit is expected
        assert keys["variant-c-tail"].guard.evaluate(request.names) is (size == "tail")


def test_p0_multiple_matches_are_dispatch_overlap_without_priority(tmp_path):
    """@brief 多匹配是 dispatch_overlap：不排序、不按声明顺序猜。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    overlapping = portfolio_document(plan.digest)
    overlapping["coverage_plan"]["buckets"].append(
        {"bucket_key": "tail_plain", "guard_digest": guard_digest(["lt", "N", 32], where="tail_plain")}
    )
    overlapping["variants"].append(
        {
            "variant_key": "variant-d-overlap",
            "bucket_key": "tail_plain",
            "guard": {"expr": ["lt", "N", 32], "digest": guard_digest(["lt", "N", 32], where="variant-d")},
            "entries": ["device/variant_b.metal"],
            "geometry_entry": None,
        }
    )
    portfolio = parse_portfolio(parent_entries(overlapping), domain_plan=plan)
    request, instance, abi = _request(contract, plan, "tail")
    record = _resolver(contract, plan, instances, portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "dispatch_overlap"
    assert record.provider is None and record.variant_id is None
    matched = [guard.variant_id for guard in record.guards if guard.result == "true"]
    assert len(matched) == 2
    assert "exact dispatch refuses to rank" in (record.detail or "")
    _expect_schema(record)
    # Reordering the variants does not change the outcome: there is no priority order.
    reversed_document = copy.deepcopy(overlapping)
    reversed_document["variants"].reverse()
    reversed_portfolio = parse_portfolio(parent_entries(reversed_document), domain_plan=plan)
    reversed_record = _resolver(contract, plan, instances, reversed_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert reversed_record.outcome == "dispatch_overlap"
    assert {guard.variant_id for guard in reversed_record.guards} == {
        guard.variant_id for guard in record.guards
    }
    # An unvalidated match is unknown before an overlap: evidence uncertainty is checked first.
    evidence = evidence_all_valid(portfolio)
    overlap_variant = next(
        variant for variant in reversed_portfolio.variants if variant.variant_key == "variant-d-overlap"
    )
    evidence[overlap_variant.variant_id] = "unknown"
    unknown = _resolver(contract, plan, instances, reversed_portfolio, evidence=evidence).resolve(request, preflight=_hook(instance, abi))
    assert unknown.outcome == "dispatch_unknown"
    assert unknown.provider is None


def test_p0_a_coverage_hole_uses_the_declared_fallback_or_is_unmatched(tmp_path):
    """@brief 零匹配：原因固定为 outside_candidate_coverage，只有声明且验证过的 fallback 才接管。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    request, instance, abi = _request(contract, plan, "below")
    record = _resolver(contract, plan, instances, portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "fallback"
    assert record.provider == "fallback"
    assert record.fallback_reason == "outside_candidate_coverage"
    assert record.variant_id is None
    assert record.fallback_id == portfolio.fallback_id
    # A fallback result is not the candidate's correctness/speedup: no measurement fields exist.
    document = record.document()
    for measurement in ("speedup", "median_us", "correctness", "published", "active"):
        assert measurement not in document
    _expect_schema(record)
    # Fallback evidence unknown → dispatch_unknown, not "fallback happened".
    evidence = evidence_all_valid(portfolio)
    evidence[portfolio.fallback_id] = "unknown"
    record = _resolver(contract, plan, instances, portfolio, evidence=evidence).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "dispatch_unknown"
    assert record.provider is None
    assert "fallback's independent evidence is not validated" in (record.detail or "")
    _expect_schema(record)
    # The fallback's declared reasons must include the derived one.
    narrow = portfolio_document(plan.digest)
    narrow["fallback"]["reasons"] = ["resource_unavailable"]
    narrow_portfolio = parse_portfolio(parent_entries(narrow), domain_plan=plan)
    record = _resolver(contract, plan, instances, narrow_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "dispatch_unmatched"
    assert record.provider is None and record.fallback_reason is None
    assert "no domain fallback declares 'outside_candidate_coverage'" in (record.detail or "")
    # No fallback at all is unmatched.
    absent = portfolio_document(plan.digest)
    del absent["fallback"]
    absent_portfolio = parse_portfolio(parent_entries(absent), domain_plan=plan)
    assert absent_portfolio.fallback is None and absent_portfolio.fallback_id is None
    record = _resolver(contract, plan, instances, absent_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "dispatch_unmatched"
    assert record.provider is None and record.fallback_reason is None
    # A refusing fallback is an explicit refusal: unmatched with the derived reason, never a run.
    refuse = portfolio_document(plan.digest)
    refuse["fallback"] = {
        "kind": "refuse",
        "implementation_id": None,
        "implementation_version": None,
        "entries": [],
        "geometry_entry": None,
        "reasons": ["outside_candidate_coverage", "resource_unavailable"],
    }
    refuse_portfolio = parse_portfolio(parent_entries(refuse), domain_plan=plan)
    record = _resolver(contract, plan, instances, refuse_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "dispatch_unmatched"
    assert record.provider is None
    assert record.fallback_reason == "outside_candidate_coverage"
    assert "explicitly refuses" in (record.detail or "")
    _expect_schema(record)


def test_p0_resource_preflight_diverts_before_the_variant_runs(tmp_path):
    """@brief 资源超限由 CPU 预检发现：声明了 resource_unavailable 的 fallback 才接管。"""
    contract, instances, plan, _ = _setup(tmp_path)
    over_limit = geometry_bytes(geometry_document("variant_a", threads=2048)).decode("utf-8")
    over_entries = {"host/geometry_a.json": ("host_tiling", "application/json", over_limit.encode())}
    document = portfolio_document(plan.digest)
    portfolio = parse_portfolio(parent_entries(document, extra_sources=over_entries), domain_plan=plan)
    request, instance, abi = _request(contract, plan, "threshold")
    record = _resolver(contract, plan, instances, portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "fallback"
    assert record.provider == "fallback"
    assert record.fallback_reason == "resource_unavailable"
    assert record.variant_id == portfolio.variants[0].variant_id
    assert record.fallback_id == portfolio.fallback_id
    assert "exceeds the target limits" in (record.detail or "")
    assert record.preflight.status in ("ok", "not_required")
    _expect_schema(record)
    # Without a declared resource_unavailable reason the matched variant simply fails: it never
    # enters evaluate_task and never borrows a fallback it did not register.
    narrow = portfolio_document(plan.digest)
    narrow["fallback"]["reasons"] = ["outside_candidate_coverage"]
    narrow_portfolio = parse_portfolio(parent_entries(narrow, extra_sources=over_entries), domain_plan=plan)
    record = _resolver(contract, plan, instances, narrow_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "variant_unavailable"
    assert record.provider is None
    assert record.variant_id == narrow_portfolio.variants[0].variant_id
    assert record.fallback_id is None
    assert record.preflight.status == "resource_unavailable"
    assert "does not declare resource_unavailable" in (record.detail or "")
    _expect_schema(record)
    # An invalid geometry is a candidate defect, not a resource switch.
    invalid = geometry_bytes(
        {"schema_version": 1, "source": "candidate", "stages": [
            {"kernel": "variant_a", "grid": ["NO_SUCH_NAME", 1, 1], "threads": [256, 1, 1]}
        ]}
    ).decode("utf-8")
    invalid_entries = {"host/geometry_a.json": ("host_tiling", "application/json", invalid.encode())}
    invalid_portfolio = parse_portfolio(parent_entries(document, extra_sources=invalid_entries), domain_plan=plan)
    record = _resolver(contract, plan, instances, invalid_portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "variant_unavailable"
    assert record.preflight.status == "invalid"
    assert "invalid geometry" in (record.detail or "")
    # Only the selected provider is preflighted, and only when it declares a geometry.
    calls: list[str] = []
    base = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    counting = lambda geometry: calls.append("preflight") or preflight_geometry(
        geometry, instance=instance, contract=abi, limits=LIMITS
    )
    record = _resolver(contract, plan, instances, base).resolve(_request(contract, plan, "threshold")[0], preflight=counting)
    assert record.outcome == "resolved" and len(calls) == 1
    calls.clear()
    record = _resolver(contract, plan, instances, base).resolve(_request(contract, plan, "large")[0], preflight=counting)
    assert record.outcome == "resolved" and calls == []
    assert record.preflight.status == "not_required"
    _expect_schema(record)


def test_p0_a_fallback_resource_failure_never_chains(tmp_path):
    """@brief fallback 自身资源不足直接失败：不得再次回退。"""
    contract, instances, plan, _ = _setup(tmp_path)
    document = bundle_fallback_document(plan.digest)
    over_limit = geometry_bytes(geometry_document("generic", threads=2048))
    entries = {"host/geometry_generic.json": ("host_tiling", "application/json", over_limit)}
    portfolio = parse_portfolio(parent_entries(document, extra_sources=entries), domain_plan=plan)
    assert portfolio.fallback is not None
    # Zero match → the bundle fallback is selected by reason, then its own preflight fails.
    request, instance, abi = _request(contract, plan, "below")
    record = _resolver(contract, plan, instances, portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "variant_unavailable"
    assert record.provider is None
    assert record.fallback_id == portfolio.fallback_id
    assert record.fallback_reason == "outside_candidate_coverage"
    assert record.preflight.status == "resource_unavailable"
    assert "no chained fallback" in (record.detail or "")
    _expect_schema(record)
    # The resource reason from a variant also ends at the same unavailable fallback.
    request, instance, abi = _request(contract, plan, "threshold")
    over_a = geometry_bytes(geometry_document("variant_a", threads=2048))
    entries = {
        "host/geometry_a.json": ("host_tiling", "application/json", over_a),
        "host/geometry_generic.json": ("host_tiling", "application/json", over_limit),
    }
    portfolio = parse_portfolio(parent_entries(document, extra_sources=entries), domain_plan=plan)
    record = _resolver(contract, plan, instances, portfolio).resolve(request, preflight=_hook(instance, abi))
    assert record.outcome == "variant_unavailable"
    assert record.provider is None
    assert record.variant_id == portfolio.variants[0].variant_id
    assert record.fallback_id == portfolio.fallback_id
    assert record.fallback_reason == "resource_unavailable"
    assert record.preflight.status == "resource_unavailable"
    _expect_schema(record)


def test_p0_a_declared_geometry_requires_the_cpu_preflight(tmp_path):
    """@brief 有显式几何的 provider 必须先被 resolve_geometry 预检；预检分类固定。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    request, instance, abi = _request(contract, plan, "threshold")
    with pytest.raises(TaskContractError, match="must be preflighted"):
        _resolver(contract, plan, instances, portfolio).resolve(request)
    # A single-source provider owns its launch in the source; dispatch cannot preflight it here.
    record = _resolver(contract, plan, instances, portfolio).resolve(_request(contract, plan, "large")[0], preflight=None)
    assert record.outcome == "resolved"
    assert record.preflight.status == "not_required"
    ok = preflight_geometry(
        portfolio.variants[0].geometry, instance=instance, contract=abi, limits=LIMITS
    )
    assert ok.status == "ok" and ok.resolved is not None
    assert ok.resolved.active_stages[0].grid == (1, 1, 1)
    threads = preflight_geometry(
        parse_candidate_geometry(geometry_document("variant_a", threads=2048)),
        instance=instance,
        contract=abi,
        limits=LIMITS,
    )
    assert threads.status == "resource_unavailable" and threads.resolved is None
    memory = preflight_geometry(
        parse_candidate_geometry(geometry_document("variant_a", threadgroup_bytes=1 << 20)),
        instance=instance,
        contract=abi,
        limits=LIMITS,
    )
    assert memory.status == "resource_unavailable"
    invalid = preflight_geometry(
        parse_candidate_geometry(
            {"schema_version": 1, "source": "candidate", "stages": [
                {"kernel": "variant_a", "grid": ["NO_SUCH_NAME", 1, 1], "threads": [256, 1, 1]}
            ]}
        ),
        instance=instance,
        contract=abi,
        limits=LIMITS,
    )
    assert invalid.status == "invalid"
    assert GeometryPreflight("not_required").document() == {"status": "not_required", "detail": None}
    with pytest.raises(TaskContractError, match="unknown preflight status"):
        GeometryPreflight("skipped")


def test_p0_a_domain_plan_cannot_relabel_another_opspec_or_derived_abi(tmp_path):
    """@brief 计划必须由权威 OpSpec/target/instances 重算：opspec/ABI/代际漂移都拒绝。"""
    import dataclasses

    contract, instances, plan, portfolio = _setup(tmp_path)

    def forged(mutator):
        document = plan.document()
        mutator(document)
        forged_plan = parse_semantic_domain_plan(document)
        forged_portfolio = parse_portfolio(
            parent_entries(portfolio_document(forged_plan.digest)), domain_plan=forged_plan
        )
        return forged_plan, forged_portfolio

    def rewrite_opspec(document):
        document["opspec_digest"] = "f" * 64

    def rewrite_generation(document):
        document["target_generation_digest"] = "f" * 64

    def rewrite_all_abi(document):
        for point in document["instances"]:
            point["target_abi_digest"] = "a" * 64

    def rewrite_one_abi(document):
        document["instances"][0]["target_abi_digest"] = "a" * 64

    def rewrite_semantic(document):
        for point in document["instances"]:
            point["semantic_instance_digest"] = "a" * 64

    authority = "authoritative OpSpec/target/instances"
    for mutator in (rewrite_opspec, rewrite_generation, rewrite_all_abi, rewrite_one_abi):
        forged_plan, forged_portfolio = forged(mutator)
        with pytest.raises(TaskContractError, match=authority):
            ExactDispatchResolver(
                domain_plan=forged_plan,
                portfolio=forged_portfolio,
                evidence=evidence_all_valid(forged_portfolio),
                spec=contract.spec,
                target=PROFILE,
                instances=instances,
            )
        with pytest.raises(TaskContractError, match=authority):
            forged_portfolio.registration_report(
                forged_plan, instances, spec=contract.spec, target=PROFILE
            )
    # A per-point semantic digest that no authoritative instance hashes to is refused too.
    forged_plan, forged_portfolio = forged(rewrite_semantic)
    with pytest.raises(TaskContractError, match="does not hash to the domain point"):
        ExactDispatchResolver(
            domain_plan=forged_plan,
            portfolio=forged_portfolio,
            evidence=evidence_all_valid(forged_portfolio),
            spec=contract.spec,
            target=PROFILE,
            instances=instances,
        )
    with pytest.raises(TaskContractError, match="does not match the domain plan's semantics"):
        forged_portfolio.registration_report(forged_plan, instances, spec=contract.spec, target=PROFILE)
    # A hand-built instance cannot stand in for the OpSpec's own derivation, even when the plan
    # was normalized from it and is therefore self-consistent.
    forged_instance = dataclasses.replace(
        instances[-1], symbols={**instances[-1].symbols, "N": 4096}
    )
    forged_instances = tuple(instances[:-1]) + (forged_instance,)
    self_consistent = normalize_semantic_domain_plan(contract.spec, forged_instances, target=PROFILE)
    self_portfolio = parse_portfolio(
        parent_entries(portfolio_document(self_consistent.digest)), domain_plan=self_consistent
    )
    with pytest.raises(TaskContractError, match="not the one the authoritative OpSpec derives"):
        ExactDispatchResolver(
            domain_plan=self_consistent,
            portfolio=self_portfolio,
            evidence=evidence_all_valid(self_portfolio),
            spec=contract.spec,
            target=PROFILE,
            instances=forged_instances,
        )
    with pytest.raises(TaskContractError, match="not the one the authoritative OpSpec derives"):
        self_portfolio.registration_report(
            self_consistent, forged_instances, spec=contract.spec, target=PROFILE
        )


def test_p0_finite_registration_reports_points_overlaps_and_empty_variants(tmp_path):
    """@brief 注册期只做有限点集查重：报告 overlap/空 variant/不可判定 guard，不是静态证明。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    report = portfolio.registration_report(plan, instances, spec=contract.spec, target=PROFILE)
    assert report.ok is True
    assert len(report.points) == len(plan.instances) == 10
    by_key = {point.instance_key: point.matched for point in report.points}
    keys = {variant.variant_key: variant.variant_id for variant in portfolio.variants}
    assert by_key["fp16/threshold/golden"] == (keys["variant-a-bundle"],)
    assert by_key["fp16/above/benchmark"] == (keys["variant-a-bundle"],)
    assert by_key["fp16/tail/golden"] == (keys["variant-c-tail"],)
    assert by_key["fp16/large/golden"] == (keys["variant-b-single"],)
    # A coverage hole is a hole, not an overlap: the fallback owns it at runtime.
    assert by_key["fp16/below/golden"] == ()
    assert all(len(point.matched) <= 1 for point in report.points)
    assert report.overlaps == () and report.empty_variants == () and report.unknown_guards == ()
    document = report.document()
    assert document["coverage_kind"] == "exhaustive_finite_point_check"
    assert "ok" in document and document["ok"] is True
    for forbidden in ("active", "published", "speedup", "proven"):
        assert forbidden not in document
    # A variant whose guard is false on every finite point is an empty coverage claim.
    empty = portfolio_document(plan.digest)
    empty["coverage_plan"]["buckets"].append(
        {"bucket_key": "never", "guard_digest": guard_digest(["gt", "N", 4096], where="never")}
    )
    empty["variants"].append(
        {
            "variant_key": "variant-e-never",
            "bucket_key": "never",
            "guard": {"expr": ["gt", "N", 4096], "digest": guard_digest(["gt", "N", 4096], where="variant-e")},
            "entries": ["device/variant_b.metal"],
            "geometry_entry": None,
        }
    )
    empty_portfolio = parse_portfolio(parent_entries(empty), domain_plan=plan)
    report = empty_portfolio.registration_report(plan, instances, spec=contract.spec, target=PROFILE)
    assert report.ok is False
    assert report.empty_variants == (empty_portfolio.variants[-1].variant_id,)
    # An undecidable guard is reported, never counted as a match.
    unknown = portfolio_document(plan.digest)
    _set_guard(unknown, "variant-c-tail", 2, ["eq", "K", 3])
    unknown_portfolio = parse_portfolio(parent_entries(unknown), domain_plan=plan)
    report = unknown_portfolio.registration_report(plan, instances, spec=contract.spec, target=PROFILE)
    assert report.ok is False
    assert report.unknown_guards
    # An unknown guard is never counted as a match for that point.
    for point in report.points:
        assert not (set(point.unknown) & set(point.matched))
    # An overlapping point is reported by its canonical instance key.
    overlap = portfolio_document(plan.digest)
    overlap["coverage_plan"]["buckets"].append(
        {"bucket_key": "tail_plain", "guard_digest": guard_digest(["lt", "N", 32], where="tail_plain")}
    )
    overlap["variants"].append(
        {
            "variant_key": "variant-d-overlap",
            "bucket_key": "tail_plain",
            "guard": {"expr": ["lt", "N", 32], "digest": guard_digest(["lt", "N", 32], where="variant-d")},
            "entries": ["device/variant_b.metal"],
            "geometry_entry": None,
        }
    )
    overlap_portfolio = parse_portfolio(parent_entries(overlap), domain_plan=plan)
    report = overlap_portfolio.registration_report(plan, instances, spec=contract.spec, target=PROFILE)
    assert report.ok is False
    assert report.overlaps == ("fp16/tail/benchmark", "fp16/tail/golden")
    # The report cannot be produced from a different domain or a mismatched instance set.
    with pytest.raises(TaskContractError, match="different semantic domain"):
        portfolio.registration_report(
            normalize_semantic_domain_plan(contract.spec, instances[:1], target=PROFILE),
            instances,
            spec=contract.spec,
            target=PROFILE,
        )
    with pytest.raises(TaskContractError, match="missing the named instance"):
        portfolio.registration_report(plan, instances[:1], spec=contract.spec, target=PROFILE)


def test_p0_the_record_is_a_composition_decision_not_runtime_or_perf_evidence(tmp_path):
    """@brief PortfolioDispatchRecord：命名独立、schema 闭集、无运行时/性能/激活字段。"""
    contract, instances, plan, portfolio = _setup(tmp_path)
    assert PortfolioDispatchRecord.__name__ == "PortfolioDispatchRecord"
    assert DISPATCH_RECORD_SCHEMA_ID == "tcenv.portfolio.dispatch-record.1"
    import tcenv.portfolio.dispatch as dispatch_module

    assert not hasattr(dispatch_module, "DispatchRecord")
    source = Path(dispatch_module.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        assert not stripped.startswith(("import v2_runtime", "from .v2_runtime", "from tcenv.backends.metal.v2_runtime"))
    records: list[PortfolioDispatchRecord] = []
    resolver = _resolver(contract, plan, instances, portfolio)
    records.append(resolver.resolve({}))
    records.append(
        resolver.resolve(
            DispatchRequest(
                op_name="portfolio_pointwise",
                dtype="fp16",
                size="not_a_named_size",
                usage="golden",
                symbols={"N": 1},
                attrs={},
                semantic_instance_digest="a" * 64,
                target_abi_digest="b" * 64,
            )
        )
    )
    for size in ("below", "threshold", "tail", "large"):
        request, instance, abi = _request(contract, plan, size)
        records.append(resolver.resolve(request, preflight=_hook(instance, abi)))
    for record in records:
        _expect_schema(record)
        document = record.document()
        assert document["record_version"] == 1
        assert document["dispatch_rules_version"] == 1
        assert record.identity == fingerprint(document)
        for forbidden in ("active", "publication_allowed", "speedup", "median_us", "eval_report", "manifest"):
            assert forbidden not in document
        if record.outcome not in ("semantic_unresolved", "unsupported_contract_capability"):
            assert len(record.guards) == len(portfolio.variants)
        assert list(record.evidence) == sorted(record.evidence)
    # The resolved record names the variant and keeps every guard outcome for replay.
    resolved = records[-1]
    assert resolved.outcome == "resolved"
    assert resolved.variant_id == portfolio.variants[1].variant_id
    assert resolved.request["instance_key"] == "fp16/large/golden"
    assert resolved.request["symbols"] == {"N": 1024}
    # Out-of-domain requests never even evaluate guards.
    unsupported = [record for record in records if record.outcome == "unsupported_contract_capability"]
    assert unsupported and unsupported[0].guards == ()
    # A record cannot let a fallback masquerade as the candidate (or vice versa).
    fields = records[-1].document()
    with pytest.raises(TaskContractError, match="provider=candidate must be a resolved outcome"):
        PortfolioDispatchRecord(
            domain_plan_digest=fields["domain_plan_digest"],
            portfolio_digest=fields["portfolio_digest"],
            request=fields["request"],
            outcome="fallback",
            provider="candidate",
            variant_id=fields["variant_id"],
        )
    with pytest.raises(TaskContractError, match="longer than"):
        PortfolioDispatchRecord(
            domain_plan_digest=fields["domain_plan_digest"],
            portfolio_digest=fields["portfolio_digest"],
            request=fields["request"],
            outcome="resolved",
            detail="x" * 4001,
        )
    with pytest.raises(TaskContractError, match="provider=fallback must be a fallback outcome"):
        PortfolioDispatchRecord(
            domain_plan_digest=fields["domain_plan_digest"],
            portfolio_digest=fields["portfolio_digest"],
            request=fields["request"],
            outcome="resolved",
            provider="fallback",
            fallback_id=fields["variant_id"],
        )
    # The decision record references no projection: projection is a separate mechanical step.
    assert "candidate_manifest" not in json.dumps(resolved.document())
    assert FALLBACK_REASONS == {"outside_candidate_coverage", "resource_unavailable", "variant_unavailable"}
