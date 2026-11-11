"""P2-B: fresh, independently rebuilt confirmation for one portfolio subject.

Each subject is evaluated twice through the existing bounded v2 worker (a real build each time),
then compared with the R5 identity comparator.  The expected identity is rebuilt from the task
contract, selected projection/geometry and the registered builder's current probe — never copied
from the search report.  Original schema-4 reports are preserved verbatim in the sidecar.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any, Mapping

from taic.policy import compare_confirmation_identity
from tcenv.backends.metal.backend import MetalBackend
from tcenv.backends.metal.launch import launch_math
from tcenv.backends.metal.source_builder import MetalSourceBuilder
from tcenv.backends.metal.track_adapter import (
    MetalTrackBuilder,
    MetalTrackEvaluator,
    combined_candidate_source,
    metal_contract,
    msl_single_geometry_resolver,
    v2_evaluation_task,
)
from tcenv.bench import BENCH_ATTEMPTS
from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
from tcenv.evaluate import (
    _evaluation_fingerprint,
    baseline_environment_digest,
    baseline_implementation_digest,
)
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import CandidateProjection, project_provider
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver
from tcenv.schema import PROTOCOL_VERSION
from tcenv.target import TargetProfile, load_target
from tcenv.taskcontract.locator import TaskLocator
from tcenv.taskcontract.portfolio_runner import _preflight_hook, locator_for_instance, run_portfolio_task
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from tcenv.track.delivery import candidate_geometry
from tcenv.track.geometry import ResolvedGeometry, geometry_pair, select_geometry
from tcenv.track.registry import load_track_registry, require_msl_track
from tcenv.track import tilelang as tilelang_contract
from tcenv.track import tilelang_confirmation as fresh_contract
from .track_factory import adapter_for_track
from tcenv.remote.artifact_builder import BuildProvenance
from taic.portfolio_confirmation import (
    CONFIRMATION_SCHEMA_ID,
    SEARCH_SEED,
    _finalize_subject_document,
    confirmation_seed_for,
    verify_subject_document,
)
from ...portfolio_validation.validation import PortfolioFreeze, ValidationPlan

def activation_expectation(
    *,
    locator: TaskLocator,
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    document: Mapping[str, Any],
    track_version: int | None = None,
    track_id: str | None = None,
):
    """@brief 为一个 subject 独立重建 dispatch record 与完整 expected（不读 sidecar expected）。

    @param track_version 该 run 冻结的选择版本；``None`` 表示按 registry 默认解析（仅用于没有
        记录版本的归档重放）。带版本的重建在缓存键里包含版本，不能与另一个版本的结果互换。
    """
    validation.validate_for(freeze)
    if locator.op_name != freeze.authority.spec.op_name:
        raise TaskContractError("activation locator names another task")
    locator.load()
    if locator.deployment_pack_digest != freeze.task_pack_digest:
        raise TaskContractError("activation task pack is not the frozen task pack")
    dtype, size, usage = document["instance_key"].split("/")
    cache_key = (locator.identity, freeze.digest, validation.digest, document["instance_key"], track_version, track_id)
    cached = _EXPECTATION_CACHE.get(cache_key)
    if cached is not None:
        return cached
    point = freeze.domain_plan.find(dtype, size, usage)
    if point is None:
        raise TaskContractError(f"{document['instance_key']}: not in the frozen domain")
    portfolio = freeze.portfolio()
    evidence = {variant_id: "validated" for variant_id in portfolio.variant_ids}
    if portfolio.fallback_id is not None:
        evidence[portfolio.fallback_id] = "validated"
    resolver = ExactDispatchResolver(
        domain_plan=freeze.domain_plan,
        portfolio=portfolio,
        evidence=evidence,
        spec=freeze.authority.spec,
        target=freeze.authority.target,
        instances=freeze.authority.instances,
        source_preflight=True,
    )
    instance = instantiate_v2(freeze.authority.spec, dtype, size)
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)
    run_contract = locator_for_instance(locator, dtype, size).load()
    if track_id == tilelang_contract.TILELANG_TRACK_ID:
        selected = load_track_registry().get(track_id, version=track_version)
        if selected.track_version != fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            raise UnsupportedContractCapability("only TileLang /3 has an activation expectation route")
        fresh_contract.audit_fresh_registration(selected)
        geometry_resolver = candidate_geometry
    else:
        geometry_resolver = msl_single_geometry_resolver(run_contract)
    hook = _preflight_hook(
        instance,
        target=freeze.authority.target,
        max_buffer_length=None,
        portfolio=portfolio,
        authority=freeze.authority,
        resolver=geometry_resolver,
        track_version=track_version,
        track_id=track_id,
    )
    record = resolver.resolve(request, preflight=hook)
    if record.outcome == "resolved":
        provider = portfolio.variant_for_provider(record.variant_id)
    else:
        provider = portfolio.fallback
    if provider is None:
        raise TaskContractError(f"{document['instance_key']}: dispatch selected no projectable provider")
    projection = project_provider(
        portfolio,
        provider,
        authority=freeze.authority,
        track_version=track_version,
        **({"track_id": track_id} if track_id is not None else {}),
    )
    if track_version is not None and int(projection.track_version) != int(track_version):
        raise TaskContractError(
            f"{document['instance_key']}: the activation projection resolved track version "
            f"{projection.track_version}, not the frozen {track_version}"
        )
    profile = load_target(freeze.authority.target.id)
    resolved = _resolved(run_contract, projection, profile)
    expected = confirmation_expected(
        run_contract,
        projection,
        resolved,
        locator=locator,
        target_id=profile.id,
        # Epoch 4 is frozen into the validation plan, so activation cannot rebuild an
        # epoch-1 expected and then reject the correctly confirmed per-launch sidecar.
        batch_submission="per_launch_set_event" if freeze.protocol_epoch == 4 else "single_buffer",
    )
    result = (record, expected)
    _EXPECTATION_CACHE[cache_key] = result
    return result


def _evaluate(
    locator, domain_plan, portfolio, request, evidence, *, mode: str, seed: int, reps: int, target_id: str,
    track_version: int | None = None,
    track_id: str = "msl-single",
    batch_submission: str = "single_buffer",
):
    """@brief 新 builder/evaluator：本 subject 的这次真实构建不由上一次复用。"""
    if track_id == tilelang_contract.TILELANG_TRACK_ID:
        if batch_submission != "single_buffer":
            raise UnsupportedContractCapability("per-launch confirmation is currently an MSL-only experiment")
        selected = load_track_registry().require_available(track_id, version=track_version)
        if selected.track_version != fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            raise UnsupportedContractCapability(
                f"track {track_id}/{selected.track_version}: only /3 has a fresh DSL evaluator route"
            )
        adapter = adapter_for_track(selected)
        builder_type = adapter.resolve("builder")
        evaluator_type = adapter.resolve("evaluator")
        geometry = adapter.resolve("geometry_resolver")
        return run_portfolio_task(
            locator=locator,
            domain_plan=domain_plan,
            portfolio=portfolio,
            request=request,
            evidence=evidence,
            builder=builder_type(target_id=target_id),
            evaluator=evaluator_type(
                target_id=target_id,
                mode=mode,
                seed=seed,
                reps=reps,
                cases=tuple(locator.load().instance.case_domain.cases),
            ),
            geometry_resolver_factory=lambda _contract: geometry,
            target_id=target_id,
            track_version=track_version,
            track_id=track_id,
        )
    evaluator_options = dict(
        target_id=target_id,
        mode=mode,
        seed=seed,
        reps=reps,
        cases=tuple(locator.load().instance.case_domain.cases),
    )
    if batch_submission != "single_buffer":
        evaluator_options["batch_submission"] = batch_submission
    return run_portfolio_task(
        locator=locator,
        domain_plan=domain_plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence,
        builder=MetalTrackBuilder(target_id=target_id),
        evaluator=MetalTrackEvaluator(**evaluator_options),
        geometry_resolver_factory=msl_single_geometry_resolver,
        target_id=target_id,
        track_version=track_version,
    )


def _resolved(contract, projection: CandidateProjection, profile: TargetProfile) -> ResolvedGeometry:
    # DSL /3 owns an explicit kernel-pipeline geometry and fresh-stable builder identity.  /1,/2
    # remain typed refusals here; no DSL subject is ever parsed by the MSL launch path.
    selected = load_track_registry().get(projection.track_id, version=projection.track_version)
    is_fresh_dsl = (
        selected.is_dsl
        and selected.track_version == fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION
    )
    if selected.is_dsl and not is_fresh_dsl:
        raise UnsupportedContractCapability(
            f"track {projection.track_id}/{projection.track_version} is not the MSL portfolio or TileLang /3 confirmation path"
        )
    if is_fresh_dsl:
        fresh_contract.audit_fresh_registration(selected)
    else:
        require_msl_track(projection.track_id, version=projection.track_version)
    instance = contract.instance
    abi = derive_tensor_memory_contract(instance, target=profile)
    if projection.track_id == "msl-bundle" or is_fresh_dsl:
        execution = candidate_geometry(projection.candidate)
    else:
        execution = msl_single_geometry_resolver(contract)(projection.candidate)
    limits = abi_limits_from_target(profile)
    pair = geometry_pair(execution, None, instance=instance, contract=abi, limits=limits)
    return select_geometry(pair).geometry


_PROBE_CACHE: dict[tuple[str, str, tuple[str, ...], str], Any] = {}
_EXPECTATION_CACHE: dict[tuple[Any, ...], Any] = {}


def _expected_from_capabilities(
    contract,
    projection: CandidateProjection,
    resolved: ResolvedGeometry,
    *,
    locator: TaskLocator,
    target_id: str,
    capabilities: Any,
    batch_submission: str = "single_buffer",
) -> dict[str, Any]:
    if not capabilities.available:
        raise TaskContractError(f"the registered Metal builder cannot establish a live expected identity: {capabilities.detail}")
    profile = load_target(target_id)
    abi = derive_tensor_memory_contract(contract.instance, target=profile)
    candidate_sha256 = projection.candidate.candidate_sha256
    provenance = BuildProvenance(
        profile=capabilities.profile,
        source_sha256=candidate_sha256,
        artifact_set_digest=None,
    ).to_dict()
    task = v2_evaluation_task(
        SimpleNamespace(
            contract=contract,
            abi=abi,
            target_id=target_id,
            locator=locator,
            selection=SimpleNamespace(geometry=resolved),
        )
    )
    compile_profile = dict(provenance["compile_profile"])
    backend = MetalBackend()
    protocol_version = PROTOCOL_VERSION if batch_submission == "single_buffer" else f"{PROTOCOL_VERSION}+{batch_submission}"
    return {
        "build": {
            "candidate_sha256": candidate_sha256,
            "opspec_digest": contract.spec.digest,
            "lowering_profile_digest": contract.fingerprint(),
            "provenance": provenance,
        },
        "evaluation": {
            "evaluation_fingerprint": _evaluation_fingerprint(task, profile.backend, target_id),
            "baseline_implementation_digest": baseline_implementation_digest(task, profile.backend),
            "baseline_environment_digest": baseline_environment_digest(task, profile.backend, target_id),
            "protocol_version": protocol_version,
            "target_id": target_id,
            "backend": profile.backend,
            "backend_version": str(backend.version),
            "device_name": compile_profile.get("device"),
            "gpu_family": compile_profile.get("gpu_family"),
        },
        "protocol_version": protocol_version,
        "target_id": target_id,
    }


def confirmation_expected(
    contract,
    projection: CandidateProjection,
    resolved: ResolvedGeometry,
    *,
    locator: TaskLocator,
    target_id: str,
    batch_submission: str = "single_buffer",
) -> dict[str, Any]:
    """@brief 从合同/几何/当前注册 builder probe 独立重建 expected（不读报告）。"""
    selected = load_track_registry().get(projection.track_id, version=projection.track_version)
    if selected.is_dsl:
        if selected.track_version != fresh_contract.TILELANG_CONFIRMATION_TRACK_VERSION:
            raise UnsupportedContractCapability(
                f"track {projection.track_id}/{projection.track_version} is not the MSL portfolio or TileLang /3 confirmation path"
            )
        fresh_contract.audit_fresh_registration(selected)
        from .tilelang_metal import TileLangFreshMetalBuilder

        capabilities = TileLangFreshMetalBuilder(selected, target_id=target_id).probe(
            projection.candidate, resolved
        )
        return _expected_from_capabilities(
            contract, projection, resolved, locator=locator, target_id=target_id, capabilities=capabilities,
            batch_submission=batch_submission,
        )

    require_msl_track(projection.track_id, version=projection.track_version)
    profile = load_target(target_id)
    abi = derive_tensor_memory_contract(contract.instance, target=profile)
    kernel_contract = metal_contract(
        task_contract=contract,
        abi=abi,
        geometry=resolved,
        target_id=target_id,
    )
    source_text = combined_candidate_source(projection.candidate)
    math = launch_math(source_text) or "safe"
    kernel_names = tuple(dict.fromkeys(stage.kernel for stage in resolved.active_stages))
    cache_key = (kernel_contract.kernel_name, math, kernel_names, target_id)
    capabilities = _PROBE_CACHE.get(cache_key)
    if capabilities is None:
        capabilities = MetalSourceBuilder(
            kernel_contract=kernel_contract,
            math=math,
            kernel_names=kernel_names,
        ).probe()
        _PROBE_CACHE[cache_key] = capabilities
    return _expected_from_capabilities(
        contract, projection, resolved, locator=locator, target_id=target_id, capabilities=capabilities,
        batch_submission=batch_submission,
    )


def confirm_subject(
    *,
    locator: TaskLocator,
    domain_plan,
    portfolio,
    request: DispatchRequest,
    evidence: Mapping[str, str],
    validation: ValidationPlan,
    freeze: PortfolioFreeze,
    track_version: int | None = None,
    track_id: str | None = None,
    batch_submission: str = "single_buffer",
) -> dict[str, Any]:
    """@brief 对一个 unseen subject 做两次独立真实构建/执行并生成确认 sidecar。

    @param track_version 该 run 冻结的选择版本；两次真实执行与 expected 重建都必须用它投影，
        不能从 live registry 默认重选（否则一个冻结在旧版本的 run 会静默换实现）。
    """
    if locator.op_name != freeze.authority.spec.op_name:
        raise TaskContractError("confirmation locator names another task")
    locator.load()
    if request.instance_key not in set(validation.unseen_instance_keys) | set(validation.tune_shards):
        raise TaskContractError("fresh confirmation may only run a mastered unseen or declared tune instance_key")
    if locator.deployment_pack_digest != freeze.task_pack_digest:
        raise TaskContractError("confirmation task pack is not the frozen task pack")
    validation.validate_for(freeze)
    if freeze.portfolio().digest != portfolio.digest:
        raise TaskContractError("confirmation portfolio is not the candidate bundle pinned by the freeze")
    fresh_seed = confirmation_seed_for(validation, request.instance_key)
    if fresh_seed == SEARCH_SEED:
        raise TaskContractError("the derived fresh seed must differ from the search seed")
    mode = "full" if request.usage == "benchmark" else "quick"
    target_id = freeze.authority.target.id
    search = _evaluate(
        locator, domain_plan, portfolio, request, evidence, mode=mode, seed=SEARCH_SEED, reps=3,
        target_id=target_id, track_version=track_version, track_id=track_id,
        batch_submission=batch_submission,
    )
    if search.result is None or search.projection is None:
        raise TaskContractError("an unseen subject did not produce a complete search evaluation")
    if track_id is not None and search.projection.track_id != track_id:
        raise TaskContractError(
            f"the confirmation projection selected {search.projection.track_id}, not the frozen {track_id}"
        )
    if track_version is not None and int(search.projection.track_version) != int(track_version):
        raise TaskContractError(
            f"the confirmation projection resolved track version {search.projection.track_version}, not the "
            f"frozen {track_version}; a confirmation never silently upgrades the Track version"
        )
    contract = locator_for_instance(locator, request.dtype, request.size).load()
    profile = load_target(target_id)
    resolved = _resolved(contract, search.projection, profile)
    expected = confirmation_expected(
        contract, search.projection, resolved, locator=locator, target_id=profile.id,
        batch_submission=batch_submission,
    )
    search_report = copy.deepcopy(search.result.report)
    candidate_sha256 = search.projection.candidate.candidate_sha256
    for attempt in range(1, BENCH_ATTEMPTS + 1):
        confirmation = _evaluate(
            locator, domain_plan, portfolio, request, evidence, mode=mode, seed=fresh_seed, reps=15,
            target_id=target_id, track_version=track_version, track_id=track_id,
            batch_submission=batch_submission,
        )
        if confirmation.result is None or confirmation.projection is None:
            raise TaskContractError("an unseen subject did not produce two complete real evaluations")
        if track_id is not None and confirmation.projection.track_id != track_id:
            raise TaskContractError(
                f"the fresh projection selected {confirmation.projection.track_id}, not the frozen {track_id}"
            )
        if track_version is not None and int(confirmation.projection.track_version) != int(track_version):
            raise TaskContractError(
                f"the confirmation projection resolved track version {confirmation.projection.track_version}, not "
                f"the frozen {track_version}"
            )
        if search.record.provider != confirmation.record.provider or search.record.variant_id != confirmation.record.variant_id:
            raise TaskContractError("search and confirmation selected different providers")
        confirmation_report = copy.deepcopy(confirmation.result.report)
        identity = compare_confirmation_identity(
            candidate_sha256=candidate_sha256,
            expected=expected,
            search=search_report,
            confirmation=confirmation_report,
            scope="local",
        )
        document = {
            "schema_id": CONFIRMATION_SCHEMA_ID,
            "instance_key": request.instance_key,
            "usage": request.usage,
            "provider": confirmation.record.provider,
            "variant_id": confirmation.record.variant_id,
            "fallback_id": confirmation.record.fallback_id,
            "candidate_sha256": candidate_sha256,
            "credits_variant": confirmation.evaluation["credits_variant"],
            "search_seed": SEARCH_SEED,
            "confirmation_seed": fresh_seed,
            "confirmation_attempts": attempt,
            "expected": expected,
            "identity": identity,
            "search_report": search_report,
            "confirmation_report": confirmation_report,
            "status": "rejected",
            "reason": "not_evaluated",
            "coverage_kind": "sampled_unseen" if request.instance_key in validation.unseen_instance_keys else "tune_confirmation",
            "confirmation_digest": "0" * 64,
        }
        verified = verify_subject_document(_finalize_subject_document(document))
        if attempt == BENCH_ATTEMPTS or verified["status"] == "confirmed" or verified["reason"] != "timing_uncertain":
            return verified
    raise AssertionError("unreachable: the bounded confirmation attempts must return a verdict")


__all__ = [
    "CONFIRMATION_SCHEMA_ID",
    "confirm_subject",
    "confirmation_expected",
    "verify_subject_document",
    "activation_expectation",
]
