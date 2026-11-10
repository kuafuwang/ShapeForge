"""P4 offline shape-portfolio simulation over the existing fake-NPU protocols.

The finite-domain portfolio is resolved with the P0 domain plan, projection identity and exact
dispatch resolver.  Each selected provider is then submitted, unchanged, through the existing
source contract (fake vendor A) and the existing bundle contract (fake vendor B).  P2 validation
and manifest negatives are applied to that same portfolio: unseen seeds stay out of the candidate
view, a fallback cannot be credited as a variant, a forged manifest digest is rejected, and a
document-only manifest cannot be activated.

Nothing here executes a Qingwei compiler, a real NPU, or a Metal device.  The report records
``simulated=true``, ``publication_allowed=false`` and ``npu_p0.status=blocked``.  Timing, speedup
and publication fields from the fake workers are dropped instead of being copied into the report.
Proto, the six RPCs, field numbers, HMAC, BlobChannel and the remote EvalReport schema are not
modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from tcenv.contract_errors import TaskContractError
from tcenv.evaluation import EvaluationRequest
from tcenv.identity import canonical_bytes, fingerprint
from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio.dispatch import DispatchRequest, ExactDispatchResolver, preflight_geometry
from tcenv.portfolio.domain import DomainAuthority, normalize_semantic_domain_plan
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.remote import contracts
from tcenv.remote.artifact_bundle import ArtifactBundle, BundleEntry, BundleError, payload_bundle, payload_bytes
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract.locator import discover_standard_tasks
from tcenv.track import REPO_ROOT
from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest
from taic.portfolio_manifest import DispatchManifest, ManifestSubject, verify_manifest_document
from tests.support.job_engine import NPU_TARGET, NPU_TASK_FINGERPRINT
from tests.support.shape_portfolio import (
    evidence_all_valid,
    load_portfolio_task,
    parent_entries,
    portfolio_document,
    portfolio_instances,
)
from tests.support.v1_tasks import V1_TASK
from tests.support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    GraphGrpcHarness,
    NpuGrpcHarness,
)

SCHEMA_ID = "local.shape-portfolio-npu-simulation.1"
TRANSPORT_SCHEMA_ID = "local.shape-portfolio-npu-simulation.transport.1"
MASTER_SEED = b"p4-npu-offline-master-seed"
GUIDANCE = {"guidance": "P4 offline NPU simulation; not a candidate-visible seed"}
EVIDENCE_PATH = (
    REPO_ROOT
    / "tests/support/golden/shared/shape-portfolio-阶段验收/P4/evidence/simulation.json"
)
FORBIDDEN_KEYS = frozenset(
    {
        "speedup",
        "speedups",
        "candidate_us",
        "elapsed_s",
        "elapsed_ns",
        "device_elapsed_ns",
        "host_us",
        "device_us",
        "latency",
        "latency_ratio",
        "timing",
        "publishable",
        "best_speedup",
        "specialized_hit_speedup",
        "effective_end_to_end_speedup",
        "confirmed",
    }
)
SOURCE_API = "tcenv.remote.npu-source.evaluator-api.1"
SOURCE_EVIDENCE = "tcenv.remote.npu-source.evidence.1"
BUNDLE_API = "tcenv.remote.npu-bundle.evaluator-api.1"
BUNDLE_EVIDENCE = "tcenv.remote.npu-bundle.evidence.1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _forbid(document: Any, *, where: str = "report") -> None:
    """@brief 拒绝把计时、加速比或发布字段写进 P4 证据。"""
    if isinstance(document, Mapping):
        found = FORBIDDEN_KEYS & set(document)
        if found:
            raise AssertionError(f"{where} carries forbidden fields {sorted(found)}")
        for key, value in document.items():
            _forbid(value, where=f"{where}.{key}")
    elif isinstance(document, list):
        for index, value in enumerate(document):
            _forbid(value, where=f"{where}[{index}]")


def _entry(path: str, role: str, media_type: str, data: bytes) -> BundleEntry:
    return BundleEntry(
        path=path,
        role=role,
        media_type=media_type,
        size_bytes=len(data),
        sha256=_sha256(data),
        data=data,
    )


def _bundle(transport: bytes, binding: bytes) -> ArtifactBundle:
    """@brief 用现有 GRAPH bundle 角色装入调度绑定；不新增 artifact role。"""
    return ArtifactBundle(
        entries=(
            _entry("graph/dispatch.json", "GRAPH_DEFINITION", "application/json", transport),
            _entry("graph/params.bin", "PARAMETER_TABLE", "application/octet-stream", binding),
        )
    )


def _preflight(instance, abi):
    limits = abi_limits_from_target(load_target(DEFAULT_TARGET_ID))

    def hook(geometry, provider=None):
        del provider
        return preflight_geometry(geometry, instance=instance, contract=abi, limits=limits)

    return hook


def _world(root: Path) -> dict[str, Any]:
    """@brief 装配 P0 有限域 portfolio、P2 freeze/plan，不做协议提交。"""
    task_root = root / "tasks"
    contract = load_portfolio_task(task_root)
    locator = next(
        item
        for item in discover_standard_tasks(task_root, code_root=REPO_ROOT)
        if item.op_name == contract.op_name
    )
    target = load_target(DEFAULT_TARGET_ID)
    instances = portfolio_instances(contract)
    draft = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(
        plan=draft, spec=contract.spec, target=target, instances=instances, contract=contract,
    )
    plan = authority.plan
    entries = parent_entries(portfolio_document(plan.digest))
    portfolio = parse_portfolio(entries, domain_plan=plan)
    freeze = PortfolioFreeze(
        protocol_epoch=1,
        authority=authority,
        candidate_entries=entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=fingerprint(GUIDANCE),
    )
    tune = tuple(sorted(key for key in freeze.instance_keys() if key.startswith("fp16/large/")))
    required = tuple(sorted(set(freeze.instance_keys()) - set(tune)))
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=tune,
        unseen_count=len(required),
        master_seed=MASTER_SEED,
        required_instance_keys=required,
    )
    resolver = ExactDispatchResolver(
        domain_plan=plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=contract.spec,
        target=target,
        instances=instances,
    )
    return {
        "contract": contract,
        "locator": locator,
        "target": target,
        "authority": authority,
        "plan": plan,
        "portfolio": portfolio,
        "freeze": freeze,
        "validation": validation,
        "resolver": resolver,
    }


def _dispatch_rows(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    portfolio = world["portfolio"]
    authority = world["authority"]
    for point in sorted(world["plan"].instances, key=lambda item: item.instance_key):
        instance = next(
            item for item in world["resolver"].instances if (item.dtype, item.size) == (point.dtype, point.size)
        )
        abi = derive_tensor_memory_contract(instance, target=world["target"])
        request = DispatchRequest.from_instance(
            instance, target_abi_digest=point.target_abi_digest, usage=point.usage,
        )
        record = world["resolver"].resolve(request, preflight=_preflight(instance, abi))
        if record.provider == "candidate":
            provider = portfolio.variant_for_provider(record.variant_id)
            variant_key = provider.variant_key
            credits = True
        elif record.provider == "fallback":
            provider = portfolio.fallback
            variant_key = None
            credits = False
        else:
            raise AssertionError(f"{point.instance_key} did not select one provider: {record.outcome}")
        projection = project_provider(portfolio, provider, authority=authority)
        rows.append(
            {
                "instance_key": point.instance_key,
                "usage": point.usage,
                "outcome": record.outcome,
                "provider": record.provider,
                "variant_key": variant_key,
                "variant_id": record.variant_id,
                "fallback_id": record.fallback_id,
                "fallback_reason": record.fallback_reason,
                "credits_variant": credits,
                "candidate_sha256": projection.candidate.candidate_sha256,
                "track_id": projection.track_id,
                "dispatch_record_digest": record.identity,
                "binding": projection.binding.document(),
            }
        )
    return rows


def _transport(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    return {
        "schema_id": TRANSPORT_SCHEMA_ID,
        "instance_keys": [row["instance_key"] for row in rows],
        "provider": first["provider"],
        "variant_key": first["variant_key"],
        "variant_id": first["variant_id"],
        "fallback_id": first["fallback_id"],
        "credits_variant": first["credits_variant"],
        "candidate_sha256": first["candidate_sha256"],
        "binding": first["binding"],
        "executed_on_npu": False,
    }


def _groups(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["provider"],
            row["variant_id"] or "",
            row["fallback_id"] or "",
            row["candidate_sha256"],
        )
        grouped.setdefault(key, []).append(row)
    return [grouped[key] for key in sorted(grouped)]


def _report_value(report, name: str):
    if isinstance(report, Mapping):
        return report.get(name)
    return getattr(report, name, None)


def _omitted(result) -> list[str]:
    found = []
    evidence = result.evidence or {}
    if "timing" in evidence:
        found.append("evidence.timing")
    report = result.report
    for name in ("speedup", "speedups", "candidate_us", "elapsed_s"):
        value = _report_value(report, name)
        if value not in (None, {}, []):
            found.append(f"report.{name}")
    return found


def _protocol_view(result, *, request_digest: str) -> dict[str, Any]:
    receipt = result.receipt or {}
    evidence = result.evidence or {}
    group = receipt.get("device_group")
    group_id = group.get("device_group_id") if isinstance(group, dict) else group
    measurement = result.provenance.measurement_kind
    if measurement != "simulated" or evidence.get("publication_allowed") is not False or evidence.get("complete") is not True:
        raise AssertionError("fake NPU result is not a completed simulation")
    return {
        "request_digest": request_digest,
        "candidate_digest": receipt["candidate_digest"],
        "device_group_id": group_id,
        "measurement_kind": measurement,
        "publication_allowed": False,
        "complete": True,
        "fixture_task": V1_TASK,
        "timing_numbers_included": False,
    }


def _submit_source(harness: NpuGrpcHarness, root: Path, label: str, payload: bytes) -> tuple[dict[str, Any], list[str]]:
    outbox = EvaluationOutbox(
        root / f"source-outbox-{label}",
        signature_key=NPU_HOST_SIGNATURE_KEY,
        transport="grpc",
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
    )
    request = bind_wire_digest(
        EvaluationRequest(
            kind="formal_eval",
            request_id=f"p4-source-{label}",
            run_id="npu-run-1",
            session_id="npu-session-1",
            task=V1_TASK,
            dtype="fp16",
            size="small",
            target_id=NPU_TARGET,
            task_fingerprint=NPU_TASK_FINGERPRINT,
            source=payload.decode("utf-8"),
            mode="quick",
            seed=7,
            reps=1,
            queue_timeout_s=10.0,
            candidate_timeout_s=10.0,
        ),
        harness.factory,
        outbox,
    )
    result = harness.factory.formal(request)
    view = _protocol_view(result, request_digest=request.request_digest)
    digest = view["candidate_digest"]
    if digest != {"algorithm": "sha256", "value": _sha256(payload)}:
        raise AssertionError("source receipt does not bind the submitted portfolio bytes")
    return view, _omitted(result)


def _submit_bundle(harness: GraphGrpcHarness, label: str, bundle: ArtifactBundle) -> tuple[dict[str, Any], list[str]]:
    harness.bundle = bundle
    request = bind_wire_digest(
        EvaluationRequest(
            kind="formal_eval",
            request_id=f"p4-bundle-{label}",
            run_id="npu-run-1",
            session_id="npu-session-1",
            task=V1_TASK,
            dtype="fp16",
            size="small",
            target_id=harness.engine.identity.target_id,
            task_fingerprint=NPU_TASK_FINGERPRINT,
            source="",
            mode="quick",
            seed=7,
            reps=1,
            queue_timeout_s=10.0,
            candidate_timeout_s=10.0,
        ),
        harness.factory,
        npu_contract=True,
    )
    result = harness.factory.formal(request)
    view = _protocol_view(result, request_digest=request.request_digest)
    view["artifact_set_digest"] = bundle.artifact_set_digest
    digest = view["candidate_digest"]
    if digest != {"algorithm": contracts.BUNDLE_DIGEST_ALGORITHM, "value": bundle.artifact_set_digest}:
        raise AssertionError("bundle receipt does not bind the artifact set")
    return view, _omitted(result)


def _label(row: Mapping[str, Any]) -> str:
    """@brief 协议 request id 用诊断名；fallback 没有 variant_key。"""
    if row["variant_key"]:
        return str(row["variant_key"])
    return "domain-fallback"


def _manifest(world: Mapping[str, Any], rows: list[dict[str, Any]]) -> tuple[DispatchManifest, dict[str, str]]:
    freeze = world["freeze"]
    validation = world["validation"]
    subjects = []
    for row in rows:
        kind = "tune_confirmation" if row["instance_key"] in validation.tune_shards else "sampled_unseen"
        subjects.append(
            ManifestSubject(
                instance_key=row["instance_key"],
                usage=row["usage"],
                provider=row["provider"],
                variant_id=row["variant_id"],
                fallback_id=row["fallback_id"],
                candidate_sha256=row["candidate_sha256"],
                credits_variant=row["credits_variant"],
                stable_confirmation_digest=fingerprint(
                    {
                        "kind": "document-only-simulated-binding",
                        "instance_key": row["instance_key"],
                        "candidate_sha256": row["candidate_sha256"],
                    }
                ),
                coverage_kind=kind,
            )
        )
    manifest = DispatchManifest(
        portfolio_freeze_digest=freeze.digest,
        validation_plan_digest=validation.digest,
        status="validated",
        activation_generation=0,
        subjects=tuple(sorted(subjects, key=lambda item: item.instance_key)),
    )
    verified = verify_manifest_document(manifest.document(), freeze, validation)
    if verified.digest != manifest.digest:
        raise AssertionError("document-only manifest did not round-trip")
    negatives: dict[str, str] = {}
    try:
        manifest.activate(freeze, validation)
    except TaskContractError as exc:
        negatives["activation_rejected"] = str(exc)
    else:
        raise AssertionError("a document-only manifest was activated")
    forged = manifest.document()
    forged["dispatch_manifest_digest"] = ("0" if forged["dispatch_manifest_digest"][0] != "0" else "1") + forged[
        "dispatch_manifest_digest"
    ][1:]
    try:
        DispatchManifest.from_document(forged)
    except TaskContractError as exc:
        negatives["forged_manifest_digest_rejected"] = str(exc)
    else:
        raise AssertionError("a forged manifest digest was accepted")
    fallback = next(row for row in rows if row["provider"] == "fallback")
    try:
        ManifestSubject(
            instance_key=fallback["instance_key"],
            usage=fallback["usage"],
            provider="fallback",
            variant_id=fallback["candidate_sha256"],
            fallback_id=fallback["fallback_id"],
            candidate_sha256=fallback["candidate_sha256"],
            credits_variant=True,
            stable_confirmation_digest=fallback["candidate_sha256"],
            coverage_kind="sampled_unseen",
        )
    except TaskContractError as exc:
        negatives["fallback_credit_rejected"] = str(exc)
    else:
        raise AssertionError("fallback credit was accepted")
    swapped = subjects[0].document()
    swapped["variant_id"] = "a" * 64
    swapped_subject = ManifestSubject.from_document(swapped)
    try:
        DispatchManifest(
            portfolio_freeze_digest=freeze.digest,
            validation_plan_digest=validation.digest,
            status="validated",
            activation_generation=0,
            subjects=(swapped_subject,),
        ).require_complete_coverage(freeze, validation)
    except TaskContractError as exc:
        negatives["identity_swap_rejected"] = str(exc)
    else:
        raise AssertionError("a foreign variant id was accepted")
    return manifest, negatives


def _closed_dispatch(world: Mapping[str, Any]) -> dict[str, Any]:
    """@brief 域外与自相矛盾的请求都不进入 fallback，也不提交协议。"""
    point = next(item for item in world["plan"].instances if item.instance_key == "fp16/threshold/golden")
    instance = next(item for item in world["resolver"].instances if item.size == "threshold")
    good = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage="golden")
    outside = world["resolver"].resolve(
        DispatchRequest(
            op_name=good.op_name,
            dtype="fp16",
            size="not_a_named_size",
            usage="golden",
            symbols={"N": 96},
            attrs={},
            semantic_instance_digest=good.semantic_instance_digest,
            target_abi_digest=good.target_abi_digest,
        )
    )
    contradictory = world["resolver"].resolve({**good.document(), "symbols": {"N": 1024}})
    for record, outcome in ((outside, "unsupported_contract_capability"), (contradictory, "semantic_unresolved")):
        if record.outcome != outcome or record.provider is not None or record.fallback_id is not None:
            raise AssertionError(f"{outcome} leaked into a provider: {record.document()}")
    return {
        "outside_domain": {
            "outcome": outside.outcome,
            "provider": None,
            "fallback_id": None,
            "protocol_submitted": False,
        },
        "contradictory_request": {
            "outcome": contradictory.outcome,
            "provider": None,
            "fallback_id": None,
            "protocol_submitted": False,
        },
    }


def simulate(root: Path | None = None) -> dict[str, Any]:
    """@brief 跑完有限域调度、两条假 NPU 协议和 P2 负例，返回可重放报告。"""
    if root is None:
        with tempfile.TemporaryDirectory(prefix="p4-npu-portfolio-") as temporary:
            return simulate(Path(temporary))
    world = _world(root)
    rows = _dispatch_rows(world)
    view = world["validation"].candidate_view(world["freeze"]).document()
    if any(key in json.dumps(view) for key in ("fresh_seed", "master_seed", "unseen")):
        raise AssertionError("candidate view leaks validation secrets")
    for secret in list(world["validation"].unseen_instance_keys) + [str(world["validation"].fresh_seeds)]:
        if secret and secret in json.dumps(view):
            raise AssertionError("candidate view contains an unseen key or seed")
    manifest, negatives = _manifest(world, rows)
    closed = _closed_dispatch(world)
    omitted: set[str] = set()
    protocol = []
    tamper_rejected = False
    with NpuGrpcHarness(root / "source") as source_harness, GraphGrpcHarness(root / "bundle") as bundle_harness:
        for group in _groups(rows):
            transport = canonical_bytes(_transport(group))
            binding = canonical_bytes(group[0]["binding"])
            bundle = _bundle(transport, binding)
            payload = payload_bytes(bundle)
            if not tamper_rejected:
                mutated = payload[:-1] + bytes([payload[-1] ^ 1])
                try:
                    payload_bundle(mutated)
                except BundleError:
                    tamper_rejected = True
                else:
                    raise AssertionError("a one-byte bundle mutation was accepted")
            label = _label(group[0])
            source_view, source_omitted = _submit_source(source_harness, root, label, transport)
            bundle_view, bundle_omitted = _submit_bundle(bundle_harness, label, bundle)
            omitted.update(source_omitted)
            omitted.update(bundle_omitted)
            public_rows = []
            for row in group:
                copied = {key: value for key, value in row.items() if key != "binding"}
                public_rows.append(copied)
            protocol.append(
                {
                    "label": label,
                    "instance_keys": [row["instance_key"] for row in group],
                    "provider": group[0]["provider"],
                    "variant_key": group[0]["variant_key"],
                    "credits_variant": group[0]["credits_variant"],
                    "candidate_sha256": group[0]["candidate_sha256"],
                    "submitted_sha256": _sha256(transport),
                    "source": source_view,
                    "bundle": bundle_view,
                    "dispatches": public_rows,
                }
            )
    if not tamper_rejected:
        raise AssertionError("bundle tamper was not checked")
    report = {
        "schema_id": SCHEMA_ID,
        "simulated": True,
        "publication_allowed": False,
        "npu_p0": {
            "status": "blocked",
            "real_device_available": False,
            "real_toolchain_validated": False,
            "correctness_on_device_proven": False,
            "performance_on_device_proven": False,
            "reason": "offline shape-portfolio protocol simulation cannot resolve NPU-P0 or publish NPU evidence",
        },
        "portfolio": {
            "op_name": world["contract"].op_name,
            "target_id": world["target"].id,
            "domain_plan_digest": world["plan"].digest,
            "portfolio_digest": world["portfolio"].digest,
            "freeze_digest": world["freeze"].digest,
            "validation_plan_digest": world["validation"].digest,
            "tune_shards": list(world["validation"].tune_shards),
            "unseen_count": len(world["validation"].unseen_instance_keys),
            "required_unseen_count": len(world["validation"].required_instance_keys),
            "candidate_view": view,
        },
        "protocol": {
            "source_api_contract_id": SOURCE_API,
            "source_evidence_contract_id": SOURCE_EVIDENCE,
            "bundle_api_contract_id": BUNDLE_API,
            "bundle_evidence_contract_id": BUNDLE_EVIDENCE,
            "fixture_task": V1_TASK,
            "fixture_boundary": (
                "preserved-v1 fake-NPU fixture identity; not the portfolio task and not an NPU execution of its kernel"
            ),
            "subjects": protocol,
        },
        "manifest": {
            "status": manifest.status,
            "activation_generation": manifest.activation_generation,
            "derived_from_confirmations": False,
            "dispatch_manifest_digest": manifest.digest,
            "stable_confirmation_digest_is_fresh_confirmation": False,
            "document": manifest.document(),
        },
        "negatives": {**closed, **negatives, "bundle_tamper_rejected": tamper_rejected},
        "omitted_protocol_fields": sorted(omitted),
        "checks": [
            "finite-domain exact dispatch selects one provider or the declared fallback",
            "fallback subjects do not credit a generated variant",
            "selected bindings are submitted through the existing source and bundle contracts",
            "receipts bind the submitted bytes and stay simulated",
            "candidate view has tune keys only",
            "document-only manifest cannot be activated",
            "forged manifest digest, fallback credit and foreign variant id are rejected",
            "one-byte bundle mutation is rejected",
            "no device correctness, timing, speedup or publication claim is recorded",
        ],
    }
    _forbid(report)
    if report["npu_p0"]["status"] != "blocked" or report["publication_allowed"] is not False:
        raise AssertionError("P4 evidence left the simulation boundary")
    return report


def encode_report(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    report = simulate()
    encoded = encode_report(report)
    if args.replay:
        expected = EVIDENCE_PATH.read_text(encoding="utf-8")
        if encoded != expected:
            raise SystemExit(f"simulation replay differs from {EVIDENCE_PATH}")
        print(
            "OK: P4 shape-portfolio NPU simulation replay matches; "
            "simulated=true, publication_allowed=false, NPU-P0=blocked"
        )
        return
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
