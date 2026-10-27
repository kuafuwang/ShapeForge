"""Shared device-free fixtures/helpers for the transport-neutral job engine."""

from __future__ import annotations

import hashlib
import secrets
import sys
import time
from pathlib import Path

from tcenv.remote.identity import ServiceIdentity, SessionGrant
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.job_types import (
    ExpectedEvaluator, ExpectedIdentities, ExternalEvaluationRequest, JobSnapshot,
)

TOKEN = "external-test-grant-token"
OTHER_TOKEN = "external-other-grant-token"
TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
TARGET = "sim-external-test"


def identity() -> ServiceIdentity:
    return ServiceIdentity(
        site_id="sim-external-site", device_id="sim-external-device",
        backend="sim-fake", backend_version="1", target_id=TARGET,
        worker_build="sim-external-build", task_pack_digest=TASK_PACK_DIGEST,
        calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}),
        task_bindings={TASK_FINGERPRINT: ("relu", "fp16", "small")},
        measurement_kind="simulated", publication_allowed=False,
    )


def grant(
    token: str = TOKEN, *, run_id: str = "run-1", session_id: str = "session-1",
    subject: str = "subject-1", kinds=frozenset({"formal_eval", "selfcheck"}),
    modes=frozenset(), expires_at: float | None = None,
    max_timed: int = 20, max_gpu_s: float = 120.0,
) -> SessionGrant:
    return SessionGrant(
        token=token, run_id=run_id, session_id=session_id, target_id=TARGET,
        allowed_kinds=frozenset(kinds), allowed_modes=frozenset(modes),
        expires_at=expires_at, max_timed=max_timed, max_gpu_s=max_gpu_s,
        subject=subject,
    )


def request(
    request_id: str = "request-1", *, source: bytes = b"OK",
    kind: str = "formal_eval", mode: str = "quick", run_id: str = "run-1",
    session_id: str = "session-1", candidate_timeout_ms: int = 3000,
    queue_timeout_ms: int = 3000, reps: int = 2,
    task: str = "relu", dtype: str = "fp16", size: str = "small",
) -> ExternalEvaluationRequest:
    return ExternalEvaluationRequest(
        kind=kind, request_id=request_id, run_id=run_id, session_id=session_id,
        task=task, dtype=dtype, size=size, target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(source).hexdigest(), source=source,
        mode=mode, seed="3", reps=reps, queue_timeout_ms=queue_timeout_ms,
        candidate_timeout_ms=candidate_timeout_ms, cases=None,
    )


def expected_identities(value: ServiceIdentity | None = None) -> ExpectedIdentities:
    value = value or identity()
    return ExpectedIdentities(ExpectedEvaluator(
        site_id=value.site_id, device_id=value.device_id, target_id=value.target_id,
        worker_build=value.worker_build, task_pack_digest=value.task_pack_digest,
        evaluator_epoch=value.external_epoch(execution_protocol_version="3"),
    ))


def store_capabilities(store, *, token: str = TOKEN, nonce: str | None = None):
    expected = expected_identities()
    nonce = secrets.token_hex(16) if nonce is None else nonce.ljust(16, "0")
    return store.get_capabilities(
        grant_token=token, client_nonce=nonce, expected=expected.evaluator.to_dict(),
        expected_identities=expected,
        provenance=identity().external_provenance(execution_protocol_version="3"),
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        required_features=[], supported_features=frozenset(),
    )


def submit_to_store(store, req: ExternalEvaluationRequest, *, token: str = TOKEN) -> JobSnapshot:
    cap = store_capabilities(store, token=token)
    snapshot, created = store.submit(
        request=req, request_digest=req.request_digest,
        expected=expected_identities(),
        provenance=identity().external_provenance(execution_protocol_version="3"),
        capability_id=cap.capability_id, required_features=[], grant_token=token,
    )
    assert created
    return snapshot


def engine(
    state_dir: Path, *, grants: list[SessionGrant] | None = None,
    worker_env: dict[str, str] | None = None, **options,
) -> JobEngine:
    return JobEngine(
        identity=identity(), grants=grants or [grant()],
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_cwd=Path(__file__).resolve().parents[2], state_dir=state_dir,
        worker_env=worker_env, **options,
    )


def capabilities(engine: JobEngine, *, token: str = TOKEN, nonce: str | None = None, valid_for_ms: int = 60_000):
    # Test labels remain readable while meeting the protocol's nonce bound;
    # omitted nonces are fresh, as a real client handshake would be.
    nonce = (secrets.token_hex(16) if nonce is None else nonce).ljust(16, "0")
    expected = engine.expected_identities
    return engine.get_capabilities(
        grant_token=token, client_nonce=nonce,
        expected=expected.evaluator.to_dict(),
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        valid_for_ms=valid_for_ms,
    )


def submit(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = TOKEN, nonce: str | None = None) -> JobSnapshot:
    cap = capabilities(engine, token=token, nonce=nonce)
    return engine.submit(
        request=req, expected=engine.expected_identities,
        capability_id=cap.capability_id, grant_token=token,
    )


def get(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = TOKEN) -> JobSnapshot:
    return engine.get_job(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, grant_token=token,
    )


def wait_state(
    engine: JobEngine, req: ExternalEvaluationRequest, states: set[str], *,
    token: str = TOKEN, timeout: float = 8.0,
) -> JobSnapshot:
    deadline = time.monotonic() + timeout
    last = get(engine, req, token=token)
    while last.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        last = get(engine, req, token=token)
    assert last.state in states, f"timed out waiting for {states}; last={last.state}, failure={last.failure}"
    return last


def retry(engine: JobEngine, req: ExternalEvaluationRequest, attempt_id: str, *, token: str = TOKEN, nonce: str | None = None) -> JobSnapshot:
    cap = capabilities(engine, token=token, nonce=nonce)
    return engine.retry(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, interrupted_attempt_id=attempt_id,
        capability_id=cap.capability_id, grant_token=token,
    )


# -- P0 NPU source contract fixtures ---------------------------------------
NPU_TARGET = "sim-npu-source"
NPU_SITE = "sim-npu-site"
NPU_DEVICE = "sim-npu-device"
NPU_WORKER_BUILD = "fake-vendor-a-worker-1"
NPU_TASK_FINGERPRINT = "d" * 64
NPU_TOKEN = "external-npu-grant-token"
NPU_FEATURES = (
    "durable-jobs", "attempt-receipts", "progress",
    "npu-source-artifact", "kernel-simt", "device-event-timing",
)


def npu_identity() -> ServiceIdentity:
    return ServiceIdentity(
        site_id=NPU_SITE, device_id=NPU_DEVICE, backend="vendor-a",
        backend_version="1", target_id=NPU_TARGET,
        worker_build=NPU_WORKER_BUILD, task_pack_digest=TASK_PACK_DIGEST,
        calibration_digest="c" * 64,
        task_fingerprints=frozenset({NPU_TASK_FINGERPRINT}),
        task_bindings={NPU_TASK_FINGERPRINT: ("relu", "fp16", "small")},
        measurement_kind="simulated", publication_allowed=False,
    )


def npu_grant(
    token: str = NPU_TOKEN, *, run_id: str = "npu-run-1",
    session_id: str = "npu-session-1", subject: str = "npu-subject-1",
    expires_at: float | None = None,
) -> SessionGrant:
    return SessionGrant(
        token=token, run_id=run_id, session_id=session_id, target_id=NPU_TARGET,
        allowed_kinds=frozenset({"formal_eval", "selfcheck"}),
        allowed_modes=frozenset({"quick", "full"}),
        expires_at=expires_at, max_timed=20, max_gpu_s=120.0, subject=subject,
    )


def npu_request(
    request_id: str = "npu-request-1", *, source: bytes = b"npu-kernel relu_kernel\n",
    run_id: str = "npu-run-1", session_id: str = "npu-session-1",
    candidate_timeout_ms: int = 3000, queue_timeout_ms: int = 3000,
    reps: int = 2, mode: str = "quick", kind: str = "formal_eval",
    generation: str = "1", blob=None,
) -> ExternalEvaluationRequest:
    from tcenv.remote.job_types import (
        CandidateDigest, DeviceGroupBinding, NpuExecutionSpec, NpuSourceRequest,
    )
    from tests.support.vendor_sim.vendor_a import VENDOR_A_GROUP

    from tests.support.vendor_sim.vendor_a import build_execution

    group = VENDOR_A_GROUP
    execution = build_execution(generation) if generation != "1" else NpuExecutionSpec(device_group=group)
    if blob is not None:
        from tcenv.remote import contracts
        from tcenv.remote.job_types import BlobDescriptor

        descriptor = blob if isinstance(blob, BlobDescriptor) else BlobDescriptor.from_dict(blob)
        npu = NpuSourceRequest(
            candidate_digest=CandidateDigest(
                contracts.BLOB_DIGEST_ALGORITHM, descriptor.sha256,
                allowed_algorithms=(contracts.BLOB_DIGEST_ALGORITHM,),
            ),
            execution=execution,
            blob=descriptor,
        )
        # A blob candidate is referenced, never embedded: no source bytes travel
        # in the control-plane request.
        source = b""
    else:
        npu = NpuSourceRequest(
            candidate_digest=CandidateDigest("sha256", hashlib.sha256(source).hexdigest()),
            execution=execution,
        )
    return ExternalEvaluationRequest(
        kind=kind, request_id=request_id, run_id=run_id,
        session_id=session_id, task="relu", dtype="fp16", size="small",
        target_id=NPU_TARGET, task_fingerprint=NPU_TASK_FINGERPRINT,
        candidate_sha256="", source=source, mode=mode, seed="3", reps=reps,
        queue_timeout_ms=queue_timeout_ms, candidate_timeout_ms=candidate_timeout_ms,
        cases=None, npu=npu,
    )


def npu_expected_identities(value: ServiceIdentity | None = None, *, contract=None) -> ExpectedIdentities:
    value = value or npu_identity()
    from tcenv.remote import contracts

    contract = contract or contracts.NPU_SOURCE_DIALECT
    return ExpectedIdentities(ExpectedEvaluator(
        site_id=value.site_id, device_id=value.device_id, target_id=value.target_id,
        worker_build=value.worker_build, task_pack_digest=value.task_pack_digest,
        evaluator_epoch=value.external_epoch(execution_protocol_version="3"),
    ), contract.api_contract_id, contract.evidence_contract_id)


def npu_features(contract=None) -> tuple[str, ...]:
    """Return the features the P0/P0b fake vendor advertises."""

    from tcenv.remote import contracts

    if contract is not None and contract.npu_blob:
        return (*NPU_FEATURES, "blob-artifact")
    return NPU_FEATURES


def npu_engine(
    state_dir: Path, *, ledger_path: Path, controller,
    grants: list[SessionGrant] | None = None,
    worker_env: dict[str, str] | None = None,
    generation: str = "1", contract=None, blob_store=None, **options,
) -> JobEngine:
    from tcenv.remote import contracts
    from tests.support.vendor_sim.vendor_a import build_execution

    contract = contract or contracts.NPU_SOURCE_DIALECT
    env = dict(worker_env or {})
    env["TCENV_VENDOR_LEDGER"] = str(ledger_path)
    env["TCENV_VENDOR_A_GENERATION"] = generation
    return JobEngine(
        identity=npu_identity(), grants=grants or [npu_grant()],
        worker_command=[sys.executable, "-m", "tests.support.vendor_sim.worker_a"],
        worker_cwd=Path(__file__).resolve().parents[2],
        state_dir=state_dir, worker_env=env,
        supported_features=frozenset(npu_features(contract)),
        supported_modes=frozenset({"quick", "full"}),
        contract=contract,
        npu_execution=build_execution(generation),
        device_controller=controller,
        blob_store=blob_store,
        **options,
    )


def npu_capabilities(
    engine: JobEngine, *, token: str = NPU_TOKEN, nonce: str | None = None,
    required_features: tuple[str, ...] | None = None,
):
    required_features = tuple(required_features if required_features is not None else npu_features(engine.contract))
    nonce = (secrets.token_hex(16) if nonce is None else nonce).ljust(16, "0")
    expected = engine.expected_identities
    return engine.get_capabilities(
        grant_token=token, client_nonce=nonce,
        expected=expected.evaluator.to_dict(),
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        required_features=list(required_features),
    )


def npu_submit(
    engine: JobEngine, req: ExternalEvaluationRequest, *,
    token: str = NPU_TOKEN, nonce: str | None = None,
) -> JobSnapshot:
    cap = npu_capabilities(engine, token=token, nonce=nonce)
    return engine.submit(
        request=req, expected=engine.expected_identities,
        capability_id=cap.capability_id, required_features=list(npu_features(engine.contract)),
        grant_token=token,
    )


def npu_get(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = NPU_TOKEN) -> JobSnapshot:
    return engine.get_job(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, grant_token=token,
    )


def npu_retry(
    engine: JobEngine, req: ExternalEvaluationRequest, attempt_id: str,
    *, token: str = NPU_TOKEN, nonce: str | None = None,
) -> JobSnapshot:
    cap = npu_capabilities(engine, token=token, nonce=nonce)
    return engine.retry(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, interrupted_attempt_id=attempt_id,
        capability_id=cap.capability_id, required_features=list(NPU_FEATURES),
        grant_token=token,
    )


# -- P1 NPU bundle contract fixtures ---------------------------------------
NPU_GRAPH_TARGET = "sim-npu-graph"
NPU_GRAPH_WORKER_BUILD = "fake-vendor-b-worker-1"
NPU_BUNDLE_TARGET = "sim-npu-bundle-source"
NPU_BUNDLE_WORKER_BUILD = "fake-vendor-a-worker-1"
#: Features advertised by the P1 bundle-capable fake vendors.
NPU_BUNDLE_FEATURES = (
    "durable-jobs", "attempt-receipts", "progress", "artifact-bundle",
    "device-event-timing",
)


def npu_bundle_features(*, blob: bool = False) -> tuple[str, ...]:
    return (*NPU_BUNDLE_FEATURES, "blob-artifact") if blob else NPU_BUNDLE_FEATURES


def npu_bundle_identity(vendor: str = "vendor-b") -> ServiceIdentity:
    """Identity of one P1 fake vendor frontend (vendor A source or vendor B graph)."""

    if vendor == "vendor-a":
        return ServiceIdentity(
            site_id=NPU_SITE, device_id=NPU_DEVICE, backend="vendor-a",
            backend_version="1", target_id=NPU_BUNDLE_TARGET,
            worker_build=NPU_BUNDLE_WORKER_BUILD, task_pack_digest=TASK_PACK_DIGEST,
            calibration_digest="c" * 64,
            task_fingerprints=frozenset({NPU_TASK_FINGERPRINT}),
            task_bindings={NPU_TASK_FINGERPRINT: ("relu", "fp16", "small")},
            measurement_kind="simulated", publication_allowed=False,
        )
    if vendor != "vendor-b":
        raise ValueError(f"unknown fake NPU vendor {vendor!r}")
    return ServiceIdentity(
        site_id=NPU_SITE, device_id="sim-npu-graph-device", backend="vendor-b",
        backend_version="1", target_id=NPU_GRAPH_TARGET,
        worker_build=NPU_GRAPH_WORKER_BUILD, task_pack_digest=TASK_PACK_DIGEST,
        calibration_digest="c" * 64,
        task_fingerprints=frozenset({NPU_TASK_FINGERPRINT}),
        task_bindings={NPU_TASK_FINGERPRINT: ("relu", "fp16", "small")},
        measurement_kind="simulated", publication_allowed=False,
    )


def npu_bundle_execution(
    vendor: str = "vendor-b", *, generation: str = "1", resources=None,
) -> "NpuExecutionSpec":
    """The version 4 execution binding one P1 fake vendor serves."""

    from tcenv.remote.job_types import NpuExecutionSpec, NpuResources

    if vendor == "vendor-b":
        from tests.support.vendor_sim.vendor_b import build_graph_execution

        return build_graph_execution(generation, resources=resources)
    from tests.support.vendor_sim.vendor_a import VENDOR_A_GROUP

    group = (
        VENDOR_A_GROUP if generation == "1"
        else type(VENDOR_A_GROUP)(
            device_group_id=VENDOR_A_GROUP.device_group_id,
            vendor_id=VENDOR_A_GROUP.vendor_id,
            profile_digest=VENDOR_A_GROUP.profile_digest,
            topology_digest=VENDOR_A_GROUP.topology_digest,
            resource_generation=generation,
        )
    )
    return NpuExecutionSpec(
        device_group=group, spec_version=4, task_family="kernel",
        resources=resources or NpuResources(),
    )


def npu_bundle_request(
    request_id: str = "npu-bundle-request-1", *, vendor: str = "vendor-b",
    bundle=None, source: bytes = b"npu-kernel relu_kernel\n",
    run_id: str = "npu-run-1", session_id: str = "npu-session-1",
    candidate_timeout_ms: int = 3000, queue_timeout_ms: int = 3000,
    reps: int = 2, mode: str = "quick", kind: str = "formal_eval",
    generation: str = "1", resources=None, execution=None,
    contract=None,
) -> ExternalEvaluationRequest:
    """Build a version 4 request: vendor B's bundle or vendor A's inline source."""

    from tcenv.remote import contracts
    from tcenv.remote.job_types import CandidateDigest, NpuSourceRequest

    contract = contract or contracts.NPU_BUNDLE_DIALECT
    execution = execution or npu_bundle_execution(vendor, generation=generation, resources=resources)
    target = NPU_GRAPH_TARGET if vendor != "vendor-a" else NPU_BUNDLE_TARGET
    if vendor == "vendor-b":
        from tests.support.vendor_sim.vendor_b import build_graph_bundle

        artifact = bundle if bundle is not None else build_graph_bundle()
        npu = NpuSourceRequest(
            candidate_digest=CandidateDigest(
                contracts.BUNDLE_DIGEST_ALGORITHM, artifact.artifact_set_digest,
                allowed_algorithms=(contracts.BUNDLE_DIGEST_ALGORITHM,),
            ),
            execution=execution,
            bundle=artifact,
        )
        source = b""
    else:
        npu = NpuSourceRequest(
            candidate_digest=CandidateDigest("sha256", hashlib.sha256(source).hexdigest()),
            execution=execution,
        )
        source = source
    return ExternalEvaluationRequest(
        kind=kind, request_id=request_id, run_id=run_id,
        session_id=session_id, task="relu", dtype="fp16", size="small",
        target_id=target, task_fingerprint=NPU_TASK_FINGERPRINT,
        candidate_sha256="", source=source, mode=mode, seed="3", reps=reps,
        queue_timeout_ms=queue_timeout_ms, candidate_timeout_ms=candidate_timeout_ms,
        cases=None, npu=npu, contract_dialect=contract,
    )


def npu_bundle_grant(
    vendor: str = "vendor-b", token: str = NPU_TOKEN, *, run_id: str = "npu-run-1",
    session_id: str = "npu-session-1", subject: str = "npu-subject-1",
    expires_at: float | None = None,
) -> SessionGrant:
    target = NPU_GRAPH_TARGET if vendor != "vendor-a" else NPU_BUNDLE_TARGET
    return SessionGrant(
        token=token, run_id=run_id, session_id=session_id, target_id=target,
        allowed_kinds=frozenset({"formal_eval", "selfcheck"}),
        allowed_modes=frozenset({"quick", "full"}),
        expires_at=expires_at, max_timed=20, max_gpu_s=120.0, subject=subject,
    )


def npu_bundle_engine(
    state_dir: Path, *, service_path: Path, controller,
    vendor: str = "vendor-b", grants: list[SessionGrant] | None = None,
    worker_env: dict[str, str] | None = None, generation: str = "1",
    blob_store=None, **options,
) -> JobEngine:
    """One P1 frontend: the bundle contract over fake vendor A or B."""

    from tcenv.remote import contracts

    env = dict(worker_env or {})
    if vendor == "vendor-b":
        env["TCENV_VENDOR_B_SERVICE"] = str(service_path)
        env.setdefault("TCENV_VENDOR_LEDGER", str(service_path))
        env["TCENV_VENDOR_B_GENERATION"] = generation
        worker = [sys.executable, "-m", "tests.support.vendor_sim.worker_b"]
    elif vendor == "vendor-a":
        env["TCENV_VENDOR_LEDGER"] = str(service_path)
        env["TCENV_VENDOR_A_GENERATION"] = generation
        worker = [sys.executable, "-m", "tests.support.vendor_sim.worker_a"]
    else:
        raise ValueError(f"unknown fake NPU vendor {vendor!r}")
    return JobEngine(
        identity=npu_bundle_identity(vendor),
        grants=grants or [npu_bundle_grant(vendor)],
        worker_command=worker,
        worker_cwd=Path(__file__).resolve().parents[2],
        state_dir=state_dir, worker_env=env,
        supported_features=frozenset(npu_bundle_features(blob=blob_store is not None)),
        supported_modes=frozenset({"quick", "full"}),
        contract=contracts.NPU_BUNDLE_DIALECT,
        npu_execution=npu_bundle_execution(vendor, generation=generation),
        device_controller=controller,
        blob_store=blob_store,
        **options,
    )


def npu_bundle_capabilities(engine: JobEngine, *, token: str = NPU_TOKEN, nonce: str | None = None):
    features = tuple(
        npu_bundle_features(blob=engine.blob_store is not None)
    )
    nonce = (secrets.token_hex(16) if nonce is None else nonce).ljust(16, "0")
    expected = engine.expected_identities
    return engine.get_capabilities(
        grant_token=token, client_nonce=nonce, expected=expected.evaluator.to_dict(),
        api_contract_ids=[expected.api_contract_id],
        evidence_contract_ids=[expected.evidence_contract_id],
        required_features=list(features),
    )


def npu_bundle_submit(
    engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = NPU_TOKEN,
) -> JobSnapshot:
    cap = npu_bundle_capabilities(engine, token=token)
    return engine.submit(
        request=req, expected=engine.expected_identities,
        capability_id=cap.capability_id,
        required_features=list(npu_bundle_features(blob=engine.blob_store is not None)),
        grant_token=token,
    )


def npu_bundle_get(engine: JobEngine, req: ExternalEvaluationRequest, *, token: str = NPU_TOKEN) -> JobSnapshot:
    return engine.get_job(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, grant_token=token,
    )


def npu_bundle_retry(
    engine: JobEngine, req: ExternalEvaluationRequest, attempt_id: str,
    *, token: str = NPU_TOKEN, nonce: str | None = None,
) -> JobSnapshot:
    cap = npu_bundle_capabilities(engine, token=token, nonce=nonce)
    return engine.retry(
        request_id=req.request_id, request_digest=req.request_digest,
        expected=engine.expected_identities, interrupted_attempt_id=attempt_id,
        capability_id=cap.capability_id,
        required_features=list(npu_bundle_features(blob=engine.blob_store is not None)),
        grant_token=token,
    )
