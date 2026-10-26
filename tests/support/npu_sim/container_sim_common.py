"""Shared fixtures for the real-VM container execution acceptance.

Test-only.  This module is never imported by a production composition root; it
builds the *production* JobEngine + DockerAttemptSupervisor with a CPU-only fake
worker backend so the container lifecycle can be exercised on a real Ubuntu VM
without a GPU, NPU, model or provider credential.
"""

from __future__ import annotations

import json
import secrets
import sys
from pathlib import Path

from tcenv.remote.container_supervisor import ContainerSupervisorConfig, DockerAttemptSupervisor
from tcenv.remote.identity import ServiceIdentity
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.job_types import ExpectedEvaluator, ExpectedIdentities, ExternalEvaluationRequest
from tcenv.remote.contracts import API_CONTRACT_ID, EVIDENCE_CONTRACT_ID

SITE = "sim-container-site"
DEVICE = "sim-container-device"
TARGET = "sim-container-target"
TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
CALIBRATION_DIGEST = "c" * 64
TOKEN = "container-sim-grant-token"
WORKER_MODULE = "tests.support.npu_sim.fake_container_worker"
WORKER_PYTHON = "python3"
IMAGE = "taic-container-worker-sim:local"


def image_digest(image: str = IMAGE, *, docker: str = "docker") -> str:
    import subprocess

    completed = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"worker image {image!r} is unavailable: {completed.stderr.strip()}")
    return completed.stdout.strip().removeprefix("sha256:")


def worker_build(digest: str) -> str:
    return f"container-sim+img:{digest}"


def identity(digest: str, *, measurement_kind: str = "simulated") -> ServiceIdentity:
    return ServiceIdentity(
        site_id=SITE, device_id=DEVICE, backend="sim-cpu", backend_version="1",
        target_id=TARGET, worker_build=worker_build(digest),
        task_pack_digest=TASK_PACK_DIGEST, calibration_digest=CALIBRATION_DIGEST,
        task_fingerprints=frozenset({TASK_FINGERPRINT}),
        task_bindings={TASK_FINGERPRINT: ("relu", "fp16", "small")},
        measurement_kind=measurement_kind, publication_allowed=False,
    )


def expected_identities(value: ServiceIdentity) -> ExpectedIdentities:
    return ExpectedIdentities(ExpectedEvaluator(
        site_id=value.site_id, device_id=value.device_id, target_id=value.target_id,
        worker_build=value.worker_build, task_pack_digest=value.task_pack_digest,
        evaluator_epoch=value.external_epoch(execution_protocol_version="3"),
    ))


def grant():
    from tcenv.remote.identity import SessionGrant

    return SessionGrant(
        token=TOKEN, run_id="container-sim-run", session_id="container-sim-session",
        target_id=TARGET, allowed_kinds=frozenset({"formal_eval", "selfcheck"}),
        allowed_modes=frozenset(), expires_at=None, max_timed=20, max_gpu_s=120.0,
        subject="container-sim-subject",
    )


def request(request_id: str, *, source: bytes = b"OK", candidate_timeout_ms: int = 30_000,
            kind: str = "formal_eval") -> ExternalEvaluationRequest:
    import hashlib

    return ExternalEvaluationRequest(
        kind=kind, request_id=request_id, run_id="container-sim-run",
        session_id="container-sim-session", task="relu", dtype="fp16", size="small",
        target_id=TARGET, task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(source).hexdigest(), source=source,
        mode="quick", seed="3", reps=2, queue_timeout_ms=30_000,
        candidate_timeout_ms=candidate_timeout_ms, cases=None,
    )


def supervisor_config(digest: str, *, docker: str = "docker", image: str = IMAGE) -> ContainerSupervisorConfig:
    return ContainerSupervisorConfig(
        image=image, image_digest=digest, docker=docker,
        env={
            "HOME": "/tmp", "PYTHONPATH": "/opt/taic",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            # Deterministic child-process token for the timeout cgroup check.
            "TCENV_SIM_CHILD_MARKER": "taic-container-sim-child",
        },
        kill_timeout_s=15.0,
    )


def supervisor(digest: str, *, docker: str = "docker", image: str = IMAGE) -> DockerAttemptSupervisor:
    return DockerAttemptSupervisor(supervisor_config(digest, docker=docker, image=image))


def engine(
    state_dir: Path, digest: str, *, docker: str = "docker", image: str = IMAGE,
    signing_key: bytes | None = None, signing_key_id: str = "",
) -> JobEngine:
    python = WORKER_PYTHON
    return JobEngine(
        identity=identity(digest), grants=[grant()],
        worker_command=[python, "-m", "tcenv.remote.legacy_worker_bridge"],
        worker_env={
            "TCENV_LEGACY_WORKER_COMMAND": json.dumps([python, "-m", WORKER_MODULE]),
        },
        supervisor=supervisor(digest, docker=docker, image=image),
        state_dir=state_dir, signing_key=signing_key, signing_key_id=signing_key_id,
        worker_cwd=Path(__file__).resolve().parents[3],
    )


def capabilities(engine_obj: JobEngine, *, nonce: str = ""):
    expected = engine_obj.expected_identities
    # Nonces are one-shot per grant; a fixed value would be refused by a
    # second exchange (including after a restart over the same durable store).
    nonce = nonce or secrets.token_hex(8)
    return engine_obj.get_capabilities(
        grant_token=TOKEN, client_nonce=nonce,
        expected=expected.evaluator.to_dict(),
        api_contract_ids=[API_CONTRACT_ID],
        evidence_contract_ids=[EVIDENCE_CONTRACT_ID],
    )


def submit(engine_obj: JobEngine, request_obj: ExternalEvaluationRequest):
    return engine_obj.submit(
        request=request_obj, expected=engine_obj.expected_identities,
        capability_id=capabilities(engine_obj).capability_id, grant_token=TOKEN,
    )


def get(engine_obj: JobEngine, request_obj: ExternalEvaluationRequest):
    return engine_obj.get_job(
        request_id=request_obj.request_id, request_digest=request_obj.request_digest,
        expected=engine_obj.expected_identities, grant_token=TOKEN,
    )


assert sys.version_info >= (3, 12)
