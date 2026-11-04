"""Production composition root for the external evaluator gRPC frontend.

This entry point is deliberately strict: TLS/mTLS material, a 0600 grant file, a
0600 signing keyring, a verified read-only task pack and the production worker
module.  Simulation backends and test workers are not reachable here; Phase 2D2
uses a separate test composition root under ``tests/support``.

Run a deployment self-check (no listener):

    python -m tcenv.remote.grpc_service --check ...same files...

Run the frontend:

    python -m tcenv.remote.grpc_service --listen 0.0.0.0:7443 ...
"""

from __future__ import annotations

import argparse
import logging
import json
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .container_supervisor import (
    ContainerSupervisorConfig,
    DockerAttemptSupervisor,
    container_worker_build,
    normalize_image_digest,
)
from . import contracts
from .grpc_server import DEFAULT_MAX_MESSAGE_BYTES, DEFAULT_MAX_WATCH_STREAMS, GrpcServerConfig, RunningServer, build_server
from .grants import load_grants_file
from .identity import ServiceIdentity
from .job_engine import JobEngine
from .keyring import load_keyring_file
from .revocation import load_revocation_file
from .task_pack import TaskPack, production_fingerprint_provider, load_task_pack

LOGGER = logging.getLogger("tcenv.remote.grpc_service")
_HEX64 = set("0123456789abcdef")


@dataclass(frozen=True)
class ProductionStack:
    engine: JobEngine
    server: RunningServer
    identity: ServiceIdentity
    task_pack: TaskPack

    def stop(self, *, grace_s: float = 5.0) -> None:
        """Stop ingress, then wait for a verified engine shutdown.

        ``JobEngine.close`` reports whether the durable store really closed.  A
        production shutdown must not return while the engine's finalizer is
        still running: that would let the next frontend overlap a store writer
        this process still owns.  Fail visibly instead.
        """

        server_error: BaseException | None = None
        try:
            self.server.stop(grace_s=grace_s)
        except BaseException as exc:  # ingress failures must not skip the engine
            server_error = exc
        wait_s = max(1.0, float(grace_s) + 1.0)
        verified = self.engine.close(wait=wait_s)
        error = self.engine.close_error
        problems: list[str] = []
        if server_error is not None:
            problems.append(f"ingress stop failed: {server_error}")
        if not verified:
            detail = "" if error is None else f": {error}"
            problems.append(
                f"job engine did not finish a verified shutdown within {wait_s:.1f}s{detail}"
            )
        if problems:
            raise TimeoutError("; ".join(problems))


def _digest(value: str, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX64 for character in value):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _task_pack(
    args: argparse.Namespace,
    *,
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> TaskPack:
    provider = fingerprint_provider or production_fingerprint_provider(args.backend)
    pack = load_task_pack(
        args.task_pack_root,
        expected_digest=args.expected_task_pack_digest or "",
        fingerprint_provider=provider,
    )
    if pack.backend != args.backend:
        raise ValueError(f"task pack backend {pack.backend!r} differs from --backend {args.backend!r}")
    if pack.target_id != args.target_id:
        raise ValueError(f"task pack target {pack.target_id!r} differs from --target-id {args.target_id!r}")
    return pack


def _worker_execution(args: argparse.Namespace) -> str:
    return getattr(args, "worker_execution", "container")


def build_supervisor(args: argparse.Namespace):
    """Build the execution supervisor this deployment selected (no fallback)."""

    mode = _worker_execution(args)
    if mode == "container":
        if not getattr(args, "worker_image", ""):
            raise ValueError("container execution requires --worker-image")
        if not getattr(args, "worker_image_digest", ""):
            raise ValueError("container execution requires --worker-image-digest")
        digest = normalize_image_digest(args.worker_image_digest)
        if not args.worker_build.endswith(f"+img:{digest}"):
            raise ValueError(
                "container execution requires --worker-build to end with "
                f"+img:{digest} (use {container_worker_build('build-name', digest)!r})"
            )
        if not getattr(args, "worker_python", "python3"):
            raise ValueError("container execution requires --worker-python")
        config = ContainerSupervisorConfig(
            image=args.worker_image,
            image_digest=digest,
            docker=getattr(args, "docker_binary", "docker") or "docker",
            docker_host=getattr(args, "docker_host", "") or "",
            runtime=getattr(args, "worker_runtime", "") or "",
            runtime_profile=getattr(args, "worker_runtime_profile", "runc") or "runc",
            user=getattr(args, "worker_user", "65534:65534"),
            pids_limit=getattr(args, "worker_pids_limit", 256),
            memory=getattr(args, "worker_memory", "2g"),
            cpus=getattr(args, "worker_cpus", "2"),
            scratch_size=getattr(args, "worker_scratch_size", "512m"),
            tmp_size=getattr(args, "worker_tmp_size", "64m"),
            env={
                "HOME": "/tmp",
                "PYTHONPATH": getattr(args, "worker_image_pythonpath", "/opt/taic") or "/opt/taic",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            },
            cli_timeout_s=float(getattr(args, "container_cli_timeout_s", 30.0)),
            kill_timeout_s=float(getattr(args, "container_kill_timeout_s", 15.0)),
        )
        return DockerAttemptSupervisor(config)
    if mode == "process":
        # Explicit, documented local-development path.  The production external
        # deployment must not silently fall back to it.
        if not getattr(args, "allow_process_worker", False):
            raise ValueError(
                "process execution is for local development only; "
                "pass --allow-process-worker with --worker-execution process"
            )
        if "+img:" in str(getattr(args, "worker_build", "")):
            # A process worker cannot have run a pinned image; signing such a
            # worker_build would claim an image that never executed.
            raise ValueError(
                "process execution cannot claim an image-pinned worker_build "
                "(+img:<digest>); use --worker-execution container"
            )
        return None
    raise ValueError("--worker-execution must be container or process")


def verify_deployment(
    args: argparse.Namespace,
    *,
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> tuple[TaskPack, ServiceIdentity, list, object, object]:
    """Validate every deployment input without opening a listener or state DB."""

    if not args.worker_module or args.worker_module.startswith("tests.") or args.worker_module.startswith("tests/"):
        raise ValueError("the production entry point cannot run a test/composition worker module")
    try:
        dialect = contracts.dialect_for(args.api_contract_id, args.evidence_contract_id)
    except contracts.ContractError as exc:
        raise ValueError(f"unsupported API/evidence contract pair: {exc}") from exc
    if dialect.npu_source:
        # P0 ships the source-only NPU contract and its fake vendor only in the
        # test composition root.  A production NPU adapter/DeviceController has
        # not been implemented or validated, so this entry must not pretend it
        # can serve one.
        raise ValueError("the P0 NPU source contract is test-only; no production NPU adapter is installed")
    if args.target_id != "apple-m4-max" or args.backend != "metal":
        raise ValueError(
            "the legacy source contract is only served for the existing Metal target "
            "apple-m4-max; use the versioned NPU contract for other targets"
        )
    supervisor = build_supervisor(args)
    pack = _task_pack(args, fingerprint_provider=fingerprint_provider)
    identity = ServiceIdentity(
        site_id=args.site_id,
        device_id=args.device_id,
        backend=args.backend,
        backend_version=args.backend_version,
        target_id=args.target_id,
        worker_build=args.worker_build,
        task_pack_digest=pack.digest,
        calibration_digest=_digest(args.calibration_digest, "--calibration-digest"),
        task_fingerprints=pack.fingerprints,
        task_bindings=pack.fingerprint_bindings,
        measurement_kind=args.measurement_kind,
        publication_allowed=bool(args.publication_allowed),
    )
    grants = load_grants_file(args.grants_file)
    if any(grant.target_id != args.target_id for grant in grants):
        raise ValueError("every grant must target the deployed --target-id")
    keyring = load_keyring_file(args.signing_keyring_file)
    keyring.require_active()
    _revoked_certificates(args)
    for path in (args.certificate_file, args.private_key_file, args.client_ca_file):
        if not Path(path).is_file():
            raise ValueError(f"TLS material is missing: {path}")
    return pack, identity, grants, keyring, supervisor


def _revoked_certificates(args: argparse.Namespace) -> frozenset[str]:
    if getattr(args, "revoked_client_certificates", None) is None:
        return frozenset()
    return load_revocation_file(args.revoked_client_certificates)


def build_stack(
    args: argparse.Namespace,
    *,
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> ProductionStack:
    pack, identity, grants, keyring, supervisor = verify_deployment(
        args, fingerprint_provider=fingerprint_provider,
    )
    active_key_id, active_key = keyring.require_active()
    if _worker_execution(args) == "container":
        python = args.worker_python
        worker_command = [python, "-m", "tcenv.remote.job_worker_bridge"]
        worker_env = {"TCENV_JOB_WORKER_COMMAND": json.dumps([python, "-m", args.worker_module])}
    else:
        worker_command = [sys.executable, "-m", "tcenv.remote.job_worker_bridge"]
        worker_env = {
            "TCENV_JOB_WORKER_COMMAND": json.dumps([sys.executable, "-m", args.worker_module]),
        }
    engine = JobEngine(
        identity=identity,
        grants=grants,
        worker_command=worker_command,
        worker_env=worker_env,
        supervisor=supervisor,
        state_dir=Path(args.state_dir),
        contract=contracts.dialect_for(args.api_contract_id, args.evidence_contract_id),
        signing_key=active_key,
        signing_key_id=active_key_id,
        signing_keyring=keyring.trusted,
        selfcheck_max_timed=args.selfcheck_max_timed,
        selfcheck_max_gpu_s=args.selfcheck_max_gpu_s,
        clock_skew_ms=args.clock_skew_ms,
    )
    try:
        server = build_server(
            engine,
            config=GrpcServerConfig(
                address=args.listen,
                max_workers=args.max_workers,
                max_message_bytes=args.max_message_bytes,
                max_watch_streams=args.max_watch_streams,
                watch_poll_s=args.watch_poll_s,
                require_client_identity=True,
                allow_insecure=False,
                certificate_file=Path(args.certificate_file),
                private_key_file=Path(args.private_key_file),
                client_ca_file=Path(args.client_ca_file),
                auth_max_failures=args.auth_max_failures,
                auth_window_s=args.auth_window_s,
                revoked_client_certificates=_revoked_certificates(args),
            ),
        )
    except Exception:
        engine.close()
        raise
    return ProductionStack(engine=engine, server=server, identity=identity, task_pack=pack)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m tcenv.remote.grpc_service")
    parser.add_argument("--listen", default="127.0.0.1:7443")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--device-id", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--backend-version", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument(
        "--api-contract-id", default=contracts.API_CONTRACT_ID,
        help="external API contract; the legacy source contract is Metal-only",
    )
    parser.add_argument(
        "--evidence-contract-id", default=contracts.EVIDENCE_CONTRACT_ID,
        help="external evidence contract paired with --api-contract-id",
    )
    parser.add_argument("--worker-build", required=True)
    parser.add_argument("--calibration-digest", required=True)
    parser.add_argument("--task-pack-root", required=True, type=Path)
    parser.add_argument("--expected-task-pack-digest", default="")
    parser.add_argument("--grants-file", required=True, type=Path)
    parser.add_argument("--signing-keyring-file", required=True, type=Path)
    parser.add_argument("--certificate-file", required=True, type=Path)
    parser.add_argument("--private-key-file", required=True, type=Path)
    parser.add_argument("--client-ca-file", required=True, type=Path)
    parser.add_argument(
        "--revoked-client-certificates", type=Path, default=None,
        help="0600 JSON list of revoked client leaf-certificate SHA-256 fingerprints",
    )
    parser.add_argument("--worker-module", default="tcenv.remote.worker")
    parser.add_argument(
        "--worker-execution", choices=["container", "process"], default="container",
        help="execution plane; the production external evaluator uses disposable containers",
    )
    parser.add_argument(
        "--allow-process-worker", action="store_true",
        help="local development only: allow one process worker instead of a container",
    )
    parser.add_argument("--worker-image", default="")
    parser.add_argument("--worker-image-digest", default="", help="pinned sha256 image id")
    parser.add_argument("--worker-image-pythonpath", default="/opt/taic")
    parser.add_argument("--worker-python", default="python3")
    parser.add_argument("--worker-runtime", default="", help="optional OCI runtime, e.g. runsc/kata-runtime")
    parser.add_argument(
        "--worker-runtime-profile", choices=["runc", "gvisor", "kata"], default="runc",
        help="recorded runtime profile; gVisor/Kata are not implemented or validated yet",
    )
    parser.add_argument("--worker-user", default="65534:65534")
    parser.add_argument("--worker-pids-limit", type=int, default=256)
    parser.add_argument("--worker-memory", default="2g")
    parser.add_argument("--worker-cpus", default="2")
    parser.add_argument("--worker-scratch-size", default="512m")
    parser.add_argument("--worker-tmp-size", default="64m")
    parser.add_argument("--docker-binary", default="docker")
    parser.add_argument("--docker-host", default="")
    parser.add_argument("--container-cli-timeout-s", type=float, default=30.0)
    parser.add_argument("--container-kill-timeout-s", type=float, default=15.0)
    parser.add_argument("--measurement-kind", choices=["measured", "simulated"], default="measured")
    parser.add_argument("--publication-allowed", action="store_true")
    parser.add_argument("--selfcheck-max-timed", type=int, default=20)
    parser.add_argument("--selfcheck-max-gpu-s", type=float, default=120.0)
    parser.add_argument("--clock-skew-ms", type=int, default=5_000)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--max-message-bytes", type=int, default=DEFAULT_MAX_MESSAGE_BYTES)
    parser.add_argument("--max-watch-streams", type=int, default=DEFAULT_MAX_WATCH_STREAMS)
    parser.add_argument("--watch-poll-s", type=float, default=0.05)
    parser.add_argument("--auth-max-failures", type=int, default=64)
    parser.add_argument("--auth-window-s", type=float, default=60.0)
    parser.add_argument("--check", action="store_true", help="verify deployment files and exit without listening")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parse_args(argv)
    if args.measurement_kind != "measured":
        raise SystemExit("the production gRPC entry point refuses simulated measurement identity")
    if args.check:
        pack, identity, grants, keyring, supervisor = verify_deployment(args)
        active_key_id, _key = keyring.require_active()
        execution = "container" if supervisor is not None else "process"
        image = (
            f" image={supervisor.config.image} image_digest={supervisor.image_digest}"
            f" runtime_profile={supervisor.runtime_profile}"
            if supervisor is not None else ""
        )
        print(
            f"ok site={identity.site_id} target={identity.target_id} backend={identity.backend} "
            f"task_pack_digest={pack.digest} fingerprints={len(pack.fingerprints)} "
            f"grants={len(grants)} active_key_id={active_key_id} "
            f"execution={execution}{image} "
            f"revoked_client_certificates={len(_revoked_certificates(args))}"
        )
        return 0
    stack = build_stack(args)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_args: stop.set())
    signal.signal(signal.SIGINT, lambda *_args: stop.set())
    LOGGER.info(
        "external evaluator listening on %s site=%s target=%s task_pack=%s",
        stack.server.address, stack.identity.site_id, stack.identity.target_id, stack.task_pack.digest,
    )
    try:
        while not stop.wait(1.0):
            if not stack.server.server.wait_for_termination(0):
                continue
            break
    finally:
        stack.stop(grace_s=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
