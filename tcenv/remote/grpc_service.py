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

from .grpc_server import DEFAULT_MAX_MESSAGE_BYTES, DEFAULT_MAX_WATCH_STREAMS, GrpcServerConfig, RunningServer, build_server
from .grants import load_grants_file
from .identity import ServiceIdentity
from .job_engine import JobEngine
from .keyring import load_keyring_file
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
        self.server.stop(grace_s=grace_s)
        self.engine.close()


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


def verify_deployment(
    args: argparse.Namespace,
    *,
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> tuple[TaskPack, ServiceIdentity, list, object]:
    """Validate every deployment input without opening a listener or state DB."""

    if not args.worker_module or args.worker_module.startswith("tests.") or args.worker_module.startswith("tests/"):
        raise ValueError("the production entry point cannot run a test/composition worker module")
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
        measurement_kind=args.measurement_kind,
        publication_allowed=bool(args.publication_allowed),
    )
    grants = load_grants_file(args.grants_file)
    if any(grant.target_id != args.target_id for grant in grants):
        raise ValueError("every grant must target the deployed --target-id")
    keyring = load_keyring_file(args.signing_keyring_file)
    keyring.require_active()
    for path in (args.certificate_file, args.private_key_file, args.client_ca_file):
        if not Path(path).is_file():
            raise ValueError(f"TLS material is missing: {path}")
    return pack, identity, grants, keyring


def build_stack(
    args: argparse.Namespace,
    *,
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> ProductionStack:
    pack, identity, grants, keyring = verify_deployment(args, fingerprint_provider=fingerprint_provider)
    active_key_id, active_key = keyring.require_active()
    engine = JobEngine(
        identity=identity,
        grants=grants,
        worker_command=[sys.executable, "-m", "tcenv.remote.legacy_worker_bridge"],
        worker_env={"TCENV_LEGACY_WORKER_COMMAND": json.dumps([sys.executable, "-m", args.worker_module])},
        state_dir=Path(args.state_dir),
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
    parser.add_argument("--worker-build", required=True)
    parser.add_argument("--calibration-digest", required=True)
    parser.add_argument("--task-pack-root", required=True, type=Path)
    parser.add_argument("--expected-task-pack-digest", default="")
    parser.add_argument("--grants-file", required=True, type=Path)
    parser.add_argument("--signing-keyring-file", required=True, type=Path)
    parser.add_argument("--certificate-file", required=True, type=Path)
    parser.add_argument("--private-key-file", required=True, type=Path)
    parser.add_argument("--client-ca-file", required=True, type=Path)
    parser.add_argument("--worker-module", default="tcenv.remote.worker")
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
        pack, identity, grants, keyring = verify_deployment(args)
        active_key_id, _key = keyring.require_active()
        print(
            f"ok site={identity.site_id} target={identity.target_id} backend={identity.backend} "
            f"task_pack_digest={pack.digest} fingerprints={len(pack.fingerprints)} "
            f"grants={len(grants)} active_key_id={active_key_id}"
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
