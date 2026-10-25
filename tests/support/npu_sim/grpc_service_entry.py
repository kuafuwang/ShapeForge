"""Test-only gRPC composition root for the simulated Docker acceptance path.

This is NOT the production entry point.  It injects the CPU fake worker from
``tests.support.fake_worker_job_engine`` and a simulated device identity so the
macOS control host can exercise real gRPC/TLS/mTLS, durable jobs, attempts and
fault recovery without a GPU, NPU or NPU SDK.

The production ``tcenv.remote.grpc_service`` entry refuses test worker modules
and simulated measurement identities, so this module cannot be reached there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tcenv.remote.grants import load_grants_file
from tcenv.remote.identity import ServiceIdentity
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.keyring import load_keyring_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.npu_sim.grpc_service_entry")
    parser.add_argument("--listen", default="0.0.0.0:7443")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--site-id", default="sim-grpc-site")
    parser.add_argument("--device-id", default="sim-grpc-device")
    parser.add_argument("--backend", default="sim-npu")
    parser.add_argument("--backend-version", default="1")
    parser.add_argument("--target-id", default="sim-grpc-target")
    parser.add_argument("--worker-build", default="sim-grpc-build-1")
    parser.add_argument("--task-pack-digest", default="b" * 64)
    parser.add_argument("--calibration-digest", default="c" * 64)
    parser.add_argument("--task-fingerprint", action="append", default=None)
    parser.add_argument("--grants-file", required=True, type=Path)
    parser.add_argument("--keyring-file", required=True, type=Path)
    parser.add_argument("--certificate-file", required=True, type=Path)
    parser.add_argument("--private-key-file", required=True, type=Path)
    parser.add_argument("--client-ca-file", required=True, type=Path)
    parser.add_argument("--selfcheck-max-timed", type=int, default=20)
    parser.add_argument("--selfcheck-max-gpu-s", type=float, default=120.0)
    parser.add_argument("--max-watch-streams", type=int, default=8)
    parser.add_argument("--clock-skew-ms", type=int, default=5_000)
    parser.add_argument("--worker-env", default="{}", help="JSON object of extra fake-worker env")
    args = parser.parse_args(argv)
    fingerprints = frozenset(args.task_fingerprint or ["a" * 64])
    identity = ServiceIdentity(
        site_id=args.site_id, device_id=args.device_id, backend=args.backend,
        backend_version=args.backend_version, target_id=args.target_id,
        worker_build=args.worker_build, task_pack_digest=args.task_pack_digest,
        calibration_digest=args.calibration_digest, task_fingerprints=fingerprints,
        task_bindings={fingerprint: ("relu", "fp16", "small") for fingerprint in fingerprints},
        measurement_kind="simulated", publication_allowed=False,
    )
    grants = load_grants_file(args.grants_file)
    keyring = load_keyring_file(args.keyring_file)
    active_key_id, active_key = keyring.require_active()
    worker_env = json.loads(args.worker_env)
    if not isinstance(worker_env, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in worker_env.items()):
        raise SystemExit("--worker-env must be a JSON object of strings")
    engine = JobEngine(
        identity=identity, grants=grants,
        worker_command=[sys.executable, "-m", "tests.support.fake_worker_job_engine"],
        worker_env=worker_env,
        state_dir=args.state_dir,
        signing_key=active_key, signing_key_id=active_key_id, signing_keyring=keyring.trusted,
        selfcheck_max_timed=args.selfcheck_max_timed,
        selfcheck_max_gpu_s=args.selfcheck_max_gpu_s,
        clock_skew_ms=args.clock_skew_ms,
    )
    try:
        server = build_server(
            engine,
            config=GrpcServerConfig(
                address=args.listen, allow_insecure=False, require_client_identity=True,
                certificate_file=args.certificate_file, private_key_file=args.private_key_file,
                client_ca_file=args.client_ca_file, max_watch_streams=args.max_watch_streams,
            ),
        )
    except Exception:
        engine.close()
        raise
    print(
        f"simulated gRPC evaluator listening on {server.address} "
        f"epoch={identity.external_epoch(execution_protocol_version='3')}",
        flush=True,
    )
    try:
        server.server.wait_for_termination()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop(grace_s=2.0)
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
