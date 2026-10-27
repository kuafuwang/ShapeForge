"""Test-only gRPC composition root for the simulated Docker acceptance path.

This is NOT the production entry point and it is NOT an NPU composition root.
It injects the CPU fake worker from ``tests.support.fake_worker_job_engine`` and
a simulated device identity so the macOS control host can exercise real
gRPC/TLS/mTLS on the **legacy Metal/source contract**: durable jobs, attempts,
restart and fault recovery.  Every NPU (fake/simulated/source) request must use
the `tcenv.remote.npu-source.*` contract and its test-only composition root
(``tests.support.vendor_sim``), so this entry refuses NPU-looking identities
instead of serving fake NPU traffic under the old contract.

The production ``tcenv.remote.grpc_service`` entry refuses test worker modules
and simulated measurement identities, so this module cannot be reached there.

Acceptance-only fault injection lives here as well: a configurable set of
request IDs returns a result envelope with the signature withheld, which proves
the host refuses unsigned results instead of only unit-testing the verifier.
The injected fault never touches the durable committed result.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

from tcenv.remote.grpc_server import EvaluatorService, GrpcServerConfig, build_server
from tcenv.remote.grants import load_grants_file
from tcenv.remote.identity import ServiceIdentity
from tcenv.remote.job_engine import JobEngine
from tcenv.remote.keyring import load_keyring_file
from tcenv.remote.revocation import load_revocation_file
from tcenv.remote.v1 import evaluator_pb2 as pb


class _UnsignedResultService(EvaluatorService):
    """Withhold the HMAC signature for named request IDs (acceptance fault)."""

    unsigned_request_ids: frozenset[str] = frozenset()

    def _strip(self, snapshot: pb.JobSnapshot) -> pb.JobSnapshot:
        if snapshot.request_id in self.unsigned_request_ids and snapshot.HasField("result"):
            snapshot.result.ClearField("signature")
        return snapshot

    @staticmethod
    def _log(method: str, started: float, note: str = "") -> None:
        print(
            f"rpc {method} {note}elapsed_ms={int((time.monotonic() - started) * 1000)} "
            f"threads={threading.active_count()}",
            flush=True,
        )

    def GetCapabilities(self, request: pb.CapabilitiesRequest, context) -> pb.Capabilities:
        started = time.monotonic()
        try:
            return super().GetCapabilities(request, context)
        finally:
            self._log("GetCapabilities", started)

    def Submit(self, request: pb.SubmitRequest, context) -> pb.JobSnapshot:
        started = time.monotonic()
        try:
            return self._strip(super().Submit(request, context))
        finally:
            request_id = request.request.request_id if request.HasField("request") else ""
            self._log("Submit", started, f"request_id={request_id} ")

    def GetJob(self, request: pb.JobRef, context) -> pb.JobSnapshot:
        started = time.monotonic()
        try:
            return self._strip(super().GetJob(request, context))
        finally:
            self._log("GetJob", started, f"request_id={request.request_id} ")

    def Cancel(self, request: pb.CancelRequest, context) -> pb.JobSnapshot:
        started = time.monotonic()
        try:
            return self._strip(super().Cancel(request, context))
        finally:
            self._log("Cancel", started)

    def Retry(self, request: pb.RetryRequest, context) -> pb.JobSnapshot:
        started = time.monotonic()
        try:
            return self._strip(super().Retry(request, context))
        finally:
            self._log("Retry", started)

    def Watch(self, request: pb.WatchRequest, context):
        started = time.monotonic()
        try:
            for event in super().Watch(request, context):
                if event.HasField("snapshot"):
                    self._strip(event.snapshot)
                yield event
        finally:
            request_id = request.job.request_id if request.HasField("job") else ""
            self._log("Watch", started, f"request_id={request_id} ")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.npu_sim.grpc_service_entry")
    parser.add_argument("--listen", default="0.0.0.0:7443")
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--site-id", default="sim-grpc-site")
    parser.add_argument("--device-id", default="sim-grpc-device")
    parser.add_argument("--backend", default="sim-cpu")
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
    parser.add_argument("--revoked-client-certificates", type=Path, default=None)
    parser.add_argument("--selfcheck-max-timed", type=int, default=20)
    parser.add_argument("--selfcheck-max-gpu-s", type=float, default=120.0)
    parser.add_argument("--selfcheck-age-s", type=float, default=60.0)
    parser.add_argument("--max-watch-streams", type=int, default=8)
    parser.add_argument("--max-queue-depth", type=int, default=1024)
    parser.add_argument("--max-db-bytes", type=int, default=1 << 30)
    parser.add_argument("--auth-max-failures", type=int, default=64)
    parser.add_argument("--auth-window-s", type=float, default=60.0)
    parser.add_argument("--clock-skew-ms", type=int, default=5_000)
    parser.add_argument(
        "--unsigned-result-for", action="append", default=[],
        help="return an unsigned result envelope for this request ID (acceptance fault)",
    )
    parser.add_argument("--worker-env", default="{}", help="JSON object of extra fake-worker env")
    args = parser.parse_args(argv)
    if "npu" in args.backend.lower() or "npu" in args.target_id.lower():
        # The legacy contract is Metal/source only; NPU fake traffic belongs to
        # the npu-source contract (tests/support/vendor_sim), not here.
        raise SystemExit(
            "this composition root serves the legacy Metal/source contract only; "
            "NPU (fake/simulated/source) requests must use the npu-source contract "
            "and its test composition root"
        )
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
    revoked = frozenset() if args.revoked_client_certificates is None else load_revocation_file(
        args.revoked_client_certificates
    )
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
        selfcheck_age_s=args.selfcheck_age_s,
        clock_skew_ms=args.clock_skew_ms,
        max_queue_depth=args.max_queue_depth,
        max_db_bytes=args.max_db_bytes,
    )
    service_factory = None
    if args.unsigned_result_for:
        service_factory = type(
            "_AcceptanceUnsignedResultService",
            (_UnsignedResultService,),
            {"unsigned_request_ids": frozenset(args.unsigned_result_for)},
        )
    try:
        server = build_server(
            engine,
            config=GrpcServerConfig(
                address=args.listen, allow_insecure=False, require_client_identity=True,
                certificate_file=args.certificate_file, private_key_file=args.private_key_file,
                client_ca_file=args.client_ca_file, max_watch_streams=args.max_watch_streams,
                auth_max_failures=args.auth_max_failures, auth_window_s=args.auth_window_s,
                revoked_client_certificates=revoked,
            ),
            service_factory=service_factory,
        )
    except Exception:
        engine.close()
        raise
    print(
        f"simulated gRPC evaluator listening on {server.address} "
        f"epoch={identity.external_epoch(execution_protocol_version='3')} "
        f"revoked_client_certificates={len(revoked)}",
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
