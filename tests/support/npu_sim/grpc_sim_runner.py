"""Phase 2D2 simulated gRPC/TLS acceptance: macOS host → Docker Linux → evaluator.

Runs the real external gRPC adapter over real TLS/mTLS against a Linux container
built from the test-only composition root, injects transport/auth/restart faults
and writes ``runs/grpc-sim-<date>/report.json``.  No GPU, NPU, NPU SDK, real
model, SSH data plane or publication claim is involved.

Usage (explicit authorization):

    TAIC_DOCKER_GRPC_SIM=1 uv run python -m tests.support.npu_sim.grpc_sim_runner

The report is a diagnostic artifact: ``simulated=true`` and
``publication_allowed=false``.  It is not a search run, has no run_config.json
and never writes confirmed/catalog/brief state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import grpc

from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.grpc_client import (
    GrpcEvaluatorFactory,
    GrpcExpected,
    SignatureKeyring,
    grpc_evaluator_factory_from_config,
    load_client_credentials,
)
from tcenv.remote.grants import SessionGrant, write_grants_file
from tcenv.remote.keyring import KeyringFile, new_key, write_keyring_file
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.v1 import evaluator_pb2_grpc as pb_grpc
from tests.support.npu_sim.tcp_proxy import FaultProxy
from tests.support.remote_pki import make_pki

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "taic-remote-grpc-sim:local"
SITE = "sim-grpc-site"
DEVICE = "sim-grpc-device"
TARGET = "sim-grpc-target"
BUILD = "sim-grpc-build-1"
TASK_PACK = "b" * 64
CALIBRATION = "c" * 64
FINGERPRINT = "a" * 64
CONTAINER_PORT = 7443


def _docker_env() -> dict[str, str]:
    env = dict(os.environ)
    if "DOCKER_HOST" in env:
        return env
    # Resolve the current context before isolating the CLI config: Docker
    # Desktop writes buildx state into ~/.docker, which some fences deny.
    try:
        completed = subprocess.run(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
            text=True, capture_output=True, check=False,
        )
        host = completed.stdout.strip()
    except OSError:
        host = ""
    if host:
        env["DOCKER_HOST"] = host
    if "DOCKER_CONFIG" not in env:
        candidate = Path(os.environ.get("TMPDIR") or "/tmp") / "tcenv-docker-config"
        candidate.mkdir(parents=True, exist_ok=True)
        env["DOCKER_CONFIG"] = str(candidate)
    return env


class Docker:
    def __init__(self) -> None:
        self.env = _docker_env()

    def run(self, *arguments: str, check: bool = True, timeout: float = 900.0) -> subprocess.CompletedProcess:
        completed = subprocess.run(
            ["docker", *arguments], text=True, capture_output=True,
            env=self.env, check=False, timeout=timeout,
        )
        if check and completed.returncode != 0:
            raise RuntimeError(f"docker {' '.join(arguments)} failed: {completed.stdout}{completed.stderr}")
        return completed

    def image_id(self, image: str) -> str:
        return self.run("image", "inspect", "--format", "{{.Id}}", image).stdout.strip()

    def container_logs(self, name: str) -> str:
        return self.run("logs", name, check=False).stdout + self.run("logs", name, check=False).stderr

    def remove(self, name: str) -> None:
        self.run("rm", "-f", name, check=False)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(request_id: str, source: str, *, kind: str = "formal_eval", mode: str = "quick", run_id: str = "run-1", session_id: str = "session-1", queue_s: float = 10.0, candidate_s: float = 15.0, progress_path: Path | None = None) -> EvaluationRequest:
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source, kind=kind, mode=mode,
        run_id=run_id, session_id=session_id, request_id=request_id, target_id=TARGET,
        task_fingerprint=FINGERPRINT, queue_timeout_s=queue_s,
        candidate_timeout_s=candidate_s, progress_path=progress_path,
    )


class Sim:
    def __init__(self, *, root: Path, docker: Docker) -> None:
        self.docker = docker
        self.root = root
        self.config = root / "container-config"
        self.state = root / "container-state"
        self.restored_state = root / "container-state-restored"
        self.backup = root / "container-backup"
        for directory in (self.config, self.state, self.backup, self.restored_state):
            directory.mkdir(parents=True, exist_ok=True)
            directory.chmod(0o700)
        self.pki = make_pki(self.config / "pki")
        self.key = new_key()
        self.key_id = f"{SITE}/key-1"
        self.keyring_path = self.config / "keyring.json"
        write_keyring_file(self.keyring_path, KeyringFile({self.key_id: self.key}, frozenset(), self.key_id))
        self.grants: list[SessionGrant] = [
            SessionGrant(token="grant-formal-selfcheck", run_id="run-1", session_id="session-1", target_id=TARGET, subject="host", max_timed=1, max_gpu_s=120.0),
            SessionGrant(token="grant-run-2", run_id="run-2", session_id="session-1", target_id=TARGET, subject="host"),
            SessionGrant(token="grant-expired", run_id="run-1", session_id="session-expired", target_id=TARGET, subject="host", expires_at=time.time() - 60),
            SessionGrant(token="grant-revoke-me", run_id="run-1", session_id="session-revoke", target_id=TARGET, subject="host"),
        ]
        self.grants_path = self.config / "grants.json"
        write_grants_file(self.grants_path, self.grants)
        self.name = "taic-grpc-sim"
        self.port = 0
        self.worker_env = {
            "FAKE_JOB_COUNTER": "/state/worker-count.txt",
            "FAKE_JOB_RUNNING": "/state/running.json",
            "FAKE_JOB_HOLD_DIR": "/state/hold-once",
        }
        self.started = False

    def write_grants(self, tokens: set[str]) -> None:
        write_grants_file(self.grants_path, [grant for grant in self.grants if grant.token in tokens])

    def start(self, *, state: Path | None = None, port: int | None = None, name: str | None = None) -> int:
        name = name or self.name
        self.docker.remove(name)
        port = port or _free_port()
        self.port = port
        uid, gid = os.getuid(), os.getgid()
        arguments = [
            "run", "-d", "--name", name,
            "-p", f"127.0.0.1:{port}:{CONTAINER_PORT}",
            "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--user", f"{uid}:{gid}", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--mount", f"type=bind,src={self.config},dst=/etc/tcenv,readonly",
            "--mount", f"type=bind,src={state or self.state},dst=/state",
            IMAGE,
            "--listen", f"0.0.0.0:{CONTAINER_PORT}",
            "--state-dir", "/state",
            "--site-id", SITE, "--device-id", DEVICE,
            "--target-id", TARGET, "--worker-build", BUILD,
            "--task-pack-digest", TASK_PACK, "--calibration-digest", CALIBRATION,
            "--task-fingerprint", FINGERPRINT,
            "--grants-file", "/etc/tcenv/grants.json",
            "--keyring-file", "/etc/tcenv/keyring.json",
            "--certificate-file", "/etc/tcenv/pki/server.pem",
            "--private-key-file", "/etc/tcenv/pki/server.key",
            "--client-ca-file", "/etc/tcenv/pki/ca.pem",
            "--worker-env", json.dumps(self.worker_env),
        ]
        arguments[arguments.index("--name") + 1] = name
        completed = self.docker.run(*arguments)
        self.started = True
        return port

    def stop(self, *, grace: int = 10, name: str | None = None) -> None:
        self.docker.run("stop", "-t", str(grace), name or self.name, check=False, timeout=grace + 30)
        self.docker.remove(name or self.name)
        self.started = False

    def kill(self, *, name: str | None = None) -> None:
        self.docker.run("kill", name or self.name, check=False)
        self.docker.remove(name or self.name)
        self.started = False

    def wait_ready(self, *, timeout: float = 60.0, token: str = "grant-formal-selfcheck") -> None:
        deadline = time.monotonic() + timeout
        last: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                credentials = load_client_credentials(
                    ca_file=self.pki.ca_cert, certificate_file=self.pki.client_cert,
                    private_key_file=self.pki.client_key,
                )
                channel = grpc.secure_channel(
                    f"127.0.0.1:{self.port}", credentials,
                    options=[("grpc.ssl_target_name_override", "localhost")],
                )
                try:
                    stub = pb_grpc.EvaluatorStub(channel)
                    stub.GetCapabilities(
                        pb.CapabilitiesRequest(
                            api_contract_ids=[contracts.API_CONTRACT_ID],
                            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
                            client_nonce="ready-" + "0" * 26,
                            expected=pb.ExpectedEvaluator(
                                site_id=SITE, device_id=DEVICE, target_id=TARGET,
                                worker_build=BUILD, task_pack_digest=TASK_PACK,
                            ),
                        ),
                        metadata=(("authorization", f"Bearer {token}"),), timeout=5,
                    )
                    return
                finally:
                    channel.close()
            except Exception as exc:  # noqa: BLE001 - readiness loop
                last = exc
                time.sleep(0.3)
        raise RuntimeError(f"simulated evaluator did not become ready: {last}\n{self.docker.container_logs(self.name)[-3000:]}")

    def factory(self, *, token: str = "grant-formal-selfcheck", run_id: str = "run-1", keyring: SignatureKeyring | None = None, endpoint: str | None = None, rpc_timeout_s: float = 15.0, watch_slice_s: float = 5.0, job_margin_s: float = 20.0) -> GrpcEvaluatorFactory:
        return grpc_evaluator_factory_from_config(
            endpoint=endpoint or f"127.0.0.1:{self.port}",
            token_provider=lambda _run, _session: token,
            run_id=run_id,
            target_id=TARGET,
            ca_file=self.pki.ca_cert,
            certificate_file=self.pki.client_cert,
            private_key_file=self.pki.client_key,
            keyring_file=self.keyring_path,
            keyring=keyring,
            expected=GrpcExpected(
                site_id=SITE, device_id=DEVICE, target_id=TARGET,
                worker_build=BUILD, task_pack_digest=TASK_PACK,
            ),
            server_name_override="localhost",
            rpc_timeout_s=rpc_timeout_s,
            watch_slice_s=watch_slice_s,
            job_margin_s=job_margin_s,
        )

    def raw_channel(self, *, ca=None, cert=None, key=None, server_name: str = "localhost", endpoint: str | None = None, insecure: bool = False):
        if insecure:
            return grpc.insecure_channel(endpoint or f"127.0.0.1:{self.port}")
        credentials = load_client_credentials(
            ca_file=ca or self.pki.ca_cert,
            certificate_file=self.pki.client_cert if cert is None else cert,
            private_key_file=self.pki.client_key if key is None else key,
        )
        return grpc.secure_channel(
            endpoint or f"127.0.0.1:{self.port}", credentials,
            options=[("grpc.ssl_target_name_override", server_name)],
        )

    def raw_capabilities(self, token: str = "grant-formal-selfcheck", *, features=(), api=contracts.API_CONTRACT_ID, evidence=contracts.EVIDENCE_CONTRACT_ID, channel=None, metadata=True):
        own = channel is None
        channel = channel or self.raw_channel()
        try:
            stub = pb_grpc.EvaluatorStub(channel)
            call_metadata = (("authorization", f"Bearer {token}"),) if metadata else ()
            return stub.GetCapabilities(
                pb.CapabilitiesRequest(
                    api_contract_ids=[api], evidence_contract_ids=[evidence],
                    required_features=list(features), client_nonce=os.urandom(16).hex(),
                    expected=pb.ExpectedEvaluator(
                        site_id=SITE, device_id=DEVICE, target_id=TARGET,
                        worker_build=BUILD, task_pack_digest=TASK_PACK,
                    ),
                ),
                metadata=call_metadata, timeout=10,
            )
        finally:
            if own:
                channel.close()

    def attempts(self, request_id: str, state: Path | None = None) -> int:
        path = (state or self.state) / "jobs.sqlite3"
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            return int(connection.execute("SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,)).fetchone()[0])
        finally:
            connection.close()

    def job_source_cleared(self) -> bool:
        connection = sqlite3.connect(f"file:{self.state / 'jobs.sqlite3'}?mode=ro", uri=True)
        try:
            row = connection.execute("SELECT COUNT(*) FROM jobs WHERE state='completed' AND source IS NULL").fetchone()
            total = connection.execute("SELECT COUNT(*) FROM jobs WHERE state='completed'").fetchone()
            return row[0] == total[0] >= 1
        finally:
            connection.close()

    def stale_bridge_dirs(self) -> list[str]:
        root = self.state / "bridge-tmp"
        if not root.is_dir():
            return []
        return [entry.name for entry in root.iterdir()]


def _grpc_code(exc: BaseException) -> str:
    if isinstance(exc, grpc.RpcError):
        return exc.code().name
    return getattr(exc, "code", type(exc).__name__)


def _conformance_request(*, run_id: str = "run-1", session_id: str = "session-1", request_id: str, source: str, kind: str = "formal_eval", mode: str = "quick") -> EvaluationRequest:
    return _request(request_id, source, kind=kind, mode=mode, run_id=run_id, session_id=session_id)


class Checker:
    def __init__(self, sim: Sim) -> None:
        self.sim = sim
        self.results: list[dict] = []

    def check(self, name: str, function, category: str = "") -> dict:
        started = time.monotonic()
        try:
            evidence = function() or {}
            result = {"name": name, "category": category, "ok": True, "evidence": evidence}
        except BaseException as exc:  # noqa: BLE001 - one failed check must not stop the matrix
            result = {"name": name, "category": category, "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:2000]}
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        self.results.append(result)
        return result

    def expect_grpc_error(self, name: str, function, expected: set[str], category: str = "tls") -> dict:
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            code = _grpc_code(exc)
            if code not in expected:
                raise AssertionError(f"expected {sorted(expected)}, got {code}: {exc}")
            return {"name": name, "category": category, "ok": True, "evidence": {"grpc_code": code}, "elapsed_s": 0}
        raise AssertionError(f"{name} unexpectedly succeeded")


def run_checks(sim: Sim, checker: Checker, *, state_root: Path) -> list[dict]:
    results = checker.results

    def capabilities() -> dict:
        factory = sim.factory()
        try:
            epoch = factory.ensure_connected()
            metadata = factory.execution_metadata()
            return {"evaluator_epoch": epoch, **metadata}
        finally:
            factory.close()
    checker.check("tls_mtls_capabilities", capabilities, "tls")

    def formal_ok() -> dict:
        factory = sim.factory()
        try:
            result = factory.formal(_conformance_request(request_id="sim-formal-ok", source="OK"))
            receipt = result.receipt
            dummy = SimpleNamespace(publishable_site_ids=frozenset(), publishable_epochs=frozenset())
            from taic.orchestrator import Orchestrator
            refused = False
            try:
                Orchestrator._check_publication(dummy, result)
            except EvaluationUnavailable as exc:
                refused = "simulated" in str(exc)
            if not refused:
                raise AssertionError("simulated result was not refused by the host publication gate")
            if result.provenance.measurement_kind != "simulated" or result.provenance.publication_allowed:
                raise AssertionError("simulated provenance flags are wrong")
            return {
                "attempt_id": receipt.get("attempt_id", ""),
                "lease_fence": receipt.get("lease_fence", ""),
                "signature_key_id": receipt.get("signature_key_id", ""),
                "measurement_kind": result.provenance.measurement_kind,
                "publication_allowed": result.provenance.publication_allowed,
                "host_publication_refused": refused,
                "report_status": result.report.get("status"),
            }
        finally:
            factory.close()

    def retry_interrupted() -> dict:
        factory = sim.factory()
        try:
            try:
                factory.formal(_conformance_request(request_id="sim-retry", source="FAIL_FIRST"))
            except EvaluationUnavailable as exc:
                if not exc.retryable:
                    raise AssertionError(f"first attempt was not retryable: {exc.code}")
            else:
                raise AssertionError("FAIL_FIRST unexpectedly completed on attempt 1")
            resumed = factory.formal_resume(_conformance_request(request_id="sim-retry", source="FAIL_FIRST"))
            attempts = sim.attempts("sim-retry")
            if attempts != 2:
                raise AssertionError(f"expected 2 attempts, found {attempts}")
            return {"attempt_id": resumed.receipt.get("attempt_id"), "lease_fence": resumed.receipt.get("lease_fence"), "attempts": attempts}
        finally:
            factory.close()
    checker.check("retry_only_interrupted", retry_interrupted, "state-machine")
    checker.check("simulated_result_and_publication_gate", formal_ok, "protocol")

    def cancel_job() -> dict:
        factory = sim.factory()
        errors: list[BaseException] = []
        import threading

        def run() -> None:
            try:
                factory.formal(_conformance_request(request_id="sim-cancel", source="SLEEP"))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        running = sim.state / "running.json"
        deadline = time.monotonic() + 10
        while not running.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        factory.cancel("sim-cancel")
        thread.join(timeout=20)
        try:
            if thread.is_alive():
                raise AssertionError("cancel did not finish the client")
            state = sim.attempts("sim-cancel")
            if state != 1:
                raise AssertionError(f"cancel changed attempt count to {state}")
            return {"client_error": _grpc_code(errors[0]) if errors else "none", "attempts": state}
        finally:
            factory.close()
    checker.check("cancel_running_job", cancel_job, "state-machine")

    # TLS negatives
    def _tls(description: str, function, expected: set[str]) -> dict:
        return checker.expect_grpc_error(description, function, expected, "tls")
    cases = {
        "plaintext_rejected": lambda: sim.raw_capabilities(channel=sim.raw_channel(insecure=True)),
        "missing_client_cert": lambda: sim.raw_capabilities(channel=grpc.secure_channel(
            f"127.0.0.1:{sim.port}",
            grpc.ssl_channel_credentials(root_certificates=sim.pki.ca_cert.read_bytes()),
            options=[("grpc.ssl_target_name_override", "localhost")],
        )),
        "wrong_ca_rejected": lambda: sim.raw_capabilities(channel=sim.raw_channel(ca=sim.pki.other_ca_cert)),
        "wrong_san_rejected": lambda: sim.raw_capabilities(channel=sim.raw_channel(server_name="wrong.example")),
        "untrusted_client_cert_rejected": lambda: sim.raw_capabilities(channel=sim.raw_channel(cert=sim.pki.other_client_cert, key=sim.pki.other_client_key)),
    }
    for name, function in cases.items():
        results.append(_tls(name, function, {"UNAVAILABLE", "UNAUTHENTICATED", "PERMISSION_DENIED"}))

    # Authentication and grant scope
    results.append(_tls("missing_grant_rejected", lambda: sim.raw_capabilities(metadata=False), {"UNAUTHENTICATED"}))
    results.append(_tls("unknown_grant_rejected", lambda: sim.raw_capabilities(token="not-a-grant"), {"UNAUTHENTICATED"}))
    results.append(_tls("expired_grant_rejected", lambda: sim.raw_capabilities(token="grant-expired"), {"UNAUTHENTICATED"}))

    def wrong_scope() -> dict:
        capabilities = sim.raw_capabilities(token="grant-run-2")
        channel = sim.raw_channel()
        try:
            stub = pb_grpc.EvaluatorStub(channel)
            probe = _request("sim-wrong-scope", "OK", run_id="run-1", session_id="session-1")
            from tcenv.remote.job_types import ExternalEvaluationRequest
            external = ExternalEvaluationRequest.from_legacy(probe)
            expected = gc.expected_identities_from_provenance(
                gc.provenance_from_proto(capabilities.provenance, require_attempt=False)
            )
            submit = pb.SubmitRequest(
                request=gc.request_to_proto(external),
                request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(expected),
                capability_id=capabilities.capability_id,
            )
            try:
                stub.Submit(submit, metadata=(("authorization", "Bearer grant-run-2"),), timeout=10)
            except grpc.RpcError as exc:
                if exc.code() not in {grpc.StatusCode.PERMISSION_DENIED, grpc.StatusCode.NOT_FOUND}:
                    raise AssertionError(f"wrong-scope submit returned {exc.code()}")
                return {"grpc_code": exc.code().name}
            raise AssertionError("a run-2 grant submitted a run-1 request")
        finally:
            channel.close()
    checker.check("grant_scope_mismatch_rejected", wrong_scope, "authz")

    def revoked_grant() -> dict:
        # Revocation removes the grant from the restart configuration; the old
        # token then has no visible authority.
        sim.stop(grace=10)
        sim.write_grants({"grant-formal-selfcheck", "grant-run-2", "grant-expired"})
        sim.start()
        sim.wait_ready()
        try:
            sim.raw_capabilities(token="grant-revoke-me")
        except grpc.RpcError as exc:
            if exc.code() not in {grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED}:
                raise AssertionError(f"revoked grant returned {exc.code()}")
            return {"grpc_code": exc.code().name}
        raise AssertionError("a revoked grant was accepted")
    checker.check("revoked_grant_rejected", revoked_grant, "authz")

    # Contract / feature negatives
    def contract_negative(api: str = contracts.API_CONTRACT_ID, evidence: str = contracts.EVIDENCE_CONTRACT_ID, features=()) -> dict:
        try:
            sim.raw_capabilities(api=api, evidence=evidence, features=features)
        except grpc.RpcError as exc:
            detail = None
            try:
                from grpc_status import rpc_status
                status = rpc_status.from_call(exc)
                if status is not None:
                    for item in status.details:
                        if item.Is(pb.EvaluationErrorDetail.DESCRIPTOR):
                            detail = pb.EvaluationErrorDetail(); item.Unpack(detail)
            except Exception:
                detail = None
            if exc.code() != grpc.StatusCode.FAILED_PRECONDITION:
                raise AssertionError(f"contract negative returned {exc.code()}")
            return {"grpc_code": exc.code().name, "missing_features": list(detail.missing_features) if detail else []}
        raise AssertionError("an unsupported contract/feature was accepted")
    checker.check("unknown_api_contract_rejected", lambda: contract_negative(api="tcenv.remote.v1.evaluator-api.99"), "contract")
    checker.check("unknown_evidence_contract_rejected", lambda: contract_negative(evidence="tcenv.remote.v1.evidence.99"), "contract")
    checker.check("unknown_required_feature_rejected", lambda: contract_negative(features=("grpc-sim-unknown",)), "contract")

    # HMAC key negatives
    def wrong_keyring() -> dict:
        wrong = SignatureKeyring({sim.key_id: b"w" * 32})
        factory = sim.factory(keyring=wrong)
        try:
            try:
                factory.formal(_conformance_request(request_id="sim-wrong-key", source="OK"))
            except EvaluationUnavailable as exc:
                if exc.code != "signature_mismatch":
                    raise AssertionError(f"wrong key returned {exc.code}")
                return {"code": exc.code}
            raise AssertionError("a wrong host keyring accepted the result")
        finally:
            factory.close()
    checker.check("wrong_hmac_key_rejected", wrong_keyring, "signature")

    def revoked_key() -> dict:
        keyring = SignatureKeyring.from_file(sim.keyring_path)
        keyring.revoke(sim.key_id)
        factory = sim.factory(keyring=keyring)
        try:
            try:
                factory.formal(_conformance_request(request_id="sim-revoked-key", source="OK"))
            except EvaluationUnavailable as exc:
                if exc.code != "signature_key_revoked":
                    raise AssertionError(f"revoked key returned {exc.code}")
                return {"code": exc.code}
            raise AssertionError("a revoked key verified a result")
        finally:
            factory.close()
    checker.check("revoked_hmac_key_rejected", revoked_key, "signature")

    # Fault injection through a TCP proxy
    def reset_reconcile() -> dict:
        with FaultProxy("127.0.0.1", sim.port, mode="reset", reset_once=True) as proxy:
            factory = sim.factory(endpoint=proxy.endpoint, rpc_timeout_s=5, watch_slice_s=2, job_margin_s=10)
            try:
                factory.formal(_conformance_request(request_id="sim-reset", source="OK"))
            except EvaluationUnavailable:
                pass
            finally:
                factory.close()
        factory = sim.factory(rpc_timeout_s=10)
        try:
            result = factory.formal(_conformance_request(request_id="sim-reset", source="OK"))
            attempts = sim.attempts("sim-reset")
            if attempts != 1:
                raise AssertionError(f"reset caused {attempts} executions")
            return {"status": result.report.get("status"), "attempts": attempts}
        finally:
            factory.close()
    checker.check("connection_reset_reconciles_via_getjob", reset_reconcile, "faults")

    def half_open_deadline() -> dict:
        with FaultProxy("127.0.0.1", sim.port, mode="half_open") as proxy:
            factory = sim.factory(endpoint=proxy.endpoint, rpc_timeout_s=1, watch_slice_s=1, job_margin_s=5)
            try:
                try:
                    factory.formal(_conformance_request(request_id="sim-half-open", source="OK"))
                except EvaluationUnavailable as exc:
                    code = exc.code
                else:
                    code = "completed"
            finally:
                factory.close()
        factory = sim.factory(rpc_timeout_s=10)
        try:
            result = factory.formal(_conformance_request(request_id="sim-half-open", source="OK"))
            return {"first_attempt": code, "reconciled_status": result.report.get("status")}
        finally:
            factory.close()
    checker.check("half_open_deadline_then_getjob", half_open_deadline, "faults")

    def slow_consumer_progress() -> dict:
        progress_path = state_root / "slow-progress.json"
        with FaultProxy("127.0.0.1", sim.port, mode="slow", delay_s=0.05) as proxy:
            factory = sim.factory(endpoint=proxy.endpoint, rpc_timeout_s=10, watch_slice_s=1, job_margin_s=15)
            try:
                result = factory.formal(_request(
                    "sim-slow", "PROGRESS_MANY", candidate_s=15, progress_path=progress_path,
                ))
            finally:
                factory.close()
        if not progress_path.is_file():
            raise AssertionError("slow consumer produced no host progress file")
        return {"status": result.report.get("status"), "progress_file": progress_path.name}
    checker.check("slow_consumer_watch_reconnect", slow_consumer_progress, "faults")

    # Quota
    def quota() -> dict:
        factory = sim.factory()
        try:
            first = factory.selfcheck(_conformance_request(request_id="sim-quota-1", source="OK", kind="selfcheck", mode="timed"))
            try:
                factory.selfcheck(_conformance_request(request_id="sim-quota-2", source="OK", kind="selfcheck", mode="timed"))
            except EvaluationUnavailable as exc:
                if exc.code != "resource_exhausted":
                    raise AssertionError(f"second timed selfcheck returned {exc.code}")
                return {"first_status": first.report.get("status"), "second_code": exc.code}
            raise AssertionError("timed selfcheck quota was not enforced")
        finally:
            factory.close()
    checker.check("selfcheck_quota", quota, "quota")

    # Grpc service restart with a running job
    def restart_recovery() -> dict:
        factory = sim.factory(rpc_timeout_s=10, watch_slice_s=5, job_margin_s=20)
        import threading

        errors: list[BaseException] = []

        def run() -> None:
            try:
                factory.formal(_request("sim-restart", "HOLD_ONCE", candidate_s=60))
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        running = sim.state / "running.json"
        running.unlink(missing_ok=True)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not running.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not running.is_file():
            raise AssertionError("the restart probe never reached the worker")
        sim.stop(grace=10)
        factory.close()
        thread.join(timeout=20)
        sim.start()
        sim.wait_ready()
        attempts_before = sim.attempts("sim-restart")
        if attempts_before != 1:
            raise AssertionError(f"restart probe created {attempts_before} attempts before the stop")
        resumed_factory = sim.factory(rpc_timeout_s=10)
        try:
            result = resumed_factory.formal_resume(_request("sim-restart", "HOLD_ONCE", candidate_s=60))
            attempts_after = sim.attempts("sim-restart")
            if attempts_after != 2:
                raise AssertionError(f"explicit resume produced {attempts_after} attempts")
            return {
                "client_error": _grpc_code(errors[0]) if errors else "none",
                "attempts_before_retry": attempts_before,
                "attempts_after_retry": attempts_after,
                "lease_fence": result.receipt.get("lease_fence"),
                "note": "host closed the old channel; the durable attempt stayed interrupted until explicit Retry",
            }
        finally:
            resumed_factory.close()
    checker.check("container_restart_explicit_retry", restart_recovery, "recovery")

    # SIGKILL leaves a durable running fence; recovery must not auto-run.
    def stale_worker() -> dict:
        factory = sim.factory(rpc_timeout_s=10, watch_slice_s=5, job_margin_s=20)
        import threading

        def run() -> None:
            try:
                factory.formal(_request("sim-stale", "HOLD_ONCE", candidate_s=60))
            except BaseException:
                pass

        running = sim.state / "running.json"
        running.unlink(missing_ok=True)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not running.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not running.is_file():
            raise AssertionError("the stale probe never reached the worker")
        sim.kill()
        factory.close()
        thread.join(timeout=20)
        attempts_before = sim.attempts("sim-stale")
        if attempts_before != 1:
            raise AssertionError(f"stale probe created {attempts_before} attempts before the kill")
        sim.start()
        sim.wait_ready()
        time.sleep(1.0)
        attempts_after_recovery = sim.attempts("sim-stale")
        # Either interrupted (resumable) or quarantined (operator release) is
        # acceptable; automatic re-execution is not.
        if attempts_after_recovery != attempts_before:
            raise AssertionError("recovery silently re-executed a stale attempt")
        resumed_factory = sim.factory(rpc_timeout_s=10)
        try:
            result = resumed_factory.formal_resume(_request("sim-stale", "HOLD_ONCE", candidate_s=60))
            final_attempts = sim.attempts("sim-stale")
            return {"attempts_before": attempts_before, "attempts_after": final_attempts, "lease_fence": result.receipt.get("lease_fence")}
        except EvaluationUnavailable as exc:
            if exc.code in {"idempotency_conflict", "session_mismatch", "invalid_argument"}:
                raise AssertionError(f"retry used a different request digest/identity: {exc.code}")
            return {"attempts_before": attempts_before, "quarantined_or_nonretryable": exc.code, "retryable": exc.retryable}
        finally:
            resumed_factory.close()
    checker.check("stale_worker_fail_closed", stale_worker, "recovery")

    # Backup / restore
    def backup_restore() -> dict:
        import threading

        factory = sim.factory(rpc_timeout_s=10, watch_slice_s=5, job_margin_s=20)

        def run() -> None:
            try:
                factory.formal(_request("sim-backup-running", "SLEEP", candidate_s=60))
            except BaseException:
                pass

        running = sim.state / "running.json"
        running.unlink(missing_ok=True)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not running.is_file() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not running.is_file():
            factory.close()
            raise AssertionError("the backup probe never reached the worker")
        sim.stop(grace=10)
        thread.join(timeout=20)
        factory.close()
        docker = sim.docker
        backup_dir = sim.backup
        for entry in backup_dir.iterdir():
            entry.unlink()
        common = [
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=bind,src={sim.state},dst=/state",
            "--mount", f"type=bind,src={backup_dir},dst=/backup",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
        ]
        backup = docker.run(*common, "-m", "tcenv.remote.state_admin", "backup", "--state-dir", "/state", "--out", "/backup/state.sqlite3", "--keyring-file", "/etc/tcenv/keyring.json")
        sqlite_path = backup_dir / "state.sqlite3"
        if not sqlite_path.is_file() or sqlite_path.stat().st_size == 0:
            raise AssertionError("backup file was not produced")
        for entry in list(sim.restored_state.iterdir()):
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
        restore = docker.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=bind,src={backup_dir},dst=/backup,readonly",
            "--mount", f"type=bind,src={sim.restored_state},dst=/restored",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
            "-m", "tcenv.remote.state_admin", "restore", "--backup", "/backup/state.sqlite3", "--state-dir", "/restored",
        )
        inspect = docker.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=bind,src={sim.restored_state},dst=/restored",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
            "-m", "tcenv.remote.state_admin", "inspect", "--state-dir", "/restored", "--keyring-file", "/etc/tcenv/keyring.json",
        )
        document = json.loads(inspect.stdout)
        if not any(state in {"completed"} for state in document["states"]):
            raise AssertionError("restored backup has no completed job")
        if not document["quarantined"]:
            raise AssertionError("restore did not quarantine nonterminal jobs")
        return {"backup_bytes": sqlite_path.stat().st_size, "states": document["states"], "quarantined": len(document["quarantined"])}
    checker.check("backup_restore_quarantine", backup_restore, "recovery")

    def rollback_restart_compat() -> dict:
        # A schema-compatible restart is the supported rollback step: stop new
        # work, restart the same image/schema/state, replay a completed result
        # and accept a new one.  Rolling back to an older image is unrun.
        sim.stop(grace=10)
        sim.start()
        sim.wait_ready()
        factory = sim.factory(rpc_timeout_s=15)
        try:
            replayed = factory.formal(_conformance_request(request_id="sim-formal-ok", source="OK"))
            new = factory.formal(_conformance_request(request_id="sim-rollback-new", source="OK"))
            return {
                "older_image": False,
                "replayed_attempt": replayed.receipt.get("attempt_id"),
                "new_status": new.report.get("status"),
                "new_attempt": new.receipt.get("attempt_id"),
            }
        finally:
            factory.close()
    checker.check("rollback_restart_compat", rollback_restart_compat, "recovery")

    # Start once more for the cleanup/health checks.
    sim.start()
    sim.wait_ready()

    def cleanup() -> dict:
        stale = sim.stale_bridge_dirs()
        if stale:
            raise AssertionError(f"stale bridge temp dirs: {stale}")
        if not sim.job_source_cleared():
            raise AssertionError("completed jobs still retain candidate source bytes")
        return {"stale_bridge_dirs": stale, "completed_sources_cleared": True}
    checker.check("cleanup", cleanup, "cleanup")
    return results


def build_and_run(*, report_path: Path, keep: bool = False, fresh: bool = False) -> dict:
    started = time.monotonic()
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        for name in ("container-config", "container-state", "container-state-restored", "container-backup"):
            candidate = report_path.parent / name
            if candidate.exists():
                shutil.rmtree(candidate)
    docker = Docker()
    build_started = time.monotonic()
    docker.run("build", "-f", "tests/support/docker/Dockerfile.remote-grpc-sim", "-t", IMAGE, ".", timeout=1800)
    build_elapsed = time.monotonic() - build_started
    image_id = docker.image_id(IMAGE)
    sim = Sim(root=report_path.parent, docker=docker)
    os_release = docker.run(
        "run", "--rm", "--entrypoint", "cat", IMAGE, "/etc/os-release", check=False,
    ).stdout
    container_os = next(
        (line.split("=", 1)[1].strip().strip('"') for line in os_release.splitlines() if line.startswith("PRETTY_NAME=")),
        "unknown",
    )
    docker_version = docker.run("version", "--format", "{{.Server.Version}}", check=False).stdout.strip()
    docker_platform = docker.run("info", "--format", "{{.OSType}}/{{.Architecture}}", check=False).stdout.strip()
    report: dict = {
        "schema_version": 1,
        "simulated": True,
        "publication_allowed": False,
        "status": "failed",
        "date": time.strftime("%Y-%m-%d"),
        "environment": {
            "control_host": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
            "container_platform": docker_platform,
            "container_os": container_os,
            "container_host": "Docker Desktop Linux VM",
            "docker_engine": docker_version,
            "image": IMAGE,
            "image_id": image_id,
            "container_user": f"{os.getuid()}:{os.getgid()}",
            "read_only_root": True,
            "capabilities": "dropped",
            "device_mounts": "none",
            "ssh_data_plane": False,
        },
        "transport": {
            "protocol": "gRPC over TLS/mTLS",
            "endpoint": f"127.0.0.1:{sim.port}",
            "server_ca": "disposable test CA",
            "client_identity": "shared host mTLS certificate + per-RPC session grant",
        },
        "identities": {
            "site_id": SITE, "device_id": DEVICE, "target_id": TARGET,
            "worker_build": BUILD, "task_pack_digest": TASK_PACK,
            "calibration_digest": CALIBRATION,
            "api_contract_id": contracts.API_CONTRACT_ID,
            "evidence_contract_id": contracts.EVIDENCE_CONTRACT_ID,
        },
        "results": [],
        "not_run": [
            "real GPU", "real NPU", "NPU SDK installation", "real model inference",
            "SSH transport data plane", "hardware performance or publication numbers",
            "true rollback to a previous image version", "HA/multi-replica",
        ],
        "timings": {"image_build_s": round(build_elapsed, 3)},
    }
    try:
        sim.start()
        sim.wait_ready()
        report["transport"]["endpoint"] = f"127.0.0.1:{sim.port}"
        checker = Checker(sim)
        checks_started = time.monotonic()
        try:
            run_checks(sim, checker, state_root=report_path.parent)
        finally:
            report["results"] = checker.results
        report["timings"]["checks_s"] = round(time.monotonic() - checks_started, 3)
        failures = [result for result in checker.results if not result["ok"]]
        report["status"] = "passed" if not failures else "failed"
        report["failed_checks"] = [result["name"] for result in failures]
        capabilities = next((result for result in checker.results if result["name"] == "tls_mtls_capabilities"), None)
        if capabilities is not None and capabilities.get("evidence"):
            report["identities"]["evaluator_epoch"] = capabilities["evidence"].get("evaluator_epoch", "")
            report["transport"]["tls"] = {
                "mutual": True,
                "client_certificate": "shared host mTLS identity",
                "server_san": "localhost/127.0.0.1",
                "plaintext_rejected": next(
                    (result["ok"] for result in checker.results if result["name"] == "plaintext_rejected"), False,
                ),
                "missing_client_certificate_rejected": next(
                    (result["ok"] for result in checker.results if result["name"] == "missing_client_cert"), False,
                ),
            }
        report["attempts"] = [
            {"request_id": result["name"], **{key: value for key, value in (result.get("evidence") or {}).items() if key in {"attempt_id", "lease_fence"}}}
            for result in checker.results
            if (result.get("evidence") or {}).get("attempt_id") or (result.get("evidence") or {}).get("lease_fence")
        ]
        report["cleanup"] = {"containers_left": docker.run("ps", "-a", "--filter", "name=taic-grpc-sim", "--format", "{{.Names}}").stdout.split(), "image_retained": image_id}
    except BaseException as exc:  # noqa: BLE001 - report the blocker instead of faking a pass
        report["status"] = "blocked"
        report["failure"] = f"{type(exc).__name__}: {exc}"[:4000]
        report["container_logs"] = docker.container_logs(sim.name)[-4000:]
    finally:
        if not keep:
            sim.stop(grace=5)
            sim.docker.remove(sim.name)
        report["timings"]["total_s"] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.npu_sim.grpc_sim_runner")
    date = time.strftime("%Y%m%d")
    parser.add_argument("--report", type=Path, default=ROOT / "runs" / f"grpc-sim-{date}" / "report.json")
    parser.add_argument("--keep", action="store_true", help="keep the simulated container for inspection")
    parser.add_argument("--fresh", action="store_true", help="discard previous container state/config before running")
    args = parser.parse_args(argv)
    if not os.environ.get("TAIC_DOCKER_GRPC_SIM") and not os.environ.get("TAIC_DOCKER_NPU_SIM"):
        print("set TAIC_DOCKER_GRPC_SIM=1 to run the simulated Docker acceptance", file=sys.stderr)
        return 2
    report = build_and_run(report_path=args.report, keep=args.keep, fresh=args.fresh)
    print(json.dumps({"status": report["status"], "report": str(args.report)}, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
