"""Phase 2D2 simulated gRPC/TLS acceptance: macOS host → Docker Linux → evaluator.

Runs the real external gRPC adapter over real TLS/mTLS against a Linux container
built from the test-only composition root, injects transport/auth/restart/key
rotation faults and writes ``runs/grpc-sim-<date>/report.json``.  No GPU, NPU,
NPU SDK, real model, SSH data plane or publication claim is involved.

This is the **legacy Metal/source contract** transport acceptance: it proves
TLS/mTLS, the six RPCs, restart durability and fault handling, and its fake
identity deliberately does not claim to be NPU.  NPU (fake/simulated/source)
traffic is covered by the npu-source contract in
``tests/shared/test_remote_npu_*.py``; the package name is historical.

Usage (explicit authorization):

    TAIC_DOCKER_GRPC_SIM=1 uv run python -m tests.support.npu_sim.grpc_sim_runner

The report is a diagnostic artifact: ``simulated=true`` and
``publication_allowed=false``.  It is not a search run, has no run_config.json
and never writes confirmed/catalog/brief state.

``status=passed`` requires every entry of :data:`REQUIRED_CHECKS` to have run and
passed *and* the post-cleanup evidence to be final.  Anything else is
``failed``/``blocked``; a missing required check is never reported as passed.
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
import threading
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
from tcenv.remote.revocation import write_revocation_file
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.v1 import evaluator_pb2_grpc as pb_grpc
from tests.support.v1_tasks import V1_TASK
from tests.support.npu_sim.tcp_proxy import FaultProxy
from tests.support.remote_pki import make_pki

ROOT = Path(__file__).resolve().parents[3]
IMAGE = "taic-remote-grpc-sim:local"
IMAGE_PREVIOUS = "taic-remote-grpc-sim:previous"
SITE = "sim-grpc-site"
DEVICE = "sim-grpc-device"
TARGET = "sim-grpc-target"
BUILD = "sim-grpc-build-1"
TASK_PACK = "b" * 64
CALIBRATION = "c" * 64
FINGERPRINT = "a" * 64
CONTAINER_PORT = 7443
MAIN_NAME = "taic-grpc-sim"
LIMITS_NAME = "taic-grpc-sim-limits"
SECRET_MARKER = "SECRET_CANDIDATE_MARKER_DO_NOT_LOG"
UNSIGNED_REQUEST = "sim-unsigned"
OLD_TOKEN = "grant-formal-selfcheck"
RENEWED_TOKEN = "grant-renewed"
NEAR_EXPIRY_TOKEN = "grant-near-expiry"
FAR_EXPIRED_TOKEN = "grant-far-expired"
REVOKE_ACTIVE_TOKEN = "grant-revoke-active"

#: The complete Phase 2D matrix this runner must execute.  Every name here has
#: to appear once, pass, and not be listed in ``not_run`` for status=passed.
REQUIRED_CHECKS: dict[str, str] = {
    "tls_mtls_capabilities": "tls",
    "plaintext_rejected": "tls",
    "missing_client_cert": "tls",
    "wrong_ca_rejected": "tls",
    "wrong_san_rejected": "tls",
    "untrusted_client_cert_rejected": "tls",
    "rotated_client_cert_accepted": "tls",
    "revoked_client_cert_rejected": "tls",
    "missing_grant_rejected": "authn",
    "unknown_grant_rejected": "authn",
    "expired_grant_rejected": "authn",
    "clock_skew_within_bound_accepted": "authn",
    "clock_skew_beyond_bound_rejected": "authn",
    "grant_scope_mismatch_rejected": "authz",
    "formal_selfcheck_scope_swap_rejected": "authz",
    "revoked_grant_rejected": "authz",
    "active_attempt_cancelled_on_revocation": "authz",
    "grant_renewal_reattach": "authz",
    "unknown_api_contract_rejected": "contract",
    "unknown_evidence_contract_rejected": "contract",
    "unknown_required_feature_rejected": "contract",
    "unsigned_result_rejected": "signature",
    "wrong_hmac_key_rejected": "signature",
    "revoked_hmac_key_rejected": "signature",
    "hmac_key_rotation_historical_verification": "signature",
    "hmac_key_revocation_quarantines_history": "signature",
    "retry_only_interrupted": "state-machine",
    "simulated_result_and_publication_gate": "protocol",
    "cancel_running_job": "state-machine",
    "duplicate_submit_receipt_dedup": "state-machine",
    "duplicate_getjob_no_billing": "quota",
    "selfcheck_quota": "quota",
    "timed_lease_single_owner": "quota",
    "fairness_two_runs_selfcheck_not_starved": "quota",
    "cache_policy_disabled": "protocol",
    "selfcheck_authoritative_evidence": "evidence",
    "queue_depth_limit": "limits",
    "watch_stream_limit": "limits",
    "job_store_byte_limit": "limits",
    "auth_failure_rate_limit": "limits",
    "secret_redaction": "security",
    "connection_reset_reconciles_via_getjob": "faults",
    "half_open_deadline_then_getjob": "faults",
    "slow_consumer_watch_reconnect": "faults",
    "container_restart_explicit_retry": "recovery",
    "stale_worker_fail_closed": "recovery",
    "backup_restore_quarantine": "recovery",
    "rollback_previous_image": "recovery",
    "cleanup": "cleanup",
}

NOT_RUN = (
    "real GPU",
    "real NPU",
    "NPU SDK installation",
    "real model inference",
    "SSH transport data plane",
    "hardware performance or publication numbers",
    "HA/multi-replica",
    "CRL/OCSP network fetching (static restart-loaded certificate deny list only)",
    "schema-breaking downgrade (the previous product commit is schema compatible)",
)


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
        self.events: list[str] = []
        self._events_process: subprocess.Popen | None = None
        self._events_thread: threading.Thread | None = None
        self._events_stop = threading.Event()

    def start_events(self) -> None:
        try:
            self._events_process = subprocess.Popen(
                ["docker", "events", "--format", "{{.Time}} {{.Action}} {{.Actor.Attributes.name}} "
                 "exit={{.Actor.Attributes.exitCode}} oom={{.Actor.Attributes.oom}}",
                 "--filter", "type=container", "--filter", "event=die", "--filter", "event=oom", "--filter", "event=kill"],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=self.env,
            )
        except OSError:
            return

        def consume() -> None:
            process = self._events_process
            if process is None or process.stdout is None:
                return
            for line in process.stdout:
                if self._events_stop.is_set():
                    break
                self.events.append(line.strip())

        self._events_thread = threading.Thread(target=consume, name="tcenv-docker-events", daemon=True)
        self._events_thread.start()

    def stop_events(self) -> None:
        self._events_stop.set()
        if self._events_process is not None:
            try:
                self._events_process.terminate()
            except OSError:
                pass
        if self._events_thread is not None:
            self._events_thread.join(timeout=2)

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

    def image_revision(self, image: str) -> str:
        label = self.run(
            "image", "inspect", "--format", '{{index .Config.Labels "tcenv.sim.revision"}}', image, check=False,
        ).stdout.strip()
        if label:
            return label
        raw = self.run("image", "inspect", "--format", "{{json .Config.Env}}", image, check=False).stdout.strip()
        try:
            values = json.loads(raw)
        except ValueError:
            return ""
        for value in values or []:
            if str(value).startswith("TCENV_SIM_IMAGE_REVISION="):
                return str(value).split("=", 1)[1]
        return ""

    def container_logs(self, name: str) -> str:
        completed = self.run("logs", name, check=False)
        return completed.stdout + completed.stderr

    def remove(self, name: str) -> None:
        self.run("rm", "-f", name, check=False)

    def volume_remove(self, name: str) -> None:
        self.run("volume", "rm", "-f", name, check=False)

    def volume_create(self, name: str, *, uid: int, gid: int) -> None:
        """Create a writable named volume for the non-root frontend user."""

        self.volume_remove(name)
        self.run("volume", "create", name)
        self.run(
            "run", "--rm", "--user", "0:0", "--mount", f"type=volume,src={name},dst=/state",
            "--entrypoint", "chown", IMAGE, "-R", f"{uid}:{gid}", "/state",
        )

    def volume_query(self, volume: str, query: str, params=()) -> list[tuple]:
        script = (
            "import sqlite3,json,sys;"
            "con=sqlite3.connect('file:/state/jobs.sqlite3?mode=ro',uri=True);"
            "print(json.dumps(con.execute(sys.argv[1],json.loads(sys.argv[2])).fetchall()))"
        )
        completed = self.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=volume,src={volume},dst=/state",
            "--entrypoint", "python3", IMAGE, "-c", script, query, json.dumps(list(params)),
        )
        return [tuple(row) for row in json.loads(completed.stdout)]

    def volume_text(self, volume: str, path: str) -> str:
        completed = self.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=volume,src={volume},dst=/state",
            "--entrypoint", "cat", IMAGE, path,
        )
        return completed.stdout

    def acceptance_containers(self) -> list[str]:
        completed = self.run("ps", "-a", "--filter", "name=taic-grpc-sim", "--format", "{{.Names}}", check=False)
        return [line.strip() for line in completed.stdout.split() if line.strip()]


def _git(*arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {completed.stdout}{completed.stderr}")
    return completed.stdout.strip()


def build_previous_image(docker: "Docker", root: Path, *, label: str) -> dict:
    """Build the previous product commit as an immutable rollback image."""

    latest = _git("log", "--format=%H", "-n", "2", "--", "tcenv/remote").split()
    ref = os.environ.get("TAIC_SIM_PREVIOUS_REF") or (latest[1] if len(latest) > 1 else "HEAD~1")
    commit = _git("rev-parse", ref)
    subject = _git("log", "-1", "--format=%s", ref)
    label = f"{label}@{commit[:12]}"
    product_diff = [line for line in _git("diff", "--name-only", ref, "--", "tcenv").splitlines() if line]
    worktree = root / "previous-src"
    _git("worktree", "prune", check=False)
    if worktree.exists():
        shutil.rmtree(worktree)
    _git("worktree", "add", "--detach", str(worktree), commit)
    try:
        docker.run(
            "build", "-f", str(worktree / "tests/support/docker/Dockerfile.remote-grpc-sim"),
            "-t", IMAGE_PREVIOUS, "--label", label, str(worktree), timeout=1800,
        )
    finally:
        _git("worktree", "remove", "--force", str(worktree), check=False)
        _git("worktree", "prune", check=False)
    return {
        "ref": ref,
        "commit": commit,
        "subject": subject,
        "product_diff_files": product_diff,
        "image_id": docker.image_id(IMAGE_PREVIOUS),
    }


def volume_names(root: Path) -> dict[str, str]:
    """Per-report persistent volume names (never shared between runs)."""

    suffix = hashlib.sha1(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:10]
    return {name: f"taic-sim-{suffix}-{name}" for name in ("state", "backup", "restored", "limits")}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(
    request_id: str, source: str, *, kind: str = "formal_eval", mode: str = "quick",
    run_id: str = "run-1", session_id: str = "session-1", queue_s: float = 10.0,
    candidate_s: float = 15.0, progress_path: Path | None = None,
    cache_policy: str = "reuse",
) -> EvaluationRequest:
    return EvaluationRequest(
        task=V1_TASK, dtype="fp16", size="small", source=source, kind=kind, mode=mode,
        run_id=run_id, session_id=session_id, request_id=request_id, target_id=TARGET,
        task_fingerprint=FINGERPRINT, queue_timeout_s=queue_s,
        candidate_timeout_s=candidate_s, progress_path=progress_path,
        cache_policy=cache_policy,
    )


def _grant(token: str, *, run_id: str = "run-1", session_id: str = "session-1", **options) -> SessionGrant:
    return SessionGrant(
        token=token, run_id=run_id, session_id=session_id, target_id=TARGET,
        subject="host", **options,
    )


class Sim:
    def __init__(self, *, root: Path, docker: Docker) -> None:
        self.docker = docker
        self.root = root
        self.config = root / "container-config"
        self.config.mkdir(parents=True, exist_ok=True)
        self.config.chmod(0o700)
        # SQLite WAL and its shared-memory index must live on a Linux
        # filesystem: a Docker Desktop bind mount is not coherent with the
        # container's mmap and can SIGBUS the frontend.  Named volumes also
        # match the production "dedicated persistent volume" contract.
        names = volume_names(root)
        self.state_volume = names["state"]
        self.backup_volume = names["backup"]
        self.restored_volume = names["restored"]
        self.limits_volume = names["limits"]
        for volume in (self.state_volume, self.backup_volume, self.restored_volume, self.limits_volume):
            docker.volume_create(volume, uid=os.getuid(), gid=os.getgid())
        self.pki = make_pki(self.config / "pki")
        self.key1 = new_key()
        self.key2 = new_key()
        self.key_id1 = f"{SITE}/key-1"
        self.key_id2 = f"{SITE}/key-2"
        self.keyring_path = self.config / "keyring.json"
        self.write_keyring({self.key_id1: self.key1}, active=self.key_id1)
        self.revocation_path = self.config / "revoked-client-certs.json"
        write_revocation_file(self.revocation_path, frozenset())
        self.grants: list[SessionGrant] = [
            _grant(OLD_TOKEN, max_timed=1, max_gpu_s=120.0),
            _grant("grant-run-2", run_id="run-2"),
            _grant("grant-expired", session_id="session-expired", expires_at=time.time() - 120),
            _grant("grant-revoke-me", session_id="session-revoke"),
            _grant(REVOKE_ACTIVE_TOKEN, session_id="session-revoke-active"),
            _grant("grant-formal-only", run_id="run-3", session_id="session-kind", allowed_kinds=frozenset({"formal_eval"})),
            _grant("grant-selfcheck-only", run_id="run-3", session_id="session-kind", allowed_kinds=frozenset({"selfcheck"})),
            _grant("grant-fair", run_id="run-2", session_id="session-fair"),
            _grant("grant-lease", run_id="run-2", session_id="session-lease", max_timed=1),
            _grant("grant-billing", run_id="run-2", session_id="session-billing", max_timed=1),
        ]
        self.grants_path = self.config / "grants.json"
        self.write_grants(self.grants)
        self.default_token = OLD_TOKEN
        self._keepalive: GrpcEvaluatorFactory | None = None
        self.name = MAIN_NAME
        self.container_name = MAIN_NAME
        self.running_names: set[str] = set()
        self.port = 0
        self.worker_env = {
            "FAKE_JOB_COUNTER": "/state/worker-count.txt",
            "FAKE_JOB_RUNNING": "/state/running.json",
            "FAKE_JOB_HOLD_DIR": "/state/hold-once",
            "FAKE_JOB_FAIL_DIR": "/state/fail-once",
            "FAKE_JOB_ORDER": "/state/worker-order.jsonl",
        }
        self.started = False

    # -- configuration files -------------------------------------------------
    def write_grants(self, grants: list[SessionGrant]) -> None:
        self.grants = list(grants)
        write_grants_file(self.grants_path, self.grants)

    def grant(self, token: str) -> SessionGrant | None:
        return next((grant for grant in self.grants if grant.token == token), None)

    def write_keyring(self, keys: dict[str, bytes], *, revoked=(), active: str = "") -> None:
        write_keyring_file(self.keyring_path, KeyringFile(dict(keys), frozenset(revoked), active))

    def revoke_client_certificates(self, fingerprints: frozenset[str]) -> None:
        write_revocation_file(self.revocation_path, fingerprints)

    # -- container lifecycle -------------------------------------------------
    def start(
        self, *, image: str = IMAGE, volume: str | None = None, port: int | None = None,
        name: str | None = None, extra_args: tuple[str, ...] = (), legacy: bool = False,
    ) -> int:
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
            "--mount", f"type=volume,src={volume or self.state_volume},dst=/state",
            image,
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
        # The previous image predates the acceptance-only flags; start it with
        # only the flags that release actually implements.
        if not legacy:
            arguments.extend([
                "--revoked-client-certificates", "/etc/tcenv/revoked-client-certs.json",
                "--unsigned-result-for", UNSIGNED_REQUEST,
                "--selfcheck-age-s", "1.0",
            ])
        arguments.extend(extra_args)
        self.docker.run(*arguments)
        self.running_names.add(name)
        self.started = name == self.name
        return port

    def stop(self, *, grace: int = 2, name: str | None = None) -> None:
        target = name or self.name
        self.docker.run("stop", "-t", str(grace), target, check=False, timeout=grace + 30)
        self.docker.remove(target)
        self.running_names.discard(target)
        if target == self.name:
            self.started = False

    def kill(self, *, name: str | None = None) -> None:
        target = name or self.name
        self.docker.run("kill", target, check=False)
        self.docker.remove(target)
        self.running_names.discard(target)
        if target == self.name:
            self.started = False

    def wait_ready(self, *, timeout: float = 60.0, token: str | None = None, port: int | None = None, name: str | None = None) -> None:
        token = token or self.default_token
        endpoint = f"127.0.0.1:{port or self.port}"
        deadline = time.monotonic() + timeout
        last: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                credentials = load_client_credentials(
                    ca_file=self.pki.ca_cert, certificate_file=self.pki.client_cert,
                    private_key_file=self.pki.client_key,
                )
                channel = grpc.secure_channel(
                    endpoint, credentials,
                    options=[("grpc.ssl_target_name_override", "localhost")],
                )
                try:
                    stub = pb_grpc.EvaluatorStub(channel)
                    stub.GetCapabilities(
                        pb.CapabilitiesRequest(
                            api_contract_ids=[contracts.API_CONTRACT_ID],
                            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
                            client_nonce="ready-" + os.urandom(8).hex(),
                            expected=pb.ExpectedEvaluator(
                                site_id=SITE, device_id=DEVICE, target_id=TARGET,
                                worker_build=BUILD, task_pack_digest=TASK_PACK,
                            ),
                        ),
                        metadata=(("authorization", f"Bearer {token}"),), timeout=5,
                    )
                    self._refresh_keepalive(name or self.name)
                    return
                finally:
                    channel.close()
            except Exception as exc:  # noqa: BLE001 - readiness loop
                last = exc
                time.sleep(0.3)
        logs = self.docker.container_logs(name or self.name)[-3000:]
        raise RuntimeError(f"simulated evaluator did not become ready: {last}\n{logs}")

    # -- clients -------------------------------------------------------------
    def _refresh_keepalive(self, name: str) -> None:
        """Hold one channel open per host process, as production does."""

        if name != self.name:
            return
        try:
            if self._keepalive is not None:
                self._keepalive.close()
            self._keepalive = self.factory()
            self._keepalive.ensure_connected()
        except Exception:  # noqa: BLE001 - the readiness probe already succeeded
            self._keepalive = None

    def close_keepalive(self) -> None:
        if self._keepalive is not None:
            self._keepalive.close()
            self._keepalive = None

    def factory(
        self, *, token: str | None = None, run_id: str = "run-1",
        keyring: SignatureKeyring | None = None, endpoint: str | None = None,
        rpc_timeout_s: float = 15.0, watch_slice_s: float = 5.0, job_margin_s: float = 20.0,
        port: int | None = None,
    ) -> GrpcEvaluatorFactory:
        return grpc_evaluator_factory_from_config(
            endpoint=endpoint or f"127.0.0.1:{port or self.port}",
            token_provider=lambda _run, _session: token or self.default_token,
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

    def raw_channel(self, *, ca=None, cert=None, key=None, server_name: str = "localhost", endpoint: str | None = None, insecure: bool = False, port: int | None = None):
        if insecure:
            return grpc.insecure_channel(endpoint or f"127.0.0.1:{port or self.port}")
        credentials = load_client_credentials(
            ca_file=ca or self.pki.ca_cert,
            certificate_file=self.pki.client_cert if cert is None else cert,
            private_key_file=self.pki.client_key if key is None else key,
        )
        return grpc.secure_channel(
            endpoint or f"127.0.0.1:{port or self.port}", credentials,
            options=[("grpc.ssl_target_name_override", server_name)],
        )

    def raw_capabilities(
        self, token: str | None = None, *, features=(), api=contracts.API_CONTRACT_ID,
        evidence=contracts.EVIDENCE_CONTRACT_ID, channel=None, metadata=True,
        cert=None, key=None, port: int | None = None,
    ):
        own = channel is None
        channel = channel or self.raw_channel(cert=cert, key=key, port=port)
        try:
            stub = pb_grpc.EvaluatorStub(channel)
            call_metadata = (("authorization", f"Bearer {token or self.default_token}"),) if metadata else ()
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

    def expected(self, capabilities) -> object:
        return gc.expected_identities_from_provenance(
            gc.provenance_from_proto(capabilities.provenance, require_attempt=False)
        )

    def raw_get_job(self, request: EvaluationRequest, *, token: str | None = None, port: int | None = None, cert=None, key=None):
        from tcenv.remote.job_types import ExternalEvaluationRequest

        external = ExternalEvaluationRequest.from_legacy(request)
        capabilities = self.raw_capabilities(token, port=port, cert=cert, key=key)
        channel = self.raw_channel(port=port, cert=cert, key=key)
        try:
            stub = pb_grpc.EvaluatorStub(channel)
            return stub.GetJob(
                pb.JobRef(
                    request_id=external.request_id,
                    request_digest=external.request_digest,
                    expected=gc.expected_identities_to_proto(self.expected(capabilities)),
                ),
                metadata=(("authorization", f"Bearer {token or self.default_token}"),), timeout=15,
            )
        finally:
            channel.close()

    # -- durable-state inspection -------------------------------------------
    # Docker Desktop's bind mounts are not coherent with the container's WAL
    # mmap: a host-side read-only SQLite connection can invalidate the
    # container's shared-memory index (observed as SIGBUS).  Durable state is
    # therefore queried *inside* the container; the direct file reader is only
    # used after the container is gone (post-cleanup evidence).
    _SQL_SCRIPT = (
        "import sqlite3,json,sys;"
        "con=sqlite3.connect('file:/state/jobs.sqlite3?mode=ro',uri=True);"
        "print(json.dumps(con.execute(sys.argv[1],json.loads(sys.argv[2])).fetchall()))"
    )

    def _running_container(self, name: str | None) -> str:
        if name is not None:
            return name if name in self.running_names else ""
        return self.name if self.name in self.running_names else ""

    def sql(self, query: str, params=(), *, container: str | None = None, volume: str | None = None) -> list[tuple]:
        image = self._running_container(container)
        if image:
            completed = self.docker.run(
                "exec", image, "python3", "-c", self._SQL_SCRIPT, query, json.dumps(list(params)),
            )
            return [tuple(row) for row in json.loads(completed.stdout)]
        return self.docker.volume_query(volume or self.state_volume, query, params)

    def attempts(self, request_id: str, *, container: str | None = None, volume: str | None = None) -> int:
        rows = self.sql("SELECT COUNT(*) FROM attempts WHERE request_id=?", (request_id,), container=container, volume=volume)
        return int(rows[0][0])

    def job_state(self, request_id: str, *, container: str | None = None) -> dict[str, object]:
        rows = self.sql(
            "SELECT state,quarantined,recovery_error FROM jobs WHERE request_id=?",
            (request_id,), container=container,
        )
        if not rows:
            raise AssertionError(f"request {request_id!r} is not durable")
        return {"state": rows[0][0], "quarantined": bool(rows[0][1]), "recovery_error": rows[0][2]}

    def running_attempts(self, *, container: str | None = None) -> int:
        rows = self.sql("SELECT COUNT(*) FROM attempts WHERE state='running'", (), container=container)
        return int(rows[0][0])

    def quota(self, run_id: str, session_id: str, kind: str = "selfcheck", *, container: str | None = None) -> dict[str, int]:
        rows = self.sql(
            "SELECT reserved_count,used_count FROM quotas WHERE run_id=? AND session_id=? AND kind=? AND target_id=?",
            (run_id, session_id, kind, TARGET), container=container,
        )
        if not rows:
            return {"reserved_count": 0, "used_count": 0}
        return {"reserved_count": int(rows[0][0]), "used_count": int(rows[0][1])}

    def request_document(self, request_id: str, *, container: str | None = None) -> dict:
        rows = self.sql("SELECT request_json FROM jobs WHERE request_id=?", (request_id,), container=container)
        if not rows:
            raise AssertionError(f"request {request_id!r} is not durable")
        return json.loads(rows[0][0])

    def result_descriptor(self, request_id: str, *, container: str | None = None) -> dict:
        rows = self.sql(
            "SELECT report_descriptor_json,evidence_descriptor_json,receipt_json,signature_json FROM results WHERE request_id=?",
            (request_id,), container=container,
        )
        if not rows:
            raise AssertionError(f"request {request_id!r} has no durable result")
        return {
            "report": json.loads(rows[0][0]),
            "evidence": None if rows[0][1] is None else json.loads(rows[0][1]),
            "receipt": json.loads(rows[0][2]),
            "signature": None if rows[0][3] is None else json.loads(rows[0][3]),
        }

    def state_text(self, name: str) -> str:
        if self.name in self.running_names:
            return self.docker.run("exec", self.name, "cat", f"/state/{name}").stdout
        return self.docker.volume_text(self.state_volume, f"/state/{name}")

    def worker_count(self) -> int:
        try:
            value = self.state_text("worker-count.txt").strip()
        except RuntimeError:
            return 0
        return int(value) if value else 0

    def worker_order(self) -> list[dict]:
        try:
            text = self.state_text("worker-order.jsonl")
        except RuntimeError:
            return []
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def _grpc_code(exc: BaseException) -> str:
    if isinstance(exc, grpc.RpcError):
        return exc.code().name
    return getattr(exc, "code", type(exc).__name__)


def _conformance_request(*, run_id: str = "run-1", session_id: str = "session-1", request_id: str, source: str, kind: str = "formal_eval", mode: str = "quick", **options) -> EvaluationRequest:
    return _request(request_id, source, kind=kind, mode=mode, run_id=run_id, session_id=session_id, **options)


class Checker:
    def __init__(self, sim: Sim) -> None:
        self.sim = sim
        self.results: list[dict] = []

    def check(self, name: str, function, category: str = "") -> dict:
        started = time.monotonic()
        try:
            evidence = function() or {}
            result = {"name": name, "category": category or REQUIRED_CHECKS.get(name, ""), "ok": True, "evidence": evidence}
        except Exception as exc:  # noqa: BLE001 - one failed check must not stop the matrix
            result = {"name": name, "category": category or REQUIRED_CHECKS.get(name, ""), "ok": False, "detail": f"{type(exc).__name__}: {exc}"[:2000]}
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        try:
            if self.sim.started:
                result["container_status"] = self.sim.docker.run(
                    "inspect", "--format", "{{.State.Status}}/{{.State.ExitCode}}", self.sim.name, check=False,
                ).stdout.strip()
        except Exception:  # noqa: BLE001 - diagnostics only
            result["container_status"] = "unknown"
        if any(existing["name"] == name for existing in self.results):
            raise AssertionError(f"duplicate acceptance check name: {name}")
        self.results.append(result)
        return result

    def expect_grpc_error(self, name: str, function, expected: set[str], category: str = "tls") -> dict:
        started = time.monotonic()
        try:
            function()
        except Exception as exc:  # noqa: BLE001
            code = _grpc_code(exc)
            if code not in expected:
                raise AssertionError(f"expected {sorted(expected)}, got {code}: {exc}")
            return {"name": name, "category": category, "ok": True, "evidence": {"grpc_code": code}, "elapsed_s": round(time.monotonic() - started, 3)}
        raise AssertionError(f"{name} unexpectedly succeeded")


def _threaded(formal_client, request: EvaluationRequest) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def run() -> None:
        try:
            if request.kind == "selfcheck":
                formal_client.selfcheck(request)
            else:
                formal_client.formal(request)
        except BaseException as exc:  # noqa: BLE001 - the probe observes the failure
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, errors


def _wait_for_running(sim: Sim, request_id: str, *, timeout: float = 15.0, container: str | None = None, errors: list[BaseException] | None = None) -> None:
    deadline = time.monotonic() + timeout
    current = "missing"
    while time.monotonic() < deadline:
        try:
            current = sim.job_state(request_id, container=container)["state"]
        except AssertionError:
            current = "missing"
        if current == "running":
            return
        time.sleep(0.1)
    details = [f"{_grpc_code(error)}: {str(error)[:200]}" for error in (errors or [])]
    try:
        jobs = sim.sql("SELECT request_id,state FROM jobs ORDER BY accepted_at_ms")
    except Exception:  # noqa: BLE001 - diagnostics only
        jobs = []
    raise AssertionError(
        f"{request_id} never reached the running state (last={current}, errors={details}, jobs={jobs})"
    )


def main_matrix(sim: Sim, checker: Checker, *, state_root: Path) -> None:
    """Execute the checks that use the primary container before rotation phases."""

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
                factory.formal(_conformance_request(request_id="sim-retry", source="FAIL_ONCE"))
            except EvaluationUnavailable as exc:
                if not exc.retryable:
                    raise AssertionError(f"first attempt was not retryable: {exc.code}")
            else:
                raise AssertionError("FAIL_ONCE unexpectedly completed on attempt 1")
            resumed = factory.formal_resume(_conformance_request(request_id="sim-retry", source="FAIL_ONCE"))
            attempts = sim.attempts("sim-retry")
            if attempts != 2:
                raise AssertionError(f"expected 2 attempts, found {attempts}")
            return {"attempt_id": resumed.receipt.get("attempt_id"), "lease_fence": resumed.receipt.get("lease_fence"), "attempts": attempts}
        finally:
            factory.close()
    checker.check("retry_only_interrupted", retry_interrupted, "state-machine")
    checker.check("simulated_result_and_publication_gate", formal_ok, "protocol")

    def duplicate_submit() -> dict:
        before = sim.worker_count()
        factory = sim.factory()
        try:
            first = factory.formal(_conformance_request(request_id="sim-dedup", source="OK"))
            second = factory.formal(_conformance_request(request_id="sim-dedup", source="OK"))
            after = sim.worker_count()
            if first.receipt["attempt_id"] != second.receipt["attempt_id"]:
                raise AssertionError("an idempotent Submit replay created a new attempt")
            if first.receipt["receipt_id"] != second.receipt["receipt_id"]:
                raise AssertionError("an idempotent Submit replay produced a different receipt")
            if after - before != 1:
                raise AssertionError(f"the replayed request executed {after - before} times")
            if sim.attempts("sim-dedup") != 1:
                raise AssertionError("duplicate Submit created a second durable attempt")
            return {"attempt_id": first.receipt["attempt_id"], "receipt_id": first.receipt["receipt_id"], "executions": after - before}
        finally:
            factory.close()
    checker.check("duplicate_submit_receipt_dedup", duplicate_submit, "state-machine")

    def cancel_job() -> dict:
        factory = sim.factory()
        try:
            thread, errors = _threaded(factory, _conformance_request(request_id="sim-cancel", source="SLEEP", candidate_s=30))
            _wait_for_running(sim, "sim-cancel", errors=errors)
            factory.cancel("sim-cancel")
            thread.join(timeout=20)
            if thread.is_alive():
                raise AssertionError("cancel did not finish the client")
            state = sim.attempts("sim-cancel")
            if state != 1:
                raise AssertionError(f"cancel changed attempt count to {state}")
            if sim.job_state("sim-cancel")["state"] != "cancelled":
                raise AssertionError("the durable job is not cancelled")
            return {"client_error": _grpc_code(errors[0]) if errors else "none", "attempts": state}
        finally:
            factory.close()
    checker.check("cancel_running_job", cancel_job, "state-machine")

    # TLS negatives
    def _tls(description: str, function, expected: set[str], category: str = "tls") -> dict:
        return checker.expect_grpc_error(description, function, expected, category)
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
        checker.results.append(_tls(name, function, {"UNAVAILABLE", "UNAUTHENTICATED", "PERMISSION_DENIED"}))

    def rotated_cert() -> dict:
        capabilities = sim.raw_capabilities(cert=sim.pki.rotated_client_cert, key=sim.pki.rotated_client_key)
        return {"capability_id": capabilities.capability_id, "client_certificate": "rotated same-CA client leaf"}
    checker.check("rotated_client_cert_accepted", rotated_cert, "tls")

    # Authentication and grant scope
    checker.results.append(_tls("missing_grant_rejected", lambda: sim.raw_capabilities(metadata=False), {"UNAUTHENTICATED"}, "authn"))
    checker.results.append(_tls("unknown_grant_rejected", lambda: sim.raw_capabilities(token="not-a-grant"), {"UNAUTHENTICATED"}, "authn"))
    checker.results.append(_tls("expired_grant_rejected", lambda: sim.raw_capabilities(token="grant-expired"), {"UNAUTHENTICATED"}, "authn"))

    def wrong_scope() -> dict:
        capabilities = sim.raw_capabilities(token="grant-run-2")
        channel = sim.raw_channel()
        try:
            stub = pb_grpc.EvaluatorStub(channel)
            probe = _request("sim-wrong-scope", "OK", run_id="run-1", session_id="session-1")
            from tcenv.remote.job_types import ExternalEvaluationRequest
            external = ExternalEvaluationRequest.from_legacy(probe)
            submit = pb.SubmitRequest(
                request=gc.request_to_proto(external),
                request_digest=external.request_digest,
                expected=gc.expected_identities_to_proto(sim.expected(capabilities)),
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

    def scope_swap() -> dict:
        outcomes = {}
        formal = sim.factory(token="grant-selfcheck-only", run_id="run-3")
        try:
            try:
                formal.formal(_conformance_request(request_id="sim-swap-formal", source="OK", run_id="run-3", session_id="session-kind"))
            except EvaluationUnavailable as exc:
                if exc.code != "permission_denied":
                    raise AssertionError(f"formal under a selfcheck-only grant returned {exc.code}")
                outcomes["formal_with_selfcheck_grant"] = exc.code
            else:
                raise AssertionError("a selfcheck-only grant submitted a formal job")
        finally:
            formal.close()
        selfcheck = sim.factory(token="grant-formal-only", run_id="run-3")
        try:
            try:
                selfcheck.selfcheck(_conformance_request(request_id="sim-swap-selfcheck", source="OK", kind="selfcheck", run_id="run-3", session_id="session-kind"))
            except EvaluationUnavailable as exc:
                if exc.code != "permission_denied":
                    raise AssertionError(f"selfcheck under a formal-only grant returned {exc.code}")
                outcomes["selfcheck_with_formal_grant"] = exc.code
            else:
                raise AssertionError("a formal-only grant submitted a selfcheck job")
        finally:
            selfcheck.close()
        return outcomes
    checker.check("formal_selfcheck_scope_swap_rejected", scope_swap, "authz")

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

    # HMAC key negatives (host-side)
    def wrong_keyring() -> dict:
        wrong = SignatureKeyring({sim.key_id1: b"w" * 32})
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
        keyring.revoke(sim.key_id1)
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

    def unsigned_result() -> dict:
        factory = sim.factory()
        try:
            try:
                factory.formal(_conformance_request(request_id=UNSIGNED_REQUEST, source="OK " + SECRET_MARKER))
            except EvaluationUnavailable as exc:
                if exc.code != "result_invalid":
                    raise AssertionError(f"an unsigned result returned {exc.code}")
                descriptor = sim.result_descriptor(UNSIGNED_REQUEST)
                if descriptor["signature"] is None:
                    raise AssertionError("the durable result lost its signature; only the RPC envelope should be unsigned")
                return {"code": exc.code, "durable_result_signed": True}
            raise AssertionError("an unsigned result was accepted")
        finally:
            factory.close()
    checker.check("unsigned_result_rejected", unsigned_result, "signature")

    def secret_redaction() -> dict:
        probe = "SECRET-GRANT-TOKEN-NEVER-ECHO"
        message = ""
        try:
            sim.raw_capabilities(token=probe)
        except grpc.RpcError as exc:
            message = exc.details() or ""
        logs = sim.docker.container_logs(sim.name)
        forbidden = {
            "grant token": [grant.token for grant in sim.grants],
            "hmac key": [sim.key1.hex()],
            "candidate source": [SECRET_MARKER],
            "probe token": [probe],
        }
        leaks = {}
        for label, values in forbidden.items():
            hits = [value for value in values if value and (value in logs or (label == "probe token" and value in message))]
            if hits:
                leaks[label] = len(hits)
        if leaks:
            raise AssertionError(f"secrets leaked into frontend logs/errors: {leaks}")
        return {"log_bytes": len(logs), "checked": sorted(forbidden), "error_echoed_token": False}
    checker.check("secret_redaction", secret_redaction, "security")

    # Cache policy and selfcheck evidence
    def cache_disabled() -> dict:
        factory = sim.factory()
        try:
            result = factory.formal(_conformance_request(request_id="sim-cache-disabled", source="OK", cache_policy="disabled"))
            document = sim.request_document("sim-cache-disabled")
            if document.get("cache_policy") != "disabled":
                raise AssertionError(f"durable request cache_policy is {document.get('cache_policy')!r}")
            if result.report.get("status") != "ok":
                raise AssertionError("cache_policy=disabled did not complete normally")
            return {"cache_policy": document["cache_policy"], "attempt_id": result.receipt["attempt_id"], "status": result.report.get("status")}
        finally:
            factory.close()
    checker.check("cache_policy_disabled", cache_disabled, "protocol")

    def selfcheck_evidence() -> dict:
        factory = sim.factory(token="grant-fair", run_id="run-2")
        try:
            result = factory.selfcheck(_conformance_request(request_id="sim-selfcheck-evidence", source="OK", kind="selfcheck", run_id="run-2", session_id="session-fair"))
            descriptor = sim.result_descriptor("sim-selfcheck-evidence")
            evidence = result.evidence if isinstance(result.evidence, dict) else {}
            if evidence.get("candidate_sha256") != _request("x", "OK").candidate_sha256:
                raise AssertionError("selfcheck evidence does not bind the exact candidate bytes")
            if descriptor["evidence"] is None or descriptor["evidence"]["schema_id"] != gc.EVIDENCE_SCHEMA_ID:
                raise AssertionError("durable selfcheck evidence descriptor is missing or has the wrong schema")
            if result.receipt.get("signature_algorithm") != contracts.RESULT_SIGNATURE_ALGORITHM:
                raise AssertionError("selfcheck evidence is not bound by the result signature")
            if result.provenance.measurement_kind != "simulated" or result.provenance.publication_allowed:
                raise AssertionError("selfcheck result claims a publishable measurement")
            return {
                "evidence_schema": descriptor["evidence"]["schema_id"],
                "candidate_sha256": evidence["candidate_sha256"],
                "signature_key_id": result.receipt.get("signature_key_id"),
                "measurement_kind": result.provenance.measurement_kind,
            }
        finally:
            factory.close()
    checker.check("selfcheck_authoritative_evidence", selfcheck_evidence, "evidence")

    # Quota, billing and scheduling
    def timed_getjob_billing() -> dict:
        factory = sim.factory(token="grant-billing", run_id="run-2")
        try:
            request = _conformance_request(request_id="sim-billing", source="OK", kind="selfcheck", mode="timed", run_id="run-2", session_id="session-billing")
            result = factory.selfcheck(request)
            before = sim.quota("run-2", "session-billing")
            for _ in range(5):
                snapshot = factory.get_job(request)
                if gc.state_from_proto(snapshot.state) != "completed":
                    raise AssertionError("GetJob did not return the completed job")
            after = sim.quota("run-2", "session-billing")
            if after != before or after["used_count"] != 1 or after["reserved_count"] != 0:
                raise AssertionError(f"repeated GetJob changed quota: before={before} after={after}")
            try:
                factory.selfcheck(_conformance_request(request_id="sim-billing-2", source="OK", kind="selfcheck", mode="timed", run_id="run-2", session_id="session-billing"))
            except EvaluationUnavailable as exc:
                if exc.code != "resource_exhausted":
                    raise AssertionError(f"the timed quota was not enforced: {exc.code}")
            else:
                raise AssertionError("a second timed selfcheck reused the same quota reservation")
            return {"attempt_id": result.receipt["attempt_id"], "quota": after, "getjob_calls": 5}
        finally:
            factory.close()
    checker.check("duplicate_getjob_no_billing", timed_getjob_billing, "quota")

    def selfcheck_quota() -> dict:
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
    checker.check("selfcheck_quota", selfcheck_quota, "quota")

    def timed_lease() -> dict:
        formal = sim.factory()
        selfcheck = sim.factory(token="grant-lease", run_id="run-2")
        running: list[int] = []
        stop = threading.Event()

        def sample() -> None:
            while not stop.is_set():
                try:
                    running.append(sim.running_attempts())
                except Exception:  # noqa: BLE001 - a sampling probe, not a verdict
                    pass
                time.sleep(0.1)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        try:
            thread, _errors = _threaded(formal, _request("sim-lease-formal", "SLEEP", candidate_s=4))
            _wait_for_running(sim, "sim-lease-formal", errors=_errors)
            selfcheck_thread, selfcheck_errors = _threaded(
                selfcheck,
                _request("sim-lease-selfcheck", "OK", kind="selfcheck", mode="timed", run_id="run-2", session_id="session-lease"),
            )
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                try:
                    queued_state = sim.job_state("sim-lease-selfcheck")["state"]
                except AssertionError:
                    queued_state = "missing"
                if queued_state == "queued":
                    break
                if queued_state == "running":
                    raise AssertionError("the timed selfcheck acquired the lease while a formal attempt was running")
                time.sleep(0.02)
            if queued_state != "queued":
                raise AssertionError(
                    f"the timed selfcheck never queued behind the lease: {queued_state} "
                    f"errors={[_grpc_code(e) for e in selfcheck_errors]}"
                )
            thread.join(timeout=30)
            if thread.is_alive():
                raise AssertionError("the formal lease probe did not finish")
            selfcheck_thread.join(timeout=25)
            if selfcheck_thread.is_alive():
                raise AssertionError("the timed selfcheck did not finish after the lease was released")
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline and sim.job_state("sim-lease-selfcheck")["state"] != "completed":
                time.sleep(0.05)
            if sim.job_state("sim-lease-selfcheck")["state"] != "completed":
                raise AssertionError("the timed selfcheck never acquired the freed lease")
            if selfcheck_errors:
                raise AssertionError(f"the timed selfcheck failed instead of waiting: {[_grpc_code(e) for e in selfcheck_errors]}")
            if max(running or [0]) > 1:
                raise AssertionError(f"two attempts held the device lease: max={max(running)}")
            return {
                "queued_while_formal_running": True,
                "max_concurrent_running": max(running or [0]),
                "formal_terminal": sim.job_state("sim-lease-formal")["state"],
                "selfcheck_terminal": sim.job_state("sim-lease-selfcheck")["state"],
                "selfcheck_error": _grpc_code(selfcheck_errors[0]) if selfcheck_errors else "none",
            }
        finally:
            stop.set()
            sampler.join(timeout=2)
            formal.close()
            selfcheck.close()
    checker.check("timed_lease_single_owner", timed_lease, "quota")

    def fairness() -> dict:
        formal = sim.factory()
        selfcheck = sim.factory(token="grant-fair", run_id="run-2")
        try:
            thread, _errors = _threaded(formal, _request("sim-fair-a", "SLEEP", candidate_s=4))
            _wait_for_running(sim, "sim-fair-a", errors=_errors)
            b_thread, _b_errors = _threaded(formal, _request("sim-fair-b", "OK"))
            c_thread, _c_errors = _threaded(
                selfcheck,
                _request("sim-fair-c", "OK", kind="selfcheck", run_id="run-2", session_id="session-fair"),
            )
            deadline = time.monotonic() + 20
            for request_id in ("sim-fair-b", "sim-fair-c"):
                while time.monotonic() < deadline:
                    try:
                        state = sim.job_state(request_id)["state"]
                    except AssertionError:
                        state = "missing"
                    if state in {"queued", "running", "completed"}:
                        break
                    time.sleep(0.02)
            time.sleep(1.3)  # exceeds the container's 1s selfcheck aging window
            d_thread, _d_errors = _threaded(formal, _request("sim-fair-d", "OK"))
            for participant in (thread, b_thread, c_thread, d_thread):
                participant.join(timeout=30)
                if participant.is_alive():
                    raise AssertionError("the fairness probe did not finish")
            deadline = time.monotonic() + 25
            for request_id in ("sim-fair-b", "sim-fair-c", "sim-fair-d"):
                while time.monotonic() < deadline and sim.job_state(request_id)["state"] not in {"completed", "failed", "cancelled"}:
                    time.sleep(0.05)
                if sim.job_state(request_id)["state"] != "completed":
                    raise AssertionError(f"{request_id} did not complete")
            order = [entry["request_id"] for entry in sim.worker_order()]
            if _b_errors or _c_errors or _d_errors:
                raise AssertionError(f"fairness probes failed: {[_grpc_code(e) for e in (*_b_errors, *_c_errors, *_d_errors)]}")
            if any(name not in order for name in ("sim-fair-b", "sim-fair-c", "sim-fair-d")):
                raise AssertionError(f"worker order is incomplete: {order}")
            if order.index("sim-fair-c") > order.index("sim-fair-d"):
                raise AssertionError(f"a later formal job jumped the aged cross-run selfcheck: {order}")
            if order.index("sim-fair-a") > order.index("sim-fair-b"):
                raise AssertionError("the earlier formal job did not run before the later one")
            if order.index("sim-fair-a") > order.index("sim-fair-c"):
                raise AssertionError("the selfcheck ran before the earlier formal work")
            return {"worker_order": order[-4:], "aged_selfcheck_ran_before_later_formal": True}
        finally:
            formal.close()
            selfcheck.close()
    checker.check("fairness_two_runs_selfcheck_not_starved", fairness, "quota")

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
        progress_path.unlink(missing_ok=True)
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
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        progress_path.unlink(missing_ok=True)
        return {
            "status": result.report.get("status"),
            "progress_file": progress_path.name,
            "progress_finished": bool(payload.get("finished")),
        }
    checker.check("slow_consumer_watch_reconnect", slow_consumer_progress, "faults")


def recovery_checks(sim: Sim, checker: Checker) -> None:
    # Restart with a running job.
    def restart_recovery() -> dict:
        factory = sim.factory(rpc_timeout_s=8, watch_slice_s=3, job_margin_s=15)
        thread, errors = _threaded(factory, _request("sim-restart", "HOLD_ONCE", candidate_s=60))
        try:
            _wait_for_running(sim, "sim-restart", errors=errors)
            sim.stop()
            factory.close()
            thread.join(timeout=10)
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
                    "note": "the durable attempt stayed interrupted until an explicit Retry",
                }
            finally:
                resumed_factory.close()
        finally:
            factory.close()
    checker.check("container_restart_explicit_retry", restart_recovery, "recovery")

    # SIGKILL leaves a durable running fence; recovery must not auto-run.
    def stale_worker() -> dict:
        factory = sim.factory(rpc_timeout_s=8, watch_slice_s=3, job_margin_s=15)
        thread, _errors = _threaded(factory, _request("sim-stale", "HOLD_ONCE", candidate_s=60))
        try:
            _wait_for_running(sim, "sim-stale", errors=_errors)
            sim.kill()
            factory.close()
            thread.join(timeout=10)
            attempts_before = sim.attempts("sim-stale")
            if attempts_before != 1:
                raise AssertionError(f"stale probe created {attempts_before} attempts before the kill")
            sim.start()
            sim.wait_ready()
            time.sleep(1.0)
            if sim.attempts("sim-stale") != attempts_before:
                raise AssertionError("recovery silently re-executed a stale attempt")
            resumed_factory = sim.factory(rpc_timeout_s=10)
            try:
                result = resumed_factory.formal_resume(_request("sim-stale", "HOLD_ONCE", candidate_s=60))
                return {"attempts_before": attempts_before, "attempts_after": sim.attempts("sim-stale"), "lease_fence": result.receipt.get("lease_fence")}
            except EvaluationUnavailable as exc:
                if exc.code in {"idempotency_conflict", "session_mismatch", "invalid_argument"}:
                    raise AssertionError(f"retry used a different request digest/identity: {exc.code}")
                return {"attempts_before": attempts_before, "quarantined_or_nonretryable": exc.code, "retryable": exc.retryable}
            finally:
                resumed_factory.close()
        finally:
            factory.close()
    checker.check("stale_worker_fail_closed", stale_worker, "recovery")

    # Backup / restore on persistent volumes.
    def backup_restore() -> dict:
        factory = sim.factory(rpc_timeout_s=8, watch_slice_s=3, job_margin_s=15)
        thread, _errors = _threaded(factory, _request("sim-backup-running", "SLEEP", candidate_s=30))
        try:
            _wait_for_running(sim, "sim-backup-running", errors=_errors)
            sim.stop()
            thread.join(timeout=10)
        finally:
            factory.close()
        docker = sim.docker
        docker.volume_create(sim.backup_volume, uid=os.getuid(), gid=os.getgid())
        docker.volume_create(sim.restored_volume, uid=os.getuid(), gid=os.getgid())
        common = [
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=volume,src={sim.state_volume},dst=/state",
            "--mount", f"type=volume,src={sim.backup_volume},dst=/backup",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
        ]
        docker.run(*common, "-m", "tcenv.remote.state_admin", "backup", "--state-dir", "/state", "--out", "/backup/state.sqlite3", "--keyring-file", "/etc/tcenv/keyring.json")
        backup_bytes = int(docker.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--mount", f"type=volume,src={sim.backup_volume},dst=/backup",
            "--entrypoint", "stat", IMAGE, "-c", "%s", "/backup/state.sqlite3",
        ).stdout.strip() or 0)
        if backup_bytes <= 0:
            raise AssertionError("backup file was not produced")
        docker.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=volume,src={sim.backup_volume},dst=/backup,readonly",
            "--mount", f"type=volume,src={sim.restored_volume},dst=/restored",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
            "-m", "tcenv.remote.state_admin", "restore", "--backup", "/backup/state.sqlite3", "--state-dir", "/restored",
        )
        inspect = docker.run(
            "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
            "--entrypoint", "python3",
            "--mount", f"type=volume,src={sim.restored_volume},dst=/restored",
            "--mount", f"type=bind,src={sim.config},dst=/etc/tcenv,readonly",
            IMAGE,
            "-m", "tcenv.remote.state_admin", "inspect", "--state-dir", "/restored", "--keyring-file", "/etc/tcenv/keyring.json",
        )
        document = json.loads(inspect.stdout)
        if not any(state == "completed" for state in document["states"]):
            raise AssertionError("restored backup has no completed job")
        if not document["quarantined"]:
            raise AssertionError("restore did not quarantine nonterminal jobs")
        sim.start()
        sim.wait_ready()
        return {
            "backup_bytes": backup_bytes,
            "states": document["states"],
            "quarantined": len(document["quarantined"]),
            "backup_volume": sim.backup_volume,
            "restored_volume": sim.restored_volume,
        }
    checker.check("backup_restore_quarantine", backup_restore, "recovery")


def limits_checks(sim: Sim, checker: Checker) -> None:
    """Exercise queue, stream, DB-byte and auth-failure limits on a small store."""

    main_port = sim.port
    sim.docker.volume_create(sim.limits_volume, uid=os.getuid(), gid=os.getgid())
    port = sim.start(
        name=LIMITS_NAME, volume=sim.limits_volume,
        extra_args=(
            "--max-queue-depth", "2", "--max-db-bytes", "1000000",
            "--max-watch-streams", "2", "--auth-max-failures", "3",
        ),
    )
    limits_keepalive = None
    try:
        sim.wait_ready(token=sim.default_token, port=port, name=LIMITS_NAME)
        limits_keepalive = sim.factory(port=port, rpc_timeout_s=8)
        limits_keepalive.ensure_connected()

        def queue_limit() -> dict:
            formal = sim.factory(port=port, rpc_timeout_s=8)
            thread = None
            queued_thread = None
            try:
                thread, hold_errors = _threaded(formal, _request("limits-hold", "SLEEP", candidate_s=25))
                _wait_for_running(sim, "limits-hold", timeout=20, container=LIMITS_NAME, errors=hold_errors)
                queued_thread, _queued_errors = _threaded(
                    formal, _request("limits-queued", "OK", queue_s=90, candidate_s=20),
                )
                deadline = time.monotonic() + 15
                depth = 0
                while time.monotonic() < deadline:
                    depth = int(sim.sql(
                        "SELECT COUNT(*) FROM jobs WHERE state IN ('accepted','queued','running')",
                        (), container=LIMITS_NAME,
                    )[0][0])
                    if depth >= 2:
                        break
                    time.sleep(0.1)
                if depth < 2:
                    raise AssertionError(f"the queued job was not accepted (depth={depth})")
                try:
                    formal.formal(_request("limits-rejected", "OK", queue_s=5))
                except EvaluationUnavailable as exc:
                    if exc.code != "resource_exhausted":
                        raise AssertionError(f"queue limit returned {exc.code}")
                    return {
                        "code": exc.code,
                        "queue_depth_limit": 2,
                        "queue_depth": depth,
                        "running": sim.job_state("limits-hold", container=LIMITS_NAME)["state"],
                    }
                raise AssertionError("the device queue depth limit was not enforced")
            finally:
                formal.cancel("limits-hold")
                formal.close()
                if thread is not None:
                    thread.join(timeout=10)
                if queued_thread is not None:
                    queued_thread.join(timeout=30)
        checker.check("queue_depth_limit", queue_limit, "limits")

        def watch_limit() -> dict:
            held: list = []
            first_events: list[int] = []
            stream_errors: list[str] = []
            threads: list[threading.Thread] = []
            hold = sim.factory(port=port, rpc_timeout_s=8)
            hold_request = _request("limits-watch-hold", "PROGRESS SLEEP", candidate_s=25)
            hold_thread, _hold_errors = _threaded(hold, hold_request)
            try:
                _wait_for_running(sim, "limits-watch-hold", timeout=20, container=LIMITS_NAME, errors=_hold_errors)
                channel = sim.raw_channel(port=port)
                stub = pb_grpc.EvaluatorStub(channel)
                capabilities = sim.raw_capabilities(port=port)
                expected = sim.expected(capabilities)
                from tcenv.remote.job_types import ExternalEvaluationRequest
                external = ExternalEvaluationRequest.from_legacy(hold_request)

                def open_stream() -> None:
                    request = pb.WatchRequest(
                        job=pb.JobRef(
                            request_id=external.request_id,
                            request_digest=external.request_digest,
                            expected=gc.expected_identities_to_proto(expected),
                        ),
                    )
                    try:
                        stream = stub.Watch(request, metadata=(("authorization", f"Bearer {sim.default_token}"),))
                    except grpc.RpcError as exc:
                        stream_errors.append(exc.code().name)
                        return
                    held.append(stream)
                    try:
                        for _event in stream:
                            first_events.append(1)
                            break
                    except grpc.RpcError as exc:
                        stream_errors.append(exc.code().name)
                        return
                    try:
                        for _event in stream:
                            pass
                    except grpc.RpcError as exc:
                        # The finally block intentionally cancels held streams.
                        # Preserve every other terminal stream error instead of
                        # leaking an unhandled exception from this test thread.
                        if exc.code() != grpc.StatusCode.CANCELLED:
                            stream_errors.append(exc.code().name)

                # The running job already has one client Watch attached (the
                # formal() probe thread), which occupies one of the two slots.
                threads = [threading.Thread(target=open_stream, daemon=True) for _ in range(2)]
                for thread in threads:
                    thread.start()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not first_events and not stream_errors:
                    time.sleep(0.05)
                if not first_events:
                    raise AssertionError(f"could not hold the configured watch streams (errors={stream_errors})")
                third = pb.WatchRequest(
                    job=pb.JobRef(
                        request_id=external.request_id,
                        request_digest=external.request_digest,
                        expected=gc.expected_identities_to_proto(expected),
                    ),
                )
                try:
                    list(stub.Watch(third, metadata=(("authorization", f"Bearer {sim.default_token}"),), timeout=5))
                except grpc.RpcError as exc:
                    if exc.code() != grpc.StatusCode.RESOURCE_EXHAUSTED:
                        raise AssertionError(f"the watch stream over the limit returned {exc.code()}")
                    return {
                        "grpc_code": exc.code().name,
                        "max_watch_streams": 2,
                        "held_streams": 2,
                        "stream_errors": stream_errors,
                    }
                raise AssertionError(f"the watch stream limit was not enforced (errors={stream_errors})")
            finally:
                for stream in held:
                    try:
                        stream.cancel()
                    except Exception:
                        pass
                for thread in threads:
                    thread.join(timeout=5)
                hold.cancel("limits-watch-hold")
                hold.close()
                hold_thread.join(timeout=10)
        checker.check("watch_stream_limit", watch_limit, "limits")

        def db_limit() -> dict:
            formal = sim.factory(port=port, rpc_timeout_s=30)
            try:
                large = "OK " + "A" * 949_000
                try:
                    formal.formal(_request("limits-db", large, candidate_s=20, queue_s=5))
                except EvaluationUnavailable as exc:
                    if exc.code != "resource_exhausted":
                        raise AssertionError(f"job-store byte limit returned {exc.code}")
                    if "byte limit" not in str(exc) and "store" not in str(exc):
                        raise AssertionError(f"the rejection was not the byte limit: {exc}")
                    return {"code": exc.code, "source_bytes": len(large), "max_db_bytes": 1_000_000}
                raise AssertionError("the job-store byte limit was not enforced")
            finally:
                formal.close()
        checker.check("job_store_byte_limit", db_limit, "limits")

        def auth_limit() -> dict:
            codes = []
            for index in range(4):
                try:
                    sim.raw_capabilities(token=f"unknown-limit-token-{index}", port=port)
                except grpc.RpcError as exc:
                    codes.append(exc.code().name)
                else:
                    codes.append("OK")
            if codes[-1] != "RESOURCE_EXHAUSTED":
                raise AssertionError(f"auth failure limit returned {codes}")
            return {"codes": codes, "auth_max_failures": 3}
        checker.check("auth_failure_rate_limit", auth_limit, "limits")
    finally:
        if limits_keepalive is not None:
            limits_keepalive.close()
        sim.stop(name=LIMITS_NAME)
        sim.port = main_port


def rotation_checks(sim: Sim, checker: Checker) -> None:
    """Grant renewal/revocation, clock skew, HMAC rotation/revocation, rollback, certs."""

    history = _request("sim-rotation-history", "OK")
    history_factory = sim.factory()
    try:
        history_result = history_factory.formal(history)
    finally:
        history_factory.close()
    history_attempt = history_result.receipt["attempt_id"]

    # Start an attempt on the grant that the next restart removes.
    revoked_factory = sim.factory(token=REVOKE_ACTIVE_TOKEN, rpc_timeout_s=8)
    revoked_thread, _errors = _threaded(
        revoked_factory,
        _request("sim-revoke-active", "SLEEP", session_id="session-revoke-active", candidate_s=30),
    )
    try:
        _wait_for_running(sim, "sim-revoke-active", errors=_errors)

        # Grant renewal + authoritative revocation + clock skew need one restart.
        sim.stop()
        near = time.time() - 5
        far = time.time() - 120
        dropped = {"grant-revoke-me", REVOKE_ACTIVE_TOKEN, OLD_TOKEN}
        renewed_grants = [grant for grant in sim.grants if grant.token not in dropped]
        renewed_grants.append(_grant(RENEWED_TOKEN, max_timed=1, max_gpu_s=120.0))
        renewed_grants.append(_grant(NEAR_EXPIRY_TOKEN, session_id="session-near", expires_at=near))
        renewed_grants.append(_grant(FAR_EXPIRED_TOKEN, session_id="session-far", expires_at=far))
        sim.write_grants(renewed_grants)
        sim.default_token = RENEWED_TOKEN
        sim.start(extra_args=("--clock-skew-ms", "30000"))
        sim.wait_ready()

        def revoked_grant() -> dict:
            try:
                sim.raw_capabilities(token="grant-revoke-me")
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.PERMISSION_DENIED:
                    raise AssertionError(f"a revoked grant returned {exc.code()}")
                state = sim.job_state("sim-revoke-active")
                if state["state"] != "cancelled":
                    raise AssertionError(f"the revoked scope's active attempt is {state['state']}")
                return {"grpc_code": exc.code().name, "removed_token": "grant-revoke-me"}
            raise AssertionError("a revoked grant was accepted")
        checker.check("revoked_grant_rejected", revoked_grant, "authz")

        def active_revocation() -> dict:
            state = sim.job_state("sim-revoke-active")
            if state["state"] != "cancelled":
                raise AssertionError(f"active attempt was not cancelled: {state}")
            if sim.running_attempts() != 0:
                raise AssertionError("a revoked scope still holds a running attempt")
            return {"request_id": "sim-revoke-active", "state": state["state"], "running_attempts": 0}
        checker.check("active_attempt_cancelled_on_revocation", active_revocation, "authz")

        def renewal_reattach() -> dict:
            factory = sim.factory(token=RENEWED_TOKEN, rpc_timeout_s=10)
            try:
                result = factory.formal(history)
                if result.receipt["attempt_id"] != history_attempt:
                    raise AssertionError("the renewed grant created a new attempt instead of re-attaching")
                old = sim.factory(token=OLD_TOKEN)
                try:
                    try:
                        old.get_job(history)
                    except EvaluationUnavailable as exc:
                        if exc.code != "grant_rejected" or "superseded" not in str(exc):
                            raise AssertionError(f"the superseded token returned {exc.code}: {exc}")
                    else:
                        raise AssertionError("the superseded token still reads the accepted job")
                finally:
                    old.close()
                return {"attempt_id": result.receipt["attempt_id"], "superseded_code": "grant_rejected"}
            finally:
                factory.close()
        checker.check("grant_renewal_reattach", renewal_reattach, "authz")

        def skew_within() -> dict:
            capabilities = sim.raw_capabilities(token=NEAR_EXPIRY_TOKEN)
            if not capabilities.capability_id:
                raise AssertionError("an expired-within-skew grant produced no capability")
            return {"clock_skew_ms": 30000, "expired_by_s": 5, "accepted": True}
        checker.check("clock_skew_within_bound_accepted", skew_within, "authn")

        def skew_beyond() -> dict:
            try:
                sim.raw_capabilities(token=FAR_EXPIRED_TOKEN)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.UNAUTHENTICATED:
                    raise AssertionError(f"a grant expired beyond skew returned {exc.code}")
                return {"clock_skew_ms": 30000, "expired_by_s": 120, "grpc_code": exc.code().name}
            raise AssertionError("a grant expired beyond the configured skew was accepted")
        checker.check("clock_skew_beyond_bound_rejected", skew_beyond, "authn")

        # Normal HMAC rotation keeps key-1 trusted for history; key-2 signs new results.
        sim.stop()
        sim.write_keyring({sim.key_id1: sim.key1, sim.key_id2: sim.key2}, active=sim.key_id2)
        sim.start(extra_args=("--clock-skew-ms", "30000"))
        sim.wait_ready()

        def rotation() -> dict:
            factory = sim.factory(token=RENEWED_TOKEN, rpc_timeout_s=10)
            try:
                historical = factory.formal(history)
                if historical.receipt["attempt_id"] != history_attempt:
                    raise AssertionError("the historical result did not replay the same attempt")
                if historical.receipt["signature_key_id"] != sim.key_id1:
                    raise AssertionError(f"historical result key is {historical.receipt['signature_key_id']}")
                fresh = factory.formal(_request("sim-rotation-new", "OK"))
                if fresh.receipt["signature_key_id"] != sim.key_id2:
                    raise AssertionError(f"new result key is {fresh.receipt['signature_key_id']}")
                return {
                    "historical_attempt_id": historical.receipt["attempt_id"],
                    "historical_key_id": historical.receipt["signature_key_id"],
                    "new_attempt_id": fresh.receipt["attempt_id"],
                    "new_key_id": fresh.receipt["signature_key_id"],
                }
            finally:
                factory.close()
        checker.check("hmac_key_rotation_historical_verification", rotation, "signature")

        # True previous-image downgrade against the same persistent state.
        previous = sim.factory(token=RENEWED_TOKEN, rpc_timeout_s=10)
        try:
            rotated_attempt = previous.formal(_request("sim-rotation-new", "OK")).receipt["attempt_id"]
        finally:
            previous.close()

        def rollback() -> dict:
            current_id = sim.docker.image_id(IMAGE)
            previous_id = sim.docker.image_id(IMAGE_PREVIOUS)
            source = getattr(sim, "previous_source", {}) or {}
            if not source.get("commit") or not source.get("product_diff_files"):
                raise AssertionError("no distinct previous product commit was built for rollback")
            if current_id == previous_id:
                raise AssertionError("the previous image has the same immutable image id")
            sim.stop()
            sim.start(image=IMAGE_PREVIOUS, extra_args=("--clock-skew-ms", "30000"), legacy=True)
            sim.wait_ready()
            factory = sim.factory(token=RENEWED_TOKEN, rpc_timeout_s=10)
            try:
                replay = factory.formal(_request("sim-rotation-new", "OK"))
                if replay.receipt["attempt_id"] != rotated_attempt:
                    raise AssertionError("the downgraded image did not replay the accepted attempt")
                fresh = factory.formal(_request("sim-rollback-fresh", "OK"))
                return {
                    "rolled_back_from_image_id": current_id,
                    "rolled_back_to_image_id": previous_id,
                    "rolled_back_from_revision": sim.docker.image_revision(IMAGE),
                    "rolled_back_to_revision": sim.docker.image_revision(IMAGE_PREVIOUS),
                    "previous_commit": source.get("commit", ""),
                    "previous_ref": source.get("ref", ""),
                    "previous_subject": source.get("subject", ""),
                    "product_diff_files": source.get("product_diff_files", []),
                    "same_persistent_schema": True,
                    "replayed_attempt_id": replay.receipt["attempt_id"],
                    "new_attempt_id": fresh.receipt["attempt_id"],
                }
            finally:
                factory.close()
        checker.check("rollback_previous_image", rollback, "recovery")

        # Back to the current image, then revoke the historical key only.
        sim.stop()
        sim.write_keyring({sim.key_id2: sim.key2}, revoked=(sim.key_id1,), active=sim.key_id2)
        sim.start(extra_args=("--clock-skew-ms", "30000"))
        sim.wait_ready()

        def revocation() -> dict:
            factory = sim.factory(token=RENEWED_TOKEN, rpc_timeout_s=10)
            try:
                try:
                    factory.formal(history)
                except EvaluationUnavailable as exc:
                    if exc.code != "result_missing":
                        raise AssertionError(f"the revoked historical key returned {exc.code}")
                    state = sim.job_state("sim-rotation-history")
                    if not state["quarantined"]:
                        raise AssertionError("the affected historical result was not quarantined")
                    snapshot = sim.raw_get_job(history)
                    if snapshot.HasField("result"):
                        raise AssertionError("a quarantined result was still served")
                    fresh = factory.formal(_request("sim-after-key-revocation", "OK"))
                    if fresh.receipt["signature_key_id"] != sim.key_id2:
                        raise AssertionError("frontend did not keep serving new results after key revocation")
                    return {
                        "historical_code": exc.code,
                        "quarantined": state["quarantined"],
                        "historical_result_withheld": True,
                        "new_key_id": fresh.receipt["signature_key_id"],
                        "frontend_restarted": True,
                    }
                raise AssertionError("the revoked historical key was still trusted")
            finally:
                factory.close()
        checker.check("hmac_key_revocation_quarantines_history", revocation, "signature")

        # Per-certificate revocation of the rotated leaf.
        sim.stop()
        from tcenv.remote.revocation import certificate_sha256

        rotated_fingerprint = certificate_sha256(sim.pki.rotated_client_cert.read_bytes())
        sim.revoke_client_certificates(frozenset({rotated_fingerprint}))
        sim.start(extra_args=("--clock-skew-ms", "30000"))
        sim.wait_ready()

        def revoked_cert() -> dict:
            try:
                sim.raw_capabilities(cert=sim.pki.rotated_client_cert, key=sim.pki.rotated_client_key)
            except grpc.RpcError as exc:
                if exc.code() != grpc.StatusCode.PERMISSION_DENIED:
                    raise AssertionError(f"a revoked client certificate returned {exc.code}")
                if "revoked" not in (exc.details() or ""):
                    raise AssertionError("the rejection was not a certificate revocation")
                sim.raw_capabilities(cert=sim.pki.client_cert, key=sim.pki.client_key)
                return {
                    "grpc_code": exc.code().name,
                    "revoked_fingerprint_sha256": rotated_fingerprint,
                    "other_same_ca_identity_served": True,
                }
            raise AssertionError("a revoked client certificate was accepted")
        checker.check("revoked_client_cert_rejected", revoked_cert, "tls")
    finally:
        revoked_factory.close()
        revoked_thread.join(timeout=10)


def _inspect_volume(docker: Docker, volume: str) -> dict:
    """Inspect one stopped state volume without a host-side SQLite connection."""

    script = (
        "import json,pathlib,sqlite3;"
        "root=pathlib.Path('/state');"
        "temp=[e.name for e in root.iterdir() if '.tmp-' in e.name or '.restore-tmp' in e.name];"
        "bridge=root/'bridge-tmp';"
        "temp += ['bridge-tmp/'+e.name for e in bridge.iterdir()] if bridge.is_dir() else [];"
        "running=workers=0;"
        "db=root/'jobs.sqlite3';"
        "running=int(sqlite3.connect(f'file:{db}?mode=ro',uri=True).execute(\"SELECT COUNT(*) FROM attempts WHERE state='running'\").fetchone()[0]) if db.is_file() else 0;"
        "workers=int(sqlite3.connect(f'file:{db}?mode=ro',uri=True).execute(\"SELECT COUNT(*) FROM attempts WHERE worker_pid IS NOT NULL AND state IN ('queued','running')\").fetchone()[0]) if db.is_file() else 0;"
        "print(json.dumps({'temp': sorted(temp), 'running': running, 'workers': workers}))"
    )
    completed = docker.run(
        "run", "--rm", "--user", f"{os.getuid()}:{os.getgid()}",
        "--mount", f"type=volume,src={volume},dst=/state",
        "--entrypoint", "python3", IMAGE, "-c", script,
    )
    return json.loads(completed.stdout)


def cleanup_evidence(sim: Sim, docker: Docker, *, kept: bool) -> dict:
    containers = docker.acceptance_containers()
    volumes = [sim.state_volume, sim.limits_volume, sim.backup_volume, sim.restored_volume]
    evidence: dict = {
        "containers_left": containers,
        "kept_for_inspection": kept,
        "images_retained": {IMAGE: docker.image_id(IMAGE), IMAGE_PREVIOUS: docker.image_id(IMAGE_PREVIOUS)},
        "volumes_retained": volumes,
        "temp_paths": {},
        "process_groups": {},
        "final": False,
    }
    if kept:
        evidence["note"] = "container kept for inspection; this is not a final cleanup proof"
        return evidence
    for label, volume in (("primary", sim.state_volume), ("limits", sim.limits_volume)):
        try:
            inspected = _inspect_volume(docker, volume)
        except Exception as exc:  # noqa: BLE001 - a missing volume is a cleanup failure
            evidence["temp_paths"][volume] = [f"inspect failed: {type(exc).__name__}"]
            evidence["process_groups"][label] = -1
            continue
        evidence["temp_paths"][volume] = inspected["temp"]
        evidence["process_groups"][label] = int(inspected["running"]) + int(inspected["workers"])
    evidence["final"] = bool(
        not containers
        and not any(evidence["temp_paths"].values())
        and not any(evidence["process_groups"].values())
    )
    return evidence


def build_and_run(*, report_path: Path, keep: bool = False, fresh: bool = False) -> dict:
    started = time.monotonic()
    report_path = Path(report_path).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if fresh:
        # Clear contents instead of deleting the directories: Docker Desktop
        # caches bind-mount inodes and a recreated path can briefly look absent
        # to the VM.
        for name in ("container-config", "container-state", "container-state-restored", "container-backup", "container-limits"):
            candidate = report_path.parent / name
            candidate.mkdir(parents=True, exist_ok=True)
            for entry in list(candidate.iterdir()):
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
    docker = Docker()
    if fresh:
        for volume in volume_names(report_path.parent).values():
            docker.volume_remove(volume)
    docker.start_events()
    report: dict = {
        "schema_version": 1,
        "simulated": True,
        "publication_allowed": False,
        "status": "blocked",
        "date": time.strftime("%Y-%m-%d"),
        "required_checks": list(REQUIRED_CHECKS),
        "results": [],
        "failed_checks": [],
        "missing_required": list(REQUIRED_CHECKS),
        "not_run": list(NOT_RUN),
        "blockers": [],
        "environment": {},
        "transport": {
            "protocol": "gRPC over TLS/mTLS",
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
        "timings": {},
    }
    sim = None
    checker = None
    build_started = time.monotonic()
    report["previous_source"] = {}
    try:
        docker.run(
            "build", "-f", "tests/support/docker/Dockerfile.remote-grpc-sim", "-t", IMAGE,
            "--label", "tcenv.sim.revision=current",
            "--build-arg", "TCENV_SIM_IMAGE_REVISION=current", ".", timeout=1800,
        )
        report["previous_source"] = build_previous_image(
            docker, report_path.parent, label="tcenv.sim.revision=previous",
        )
    except Exception as exc:  # noqa: BLE001 - report the blocker instead of faking a pass
        report["status"] = "blocked"
        report["blockers"] = [f"docker image build failed: {type(exc).__name__}: {exc}"[:2000]]
        report["timings"]["total_s"] = round(time.monotonic() - started, 3)
        docker.stop_events()
        report["docker_events"] = docker.events[-80:]
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return report
    report["timings"]["image_build_s"] = round(time.monotonic() - build_started, 3)
    try:
        image_id = docker.image_id(IMAGE)
        previous_image_id = docker.image_id(IMAGE_PREVIOUS)
        os_release = docker.run("run", "--rm", "--entrypoint", "cat", IMAGE, "/etc/os-release", check=False).stdout
        container_os = next(
            (line.split("=", 1)[1].strip().strip('"') for line in os_release.splitlines() if line.startswith("PRETTY_NAME=")),
            "unknown",
        )
        docker_version = docker.run("version", "--format", "{{.Server.Version}}", check=False).stdout.strip()
        docker_platform = docker.run("info", "--format", "{{.OSType}}/{{.Architecture}}", check=False).stdout.strip()
        report["environment"] = {
            "control_host": f"macOS {platform.mac_ver()[0]} {platform.machine()}",
            "container_platform": docker_platform,
            "container_os": container_os,
            "container_host": "Docker Desktop Linux VM",
            "docker_engine": docker_version,
            "image": IMAGE,
            "image_id": image_id,
            "previous_image_id": previous_image_id,
            "image_revision": docker.image_revision(IMAGE),
            "previous_image_revision": docker.image_revision(IMAGE_PREVIOUS),
            "container_user": f"{os.getuid()}:{os.getgid()}",
            "read_only_root": True,
            "capabilities": "dropped",
            "device_mounts": "none",
            "ssh_data_plane": False,
        }
        sim = Sim(root=report_path.parent, docker=docker)
        sim.previous_source = dict(report.get("previous_source") or {})
        sim.start()
        sim.wait_ready()
        report["transport"]["endpoint"] = f"127.0.0.1:{sim.port}"
        checker = Checker(sim)
        checks_started = time.monotonic()
        try:
            main_matrix(sim, checker, state_root=report_path.parent)
            recovery_checks(sim, checker)
            limits_checks(sim, checker)
            rotation_checks(sim, checker)
        finally:
            report["results"] = checker.results
            report["timings"]["checks_s"] = round(time.monotonic() - checks_started, 3)
    except Exception as exc:  # noqa: BLE001 - report the blocker instead of faking a pass
        report["status"] = "failed"
        report["failure"] = f"{type(exc).__name__}: {exc}"[:4000]
        if sim is not None:
            report["container_logs"] = docker.container_logs(sim.name)[-4000:]
    finally:
        if sim is not None:
            failures_before_cleanup = [
                result["name"] for result in (checker.results if checker is not None else []) if not result["ok"]
            ]
            if failures_before_cleanup:
                report["container_logs"] = docker.container_logs(sim.name)[-8000:]
            if not keep:
                sim.stop()
                docker.remove(LIMITS_NAME)
            sim.close_keepalive()
            cleanup = cleanup_evidence(sim, docker, kept=keep)
            report["cleanup"] = cleanup
            if checker is not None:
                # The cleanup result is recorded last, after the container is gone.
                checker.results.append({
                    "name": "cleanup",
                    "category": "cleanup",
                    "ok": bool(cleanup.get("final")),
                    "evidence": cleanup,
                    "elapsed_s": 0,
                })
                report["results"] = checker.results
            docker.stop_events()
            report["docker_events"] = docker.events[-80:]
            if keep:
                report["blockers"].append("container kept for inspection; the final cleanup proof did not run")
        else:
            docker.stop_events()
            report["docker_events"] = docker.events[-80:]
        report["timings"]["total_s"] = round(time.monotonic() - started, 3)
        if checker is not None:
            by_name = {result["name"]: result for result in checker.results}
            missing = [name for name in REQUIRED_CHECKS if name not in by_name]
            failed = [name for name in REQUIRED_CHECKS if name in by_name and not by_name[name]["ok"]]
            report["missing_required"] = missing
            report["failed_checks"] = failed
            report["unexpected_checks"] = sorted(set(by_name) - set(REQUIRED_CHECKS))
            blockers = list(report.get("blockers") or [])
            if missing:
                blockers.append(f"required checks did not run: {missing}")
            if not report.get("cleanup", {}).get("final"):
                blockers.append("final cleanup evidence is not final")
            report["blockers"] = blockers
            capabilities = by_name.get("tls_mtls_capabilities")
            if capabilities is not None and capabilities.get("evidence"):
                report["identities"]["evaluator_epoch"] = capabilities["evidence"].get("evaluator_epoch", "")
            report["attempts"] = [
                {
                    "request_id": result["name"],
                    **{
                        key: value for key, value in (result.get("evidence") or {}).items()
                        if key in {"attempt_id", "lease_fence"}
                    },
                }
                for result in checker.results
                if (result.get("evidence") or {}).get("attempt_id") or (result.get("evidence") or {}).get("lease_fence")
            ]
            if report.get("failure") or failed or missing:
                report["status"] = "failed"
            elif blockers:
                report["status"] = "blocked"
            else:
                report["status"] = "passed"
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
