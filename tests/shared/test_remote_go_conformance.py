"""Non-Python (Go) gRPC conformance client against the local TLS frontend.

The Go module vendors no generated stubs: it loads the published descriptor set
and verifies the request digest, receipt and HMAC result signature itself.  The
test skips only when the Go toolchain or its module cache is unavailable; with
Go present it must actually run.

The flow is executed twice against one durable state directory: a full
conformance run (including a forced interrupted attempt resumed by the explicit
Retry RPC) and a re-attachment run after normal grant renewal and HMAC key
rotation.  Revoking the active key on the host side must make the same replay
fail closed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tcenv.remote.identity import SessionGrant
from tcenv.remote.keyring import KeyringFile
from tests.support.job_engine import (
    TARGET, TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN,
    engine as make_engine, grant as make_grant, identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki
from tests.support.v1_tasks import V1_TASK

ROOT = Path(__file__).resolve().parents[2]
GO_DIR = ROOT / "tests" / "support" / "remotegrpc"
DESC = ROOT / "proto" / "tcenv" / "remote" / "v1" / "evaluator.desc"
KEY1 = b"g" * 32
KEY2 = b"h" * 32
KEY1_ID = "sim-external-site/key-1"
KEY2_ID = "sim-external-site/key-2"
RENEWED_TOKEN = "external-go-renewed-token"

pytestmark = pytest.mark.skipif(
    shutil.which("go") is None or shutil.which("openssl") is None,
    reason="Go and openssl are required for the non-Python conformance client",
)


def _go_environment() -> dict[str, str]:
    env = dict(os.environ)
    base = Path(tempfile.gettempdir())
    env.setdefault("GOMODCACHE", str(base / "go-modcache"))
    env.setdefault("GOCACHE", str(base / "go-build-cache"))
    env.setdefault("GOPATH", str(base / "gopath"))
    for name in ("GOMODCACHE", "GOCACHE", "GOPATH"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    env["GOFLAGS"] = "-mod=mod"
    return env


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    try:
        return make_pki(tmp_path_factory.mktemp("go-pki"))
    except PkiUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


class Frontend:
    def __init__(self, root: Path, pki) -> None:
        self.root = root
        self.pki = pki
        self.state = root / "state"
        self.engine = None
        self.server = None

    def start(self, *, grants: list[SessionGrant], keyring: KeyringFile) -> None:
        self.stop()
        active_key_id, active_key = keyring.require_active()
        self.engine = make_engine(
            self.state, grants=grants, signing_key=active_key, signing_key_id=active_key_id,
            signing_keyring=keyring.trusted,
            worker_env={"FAKE_JOB_FAIL_DIR": str(self.root / "fail-once")},
        )
        self.server = build_server(
            self.engine,
            config=GrpcServerConfig(
                address="127.0.0.1:0", allow_insecure=False, require_client_identity=True,
                certificate_file=self.pki.server_cert, private_key_file=self.pki.server_key,
                client_ca_file=self.pki.ca_cert, watch_poll_s=0.02,
            ),
        )

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop(grace_s=0.2)
            self.server = None
        if self.engine is not None:
            self.engine.close()
            self.engine = None


def _keyring(path: Path, keys: dict[str, bytes], *, revoked: tuple[str, ...] = (), active: str) -> Path:
    path.write_text(json.dumps({
        "schema_version": 1,
        "keys": {key_id: key.hex() for key_id, key in keys.items()},
        "revoked": list(revoked),
        "active_key_id": active,
    }), encoding="utf-8")
    path.chmod(0o600)
    return path


def _command(
    frontend: Frontend, *, token: Path, keyring: Path, source: Path, request_id: str,
    flow: str = "conformance", retry_source: Path | None = None,
    cancel_source: Path | None = None, expect_attempt_id: str = "",
    expect_key_id: str = "", out: Path,
) -> list[str]:
    pki = frontend.pki
    command = [
        "go", "run", ".",
        "--desc", str(DESC),
        "--endpoint", f"127.0.0.1:{frontend.server.port}",
        "--ca", str(pki.ca_cert), "--cert", str(pki.client_cert), "--key", str(pki.client_key),
        "--server-name", "localhost",
        "--token-file", str(token), "--keyring", str(keyring),
        "--target-id", TARGET, "--site-id", identity().site_id,
        "--device-id", identity().device_id, "--worker-build", identity().worker_build,
        "--task-pack-digest", TASK_PACK_DIGEST,
        "--run-id", "run-1", "--session-id", "session-1",
        "--task", V1_TASK, "--fingerprint", TASK_FINGERPRINT,
        "--source-file", str(source), "--request-id", request_id,
        "--candidate-timeout-ms", "20000", "--flow", flow, "--out", str(out),
    ]
    if retry_source is not None:
        command += ["--retry-source-file", str(retry_source), "--retry-request-id", f"{request_id}-retry"]
    if cancel_source is not None:
        command += ["--cancel-source-file", str(cancel_source), "--cancel-request-id", f"{request_id}-cancel"]
    if expect_attempt_id:
        command += ["--expect-attempt-id", expect_attempt_id]
    if expect_key_id:
        command += ["--expect-key-id", expect_key_id]
    return command


def _run(command: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        command, cwd=GO_DIR, text=True, capture_output=True,
        env=_go_environment(), check=False, timeout=timeout,
    )


def test_go_client_completes_conformance_retry_renewal_rotation_and_revocation(tmp_path, pki):
    frontend = Frontend(tmp_path, pki)
    source = tmp_path / "candidate.metal"
    source.write_text("OK", encoding="utf-8")
    sleeper = tmp_path / "sleeper.metal"
    sleeper.write_text("SLEEP", encoding="utf-8")
    retry_source = tmp_path / "retry.metal"
    retry_source.write_text("FAIL_ONCE", encoding="utf-8")
    token = tmp_path / "grant.token"
    token.write_text(TOKEN, encoding="utf-8")
    token.chmod(0o600)
    server_keyring = KeyringFile({KEY1_ID: KEY1}, frozenset(), KEY1_ID)
    host_keyring = _keyring(tmp_path / "host-keyring.json", {KEY1_ID: KEY1}, active=KEY1_ID)
    report_path = tmp_path / "go-report.json"
    try:
        frontend.start(grants=[make_grant()], keyring=server_keyring)
        completed = _run(_command(
            frontend, token=token, keyring=host_keyring, source=source,
            request_id="go-conformance-1", retry_source=retry_source,
            cancel_source=sleeper, out=report_path,
        ))
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert report_path.is_file()
        report = json.loads(report_path.read_text())
        assert report["ok"] is True
        assert report["client"] == "go"
        steps = {step["name"]: step for step in report["steps"]}
        for name in (
            "unauthenticated_capabilities_rejected",
            "unsupported_required_feature_rejected",
            "capabilities",
            "submit_watch_result",
            "getjob_terminal_snapshot",
            "verify_result_signature",
            "interrupted_before_retry",
            "retry_interrupted_attempt",
            "retry_completed_rejected",
            "cancel_second_job",
        ):
            assert steps[name]["ok"], steps[name]
        assert steps["verify_result_signature"]["attempt_id"]
        assert steps["verify_result_signature"]["key_id"] == KEY1_ID
        assert steps["interrupted_before_retry"]["state"] == 4
        assert steps["interrupted_before_retry"]["attempt_id"]
        assert steps["retry_interrupted_attempt"]["state"] == 5
        assert (
            steps["retry_interrupted_attempt"]["attempt_id"]
            != steps["interrupted_before_retry"]["attempt_id"]
        )
        assert (
            int(steps["retry_interrupted_attempt"]["lease_fence"])
            > int(steps["interrupted_before_retry"]["lease_fence"])
        )
        assert steps["cancel_second_job"]["state"] in {5, 7}
        historical_attempt = steps["verify_result_signature"]["attempt_id"]

        # Normal grant renewal plus a normally rotated HMAC keyring: key-1 stays
        # trusted for the historical result, key-2 signs new results.
        frontend.stop()
        renewed = make_grant(token=RENEWED_TOKEN)
        rotated = KeyringFile({KEY1_ID: KEY1, KEY2_ID: KEY2}, frozenset(), KEY2_ID)
        frontend.start(grants=[renewed], keyring=rotated)
        renewed_token = tmp_path / "renewed.token"
        renewed_token.write_text(RENEWED_TOKEN, encoding="utf-8")
        renewed_token.chmod(0o600)
        host_rotated = _keyring(
            tmp_path / "host-keyring-rotated.json",
            {KEY1_ID: KEY1, KEY2_ID: KEY2}, active=KEY2_ID,
        )
        replay_path = tmp_path / "go-reattach.json"
        replayed = _run(_command(
            frontend, token=renewed_token, keyring=host_rotated, source=source,
            request_id="go-conformance-1", flow="reattach",
            expect_attempt_id=historical_attempt, expect_key_id=KEY1_ID, out=replay_path,
        ))
        assert replayed.returncode == 0, replayed.stdout + replayed.stderr
        replay = json.loads(replay_path.read_text())
        replay_steps = {step["name"]: step for step in replay["steps"]}
        assert replay["ok"] is True
        assert replay_steps["reattach_matches_historical_attempt"]["ok"]
        assert replay_steps["result_uses_expected_key"]["ok"]
        assert replay_steps["verify_result_signature"]["key_id"] == KEY1_ID

        new_path = tmp_path / "go-rotated.json"
        fresh = _run(_command(
            frontend, token=renewed_token, keyring=host_rotated, source=source,
            request_id="go-rotated-1", flow="reattach",
            expect_key_id=KEY2_ID, out=new_path,
        ))
        assert fresh.returncode == 0, fresh.stdout + fresh.stderr
        fresh_report = json.loads(new_path.read_text())
        fresh_steps = {step["name"]: step for step in fresh_report["steps"]}
        assert fresh_steps["result_uses_expected_key"]["ok"]
        assert fresh_steps["verify_result_signature"]["key_id"] == KEY2_ID

        # Revoking the active key in the host keyring refuses the same result.
        host_revoked = _keyring(
            tmp_path / "host-keyring-revoked.json",
            {KEY1_ID: KEY1, KEY2_ID: KEY2}, revoked=(KEY2_ID,), active=KEY1_ID,
        )
        refused_path = tmp_path / "go-refused.json"
        refused = _run(_command(
            frontend, token=renewed_token, keyring=host_revoked, source=source,
            request_id="go-rotated-1", flow="reattach", out=refused_path,
        ))
        assert refused.returncode != 0
        assert "revoked" in (refused.stdout + refused.stderr).lower()
    finally:
        frontend.stop()
