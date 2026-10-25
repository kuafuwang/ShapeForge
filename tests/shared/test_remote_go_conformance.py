"""Non-Python (Go) gRPC conformance client against the local TLS frontend.

The Go module vendors no generated stubs: it loads the published descriptor set
and verifies the request digest, receipt and HMAC result signature itself.  The
test skips only when the Go toolchain or its module cache is unavailable; with
Go present it must actually run.
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
from tests.support.job_engine import (
    TARGET, TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN, engine as make_engine, identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki

ROOT = Path(__file__).resolve().parents[2]
GO_DIR = ROOT / "tests" / "support" / "remotegrpc"
DESC = ROOT / "proto" / "tcenv" / "remote" / "v1" / "evaluator.desc"
SIGNING_KEY = b"g" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"

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


def test_go_client_completes_capabilities_submit_watch_getjob_retry_cancel(tmp_path, pki):
    engine = make_engine(tmp_path / "state", signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID)
    server = build_server(
        engine,
        config=GrpcServerConfig(
            address="127.0.0.1:0", allow_insecure=False, require_client_identity=True,
            certificate_file=pki.server_cert, private_key_file=pki.server_key,
            client_ca_file=pki.ca_cert, watch_poll_s=0.02,
        ),
    )
    try:
        token = tmp_path / "grant.token"
        token.write_text(TOKEN)
        keyring = pki.keyring_file(key_id=SIGNING_KEY_ID, key=SIGNING_KEY)
        source = tmp_path / "candidate.metal"
        source.write_text("OK", encoding="utf-8")
        sleeper = tmp_path / "sleeper.metal"
        sleeper.write_text("SLEEP", encoding="utf-8")
        report_path = tmp_path / "go-report.json"
        command = [
            "go", "run", ".",
            "--desc", str(DESC),
            "--endpoint", f"127.0.0.1:{server.port}",
            "--ca", str(pki.ca_cert), "--cert", str(pki.client_cert), "--key", str(pki.client_key),
            "--server-name", "localhost",
            "--token-file", str(token), "--keyring", str(keyring),
            "--target-id", TARGET, "--run-id", "run-1", "--session-id", "session-1",
            "--task", "relu", "--fingerprint", TASK_FINGERPRINT,
            "--source-file", str(source), "--request-id", "go-conformance-1",
            "--cancel-source-file", str(sleeper), "--cancel-request-id", "go-conformance-cancel-1",
            "--candidate-timeout-ms", "20000", "--out", str(report_path),
        ]
        completed = subprocess.run(
            command, cwd=GO_DIR, text=True, capture_output=True,
            env=_go_environment(), check=False, timeout=600,
        )
    finally:
        server.stop(grace_s=0.2)
        engine.close()
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
        "retry_completed_rejected",
        "cancel_second_job",
    ):
        assert steps[name]["ok"], steps[name]
    assert steps["verify_result_signature"]["attempt_id"]
    assert steps["cancel_second_job"]["state"] in {5, 7}
