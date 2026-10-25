"""External Python client and wire-conformance flow over real TLS/mTLS."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tests.support.job_engine import (
    TARGET,
    TASK_FINGERPRINT,
    TASK_PACK_DIGEST,
    TOKEN,
    engine as make_engine,
    identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "scripts" / "remote_evaluator_client.py"
SIGNING_KEY = b"c" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"

pytestmark = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl is required for TLS material"
)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    try:
        return make_pki(tmp_path_factory.mktemp("pki"))
    except PkiUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


@pytest.fixture
def frontend(tmp_path, pki):
    engine = make_engine(
        tmp_path / "state", signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID,
        worker_env={"FAKE_JOB_RUNNING": str(tmp_path / "running.json")},
    )
    server = build_server(
        engine,
        config=GrpcServerConfig(
            address="127.0.0.1:0", allow_insecure=False, require_client_identity=True,
            certificate_file=pki.server_cert, private_key_file=pki.server_key,
            client_ca_file=pki.ca_cert, watch_poll_s=0.02,
        ),
    )
    token_file = tmp_path / "grant.token"
    token_file.write_text(TOKEN)
    keyring_file = pki.keyring_file(key_id=SIGNING_KEY_ID, key=SIGNING_KEY)
    source = tmp_path / "candidate.metal"
    source.write_text("OK", encoding="utf-8")
    sleeper = tmp_path / "sleeper.metal"
    sleeper.write_text("SLEEP", encoding="utf-8")
    yield {
        "engine": engine, "server": server, "pki": pki, "token": token_file,
        "keyring": keyring_file, "source": source, "sleeper": sleeper,
    }
    server.stop(grace_s=0.2)
    engine.close()


def _common(frontend, *extra: str) -> list[str]:
    pki = frontend["pki"]
    return [
        "--endpoint", f"127.0.0.1:{frontend['server'].port}",
        "--ca", str(pki.ca_cert),
        "--client-cert", str(pki.client_cert),
        "--client-key", str(pki.client_key),
        "--server-name", "localhost",
        "--token-file", str(frontend["token"]),
        "--keyring", str(frontend["keyring"]),
        "--target-id", TARGET,
        "--run-id", "run-1",
        "--session-id", "session-1",
        "--expected-site-id", identity().site_id,
        "--expected-device-id", identity().device_id,
        "--expected-worker-build", identity().worker_build,
        "--expected-task-pack-digest", TASK_PACK_DIGEST,
        *extra,
    ]


def _run(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CLIENT), *arguments], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )


def test_external_capabilities_and_submit_cli(frontend):
    capabilities = _run("capabilities", *_common(frontend))
    assert capabilities.returncode == 0, capabilities.stdout + capabilities.stderr
    document = json.loads(capabilities.stdout)
    assert document["ok"] is True
    assert document["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
    assert document["evidence_contract_id"] == "tcenv.remote.v1.evidence.1"
    assert document["evaluator_epoch"]

    submitted = _run(
        "submit", *_common(frontend),
        "--source-file", str(frontend["source"]),
        "--task", "relu", "--fingerprint", TASK_FINGERPRINT,
        "--request-id", "external-submit-1",
    )
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    result = json.loads(submitted.stdout)["result"]
    assert result["status"] == "ok"
    assert result["attempt_id"]
    assert result["signature_key_id"] == SIGNING_KEY_ID
    assert result["measurement_kind"] == "simulated"


def test_external_conformance_client_completes_the_flow(frontend):
    completed = _run(
        "conformance", *_common(frontend),
        "--source-file", str(frontend["source"]),
        "--task", "relu", "--fingerprint", TASK_FINGERPRINT,
        "--request-id", "conformance-primary",
        "--cancel-source-file", str(frontend["sleeper"]),
        "--cancel-request-id", "conformance-cancel",
        "--candidate-timeout-ms", "20000",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(completed.stdout)
    assert report["ok"] is True
    steps = {step["name"]: step for step in report["steps"]}
    assert steps["unauthenticated_capabilities_rejected"]["ok"]
    assert steps["unsupported_required_feature_rejected"]["ok"]
    assert steps["capabilities"]["ok"]
    assert steps["submit_watch_result"]["ok"]
    assert steps["getjob_terminal_snapshot"]["ok"]
    assert steps["retry_completed_rejected"]["ok"]
    assert steps["cancel_second_job"]["ok"]
    assert steps["cancel_second_job"]["state"] in {"cancelled", "completed"}


def test_external_client_rejects_a_wrong_client_identity(tmp_path, frontend):
    pki = frontend["pki"]
    arguments = _common(frontend)
    arguments[arguments.index("--client-cert") + 1] = str(pki.other_client_cert)
    arguments[arguments.index("--client-key") + 1] = str(pki.other_client_key)
    failed = _run("capabilities", *arguments)
    assert failed.returncode != 0
    assert "UNAVAILABLE" in failed.stdout or "unavailable" in failed.stdout.lower()


def test_external_client_requires_a_trusted_keyring(frontend, tmp_path):
    other = tmp_path / "other-keyring.json"
    other.write_text(json.dumps({
        "schema_version": 1, "keys": {"sim-external-site/key-9": "d" * 64}, "revoked": [],
    }))
    other.chmod(0o600)
    arguments = _common(frontend)
    arguments[arguments.index("--keyring") + 1] = str(other)
    failed = _run(
        "submit", *arguments,
        "--source-file", str(frontend["source"]),
        "--task", "relu", "--fingerprint", TASK_FINGERPRINT,
    )
    assert failed.returncode != 0
    assert "signature" in failed.stdout.lower()
