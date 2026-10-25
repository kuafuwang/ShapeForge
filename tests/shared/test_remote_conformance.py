"""External Python client and wire-conformance flow over real TLS/mTLS.

The flow is exercised twice: first as a full conformance run (including a forced
interrupted attempt resumed through the explicit Retry RPC), then as a
re-attachment run after normal grant renewal and HMAC key rotation.  Revoked
certificates are covered by ``test_remote_grpc_revocation.py``; revoked grants
and keys are covered here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tcenv.remote.grants import write_grants_file
from tcenv.remote.identity import SessionGrant
from tcenv.remote.keyring import KeyringFile, write_keyring_file
from tests.support.job_engine import (
    TARGET,
    TASK_FINGERPRINT,
    TASK_PACK_DIGEST,
    TOKEN,
    engine as make_engine,
    grant as make_grant,
    identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki

ROOT = Path(__file__).resolve().parents[2]
CLIENT = ROOT / "scripts" / "remote_evaluator_client.py"
KEY1 = b"c" * 32
KEY2 = b"e" * 32
KEY1_ID = "sim-external-site/key-1"
KEY2_ID = "sim-external-site/key-2"
RENEWED_TOKEN = "external-renewed-grant-token"

pytestmark = pytest.mark.skipif(
    shutil.which("openssl") is None, reason="openssl is required for TLS material"
)


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    try:
        return make_pki(tmp_path_factory.mktemp("pki"))
    except PkiUnavailable as exc:  # pragma: no cover
        pytest.skip(str(exc))


class Frontend:
    """A restartable TLS/mTLS frontend sharing one durable state directory."""

    def __init__(self, root: Path, pki) -> None:
        self.root = root
        self.pki = pki
        self.state = root / "state"
        self.engine = None
        self.server = None
        self.token_path = root / "grant.token"
        self.keyring_path = root / "host-keyring.json"
        self.counter = 0

    @property
    def port(self) -> int:
        assert self.server is not None
        return self.server.port

    def start(
        self, *, grants: list[SessionGrant], keyring: KeyringFile,
        host_keyring: KeyringFile | None = None,
    ) -> None:
        self.stop()
        self.counter += 1
        grants_path = self.root / f"grants-{self.counter}.json"
        write_grants_file(grants_path, grants)
        self.token_path.write_text(grants[0].token, encoding="utf-8")
        self.token_path.chmod(0o600)
        active_key_id, active_key = keyring.require_active()
        self.engine = make_engine(
            self.state,
            grants=grants,
            signing_key=active_key,
            signing_key_id=active_key_id,
            signing_keyring=keyring.trusted,
            worker_env={
                "FAKE_JOB_RUNNING": str(self.root / "running.json"),
                "FAKE_JOB_FAIL_DIR": str(self.root / "fail-once"),
            },
        )
        write_keyring_file(self.keyring_path, host_keyring or keyring)
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


@pytest.fixture
def frontend(tmp_path, pki):
    scenario = Frontend(tmp_path, pki)
    source = tmp_path / "candidate.metal"
    source.write_text("OK", encoding="utf-8")
    sleeper = tmp_path / "sleeper.metal"
    sleeper.write_text("SLEEP", encoding="utf-8")
    scenario.source = source
    scenario.sleeper = sleeper
    scenario.start(grants=[make_grant()], keyring=KeyringFile({KEY1_ID: KEY1}, frozenset(), KEY1_ID))
    yield scenario
    scenario.stop()


def _common(frontend: Frontend, *extra: str) -> list[str]:
    pki = frontend.pki
    return [
        "--endpoint", f"127.0.0.1:{frontend.port}",
        "--ca", str(pki.ca_cert),
        "--client-cert", str(pki.client_cert),
        "--client-key", str(pki.client_key),
        "--server-name", "localhost",
        "--token-file", str(frontend.token_path),
        "--keyring", str(frontend.keyring_path),
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


def _request_arguments(frontend: Frontend, *, request_id: str, source: Path | None = None) -> list[str]:
    return [
        "--source-file", str(source or frontend.source),
        "--task", "relu", "--fingerprint", TASK_FINGERPRINT,
        "--request-id", request_id,
    ]


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
        *_request_arguments(frontend, request_id="external-submit-1"),
    )
    assert submitted.returncode == 0, submitted.stdout + submitted.stderr
    result = json.loads(submitted.stdout)["result"]
    assert result["status"] == "ok"
    assert result["attempt_id"]
    assert result["signature_key_id"] == KEY1_ID
    assert result["measurement_kind"] == "simulated"


def test_external_conformance_client_completes_the_flow(frontend):
    retry_source = frontend.root / "retry.metal"
    retry_source.write_text("FAIL_ONCE", encoding="utf-8")
    completed = _run(
        "conformance", *_common(frontend),
        *_request_arguments(frontend, request_id="conformance-primary"),
        "--retry-source-file", str(retry_source),
        "--retry-request-id", "conformance-retry",
        "--cancel-source-file", str(frontend.sleeper),
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
    assert steps["interrupted_before_retry"]["ok"], steps["interrupted_before_retry"]
    assert steps["submit_watch_result_after_retry"]["ok"], steps["submit_watch_result_after_retry"]
    assert steps["interrupted_before_retry"]["attempt_id"]
    assert (
        steps["submit_watch_result_after_retry"]["attempt_id"]
        != steps["interrupted_before_retry"]["attempt_id"]
    )
    assert (
        int(steps["submit_watch_result_after_retry"]["lease_fence"])
        > int(steps["interrupted_before_retry"]["lease_fence"])
    )
    assert steps["cancel_second_job"]["ok"]
    assert steps["cancel_second_job"]["state"] in {"cancelled", "completed"}
    assert report["primary"]["request_digest"]
    assert report["primary"]["attempt_id"] == steps["submit_watch_result"]["attempt_id"]


def test_external_client_rejects_a_wrong_client_identity(tmp_path, frontend):
    pki = frontend.pki
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
        *_request_arguments(frontend, request_id="wrong-keyring"),
    )
    assert failed.returncode != 0
    assert "signature" in failed.stdout.lower()


def test_external_grant_renewal_and_hmac_rotation_and_revocation(tmp_path, pki):
    """Renewed grants re-attach; normal rotation keeps history; revocation fails closed."""

    scenario = Frontend(tmp_path, pki)
    scenario.source = tmp_path / "candidate.metal"
    scenario.source.write_text("OK", encoding="utf-8")
    scenario.sleeper = tmp_path / "sleeper.metal"
    scenario.sleeper.write_text("SLEEP", encoding="utf-8")
    retry_source = tmp_path / "retry.metal"
    retry_source.write_text("FAIL_ONCE", encoding="utf-8")
    try:
        key1 = KeyringFile({KEY1_ID: KEY1}, frozenset(), KEY1_ID)
        scenario.start(grants=[make_grant()], keyring=key1)
        primary = _run(
            "conformance", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-primary"),
            "--retry-source-file", str(retry_source),
            "--retry-request-id", "rotation-retry",
            "--candidate-timeout-ms", "20000",
        )
        assert primary.returncode == 0, primary.stdout + primary.stderr
        first = json.loads(primary.stdout)
        assert first["ok"] is True
        historical_attempt = first["primary"]["attempt_id"]
        historical_digest = first["primary"]["request_digest"]
        assert first["primary"]["signature_key_id"] == KEY1_ID

        # Normal renewal: a new token for the same subject/run/session scope and
        # a normally rotated keyring that keeps key-1 for historical results.
        scenario.stop()
        renewed = make_grant(token=RENEWED_TOKEN)
        rotated = KeyringFile({KEY1_ID: KEY1, KEY2_ID: KEY2}, frozenset(), KEY2_ID)
        scenario.start(grants=[renewed], keyring=rotated)

        old_token = tmp_path / "old.token"
        old_token.write_text(TOKEN, encoding="utf-8")
        old_token.chmod(0o600)
        stale = _run(
            "capabilities", *_common(scenario),
            "--token-file", str(old_token),
        )
        assert stale.returncode != 0
        assert "superseded" in stale.stdout.lower() or "UNAUTHENTICATED" in stale.stdout

        # The renewed grant re-attaches to the accepted job and still verifies
        # the historical key-1 signature.
        reattached = _run(
            "reattach", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-primary"),
            "--candidate-timeout-ms", "20000",
            "--expect-attempt-id", historical_attempt,
            "--expect-key-id", KEY1_ID,
        )
        assert reattached.returncode == 0, reattached.stdout + reattached.stderr
        reattach_document = json.loads(reattached.stdout)
        assert reattach_document["ok"] is True
        assert reattach_document["result"]["attempt_id"] == historical_attempt
        assert reattach_document["result"]["signature_key_id"] == KEY1_ID

        # A new job is signed with the rotated active key.
        fresh = _run(
            "submit", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-new"),
        )
        assert fresh.returncode == 0, fresh.stdout + fresh.stderr
        fresh_result = json.loads(fresh.stdout)["result"]
        assert fresh_result["signature_key_id"] == KEY2_ID
        new_attempt = fresh_result["attempt_id"]

        # Host-side revocation of the active key refuses the new result even
        # though the server still serves it.
        scenario.stop()
        scenario.start(grants=[renewed], keyring=rotated)
        revoked_host = KeyringFile({KEY1_ID: KEY1, KEY2_ID: KEY2}, frozenset({KEY2_ID}), "")
        write_keyring_file(scenario.keyring_path, revoked_host)
        refused = _run(
            "reattach", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-new"),
        )
        assert refused.returncode != 0
        refused_document = json.loads(refused.stdout)
        assert refused_document["code"] == "signature_key_revoked", refused_document

        # Server-side revocation of the historical key withholds only the
        # affected result; the frontend still restarts and serves new jobs.
        scenario.stop()
        revoked_server = KeyringFile({KEY2_ID: KEY2}, frozenset({KEY1_ID}), KEY2_ID)
        scenario.start(grants=[renewed], keyring=revoked_server, host_keyring=rotated)
        withheld = _run(
            "reattach", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-primary"),
            "--candidate-timeout-ms", "20000",
        )
        assert withheld.returncode != 0
        withheld_document = json.loads(withheld.stdout)
        assert withheld_document["code"] == "result_missing", withheld_document
        after_revocation = _run(
            "submit", *_common(scenario),
            *_request_arguments(scenario, request_id="rotation-after-revocation"),
        )
        assert after_revocation.returncode == 0, after_revocation.stdout + after_revocation.stderr
        assert json.loads(after_revocation.stdout)["result"]["signature_key_id"] == KEY2_ID
        assert historical_digest
        assert new_attempt
    finally:
        scenario.stop()
