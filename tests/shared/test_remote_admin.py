"""Provisioning/rotation tooling and the production gRPC composition root."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.remote.grpc_server import load_client_credentials, load_server_credentials
from tcenv.remote.grants import load_grants_file
from tcenv.remote.keyring import load_keyring_file
from tcenv.remote.task_pack import production_fingerprint_provider

ROOT = Path(__file__).resolve().parents[2]
ADMIN = ROOT / "scripts" / "remote_evaluator_admin.py"


def _admin(*arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(
        [sys.executable, str(ADMIN), *arguments], cwd=ROOT, text=True,
        capture_output=True, check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(completed.stdout + completed.stderr)
    return completed


def test_keyring_init_rotate_revoke_keeps_history_explicit(tmp_path):
    path = tmp_path / "keyring.json"
    _admin("keyring-init", "--out", str(path), "--key-id", "site/key-1")
    first = load_keyring_file(path)
    assert first.active_key_id == "site/key-1"
    assert path.stat().st_mode & 0o077 == 0
    _admin("keyring-rotate", "--file", str(path), "--key-id", "site/key-2")
    rotated = load_keyring_file(path)
    assert rotated.active_key_id == "site/key-2"
    assert set(rotated.trusted) == {"site/key-1", "site/key-2"}
    # Revoking the active key requires an explicit replacement.
    failed = _admin("keyring-revoke", "--file", str(path), "--key-id", "site/key-2", check=False)
    assert failed.returncode != 0 and "new-active" in (failed.stdout + failed.stderr)
    _admin("keyring-revoke", "--file", str(path), "--key-id", "site/key-2", "--new-active", "site/key-1")
    revoked = load_keyring_file(path)
    assert revoked.active_key_id == "site/key-1"
    assert "site/key-2" in revoked.revoked and "site/key-2" not in revoked.trusted
    shown = _admin("keyring-show", "--file", str(path))
    assert "revoked" in shown.stdout and "site/key-2" in shown.stdout


def test_grant_mint_appends_grants_and_writes_a_private_token_file(tmp_path):
    grants_path = tmp_path / "grants.json"
    token_path = tmp_path / "run-1-session-1.token"
    _admin(
        "grant-mint", "--out", str(grants_path), "--token-out", str(token_path),
        "--run-id", "run-1", "--session-id", "session-1", "--target-id", "target-1",
        "--subject", "host-evaluator", "--kinds", "formal_eval", "--expires-in", "60",
    )
    _admin(
        "grant-mint", "--out", str(grants_path), "--append",
        "--token-out", str(tmp_path / "run-1-session-2.token"),
        "--run-id", "run-1", "--session-id", "session-2", "--target-id", "target-1",
        "--subject", "host-evaluator", "--kinds", "selfcheck", "--modes", "timed",
        "--expires-in", "0",
    )
    grants = load_grants_file(grants_path)
    assert [grant.session_id for grant in grants] == ["session-1", "session-2"]
    assert grants[1].expires_at is None and grants[0].expires_at is not None
    assert grants[1].allowed_kinds == frozenset({"selfcheck"})
    assert token_path.stat().st_mode & 0o077 == 0
    assert token_path.read_text().strip() == grants[0].token
    shown = _admin("grants-show", "--file", str(grants_path))
    assert "session-1" in shown.stdout and grants[0].token not in shown.stdout
    duplicate = _admin(
        "grant-mint", "--out", str(grants_path), "--append",
        "--run-id", "run-1", "--session-id", "session-1", "--target-id", "target-1",
        "--subject", "host-evaluator", "--kinds", "formal_eval", check=False,
    )
    assert duplicate.returncode != 0 and "already exists" in (duplicate.stdout + duplicate.stderr)
    renewed_token = tmp_path / "renewed.token"
    _admin(
        "grant-mint", "--out", str(grants_path), "--append", "--renew",
        "--token-out", str(renewed_token),
        "--run-id", "run-1", "--session-id", "session-1", "--target-id", "target-1",
        "--subject", "host-evaluator", "--kinds", "formal_eval", "--expires-in", "60",
    )
    rotated = load_grants_file(grants_path)
    matching = [grant for grant in rotated if grant.session_id == "session-1"]
    assert len(matching) == 1
    assert matching[0].token == renewed_token.read_text().strip()
    assert matching[0].token != grants[0].token


@pytest.mark.skipif(shutil.which("openssl") is None, reason="openssl is required for certificate tooling")
def test_ca_and_certificate_tooling_produces_loadable_tls_material(tmp_path):
    pki = tmp_path / "pki"
    _admin("ca-init", "--out", str(pki), "--days", "2")
    _admin(
        "cert-issue", "--ca", str(pki), "--out", str(pki), "--name", "evaluator",
        "--kind", "server", "--san", "DNS:localhost,IP:127.0.0.1", "--days", "2",
    )
    _admin(
        "cert-issue", "--ca", str(pki), "--out", str(pki), "--name", "host",
        "--kind", "client", "--days", "2",
    )
    server = load_server_credentials(
        certificate_file=pki / "evaluator.pem", private_key_file=pki / "evaluator.key",
        client_ca_file=pki / "ca.pem",
    )
    client = load_client_credentials(
        ca_file=pki / "ca.pem", certificate_file=pki / "host.pem", private_key_file=pki / "host.key",
    )
    assert server is not None and client is not None
    assert (pki / "evaluator.key").stat().st_mode & 0o077 == 0
    assert (pki / "host.key").stat().st_mode & 0o077 == 0


def _task_pack(root: Path) -> tuple[Path, str]:
    from tcenv.remote.task_pack import load_task_pack

    (root / "sources").mkdir(parents=True)
    (root / "sources" / "relu.txt").write_text("relu reference\n", encoding="utf-8")
    (root / "pack.json").write_text(json.dumps({
        "schema_version": 1, "backend": "metal", "target_id": "apple-m4-max",
        "reference_versions": {"naive": "v1", "baseline": "v1"},
    }), encoding="utf-8")
    fingerprint = production_fingerprint_provider("metal")({
        "task": "relu", "dtype": "fp16", "size": "small",
    })
    (root / "fingerprints.json").write_text(json.dumps({
        "schema_version": 1,
        "tasks": [{"task": "relu", "dtype": "fp16", "size": "small", "fingerprint": fingerprint}],
    }), encoding="utf-8")
    pack = load_task_pack(root)
    return root, pack.digest


def test_production_grpc_entry_check_validates_pack_keyring_and_grants(tmp_path):
    pack_root, digest = _task_pack(tmp_path / "pack")
    keyring = tmp_path / "keyring.json"
    _admin("keyring-init", "--out", str(keyring), "--key-id", "sim-site/key-1")
    grants = tmp_path / "grants.json"
    _admin(
        "grant-mint", "--out", str(grants), "--run-id", "run-1", "--session-id", "session-1",
        "--target-id", "apple-m4-max", "--kinds", "formal_eval",
    )
    tls = tmp_path / "tls"
    tls.mkdir()
    for name in ("server.pem", "server.key", "ca.pem"):
        path = tls / name
        path.write_text("placeholder")
        path.chmod(0o600)
    base = [
        "--check",
        "--state-dir", str(tmp_path / "state"),
        "--site-id", "sim-site", "--device-id", "device-1",
        "--backend", "metal", "--backend-version", "1",
        "--target-id", "apple-m4-max", "--worker-build", "build-1",
        "--calibration-digest", "c" * 64,
        "--task-pack-root", str(pack_root), "--expected-task-pack-digest", digest,
        "--grants-file", str(grants), "--signing-keyring-file", str(keyring),
        "--certificate-file", str(tls / "server.pem"),
        "--private-key-file", str(tls / "server.key"),
        "--client-ca-file", str(tls / "ca.pem"),
    ]
    process_base = base + ["--worker-execution", "process", "--allow-process-worker"]
    completed = subprocess.run(
        [sys.executable, "-m", "tcenv.remote.grpc_service", *process_base],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "task_pack_digest=" + digest in completed.stdout
    assert "active_key_id=sim-site/key-1" in completed.stdout
    assert "execution=process" in completed.stdout
    assert not (tmp_path / "state").exists()

    # The production external entry point uses disposable containers by
    # default and refuses every ambiguous configuration instead of falling back.
    digest_hex = "d" * 64
    container_base = list(base)
    container_base[container_base.index("--worker-build") + 1] = f"build-1+img:{digest_hex}"
    container_base += [
        "--worker-image", "taic-worker:local", "--worker-image-digest", digest_hex,
    ]
    container_check = subprocess.run(
        [sys.executable, "-m", "tcenv.remote.grpc_service", *container_base],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert container_check.returncode == 0, container_check.stdout + container_check.stderr
    assert "execution=container" in container_check.stdout
    assert f"image_digest={digest_hex}" in container_check.stdout
    assert not (tmp_path / "state").exists()

    refusals = [
        ["--worker-module", "tests.support.fake_worker_job_engine"],
        ["--measurement-kind", "simulated"],
        # No silent fallback: an unset image, an unpinned digest, a build id
        # that does not bind the digest, or process mode without opt-in.
        ["--worker-image", ""],
        ["--worker-image-digest", ""],
        ["--worker-build", "build-1"],
        ["--worker-execution", "process"],
        ["--worker-runtime-profile", "gvisor", "--worker-runtime", "runsc", "--worker-image-digest", "x"],
    ]
    for replacement in refusals:
        arguments = [argument for argument in container_base if argument not in ("--check",)]
        arguments.insert(0, "--check")
        index = 0
        while index < len(arguments):
            if arguments[index] == replacement[0]:
                del arguments[index:index + 2]
            else:
                index += 1
        arguments.extend(replacement)
        refused = subprocess.run(
            [sys.executable, "-m", "tcenv.remote.grpc_service", *arguments],
            cwd=ROOT, text=True, capture_output=True, check=False,
        )
        assert refused.returncode != 0, (replacement, refused.stdout, refused.stderr)

    # A local process worker must not claim an image-pinned build identity: the
    # signed result would name an image that never executed.
    process_with_image = list(base) + [
        "--worker-execution", "process", "--allow-process-worker",
    ]
    process_with_image[process_with_image.index("--worker-build") + 1] = f"build-1+img:{digest_hex}"
    refused = subprocess.run(
        [sys.executable, "-m", "tcenv.remote.grpc_service", *process_with_image],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert refused.returncode != 0
    assert "image-pinned" in refused.stdout + refused.stderr


def test_generated_secret_files_are_never_group_readable(tmp_path):
    grants = tmp_path / "grants.json"
    _admin(
        "grant-mint", "--out", str(grants), "--run-id", "run-1", "--session-id", "s",
        "--target-id", "t", "--kinds", "formal_eval",
    )
    assert os.stat(grants).st_mode & 0o077 == 0
