"""Per-certificate client revocation for the external evaluator frontend.

Python's gRPC TLS stack has no CRL/OCSP support, so the frontend enforces the
deployment's static leaf-certificate deny list (SHA-256 of DER) before any
session grant is authorized.  These tests use real TLS/mTLS material and an
actually revoked client certificate.
"""

from __future__ import annotations

import json
import secrets

import grpc
import pytest

from tcenv.remote import contracts
from tcenv.remote.grpc_server import GrpcServerConfig, build_server
from tcenv.remote.revocation import (
    RevocationError,
    certificate_sha256,
    load_revocation_file,
    write_revocation_file,
)
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.v1 import evaluator_pb2_grpc as pb_grpc
from tests.support.job_engine import (
    TARGET, TASK_PACK_DIGEST, TOKEN, engine as make_engine, identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("openssl") is None,
    reason="openssl is required for disposable test certificates",
)

SIGNING_KEY = b"r" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    try:
        return make_pki(tmp_path_factory.mktemp("revocation-pki"))
    except PkiUnavailable as exc:  # pragma: no cover - only on hosts without openssl
        pytest.skip(str(exc))


def _server(tmp_path, pki, *, revoked=frozenset()):
    engine = make_engine(
        tmp_path / "state", signing_key=SIGNING_KEY, signing_key_id=SIGNING_KEY_ID,
    )
    server = build_server(
        engine,
        config=GrpcServerConfig(
            address="127.0.0.1:0",
            allow_insecure=False,
            require_client_identity=True,
            certificate_file=pki.server_cert,
            private_key_file=pki.server_key,
            client_ca_file=pki.ca_cert,
            watch_poll_s=0.02,
            revoked_client_certificates=frozenset(revoked),
        ),
    )
    return engine, server


def _channel(pki, port: int, *, certificate, key):
    from tcenv.remote.grpc_client import load_client_credentials

    return grpc.secure_channel(
        f"127.0.0.1:{port}",
        load_client_credentials(
            ca_file=pki.ca_cert, certificate_file=certificate, private_key_file=key,
        ),
        options=[("grpc.ssl_target_name_override", "localhost")],
    )


def _capabilities(port: int, pki, *, certificate, key):
    channel = _channel(pki, port, certificate=certificate, key=key)
    try:
        stub = pb_grpc.EvaluatorStub(channel)
        return stub.GetCapabilities(
            pb.CapabilitiesRequest(
                api_contract_ids=[contracts.API_CONTRACT_ID],
                evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
                client_nonce=secrets.token_hex(16),
                expected=pb.ExpectedEvaluator(
                    site_id=identity().site_id, device_id=identity().device_id,
                    target_id=TARGET, worker_build=identity().worker_build,
                    task_pack_digest=TASK_PACK_DIGEST,
                ),
            ),
            metadata=(("authorization", f"Bearer {TOKEN}"),),
            timeout=10,
        )
    finally:
        channel.close()


def test_normal_client_certificate_rotation_is_accepted(tmp_path, pki):
    engine, server = _server(tmp_path, pki)
    try:
        for certificate, key in (
            (pki.client_cert, pki.client_key),
            (pki.rotated_client_cert, pki.rotated_client_key),
        ):
            capabilities = _capabilities(server.port, pki, certificate=certificate, key=key)
            assert capabilities.capability_id
    finally:
        server.stop(grace_s=0.2)
        engine.close()


def test_an_actually_revoked_client_certificate_is_rejected(tmp_path, pki):
    fingerprint = certificate_sha256(pki.rotated_client_cert.read_bytes())
    revoked = write_revocation_file(tmp_path / "revoked.json", frozenset({fingerprint}))
    assert load_revocation_file(revoked) == frozenset({fingerprint})
    engine, server = _server(tmp_path, pki, revoked=frozenset({fingerprint}))
    try:
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(
                server.port, pki,
                certificate=pki.rotated_client_cert, key=pki.rotated_client_key,
            )
        assert excinfo.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert "revoked" in excinfo.value.details()
        # The non-revoked identity on the same CA is still served.
        capabilities = _capabilities(
            server.port, pki, certificate=pki.client_cert, key=pki.client_key,
        )
        assert capabilities.capability_id
    finally:
        server.stop(grace_s=0.2)
        engine.close()


def test_revocation_list_is_strict_and_private(tmp_path):
    path = tmp_path / "revoked.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "revoked_client_certificates_sha256": ["a" * 64],
        "extra": True,
    }), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(RevocationError, match="exactly"):
        load_revocation_file(path)
    path.write_text(json.dumps({
        "schema_version": 1,
        "revoked_client_certificates_sha256": ["A" * 64],
    }), encoding="utf-8")
    with pytest.raises(RevocationError, match="lowercase hex"):
        load_revocation_file(path)
    write_revocation_file(path, frozenset())
    path.chmod(0o644)
    with pytest.raises(RevocationError, match="0600"):
        load_revocation_file(path)


def test_certificate_fingerprint_matches_the_der_bytes(pki):
    import hashlib
    import ssl

    pem = pki.client_cert.read_text(encoding="ascii")
    der = ssl.PEM_cert_to_DER_cert(pem)
    assert certificate_sha256(pem) == hashlib.sha256(der).hexdigest()
    assert certificate_sha256(pki.client_cert.read_bytes()) == certificate_sha256(pem)
