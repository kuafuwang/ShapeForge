"""TLS/mTLS tests for the external evaluator gRPC ingress."""

from __future__ import annotations

import hashlib

import grpc
import pytest

from tcenv.remote import contracts
from tcenv.remote import grpc_contract as gc
from tcenv.remote.grpc_client import GrpcEvaluatorFactory, GrpcExpected, SignatureKeyring, load_client_credentials
from tcenv.remote.grpc_server import GrpcServerConfig, build_server, load_server_credentials
from tcenv.remote.v1 import evaluator_pb2 as pb
from tcenv.remote.v1 import evaluator_pb2_grpc as pb_grpc
from tests.support.job_engine import (
    TARGET,
    TASK_FINGERPRINT,
    TASK_PACK_DIGEST,
    TOKEN,
    engine as make_engine,
    identity,
)
from tests.support.remote_pki import PkiUnavailable, make_pki

pytestmark = pytest.mark.skipif(
    __import__("shutil").which("openssl") is None,
    reason="openssl is required for disposable test certificates",
)

SIGNING_KEY = b"t" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"


@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    try:
        return make_pki(tmp_path_factory.mktemp("pki"))
    except PkiUnavailable as exc:  # pragma: no cover - only on hosts without openssl
        pytest.skip(str(exc))


@pytest.fixture
def tls_server(tmp_path, pki):
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
        ),
    )
    yield engine, server, pki
    server.stop(grace_s=0.2)
    engine.close()


def _channel(address: str, pki, *, ca=None, certificate=None, key=None, server_name="localhost"):
    credentials = load_client_credentials(
        ca_file=ca or pki.ca_cert,
        certificate_file=pki.client_cert if certificate is None else certificate,
        private_key_file=pki.client_key if key is None else key,
    )
    return grpc.secure_channel(
        address, credentials, options=[("grpc.ssl_target_name_override", server_name)],
    )


def _capabilities(stub, *, metadata=(("authorization", f"Bearer {TOKEN}"),)):
    return stub.GetCapabilities(
        pb.CapabilitiesRequest(
            api_contract_ids=[contracts.API_CONTRACT_ID],
            evidence_contract_ids=[contracts.EVIDENCE_CONTRACT_ID],
            client_nonce="e" * 16,
        ),
        metadata=metadata,
        timeout=10,
    )


def test_mtls_round_trip_through_the_grpc_client(tmp_path, tls_server):
    engine, server, pki = tls_server
    factory = GrpcEvaluatorFactory(
        endpoint=f"127.0.0.1:{server.port}",
        token_provider=lambda _run, _session: TOKEN,
        run_id="run-1",
        target_id=TARGET,
        expected=GrpcExpected(
            site_id=identity().site_id, device_id=identity().device_id,
            target_id=TARGET, worker_build=identity().worker_build,
            task_pack_digest=TASK_PACK_DIGEST,
        ),
        keyring=SignatureKeyring({SIGNING_KEY_ID: SIGNING_KEY}),
        credentials=load_client_credentials(
            ca_file=pki.ca_cert, certificate_file=pki.client_cert, private_key_file=pki.client_key,
        ),
        server_name_override="localhost",
    )
    try:
        from tcenv.evaluation import EvaluationRequest

        request = EvaluationRequest(
            task="relu", dtype="fp16", size="small", source="OK",
            target_id=TARGET, task_fingerprint=TASK_FINGERPRINT,
            run_id="run-1", session_id="session-1", request_id="tls-round-trip",
        )
        result = factory.formal(request)
        assert result.report["status"] == "ok"
        assert result.receipt["attempt_id"]
    finally:
        factory.close()


@pytest.mark.parametrize("failure", ["missing_client_cert", "wrong_ca", "wrong_san", "untrusted_client_cert"])
def test_mtls_rejects_untrusted_or_missing_peers(tls_server, failure):
    _engine, server, pki = tls_server
    address = f"127.0.0.1:{server.port}"
    if failure == "missing_client_cert":
        credentials = grpc.ssl_channel_credentials(root_certificates=pki.ca_cert.read_bytes())
        channel = grpc.secure_channel(
            address, credentials, options=[("grpc.ssl_target_name_override", "localhost")],
        )
    elif failure == "wrong_ca":
        channel = _channel(address, pki, ca=pki.other_ca_cert)
    elif failure == "wrong_san":
        channel = _channel(address, pki, server_name="wrong.example")
    else:
        channel = _channel(
            address, pki, certificate=pki.other_client_cert, key=pki.other_client_key,
        )
    try:
        stub = pb_grpc.EvaluatorStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(stub)
        assert excinfo.value.code() in {
            grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAUTHENTICATED,
            grpc.StatusCode.PERMISSION_DENIED,
        }
    finally:
        channel.close()


def test_plaintext_client_cannot_reach_the_tls_frontend(tls_server):
    _engine, server, _pki = tls_server
    channel = grpc.insecure_channel(f"127.0.0.1:{server.port}")
    try:
        stub = pb_grpc.EvaluatorStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(stub)
        assert excinfo.value.code() in {grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.UNAUTHENTICATED}
    finally:
        channel.close()


def test_tls_still_requires_a_per_rpc_session_grant(tls_server):
    _engine, server, pki = tls_server
    channel = _channel(f"127.0.0.1:{server.port}", pki)
    try:
        stub = pb_grpc.EvaluatorStub(channel)
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(stub, metadata=())
        assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
        # A syntactically valid but unknown grant is rejected by the engine.
        with pytest.raises(grpc.RpcError) as excinfo:
            _capabilities(stub, metadata=(("authorization", "Bearer not-a-grant"),))
        assert excinfo.value.code() == grpc.StatusCode.UNAUTHENTICATED
    finally:
        channel.close()


def test_secret_file_permissions_are_enforced(tmp_path, pki):
    open_key = tmp_path / "open.key"
    open_key.write_bytes(pki.server_key.read_bytes())
    open_key.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        load_server_credentials(
            certificate_file=pki.server_cert, private_key_file=open_key,
            client_ca_file=pki.ca_cert,
        )
    with pytest.raises(ValueError, match="client certificate"):
        GrpcServerConfig(
            certificate_file=pki.server_cert, private_key_file=pki.server_key,
            client_ca_file=None, require_client_identity=True, allow_insecure=False,
        ).credentials()


def test_server_certificate_and_key_are_required_for_production(tmp_path):
    with pytest.raises(ValueError, match="certificate"):
        GrpcServerConfig(allow_insecure=False).credentials()
