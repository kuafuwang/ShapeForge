"""Tests for the explicit gRPC remote composition and self-check bridge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from taic.remote_runtime import RemoteEvaluationConfig
from tcenv.remote.grpc_client import GrpcEvaluatorFactory
from tests.support.remote_pki import make_pki
from tests.support.v1_tasks import V1_TASK

TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
TOKEN = "test-session-token"
SIGNING_KEY = b"t" * 32
SIGNING_KEY_ID = "sim-external-site/key-1"
FORMAL_TOKEN = "grpc-e2e-formal-grant"
SELFCHECK_TOKEN = "grpc-e2e-selfcheck-grant"
TARGET = "sim-external-test"


def _grpc_document(tmp_path: Path, pki, **overrides) -> Path:
    token_file = tmp_path / "grant.token"
    token_file.write_text(TOKEN)
    selfcheck_file = tmp_path / "selfcheck-grant.token"
    selfcheck_file.write_text("selfcheck-token")
    keyring = pki.keyring_file(key_id="sim-site/key-1", key=b"k" * 32)
    document = {
        "transport": "grpc",
        "endpoint": "127.0.0.1:1",
        "tls_server_name": "localhost",
        "token_file": str(token_file),
        "selfcheck_token_file": str(selfcheck_file),
        "ca_file": str(pki.ca_cert),
        "client_certificate_file": str(pki.client_cert),
        "client_private_key_file": str(pki.client_key),
        "signature_keyring_file": str(keyring),
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_device_id": "sim-device",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
        "publishable_site_ids": [],
        "socket_root": tempfile.mkdtemp(prefix="taic-remote-", dir="/tmp"),
    }
    document.update(overrides)
    path = tmp_path / "remote-grpc.json"
    path.write_text(json.dumps(document))
    return path


def test_grpc_remote_config_builds_shared_channel_factories(tmp_path):
    from tcenv.remote.grpc_client import GrpcEvaluatorFactory
    from tests.support.remote_pki import make_pki

    pki = make_pki(tmp_path / "pki")
    config = RemoteEvaluationConfig.from_file(_grpc_document(tmp_path, pki))
    assert config.transport == "grpc"
    formal = config.make_evaluator("run-1")
    selfcheck = config.make_selfcheck_evaluator("run-1")
    try:
        assert isinstance(formal, GrpcEvaluatorFactory)
        assert isinstance(selfcheck, GrpcEvaluatorFactory)
        metadata = formal.execution_metadata()
        assert metadata["transport"] == "grpc"
        assert metadata["api_contract_id"] == "tcenv.remote.v1.evaluator-api.1"
        assert metadata["evidence_contract_id"] == "tcenv.remote.v1.evidence.1"
        assert metadata["result_signature_key_ids"] == ["sim-site/key-1"]
        # One host channel per endpoint/TLS identity: formal and selfcheck
        # factories share it, and closing both releases the registry entry.
        assert formal._channel is selfcheck._channel
        from tcenv.remote import grpc_client as grpc_client_module
        assert grpc_client_module._SHARED_CHANNELS
        # A renewed token file is re-read per call, never cached in the factory.
        (tmp_path / "grant.token").write_text("renewed-grant")
        assert formal.token_provider("run-1", "session-1") == "renewed-grant"
    finally:
        formal.close()
        selfcheck.close()
    assert not grpc_client_module._SHARED_CHANNELS


def test_grpc_remote_config_rejects_missing_tls_material_or_mixed_transports(tmp_path):
    from tests.support.remote_pki import make_pki

    pki = make_pki(tmp_path / "pki")
    base = json.loads(_grpc_document(tmp_path, pki).read_text())
    for field in ("ca_file", "signature_keyring_file"):
        document = dict(base)
        document.pop(field)
        path = tmp_path / f"missing-{field}.json"
        path.write_text(json.dumps(document))
        try:
            RemoteEvaluationConfig.from_file(path)
        except ValueError as exc:
            assert field in str(exc)
        else:
            raise AssertionError(f"gRPC config without {field} was accepted")
    mixed = dict(base)
    mixed["command"] = ["python3", "-c", "pass"]
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(mixed))
    try:
        RemoteEvaluationConfig.from_file(path)
    except ValueError as exc:
        assert "command" in str(exc)
    else:
        raise AssertionError("a gRPC config with a command was accepted")
    half_cert = dict(base)
    half_cert.pop("client_private_key_file")
    path = tmp_path / "half-cert.json"
    path.write_text(json.dumps(half_cert))
    try:
        RemoteEvaluationConfig.from_file(path)
    except ValueError as exc:
        assert "together" in str(exc)
    else:
        raise AssertionError("half a client identity was accepted")


def test_remote_config_rejects_unknown_fields(tmp_path):
    """@brief 配置中的未知字段必须在创建连接前拒绝。"""
    pki = make_pki(tmp_path / "pki")
    base = json.loads(_grpc_document(tmp_path, pki).read_text())
    base["shell"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="unknown remote evaluation fields"):
        RemoteEvaluationConfig.from_file(path)


def test_remote_config_requires_an_explicit_grpc_transport(tmp_path):
    """@brief 省略 transport 或写 legacy-ssh 必须拒绝，不得默认或静默换协议。"""
    pki = make_pki(tmp_path / "pki")
    base = json.loads(_grpc_document(tmp_path, pki).read_text())
    cases = [
        ("missing", lambda document: document.pop("transport"), "must declare transport='grpc'"),
        ("legacy", lambda document: document.update(transport="legacy-ssh"), "no longer supported"),
        ("other", lambda document: document.update(transport="local"), "must be 'grpc'"),
        ("command", lambda document: document.update(command=["ssh", "evaluator"]), "does not accept a command"),
        ("pool-knob", lambda document: document.update(idle_timeout_s=1.0), "unknown remote evaluation fields"),
    ]
    for name, mutate, match in cases:
        document = dict(base)
        mutate(document)
        path = tmp_path / f"rejected-{name}.json"
        path.write_text(json.dumps(document))
        with pytest.raises(ValueError, match=match):
            RemoteEvaluationConfig.from_file(path)


def test_remote_evaluation_config_cannot_be_constructed_with_legacy_ssh(tmp_path):
    """@brief 直接构造 dataclass 也不能绕开 gRPC-only 约束。"""
    with pytest.raises(ValueError, match="only transport='grpc'"):
        RemoteEvaluationConfig(
            token_file=tmp_path / "grant.token",
            target_id="sim-npu-test",
            transport="legacy-ssh",
        )


def test_legacy_remote_data_plane_modules_are_gone():
    """@brief 旧 SSH 客户端和服务端入口不再可导入。"""
    import importlib.util

    assert importlib.util.find_spec("tcenv.remote.client") is None
    assert importlib.util.find_spec("tcenv.remote.service") is None


def test_grpc_selfcheck_bridge_uses_the_grpc_factory_and_releases_it(tmp_path):
    """@brief 显式 gRPC 配置能启动 per-session bridge，并在停止时释放共享 channel。"""
    from tcenv.remote import grpc_client as grpc_client_module

    pki = make_pki(tmp_path / "pki")
    config = RemoteEvaluationConfig.from_file(_grpc_document(tmp_path, pki))
    work = tmp_path / "work"
    work.mkdir()
    (work / "report.json").write_text(json.dumps({"speedup": 1.0, "benchmark": {"candidate": {"cv": 0.01}}}))
    factory = config.make_bridge_factory(
        task=V1_TASK, dtype="fp16", size="small",
        task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
    )
    server, token_file = factory("session-1", work, None)
    try:
        assert isinstance(server.evaluator, GrpcEvaluatorFactory)
        assert server.config.close_evaluator_on_stop is True
        assert Path(token_file).is_file()
        assert grpc_client_module._SHARED_CHANNELS
    finally:
        server.stop()
    assert not Path(token_file).exists()
    assert not grpc_client_module._SHARED_CHANNELS


@pytest.fixture
def mtls_evaluator(tmp_path):
    """A real mTLS gRPC frontend over the durable engine and fake private worker."""
    import shutil

    from tcenv.remote.grpc_server import GrpcServerConfig, build_server
    from tests.support.job_engine import engine as make_engine, grant
    from tests.support.remote_pki import PkiUnavailable

    if shutil.which("openssl") is None:
        pytest.skip("openssl is required for disposable test certificates")
    try:
        pki = make_pki(tmp_path / "pki")
    except PkiUnavailable as exc:
        pytest.skip(str(exc))
    engine = make_engine(
        tmp_path / "state",
        grants=[
            grant(FORMAL_TOKEN, kinds=frozenset({"formal_eval"}), subject="formal"),
            grant(SELFCHECK_TOKEN, kinds=frozenset({"selfcheck"}), subject="selfcheck"),
        ],
        signing_key=SIGNING_KEY,
        signing_key_id=SIGNING_KEY_ID,
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
    try:
        yield engine, server, pki
    finally:
        server.stop(grace_s=0.2)
        engine.close()


def _mtls_document(tmp_path: Path, pki, server, *, selfcheck_token: str | None = SELFCHECK_TOKEN) -> Path:
    from tests.support.job_engine import identity

    formal_file = tmp_path / "formal.token"
    formal_file.write_text(FORMAL_TOKEN)
    keyring = pki.keyring_file(key_id=SIGNING_KEY_ID, key=SIGNING_KEY)
    document = {
        "transport": "grpc",
        "endpoint": f"127.0.0.1:{server.port}",
        "tls_server_name": "localhost",
        "token_file": str(formal_file),
        "ca_file": str(pki.ca_cert),
        "client_certificate_file": str(pki.client_cert),
        "client_private_key_file": str(pki.client_key),
        "signature_keyring_file": str(keyring),
        "target_id": TARGET,
        "expected_site_id": identity().site_id,
        "expected_device_id": identity().device_id,
        "expected_worker_build": identity().worker_build,
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
        "publishable_site_ids": [],
        "socket_root": tempfile.mkdtemp(prefix="taic-remote-", dir="/tmp"),
        "selfcheck_reps": 2,
        "selfcheck_max_timed": 2,
        "selfcheck_max_gpu_s": 10.0,
    }
    if selfcheck_token is not None:
        selfcheck_file = tmp_path / "selfcheck.token"
        selfcheck_file.write_text(selfcheck_token)
        document["selfcheck_token_file"] = str(selfcheck_file)
    path = tmp_path / "remote-grpc-mtls.json"
    path.write_text(json.dumps(document))
    return path


def _mtls_request(request_id: str = "grpc-e2e-formal"):
    from tcenv.evaluation import EvaluationRequest

    return EvaluationRequest(
        kind="formal_eval",
        request_id=request_id,
        run_id="run-1",
        session_id="session-1",
        task=V1_TASK,
        dtype="fp16",
        size="small",
        target_id=TARGET,
        task_fingerprint=TASK_FINGERPRINT,
        source="OK",
        mode="quick",
        seed=3,
        reps=2,
        queue_timeout_s=10.0,
        candidate_timeout_s=10.0,
        cache_policy="disabled",
    )


def test_grpc_selfcheck_runs_through_the_unix_bridge_with_its_own_grant(tmp_path, mtls_evaluator):
    """@brief 真实 mTLS RPC 经 Unix bridge 执行 selfcheck，并保留 token/seed 宿主边界。"""
    from taic.selfcheck_bridge import SelfCheckClient
    from tcenv.evaluation import EvaluationUnavailable

    _engine, server, pki = mtls_evaluator
    config = RemoteEvaluationConfig.from_file(_mtls_document(tmp_path, pki, server))
    formal = config.make_evaluator("run-1")
    selfcheck_probe = config.make_selfcheck_evaluator("run-1")
    try:
        result = formal.formal(_mtls_request())
        assert result.report["build"]["build_kind"] == "simulator"
        with pytest.raises(EvaluationUnavailable) as caught:
            selfcheck_probe.formal(_mtls_request("grpc-e2e-probe-formal"))
        message = str(caught.value).lower()
        assert "selfcheck" in message or "formal" in message or "kind" in message

        work = tmp_path / "work"
        work.mkdir()
        seed_json = json.dumps({"speedup": 1.0, "benchmark": {"candidate": {"cv": 0.01}}})
        (work / "report.json").write_text(seed_json)
        factory = config.make_bridge_factory(
            task=V1_TASK, dtype="fp16", size="small",
            task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
        )
        bridge, token_file = factory("session-1", work, formal)
        try:
            assert not bridge._usage_path.is_relative_to(work)
            assert bridge.config.seed_report_path is not None
            assert not bridge.config.seed_report_path.is_relative_to(work)
            (work / "report.json").write_text(json.dumps({"speedup": 1000.0}))
            assert bridge.config.seed_report_path.read_text() == seed_json
            checked = SelfCheckClient(bridge.socket_path, token_file=token_file).check(
                "kernel void demo() {}", "quick",
            )
            assert checked.report["build"]["build_kind"] == "simulator"
            assert checked.report["status"] == "ok"
            assert checked.receipt["attempt_id"]
            # The credential is host-owned, outside the agent-writable
            # workspace, so a session prune cannot archive or delete it.
            assert not Path(token_file).is_relative_to(work)
            assert Path(token_file).parent == bridge.socket_path.parent
            from taic.kernel_agent import prune_work
            (work / "scratch.txt").write_text("x")
            pruned = prune_work(work, label="init", keep={"candidate.metal"})
            assert "token" not in pruned.get("archived", [])
            assert Path(token_file).is_file()
        finally:
            bridge.stop()
        assert not Path(token_file).exists()
    finally:
        selfcheck_probe.close()
        formal.close()


def test_grpc_selfcheck_requires_its_own_grant(tmp_path, mtls_evaluator):
    """@brief 缺少独立 selfcheck grant 时禁止复用 formal evaluator。"""
    _engine, server, pki = mtls_evaluator
    config = RemoteEvaluationConfig.from_file(_mtls_document(tmp_path, pki, server, selfcheck_token=None))
    with pytest.raises(ValueError, match="selfcheck_token_file"):
        config.make_bridge_factory(
            task=V1_TASK, dtype="fp16", size="small",
            task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
        )
    with pytest.raises(ValueError, match="selfcheck_token_file"):
        config.make_selfcheck_evaluator("run-1")


def test_grpc_selfcheck_does_not_fall_back_to_the_formal_grant(tmp_path, mtls_evaluator):
    """@brief 把 formal grant 当 selfcheck grant 时由真实 RPC 拒绝，而不是静默复用。"""
    from taic.selfcheck_bridge import SelfCheckBridgeError, SelfCheckClient

    _engine, server, pki = mtls_evaluator
    config = RemoteEvaluationConfig.from_file(
        _mtls_document(tmp_path, pki, server, selfcheck_token=FORMAL_TOKEN),
    )
    work = tmp_path / "work"
    work.mkdir()
    bridge, token_file = config.make_bridge_factory(
        task=V1_TASK, dtype="fp16", size="small",
        task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
    )("session-1", work, None)
    try:
        with pytest.raises(SelfCheckBridgeError) as caught:
            SelfCheckClient(bridge.socket_path, token_file=token_file).check(
                "kernel void demo() {}", "quick",
            )
        message = str(caught.value).lower()
        assert "selfcheck" in message or "kind" in message or "grant" in message
    finally:
        bridge.stop()
