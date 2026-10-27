"""Tests for the explicit gRPC remote composition and self-check bridge."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from taic.remote_runtime import RemoteEvaluationConfig
from tcenv.remote.grpc_client import GrpcEvaluatorFactory
from tests.support.remote_pki import make_pki

TASK_FINGERPRINT = "a" * 64
TASK_PACK_DIGEST = "b" * 64
TOKEN = "test-session-token"


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
        task="relu", dtype="fp16", size="small",
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
