"""Tests for the explicit remote composition, using the real service and fake device."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

from taic.remote_runtime import RemoteEvaluationConfig
from taic.selfcheck_bridge import SelfCheckClient
from tcenv.evaluation import EvaluationRequest
from tests.support.npu_sim.service_entry import TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN

ROOT = Path(__file__).resolve().parents[2]


def _config(tmp_path: Path) -> RemoteEvaluationConfig:
    token_file = tmp_path / "grant.token"
    token_file.write_text(TOKEN)
    selfcheck_file = tmp_path / "selfcheck-grant.token"
    selfcheck_file.write_text("selfcheck-token")
    command = [
        sys.executable, "-m", "tests.support.npu_sim.service_entry",
        "--state-dir", str(tmp_path / "state" / "{evaluation_kind}" / "{session_id}"),
        "--run-id", "{run_id}", "--session-id", "{session_id}", "--token", TOKEN,
        "--selfcheck-token", "selfcheck-token",
    ]
    path = tmp_path / "remote.json"
    path.write_text(json.dumps({
        "command": command,
        "token_file": str(token_file),
        "selfcheck_token_file": str(selfcheck_file),
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
        "publishable_site_ids": [],
        "socket_root": tempfile.mkdtemp(prefix="taic-remote-", dir="/tmp"),
        "timeout_s": 2,
    }))
    return RemoteEvaluationConfig.from_file(path)


def _request() -> EvaluationRequest:
    source = "kernel void demo() {}"
    return EvaluationRequest(
        task="relu", dtype="fp16", size="small", source=source,
        candidate_sha256=hashlib.sha256(source.encode()).hexdigest(),
        kind="formal_eval", mode="quick", seed=3, reps=2,
        run_id="run-1", session_id="session-1", target_id="sim-npu-test",
        task_fingerprint=TASK_FINGERPRINT, cache_policy="disabled",
        candidate_timeout_s=2, queue_timeout_s=2,
    )


def test_explicit_remote_config_builds_pool_and_per_session_bridge(tmp_path):
    """@brief 配置文件创建真实 pool 和 host bridge，自测保留 simulated 标记并清理 token。"""
    config = _config(tmp_path)
    evaluator = config.make_evaluator("run-1")
    try:
        result = evaluator.formal(_request())
        assert result.report["simulated"] is True
        factory = config.make_bridge_factory(task="relu", dtype="fp16", size="small", task_fingerprint=TASK_FINGERPRINT, run_id="run-1")
        work = tmp_path / "work"
        work.mkdir()
        seed_json = json.dumps({"speedup": 1.0, "benchmark": {"candidate": {"cv": 0.01}}})
        (work / "report.json").write_text(seed_json)
        server, token_file = factory("session-1", work, evaluator)
        assert not server._usage_path.is_relative_to(work)
        assert server.config.seed_report_path is not None
        assert not server.config.seed_report_path.is_relative_to(work)
        (work / "report.json").write_text(json.dumps({"speedup": 1000.0}))
        assert server.config.seed_report_path.read_text() == seed_json
        try:
            checked = SelfCheckClient(server.socket_path, token_file=token_file).check("kernel void demo() {}", "quick")
            assert checked.report["simulated"] is True
            # The credential is host-owned, outside the agent-writable
            # workspace, so a session prune cannot archive or delete it.
            assert not Path(token_file).is_relative_to(work)
            assert Path(token_file).parent == server.socket_path.parent
            from taic.kernel_agent import prune_work
            (work / "scratch.txt").write_text("x")
            pruned = prune_work(work, label="init", keep={"candidate.metal"})
            assert "token" not in pruned.get("archived", [])
            assert Path(token_file).is_file()
        finally:
            server.stop()
        assert not Path(token_file).exists()
    finally:
        evaluator.close()


def test_remote_config_can_use_separate_formal_and_selfcheck_grants(tmp_path):
    """@brief 正式与自测可以使用不同远端 grant；自测 grant 不能冒充 formal。"""
    formal_token = tmp_path / "formal.token"
    selfcheck_token = tmp_path / "selfcheck.token"
    formal_token.write_text("formal-token")
    selfcheck_token.write_text("selfcheck-token")
    command = [
        sys.executable, "-m", "tests.support.npu_sim.service_entry",
        "--state-dir", str(tmp_path / "state" / "{evaluation_kind}" / "{session_id}"),
        "--run-id", "{run_id}", "--session-id", "{session_id}",
        "--formal-token", "formal-token", "--selfcheck-token", "selfcheck-token",
    ]
    path = tmp_path / "remote.json"
    path.write_text(json.dumps({
        "command": command,
        "token_file": str(formal_token),
        "selfcheck_token_file": str(selfcheck_token),
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
        "publishable_site_ids": [],
        "socket_root": tempfile.mkdtemp(prefix="taic-remote-", dir="/tmp"),
        "timeout_s": 2,
    }))
    config = RemoteEvaluationConfig.from_file(path)
    evaluator = config.make_evaluator("run-1")
    selfcheck_pool = config.make_selfcheck_evaluator("run-1")
    try:
        assert evaluator.formal(_request()).report["simulated"] is True
        from tcenv.remote.client import RemoteError
        try:
            selfcheck_pool.formal(_request())
        except RemoteError as exc:
            assert "selfcheck" in str(exc) or "formal" in str(exc)
        else:
            raise AssertionError("a selfcheck-only grant submitted formal evaluation")
        # The bridge owns a new frontend for the same per-kind state path;
        # release this explicit probe before opening that frontend.
        selfcheck_pool.close()
        work = tmp_path / "work"
        work.mkdir()
        server, token_file = config.make_bridge_factory(
            task="relu", dtype="fp16", size="small",
            task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
        )("session-1", work, evaluator)
        try:
            checked = SelfCheckClient(server.socket_path, token_file=token_file).check("kernel void demo() {}", "quick")
            assert checked.report["simulated"] is True
        finally:
            server.stop()
    finally:
        selfcheck_pool.close()
        evaluator.close()


def test_remote_selfcheck_never_falls_back_to_the_formal_grant(tmp_path):
    """@brief 缺少独立 selfcheck grant 时必须拒绝，而不是复用 formal evaluator。"""
    from dataclasses import replace
    import pytest

    config = replace(_config(tmp_path), selfcheck_token_file=None)
    with pytest.raises(ValueError, match="selfcheck_token_file"):
        config.make_bridge_factory(
            task="relu", dtype="fp16", size="small",
            task_fingerprint=TASK_FINGERPRINT, run_id="run-1",
        )
    with pytest.raises(ValueError, match="selfcheck_token_file"):
        config.make_selfcheck_evaluator("run-1")


def test_remote_config_rejects_unknown_fields(tmp_path):
    """@brief 配置中的未知字段必须在创建连接前拒绝。"""
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"command": ["echo"], "token_file": "token", "target_id": "sim-npu-test", "shell": True}))
    try:
        RemoteEvaluationConfig.from_file(path)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown remote config field was accepted")


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
        # A renewed token file is re-read per call, never cached in the factory.
        (tmp_path / "grant.token").write_text("renewed-grant")
        assert formal.token_provider("run-1", "session-1") == "renewed-grant"
    finally:
        formal.close()
        selfcheck.close()


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
    mixed["command"] = [sys.executable, "-c", "pass"]
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
