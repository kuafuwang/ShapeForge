"""CPU-only tests for the remote evaluator client against the real service core."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path

import pytest

from tcenv.evaluation import EvaluationRequest
from tcenv.remote.client import CommandTransport, RemoteError, RemoteEvaluator, RemoteEvaluatorPool
from tests.support.npu_sim.service_entry import TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN

ROOT = Path(__file__).resolve().parents[2]


def _request(*, request_id: str = "client-req-1", kind: str = "formal_eval", mode: str = "quick", source: str = "kernel void demo() {}", session_id: str = "session-1", candidate_timeout_s: float = 2.0, progress_path: Path | None = None) -> EvaluationRequest:
    return EvaluationRequest(
        request_id=request_id,
        run_id="run-1",
        session_id=session_id,
        task="relu",
        dtype="fp16",
        size="small",
        target_id="sim-npu-test",
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        kind=kind,
        mode=mode,
        seed=3,
        reps=2,
        candidate_timeout_s=candidate_timeout_s,
        queue_timeout_s=2.0,
        cache_policy="disabled",
        progress_path=progress_path,
    )


def _evaluator(tmp_path: Path, **overrides) -> RemoteEvaluator:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    command = [sys.executable, "-m", "tests.support.npu_sim.service_entry", "--state-dir", str(tmp_path / "state")]
    signing_key = overrides.pop("signing_key", None)
    if signing_key is not None:
        key_path = tmp_path / "signing.key"
        key_path.write_bytes(signing_key)
        command.extend(["--signature-key-file", str(key_path)])
    options = {
        "token": TOKEN,
        "run_id": "run-1",
        "session_id": "session-1",
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": TASK_PACK_DIGEST,
        "task_fingerprint": TASK_FINGERPRINT,
    }
    if signing_key is not None:
        options["signature_key"] = signing_key
    options.update(overrides)
    return RemoteEvaluator(CommandTransport(command, env=env), **options)


def test_remote_evaluator_pins_epoch_and_returns_bound_result(tmp_path):
    """@brief hello 后固定 epoch，正式请求返回绑定候选摘要的结果。"""
    with _evaluator(tmp_path) as evaluator:
        provenance = evaluator.connect()
        request = _request()
        result = evaluator.formal(request)
    assert result.provenance.epoch == provenance.epoch
    assert result.provenance.request_digest == request.request_digest
    assert result.evidence["candidate_sha256"] == request.candidate_sha256
    assert result.report["status"] == "ok"


def test_remote_evaluator_selfcheck_uses_selfcheck_kind(tmp_path):
    """@brief selfcheck 走 selfcheck kind，不会被当成正式评测。"""
    with _evaluator(tmp_path) as evaluator:
        result = evaluator.selfcheck(_request(request_id="self-1", kind="selfcheck", mode="quick"))
    assert result.report["mode"] == "quick"


def test_remote_evaluator_rejects_wrong_identity_before_request(tmp_path):
    """@brief 期望的 worker build 不匹配时，hello 必须失败。"""
    evaluator = _evaluator(tmp_path, expected_worker_build="wrong-build")
    try:
        with pytest.raises(RemoteError, match="worker build"):
            evaluator.connect()
    finally:
        evaluator.close()


def test_remote_evaluator_rejects_kind_mismatch(tmp_path):
    """@brief formal() 不能发送 selfcheck 请求。"""
    with _evaluator(tmp_path) as evaluator:
        with pytest.raises(ValueError, match="formal"):
            evaluator.formal(_request(kind="selfcheck"))


def test_remote_evaluator_surfaces_service_error(tmp_path):
    """@brief validate 没有映射时，服务端错误帧必须变成带 code 的 RemoteError。"""
    with _evaluator(tmp_path) as evaluator:
        with pytest.raises(RemoteError) as exc_info:
            evaluator.formal(_request(mode="validate"))
    assert exc_info.value.code == "protocol_error"
    assert exc_info.value.retryable is False


def test_remote_evaluator_rejects_wrong_target_at_hello(tmp_path):
    """@brief 客户端 target 与服务端能力不一致时，在发请求前失败。"""
    evaluator = _evaluator(tmp_path, target_id="other-target")
    try:
        with pytest.raises(RemoteError, match="target"):
            evaluator.connect()
    finally:
        evaluator.close()


def test_pool_binds_each_session_to_its_own_connection(tmp_path):
    """@brief pool 允许不同 hmz session 使用各自 grant，不再要求固定 session_id。"""
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def factory(key):
        run_id, session_id, _target = key
        command = [sys.executable, "-m", "tests.support.npu_sim.service_entry", "--state-dir", str(tmp_path / session_id), "--run-id", run_id, "--session-id", session_id, "--token", TOKEN]
        return CommandTransport(command, env=env)

    pool = RemoteEvaluatorPool(factory, token=TOKEN, run_id="run-1", target_id="sim-npu-test", expected_site_id="sim-site", expected_worker_build="sim-build-1", expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT, timeout_s=2)
    try:
        first = pool.formal(_request(request_id="pool-1", session_id="session-1"))
        second = pool.formal(_request(request_id="pool-2", session_id="session-2"))
        assert first.report["status"] == "ok" and second.report["status"] == "ok"
        assert len(pool._clients) == 2
    finally:
        pool.close()


def test_progress_frames_update_the_host_progress_file(tmp_path):
    """@brief 服务端 progress 只能落到 host 指定路径，不改变请求结果。"""
    import json

    progress = tmp_path / "eval_progress.json"
    with _evaluator(tmp_path) as evaluator:
        result = evaluator.formal(_request(request_id="progress-1", source="REMOTE_PROGRESS", progress_path=progress))
    assert result.report["status"] == "ok"
    document = json.loads(progress.read_text())
    assert document["stage"] == "done"
    assert document["finished"] is True
    assert "TCENV_REMOTE_PROGRESS_PATH" not in document


def _pool_for_lifecycle(tmp_path: Path, **options) -> RemoteEvaluatorPool:
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}

    def factory(key):
        run_id, session_id, _target = key
        command = [
            sys.executable, "-m", "tests.support.npu_sim.service_entry",
            "--state-dir", str(tmp_path / session_id),
            "--run-id", run_id, "--session-id", session_id, "--token", TOKEN,
        ]
        return CommandTransport(command, env=env)

    return RemoteEvaluatorPool(
        factory, token=TOKEN, run_id="run-1", target_id="sim-npu-test",
        expected_site_id="sim-site", expected_worker_build="sim-build-1",
        expected_task_pack_digest=TASK_PACK_DIGEST, task_fingerprint=TASK_FINGERPRINT,
        timeout_s=2, **options,
    )


def test_pool_retires_only_idle_connections(tmp_path):
    """@brief idle 连接会回收重建，活跃请求在 idle/LRU 边界也不会被关闭。"""
    pool = _pool_for_lifecycle(tmp_path, idle_timeout_s=0.02, max_clients=1)
    try:
        assert pool.formal(_request(request_id="idle-1", session_id="session-1")).report["status"] == "ok"
        first = next(iter(pool._clients.values()))
        first._active_request_id = "active"
        pool._request_clients["active"] = first
        time.sleep(0.04)
        pool.reap_idle()
        assert first._closed is False
        first._active_request_id = ""
        pool._request_clients.clear()
        pool.reap_idle()
        assert first._closed is True
        assert pool.formal(_request(request_id="idle-2", session_id="session-2")).report["status"] == "ok"
        assert len(pool._clients) == 1
        assert first not in pool._clients.values()
    finally:
        pool.close()


def test_pool_lru_limit_closes_the_oldest_idle_connection(tmp_path):
    """@brief 连接数超过上限时关闭最老的空闲连接，不无限积累 SSH 子进程。"""
    pool = _pool_for_lifecycle(tmp_path, idle_timeout_s=None, max_clients=1)
    try:
        assert pool.formal(_request(request_id="lru-1", session_id="session-1")).report["status"] == "ok"
        first = next(iter(pool._clients.values()))
        assert pool.formal(_request(request_id="lru-2", session_id="session-2")).report["status"] == "ok"
        assert first._closed is True
        assert len(pool._clients) == 1
    finally:
        pool.close()


def test_transport_timeout_closes_the_request(tmp_path):
    """@brief 服务端没有响应时，客户端在 transport timeout 后失败而不是永久等待。"""
    evaluator = _evaluator(tmp_path, timeout_s=0.05)
    try:
        with pytest.raises(RemoteError) as exc_info:
            evaluator.formal(_request(source="REMOTE_SLEEP", candidate_timeout_s=2.0))
        assert exc_info.value.code == "transport_timeout"
    finally:
        evaluator.close()


def test_signature_is_verified_and_receipt_replay_is_scoped_by_digest(tmp_path):
    """@brief 配置密钥后签名必须验证；同 receipt 跨不同 digest 必须拒绝。"""
    key = b"test-signing-key"
    with _evaluator(tmp_path, signing_key=key) as evaluator:
        result = evaluator.formal(_request(request_id="signed-1", source="REPLAY_RECEIPT"))
        assert result.signature
        assert result.receipt["receipt_id"]
        with pytest.raises(RemoteError) as exc_info:
            evaluator.formal(_request(request_id="signed-2", source="REPLAY_RECEIPT", mode="full"))
        assert exc_info.value.code == "receipt_replay"


def test_a_signed_result_is_rejected_without_a_verification_key(tmp_path):
    """@brief 没有配置验证密钥时，带签名的结果不能被当作已验证证据。"""
    key = b"test-signing-key"
    with _evaluator(tmp_path, signing_key=key) as evaluator:
        evaluator.signature_key = None
        with pytest.raises(RemoteError) as exc_info:
            evaluator.formal(_request(request_id="unsigned-host"))
        assert exc_info.value.code == "signature_unverified"


def test_a_wrong_signature_key_is_rejected(tmp_path):
    """@brief 客户端密钥与服务端不一致时，签名验证必须失败。"""
    with _evaluator(tmp_path, signing_key=b"server-key") as evaluator:
        evaluator.signature_key = b"wrong-key"
        with pytest.raises(RemoteError) as exc_info:
            evaluator.formal(_request(request_id="wrong-sign"))
        assert exc_info.value.code == "signature_mismatch"
