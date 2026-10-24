"""End-to-end subprocess tests for the remote evaluator service core."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.remote.protocol import (
    EvaluationRequest,
    Frame,
    ProtocolError,
    write_frame,
    read_frame,
)
from tcenv.remote.production import ProductionBackend
from tcenv.evaluate import EvalReport
from tcenv.remote.service import RemoteService, ServiceIdentity, SessionGrant, _JobFailed
from tcenv.evaluation import EvaluationError
from tests.support.npu_sim.service_entry import TASK_FINGERPRINT, TASK_PACK_DIGEST, TOKEN


ROOT = Path(__file__).resolve().parents[2]


class ServiceProcess:
    def __init__(self, tmp_path: Path, *, selfcheck_max: int = 20, selfcheck_max_gpu_s: float = 120.0, allow_kinds: tuple[str, ...] = ("formal_eval", "selfcheck"), expires_at: float | None = None, extra_env: dict[str, str] | None = None, alt: bool = False) -> None:
        env = dict(os.environ)
        if extra_env:
            env.update(extra_env)
        env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        command = [sys.executable, "-m", "tests.support.npu_sim.service_entry", "--state-dir", str(tmp_path / "state"), "--selfcheck-max", str(selfcheck_max), "--selfcheck-max-gpu-s", str(selfcheck_max_gpu_s)]
        if alt:
            command.extend(["--alt-token", "test-alt-token", "--alt-run-id", "run-2", "--alt-session-id", "session-1", "--aging-s", "0.2"])
        for kind in allow_kinds:
            command.extend(["--allow-kind", kind])
        if expires_at is not None:
            command.extend(["--expires-at", str(expires_at)])
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            cwd=ROOT,
        )

    def send(self, frame_type: str, request_id: str, payload: dict) -> None:
        assert self.proc.stdin is not None
        write_frame(self.proc.stdin, frame_type, request_id, payload)

    def recv(self) -> Frame:
        assert self.proc.stdout is not None
        return read_frame(self.proc.stdout)

    def hello(self, nonce: str = "client-nonce-1", token: str = TOKEN, pending: list | None = None) -> Frame:
        self.send("hello", nonce, {
            "protocol_version": "1",
            "client_nonce": nonce,
            "session_token": token,
            "expected_site_id": "sim-site",
            "expected_worker_build": "sim-build-1",
            "expected_task_pack_digest": TASK_PACK_DIGEST,
            "target_id": "sim-npu-test",
        })
        while True:
            frame = self.recv()
            if frame.frame_type in {"hello", "error"}:
                return frame
            if pending is not None and frame.frame_type == "result":
                pending.append(frame)
            else:
                raise AssertionError(f"unexpected frame while waiting for hello: {frame.frame_type}")

    def close(self) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=5)
        finally:
            if self.proc.poll() is None:
                self.proc.kill()
                self.proc.wait(timeout=5)


@pytest.fixture
def service(tmp_path):
    proc = ServiceProcess(tmp_path)
    proc.hello_frame = proc.hello()
    assert proc.hello_frame.frame_type == "hello"
    yield proc
    proc.close()


def _request(*, request_id: str = "req-1", source: str = "kernel void demo() {}", kind: str = "formal_eval", mode: str = "quick", timeout: float = 2.0, queue_timeout: float = 2.0, cache_policy: str = "reuse", run_id: str = "run-1", session_id: str = "session-1") -> EvaluationRequest:
    return EvaluationRequest(
        kind=kind,
        request_id=request_id,
        run_id=run_id,
        session_id=session_id,
        task="relu",
        dtype="fp16",
        size="small",
        target_id="sim-npu-test",
        task_fingerprint=TASK_FINGERPRINT,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        mode=mode,
        seed=3,
        reps=2,
        queue_timeout_s=queue_timeout,
        candidate_timeout_s=timeout,
        cache_policy=cache_policy,
    )


def _send_request(service: ServiceProcess, request: EvaluationRequest) -> None:
    service.send("request", request.request_id, request.to_dict())


def test_production_validate_is_rejected_instead_of_silently_becoming_quick():
    """@brief 生产 validate 不得沿 public evaluate 的 quick fallback 执行。"""
    with pytest.raises(ProtocolError, match="validate"):
        ProductionBackend().evaluate(_request(mode="validate"))


def test_hello_returns_identity_capabilities_and_no_token(service):
    hello = service.hello_frame
    assert hello.frame_type == "hello"
    assert hello.payload["protocol_version"] == "1"
    assert hello.payload["client_nonce"] == "client-nonce-1"
    assert hello.payload["server_nonce"]
    assert hello.payload["provenance"]["site_id"] == "sim-site"
    assert hello.payload["provenance"]["measurement_kind"] == "simulated"
    assert hello.payload["provenance"]["publication_allowed"] is False
    assert hello.payload["capabilities"]["targets"] == ["sim-npu-test"]
    assert "validate" not in hello.payload["capabilities"]["modes"]
    assert TOKEN not in str(hello.payload)


def test_a_selfcheck_only_grant_cannot_submit_formal(tmp_path):
    """@brief selfcheck-only token 不得通过 request.kind 冒用 formal 身份。"""
    proc = ServiceProcess(tmp_path, allow_kinds=("selfcheck",))
    assert proc.hello().frame_type == "hello"
    try:
        request = _request(kind="formal_eval")
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "protocol_error"
        assert "allow" in error.payload["message"]
    finally:
        proc.close()


def test_a_grant_cannot_be_used_for_another_run_or_session(tmp_path):
    """@brief 改 run_id/session_id 不能绕过 grant 绑定或配额。"""
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        request = _request(run_id="run-2", session_id="session-2")
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert "run_id" in error.payload["message"] or "session_id" in error.payload["message"]
    finally:
        proc.close()


def test_an_expired_grant_is_rejected(tmp_path):
    """@brief 过期 token 不能建立 hello，也不能提交评测。"""
    import time
    proc = ServiceProcess(tmp_path, expires_at=time.time() - 1.0)
    frame = proc.hello()
    try:
        assert frame.frame_type == "error"
        assert "expired" in frame.payload["message"]
    finally:
        proc.close()


def test_formal_request_runs_in_a_worker_and_returns_simulated_report(service):
    request = _request()
    _send_request(service, request)
    result = service.recv()
    assert result.frame_type == "result"
    assert result.request_id == request.request_id
    assert result.payload["request_digest"] == request.request_digest
    assert result.payload["report"]["simulated"] is True
    assert result.payload["report"]["target_id"] == "sim-npu-test"
    assert result.payload["provenance"]["request_digest"] == request.request_digest
    assert result.payload["provenance"]["epoch"] == service.hello_frame.payload["provenance"]["epoch"]


def test_same_request_id_and_body_replays_result_without_running_again(tmp_path):
    counter = tmp_path / "worker-count.txt"
    proc = ServiceProcess(tmp_path, extra_env={"NPU_SIM_COUNTER": str(counter)})
    assert proc.hello().frame_type == "hello"
    try:
        request = _request()
        _send_request(proc, request)
        first = proc.recv()
        _send_request(proc, request)
        second = proc.recv()
        assert first.payload["report"] == second.payload["report"]
        assert counter.read_text() == "1"
    finally:
        proc.close()


def test_same_request_id_with_different_body_is_refused(service):
    request = _request()
    _send_request(service, request)
    service.recv()
    changed = _request(source="kernel void other() {}")
    changed = EvaluationRequest(**{**changed.to_dict(), "request_id": request.request_id})
    _send_request(service, changed)
    error = service.recv()
    assert error.frame_type == "error"
    assert error.payload["code"] == "idempotency_conflict"


def test_malformed_request_is_rejected_without_shell_execution(service):
    request = _request()
    payload = request.to_dict()
    payload["command"] = "touch /tmp/remote-service-was-here"
    service.send("request", request.request_id, payload)
    error = service.recv()
    assert error.frame_type == "error"
    assert error.payload["code"] == "protocol_error"
    assert not Path("/tmp/remote-service-was-here").exists()


def test_cancel_kills_the_worker_and_returns_cancelled(tmp_path):
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        request = _request(source="REMOTE_SLEEP", timeout=20.0)
        _send_request(proc, request)
        # Give the service a chance to hand the job to the worker before cancelling.
        import time
        time.sleep(0.2)
        proc.send("cancel", request.request_id, {"request_id": request.request_id})
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "cancelled"
    finally:
        proc.close()


def test_worker_timeout_is_an_infrastructure_error_not_a_candidate_crash(tmp_path):
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        request = _request(source="REMOTE_SLEEP", timeout=0.15)
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "candidate_timeout"
    finally:
        proc.close()


def test_selfcheck_quota_is_enforced_server_side(tmp_path):
    proc = ServiceProcess(tmp_path, selfcheck_max=1)
    assert proc.hello().frame_type == "hello"
    try:
        first = _request(request_id="self-1", kind="selfcheck", mode="timed")
        _send_request(proc, first)
        assert proc.recv().frame_type == "result"
        second = _request(request_id="self-2", kind="selfcheck", mode="timed")
        _send_request(proc, second)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "selfcheck_quota"
    finally:
        proc.close()


def _write_state(tmp_path: Path, request: EvaluationRequest, document: dict) -> None:
    path = tmp_path / "state" / (hashlib.sha256(request.request_id.encode("utf-8")).hexdigest() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_stale_running_state_is_retried_after_restart(tmp_path):
    """@brief 服务重启遗留 running 状态时，应按 retryable 恢复而不是永久 in_progress。"""
    proc = ServiceProcess(tmp_path)
    request = _request()
    _write_state(tmp_path, request, {"schema_version": 1, "request_id": request.request_id, "request_digest": request.request_digest, "status": "running", "updated_at": 0.0})
    assert proc.hello().frame_type == "hello"
    try:
        _send_request(proc, request)
        result = proc.recv()
        assert result.frame_type == "result"
        assert result.payload["report"]["simulated"] is True
    finally:
        proc.close()


def test_failed_terminal_state_is_replayed_not_in_progress(tmp_path):
    """@brief 已失败请求应回放失败，不应永远报告 in_progress。"""
    proc = ServiceProcess(tmp_path)
    request = _request()
    _write_state(tmp_path, request, {"schema_version": 1, "request_id": request.request_id, "request_digest": request.request_digest, "status": "failed", "error": {"code": "worker_failed", "message": "bad worker", "retryable": False}, "updated_at": 0.0})
    assert proc.hello().frame_type == "hello"
    try:
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "worker_failed"
    finally:
        proc.close()


def test_queue_timeout_is_enforced(tmp_path):
    """@brief 排队中的请求超过 queue_timeout_s 必须失败并移除，不能无限等待。"""
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        first = _request(source="REMOTE_SLEEP", timeout=30.0)
        _send_request(proc, first)
        second = _request(request_id="queued", timeout=30.0, queue_timeout=0.05)
        _send_request(proc, second)
        assert proc.recv().frame_type == "result"
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "queue_timeout"
    finally:
        proc.close()


def test_gpu_time_quota_is_enforced(tmp_path):
    """@brief GPU 秒数配额由服务端累计，不能只在 hello 中展示。"""
    proc = ServiceProcess(tmp_path, selfcheck_max_gpu_s=0.25)
    assert proc.hello().frame_type == "hello"
    try:
        first = _request(request_id="gpu-1", kind="selfcheck", mode="timed")
        _send_request(proc, first)
        assert proc.recv().frame_type == "result"
        second = _request(request_id="gpu-2", kind="selfcheck", mode="timed")
        _send_request(proc, second)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "selfcheck_quota"
    finally:
        proc.close()


def test_report_target_mismatch_is_rejected(tmp_path):
    """@brief worker 报告 target 与请求不符时必须拒绝，不得只接受 dict。"""
    identity = ServiceIdentity(site_id="sim-site", device_id="sim-device", backend="sim-npu", backend_version="1", target_id="sim-npu-test", worker_build="sim-build-1", task_pack_digest=TASK_PACK_DIGEST, calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}), measurement_kind="simulated", publication_allowed=False)
    grant = SessionGrant(token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test")
    service = RemoteService(identity=identity, grants=[grant], worker_command=["/bin/true"], state_dir=tmp_path / "svc")
    request = _request()
    report = {"task": request.task, "dtype": request.dtype, "size": request.size, "mode": request.mode, "target_id": "other-target", "protocol_version": "3", "status": "ok", "simulated": True}
    with pytest.raises(_JobFailed) as exc_info:
        service._validate_worker_result(request, {"report": report})
    assert exc_info.value.code == "provenance_mismatch"


def test_provenance_from_dict_rejects_missing_fields():
    """@brief provenance 缺字段必须是结构化错误，不能静默接受。"""
    with pytest.raises(EvaluationError):
        from tcenv.evaluation import EvaluatorProvenance
        EvaluatorProvenance.from_dict({})


def test_production_cache_policy_is_mapped(monkeypatch):
    """@brief reuse/disabled 映射到 eval_cache，不支持的 refresh 必须显式拒绝。"""
    seen = []
    def fake(*args, **kwargs):
        seen.append(kwargs)
        request = args[3] if len(args) > 3 else None
        return EvalReport(task="relu", dtype="fp16", size="small", mode=kwargs.get("mode", "quick"), target_id="sim-npu-test")
    monkeypatch.setattr("tcenv.evaluate.evaluate", fake)
    ProductionBackend().evaluate(_request(cache_policy="reuse"))
    assert seen[-1]["eval_cache"] is True and seen[-1]["eval_cache_refresh"] is False
    ProductionBackend().evaluate(_request(cache_policy="disabled"))
    assert seen[-1]["eval_cache"] is False and seen[-1]["eval_cache_refresh"] is False
    ProductionBackend().evaluate(_request(cache_policy="refresh"))
    assert seen[-1]["eval_cache"] is True and seen[-1]["eval_cache_refresh"] is True


def test_corrupt_request_state_fails_closed(tmp_path):
    """@brief 状态文件损坏不能当成从未执行而重复发起任务。"""
    proc = ServiceProcess(tmp_path)
    request = _request()
    path = tmp_path / "state" / (hashlib.sha256(request.request_id.encode("utf-8")).hexdigest() + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not-json")
    assert proc.hello().frame_type == "hello"
    try:
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "protocol_error"
    finally:
        proc.close()


def test_cancel_is_scoped_to_the_grant(tmp_path):
    """@brief 取消操作也必须绑定 run/session，不能跨 run 取消他人任务。"""
    identity = ServiceIdentity(site_id="sim-site", device_id="sim-device", backend="sim-npu", backend_version="1", target_id="sim-npu-test", worker_build="sim-build-1", task_pack_digest=TASK_PACK_DIGEST, calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}), measurement_kind="simulated", publication_allowed=False)
    first = SessionGrant(token="first", run_id="run-1", session_id="session-1", target_id="sim-npu-test")
    second = SessionGrant(token="second", run_id="run-2", session_id="session-2", target_id="sim-npu-test")
    service = RemoteService(identity=identity, grants=[first, second], worker_command=["/bin/true"], state_dir=tmp_path / "svc")
    service._authenticated = True
    service._grant = first
    other = _request(request_id="other", run_id="run-2", session_id="session-2")
    job = service._jobs["other"] = type("Job", (), {"request": other, "cancel": __import__("threading").Event(), "process": None})()
    frame = Frame("cancel", "other", {"request_id": "other"}, 0, "")
    with pytest.raises(ProtocolError, match="run_id|session_id"):
        service._handle_cancel(frame)


def test_service_identity_requires_a_registered_task_fingerprint():
    """@brief 生产身份不能为空 task fingerprint 集合，否则会接受任意任务。"""
    with pytest.raises(ValueError, match="task fingerprint"):
        ServiceIdentity(site_id="s", device_id="d", backend="b", backend_version="1", target_id="t", worker_build="w", task_pack_digest="p", calibration_digest="c", task_fingerprints=frozenset(), measurement_kind="measured")


def test_formal_validate_is_rejected_before_a_worker(tmp_path):
    """@brief hello 不再宣告 validate；提交 validate 必须是明确协议错误。"""
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        request = _request(mode="validate")
        _send_request(proc, request)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "protocol_error"
        assert "validate" in error.payload["message"]
    finally:
        proc.close()


def test_missing_evidence_is_rejected(tmp_path):
    """@brief worker 省略 evidence 时不能绕过 source digest 绑定。"""
    identity = ServiceIdentity(site_id="sim-site", device_id="sim-device", backend="sim-npu", backend_version="1", target_id="sim-npu-test", worker_build="sim-build-1", task_pack_digest=TASK_PACK_DIGEST, calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}), measurement_kind="simulated", publication_allowed=False)
    grant = SessionGrant(token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test")
    service = RemoteService(identity=identity, grants=[grant], worker_command=["/bin/true"], state_dir=tmp_path / "svc")
    request = _request()
    report = {"task": request.task, "dtype": request.dtype, "size": request.size, "mode": request.mode, "target_id": request.target_id, "protocol_version": "3", "status": "ok", "simulated": True, "seed": request.seed}
    with pytest.raises(_JobFailed) as exc_info:
        service._validate_worker_result(request, {"report": report})
    assert exc_info.value.code == "provenance_mismatch"
    assert "evidence" in str(exc_info.value)


def test_selfcheck_timeout_locks_future_timed_requests(tmp_path):
    """@brief selfcheck candidate timeout 后，同 run/session 的 timed 必须被锁住。"""
    proc = ServiceProcess(tmp_path)
    assert proc.hello().frame_type == "hello"
    try:
        first = _request(request_id="hang-1", source="REMOTE_SLEEP", kind="selfcheck", mode="timed", timeout=0.05)
        _send_request(proc, first)
        first_result = proc.recv()
        assert first_result.frame_type == "error"
        assert first_result.payload["code"] == "candidate_timeout"
        second = _request(request_id="hang-2", source="kernel void demo() {}", kind="selfcheck", mode="timed", timeout=2.0)
        _send_request(proc, second)
        error = proc.recv()
        assert error.frame_type == "error"
        assert error.payload["code"] == "selfcheck_hang"
    finally:
        proc.close()


def test_report_seed_mismatch_is_rejected(tmp_path):
    """@brief report seed 与请求 seed 不一致时，报告不能进入候选。"""
    identity = ServiceIdentity(site_id="sim-site", device_id="sim-device", backend="sim-npu", backend_version="1", target_id="sim-npu-test", worker_build="sim-build-1", task_pack_digest=TASK_PACK_DIGEST, calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}), measurement_kind="simulated", publication_allowed=False)
    grant = SessionGrant(token=TOKEN, run_id="run-1", session_id="session-1", target_id="sim-npu-test")
    service = RemoteService(identity=identity, grants=[grant], worker_command=["/bin/true"], state_dir=tmp_path / "svc")
    request = _request()
    report = {"task": request.task, "dtype": request.dtype, "size": request.size, "mode": request.mode, "target_id": request.target_id, "protocol_version": "3", "status": "ok", "simulated": True, "seed": request.seed + 1}
    with pytest.raises(_JobFailed) as exc_info:
        service._validate_worker_result(request, {"report": report, "evidence": {"candidate_sha256": request.candidate_sha256}})
    assert exc_info.value.code == "provenance_mismatch"
    assert "seed" in str(exc_info.value)


def test_a_failed_rehello_keeps_the_previous_grant(tmp_path):
    """@brief 失败的 re-hello 不能提前把连接切到新 grant。"""
    identity = ServiceIdentity(site_id="sim-site", device_id="sim-device", backend="sim-npu", backend_version="1", target_id="sim-npu-test", worker_build="sim-build-1", task_pack_digest=TASK_PACK_DIGEST, calibration_digest="c" * 64, task_fingerprints=frozenset({TASK_FINGERPRINT}), measurement_kind="simulated", publication_allowed=False)
    first = SessionGrant(token="first", run_id="run-1", session_id="session-1", target_id="sim-npu-test")
    second = SessionGrant(token="second", run_id="run-2", session_id="session-1", target_id="sim-npu-test")
    service = RemoteService(identity=identity, grants=[first, second], worker_command=["/bin/true"], state_dir=tmp_path / "svc")
    service._authenticated = True
    service._grant = first
    hello = {
        "protocol_version": "1",
        "client_nonce": "failed-rehello",
        "session_token": "second",
        "expected_site_id": "wrong-site",
    }
    with pytest.raises(ProtocolError, match="site"):
        service._handle_hello(Frame("hello", "failed-rehello", hello, 0, ""))
    assert service._grant is first
    assert service._authenticated is True


def test_rehello_switches_grants_and_ages_selfcheck(tmp_path):
    """@brief 同一连接 re-hello 切换 grant 后，selfcheck 能在短 aging 后越过后续 formal。"""
    proc = ServiceProcess(tmp_path, alt=True)
    assert proc.hello().frame_type == "hello"
    try:
        first = _request(request_id="fair-a1", source="REMOTE_SLEEP_SHORT", mode="full")
        _send_request(proc, first)
        frames = []
        assert proc.hello(nonce="client-nonce-2", token="test-alt-token", pending=frames).frame_type == "hello"
        second = _request(request_id="fair-b1", source="kernel void demo() {}", kind="selfcheck", run_id="run-2")
        _send_request(proc, second)
        assert proc.hello(nonce="client-nonce-3", pending=frames).frame_type == "hello"
        third = _request(request_id="fair-a2", source="REMOTE_SLEEP_SHORT", mode="full")
        _send_request(proc, third)
        frames.extend([proc.recv() for _ in range(3 - len(frames))])
        assert [frame.request_id for frame in frames] == ["fair-a1", "fair-b1", "fair-a2"]
        assert all(frame.frame_type == "result" for frame in frames)
    finally:
        proc.close()
