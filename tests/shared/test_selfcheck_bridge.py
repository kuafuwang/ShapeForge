"""CPU-only tests for the host-controlled self-check bridge."""

from __future__ import annotations

import json
import socket
import tempfile
import threading
import uuid
from pathlib import Path

import pytest

from taic.selfcheck_bridge import SelfCheckBridgeError, SelfCheckClient, SelfCheckConfig, SelfCheckServer
from tcenv.evaluation import EvaluationRequest, EvaluationResult, EvaluatorProvenance
from tcenv.evaluate import EvalReport


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def formal(self, request):
        raise AssertionError("bridge must never send formal evaluation")

    def selfcheck(self, request):
        self.requests.append(request)
        report = EvalReport(task=request.task, dtype=request.dtype, size=request.size, mode=request.mode, seed=1, status="ok", message="ok", elapsed_s=0.5)
        provenance = EvaluatorProvenance(
            schema_version=1, epoch="e" * 64, site_id="local", device_id=request.target_id,
            backend="metal", backend_version="1", target_id=request.target_id,
            worker_build="w", task_pack_digest="p" * 64, calibration_digest="c" * 64,
            protocol_version="1", request_digest=request.request_digest,
            measurement_kind="measured", publication_allowed=True,
        )
        return EvaluationResult(report=report, provenance=provenance, evidence={"candidate_sha256": request.candidate_sha256})


def _short_socket() -> Path:
    return Path(tempfile.gettempdir()) / f"taic-sc-{uuid.uuid4().hex}.sock"


def _config(tmp_path: Path, **overrides) -> SelfCheckConfig:
    values = dict(
        task="relu", dtype="fp16", size="small", target_id="apple-m4-max",
        task_fingerprint="a" * 64, run_id="run-1", session_id="session-1", token="test-token",
        reps=7, self_check_dir=tmp_path / "checks",
    )
    values.update(overrides)
    return SelfCheckConfig(**values)


def test_bridge_sends_only_selfcheck_with_host_owned_parameters(tmp_path):
    """@brief agent 只能提交源码和 quick/timed，task/reps 由 host 配置决定。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path), executor, _short_socket())
    with server:
        result = SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "quick")
    assert result.report["status"] == "ok"
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.kind == "selfcheck" and request.mode == "quick"
    assert request.task == "relu" and request.reps == 7 and request.run_id == "run-1"
    assert request.cache_policy == "disabled"


def test_bridge_rejects_a_mode_outside_host_config(tmp_path):
    """@brief host 只允许 quick 时，timed 请求必须在进入 evaluator 前被拒。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path, allowed_modes=frozenset({"quick"})), executor, _short_socket())
    with server:
        with pytest.raises(SelfCheckBridgeError, match="not allowed"):
            SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "timed")
    assert executor.requests == []


def test_bridge_rejects_extra_request_fields(tmp_path):
    """@brief bridge 不接受 agent 自带的 task、target 或配额字段。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path), executor, _short_socket())
    with server:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(str(server.socket_path))
            sock.sendall((json.dumps({"mode": "quick", "source": "kernel void demo() {}", "token": "test-token", "task": "gemm"}) + "\n").encode())
            response = json.loads(sock.makefile("rb").readline())
    assert response["ok"] is False
    assert response["error"]["code"] == "invalid_request"
    assert executor.requests == []


def test_bridge_socket_is_private_and_removed_on_stop(tmp_path):
    """@brief bridge socket 只允许当前用户访问，停止后删除。"""
    server = SelfCheckServer(_config(tmp_path), FakeExecutor(), _short_socket())
    path = server.start()
    try:
        assert path.stat().st_mode & 0o077 == 0
    finally:
        server.stop()
    assert not path.exists()


def test_selfcheck_client_rejects_non_object_response(tmp_path):
    """@brief 非对象的 bridge 响应必须变成明确的协议错误。"""
    path = _short_socket()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        conn.recv(4096)
        conn.sendall(b"[]\n")
        conn.close()
        listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        with pytest.raises(SelfCheckBridgeError, match="must be an object"):
            SelfCheckClient(path, token="test-token").check("kernel void demo() {}", "quick")
    finally:
        path.unlink(missing_ok=True)
        thread.join(timeout=2)


def test_bridge_rejects_a_wrong_session_token(tmp_path):
    """@brief 错误 token 不能调用 evaluator，也不能消耗自测配额。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path), executor, _short_socket())
    with server:
        with pytest.raises(SelfCheckBridgeError, match="token"):
            SelfCheckClient(server.socket_path, token="wrong-token").check("kernel void demo() {}", "quick")
    assert executor.requests == []


def test_bridge_quota_and_hang_usage_survive_restart(tmp_path):
    """@brief host 自己记录次数和 hang；重启后仍不能继续 timed。"""
    executor = FakeExecutor()
    socket_path = _short_socket()
    config = _config(tmp_path, max_timed=1)
    server = SelfCheckServer(config, executor, socket_path)
    usage_path = server._usage_path
    with server:
        SelfCheckClient(socket_path, token="test-token").check("kernel void demo() {}", "timed")
        with pytest.raises(SelfCheckBridgeError, match="quota"):
            SelfCheckClient(socket_path, token="test-token").check("kernel void demo() {}", "timed")
    restarted = SelfCheckServer(config, executor, socket_path)
    with restarted:
        with pytest.raises(SelfCheckBridgeError, match="quota"):
            SelfCheckClient(socket_path, token="test-token").check("kernel void demo() {}", "timed")
    assert json.loads(usage_path.read_text())["count"] == 1


def test_bridge_remote_error_does_not_kill_accept_loop(tmp_path):
    """@brief 远程不可用返回结构化错误，accept loop 仍能服务后续请求。"""
    from tcenv.remote.client import RemoteError

    class Failing:
        def selfcheck(self, request):
            raise RemoteError("transport_closed", "remote went away")

    server = SelfCheckServer(_config(tmp_path), Failing(), _short_socket())
    with server:
        for _ in range(2):
            with pytest.raises(SelfCheckBridgeError, match="remote went away"):
                SelfCheckClient(server.socket_path, token="test-token", timeout_s=1).check("kernel void demo() {}", "quick")
        assert server._thread is not None and server._thread.is_alive()


def test_bridge_stop_interrupts_an_idle_connection(tmp_path):
    """@brief idle client 不得让 stop 留下线程或等待连接超时。"""
    server = SelfCheckServer(_config(tmp_path), FakeExecutor(), _short_socket())
    server.start()
    idle = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    idle.connect(str(server.socket_path))
    thread = server._thread
    try:
        started = __import__("time").monotonic()
        server.stop()
        assert __import__("time").monotonic() - started < 1.0
        assert thread is not None and not thread.is_alive()
    finally:
        idle.close()


def test_bridge_allows_simulated_selfcheck_but_keeps_marker(tmp_path):
    """@brief 模拟自测可用于反馈，结果仍保留 simulated 标记，不能伪装实测。"""
    class SimulatedExecutor(FakeExecutor):
        def selfcheck(self, request):
            self.requests.append(request)
            report = {"task": request.task, "dtype": request.dtype, "size": request.size, "mode": request.mode, "status": "ok", "simulated": True, "target_id": request.target_id}
            provenance = EvaluatorProvenance(
                schema_version=1, epoch="e" * 64, site_id="sim", device_id="sim", backend="sim-npu", backend_version="1", target_id=request.target_id,
                worker_build="sim", task_pack_digest="p" * 64, calibration_digest="c" * 64, protocol_version="1", request_digest=request.request_digest,
                measurement_kind="simulated", publication_allowed=False,
            )
            return EvaluationResult(report=report, provenance=provenance, evidence={"candidate_sha256": request.candidate_sha256})

    server = SelfCheckServer(_config(tmp_path), SimulatedExecutor(), _short_socket())
    with server:
        result = SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "quick")
    assert result.report["simulated"] is True


def test_bridge_validate_is_host_controlled_and_reaches_the_evaluator(tmp_path):
    """@brief validate 只能由 host 策略开放，并且作为 request.validate 传给 evaluator。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path, allow_validate=True), executor, _short_socket())
    with server:
        result = SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "timed", validate=True)
    assert result.report["status"] == "ok"
    assert executor.requests[0].validate is True

    blocked = SelfCheckServer(_config(tmp_path, allow_validate=False), FakeExecutor(), _short_socket())
    with blocked:
        with pytest.raises(SelfCheckBridgeError, match="validation"):
            SelfCheckClient(blocked.socket_path, token="test-token").check("kernel void demo() {}", "timed", validate=True)
    with pytest.raises(ValueError, match="validation"):
        SelfCheckClient(blocked.socket_path, token="test-token").check("kernel void demo() {}", "quick", validate=True)


def test_bridge_host_writes_authoritative_selfcheck_row_and_verdict(tmp_path):
    """@brief timed selfcheck 由 host 写权威行，并用宿主种子报告生成换种子结论。"""
    seed_path = tmp_path / "seed-report.json"
    seed_path.write_text(json.dumps({"speedup": 1.0, "benchmark": {"candidate": {"cv": 0.01}}}), encoding="utf-8")

    class FasterExecutor(FakeExecutor):
        def selfcheck(self, request):
            result = super().selfcheck(request)
            result.report.speedup = 2.0
            result.report.benchmark = {"candidate": {"cv": 0.01}}
            return result

    config = _config(tmp_path, seed_report_path=seed_path)
    server = SelfCheckServer(config, FasterExecutor(), _short_socket())
    with server:
        client = SelfCheckClient(server.socket_path, token="test-token")
        client.check("kernel void demo() {}", "timed")
        assert "够换种子" in client.last_presentation["verdict"]
    rows = (tmp_path / "checks" / "self_checks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["status"] == "ok" and row["kernel_sha256"]


def test_bridge_cli_reads_token_file_and_workspace_relative_kernel(tmp_path, monkeypatch, capsys):
    """@brief CLI 从 token 文件取凭据，只接受 workspace 相对候选路径。"""
    from taic.selfcheck_bridge import main

    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path), executor, _short_socket())
    token_file = tmp_path / "selfcheck.token"
    token_file.write_text("test-token")
    (tmp_path / "candidate.metal").write_text("kernel void demo() {}")
    monkeypatch.chdir(tmp_path)
    with server:
        status = main(["--socket", str(server.socket_path), "--token-file", str(token_file), "--kernel", "candidate.metal", "--quick"])
    assert status == 0
    assert len(executor.requests) == 1
    assert "ok" in capsys.readouterr().out


def test_bridge_cli_reads_the_session_token_from_the_environment(tmp_path, monkeypatch, capsys):
    """@brief host bridge 通过会话环境提供 token，CLI 不需要把凭据写进 argv。"""
    from taic.selfcheck_bridge import main

    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path), executor, _short_socket())
    (tmp_path / "candidate.metal").write_text("kernel void demo() {}")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAIC_SELFCHECK_SOCKET", str(server.socket_path))
    monkeypatch.setenv("TAIC_SELFCHECK_TOKEN", "test-token")
    monkeypatch.delenv("TAIC_SELFCHECK_TOKEN_FILE", raising=False)
    with server:
        status = main(["--kernel", "candidate.metal", "--quick"])
    assert status == 0
    assert len(executor.requests) == 1
    assert "ok" in capsys.readouterr().out


def test_server_provisions_and_removes_a_0600_token_file(tmp_path):
    """@brief host 生成 0600 token 文件，stop 后删除，不把 token 写进 argv。"""
    token_path = tmp_path / "selfcheck.token"
    config = _config(tmp_path, token_file=token_path)
    server = SelfCheckServer(config, FakeExecutor(), _short_socket())
    server.start()
    try:
        assert token_path.read_text() == "test-token"
        assert token_path.stat().st_mode & 0o077 == 0
        result = SelfCheckClient(server.socket_path, token_file=token_path).check("kernel void demo() {}", "quick")
        assert result.report["status"] == "ok"
    finally:
        server.stop()
    assert not token_path.exists()


def test_bridge_rejects_a_usage_path_inside_the_agent_workspace(tmp_path):
    """@brief host-owned quota 不能放在 agent 可改的 self_check_dir 里。"""
    config = _config(tmp_path, usage_path=tmp_path / "checks" / "usage.json")
    with pytest.raises(ValueError, match="usage_path"):
        SelfCheckServer(config, FakeExecutor(), _short_socket())
