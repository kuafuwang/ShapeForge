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
from tests.support.v2_specimen import V2_TASK


class FakeExecutor:
    def __init__(self) -> None:
        self.requests: list[EvaluationRequest] = []

    def formal(self, request):
        raise AssertionError("bridge must never send formal evaluation")

    def selfcheck(self, request):
        self.requests.append(request)
        # This executor stands in for the *measured* local Metal executor, so its report carries
        # the native structural fixture (explicitly; the helper default is the simulator).
        from tests.support.v2_specimen import v2_structural_evaluation_task
        from tests.support.report_build import attach_build

        task_obj = v2_structural_evaluation_task(request.task, request.dtype, request.size)
        report = EvalReport(
            task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
            seed=1, status="ok", message="ok", elapsed_s=0.5, target_id=request.target_id,
            backend="metal", backend_version="1",
        )
        attach_build(report, task_obj, source=request.source.encode("utf-8"))
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
        task=V2_TASK, dtype="fp16", size="small", target_id="apple-m4-max",
        task_fingerprint="a" * 64, run_id="run-1", session_id="session-1", token="test-token",
        reps=7, self_check_dir=tmp_path / "checks", workspace=tmp_path,
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
    assert request.task == V2_TASK and request.reps == 7 and request.run_id == "run-1"
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


def test_selfcheck_client_response_timeout_is_enforced(tmp_path):
    """@brief bridge 收下请求但不回应时，client 必须在自己的 timeout 后退出。"""
    import time

    path = _short_socket()
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    accepted = threading.Event()
    release = threading.Event()

    def serve():
        conn, _ = listener.accept()
        conn.recv(4096)
        accepted.set()
        release.wait(2)
        conn.close()
        listener.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    started = time.monotonic()
    try:
        with pytest.raises(SelfCheckBridgeError, match="timed out") as exc_info:
            SelfCheckClient(path, token="test-token", timeout_s=0.05).check("kernel void demo() {}", "quick")
        assert exc_info.value.retryable is True
        assert accepted.is_set()
        assert time.monotonic() - started < 1.0
    finally:
        release.set()
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
    from tcenv.evaluation import EvaluationUnavailable

    class Failing:
        def selfcheck(self, request):
            raise EvaluationUnavailable("transport_closed", "remote went away")

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
            from tests.support.report_build import v2_report_for_request

            report = v2_report_for_request(request, status="ok")
            provenance = EvaluatorProvenance(
                schema_version=1, epoch="e" * 64, site_id="sim", device_id="sim", backend="sim-npu", backend_version="1", target_id=request.target_id,
                worker_build="sim", task_pack_digest="p" * 64, calibration_digest="c" * 64, protocol_version="1", request_digest=request.request_digest,
                measurement_kind="simulated", publication_allowed=False,
            )
            return EvaluationResult(report=report, provenance=provenance, evidence={"candidate_sha256": request.candidate_sha256})

    server = SelfCheckServer(_config(tmp_path), SimulatedExecutor(), _short_socket())
    with server:
        result = SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "quick")
    # The simulated marker is carried by the evaluator provenance and the report's build kind,
    # not by a retired top-level report key.
    assert result.report["build"]["build_kind"] == "simulator"
    assert result.report.get("simulated") is None


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


def test_bridge_carries_the_host_contract_launch_policy(tmp_path):
    """@brief remote selfcheck 与 formal 使用同一条 host-owned launch contract 门禁。"""
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path, require_contract_launch=True), executor, _short_socket())
    with server:
        SelfCheckClient(server.socket_path, token="test-token").check("kernel void demo() {}", "quick")
    assert executor.requests[0].require_contract_launch is True


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
    executor = FasterExecutor()
    server = SelfCheckServer(config, executor, _short_socket())
    with server:
        client = SelfCheckClient(server.socket_path, token="test-token")
        client.check("kernel void demo() {}", "timed")
        assert "够换种子" in client.last_presentation["verdict"]
    rows = (tmp_path / "checks" / "self_checks.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["status"] == "ok" and row["kernel_sha256"]
    assert row["request_id"] == executor.requests[0].request_id
    assert row["request_digest"] == executor.requests[0].request_digest
    assert row["measurement_kind"] == "measured" and row["simulated"] is False


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


def test_bridge_cli_rejects_symlinked_and_oversized_candidates(tmp_path, capsys):
    """@brief host 候选解析拒绝符号链接，并在读入内存前拒绝超大文件。"""
    from taic.selfcheck_bridge import MAX_SOURCE_BYTES, main

    actual = tmp_path / "actual.metal"
    actual.write_text("kernel void demo() {}")
    linked = tmp_path / "linked.metal"
    linked.symlink_to(actual)
    token_file = tmp_path / "token"
    token_file.write_text("test-token")
    server = SelfCheckServer(_config(tmp_path), FakeExecutor(), _short_socket())
    with server:
        assert main(["--socket", str(server.socket_path), "--token-file", str(token_file), "--kernel", "linked.metal", "--workspace", str(tmp_path), "--quick"]) == 2
        assert "symbolic links" in capsys.readouterr().err

        large = tmp_path / "large.metal"
        large.write_bytes(b" " * (MAX_SOURCE_BYTES + 1))
        assert main(["--socket", str(server.socket_path), "--token-file", str(token_file), "--kernel", "large.metal", "--workspace", str(tmp_path), "--quick"]) == 2
        assert "source exceeds" in capsys.readouterr().err

        # A caller-selected root cannot redirect the host's session workspace.
        assert main(["--socket", str(server.socket_path), "--token-file", str(token_file), "--kernel", "actual.metal", "--workspace", "/", "--quick"]) == 2
        assert "workspace does not match" in capsys.readouterr().err


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


def test_the_selfcheck_fixture_build_kinds_are_selected_without_a_socket():
    """helper 级断言：实测替身是 native 结构 fixture，模拟路由默认是 simulator。

    不启动 server、不绑定 AF_UNIX，避免沙箱路径过长把这组语义检查遮住。
    """
    from tests.support.report_build import v2_report_for_request

    request = EvaluationRequest(
        task=V2_TASK, dtype="fp16", size="small", source="kernel void demo() {}\n",
        target_id="apple-m4-max",
    )
    measured = FakeExecutor().selfcheck(request).report
    assert measured.build["build_kind"] == "native"
    assert measured.build["provenance"]["builder_id"] == "metal-native-structural-fixture"

    simulated = v2_report_for_request(request, status="ok")
    assert simulated["build"]["build_kind"] == "simulator"
    assert simulated["build"]["provenance"]["builder_id"] == "fake-npu-builder"


def test_the_bridge_runs_a_paired_compare_and_debits_its_own_quota(tmp_path, monkeypatch):
    """@brief 桥自己跑两核比较并扣宿主配额。任务书里还没有这条命令。"""
    from taic.kernel_agent import SessionBrief, render_contract

    calls = {"n": 0}

    def fake_pair(*_args, **kwargs):
        calls["n"] += 1
        return {
            "applied": True, "elapsed_s": 0.25, "sha_a": kwargs["sha_a"], "sha_b": kwargs["sha_b"],
            "changes_seed": False, "gpu": True, "backend": "metal",
        }

    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr("taic.paired_compare.run_host_pair", fake_pair)
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "left.metal").write_text("kernel void left() {}\n", encoding="utf-8")
    (work / "right.metal").write_text("kernel void right() {}\n", encoding="utf-8")
    forged = work / ".selfcheck" / "host_paired_comparisons.jsonl"
    forged.parent.mkdir()
    forged.write_text('{"applied": true, "host_evidence": true}\n', encoding="utf-8")
    executor = FakeExecutor()
    server = SelfCheckServer(_config(tmp_path, workspace=work, self_check_dir=work / ".selfcheck", max_timed=1), executor, _short_socket())
    with server:
        measured = SelfCheckClient(server.socket_path, token="test-token").compare(
            "left.metal", "right.metal", workspace=str(work),
        )
        with pytest.raises(SelfCheckBridgeError, match="quota"):
            SelfCheckClient(server.socket_path, token="test-token").compare(
                "left.metal", "right.metal", workspace=str(work),
            )
    assert calls["n"] == 1
    assert measured["applied"] is True
    assert executor.requests == []
    usage = json.loads(server._usage_path.read_text(encoding="utf-8"))
    assert usage["count"] == 1
    assert usage["gpu_s"] == pytest.approx(0.25)
    assert not (host_dir / "host_paired_comparisons.jsonl").exists()
    nomination = json.loads((work / ".selfcheck" / "paired_comparisons.jsonl").read_text(encoding="utf-8"))
    assert nomination["host_evidence"] is False
    assert len(nomination["sha_a"]) == 64 and len(nomination["sha_b"]) == 64
    quota = json.loads((work / ".selfcheck" / "self_checks.jsonl").read_text(encoding="utf-8"))
    assert quota["kind"] == "paired_compare_quota"
    assert "sha_a" not in quota and "sha_b" not in quota
    assert forged.read_text(encoding="utf-8") == '{"applied": true, "host_evidence": true}\n'
    brief = SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=True,
        reps=15, min_improvement=1.005, noise_z=2.0, work_dir=str(work),
        selfcheck_socket=str(server.socket_path), selfcheck_token_file=str(tmp_path / "token"),
    )
    task_book = render_contract(brief)
    assert "--paired" not in task_book
    assert "paired-compare" not in task_book
    # The unused task-book hint builder is gone. A session ranks variants with the
    # read-only `tcenv ab`; E1's paired text still needs its own named approval, so the
    # local `paired-compare` command must stay out of TASK.md as well.
    local_brief = SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=True,
        reps=15, min_improvement=1.005, noise_z=2.0, work_dir=str(work), v2=True,
    )
    assert "paired-compare" not in render_contract(local_brief)
    assert "tcenv.cli ab" in render_contract(SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=True,
        reps=15, min_improvement=1.005, noise_z=2.0, work_dir=str(work), v2=True, sweep_tool=True,
    ))


def test_two_bridge_nominations_keep_one_host_evidence(tmp_path, monkeypatch):
    """@brief 两次桥请求各写提名和配额行，权威证据只由 host_select 的一次重跑产生。"""
    from taic.paired_compare import host_select

    calls = {"pair": 0, "rerun": 0}

    def fake_pair(*_args, **kwargs):
        calls["pair"] += 1
        return {
            "applied": True, "elapsed_s": 0.1, "sha_a": kwargs["sha_a"], "sha_b": kwargs["sha_b"],
            "changes_seed": False, "conclusion": "no_diff",
        }

    host_dir = tmp_path / "host-authority"
    monkeypatch.setattr("taic.paired_compare.run_host_pair", fake_pair)
    monkeypatch.setattr(
        "taic.paired_compare.host_evidence_path",
        lambda _directory: host_dir / "host_paired_comparisons.jsonl",
    )
    work = tmp_path / "work"
    work.mkdir()
    (work / "left.metal").write_text("kernel void left() {}\n", encoding="utf-8")
    (work / "right.metal").write_text("kernel void right() {}\n", encoding="utf-8")
    (work / "third.metal").write_text("kernel void third() {}\n", encoding="utf-8")
    directory = work / ".selfcheck"
    server = SelfCheckServer(
        _config(tmp_path, workspace=work, self_check_dir=directory, max_timed=5),
        FakeExecutor(), _short_socket(),
    )
    with server:
        SelfCheckClient(server.socket_path, token="test-token").compare("left.metal", "right.metal", workspace=str(work))
        SelfCheckClient(server.socket_path, token="test-token").compare("left.metal", "third.metal", workspace=str(work))
    nominations = [json.loads(line) for line in (directory / "paired_comparisons.jsonl").read_text(encoding="utf-8").splitlines()]
    quotas = [json.loads(line) for line in (directory / "self_checks.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(nominations) == 2
    assert len(quotas) == 2
    assert all(item["host_evidence"] is False and len(item["sha_a"]) == 64 for item in nominations)
    assert all("sha_a" not in item and item["kind"] == "paired_compare_quota" for item in quotas)
    assert not (host_dir / "host_paired_comparisons.jsonl").exists()

    def rerun(selected):
        calls["rerun"] += 1
        item = selected[0]
        return [{
            "applied": True, "elapsed_s": 0.01, "sha_a": item["sha_a"], "sha_b": item["sha_b"],
            "changes_seed": False,
        }]

    saved = host_select(directory, rerun=rerun)
    evidence = (host_dir / "host_paired_comparisons.jsonl").read_text(encoding="utf-8").splitlines()
    assert calls == {"pair": 2, "rerun": 1}
    assert saved["reviewed"] == 1
    assert len(evidence) == 1
    assert json.loads(evidence[0])["host_evidence"] is True


def test_a_remote_bridge_rejects_paired_before_the_local_worker(tmp_path, monkeypatch):
    """@brief 远端桥只允许 quick/timed。paired 在进入 worker 之前被拒，也不扣已有配额。"""
    from taic.remote_runtime import RemoteEvaluationConfig
    from tests.support.remote_pki import make_pki

    calls = {"n": 0}

    def refuse(*_args, **_kwargs):
        calls["n"] += 1
        raise AssertionError("run_host_pair")

    monkeypatch.setattr("taic.paired_compare.run_host_pair", refuse)
    monkeypatch.setattr("tcenv.v2_worker.run_bounded_v2", refuse)
    pki = make_pki(tmp_path / "pki")
    token_file = tmp_path / "grant.token"
    token_file.write_text("grant")
    selfcheck_file = tmp_path / "selfcheck-grant.token"
    selfcheck_file.write_text("selfcheck-token")
    document = {
        "transport": "grpc",
        "endpoint": "127.0.0.1:1",
        "tls_server_name": "localhost",
        "token_file": str(token_file),
        "selfcheck_token_file": str(selfcheck_file),
        "ca_file": str(pki.ca_cert),
        "client_certificate_file": str(pki.client_cert),
        "client_private_key_file": str(pki.client_key),
        "signature_keyring_file": str(pki.keyring_file(key_id="sim-site/key-1", key=b"k" * 32)),
        "target_id": "sim-npu-test",
        "expected_site_id": "sim-site",
        "expected_device_id": "sim-device",
        "expected_worker_build": "sim-build-1",
        "expected_task_pack_digest": "b" * 64,
        "task_fingerprint": "a" * 64,
        "publishable_site_ids": [],
        # AF_UNIX 路径上限 104 字节；tempfile.gettempdir() 会跟着 TMPDIR 变长（pytest 的
        # tmp_path、CI/沙箱前缀都能把它顶过上限），所以和同目录另外两处一样固定用 /tmp。
        "socket_root": tempfile.mkdtemp(prefix="taic-remote-", dir="/tmp"),
    }
    config_path = tmp_path / "remote-grpc.json"
    config_path.write_text(json.dumps(document))
    config = RemoteEvaluationConfig.from_file(config_path)
    work = tmp_path / "run" / "work"
    work.mkdir(parents=True)
    factory = config.make_bridge_factory(
        task=V2_TASK, dtype="fp16", size="small",
        task_fingerprint="a" * 64, run_id="run-1",
    )
    server, bridge_token = factory("session-1", work, None)
    try:
        assert server.config.allowed_modes == frozenset({"quick", "timed"})
        server._usage_path.parent.mkdir(parents=True, exist_ok=True)
        server._usage_path.write_text(json.dumps({"count": 4, "gpu_s": 1.25, "hang": False}), encoding="utf-8")
        with pytest.raises(SelfCheckBridgeError, match="not allowed") as caught:
            SelfCheckClient(server.socket_path, token=Path(bridge_token).read_text(encoding="utf-8").strip()).compare(
                "left.metal", "right.metal",
            )
        assert caught.value.code == "mode_not_allowed"
        saved = json.loads(server._usage_path.read_text(encoding="utf-8"))
        assert saved == {"count": 4, "gpu_s": 1.25, "hang": False}
        assert server._usage["count"] == 0
        assert calls["n"] == 0
        assert not (work / ".selfcheck" / "self_checks.jsonl").exists()
    finally:
        server.stop()
