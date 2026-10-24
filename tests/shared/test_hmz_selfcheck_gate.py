"""Real hmz online=NONE fence acceptance for the host self-check bridge.

The model is replaced by a fake ACP CLI, and the evaluator is a deterministic
in-process fake: this test starts a real ``hmz exec`` flow and exercises the
real ``tcenv`` selfcheck bridge over a Unix socket.  It never contacts a
provider, device or GPU.
"""

from __future__ import annotations

import json
import shutil
import socket
import sys
import tempfile
import threading
import uuid
from pathlib import Path

import pytest

pytest.importorskip("hmz")


ROOT = Path(__file__).resolve().parents[2]
TASK_FINGERPRINT = "a" * 64


class _RecordingExecutor:
    """A deterministic stand-in for a device evaluator."""

    def __init__(self) -> None:
        self.requests = []

    def formal(self, request):
        raise AssertionError("the bridge must never submit formal evaluation")

    def selfcheck(self, request):
        from tcenv.evaluate import EvalReport
        from tcenv.evaluation import EvaluationResult, EvaluatorProvenance

        self.requests.append(request)
        report = EvalReport(
            task=request.task,
            dtype=request.dtype,
            size=request.size,
            mode=request.mode,
            status="ok",
            stage="correct",
            seed=1,
            target_id=request.target_id,
            elapsed_s=0.0,
        )
        provenance = EvaluatorProvenance(
            schema_version=1,
            epoch="e" * 64,
            site_id="fake-site",
            device_id=request.target_id,
            backend="fake",
            backend_version="1",
            target_id=request.target_id,
            worker_build="fake-build",
            task_pack_digest="b" * 64,
            calibration_digest="c" * 64,
            protocol_version="1",
            request_digest=request.request_digest,
            measurement_kind="measured",
            publication_allowed=False,
        )
        return EvaluationResult(
            report=report,
            provenance=provenance,
            evidence={"candidate_sha256": request.candidate_sha256},
        )


def _brief(work: Path, socket_path: Path, token_file: Path):
    from taic.kernel_agent import SessionBrief

    return SessionBrief(
        repo=ROOT,
        task_name="relu",
        dtype="fp16",
        size="small",
        kernel_name="demo",
        triton_source="",
        description="a fake ACP bridge acceptance",
        contract_text="",
        abi_stub="kernel void demo() {}",
        rtol=1e-3,
        atol_rel=1e-3,
        seedless=True,
        target_id="apple-m4-max",
        work_dir=str(work),
        selfcheck_socket=str(socket_path),
        selfcheck_token_file=str(token_file),
    )


@pytest.mark.slow
def test_real_hmz_online_none_can_use_only_its_own_selfcheck_bridge(tmp_path, monkeypatch):
    """@brief 真实 hmz 围栏里，bridge 只能被本 session 通过环境提供的 token 使用。"""
    from taic.kernel_agent import Prices, run_kernel_agent
    from taic.selfcheck_bridge import SelfCheckBridgeError, SelfCheckClient, SelfCheckConfig, SelfCheckServer

    # Avoid importing a private test helper; the prices are deliberately zero
    # because this session must not call a provider.
    prices = Prices(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0)
    hmz = shutil.which("hmz")
    if not hmz:
        pytest.skip("hmz is not installed")
    if not shutil.which("uv"):
        pytest.skip("uv is needed by the rendered selfcheck command")

    home = tmp_path / "hmz-home"
    home.mkdir()
    agent_state = tmp_path / "agent-state"
    agent_script = ROOT / "tests" / "support" / "hmz_selfcheck_agent.py"
    settings = {
        "clis": {
            "fake-acp": {
                "command": [sys.executable, str(agent_script)],
                # ACP requires a declared model host even though this fake
                # never opens a network connection. TEST-NET-1 is not routed.
                "hosts": ["192.0.2.1"],
                "state": [str(agent_state)],
            }
        }
    }
    (home / "settings.yaml").write_text(json.dumps(settings), encoding="utf-8")
    monkeypatch.setenv("HUMANIZE_HOME", str(home))

    host_cache = Path.home() / ".cache"
    host_cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    bridge_root = Path(tempfile.mkdtemp(prefix="taic-hmz-", dir=str(host_cache)))
    socket_path = bridge_root / "session.sock"
    token_file = bridge_root / "session.token"
    token = "session-selfcheck-token"
    other_socket = bridge_root / "other.sock"
    other_token_file = bridge_root / "other.token"
    other_token = "other-session-token"

    executor = _RecordingExecutor()
    other_executor = _RecordingExecutor()
    server = SelfCheckServer(
        SelfCheckConfig(
            task="relu",
            dtype="fp16",
            size="small",
            target_id="apple-m4-max",
            task_fingerprint=TASK_FINGERPRINT,
            run_id="run-1",
            session_id="session-1",
            token=token,
            token_file=token_file,
            workspace=tmp_path / "agent_work" / "init",
            usage_path=bridge_root / "usage.json",
        ),
        executor,
        socket_path,
    )
    other_server = SelfCheckServer(
        SelfCheckConfig(
            task="relu",
            dtype="fp16",
            size="small",
            target_id="apple-m4-max",
            task_fingerprint=TASK_FINGERPRINT,
            run_id="run-2",
            session_id="session-2",
            token=other_token,
            token_file=other_token_file,
            usage_path=bridge_root / "other-usage.json",
        ),
        other_executor,
        other_socket,
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    tcp_port = listener.getsockname()[1]
    accepting = threading.Event()
    accepting.set()

    def accept_once() -> None:
        while accepting.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            conn.close()

    acceptor = threading.Thread(target=accept_once, name="hmz-tcp-probe", daemon=True)
    acceptor.start()
    server.start()
    other_server.start()
    monkeypatch.setenv("TAIC_TEST_HMZ_OTHER_SOCKET", str(other_socket))
    monkeypatch.setenv("TAIC_TEST_HMZ_TCP_PORT", str(tcp_port))
    try:
        from taic import kernel_agent

        monkeypatch.setattr(kernel_agent, "agent_spec", lambda *_args: "fake-acp/fake-model")
        work = tmp_path / "agent_work" / "init"
        brief = _brief(work, socket_path, token_file)
        result = run_kernel_agent(
            brief,
            work=work,
            provider="fake",
            model="fake",
            effort=None,
            prices=prices,
            turn_output_tokens=1000,
            turn_minutes=1,
            hmz=hmz,
            timeout_s=45,
        )
        assert result.source and "kernel void demo" in result.source, result.error
        evidence = json.loads(result.notes)
        assert evidence["command_found"] is True
        assert evidence["selfcheck_rc"] == 0, evidence
        assert evidence["socket_env_present"] is True
        assert evidence["token_env_present"] is True
        assert evidence["token_file_env_present"] is False
        assert evidence["socket_in_task"] is False
        assert evidence["token_in_task"] is False
        assert evidence["socket_in_command"] is False
        assert evidence["token_file_in_command"] is False
        assert evidence["workspace_token_file_exists"] is False
        assert evidence["socket_mode"] is not None and evidence["socket_mode"] & 0o077 == 0
        assert evidence["socket_dir_mode"] is not None and evidence["socket_dir_mode"] & 0o077 == 0
        assert evidence["outside_write_refused"] is True, evidence
        assert evidence["network_connect_succeeded"] is False, evidence
        assert evidence["cross_session_error"] == "unauthorized", evidence
        assert evidence["formal_mode_error"] == "mode_not_allowed", evidence
        assert evidence["wrong_token_error"] == "unauthorized", evidence
        assert len(executor.requests) == 1
        assert other_executor.requests == []
    finally:
        server.stop()
        other_server.stop()
        accepting.clear()
        listener.close()
        acceptor.join(timeout=2)
        shutil.rmtree(bridge_root, ignore_errors=True)

    assert not socket_path.exists()
    assert not token_file.exists()
    assert not other_socket.exists()
    assert not other_token_file.exists()
    shutil.rmtree(bridge_root, ignore_errors=True)
    with pytest.raises(SelfCheckBridgeError):
        SelfCheckClient(socket_path, token=token, timeout_s=1).check("kernel void demo() {}", "quick")
