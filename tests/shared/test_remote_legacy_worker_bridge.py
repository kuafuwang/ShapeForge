"""Real-subprocess tests for the private JobEngine-to-legacy worker bridge."""

from __future__ import annotations

import hashlib
import json
import os
import select
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.remote.job_types import Attempt, ExternalEvaluationRequest
from tcenv.remote.legacy_worker_bridge import _translate_progress
from tcenv.remote.protocol import FrameDecoder, read_frame
from tcenv.remote.worker_protocol import (
    MAX_WORKER_FRAME_PAYLOAD,
    decode_worker_frame,
    encode_request,
)

from ..support.fake_worker_legacy_bridge import LONG_DETAIL
from ..support.job_engine import request as make_request
from ..support.process_state import process_state

ROOT = Path(__file__).resolve().parents[2]
NPU_WORKER = [sys.executable, "-m", "tests.support.npu_sim.worker_entry"]
FAKE_WORKER = [sys.executable, "-m", "tests.support.fake_worker_legacy_bridge"]


def _attempt() -> Attempt:
    return Attempt("bridge-attempt-1", 9, "running", 0, False, 0)


def _request(request_id: str = "bridge-request-1", *, source: bytes = b"OK") -> ExternalEvaluationRequest:
    return make_request(request_id, source=source, candidate_timeout_ms=5000)


def _spawn_bridge(
    tmp_path: Path,
    command: list[str] | None,
    *,
    command_json: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen, Path]:
    temp_root = tmp_path / "bridge-tmp"
    temp_root.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["TMPDIR"] = str(temp_root)
    env.pop("TCENV_LEGACY_WORKER_COMMAND", None)
    env.pop("TCENV_REMOTE_STATE_DIR", None)
    env.pop("FAKE_LEGACY_BEHAVIOR", None)
    env.pop("FAKE_LEGACY_RECORD", None)
    env.pop("FAKE_LEGACY_RELEASE", None)
    env.pop("NPU_SIM_WORKER_EXIT", None)
    if command_json is not None:
        env["TCENV_LEGACY_WORKER_COMMAND"] = command_json
    elif command is not None:
        env["TCENV_LEGACY_WORKER_COMMAND"] = json.dumps(command)
    if extra_env:
        env.update(extra_env)
    process = subprocess.Popen(
        [sys.executable, "-m", "tcenv.remote.legacy_worker_bridge"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        cwd=ROOT,
        env=env,
        # The test gives the bridge its own group.  The production bridge must
        # not give its legacy child another one.
        start_new_session=True,
    )
    return process, temp_root


def _destroy_bridge(process: subprocess.Popen) -> None:
    try:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _communicate(
    process: subprocess.Popen,
    request: ExternalEvaluationRequest,
    attempt: Attempt,
    *,
    timeout: float = 15,
) -> tuple[int, bytes, bytes]:
    output, error = process.communicate(encode_request(request, attempt), timeout=timeout)
    assert process.returncode is not None
    return process.returncode, output or b"", error or b""


def _run_bridge(
    tmp_path: Path,
    request: ExternalEvaluationRequest,
    attempt: Attempt,
    command: list[str] | None,
    *,
    command_json: str | None = None,
    extra_env: dict[str, str] | None = None,
) -> tuple[int, bytes, bytes, Path]:
    process, temp_root = _spawn_bridge(
        tmp_path, command, command_json=command_json, extra_env=extra_env,
    )
    try:
        returncode, output, error = _communicate(process, request, attempt)
        return returncode, output, error, temp_root
    finally:
        _destroy_bridge(process)


def _decode_engine_frames(
    raw: bytes, request: ExternalEvaluationRequest, attempt: Attempt,
) -> list[tuple[object, str, dict]]:
    decoder = FrameDecoder(max_payload=MAX_WORKER_FRAME_PAYLOAD)
    frames = decoder.feed(raw)
    decoder.finish()
    decoded = []
    for frame in frames:
        kind, payload = decode_worker_frame(frame, attempt, request_id=request.request_id)
        decoded.append((frame, kind, payload))
    return decoded


def _send_request(
    process: subprocess.Popen, request: ExternalEvaluationRequest, attempt: Attempt,
) -> None:
    assert process.stdin is not None
    process.stdin.write(encode_request(request, attempt))
    process.stdin.flush()
    process.stdin.close()


def _recv_engine_frame(
    process: subprocess.Popen, request: ExternalEvaluationRequest, attempt: Attempt,
    *, timeout: float = 10,
):
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    assert readable, "timed out waiting for a bridge frame"
    frame = read_frame(process.stdout, max_payload=MAX_WORKER_FRAME_PAYLOAD)
    kind, payload = decode_worker_frame(frame, attempt, request_id=request.request_id)
    return frame, kind, payload


def _ps_state(pid: int) -> str | None:
    """State/reaped probe; never executes ``ps``, which a fence can deny."""

    return process_state(pid)


def test_npu_worker_result_progress_binding_and_temp_cleanup(tmp_path):
    request = _request(source=b"// REMOTE_PROGRESS\r\nkernel void demo() {} \r\n")
    attempt = _attempt()
    returncode, output, error, temp_root = _run_bridge(
        tmp_path, request, attempt, NPU_WORKER,
    )
    assert returncode == 0, error.decode("utf-8", errors="replace")
    decoded = _decode_engine_frames(output, request, attempt)
    assert decoded[-1][1] == "result"
    assert sum(kind == "result" for _frame, kind, _payload in decoded) == 1
    progress = [payload for _frame, kind, payload in decoded if kind == "progress"]
    assert progress, "REMOTE_PROGRESS must produce at least one engine progress frame"
    # The fake has no delay after its terminal write; the final drain, rather
    # than a lucky polling iteration, must preserve this snapshot.
    assert progress[-1]["stage"] == "finalizing"
    assert progress[-1]["detail"] == "fake progress complete"
    assert isinstance(progress[-1]["observed_at_ms"], int)
    result_frame, _kind, result = decoded[-1]
    assert result_frame.request_id == request.request_id
    assert result["report"]["simulated"] is True
    assert result["report"]["status"] == "ok"
    assert result["evidence"] == {
        "request_id": request.request_id,
        "candidate_sha256": request.candidate_sha256,
        "measurement_kind": "simulated",
    }
    assert set(result_frame.payload) == {
        "worker_protocol", "attempt_id", "lease_fence", "report", "evidence",
    }
    assert list(temp_root.iterdir()) == []
    assert str(temp_root).encode() not in output
    assert b"TCENV_REMOTE_PROGRESS_PATH" not in output


def test_exact_legacy_request_same_process_group_and_success_reap(tmp_path):
    source_text = "// π 🚀\r\nkernel void demo() {} \r\n"
    request = _request(source=source_text.encode("utf-8"))
    attempt = _attempt()
    record_path = tmp_path / "legacy-record.json"
    process, temp_root = _spawn_bridge(
        tmp_path, FAKE_WORKER,
        extra_env={"FAKE_LEGACY_RECORD": str(record_path)},
    )
    try:
        returncode, output, error = _communicate(process, request, attempt)
        assert returncode == 0, error.decode("utf-8", errors="replace")
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["source"] == source_text
        assert record["request_id"] == request.request_id
        assert record["candidate_sha256"] == request.candidate_sha256
        assert record["stdin_eof"] is True
        assert record["parent_pid"] == process.pid
        assert record["pgid"] == process.pid
        assert record["parent_pgid"] == process.pid
        # communicate() returning is not enough: ps must not show either a
        # live child or an unreaped zombie after the successful bridge exit.
        assert _ps_state(record["pid"]) is None
        progress_path = Path(record["progress_path"])
        assert not progress_path.exists()
        assert not progress_path.parent.exists()
        assert list(temp_root.iterdir()) == []

        decoded = _decode_engine_frames(output, request, attempt)
        assert len(decoded) == 1
        result_frame, kind, result = decoded[0]
        assert kind == "result"
        assert result["report"]["legacy_report_marker"] == "report-from-legacy-worker"
        assert result["evidence"]["legacy_evidence_marker"] == "evidence-from-legacy-worker"
        assert "receipt" not in result_frame.payload
        assert "signature" not in result_frame.payload
    finally:
        _destroy_bridge(process)


def test_state_scoped_temp_root_prunes_residue_and_cleans_the_attempt(tmp_path):
    """@brief 临时目录由 state 目录限定：SIGKILL 残留会在下一次尝试前清掉。"""
    request = _request()
    record_path = tmp_path / "scoped-record.json"
    state_dir = tmp_path / "state"
    root = state_dir / "bridge-tmp"
    stale = root / "attempt-orphan"
    stale.mkdir(parents=True)
    (stale / "progress.json").write_text("{}", encoding="utf-8")
    unrelated = root / "not-ours.txt"
    unrelated.write_text("keep", encoding="utf-8")

    returncode, output, error, _temp_root = _run_bridge(
        tmp_path, request, _attempt(), FAKE_WORKER,
        extra_env={
            "FAKE_LEGACY_RECORD": str(record_path),
            "TCENV_REMOTE_STATE_DIR": str(state_dir),
        },
    )
    assert returncode == 0, error.decode("utf-8", errors="replace")
    progress_path = Path(json.loads(record_path.read_text(encoding="utf-8"))["progress_path"])
    assert progress_path.parent.parent == root
    assert progress_path.parent.name.startswith("attempt-")
    assert not progress_path.parent.exists()
    assert not stale.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert list(root.iterdir()) == [unrelated]


def test_non_lowercase_state_dir_is_not_used_for_temp_root(tmp_path):
    """@brief 非绝对路径不能悄悄把临时目录指到当前工作目录之外的相对位置。"""
    from tcenv.remote.legacy_worker_bridge import _bridge_temp_root

    assert _bridge_temp_root({"TCENV_REMOTE_STATE_DIR": "relative/state"}) is None
    assert _bridge_temp_root({}) is None


def test_nonexistent_child_command_fails_without_a_result(tmp_path):
    request = _request()
    returncode, output, error, _temp_root = _run_bridge(
        tmp_path, request, _attempt(), [str(tmp_path / "does-not-exist")],
    )
    assert returncode != 0
    assert output == b""
    assert b"could not start legacy worker" in error


def test_nonzero_child_is_reaped_and_emits_no_result(tmp_path):
    request = _request()
    record_path = tmp_path / "failed-legacy-record.json"
    process, _temp_root = _spawn_bridge(
        tmp_path, FAKE_WORKER,
        extra_env={
            "FAKE_LEGACY_BEHAVIOR": "nonzero",
            "FAKE_LEGACY_RECORD": str(record_path),
        },
    )
    try:
        returncode, output, error = _communicate(process, request, _attempt())
        assert returncode != 0
        assert output == b""
        assert b"legacy worker exited 23" in error
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert _ps_state(record["pid"]) is None
    finally:
        _destroy_bridge(process)


@pytest.mark.parametrize("configured", [
    None,
    "",
    "not-json",
    "null",
    "{}",
    "[]",
    '[""]',
    '["python", ""]',
    '["python", 3]',
    '[true]',
])
def test_malformed_or_missing_legacy_command_is_rejected(tmp_path, configured):
    returncode, output, error, temp_root = _run_bridge(
        tmp_path, _request(), _attempt(), None, command_json=configured,
    )
    assert returncode != 0
    assert output == b""
    assert b"TCENV_LEGACY_WORKER_COMMAND" in error
    assert list(temp_root.iterdir()) == []


@pytest.mark.parametrize("behavior,error_fragment", [
    ("wrong_request", b"request_id"),
    ("multiple", b"exactly one result"),
    ("malformed", b"malformed frames"),
    ("missing_report", b"report"),
    ("invalid_report", b"report"),
    ("wrong_type", b"exactly one result"),
    ("invalid_evidence", b"evidence"),
])
def test_invalid_legacy_results_fail_without_an_engine_result(
    tmp_path, behavior, error_fragment,
):
    returncode, output, error, temp_root = _run_bridge(
        tmp_path, _request(), _attempt(), FAKE_WORKER,
        extra_env={"FAKE_LEGACY_BEHAVIOR": behavior},
    )
    assert returncode != 0
    assert output == b""
    assert error_fragment in error
    assert list(temp_root.iterdir()) == []


def test_bench_done_progress_is_bounded_and_terminal_is_before_result(tmp_path):
    request = _request()
    attempt = _attempt()
    release = tmp_path / "release-progress"
    record_path = tmp_path / "progress-legacy-record.json"
    process, temp_root = _spawn_bridge(
        tmp_path, FAKE_WORKER,
        extra_env={
            "FAKE_LEGACY_BEHAVIOR": "progress_handshake",
            "FAKE_LEGACY_RELEASE": str(release),
            "FAKE_LEGACY_RECORD": str(record_path),
        },
    )
    try:
        _send_request(process, request, attempt)
        _frame, kind, progress = _recv_engine_frame(process, request, attempt)
        assert kind == "progress"
        assert progress == {
            "stage": "timing",
            "detail": LONG_DETAIL[:300],
            "observed_at_ms": 1_234_567,
        }
        assert len(progress["detail"]) == 300
        assert str(temp_root) not in progress["detail"]
        # Hold the legacy child in bench until the first frame is consumed, so
        # this covers bench without relying on scheduler-sensitive sleeps.
        release.write_text("release", encoding="utf-8")

        _frame, kind, terminal = _recv_engine_frame(process, request, attempt)
        assert kind == "progress"
        assert terminal == {
            "stage": "finalizing",
            "detail": "terminal done",
            "observed_at_ms": 5000,
        }
        result_frame, kind, result = _recv_engine_frame(process, request, attempt)
        assert kind == "result"
        assert result["report"]["status"] == "ok"
        assert result_frame.request_id == request.request_id
        assert process.wait(timeout=10) == 0
        assert process.stdout is not None
        assert process.stdout.read() == b""
        assert process.stderr is not None
        assert process.stderr.read() == b""
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert _ps_state(record["pid"]) is None
        assert list(temp_root.iterdir()) == []
    finally:
        _destroy_bridge(process)


def test_unknown_progress_stage_uses_safe_fallback(tmp_path):
    returncode, output, error, _temp_root = _run_bridge(
        tmp_path, _request(), _attempt(), FAKE_WORKER,
        extra_env={"FAKE_LEGACY_BEHAVIOR": "unknown_progress"},
    )
    assert returncode == 0, error.decode("utf-8", errors="replace")
    decoded = _decode_engine_frames(output, _request(), _attempt())
    assert [kind for _frame, kind, _payload in decoded] == ["progress", "result"]
    assert decoded[0][2] == {
        "stage": "queued",
        "detail": "unknown legacy stage",
    }


def test_every_documented_legacy_stage_alias_maps_to_an_engine_stage():
    expected = {
        "queued": "queued",
        "static": "static_check",
        "validate": "static_check",
        "static_check": "static_check",
        "compile": "compile",
        "correctness": "correctness",
        "bench": "timing",
        "timing": "timing",
        "done": "finalizing",
        "finalizing": "finalizing",
        "finished": "finalizing",
    }
    for legacy_stage, engine_stage in expected.items():
        assert _translate_progress({
            "stage": legacy_stage, "message": "detail", "updated_at": 1.25,
        }) == {
            "stage": engine_stage, "detail": "detail", "observed_at_ms": 1250,
        }


def test_sustained_stderr_flood_keeps_the_report_and_reaps_the_child(tmp_path):
    """8 MiB of stderr must cost bounded memory, not the valid result."""

    request = _request(source=b"// STDERR_FLOOD\r\nkernel void demo() {} \r\n")
    attempt = _attempt()
    record_path = tmp_path / "flood-record.json"
    returncode, output, error, temp_root = _run_bridge(
        tmp_path, request, attempt, FAKE_WORKER,
        extra_env={
            "FAKE_LEGACY_RECORD": str(record_path),
            "FAKE_LEGACY_BEHAVIOR": "stderr_flood",
            "FAKE_LEGACY_FLOOD_BLOCKS": "128",
        },
    )
    assert returncode == 0, error.decode("utf-8", errors="replace")[-2000:]
    decoded = _decode_engine_frames(output, request, attempt)
    assert len(decoded) == 1 and decoded[0][1] == "result"
    assert decoded[0][2]["report"]["legacy_report_marker"] == "report-from-legacy-worker"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert _ps_state(record["pid"]) is None, "the flooding child was not reaped"
    assert list(temp_root.iterdir()) == []


def test_sustained_stdout_flood_fails_bounded_and_reaps_the_child(tmp_path):
    """4 MiB of non-frame stdout must fail the attempt, not buffer or leak."""

    request = _request(source=b"// STDOUT_FLOOD\r\nkernel void demo() {} \r\n")
    attempt = _attempt()
    record_path = tmp_path / "stdout-flood-record.json"
    returncode, output, error, temp_root = _run_bridge(
        tmp_path, request, attempt, FAKE_WORKER,
        extra_env={
            "FAKE_LEGACY_RECORD": str(record_path),
            "FAKE_LEGACY_BEHAVIOR": "stdout_garbage_flood",
            "FAKE_LEGACY_FLOOD_BLOCKS": "64",
        },
    )
    assert returncode == 1
    assert output == b"", "a failed bridge must never emit an engine result"
    message = error.decode("utf-8", errors="replace")
    assert "malformed frames" in message
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert _ps_state(record["pid"]) is None, "the flooding child was not reaped"
    assert list(temp_root.iterdir()) == []


def test_a_descendant_holding_the_pipe_cannot_block_the_bridge(tmp_path, monkeypatch):
    """The bounded join must bound the call even when a descendant holds stdout."""

    import io
    import threading
    import time

    import tcenv.remote.legacy_worker_bridge as bridge

    monkeypatch.setattr(bridge, "_PIPE_JOIN_S", 0.05)
    holder_file = tmp_path / "holder.pid"
    request = _request(source=b"// STREAM\r\nkernel void demo() {} \r\n")
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "exit_with_pipe_holder"
    environ["FAKE_LEGACY_HOLDER_PID"] = str(holder_file)
    environ["FAKE_LEGACY_HOLD_S"] = "5"
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    started = time.monotonic()
    try:
        with pytest.raises(bridge._BridgeFailure, match="did not finish"):
            bridge._run_legacy_worker(
                FAKE_WORKER, request, output,
                attempt_id=_attempt().attempt_id,
                lease_fence=_attempt().lease_fence,
                environ=environ,
                write_lock=threading.Lock(),
            )
    finally:
        holder = holder_file.read_text(encoding="utf-8") if holder_file.is_file() else ""
        if holder:
            try:
                os.kill(int(holder), signal.SIGKILL)
            except ProcessLookupError:
                pass
    elapsed = time.monotonic() - started
    # A blocked BufferedReader.close() would wait for the 5s holder; the bridge
    # must return after its bounded joins instead.
    assert elapsed < 2.0, elapsed
    assert output.getvalue() == b""


def test_a_request_write_epipe_with_a_success_result_is_not_accepted(tmp_path):
    """A child that never reads the request cannot have its result accepted."""

    import io
    import threading

    import tcenv.remote.legacy_worker_bridge as bridge

    request_obj = _request(source=b"x" * 900_000)
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "result_without_reading"
    environ["FAKE_LEGACY_REQUEST_ID"] = request_obj.request_id
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    with pytest.raises(bridge._BridgeFailure, match="request write failed"):
        bridge._run_legacy_worker(
            FAKE_WORKER, request_obj, output,
            attempt_id=_attempt().attempt_id,
            lease_fence=_attempt().lease_fence,
            environ=environ,
            write_lock=threading.Lock(),
        )
    assert output.getvalue() == b""


def test_a_descendant_holding_stdin_cannot_block_the_bridge(tmp_path, monkeypatch):
    """The child exits without reading a large request; its descendant holds stdin."""

    import io
    import threading
    import time

    import tcenv.remote.legacy_worker_bridge as bridge

    monkeypatch.setattr(bridge, "_PIPE_JOIN_S", 0.2)
    holder_file = tmp_path / "holder.pid"
    # Large enough to fill the pipe buffer and block the writer.
    request_obj = _request(source=b"// STREAM\r\n" + b"x" * 900_000)
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "exit_with_stdin_holder"
    environ["FAKE_LEGACY_HOLDER_PID"] = str(holder_file)
    environ["FAKE_LEGACY_HOLD_S"] = "5"
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    outcome: list[BaseException] = []

    def run() -> None:
        try:
            bridge._run_legacy_worker(
                FAKE_WORKER, request_obj, output,
                attempt_id=_attempt().attempt_id,
                lease_fence=_attempt().lease_fence,
                environ=environ,
                write_lock=threading.Lock(),
            )
        except BaseException as exc:  # noqa: BLE001 - the thread reports the failure
            outcome.append(exc)

    started = time.monotonic()
    thread = threading.Thread(target=run, name="legacy-writer-probe", daemon=True)
    thread.start()
    thread.join(5.0)
    elapsed = time.monotonic() - started
    holder = holder_file.read_text(encoding="utf-8") if holder_file.is_file() else ""
    if holder:
        try:
            os.kill(int(holder), signal.SIGKILL)
        except ProcessLookupError:
            pass
    thread.join(5.0)
    # The writer stays blocked until the holder dies; closing its buffered
    # stdin from the bridge would wait on the writer's lock instead of failing.
    assert not thread.is_alive()
    assert outcome and isinstance(outcome[0], bridge._BridgeFailure)
    assert "request writer did not finish" in str(outcome[0])
    assert elapsed < 2.0, elapsed
    assert output.getvalue() == b""


def test_a_stderr_reader_failure_fails_the_bridge(tmp_path, monkeypatch):
    """Both readers must be error-free, not only stdout."""

    import io
    import threading

    import tcenv.remote.legacy_worker_bridge as bridge

    original_run = bridge._LegacyPipe._run

    def run_or_fail(self):
        if self._decode:
            return original_run(self)
        self._fail("could not read legacy worker output: injected stderr failure")

    monkeypatch.setattr(bridge._LegacyPipe, "_run", run_or_fail)
    request_obj = _request(source=b"// STREAM\r\nkernel void demo() {} \r\n")
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "success"
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    with pytest.raises(bridge._BridgeFailure, match="injected stderr failure"):
        bridge._run_legacy_worker(
            FAKE_WORKER, request_obj, output,
            attempt_id=_attempt().attempt_id,
            lease_fence=_attempt().lease_fence,
            environ=environ,
            write_lock=threading.Lock(),
        )
    assert output.getvalue() == b""


def test_streaming_pipes_keep_their_buffers_bounded():
    import io

    from tcenv.remote.legacy_worker_bridge import _MAX_BRIDGE_STDERR_TAIL, _LegacyPipe

    stderr_pipe = _LegacyPipe(io.BytesIO(b"e" * (4 * 1024 * 1024)), decode=False)
    stderr_pipe.start()
    assert stderr_pipe.join(30)
    assert stderr_pipe.error == ""
    assert len(stderr_pipe.tail.encode("utf-8")) <= _MAX_BRIDGE_STDERR_TAIL

    stdout_pipe = _LegacyPipe(io.BytesIO(b"g" * (4 * 1024 * 1024)), decode=True)
    stdout_pipe.start()
    assert stdout_pipe.join(30)
    # The first four bytes are not a frame header, so the decoder fails at once
    # and the remaining stream is drained instead of retained.
    assert "malformed frames" in stdout_pipe.error
    assert stdout_pipe.results == []


class _ExplodingStream:
    """Yields one complete frame, then fails like a decoder stack overflow."""

    def __init__(self, first: bytes) -> None:
        self._first = first
        self._exploded = False

    def read(self, _size: int) -> bytes:
        if self._first:
            data, self._first = self._first, b""
            return data
        if not self._exploded:
            self._exploded = True
            raise RecursionError("injected decoder recursion")
        return b""


class _BlockingStream:
    """Never returns EOF until released; models a reader stuck before finish."""

    def __init__(self, release) -> None:
        self._release = release
        self._first = b""

    def read(self, _size: int) -> bytes:
        self._release.wait(10.0)
        return self._first


def test_a_reader_exception_after_a_frame_is_an_error_not_a_result():
    import io

    from tcenv.remote.legacy_worker_bridge import _LegacyPipe
    from tcenv.remote.protocol import encode_frame

    frame = encode_frame(
        "result", "bridge-request-1",
        {"report": {"status": "ok"}, "evidence": {"candidate_sha256": "a" * 64}},
    )
    pipe = _LegacyPipe(_ExplodingStream(frame), decode=True)
    pipe.start()
    assert pipe.join(10.0) is True
    # The frame was decoded, but the reader never finished cleanly: accepting it
    # would commit a partial stream, so the error must be recorded.
    assert len(pipe.results) == 1
    assert "RecursionError" in pipe.error


def test_a_reader_that_never_finishes_is_not_reported_as_complete():
    import threading

    from tcenv.remote.legacy_worker_bridge import _LegacyPipe

    release = threading.Event()
    pipe = _LegacyPipe(_BlockingStream(release), decode=True)
    pipe.start()
    assert pipe.join(0.1) is False
    release.set()
    assert pipe.join(10.0) is True
    assert pipe.error == ""
    assert pipe.results == []


def test_the_bridge_refuses_a_result_when_the_reader_did_not_finish(tmp_path, monkeypatch):
    import io
    import threading

    import tcenv.remote.legacy_worker_bridge as bridge

    original_join = bridge._LegacyPipe.join

    def unfinished_decode_reader(self, timeout=bridge._PIPE_JOIN_S):
        finished = original_join(self, timeout)
        return False if self._decode else finished

    monkeypatch.setattr(bridge._LegacyPipe, "join", unfinished_decode_reader)
    request = _request(source=b"// STREAM\r\nkernel void demo() {} \r\n")
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "success"
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    with pytest.raises(bridge._BridgeFailure, match="did not finish"):
        bridge._run_legacy_worker(
            FAKE_WORKER, request, output,
            attempt_id=_attempt().attempt_id,
            lease_fence=_attempt().lease_fence,
            environ=environ,
            write_lock=threading.Lock(),
        )
    assert output.getvalue() == b""


def test_the_bridge_streams_the_child_instead_of_buffering_with_communicate(tmp_path, monkeypatch):
    """A 3-line regression for blocker 7: ``communicate()`` is gone."""

    import io
    import threading

    import tcenv.remote.legacy_worker_bridge as bridge

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "the bridge must stream the child's pipes, not buffer them with communicate()"
        )

    monkeypatch.setattr(subprocess.Popen, "communicate", forbidden)
    request = _request(source=b"// STREAM\r\nkernel void demo() {} \r\n")
    output = io.BytesIO()
    environ = dict(os.environ)
    environ["PYTHONPATH"] = str(ROOT) + os.pathsep + environ.get("PYTHONPATH", "")
    environ["FAKE_LEGACY_BEHAVIOR"] = "success"
    environ.pop("TCENV_REMOTE_STATE_DIR", None)
    report, evidence = bridge._run_legacy_worker(
        FAKE_WORKER, request, output,
        attempt_id=_attempt().attempt_id,
        lease_fence=_attempt().lease_fence,
        environ=environ,
        write_lock=threading.Lock(),
    )
    assert report["legacy_report_marker"] == "report-from-legacy-worker"
    assert evidence["legacy_evidence_marker"] == "evidence-from-legacy-worker"
    # ``_run_legacy_worker`` returns the validated report/evidence; the engine
    # result frame is written by ``main`` and never carries the legacy bytes.
    assert output.getvalue() == b""
    assert report["legacy_report_marker"] == "report-from-legacy-worker"


def _production_request(*, cache_policy: str):
    from tcenv.remote.protocol import EvaluationRequest

    source = "kernel void demo() {}"
    return EvaluationRequest(
        kind="formal_eval",
        request_id=f"production-{cache_policy}",
        run_id="run-1",
        session_id="session-1",
        task="relu",
        dtype="fp16",
        size="small",
        target_id="sim-npu-test",
        task_fingerprint="a" * 64,
        candidate_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source=source,
        mode="quick",
        seed=3,
        reps=2,
        queue_timeout_s=2.0,
        candidate_timeout_s=2.0,
        cache_policy=cache_policy,
    )


def test_production_cache_policy_is_mapped(monkeypatch):
    """@brief reuse/refresh/disabled 映射到 eval_cache，不静默改变评测缓存语义。"""
    from tcenv.evaluate import EvalReport
    from tcenv.remote.production import ProductionBackend

    seen = []
    def fake(*args, **kwargs):
        seen.append(kwargs)
        request = args[3] if len(args) > 3 else None
        del request
        return EvalReport(task="relu", dtype="fp16", size="small", mode=kwargs.get("mode", "quick"), target_id="sim-npu-test")
    monkeypatch.setattr("tcenv.evaluate.evaluate", fake)
    ProductionBackend().evaluate(_production_request(cache_policy="reuse"))
    assert seen[-1]["eval_cache"] is True and seen[-1]["eval_cache_refresh"] is False
    ProductionBackend().evaluate(_production_request(cache_policy="disabled"))
    assert seen[-1]["eval_cache"] is False and seen[-1]["eval_cache_refresh"] is False
    ProductionBackend().evaluate(_production_request(cache_policy="refresh"))
    assert seen[-1]["eval_cache"] is True and seen[-1]["eval_cache_refresh"] is True
