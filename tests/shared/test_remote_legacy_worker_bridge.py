"""Real-subprocess tests for the private JobEngine-to-legacy worker bridge."""

from __future__ import annotations

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
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "stat="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


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
