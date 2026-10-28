"""Scriptable CPU-only job worker used by job-worker bridge tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tcenv.remote.protocol import EvaluationRequest, write_frame, read_frame

LONG_DETAIL = "benchmark:" + "x" * 1000 + ":END"


def _record(request: EvaluationRequest, stdin_eof: bool) -> None:
    destination = os.environ.get("FAKE_JOB_RECORD")
    if not destination:
        return
    parent = os.getppid()
    Path(destination).write_text(json.dumps({
        "pid": os.getpid(),
        "parent_pid": parent,
        "pgid": os.getpgid(0),
        "parent_pgid": os.getpgid(parent),
        "request_id": request.request_id,
        "source": request.source,
        "candidate_sha256": request.candidate_sha256,
        "stdin_eof": stdin_eof,
        "progress_path": os.environ.get("TCENV_REMOTE_PROGRESS_PATH", ""),
    }), encoding="utf-8")


def _progress(document: dict[str, Any]) -> None:
    path = os.environ.get("TCENV_REMOTE_PROGRESS_PATH")
    if not path:
        raise RuntimeError("bridge did not provide a job progress path")
    Path(path).write_text(json.dumps(document), encoding="utf-8")


def _result(request: EvaluationRequest, *, request_id: str | None = None) -> None:
    report = {
        "task": request.task,
        "dtype": request.dtype,
        "size": request.size,
        "mode": request.mode,
        "seed": request.seed,
        "status": "ok",
        "stage": "done",
        "candidate_us": 1.0,
        "speedup": 1.0,
        "elapsed_s": 0.0,
        "target_id": request.target_id,
        "protocol_version": "3",
        "simulated": True,
        "job_report_marker": "report-from-job-worker",
    }
    evidence = {
        "request_id": request.request_id,
        "candidate_sha256": request.candidate_sha256,
        "measurement_kind": "simulated",
        "job_evidence_marker": "evidence-from-job-worker",
    }
    behavior = os.environ.get("FAKE_JOB_BEHAVIOR", "success")
    if behavior == "target_mismatch":
        report["target_id"] = "other-target"
    elif behavior == "seed_mismatch":
        report["seed"] = (request.seed or 0) + 1
    elif behavior == "missing_evidence":
        evidence = None
    elif behavior == "candidate_crash":
        report["status"] = "crash"
        report["elapsed_s"] = 0.25
    write_frame(sys.stdout.buffer, "result", request_id or request.request_id, {
        "report": report,
        "evidence": evidence,
        "receipt": {"receipt_id": "job-receipt-must-not-cross-the-bridge"},
        "signature": "job-signature-must-not-cross-the-bridge",
    })


def _wait_for_release() -> bool:
    release = os.environ.get("FAKE_JOB_RELEASE")
    if not release:
        return False
    deadline = time.monotonic() + 10.0
    while not Path(release).exists():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def main() -> int:
    behavior = os.environ.get("FAKE_JOB_BEHAVIOR", "success")
    if behavior == "result_without_reading":
        # Emit a success result without reading the request, then exit: the
        # bridge's large write gets EPIPE and must not accept the result.
        request_id = os.environ.get("FAKE_JOB_REQUEST_ID", "bridge-request-1")
        write_frame(sys.stdout.buffer, "result", request_id, {
            "report": {"status": "ok", "job_report_marker": "unread"},
            "evidence": None,
        })
        sys.stdout.buffer.flush()
        return 0
    if behavior == "exit_with_stdin_holder":
        # Exit without reading the request while a descendant keeps the stdin
        # read end open: the bridge's writer blocks on a full pipe and must not
        # be "closed" from another thread.  Only stdin is inherited; stdout and
        # stderr are closed in the descendant so the readers still see EOF.
        holder = subprocess.Popen(
            [sys.executable, "-c", "import sys, time; time.sleep(float(sys.argv[1]))",
             os.environ.get("FAKE_JOB_HOLD_S", "5")],
            stdin=sys.stdin, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        destination = os.environ.get("FAKE_JOB_HOLDER_PID")
        if destination:
            Path(destination).write_text(str(holder.pid), encoding="utf-8")
        return 0
    frame = read_frame(sys.stdin.buffer)
    if frame.frame_type != "request":
        raise RuntimeError("fake job worker expected a request")
    request = EvaluationRequest.from_dict(frame.payload)
    if request.request_id != frame.request_id:
        raise RuntimeError("fake job worker request id mismatch")
    # communicate(input=...) must send exactly one frame and then close stdin.
    stdin_eof = sys.stdin.buffer.read(1) == b""
    _record(request, stdin_eof)
    behavior = os.environ.get("FAKE_JOB_BEHAVIOR", "success")

    if behavior == "nonzero":
        return 23
    if behavior == "sleep":
        time.sleep(60.0)
        return 0
    if behavior == "malformed":
        sys.stdout.buffer.write(b"not-a-job-frame")
        sys.stdout.buffer.flush()
        return 0
    if behavior == "exit_with_pipe_holder":
        # A descendant inherits stdout/stderr and keeps them open after this
        # worker exits: the bridge's readers cannot reach EOF, so the bounded
        # join must fail instead of blocking on a stream close.
        holder = subprocess.Popen([
            sys.executable, "-c",
            "import sys,time; time.sleep(float(sys.argv[1]))",
            os.environ.get("FAKE_JOB_HOLD_S", "2"),
        ])
        destination = os.environ.get("FAKE_JOB_HOLDER_PID")
        if destination:
            Path(destination).write_text(str(holder.pid), encoding="utf-8")
        _result(request)
        return 0
    if behavior == "wrong_request":
        _result(request, request_id="another-request")
        return 0
    if behavior == "multiple":
        _result(request)
        _result(request)
        return 0
    if behavior == "missing_report":
        write_frame(sys.stdout.buffer, "result", request.request_id, {"evidence": {}})
        return 0
    if behavior == "invalid_report":
        write_frame(sys.stdout.buffer, "result", request.request_id, {
            "report": [], "evidence": None,
        })
        return 0
    if behavior == "wrong_type":
        write_frame(sys.stdout.buffer, "progress", request.request_id, {"stage": "done"})
        return 0
    if behavior == "invalid_evidence":
        write_frame(sys.stdout.buffer, "result", request.request_id, {
            "report": {"status": "ok"}, "evidence": [],
        })
        return 0
    if behavior == "stderr_flood":
        # A sustained stderr stream must not be buffered by the bridge, and it
        # must not cost the attempt its valid report.
        block = b"e" * 65536
        for _ in range(int(os.environ.get("FAKE_JOB_FLOOD_BLOCKS", "128"))):
            sys.stderr.buffer.write(block)
        sys.stderr.buffer.flush()
        _result(request)
        return 0
    if behavior == "stdout_garbage_flood":
        # Unbounded stdout that is not a frame: the bridge must fail bounded and
        # still reap this child instead of buffering whatever it writes.
        block = b"g" * 65536
        for _ in range(int(os.environ.get("FAKE_JOB_FLOOD_BLOCKS", "64"))):
            sys.stdout.buffer.write(block)
        sys.stdout.buffer.flush()
        return 0

    if behavior == "progress_handshake":
        _progress({
            "stage": "bench",
            "status": "running",
            "message": LONG_DETAIL,
            "updated_at": 1234.567,
            "path_must_not_cross": os.environ.get("TCENV_REMOTE_PROGRESS_PATH", ""),
        })
        if not _wait_for_release():
            return 24
        _progress({
            "stage": "done",
            "status": "ok",
            "message": "terminal done",
            "updated_at": 5,
            "finished": True,
        })
    elif behavior == "unknown_progress":
        _progress({
            "stage": "not-a-real-stage",
            "message": "unknown job stage",
            "updated_at": "not-a-timestamp",
        })
    elif behavior not in {"success", "target_mismatch", "seed_mismatch", "missing_evidence", "candidate_crash"}:
        raise RuntimeError(f"unknown fake behavior {behavior!r}")

    _result(request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
