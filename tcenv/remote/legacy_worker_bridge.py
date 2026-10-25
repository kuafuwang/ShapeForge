"""Private JobEngine-worker adapter for the existing legacy evaluator worker.

The bridge itself is the JobEngine worker.  It accepts one private worker
request, starts ``TCENV_LEGACY_WORKER_COMMAND`` without a shell or a new process
group, and translates the legacy worker's one-result protocol.  In particular,
the child inherits this process's group so JobEngine timeout/cancellation kills
both processes together.

Legacy progress files are deliberately treated as untrusted, lossy telemetry.
Oversized or partially written snapshots are ignored.  Unknown stages map to
``queued`` (the safe stage which claims no evaluation advancement), while only
bounded stage/detail/timestamp fields cross the private worker boundary.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, BinaryIO, Mapping

from .job_types import ExternalEvaluationRequest
from .protocol import FrameDecoder, ProtocolError, encode_frame
from .worker_protocol import STATE_DIR_ENV, read_request, write_progress, write_result

_LEGACY_COMMAND_ENV = "TCENV_LEGACY_WORKER_COMMAND"
_PROGRESS_ENV = "TCENV_REMOTE_PROGRESS_PATH"
_BRIDGE_TMP_DIRNAME = "bridge-tmp"
_ATTEMPT_TMP_PREFIX = "attempt-"
_MAX_PROGRESS_BYTES = 65_536
_PROGRESS_POLL_S = 0.05
_MAX_DETAIL_CHARS = 300
_MAX_OBSERVED_AT_MS = 2**63 - 1

_STAGE_MAP = {
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


class _BridgeFailure(ValueError):
    """A sanitized, nonzero bridge outcome; no engine result is written."""


def _legacy_command(environ: Mapping[str, str]) -> list[str]:
    raw = environ.get(_LEGACY_COMMAND_ENV)
    if raw is None:
        raise _BridgeFailure(f"{_LEGACY_COMMAND_ENV} is missing")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _BridgeFailure(f"{_LEGACY_COMMAND_ENV} is not a JSON array") from exc
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(argument, str) or not argument for argument in value)
        or any("\x00" in argument for argument in value)
    ):
        raise _BridgeFailure(
            f"{_LEGACY_COMMAND_ENV} must be a JSON array of non-empty strings"
        )
    return value


def _prune_stale_attempt_dirs(root: Path) -> None:
    """Remove attempt temp dirs left behind by a SIGKILLed bridge.

    One frontend runs at most one attempt at a time and the engine confirms the
    previous worker group is gone before spawning the next, so every entry under
    this state-scoped root is stale when a new bridge starts.  Only our own
    prefix is removed; symlinks are unlinked, never followed.
    """

    try:
        entries = list(os.scandir(root))
    except OSError:
        return
    for entry in entries:
        if not entry.name.startswith(_ATTEMPT_TMP_PREFIX):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                shutil.rmtree(entry.path, ignore_errors=True)
            else:
                os.unlink(entry.path)
        except OSError:
            continue


def _bridge_temp_root(environ: Mapping[str, str]) -> Path | None:
    """Return the state-scoped temp root for one attempt, or ``None``.

    Residue under a process-wide temp directory cannot be attributed or cleaned
    safely; the engine passes its persistent state directory so leftovers are
    scoped to this single-active frontend.  A missing or unusable root falls back
    to the standard temp directory instead of failing the candidate.
    """

    raw = environ.get(STATE_DIR_ENV)
    if not raw:
        return None
    try:
        state_dir = Path(raw)
        if not state_dir.is_absolute():
            return None
        root = state_dir / _BRIDGE_TMP_DIRNAME
        root.mkdir(mode=0o700, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            return None
        os.chmod(root, 0o700)
    except OSError:
        return None
    _prune_stale_attempt_dirs(root)
    return root


def _translate_progress(document: Any) -> dict[str, Any] | None:
    """Project one legacy JSON object onto the bounded engine progress schema."""

    if not isinstance(document, dict):
        return None
    legacy_stage = document.get("stage")
    # Unknown (including non-text) stages deliberately make no advancement
    # claim.  Malformed telemetry must not crash an otherwise valid evaluation.
    stage = _STAGE_MAP.get(legacy_stage, "queued") if isinstance(legacy_stage, str) else "queued"
    message = document.get("message")
    progress: dict[str, Any] = {
        "stage": stage,
        "detail": message[:_MAX_DETAIL_CHARS] if isinstance(message, str) else "",
    }
    updated_at = document.get("updated_at")
    if not isinstance(updated_at, bool) and isinstance(updated_at, int):
        if 0 <= updated_at <= _MAX_OBSERVED_AT_MS // 1000:
            progress["observed_at_ms"] = updated_at * 1000
    elif not isinstance(updated_at, bool) and isinstance(updated_at, float):
        if math.isfinite(updated_at) and 0 <= updated_at <= _MAX_OBSERVED_AT_MS / 1000:
            progress["observed_at_ms"] = int(updated_at * 1000)
    return progress


class _ProgressRelay:
    """Poll a bounded legacy snapshot and serialize writes with the result."""

    def __init__(
        self,
        path: Path,
        writer: BinaryIO,
        *,
        request_id: str,
        attempt_id: str,
        lease_fence: int,
        write_lock: threading.Lock,
    ) -> None:
        self._path = path
        self._writer = writer
        self._request_id = request_id
        self._attempt_id = attempt_id
        self._lease_fence = lease_fence
        self._write_lock = write_lock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last: dict[str, Any] | None = None
        self._error: Exception | None = None

    def _snapshot(self) -> dict[str, Any] | None:
        try:
            info = self._path.stat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_PROGRESS_BYTES:
                return None
            # The second bound closes the race where a writer grows the file
            # after stat().  No legacy-controlled unbounded read is performed.
            with self._path.open("rb") as handle:
                raw = handle.read(_MAX_PROGRESS_BYTES + 1)
            if len(raw) > _MAX_PROGRESS_BYTES:
                return None
            document = json.loads(raw.decode("utf-8"))
        except (OSError, ValueError, RecursionError):
            # Missing files and transient partial writes are normal while the
            # evaluator replaces its snapshot.  A later poll/final drain retries.
            return None
        return _translate_progress(document)

    def relay_one(self) -> None:
        progress = self._snapshot()
        if progress is None or progress == self._last:
            return
        with self._write_lock:
            write_progress(
                self._writer,
                request_id=self._request_id,
                attempt_id=self._attempt_id,
                lease_fence=self._lease_fence,
                progress=progress,
            )
        self._last = progress

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.relay_one()
            except Exception as exc:  # propagated to the supervising thread
                self._error = exc
                self._stop.set()
                return
            self._stop.wait(_PROGRESS_POLL_S)

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="tcenv-legacy-worker-progress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise _BridgeFailure(f"could not write engine progress: {self._error}") from self._error


def _decode_legacy_result(output: bytes, request_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    decoder = FrameDecoder()
    try:
        frames = decoder.feed(output)
        decoder.finish()
    except ProtocolError as exc:
        raise _BridgeFailure(f"legacy worker emitted malformed frames: {exc}") from exc
    if len(frames) != 1 or frames[0].frame_type != "result":
        raise _BridgeFailure("legacy worker did not emit exactly one result frame")
    frame = frames[0]
    if frame.request_id != request_id:
        raise _BridgeFailure("legacy worker result request_id does not match the request")
    report = frame.payload.get("report")
    if not isinstance(report, dict):
        raise _BridgeFailure("legacy worker result has no valid report object")
    evidence = frame.payload.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise _BridgeFailure("legacy worker evidence must be an object or null")
    # Legacy receipts/signatures intentionally stop here.  JobEngine binds and
    # signs its own attempt-scoped external receipt from report/evidence only.
    return report, evidence


def _close_process(process: subprocess.Popen) -> None:
    """Kill an unfinished direct child, reap it, and close every pipe."""

    try:
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        # A successful communicate() already waited.  After an exceptional
        # path this wait is mandatory: a bridge failure must not leave a zombie.
        process.wait()
    finally:
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass


def _run_legacy_worker(
    command: list[str],
    request: ExternalEvaluationRequest,
    writer: BinaryIO,
    *,
    attempt_id: str,
    lease_fence: int,
    environ: Mapping[str, str],
    write_lock: threading.Lock,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    legacy = request.to_legacy()
    legacy_request = encode_frame("request", legacy.request_id, legacy.to_dict())
    temp_root = _bridge_temp_root(environ)
    with tempfile.TemporaryDirectory(prefix=_ATTEMPT_TMP_PREFIX, dir=temp_root) as temporary:
        progress_path = Path(temporary) / "progress.json"
        child_env = dict(environ)
        child_env[_PROGRESS_ENV] = str(progress_path)
        relay = _ProgressRelay(
            progress_path,
            writer,
            request_id=request.request_id,
            attempt_id=attempt_id,
            lease_fence=lease_fence,
            write_lock=write_lock,
        )
        process: subprocess.Popen | None = None
        try:
            try:
                # No start_new_session/popen_group: this child must remain in
                # the group JobEngine kills on timeout or cancellation.
                process = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=child_env,
                    shell=False,
                )
            except OSError as exc:
                raise _BridgeFailure(f"could not start legacy worker: {exc}") from exc
            relay.start()
            output, _stderr = process.communicate(input=legacy_request)
            # communicate() has reaped the child and reached stdout EOF.  Stop
            # asynchronous polling, but retain the file for one terminal drain.
            relay.stop()
            relay.raise_if_failed()
            if process.returncode != 0:
                raise _BridgeFailure(f"legacy worker exited {process.returncode}")
            report, evidence = _decode_legacy_result(output or b"", request.request_id)
            # The terminal write often happens immediately before child exit.
            # Drain it only after the result is known valid, and always before
            # the engine result frame is written by main().
            relay.relay_one()
            return report, evidence
        finally:
            relay.stop()
            if process is not None:
                _close_process(process)


def main(argv: list[str] | None = None) -> int:
    """Run one bridge attempt; import alone never starts a worker."""

    arguments = sys.argv[1:] if argv is None else list(argv)
    if arguments:
        print("legacy worker bridge accepts no command-line arguments", file=sys.stderr, flush=True)
        return 2
    write_lock = threading.Lock()
    try:
        attempt_id, request, request_id, lease_fence = read_request(sys.stdin.buffer)
        command = _legacy_command(os.environ)
        report, evidence = _run_legacy_worker(
            command,
            request,
            sys.stdout.buffer,
            attempt_id=attempt_id,
            lease_fence=lease_fence,
            environ=os.environ,
            write_lock=write_lock,
        )
        with write_lock:
            write_result(
                sys.stdout.buffer,
                request_id=request_id,
                attempt_id=attempt_id,
                lease_fence=lease_fence,
                report=report,
                evidence=evidence,
            )
        return 0
    except Exception as exc:  # a worker/protocol failure is a nonzero attempt, never a result
        detail = str(exc) or type(exc).__name__
        print(f"legacy worker bridge failed: {detail[:1000]}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
