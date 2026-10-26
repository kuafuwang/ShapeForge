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
# The legacy child's pipes are read as a stream: a sustained flood costs a
# fixed-size chunk buffer plus one result frame, never the whole stream.
_BRIDGE_CHUNK = 65_536
_MAX_BRIDGE_STDOUT_BYTES = 16_000_000
_MAX_BRIDGE_STDERR_TAIL = 8_192
_PIPE_JOIN_S = 10.0

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


def _decode_legacy_result_frame(frame: Any, request_id: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate the one result frame a legacy worker may return."""

    if frame.frame_type != "result":
        raise _BridgeFailure("legacy worker did not emit exactly one result frame")
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


class _LegacyPipe:
    """One bounded streaming reader for a legacy child pipe.

    ``decode`` marks the stdout stream: frames are decoded as they arrive and at
    most one result frame is retained.  Once a limit or a protocol error is hit
    the reader keeps *draining* (without buffering) so a flooding child is never
    blocked on a full pipe, which would turn a bounded failure into a hang.  The
    stderr stream only keeps a fixed-size tail and never fails the attempt.
    """

    def __init__(self, stream: BinaryIO, *, decode: bool) -> None:
        self._stream = stream
        self._decode = decode
        self._decoder = FrameDecoder() if decode else None
        self._results: list[Any] = []
        self._error = ""
        self._bytes = 0
        self._tail = bytearray()
        self._thread = threading.Thread(
            target=self._run, name="tcenv-legacy-worker-pipe", daemon=True,
        )

    @property
    def results(self) -> list[Any]:
        return self._results

    @property
    def error(self) -> str:
        return self._error

    @property
    def tail(self) -> str:
        return self._tail.decode("utf-8", errors="replace")

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> bool:
        # Resolve the bound at call time so the pipeline limit stays a single
        # module constant (and a short bounded join can be exercised for real).
        self._thread.join(timeout=_PIPE_JOIN_S if timeout is None else timeout)
        return not self._thread.is_alive()

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _fail(self, message: str) -> None:
        if not self._error:
            self._error = message

    def _run(self) -> None:
        # The reader thread must never die with an uncaught exception: a partial
        # frame sequence would then look like a complete, successful read.  Any
        # failure (including a decoder RecursionError) becomes a bounded error
        # that the caller checks before accepting a result.
        try:
            while True:
                try:
                    chunk = self._stream.read(_BRIDGE_CHUNK)
                except (OSError, ValueError) as exc:
                    self._fail(f"could not read legacy worker output: {exc}")
                    return
                if not chunk:
                    break
                self._consume(chunk)
            if self._decoder is not None and not self._error:
                try:
                    self._decoder.finish()
                except ProtocolError as exc:
                    self._fail(f"legacy worker emitted malformed frames: {exc}")
        except BaseException as exc:  # noqa: BLE001 - the thread boundary must not swallow
            self._fail(
                "legacy worker output reader failed: "
                f"{type(exc).__name__}: {exc}"[:1000]
            )

    def _consume(self, chunk: bytes) -> None:
        self._bytes += len(chunk)
        if self._decode and not self._error and self._bytes > _MAX_BRIDGE_STDOUT_BYTES:
            self._fail("legacy worker output exceeded the bridge limit")
        if self._decoder is None:
            self._tail = (self._tail + chunk)[-_MAX_BRIDGE_STDERR_TAIL:]
            return
        if self._error:
            return  # drain and discard, never buffer a flood
        try:
            frames = self._decoder.feed(chunk)
        except ProtocolError as exc:
            self._fail(f"legacy worker emitted malformed frames: {exc}")
            return
        for frame in frames:
            if frame.frame_type != "result":
                self._fail("legacy worker did not emit exactly one result frame")
                return
            self._results.append(frame)
            if len(self._results) > 1:
                self._fail("legacy worker did not emit exactly one result frame")
                return


def _close_process(
    process: subprocess.Popen, *, close_reads: bool = True, close_writes: bool = True,
) -> None:
    """Close the pipes of an already reaped direct child.

    ``BufferedReader.close`` waits for the buffer lock a reader thread holds
    while blocked in ``read``.  A descendant that inherited stdout/stderr can
    keep that read blocked long after this child exited, so the read sides may
    only be closed once their readers proved they finished.
    """

    streams = []
    if close_writes:
        streams.append(process.stdin)
    if close_reads:
        streams += [process.stdout, process.stderr]
    for stream in streams:
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
        stdout_pipe: _LegacyPipe | None = None
        stderr_pipe: _LegacyPipe | None = None
        # Only proven-done writers may have their stdin closed.
        writer_done = False
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
            assert process.stdin is not None and process.stdout is not None and process.stderr is not None
            relay.start()
            stdout_pipe = _LegacyPipe(process.stdout, decode=True)
            stderr_pipe = _LegacyPipe(process.stderr, decode=False)
            stdout_pipe.start()
            stderr_pipe.start()

            def write_request() -> None:
                try:
                    process.stdin.write(legacy_request)
                    process.stdin.flush()
                    process.stdin.close()
                except (BrokenPipeError, OSError, ValueError):
                    # A child that died before reading stdin is reported by its
                    # exit status, exactly as communicate() reported it.
                    pass

            writer = threading.Thread(
                target=write_request, name="tcenv-legacy-worker-stdin", daemon=True,
            )
            writer.start()
            # Reap the child first; the pipes then reach EOF and every reader
            # finishes.  A child that ignores its request is bounded by the
            # engine's candidate deadline, which kills the whole process group.
            process.wait()
            # A reader that did not finish before the bounded join is *not* a
            # completed stream: accepting the frames it happened to retain
            # would commit a partial, unvalidated result.
            stdout_done = stdout_pipe.join()
            stderr_done = stderr_pipe.join()
            writer.join(timeout=_PIPE_JOIN_S)
            # A writer still blocked in ``write`` means the request was never
            # fully delivered (a descendant can hold the stdin read end after
            # this child exits).  Closing its buffered stdin would wait on the
            # writer's lock, so the post-conditions below fail instead.
            writer_done = not writer.is_alive()
            # The readers reached EOF: the child is reaped.  Stop asynchronous
            # polling, but retain the file for one terminal drain.
            relay.stop()
            relay.raise_if_failed()
        finally:
            relay.stop()
            if process is not None:
                try:
                    if process.poll() is None:
                        try:
                            process.kill()
                        except ProcessLookupError:
                            pass
                    process.wait()
                finally:
                    # Kill/reap before joining readers and closing pipes, so a
                    # failed attempt still ends with a reaped child and EOF
                    # rather than a reader blocked on a closed stream.
                    for pipe in (stdout_pipe, stderr_pipe):
                        if pipe is not None:
                            pipe.join(timeout=_PIPE_JOIN_S)
                    readers_done = all(
                        pipe is None or not pipe.is_alive()
                        for pipe in (stdout_pipe, stderr_pipe)
                    )
                    # A descendant can inherit the pipe and keep a reader
                    # blocked after this child was reaped; closing its read
                    # stream here would wait on the reader's buffer lock and
                    # defeat the bounded join.
                    _close_process(
                        process, close_reads=readers_done, close_writes=writer_done,
                    )
        # Post-conditions are evaluated only after the child was reaped, every
        # pipe was closed and the readers were joined: a reader exception or a
        # stream that never reached EOF must fail instead of yielding the one
        # frame it managed to decode.
        assert process is not None and stdout_pipe is not None
        if not writer_done:
            raise _BridgeFailure(
                "legacy worker request writer did not finish before the bounded join deadline"
            )
        if process.returncode != 0:
            raise _BridgeFailure(f"legacy worker exited {process.returncode}")
        if not stdout_done:
            raise _BridgeFailure(
                "legacy worker output reader did not finish before the bounded join deadline"
            )
        if not stderr_done:
            raise _BridgeFailure(
                "legacy worker stderr reader did not finish before the bounded join deadline"
            )
        if stdout_pipe.error:
            raise _BridgeFailure(stdout_pipe.error)
        if len(stdout_pipe.results) != 1:
            raise _BridgeFailure("legacy worker did not emit exactly one result frame")
        report, evidence = _decode_legacy_result_frame(
            stdout_pipe.results[0], request.request_id,
        )
        # The terminal write often happens immediately before child exit.
        # Drain it only after the result is known valid, and always before the
        # engine result frame is written by main().
        relay.relay_one()
        return report, evidence


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
