"""Host-controlled self-check bridge for one kernel session.

The agent may ask for ``quick`` or ``timed`` self-checks, but it does not choose
the task, target, repetitions, quota or evaluator. The host owns those values,
receives candidate source over a per-session Unix socket, and calls the same
:class:`tcenv.evaluation.EvaluationExecutor` used for formal evaluation.

A socket is a transport, not the whole authorization boundary. Every request
also carries the session token from a host-provisioned token file. The server
keeps its own quota/hang usage so local self-checks cannot bypass the limits by
omitting ``self_check_dir``.
"""

from __future__ import annotations

import argparse
import hmac
import json
import math
import os
import socket
import stat
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tcenv.evaluate import EvalReport
from tcenv.evaluation import EvaluationError, EvaluationExecutor, EvaluationRequest, EvaluationResult, EvaluationUnavailable
from tcenv.report import format_feedback
from tcenv.schema import write_json_atomic

MAX_MESSAGE_BYTES = 1_100_000
MAX_SOURCE_BYTES = 1_000_000


class SelfCheckBridgeError(RuntimeError):
    """A self-check request was rejected before or during evaluation."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = bool(retryable)


@dataclass(frozen=True)
class SelfCheckConfig:
    """The host-owned limits and credentials for one session's bridge."""

    task: str
    dtype: str
    size: str
    target_id: str
    task_fingerprint: str
    run_id: str
    session_id: str
    token: str
    token_file: Path | None = None
    reps: int = 15
    cases: tuple[str, ...] | None = None
    self_check_dir: Path | None = None
    usage_path: Path | None = None
    quick_timeout_s: float = 420.0
    timed_timeout_s: float = 420.0
    max_timed: int = 20
    max_gpu_s: float = 120.0
    allowed_modes: frozenset[str] = frozenset({"quick", "timed"})
    max_source_bytes: int = MAX_SOURCE_BYTES
    close_evaluator_on_stop: bool = False

    def __post_init__(self) -> None:
        for name in ("task", "dtype", "size", "target_id", "run_id", "session_id", "token"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"selfcheck config {name} must be non-empty")
        if not isinstance(self.task_fingerprint, str) or len(self.task_fingerprint) != 64:
            raise ValueError("selfcheck config task_fingerprint must be 64 hex characters")
        if isinstance(self.reps, bool) or not isinstance(self.reps, int) or not 1 <= self.reps <= 10_000:
            raise ValueError("selfcheck reps must be positive")
        if isinstance(self.max_timed, bool) or not isinstance(self.max_timed, int) or self.max_timed < 1:
            raise ValueError("selfcheck max_timed must be a positive integer")
        if isinstance(self.max_gpu_s, bool) or not isinstance(self.max_gpu_s, (int, float)) or not math.isfinite(self.max_gpu_s) or self.max_gpu_s <= 0:
            raise ValueError("selfcheck max_gpu_s must be finite and positive")
        for name in ("quick_timeout_s", "timed_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
                raise ValueError(f"selfcheck {name} must be finite and positive")
        if not self.allowed_modes or not self.allowed_modes <= {"quick", "timed"}:
            raise ValueError("selfcheck modes must be quick and/or timed")
        if self.cases is not None and (not isinstance(self.cases, tuple) or any(not isinstance(case, str) or not case for case in self.cases)):
            raise ValueError("selfcheck cases must be a tuple of non-empty strings")
        if not isinstance(self.max_source_bytes, int) or isinstance(self.max_source_bytes, bool) or self.max_source_bytes <= 0:
            raise ValueError("selfcheck max_source_bytes must be positive")

    def request(self, source: str, mode: str) -> EvaluationRequest:
        if mode not in self.allowed_modes:
            raise SelfCheckBridgeError("mode_not_allowed", f"selfcheck mode {mode!r} is not allowed")
        if not isinstance(source, str):
            raise SelfCheckBridgeError("invalid_source", "source must be text")
        if len(source.encode("utf-8")) > self.max_source_bytes:
            raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
        return EvaluationRequest(
            task=self.task,
            dtype=self.dtype,
            size=self.size,
            source=source,
            kind="selfcheck",
            mode=mode,
            reps=self.reps,
            cases=self.cases,
            seed=None,
            self_check_dir=self.self_check_dir,
            target_id=self.target_id,
            run_id=self.run_id,
            session_id=self.session_id,
            task_fingerprint=self.task_fingerprint,
            candidate_timeout_s=self.timed_timeout_s if mode == "timed" else self.quick_timeout_s,
            queue_timeout_s=30.0,
            cache_policy="disabled",
            eval_cache=False,
        )


@dataclass
class SelfCheckServer:
    """Serve one session's self-check socket in a background thread."""

    config: SelfCheckConfig
    evaluator: EvaluationExecutor
    socket_path: Path
    _socket: socket.socket | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _connections: set[socket.socket] = field(default_factory=set, init=False, repr=False)
    _connections_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _active_requests: set[str] = field(default_factory=set, init=False, repr=False)
    _usage: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _usage_path: Path = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.socket_path = Path(self.socket_path)
        if not self.socket_path.name:
            raise ValueError("selfcheck socket path needs a filename")
        if self.config.usage_path is not None:
            self._usage_path = Path(self.config.usage_path)
            if self.config.self_check_dir is not None:
                try:
                    inside = self._usage_path.resolve().is_relative_to(Path(self.config.self_check_dir).resolve())
                except OSError:
                    inside = False
                if inside:
                    raise ValueError("selfcheck usage_path must not be inside the agent-writable self_check_dir")
        else:
            # Host-authoritative quota records must not live in the workspace the
            # agent can edit. The socket root is host-owned by construction.
            self._usage_path = self.socket_path.with_name(self.socket_path.name + ".usage.json")

    def start(self) -> Path:
        """Load usage, bind the per-session socket, set 0600, and start accepting."""
        if self._socket is not None:
            return self.socket_path
        self._load_usage()
        if self.config.token_file is not None:
            token_path = Path(self.config.token_file)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            fd = os.open(token_path, flags, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(self.config.token)
            os.chmod(token_path, 0o600)
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            self.socket_path.unlink()
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old_umask = os.umask(0o077)
        try:
            sock.bind(str(self.socket_path))
        except BaseException:
            sock.close()
            if self.config.token_file is not None:
                try:
                    Path(self.config.token_file).unlink()
                except OSError:
                    pass
            raise
        finally:
            os.umask(old_umask)
        os.chmod(self.socket_path, stat.S_IRUSR | stat.S_IWUSR)
        sock.listen(4)
        sock.settimeout(0.2)
        self._socket = sock
        self._thread = threading.Thread(target=self._serve, name="taic-selfcheck", daemon=True)
        self._thread.start()
        return self.socket_path

    def _load_usage(self) -> None:
        if not self._usage_path.is_file():
            self._usage = {"count": 0, "gpu_s": 0.0, "hang": False}
            return
        try:
            document = json.loads(self._usage_path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("selfcheck usage is not an object")
            count = document.get("count", 0)
            gpu_s = document.get("gpu_s", 0.0)
            hang = document.get("hang", False)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("invalid selfcheck count")
            if isinstance(gpu_s, bool) or not isinstance(gpu_s, (int, float)) or not math.isfinite(gpu_s) or gpu_s < 0:
                raise ValueError("invalid selfcheck GPU time")
            if not isinstance(hang, bool):
                raise ValueError("invalid selfcheck hang flag")
            self._usage = {"count": count, "gpu_s": float(gpu_s), "hang": hang}
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read selfcheck bridge usage: {exc}") from exc

    def _save_usage(self) -> None:
        self._usage_path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._usage_path, self._usage)

    def _quota_error(self, mode: str) -> SelfCheckBridgeError | None:
        if mode != "timed":
            return None
        if self._usage.get("hang") is True:
            return SelfCheckBridgeError("selfcheck_hang", "a previous self-check timed out or crashed")
        if self._usage.get("count", 0) >= self.config.max_timed:
            return SelfCheckBridgeError("selfcheck_quota", "self-check count quota is exhausted")
        if self._usage.get("gpu_s", 0.0) >= self.config.max_gpu_s:
            return SelfCheckBridgeError("selfcheck_quota", "self-check GPU-time quota is exhausted")
        return None

    def _record_usage(self, request: EvaluationRequest, result: EvaluationResult) -> None:
        if request.mode != "timed":
            return
        self._usage["count"] = int(self._usage.get("count", 0)) + 1
        report = result.report if isinstance(result.report, dict) else result.report.to_json()
        elapsed = report.get("elapsed_s", 0.0)
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            self._usage["gpu_s"] = float(self._usage.get("gpu_s", 0.0)) + float(elapsed)
        if report.get("status") in {"timeout", "crash"}:
            self._usage["hang"] = True
        self._save_usage()

    def _serve(self) -> None:
        while not self._stop.is_set():
            sock = self._socket
            if sock is None:
                break
            try:
                conn, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with self._connections_lock:
                self._connections.add(conn)
            try:
                # Poll briefly so stop() can interrupt an idle client even on
                # platforms where closing a socket in another thread does not wake recv().
                conn.settimeout(0.2)
                self._handle(conn)
            except Exception as exc:  # noqa: BLE001 - one client must not kill the accept loop
                try:
                    _write_document(conn, {"ok": False, "error": {"code": "evaluation_unavailable", "message": str(exc)[:500], "retryable": True}})
                except OSError:
                    pass
            finally:
                with self._connections_lock:
                    self._connections.discard(conn)
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle(self, conn: socket.socket) -> None:
        try:
            raw = _read_line(conn, MAX_MESSAGE_BYTES, stop_event=self._stop)
            document = json.loads(raw.decode("utf-8"))
            if not isinstance(document, dict) or set(document) != {"mode", "source", "token"}:
                raise SelfCheckBridgeError("invalid_request", "request fields must be exactly mode, source and token")
            token = document.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token, self.config.token):
                raise SelfCheckBridgeError("unauthorized", "selfcheck session token was refused")
            request = self.config.request(document["source"], document["mode"])
            quota_error = self._quota_error(request.mode)
            if quota_error is not None:
                raise quota_error
            self._active_requests.add(request.request_id)
            try:
                result = self.evaluator.selfcheck(request)
            except EvaluationUnavailable as exc:
                raise SelfCheckBridgeError(exc.code, str(exc), retryable=exc.retryable) from exc
            except Exception as exc:  # noqa: BLE001 - infrastructure failure is a structured bridge error
                raise SelfCheckBridgeError("evaluation_unavailable", str(exc)[:500], retryable=True) from exc
            finally:
                self._active_requests.discard(request.request_id)
            if not isinstance(result, EvaluationResult):
                raise SelfCheckBridgeError("evaluation_unavailable", "evaluator returned an invalid result", retryable=True)
            self._record_usage(request, result)
            _write_document(conn, {"ok": True, "result": result.to_dict()})
        except SelfCheckBridgeError as exc:
            _write_document(conn, {"ok": False, "error": {"code": exc.code, "message": str(exc)[:1000], "retryable": exc.retryable}})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            try:
                _write_document(conn, {"ok": False, "error": {"code": "invalid_request", "message": str(exc)[:500], "retryable": False}})
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        sock = self._socket
        self._socket = None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        with self._connections_lock:
            connections = list(self._connections)
            self._connections.clear()
        cancel = getattr(self.evaluator, "cancel", None)
        if callable(cancel):
            for request_id in list(self._active_requests):
                try:
                    cancel(request_id)
                except Exception:
                    pass
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        thread = self._thread
        self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2)
        if self.config.close_evaluator_on_stop:
            close = getattr(self.evaluator, "close", None)
            if callable(close):
                close()
        try:
            self.socket_path.unlink()
        except OSError:
            pass
        if self.config.token_file is not None:
            try:
                Path(self.config.token_file).unlink()
            except OSError:
                pass

    def __enter__(self) -> "SelfCheckServer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()


class SelfCheckClient:
    """The narrow client used by ``taic-selfcheck`` inside a session workspace."""

    def __init__(self, socket_path: Path | str, *, token: str = "", token_file: Path | str = "", timeout_s: float = 480.0) -> None:
        self.socket_path = str(socket_path)
        self.token = token
        self.token_file = str(token_file) if token_file else ""
        self.timeout_s = float(timeout_s)

    def _token(self) -> str:
        if self.token:
            return self.token
        if self.token_file:
            try:
                token = Path(self.token_file).read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise SelfCheckBridgeError("unauthorized", f"cannot read selfcheck token file: {exc}") from exc
            if token:
                return token
        raise SelfCheckBridgeError("unauthorized", "a selfcheck session token is required")

    def check(self, source: str, mode: str) -> EvaluationResult:
        if mode not in {"quick", "timed"}:
            raise ValueError("mode must be quick or timed")
        if not isinstance(source, str):
            raise TypeError("source must be text")
        payload = json.dumps({"mode": mode, "source": source, "token": self._token()}, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_s)
                sock.connect(self.socket_path)
                sock.sendall(payload + b"\n")
                raw = _read_line(sock, MAX_MESSAGE_BYTES)
        except (OSError, socket.timeout) as exc:
            raise SelfCheckBridgeError("evaluation_unavailable", f"selfcheck bridge unavailable: {exc}", retryable=True) from exc
        document = json.loads(raw.decode("utf-8"))
        if not isinstance(document, dict):
            raise SelfCheckBridgeError("invalid_response", "bridge response must be an object")
        if document.get("ok") is not True:
            error = document.get("error") if isinstance(document.get("error"), dict) else {}
            raise SelfCheckBridgeError(str(error.get("code") or "selfcheck_failed"), str(error.get("message") or "selfcheck failed"), retryable=error.get("retryable") is True)
        try:
            return EvaluationResult.from_dict(document["result"])
        except (EvaluationError, KeyError, TypeError) as exc:
            raise SelfCheckBridgeError("invalid_response", f"bridge returned an invalid result: {exc}") from exc


def _read_line(stream: socket.socket, limit: int, *, stop_event: threading.Event | None = None) -> bytes:
    chunks = bytearray()
    while True:
        if len(chunks) > limit:
            raise SelfCheckBridgeError("message_too_large", "selfcheck message exceeds the limit")
        try:
            chunk = stream.recv(min(65_536, limit + 1 - len(chunks)))
        except socket.timeout:
            if stop_event is not None and stop_event.is_set():
                raise SelfCheckBridgeError("connection_closed", "selfcheck bridge is stopping")
            continue
        if not chunk:
            raise SelfCheckBridgeError("connection_closed", "selfcheck bridge closed before a response")
        chunks.extend(chunk)
        if b"\n" in chunk:
            line, _, rest = chunks.partition(b"\n")
            if rest:
                raise SelfCheckBridgeError("invalid_request", "extra bytes after the request")
            return bytes(line)


def _write_document(stream: socket.socket, document: dict) -> None:
    stream.sendall(json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")


def _workspace_kernel(value: str, workspace: str = "") -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("kernel path must be relative to the session workspace")
    root = Path(workspace or Path.cwd()).resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("kernel path escapes the session workspace")
    return resolved


def main(argv: list[str] | None = None) -> int:
    """Run one self-check; task/target/reps stay host-owned."""
    parser = argparse.ArgumentParser(prog="taic-selfcheck")
    parser.add_argument("--socket", default=os.environ.get("TAIC_SELFCHECK_SOCKET", ""))
    parser.add_argument("--token-file", default=os.environ.get("TAIC_SELFCHECK_TOKEN_FILE", ""))
    parser.add_argument("--workspace", default="")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--json", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_const", const="quick", dest="mode")
    group.add_argument("--timed", action="store_const", const="timed", dest="mode")
    args = parser.parse_args(argv)
    if not args.socket:
        print("TAIC_SELFCHECK_SOCKET or --socket is required", file=sys.stderr)
        return 2
    try:
        source = _workspace_kernel(args.kernel, args.workspace).read_text(encoding="utf-8")
        result = SelfCheckClient(
            args.socket,
            token=os.environ.get("TAIC_SELFCHECK_TOKEN", ""),
            token_file=args.token_file,
        ).check(source, args.mode)
        report = result.report if isinstance(result.report, EvalReport) else result.report
        if isinstance(report, dict) and report.get("simulated") is True:
            if args.json:
                print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(f"simulated selfcheck: {report.get('status')} {report.get('message', '')}")
            return 0 if report.get("status") == "ok" else 1
        if not isinstance(report, EvalReport):
            report = EvalReport.from_json(report)
    except (OSError, ValueError, SelfCheckBridgeError, EvaluationError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_feedback(report))
    return 0 if report.status == "ok" else 1


__all__ = [
    "SelfCheckBridgeError",
    "SelfCheckClient",
    "SelfCheckConfig",
    "SelfCheckServer",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
