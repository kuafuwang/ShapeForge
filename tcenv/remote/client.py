"""Remote evaluator client.

The client owns the host side of the ``tcenv.remote`` frame protocol.  It is a
small, deliberately non-general transport wrapper: a command/SSH process or an
in-memory stream can carry the same bytes.  The service remains the authority for
queueing, isolation and measurement; the client only pins the evaluator identity,
sends one request at a time and validates the returned envelope.
"""

from __future__ import annotations

import os
import queue
import secrets
import subprocess
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import BinaryIO, Callable, Protocol

from tcenv.children import forget, kill_group, popen_group
from tcenv.evaluation import (
    EvaluationError,
    EvaluationRequest,
    EvaluationResult,
    EvaluationUnavailable,
    EvaluatorProvenance,
    validate_receipt,
    verify_result_signature,
)
from tcenv.evaluate import EvalReport

from .protocol import (
    PROTOCOL_VERSION,
    Frame,
    HelloRequest,
    ProtocolError,
    read_frame,
    write_frame,
)

MAX_STDERR = 8_000


class RemoteError(EvaluationUnavailable):
    """A remote request failed; ``retryable`` is the service's classification."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, request_id: str = "") -> None:
        super().__init__(code, message, retryable=retryable, request_id=request_id)


class Transport(Protocol):
    """A connected bidirectional byte stream."""

    reader: BinaryIO
    writer: BinaryIO

    def close(self) -> None: ...


class CommandTransport:
    """A child process carrying the remote frame stream on stdin/stdout."""

    def __init__(
        self,
        command: list[str],
        *,
        cwd: Path | str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("transport command must be a non-empty argv list")
        self.command = list(command)
        self.cwd = str(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.reader: BinaryIO = None  # type: ignore[assignment]
        self.writer: BinaryIO = None  # type: ignore[assignment]
        self._proc: subprocess.Popen | None = None
        self._stderr_tail = ""

    def __enter__(self) -> "CommandTransport":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None:
            return
        env = os.environ.copy()
        if self.env:
            env.update(self.env)
        try:
            proc = popen_group(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
                env=env,
            )
        except OSError as exc:
            raise RemoteError("transport_unavailable", f"could not start transport: {exc}") from exc
        assert proc.stdin is not None and proc.stdout is not None
        self._proc = proc
        self.reader = proc.stdout
        self.writer = proc.stdin

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except OSError:
            pass
        try:
            kill_group(proc)
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
            if proc.stderr is not None:
                self._stderr_tail = proc.stderr.read().decode("utf-8", errors="replace")[-MAX_STDERR:]
        except (OSError, subprocess.TimeoutExpired, ValueError):
            try:
                kill_group(proc)
            except OSError:
                pass
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except (OSError, AttributeError):
                pass
        forget(proc)
        self.reader = None  # type: ignore[assignment]
        self.writer = None  # type: ignore[assignment]

    @property
    def stderr_tail(self) -> str:
        return self._stderr_tail


class _StreamTransport:
    """Test/embedded transport over caller-owned binary streams."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO, *, on_close=None) -> None:
        self.reader = reader
        self.writer = writer
        self._on_close = on_close

    def close(self) -> None:
        if self._on_close is not None:
            self._on_close()
            self._on_close = None


class RemoteEvaluator:
    """An ``EvaluationExecutor`` backed by one authenticated remote connection.

    The evaluator pins the server's epoch at hello.  Every result must carry the
    same epoch, request digest, target and candidate digest; an unexpected change
    is an infrastructure failure, not a candidate result.
    """

    def __init__(
        self,
        transport: Transport,
        *,
        token: str,
        run_id: str,
        session_id: str,
        expected_site_id: str = "",
        expected_worker_build: str = "",
        expected_task_pack_digest: str = "",
        target_id: str = "apple-m4-max",
        task_fingerprint: str = "",
        timeout_s: float | None = 480.0,
        signature_key: bytes | None = None,
    ) -> None:
        if not isinstance(token, str) or not token:
            raise ValueError("remote token is required")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("remote run_id is required")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("remote session_id is required")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("remote target_id is required")
        if timeout_s is not None and (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not 0 < timeout_s <= 86_400):
            raise ValueError("remote timeout_s must be positive and finite")
        if signature_key is not None and (not isinstance(signature_key, bytes) or not signature_key):
            raise ValueError("signature_key must be non-empty bytes or None")
        self.transport = transport
        self.token = token
        self.run_id = run_id
        self.session_id = session_id
        self.expected_site_id = expected_site_id
        self.expected_worker_build = expected_worker_build
        self.expected_task_pack_digest = expected_task_pack_digest
        self.target_id = target_id
        self.task_fingerprint = task_fingerprint
        self.timeout_s = None if timeout_s is None else float(timeout_s)
        self.signature_key = signature_key
        self._receipts: dict[str, str] = {}
        self._provenance: EvaluatorProvenance | None = None
        self._server_nonce = ""
        self._connected = False
        self._closed = False
        self._active_request_id = ""
        self.last_progress: Frame | None = None

    @property
    def provenance(self) -> EvaluatorProvenance | None:
        return self._provenance

    @property
    def epoch(self) -> str:
        return "" if self._provenance is None else self._provenance.epoch

    def ensure_epoch(self, request: EvaluationRequest | None = None) -> str:
        """Connect without submitting work so outbox replay can check the epoch."""
        del request
        return self.connect().epoch

    def __enter__(self) -> "RemoteEvaluator":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._active_request_id and self._connected:
            try:
                self._send("cancel", self._active_request_id, {"request_id": self._active_request_id})
            except RemoteError:
                pass
        self._connected = False
        self.transport.close()

    def cancel(self, request_id: str = "") -> None:
        """Ask the service to cancel the active (or named) request."""
        request_id = request_id or self._active_request_id
        if not request_id:
            return
        self._send("cancel", request_id, {"request_id": request_id})

    def _read_frame(self) -> Frame:
        if self.timeout_s is None:
            try:
                return read_frame(self.transport.reader)
            except ProtocolError as exc:
                raise RemoteError("transport_protocol_error", str(exc)) from exc
            except (OSError, EOFError) as exc:
                raise RemoteError("transport_closed", f"remote stream closed: {exc}") from exc
        # A pipe read cannot be interrupted by a socket-style timeout. Use a
        # daemon reader and close the transport on deadline, which also kills the
        # child process group for CommandTransport.
        result: queue.Queue = queue.Queue(maxsize=1)

        def read_one() -> None:
            try:
                result.put(("frame", read_frame(self.transport.reader)))
            except BaseException as exc:  # noqa: BLE001 - propagated below in the caller thread
                result.put(("error", exc))

        thread = threading.Thread(target=read_one, name="tcenv-remote-read", daemon=True)
        thread.start()
        try:
            kind, value = result.get(timeout=self.timeout_s)
        except queue.Empty as exc:
            self.close()
            raise RemoteError("transport_timeout", f"remote evaluator did not answer within {self.timeout_s:.0f}s", retryable=True) from exc
        if kind == "error":
            if isinstance(value, ProtocolError):
                raise RemoteError("transport_protocol_error", str(value)) from value
            if isinstance(value, (OSError, EOFError)):
                raise RemoteError("transport_closed", f"remote stream closed: {value}") from value
            raise value
        return value

    def _send(self, frame_type: str, request_id: str, payload: dict) -> None:
        try:
            write_frame(self.transport.writer, frame_type, request_id, payload)
        except (OSError, ProtocolError) as exc:
            raise RemoteError("transport_write_failed", f"could not send {frame_type}: {exc}") from exc

    def connect(self) -> EvaluatorProvenance:
        """Connect and always clean up a failed handshake."""
        try:
            return self._connect()
        except Exception:
            self.close()
            raise

    def _connect(self) -> EvaluatorProvenance:
        """Send hello and pin the server identity and capabilities."""
        if self._connected:
            assert self._provenance is not None
            return self._provenance
        if self._closed:
            raise RemoteError("transport_closed", "remote evaluator is closed")
        start = getattr(self.transport, "start", None)
        if callable(start):
            start()
        client_nonce = secrets.token_hex(16)
        hello = HelloRequest(
            protocol_version=PROTOCOL_VERSION,
            client_nonce=client_nonce,
            session_token=self.token,
            expected_site_id=self.expected_site_id,
            expected_worker_build=self.expected_worker_build,
            expected_task_pack_digest=self.expected_task_pack_digest,
            target_id=self.target_id,
        )
        self._send("hello", client_nonce, hello.__dict__)
        frame = self._read_frame()
        if frame.frame_type == "error":
            raise self._error_from(frame)
        if frame.frame_type != "hello" or frame.request_id != client_nonce:
            raise RemoteError("hello_rejected", "server did not answer hello with the same nonce")
        payload = frame.payload
        if payload.get("protocol_version") != PROTOCOL_VERSION:
            raise RemoteError("protocol_mismatch", "server protocol version differs")
        if payload.get("client_nonce") != client_nonce:
            raise RemoteError("hello_rejected", "server echoed a different client nonce")
        server_nonce = payload.get("server_nonce")
        if not isinstance(server_nonce, str) or not server_nonce:
            raise RemoteError("hello_rejected", "server did not provide a nonce")
        provenance_doc = payload.get("provenance")
        if not isinstance(provenance_doc, dict):
            raise RemoteError("hello_rejected", "server did not provide provenance")
        capabilities = payload.get("capabilities")
        if not isinstance(capabilities, dict):
            raise RemoteError("hello_rejected", "server did not provide capabilities")
        if capabilities.get("protocol_version") != PROTOCOL_VERSION:
            raise RemoteError("protocol_mismatch", "capabilities protocol version differs")
        if self.target_id not in (capabilities.get("targets") or []):
            raise RemoteError("target_mismatch", "server does not advertise the requested target")
        if self.expected_site_id and capabilities.get("site_id") != self.expected_site_id:
            raise RemoteError("site_mismatch", "server site differs from expectation")
        if self.expected_worker_build and provenance_doc.get("worker_build") != self.expected_worker_build:
            raise RemoteError("worker_build_mismatch", "server worker build differs from expectation")
        if self.expected_task_pack_digest and capabilities.get("task_pack_digest") != self.expected_task_pack_digest:
            raise RemoteError("task_pack_mismatch", "server task pack differs from expectation")
        try:
            provenance = EvaluatorProvenance.from_dict(provenance_doc)
        except EvaluationError as exc:
            raise RemoteError("hello_rejected", f"invalid server provenance: {exc}") from exc
        if provenance.target_id != self.target_id:
            raise RemoteError("target_mismatch", "server provenance target differs")
        if provenance.site_id != capabilities.get("site_id"):
            raise RemoteError("site_mismatch", "server capabilities and provenance disagree on site")
        if provenance.device_id != capabilities.get("device_id"):
            raise RemoteError("device_mismatch", "server capabilities and provenance disagree on device")
        if provenance.task_pack_digest != capabilities.get("task_pack_digest"):
            raise RemoteError("task_pack_mismatch", "server capabilities and provenance disagree on task pack")
        if provenance.protocol_version != PROTOCOL_VERSION:
            raise RemoteError("protocol_mismatch", "server provenance protocol differs")
        self._server_nonce = server_nonce
        self._provenance = provenance
        self._connected = True
        return provenance

    def _error_from(self, frame: Frame) -> RemoteError:
        payload = frame.payload
        code = payload.get("code")
        message = payload.get("message")
        retryable = payload.get("retryable", False)
        if not isinstance(code, str) or not code:
            code = "remote_error"
        if not isinstance(message, str):
            message = "remote evaluator returned an error"
        return RemoteError(code, message[:MAX_STDERR], retryable=retryable is True, request_id=frame.request_id)

    def _bind_request(self, request: EvaluationRequest) -> EvaluationRequest:
        if not isinstance(request, EvaluationRequest):
            raise TypeError("request must be an EvaluationRequest")
        changes = {}
        if not request.run_id:
            changes["run_id"] = self.run_id
        elif request.run_id != self.run_id:
            raise RemoteError("session_mismatch", "request run_id differs from this evaluator")
        if not request.session_id:
            changes["session_id"] = self.session_id
        elif request.session_id != self.session_id:
            raise RemoteError("session_mismatch", "request session_id differs from this evaluator")
        if not request.target_id:
            changes["target_id"] = self.target_id
        elif request.target_id != self.target_id:
            raise RemoteError("target_mismatch", "request target differs from this evaluator")
        if self.task_fingerprint:
            if request.task_fingerprint and request.task_fingerprint != self.task_fingerprint:
                raise RemoteError("task_mismatch", "request task fingerprint differs from this evaluator")
            if not request.task_fingerprint:
                changes["task_fingerprint"] = self.task_fingerprint
        return replace(request, **changes) if changes else request

    def submit(self, request: EvaluationRequest) -> EvaluationResult:
        """Send one request and return the validated result envelope."""
        self.connect()
        request = self._bind_request(request)
        self._active_request_id = request.request_id
        try:
            self._send("request", request.request_id, request.to_dict())
            while True:
                frame = self._read_frame()
                if frame.request_id != request.request_id:
                    raise RemoteError(
                        "response_mismatch",
                        f"server answered request {frame.request_id!r}, expected {request.request_id!r}",
                    )
                if frame.frame_type == "progress":
                    # Progress is advisory. It never completes or changes a request.
                    self.last_progress = frame
                    if request.progress_path is not None:
                        try:
                            from tcenv.events import publish

                            publish(request.progress_path, dict(frame.payload))
                        except Exception:
                            pass
                    continue
                if frame.frame_type == "error":
                    raise self._error_from(frame)
                if frame.frame_type != "result":
                    raise RemoteError("protocol_error", f"unexpected frame type {frame.frame_type!r}")
                return self._validate_result(request, frame)
        finally:
            self._active_request_id = ""

    def _validate_result(self, request: EvaluationRequest, frame: Frame) -> EvaluationResult:
        try:
            result = EvaluationResult.from_dict(frame.payload)
        except EvaluationError as exc:
            raise RemoteError("result_invalid", f"invalid evaluation result: {exc}") from exc
        provenance = result.provenance
        assert self._provenance is not None
        if provenance.epoch != self._provenance.epoch:
            raise RemoteError("epoch_drift", "server epoch changed during a request")
        if provenance.request_digest != request.request_digest:
            raise RemoteError("result_invalid", "result request digest does not match the request")
        if provenance.target_id != request.target_id:
            raise RemoteError("target_mismatch", "result target does not match the request")
        if provenance.site_id != self._provenance.site_id:
            raise RemoteError("site_mismatch", "result site changed during a request")
        if provenance.task_pack_digest != self._provenance.task_pack_digest:
            raise RemoteError("task_pack_mismatch", "result task pack changed during a request")
        if provenance.worker_build != self._provenance.worker_build:
            raise RemoteError("worker_build_mismatch", "result worker build changed during a request")
        evidence = result.evidence
        if not isinstance(evidence, dict) or evidence.get("candidate_sha256") != request.candidate_sha256:
            raise RemoteError("provenance_mismatch", "result evidence does not bind the candidate source")
        try:
            receipt_id = validate_receipt(result, request)
        except EvaluationError as exc:
            raise RemoteError("receipt_invalid", str(exc)) from exc
        if self.signature_key is not None:
            if not verify_result_signature(result, self.signature_key):
                raise RemoteError("signature_mismatch", "result signature does not verify")
        elif result.signature:
            raise RemoteError("signature_unverified", "result carries a signature but no verification key is configured")
        previous_digest = self._receipts.get(receipt_id)
        if previous_digest is not None and previous_digest != request.request_digest:
            raise RemoteError("receipt_replay", "receipt id was already used for a different request digest")
        self._receipts[receipt_id] = request.request_digest
        report = result.report
        if isinstance(report, dict):
            if report.get("simulated") is True:
                # Simulated reports intentionally carry an extra marker that the
                # production EvalReport schema does not know. The service still
                # verifies their task/target fields; the search layer refuses
                # simulated provenance before it can become a candidate.
                if report.get("task") != request.task or report.get("dtype") != request.dtype or report.get("size") != request.size:
                    raise RemoteError("result_invalid", "simulated report does not match the request")
                if report.get("target_id") and report.get("target_id") != request.target_id:
                    raise RemoteError("target_mismatch", "simulated report target does not match")
            else:
                try:
                    parsed = EvalReport.from_json(report)
                except Exception as exc:  # noqa: BLE001 - malformed remote data is a protocol result, not a kernel failure
                    raise RemoteError("result_invalid", f"invalid report: {exc}") from exc
                if parsed.task != request.task or parsed.dtype != request.dtype or parsed.size != request.size:
                    raise RemoteError("result_invalid", "report task/dtype/size does not match the request")
                if parsed.target_id and parsed.target_id != request.target_id:
                    raise RemoteError("target_mismatch", "report target does not match the request")
        return result

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        return self.submit(request)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        return self.submit(request)


def remote_evaluator_from_command(
    command: list[str],
    *,
    token: str,
    run_id: str,
    session_id: str,
    target_id: str,
    expected_site_id: str = "",
    expected_worker_build: str = "",
    expected_task_pack_digest: str = "",
    task_fingerprint: str = "",
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = 480.0,
    signature_key: bytes | None = None,
) -> RemoteEvaluator:
    """Convenience factory for the common SSH/forced-command deployment."""
    return RemoteEvaluator(
        CommandTransport(command, cwd=cwd, env=env),
        token=token,
        run_id=run_id,
        session_id=session_id,
        target_id=target_id,
        expected_site_id=expected_site_id,
        expected_worker_build=expected_worker_build,
        expected_task_pack_digest=expected_task_pack_digest,
        task_fingerprint=task_fingerprint,
        timeout_s=timeout_s,
        signature_key=signature_key,
    )


class RemoteEvaluatorPool:
    """Create one authenticated evaluator per run/session/target.

    ``RemoteEvaluator`` intentionally represents one grant. The pool is the
    adapter for a search, whose hmz session labels change between rounds. A
    transport factory receives ``(run_id, session_id, target_id)`` and a token
    provider receives ``(run_id, session_id)``; both are host-controlled.
    """

    def __init__(
        self,
        transport_factory: Callable[[tuple[str, str, str]], Transport],
        *,
        token: str | Callable[[str, str], str],
        run_id: str,
        session_id: str = "",
        target_id: str = "apple-m4-max",
        expected_site_id: str = "",
        expected_worker_build: str = "",
        expected_task_pack_digest: str = "",
        task_fingerprint: str = "",
        timeout_s: float | None = 480.0,
        signature_key: bytes | None = None,
        idle_timeout_s: float | None = 300.0,
        max_clients: int | None = 32,
    ) -> None:
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("remote pool run_id is required")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("remote pool target_id is required")
        if not callable(transport_factory):
            raise ValueError("remote pool transport_factory must be callable")
        if isinstance(token, str) and not token:
            raise ValueError("remote pool token or token provider is required")
        if not isinstance(token, str) and not callable(token):
            raise ValueError("remote pool token must be text or a provider")
        if idle_timeout_s is not None and (isinstance(idle_timeout_s, bool) or not isinstance(idle_timeout_s, (int, float)) or not idle_timeout_s > 0):
            raise ValueError("remote pool idle_timeout_s must be positive or None")
        if max_clients is not None and (isinstance(max_clients, bool) or not isinstance(max_clients, int) or max_clients < 1):
            raise ValueError("remote pool max_clients must be positive or None")
        self.idle_timeout_s = None if idle_timeout_s is None else float(idle_timeout_s)
        self.max_clients = max_clients
        self.transport_factory = transport_factory
        self.token = token
        self.run_id = run_id
        self.session_id = session_id
        self.target_id = target_id
        self.expected_site_id = expected_site_id
        self.expected_worker_build = expected_worker_build
        self.expected_task_pack_digest = expected_task_pack_digest
        self.task_fingerprint = task_fingerprint
        self.timeout_s = timeout_s
        self.signature_key = signature_key
        self._clients: dict[tuple[str, str, str], RemoteEvaluator] = {}
        self._request_clients: dict[str, RemoteEvaluator] = {}
        self._receipts: dict[str, str] = {}
        self._last: RemoteEvaluator | None = None
        self._last_used: dict[tuple[str, str, str], float] = {}
        self._closed = False

    def _touch(self, client: RemoteEvaluator) -> None:
        self._last = client
        for key, held in self._clients.items():
            if held is client:
                self._last_used[key] = time.monotonic()
                return

    def _close_key(self, key: tuple[str, str, str], client: RemoteEvaluator) -> None:
        if client._active_request_id or client in self._request_clients.values():
            return
        try:
            client.close()
        finally:
            self._clients.pop(key, None)
            self._last_used.pop(key, None)
            if self._last is client:
                self._last = None

    def reap_idle(self) -> None:
        """Close only inactive clients past the idle timeout or LRU limit."""
        now = time.monotonic()
        for key, client in list(self._clients.items()):
            if client._closed:
                self._clients.pop(key, None)
                self._last_used.pop(key, None)
                if self._last is client:
                    self._last = None
        if self.idle_timeout_s is not None:
            for key, last_used in list(self._last_used.items()):
                client = self._clients.get(key)
                if client is None:
                    self._last_used.pop(key, None)
                elif now - last_used >= self.idle_timeout_s:
                    self._close_key(key, client)
        if self.max_clients is None:
            return
        while len(self._clients) > self.max_clients:
            candidates = [
                (last_used, key, client)
                for key, last_used in self._last_used.items()
                if (client := self._clients.get(key)) is not None
                and not client._active_request_id
                and client not in self._request_clients.values()
            ]
            if not candidates:
                return
            _last_used, key, client = min(candidates)
            self._close_key(key, client)

    @property
    def provenance(self) -> EvaluatorProvenance | None:
        return None if self._last is None else self._last.provenance

    @property
    def epoch(self) -> str:
        return "" if self._last is None else self._last.epoch

    def ensure_epoch(self, request: EvaluationRequest) -> str:
        """Hello the session before deciding whether an outbox result is reusable."""
        return self._client_for(request).epoch

    def _client_for(self, request: EvaluationRequest) -> RemoteEvaluator:
        if self._closed:
            raise RemoteError("transport_closed", "remote evaluator pool is closed")
        self.reap_idle()
        run_id = request.run_id or self.run_id
        if request.run_id and request.run_id != self.run_id:
            raise RemoteError("session_mismatch", "request run_id differs from this evaluator pool")
        session_id = request.session_id or self.session_id
        target_id = request.target_id or self.target_id
        if not session_id:
            raise RemoteError("session_mismatch", "request has no session id for a pooled evaluator")
        if target_id != self.target_id:
            raise RemoteError("target_mismatch", "pool does not serve this target")
        key = (run_id, session_id, target_id)
        client = self._clients.get(key)
        if client is not None and client._closed:
            self._clients.pop(key, None)
            self._last_used.pop(key, None)
            if self._last is client:
                self._last = None
            client = None
        if client is not None:
            self._touch(client)
            return client
        token = self.token(run_id, session_id) if callable(self.token) else self.token
        if not isinstance(token, str) or not token:
            raise RemoteError("session_mismatch", f"no grant token for {run_id}/{session_id}")
        client = RemoteEvaluator(
            self.transport_factory(key),
            token=token,
            run_id=run_id,
            session_id=session_id,
            target_id=target_id,
            expected_site_id=self.expected_site_id,
            expected_worker_build=self.expected_worker_build,
            expected_task_pack_digest=self.expected_task_pack_digest,
            task_fingerprint=self.task_fingerprint,
            timeout_s=self.timeout_s,
            signature_key=self.signature_key,
        )
        try:
            client.connect()
        except Exception:
            client.close()
            raise
        self._clients[key] = client
        self._touch(client)
        self.reap_idle()
        return client

    def formal(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "formal_eval":
            raise ValueError("formal() requires kind='formal_eval'")
        client = self._client_for(request)
        self._request_clients[request.request_id] = client
        try:
            result = client.formal(request)
            try:
                receipt_id = validate_receipt(result, request)
            except EvaluationError as exc:
                raise RemoteError("receipt_invalid", str(exc)) from exc
            previous = self._receipts.get(receipt_id)
            if previous is not None and previous != request.request_digest:
                raise RemoteError("receipt_replay", "receipt id was already used for a different request digest")
            self._receipts[receipt_id] = request.request_digest
            return result
        finally:
            self._request_clients.pop(request.request_id, None)
            self._touch(client)

    def selfcheck(self, request: EvaluationRequest) -> EvaluationResult:
        if request.kind != "selfcheck":
            raise ValueError("selfcheck() requires kind='selfcheck'")
        client = self._client_for(request)
        self._request_clients[request.request_id] = client
        try:
            result = client.selfcheck(request)
            try:
                receipt_id = validate_receipt(result, request)
            except EvaluationError as exc:
                raise RemoteError("receipt_invalid", str(exc)) from exc
            previous = self._receipts.get(receipt_id)
            if previous is not None and previous != request.request_digest:
                raise RemoteError("receipt_replay", "receipt id was already used for a different request digest")
            self._receipts[receipt_id] = request.request_digest
            return result
        finally:
            self._request_clients.pop(request.request_id, None)
            self._touch(client)

    def cancel(self, request_id: str) -> None:
        """Cancel only the session that owns request_id, if it is still active."""
        client = self._request_clients.get(request_id)
        if client is not None:
            client.cancel(request_id)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for client in self._clients.values():
            try:
                client.close()
            except Exception:
                pass
        self._clients.clear()
        self._request_clients.clear()
        self._receipts.clear()
        self._last_used.clear()
        self._last = None

    def __enter__(self) -> "RemoteEvaluatorPool":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def remote_evaluator_pool_from_command(
    command: list[str] | Callable[[tuple[str, str, str]], list[str]],
    *,
    token: str | Callable[[str, str], str],
    run_id: str,
    session_id: str = "",
    target_id: str,
    expected_site_id: str = "",
    expected_worker_build: str = "",
    expected_task_pack_digest: str = "",
    task_fingerprint: str = "",
    cwd: Path | str | None = None,
    env: dict[str, str] | None = None,
    timeout_s: float | None = 480.0,
    signature_key: bytes | None = None,
    idle_timeout_s: float | None = 300.0,
    max_clients: int | None = 32,
) -> RemoteEvaluatorPool:
    """Build a per-session pool for SSH/forced-command transports."""
    def factory(key: tuple[str, str, str]) -> CommandTransport:
        argv = command(key) if callable(command) else command
        return CommandTransport(argv, cwd=cwd, env=env)

    return RemoteEvaluatorPool(
        factory,
        token=token,
        run_id=run_id,
        session_id=session_id,
        target_id=target_id,
        expected_site_id=expected_site_id,
        expected_worker_build=expected_worker_build,
        expected_task_pack_digest=expected_task_pack_digest,
        task_fingerprint=task_fingerprint,
        timeout_s=timeout_s,
        signature_key=signature_key,
        idle_timeout_s=idle_timeout_s,
        max_clients=max_clients,
    )
