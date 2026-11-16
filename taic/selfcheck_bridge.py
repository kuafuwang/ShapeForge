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
import hashlib
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

from .evaluation_outbox import bind_wire_digest

MAX_MESSAGE_BYTES = 1_100_000
MAX_SOURCE_BYTES = 1_000_000


class SelfCheckBridgeError(RuntimeError):
    """A self-check request was rejected before or during evaluation."""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = bool(retryable)


def _uses_npu_contract(evaluator: object) -> bool:
    """True when the host evaluator serves the NPU source contract."""

    contract = getattr(evaluator, "contract", None)
    if contract is not None and getattr(contract, "npu_source", False):
        return True
    from tcenv.remote.contracts import NPU_API_CONTRACT_IDS

    return getattr(evaluator, "api_contract_id", "") in NPU_API_CONTRACT_IDS


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
    workspace: Path | None = None
    reps: int = 15
    cases: tuple[str, ...] | None = None
    self_check_dir: Path | None = None
    usage_path: Path | None = None
    quick_timeout_s: float = 420.0
    timed_timeout_s: float = 420.0
    max_timed: int = 20
    max_gpu_s: float = 120.0
    allowed_modes: frozenset[str] = frozenset({"quick", "timed", "paired"})
    track_id: str = "msl-single"
    allow_validate: bool = True
    require_contract_launch: bool = False
    seed_report_path: Path | None = None
    min_improvement: float = 1.005
    noise_z: float = 2.0
    max_source_bytes: int = MAX_SOURCE_BYTES
    close_evaluator_on_stop: bool = False

    def __post_init__(self) -> None:
        if self.workspace is None and self.self_check_dir is not None:
            # Backward-compatible host-owned default for work/.selfcheck.
            object.__setattr__(self, "workspace", self.self_check_dir.parent)
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
        if not self.allowed_modes or not self.allowed_modes <= {"quick", "timed", "paired"}:
            raise ValueError("selfcheck modes must be quick, timed, and/or paired")
        if not isinstance(self.track_id, str) or not self.track_id:
            raise ValueError("selfcheck track_id must be non-empty")
        if self.cases is not None and (not isinstance(self.cases, tuple) or any(not isinstance(case, str) or not case for case in self.cases)):
            raise ValueError("selfcheck cases must be a tuple of non-empty strings")
        if not isinstance(self.max_source_bytes, int) or isinstance(self.max_source_bytes, bool) or self.max_source_bytes <= 0:
            raise ValueError("selfcheck max_source_bytes must be positive")
        if not isinstance(self.allow_validate, bool) or not isinstance(self.require_contract_launch, bool):
            raise ValueError("selfcheck validation and contract-launch policies must be boolean")
        if isinstance(self.min_improvement, bool) or not isinstance(self.min_improvement, (int, float)) or not math.isfinite(self.min_improvement) or self.min_improvement < 1.0:
            raise ValueError("selfcheck min_improvement must be finite and at least one")
        if isinstance(self.noise_z, bool) or not isinstance(self.noise_z, (int, float)) or not math.isfinite(self.noise_z) or self.noise_z < 0:
            raise ValueError("selfcheck noise_z must be finite and non-negative")

    def request(self, source: str, mode: str, *, validate: bool = False) -> EvaluationRequest:
        if mode not in self.allowed_modes:
            raise SelfCheckBridgeError("mode_not_allowed", f"selfcheck mode {mode!r} is not allowed")
        if not isinstance(validate, bool):
            raise SelfCheckBridgeError("invalid_request", "validate must be boolean")
        if validate and (mode != "timed" or not self.allow_validate):
            raise SelfCheckBridgeError("validate_not_allowed", "validation is not allowed for this selfcheck")
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
            target_id=self.target_id,
            run_id=self.run_id,
            session_id=self.session_id,
            task_fingerprint=self.task_fingerprint,
            candidate_timeout_s=self.timed_timeout_s if mode == "timed" else self.quick_timeout_s,
            queue_timeout_s=30.0,
            cache_policy="disabled",
            eval_cache=False,
            validate=validate,
            require_contract_launch=self.require_contract_launch,
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
        if isinstance(result.report, EvalReport):
            report = result.report.to_json()
        else:
            # A decoded report must be a legal schema 4 before any of its fields are read; the
            # retired ``simulated`` key is not an input and there is no bypass.
            report = EvalReport.from_json(result.report).to_json()
        elapsed = report.get("elapsed_s", 0.0)
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
            self._usage["gpu_s"] = float(self._usage.get("gpu_s", 0.0)) + float(elapsed)
        if report.get("status") in {"timeout", "crash"}:
            self._usage["hang"] = True
        self._save_usage()
        if self.config.self_check_dir is not None:
            # The host, not the remote worker, writes the authoritative row.
            # ``tcenv.selfcheck.append`` also mirrors it to the host-only log.
            from tcenv.selfcheck import append
            from tcenv.schema import PROTOCOL_VERSION
            from .evidence_level import evidence_fields, grade_evidence

            measurement_kind = result.provenance.measurement_kind

            bench = report.get("benchmark") if isinstance(report.get("benchmark"), dict) else {}
            candidate = bench.get("candidate") if isinstance(bench.get("candidate"), dict) else {}
            receipt = result.receipt if isinstance(result.receipt, dict) else {}
            append(Path(self.config.self_check_dir), {
                "status": report.get("status"),
                "candidate_us": report.get("candidate_us"),
                "cv": candidate.get("cv"),
                "speedup": report.get("speedup"),
                "elapsed_s": elapsed,
                "task": request.task,
                "dtype": request.dtype,
                "size": request.size,
                "target_id": request.target_id,
                "protocol_version": PROTOCOL_VERSION,
                "kernel_sha256": hashlib.sha256(request.source.encode("utf-8")).hexdigest(),
                "request_id": request.request_id,
                "request_digest": request.request_digest,
                "receipt_id": receipt.get("receipt_id"),
                "measurement_kind": result.provenance.measurement_kind,
                **evidence_fields(grade_evidence(
                    "host_bridge_self_check",
                    record={"measurement_kind": measurement_kind},
                    simulated=measurement_kind == "simulated",
                )),
                # The row's simulated flag is derived from the evaluator provenance only; the
                # build kind is what the report's one build record says (None = no build ran).
                "simulated": result.provenance.measurement_kind == "simulated",
                "build_kind": (
                    report["build"].get("build_kind")
                    if isinstance(report.get("build"), dict) else None
                ),
            })

    def _presentation(self, request: EvaluationRequest, result: EvaluationResult) -> dict[str, str]:
        if request.mode != "timed" or self.config.seed_report_path is None:
            return {}
        try:
            seed = json.loads(Path(self.config.seed_report_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(seed, dict):
            return {}
        from tcenv.verdict import gpu_timestamp_line, replacement_line

        report = result.report
        return {
            "gpu_timestamp": gpu_timestamp_line(report, seed),
            "verdict": replacement_line(
                report,
                seed,
                min_improvement=self.config.min_improvement,
                noise_z=self.config.noise_z,
            ),
        }

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
            if not isinstance(document, dict) or not {"mode", "token"} <= set(document):
                raise SelfCheckBridgeError("invalid_request", "request fields must be mode, token, source or kernel, and optional validate")
            extra = set(document) - {"mode", "source", "kernel", "workspace", "token", "validate"}
            if document.get("mode") == "paired":
                if extra - {"kernel_b", "source_b"}:
                    raise SelfCheckBridgeError("invalid_request", "a paired request may only add kernel_b or source_b")
                if "paired" not in self.config.allowed_modes:
                    raise SelfCheckBridgeError("mode_not_allowed", "selfcheck mode 'paired' is not allowed")
                self._handle_paired(conn, document)
                return
            if extra:
                raise SelfCheckBridgeError("invalid_request", "request fields must be mode, token, source or kernel, and optional validate")
            if ("source" in document) == ("kernel" in document):
                raise SelfCheckBridgeError("invalid_request", "exactly one of source and kernel is required")
            token = document.get("token")
            if not isinstance(token, str) or not hmac.compare_digest(token, self.config.token):
                raise SelfCheckBridgeError("unauthorized", "selfcheck session token was refused")
            if "kernel" in document:
                if self.config.workspace is None or not isinstance(document["kernel"], str):
                    raise SelfCheckBridgeError("invalid_request", "a host workspace and relative kernel path are required")
                if "workspace" in document:
                    supplied_workspace = document["workspace"]
                    if not isinstance(supplied_workspace, str) or Path(supplied_workspace).resolve() != self.config.workspace.resolve():
                        raise SelfCheckBridgeError("invalid_request", "workspace does not match the host session")
                source = _read_workspace_kernel(document["kernel"], self.config.workspace, self.config.max_source_bytes)
            else:
                source = document["source"]
            request = self.config.request(source, document["mode"], validate=document.get("validate", False))
            quota_error = self._quota_error(request.mode)
            if quota_error is not None:
                raise quota_error
            self._active_requests.add(request.request_id)
            try:
                # Selfcheck has no durable outbox row, but the NPU contract still
                # requires the exact wire identity (CandidateDigest + device
                # group) before the adapter submits.
                request = bind_wire_digest(
                    request, self.evaluator, npu_contract=_uses_npu_contract(self.evaluator),
                )
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
            _write_document(conn, {"ok": True, "result": result.to_dict(), "presentation": self._presentation(request, result)})
        except SelfCheckBridgeError as exc:
            _write_document(conn, {"ok": False, "error": {"code": exc.code, "message": str(exc)[:1000], "retryable": exc.retryable}})
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            try:
                _write_document(conn, {"ok": False, "error": {"code": "invalid_request", "message": str(exc)[:500], "retryable": False}})
            except OSError:
                pass

    def _handle_paired(self, conn: socket.socket, document: dict) -> None:
        """@brief 宿主读取两份候选并计时，但结果只是提名。

        两个 SHA 写进 ``paired_comparisons.jsonl``，配额行写进 ``self_checks.jsonl``
        且不含 SHA。这里不写宿主证据。唯一的 K=1 权威证据由场次结束时的
        ``host_select`` 重跑产生。
        """
        token = document.get("token")
        if not isinstance(token, str) or not hmac.compare_digest(token, self.config.token):
            raise SelfCheckBridgeError("unauthorized", "selfcheck session token was refused")
        if "paired" not in self.config.allowed_modes:
            raise SelfCheckBridgeError("mode_not_allowed", "selfcheck mode 'paired' is not allowed")
        source_a, source_b = self._paired_sources(document)
        quota_error = self._quota_error("timed")
        if quota_error is not None:
            raise quota_error
        sha_a = hashlib.sha256(source_a.encode("utf-8")).hexdigest()
        sha_b = hashlib.sha256(source_b.encode("utf-8")).hexdigest()
        path_a, path_b = self._nomination_paths(document, source_a, source_b, sha_a, sha_b)
        row = {"sha_a": sha_a, "sha_b": sha_b, "path_a": path_a, "path_b": path_b, "conclusion": ""}
        try:
            measured = self._run_paired(source_a, source_b)
        except SelfCheckBridgeError as exc:
            self._record_paired_nomination(row, {"status": "crash", "reason": str(exc)})
            self._debit_paired({"status": "crash"}, failed=True)
            raise
        except Exception as exc:  # noqa: BLE001 - the bridge reports a structured failure
            self._record_paired_nomination(row, {"status": "crash", "reason": str(exc)})
            self._debit_paired({"status": "crash"}, failed=True)
            raise SelfCheckBridgeError("evaluation_unavailable", str(exc)[:500], retryable=True) from exc
        if not isinstance(measured, dict):
            measured = {"status": "crash", "reason": "paired compare returned no document"}
        measured.setdefault("sha_a", sha_a)
        measured.setdefault("sha_b", sha_b)
        row["conclusion"] = str(measured.get("conclusion") or "")
        self._record_paired_nomination(row, measured)
        self._debit_paired(measured)
        _write_document(conn, {"ok": True, "paired": measured})

    def _nomination_directory(self) -> Path | None:
        """@brief 本场提名目录。没有自测目录时用工作区里的 ``.selfcheck``。"""
        if self.config.self_check_dir is not None:
            return Path(self.config.self_check_dir)
        if self.config.workspace is not None:
            return Path(self.config.workspace) / ".selfcheck"
        return None

    def _nomination_paths(self, document: dict, source_a: str, source_b: str, sha_a: str, sha_b: str) -> tuple[str, str]:
        """@brief 提名要指向场次结束后仍在的文件，供宿主选择重算 SHA。"""
        if "kernel" in document and self.config.workspace is not None:
            root = Path(self.config.workspace)
            return (str((root / document["kernel"]).resolve()), str((root / document["kernel_b"]).resolve()))
        import tempfile

        directory = self._nomination_directory()
        folder = (directory if directory is not None else Path(tempfile.gettempdir())) / "paired_sources"
        folder.mkdir(parents=True, exist_ok=True)
        path_a = folder / f"{sha_a}.metal"
        path_b = folder / f"{sha_b}.metal"
        path_a.write_text(source_a, encoding="utf-8")
        path_b.write_text(source_b, encoding="utf-8")
        return str(path_a), str(path_b)

    def _record_paired_nomination(self, row: dict, document: dict) -> None:
        """@brief 把这次桥请求写成提名和配额行。不写宿主证据。"""
        directory = self._nomination_directory()
        if directory is None:
            return
        from taic.paired_compare import record_supported_attempt

        record_supported_attempt(directory, row, document)

    def _paired_sources(self, document: dict) -> tuple[str, str]:
        has_kernel = "kernel" in document or "kernel_b" in document
        has_source = "source" in document or "source_b" in document
        if has_kernel == has_source:
            raise SelfCheckBridgeError("invalid_request", "a paired request needs both kernels or both sources")
        if has_kernel:
            if self.config.workspace is None or not isinstance(document.get("kernel"), str) or not isinstance(document.get("kernel_b"), str):
                raise SelfCheckBridgeError("invalid_request", "a host workspace and two relative kernel paths are required")
            if "workspace" in document:
                supplied = document["workspace"]
                if not isinstance(supplied, str) or Path(supplied).resolve() != self.config.workspace.resolve():
                    raise SelfCheckBridgeError("invalid_request", "workspace does not match the host session")
            return (
                _read_workspace_kernel(document["kernel"], self.config.workspace, self.config.max_source_bytes),
                _read_workspace_kernel(document["kernel_b"], self.config.workspace, self.config.max_source_bytes),
            )
        source_a = document.get("source")
        source_b = document.get("source_b")
        if not isinstance(source_a, str) or not isinstance(source_b, str):
            raise SelfCheckBridgeError("invalid_source", "both sources must be text")
        for source in (source_a, source_b):
            if len(source.encode("utf-8")) > self.config.max_source_bytes:
                raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
        return source_a, source_b

    def _run_paired(self, source_a: str, source_b: str) -> dict:
        import tempfile

        from taic.paired_compare import run_host_pair

        sha_a = hashlib.sha256(source_a.encode("utf-8")).hexdigest()
        sha_b = hashlib.sha256(source_b.encode("utf-8")).hexdigest()
        with tempfile.TemporaryDirectory(prefix="taic-paired-") as temporary:
            root = Path(temporary)
            path_a = root / "a.metal"
            path_b = root / "b.metal"
            path_a.write_text(source_a, encoding="utf-8")
            path_b.write_text(source_b, encoding="utf-8")
            measured = run_host_pair(
                path_a, path_b, sha_a=sha_a, sha_b=sha_b, seed=7,
                task=self.config.task, dtype=self.config.dtype, size=self.config.size,
                track_id=self.config.track_id, reps=4,
            )
        if not isinstance(measured, dict):
            raise SelfCheckBridgeError("evaluation_unavailable", "paired compare returned no document", retryable=True)
        measured.setdefault("sha_a", sha_a)
        measured.setdefault("sha_b", sha_b)
        return measured

    def _debit_paired(self, measured: dict, *, failed: bool = False) -> None:
        """@brief 配对比较占用一次 timed 配额。两个 SHA 不进这条记账。"""
        self._usage["count"] = int(self._usage.get("count", 0)) + 1
        elapsed = measured.get("elapsed_s", None)
        if (
            "elapsed_s" in measured
            and isinstance(elapsed, (int, float))
            and not isinstance(elapsed, bool)
            and math.isfinite(elapsed)
            and elapsed >= 0
        ):
            self._usage["gpu_s"] = float(self._usage.get("gpu_s", 0.0)) + float(elapsed)
        if failed or measured.get("status") in {"timeout", "crash"}:
            self._usage["hang"] = True
        self._save_usage()

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
        self.last_presentation: dict[str, str] = {}

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

    def check(self, source: str, mode: str, *, validate: bool = False) -> EvaluationResult:
        if not isinstance(source, str):
            raise TypeError("source must be text")
        return self._check({"source": source}, mode, validate=validate)

    def check_path(self, kernel: str, mode: str, *, validate: bool = False, workspace: str = "") -> EvaluationResult:
        """Ask the host to read a candidate from its configured workspace."""
        if not isinstance(kernel, str):
            raise TypeError("kernel path must be text")
        candidate = {"kernel": kernel, "workspace": workspace} if workspace else {"kernel": kernel}
        return self._check(candidate, mode, validate=validate)

    def compare(self, kernel: str, kernel_b: str, *, workspace: str = "") -> dict:
        """Ask the host to time two workspace candidates. The client does not time them."""
        if not isinstance(kernel, str) or not isinstance(kernel_b, str):
            raise TypeError("kernel paths must be text")
        payload = {"mode": "paired", "kernel": kernel, "kernel_b": kernel_b, "token": self._token()}
        if workspace:
            payload["workspace"] = workspace
        document = self._exchange(payload)
        paired = document.get("paired")
        if not isinstance(paired, dict):
            raise SelfCheckBridgeError("evaluation_unavailable", "paired compare returned no document", retryable=True)
        return paired

    def _check(self, candidate: dict[str, str], mode: str, *, validate: bool) -> EvaluationResult:
        if mode not in {"quick", "timed"}:
            raise ValueError("mode must be quick or timed")
        if validate and mode != "timed":
            raise ValueError("validation requires a timed selfcheck")
        payload = {"mode": mode, **candidate, "token": self._token(), "validate": bool(validate)}
        return self._result(self._exchange(payload))

    def _exchange(self, payload: dict) -> dict:
        raw_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(raw_payload) > MAX_MESSAGE_BYTES:
            raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout_s)
                sock.connect(self.socket_path)
                sock.sendall(raw_payload + b"\n")
                raw = _read_line(sock, MAX_MESSAGE_BYTES)
        except (OSError, socket.timeout) as exc:
            raise SelfCheckBridgeError("evaluation_unavailable", f"selfcheck bridge unavailable: {exc}", retryable=True) from exc
        from tcenv.schema import SchemaError, loads_strict

        try:
            document = loads_strict(raw.decode("utf-8"), what="selfcheck response")
        except (UnicodeDecodeError, SchemaError) as exc:
            raise SelfCheckBridgeError("invalid_response", f"selfcheck response is not valid JSON: {exc}") from exc
        if not isinstance(document, dict):
            raise SelfCheckBridgeError("invalid_response", "bridge response must be an object")
        if document.get("ok") is not True:
            error = document.get("error") if isinstance(document.get("error"), dict) else {}
            raise SelfCheckBridgeError(
                str(error.get("code") or "selfcheck_failed"),
                str(error.get("message") or "selfcheck failed"),
                retryable=error.get("retryable") is True,
            )
        return document

    def _result(self, document: dict) -> EvaluationResult:
        presentation = document.get("presentation")
        self.last_presentation = {str(k): str(v) for k, v in presentation.items() if isinstance(v, str)} if isinstance(presentation, dict) else {}
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
            if stop_event is None:
                raise SelfCheckBridgeError("evaluation_unavailable", "selfcheck bridge response timed out", retryable=True)
            if stop_event.is_set():
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
    candidate = root / path
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("kernel path must not contain symbolic links")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("kernel path escapes the session workspace")
    return resolved


def _read_workspace_kernel(value: str, workspace: Path, limit: int) -> str:
    """Read one bounded regular file without following agent-raced symlinks."""
    path = Path(value)
    if not path.parts:
        raise ValueError("kernel path must name a file")
    _workspace_kernel(value, str(workspace))  # legacy validation and diagnostics
    if any(part == ".." for part in path.parts):
        raise ValueError("kernel path must not contain '..'")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory = getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(workspace, os.O_RDONLY | directory | nofollow)
    current_fd = root_fd
    try:
        for part in path.parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | directory | nofollow, dir_fd=current_fd)
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        flags = os.O_RDONLY | nofollow | getattr(os, "O_NONBLOCK", 0)
        file_fd = os.open(path.parts[-1], flags, dir_fd=current_fd)
        try:
            info = os.fstat(file_fd)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("kernel path must name a regular file")
            if info.st_size > limit:
                raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
            with os.fdopen(file_fd, "rb") as handle:
                file_fd = -1
                data = handle.read(limit + 1)
        finally:
            if file_fd >= 0:
                os.close(file_fd)
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    if len(data) > limit:
        raise SelfCheckBridgeError("source_too_large", "source exceeds the selfcheck limit")
    return data.decode("utf-8")


def main(argv: list[str] | None = None) -> int:
    """Run one self-check; task/target/reps stay host-owned."""
    parser = argparse.ArgumentParser(prog="taic-selfcheck")
    parser.add_argument("--socket", default=os.environ.get("TAIC_SELFCHECK_SOCKET", ""))
    parser.add_argument("--token-file", default=os.environ.get("TAIC_SELFCHECK_TOKEN_FILE", ""))
    parser.add_argument("--workspace", default="")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--kernel-b", default="")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--validate", action="store_true")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--quick", action="store_const", const="quick", dest="mode")
    group.add_argument("--timed", action="store_const", const="timed", dest="mode")
    group.add_argument("--paired", action="store_const", const="paired", dest="mode")
    args = parser.parse_args(argv)
    if args.validate and args.mode != "timed":
        print("--validate belongs to a timed selfcheck", file=sys.stderr)
        return 2
    if args.mode == "paired" and not args.kernel_b:
        print("--kernel-b is required for a paired compare", file=sys.stderr)
        return 2
    if not args.socket:
        print("TAIC_SELFCHECK_SOCKET or --socket is required", file=sys.stderr)
        return 2
    try:
        client = SelfCheckClient(
            args.socket,
            token=os.environ.get("TAIC_SELFCHECK_TOKEN", ""),
            token_file=args.token_file,
        )
        # The host checks --workspace against its session binding and reads the
        # candidate itself; this process never chooses the authoritative root.
        if args.mode == "paired":
            paired = client.compare(args.kernel, args.kernel_b, workspace=args.workspace)
            print(json.dumps(paired, ensure_ascii=False, sort_keys=True))
            return 0 if paired.get("applied") else 1
        result = client.check_path(args.kernel, args.mode, validate=args.validate, workspace=args.workspace)
        report = result.report if isinstance(result.report, EvalReport) else result.report
        if not isinstance(report, EvalReport):
            report = EvalReport.from_json(report)
    except (OSError, ValueError, SelfCheckBridgeError, EvaluationError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(format_feedback(report))
        for key in ("gpu_timestamp", "verdict"):
            line = client.last_presentation.get(key)
            if line:
                print(line)
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
