"""Explicit production composition for remote evaluation and self-check.

The default search remains local. This module is used only when the caller
selects ``--remote-evaluator-config``. It builds a per-session RemoteEvaluatorPool
and a per-session SelfCheckServer factory; no provider credential is read or
stored here. The remote grant token and the agent bridge token are separate
credentials.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from tcenv.remote.client import RemoteError, RemoteEvaluatorPool, remote_evaluator_pool_from_command

from .selfcheck_bridge import SelfCheckConfig, SelfCheckServer


def _default_socket_root() -> Path:
    """A host-owned bridge root the fenced agent cannot write.

    ``/tmp`` is deliberately avoided: the hmz fence permits ordinary temp-file
    writes there, so a same-UID agent could unlink or replace the bridge's
    socket and token.  The user cache is still short enough for an AF_UNIX
    socket path and is outside the agent's writable workspace.
    """
    cache = Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache"))
    return cache / "taic-selfcheck"


@dataclass(frozen=True)
class RemoteEvaluationConfig:
    """Validated host-owned remote evaluation settings."""

    command: tuple[str, ...]
    token_file: Path
    target_id: str
    selfcheck_token_file: Path | None = None
    signature_key_file: Path | None = None
    expected_site_id: str = ""
    expected_worker_build: str = ""
    expected_task_pack_digest: str = ""
    task_fingerprint: str = ""
    publishable_site_ids: frozenset[str] = frozenset()
    publishable_epochs: frozenset[str] | None = None
    socket_root: Path = _default_socket_root()
    selfcheck_reps: int = 15
    selfcheck_max_timed: int = 20
    selfcheck_max_gpu_s: float = 120.0
    selfcheck_allow_validate: bool = True
    selfcheck_min_improvement: float = 1.005
    selfcheck_noise_z: float = 2.0
    timeout_s: float = 480.0

    @classmethod
    def from_file(cls, path: Path | str) -> "RemoteEvaluationConfig":
        path = Path(path).resolve()
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read remote evaluation config: {exc}") from exc
        if not isinstance(document, dict):
            raise ValueError("remote evaluation config must be an object")
        allowed = {
            "command", "token_file", "selfcheck_token_file", "signature_key_file", "target_id", "expected_site_id", "expected_worker_build",
            "expected_task_pack_digest", "task_fingerprint", "publishable_site_ids", "publishable_epochs",
            "socket_root", "selfcheck_reps", "selfcheck_max_timed", "selfcheck_max_gpu_s",
            "selfcheck_allow_validate", "selfcheck_min_improvement", "selfcheck_noise_z", "timeout_s",
        }
        unknown = set(document) - allowed
        if unknown:
            raise ValueError(f"unknown remote evaluation fields: {sorted(unknown)}")
        command = document.get("command")
        if not isinstance(command, list) or not command or any(not isinstance(part, str) or not part for part in command):
            raise ValueError("remote command must be a non-empty argv list")
        token_file = document.get("token_file")
        target_id = document.get("target_id")
        if not isinstance(token_file, str) or not token_file:
            raise ValueError("remote token_file is required")
        if not isinstance(target_id, str) or not target_id:
            raise ValueError("remote target_id is required")
        def resolve(value: str) -> Path:
            candidate = Path(value)
            return candidate if candidate.is_absolute() else path.parent / candidate
        sites = document.get("publishable_site_ids", [])
        epochs = document.get("publishable_epochs")
        if not isinstance(sites, list) or any(not isinstance(site, str) or not site for site in sites):
            raise ValueError("publishable_site_ids must be a list of non-empty strings")
        if epochs is not None and (not isinstance(epochs, list) or any(not isinstance(epoch, str) or not epoch for epoch in epochs)):
            raise ValueError("publishable_epochs must be null or a list of non-empty strings")
        socket_root = document.get("socket_root", str(_default_socket_root()))
        if not isinstance(socket_root, str) or not socket_root:
            raise ValueError("socket_root must be a path")
        for name in ("selfcheck_reps", "selfcheck_max_timed"):
            value = document.get(name, 15 if name == "selfcheck_reps" else 20)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("selfcheck_max_gpu_s", "timeout_s"):
            value = document.get(name, 120.0 if name == "selfcheck_max_gpu_s" else 480.0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 86_400:
                raise ValueError(f"{name} must be positive")
        if not isinstance(document.get("selfcheck_allow_validate", True), bool):
            raise ValueError("selfcheck_allow_validate must be boolean")
        for name, default, lower in (("selfcheck_min_improvement", 1.005, 1.0), ("selfcheck_noise_z", 2.0, 0.0)):
            value = document.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < lower:
                raise ValueError(f"{name} is outside the limit")
        return cls(
            command=tuple(command),
            token_file=resolve(token_file),
            selfcheck_token_file=resolve(document["selfcheck_token_file"]) if document.get("selfcheck_token_file") else None,
            signature_key_file=resolve(document["signature_key_file"]) if document.get("signature_key_file") else None,
            target_id=target_id,
            expected_site_id=str(document.get("expected_site_id") or ""),
            expected_worker_build=str(document.get("expected_worker_build") or ""),
            expected_task_pack_digest=str(document.get("expected_task_pack_digest") or ""),
            task_fingerprint=str(document.get("task_fingerprint") or ""),
            publishable_site_ids=frozenset(sites),
            publishable_epochs=None if epochs is None else frozenset(epochs),
            socket_root=resolve(socket_root),
            selfcheck_reps=document.get("selfcheck_reps", 15),
            selfcheck_max_timed=document.get("selfcheck_max_timed", 20),
            selfcheck_max_gpu_s=float(document.get("selfcheck_max_gpu_s", 120.0)),
            selfcheck_allow_validate=document.get("selfcheck_allow_validate", True),
            selfcheck_min_improvement=float(document.get("selfcheck_min_improvement", 1.005)),
            selfcheck_noise_z=float(document.get("selfcheck_noise_z", 2.0)),
            timeout_s=float(document.get("timeout_s", 480.0)),
        )

    def _format(self, value: str, run_id: str, session_id: str) -> str:
        return value.format(run_id=run_id, session_id=session_id, target_id=self.target_id)

    def command_for_session(self, key: tuple[str, str, str]) -> list[str]:
        run_id, session_id, _target = key
        return [self._format(part, run_id, session_id) for part in self.command]

    def token_for_session(self, run_id: str, session_id: str) -> str:
        path = Path(self._format(str(self.token_file), run_id, session_id))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RemoteError("session_mismatch", f"cannot read the remote session grant: {exc}", retryable=True) from exc
        if not token:
            raise RemoteError("session_mismatch", "remote session grant is empty")
        return token

    def selfcheck_token_for_session(self, run_id: str, session_id: str) -> str:
        path = self.selfcheck_token_file or self.token_file
        path = Path(self._format(str(path), run_id, session_id))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RemoteError("session_mismatch", f"cannot read the selfcheck session grant: {exc}", retryable=True) from exc
        if not token:
            raise RemoteError("session_mismatch", "selfcheck session grant is empty")
        return token

    def make_selfcheck_evaluator(self, run_id: str) -> RemoteEvaluatorPool:
        """Create the pool for selfcheck requests, optionally with a separate grant."""
        return remote_evaluator_pool_from_command(
            self.command_for_session,
            token=self.selfcheck_token_for_session,
            run_id=run_id,
            target_id=self.target_id,
            expected_site_id=self.expected_site_id,
            expected_worker_build=self.expected_worker_build,
            expected_task_pack_digest=self.expected_task_pack_digest,
            task_fingerprint=self.task_fingerprint,
            timeout_s=self.timeout_s,
            signature_key=self._signature_key(),
        )

    def _signature_key(self) -> bytes | None:
        if self.signature_key_file is None:
            return None
        try:
            key = self.signature_key_file.read_bytes()
        except OSError as exc:
            raise ValueError(f"cannot read remote signature key file: {exc}") from exc
        if not key:
            raise ValueError("remote signature key file must not be empty")
        return key

    def make_evaluator(self, run_id: str) -> RemoteEvaluatorPool:
        """Create a pool whose connections are keyed by run/session/target."""
        return remote_evaluator_pool_from_command(
            self.command_for_session,
            token=self.token_for_session,
            run_id=run_id,
            target_id=self.target_id,
            expected_site_id=self.expected_site_id,
            expected_worker_build=self.expected_worker_build,
            expected_task_pack_digest=self.expected_task_pack_digest,
            task_fingerprint=self.task_fingerprint,
            timeout_s=self.timeout_s,
            signature_key=self._signature_key(),
        )

    def make_bridge_factory(
        self,
        *,
        task: str,
        dtype: str,
        size: str,
        task_fingerprint: str,
        run_id: str,
    ) -> Callable[[str, Path, Any], tuple[SelfCheckServer, str]]:
        """Return the per-session bridge factory consumed by Orchestrator."""
        def factory(session_id: str, work: Path, evaluator: Any) -> tuple[SelfCheckServer, str]:
            token = secrets.token_hex(32)
            self.socket_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self.socket_root.chmod(0o700)
            except OSError:
                pass
            socket_name = hashlib.sha256(f"{run_id}\0{session_id}".encode("utf-8")).hexdigest()[:32] + ".sock"
            # Keep the capability file outside the agent-writable workspace. The
            # bridge writes it 0600 and removes it on stop; the agent receives
            # the value through its session environment.
            token_file = self.socket_root / (socket_name + ".token")
            selfcheck_evaluator = self.make_selfcheck_evaluator(run_id) if self.selfcheck_token_file is not None else evaluator
            config = SelfCheckConfig(
                task=task,
                dtype=dtype,
                size=size,
                target_id=self.target_id,
                task_fingerprint=task_fingerprint,
                run_id=run_id,
                session_id=session_id,
                token=token,
                token_file=token_file,
                reps=self.selfcheck_reps,
                self_check_dir=work / ".selfcheck",
                seed_report_path=work / "report.json",
                allow_validate=self.selfcheck_allow_validate,
                min_improvement=self.selfcheck_min_improvement,
                noise_z=self.selfcheck_noise_z,
                usage_path=work.parent.parent / "selfcheck_usage" / f"{hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:20]}.json",
                max_timed=self.selfcheck_max_timed,
                max_gpu_s=self.selfcheck_max_gpu_s,
                close_evaluator_on_stop=self.selfcheck_token_file is not None,
            )
            server = SelfCheckServer(config, selfcheck_evaluator, self.socket_root / socket_name)
            try:
                server.start()
            except Exception:
                server.stop()
                raise
            return server, str(token_file)
        return factory
