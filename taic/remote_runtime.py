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
    publishable_epochs: frozenset[str] = frozenset()
    socket_root: Path = _default_socket_root()
    selfcheck_reps: int = 15
    selfcheck_max_timed: int = 20
    selfcheck_max_gpu_s: float = 120.0
    selfcheck_allow_validate: bool = True
    selfcheck_min_improvement: float = 1.005
    selfcheck_noise_z: float = 2.0
    idle_timeout_s: float | None = 300.0
    max_clients: int | None = 32
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
            "selfcheck_allow_validate", "selfcheck_min_improvement", "selfcheck_noise_z",
            "idle_timeout_s", "max_clients", "timeout_s",
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
        epochs = [] if document.get("publishable_epochs") is None else document.get("publishable_epochs")
        if not isinstance(sites, list) or any(not isinstance(site, str) or not site for site in sites):
            raise ValueError("publishable_site_ids must be a list of non-empty strings")
        if not isinstance(epochs, list) or any(not isinstance(epoch, str) or not epoch for epoch in epochs):
            raise ValueError("publishable_epochs must be a list of non-empty strings")
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
        idle = document.get("idle_timeout_s", 300.0)
        if idle is not None and (isinstance(idle, bool) or not isinstance(idle, (int, float)) or not math.isfinite(idle) or idle <= 0):
            raise ValueError("idle_timeout_s must be positive or null")
        max_clients = document.get("max_clients", 32)
        if max_clients is not None and (isinstance(max_clients, bool) or not isinstance(max_clients, int) or max_clients < 1):
            raise ValueError("max_clients must be positive or null")
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
            publishable_epochs=frozenset(epochs),
            socket_root=resolve(socket_root),
            selfcheck_reps=document.get("selfcheck_reps", 15),
            selfcheck_max_timed=document.get("selfcheck_max_timed", 20),
            selfcheck_max_gpu_s=float(document.get("selfcheck_max_gpu_s", 120.0)),
            selfcheck_allow_validate=document.get("selfcheck_allow_validate", True),
            selfcheck_min_improvement=float(document.get("selfcheck_min_improvement", 1.005)),
            selfcheck_noise_z=float(document.get("selfcheck_noise_z", 2.0)),
            idle_timeout_s=idle,
            max_clients=max_clients,
            timeout_s=float(document.get("timeout_s", 480.0)),
        )

    def _format(self, value: str, run_id: str, session_id: str, evaluation_kind: str) -> str:
        return value.format(
            run_id=run_id,
            session_id=session_id,
            target_id=self.target_id,
            evaluation_kind=evaluation_kind,
        )

    def command_for_session(self, key: tuple[str, str, str], evaluation_kind: str = "formal_eval") -> list[str]:
        """Build one legacy frontend command.

        Concurrent formal/selfcheck processes are independent legacy frontends.
        Configurations which keep their SQLite state in the command can use the
        ``{evaluation_kind}`` placeholder to give each frontend its own
        single-active store instead of violating the owner lock.
        """

        run_id, session_id, _target = key
        return [self._format(part, run_id, session_id, evaluation_kind) for part in self.command]

    def token_for_session(self, run_id: str, session_id: str) -> str:
        path = Path(self._format(str(self.token_file), run_id, session_id, "formal_eval"))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RemoteError("session_mismatch", f"cannot read the remote session grant: {exc}", retryable=True) from exc
        if not token:
            raise RemoteError("session_mismatch", "remote session grant is empty")
        return token

    def selfcheck_token_for_session(self, run_id: str, session_id: str) -> str:
        if self.selfcheck_token_file is None:
            raise RemoteError("selfcheck_grant_required", "remote selfcheck requires a separate selfcheck_token_file")
        path = Path(self._format(str(self.selfcheck_token_file), run_id, session_id, "selfcheck"))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise RemoteError("session_mismatch", f"cannot read the selfcheck session grant: {exc}", retryable=True) from exc
        if not token:
            raise RemoteError("session_mismatch", "selfcheck session grant is empty")
        return token

    def make_selfcheck_evaluator(self, run_id: str) -> RemoteEvaluatorPool:
        """Create a selfcheck-only pool; never silently reuse the formal grant."""
        if self.selfcheck_token_file is None:
            raise ValueError("remote selfcheck requires selfcheck_token_file")
        return remote_evaluator_pool_from_command(
            lambda key: self.command_for_session(key, evaluation_kind="selfcheck"),
            token=self.selfcheck_token_for_session,
            run_id=run_id,
            target_id=self.target_id,
            expected_site_id=self.expected_site_id,
            expected_worker_build=self.expected_worker_build,
            expected_task_pack_digest=self.expected_task_pack_digest,
            task_fingerprint=self.task_fingerprint,
            timeout_s=self.timeout_s,
            signature_key=self._signature_key(),
            idle_timeout_s=self.idle_timeout_s,
            max_clients=self.max_clients,
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
            idle_timeout_s=self.idle_timeout_s,
            max_clients=self.max_clients,
        )

    def make_bridge_factory(
        self,
        *,
        task: str,
        dtype: str,
        size: str,
        task_fingerprint: str,
        run_id: str,
        require_contract_launch: bool = False,
        min_improvement: float | None = None,
        noise_z: float | None = None,
    ) -> Callable[[str, Path, Any], tuple[SelfCheckServer, str]]:
        """Return the per-session bridge factory consumed by Orchestrator."""
        if self.selfcheck_token_file is None:
            raise ValueError("remote selfcheck requires selfcheck_token_file; formal fallback is disabled")
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
            usage_path = work.parent.parent / "selfcheck_usage" / f"{hashlib.sha256(session_id.encode('utf-8')).hexdigest()[:20]}.json"
            # Orchestrator stages report.json from the trusted SessionBrief
            # before invoking this factory. Snapshot it outside the workspace;
            # reading it lazily would let the agent rewrite the host verdict.
            seed_report_path = usage_path.with_suffix(".seed.json")
            try:
                seed_report = (work / "report.json").read_text(encoding="utf-8")
            except OSError:
                seed_report_path = None
            else:
                seed_report_path.parent.mkdir(parents=True, exist_ok=True)
                seed_report_path.write_text(seed_report, encoding="utf-8")
            selfcheck_evaluator = self.make_selfcheck_evaluator(run_id)
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
                workspace=work,
                reps=self.selfcheck_reps,
                self_check_dir=work / ".selfcheck",
                seed_report_path=seed_report_path,
                allow_validate=self.selfcheck_allow_validate,
                require_contract_launch=require_contract_launch,
                min_improvement=self.selfcheck_min_improvement if min_improvement is None else min_improvement,
                noise_z=self.selfcheck_noise_z if noise_z is None else noise_z,
                usage_path=usage_path,
                max_timed=self.selfcheck_max_timed,
                max_gpu_s=self.selfcheck_max_gpu_s,
                close_evaluator_on_stop=True,
            )
            server = SelfCheckServer(config, selfcheck_evaluator, self.socket_root / socket_name)
            try:
                server.start()
            except Exception:
                server.stop()
                raise
            return server, str(token_file)
        return factory
