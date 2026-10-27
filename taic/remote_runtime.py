"""Explicit production composition for remote evaluation and self-check.

The default search remains local. This module is used only when the caller
selects ``--remote-evaluator-config``. Remote evaluation is gRPC only: the
legacy SSH frame transport was removed, so a config that omits ``transport`` or
names ``legacy-ssh`` is rejected instead of falling back. The module builds a
shared-channel gRPC factory and a per-session SelfCheckServer; no provider
credential is read or stored here. The remote grant token and the agent bridge
token are separate credentials.
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

from tcenv.evaluation import EvaluationUnavailable

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

    token_file: Path
    target_id: str
    transport: str
    endpoint: str = ""
    tls_server_name: str = ""
    ca_file: Path | None = None
    client_certificate_file: Path | None = None
    client_private_key_file: Path | None = None
    signature_keyring_file: Path | None = None
    api_contract_ids: tuple[str, ...] = ()
    evidence_contract_ids: tuple[str, ...] = ()
    required_features: tuple[str, ...] = ()
    rpc_timeout_s: float = 30.0
    watch_slice_s: float = 30.0
    job_margin_s: float = 30.0
    expected_device_id: str = ""
    expected_epoch: str = ""
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

    def __post_init__(self) -> None:
        if self.transport != "grpc":
            raise ValueError(
                f"remote transport {self.transport!r} is not supported; "
                "only transport='grpc' is accepted"
            )

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
            "transport", "command", "endpoint", "tls_server_name", "ca_file",
            "client_certificate_file", "client_private_key_file", "signature_keyring_file",
            "api_contract_ids", "evidence_contract_ids", "required_features",
            "rpc_timeout_s", "watch_slice_s", "job_margin_s", "expected_device_id", "expected_epoch",
            "token_file", "selfcheck_token_file", "signature_key_file", "target_id", "expected_site_id", "expected_worker_build",
            "expected_task_pack_digest", "task_fingerprint", "publishable_site_ids", "publishable_epochs",
            "socket_root", "selfcheck_reps", "selfcheck_max_timed", "selfcheck_max_gpu_s",
            "selfcheck_allow_validate", "selfcheck_min_improvement", "selfcheck_noise_z",
        }
        unknown = set(document) - allowed
        if unknown:
            raise ValueError(f"unknown remote evaluation fields: {sorted(unknown)}")
        transport = document.get("transport")
        if transport is None:
            raise ValueError(
                "remote evaluation config must declare transport='grpc'; "
                "legacy-ssh is no longer supported"
            )
        if transport == "legacy-ssh":
            raise ValueError(
                "remote transport 'legacy-ssh' is no longer supported; use transport='grpc'"
            )
        if transport != "grpc":
            raise ValueError("remote transport must be 'grpc'")
        if document.get("command") is not None:
            raise ValueError("gRPC transport does not accept a command")
        for name in ("expected_site_id", "expected_device_id"):
            value = document.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"gRPC transport requires {name}")
        for name in ("endpoint", "ca_file", "signature_keyring_file"):
            value = document.get(name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"gRPC transport requires {name}")
        endpoint = document["endpoint"]
        if any(character in endpoint for character in "/ \t\n"):
            raise ValueError("gRPC endpoint must be host:port without credentials")
        certificate = document.get("client_certificate_file")
        private_key = document.get("client_private_key_file")
        if (certificate is None) != (private_key is None):
            raise ValueError("client_certificate_file and client_private_key_file must be supplied together")
        if certificate is not None and (not isinstance(certificate, str) or not certificate or not isinstance(private_key, str) or not private_key):
            raise ValueError("client certificate paths must be non-empty strings")
        for name in ("api_contract_ids", "evidence_contract_ids", "required_features"):
            value = document.get(name)
            if value is not None and (not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value)):
                raise ValueError(f"{name} must be a list of non-empty strings")
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
        value = document.get("selfcheck_max_gpu_s", 120.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 86_400:
            raise ValueError("selfcheck_max_gpu_s must be positive")
        if not isinstance(document.get("selfcheck_allow_validate", True), bool):
            raise ValueError("selfcheck_allow_validate must be boolean")
        for name, default, lower in (("selfcheck_min_improvement", 1.005, 1.0), ("selfcheck_noise_z", 2.0, 0.0)):
            value = document.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < lower:
                raise ValueError(f"{name} is outside the limit")
        for name, default in (("rpc_timeout_s", 30.0), ("watch_slice_s", 30.0), ("job_margin_s", 30.0)):
            value = document.get(name, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 86_400:
                raise ValueError(f"{name} must be positive and bounded")
        return cls(
            transport=transport,
            endpoint=str(document.get("endpoint") or ""),
            tls_server_name=str(document.get("tls_server_name") or ""),
            ca_file=resolve(document["ca_file"]) if document.get("ca_file") else None,
            client_certificate_file=resolve(document["client_certificate_file"]) if document.get("client_certificate_file") else None,
            client_private_key_file=resolve(document["client_private_key_file"]) if document.get("client_private_key_file") else None,
            signature_keyring_file=resolve(document["signature_keyring_file"]) if document.get("signature_keyring_file") else None,
            api_contract_ids=tuple(document.get("api_contract_ids") or ()),
            evidence_contract_ids=tuple(document.get("evidence_contract_ids") or ()),
            required_features=tuple(document.get("required_features") or ()),
            rpc_timeout_s=float(document.get("rpc_timeout_s", 30.0)),
            watch_slice_s=float(document.get("watch_slice_s", 30.0)),
            job_margin_s=float(document.get("job_margin_s", 30.0)),
            expected_device_id=str(document.get("expected_device_id") or ""),
            expected_epoch=str(document.get("expected_epoch") or ""),
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
        )

    def _format(self, value: str, run_id: str, session_id: str, evaluation_kind: str) -> str:
        return value.format(
            run_id=run_id,
            session_id=session_id,
            target_id=self.target_id,
            evaluation_kind=evaluation_kind,
        )

    def token_for_session(self, run_id: str, session_id: str) -> str:
        path = Path(self._format(str(self.token_file), run_id, session_id, "formal_eval"))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EvaluationUnavailable("session_mismatch", f"cannot read the remote session grant: {exc}", retryable=True) from exc
        if not token:
            raise EvaluationUnavailable("session_mismatch", "remote session grant is empty")
        return token

    def selfcheck_token_for_session(self, run_id: str, session_id: str) -> str:
        if self.selfcheck_token_file is None:
            raise EvaluationUnavailable("selfcheck_grant_required", "remote selfcheck requires a separate selfcheck_token_file")
        path = Path(self._format(str(self.selfcheck_token_file), run_id, session_id, "selfcheck"))
        try:
            token = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise EvaluationUnavailable("session_mismatch", f"cannot read the selfcheck session grant: {exc}", retryable=True) from exc
        if not token:
            raise EvaluationUnavailable("session_mismatch", "selfcheck session grant is empty")
        return token

    def make_selfcheck_evaluator(self, run_id: str):
        """Create a selfcheck-only evaluator; never silently reuse the formal grant."""
        if self.selfcheck_token_file is None:
            raise ValueError("remote selfcheck requires selfcheck_token_file")
        return self._make_grpc_factory(run_id, selfcheck=True)

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

    def _make_grpc_factory(self, run_id: str, *, selfcheck: bool):
        if self.ca_file is None or self.signature_keyring_file is None:
            raise ValueError("gRPC transport requires ca_file and signature_keyring_file")
        from tcenv.remote.grpc_client import DEFAULT_REQUIRED_FEATURES, GrpcExpected, grpc_evaluator_factory_from_config
        from tcenv.remote import contracts

        required = self.required_features or DEFAULT_REQUIRED_FEATURES
        return grpc_evaluator_factory_from_config(
            endpoint=self.endpoint,
            token_provider=self.selfcheck_token_for_session if selfcheck else self.token_for_session,
            run_id=run_id,
            target_id=self.target_id,
            ca_file=self.ca_file,
            certificate_file=self.client_certificate_file,
            private_key_file=self.client_private_key_file,
            keyring_file=self.signature_keyring_file,
            legacy_signature_key=self._signature_key(),
            expected=GrpcExpected(
                site_id=self.expected_site_id,
                device_id=self.expected_device_id,
                target_id=self.target_id,
                worker_build=self.expected_worker_build,
                task_pack_digest=self.expected_task_pack_digest,
                evaluator_epoch=self.expected_epoch,
            ),
            api_contract_ids=self.api_contract_ids or (contracts.API_CONTRACT_ID,),
            evidence_contract_ids=self.evidence_contract_ids or (contracts.EVIDENCE_CONTRACT_ID,),
            required_features=required,
            server_name_override=self.tls_server_name,
            rpc_timeout_s=self.rpc_timeout_s,
            watch_slice_s=self.watch_slice_s,
            job_margin_s=self.job_margin_s,
        )

    def make_evaluator(self, run_id: str):
        """Create the gRPC executor; the shared channel is keyed by endpoint/TLS."""
        return self._make_grpc_factory(run_id, selfcheck=False)

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
