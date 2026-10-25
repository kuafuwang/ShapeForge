"""Optional Phase 4 pieces for an agent that runs outside the control host.

Phase 3 proves a workspace transport with the agent still on the control host.  This module
adds the two things an agent on the far side needs and one thing the trusted evaluation side
must never lose:

- a **scoped, short-lived credential** the agent environment can use without holding a
  long-term provider key, minted and revoked by the control host and never written to the
  workspace, TASK or a log;
- a **credential seam** the runner calls to hand that credential to the session's
  environment: a deterministic loopback fake provider for tests, or a host-owned helper for
  an operator-provisioned host;
- a **trusted-evaluation fence** that refuses a placement where the agent workspace overlaps
  the evaluator's code, locks, reports, receipts or self-check authority directory.

Everything here is opt-in.  ``harness=local`` (the default and Phase 3) never reaches it.
The fence is a configuration check, not a sandbox: it proves the runner cannot be pointed at
an overlapping layout, not that a compromised remote host behaves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Protocol

from .redact import redact_text

CREDENTIAL_SCHEMA_VERSION = 1
_SEAM_MODES = frozenset({"none", "loopback-fake", "external-helper"})
_ABSOLUTE = re.compile(r"/[A-Za-z0-9._+/-]*")


class IsolationError(RuntimeError):
    """A typed reason an isolated-agent placement is not acceptable."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(f"{kind}: {redact_text(message, limit=600)}")


def _absolute_paths(values: Any, *, what: str) -> tuple[str, ...]:
    """Validate a list of absolute host paths, refusing anything ambiguous."""
    if values is None:
        return ()
    if not isinstance(values, (list, tuple)):
        raise IsolationError("config", f"{what} must be a list of absolute paths")
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.startswith("/") or ".." in value.split("/"):
            raise IsolationError("config", f"{what} must hold absolute paths without '..'")
        if any(ord(character) < 0x20 for character in value) or not _ABSOLUTE.fullmatch(value):
            raise IsolationError("config", f"{what} has a character outside the allowlist")
        result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class IsolatedAgentConfig:
    """Host-owned settings for an agent whose CLI runs on the workspace host.

    ``credential_seam`` is the only way a session gets a provider credential.  The loopback
    fake is a deterministic test provider; ``external-helper`` names a host-owned command the
    operator supplies, which receives the short-lived token on stdin and must provision it in
    the remote environment without copying a long-term key.  The helper's behaviour is
    operator-attested: this project can verify what it itself writes, not what the helper
    does on the far side.
    """

    agent_kind: str = "pi"
    credential_seam: str = "none"
    credential_ttl_s: float = 600.0
    credential_max_uses: int = 32
    helper: tuple[str, ...] = ()
    trusted_evaluator_paths: tuple[str, ...] = ()
    simulated: bool = False

    def __post_init__(self) -> None:
        if self.agent_kind != "pi":
            raise IsolationError("config", "the only supported agent kind is pi")
        if self.credential_seam not in _SEAM_MODES:
            raise IsolationError("config", f"unknown credential_seam {self.credential_seam!r}")
        if isinstance(self.credential_ttl_s, bool) or not isinstance(self.credential_ttl_s, (int, float)) or not 0 < float(self.credential_ttl_s) <= 86_400:
            raise IsolationError("config", "credential_ttl_s must be positive and bounded")
        if isinstance(self.credential_max_uses, bool) or not isinstance(self.credential_max_uses, int) or not 1 <= self.credential_max_uses <= 1_000_000:
            raise IsolationError("config", "credential_max_uses must be a positive integer")
        if not isinstance(self.simulated, bool):
            raise IsolationError("config", "simulated must be a boolean")
        if self.credential_seam == "external-helper":
            if not self.helper or any(not isinstance(part, str) or not part for part in self.helper):
                raise IsolationError("config", "external-helper needs a non-empty argv helper")
            if self.simulated:
                raise IsolationError("config", "external-helper is a real seam and cannot be marked simulated")
        if self.credential_seam == "loopback-fake" and not self.simulated:
            raise IsolationError("config", "loopback-fake is a deterministic test provider; set simulated=true explicitly")
        _absolute_paths(self.trusted_evaluator_paths, what="trusted_evaluator_paths")

    @classmethod
    def from_dict(cls, document: Any) -> "IsolatedAgentConfig":
        if document is None:
            return cls()
        if not isinstance(document, dict):
            raise IsolationError("config", "isolation must be an object")
        allowed = {"agent_kind", "credential_seam", "credential_ttl_s", "credential_max_uses", "helper", "trusted_evaluator_paths", "simulated"}
        unknown = set(document) - allowed
        if unknown:
            raise IsolationError("config", f"unknown isolation fields: {sorted(unknown)}")
        helper = document.get("helper") or ()
        if not isinstance(helper, (list, tuple)) or any(not isinstance(part, str) for part in helper):
            raise IsolationError("config", "helper must be an argv list")
        return cls(
            agent_kind=str(document.get("agent_kind", "pi")),
            credential_seam=str(document.get("credential_seam", "none")),
            credential_ttl_s=float(document.get("credential_ttl_s", 600.0)),
            credential_max_uses=document.get("credential_max_uses", 32),
            helper=tuple(helper),
            trusted_evaluator_paths=_absolute_paths(document.get("trusted_evaluator_paths"), what="trusted_evaluator_paths"),
            simulated=document.get("simulated", False),
        )

    @property
    def requires_remote_agent(self) -> bool:
        return self.credential_seam != "none"


# ------------------------------------------------------------------------ scoped credential


@dataclass(frozen=True)
class ScopedCredential:
    """One short-lived credential bound to a single session.

    The token itself is never persisted and never put in a manifest, TASK, argv or log; only
    its hash is kept by the authority.  ``repr`` omits it so an accidental diagnostic cannot
    leak it either.
    """

    credential_id: str
    token: str = field(repr=False)
    scope: dict[str, str]
    expires_at: float
    max_uses: int

    def as_ref(self) -> dict[str, Any]:
        """The non-secret reference the run record may hold."""
        return {
            "credential_id": self.credential_id,
            "scope": dict(self.scope),
            "expires_in_s": max(0.0, self.expires_at - time.time()),
            "max_uses": self.max_uses,
            "token_at_rest": False,
        }


class ScopedCredentialAuthority:
    """Mint, verify, count and revoke session-scoped credentials in host memory.

    A process crash drops every credential with the process; nothing needs a durable store
    and nothing can be replayed after the run.  The hash is kept so verification never needs
    the token at rest.
    """

    def __init__(self, *, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._records: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.usage: dict[str, dict[str, int]] = {}

    def mint(self, *, run_id: str, session_label: str, runtime: str, ttl_s: float, max_uses: int) -> ScopedCredential:
        """Mint one credential for exactly this run/session/runtime."""
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        credential_id = digest[:32]
        scope = {"run_id": str(run_id), "session_label": str(session_label), "runtime": str(runtime)}
        expires_at = self._clock() + float(ttl_s)
        with self._lock:
            self._records[credential_id] = {
                "digest": digest,
                "scope": scope,
                "expires_at": expires_at,
                "max_uses": int(max_uses),
                "uses": 0,
                "revoked": False,
            }
        return ScopedCredential(credential_id=credential_id, token=token, scope=scope, expires_at=expires_at, max_uses=int(max_uses))

    def verify(
        self,
        token: str,
        *,
        now: float | None = None,
        scope: dict[str, str] | None = None,
    ) -> str | None:
        """Return the credential id for a live token in ``scope``, or ``None``.

        Expiry, revocation and a scope mismatch are all refusals.  A token minted for run A
        must not be accepted for run B even if it leaks, so every caller that knows the
        expected run/session/runtime passes it here.  A token is not consumed here: a fake
        provider may call this and then :meth:`consume`, so that a rejected request is not
        billed against the use budget.
        """
        if not isinstance(token, str) or not token:
            return None
        digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
        stamp = self._clock() if now is None else float(now)
        with self._lock:
            for credential_id, record in self._records.items():
                if not secret_compare(record["digest"], digest):
                    continue
                if record["revoked"] or stamp >= record["expires_at"]:
                    return None
                if record["uses"] >= record["max_uses"]:
                    return None
                if scope is not None:
                    recorded = record["scope"]
                    if any(str(recorded.get(key)) != str(value) for key, value in scope.items()):
                        return None
                return credential_id
        return None

    def consume(self, credential_id: str) -> bool:
        """Count one use, refusing a credential that is out of budget."""
        with self._lock:
            record = self._records.get(credential_id)
            if record is None or record["revoked"] or record["uses"] >= record["max_uses"]:
                return False
            record["uses"] += 1
            self.usage.setdefault(credential_id, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})["requests"] += 1
            return True

    def revoke(self, credential_id: str) -> bool:
        """Revoke one credential; returns whether it was still live."""
        with self._lock:
            record = self._records.get(credential_id)
            if record is None:
                return False
            record["revoked"] = True
            return True

    def live_ids(self) -> tuple[str, ...]:
        """Every credential that could still be used, for cleanup evidence."""
        stamp = self._clock()
        with self._lock:
            return tuple(
                credential_id
                for credential_id, record in self._records.items()
                if not record["revoked"] and stamp < record["expires_at"] and record["uses"] < record["max_uses"]
            )

    def record_usage(self, credential_id: str, *, prompt_tokens: int, completion_tokens: int) -> None:
        """Attribute token usage to one credential, for reconciliation evidence."""
        with self._lock:
            usage = self.usage.setdefault(credential_id, {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0})
            usage["prompt_tokens"] += int(prompt_tokens)
            usage["completion_tokens"] += int(completion_tokens)


def secret_compare(left: str, right: str) -> bool:
    """Constant-time comparison of two hex digests."""
    return secrets.compare_digest(str(left), str(right))


# ------------------------------------------------------------------------ fake provider


class LoopbackFakeProvider:
    """A deterministic OpenAI-compatible endpoint on loopback, for isolated-agent tests.

    It never reaches a network, never reads a provider key and answers with a fixed
    completion.  Every request must carry a live scoped token; the token is consumed before
    the answer, so quota behaviour is exercised too.
    """

    def __init__(self, authority: ScopedCredentialAuthority, *, expected_scope: dict[str, str] | None = None) -> None:
        self.authority = authority
        self.expected_scope = dict(expected_scope) if expected_scope else None
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.requests: list[dict[str, Any]] = []

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise IsolationError("seam", "the fake provider is not running")
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> "LoopbackFakeProvider":
        if self._server is not None:
            return self
        authority = self.authority
        requests = self.requests

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def log_message(self, *_args) -> None:  # noqa: ANN002 - silence the default log
                return

            def do_POST(self) -> None:  # noqa: N802 - the stdlib name
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(min(length, 1_000_000))
                header = self.headers.get("Authorization") or ""
                token = header[7:].strip() if header.startswith("Bearer ") else ""
                credential_id = authority.verify(token, scope=provider.expected_scope)
                if credential_id is None or not authority.consume(credential_id):
                    self._reply(401, {"error": {"message": "unauthorized", "type": "invalid_request_error"}})
                    return
                try:
                    payload = json.loads(body.decode("utf-8")) if body else {}
                except ValueError:
                    payload = {}
                prompt = str(payload.get("prompt") or "") if isinstance(payload, dict) else ""
                prompt_tokens = max(1, len(prompt.split()))
                completion_tokens = 2
                authority.record_usage(credential_id, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                requests.append({
                    "credential_id": credential_id,
                    "path": self.path,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                })
                self._reply(
                    200,
                    {
                        "id": "fake-completion",
                        "object": "chat.completion",
                        "model": "fake",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "fake"}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    },
                )

            def _reply(self, status: int, document: dict[str, Any]) -> None:
                body = json.dumps(document).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        provider = self
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, name="taic-fake-provider", daemon=True)
        thread.start()
        self._server, self._thread = server, thread
        return self

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server, self._thread = None, None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)


# ------------------------------------------------------------------------------- seams


class CredentialSeam(Protocol):
    """How one session's scoped credential reaches the agent environment."""

    mode: str

    def provision(self, credential: ScopedCredential, *, env_spec: str) -> dict[str, Any]:
        """Hand ``credential`` to the session and return the non-secret evidence."""
        ...

    def revoke(self, credential: ScopedCredential) -> None:
        """Invalidate the seam's copy and the credential itself."""
        ...

    def close(self) -> None:
        """Release any process or listener this seam owns."""
        ...


def _check_credential_scope(credential: ScopedCredential, expected: dict[str, str] | None) -> None:
    """Refuse a credential whose own scope does not match the session the seam serves."""
    if expected is None:
        return
    if any(str(credential.scope.get(key)) != str(value) for key, value in expected.items()):
        raise IsolationError("credential", "the scoped credential belongs to another session")


class NoCredentialSeam:
    """The default: no agent runs outside the control host, so there is nothing to hand out."""

    mode = "none"

    def provision(self, credential: ScopedCredential, *, env_spec: str) -> dict[str, Any]:
        raise IsolationError("credential", "harness=local never provisions a remote agent credential")

    def revoke(self, credential: ScopedCredential) -> None:
        return

    def close(self) -> None:
        return


class LoopbackFakeSeam:
    """A deterministic seam: a loopback fake provider and an in-process token reference."""

    mode = "loopback-fake"

    def __init__(
        self,
        authority: ScopedCredentialAuthority,
        *,
        provider: LoopbackFakeProvider | None = None,
        expected_scope: dict[str, str] | None = None,
    ) -> None:
        self.authority = authority
        self.expected_scope = dict(expected_scope) if expected_scope else None
        self.provider = provider or LoopbackFakeProvider(authority, expected_scope=self.expected_scope)

    def provision(self, credential: ScopedCredential, *, env_spec: str) -> dict[str, Any]:
        _check_credential_scope(credential, getattr(self, "expected_scope", None))
        self.provider.start()
        return {
            "mode": self.mode,
            "base_url": self.provider.base_url,
            "credential": credential.as_ref(),
            "token_in_env": False,
            # The loopback provider is a test double: nothing is handed to the far side, so
            # this record must not be read as evidence that a remote agent got a credential.
            "provisioned_to_agent": False,
            "agent_env": "in-process-test-double",
            "simulated": True,
            "env_spec_hash": hashlib.sha256(env_spec.encode("utf-8")).hexdigest(),
        }

    def revoke(self, credential: ScopedCredential) -> None:
        self.authority.revoke(credential.credential_id)

    def close(self) -> None:
        self.provider.close()


class ExternalHelperSeam:
    """A host-owned helper provisions the short-lived token on the far side.

    The helper is the operator's code; this project can only bound it, pass the token on
    stdin rather than argv, and revoke the credential afterwards.  It must not copy a
    long-term provider key into the remote environment; that promise is operator-attested.
    """

    mode = "external-helper"

    def __init__(
        self,
        authority: ScopedCredentialAuthority,
        helper: tuple[str, ...],
        *,
        timeout_s: float = 30.0,
        expected_scope: dict[str, str] | None = None,
        provider_base_url: str = "",
    ) -> None:
        if not helper:
            raise IsolationError("config", "external-helper needs an argv helper")
        self.authority = authority
        self.helper = tuple(helper)
        self.timeout_s = float(timeout_s)
        self.expected_scope = dict(expected_scope) if expected_scope else None
        self.provider_base_url = str(provider_base_url or "")
        self.calls: list[str] = []

    def _run(self, action: str, credential: ScopedCredential, *, env_spec: str) -> dict[str, Any]:
        document = {
            "schema_version": CREDENTIAL_SCHEMA_VERSION,
            "action": action,
            "credential_id": credential.credential_id,
            "token": credential.token,
            "scope": dict(credential.scope),
            "expires_at": credential.expires_at,
            "max_uses": credential.max_uses,
            "env_spec": env_spec,
        }
        if self.provider_base_url:
            document["provider_base_url"] = self.provider_base_url
        try:
            completed = subprocess.run(
                list(self.helper),
                input=json.dumps(document),
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise IsolationError("timeout", f"the credential helper did not finish within {self.timeout_s:g}s") from exc
        except OSError as exc:
            raise IsolationError("unavailable", f"could not start the credential helper: {exc}") from exc
        self.calls.append(action)
        if completed.returncode != 0:
            detail = redact_text((completed.stderr or completed.stdout or "").strip()[-400:], limit=400)
            # A helper may echo what it was given; the token must not reach a log or a result.
            detail = detail.replace(credential.token, "[redacted]")
            raise IsolationError("helper", f"the credential helper failed with exit {completed.returncode}: {detail}")
        try:
            answer = json.loads(completed.stdout or "{}")
        except ValueError:
            answer = {}
        return answer if isinstance(answer, dict) else {}

    def provision(self, credential: ScopedCredential, *, env_spec: str) -> dict[str, Any]:
        _check_credential_scope(credential, getattr(self, "expected_scope", None))
        answer = self._run("provision", credential, env_spec=env_spec)
        return {
            "mode": self.mode,
            "credential": credential.as_ref(),
            "token_in_env": bool(answer.get("token_in_env", True)),
            "provisioned_to_agent": True,
            "simulated": False,
            "helper": self.helper[0],
            "env_spec_hash": hashlib.sha256(env_spec.encode("utf-8")).hexdigest(),
        }

    def revoke(self, credential: ScopedCredential) -> None:
        try:
            self._run("revoke", credential, env_spec="")
        finally:
            self.authority.revoke(credential.credential_id)

    def close(self) -> None:
        return


def build_seam(
    config: IsolatedAgentConfig | None,
    *,
    authority: ScopedCredentialAuthority | None = None,
    scope: dict[str, str] | None = None,
) -> CredentialSeam:
    """Build the seam named by ``config``, or the no-op default."""
    if config is None or config.credential_seam == "none":
        return NoCredentialSeam()
    authority = authority or ScopedCredentialAuthority()
    if config.credential_seam == "loopback-fake":
        return LoopbackFakeSeam(authority, expected_scope=scope)
    if config.credential_seam == "external-helper":
        return ExternalHelperSeam(authority, config.helper, expected_scope=scope)
    raise IsolationError("config", f"unknown credential_seam {config.credential_seam!r}")


# -------------------------------------------------------------------------------- fence


@dataclass(frozen=True)
class TrustedEvaluationFence:
    """A structural check that the agent workspace cannot contain evaluator artifacts.

    ``trusted_paths`` are host-side paths the evaluation and self-check side owns: package
    code, the device lock, candidate reports, the outbox, receipts and the authority
    directory.  The check refuses any layout where the workspace and a trusted path contain
    one another.  It is a placement invariant, not an enforcement mechanism on the remote
    host.
    """

    trusted_paths: tuple[Path, ...]
    host_workspace: Path

    def __post_init__(self) -> None:
        workspace = Path(self.host_workspace).resolve()
        for path in self.trusted_paths:
            if not Path(path).is_absolute():
                raise IsolationError("fence", f"a trusted evaluation path is not absolute: {path}")
            resolved = Path(path).resolve()
            if resolved == workspace or resolved in workspace.parents or workspace in resolved.parents:
                raise IsolationError(
                    "fence",
                    f"trusted evaluation path {resolved} overlaps the agent workspace {workspace}",
                )

    def check(self) -> None:
        """Re-run the invariant; callers can use it as an explicit startup assertion."""
        for path in self.trusted_paths:
            resolved = Path(path).resolve()
            if resolved == self.host_workspace or resolved in self.host_workspace.parents or self.host_workspace in resolved.parents:
                raise IsolationError("fence", f"trusted evaluation path {resolved} overlaps the agent workspace")

    def refuse_write(self, target: Path) -> None:
        """Refuse a write whose resolved target is a trusted artifact."""
        resolved = Path(target).resolve()
        for path in self.trusted_paths:
            trusted = Path(path).resolve()
            if resolved == trusted or trusted in resolved.parents:
                raise IsolationError("fence", f"the agent workspace may not write the trusted evaluator artifact {trusted}")
            if resolved in trusted.parents and trusted != resolved:
                # Writing a directory above a trusted artifact could replace it.
                raise IsolationError("fence", f"the agent workspace may not write above the trusted evaluator artifact {trusted}")


def default_trusted_paths(run_dir: Path, *, repo_root: Path) -> tuple[Path, ...]:
    """The evaluator artifacts a run owns, as host-side paths.

    These are the files and directories the search and the trusted evaluator write: the
    candidate store, the evaluation outbox, the manifests and the ledgers, the evaluator
    package itself and the self-check authority directory.  The agent workspace is a
    sibling of the first group, so a correct layout is disjoint by construction.
    """
    from tcenv.selfcheck import authority_root

    run = Path(run_dir)
    fixed = [
        run / "candidates",
        run / "evaluation_outbox",
        run / "execution.json",
        run / "transcript.jsonl",
        run / "ledger.json",
        run / "result.json",
        run / "state.json",
        run / "control.json",
        run / "run_config.json",
        Path(repo_root) / "tcenv",
        Path(repo_root) / "taic",
        Path(repo_root) / "tests",
        authority_root(),
    ]
    return tuple(fixed)


__all__ = [
    "CREDENTIAL_SCHEMA_VERSION",
    "CredentialSeam",
    "ExternalHelperSeam",
    "IsolatedAgentConfig",
    "IsolationError",
    "LoopbackFakeProvider",
    "LoopbackFakeSeam",
    "NoCredentialSeam",
    "ScopedCredential",
    "ScopedCredentialAuthority",
    "TrustedEvaluationFence",
    "build_seam",
    "default_trusted_paths",
]
