"""Optional SSH workspace transport for one kernel-writing session.

The default search never imports this module's transport: ``kernel_once`` keeps its
``LocalEnv`` role, ``run_kernel_agent`` runs ``hmz exec`` in the canonical
``agent_work/<label>`` directory, and the outer loop evaluates the submission through the
same ``EvaluationExecutor``.  Everything here is reached only from an explicit
``--remote-workspace-config`` and it fails closed: an unavailable host, an unregistered
runtime or a receipt that belongs to another session never falls back to local.

The transport is deliberately small.  It stages a bounded, declared input manifest onto a
pre-provisioned host, reads a bounded artifact manifest back, and removes the session
directory only when the receipt it wrote there still matches.  It never carries a shell
command from the agent: every remote command is one of the fixed forms in this file, and
every path is validated before it is quoted into one.

Phase 3 proves transport and canonical artifact layout, not isolation.  Whether the inner
loop can reach control_host files is a question for the hmz fence on a real host; this
module records the evidence it can observe and nothing more.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import signal
import stat
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Sequence

from .redact import redact_text

SCHEMA_VERSION = 1

_RUNTIME_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}")
_RUN_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_SEGMENT = re.compile(r"[A-Za-z0-9._+-]{1,128}")
_SESSION_JSON = ".taic/session.json"
_MAX_DIAGNOSTIC_BYTES = 4_000
#: The receipt is our own bounded marker, not agent data, so it has its own small cap
#: instead of being held to the per-artifact byte budget.
_MAX_RECEIPT_BYTES = 65_536
_KINDS = frozenset(
    {
        "config",
        "unavailable",
        "timeout",
        "disconnected",
        "protocol",
        "path",
        "conflict",
        "manifest",
        "cleanup",
    }
)


class WorkspaceError(RuntimeError):
    """One bounded reason the remote workspace could not be used.

    ``kind`` is stable and typed so the caller can map it to an agent or harness failure
    without parsing text.  ``message`` is redacted and bounded: it may quote the remote
    diagnostic, never a credential.
    """

    def __init__(self, kind: str, message: str, *, returncode: int | None = None) -> None:
        if kind not in _KINDS:
            raise ValueError(f"unknown workspace error kind {kind!r}")
        self.kind = kind
        self.returncode = returncode
        self._raw = str(message)
        super().__init__(f"{kind}: {redact_text(message, limit=600)}")

    @property
    def message(self) -> str:
        """The redacted, bounded diagnostic this error carries."""
        return redact_text(self._raw, limit=600)


def _sha256(data: bytes) -> str:
    """Hex SHA-256 of exactly these bytes."""
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, UTF-8."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _safe_relative(path: str) -> str:
    """Validate one POSIX relative path from a manifest.

    Refuses absolute paths, ``..``, empty segments, backslashes and control characters.
    The remote side never sees a path that this function has not accepted, and the host
    side never reads one outside the canonical work directory.

    @raises WorkspaceError with kind ``path``.
    """
    text = str(path or "")
    if not text or text != text.strip():
        raise WorkspaceError("path", "an empty or padded manifest path was refused")
    if text.startswith("/") or text.startswith("\\") or "\\" in text:
        raise WorkspaceError("path", f"an absolute or non-POSIX manifest path was refused: {text!r}")
    if any(ord(character) < 0x20 for character in text) or "\x7f" in text:
        raise WorkspaceError("path", f"a control character in a manifest path was refused: {text!r}")
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise WorkspaceError("path", f"a climbing or empty path segment was refused: {text!r}")
    for part in parts:
        if not _SEGMENT.fullmatch(part):
            raise WorkspaceError("path", f"a manifest path segment outside the allowlist was refused: {part!r}")
    return "/".join(parts)


@dataclass(frozen=True)
class ManifestEntry:
    """One regular file in a staged or collected manifest."""

    path: str
    sha256: str
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True)
class WorkspaceManifest:
    """A bounded, ordered list of regular files and the digest of that list."""

    entries: tuple[ManifestEntry, ...]
    total_bytes: int
    digest: str

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(entry.path for entry in self.entries)


def build_manifest(
    work: Path,
    paths: Iterable[str],
    *,
    max_file_bytes: int = 2_000_000,
    max_total_bytes: int = 8_000_000,
) -> WorkspaceManifest:
    """Build the manifest for ``paths`` under ``work``.

    Only regular files are accepted.  A symlink, directory, device or socket is refused: a
    staged workspace must not be a way to smuggle an arbitrary control_host file, and a
    collected artifact must not be a way to read one either.

    @param work canonical host work directory.
    @param paths POSIX relative paths to include.
    @param max_file_bytes per-file byte cap.
    @param max_total_bytes total byte cap.
    @return the manifest, ordered by path.
    @raises WorkspaceError for an unsafe path, a missing file, a non-regular file or a cap.
    """
    root = Path(work)
    seen: set[str] = set()
    entries: list[ManifestEntry] = []
    total = 0
    for raw in paths:
        relative = _safe_relative(raw)
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise WorkspaceError("manifest", f"manifest input is missing: {relative}") from exc
        except OSError as exc:
            raise WorkspaceError("manifest", f"cannot stat manifest input {relative}: {exc.strerror}") from exc
        if stat.S_ISLNK(info.st_mode):
            raise WorkspaceError("path", f"manifest input is a symlink: {relative}")
        if not stat.S_ISREG(info.st_mode):
            raise WorkspaceError("path", f"manifest input is not a regular file: {relative}")
        if info.st_size > max_file_bytes:
            raise WorkspaceError("manifest", f"manifest input is over {max_file_bytes} bytes: {relative}")
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise WorkspaceError("manifest", f"cannot read manifest input {relative}: {exc.strerror}") from exc
        if len(data) != info.st_size:
            raise WorkspaceError("manifest", f"manifest input changed while it was read: {relative}")
        total += len(data)
        if total > max_total_bytes:
            raise WorkspaceError("manifest", f"manifest is over {max_total_bytes} bytes")
        entries.append(ManifestEntry(path=relative, sha256=_sha256(data), size=len(data)))
    entries.sort(key=lambda entry: entry.path)
    document = {"schema_version": SCHEMA_VERSION, "files": [entry.as_dict() for entry in entries]}
    return WorkspaceManifest(entries=tuple(entries), total_bytes=total, digest=_sha256(_canonical_json(document)))


def check_relative_destination(work: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``work`` and refuse anything that escapes it.

    The host-side collection path is checked before a byte is written: the agent's file
    layout is untrusted, so ``a/../../..`` and symlinked parents are refused here.
    """
    clean = _safe_relative(relative)
    root = Path(work).resolve()
    target = (root / clean).resolve(strict=False)
    if target != root and root not in target.parents:
        raise WorkspaceError("path", f"collected path escapes the work directory: {relative}")
    return target


def workspace_root(host_work: Path) -> str:
    """One stable identifier for the canonical host work directory."""
    return _sha256(str(Path(host_work).resolve()).encode("utf-8"))


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class RemoteWorkspaceConfig:
    """Host-owned settings for the optional SSH workspace.

    The config names a runtime saved in hmz's own settings (``runtimes: ssh: <name>``),
    never a credential: the ssh identity, port and host belong to that runtime entry and
    stay in the operator's files.  ``harness`` records where the agent CLI is expected to
    run and is checked against the runtime's saved setting before any session starts, so a
    Phase 3 (``local``) run cannot silently become a Phase 4 (``env``) run or the reverse.
    """

    runtime: str
    root: str
    harness: str = "local"
    ssh_binary: str = "ssh"
    settings_file: Path | None = None
    connect_timeout_s: float = 15.0
    command_timeout_s: float = 60.0
    max_file_bytes: int = 2_000_000
    max_total_bytes: int = 8_000_000
    keep_workspace: bool = False
    #: Set only for the optional Phase 4 remote-agent placement.
    isolation: Any = None

    def __post_init__(self) -> None:
        if not _RUNTIME_NAME.fullmatch(str(self.runtime)):
            raise WorkspaceError("config", "remote workspace runtime must be a saved hmz runtime name")
        root = str(self.root)
        if not root.startswith("/") or root.startswith("//"):
            raise WorkspaceError("config", "remote workspace root must be an absolute POSIX path")
        if any(ord(character) < 0x20 for character in root) or "\\" in root:
            raise WorkspaceError("config", "remote workspace root has a control character or backslash")
        parts = [part for part in root.split("/") if part]
        if len(parts) < 2 or any(part in (".", "..") for part in parts):
            raise WorkspaceError("config", "remote workspace root must be a plain directory at least two levels deep")
        if not re.fullmatch(r"[A-Za-z0-9._+/-]+", root):
            raise WorkspaceError("config", "remote workspace root is outside the allowlist")
        if self.harness not in ("local", "env"):
            raise WorkspaceError("config", "remote workspace harness must be local or env")
        if not re.fullmatch(r"[A-Za-z0-9._+/-]+", str(self.ssh_binary)):
            raise WorkspaceError("config", "ssh binary must be a plain path or name")
        for name in ("connect_timeout_s", "command_timeout_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < float(value) <= 3_600:
                raise WorkspaceError("config", f"{name} must be positive and bounded")
        for name in ("max_file_bytes", "max_total_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise WorkspaceError("config", f"{name} must be a positive integer")
        if self.max_file_bytes > self.max_total_bytes:
            raise WorkspaceError("config", "max_file_bytes cannot exceed max_total_bytes")
        if not isinstance(self.keep_workspace, bool):
            raise WorkspaceError("config", "keep_workspace must be a boolean")
        from .agent_isolation import IsolatedAgentConfig, IsolationError

        if self.isolation is not None and not isinstance(self.isolation, IsolatedAgentConfig):
            raise WorkspaceError("config", "isolation must be an IsolatedAgentConfig")
        try:
            if self.harness == "env":
                if self.isolation is None or not self.isolation.requires_remote_agent:
                    raise WorkspaceError(
                        "config",
                        "harness=env needs an explicit isolation credential seam; there is no "
                        "silent remote-agent credential",
                    )
            elif self.isolation is not None and self.isolation.credential_seam != "none":
                raise WorkspaceError("config", "harness=local cannot provision a remote agent credential")
        except IsolationError as exc:
            raise WorkspaceError("config", exc.args[0]) from exc

    @classmethod
    def from_file(cls, path: Path | str) -> "RemoteWorkspaceConfig":
        """Read one explicit JSON config; unknown fields are refused, not ignored."""
        source = Path(path).resolve()
        try:
            document = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise WorkspaceError("config", f"cannot read remote workspace config: {exc}") from exc
        if not isinstance(document, dict):
            raise WorkspaceError("config", "remote workspace config must be an object")
        allowed = {
            "runtime", "root", "harness", "ssh_binary", "settings_file",
            "connect_timeout_s", "command_timeout_s", "max_file_bytes", "max_total_bytes",
            "keep_workspace", "isolation",
        }
        unknown = set(document) - allowed
        if unknown:
            raise WorkspaceError("config", f"unknown remote workspace fields: {sorted(unknown)}")
        for name in ("runtime", "root"):
            value = document.get(name)
            if not isinstance(value, str) or not value:
                raise WorkspaceError("config", f"remote workspace {name} is required")
        settings = document.get("settings_file")
        resolved: Path | None = None
        if settings is not None:
            if not isinstance(settings, str) or not settings:
                raise WorkspaceError("config", "settings_file must be a path")
            candidate = Path(settings)
            resolved = candidate if candidate.is_absolute() else source.parent / candidate
        for name in ("connect_timeout_s", "command_timeout_s"):
            value = document.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))):
                raise WorkspaceError("config", f"{name} must be a number")
        for name in ("max_file_bytes", "max_total_bytes"):
            value = document.get(name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise WorkspaceError("config", f"{name} must be an integer")
        if not isinstance(document.get("keep_workspace", False), bool):
            raise WorkspaceError("config", "keep_workspace must be a boolean")
        from .agent_isolation import IsolatedAgentConfig, IsolationError

        try:
            isolation = IsolatedAgentConfig.from_dict(document["isolation"]) if "isolation" in document else None
        except IsolationError as exc:
            raise WorkspaceError("config", exc.args[0]) from exc
        return cls(
            runtime=document["runtime"],
            root=document["root"],
            harness=document.get("harness", "local"),
            ssh_binary=document.get("ssh_binary", "ssh"),
            settings_file=resolved,
            connect_timeout_s=float(document.get("connect_timeout_s", 15.0)),
            command_timeout_s=float(document.get("command_timeout_s", 60.0)),
            max_file_bytes=int(document.get("max_file_bytes", 2_000_000)),
            max_total_bytes=int(document.get("max_total_bytes", 8_000_000)),
            keep_workspace=document.get("keep_workspace", False),
            isolation=isolation,
        )


@dataclass(frozen=True)
class SavedRuntime:
    """The parts of one saved hmz ssh runtime this transport reads.

    Only destination facts are kept.  Key material is referenced by path exactly as the
    operator wrote it and is never copied, logged or put in a manifest.
    """

    name: str
    harness: str
    host: str = ""
    user: str = ""
    port: int = 0
    identity_file: str = ""
    proxy_jump: str = ""
    alias: str = ""
    config: str = ""
    workdir: str = ""
    options: dict[str, str] = field(default_factory=dict)

    @property
    def destination(self) -> str:
        """The host argument for ``ssh``: the imported alias, or ``user@host``."""
        if self.alias:
            return self.alias
        return f"{self.user}@{self.host}" if self.user else self.host

    def argv(self, binary: str, command: str, *, connect_timeout_s: float) -> list[str]:
        """The fixed ssh command that reaches this runtime and runs ``command``."""
        argv = [binary, "-T", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={int(connect_timeout_s)}"]
        if self.alias:
            # An imported runtime keeps its ssh config: `ssh` is told nothing it would
            # otherwise have read, and the config goes on being what it says.
            if self.config:
                argv.extend(["-F", self.config])
        else:
            if self.port:
                argv.extend(["-p", str(self.port)])
            if self.identity_file:
                argv.extend(["-i", self.identity_file])
            if self.proxy_jump:
                argv.extend(["-J", self.proxy_jump])
            for key, value in sorted(self.options.items()):
                argv.extend(["-o", f"{key}={value}"])
        argv.append(self.destination)
        argv.append(command)
        return argv


def settings_path(config: RemoteWorkspaceConfig) -> Path:
    """Where hmz keeps its settings, overridable for a test or a second home."""
    if config.settings_file is not None:
        return Path(config.settings_file)
    explicit = os.environ.get("HUMANIZE_HOME")
    home = Path(explicit) if explicit else Path.home() / ".hmz"
    return home / "settings.yaml"


def read_runtime(config: RemoteWorkspaceConfig) -> SavedRuntime:
    """Read one saved ssh runtime and refuse it unless it can carry this placement.

    This is a read-only preflight, run before any session starts.  It is what keeps a
    Phase 3 transport config from silently running the agent on the host (or the reverse):
    the runtime's saved ``harness`` setting must equal the config's.

    @raises WorkspaceError with kind ``config`` for a missing runtime, a non-ssh or
        malformed entry, or a harness mismatch.
    """
    path = settings_path(config)
    try:
        import yaml  # noqa: PLC0415 -- hmz's own dependency; absent only without hmz
    except ImportError as exc:  # pragma: no cover - hmz requires pyyaml
        raise WorkspaceError("config", "cannot read hmz settings: pyyaml is not installed") from exc
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise WorkspaceError("config", f"hmz has no settings file at {path}; save the ssh runtime first") from exc
    except (OSError, yaml.YAMLError) as exc:
        raise WorkspaceError("config", f"cannot read hmz settings at {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise WorkspaceError("config", f"hmz settings at {path} are not an object")
    runtimes = document.get("runtimes")
    ssh = runtimes.get("ssh") if isinstance(runtimes, dict) else None
    entry = ssh.get(config.runtime) if isinstance(ssh, dict) else None
    if not isinstance(entry, dict):
        raise WorkspaceError("config", f"hmz has no saved ssh runtime {config.runtime!r} in {path}")
    harness = entry.get("harness", "env")
    if harness not in ("local", "env"):
        raise WorkspaceError("config", f"runtime {config.runtime!r} has an unknown harness setting {harness!r}")
    if harness != config.harness:
        raise WorkspaceError(
            "config",
            f"runtime {config.runtime!r} saves harness={harness!r}, but this config asks for "
            f"harness={config.harness!r}; refusing to change where the agent runs",
        )
    host = str(entry.get("host") or "")
    alias = str(entry.get("alias") or "")
    if not host and not alias:
        raise WorkspaceError("config", f"runtime {config.runtime!r} names neither a host nor an alias")
    port = entry.get("port") or 0
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
        raise WorkspaceError("config", f"runtime {config.runtime!r} has an invalid port")
    options = entry.get("options") or {}
    if not isinstance(options, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in options.items()):
        raise WorkspaceError("config", f"runtime {config.runtime!r} has invalid ssh options")
    for key in options:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9]*", key):
            raise WorkspaceError("config", f"runtime {config.runtime!r} has an invalid ssh option name")
    saved_workdir = str(entry.get("workdir") or "").rstrip("/")
    if saved_workdir and saved_workdir != config.root.rstrip("/"):
        raise WorkspaceError(
            "config",
            f"runtime {config.runtime!r} saves workdir {saved_workdir!r}, but this config stages "
            f"under {config.root!r}; refusing to stage one directory and run the agent in another",
        )
    return SavedRuntime(
        name=config.runtime,
        harness=harness,
        host=host,
        user=str(entry.get("user") or ""),
        port=int(port),
        identity_file=str(entry.get("identity_file") or ""),
        proxy_jump=str(entry.get("proxy_jump") or ""),
        alias=alias,
        config=str(entry.get("config") or ""),
        workdir=saved_workdir,
        options={str(k): str(v) for k, v in options.items()},
    )


# ------------------------------------------------------------------------------ receipts


@dataclass(frozen=True)
class WorkspaceReceipt:
    """The identity a remote session directory belongs to.

    The receipt is written before staging and read back before cleanup.  A mismatch is a
    conflict, never a silent overwrite: two runs must not share or remove each other's
    workspace, and a resumed session must be able to prove which one it is re-entering.
    """

    document: dict[str, Any]
    receipt_id: str

    #: The fields the receipt id is computed over; ``receipt_id`` itself is not one of them.
    IDENTITY_KEYS = (
        "schema_version", "run_id", "session_label", "session_id",
        "workspace", "manifest_digest", "prompt_surface_sha256", "epic",
    )

    @classmethod
    def build(
        cls,
        *,
        run_id: str,
        session_label: str,
        session_id: str,
        runtime: str,
        workdir: str,
        host_work: Path,
        manifest: WorkspaceManifest,
        prompt_surface_sha256: str,
        epic: str = "",
    ) -> "WorkspaceReceipt":
        identity = {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "session_label": session_label,
            "session_id": session_id,
            "workspace": {
                "backend": "ssh",
                "runtime": runtime,
                "workdir": workdir,
                "host_work_sha256": workspace_root(host_work),
            },
            "manifest_digest": manifest.digest,
            "prompt_surface_sha256": prompt_surface_sha256,
            "epic": epic,
        }
        receipt_id = _sha256(_canonical_json(identity))
        document = dict(identity)
        document["receipt_id"] = receipt_id
        return cls(document=document, receipt_id=receipt_id)

    @classmethod
    def from_document(cls, document: object) -> "WorkspaceReceipt":
        """Validate one stored receipt and recompute its id from its own fields.

        A remote agent can edit ``.taic/session.json``.  The stored ``receipt_id`` is
        therefore never trusted on its own: the digest is recomputed over the identity
        fields, and a mismatch is a conflict (a broken receipt eligible for marker-based
        cleanup), not an authoritative claim about which session owns the directory.

        @raises WorkspaceError with kind ``conflict`` for a malformed or tampered receipt.
        """
        if not isinstance(document, dict):
            raise WorkspaceError("conflict", "the remote session receipt is not an object")
        identity: dict[str, Any] = {}
        for key in cls.IDENTITY_KEYS:
            if key not in document:
                raise WorkspaceError("conflict", f"the remote session receipt has no {key}")
            identity[key] = document[key]
        if not isinstance(identity["schema_version"], int) or identity["schema_version"] != SCHEMA_VERSION:
            raise WorkspaceError("conflict", "the remote session receipt has an unknown schema version")
        for key in ("run_id", "session_label", "session_id", "manifest_digest", "prompt_surface_sha256", "epic"):
            if not isinstance(identity[key], str):
                raise WorkspaceError("conflict", f"the remote session receipt has a non-string {key}")
        workspace = identity["workspace"]
        if not isinstance(workspace, dict) or any(
            not isinstance(workspace.get(key), str)
            for key in ("backend", "runtime", "workdir", "host_work_sha256")
        ):
            raise WorkspaceError("conflict", "the remote session receipt has a malformed workspace")
        stored = document.get("receipt_id")
        computed = _sha256(_canonical_json(identity))
        if not isinstance(stored, str) or stored != computed:
            raise WorkspaceError("conflict", "the remote session receipt id does not match its contents")
        return cls(document=dict(document), receipt_id=computed)

    def json_bytes(self) -> bytes:
        return _canonical_json(self.document) + b"\n"


# ----------------------------------------------------------------------------- transport


def _require_absolute_workdir(root: str, run_token: str, label: str) -> str:
    """Build ``<root>/<run_token>/<label>`` and refuse anything that is not one."""
    for value, what in ((run_token, "run token"), (label, "session label")):
        if not _RUN_TOKEN.fullmatch(str(value)):
            raise WorkspaceError("path", f"the {what} {value!r} is outside the allowlist")
    workdir = PurePosixPath(root) / run_token / label
    text = str(workdir)
    if not text.startswith("/") or ".." in text.split("/") or text == "/":
        raise WorkspaceError("path", f"the remote workdir {text!r} was refused")
    return text


def _quote(path: str) -> str:
    """Quote one validated remote path for a fixed shell command."""
    return shlex.quote(path)


def _bounded_text(raw: bytes, limit: int = _MAX_DIAGNOSTIC_BYTES) -> str:
    """Decode a remote diagnostic and keep at most ``limit`` bytes, redacted."""
    text = raw[:limit].decode("utf-8", "replace").strip()
    return redact_text(text, limit=600)


def _read_bounded(stream, limit: int) -> bytes:
    """Read at most ``limit`` bytes from a pipe."""
    chunks: list[bytes] = []
    read = 0
    while read < limit:
        chunk = stream.read(min(65_536, limit - read))
        if not chunk:
            break
        chunks.append(chunk)
        read += len(chunk)
    return b"".join(chunks)


class SshWorkspaceTransport:
    """Fixed-command SSH file transport for one session workspace.

    Every method runs its own bounded ``ssh`` process, so a disconnect leaves a known
    state: either the fixed command completed or the return code says it did not.  The
    transport keeps no ssh master connection and no long-lived process, which is what lets
    a timeout kill the whole process group and leave nothing behind on this machine.
    """

    def __init__(
        self,
        config: RemoteWorkspaceConfig,
        *,
        runtime: SavedRuntime | None = None,
        popen: Callable[..., subprocess.Popen] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.runtime = runtime or read_runtime(config)
        self._popen = popen or subprocess.Popen
        self._sleep = sleep
        self.commands: list[str] = []

    # -- one bounded process ---------------------------------------------------------

    def _run(self, command: str, *, stdin_bytes: bytes | None = None, timeout_s: float | None = None) -> tuple[int, bytes, bytes]:
        """Run one fixed remote command; return (returncode, stdout, stderr).

        On timeout the local process group is killed and a typed ``timeout`` is raised.  A
        non-zero return code is returned to the caller, which decides whether it means a
        missing file, a refused command or a broken host.
        """
        self.commands.append(command)
        argv = self.runtime.argv(self.config.ssh_binary, command, connect_timeout_s=self.config.connect_timeout_s)
        timeout = float(timeout_s if timeout_s is not None else self.config.command_timeout_s)
        try:
            proc = self._popen(
                argv,
                stdin=subprocess.PIPE if stdin_bytes is not None else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise WorkspaceError("unavailable", f"could not start ssh: {exc}") from exc
        try:
            out_chunks: list[bytes] = []
            err_chunks: list[bytes] = []
            write_error: list[BaseException] = []
            # Every pipe is served by its own thread so neither a full input pipe nor a
            # silent host can deadlock the caller, and each read is bounded: a hostile file
            # cannot turn the answer into unbounded memory.
            import threading  # noqa: PLC0415 -- one small import per command

            def drain(pipe, sink: list[bytes]) -> None:
                if pipe is None:
                    return
                sink.append(_read_bounded(pipe, self.config.max_file_bytes + 65_536))

            def feed(pipe, payload: bytes) -> None:
                if pipe is None:
                    return
                try:
                    pipe.write(payload)
                except (BrokenPipeError, OSError) as exc:
                    write_error.append(exc)
                finally:
                    try:
                        pipe.close()
                    except OSError:
                        pass

            threads = [
                threading.Thread(target=drain, args=(proc.stdout, out_chunks), daemon=True),
                threading.Thread(target=drain, args=(proc.stderr, err_chunks), daemon=True),
            ]
            if stdin_bytes is not None:
                threads.append(threading.Thread(target=feed, args=(proc.stdin, stdin_bytes), daemon=True))
            for thread in threads:
                thread.start()
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired as exc:
                _kill_group(proc)
                raise WorkspaceError("timeout", f"the remote command did not finish within {timeout:g}s") from exc
            for thread in threads:
                thread.join(timeout=5)
            if write_error and int(code) == 0:
                raise WorkspaceError("disconnected", f"the remote side closed the input pipe: {write_error[0]}")
            return int(code), b"".join(out_chunks), b"".join(err_chunks)
        finally:
            _kill_group(proc)
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except OSError:
                    pass

    def _checked(self, command: str, *, stdin_bytes: bytes | None = None, timeout_s: float | None = None, what: str = "remote command") -> bytes:
        """Run ``command`` and raise unless it exits zero."""
        code, out, err = self._run(command, stdin_bytes=stdin_bytes, timeout_s=timeout_s)
        if code != 0:
            detail = _bounded_text(err) or _bounded_text(out) or "no diagnostic"
            raise WorkspaceError("protocol", f"{what} failed with exit {code}: {detail}", returncode=code)
        return out

    # -- fixed commands --------------------------------------------------------------

    def _segments(self, workdir: str, extra: Iterable[str] = ()) -> list[str]:
        """Split ``workdir`` into checked components below the configured root."""
        root = self.config.root.rstrip("/") or "/"
        prefix = root + "/"
        if not workdir.startswith(prefix):
            raise WorkspaceError("path", f"the remote workdir {workdir!r} is not under the configured root")
        parts = [part for part in workdir[len(prefix):].split("/") if part]
        for part in parts:
            if not _RUN_TOKEN.fullmatch(part):
                raise WorkspaceError("path", f"the remote path component {part!r} is outside the allowlist")
        for part in extra:
            text = str(part)
            if text in (".", "..") or not _SEGMENT.fullmatch(text):
                raise WorkspaceError("path", f"the remote path component {text!r} is outside the allowlist")
            parts.append(text)
        return parts

    def _walk(self, workdir: str, extra: Iterable[str] = (), *, final: str = "") -> str:
        """A fixed shell walk from the configured root, refusing every symlinked component.

        A leaf-only ``test ! -L`` is not enough: a symlinked session directory or a symlinked
        subdirectory would still redirect a read.  The walk covers every component below the
        operator-owned root; nothing an agent can create changes where a path resolves.
        """
        parts = self._segments(workdir, extra)
        pieces = [f"p={_quote(self.config.root)}", '[ ! -L "$p" ] || { echo taic: the remote workspace root is a symlink >&2; exit 4; }']
        for part in parts:
            pieces.append(f'p="$p"/{_quote(part)}')
            pieces.append('[ ! -L "$p" ] || { echo taic: a path component below the remote workspace root is a symlink >&2; exit 4; }')
        if final:
            pieces.append(final)
        return "; ".join(pieces)

    def make_workdir(self, workdir: str) -> None:
        """Create the session workdir and its marker parent.

        A symlink where one of those directories belongs is refused: an agent that replaced
        its session directory between two attempts must not be able to redirect a receipt or
        a staged input to another path on the host.
        """
        # The walk runs before every mkdir: a symlinked parent is refused before anything is
        # created through it, and again after creation to close the obvious race.
        self._checked(
            f"umask 077; {self._walk(workdir)}; mkdir -p -- {_quote(workdir)} || exit 1; "
            f"{self._walk(workdir)}; {self._walk(workdir, ['.taic'])}; "
            f"mkdir -p -- {_quote(workdir + '/.taic')} || exit 1; {self._walk(workdir, ['.taic'])}",
            what="create the remote workspace",
        )

    def write_receipt(self, workdir: str, receipt: WorkspaceReceipt) -> None:
        """Write the receipt atomically, before any input file is staged."""
        self._write(workdir, _SESSION_JSON, receipt.json_bytes())

    def read_receipt(self, workdir: str) -> WorkspaceReceipt | None:
        """Read the receipt, or ``None`` when this directory has none."""
        data = self._read(workdir, _SESSION_JSON, limit=_MAX_RECEIPT_BYTES)
        if data is None:
            return None
        try:
            document = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise WorkspaceError("conflict", f"the remote session receipt is not readable JSON: {exc}") from exc
        return WorkspaceReceipt.from_document(document)

    def stage(self, workdir: str, host_work: Path, manifest: WorkspaceManifest, receipt: WorkspaceReceipt) -> None:
        """Stage every manifest file, verifying the receipt first.

        The receipt is read before and after staging.  A directory that belongs to another
        session is refused; the same receipt is written again and the files are replaced
        atomically, which makes a re-stage of an interrupted session idempotent.
        """
        self.make_workdir(workdir)
        existing = self.read_receipt(workdir)
        if existing is not None and existing.receipt_id != receipt.receipt_id:
            raise WorkspaceError(
                "conflict",
                f"the remote workspace {workdir} belongs to another session receipt",
            )
        self.write_receipt(workdir, receipt)
        for entry in manifest.entries:
            source = Path(host_work) / entry.path
            try:
                data = source.read_bytes()
            except OSError as exc:
                raise WorkspaceError("manifest", f"cannot read staged input {entry.path}: {exc.strerror}") from exc
            if len(data) != entry.size or _sha256(data) != entry.sha256:
                raise WorkspaceError("manifest", f"staged input changed after the manifest was built: {entry.path}")
            self._write(workdir, entry.path, data)
        confirmed = self.read_receipt(workdir)
        if confirmed is None or confirmed.receipt_id != receipt.receipt_id:
            raise WorkspaceError("protocol", "the remote session receipt did not survive staging")

    def collect(self, workdir: str, paths: Iterable[str]) -> dict[str, bytes]:
        """Read the declared artifact paths, skipping missing ones.

        Each file is sized before it is read and refused if it is over the per-file cap.
        """
        found: dict[str, bytes] = {}
        total = 0
        for raw in paths:
            relative = _safe_relative(raw)
            data = self._read(workdir, relative)
            if data is None:
                continue
            total += len(data)
            if total > self.config.max_total_bytes:
                raise WorkspaceError(
                    "manifest",
                    f"the collected artifacts are over {self.config.max_total_bytes} bytes",
                )
            found[relative] = data
        return found

    def list_missing(self, workdir: str, paths: Iterable[str]) -> list[str]:
        """Which declared paths are absent, in one fixed command per path."""
        missing: list[str] = []
        for raw in paths:
            relative = _safe_relative(raw)
            code, _out, _err = self._run(f"test -f {_quote(workdir + '/' + relative)}")
            if code != 0:
                missing.append(relative)
        return missing

    def exists(self, workdir: str) -> bool:
        """Whether the session directory is still there."""
        code, _out, _err = self._run(f"test -e {_quote(workdir)}")
        return code == 0

    def remove(self, workdir: str, receipt: WorkspaceReceipt | None) -> bool:
        """Remove the session directory when it is one of ours.

        Returns whether it was removed.  A directory with no marker at all is left alone: it
        is not ours to delete.  A broken or symlinked receipt cannot make cleanup refuse; the
        marker still proves this is a session directory under our root, and ``rm -rf`` on a
        symlink removes the link rather than following it.
        """
        current: WorkspaceReceipt | None = None
        broken = False
        try:
            current = self.read_receipt(workdir)
        except WorkspaceError as exc:
            # A symlinked or tampered receipt cannot make cleanup refuse.  The directory is
            # still under our run token and carries a marker; `rm -rf` does not follow the
            # link, so removing it is safe.
            if exc.kind not in ("path", "conflict"):
                raise
            broken = True
        if current is None and not broken:
            if not self.exists(workdir):
                return False
            marker_code, _out, _err = self._run(
                f"test -L {_quote(workdir + '/.taic')} || test -e {_quote(workdir + '/.taic')}"
            )
            if marker_code != 0:
                raise WorkspaceError("conflict", f"the remote workspace {workdir} has no receipt; leaving it alone")
        if receipt is not None and current is not None and current.receipt_id != receipt.receipt_id:
            raise WorkspaceError("conflict", f"the remote workspace {workdir} belongs to another session receipt")
        self._checked(f"rm -rf -- {_quote(workdir)}", what="remove the remote workspace")
        if self.exists(workdir):
            raise WorkspaceError("cleanup", f"the remote workspace {workdir} survived removal")
        return True

    # -- orphan reaping --------------------------------------------------------------

    def list_receipts(self, run_token: str) -> list[tuple[str, WorkspaceReceipt | None]]:
        """List the session directories this run's token owns on the host.

        The command is fixed and bounded; every returned path is checked to be a session
        directory under ``<root>/<run token>`` before its receipt is read.
        """
        if not _RUN_TOKEN.fullmatch(str(run_token)):
            raise WorkspaceError("path", f"the run token {run_token!r} is outside the allowlist")
        root = str(PurePosixPath(self.config.root) / run_token)
        code, out, err = self._run(f"find {_quote(root)} -mindepth 3 -maxdepth 3 -name session.json -print")
        if code != 0:
            _code, _out, _err = self._run(f"test -e {_quote(root)}")
            if _code != 0:
                return []
            raise WorkspaceError("unavailable", f"cannot list the remote run root: {_bounded_text(err)}")
        found: list[tuple[str, WorkspaceReceipt | None]] = []
        prefix = root + "/"
        for line in out.decode("utf-8", "replace").splitlines():
            path = line.strip()
            if not path.startswith(prefix) or not path.endswith("/.taic/session.json"):
                continue
            workdir = path[: -len("/.taic/session.json")]
            label = workdir[len(prefix):].split("/", 1)[0]
            if not _RUN_TOKEN.fullmatch(label) or "/" in label:
                continue
            try:
                receipt = self.read_receipt(workdir)
            except WorkspaceError:
                # A broken or symlinked receipt must not make its directory unreapable.
                receipt = None
            found.append((workdir, receipt))
            if len(found) >= 256:
                break
        return found

    def reap_orphans(self, run_token: str, run_id: str, active_labels: Iterable[str]) -> list[str]:
        """Remove this run's session directories whose labels are no longer active.

        A directory whose receipt belongs to another run is left alone; so is one without a
        receipt.  Returns the labels that were removed.
        """
        active = {str(label) for label in active_labels}
        removed: list[str] = []
        for workdir, receipt in self.list_receipts(run_token):
            if receipt is not None and str(receipt.document.get("run_id") or "") != str(run_id):
                continue
            label = workdir.rsplit("/", 1)[-1]
            if label in active:
                continue
            if self.remove(workdir, receipt):
                removed.append(label)
        return removed

    # -- file primitives -------------------------------------------------------------

    def _write(self, workdir: str, relative: str, data: bytes) -> None:
        """Atomically replace one file under ``workdir``.

        The temporary name comes from ``mktemp``, so a file or symlink an agent left at a
        predictable temporary path cannot capture the write; a symlinked parent directory is
        refused for the same reason.
        """
        clean = _safe_relative(relative)
        target = workdir + "/" + clean
        parent_parts = clean.split("/")[:-1]
        parent = target.rsplit("/", 1)[0]
        command = (
            f"umask 077; mkdir -p -- {_quote(parent)}; {self._walk(workdir, parent_parts)}; "
            f"temp=$(mktemp -- {_quote(parent + '/.taic.XXXXXX')}) && "
            f"{{ cat > \"$temp\" && mv -f -- \"$temp\" {_quote(target)}; }} || "
            f"{{ [ -z \"$temp\" ] || rm -f -- \"$temp\"; exit 1; }}"
        )
        self._checked(command, stdin_bytes=data, what=f"write {clean}")

    def _read(self, workdir: str, relative: str, *, limit: int | None = None) -> bytes | None:
        """Read one file under ``workdir``, or ``None`` when it is not there.

        ``limit`` overrides the per-artifact byte cap for transport-owned markers.
        """
        cap = self.config.max_file_bytes if limit is None else int(limit)
        clean = _safe_relative(relative)
        target = workdir + "/" + clean
        code, _out, err = self._run(self._walk(workdir, clean.split("/"), final='[ -f "$p" ] || exit 3'))
        if code == 4:
            raise WorkspaceError("path", f"the remote artifact {clean} has a symlinked component and was refused")
        if code != 0:
            if code == 3:
                return None
            raise WorkspaceError("unavailable", f"cannot reach the host to read {clean}: {_bounded_text(err)}", returncode=code)
        code, out, err = self._run(f"wc -c -- {_quote(target)}")
        if code != 0:
            text = _bounded_text(err) or "no diagnostic"
            raise WorkspaceError("protocol", f"cannot size {clean}: {text}", returncode=code)
        try:
            size = int(out.decode("ascii", "replace").split()[0])
        except (IndexError, ValueError) as exc:
            raise WorkspaceError("protocol", f"cannot parse the size of {clean}") from exc
        if size > cap:
            raise WorkspaceError("manifest", f"{clean} is over {cap} bytes")
        code, data, err = self._run(f"cat -- {_quote(target)}", timeout_s=self.config.command_timeout_s)
        if code != 0:
            detail = _bounded_text(err) or "no diagnostic"
            raise WorkspaceError("protocol", f"cannot read {clean}: {detail}", returncode=code)
        if len(data) > cap:
            raise WorkspaceError("manifest", f"{clean} grew past {cap} bytes while being read")
        if len(data) != size:
            raise WorkspaceError("protocol", f"{clean} changed size while being read")
        return data


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the ssh process group; a disconnect must leave no local writer behind."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        try:
            proc.kill()
        except OSError:
            pass


# ------------------------------------------------------------------------------- session


def run_token(run_dir: Path) -> str:
    """A stable directory name for one run: a readable prefix and a path digest."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(run_dir).name).strip("-.")[:40] or "run"
    digest = _sha256(str(Path(run_dir).resolve()).encode("utf-8"))[:12]
    return f"{name}-{digest}"


@dataclass(frozen=True)
class RemoteSessionLayout:
    """Where one session lives on both sides of the transport."""

    host_work: Path
    workdir: str
    run_token: str
    session_label: str


def session_layout(config: RemoteWorkspaceConfig, *, run_dir: Path, host_work: Path, session_label: str) -> RemoteSessionLayout:
    """Build and validate the remote and host paths for one session."""
    token = run_token(run_dir)
    workdir = _require_absolute_workdir(config.root, token, session_label)
    return RemoteSessionLayout(host_work=Path(host_work), workdir=workdir, run_token=token, session_label=session_label)


def hmz_env_spec(config: RemoteWorkspaceConfig, workdir: str) -> str:
    """The single ``-e`` mapping that puts the flow's workspace on the runtime.

    ``kernel_once`` declares ``remote_workspace`` as an optional environment role; the role
    is absent for local runs, so the default argv does not grow this flag.
    """
    if not workdir.startswith("/") or ".." in workdir.split("/"):
        raise WorkspaceError("path", f"the remote workdir {workdir!r} was refused")
    return f"remote_workspace=ssh@{config.runtime}{workdir}"


def prompt_surface_digest(work: Path, flow_dir: Path) -> tuple[str, str, str]:
    """Hash the prompt surfaces this session actually offers.

    ``TASK.md`` is rendered per session and is the contract; the flow's ``skills/`` tree is
    delivered to the agent's virtual home, so it is part of the same surface even though it
    does not live in the workspace.  Both are recorded per session so a remote run can be
    trusted separately from a local one.

    @return ``(task_sha256, skill_sha256, combined_sha256)``.
    """
    task = Path(work) / "TASK.md"
    try:
        task_bytes = task.read_bytes()
    except OSError as exc:
        raise WorkspaceError("manifest", f"cannot read the rendered TASK.md: {exc.strerror}") from exc
    skill_digest = hashlib.sha256()
    skill_root = Path(flow_dir) / "skills"
    if skill_root.is_dir():
        for path in sorted(skill_root.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(skill_root).as_posix()
            skill_digest.update(relative.encode("utf-8") + b"\0")
            try:
                skill_digest.update(path.read_bytes())
            except OSError:
                # The local runner never hashed skill files before this profile existed; an
                # unreadable one is recorded as such rather than turning into a new failure
                # before the session starts.
                skill_digest.update(b"<unreadable>")
    skill_sha = skill_digest.hexdigest()
    task_sha = _sha256(task_bytes)
    combined = _sha256(_canonical_json({"task_sha256": task_sha, "skill_sha256": skill_sha}))
    return task_sha, skill_sha, combined


# --------------------------------------------------------------------------------- epic


@dataclass(frozen=True)
class EpicBinding:
    """One hmz epic bound to a session by workspace, environment and start time."""

    epic: str
    workspace: str
    envs: tuple[str, ...]
    used: tuple[str, ...]
    sessions: tuple[str, ...]
    cost_usd: float
    began_at: str
    #: Where hmz says the agent's harness ran: `local`, `env`, or `<backend>:<name>`.
    harness: str = ""
    #: The account the session ran as, as hmz recorded it.
    provider: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "epic": self.epic,
            "workspace": self.workspace,
            "envs": list(self.envs),
            "used": list(self.used),
            "sessions": list(self.sessions),
            "cost_usd": self.cost_usd,
            "began_at": self.began_at,
            "harness": self.harness,
            "provider": self.provider,
        }


def _epic_events(path: Path) -> list[dict[str, Any]]:
    """Read one epic journal, skipping anything that is not an object."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def epic_for_session(
    *,
    host_work: Path,
    env_spec: str,
    not_before: float,
    root: Path | None = None,
) -> EpicBinding | None:
    """Find the hmz epic that belongs to one remote session.

    The binding is exact: the epic's ``began.workspace`` must be this canonical host work
    directory, its environment list must name this remote workspace, and it must have begun
    no earlier than this attempt.  A near miss returns ``None``; the caller turns that into
    ``cost_incomplete`` instead of borrowing another session's bill.

    @param host_work canonical host directory the session ran in.
    @param env_spec the ``-e`` mapping this session used.
    @param not_before monotonic-free wall-clock start of this attempt, seconds since epoch.
    @param root epic root, defaulting to ``~/.hmz/epics``.
    """
    base = Path(root) if root is not None else Path.home() / ".hmz" / "epics"
    if not base.is_dir():
        return None
    wanted = str(Path(host_work).resolve())
    best: EpicBinding | None = None
    for path in base.rglob("epic.jsonl"):
        events = _epic_events(path)
        began = next((event for event in events if event.get("event") == "began"), None)
        if began is None:
            continue
        workspace = str(began.get("workspace") or "")
        try:
            if not workspace or str(Path(workspace).resolve()) != wanted:
                continue
        except (OSError, ValueError):
            continue
        envs = tuple(str(item) for item in (began.get("envs") or []) if isinstance(item, str))
        used = tuple(str(item) for item in (began.get("used") or []) if isinstance(item, str))
        candidates = used or envs
        if env_spec not in candidates:
            continue
        began_at = str(began.get("at") or "")
        epoch = _epoch_seconds(began_at)
        if epoch is None:
            # Without a parseable start time this epic cannot be attributed to an attempt.
            continue
        if epoch + 2.0 < not_before:
            continue
        cost = 0.0
        for event in events:
            if event.get("event") != "usage":
                continue
            number = event.get("cost")
            if isinstance(number, (int, float)) and not isinstance(number, bool):
                cost += float(number)
        opened = next((event for event in events if event.get("event") == "opened"), {})
        sessions = tuple(
            str(event.get("session"))
            for event in events
            if event.get("event") == "opened" and isinstance(event.get("session"), str)
        )
        binding = EpicBinding(
            epic=path.parent.name,
            workspace=workspace,
            envs=envs,
            used=used,
            sessions=sessions,
            cost_usd=cost,
            began_at=began_at,
            harness=str(opened.get("harness") or ""),
            provider=str(opened.get("provider") or ""),
        )
        if best is None or began_at > best.began_at:
            best = binding
    return best


def _epoch_seconds(stamp: str) -> float | None:
    """Parse hmz's UTC ``.``-fraction ISO timestamp (``2026-10-23T16:43:42.132Z``)."""
    if not stamp:
        return None
    try:
        from datetime import datetime, timezone

        text = stamp.replace("Z", "+00:00")
        return datetime.fromisoformat(text).astimezone(timezone.utc).timestamp()
    except ValueError:
        return None


__all__ = [
    "EpicBinding",
    "ManifestEntry",
    "RemoteSessionLayout",
    "RemoteWorkspaceConfig",
    "SavedRuntime",
    "SshWorkspaceTransport",
    "WorkspaceError",
    "WorkspaceManifest",
    "WorkspaceReceipt",
    "build_manifest",
    "check_relative_destination",
    "epic_for_session",
    "hmz_env_spec",
    "prompt_surface_digest",
    "read_runtime",
    "run_token",
    "session_layout",
    "settings_path",
    "workspace_root",
]
