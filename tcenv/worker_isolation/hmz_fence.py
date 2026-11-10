"""Lower-level hmz Profile/Proxy fencing for untrusted embedded DSL builds.

ShapeForge owns the strict attempt contract: explicit environment, fresh
HOME/caches, read-only inputs/toolchain, attempt/build-only persistent writes,
resource limits, an outer process-group watchdog, and trusted artifact/evidence
collection.  A trusted driver run by hmz's own uv-tool interpreter owns the
lower-level ``Proxy(()) + seatbelt.Profile(port=proxy.port)`` lifecycle.  The
project Python never imports that separate environment's hmz modules.

This is the only production route.  It does not invoke ``hmz internal fence``
or ``wrap._seatbelted`` and therefore does not inherit wrap's per-user C/mds
write grant.  Missing/drifted hmz, interpreter, driver/shim identity or Seatbelt
is a typed ``backend_capability_unavailable``; there is no unfenced fallback.

Static checks, mocked processes and ordinary CPU shim/watchdog tests are not
attack-containment evidence.  MT0's outer-control-plane probe remains separate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ..children import forget, kill_group, popen_group, scrubbed_env
from ..evaluate import supervise
from ..identity import fingerprint
from .child import LIMIT_REPORT_SCHEMA_ID, REPORT_FD_ENV, SHIM_SETUP_FAILED

SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.result.1"
CAPABILITY_SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.installation.1"
REQUEST_SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.request.1"
DRIVER_REPORT_SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.driver-report.1"
HMZ_PROTOCOL_ID = "hmz-lower-level-seatbelt-profile-proxy-1"
HMZ_VERSION = "0.2.2"
HMZ_PYTHON_VERSION = "3.14.4"
HMZ_ENTRYPOINT_SHA256 = "82d06d0c2396de31c65ed4bb69ae4fc903e9ca3787fdd172f079113a96925d94"
DRIVER_SETUP_FAILED = 126
DRIVER_REPORT_FD_ENV = "SF_HMZ_DRIVER_REPORT_FD"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
REPO_ROOT = Path(__file__).resolve().parents[2]
DRIVER_PATH = Path(__file__).with_name("hmz_profile_driver.py")
CHILD_PATH = Path(__file__).with_name("child.py")

# Filled from the audited files; inspect_hmz refuses drift before any launch.
DRIVER_SOURCE_SHA256 = "451ba7ab6a3fc316df6d0d5ea431cceaa9a5f798e13da3301608da391b1dab28"
CHILD_SOURCE_SHA256 = "ecd7120f9c8c6f5ad966b1677694efafae2a10217cb44afdd961d5362156258b"

# Only lower-level runtime dependencies are pinned.  wrap.py is included as an
# audit anchor, never imported or executed by the production route.
HMZ_SOURCE_SHA256 = {
    "hmz/__init__.py": "6c4552bd5c3d84f4fb4d436c48fe3051076a72bcdcc98a934fd2c6bd03c89c01",
    "hmz/coganchor/__init__.py": "65440e88cd750d7a38d3b5def5e47bc3fe19c0f41c2e453db49ab03cbc435d83",
    "hmz/coganchor/darwin/__init__.py": "e30875525b477c7834bc644a0bb33f587b9038fc5d82f270873e4869ce22977c",
    "hmz/coganchor/darwin/seatbelt.py": "aad72f734bb7eb5c2ef21d8b3f5a27bdb346729f95637d1247c5756d2bc44d20",
    "hmz/coganchor/fence/__init__.py": "9220e7069c2b2119d60fcd9f8383b25143c32e709c36a5f02a01af8e032a3262",
    "hmz/coganchor/fence/proxy.py": "2efb0a41be47df39296419dd820bef69842ad8274516318c8bb01de82b5088d9",
    "hmz/coganchor/fence/wrap.py": "0e0aeda41cc042dde2ce5dacedf8f3d6a46182ae3250dffb00ed5537de90ccb3",
    "hmz/coganchor/proto.py": "976250da95e6a1b1e85f1d8e063729485ca159ba54cb37fc3ba606d0633d824b",
}

_DARWIN_DEVICES = (
    "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/tty",
    "/dev/ptmx", "/dev/fd", "/dev/dtracehelper",
)
_SYSTEM_READ_CANDIDATES = (
    "/usr", "/bin", "/sbin", "/System", "/private/var/select",
    "/private/var/db/timezone", "/private/var/db/dyld", "/etc/ssl",
    "/etc/resolv.conf", "/etc/hosts", "/etc/services", "/etc/protocols",
    "/etc/passwd", "/etc/group", "/etc/shells", "/etc/localtime",
    "/opt/homebrew", "/Applications/Xcode.app", "/Library/Developer",
)
_PROXY_ENV = frozenset({
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
    "NO_PROXY", "no_proxy", "NODE_USE_ENV_PROXY",
})
_SENSITIVE_ENV = frozenset({
    "SSH_AUTH_SOCK", "SSH_AGENT_PID", "SSH_ASKPASS", "GIT_ASKPASS",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE", "GOOGLE_APPLICATION_CREDENTIALS",
    "GH_CONFIG_DIR", "DOCKER_CONFIG", "KUBECONFIG", "NETRC", "CURL_HOME",
    "PGPASSFILE", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONINSPECT", "PYTHONUSERBASE",
    "BASH_ENV", "ENV", "SHELLOPTS", "PS4", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "NODE_OPTIONS", "RUBYOPT", "PERL5OPT",
})
_SECRET_VALUE = re.compile(r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}|\beyJ[A-Za-z0-9_-]{12,}\.)")


class HmzCapabilityUnavailable(RuntimeError):
    """The required lower-level hmz/Seatbelt boundary cannot be established."""

    code = "backend_capability_unavailable"
    error_code = "backend_capability_unavailable"

    def __init__(self, detail: str, *, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(f"backend_capability_unavailable: {detail}")
        self.detail = detail
        self.evidence = dict(evidence or {})


class ArtifactContractError(ValueError):
    """The fenced command returned an escaping or otherwise invalid artifact."""


@dataclass(frozen=True)
class ResourceLimits:
    """Inherited rlimits plus the ShapeForge-owned wall/artifact bounds."""

    wall_seconds: float = 60.0
    cpu_seconds: int = 60
    address_space_bytes: int | None = None
    file_bytes: int = 64 * 1024**2
    open_files: int = 128
    processes: int | None = None
    artifact_total_bytes: int = 256 * 1024**2

    def __post_init__(self) -> None:
        if isinstance(self.wall_seconds, bool) or not isinstance(self.wall_seconds, (int, float)):
            raise ValueError("wall_seconds must be a finite positive number")
        if not math.isfinite(float(self.wall_seconds)) or self.wall_seconds <= 0:
            raise ValueError("wall_seconds must be a finite positive number")
        for name in (
            "cpu_seconds", "address_space_bytes", "file_bytes", "open_files",
            "processes", "artifact_total_bytes",
        ):
            value = getattr(self, name)
            if name in {"address_space_bytes", "processes"} and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer or None where supported")

    def document(self, report: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {
            "wall_seconds": float(self.wall_seconds),
            "cpu_seconds": self.cpu_seconds,
            "address_space_bytes_requested": self.address_space_bytes,
            "file_bytes_per_file": self.file_bytes,
            "open_files": self.open_files,
            "processes_per_uid_requested": self.processes,
            "artifact_total_bytes": self.artifact_total_bytes,
            "evidence_status": "trusted_shim_observed" if report is not None else "requested_only_not_applied",
            "applied": list(report.get("applied", [])) if report is not None else [],
            "unsupported": list(report.get("unsupported", [])) if report is not None else [],
            "not_configured": list(report.get("not_configured", [])) if report is not None else [],
            "required_failures": list(report.get("required_failures", [])) if report is not None else [],
            "memory_cap_note": "Darwin RLIMIT_AS is not enforced by this contract; no outer RSS cap is currently implemented",
            "process_cap_note": "RLIMIT_NPROC is per-UID on macOS and disabled by default; it is not an attempt-local process quota",
        }


def _artifact_relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not name or "\x00" in name or "\\" in name:
        raise ValueError("artifact names must be non-empty POSIX relative paths")
    if any(part in {"", ".", ".."} for part in name.split("/")):
        raise ValueError(f"unsafe artifact path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or len(name) > 240:
        raise ValueError(f"unsafe artifact path: {name!r}")
    return path


def _validate_relative_set(names: tuple[str, ...], *, what: str) -> None:
    if len(set(names)) != len(names):
        raise ValueError(f"{what} paths must be unique")
    normalized = [_artifact_relative(name).as_posix() for name in names]
    occupied = set(normalized)
    for name in normalized:
        parts = name.split("/")
        for index in range(1, len(parts)):
            if "/".join(parts[:index]) in occupied:
                raise ValueError(f"{what} paths contain a file/directory collision: {name}")


def _validate_extra_env(extra_env: Mapping[str, str]) -> None:
    reserved = {
        "HOME", "PATH", "PYTHONPATH", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME", "XDG_DATA_HOME", "PYTHONPYCACHEPREFIX", "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED", "CLANG_MODULE_CACHE_PATH", "LANG", "LC_ALL",
        "UV_CACHE_DIR", "PIP_CACHE_DIR", "GOCACHE", "npm_config_cache", REPORT_FD_ENV,
        DRIVER_REPORT_FD_ENV, *_PROXY_ENV,
    }
    overlap = reserved.intersection(extra_env)
    if overlap:
        raise ValueError(f"extra_env cannot replace generated/driver entries: {sorted(overlap)}")
    if scrubbed_env(dict(extra_env)) != dict(extra_env):
        raise ValueError("secret-shaped environment variables are not an isolation input")
    for key, value in extra_env.items():
        if (
            not isinstance(key, str) or not key or "=" in key or "\x00" in key
            or not isinstance(value, str) or "\x00" in value
        ):
            raise ValueError("extra_env entries must be NUL-free strings")
        if key.upper() in _SENSITIVE_ENV or _SECRET_VALUE.search(value):
            raise ValueError(f"real credentials are not allowed in extra_env: {key}")


@dataclass(frozen=True)
class AttemptSpec:
    """One closed, offline embedded-frontend attempt supplied by a trusted worker."""

    attempt_root: Path
    argv: tuple[str, ...]
    artifacts: tuple[str, ...] = ()
    input_files: Mapping[str, bytes] = field(default_factory=dict)
    read_only_roots: tuple[Path, ...] = ()
    toolchain_roots: tuple[Path, ...] = ()
    python_path: tuple[Path, ...] = ()
    extra_env: Mapping[str, str] = field(default_factory=dict)
    limits: ResourceLimits = field(default_factory=ResourceLimits)

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt_root", Path(self.attempt_root))
        object.__setattr__(self, "argv", tuple(self.argv))
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "input_files", dict(self.input_files))
        for name in ("read_only_roots", "toolchain_roots", "python_path"):
            object.__setattr__(self, name, tuple(Path(value) for value in getattr(self, name)))
        object.__setattr__(self, "extra_env", dict(self.extra_env))
        if not self.argv or any(not isinstance(item, str) or not item or "\x00" in item for item in self.argv):
            raise ValueError("argv must contain non-empty strings without NUL")
        if not Path(self.argv[0]).is_absolute():
            raise ValueError("the frontend executable must be an absolute path")
        _validate_relative_set(self.artifacts, what="artifact")
        _validate_relative_set(tuple(self.input_files), what="input")
        for name, data in self.input_files.items():
            if not isinstance(data, bytes):
                raise ValueError("input_files values must be bytes")
            if len(data) > self.limits.file_bytes:
                raise ValueError("an input file exceeds the per-file resource limit")
        _validate_extra_env(self.extra_env)


@dataclass(frozen=True)
class HmzInstallation:
    """Read-only, source-audited hmz uv-tool and lower-level protocol identity."""

    requested_executable: str
    executable: Path
    interpreter: Path
    resolved_interpreter: Path
    tool_root: Path
    version: str
    python_version: str
    site_packages: Path
    source_sha256: Mapping[str, str]
    entrypoint_sha256: str
    driver_sha256: str
    child_sha256: str
    protocol_digest: str
    sandbox_exec: Path

    def document(self) -> dict[str, Any]:
        return {
            "schema_id": CAPABILITY_SCHEMA_ID,
            "protocol_id": HMZ_PROTOCOL_ID,
            "requested_executable": self.requested_executable,
            "executable": str(self.executable),
            "interpreter": str(self.interpreter),
            "resolved_interpreter": str(self.resolved_interpreter),
            "tool_root": str(self.tool_root),
            "version": self.version,
            "python_version": self.python_version,
            "site_packages": str(self.site_packages),
            "source_sha256": dict(sorted(self.source_sha256.items())),
            "entrypoint_sha256": self.entrypoint_sha256,
            "driver_path": str(DRIVER_PATH.resolve()),
            "driver_sha256": self.driver_sha256,
            "child_path": str(CHILD_PATH.resolve()),
            "child_sha256": self.child_sha256,
            "protocol_digest": self.protocol_digest,
            "sandbox_exec": str(self.sandbox_exec),
            "sandbox_exec_present_executable": self.sandbox_exec.is_file() and os.access(self.sandbox_exec, os.X_OK),
            "inspection": "read-only filesystem/source inspection; no hmz, driver or sandbox-exec execution",
            "runtime_route": "trusted driver imports lower-level Proxy and Profile using hmz's own interpreter",
            "internal_fence_or_wrap_used": False,
            "implicit_mds_write_roots": [],
            "network_semantics": "online=false; driver starts Proxy(()) and passes its actual port to Profile",
            "local_ipc_boundary": "hmz's loopback/Unix-socket/Mach-service semantics remain; no claim that all local IPC disappears",
            "runtime_attack_evidence": "not established by installation inspection",
        }


@dataclass(frozen=True)
class PreparedAttempt:
    spec: AttemptSpec
    installation: HmzInstallation
    paths: Mapping[str, Path]
    read_roots: tuple[Path, ...]
    policy: Mapping[str, Any]
    request: Mapping[str, Any]
    request_sha256: str
    argv: tuple[str, ...]
    fenced_argv: tuple[str, ...]
    environment: Mapping[str, str]
    spec_digest: str


@dataclass(frozen=True)
class IsolationResult:
    """A trusted worker result, never candidate-self-reported success."""

    status: str
    returncode: int | None
    manifest_path: Path
    request_path: Path
    stdout_path: Path
    stderr_path: Path
    document: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "completed" and self.returncode == 0

    @property
    def policy_path(self) -> Path:
        """Compatibility name for the trusted request path, not an hmz CLI policy."""

        return self.request_path


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _capability(detail: str, **evidence: Any) -> HmzCapabilityUnavailable:
    return HmzCapabilityUnavailable(detail, evidence=evidence)


def _metadata_field(text: str, name: str) -> str:
    values = [line[len(name) + 2:].strip() for line in text.splitlines() if line.startswith(name + ": ")]
    if len(values) != 1 or not values[0]:
        raise _capability(f"hmz METADATA must contain exactly one {name} field")
    return values[0]


def _pinned_file(path: Path, expected: str, *, what: str) -> str:
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise _capability(f"{what} cannot be read: {path}: {exc}") from exc
    if actual != expected:
        raise _capability(f"{what} source drift", path=str(path), expected_sha256=expected, actual_sha256=actual)
    return actual


def inspect_hmz(executable: str | Path | None = None) -> HmzInstallation:
    """Inspect and pin the uv-tool/interpreter/driver without executing them."""

    if sys.platform != "darwin":
        raise _capability("the selected lower-level Profile boundary is macOS Seatbelt; no other-platform fallback is allowed")
    if not SANDBOX_EXEC.is_file() or not os.access(SANDBOX_EXEC, os.X_OK):
        raise _capability("Seatbelt executable is absent or not executable", sandbox_exec=str(SANDBOX_EXEC))
    if executable is None:
        found = shutil.which("hmz")
        if found is None:
            raise _capability("hmz executable was not found on PATH")
    else:
        found = os.fspath(executable)
        if not Path(found).is_absolute() and not any(separator and separator in found for separator in (os.sep, os.altsep)):
            found = shutil.which(found) or found
    requested = str(found)
    try:
        resolved = Path(found).resolve(strict=True)
    except OSError as exc:
        raise _capability(f"hmz executable cannot be resolved: {exc}", requested_executable=requested) from exc
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise _capability("hmz is not an executable regular file", executable=str(resolved))
    entrypoint_sha256 = _pinned_file(resolved, HMZ_ENTRYPOINT_SHA256, what="hmz entrypoint")
    try:
        entrypoint = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise _capability(f"hmz entrypoint cannot be inspected: {exc}") from exc
    if "from hmz.cli import main" not in entrypoint:
        raise _capability("hmz entrypoint does not dispatch to the audited hmz.cli main")

    tool_root = resolved.parent.parent
    interpreter = tool_root / "bin" / "python"
    try:
        if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise OSError("not an executable regular file")
        resolved_interpreter = interpreter.resolve(strict=True)
        pyvenv = (tool_root / "pyvenv.cfg").read_text(encoding="utf-8")
    except OSError as exc:
        raise _capability(f"hmz uv-tool interpreter layout is absent or ambiguous: {exc}") from exc
    settings = dict(
        (key.strip(), value.strip())
        for line in pyvenv.splitlines() if "=" in line
        for key, value in [line.split("=", 1)]
    )
    if (
        settings.get("version_info") != HMZ_PYTHON_VERSION
        or settings.get("implementation") != "CPython"
        or settings.get("include-system-site-packages") != "false"
    ):
        raise _capability(
            "hmz interpreter version/layout drift",
            expected_python=HMZ_PYTHON_VERSION, actual_pyvenv=settings,
        )

    site_dirs = sorted(path for path in tool_root.glob("lib/python*/site-packages") if path.is_dir())
    distributions = [path for site in site_dirs for path in site.glob("hmz-*.dist-info") if path.is_dir()]
    if len(site_dirs) != 1 or len(distributions) != 1:
        raise _capability("hmz uv-tool layout is absent or ambiguous; refusing to import or guess its environment")
    site_packages = site_dirs[0]
    if site_packages != tool_root / "lib" / "python3.14" / "site-packages":
        raise _capability("hmz Python 3.14 site-packages layout drift")
    try:
        metadata = (distributions[0] / "METADATA").read_text(encoding="utf-8")
    except OSError as exc:
        raise _capability(f"hmz METADATA cannot be read: {exc}") from exc
    if _metadata_field(metadata, "Name") != "hmz":
        raise _capability("the discovered distribution is not hmz")
    version = _metadata_field(metadata, "Version")
    if version != HMZ_VERSION:
        raise _capability(
            f"hmz version drift: expected {HMZ_VERSION}, found {version}",
            expected_version=HMZ_VERSION, actual_version=version,
        )
    actual_hashes = {
        relative: _pinned_file(site_packages / relative, expected, what=f"hmz protocol source {relative}")
        for relative, expected in HMZ_SOURCE_SHA256.items()
    }
    driver_sha256 = _pinned_file(DRIVER_PATH, DRIVER_SOURCE_SHA256, what="trusted driver")
    child_sha256 = _pinned_file(CHILD_PATH, CHILD_SOURCE_SHA256, what="trusted shim")
    protocol_digest = fingerprint({
        "protocol_id": HMZ_PROTOCOL_ID,
        "layout": "hmz-uv-tool-bin-python-lower-level-1",
        "version": version,
        "python_version": HMZ_PYTHON_VERSION,
        "entrypoint_sha256": entrypoint_sha256,
        "source_sha256": actual_hashes,
        "driver_sha256": driver_sha256,
        "child_sha256": child_sha256,
    })
    return HmzInstallation(
        requested_executable=requested,
        executable=resolved,
        interpreter=interpreter.absolute(),
        resolved_interpreter=resolved_interpreter,
        tool_root=tool_root.resolve(),
        version=version,
        python_version=HMZ_PYTHON_VERSION,
        site_packages=site_packages.resolve(),
        source_sha256=actual_hashes,
        entrypoint_sha256=entrypoint_sha256,
        driver_sha256=driver_sha256,
        child_sha256=child_sha256,
        protocol_digest=protocol_digest,
        sandbox_exec=SANDBOX_EXEC,
    )


def _existing_directory(path: Path, *, what: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{what} cannot be resolved: {path}: {exc}") from exc
    if not resolved.is_dir():
        raise ValueError(f"{what} is not a directory: {path}")
    return resolved


def _prepare_attempt(spec: AttemptSpec) -> dict[str, Path]:
    root = spec.attempt_root
    if root.is_symlink():
        raise ValueError("attempt_root cannot be a symlink")
    prospective = root.resolve(strict=False)
    for forbidden in (REPO_ROOT, *spec.read_only_roots, *spec.toolchain_roots, *spec.python_path):
        resolved = _existing_directory(forbidden, what="read-only root")
        if _inside(prospective, resolved):
            raise ValueError("the writable attempt cannot be inside a read-only repository/toolchain root")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ValueError("attempt_root must be absent or an empty directory")
        root.chmod(0o700)
    else:
        root.mkdir(mode=0o700, parents=True)
    root = root.resolve(strict=True)
    paths = {
        "root": root,
        "input": root / "input",
        "build": root / "build",
        "work": root / "build" / "work",
        "home": root / "build" / "home",
        "cache": root / "build" / "cache",
        "config": root / "build" / "config",
        "data": root / "build" / "data",
        "tmp": root / "build" / "tmp",
        "pycache": root / "build" / "pycache",
        "artifacts": root / "build" / "artifacts",
        "request": root / "hmz-profile-request.json",
        "driver_report": root / "driver-report.json",
        "limits": root / "limits.json",
        "stdout": root / "stdout.txt",
        "stderr": root / "stderr.txt",
        "heartbeat": root / "heartbeat",
        "result": root / "result.json",
    }
    for key in ("input", "work", "home", "cache", "config", "data", "tmp", "pycache", "artifacts"):
        paths[key].mkdir(mode=0o700, parents=True)
    for names, base in ((spec.artifacts, paths["artifacts"]), (tuple(spec.input_files), paths["input"])):
        for name in names:
            relative = _artifact_relative(name)
            base.joinpath(*relative.parts[:-1]).mkdir(mode=0o700, parents=True, exist_ok=True)
    for name, data in spec.input_files.items():
        destination = paths["input"].joinpath(*_artifact_relative(name).parts)
        with destination.open("xb") as output:
            output.write(data)
    return paths


def _system_read_roots() -> tuple[Path, ...]:
    return tuple(path.resolve() for candidate in _SYSTEM_READ_CANDIDATES if (path := Path(candidate)).exists())


def _read_roots(spec: AttemptSpec, paths: Mapping[str, Path], executable: Path) -> tuple[Path, ...]:
    candidates = [
        CHILD_PATH.resolve().parent, paths["input"], *_system_read_roots(),
        *(_existing_directory(path, what="read-only root") for path in spec.read_only_roots),
        *(_existing_directory(path, what="toolchain root") for path in spec.toolchain_roots),
        *(_existing_directory(path, what="Python path") for path in spec.python_path),
        executable.parent.resolve(), Path(sys.executable).resolve().parent,
    ]
    for prefix in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix):
        path = Path(prefix)
        if path.is_dir():
            candidates.append(path.resolve())
    roots = tuple(dict.fromkeys(candidates))
    for root in roots:
        if root.is_dir() and _inside(paths["root"], root):
            raise ValueError(f"attempt root is inside a read-only root: {root}")
    return roots


def _environment(spec: AttemptSpec, paths: Mapping[str, Path], executable: Path) -> dict[str, str]:
    path_entries = [
        str(executable.parent), str(Path(sys.executable).resolve().parent),
        "/opt/homebrew/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin",
    ]
    python_entries = [str(paths["input"]), *(str(path.resolve()) for path in spec.python_path)]
    environment = {
        "PATH": os.pathsep.join(dict.fromkeys(path_entries)),
        "HOME": str(paths["home"]),
        "TMPDIR": str(paths["tmp"]) + os.sep,
        "TMP": str(paths["tmp"]),
        "TEMP": str(paths["tmp"]),
        "XDG_CACHE_HOME": str(paths["cache"]),
        "XDG_CONFIG_HOME": str(paths["config"]),
        "XDG_DATA_HOME": str(paths["data"]),
        "PYTHONPATH": os.pathsep.join(dict.fromkeys(python_entries)),
        "PYTHONPYCACHEPREFIX": str(paths["pycache"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "CLANG_MODULE_CACHE_PATH": str(paths["cache"] / "clang"),
        "UV_CACHE_DIR": str(paths["cache"] / "uv"),
        "PIP_CACHE_DIR": str(paths["cache"] / "pip"),
        "GOCACHE": str(paths["cache"] / "go-build"),
        "npm_config_cache": str(paths["cache"] / "npm"),
        "LANG": "C",
        "LC_ALL": "C",
    }
    environment.update(spec.extra_env)
    expected = {
        "PATH", "HOME", "TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME",
        "XDG_DATA_HOME", "PYTHONPATH", "PYTHONPYCACHEPREFIX", "PYTHONNOUSERSITE",
        "PYTHONUNBUFFERED", "CLANG_MODULE_CACHE_PATH", "UV_CACHE_DIR", "PIP_CACHE_DIR",
        "GOCACHE", "npm_config_cache", "LANG", "LC_ALL", *spec.extra_env.keys(),
    }
    if set(environment) != expected:
        raise AssertionError("the environment must be a fresh explicit allowlist")
    return environment


def _policy(paths: Mapping[str, Path], read_roots: tuple[Path, ...]) -> dict[str, Any]:
    build_writes = [paths[key] for key in ("work", "home", "cache", "config", "data", "tmp", "pycache", "artifacts")]
    writes = [Path(device) for device in _DARWIN_DEVICES if Path(device).exists()]
    writes.extend(build_writes)
    policy = {
        "read": [str(path) for path in read_roots],
        "write": [str(path.resolve()) for path in writes],
        "online": False,
        "hosts": [],
    }
    if policy["online"] or policy["hosts"]:
        raise AssertionError("offline attempt policy was widened")
    if os.sep in policy["write"]:
        raise AssertionError("whole-filesystem write is never a valid attempt")
    for path in build_writes:
        if not _inside(path, paths["build"]):
            raise AssertionError("persistent writes must stay under attempt/build")
    return policy


def _write_json(path: Path, document: Mapping[str, Any], *, atomic: bool = False) -> None:
    raw = (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if not atomic:
        path.write_bytes(raw)
        return
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(raw)
    os.replace(temporary, path)


def _artifact_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactContractError("the artifact root was replaced or is not a directory")
    found: list[Path] = []
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            if path.is_symlink():
                raise ArtifactContractError(f"symlinked artifact path is forbidden: {path}")
        for name in files:
            path = current_path / name
            if not stat.S_ISREG(path.lstat().st_mode):
                raise ArtifactContractError(f"artifact is not a regular file: {path}")
            found.append(path)
    return sorted(found)


def collect_artifacts(
    root: Path, declared: tuple[str, ...], *, per_file_bytes: int, total_bytes: int,
) -> list[dict[str, Any]]:
    """Collect exactly the declared regular files with stable identities and hashes."""

    for name, value in (("per_file_bytes", per_file_bytes), ("total_bytes", total_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_relative_set(declared, what="artifact")
    files = _artifact_files(root)
    actual = {path.relative_to(root).as_posix() for path in files}
    expected = set(declared)
    if actual != expected:
        raise ArtifactContractError(
            f"artifact set mismatch: undeclared={sorted(actual - expected)}, missing={sorted(expected - actual)}"
        )
    documents = []
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        before = path.lstat()
        if before.st_nlink != 1:
            raise ArtifactContractError(f"hardlinked artifact is forbidden: {relative}")
        if before.st_size > per_file_bytes:
            raise ArtifactContractError(f"artifact exceeds its byte cap: {relative}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        try:
            opened = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino) or not stat.S_ISREG(opened.st_mode):
                raise ArtifactContractError(f"artifact changed identity during collection: {relative}")
            digest = hashlib.sha256()
            size = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, per_file_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > per_file_bytes:
                    raise ArtifactContractError(f"artifact grew beyond its byte cap: {relative}")
                digest.update(chunk)
            after = os.fstat(descriptor)
            final = path.lstat()
            for first, second in ((before, after), (after, final)):
                first_identity = (
                    first.st_dev, first.st_ino, first.st_nlink, first.st_mode,
                    first.st_size, first.st_mtime_ns, first.st_ctime_ns,
                )
                second_identity = (
                    second.st_dev, second.st_ino, second.st_nlink, second.st_mode,
                    second.st_size, second.st_mtime_ns, second.st_ctime_ns,
                )
                if first_identity != second_identity:
                    raise ArtifactContractError(f"artifact changed during collection: {relative}")
        finally:
            os.close(descriptor)
        total += size
        if total > total_bytes:
            raise ArtifactContractError("declared artifacts exceed their total byte cap")
        documents.append({"path": relative, "bytes": size, "sha256": digest.hexdigest()})
    return documents


def _verify_inputs(spec: AttemptSpec, root: Path) -> None:
    documents = collect_artifacts(
        root, tuple(spec.input_files), per_file_bytes=spec.limits.file_bytes,
        total_bytes=max(1, sum(map(len, spec.input_files.values()))),
    )
    for document in documents:
        expected = spec.input_files[document["path"]]
        if document["bytes"] != len(expected) or document["sha256"] != hashlib.sha256(expected).hexdigest():
            raise ArtifactContractError(f"read-only input changed during the attempt: {document['path']}")


def _read_json_report(path: Path, schema: str) -> dict[str, Any] | None:
    try:
        raw = path.read_bytes()
        if len(raw) > 128 * 1024:
            return None
        document = json.loads(raw)
    except (OSError, UnicodeError, ValueError):
        return None
    if not isinstance(document, dict) or document.get("schema_id") != schema:
        return None
    return document


def _read_limit_report(path: Path) -> dict[str, Any] | None:
    document = _read_json_report(path, LIMIT_REPORT_SCHEMA_ID)
    if document is None or document.get("status") not in {"ready", "failed"}:
        return None
    for name in ("applied", "unsupported", "not_configured", "required_failures"):
        if not isinstance(document.get(name), list):
            return None
    if document["status"] == "failed" and not document["required_failures"]:
        return None
    return document


def _valid_driver_report(prepared: PreparedAttempt, report: Mapping[str, Any] | None) -> bool:
    if report is None or report.get("status") != "ready":
        return False
    if report.get("protocol_id") != HMZ_PROTOCOL_ID:
        return False
    if report.get("request_sha256") != prepared.request_sha256 or report.get("protocol_digest") != prepared.installation.protocol_digest:
        return False
    if report.get("online") is not False or report.get("hosts") != [] or report.get("profile_command_spawned") is not True:
        return False
    if report.get("proxy_stopped") is not True or report.get("attack_containment_proven") is not False:
        return False
    proxy = report.get("proxy")
    profile = report.get("profile")
    if not isinstance(proxy, dict) or not isinstance(profile, dict):
        return False
    port = proxy.get("port")
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        return False
    if proxy.get("allowed_hosts") != [] or proxy.get("bind") != "127.0.0.1" or profile.get("port") != port:
        return False
    if profile.get("read") != prepared.policy["read"] or profile.get("write") != prepared.policy["write"]:
        return False
    if report.get("internal_fence_or_wrap_used") is not False or report.get("mds_write_root_added") is not False:
        return False
    if report.get("lower_level_route") != "hmz.coganchor.fence.proxy.Proxy + hmz.coganchor.darwin.seatbelt.Profile":
        return False
    if not isinstance(report.get("profile_command_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", report["profile_command_sha256"]):
        return False
    if not isinstance(profile.get("rules_sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", profile["rules_sha256"]):
        return False
    fenced_returncode = report.get("fenced_returncode")
    return isinstance(fenced_returncode, int) and not isinstance(fenced_returncode, bool)


class HmzFenceExecutor:
    """Run an embedded frontend through the one lower-level hmz Profile route."""

    def __init__(
        self,
        hmz_executable: str | Path | None = None,
        *,
        installation_loader: Callable[[], HmzInstallation] | None = None,
    ) -> None:
        self._installation_loader = installation_loader or (lambda: inspect_hmz(hmz_executable))

    def prepare(self, spec: AttemptSpec) -> PreparedAttempt:
        installation = self._installation_loader()
        paths = _prepare_attempt(spec)
        try:
            executable = Path(spec.argv[0]).resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"frontend executable cannot be resolved: {exc}") from exc
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("frontend executable is not an executable regular file")
        if _inside(executable, paths["root"]):
            raise ValueError("the trusted frontend executable cannot be inside the writable attempt")
        read_roots = _read_roots(spec, paths, executable)
        policy = _policy(paths, read_roots)
        environment = _environment(spec, paths, executable)
        limits = spec.limits
        optional_limits: list[str] = []
        if limits.address_space_bytes is not None:
            optional_limits.extend(("--address-space-bytes", str(limits.address_space_bytes)))
        if limits.processes is not None:
            optional_limits.extend(("--processes", str(limits.processes)))
        fenced_argv = (
            str(Path(sys.executable).resolve()), "-I", str(CHILD_PATH.resolve()),
            "--cpu-seconds", str(limits.cpu_seconds),
            *optional_limits,
            "--file-bytes", str(limits.file_bytes),
            "--open-files", str(limits.open_files),
            "--", *spec.argv,
        )
        spec_document = {
            "argv": list(spec.argv),
            "artifacts": list(spec.artifacts),
            "input_files": {
                name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                for name, data in sorted(spec.input_files.items())
            },
            "attempt_root": str(paths["root"]),
            "read_only_roots": [str(path) for path in read_roots],
            "python_path": [str(path) for path in spec.python_path],
            "extra_env_keys": sorted(spec.extra_env),
            "online": False,
            "limits": limits.document(),
        }
        request = {
            "schema_id": REQUEST_SCHEMA_ID,
            "protocol_id": HMZ_PROTOCOL_ID,
            "protocol_digest": installation.protocol_digest,
            "installation": {
                "version": installation.version,
                "python_version": installation.python_version,
                "tool_root": str(installation.tool_root),
                "site_packages": str(installation.site_packages),
                "resolved_interpreter": str(installation.resolved_interpreter),
                "source_sha256": dict(sorted(installation.source_sha256.items())),
                "driver_sha256": installation.driver_sha256,
                "child_sha256": installation.child_sha256,
            },
            "attempt_root": str(paths["root"]),
            "profile": {"read": policy["read"], "write": policy["write"]},
            "online": False,
            "hosts": [],
            "argv": list(fenced_argv),
            "cwd": str(paths["work"]),
            "environment": environment,
        }
        _write_json(paths["request"], request)
        request_sha256 = hashlib.sha256(paths["request"].read_bytes()).hexdigest()
        argv = (
            str(installation.interpreter), "-I", str(DRIVER_PATH.resolve()),
            "--request", str(paths["request"]),
        )
        return PreparedAttempt(
            spec=spec, installation=installation, paths=paths, read_roots=read_roots,
            policy=policy, request=request, request_sha256=request_sha256,
            argv=argv, fenced_argv=fenced_argv, environment=environment,
            spec_digest=fingerprint(spec_document),
        )

    def run(self, spec: AttemptSpec) -> IsolationResult:
        prepared = self.prepare(spec)
        paths = prepared.paths
        reason = ""
        returncode: int | None = None
        with paths["stdout"].open("w") as stdout, paths["stderr"].open("w") as stderr:
            driver_fd = os.open(paths["driver_report"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                limit_fd = os.open(paths["limits"], os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except BaseException:
                os.close(driver_fd)
                raise
            try:
                environment = dict(prepared.environment)
                environment[DRIVER_REPORT_FD_ENV] = str(driver_fd)
                environment[REPORT_FD_ENV] = str(limit_fd)
                try:
                    proc = popen_group(
                        list(prepared.argv), stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                        env=environment, cwd=paths["work"], pass_fds=(driver_fd, limit_fd),
                    )
                except OSError as exc:
                    document = self._document(prepared, "backend_capability_unavailable", None, "", [], None, None)
                    _write_json(paths["result"], document, atomic=True)
                    raise _capability(
                        f"the pinned hmz interpreter/driver could not be started: {type(exc).__name__}: {exc}",
                        result=document,
                    ) from exc
            finally:
                os.close(limit_fd)
                os.close(driver_fd)
            try:
                reason = supervise(
                    proc, paths["heartbeat"], launch_deadline_s=float(spec.limits.wall_seconds),
                    timeout=float(spec.limits.wall_seconds), compile_deadline_s=float(spec.limits.wall_seconds),
                )
                returncode = proc.returncode
            finally:
                # Driver, sandbox-exec, shim and ordinary descendants share this owned
                # group.  Proxy threads die with the driver.  setsid containment remains
                # an outer-probe question, not a claim made by this adapter.
                kill_group(proc)
                forget(proc)
        driver_report = _read_json_report(paths["driver_report"], DRIVER_REPORT_SCHEMA_ID)
        limit_report = _read_limit_report(paths["limits"])
        if not reason and _valid_driver_report(prepared, driver_report):
            returncode = driver_report["fenced_returncode"]

        def capability_failure(detail: str) -> None:
            document = self._document(
                prepared, "backend_capability_unavailable", returncode, reason, [], limit_report, driver_report,
            )
            document["detail"] = detail
            _write_json(paths["result"], document, atomic=True)
            raise HmzCapabilityUnavailable(detail, evidence=document)

        if not reason:
            if not _valid_driver_report(prepared, driver_report):
                capability_failure("the trusted driver did not establish and cleanly close the requested Profile/Proxy boundary")
            if limit_report is None:
                capability_failure("the trusted shim did not return a valid resource-limit report; no unfenced execution is accepted")
            if returncode == SHIM_SETUP_FAILED and limit_report["status"] == "failed":
                capability_failure("the trusted shim could not apply a required resource limit; frontend was not executed")
            if limit_report["status"] != "ready":
                capability_failure("the trusted shim resource-limit report is inconsistent with the process outcome")
        status = "timeout" if reason else ("completed" if returncode == 0 else "process_failed")
        artifacts: list[dict[str, Any]] = []
        if status == "completed":
            try:
                _verify_inputs(spec, paths["input"])
                artifacts = collect_artifacts(
                    paths["artifacts"], spec.artifacts,
                    per_file_bytes=spec.limits.file_bytes, total_bytes=spec.limits.artifact_total_bytes,
                )
            except ArtifactContractError:
                document = self._document(prepared, "artifact_invalid", returncode, reason, [], limit_report, driver_report)
                _write_json(paths["result"], document, atomic=True)
                raise
        document = self._document(prepared, status, returncode, reason, artifacts, limit_report, driver_report)
        _write_json(paths["result"], document, atomic=True)
        return IsolationResult(
            status, returncode, paths["result"], paths["request"],
            paths["stdout"], paths["stderr"], document,
        )

    def _document(
        self, prepared: PreparedAttempt, status: str, returncode: int | None,
        reason: str, artifacts: list[dict[str, Any]], limit_report: Mapping[str, Any] | None,
        driver_report: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        paths = prepared.paths
        return {
            "schema_id": SCHEMA_ID,
            "status": status,
            "returncode": returncode,
            "watchdog_reason": reason or None,
            "spec_digest": prepared.spec_digest,
            "hmz": prepared.installation.document(),
            "request_sha256": prepared.request_sha256,
            "request_path": str(paths["request"]),
            "policy": dict(prepared.policy),
            "launch_argv": list(prepared.argv),
            "fenced_argv": list(prepared.fenced_argv),
            "candidate_argv": list(prepared.spec.argv),
            "environment_keys": sorted(prepared.environment),
            "control_environment_removed_before_frontend": [DRIVER_REPORT_FD_ENV, REPORT_FD_ENV],
            "driver_added_offline_proxy_keys": sorted(_PROXY_ENV),
            "environment_basis": "fresh explicit allowlist; host environment is never inherited",
            "attempt_root": str(paths["root"]),
            "artifacts": artifacts,
            "artifact_manifest_digest": fingerprint({"artifacts": artifacts}),
            "artifact_channel": "trusted worker collects declared files only after process-group reaping",
            "driver_report": dict(driver_report) if driver_report is not None else None,
            "process_boundary": {
                "os_fence_owner": "hmz lower-level seatbelt.Profile with empty-host Proxy, started by trusted hmz-interpreter driver",
                "production_route_count": 1,
                "internal_fence_or_wrap_used": False,
                "implicit_mds_write_roots": [],
                "python_clang_descendants_inside_same_profile": True,
                "shapeforge_owned_process_group": True,
                "ordinary_descendants_reaped": True,
                "setsid_escape_proven_contained": False,
                "attack_containment_proven_by_this_result": False,
            },
            "resource_limits": prepared.spec.limits.document(limit_report),
            "limit_report": dict(limit_report) if limit_report is not None else None,
            "returncode_classification": {
                "driver_setup_failure_exit": DRIVER_SETUP_FAILED,
                "shim_setup_failure_exit": SHIM_SETUP_FAILED,
                "rule": "a valid final driver report and valid ready shim report are required; frontend 125/126 never becomes success",
            },
            "residual_risks": [
                "real Seatbelt/file/network/credential/descendant attacks require the outer-control-plane probe",
                "offline mode intentionally uses a loopback Proxy and preserves hmz's documented Unix-socket/Mach-service boundaries; not all local IPC is proven absent",
                "setsid/new-session descendants require separate process-containment evidence",
                "Darwin RLIMIT_AS is not an enforced memory cap and no outer RSS cap is currently implemented",
                "RLIMIT_NPROC is per-UID on macOS and RLIMIT_FSIZE is per file, not a filesystem quota",
                "trusted dependency installation is a separate phase and is not candidate online access",
                "compiler, kernel and driver vulnerabilities are outside this CPU attempt contract",
            ],
            "stdout_path": str(paths["stdout"]),
            "stderr_path": str(paths["stderr"]),
        }


__all__ = [
    "ArtifactContractError", "AttemptSpec", "CAPABILITY_SCHEMA_ID", "CHILD_SOURCE_SHA256",
    "DRIVER_REPORT_SCHEMA_ID", "DRIVER_SOURCE_SHA256", "HMZ_PROTOCOL_ID", "HMZ_PYTHON_VERSION",
    "HMZ_SOURCE_SHA256", "HMZ_VERSION", "HmzCapabilityUnavailable", "HmzFenceExecutor",
    "HmzInstallation", "IsolationResult", "PreparedAttempt", "REPO_ROOT", "REQUEST_SCHEMA_ID",
    "ResourceLimits", "SCHEMA_ID", "collect_artifacts", "inspect_hmz",
]
