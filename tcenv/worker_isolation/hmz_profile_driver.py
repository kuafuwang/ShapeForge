#!/usr/bin/env python3
"""Trusted hmz-interpreted driver for one lower-level Seatbelt Profile attempt.

This file is executed by the pinned hmz uv-tool interpreter, outside Seatbelt.
It deliberately imports only hmz's lower-level ``Proxy`` and ``Profile`` after
validating the interpreter, installation and source bytes.  It never calls the
internal-fence CLI or ``wrap._seatbelted`` (which would add per-user C/mds).

The request is a strict trusted-attempt document.  Candidate argv is opaque data
inside the trusted rlimit-shim argv; it is never a driver control plane.  The
only process spawned here is ``Profile.command(shim_argv)`` while an empty-host
Proxy is running.  Thus no candidate code runs before sandbox-exec applies the
Profile.  ShapeForge's outer worker owns the process group and wall watchdog;
this driver owns Proxy start/stop and reports the actual loopback port.

No result from this driver is attack-containment evidence.  In particular, the
Profile retains hmz's documented Unix-socket/Mach-service and bind semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

REQUEST_SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.request.1"
REPORT_SCHEMA_ID = "tcenv.worker-isolation.hmz-profile.driver-report.1"
PROTOCOL_ID = "hmz-lower-level-seatbelt-profile-proxy-1"
DRIVER_REPORT_FD_ENV = "SF_HMZ_DRIVER_REPORT_FD"
LIMIT_REPORT_FD_ENV = "SF_LIMIT_REPORT_FD"
DRIVER_SETUP_FAILED = 126
EXPECTED_HMZ_VERSION = "0.2.2"
EXPECTED_PYTHON_VERSION = "3.14.4"
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

# Audited hmz bytes.  wrap.py is pinned as audit evidence but is never imported.
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

_PROXY_ENV = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy",
)
_RESERVED_ENV = frozenset({
    DRIVER_REPORT_FD_ENV, LIMIT_REPORT_FD_ENV,
    *_PROXY_ENV, "NO_PROXY", "no_proxy", "NODE_USE_ENV_PROXY",
})
_SECRET_NAME = re.compile(
    r"(^|_)(API_?KEY|APIKEY|KEYS?|TOKENS?|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?)($|_)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(r"(?:\bsk-[A-Za-z0-9_-]{16,}|\bAKIA[0-9A-Z]{16}|\beyJ[A-Za-z0-9_-]{12,}\.)")
_SENSITIVE_ENV = frozenset({
    "SSH_AUTH_SOCK", "SSH_AGENT_PID", "SSH_ASKPASS", "GIT_ASKPASS",
    "AWS_SHARED_CREDENTIALS_FILE", "AWS_CONFIG_FILE", "GOOGLE_APPLICATION_CREDENTIALS",
    "GH_CONFIG_DIR", "DOCKER_CONFIG", "KUBECONFIG", "NETRC", "CURL_HOME",
    "PGPASSFILE", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONINSPECT", "PYTHONUSERBASE",
    "BASH_ENV", "ENV", "SHELLOPTS", "PS4", "LD_PRELOAD", "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH", "DYLD_FALLBACK_LIBRARY_PATH",
    "NODE_OPTIONS", "RUBYOPT", "PERL5OPT",
})
_HEX256 = re.compile(r"^[0-9a-f]{64}$")


class DriverError(RuntimeError):
    """The strict driver/profile boundary cannot be established."""


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DriverError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_request(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DriverError(f"request cannot be read: {exc}") from exc
    if len(raw) > 1024 * 1024:
        raise DriverError("request exceeds its one-MiB bound")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs)
    except (UnicodeError, ValueError) as exc:
        raise DriverError(f"request is not strict JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DriverError("request must be a JSON object")
    return value


def _keys(value: Mapping[str, Any], expected: set[str], *, what: str) -> None:
    if set(value) != expected:
        raise DriverError(
            f"{what} fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _absolute_paths(value: Any, *, what: str, allow_root: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DriverError(f"{what} must be a list of absolute paths")
    result = []
    for item in value:
        path = Path(item)
        if not item or "\x00" in item or not path.is_absolute() or ".." in path.parts:
            raise DriverError(f"unsafe {what} entry: {item!r}")
        if not allow_root and path == Path(path.anchor):
            raise DriverError(f"whole-filesystem {what} is forbidden")
        result.append(item)
    if len(set(result)) != len(result):
        raise DriverError(f"{what} entries must be unique")
    return tuple(result)


def _under(path: str, roots: tuple[str, ...]) -> bool:
    resolved = Path(path).resolve(strict=False)
    return any(resolved == Path(root).resolve(strict=False) or Path(root).resolve(strict=False) in resolved.parents for root in roots)


def validate_request(request: Mapping[str, Any]) -> None:
    """Validate the closed attempt/argv schema without executing candidate code."""

    _keys(request, {
        "schema_id", "protocol_id", "protocol_digest", "installation", "attempt_root",
        "profile", "online", "hosts", "argv", "cwd", "environment",
    }, what="request")
    if request["schema_id"] != REQUEST_SCHEMA_ID or request["protocol_id"] != PROTOCOL_ID:
        raise DriverError("request schema/protocol identity mismatch")
    if not isinstance(request["protocol_digest"], str) or not _HEX256.fullmatch(request["protocol_digest"]):
        raise DriverError("protocol_digest must be a lower-case SHA-256")
    if request["online"] is not False or request["hosts"] != []:
        raise DriverError("untrusted candidate construction is always online=false with an empty host allowlist")

    installation = request["installation"]
    if not isinstance(installation, dict):
        raise DriverError("installation must be an object")
    _keys(installation, {
        "version", "python_version", "tool_root", "site_packages", "resolved_interpreter",
        "source_sha256", "driver_sha256", "child_sha256",
    }, what="installation")
    if installation["version"] != EXPECTED_HMZ_VERSION or installation["python_version"] != EXPECTED_PYTHON_VERSION:
        raise DriverError("hmz or hmz-interpreter version drift")
    _absolute_paths([installation[name] for name in ("tool_root", "site_packages", "resolved_interpreter")], what="installation path")
    if installation["source_sha256"] != HMZ_SOURCE_SHA256:
        raise DriverError("request hmz source identity differs from the driver's pinned identity")
    for name in ("driver_sha256", "child_sha256"):
        if not isinstance(installation[name], str) or not _HEX256.fullmatch(installation[name]):
            raise DriverError(f"{name} must be a lower-case SHA-256")

    _absolute_paths([request["attempt_root"]], what="attempt root")
    build = Path(request["attempt_root"]) / "build"

    profile = request["profile"]
    if not isinstance(profile, dict):
        raise DriverError("profile must be an object")
    _keys(profile, {"read", "write"}, what="profile")
    reads = _absolute_paths(profile["read"], what="read root")
    writes = _absolute_paths(profile["write"], what="write root")
    devices = {
        "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom", "/dev/tty",
        "/dev/ptmx", "/dev/fd", "/dev/dtracehelper",
    }
    if any(path not in devices and not _under(path, (str(build),)) for path in writes):
        raise DriverError("persistent Profile writes must stay under attempt/build; no per-user mds grant is allowed")
    if not reads or not writes:
        raise DriverError("Profile read/write roots cannot be empty")

    argv = request["argv"]
    if (
        not isinstance(argv, list) or not argv or len(argv) > 4096
        or not all(isinstance(item, str) and item and "\x00" not in item for item in argv)
    ):
        raise DriverError("argv must be a bounded list of non-empty NUL-free strings")
    if not Path(argv[0]).is_absolute() or len(argv) < 4 or argv[1] != "-I":
        raise DriverError("the trusted shim must start with an absolute Python followed by -I and its script")
    if _under(argv[2], (str(request["attempt_root"]),)):
        raise DriverError("the trusted shim cannot be inside the writable attempt")
    if "--" not in argv[3:]:
        raise DriverError("trusted shim argv has no candidate delimiter")
    candidate = argv[argv.index("--", 3) + 1:]
    if not candidate or not Path(candidate[0]).is_absolute():
        raise DriverError("candidate argv must follow -- and start with an absolute executable")

    cwd = request["cwd"]
    if not isinstance(cwd, str) or not Path(cwd).is_absolute() or Path(cwd) != build / "work":
        raise DriverError("cwd must be exactly attempt/build/work")
    environment = request["environment"]
    if not isinstance(environment, dict) or len(environment) > 256:
        raise DriverError("environment must be a bounded explicit object")
    for key, value in environment.items():
        if (
            not isinstance(key, str) or not key or "=" in key or "\x00" in key
            or not isinstance(value, str) or "\x00" in value
        ):
            raise DriverError("environment entries must be NUL-free strings")
        if key in _RESERVED_ENV or key.upper() in _SENSITIVE_ENV or _SECRET_NAME.search(key) or _SECRET_VALUE.search(value):
            raise DriverError(f"secret/proxy/control environment is not a candidate input: {key}")
    for required in ("PATH", "HOME", "TMPDIR", "TMP", "TEMP", "PYTHONPATH"):
        if required not in environment:
            raise DriverError(f"explicit environment is missing {required}")
    expected_paths = {
        "HOME": str(build / "home"),
        "TMPDIR": str(build / "tmp") + os.sep,
        "TMP": str(build / "tmp"),
        "TEMP": str(build / "tmp"),
        "XDG_CACHE_HOME": str(build / "cache"),
        "XDG_CONFIG_HOME": str(build / "config"),
        "XDG_DATA_HOME": str(build / "data"),
        "PYTHONPYCACHEPREFIX": str(build / "pycache"),
        "CLANG_MODULE_CACHE_PATH": str(build / "cache" / "clang"),
        "UV_CACHE_DIR": str(build / "cache" / "uv"),
        "PIP_CACHE_DIR": str(build / "cache" / "pip"),
        "GOCACHE": str(build / "cache" / "go-build"),
        "npm_config_cache": str(build / "cache" / "npm"),
    }
    for name, expected in expected_paths.items():
        if environment.get(name) != expected:
            raise DriverError(f"{name} must be exactly its attempt/build location")
    for name, expected in (("PYTHONNOUSERSITE", "1"), ("PYTHONUNBUFFERED", "1"), ("LANG", "C"), ("LC_ALL", "C")):
        if environment.get(name) != expected:
            raise DriverError(f"{name} differs from the strict attempt environment")


def _sha256(path: Path) -> str:
    try:
        if not path.is_file():
            raise OSError("not a regular file")
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DriverError(f"pinned source cannot be read: {path}: {exc}") from exc


def _validate_installation(request: Mapping[str, Any]) -> None:
    """Check real interpreter/source identity before importing any hmz module."""

    installation = request["installation"]
    if sys.platform != "darwin":
        raise DriverError("the lower-level Profile route requires macOS Seatbelt")
    if not SANDBOX_EXEC.is_file() or not os.access(SANDBOX_EXEC, os.X_OK):
        raise DriverError("Seatbelt executable is absent or not executable")
    if tuple(map(int, EXPECTED_PYTHON_VERSION.split("."))) != sys.version_info[:3]:
        raise DriverError("the running interpreter is not the pinned hmz Python")
    if Path(sys.prefix).resolve() != Path(installation["tool_root"]).resolve():
        raise DriverError("the driver is not running in the pinned hmz uv-tool environment")
    if Path(sys.executable).resolve() != Path(installation["resolved_interpreter"]).resolve():
        raise DriverError("the running executable is not the pinned hmz interpreter")
    site = Path(installation["site_packages"])
    expected_site = Path(installation["tool_root"]) / "lib" / "python3.14" / "site-packages"
    if site.resolve() != expected_site.resolve():
        raise DriverError("hmz site-packages layout drift")
    try:
        version = importlib.metadata.version("hmz")
    except importlib.metadata.PackageNotFoundError as exc:
        raise DriverError("hmz distribution metadata is unavailable") from exc
    if version != EXPECTED_HMZ_VERSION:
        raise DriverError("hmz runtime metadata version drift")
    for relative, expected in HMZ_SOURCE_SHA256.items():
        actual = _sha256(site / relative)
        if actual != expected:
            raise DriverError(f"hmz source drift before import: {relative}")
    if _sha256(Path(__file__).resolve()) != installation["driver_sha256"]:
        raise DriverError("trusted driver source identity drift")
    child = Path(request["argv"][2]).resolve()
    if _sha256(child) != installation["child_sha256"]:
        raise DriverError("trusted shim source identity drift")


def _load_lower_level(request: Mapping[str, Any]):
    """Import only the audited lower-level modules, never the internal wrapper."""

    from hmz.coganchor.darwin import seatbelt
    from hmz.coganchor.fence.proxy import Proxy

    if "hmz.coganchor.fence.wrap" in sys.modules:
        raise DriverError("the mds-granting internal wrapper was imported")
    site = Path(request["installation"]["site_packages"]).resolve()
    expected = {
        "seatbelt": site / "hmz/coganchor/darwin/seatbelt.py",
        "proxy": site / "hmz/coganchor/fence/proxy.py",
    }
    if Path(seatbelt.__file__).resolve() != expected["seatbelt"].resolve():
        raise DriverError("seatbelt was imported from an unexpected path")
    if Path(sys.modules[Proxy.__module__].__file__).resolve() != expected["proxy"].resolve():
        raise DriverError("Proxy was imported from an unexpected path")
    return seatbelt, Proxy


def _write_report(descriptor: int, report: Mapping[str, Any]) -> None:
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("could not write the complete driver report")
        view = view[written:]
    os.set_inheritable(descriptor, False)


def _execute(
    request: Mapping[str, Any], *, report_fd: int, limit_fd: int,
    lower_level=None, popen_factory=subprocess.Popen,
) -> int:
    """Run only Profile.command(shim argv), with a real empty-host Proxy port."""

    report: dict[str, Any] = {
        "schema_id": REPORT_SCHEMA_ID,
        "status": "preparing",
        "request_sha256": hashlib.sha256(
            (json.dumps(request, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
        ).hexdigest(),
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": request["protocol_digest"],
        "online": False,
        "hosts": [],
        "profile_command_spawned": False,
        "proxy_stopped": False,
        "attack_containment_proven": False,
    }
    proxy = None
    process = None
    returncode = DRIVER_SETUP_FAILED
    try:
        if lower_level is None:
            lower_level = _load_lower_level(request)
        seatbelt, proxy_type = lower_level
        proxy = proxy_type(())
        proxy.start()
        port = proxy.port
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise DriverError("the offline Proxy did not provide a real loopback port")
        # port=None would leave networking unrestricted; it is never used here.
        profile = seatbelt.Profile(
            read=tuple(request["profile"]["read"]),
            write=tuple(request["profile"]["write"]),
            port=port,
        )
        rules, params = profile.rules()
        command = profile.command(request["argv"])
        if not command or Path(command[0]) != SANDBOX_EXEC:
            raise DriverError("hmz Profile.command did not select its pinned Seatbelt executable")
        environment = dict(request["environment"])
        environment[LIMIT_REPORT_FD_ENV] = str(limit_fd)
        proxy_url = f"http://127.0.0.1:{port}"
        environment.update(dict.fromkeys(_PROXY_ENV, proxy_url))
        environment.update(NO_PROXY="", no_proxy="", NODE_USE_ENV_PROXY="1")
        report.update({
            "proxy": {
                "bind": "127.0.0.1", "port": port, "allowed_hosts": [],
                "semantics": "empty-host HTTP proxy; not a claim that Unix sockets, Mach services or all local IPC disappear",
            },
            "profile": {
                "read": list(request["profile"]["read"]),
                "write": list(request["profile"]["write"]),
                "port": port,
                "rules_sha256": hashlib.sha256(
                    json.dumps({"rules": rules, "params": params}, sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "profile_command_sha256": hashlib.sha256(
                json.dumps(command, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "lower_level_route": "hmz.coganchor.fence.proxy.Proxy + hmz.coganchor.darwin.seatbelt.Profile",
            "internal_fence_or_wrap_used": False,
            "mds_write_root_added": False,
        })
        _write_report(report_fd, report)
        process = popen_factory(
            command,
            stdin=subprocess.DEVNULL,
            env=environment,
            cwd=request["cwd"],
            pass_fds=(limit_fd,),
            close_fds=True,
        )
        report["profile_command_spawned"] = True
        report["status"] = "running"
        _write_report(report_fd, report)
        returncode = process.wait()
        report["fenced_returncode"] = returncode
        report["status"] = "ready"
    except BaseException as exc:
        returncode = DRIVER_SETUP_FAILED
        report["status"] = "failed"
        report["error_code"] = "backend_capability_unavailable"
        report["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        if process is not None and process.poll() is None:
            try:
                process.kill()
                process.wait()
            except BaseException:
                pass
    finally:
        if proxy is not None:
            try:
                proxy.stop()
                report["proxy_stopped"] = True
            except BaseException as exc:
                returncode = DRIVER_SETUP_FAILED
                report["status"] = "failed"
                report["error_code"] = "backend_capability_unavailable"
                report["error"] = f"proxy stop failed: {type(exc).__name__}: {exc}"[:4000]
        try:
            _write_report(report_fd, report)
        except BaseException as exc:
            print(f"hmz profile driver: could not write final report: {exc}", file=sys.stderr)
            returncode = DRIVER_SETUP_FAILED
    if report["status"] == "failed":
        print(f"hmz profile driver: {report.get('error', 'setup failed')}", file=sys.stderr)
    return returncode


def _descriptor(name: str) -> int:
    value = os.environ.get(name)
    if value is None:
        raise DriverError(f"missing control descriptor {name}")
    try:
        descriptor = int(value)
    except ValueError as exc:
        raise DriverError(f"{name} must be an integer") from exc
    if descriptor < 0:
        raise DriverError(f"{name} must be non-negative")
    return descriptor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report_fd = _descriptor(DRIVER_REPORT_FD_ENV)
        limit_fd = _descriptor(LIMIT_REPORT_FD_ENV)
        os.environ.pop(DRIVER_REPORT_FD_ENV, None)
    except DriverError as exc:
        print(f"hmz profile driver: {exc}", file=sys.stderr)
        return DRIVER_SETUP_FAILED
    try:
        request = _load_request(args.request)
        validate_request(request)
        _validate_installation(request)
        lower_level = _load_lower_level(request)
    except BaseException as exc:
        report = {
            "schema_id": REPORT_SCHEMA_ID,
            "status": "failed",
            "protocol_id": PROTOCOL_ID,
            "online": False,
            "hosts": [],
            "profile_command_spawned": False,
            "proxy_stopped": False,
            "candidate_or_shim_spawned": False,
            "error_code": "backend_capability_unavailable",
            "error": f"{type(exc).__name__}: {exc}"[:4000],
            "attack_containment_proven": False,
        }
        try:
            _write_report(report_fd, report)
        except BaseException:
            pass
        print(f"hmz profile driver: {type(exc).__name__}: {exc}", file=sys.stderr)
        return DRIVER_SETUP_FAILED
    return _execute(request, report_fd=report_fd, limit_fd=limit_fd, lower_level=lower_level)


if __name__ == "__main__":
    raise SystemExit(main())
