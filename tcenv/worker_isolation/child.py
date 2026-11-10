"""Trusted in-fence shim: install supported rlimits, report them, then exec.

The outer trusted driver starts this shim through hmz's lower-level
``seatbelt.Profile.command()``, so the shim, Python/Clang frontend and their
descendants all stay in the same Profile. ShapeForge's outer worker still owns
the wall deadline and process group. This module has no candidate parser,
child-process launcher or fallback.

Darwin does not provide a useful RLIMIT_AS enforcement boundary: setting a value
below the process's current vsize can fail with EINVAL, and the kernel does not
enforce the resulting cap as ShapeForge requires.  AS is therefore optional and
reported as unsupported on Darwin rather than silently claimed.  RLIMIT_NPROC is
per-UID on macOS and is disabled unless explicitly requested, so a low per-attempt
number cannot accidentally forbid every frontend/compiler fork.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
from typing import Any

SHIM_SETUP_FAILED = 125
LIMIT_REPORT_SCHEMA_ID = "tcenv.worker-isolation.limits.1"
REPORT_FD_ENV = "SF_LIMIT_REPORT_FD"


def _positive(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("resource limits must be positive")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cpu-seconds", type=_positive, required=True)
    parser.add_argument("--address-space-bytes", type=_positive)
    parser.add_argument("--file-bytes", type=_positive, required=True)
    parser.add_argument("--open-files", type=_positive, required=True)
    parser.add_argument("--processes", type=_positive)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    return parser


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _apply(
    report: dict[str, Any], *, name: str, resource_name: str, value: int,
    required: bool, scope: str,
) -> None:
    requested = [value, value]
    try:
        resource.setrlimit(getattr(resource, resource_name), (value, value))
        actual = list(resource.getrlimit(getattr(resource, resource_name)))
    except (OSError, ValueError, OverflowError, AttributeError) as exc:
        row = {"name": name, "requested": requested, "scope": scope, "error": _error(exc)}
        if required:
            report["required_failures"].append(row)
        else:
            report["unsupported"].append(row)
        return
    if actual != requested:
        row = {
            "name": name, "requested": requested, "actual": actual, "scope": scope,
            "error": "the kernel accepted a different limit than requested",
        }
        if required:
            report["required_failures"].append(row)
        else:
            report["unsupported"].append(row)
        return
    report["applied"].append({"name": name, "actual": actual, "scope": scope})


def _write_report(descriptor: int, report: dict[str, Any]) -> None:
    if descriptor < 0:
        return
    raw = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("could not write the complete resource-limit report")
        view = view[written:]
    # The frontend must not inherit a writer into the control-owned attempt root.
    os.set_inheritable(descriptor, False)


def _report_fd() -> int:
    written = os.environ.pop(REPORT_FD_ENV, None)
    if written is None:
        return -1
    try:
        descriptor = int(written)
    except ValueError as exc:
        raise ValueError(f"{REPORT_FD_ENV} must be an integer") from exc
    if descriptor < 0:
        raise ValueError(f"{REPORT_FD_ENV} must name a non-negative descriptor")
    return descriptor


def main() -> int:
    args = _parser().parse_args()
    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv.pop(0)
    if not argv:
        raise SystemExit("missing frontend argv")
    try:
        descriptor = _report_fd()
    except ValueError as exc:
        print(f"worker isolation shim: {exc}", file=sys.stderr)
        return SHIM_SETUP_FAILED
    report: dict[str, Any] = {
        "schema_id": LIMIT_REPORT_SCHEMA_ID,
        "platform": sys.platform,
        "status": "ready",
        "applied": [],
        "unsupported": [],
        "not_configured": [],
        "required_failures": [],
        "frontend_exec_attempted": False,
    }
    try:
        _apply(report, name="cpu_seconds", resource_name="RLIMIT_CPU", value=args.cpu_seconds, required=True, scope="per_process_and_descendants")
        _apply(report, name="file_bytes", resource_name="RLIMIT_FSIZE", value=args.file_bytes, required=True, scope="per_file")
        _apply(report, name="open_files", resource_name="RLIMIT_NOFILE", value=args.open_files, required=True, scope="per_process_descriptor_table")
        _apply(report, name="core_bytes", resource_name="RLIMIT_CORE", value=0, required=True, scope="per_process_and_descendants")
        if args.address_space_bytes is None:
            report["not_configured"].append({
                "name": "address_space_bytes",
                "reason": "no memory cap is requested or claimed; outer RSS enforcement is not implemented",
            })
        elif sys.platform == "darwin":
            report["unsupported"].append({
                "name": "address_space_bytes",
                "requested": [args.address_space_bytes, args.address_space_bytes],
                "scope": "darwin_per_process_vsize",
                "error": "Darwin RLIMIT_AS is not a reliable enforced memory cap and can reject a value below current vsize with EINVAL",
            })
        else:
            _apply(
                report, name="address_space_bytes", resource_name="RLIMIT_AS",
                value=args.address_space_bytes, required=False, scope="per_process_virtual_address_space",
            )
        if args.processes is None:
            report["not_configured"].append({
                "name": "processes",
                "reason": "no process-count cap is requested; macOS RLIMIT_NPROC is per-UID, not per-attempt",
            })
        else:
            _apply(
                report, name="processes", resource_name="RLIMIT_NPROC",
                value=args.processes, required=False, scope="per_uid_on_darwin_not_per_attempt",
            )
        if report["required_failures"]:
            report["status"] = "failed"
        _write_report(descriptor, report)
    except (OSError, ValueError) as exc:
        print(f"worker isolation shim: could not report resource setup: {_error(exc)}", file=sys.stderr)
        return SHIM_SETUP_FAILED
    if report["required_failures"]:
        return SHIM_SETUP_FAILED
    report["frontend_exec_attempted"] = True
    try:
        # Best effort update before exec.  Failure to rewrite this diagnostic does
        # not weaken the already-applied limits or justify running without them.
        _write_report(descriptor, report)
    except (OSError, ValueError):
        pass
    try:
        os.execvpe(argv[0], argv, os.environ)
    except OSError as exc:
        print(f"worker isolation shim: could not exec {argv[0]}: {_error(exc)}", file=sys.stderr)
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
