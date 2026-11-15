#!/usr/bin/env python3
"""Future outer-control lower-level Profile probe. NOT RUN in this package.

Requires a separate explicit outside-Seatbelt authorization. All targets are
artificial outer-owned canaries/listeners; no real credentials, external network,
GPU/NPU or model is used. This file is frozen as a plan, not attack evidence.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

from tcenv.worker_isolation import (
    ArtifactContractError, AttemptSpec, HmzCapabilityUnavailable, HmzFenceExecutor,
    ResourceLimits, inspect_hmz,
)

ACK = "outside-seatbelt"


def _write(path: Path, document) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _spec(root: Path, source: str, outside: Path, port: int, mds: Path, *, clang: str, wall=4.0, artifacts=("report.json",)):
    return AttemptSpec(
        attempt_root=root / "attempt",
        argv=(sys.executable, str(root / "attempt" / "input" / "probe.py")),
        input_files={"probe.py": source.encode()},
        artifacts=artifacts,
        extra_env={
            "PROBE_OUTSIDE": str(outside), "PROBE_PORT": str(port),
            "PROBE_MDS": str(mds), "PROBE_CLANG": clang,
        },
        limits=ResourceLimits(wall_seconds=wall, cpu_seconds=3),
    )


def _exception(exc: BaseException) -> dict:
    return {"type": type(exc).__name__, "detail": str(exc)[:2000]}


source = r'''
import json, os, socket, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit
root = Path(__file__).resolve().parents[1]
outside = Path(os.environ["PROBE_OUTSIDE"])
observed = {"home": os.environ.get("HOME"), "host_canary": os.environ.get("SHAPEFORGE_OUTER_PROBE_CANARY")}
def attempt(name, action):
    try:
        value = action()
    except Exception as exc:
        observed[name] = {"result": "denied", "error": type(exc).__name__}
    else:
        observed[name] = {"result": "succeeded", "value": value}
attempt("outside_read", lambda: (outside / "secret.txt").read_bytes())
attempt("outside_write", lambda: (outside / "outside-write.txt").write_text("escape"))
attempt("mds_write", lambda: Path(os.environ["PROBE_MDS"]).write_text("escape"))
def non_proxy():
    with socket.create_connection(("127.0.0.1", int(os.environ["PROBE_PORT"])), timeout=0.5):
        pass
attempt("non_proxy_loopback", non_proxy)
def empty_host_proxy():
    port = urlsplit(os.environ["HTTPS_PROXY"]).port
    with socket.create_connection(("127.0.0.1", port), timeout=1) as stream:
        target = int(os.environ["PROBE_PORT"])
        stream.sendall(f"CONNECT 127.0.0.1:{target} HTTP/1.1\r\nHost: 127.0.0.1:{target}\r\n\r\n".encode())
        response = stream.recv(4096)
    observed["proxy_port"] = port
    return int(response.split(b" ", 2)[1])
attempt("empty_host_proxy", empty_host_proxy)
child = "from pathlib import Path; import sys; p=Path(sys.argv[1]);\n" \
        "try: p.write_text('child escape')\nexcept OSError: sys.exit(13)\nelse: sys.exit(0)"
observed["python_child_returncode"] = subprocess.run([sys.executable, "-c", child, str(outside / "child-write.txt")]).returncode
clang = subprocess.run([os.environ["PROBE_CLANG"], "--version"], capture_output=True, text=True)
observed["clang_returncode"] = clang.returncode
observed["clang_stdout"] = clang.stdout
observed["clang_stderr"] = clang.stderr
(root / "build" / "artifacts" / "report.json").write_text(json.dumps(observed, sort_keys=True))
'''

sleeping = r'''
import subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
code = "import pathlib,sys,time; time.sleep(1); pathlib.Path(sys.argv[1]).write_text('survived')"
subprocess.Popen([sys.executable, "-c", code, str(root / "build" / "work" / "ordinary-survived")])
time.sleep(10)
'''

detached = r'''
import json, os, subprocess, sys, time
from pathlib import Path
root = Path(__file__).resolve().parents[1]
code = "import pathlib,sys,time; time.sleep(0.5); out=pathlib.Path(sys.argv[1]); work=pathlib.Path(sys.argv[2]);\n" \
       "try: out.write_text('detached escape'); outcome='succeeded'\nexcept OSError: outcome='denied'\n" \
       "(work/'detached-outcome').write_text(outcome); (work/'detached-alive').write_text('alive')"
proc = subprocess.Popen([sys.executable, "-c", code, os.environ["PROBE_OUTSIDE"] + "/detached-write.txt", str(root / "build" / "work")], start_new_session=True)
(root / "build" / "work" / "detached.pid").write_text(str(proc.pid))
(root / "build" / "artifacts" / "report.json").write_text(json.dumps({"pid": proc.pid}))
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-control-ack", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "schema_id": "shapeforge.mt0-isolation-profile.outer-probe-result.1",
        "real_os_fence_executed": False,
        "publication_allowed": False,
        "production_ready": False,
        "mt0_status": "blocked",
    }
    if args.outer_control_ack != ACK:
        report.update(status="refused", detail="explicit outer-control acknowledgement is required")
        _write(args.output, report)
        return 2
    with tempfile.TemporaryDirectory(prefix="shapeforge-profile-outer-probe-") as directory:
        root = Path(directory).resolve()
        outside = root / "outside-canaries"
        outside.mkdir()
        (outside / "secret.txt").write_text("artificial-read-canary\n")
        outside_write = outside / "outside-write.txt"
        outside_write.write_text("precheck")
        outside_write.unlink()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            control, _ = listener.accept()
            control.close()
        cache = os.confstr(65538)
        if not cache:
            raise RuntimeError("the outer mds negative control needs the per-user cache location")
        mds_write = Path(cache, "mds", f"shapeforge-profile-outer-probe-{os.getpid()}.txt")
        mds_write.parent.mkdir(parents=True, exist_ok=True)
        mds_write.write_text("precheck")
        mds_write.unlink()
        os.environ["SHAPEFORGE_OUTER_PROBE_CANARY"] = "artificial-environment-canary"
        detached_pid = None
        try:
            clang = subprocess.check_output(["xcrun", "-n", "-f", "clang"], text=True).strip()
            installation = inspect_hmz()
            executor = HmzFenceExecutor(installation_loader=lambda: installation)
            report["hmz"] = installation.document()
            result = executor.run(_spec(root / "main", source, outside, port, mds_write, clang=clang))
            observed = json.loads((root / "main" / "attempt" / "build" / "artifacts" / "report.json").read_text())
            driver_report = result.document["driver_report"]
            checks = {
                "positive_artifact": result.ok,
                "temporary_home": Path(observed["home"]).is_relative_to(root / "main" / "attempt" / "build"),
                "host_environment_not_inherited": observed["host_canary"] is None,
                "outside_read_denied": observed["outside_read"]["result"] == "denied",
                "outside_write_denied": observed["outside_write"]["result"] == "denied",
                "mds_write_denied": observed["mds_write"]["result"] == "denied",
                "mds_parent_negative_control": True,
                "non_proxy_loopback_denied": observed["non_proxy_loopback"]["result"] == "denied",
                "empty_host_proxy_refused_upstream": observed["empty_host_proxy"] == {"result": "succeeded", "value": 403},
                "actual_proxy_port_matches_evidence": observed["proxy_port"] == driver_report["profile"]["port"] == driver_report["proxy"]["port"],
                "python_child_still_fenced": observed["python_child_returncode"] == 13,
                "clang_child_usable": observed["clang_returncode"] == 0,
                "single_lower_level_driver_identity": (
                    result.document["hmz"]["protocol_id"] == "hmz-lower-level-seatbelt-profile-proxy-1"
                    and result.document["launch_argv"][1] == "-I"
                    and result.document["process_boundary"]["production_route_count"] == 1
                    and result.document["process_boundary"]["internal_fence_or_wrap_used"] is False
                    and result.document["process_boundary"]["implicit_mds_write_roots"] == []
                ),
            }
            report["real_os_fence_executed"] = True
            report["main_observations"] = observed

            timeout = executor.run(_spec(root / "timeout", sleeping, outside, port, mds_write, clang=clang, wall=0.3, artifacts=()))
            time.sleep(1.2)
            checks["ordinary_child_reaped"] = timeout.status == "timeout" and not (root / "timeout" / "attempt" / "build" / "work" / "ordinary-survived").exists()

            detached_result = executor.run(_spec(root / "detached", detached, outside, port, mds_write, clang=clang))
            detached_pid = int((root / "detached" / "attempt" / "build" / "work" / "detached.pid").read_text())
            time.sleep(1.0)
            work = root / "detached" / "attempt" / "build" / "work"
            checks["setsid_outside_write_denied"] = detached_result.ok and not (outside / "detached-write.txt").exists()
            checks["setsid_fence_inherited"] = (work / "detached-outcome").read_text() == "denied"
            checks["setsid_process_group_gap_explicitly_measured"] = (work / "detached-alive").exists()

            symlink = "import os\nfrom pathlib import Path\n(Path(__file__).resolve().parents[1] / 'build' / 'artifacts' / 'report.json').symlink_to(Path(os.environ['PROBE_OUTSIDE']) / 'linked')\n"
            try:
                executor.run(_spec(root / "symlink", symlink, outside, port, mds_write, clang=clang))
            except ArtifactContractError:
                checks["artifact_symlink_rejected"] = True
            else:
                checks["artifact_symlink_rejected"] = False
            report["checks"] = checks
            report["local_ipc_boundary"] = "source-audited Unix-socket/Mach-service residual; no all-local-IPC-denied claim"
            report["status"] = "outer_probe_observed" if all(checks.values()) else "outer_probe_failed"
            report["independent_review_required"] = True
            returncode = 0 if all(checks.values()) else 1
        except HmzCapabilityUnavailable as exc:
            report.update(
                status="backend_capability_unavailable",
                attack_evidence_status="pending_outer_probe",
                detail=exc.detail, evidence=exc.evidence,
            )
            returncode = 2
        except Exception as exc:
            report.update(status="outer_probe_error", error=_exception(exc), independent_review_required=True)
            returncode = 1
        finally:
            listener.close()
            if detached_pid is not None:
                with contextlib.suppress(OSError):
                    os.killpg(detached_pid, 9)
            for path in (outside_write, outside / "child-write.txt", outside / "detached-write.txt", mds_write):
                with contextlib.suppress(OSError):
                    path.unlink()
            report["negative_controls"] = {
                "artificial_secret_was_readable": True,
                "artificial_outside_directory_was_writable": True,
                "artificial_mds_parent_was_writable": True,
                "loopback_listener_bound_and_accepted_outer_control": True,
                "real_credentials_or_external_network_used": False,
            }
            _write(args.output, report)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
