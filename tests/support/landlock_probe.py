#!/usr/bin/env python3
"""Real hmz Landlock/seccomp attack probe for a disposable Linux host.

Requires an explicit --ack=landlock. Uses only artificial canaries and loopback listeners; no
real credential, external network, GPU, NPU or model. Run with a CPython that provides
os.pidfd_open and the installed hmz 0.2.2 package on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
from urllib.parse import urlsplit

from hmz.coganchor.fence import Fence, enforceable, landlocked
from hmz.coganchor.fence.loopback import supervisable
from hmz.coganchor.linux import landlock, seccomp

ACK = "landlock"

MAIN = r'''
import json, os, socket, subprocess, sys
from pathlib import Path
from urllib.parse import urlsplit
work = Path(os.environ["PROBE_WORK"])
outside = Path(os.environ["PROBE_OUTSIDE"])
out = {"home": os.environ.get("HOME"), "host_canary": os.environ.get("SHAPEFORGE_OUTER_PROBE_CANARY")}
def attempt(name, action):
    try:
        value = action()
    except Exception as exc:
        out[name] = {"result": "denied", "error": type(exc).__name__}
    else:
        out[name] = {"result": "succeeded", "value": value}
attempt("outside_read", lambda: (outside / "secret.txt").read_text())
attempt("outside_write", lambda: (outside / "outside-write.txt").write_text("escape"))
attempt("allowed_write", lambda: (work / "allowed.txt").write_text("allowed"))
def non_proxy():
    with socket.create_connection(("127.0.0.1", int(os.environ["PROBE_PORT"])), timeout=0.5):
        pass
attempt("non_proxy_loopback", non_proxy)
def udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"probe", ("127.0.0.1", int(os.environ["PROBE_UDP_PORT"])))
    finally:
        sock.close()
attempt("udp_loopback", udp)
def empty_proxy():
    port = urlsplit(os.environ["HTTPS_PROXY"]).port
    with socket.create_connection(("127.0.0.1", port), timeout=1) as stream:
        target = int(os.environ["PROBE_PORT"])
        stream.sendall(f"CONNECT 127.0.0.1:{target} HTTP/1.1\r\nHost: 127.0.0.1:{target}\r\n\r\n".encode())
        response = stream.recv(4096)
    out["proxy_port"] = port
    return int(response.split(b" ", 2)[1])
attempt("empty_host_proxy", empty_proxy)
child = "from pathlib import Path; import sys; p=Path(sys.argv[1]);\n" \
        "try: p.write_text('child escape')\nexcept OSError: sys.exit(13)\nelse: sys.exit(0)"
out["python_child_returncode"] = subprocess.run([sys.executable, "-c", child, str(outside / "child-write.txt")]).returncode
(work / "main-result.json").write_text(json.dumps(out, sort_keys=True))
'''

TIMEOUT = r'''
import subprocess, sys, time
from pathlib import Path
work = Path(os.environ["PROBE_WORK"])
code = "import pathlib,sys,time; time.sleep(1); pathlib.Path(sys.argv[1]).write_text('survived')"
subprocess.Popen([sys.executable, "-c", code, str(work / "ordinary-survived")])
time.sleep(10)
'''

DETACHED = r'''
import json, os, socket, time
from pathlib import Path
work = Path(os.environ["PROBE_WORK"])
outside = Path(os.environ["PROBE_OUTSIDE"])
first = os.fork()
if first > 0:
    time.sleep(1.5)
    os._exit(0)
os.setsid()
(work / "daemon-group.json").write_text(json.dumps({"pgid": os.getpgid(0), "sid": os.getsid(0), "first_pid": first}))
second = os.fork()
if second > 0:
    os._exit(0)
time.sleep(0.5)
def attempt(name, action):
    try:
        action()
    except OSError as exc:
        return {"result": "denied", "error": type(exc).__name__}
    return {"result": "succeeded"}
outcome = attempt("outside_write", lambda: (outside / "detached-write.txt").write_text("detached escape"))
def non_proxy():
    with socket.create_connection(("127.0.0.1", int(os.environ["PROBE_PORT"])), timeout=0.5):
        pass
def udp():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"probe", ("127.0.0.1", int(os.environ["PROBE_UDP_PORT"])))
    finally:
        sock.close()
network = attempt("non_proxy_loopback", non_proxy)
udp_result = attempt("udp_loopback", udp)
(work / "daemon-outcome.json").write_text(json.dumps({
    "pid": os.getpid(), "outside_write": outcome["result"],
    "non_proxy_loopback": network["result"], "udp_loopback": udp_result["result"],
}))
(work / "daemon-alive").write_text("alive")
time.sleep(30)
'''


def _write(path: Path, document) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _kill_group(pgid: int) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(pgid, signal.SIGKILL)


def _alive(pid: int) -> bool:
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    if state == "Z":
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ack", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.ack != ACK:
        raise SystemExit("explicit --ack=landlock is required")
    if sys.platform != "linux":
        raise SystemExit("this probe must run on Linux")
    if not enforceable(net=True) or not landlocked(net=True):
        raise SystemExit("the full Landlock/seccomp/pidfd boundary is unavailable")
    hmz_root = Path(importlib.metadata.distribution("hmz").locate_file(""))
    source_names = (
        "hmz/__init__.py", "hmz/coganchor/__init__.py", "hmz/coganchor/darwin/__init__.py",
        "hmz/coganchor/darwin/seatbelt.py", "hmz/coganchor/fence/__init__.py",
        "hmz/coganchor/fence/proxy.py", "hmz/coganchor/fence/wrap.py", "hmz/coganchor/proto.py",
        "hmz/coganchor/linux/landlock.py", "hmz/coganchor/linux/seccomp.py",
        "hmz/coganchor/fence/loopback.py",
    )
    report = {
        "schema_id": "shapeforge.hmz-landlock-probe.1",
        "hmz": {
            "version": importlib.metadata.version("hmz"),
            "entrypoint_argv": [sys.executable, "-Pm", "hmz", "internal", "fence"],
            "entrypoint_sha256": hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest(),
            "distribution_metadata_sha256": hashlib.sha256(
                (hmz_root / "hmz-0.2.2.dist-info" / "METADATA").read_bytes()
            ).hexdigest(),
            "source_sha256": {
                name: hashlib.sha256((hmz_root / name).read_bytes()).hexdigest()
                for name in source_names
            },
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "status": "outer_probe_error",
        "real_os_fence_executed": True,
        "real_credentials_or_external_network_used": False,
        "publication_allowed": False,
        "production_ready": False,
        "preflight": {
            "landlock_abi": landlock.abi(),
            "seccomp_notifiable": seccomp.notifiable(),
            "pidfd_supervisable": supervisable(),
            "enforceable_net": True,
            "python": sys.version,
        },
    }
    with tempfile.TemporaryDirectory(prefix="shapeforge-landlock-probe-") as directory:
        root = Path(directory)
        work = root / "work"
        outside = Path(f"/home/kuafu/shapeforge-landlock-probe-outside-{os.getpid()}")
        work.mkdir()
        outside.mkdir()
        (outside / "secret.txt").write_text("artificial-read-canary\n")
        negative_control_read = (outside / "secret.txt").read_text()
        (outside / "outside-write.txt").write_text("precheck")
        (outside / "outside-write.txt").unlink()
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            control, _ = listener.accept()
            control.close()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("127.0.0.1", 0))
        udp_port = udp.getsockname()[1]
        udp.close()
        os.environ["SHAPEFORGE_OUTER_PROBE_CANARY"] = "artificial-environment-canary"
        environment = dict(os.environ)
        environment.update({
            "PYTHONPATH": os.environ["PYTHONPATH"],
            "PROBE_WORK": str(work), "PROBE_OUTSIDE": str(outside),
            "PROBE_PORT": str(port), "PROBE_UDP_PORT": str(udp_port),
        })
        # ShapeForge's attempt boundary, not an assertion that hmz internal fence sanitizes HOME
        # or arbitrary environment variables for us.  The outer canary remains outside this attempt.
        environment.pop("SHAPEFORGE_OUTER_PROBE_CANARY", None)
        environment["HOME"] = str(work / "home")
        environment["TMPDIR"] = str(work / "tmp")
        (work / "home").mkdir()
        (work / "tmp").mkdir()
        policy = Fence.of(
            local="all", user="none", system="none", online=False,
            workdir=work, home=work / "home", hosts=(),
        )
        policy_path = work / "policy.json"
        policy_path.write_text(policy.dumps())
        report["policy_sha256"] = hashlib.sha256(policy_path.read_bytes()).hexdigest()
        base = [sys.executable, "-Pm", "hmz", "internal", "fence", f"--policy=@{policy_path}", "--"]
        detached_group = None
        wrapper_pid = None
        try:
            probe_path = work / "main.py"
            probe_path.write_text(MAIN)
            completed = subprocess.run(
                [*base, "/usr/bin/python3", "-I", str(probe_path)], env=environment,
                text=True, capture_output=True, timeout=30, check=False,
            )
            report["main_returncode"] = completed.returncode
            report["main_stdout"] = completed.stdout[-4000:]
            report["main_stderr"] = completed.stderr[-4000:]
            observed = json.loads((work / "main-result.json").read_text())
            report["main_observations"] = observed
            checks = {
                "positive_artifact": completed.returncode == 0,
                "temporary_home": observed["home"].startswith(str(work)),
                "host_environment_not_inherited": observed["host_canary"] is None,
                "outside_read_denied": observed["outside_read"]["result"] == "denied",
                "outside_write_denied": observed["outside_write"]["result"] == "denied",
                "allowed_write_succeeded": observed["allowed_write"]["result"] == "succeeded",
                "non_proxy_loopback_denied": observed["non_proxy_loopback"]["result"] == "denied",
                "udp_loopback_denied": observed["udp_loopback"]["result"] == "denied",
                "empty_host_proxy_refused_upstream": observed["empty_host_proxy"] == {"result": "succeeded", "value": 403},
                "python_child_still_fenced": observed["python_child_returncode"] == 13,
            }

            timeout_path = work / "timeout.py"
            timeout_path.write_text(TIMEOUT)
            wrapper = subprocess.Popen(
                [*base, "/usr/bin/python3", "-I", str(timeout_path)], env=environment,
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            wrapper_pid = wrapper.pid
            time.sleep(0.7)
            _kill_group(wrapper_pid)
            wrapper.wait(timeout=10)
            time.sleep(1.2)
            checks["ordinary_child_killed_by_outer_control"] = not (work / "ordinary-survived").exists()

            detached_path = work / "detached.py"
            detached_path.write_text(DETACHED)
            subprocess.run(
                [*base, "/usr/bin/python3", "-I", str(detached_path)], env=environment,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False,
            )
            deadline = time.monotonic() + 10
            while not (work / "daemon-outcome.json").exists() and time.monotonic() < deadline:
                time.sleep(0.05)
            group_document = json.loads((work / "daemon-group.json").read_text())
            detached_group = int(group_document["pgid"])
            daemon_document = json.loads((work / "daemon-outcome.json").read_text())
            daemon_pid = int(daemon_document["pid"])
            checks["detached_outside_write_denied"] = daemon_document["outside_write"] == "denied"
            checks["detached_non_proxy_loopback_denied"] = daemon_document["non_proxy_loopback"] == "denied"
            checks["detached_udp_loopback_denied"] = daemon_document["udp_loopback"] == "denied"
            checks["detached_process_group_gap_explicitly_measured"] = (work / "daemon-alive").exists() and _alive(daemon_pid)
            _kill_group(detached_group)
            time.sleep(0.3)
            checks["detached_daemon_killed_by_outer_control"] = not _alive(daemon_pid)
            report["checks"] = checks
            report["status"] = "outer_probe_observed" if all(checks.values()) else "outer_probe_failed"
            report["residual_boundaries"] = {
                "detached_process_group_gap_closed": False,
                "outer_detached_group_cleanup_required": True,
                "loopback_unix_socket_residual_remains": True,
                "all_local_ipc_eliminated": False,
            }
            report["negative_controls"] = {
                "artificial_secret_was_readable": negative_control_read == "artificial-read-canary\n",
                "artificial_outside_directory_was_writable": True,
                "loopback_listener_bound_and_accepted_outer_control": True,
                "real_credentials_or_external_network_used": False,
            }
            return 0 if all(checks.values()) else 1
        finally:
            listener.close()
            if wrapper_pid is not None:
                _kill_group(wrapper_pid)
            if detached_group is not None:
                _kill_group(detached_group)
            for name in ("outside-write.txt", "child-write.txt", "detached-write.txt"):
                with contextlib.suppress(OSError):
                    (outside / name).unlink()
            with contextlib.suppress(OSError):
                outside.rmdir()
            _write(args.output, report)


if __name__ == "__main__":
    raise SystemExit(main())
