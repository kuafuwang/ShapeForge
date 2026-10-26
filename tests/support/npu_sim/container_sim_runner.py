"""Real Ubuntu VM acceptance for the container execution plane (test-only).

This runner executes the *production* ``JobEngine`` + ``DockerAttemptSupervisor``
with a CPU-only fake worker on a real Ubuntu ARM VM.  It is not the simulated
Phase 2D2 gRPC matrix and it is not a search run: no GPU/NPU, no device SDK, no
model/provider call, no publication claim.  The report always carries
``simulated=true`` and ``publication_allowed=false``, writes no
``run_config.json`` and never touches confirmed/catalog/brief state.

Run it on the VM (test-only composition root), with an explicit authorization
flag, from a staging checkout of this repository:

    TAIC_CONTAINER_SIM=1 python3 -m tests.support.npu_sim.container_sim_runner \\
        --out runs/container-sim-<date> --build-image

The worker image is built locally from an already-present base image
(``taic-remote-sim:local``); no registry/network access is required.

When the runner itself runs inside a harness container it needs the host Docker
CLI + socket, the host docker group, the host cgroup namespace
(``--cgroupns host``) and the host PID namespace (``--pid=host``): the cgroup
lifecycle check and the host survivor scans are meaningless in a private
namespace.  The runner proves both (a self-spawned ``ps`` control and the live
worker container's host PID) before trusting an empty survivor scan.

Required checks cover: a real 60s-sleep timeout that kills parent and child and
leaves no container/process/cgroup/temp path; Cancel of an actively executing
worker that rejects a late result; frontend SIGKILL/restart (verified from the
daemon after the kill) that cleans the container and marks the attempt
interrupted (or quarantines when cleanup cannot be verified) without
auto-running it again; a stale-fence result refused while a *newer* attempt is
executing; a normal fake report/evidence/receipt that validates; the hardened
worker posture (no Docker socket, no control-host home/repo mount, no long-term
credentials, no network, read-only root); and the production entry's refusal of
test composition roots.  ``status=passed`` needs every required check to run and
pass plus final cleanup evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from tcenv.remote.container_supervisor import DockerAttemptSupervisor
from tcenv.remote.job_types import JobError
from tcenv.remote import contracts
from tests.support.npu_sim import container_sim_common as common

ROOT = Path(__file__).resolve().parents[3]
OWNER_FILTER = "label=tcenv.owner=external-evaluator"
CHILD_TOKEN_PREFIX = "taic-container-sim-child"
REQUIRED_CHECKS = (
    "preflight_docker_and_pinned_image",
    "timeout_kills_parent_and_child",
    "timeout_delete_removes_container_cgroup_and_temp",
    "cancel_kills_worker_and_rejects_old_result",
    "crash_restart_cleans_container_and_marks_interrupted",
    "crash_restart_never_reruns_the_attempt",
    "stale_fence_cannot_commit_while_a_newer_attempt_runs",
    "crash_quarantine_blocks_retry_when_cleanup_is_unverified",
    "normal_report_evidence_and_receipt_validate",
    "worker_has_no_socket_home_repo_or_credentials",
    "production_entry_rejects_test_worker_and_fake_backend",
    "final_cleanup_has_no_containers_processes_or_temp",
)


def _run(argv: list[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        argv, text=True, capture_output=True, timeout=timeout, check=False,
    )


def _docker(*args: str, timeout: float = 60.0) -> subprocess.CompletedProcess:
    return _run(["docker", *args], timeout=timeout)


def _ps_tokens(token: str) -> list[str]:
    completed = _run(["ps", "-eo", "pid,args"], timeout=30.0)
    if completed.returncode != 0:
        return [f"<ps unavailable: {completed.stderr.strip()}>"]
    return [line.strip() for line in completed.stdout.splitlines() if token in line and "ps -eo" not in line]


def _ps_tokens_settled(token: str, *, timeout_s: float = 10.0) -> list[str]:
    """Processes can take a moment to be reaped; report survivors only."""

    deadline = time.monotonic() + timeout_s
    survivors = _ps_tokens(token)
    while survivors and time.monotonic() < deadline:
        time.sleep(0.2)
        survivors = _ps_tokens(token)
    return survivors


def _cgroup_paths(container_id: str) -> list[Path]:
    return [
        Path(f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope"),
        Path(f"/sys/fs/cgroup/docker/{container_id}"),
    ]


def _cgroup_exists(container_id: str) -> bool:
    return any(path.exists() for path in _cgroup_paths(container_id))


def _wait(predicate, deadline_s: float, description: str, interval: float = 0.1):
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise RuntimeError(f"timed out waiting for {description}")


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        self.checks: dict[str, dict] = {}
        self.started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.environment: dict = {}
        self.blockers: list[str] = []

    def record(self, name: str, status: str, detail: str, **evidence) -> None:
        self.checks[name] = {
            "name": name,
            "required": name in REQUIRED_CHECKS,
            "status": status,
            "detail": detail,
            "evidence": evidence,
        }
        print(f"[{status}] {name}: {detail}", flush=True)

    def passed(self, name: str, detail: str, **evidence) -> None:
        self.record(name, "passed", detail, **evidence)

    def failed(self, name: str, detail: str, **evidence) -> None:
        self.record(name, "failed", detail, **evidence)

    def skip(self, name: str, detail: str) -> None:
        self.record(name, "not_run", detail)

    def write(self, *, cleanup: dict, extra: dict | None = None) -> dict:
        missing = [name for name in REQUIRED_CHECKS if name not in self.checks]
        not_passed = [
            name for name in REQUIRED_CHECKS
            if name in self.checks and self.checks[name]["status"] != "passed"
        ]
        failed = [
            name for name in REQUIRED_CHECKS
            if name in self.checks and self.checks[name]["status"] == "failed"
        ]
        not_run = [
            name for name in REQUIRED_CHECKS
            if name not in self.checks or self.checks[name]["status"] == "not_run"
        ]
        # A blocker recorded after a check seems to have passed must not be
        # folded back into ``status=passed``.
        if not missing and not not_passed and cleanup.get("final") and not self.blockers:
            status = "passed"
        elif failed:
            status = "failed"
        elif self.blockers or not_run:
            status = "blocked"
        else:
            status = "failed"
        document = {
            "report_schema": "taic.container-sim.v1",
            "kind": "container-execution-acceptance",
            "simulated": True,
            "publication_allowed": False,
            "hardware_performance_claims": False,
            "device": "cpu-fake",
            "status": status,
            "started_at": self.started_at,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "environment": self.environment,
            "checks": [self.checks[name] for name in self.checks],
            "required_checks": list(REQUIRED_CHECKS),
            "not_run": missing + [name for name in REQUIRED_CHECKS if name in self.checks and self.checks[name]["status"] == "not_run"],
            "blockers": self.blockers,
            "cleanup": cleanup,
            "extra": extra or {},
        }
        self.out.mkdir(parents=True, exist_ok=True)
        target = self.out / "report.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        return document


def _write_evidence(out: Path, name: str, argv: list[str]) -> dict:
    completed = _run(argv, timeout=120.0)
    path = out / f"evidence-{name}.json"
    document = {
        "command": argv,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-60_000:],
        "stderr": completed.stderr[-10_000:],
    }
    try:
        document["parsed"] = json.loads(completed.stdout)
    except ValueError:
        pass
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"file": path.name, "returncode": completed.returncode}


def _inspect_container(container_id: str) -> dict | None:
    completed = _docker("inspect", "--type", "container", container_id)
    if completed.returncode != 0:
        return None
    try:
        return json.loads(completed.stdout)[0]
    except (ValueError, IndexError):
        return None


def _container_state(container_id: str) -> str:
    completed = _docker("inspect", "--type", "container", container_id)
    if completed.returncode != 0:
        text = (completed.stdout + completed.stderr).lower()
        if "no such" in text or "not found" in text:
            return "absent"
        raise RuntimeError(
            f"docker inspect failed for {container_id}: {completed.stderr.strip()[:300]}"
        )
    try:
        document = json.loads(completed.stdout)[0]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"docker inspect output for {container_id} is invalid: {exc}") from exc
    return str((document.get("State") or {}).get("Status") or "unknown")


def _positive_process_control() -> dict:
    """Prove this process can see host processes before trusting a null scan.

    The survivor checks conclude "nothing left" from an empty ``ps`` match; that
    is only evidence if the same scan can see a process we know exists.
    """

    token = f"taic-container-sim-ps-control-{os.getpid()}-{int(time.time())}"
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(20)", token],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        seen = _wait(lambda: _ps_tokens(token), 10.0, "the ps positive control")
    finally:
        child.kill()
        child.wait(timeout=10)
    return {"token": token, "visible": bool(seen), "scan": seen[:4]}


def _host_pid_visibility(container_id: str) -> dict:
    """Prove a *host* process of the worker container is visible to ``ps``.

    The survivor scans conclude "no process left" from an empty host ``ps``
    match.  Inside a harness container that only sees its own PID namespace the
    scan is blind, so the acceptance must show it can see the worker's host PID
    (run the frontend with ``--pid=host``).
    """

    document = _inspect_container(container_id)
    pid = 0
    if isinstance(document, dict):
        value = (document.get("State") or {}).get("Pid")
        if isinstance(value, int) and not isinstance(value, bool):
            pid = value
    completed = _run(["ps", "-p", str(pid), "-o", "pid,args"], timeout=30.0)
    return {
        "container_id": container_id,
        "host_pid": pid,
        "returncode": completed.returncode,
        "visible": bool(pid) and completed.returncode == 0 and str(pid) in completed.stdout,
        "scan": completed.stdout[-2000:],
        "stderr": completed.stderr[-500:],
    }


def _attach_visible(container_id: str) -> list[str]:
    """The frontend's ``docker start --attach`` client for this container."""

    completed = _run(["ps", "-eo", "pid,args"], timeout=30.0)
    if completed.returncode != 0:
        return [f"<ps unavailable: {completed.stderr.strip()}>"]
    return [
        line.strip() for line in completed.stdout.splitlines()
        if container_id in line and "docker start --attach" in line and "ps -eo" not in line
    ]


def _labeled_containers() -> list[str]:
    completed = _docker("ps", "--all", "--quiet", "--filter", OWNER_FILTER)
    if completed.returncode != 0:
        raise RuntimeError(f"docker ps failed: {completed.stderr.strip()}")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def _build_image(out: Path, image: str, base: str) -> dict:
    stage = out / "image-src"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "tests" / "support" / "npu_sim").mkdir(parents=True)
    shutil.copytree(ROOT / "tcenv", stage / "tcenv")
    for source, target in (
        (ROOT / "tests" / "__init__.py", stage / "tests" / "__init__.py"),
        (ROOT / "tests" / "support" / "__init__.py", stage / "tests" / "support" / "__init__.py"),
        (ROOT / "tests" / "support" / "npu_sim" / "__init__.py", stage / "tests" / "support" / "npu_sim" / "__init__.py"),
        (ROOT / "tests" / "support" / "npu_sim" / "fake_container_worker.py", stage / "tests" / "support" / "npu_sim" / "fake_container_worker.py"),
        (ROOT / "pyproject.toml", stage / "pyproject.toml"),
    ):
        if not source.exists():
            raise RuntimeError(f"missing build input {source}")
        shutil.copy2(source, target)
    (stage / "Dockerfile").write_text(
        "\n".join([
            f"FROM {base}",
            "ENV PYTHONPATH=/opt/taic HOME=/tmp PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1",
            "USER root",
            "RUN rm -rf /opt/taic/tcenv /opt/taic/tests",
            "COPY tcenv /opt/taic/tcenv",
            "COPY tests /opt/taic/tests",
            "COPY pyproject.toml /opt/taic/pyproject.toml",
            "USER 65534:65534",
            "WORKDIR /opt/taic",
            "",
        ]),
        encoding="utf-8",
    )
    build = _run(
        ["docker", "build", "--network", "none", "--tag", image, str(stage)],
        timeout=600.0,
    )
    (out / "image-build.log").write_text(build.stdout + "\n" + build.stderr, encoding="utf-8")
    if build.returncode != 0:
        raise RuntimeError(f"docker build failed; see {out / 'image-build.log'}")
    return {"image": image, "base": base, "log": "image-build.log", "digest": common.image_digest(image)}


def _wait_job(engine, request_obj, states: set[str], *, timeout_s: float = 90.0):
    return _wait(
        lambda: (lambda snapshot: snapshot if snapshot.state in states else None)(common.get(engine, request_obj)),
        timeout_s, f"job {request_obj.request_id} in {sorted(states)}",
    )


def _wait_attempt_container(engine, request_obj, *, timeout_s: float = 90.0, running: bool = True):
    def attached():
        snapshot = common.get(engine, request_obj)
        attempt = snapshot.current_attempt
        if attempt is None or not attempt.container_id:
            return None
        if not running:
            return attempt
        if _container_state(attempt.container_id) == "running":
            return attempt
        return None

    return _wait(attached, timeout_s, f"container for {request_obj.request_id}")


def _frontend_runtime() -> str:
    """Where the acceptance frontend itself runs (host or a harness container)."""

    if Path("/.dockerenv").exists():
        return "docker-container"
    if Path("/run/.containerenv").exists():
        return "podman-container"
    return "host"


def _check_preflight(report: Report, out: Path, image: str, digest: str) -> None:
    server = _docker("version", "--format", "{{.Server.Version}}")
    if server.returncode != 0:
        report.blockers.append(f"docker unavailable: {server.stderr.strip()}")
        report.skip("preflight_docker_and_pinned_image", server.stderr.strip()[:300])
        raise SystemExit("docker is unavailable")
    info = _docker("info", "--format", "{{.Driver}}|{{.CgroupDriver}}|{{.CgroupVersion}}")
    supervisor = common.supervisor(digest, image=image)
    report.environment.update({
        "hostname": socket.gethostname(),
        "frontend_runtime": _frontend_runtime(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "docker_server": server.stdout.strip(),
        "docker_info": info.stdout.strip(),
        "image": image,
        "image_digest": digest,
        "daemon_id": supervisor.daemon_identity(),
        "worker_build": common.worker_build(digest),
        "control_plane": "same host (VM-local Docker socket, no SSH data plane)",
    })
    identity_error = supervisor.verify_image()
    if identity_error:
        report.failed("preflight_docker_and_pinned_image", identity_error)
        raise SystemExit(identity_error)
    report.passed(
        "preflight_docker_and_pinned_image",
        f"docker {server.stdout.strip()} ({info.stdout.strip()}) and image {image} pinned to {digest}",
        image=image, image_digest=digest,
    )


def _check_timeout(report: Report, out: Path, state: Path, digest: str) -> None:
    engine = common.engine(state, digest)
    request_obj = common.request("sim-timeout", source=b"SLEEP60", candidate_timeout_ms=20_000)
    token = f"{CHILD_TOKEN_PREFIX}-{request_obj.request_id}"
    try:
        common.submit(engine, request_obj)
        attempt = _wait_attempt_container(engine, request_obj)
        container_id = attempt.container_id

        def child_visible() -> str | None:
            completed = _docker("top", container_id, "-o", "pid,args")
            return completed.stdout if token in completed.stdout else None

        top_stdout = _wait(child_visible, 15.0, "the sleeping child to appear in docker top")
        ps_control = _positive_process_control()
        host_pid = _host_pid_visibility(container_id)
        attach_before = _attach_visible(container_id)
        cgroup_before = _cgroup_exists(container_id)
        _write_evidence(out, "timeout-top", ["docker", "top", container_id, "-o", "pid,args"])
        (out / "evidence-timeout-attach.json").write_text(
            json.dumps({"attach": attach_before, "ps_control": ps_control}, indent=2),
            encoding="utf-8",
        )

        if token not in top_stdout:
            report.failed(
                "timeout_kills_parent_and_child",
                "the sleeping child process was not visible inside the running container",
                top=top_stdout[-4000:],
            )
        else:
            report.passed(
                "timeout_kills_parent_and_child",
                "parent worker and its 120s-sleep child ran inside the container before the deadline",
                container_id=container_id, child_token=token, top=top_stdout[-4000:],
            )
        final = _wait_job(engine, request_obj, {"completed"}, timeout_s=120.0)
        report_json = final.result.report_json() if final.result is not None else {}
        if report_json.get("status") != "timeout":
            report.failed(
                "timeout_kills_parent_and_child",
                f"timeout attempt did not produce a timeout report: {report_json.get('status')!r}",
            )
        state_after = _container_state(container_id)
        cgroup_after = _cgroup_exists(container_id)
        survivors = _ps_tokens_settled(token)
        attach_survivors = _ps_tokens_settled(container_id)
        labeled = _labeled_containers()
        temp_leftovers = list((state / "container-env").glob("*")) if (state / "container-env").is_dir() else []
        problems = []
        if state_after != "absent":
            problems.append(f"container state after delete is {state_after}")
        if cgroup_after:
            problems.append("container cgroup survived")
        if not cgroup_before:
            problems.append("container cgroup was never observed (cannot prove cgroup lifecycle)")
        if survivors:
            problems.append(f"child processes survived: {survivors}")
        if attach_survivors:
            problems.append(f"container processes/attach client survived: {attach_survivors}")
        if labeled:
            problems.append(f"label-matched containers remain: {labeled}")
        if temp_leftovers:
            problems.append(f"per-attempt env/temp files remain: {[p.name for p in temp_leftovers]}")
        if not attach_before:
            problems.append("the frontend's attach client was never observed while the worker ran")
        if not ps_control["visible"]:
            problems.append("the ps positive control was not visible; empty survivor scans prove nothing")
        if not host_pid["visible"]:
            problems.append(
                "the worker container's host PID was not visible; run the frontend with --pid=host "
                "or the empty survivor scans prove nothing"
            )
        evidence = {
            "container_id": container_id, "container_state_after": state_after,
            "cgroup_before": cgroup_before, "cgroup_after": cgroup_after,
            "survivor_processes": survivors, "attach_survivors": attach_survivors,
            "labeled_containers": labeled, "temp_leftovers": [p.name for p in temp_leftovers],
            "attach_before_timeout": attach_before[-2000:],
            "ps_positive_control": ps_control,
            "host_pid_visibility": host_pid,
        }
        if problems:
            report.failed("timeout_delete_removes_container_cgroup_and_temp", "; ".join(problems), **evidence)
        else:
            report.passed(
                "timeout_delete_removes_container_cgroup_and_temp",
                "timeout Kill+Delete removed the container, its cgroup and every process/temp path",
                **evidence,
            )
    finally:
        engine.close()


def _check_cancel(report: Report, state: Path, digest: str) -> None:
    engine = common.engine(state, digest)
    request_obj = common.request("sim-cancel", source=b"SLEEP60", candidate_timeout_ms=600_000)
    token = f"{CHILD_TOKEN_PREFIX}-{request_obj.request_id}"
    try:
        common.submit(engine, request_obj)
        attempt = _wait_attempt_container(engine, request_obj)
        ps_control = _positive_process_control()
        host_pid = _host_pid_visibility(attempt.container_id)

        # Cancel must land while the worker is executing the candidate, not
        # merely while a container record exists: require the child the fake
        # worker spawns after it read the private request.
        def child_visible() -> str | None:
            completed = _docker("top", attempt.container_id, "-o", "pid,args")
            return completed.stdout if token in completed.stdout else None

        top_before_cancel = _wait(child_visible, 30.0, "the cancelled worker's child")
        cancelled = engine.cancel(
            request_id=request_obj.request_id, request_digest=request_obj.request_digest,
            expected=engine.expected_identities, grant_token=common.TOKEN,
        )
        snapshot = common.get(engine, request_obj)
        problems = []
        if token not in top_before_cancel:
            problems.append("the worker was not executing the candidate before Cancel")
        if not ps_control["visible"]:
            problems.append("the ps positive control was not visible; empty survivor scans prove nothing")
        if not host_pid["visible"]:
            problems.append("the worker container's host PID was not visible before Cancel")
        if cancelled.state != "cancelled" or snapshot.state != "cancelled":
            problems.append(f"cancel returned {cancelled.state}/{snapshot.state}")
        if _container_state(attempt.container_id) != "absent":
            problems.append("container survived Cancel")
        survivors = _ps_tokens_settled(token)
        if survivors:
            problems.append(f"worker child survived Cancel: {survivors}")
        # A late result bound to the cancelled attempt/fence must be refused.
        forged = {
            "task": request_obj.task, "dtype": request_obj.dtype, "size": request_obj.size,
            "mode": request_obj.mode, "seed": request_obj.seed_int, "status": "ok",
            "stage": "done", "target_id": request_obj.target_id,
            "protocol_version": "3", "simulated": True,
        }
        late = engine.store.complete_attempt(
            request_obj.request_id, snapshot.current_attempt, report=forged,
            evidence={"candidate_sha256": request_obj.candidate_sha256},
        )
        if late.state != "cancelled" or late.result is not None:
            problems.append(f"old-fence result was not rejected: state={late.state}")
        if common.get(engine, request_obj).result is not None:
            problems.append("a result exists for a cancelled job")

        evidence = {
            "container_id": attempt.container_id, "cancel_state": cancelled.state,
            "late_commit_state": late.state, "survivor_processes": survivors,
            "child_token_before_cancel": token in top_before_cancel,
            "top_before_cancel": top_before_cancel[-2000:],
            "ps_positive_control": ps_control,
            "host_pid_visibility": host_pid,
        }
        if problems:
            report.failed("cancel_kills_worker_and_rejects_old_result", "; ".join(problems), **evidence)
        else:
            report.passed(
                "cancel_kills_worker_and_rejects_old_result",
                "Cancel killed and deleted an actively executing worker synchronously; the old attempt/fence cannot commit",
                **evidence,
            )
    finally:
        engine.close()


def _crash_state(out: Path, state: Path, digest: str, request_id: str) -> dict:
    ready = out / f"crash-{request_id}-ready.json"
    log = out / f"crash-{request_id}-frontend.log"
    token = f"{CHILD_TOKEN_PREFIX}-{request_id}"
    if ready.exists():
        ready.unlink()
    environment = dict(os.environ, PYTHONPATH=str(ROOT))
    with log.open("wb") as handle:
        child = subprocess.Popen(
            [
                sys.executable, "-m", "tests.support.npu_sim.container_frontend_child",
                "--state-dir", str(state), "--digest", digest, "--docker", "docker",
                "--request-id", request_id, "--ready", str(ready),
            ],
            cwd=str(ROOT), env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
    try:
        _wait(ready.is_file, 120.0, f"crash frontend {request_id} to attach its container")
        info = json.loads(ready.read_text(encoding="utf-8"))
        if _container_state(info["container_id"]) != "running":
            raise RuntimeError("crashed frontend's container is not running")

        # The crash must land while the worker is executing the candidate: the
        # old container has to be live (and in the middle of the work) when the
        # frontend dies, or the recovery check would be vacuous.
        def child_visible() -> str | None:
            completed = _docker("top", info["container_id"], "-o", "pid,args")
            return completed.stdout if token in completed.stdout else None

        info["top_before_crash"] = _wait(child_visible, 30.0, "the crashing worker's child")
        info["child_token"] = token
        info["cgroup_before_crash"] = _cgroup_exists(info["container_id"])
        (out / f"evidence-crash-{request_id}-top.json").write_text(
            json.dumps({"top": info["top_before_crash"], "child_token": token}, indent=2),
            encoding="utf-8",
        )
        child.kill()  # SIGKILL: no handlers, no cleanup, a real frontend crash
        info["frontend_returncode"] = child.wait(timeout=30)
        # Liveness of the old worker must be established *after* the frontend
        # died, from the daemon, not from the frontend's own last words.
        info["was_running_after_crash"] = _container_state(info["container_id"]) == "running"
        _write_evidence(out, f"crash-{request_id}-inspect", ["docker", "inspect", "--type", "container", info["container_id"]])
    finally:
        if child.poll() is None:
            child.kill()
    info["frontend_log"] = log.name
    return info


def _check_crash(report: Report, out: Path, state: Path, digest: str) -> None:
    info = _crash_state(out, state, digest, "sim-crash")
    container_id = info["container_id"]
    still_running = info["was_running_after_crash"]
    recovered = common.engine(state, digest)
    try:
        snapshot = _wait(
            lambda: (lambda value: value if value.state == "interrupted" else None)(common.get(recovered, _request_from_info(info))),
            90.0, "crash recovery to finish",
        )
        problems = []
        if snapshot.quarantined:
            problems.append(f"recovery quarantined: {snapshot.recovery_error}")
        if snapshot.current_attempt.attempt_id != info["attempt_id"]:
            problems.append("recovery changed the attempt id")
        if _container_state(container_id) != "absent":
            problems.append("the crashed container was not cleaned")
        attach = _ps_tokens_settled(container_id)
        if attach:
            problems.append(f"the old attach client survived: {attach}")
        rows = recovered.store._conn.execute(
            "SELECT attempt_id FROM attempts WHERE request_id=?", (info["request_id"],),
        ).fetchall()
        cgroup_after = _cgroup_exists(container_id)
        evidence = {
            "container_id": container_id, "was_running_after_crash": still_running,
            "attempt_id": info["attempt_id"], "attempt_rows": len(rows),
            "failure": snapshot.failure.code if snapshot.failure else None,
            "recovery_error": snapshot.recovery_error, "frontend_log": info["frontend_log"],
            "child_token": info.get("child_token"),
            "child_running_before_crash": bool(info.get("top_before_crash")),
            "frontend_signal_exit": info.get("frontend_returncode"),
            "cgroup_before_crash": info.get("cgroup_before_crash"),
            "cgroup_after_recovery": cgroup_after,
        }
        if not info.get("child_token") or info["child_token"] not in (info.get("top_before_crash") or ""):
            problems.append("the old worker was not executing the candidate when the frontend crashed")
        if info.get("frontend_returncode") != -9:
            problems.append(f"the frontend was not killed by SIGKILL: {info.get('frontend_returncode')}")
        if not still_running:
            problems.append("the old container was not verified running after the frontend died")
        if not info.get("cgroup_before_crash"):
            problems.append("the old container cgroup was never observed while it ran")
        if cgroup_after:
            problems.append("the old container cgroup survived recovery")
        if problems:
            report.failed("crash_restart_cleans_container_and_marks_interrupted", "; ".join(problems), **evidence)
        else:
            report.passed(
                "crash_restart_cleans_container_and_marks_interrupted",
                "a SIGKILLed frontend's container was cleaned before the attempt was marked interrupted",
                **evidence,
            )
        if len(rows) != 1 or snapshot.state != "interrupted":
            report.failed(
                "crash_restart_never_reruns_the_attempt",
                f"recovery produced {len(rows)} attempt rows and state {snapshot.state}",
                **evidence,
            )
        else:
            report.passed(
                "crash_restart_never_reruns_the_attempt",
                "the running attempt was marked interrupted, not re-queued or re-executed",
                **evidence,
            )
    finally:
        recovered.close()


def _request_from_info(info: dict):
    """Rebuild the exact crashed request (same digest) from its id."""

    request_obj = common.request(
        info["request_id"], source=b"SLEEP60", candidate_timeout_ms=600_000,
    )
    if request_obj.request_digest != info["request_digest"]:
        raise RuntimeError("reconstructed crash request digest differs from the durable job")
    return request_obj


def _check_stale_fence(report: Report, out: Path, state: Path, digest: str) -> None:
    """A late result for an interrupted attempt must not touch the newer run."""

    info = json.loads((out / "crash-sim-crash-ready.json").read_text(encoding="utf-8"))
    engine = common.engine(state, digest)
    try:
        request_obj = _request_from_info(info)
        interrupted = _wait(
            lambda: (lambda value: value if value.state == "interrupted" else None)(
                common.get(engine, request_obj)
            ),
            90.0, "the interrupted attempt before Retry",
        )
        old_attempt = interrupted.current_attempt
        cap = common.capabilities(engine)
        retried = engine.retry(
            request_id=request_obj.request_id, request_digest=request_obj.request_digest,
            expected=engine.expected_identities,
            interrupted_attempt_id=old_attempt.attempt_id,
            capability_id=cap.capability_id, grant_token=common.TOKEN,
        )
        newer = _wait_attempt_container(engine, request_obj)
        token = f"{CHILD_TOKEN_PREFIX}-{request_obj.request_id}"

        def newer_child_visible() -> str | None:
            completed = _docker("top", newer.container_id, "-o", "pid,args")
            return completed.stdout if token in completed.stdout else None

        top_newer = _wait(newer_child_visible, 30.0, "the newer attempt's child")
        host_pid = _host_pid_visibility(newer.container_id)
        _write_evidence(out, "stale-fence-top", ["docker", "top", newer.container_id, "-o", "pid,args"])
        was_running = _container_state(newer.container_id) == "running"
        forged = {
            "task": request_obj.task, "dtype": request_obj.dtype, "size": request_obj.size,
            "mode": request_obj.mode, "seed": request_obj.seed_int, "status": "ok",
            "stage": "done", "target_id": request_obj.target_id,
            "protocol_version": "3", "simulated": True,
        }
        stale = engine.store.complete_attempt(
            request_obj.request_id, old_attempt, report=forged,
            evidence={"candidate_sha256": request_obj.candidate_sha256},
        )
        problems = []
        if old_attempt.attempt_id == newer.attempt_id:
            problems.append("Retry did not create a newer attempt")
        if not top_newer:
            problems.append("the newer attempt was not executing before the stale commit")
        if not host_pid["visible"]:
            problems.append("the newer container's host PID was not visible")
        if not was_running:
            problems.append("the newer attempt's container was not running")
        if stale.result is not None:
            problems.append("the stale fence committed a result while a newer attempt ran")
        if stale.state != "running":
            problems.append(f"the stale fence changed the newer job state to {stale.state}")
        if stale.current_attempt is None or stale.current_attempt.attempt_id != newer.attempt_id:
            problems.append("the stale fence changed the current attempt")
        if common.get(engine, request_obj).result is not None:
            problems.append("a result exists before the newer attempt finished")
        engine.cancel(
            request_id=request_obj.request_id, request_digest=request_obj.request_digest,
            expected=engine.expected_identities, grant_token=common.TOKEN,
        )
        if _container_state(newer.container_id) != "absent":
            problems.append("the newer attempt's container survived Cancel")
        evidence = {
            "old_attempt_id": old_attempt.attempt_id,
            "newer_attempt_id": newer.attempt_id,
            "retry_state": retried.state,
            "newer_container_id": newer.container_id,
            "newer_child_token": token,
            "newer_was_running": was_running,
            "stale_commit_state": stale.state,
            "stale_commit_result": stale.result is not None,
            "host_pid_visibility": host_pid,
            "ps_positive_control": _positive_process_control(),
        }
        (out / "stale-fence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
        if problems:
            report.failed(
                "stale_fence_cannot_commit_while_a_newer_attempt_runs",
                "; ".join(problems), **evidence,
            )
        else:
            report.passed(
                "stale_fence_cannot_commit_while_a_newer_attempt_runs",
                "a result for the interrupted attempt was refused while the newer attempt was executing",
                **evidence,
            )
    finally:
        engine.close()


def _check_crash_quarantine(report: Report, out: Path, state: Path, digest: str) -> None:
    info = _crash_state(out, state, digest, "sim-quarantine")
    container_id = info["container_id"]
    wrapper = out / "docker-fail-rm"
    real_docker = shutil.which("docker")
    if not real_docker:
        raise RuntimeError("docker CLI is not on PATH for the fault-injection wrapper")
    wrapper.write_text(
        "\n".join([
            "#!/usr/bin/env python3",
            "import os, sys",
            "args = sys.argv[1:]",
            "while args and args[0] == '--host':",
            "    args = args[2:]",
            "if args and args[0] == 'rm':",
            "    sys.stderr.write('Error response from daemon: injected rm failure\\n')",
            "    raise SystemExit(1)",
            f"os.execv({real_docker!r}, ['docker', *sys.argv[1:]])",
            "",
        ]),
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    quarantined_engine = common.engine(state, digest, docker=str(wrapper))
    try:
        request_obj = _request_from_info(info)
        snapshot = _wait(
            lambda: (lambda value: value if value.quarantined else None)(common.get(quarantined_engine, request_obj)),
            90.0, "crash quarantine",
        )
        problems = []
        # The quarantine reason must be the failure the wrapper actually
        # injected: any other "could not delete" (a missing label, a name
        # conflict, an unrelated daemon error) would otherwise pass silently.
        if "could not delete" not in snapshot.recovery_error:
            problems.append(f"unexpected quarantine reason: {snapshot.recovery_error!r}")
        if "injected rm failure" not in snapshot.recovery_error:
            problems.append(
                "quarantine was not caused by the injected rm failure: "
                f"{snapshot.recovery_error!r}"
            )
        cap = common.capabilities(quarantined_engine)
        try:
            quarantined_engine.retry(
                request_id=request_obj.request_id, request_digest=request_obj.request_digest,
                expected=quarantined_engine.expected_identities,
                interrupted_attempt_id=snapshot.current_attempt.attempt_id,
                capability_id=cap.capability_id, grant_token=common.TOKEN,
            )
            problems.append("Retry was accepted while cleanup was unverified")
        except JobError as exc:
            if exc.code != "failed_precondition":
                problems.append(f"Retry failed with {exc.code}")
        evidence = {
            "container_id": container_id, "quarantined": snapshot.quarantined,
            "recovery_error": snapshot.recovery_error, "injected_fault": "rm_failure",
        }
        if problems:
            report.failed(
                "crash_quarantine_blocks_retry_when_cleanup_is_unverified",
                "; ".join(problems), **evidence,
            )
        else:
            report.passed(
                "crash_quarantine_blocks_retry_when_cleanup_is_unverified",
                "an unverified container delete quarantined the job and Retry was refused",
                **evidence,
            )
    finally:
        quarantined_engine.close()
    # Operator path: a real supervisor must clean the quarantined container and
    # leave the attempt interrupted (explicit Retry only).
    real = common.engine(state, digest)
    try:
        request_obj = _request_from_info(info)
        interrupted = _wait(
            lambda: (lambda value: value if value.state == "interrupted" else None)(
                common.get(real, request_obj)
            ),
            90.0, "operator recovery",
        )
        released = real.release_quarantine(request_obj.request_id)
        if _container_state(container_id) != "absent":
            report.failed(
                "crash_quarantine_blocks_retry_when_cleanup_is_unverified",
                "operator release did not remove the quarantined container",
            )
        elif released.state != "interrupted" or released.quarantined:
            report.failed(
                "crash_quarantine_blocks_retry_when_cleanup_is_unverified",
                f"operator release left state={released.state} quarantined={released.quarantined}",
            )
        else:
            report.checks["crash_quarantine_blocks_retry_when_cleanup_is_unverified"]["evidence"].update({
                "operator_recovery_state": interrupted.state,
                "operator_released_state": released.state,
                "operator_released_quarantine": released.quarantined,
            })
    finally:
        real.close()


def _check_normal(report: Report, out: Path, state: Path, digest: str) -> None:
    signing_key = b"container-sim-signing-key-32-bytes!"
    engine = common.engine(
        state, digest, signing_key=signing_key, signing_key_id="sim-container-site/key-1",
    )
    request_obj = common.request("sim-normal")
    observed: dict = {}
    signature_verified = False
    original_attach = engine.store.attach_container

    def attach_and_observe(*args, **kwargs):
        container_id = kwargs.get("container_id", "")
        observed["state_at_attach"] = _container_state(container_id)
        observed["labeled_at_attach"] = _labeled_containers()
        return original_attach(*args, **kwargs)

    engine.store.attach_container = attach_and_observe
    try:
        common.submit(engine, request_obj)
        running = _wait_attempt_container(engine, request_obj)
        cgroup_running = _cgroup_exists(running.container_id)
        observed["cgroup_running"] = cgroup_running
        observed["daemon_id"] = engine.supervisor.daemon_identity()
        final = _wait_job(engine, request_obj, {"completed"}, timeout_s=120.0)
        problems = []
        result = final.result
        attempt = final.current_attempt
        if result is None:
            problems.append("no result was committed")
        else:
            report_json = result.report_json()
            evidence_json = result.evidence_json() or {}
            receipt = result.receipt
            provenance = result.provenance
            if report_json.get("status") != "ok" or report_json.get("simulated") is not True:
                problems.append(f"unexpected report: {report_json.get('status')}, simulated={report_json.get('simulated')}")
            if evidence_json.get("candidate_sha256") != request_obj.candidate_sha256:
                problems.append("evidence does not bind the candidate")
            if receipt.get("attempt_id") != attempt.attempt_id:
                problems.append("receipt is not bound to the attempt")
            if receipt.get("request_digest") != request_obj.request_digest:
                problems.append("receipt does not bind the request digest")
            if receipt.get("candidate_sha256") != request_obj.candidate_sha256:
                problems.append("receipt does not bind the candidate")
            if receipt.get("lease_fence") != str(attempt.lease_fence):
                problems.append("receipt fence differs from the attempt")
            if provenance.get("worker_build") != common.worker_build(digest):
                problems.append("provenance does not bind the pinned image digest")
            if result.signature is None:
                problems.append("result is unsigned")
            elif result.signature.get("key_id") != "sim-container-site/key-1":
                problems.append("result signature key id differs")
            else:
                # Recompute the HMAC independently: the acceptance must verify
                # the signature bytes, not merely the key id it claims.
                signed = {
                    "provenance": result.provenance,
                    "report": result.report_descriptor,
                    "evidence": result.evidence_descriptor,
                    "receipt": receipt,
                }
                key_id = result.signature["key_id"]
                recomputed = contracts.sign_result(signed, key_id=key_id, key=signing_key)
                if recomputed != result.signature.get("signature"):
                    problems.append("result HMAC does not verify against the simulated key")
                elif not contracts.verify_result_signature(
                    signed, key_id=key_id, key=signing_key,
                    signature=result.signature["signature"],
                ):
                    problems.append("contract result-signature verification failed")
                else:
                    signature_verified = True
            if attempt.execution_kind != "container":
                problems.append("attempt execution kind is not container")
            if attempt.container_image_digest != digest:
                problems.append("attempt image digest differs")
            if attempt.container_runtime != "runc":
                problems.append("attempt runtime profile differs")
            if observed.get("state_at_attach") != "created":
                problems.append(f"container state at durable attach was {observed.get('state_at_attach')!r}, not created")
            labels_at_attach = observed.get("labeled_at_attach") or []
            if sorted(labels_at_attach) != [attempt.container_id[:12]]:
                problems.append(
                    f"exactly one owned container must exist at the durable attach: {labels_at_attach}"
                )
            if not cgroup_running:
                problems.append("the worker cgroup was never observed while the attempt ran")
            if _cgroup_exists(attempt.container_id):
                problems.append("the worker cgroup survived a completed attempt")
            if _container_state(attempt.container_id) != "absent":
                problems.append("container survived a completed attempt")
            if attempt.container_daemon != observed.get("daemon_id"):
                problems.append("the attempt did not persist the executing daemon identity")
            leftovers = [p.name for p in (state / "container-env").glob("*")] if (state / "container-env").is_dir() else []
            if leftovers:
                problems.append(f"per-attempt env files remain: {leftovers}")
        evidence = {
            "attempt_id": attempt.attempt_id if attempt else "",
            "container_id": attempt.container_id if attempt else "",
            "state_at_durable_attach": observed.get("state_at_attach"),
            "labeled_at_attach": observed.get("labeled_at_attach"),
            "receipt_id": result.receipt.get("receipt_id") if result else "",
            "signature_key_id": result.signature.get("key_id") if result and result.signature else "",
            "signature_verified": signature_verified,
            "worker_build": result.provenance.get("worker_build") if result else "",
            "report_status": result.report_json().get("status") if result else "",
            "cgroup_running": cgroup_running,
            "daemon_id": observed.get("daemon_id"),
        }
        if problems:
            report.failed("normal_report_evidence_and_receipt_validate", "; ".join(problems), **evidence)
        else:
            report.passed(
                "normal_report_evidence_and_receipt_validate",
                "fake report/evidence/attempt receipt validated; the container was created before its durable attach and removed after commit",
                **evidence,
            )
    finally:
        engine.close()


def _check_posture(report: Report, out: Path, state: Path, digest: str) -> None:
    engine = common.engine(state, digest)
    request_obj = common.request("sim-posture", source=b"SLEEP60", candidate_timeout_ms=600_000)
    try:
        common.submit(engine, request_obj)
        attempt = _wait_attempt_container(engine, request_obj)
        container_id = attempt.container_id
        inspect = _inspect_container(container_id) or {}
        host = inspect.get("HostConfig") or {}
        config = inspect.get("Config") or {}
        env_names = [item.split("=", 1)[0] for item in (config.get("Env") or [])]
        _write_evidence(out, "posture-inspect", ["docker", "inspect", container_id])
        probe = (
            "import json,os,socket,sys;"
            "r={'uid':os.getuid(),'gid':os.getgid(),"
            "'docker_socket':os.path.exists('/var/run/docker.sock'),"
            "'host_home':os.path.exists('/home/kuafu'),"
            "'host_repo':os.path.exists('/home/kuafu/taic-repo') or os.path.exists('/home/kuafu/tcsrc'),"
            "'repo_mount':os.path.exists('/workspace'),"
            "'cwd':os.getcwd(),'root_writable':True,'scratch_writable':True,"
            "'network':False,"
            "'cgroup':open('/proc/1/cgroup').read().strip().splitlines()[-1]};"
            "\ntry:\n open('/root-probe','w').write('x')\nexcept OSError:\n r['root_writable']=False\n"
            "try:\n open('/scratch/scratch-probe','w').write('x')\nexcept OSError:\n r['scratch_writable']=False\n"
            "try:\n socket.create_connection(('1.1.1.1',80),timeout=1).close()\nexcept OSError:\n r['network']=True\n"
            "print(json.dumps(r))"
        )
        exec_result = _docker("exec", container_id, "python3", "-c", probe)
        exec_document: dict = {}
        try:
            exec_document = json.loads(exec_result.stdout)
        except ValueError:
            pass
        (out / "posture-in-container.json").write_text(
            json.dumps({"returncode": exec_result.returncode, "stdout": exec_result.stdout, "stderr": exec_result.stderr}, indent=2),
            encoding="utf-8",
        )
        problems = []
        expected_checks = (
            (host.get("NetworkMode") == "none", "network is not none"),
            (host.get("ReadonlyRootfs") is True, "root filesystem is not read-only"),
            (list(host.get("CapDrop") or []) == ["ALL"], "capabilities were not dropped"),
            (host.get("Privileged") is not True, "container is privileged"),
            (not (host.get("Binds") or []), "container has a bind mount"),
            (not (host.get("PidMode") or ""), "container shares the host pid namespace"),
            (isinstance(host.get("PidsLimit"), int) and host["PidsLimit"] > 0, "pids limit missing"),
            (isinstance(host.get("Memory"), int) and host["Memory"] > 0, "memory limit missing"),
            (isinstance(host.get("NanoCpus"), int) and host["NanoCpus"] > 0, "cpu limit missing"),
            (config.get("User") == "65534:65534", "worker user is not the pinned non-root user"),
            (all(m.get("Type") == "tmpfs" for m in (inspect.get("Mounts") or [])), "non-tmpfs mount present"),
            (not any(name in {"DOCKER_HOST", "SSH_AUTH_SOCK"} for name in env_names), "host docker env leaked"),
            (not any(name.upper().endswith(("API_KEY", "TOKEN", "SECRET", "PASSWORD")) for name in env_names), "credential-shaped env names"),
            (exec_document.get("uid") not in (None, 0), "container process is root"),
            (exec_document.get("root_writable") is False, "root filesystem is writable in the container"),
            (exec_document.get("scratch_writable") is True, "per-attempt scratch tmpfs is not writable"),
            (exec_document.get("docker_socket") is False, "docker socket is present in the worker"),
            (exec_document.get("host_home") is False, "control-host home is visible in the worker"),
            (exec_document.get("host_repo") is False, "control-host repo is visible in the worker"),
            (exec_document.get("network") is True, "worker network appears reachable"),
        )
        for ok, message in expected_checks:
            if not ok:
                problems.append(message)
        evidence = {
            "container_id": container_id,
            "host_config": {
                key: host.get(key) for key in (
                    "NetworkMode", "ReadonlyRootfs", "CapDrop", "SecurityOpt", "Privileged",
                    "Binds", "PidMode", "PidsLimit", "Memory", "NanoCpus", "Runtime",
                )
            },
            "user": config.get("User"),
            "mounts": [m.get("Type") for m in (inspect.get("Mounts") or [])],
            "tmpfs": sorted((host.get("Tmpfs") or {}).keys()),
            "env_names": env_names,
            "in_container": exec_document,
            "evidence_files": ["evidence-posture-inspect.json", "posture-in-container.json"],
        }
        if problems:
            report.failed("worker_has_no_socket_home_repo_or_credentials", "; ".join(problems), **evidence)
        else:
            report.passed(
                "worker_has_no_socket_home_repo_or_credentials",
                "worker container has no Docker socket, control-host home/repo, credentials or network",
                **evidence,
            )
        engine.cancel(
            request_id=request_obj.request_id, request_digest=request_obj.request_digest,
            expected=engine.expected_identities, grant_token=common.TOKEN,
        )
        if _container_state(container_id) != "absent":
            report.failed("worker_has_no_socket_home_repo_or_credentials", "posture container survived Cancel")
    finally:
        engine.close()


def _check_production_entry(report: Report, out: Path, digest: str, image: str) -> None:
    """Exercise the production entry inside a gRPC-capable *test* container.

    The check container is not the worker: it only imports the production
    modules and proves that the production composition root builds the container
    supervisor and refuses test/fake workers.  The worker image never contains
    grpcio and never runs this module.
    """

    entry_image = os.environ.get("TAIC_CONTAINER_SIM_ENTRY_IMAGE", "taic-remote-grpc-sim:local")
    # Bind sources are resolved by the Docker daemon on the host.  When this
    # runner itself runs inside a harness container, the host path differs from
    # the runner's own ROOT.
    host_root = os.environ.get("TAIC_CONTAINER_SIM_HOST_ROOT", str(ROOT))
    available = _docker("image", "inspect", "--format", "{{.Id}}", entry_image)
    if available.returncode != 0:
        report.blockers.append(f"production-entry check image {entry_image!r} is unavailable")
        report.skip(
            "production_entry_rejects_test_worker_and_fake_backend",
            f"{entry_image} is unavailable for the grpc-capable entry check",
        )
        return
    # ``verify_deployment`` validates the worker module and the execution mode
    # before it reads any deployment file, so placeholder paths are enough to
    # reach both refusals without a real task pack or TLS material.
    deployment_args = [
        "--check",
        "--state-dir", "/tmp/entry-check-state",
        "--site-id", "s", "--device-id", "d", "--backend", "b", "--backend-version", "1",
        "--target-id", "t", "--worker-build", common.worker_build(digest),
        "--calibration-digest", "c" * 64,
        "--task-pack-root", "/tmp/entry-check-pack",
        "--grants-file", "/tmp/entry-check-grants.json",
        "--signing-keyring-file", "/tmp/entry-check-keyring.json",
        "--certificate-file", "/tmp/entry-check-cert.pem",
        "--private-key-file", "/tmp/entry-check-key.pem",
        "--client-ca-file", "/tmp/entry-check-ca.pem",
    ]
    run_prefix = [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--tmpfs", "/tmp:rw,size=16m", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--user", "65534:65534",
        "--workdir", "/opt/taic", "-e", "PYTHONPATH=/opt/taic", "-e", "HOME=/tmp",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        "-v", f"{host_root}:/opt/taic:ro", "--entrypoint", "python3", entry_image,
        "-m", "tcenv.remote.grpc_service",
    ]
    refusal = _run(
        run_prefix + deployment_args + [
            "--worker-module", "tests.support.npu_sim.fake_container_worker",
        ],
        timeout=120.0,
    )
    refuse_process = _run(
        run_prefix + deployment_args + ["--worker-execution", "process"],
        timeout=120.0,
    )
    snippet = (
        "import inspect, json, sys;"
        "from types import SimpleNamespace;"
        "import tcenv.remote.grpc_service as gs, tcenv.remote.production as p, tcenv.remote.worker;"
        "bad=[m for m in sys.modules if m=='tests' or m.startswith('tests.')];"
        "assert not bad, bad;"
        "src=inspect.getsource(p);"
        "assert 'tests.support' not in src and 'npu_sim' not in src;"
        "args=SimpleNamespace(worker_execution='container',worker_image=sys.argv[1],"
        "worker_image_digest=sys.argv[2],worker_build=sys.argv[3],worker_python='python3',"
        "docker_binary='docker',docker_host='',worker_runtime='',worker_runtime_profile='runc',"
        "worker_user='65534:65534',worker_pids_limit=256,worker_memory='2g',worker_cpus='2',"
        "worker_scratch_size='512m',worker_tmp_size='64m',worker_image_pythonpath='/opt/taic',"
        "container_cli_timeout_s=30.0,container_kill_timeout_s=15.0);"
        "s=gs.build_supervisor(args);"
        "print(json.dumps({'type':type(s).__name__,'digest':s.config.image_digest,"
        "'image':s.config.image,'profile':s.runtime_profile}))"
    )
    dynamic = _run(
        [
            "docker", "run", "--rm", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=16m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--user", "65534:65534",
            "--workdir", "/opt/taic", "-e", "PYTHONPATH=/opt/taic", "-e", "HOME=/tmp",
            "-v", f"{host_root}:/opt/taic:ro", "--entrypoint", "python3", entry_image,
            "-c", snippet, image, digest, common.worker_build(digest),
        ],
        timeout=120.0,
    )
    try:
        built = json.loads(dynamic.stdout.strip().splitlines()[-1]) if dynamic.returncode == 0 else {}
    except (ValueError, IndexError):
        built = {}
    problems = []
    if built.get("type") != "DockerAttemptSupervisor":
        problems.append(f"the production composition root built {built.get('type')!r}")
    if built.get("digest") != digest:
        problems.append("the production supervisor did not pin the acceptance image digest")
    if dynamic.returncode != 0:
        problems.append(f"production modules import test support: {dynamic.stderr.strip()[:300]}")
    if refusal.returncode == 0 or "test/composition worker" not in (refusal.stderr + refusal.stdout):
        problems.append("the production entry accepted a test worker module")
    if refuse_process.returncode == 0 or "local development only" not in (refuse_process.stderr + refuse_process.stdout):
        problems.append("the production entry did not refuse process execution without an explicit opt-in")
    evidence = {
        "entry_check_image": entry_image,
        "supervisor": built.get("type"),
        "pinned_digest": built.get("digest"),
        "production_module_import_rc": dynamic.returncode,
        "test_worker_refusal_rc": refusal.returncode,
        "process_mode_refusal_rc": refuse_process.returncode,
        "test_worker_refusal_message": (refusal.stderr + refusal.stdout).strip()[-300:],
        "process_mode_refusal_message": (refuse_process.stderr + refuse_process.stdout).strip()[-300:],
    }
    (out / "production-entry-checks.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if problems:
        report.failed("production_entry_rejects_test_worker_and_fake_backend", "; ".join(problems), **evidence)
    else:
        report.passed(
            "production_entry_rejects_test_worker_and_fake_backend",
            "the production gRPC root builds the container supervisor and refuses test/fake workers and implicit process mode",
            **evidence,
        )


def _final_cleanup(report: Report, out: Path, states: list[Path]) -> dict:
    leftover = _labeled_containers()
    removed = []
    for container_id in leftover:
        _docker("rm", "--force", container_id)
        removed.append(container_id)
    remaining = _labeled_containers()
    by_name = _docker("ps", "--all", "--quiet", "--filter", "name=tcenv-attempt-")
    if by_name.returncode != 0:
        raise RuntimeError(f"docker ps (name filter) failed: {by_name.stderr.strip()[:300]}")
    named = [line for line in by_name.stdout.splitlines() if line.strip()]
    tokens = _ps_tokens_settled(CHILD_TOKEN_PREFIX)
    attach = _ps_tokens_settled("docker start --attach --interactive")
    temp: list[str] = []
    for state in states:
        env_dir = state / "container-env"
        if env_dir.is_dir():
            temp.extend(str(path) for path in env_dir.glob("*"))
    scratch = list(Path("/tmp").glob("tcenv-container-sim-*"))
    ps_control = _positive_process_control()
    evidence = {
        "removed_containers": removed,
        "remaining_labeled_containers": remaining,
        "remaining_named_containers": named,
        "survivor_processes": tokens,
        "survivor_attach_clients": attach,
        "temp_files": temp,
        "scratch_dirs": [str(path) for path in scratch],
        "ps_positive_control": ps_control,
    }
    problems = []
    for key in (
        "remaining_labeled_containers", "remaining_named_containers",
        "survivor_processes", "survivor_attach_clients", "temp_files", "scratch_dirs",
    ):
        if evidence[key]:
            problems.append(f"{key}: {evidence[key]}")
    if not ps_control["visible"]:
        problems.append("the ps positive control was not visible; empty process scans prove nothing")
    final = not problems
    (out / "final-cleanup.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    if final:
        report.passed("final_cleanup_has_no_containers_processes_or_temp", "no container, process or temp path remains", **evidence)
    else:
        report.failed("final_cleanup_has_no_containers_processes_or_temp", "; ".join(problems), **evidence)
    return {"final": final, **evidence}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="container_sim_runner")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image", default=common.IMAGE)
    parser.add_argument("--base-image", default="taic-remote-sim:local")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument(
        "--image-build-log", default="",
        help="path to the host-side image build log when the image was prebuilt",
    )
    parser.add_argument("--keep-state", action="store_true")
    args = parser.parse_args(argv)
    if os.environ.get("TAIC_CONTAINER_SIM") != "1":
        raise SystemExit("set TAIC_CONTAINER_SIM=1 to authorize the real-VM container acceptance")

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = Report(out)
    log_handle = (out / "container-sim.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message}"
        print(line, flush=True)
        log_handle.write(line + "\n")
        log_handle.flush()

    states = [out / name for name in ("state-timeout", "state-cancel", "state-crash", "state-quarantine", "state-normal", "state-posture")]
    for state in states:
        if state.exists():
            shutil.rmtree(state)
        state.mkdir(parents=True, exist_ok=True)
    try:
        log(f"acceptance start out={out} image={args.image}")
        if args.build_image:
            build = _build_image(out, args.image, args.base_image)
            log(f"built worker image: {build}")
        else:
            report.environment["image_build"] = (
                "prebuilt on the host (see log)" if args.image_build_log else "reused existing image"
            )
            report.environment["image_build_log"] = args.image_build_log
        digest = common.image_digest(args.image)
        log(f"worker image digest {digest}")
        if args.build_image:
            report.environment["image_build_log"] = "image-build.log"
        _check_preflight(report, out, args.image, digest)
        _check_timeout(report, out, states[0], digest)
        _check_cancel(report, states[1], digest)
        _check_crash(report, out, states[2], digest)
        _check_stale_fence(report, out, states[2], digest)
        _check_crash_quarantine(report, out, states[3], digest)
        _check_normal(report, out, states[4], digest)
        _check_posture(report, out, states[5], digest)
        _check_production_entry(report, out, digest, args.image)
        log("all execution checks finished")
    except SystemExit as exc:
        if str(exc):
            report.blockers.append(str(exc))
        log(f"acceptance blocked: {exc}")
    except Exception as exc:  # any unexpected failure makes the report failed
        report.blockers.append(f"{type(exc).__name__}: {exc}")
        log(f"acceptance failed: {type(exc).__name__}: {exc}")
    finally:
        cleanup = _final_cleanup(report, out, states)
        extra = {
            "no_run_config_json": not (out / "run_config.json").exists(),
            "no_confirmed_or_catalog_written": not any(
                (out / name).exists() for name in ("confirmed.json", "catalog.json", "brief.json")
            ),
            "note": "CPU-only fake backend, simulated=true, no hardware performance claim",
        }
        document = report.write(cleanup=cleanup, extra=extra)
        log(f"report written status={document['status']} cleanup_final={cleanup['final']}")
        log_handle.close()
        if not args.keep_state:
            for state in states:
                # Keep the durable databases as evidence but remove large scratch.
                scratch = state / "container-env"
                if scratch.is_dir():
                    shutil.rmtree(scratch, ignore_errors=True)
    return 0 if document["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
