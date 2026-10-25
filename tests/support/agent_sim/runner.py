"""Opt-in Docker isolated-agent simulation (Phase 4, simulated only).

It starts a pre-provisioned ssh host in a container (Ubuntu 24.04 + sshd + Python 3.12), runs
the flow with ``harness=env`` so the fake ACP agent executes **inside** that container, hands
it a short-lived scoped token through a host-side helper (``docker exec``), and lets it call a
control-host fake provider over ``host.docker.internal``.  The workspace is staged and
collected by the real SSH transport.  No model, provider key, GPU or NPU is used.

The fixture flow deliberately relaxes ``online`` to ``ALL`` so the fake agent can reach the
control-host provider; the production flow keeps ``online=NONE``.  This run therefore proves
isolation placement, credential delivery and artifact flow, not the production network fence.

Run directly (writes an acceptance artifact):

    python -m tests.support.agent_sim.runner --report runs/remote-agent-sim-<date>/report.json

or through the opt-in pytest:

    TAIC_DOCKER_AGENT_SIM=1 uv run pytest -q -n 0 tests/shared/test_agent_sim_docker.py

``status=passed`` is accepted only when every required check actually ran and passed and the
cleanup left no container or credential behind.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SIM = Path(__file__).resolve().parent
FLOW = SIM / "flow"
IMAGE = "taic-agent-sim:local"
CONTAINER_PREFIX = "taic-agent-sim-"
REQUIRED_CHECKS = (
    "image_built",
    "ssh_reachable",
    "remote_agent_ran_in_container",
    "credential_provisioned_to_agent",
    "provider_called_with_scoped_token",
    "submission_collected",
    "canonical_layout_materialized",
    "harness_recorded_env",
    "credential_revoked",
    "credential_file_removed",
    "remote_workspace_cleaned",
    "no_orphan_container",
    "cost_incomplete_for_fake_agent",
)


def _run(argv, *, input_text: str | None = None, timeout: float = 120, env=None, check: bool = True):
    completed = subprocess.run(
        argv, input=input_text, capture_output=True, text=True, timeout=timeout,
        env=env or os.environ.copy(), check=False,
    )
    if check and completed.returncode != 0:
        raise RuntimeError(
            f"{' '.join(argv[:3])} failed ({completed.returncode}): "
            f"{(completed.stderr or completed.stdout or '')[-800:]}"
        )
    return completed


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ssh(argv, port: int, key: Path, command: str, *, check: bool = False, timeout: float = 60):
    return _run(
        [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
            "-i", str(key), "-p", str(port), "agent@127.0.0.1", command,
        ],
        check=check, timeout=timeout,
    )


def _build_image() -> str:
    env = dict(os.environ)
    env["DOCKER_BUILDKIT"] = "0"
    _run(["docker", "build", "-t", IMAGE, str(SIM)], env=env, timeout=900)
    return _run(["docker", "image", "inspect", "-f", "{{.Id}}", IMAGE]).stdout.strip()


def _start_container(name: str, port: int, public_key: Path) -> None:
    _run(["docker", "rm", "-f", name], check=False, timeout=60)
    _run(
        [
            "docker", "run", "-d", "--name", name,
            # The test host needs to install its own Landlock/seccomp fence and to use
            # ptrace; the nested container profile would otherwise block it.  The container
            # runs only the fake agent and holds no real credential.
            "--security-opt", "seccomp=unconfined",
            "-p", f"127.0.0.1:{port}:22",
            "-v", f"{public_key}:/home/agent/.ssh/authorized_keys:ro",
            IMAGE,
        ],
        timeout=120,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        if _ssh([], port, public_key.with_suffix(""), "true").returncode == 0:
            return
        time.sleep(0.5)
    raise RuntimeError("the simulated ssh host did not accept a connection within 30s")


def _helper_script(path: Path, container: str) -> None:
    """A host-side helper: stdin JSON in, remote credential file written or removed."""
    path.write_text(
        "import json, subprocess, sys\n"
        "doc = json.load(sys.stdin)\n"
        f"container = {container!r}\n"
        "if doc.get('action') == 'revoke':\n"
        "    subprocess.run(['docker', 'exec', container, 'sh', '-c', 'rm -f /run/taic/credential.json'], check=False)\n"
        "    print(json.dumps({'ok': True}))\n"
        "    raise SystemExit(0)\n"
        "payload = json.dumps({'base_url': doc['provider_base_url'], 'token': doc['token']})\n"
        "completed = subprocess.run(\n"
        "    ['docker', 'exec', '-i', container, 'sh', '-c', 'mkdir -p /run/taic && cat > /run/taic/credential.json'],\n"
        "    input=payload, text=True, capture_output=True, check=False,\n"
        ")\n"
        "if completed.returncode != 0:\n"
        "    sys.stderr.write(completed.stderr[-400:])\n"
        "    raise SystemExit(completed.returncode)\n"
        "print(json.dumps({'token_in_env': True}))\n",
        encoding="utf-8",
    )


def _settings(path: Path, port: int, key: Path) -> None:
    path.write_text(
        "clis:\n"
        "  fake-acp:\n"
        "    command: [\"python3\", \"/opt/taic/fake_acp.py\"]\n"
        "    hosts: [\"host.docker.internal\"]\n"
        "    state: [\"/tmp/fake-acp-state\"]\n"
        "runtimes:\n"
        "  ssh:\n"
        "    gpu-box:\n"
        "      host: 127.0.0.1\n"
        "      user: agent\n"
        f"      port: {port}\n"
        f"      identity_file: {key}\n"
        "      proxy_jump: ''\n"
        "      options:\n"
        "        StrictHostKeyChecking: 'no'\n"
        "        UserKnownHostsFile: /dev/null\n"
        "        LogLevel: ERROR\n"
        "      alias: ''\n"
        "      config: ''\n"
        "      workdir: /srv/taic\n"
        "      fallback: []\n"
        "      made: typed\n"
        "      harness: env\n",
        encoding="utf-8",
    )


def _brief(work: Path):
    from taic.kernel_agent import SessionBrief

    return SessionBrief(
        repo=ROOT, task_name="relu", dtype="fp16", size="small", kernel_name="demo",
        triton_source="def relu(x): return x", description="isolated fake agent simulation",
        contract_text="kernel demo", abi_stub="kernel void demo()", rtol=1e-3, atol_rel=1e-3,
        seedless=True, label="init", work_dir=str(work),
    )


def run_simulation(*, report_path: Path | None = None, keep_container: bool = False) -> dict:
    """Build, run and clean one isolated fake-agent session; return the report document."""
    from taic.agent_isolation import ExternalHelperSeam, IsolatedAgentConfig, LoopbackFakeProvider, ScopedCredentialAuthority
    from taic.session_runtime import HmzSessionPlacement, SelfCheckBinding, SessionExecution, SshWorkspaceKernelSessionRunner
    from taic.kernel_agent import Prices
    from taic.workspace_env import RemoteWorkspaceConfig, SshWorkspaceTransport, session_layout

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "status": "passed" if ok else "failed", "detail": detail})
        return ok

    root = Path(tempfile.mkdtemp(prefix="taic-agent-sim-"))
    container = f"{CONTAINER_PREFIX}{uuid.uuid4().hex[:10]}"
    key = root / "id_ed25519"
    port = _free_port()
    provider = None
    image_id = ""
    run_id = "run"  # the runner derives the session run id from <run>/agent_work/<label>
    failure = ""
    try:
        _run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], timeout=60)
        image_id = _build_image()
        check("image_built", bool(image_id), image_id[:20])
        _start_container(container, port, key.with_suffix(".pub"))
        reachable = _ssh([], port, key, "python3 --version; command -v bash").returncode == 0
        check("ssh_reachable", reachable, f"127.0.0.1:{port}")

        home = root / "hmz-home"
        home.mkdir()
        settings = home / "settings.yaml"
        _settings(settings, port, key)
        os.environ["HUMANIZE_HOME"] = str(home)

        authority = ScopedCredentialAuthority()
        scope = {"run_id": run_id, "session_label": "init", "runtime": "gpu-box"}
        provider = LoopbackFakeProvider(authority, expected_scope=scope).start()
        base_url = provider.base_url.replace("127.0.0.1", "host.docker.internal")
        helper = root / "helper.py"
        _helper_script(helper, container)
        isolation = IsolatedAgentConfig(
            credential_seam="external-helper", helper=(sys.executable, str(helper)),
            credential_ttl_s=120, credential_max_uses=4,
        )
        config = RemoteWorkspaceConfig(
            runtime="gpu-box", root="/srv/taic", harness="env", settings_file=settings,
            isolation=isolation,
        )

        def seam_factory(cfg, *, authority, scope):
            return ExternalHelperSeam(
                authority, tuple(cfg.helper), expected_scope=scope, provider_base_url=base_url,
            )

        brief = _brief(root / "run" / "agent_work" / "init")
        runner = SshWorkspaceKernelSessionRunner(
            config, transport=SshWorkspaceTransport(config), epic_root=home / "epics",
            authority=authority, seam_factory=seam_factory, flow=FLOW,
            assistant_spec="fake-acp/fake-model",
        )
        work = Path(brief.work_dir)
        execution = SessionExecution(
            placement=HmzSessionPlacement(harness_placement="env", workspace_env="ssh", endpoint="gpu-box"),
            paths=runner.paths(work, "init"),
            prices=Prices(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
            turn_output_tokens=1000, turn_minutes=2,
            hmz_binary=shutil.which("hmz"),
            selfcheck=SelfCheckBinding(mode="local_cli"),
        )
        result = runner.execute(brief, execution=execution, provider="fake", model="fake", effort=None)
        session_detail = {
            "status": result.status,
            "error": result.error,
            "diagnostics": {k: v for k, v in (result.diagnostics or {}).items() if k != "response_preview"},
            "identity": result.identity,
        }

        source = result.source or ""
        check("remote_agent_ran_in_container", bool(source.startswith("kernel void demo")), source[:40])
        credential = (result.identity or {}).get("credential", {})
        check("credential_provisioned_to_agent", credential.get("provisioned_to_agent") is True, credential.get("mode", ""))
        notes_path = work / "NOTES.md"
        notes_text = notes_path.read_text(encoding="utf-8", errors="replace")[:2000] if notes_path.is_file() else ""
        notes = json.loads(notes_text) if notes_text else {}
        session_detail["notes"] = notes_text
        check(
            "provider_called_with_scoped_token",
            notes.get("credential_present") is True and notes.get("provider_status") == 200 and notes.get("provider_body") == "fake",
            json.dumps({k: notes.get(k) for k in ("credential_present", "provider_status", "provider_body", "provider_error")}),
        )
        check("submission_collected", bool(result.source), (result.identity or {}).get("collected", []))
        check(
            "canonical_layout_materialized",
            (work / "TASK.md").is_file() and (work / "NOTES.md").is_file() and (work / "relu-fp16-small-init.metal").is_file(),
            str(work),
        )
        check(
            "harness_recorded_env",
            (result.identity or {}).get("harness_recorded") == "env",
            str((result.identity or {}).get("harness_recorded")),
        )
        check("credential_revoked", credential.get("revoked") is True and "live_after_cleanup" not in credential, str(credential.get("revoked")))
        marker = _ssh([], port, key, "test -e /run/taic/credential.json")
        check("credential_file_removed", marker.returncode != 0)
        workdir = session_layout(config, run_dir=root / "run", host_work=work, session_label="init").workdir
        remote_exists = _ssh([], port, key, f"test -e {workdir}")
        check("remote_workspace_cleaned", remote_exists.returncode != 0, workdir)
        check(
            "cost_incomplete_for_fake_agent",
            result.cost_incomplete is True and result.cost_usd is None,
            f"cost_incomplete={result.cost_incomplete}",
        )
    except Exception as exc:  # noqa: BLE001 - the report records the failure
        failure = f"{type(exc).__name__}: {exc}"
    finally:
        if provider is not None:
            provider.close()
        if keep_container:
            container_left = False
        else:
            _run(["docker", "rm", "-f", container], check=False, timeout=60)
            container_left = _run(["docker", "ps", "-a", "--format", "{{.Names}}"], check=False).stdout.splitlines()
            container_left = any(name == container for name in container_left)
        shutil.rmtree(root, ignore_errors=True)
        check("no_orphan_container", not container_left, container)

    failed = [item["name"] for item in checks if item["status"] != "passed"]
    report = {
        "schema_version": 1,
        "coverage": "isolated_agent_docker_fake",
        "simulated": True,
        "publication_allowed": False,
        "status": "passed" if not failure and not failed and {item["name"] for item in checks} == set(REQUIRED_CHECKS) else "failed",
        "failure": failure,
        "failed_checks": failed,
        "missing_required": sorted(set(REQUIRED_CHECKS) - {item["name"] for item in checks}),
        "required_checks": list(REQUIRED_CHECKS),
        "environment": {
            "control_host": "macOS/arm64",
            "transport": "SSH to a pre-provisioned Docker sshd host",
            "agent_host": "ubuntu:24.04 container",
            "agent_host_image_id": image_id,
            "agent_cli": "fake-acp (deterministic test fixture)",
            "agent_flow": "tests/support/agent_sim/flow (online=ALL test fixture)",
            "provider": "control-host loopback fake provider",
            "credential": "short-lived scoped token via a host-side docker-exec helper",
            "harness": "env",
            "container_security": "seccomp=unconfined so the test host can install its own fence",
        },
        "results": checks,
        "session": session_detail if "session_detail" in locals() else {},
        "not_run": [
            "real pi CLI and real provider",
            "production flow with online=NONE (this fixture relaxes online to ALL)",
            "real Ubuntu VM over SSH (blocked: no private key)",
            "hmz fence escape test on a real host",
            "GPU/NPU and performance measurement",
        ],
    }
    if report_path is not None:
        report_path = Path(report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tests.support.agent_sim.runner")
    parser.add_argument("--report", default=None, help="write the diagnostic report here")
    parser.add_argument("--keep-container", action="store_true")
    args = parser.parse_args(argv)
    report = run_simulation(
        report_path=Path(args.report) if args.report else None,
        keep_container=args.keep_container,
    )
    print(json.dumps({"status": report["status"], "failed_checks": report["failed_checks"], "failure": report["failure"]}))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
