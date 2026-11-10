"""Static/CPU-mock tests for lower-level hmz Profile fencing; no real fence."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import pytest

import tcenv.worker_isolation.hmz_fence as isolation
from tcenv.worker_isolation import hmz_profile_driver as driver
from tcenv.children import forget as forget_child, kill_group as kill_child_group, popen_group as real_popen_group
from tcenv.evaluate import supervise as real_supervise
from tcenv.worker_isolation.child import LIMIT_REPORT_SCHEMA_ID, REPORT_FD_ENV
from tcenv.worker_isolation import (
    ArtifactContractError,
    AttemptSpec,
    HmzCapabilityUnavailable,
    HmzFenceExecutor,
    HmzInstallation,
    ResourceLimits,
    collect_artifacts,
)


def _installation(root: Path) -> HmzInstallation:
    executable = root / "fake-hmz"
    executable.write_text("# fake: never executed\n")
    executable.chmod(0o755)
    interpreter = root / "uv-tool" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("# fake interpreter: never executed\n")
    interpreter.chmod(0o755)
    sandbox = root / "fake-sandbox-exec"
    sandbox.write_text("# fake: only an existence fixture\n")
    sandbox.chmod(0o755)
    return HmzInstallation(
        requested_executable=str(executable), executable=executable,
        interpreter=interpreter, resolved_interpreter=interpreter,
        tool_root=root / "uv-tool", version=isolation.HMZ_VERSION,
        python_version=isolation.HMZ_PYTHON_VERSION, site_packages=root,
        source_sha256=isolation.HMZ_SOURCE_SHA256, entrypoint_sha256="2" * 64,
        driver_sha256=isolation.DRIVER_SOURCE_SHA256, child_sha256=isolation.CHILD_SOURCE_SHA256,
        protocol_digest="1" * 64, sandbox_exec=sandbox,
    )


def _spec(root: Path, **changes) -> AttemptSpec:
    values = {
        "attempt_root": root / "attempt",
        "argv": (sys.executable, str(root / "attempt" / "input" / "frontend.py"), "--preserve-me"),
        "artifacts": ("result.json",),
        "input_files": {"frontend.py": b"# fixed frontend fixture\n"},
    }
    values.update(changes)
    return AttemptSpec(**values)


def _executor(installation: HmzInstallation) -> HmzFenceExecutor:
    return HmzFenceExecutor(installation_loader=lambda: installation)


def test_policy_is_offline_attempt_only_without_mds_or_whole_repo_read(tmp_path, monkeypatch):
    monkeypatch.setenv("SHAPEFORGE_HOST_API_KEY", "must-not-leak")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/host/agent.sock")
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    spec = _spec(tmp_path, toolchain_roots=(toolchain,), extra_env={"FRONTEND_MODE": "fixture"})
    prepared = _executor(_installation(tmp_path)).prepare(spec)
    policy = prepared.policy
    build = tmp_path / "attempt" / "build"
    assert policy["online"] is False and policy["hosts"] == []
    assert str(Path.home()) not in policy["read"]
    assert str(isolation.REPO_ROOT) not in policy["read"]
    assert str(toolchain) in policy["read"]
    assert str(tmp_path / "attempt" / "input") in policy["read"]
    assert str(isolation.CHILD_PATH.parent) in policy["read"]
    assert "/" not in policy["write"]
    for written in policy["write"]:
        assert written.startswith("/dev/") or Path(written).is_relative_to(build)
        assert "mds" not in Path(written).parts
    assert prepared.environment["HOME"] == str(build / "home")
    assert prepared.environment["TMPDIR"] == str(build / "tmp") + os.sep
    assert prepared.environment["FRONTEND_MODE"] == "fixture"
    assert "SHAPEFORGE_HOST_API_KEY" not in prepared.environment
    assert "SSH_AUTH_SOCK" not in prepared.environment
    for name in (
        "TMP", "TEMP", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME",
        "PYTHONPYCACHEPREFIX", "UV_CACHE_DIR", "PIP_CACHE_DIR", "GOCACHE", "npm_config_cache",
    ):
        assert Path(prepared.environment[name]).is_relative_to(build)


def test_driver_argv_and_single_in_fence_rlimit_shim_are_pinned(tmp_path):
    spec = _spec(tmp_path)
    installation = _installation(tmp_path)
    prepared = _executor(installation).prepare(spec)
    assert prepared.argv == (
        str(installation.interpreter), "-I", str(isolation.DRIVER_PATH.resolve()),
        "--request", str(tmp_path / "attempt" / "hmz-profile-request.json"),
    )
    fenced = prepared.fenced_argv
    assert fenced[:3] == (str(Path(sys.executable).resolve()), "-I", str(isolation.CHILD_PATH.resolve()))
    assert "--address-space-bytes" not in fenced and "--processes" not in fenced
    assert fenced[fenced.index("--cpu-seconds") + 1] == "60"
    assert fenced[-len(spec.argv):] == spec.argv
    assert fenced[-len(spec.argv) - 1] == "--"
    assert not any("internal" == word or "sandbox-exec" in word for word in prepared.argv)
    request = json.loads((tmp_path / "attempt" / "hmz-profile-request.json").read_text())
    assert request == dict(prepared.request)
    assert request["schema_id"] == isolation.REQUEST_SCHEMA_ID
    assert request["online"] is False and request["hosts"] == []
    assert request["installation"]["source_sha256"] == isolation.HMZ_SOURCE_SHA256
    assert len(prepared.spec_digest) == len(prepared.request_sha256) == 64
    with pytest.raises(ValueError, match="generated/driver"):
        _spec(tmp_path, extra_env={REPORT_FD_ENV: "3"})
    with pytest.raises(ValueError, match="generated/driver"):
        _spec(tmp_path, extra_env={isolation.DRIVER_REPORT_FD_ENV: "3"})


def test_controlled_clang_frontend_and_arguments_are_not_banned(tmp_path):
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    clang = toolchain / "clang"
    clang.write_text("# fake executable; no process is launched\n")
    clang.chmod(0o755)
    spec = _spec(tmp_path, argv=(str(clang), "-cc1", "--custom-dsl-frontend"), toolchain_roots=(toolchain,))
    prepared = _executor(_installation(tmp_path)).prepare(spec)
    assert prepared.fenced_argv[-3:] == (str(clang), "-cc1", "--custom-dsl-frontend")
    assert str(toolchain) in prepared.policy["read"] and str(toolchain) not in prepared.policy["write"]
    child_source = isolation.CHILD_PATH.read_text()
    assert "os.execvpe" in child_source and "subprocess" not in child_source


@pytest.mark.parametrize(
    "name",
    ["API_KEY", "SSH_AUTH_SOCK", "GOOGLE_APPLICATION_CREDENTIALS", "HTTP_PROXY", "HOME", "DYLD_INSERT_LIBRARIES"],
)
def test_secret_credential_and_generated_environment_inputs_are_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        _spec(tmp_path, extra_env={name: "host-value"})


def test_credential_shaped_value_is_rejected_even_under_an_innocent_name(tmp_path):
    with pytest.raises(ValueError, match="real credentials"):
        _spec(tmp_path, extra_env={"FRONTEND_LABEL": "sk-abcdefghijklmnop1234"})


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a\\b", "a//b", "./a", "a/./b", "."])
def test_artifact_declaration_path_escapes_are_rejected(tmp_path, path):
    with pytest.raises(ValueError):
        _spec(tmp_path, artifacts=(path,))


def test_attempt_cannot_be_writable_inside_a_read_only_root(tmp_path):
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    with pytest.raises(ValueError, match="cannot be inside"):
        _executor(_installation(tmp_path)).prepare(_spec(tmp_path, attempt_root=read_only / "attempt", read_only_roots=(read_only,)))


def test_positive_artifact_manifest_has_exact_set_size_and_sha(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "a.bin").write_bytes(b"\x00\x01abc")
    (root / "nested" / "b.txt").write_text("payload\n")
    documents = collect_artifacts(root, ("a.bin", "nested/b.txt"), per_file_bytes=100, total_bytes=100)
    assert documents == [
        {"path": "a.bin", "bytes": 5, "sha256": hashlib.sha256(b"\x00\x01abc").hexdigest()},
        {"path": "nested/b.txt", "bytes": 8, "sha256": hashlib.sha256(b"payload\n").hexdigest()},
    ]


@pytest.mark.parametrize("kind", ["undeclared", "missing", "symlink", "hardlink", "file_size", "total_size"])
def test_artifact_escape_set_and_size_negative_cases(tmp_path, kind):
    root = tmp_path / "artifacts"
    root.mkdir()
    declared = ("wanted.bin",)
    (root / "wanted.bin").write_bytes(b"wanted")
    per_file, total = 100, 100
    if kind == "undeclared":
        (root / "extra.bin").write_bytes(b"extra")
    elif kind == "missing":
        (root / "wanted.bin").unlink()
    elif kind == "symlink":
        (root / "wanted.bin").unlink()
        (root / "wanted.bin").symlink_to(tmp_path / "outside.bin")
    elif kind == "hardlink":
        outside = tmp_path / "outside.bin"
        outside.write_bytes(b"wanted")
        (root / "wanted.bin").unlink()
        os.link(outside, root / "wanted.bin")
    elif kind == "file_size":
        per_file = 2
    else:
        total = 2
    with pytest.raises(ArtifactContractError):
        collect_artifacts(root, declared, per_file_bytes=per_file, total_bytes=total)


def _limit_report(status="ready"):
    return {
        "schema_id": LIMIT_REPORT_SCHEMA_ID,
        "platform": sys.platform,
        "status": status,
        "applied": [
            {"name": "cpu_seconds", "actual": [60, 60], "scope": "per_process_and_descendants"},
            {"name": "file_bytes", "actual": [67108864, 67108864], "scope": "per_file"},
            {"name": "open_files", "actual": [128, 128], "scope": "per_process_descriptor_table"},
            {"name": "core_bytes", "actual": [0, 0], "scope": "per_process_and_descendants"},
        ],
        "unsupported": [],
        "not_configured": [{"name": "address_space_bytes"}, {"name": "processes"}],
        "required_failures": [{"name": "open_files", "error": "fixture"}] if status == "failed" else [],
        "frontend_exec_attempted": status == "ready",
    }


def _driver_report(prepared, returncode, *, port=43123, status="ready"):
    report = {
        "schema_id": isolation.DRIVER_REPORT_SCHEMA_ID,
        "status": status,
        "request_sha256": prepared.request_sha256,
        "protocol_id": isolation.HMZ_PROTOCOL_ID,
        "protocol_digest": prepared.installation.protocol_digest,
        "online": False,
        "hosts": [],
        "profile_command_spawned": status == "ready",
        "proxy_stopped": True,
        "attack_containment_proven": False,
        "proxy": {"bind": "127.0.0.1", "port": port, "allowed_hosts": []},
        "profile": {
            "read": prepared.policy["read"], "write": prepared.policy["write"], "port": port,
            "rules_sha256": "4" * 64,
        },
        "internal_fence_or_wrap_used": False,
        "mds_write_root_added": False,
        "lower_level_route": "hmz.coganchor.fence.proxy.Proxy + hmz.coganchor.darwin.seatbelt.Profile",
        "profile_command_sha256": "3" * 64,
        "fenced_returncode": returncode,
    }
    if status != "ready":
        report["error_code"] = "backend_capability_unavailable"
    return report


def _mock_run(
    tmp_path, monkeypatch, *, returncode=0, reason="", mutate=None,
    limit_status="ready", write_limit=True, write_driver=True, driver_status="ready",
):
    calls = {"kill": [], "forget": [], "popen": None}

    class Process:
        pid = 424242
        returncode = None

    process = Process()

    def popen(argv, **kwargs):
        calls["popen"] = (argv, kwargs)
        request_path = Path(argv[argv.index("--request") + 1])
        request = json.loads(request_path.read_text())
        # Recreate only the immutable view needed by the report; no second launch/path.
        policy = {"read": request["profile"]["read"], "write": request["profile"]["write"]}
        class View:
            request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
            installation = _installation(tmp_path)
        View.policy = policy
        if write_driver:
            document = _driver_report(View, returncode, status=driver_status)
            os.write(kwargs["pass_fds"][0], (json.dumps(document) + "\n").encode())
        if write_limit:
            os.write(kwargs["pass_fds"][1], (json.dumps(_limit_report(limit_status)) + "\n").encode())
        if mutate is not None:
            mutate(tmp_path / "attempt")
        return process

    def supervise(proc, heartbeat, **kwargs):
        assert proc is process
        proc.returncode = returncode
        return reason

    monkeypatch.setattr(isolation, "popen_group", popen)
    monkeypatch.setattr(isolation, "supervise", supervise)
    monkeypatch.setattr(isolation, "kill_group", lambda proc: calls["kill"].append(proc.pid))
    monkeypatch.setattr(isolation, "forget", lambda proc: calls["forget"].append(proc.pid))
    return calls


def test_mocked_success_writes_worker_owned_artifact_and_profile_evidence(tmp_path, monkeypatch):
    def write_artifact(attempt):
        (attempt / "build" / "artifacts" / "result.json").write_text('{"ok":true}\n')

    calls = _mock_run(tmp_path, monkeypatch, mutate=write_artifact)
    result = _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert result.ok and result.status == "completed"
    assert result.document["artifacts"][0]["path"] == "result.json"
    assert result.document["artifacts"][0]["sha256"] == hashlib.sha256(b'{"ok":true}\n').hexdigest()
    assert result.document["process_boundary"]["production_route_count"] == 1
    assert result.document["process_boundary"]["internal_fence_or_wrap_used"] is False
    assert result.document["process_boundary"]["implicit_mds_write_roots"] == []
    assert result.document["process_boundary"]["attack_containment_proven_by_this_result"] is False
    assert result.document["driver_report"]["proxy"]["port"] == 43123
    assert result.document["driver_report"]["profile"]["port"] == 43123
    assert result.document["hmz"]["inspection"].startswith("read-only")
    assert result.document["resource_limits"]["evidence_status"] == "trusted_shim_observed"
    assert calls["kill"] == calls["forget"] == [424242]
    assert calls["popen"][1]["stdin"] is isolation.subprocess.DEVNULL
    assert calls["popen"][1]["cwd"] == tmp_path / "attempt" / "build" / "work"


def test_timeout_kills_and_forgets_owned_process_group_without_accepting_artifacts(tmp_path, monkeypatch):
    calls = _mock_run(tmp_path, monkeypatch, returncode=-9, reason="overall timeout")
    result = _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert result.status == "timeout" and not result.ok
    assert result.document["watchdog_reason"] == "overall timeout"
    assert result.document["artifacts"] == []
    assert calls["kill"] == calls["forget"] == [424242]


@pytest.mark.parametrize("returncode", [7, 125, 126])
def test_nonzero_frontend_is_not_an_artifact_success(tmp_path, monkeypatch, returncode):
    _mock_run(tmp_path, monkeypatch, returncode=returncode)
    result = _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert result.status == "process_failed" and result.returncode == returncode
    assert result.document["artifacts"] == []


def test_driver_setup_failure_is_typed_capability_failure_without_fallback(tmp_path, monkeypatch):
    calls = _mock_run(tmp_path, monkeypatch, returncode=126, driver_status="failed")
    with pytest.raises(HmzCapabilityUnavailable) as caught:
        _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert caught.value.code == "backend_capability_unavailable"
    assert caught.value.evidence["status"] == "backend_capability_unavailable"
    assert calls["kill"] == calls["forget"] == [424242]
    assert json.loads((tmp_path / "attempt" / "result.json").read_text())["returncode"] == 126


def test_shim_125_with_required_failure_report_is_typed_capability_failure(tmp_path, monkeypatch):
    _mock_run(tmp_path, monkeypatch, returncode=125, limit_status="failed")
    with pytest.raises(HmzCapabilityUnavailable, match="required resource limit") as caught:
        _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert caught.value.evidence["limit_report"]["required_failures"][0]["name"] == "open_files"


def test_missing_shim_limit_report_is_typed_capability_failure(tmp_path, monkeypatch):
    _mock_run(tmp_path, monkeypatch, returncode=0, write_limit=False)
    with pytest.raises(HmzCapabilityUnavailable, match="valid resource-limit report"):
        _executor(_installation(tmp_path)).run(_spec(tmp_path))


def test_missing_driver_report_is_typed_capability_failure(tmp_path, monkeypatch):
    _mock_run(tmp_path, monkeypatch, returncode=0, write_driver=False)
    with pytest.raises(HmzCapabilityUnavailable, match="trusted driver"):
        _executor(_installation(tmp_path)).run(_spec(tmp_path))


def test_mocked_input_tamper_is_rejected_before_artifact_success(tmp_path, monkeypatch):
    def tamper(attempt):
        (attempt / "input" / "frontend.py").write_text("changed")
        (attempt / "build" / "artifacts" / "result.json").write_text("{}")

    _mock_run(tmp_path, monkeypatch, mutate=tamper)
    with pytest.raises(ArtifactContractError, match="read-only input changed"):
        _executor(_installation(tmp_path)).run(_spec(tmp_path))
    assert json.loads((tmp_path / "attempt" / "result.json").read_text())["status"] == "artifact_invalid"


def _run_real_shim(
    tmp_path, frontend, *, address_space=None, processes=None, preexec_fn=None,
):
    """Execute only the trusted CPU shim, never hmz, the driver or an OS fence."""

    report_path = tmp_path / "real-limits.json"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    argv = [
        sys.executable, "-m", "tcenv.worker_isolation.child",
        "--cpu-seconds", "30", "--file-bytes", "67108864", "--open-files", "128",
    ]
    if address_space is not None:
        argv.extend(("--address-space-bytes", str(address_space)))
    if processes is not None:
        argv.extend(("--processes", str(processes)))
    argv.extend(("--", *frontend))
    environment = os.environ.copy()
    environment[REPORT_FD_ENV] = str(descriptor)
    options = {"preexec_fn": preexec_fn} if preexec_fn is not None else {}
    try:
        proc = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=Path(__file__).resolve().parents[2], env=environment,
            pass_fds=(descriptor,), **options,
        )
    finally:
        os.close(descriptor)
    try:
        stdout, stderr = proc.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr), json.loads(report_path.read_text())


def test_real_shim_echo_succeeds_and_unsupported_darwin_as_is_only_recorded(tmp_path):
    completed, report = _run_real_shim(tmp_path, ["/bin/echo", "ok"], address_space=2 * 1024**3)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"
    assert report["status"] == "ready"
    assert {row["name"] for row in report["applied"]} == {"cpu_seconds", "file_bytes", "open_files", "core_bytes"}
    if sys.platform == "darwin":
        unsupported = next(row for row in report["unsupported"] if row["name"] == "address_space_bytes")
        assert "EINVAL" in unsupported["error"]
        assert not any(row["name"] == "address_space_bytes" for row in report["applied"])


def test_real_shim_frontend_observes_required_limits_and_forks_a_grandchild(tmp_path):
    frontend = tmp_path / "observe_limits.py"
    frontend.write_text(
        "import json, os, resource, subprocess, sys\n"
        "child = subprocess.run([sys.executable, '-c', \"print('grandchild-ok')\"], capture_output=True, text=True, timeout=5)\n"
        "print(json.dumps({\n"
        " 'cpu': list(resource.getrlimit(resource.RLIMIT_CPU)),\n"
        " 'fsize': list(resource.getrlimit(resource.RLIMIT_FSIZE)),\n"
        " 'nofile': list(resource.getrlimit(resource.RLIMIT_NOFILE)),\n"
        " 'grandchild_rc': child.returncode,\n"
        " 'grandchild_stdout': child.stdout,\n"
        " 'report_fd_env': os.environ.get('SF_LIMIT_REPORT_FD'),\n"
        "}))\n"
    )
    completed, report = _run_real_shim(tmp_path, [sys.executable, str(frontend)])
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert observed["cpu"] == [30, 30]
    assert observed["fsize"] == [67108864, 67108864]
    assert observed["nofile"] == [128, 128]
    assert observed["grandchild_rc"] == 0
    assert observed["grandchild_stdout"] == "grandchild-ok\n"
    assert observed["report_fd_env"] is None
    assert {row["name"] for row in report["not_configured"]} == {"address_space_bytes", "processes"}


def test_real_shim_required_limit_failure_uses_125_and_never_executes_frontend(tmp_path):
    def lower_nofile_hard_limit():
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    completed, report = _run_real_shim(
        tmp_path, ["/bin/echo", "SHOULD_NOT_PRINT"], preexec_fn=lower_nofile_hard_limit,
    )
    assert completed.returncode == 125
    assert "SHOULD_NOT_PRINT" not in completed.stdout
    assert report["status"] == "failed" and report["frontend_exec_attempted"] is False
    assert report["required_failures"][0]["name"] == "open_files"


def test_real_popen_group_and_supervise_reap_a_timed_out_grandchild(tmp_path):
    sentinel = tmp_path / "grandchild-survived"
    frontend = tmp_path / "timeout_parent.py"
    grandchild = (
        "import pathlib, sys, time; time.sleep(0.6); "
        f"pathlib.Path({str(sentinel)!r}).write_text('survived')"
    )
    frontend.write_text(
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}])\n"
        "time.sleep(30)\n"
    )
    with (tmp_path / "stdout").open("w") as stdout, (tmp_path / "stderr").open("w") as stderr:
        proc = real_popen_group(
            [sys.executable, str(frontend)], stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
        try:
            reason = real_supervise(
                proc, tmp_path / "heartbeat", launch_deadline_s=0.2,
                timeout=0.2, compile_deadline_s=0.2, poll_s=0.02,
            )
        finally:
            kill_child_group(proc)
            forget_child(proc)
    assert reason == "timeout" and proc.returncode != 0
    time.sleep(0.8)
    assert not sentinel.exists()


def _uv_tool(tmp_path, monkeypatch, *, version="0.2.2"):
    tool = tmp_path / "uv-tool"
    executable = tool / "bin" / "hmz"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nfrom hmz.cli import main\n")
    executable.chmod(0o755)
    interpreter = tool / "bin" / "python"
    interpreter.write_text("# fake interpreter; inspection only\n")
    interpreter.chmod(0o755)
    (tool / "pyvenv.cfg").write_text(
        "implementation = CPython\nversion_info = 3.14.4\ninclude-system-site-packages = false\n"
    )
    site = tool / "lib" / "python3.14" / "site-packages"
    metadata = site / f"hmz-{version}.dist-info" / "METADATA"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(f"Metadata-Version: 2.4\nName: hmz\nVersion: {version}\n")
    hashes = {}
    for relative in isolation.HMZ_SOURCE_SHA256:
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture source: {relative}\n")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    sandbox = tmp_path / "sandbox-exec"
    sandbox.write_text("fixture")
    sandbox.chmod(0o755)
    monkeypatch.setattr(isolation, "SANDBOX_EXEC", sandbox)
    monkeypatch.setattr(isolation.sys, "platform", "darwin")
    monkeypatch.setattr(isolation, "HMZ_SOURCE_SHA256", hashes)
    monkeypatch.setattr(isolation, "HMZ_ENTRYPOINT_SHA256", hashlib.sha256(executable.read_bytes()).hexdigest())
    return executable, site, hashes, tool


def test_read_only_hmz_inspection_binds_interpreter_layout_sources_and_driver(tmp_path, monkeypatch):
    executable, site, hashes, tool = _uv_tool(tmp_path, monkeypatch)
    installation = isolation.inspect_hmz(executable)
    assert installation.version == "0.2.2" and installation.python_version == "3.14.4"
    assert installation.source_sha256 == hashes and installation.site_packages == site
    assert installation.interpreter == tool / "bin" / "python"
    assert installation.resolved_interpreter == (tool / "bin" / "python").resolve()
    assert installation.driver_sha256 == hashlib.sha256(isolation.DRIVER_PATH.read_bytes()).hexdigest()
    assert installation.child_sha256 == hashlib.sha256(isolation.CHILD_PATH.read_bytes()).hexdigest()
    assert len(installation.protocol_digest) == 64
    document = installation.document()
    assert document["internal_fence_or_wrap_used"] is False
    assert document["implicit_mds_write_roots"] == []
    assert document["runtime_attack_evidence"].startswith("not established")


def test_missing_hmz_is_typed_capability_unavailable(tmp_path, monkeypatch):
    _uv_tool(tmp_path, monkeypatch)
    monkeypatch.setattr(isolation.shutil, "which", lambda name: None)
    with pytest.raises(HmzCapabilityUnavailable, match="not found"):
        isolation.inspect_hmz()


def test_hmz_version_drift_fails_closed(tmp_path, monkeypatch):
    executable, _, _, _ = _uv_tool(tmp_path, monkeypatch, version="0.2.3")
    with pytest.raises(HmzCapabilityUnavailable, match="version drift"):
        isolation.inspect_hmz(executable)


def test_hmz_protocol_source_drift_fails_closed(tmp_path, monkeypatch):
    executable, site, _, _ = _uv_tool(tmp_path, monkeypatch)
    (site / "hmz" / "coganchor" / "darwin" / "seatbelt.py").write_text("drifted")
    with pytest.raises(HmzCapabilityUnavailable, match="source drift"):
        isolation.inspect_hmz(executable)


def test_driver_source_drift_fails_closed(tmp_path, monkeypatch):
    executable, _, _, _ = _uv_tool(tmp_path, monkeypatch)
    monkeypatch.setattr(isolation, "DRIVER_SOURCE_SHA256", "0" * 64)
    with pytest.raises(HmzCapabilityUnavailable, match="trusted driver source drift"):
        isolation.inspect_hmz(executable)


def test_shim_source_drift_fails_closed(tmp_path, monkeypatch):
    executable, _, _, _ = _uv_tool(tmp_path, monkeypatch)
    monkeypatch.setattr(isolation, "CHILD_SOURCE_SHA256", "0" * 64)
    with pytest.raises(HmzCapabilityUnavailable, match="trusted shim source drift"):
        isolation.inspect_hmz(executable)


def test_missing_hmz_interpreter_fails_closed(tmp_path, monkeypatch):
    executable, _, _, tool = _uv_tool(tmp_path, monkeypatch)
    (tool / "bin" / "python").unlink()
    with pytest.raises(HmzCapabilityUnavailable, match="interpreter layout"):
        isolation.inspect_hmz(executable)


def test_hmz_python_version_drift_fails_closed(tmp_path, monkeypatch):
    executable, _, _, tool = _uv_tool(tmp_path, monkeypatch)
    (tool / "pyvenv.cfg").write_text(
        "implementation = CPython\nversion_info = 3.13.0\ninclude-system-site-packages = false\n"
    )
    with pytest.raises(HmzCapabilityUnavailable, match="interpreter version/layout drift"):
        isolation.inspect_hmz(executable)


def test_missing_seatbelt_is_static_typed_refusal_not_a_probe(tmp_path, monkeypatch):
    executable, _, _, _ = _uv_tool(tmp_path, monkeypatch)
    monkeypatch.setattr(isolation, "SANDBOX_EXEC", tmp_path / "absent-sandbox-exec")
    with pytest.raises(HmzCapabilityUnavailable, match="absent or not executable"):
        isolation.inspect_hmz(executable)


def _profile_request(tmp_path):
    prepared = _executor(_installation(tmp_path)).prepare(_spec(tmp_path))
    return prepared.request


def test_driver_strict_request_rejects_online_hosts_and_candidate_control_fields(tmp_path):
    request = _profile_request(tmp_path)
    driver.validate_request(request)
    for key, value in (("online", True), ("hosts", ["example.com"]), ("candidate_control", "import this")):
        changed = dict(request)
        changed[key] = value
        with pytest.raises(driver.DriverError):
            driver.validate_request(changed)
    inside = dict(request)
    inside["argv"] = [*request["argv"]]
    inside["argv"][2] = str(Path(request["attempt_root"]) / "input" / "child.py")
    with pytest.raises(driver.DriverError, match="cannot be inside"):
        driver.validate_request(inside)


def test_driver_strict_request_rejects_external_mds_write_and_secret_environment(tmp_path):
    request = _profile_request(tmp_path)
    external = dict(request)
    external["profile"] = {"read": request["profile"]["read"], "write": [*request["profile"]["write"], "/private/var/folders/example/C/mds"]}
    with pytest.raises(driver.DriverError, match="attempt/build"):
        driver.validate_request(external)
    secret = dict(request)
    secret["environment"] = {**request["environment"], "API_KEY": "not-allowed"}
    with pytest.raises(driver.DriverError, match="secret/proxy/control"):
        driver.validate_request(secret)
    cache = dict(request)
    cache["environment"] = {**request["environment"], "XDG_CACHE_HOME": str(tmp_path / "outside-cache")}
    with pytest.raises(driver.DriverError, match="exactly its attempt/build location"):
        driver.validate_request(cache)


def test_driver_strict_request_rejects_source_and_driver_identity_drift(tmp_path, monkeypatch):
    request = _profile_request(tmp_path)
    changed = dict(request)
    tool_root = Path(request["installation"]["tool_root"])
    site = tool_root / "lib" / "python3.14" / "site-packages"
    changed["installation"] = {
        **request["installation"], "site_packages": str(site), "driver_sha256": "0" * 64,
    }
    driver.validate_request(changed)

    class FakeSys:
        platform = "darwin"
        version_info = (3, 14, 4)
        prefix = str(tool_root)
        executable = request["installation"]["resolved_interpreter"]

    def fake_sha256(path):
        path = Path(path)
        if path == Path(driver.__file__).resolve():
            return isolation.DRIVER_SOURCE_SHA256
        if path == Path(request["argv"][2]).resolve():
            return isolation.CHILD_SOURCE_SHA256
        for relative, digest in isolation.HMZ_SOURCE_SHA256.items():
            if path == site / relative:
                return digest
        raise AssertionError(f"unexpected hashed path: {path}")

    sandbox = tmp_path / "driver-sandbox-fixture"
    sandbox.write_text("fixture")
    sandbox.chmod(0o755)
    monkeypatch.setattr(driver, "sys", FakeSys)
    monkeypatch.setattr(driver, "SANDBOX_EXEC", sandbox)
    monkeypatch.setattr(driver, "_sha256", fake_sha256)
    monkeypatch.setattr(driver.importlib.metadata, "version", lambda name: "0.2.2")
    with pytest.raises(driver.DriverError, match="driver source identity drift"):
        driver._validate_installation(changed)

    source = dict(request)
    source["installation"] = {**request["installation"], "source_sha256": {}}
    with pytest.raises(driver.DriverError, match="source identity"):
        driver.validate_request(source)


def _fake_lower_level(events, *, port=43123):
    class Profile:
        def __init__(self, *, read, write, port):
            events.append(("profile", read, write, port))
            self.read, self.write, self.port = read, write, port

        def rules(self):
            events.append(("rules",))
            return "(version 1)(allow default)", {"fixture": "1"}

        def command(self, argv):
            events.append(("command", tuple(argv)))
            return ["/usr/bin/sandbox-exec", "-p", "fake-profile", "--", *argv]

    class Proxy:
        def __init__(self, allowed):
            events.append(("proxy-init", tuple(allowed)))
            self.port = port

        def start(self):
            events.append(("proxy-start",))

        def stop(self):
            events.append(("proxy-stop",))

    class Seatbelt:
        pass

    Seatbelt.Profile = Profile
    return Seatbelt, Proxy


def _report_fd(tmp_path):
    path = tmp_path / "fake-driver-report.json"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return path, descriptor


def test_driver_mock_uses_empty_host_proxy_actual_port_and_one_profile_command(tmp_path):
    request = _profile_request(tmp_path)
    events = []

    class Process:
        def wait(self):
            events.append(("wait",))
            return 0

        def poll(self):
            return 0

    def popen(command, **kwargs):
        events.append(("popen", tuple(command), kwargs))
        return Process()

    report_path, descriptor = _report_fd(tmp_path)
    try:
        returncode = driver._execute(
            request, report_fd=descriptor, limit_fd=17,
            lower_level=_fake_lower_level(events), popen_factory=popen,
        )
    finally:
        os.close(descriptor)
    report = json.loads(report_path.read_text())
    assert returncode == 0 and report["status"] == "ready"
    assert report["proxy"]["allowed_hosts"] == [] and report["proxy"]["port"] == 43123
    assert report["profile"]["port"] == 43123 and report["profile_command_spawned"] is True
    assert report["proxy_stopped"] is True and report["mds_write_root_added"] is False
    assert events[0] == ("proxy-init", ()) and events[1] == ("proxy-start",)
    assert events[2][0] == "profile" and events[2][3] == 43123
    popen = next(event for event in events if event[0] == "popen")
    assert popen[1][0] == "/usr/bin/sandbox-exec"
    assert popen[1][-len(request["argv"]):] == tuple(request["argv"])
    assert popen[2]["env"]["HTTPS_PROXY"] == "http://127.0.0.1:43123"
    assert popen[2]["env"]["NO_PROXY"] == "" and popen[2]["env"]["SF_LIMIT_REPORT_FD"] == "17"
    assert isolation.DRIVER_REPORT_FD_ENV not in popen[2]["env"]
    assert events[-1] == ("proxy-stop",)
    assert events.index(next(event for event in events if event[0] == "command")) < events.index(popen)


def test_driver_mock_port_none_fails_closed_before_profile_or_process(tmp_path):
    request = _profile_request(tmp_path)
    events = []
    report_path, descriptor = _report_fd(tmp_path)

    def forbidden_popen(*args, **kwargs):
        events.append(("forbidden-popen",))
        raise AssertionError("must not spawn")

    try:
        returncode = driver._execute(
            request, report_fd=descriptor, limit_fd=17,
            lower_level=_fake_lower_level(events, port=None), popen_factory=forbidden_popen,
        )
    finally:
        os.close(descriptor)
    report = json.loads(report_path.read_text())
    assert returncode == 126 and report["status"] == "failed"
    assert report["profile_command_spawned"] is False and report["proxy_stopped"] is True
    assert not any(event[0] in {"profile", "forbidden-popen"} for event in events)


def test_driver_mock_spawn_failure_stops_proxy_and_never_accepts_candidate(tmp_path):
    request = _profile_request(tmp_path)
    events = []
    report_path, descriptor = _report_fd(tmp_path)

    def fail_spawn(*args, **kwargs):
        events.append(("spawn-failed",))
        raise OSError("fixture spawn failure")

    try:
        returncode = driver._execute(
            request, report_fd=descriptor, limit_fd=17,
            lower_level=_fake_lower_level(events), popen_factory=fail_spawn,
        )
    finally:
        os.close(descriptor)
    report = json.loads(report_path.read_text())
    assert returncode == 126 and report["status"] == "failed"
    assert report["error_code"] == "backend_capability_unavailable"
    assert report["profile_command_spawned"] is False and report["proxy_stopped"] is True
    assert events[-1] == ("proxy-stop",)


def test_no_internal_fence_second_renderer_or_project_hmz_import():
    adapter_source = Path(isolation.__file__).read_text()
    driver_source = Path(driver.__file__).read_text()
    assert "subprocess.run(" not in adapter_source and "subprocess.run(" not in driver_source
    assert "sandbox_apply" not in adapter_source and "sandbox_apply" not in driver_source
    assert "def render_profile" not in adapter_source and "(deny default)" not in adapter_source
    tree = ast.parse(driver_source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names if alias.name.startswith("hmz"))
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("hmz"):
            imported.add(node.module)
    assert imported == {"hmz.coganchor.darwin", "hmz.coganchor.fence.proxy"}
    assert not (Path(isolation.__file__).parent / "macos_seatbelt.py").exists()
