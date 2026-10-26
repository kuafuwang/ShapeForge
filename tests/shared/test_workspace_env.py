"""Offline tests for the optional SSH workspace transport.

Nothing here starts ssh, a container, a model or a GPU.  The real transport is exercised
against a local stand-in for the remote command runner: the exact fixed commands the
transport builds are executed locally, so manifest staging, receipt idempotency, bounded
reads and cleanup are tested as bytes rather than mocked at the Python API.
"""

from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest

from taic.workspace_env import (
    EpicBinding,
    ManifestEntry,
    RemoteWorkspaceConfig,
    SavedRuntime,
    SshWorkspaceTransport,
    WorkspaceError,
    WorkspaceManifest,
    WorkspaceReceipt,
    build_manifest,
    check_relative_destination,
    epic_for_session,
    hmz_env_spec,
    prompt_surface_digest,
    read_runtime,
    run_token,
    session_layout,
    settings_path,
)


def _config(tmp_path: Path, *, harness: str = "local", root: str | None = None, **overrides) -> RemoteWorkspaceConfig:
    """Build one explicit config with a test root and a settings file beside it."""
    values = {
        "runtime": "gpu-box",
        "root": root or str(tmp_path / "srv" / "taic"),
        "harness": harness,
        "settings_file": tmp_path / "hmz-settings.yaml",
        "command_timeout_s": 20.0,
    }
    values.update(overrides)
    return RemoteWorkspaceConfig(**values)


def _settings(tmp_path: Path, *, harness: str = "local", port: int = 2222, host: str = "127.0.0.1", root: str | None = None) -> Path:
    """Write one saved hmz ssh runtime the way hmz's settings file keeps it."""
    path = tmp_path / "hmz-settings.yaml"
    saved_root = root or str(tmp_path / "srv" / "taic")
    path.write_text(
        "runtimes:\n"
        "  ssh:\n"
        "    gpu-box:\n"
        f"      host: {host}\n"
        "      user: eval\n"
        f"      port: {port}\n"
        "      identity_file: ''\n"
        "      proxy_jump: ''\n"
        "      options: {}\n"
        "      alias: ''\n"
        "      config: ''\n"
        f"      workdir: {saved_root}\n"
        "      fallback: []\n"
        "      made: typed\n"
        f"      harness: {harness}\n",
        encoding="utf-8",
    )
    return path


def _manifest(work: Path, paths, **kwargs) -> WorkspaceManifest:
    return build_manifest(work, paths, **kwargs)


def _receipt(work: Path, manifest: WorkspaceManifest, *, label: str = "init", runtime: str = "gpu-box") -> WorkspaceReceipt:
    return WorkspaceReceipt.build(
        run_id="run-1",
        session_label=label,
        session_id=label,
        runtime=runtime,
        workdir="/srv/taic/run-1/init",
        host_work=work,
        manifest=manifest,
        prompt_surface_sha256="a" * 64,
    )


# ------------------------------------------------------------------------------- manifest


def test_manifest_hashes_only_regular_files_and_is_order_independent(tmp_path):
    """A manifest is a sorted, hashed list of regular files and nothing else."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "TASK.md").write_text("task", encoding="utf-8")
    (work / "kernel.metal").write_text("kernel void k() {}", encoding="utf-8")
    first = _manifest(work, ["TASK.md", "kernel.metal"])
    second = _manifest(work, ["kernel.metal", "TASK.md", "TASK.md"])
    assert first.digest == second.digest
    assert first.paths == ("TASK.md", "kernel.metal")
    assert first.total_bytes == 4 + len("kernel void k() {}")


def test_manifest_refuses_absolute_parent_and_control_character_paths(tmp_path):
    """A manifest never names a path outside the canonical work directory."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "ok.md").write_text("x", encoding="utf-8")
    for bad in ("/etc/passwd", "../ok.md", "a/../../ok.md", "a//b", "a/./b", "a\\b", "a\x00b", " a.md"):
        with pytest.raises(WorkspaceError, match="path"):
            _manifest(work, [bad])


def test_manifest_refuses_a_symlink_even_when_it_points_inside(tmp_path):
    """A symlinked input would let control_host files ride into the workspace."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "real.md").write_text("x", encoding="utf-8")
    (work / "link.md").symlink_to(work / "real.md")
    with pytest.raises(WorkspaceError, match="symlink"):
        _manifest(work, ["link.md"])


def test_manifest_enforces_per_file_and_total_caps(tmp_path):
    """A bounded manifest is what keeps staging from becoming an arbitrary copy."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.md").write_text("x" * 10, encoding="utf-8")
    (work / "b.md").write_text("x" * 10, encoding="utf-8")
    with pytest.raises(WorkspaceError, match="over"):
        _manifest(work, ["a.md"], max_file_bytes=5)
    with pytest.raises(WorkspaceError, match="over"):
        _manifest(work, ["a.md", "b.md"], max_total_bytes=15)


def test_manifest_names_a_missing_input(tmp_path):
    """A missing input is a manifest error, not an empty staging run."""
    work = tmp_path / "work"
    work.mkdir()
    with pytest.raises(WorkspaceError, match="missing"):
        _manifest(work, ["absent.md"])


def test_collected_paths_cannot_escape_the_host_work_directory(tmp_path):
    """A malicious artifact name never writes outside the canonical directory."""
    work = tmp_path / "run" / "agent_work" / "init"
    work.mkdir(parents=True)
    assert check_relative_destination(work, "NOTES.md") == (work / "NOTES.md").resolve()
    for bad in ("../escape.md", "/tmp/escape.md", "a/../../escape.md"):
        with pytest.raises(WorkspaceError, match="path"):
            check_relative_destination(work, bad)


# -------------------------------------------------------------------------------- config


def test_the_default_config_is_explicit_and_local(tmp_path):
    """The module has no implicit remote default: a config names its runtime and root."""
    config = _config(tmp_path)
    assert config.harness == "local"
    with pytest.raises(WorkspaceError, match="runtime"):
        RemoteWorkspaceConfig(runtime="../gpu", root="/srv/taic")
    with pytest.raises(WorkspaceError, match="root"):
        RemoteWorkspaceConfig(runtime="gpu-box", root="relative/path")
    with pytest.raises(WorkspaceError, match="root"):
        RemoteWorkspaceConfig(runtime="gpu-box", root="/")
    with pytest.raises(WorkspaceError, match="harness"):
        RemoteWorkspaceConfig(runtime="gpu-box", root="/srv/taic", harness="remote")


def test_config_from_file_refuses_unknown_fields_and_resolves_the_settings_file(tmp_path):
    """Unknown keys are refused rather than silently ignored."""
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"runtime": "gpu-box", "root": "/srv/taic", "settings_file": "hmz.yaml"}))
    config = RemoteWorkspaceConfig.from_file(path)
    assert config.settings_file == tmp_path / "hmz.yaml"
    path.write_text(json.dumps({"runtime": "gpu-box", "root": "/srv/taic", "transport": "docker"}))
    with pytest.raises(WorkspaceError, match="unknown"):
        RemoteWorkspaceConfig.from_file(path)


def test_the_saved_runtime_harness_must_match_the_config(tmp_path):
    """A Phase 3 config cannot silently run the agent on the host, or the reverse."""
    from taic.agent_isolation import IsolatedAgentConfig

    _settings(tmp_path, harness="local")
    config = _config(tmp_path)
    runtime = read_runtime(config)
    assert runtime.harness == "local"
    assert runtime.destination == "eval@127.0.0.1"
    env = dict(harness="env", isolation=IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True))
    with pytest.raises(WorkspaceError, match="harness"):
        read_runtime(_config(tmp_path, **env))
    _settings(tmp_path, harness="env")
    assert read_runtime(_config(tmp_path, **env)).harness == "env"
    with pytest.raises(WorkspaceError, match="harness"):
        read_runtime(_config(tmp_path))


def test_a_missing_runtime_or_settings_file_fails_closed(tmp_path):
    """There is no fallback to a raw host or to local when the runtime is not saved."""
    with pytest.raises(WorkspaceError, match="no settings file"):
        read_runtime(_config(tmp_path))
    _settings(tmp_path)
    with pytest.raises(WorkspaceError, match="no saved ssh runtime"):
        read_runtime(RemoteWorkspaceConfig(runtime="other", root=str(tmp_path / "srv" / "taic"), settings_file=tmp_path / "hmz-settings.yaml"))


def test_settings_path_follows_humanize_home(monkeypatch, tmp_path):
    """The preflight reads the same settings home hmz itself uses."""
    monkeypatch.setenv("HUMANIZE_HOME", str(tmp_path / "hmz-home"))
    config = RemoteWorkspaceConfig(runtime="gpu-box", root="/srv/taic")
    assert settings_path(config) == tmp_path / "hmz-home" / "settings.yaml"


def test_the_env_spec_names_the_optional_flow_role(tmp_path):
    """One `-e` mapping, no provider and no credential."""
    config = _config(tmp_path)
    assert hmz_env_spec(config, "/srv/taic/run-1/init") == "remote_workspace=ssh@gpu-box/srv/taic/run-1/init"
    with pytest.raises(WorkspaceError, match="path"):
        hmz_env_spec(config, "/srv/taic/../etc")


def test_session_layout_is_deterministic_and_bounded(tmp_path):
    """The remote directory is derived from the run and the label, never from agent text."""
    config = _config(tmp_path)
    first = session_layout(config, run_dir=tmp_path / "run-a", host_work=tmp_path / "run-a" / "agent_work" / "init", session_label="init")
    second = session_layout(config, run_dir=tmp_path / "run-a", host_work=tmp_path / "run-a" / "agent_work" / "init", session_label="init")
    assert first == second
    assert first.workdir.startswith(config.root + "/")
    assert first.workdir.endswith("/init")
    with pytest.raises(WorkspaceError, match="label"):
        session_layout(config, run_dir=tmp_path / "run-a", host_work=tmp_path / "w", session_label="../../etc")
    assert run_token(tmp_path / "a run with spaces") != run_token(tmp_path / "another")


def test_prompt_surface_digest_covers_task_and_flow_skills(tmp_path):
    """The remote session records its own TASK and skill hashes."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "TASK.md").write_text("task", encoding="utf-8")
    flow = tmp_path / "flow"
    (flow / "skills" / "taic-tools").mkdir(parents=True)
    (flow / "skills" / "taic-tools" / "SKILL.md").write_text("skill", encoding="utf-8")
    first = prompt_surface_digest(work, flow)
    import re

    assert re.fullmatch(r"[0-9a-f]{64}", first[0])
    assert re.fullmatch(r"[0-9a-f]{64}", first[1])
    assert first[0] != first[1]
    (flow / "skills" / "taic-tools" / "SKILL.md").write_text("skill 2", encoding="utf-8")
    assert prompt_surface_digest(work, flow)[1] != first[1]


# ------------------------------------------------------------------------------ receipts


def test_a_receipt_binds_run_session_runtime_and_manifest(tmp_path):
    """The receipt id is a function of the session identity and its staged manifest."""
    work = tmp_path / "work"
    work.mkdir()
    (work / "TASK.md").write_text("task", encoding="utf-8")
    manifest = _manifest(work, ["TASK.md"])
    base = _receipt(work, manifest)
    assert base.document["receipt_id"] == base.receipt_id
    assert base.document["run_id"] == "run-1"
    assert base.document["workspace"]["host_work_sha256"] == _receipt(work, manifest).document["workspace"]["host_work_sha256"]
    other_label = _receipt(work, manifest, label="r2.p0")
    assert other_label.receipt_id != base.receipt_id
    other_runtime = _receipt(work, manifest, runtime="other-box")
    assert other_runtime.receipt_id != base.receipt_id


# ---------------------------------------------------------------------------- transport


class _FakeSsh:
    """Execute the last argv element as the remote command, locally.

    That is exactly the contract the transport relies on: one fixed command string, stdin
    and two output pipes.  It exercises the real command text and real file bytes without a
    host or a credential.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        path.write_text(
            "#!/usr/bin/env python3\n"
            "import subprocess, sys\n"
            "command = sys.argv[-1]\n"
            "raise SystemExit(subprocess.run(command, shell=True).returncode)\n",
            encoding="utf-8",
        )
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        self.commands: list[str] = []

    def transport(self, config: RemoteWorkspaceConfig, **kwargs) -> SshWorkspaceTransport:
        runtime = SavedRuntime(name=config.runtime, harness=config.harness, host="fake", user="eval")
        config = RemoteWorkspaceConfig(**{**config.__dict__, "ssh_binary": str(self.path)})
        return SshWorkspaceTransport(config, runtime=runtime, **kwargs)


def _work(tmp_path: Path) -> tuple[Path, RemoteWorkspaceConfig, Path]:
    """A staged work directory, a config rooted under tmp, and the remote mirror root."""
    work = tmp_path / "run" / "agent_work" / "init"
    work.mkdir(parents=True)
    (work / "TASK.md").write_text("task", encoding="utf-8")
    (work / "kernel.metal").write_text("kernel void k() {}", encoding="utf-8")
    config = _config(tmp_path)
    return work, config, Path(config.root)


def test_the_real_transport_stages_validates_and_collects(tmp_path):
    """Stage writes the receipt and bytes; collect reads back the declared files."""
    work, config, _root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md", "kernel.metal"])
    receipt = _receipt(work, manifest)
    workdir = "/" + str((Path(config.root) / "run-1" / "init")).lstrip("/")
    transport.stage(workdir, work, manifest, receipt)
    assert any("cat >" in command for command in transport.commands)
    assert transport.read_receipt(workdir).receipt_id == receipt.receipt_id
    collected = transport.collect(workdir, ["TASK.md", "absent.md", "kernel.metal"])
    assert collected == {"TASK.md": b"task", "kernel.metal": b"kernel void k() {}"}
    assert transport.remove(workdir, receipt) is True
    assert transport.exists(workdir) is False


def test_a_conflicting_receipt_is_never_overwritten_or_removed(tmp_path):
    """Another session's directory is left alone; the caller gets a typed conflict."""
    work, config, _root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    workdir = "/" + str((Path(config.root) / "run-1" / "init")).lstrip("/")
    transport.stage(workdir, work, manifest, _receipt(work, manifest, label="init"))
    with pytest.raises(WorkspaceError, match="another session"):
        transport.stage(workdir, work, manifest, _receipt(work, manifest, label="r1.p0"))
    with pytest.raises(WorkspaceError, match="another session"):
        transport.remove(workdir, _receipt(work, manifest, label="r1.p0"))
    assert transport.exists(workdir) is True
    assert transport.remove(workdir, _receipt(work, manifest, label="init")) is True


def test_an_unmarked_directory_is_not_removed(tmp_path):
    """Cleanup only removes a directory that carries our receipt."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    workdir = str(root / "run-1" / "init")
    Path(workdir).mkdir(parents=True)
    with pytest.raises(WorkspaceError, match="no receipt"):
        transport.remove(workdir, _receipt(work, _manifest(work, ["TASK.md"])))
    assert Path(workdir).is_dir()


def test_collect_refuses_a_file_over_the_cap(tmp_path):
    """A hostile candidate file cannot turn into unbounded host memory."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(RemoteWorkspaceConfig(**{**config.__dict__, "max_file_bytes": 16}))
    manifest = _manifest(work, ["TASK.md"], max_file_bytes=16)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, _receipt(work, manifest))
    Path(workdir, "big.metal").write_text("x" * 17, encoding="utf-8")
    with pytest.raises(WorkspaceError, match="over"):
        transport.collect(workdir, ["big.metal"])


def test_a_timeout_kills_the_local_process_group(tmp_path):
    """A hung ssh is a typed timeout and leaves no local writer."""
    work, config, _root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    with pytest.raises(WorkspaceError) as raised:
        transport._run("sleep 30", timeout_s=0.2)
    assert raised.value.kind == "timeout"


def test_a_broken_command_is_a_typed_protocol_error(tmp_path):
    """A non-zero fixed command is a protocol failure, not a candidate result."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    # The configured root is a regular file, so the fixed mkdir command cannot succeed.
    root.parent.mkdir(parents=True, exist_ok=True)
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(WorkspaceError) as raised:
        transport.make_workdir(str(root / "run-1" / "init"))
    assert raised.value.kind == "protocol"


# ---------------------------------------------------------------------------------- epic


def _write_epic(root: Path, *, workspace: Path, env: str, cost: float = 0.25, label: str = "e1", at: str | None = None) -> Path:
    """Write one hmz epic journal with the fields the binding reads."""
    folder = root / label / "20261026T000000.000Z-deadbeef"
    folder.mkdir(parents=True)
    stamp = at or time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    lines = [
        {"event": "began", "at": stamp, "workspace": str(workspace), "envs": [env], "used": [env], "flow": "kernel_once"},
        {"event": "opened", "at": stamp, "session": "pi-session-1", "provider": "local"},
        {"event": "usage", "at": stamp, "cost": cost, "output_tokens": 10},
        {"event": "ended", "at": stamp, "how": "done"},
    ]
    (folder / "epic.jsonl").write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    return folder / "epic.jsonl"


def test_an_epic_binds_only_to_the_same_workspace_and_remote_env(tmp_path):
    """Cost is borrowed by exact identity; a local epic never pays for a remote session."""
    work = tmp_path / "run" / "agent_work" / "init"
    work.mkdir(parents=True)
    epics = tmp_path / "epics"
    env = "remote_workspace=ssh@gpu-box/srv/taic/run-1/init"
    local = _write_epic(epics, workspace=work, env="", label="local", cost=9.0)
    assert local.is_file()
    binding = epic_for_session(host_work=work, env_spec=env, not_before=time.time() - 60, root=epics)
    assert binding is None
    _write_epic(epics, workspace=work, env=env, label="remote", cost=0.25)
    binding = epic_for_session(host_work=work, env_spec=env, not_before=time.time() - 60, root=epics)
    assert isinstance(binding, EpicBinding)
    assert binding.cost_usd == pytest.approx(0.25)
    assert binding.sessions == ("pi-session-1",)
    assert binding.envs == (env,)


def test_an_epic_that_began_before_this_attempt_is_not_borrowed(tmp_path):
    """A sibling session's earlier bill is never attributed to this attempt."""
    work = tmp_path / "run" / "agent_work" / "init"
    work.mkdir(parents=True)
    epics = tmp_path / "epics"
    env = "remote_workspace=ssh@gpu-box/srv/taic/run-1/init"
    _write_epic(epics, workspace=work, env=env, cost=1.0, at="2020-01-01T00:00:00.000Z")
    assert epic_for_session(host_work=work, env_spec=env, not_before=time.time() - 60, root=epics) is None


def test_no_epic_root_means_no_binding(tmp_path):
    """A host without an epic directory yields cost_incomplete, not a guessed bill."""
    work = tmp_path / "run" / "agent_work" / "init"
    work.mkdir(parents=True)
    assert epic_for_session(host_work=work, env_spec="remote_workspace=ssh@gpu-box/x", not_before=0.0, root=tmp_path / "absent") is None


# ------------------------------------------------------------------------------ surface


def test_the_remote_task_does_not_name_a_control_host_path(tmp_path):
    """The transport-only TASK prints no repo path and no self-check command."""
    from taic.kernel_agent import SessionBrief, render_contract
    from taic.workspace_prompt import render_remote_contract

    brief = SessionBrief(
        repo=tmp_path, task_name="relu", dtype="fp16", size="small", kernel_name="relu_kernel",
        triton_source="def relu(x): return x", description="relu", contract_text="kernel relu_kernel",
        abi_stub="kernel void relu_kernel()", rtol=1e-3, atol_rel=1e-3, label="init",
        work_dir=str(tmp_path / "agent_work" / "init"),
    )
    local = render_contract(brief)
    remote = render_remote_contract(brief)
    assert str(tmp_path) in local
    assert str(tmp_path) not in remote
    assert "--timed" not in remote and "taic.selfcheck_bridge" not in remote
    assert "# 本场没有自测通道" in remote
    # The local renderer is unchanged by the optional profile.
    assert render_contract(brief) == local
    assert local != remote


def test_staging_refuses_a_symlinked_session_directory(tmp_path):
    """An agent that replaced its session directory cannot redirect a receipt or input."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, receipt)
    victim = root / "victim"
    victim.mkdir()
    import shutil

    shutil.rmtree(Path(workdir) / ".taic")
    (Path(workdir) / ".taic").symlink_to(victim)
    with pytest.raises(WorkspaceError, match="create the remote workspace"):
        transport.stage(workdir, work, manifest, receipt)
    assert list(victim.iterdir()) == []


def test_collection_refuses_a_symlinked_artifact(tmp_path):
    """A symlinked submission cannot pull another host file into the canonical workspace."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, receipt)
    secret = root / "secret.txt"
    secret.write_text("not a kernel", encoding="utf-8")
    (Path(workdir) / "kernel.metal").symlink_to(secret)
    with pytest.raises(WorkspaceError, match="symlink"):
        transport.collect(workdir, ["kernel.metal"])


def test_the_saved_runtime_argv_carries_its_own_destination_options(tmp_path):
    """A typed runtime reaches ssh with its port, identity, jump and options; an alias keeps its config."""
    typed = SavedRuntime(
        name="gpu-box", harness="local", host="example", user="eval", port=2222,
        identity_file="/keys/id", proxy_jump="jump", options={"ServerAliveInterval": "30"},
    )
    argv = typed.argv("ssh", "echo hi", connect_timeout_s=5)
    assert argv[:4] == ["ssh", "-T", "-o", "BatchMode=yes"]
    assert ["-p", "2222"] == argv[argv.index("-p"):argv.index("-p") + 2]
    assert ["-i", "/keys/id"] == argv[argv.index("-i"):argv.index("-i") + 2]
    assert ["-J", "jump"] == argv[argv.index("-J"):argv.index("-J") + 2]
    assert argv[argv.index("-o", argv.index("-J")) + 1] == "ServerAliveInterval=30"
    assert argv[-2:] == ["eval@example", "echo hi"]
    alias = SavedRuntime(name="box", harness="local", host="ignored", alias="box-alias", config="/tmp/ssh_config")
    argv = alias.argv("ssh", "echo hi", connect_timeout_s=5)
    assert argv[-2:] == ["box-alias", "echo hi"]
    assert ["-F", "/tmp/ssh_config"] == argv[argv.index("-F"):argv.index("-F") + 2]
    assert "-p" not in argv


def test_collection_refuses_a_symlinked_session_directory_or_subdirectory(tmp_path):
    """A symlinked workdir or subdirectory cannot redirect a read to another host path."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, receipt)

    victim = root / "victim"
    victim.mkdir(parents=True)
    (victim / "kernel.metal").write_text("stolen", encoding="utf-8")
    (Path(workdir) / "sub").symlink_to(victim)
    with pytest.raises(WorkspaceError, match="symlinked component"):
        transport.collect(workdir, ["sub/kernel.metal"])

    import shutil

    shutil.rmtree(workdir)
    Path(workdir).symlink_to(victim)
    with pytest.raises(WorkspaceError):
        transport.collect(workdir, ["kernel.metal"])


def test_a_broken_receipt_cannot_block_cleanup_or_reaping(tmp_path):
    """A symlinked receipt is still a session directory of ours; removal and the sweep proceed."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    from taic.workspace_env import run_token

    token = run_token(tmp_path / "run")
    root_dir = Path(config.root) / token
    good = WorkspaceReceipt.build(
        run_id="run", session_label="init", session_id="init", runtime="gpu-box",
        workdir=str(root_dir / "init"), host_work=work, manifest=manifest, prompt_surface_sha256="a" * 64,
    )
    transport.stage(str(root_dir / "init"), work, manifest, good)
    (root_dir / "init" / ".taic" / "session.json").unlink()
    (root_dir / "init" / ".taic" / "session.json").symlink_to(root / "elsewhere.json")
    listed = transport.list_receipts(token)
    assert listed and listed[0][1] is None
    assert transport.remove(str(root_dir / "init"), good) is True
    assert not (root_dir / "init").exists()

    transport.stage(str(root_dir / "init2"), work, manifest, _receipt(work, manifest, label="init2"))
    (root_dir / "init2" / ".taic" / "session.json").unlink()
    (root_dir / "init2" / ".taic" / "session.json").symlink_to(root / "elsewhere.json")
    removed = transport.reap_orphans(token, "run", set())
    assert "init2" in removed


def test_keep_workspace_must_be_a_real_boolean(tmp_path):
    """JSON `"keep_workspace": "false"` must not silently turn into True and retain a host workspace."""
    path = tmp_path / "workspace.json"
    path.write_text(json.dumps({"runtime": "gpu-box", "root": "/srv/taic", "keep_workspace": "false"}))
    with pytest.raises(WorkspaceError, match="keep_workspace"):
        RemoteWorkspaceConfig.from_file(path)
    path.write_text(json.dumps({"runtime": "gpu-box", "root": "/srv/taic", "keep_workspace": False}))
    assert RemoteWorkspaceConfig.from_file(path).keep_workspace is False


def test_a_tampered_receipt_is_a_conflict_and_still_reapable(tmp_path):
    """A stored receipt_id is never trusted without recomputation."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, receipt)
    marker = Path(workdir) / ".taic" / "session.json"
    document = json.loads(marker.read_text())
    document["receipt_id"] = "0" * 64
    marker.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(WorkspaceError, match="does not match"):
        transport.read_receipt(workdir)
    assert transport.list_receipts("run-1") == [(workdir, None)]
    assert transport.remove(workdir, receipt) is True
    assert not Path(workdir).exists()


def test_a_tampered_receipt_is_reaped_by_its_marker(tmp_path):
    """A tampered receipt cannot make its directory invisible to the orphan sweep."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    from taic.workspace_env import run_token

    token = run_token(tmp_path / "run")
    root_dir = Path(config.root) / token
    receipt = WorkspaceReceipt.build(
        run_id="run", session_label="init", session_id="init", runtime="gpu-box",
        workdir=str(root_dir / "init"), host_work=work, manifest=manifest, prompt_surface_sha256="a" * 64,
    )
    transport.stage(str(root_dir / "init"), work, manifest, receipt)
    marker = root_dir / "init" / ".taic" / "session.json"
    document = json.loads(marker.read_text())
    document["run_id"] = "another-run"
    marker.write_text(json.dumps(document), encoding="utf-8")
    assert "init" in transport.reap_orphans(token, "run", set())


def test_collection_has_a_cumulative_byte_cap(tmp_path):
    """Per-file caps are not enough: the whole collected set is bounded too."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(RemoteWorkspaceConfig(**{**config.__dict__, "max_file_bytes": 100, "max_total_bytes": 100}))
    manifest = _manifest(work, ["TASK.md"], max_file_bytes=100)
    workdir = str(root / "run-1" / "init")
    transport.stage(workdir, work, manifest, _receipt(work, manifest))
    for index in range(3):
        (Path(workdir) / f"part{index}.md").write_text("x" * 40, encoding="utf-8")
    with pytest.raises(WorkspaceError, match="over 100 bytes"):
        transport.collect(workdir, ["part0.md", "part1.md", "part2.md"])


def test_a_runtime_saved_workdir_must_match_the_configured_root(tmp_path):
    """Staging one directory while the runtime runs the agent in another is refused."""
    _settings(tmp_path, harness="local", root="/srv/somewhere-else")
    with pytest.raises(WorkspaceError, match="saves workdir"):
        read_runtime(_config(tmp_path))


def test_a_symlinked_run_token_is_refused_before_creating_anything(tmp_path):
    """make_workdir walks existing components before mkdir, so a symlinked parent creates nothing."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    victim = root / "victim"
    victim.mkdir(parents=True)
    token = run_token(tmp_path / "run")
    (Path(config.root) / token).symlink_to(victim)
    with pytest.raises(WorkspaceError):
        transport.make_workdir(str(Path(config.root) / token / "init"))
    assert list(victim.iterdir()) == []


def test_removal_refuses_a_symlinked_run_token_and_keeps_the_outside_victim(tmp_path):
    """Cleanup re-walks every parent first: `rm -rf` must never follow a symlinked run token."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    token = run_token(tmp_path / "run")
    victim = tmp_path / "outside" / "victim"
    (victim / "init").mkdir(parents=True)
    (victim / "init" / "keep.txt").write_text("precious", encoding="utf-8")
    Path(config.root).mkdir(parents=True, exist_ok=True)
    (Path(config.root) / token).symlink_to(victim)
    workdir = str(Path(config.root) / token / "init")
    with pytest.raises(WorkspaceError) as raised:
        transport.remove(workdir, receipt)
    assert raised.value.kind == "path"
    assert (victim / "init" / "keep.txt").read_text(encoding="utf-8") == "precious"
    assert (Path(config.root) / token).is_symlink()


def test_a_write_creates_nothing_through_a_symlinked_label(tmp_path):
    """A receipt write walks the parent before mkdir, so no directory appears outside the root."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    token = run_token(tmp_path / "run")
    victim = tmp_path / "outside" / "victim"
    victim.mkdir(parents=True)
    (Path(config.root) / token).mkdir(parents=True)
    (Path(config.root) / token / "init").symlink_to(victim)
    workdir = str(Path(config.root) / token / "init")
    with pytest.raises(WorkspaceError) as raised:
        transport.write_receipt(workdir, receipt)
    assert raised.value.kind == "path"
    assert list(victim.rglob("*")) == []


def test_removal_unlinks_a_symlinked_label_leaf_and_keeps_the_victim(tmp_path):
    """A leaf symlink is unlinked, never followed: the directory it names survives intact."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    token = run_token(tmp_path / "run")
    victim = tmp_path / "outside" / "victim"
    victim.mkdir(parents=True)
    (victim / "keep.txt").write_text("precious", encoding="utf-8")
    (Path(config.root) / token).mkdir(parents=True)
    (Path(config.root) / token / "init").symlink_to(victim)
    workdir = str(Path(config.root) / token / "init")
    assert transport.remove(workdir, receipt) is True
    assert not (Path(config.root) / token / "init").is_symlink()
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_a_trailing_slash_cannot_turn_a_leaf_symlink_into_a_recursive_delete(tmp_path):
    """`rm -rf dir/` follows a leaf symlink; the target is rebuilt from checked components."""
    work, config, root = _work(tmp_path)
    fake = _FakeSsh(tmp_path / "fake-ssh")
    transport = fake.transport(config)
    manifest = _manifest(work, ["TASK.md"])
    receipt = _receipt(work, manifest)
    token = run_token(tmp_path / "run")
    victim = tmp_path / "outside" / "victim"
    victim.mkdir(parents=True)
    (victim / "keep.txt").write_text("precious", encoding="utf-8")
    (Path(config.root) / token).mkdir(parents=True)
    (Path(config.root) / token / "init").symlink_to(victim)
    workdir = str(Path(config.root) / token / "init")
    assert transport.remove(workdir + "/", receipt) is True
    assert not (Path(config.root) / token / "init").is_symlink()
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "precious"
