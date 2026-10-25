"""Offline tests for the optional isolated-agent pieces (Phase 4).

No model, provider, GPU or remote host is used.  The scoped credential, its seam, the
loopback fake provider and the trusted-evaluation fence are exercised directly; the runner
is exercised with an in-process fake transport and a fixed invoker, and the real transport's
orphan sweep against the same fixed-command stand-in used by the Phase 3 tests.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path

import pytest

from taic.agent_isolation import (
    ExternalHelperSeam,
    IsolatedAgentConfig,
    IsolationError,
    LoopbackFakeProvider,
    LoopbackFakeSeam,
    NoCredentialSeam,
    ScopedCredentialAuthority,
    TrustedEvaluationFence,
    build_seam,
    default_trusted_paths,
)
from taic.kernel_agent import FLOW, HmzInvocation, Prices, SessionBrief
from taic.session_runtime import (
    HmzSessionPlacement,
    SelfCheckBinding,
    SessionExecution,
    SessionPaths,
    SshWorkspaceKernelSessionRunner,
)
from taic.workspace_env import (
    RemoteWorkspaceConfig,
    SavedRuntime,
    SshWorkspaceTransport,
    WorkspaceError,
    WorkspaceReceipt,
    build_manifest,
    session_layout,
)

PRICES = Prices(input=0.3, output=1.2, cache_read=0.006, cache_write=0.0)
KERNEL = "kernel void conv2d_kernel() { /* fixed by the fake agent */ }\n"


# -------------------------------------------------------------------- credential + seams


def test_a_scoped_credential_is_short_lived_bounded_and_never_at_rest():
    """The authority keeps only a digest; expiry, revocation and the use budget all refuse."""
    clock = {"now": 1_000.0}
    authority = ScopedCredentialAuthority(clock=lambda: clock["now"])
    credential = authority.mint(run_id="run-1", session_label="r1.p0", runtime="gpu-box", ttl_s=60, max_uses=2)
    assert credential.token not in repr(credential)
    ref = credential.as_ref()
    assert set(ref) == {"credential_id", "scope", "expires_in_s", "max_uses", "token_at_rest"}
    assert credential.token not in json.dumps(ref)
    assert authority.verify(credential.token) == credential.credential_id
    assert authority.consume(credential.credential_id) is True
    assert authority.consume(credential.credential_id) is True
    assert authority.consume(credential.credential_id) is False
    assert authority.verify(credential.token) is None  # out of budget
    second = authority.mint(run_id="run-1", session_label="r1.p0", runtime="gpu-box", ttl_s=60, max_uses=2)
    clock["now"] += 61
    assert authority.verify(second.token) is None
    third = authority.mint(run_id="run-1", session_label="r1.p0", runtime="gpu-box", ttl_s=60, max_uses=2)
    assert authority.revoke(third.credential_id) is True
    assert authority.verify(third.token) is None
    assert authority.live_ids() == ()
    assert authority.verify("not-a-token") is None


def _post(url: str, token: str | None, payload: dict | None = None) -> tuple[int, dict]:
    request = urllib.request.Request(
        url + "/chat/completions",
        data=json.dumps(payload or {"prompt": "hello there"}).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def test_the_loopback_provider_needs_a_live_token_and_counts_usage():
    """A deterministic provider on 127.0.0.1: no token is refused, and usage is attributed."""
    authority = ScopedCredentialAuthority()
    provider = LoopbackFakeProvider(authority).start()
    try:
        assert provider.base_url.startswith("http://127.0.0.1:")
        credential = authority.mint(run_id="run-1", session_label="init", runtime="gpu-box", ttl_s=60, max_uses=1)
        assert _post(provider.base_url, None)[0] == 401
        assert _post(provider.base_url, "wrong")[0] == 401
        status, body = _post(provider.base_url, credential.token)
        assert status == 200 and body["choices"][0]["message"]["content"] == "fake"
        assert provider.requests and provider.requests[0]["credential_id"] == credential.credential_id
        assert authority.usage[credential.credential_id]["requests"] == 1
        assert _post(provider.base_url, credential.token)[0] == 401  # max_uses reached
        another = authority.mint(run_id="run-1", session_label="init", runtime="gpu-box", ttl_s=60, max_uses=4)
        authority.revoke(another.credential_id)
        assert _post(provider.base_url, another.token)[0] == 401
    finally:
        provider.close()


def _helper(tmp_path: Path) -> Path:
    """A host helper that records the stdin document and exits as told."""
    log = tmp_path / "helper.jsonl"
    script = tmp_path / "helper.py"
    script.write_text(
        "import json, sys\n"
        f"open({str(log)!r}, 'a').write(json.dumps(json.load(sys.stdin)) + '\\n')\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    return script


def test_the_external_helper_gets_the_token_on_stdin_and_is_revoked(tmp_path):
    """The token never enters argv, and revoke reaches the helper and the authority."""
    script = _helper(tmp_path)
    authority = ScopedCredentialAuthority()
    seam = ExternalHelperSeam(authority, (sys.executable, str(script)))
    credential = authority.mint(run_id="run-1", session_label="init", runtime="gpu-box", ttl_s=60, max_uses=4)
    evidence = seam.provision(credential, env_spec="remote_workspace=ssh@gpu-box/srv/taic/r/i")
    assert evidence["mode"] == "external-helper" and evidence["token_in_env"] is True
    assert evidence["provisioned_to_agent"] is True
    assert credential.token not in " ".join(seam.helper)
    seam.revoke(credential)
    assert authority.verify(credential.token) is None
    rows = [json.loads(line) for line in (tmp_path / "helper.jsonl").read_text().splitlines()]
    assert [row["action"] for row in rows] == ["provision", "revoke"]
    assert rows[0]["token"] == credential.token
    assert rows[0]["scope"] == {"run_id": "run-1", "session_label": "init", "runtime": "gpu-box"}


def test_a_failing_or_hung_helper_is_a_typed_error(tmp_path):
    """Infrastructure failure is typed and never becomes a candidate failure."""
    failing = tmp_path / "failing.py"
    failing.write_text("import sys; sys.exit(7)\n", encoding="utf-8")
    authority = ScopedCredentialAuthority()
    seam = ExternalHelperSeam(authority, (sys.executable, str(failing)), timeout_s=5)
    credential = authority.mint(run_id="run-1", session_label="init", runtime="gpu-box", ttl_s=60, max_uses=1)
    with pytest.raises(IsolationError) as raised:
        seam.provision(credential, env_spec="x")
    assert raised.value.kind == "helper"
    slow = tmp_path / "slow.py"
    slow.write_text("import time; time.sleep(30)\n", encoding="utf-8")
    with pytest.raises(IsolationError) as raised:
        ExternalHelperSeam(authority, (sys.executable, str(slow)), timeout_s=0.2).provision(credential, env_spec="x")
    assert raised.value.kind == "timeout"


def test_seam_selection_is_explicit():
    """No config means no seam; simulated and real seams cannot be confused."""
    assert isinstance(build_seam(None), NoCredentialSeam)
    loopback = build_seam(IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True))
    assert isinstance(loopback, LoopbackFakeSeam)
    loopback.close()
    with pytest.raises(IsolationError, match="simulated"):
        IsolatedAgentConfig(credential_seam="loopback-fake")
    with pytest.raises(IsolationError, match="argv"):
        IsolatedAgentConfig(credential_seam="external-helper")
    with pytest.raises(IsolationError, match="simulated"):
        IsolatedAgentConfig(credential_seam="external-helper", helper=("helper",), simulated=True)
    with pytest.raises(IsolationError, match="absolute"):
        IsolatedAgentConfig(trusted_evaluator_paths=("relative/path",))


def test_an_isolated_config_needs_a_runtime_saved_for_the_same_harness(tmp_path):
    """The workspace config refuses an env harness without a seam, or a seam with local harness."""
    with pytest.raises(WorkspaceError, match="credential seam"):
        RemoteWorkspaceConfig(runtime="gpu-box", root=str(tmp_path / "srv" / "taic"), harness="env", settings_file=tmp_path / "s.yaml")
    with pytest.raises(WorkspaceError, match="harness=local"):
        RemoteWorkspaceConfig(
            runtime="gpu-box", root=str(tmp_path / "srv" / "taic"), harness="local",
            settings_file=tmp_path / "s.yaml",
            isolation=IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True),
        )
    ok = RemoteWorkspaceConfig(
        runtime="gpu-box", root=str(tmp_path / "srv" / "taic"), harness="env",
        settings_file=tmp_path / "s.yaml",
        isolation=IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True),
    )
    assert ok.harness == "env" and ok.isolation.requires_remote_agent


# -------------------------------------------------------------------------------- fence


def test_the_fence_refuses_an_overlapping_workspace(tmp_path):
    """The agent workspace may not contain, or sit inside, a trusted evaluation artifact."""
    run = tmp_path / "run"
    workspace = run / "agent_work"
    workspace.mkdir(parents=True)
    (run / "candidates").mkdir()
    fence = TrustedEvaluationFence(trusted_paths=(run / "candidates", run / "execution.json"), host_workspace=workspace)
    fence.check()
    with pytest.raises(IsolationError, match="overlaps"):
        TrustedEvaluationFence(trusted_paths=(workspace / "x",), host_workspace=workspace)
    with pytest.raises(IsolationError, match="overlaps"):
        TrustedEvaluationFence(trusted_paths=(run,), host_workspace=workspace)


def test_materializing_a_collected_path_cannot_escape_the_workspace(tmp_path):
    """Even a hostile artifact name cannot write outside the canonical agent_work directory."""
    from taic.session_runtime import SshWorkspaceKernelSessionRunner
    from taic.workspace_env import check_relative_destination

    for bad in ("../candidates/001/report.json", "/tmp/escape", "a/../../escape"):
        with pytest.raises(WorkspaceError):
            check_relative_destination(tmp_path / "run" / "agent_work" / "init", bad)
    host = _Host({"tmp_path": tmp_path})
    config = _config(tmp_path)
    runner = SshWorkspaceKernelSessionRunner(config, transport=host, epic_root=tmp_path / "epics")
    with pytest.raises(WorkspaceError):
        runner._materialize(tmp_path / "run" / "agent_work" / "init", {"../candidates/x": b"x"})


def test_the_fence_refuses_a_write_into_a_trusted_artifact(tmp_path):
    """A collected or staged path can never land on the evaluator's files."""
    run = tmp_path / "run"
    workspace = run / "agent_work"
    workspace.mkdir(parents=True)
    (run / "candidates").mkdir()
    fence = TrustedEvaluationFence(trusted_paths=(run / "candidates",), host_workspace=workspace)
    fence.refuse_write(workspace / "init" / "submit.metal")
    for target in (run / "candidates" / "001", run / "candidates", run):
        with pytest.raises(IsolationError, match="trusted"):
            fence.refuse_write(target)


def test_default_trusted_paths_are_disjoint_from_the_agent_workspace(tmp_path):
    """The shipped layout keeps evaluator artifacts out of the agent workspace."""
    run = tmp_path / "run"
    workspace = run / "agent_work"
    workspace.mkdir(parents=True)
    fence = TrustedEvaluationFence(trusted_paths=default_trusted_paths(run, repo_root=Path(__file__).resolve().parents[2]), host_workspace=workspace)
    fence.check()
    assert run / "candidates" in fence.trusted_paths


# ------------------------------------------------------------------- runner end to end


def _brief(tmp_path: Path, *, label: str = "init") -> SessionBrief:
    work = tmp_path / "run" / "agent_work" / label
    return SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3,
        label=label, work_dir=str(work.resolve()),
        seed_source="kernel void conv2d_kernel() { /* seed */ }\n",
    )


def _config(tmp_path: Path, *, isolation: IsolatedAgentConfig | None = None) -> RemoteWorkspaceConfig:
    return RemoteWorkspaceConfig(
        runtime="gpu-box",
        root=str(tmp_path / "srv" / "taic"),
        harness="env" if isolation is not None else "local",
        settings_file=tmp_path / "hmz-settings.yaml",
        isolation=isolation,
    )


def _execution(work: Path, config: RemoteWorkspaceConfig) -> SessionExecution:
    host = work.resolve()
    visible = session_layout(config, run_dir=host.parent.parent, host_work=host, session_label=host.name).workdir
    return SessionExecution(
        placement=HmzSessionPlacement(harness_placement=config.harness, workspace_env="ssh", endpoint=config.runtime),
        paths=SessionPaths(host_work=host, agent_work=visible),
        prices=PRICES,
        turn_output_tokens=1000,
        turn_minutes=1,
        hmz_binary="/fake/hmz",
        selfcheck=SelfCheckBinding(mode="local_cli"),
    )


class _Host:
    """A compact Phase 3-shaped fake transport."""

    def __init__(self, state: dict) -> None:
        self.state = state
        self.staged = False
        self.removed: list[str] = []

    def stage(self, workdir, host_work, manifest, receipt) -> None:
        self.state["workdir"] = workdir
        target = Path(workdir)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".taic").mkdir(exist_ok=True)
        (target / ".taic" / "session.json").write_text(json.dumps(receipt.document))
        for entry in manifest.entries:
            (target / entry.path).write_bytes((Path(host_work) / entry.path).read_bytes())
        self.staged = True

    def collect(self, workdir, paths):
        found = {}
        for relative in paths:
            path = Path(workdir) / relative
            if path.is_file():
                found[relative] = path.read_bytes()
        return found

    def remove(self, workdir, receipt) -> bool:
        if not Path(workdir).exists():
            return False
        shutil.rmtree(workdir)
        self.removed.append(workdir)
        return True


def _epic(tmp_path: Path, *, host_work: Path, env_spec: str, harness: str, cost: float = 0.002) -> None:
    folder = tmp_path / "epics" / "epic-1" / "20261026T000000.000Z-deadbeef"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    events = [
        {"event": "began", "at": stamp, "workspace": str(host_work.resolve()), "envs": [env_spec], "used": [env_spec]},
        {"event": "opened", "at": stamp, "session": "pi-remote-1", "harness": harness, "provider": "local"},
        {"event": "usage", "at": stamp, "cost": cost},
    ]
    (folder / "epic.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")


def _invoke(state: dict, *, status: str = "ok", record: dict | None = None):
    """A fake remote agent: writes the submission and (optionally) uses the provider token."""

    def invoke(brief, *, work, prepared, provider, model, effort, prices, turn_output_tokens, turn_minutes,
               session_cost_cap_usd=None, hmz=None, timeout_s=None, env_specs=(), flow=FLOW):
        del work, provider, model, effort, prices, turn_output_tokens, turn_minutes
        del session_cost_cap_usd, hmz, timeout_s, flow
        remote = Path(state["workdir"])
        if status in ("ok", "timeout"):
            (remote / prepared.submit).write_text(f"kernel void {brief.kernel_name}() {{ /* fixed */ }} // speed=1.2\n", encoding="utf-8")
        (remote / "NOTES.md").write_text("one version\n", encoding="utf-8")
        if record is not None and record.get("base_url"):
            status_code, body = _post(record["base_url"], record["token"])
            record["status"] = status_code
            record["body"] = body
        state["env_spec"] = env_specs[0]
        if state.get("write_epic"):
            _epic(
                state["tmp_path"], host_work=Path(brief.work_dir), env_spec=env_specs[0],
                harness=state.get("epic_harness", "env"), cost=0.002,
            )
        return HmzInvocation(
            status="timeout" if status == "timeout" else ("failed" if status == "crash" else "ok"),
            returncode=None if status == "timeout" else (1 if status == "crash" else 0),
            stdout=json.dumps({"kind": "result" if status != "crash" else "failed", "spent": "Usage(output=1000)"}),
            stderr="hmz crashed" if status == "crash" else "",
            error="hmz crashed" if status == "crash" else "",
            spent={"output": 1000},
            saw_tokens=True,
            cost_usd=0.002,
            elapsed_s=0.1,
            argv=(),
        )

    return invoke


def _runner(tmp_path: Path, host: _Host, state: dict, *, config, invoke) -> SshWorkspaceKernelSessionRunner:
    return SshWorkspaceKernelSessionRunner(
        config, transport=host, invoke=invoke, epic_root=tmp_path / "epics",
    )


def test_a_remote_agent_session_gets_a_scoped_token_and_cleans_up(tmp_path):
    """The agent environment uses a short-lived token, and nothing survives the session."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True, credential_max_uses=4)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path, "write_epic": True, "epic_harness": "env"}
    host = _Host(state)
    record: dict = {}
    authority = ScopedCredentialAuthority()

    def seam_factory(cfg, *, authority):
        outer = LoopbackFakeSeam(authority)

        class _Recording:
            mode = outer.mode

            def provision(self, credential, *, env_spec):
                record["token"] = credential.token
                evidence = outer.provision(credential, env_spec=env_spec)
                record["base_url"] = evidence["base_url"]
                return evidence

            def revoke(self, credential):
                outer.revoke(credential)

            def close(self):
                outer.close()

        return _Recording()

    runner = SshWorkspaceKernelSessionRunner(
        config, transport=host, invoke=_invoke(state, record=record), epic_root=tmp_path / "epics",
        authority=authority, seam_factory=seam_factory,
    )
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.source and result.source.startswith("kernel void conv2d_kernel")
    assert record["status"] == 200 and record["body"]["choices"][0]["message"]["content"] == "fake"
    assert result.identity["credential"]["mode"] == "loopback-fake"
    assert result.identity["credential"]["provisioned_to_agent"] is False
    assert result.identity["credential"]["agent_env"] == "in-process-test-double"
    assert result.identity["credential"]["credential"]["token_at_rest"] is False
    assert record["token"] not in json.dumps(result.identity)
    assert result.identity["credential"]["revoked"] is True
    assert result.identity["credential"]["seam_closed"] is True
    assert "live_after_cleanup" not in result.identity["credential"]
    assert result.identity["harness_recorded"] == "env"
    assert result.identity["cost_bound"] is True
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_a_harness_mismatch_stops_the_cost_ledger(tmp_path):
    """If hmz says the harness ran locally while this config says env, the bill is not trusted."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path, "write_epic": True, "epic_harness": "local"}
    runner = _runner(tmp_path, _Host(state), state, config=config, invoke=_invoke(state))
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.source is not None
    assert result.cost_usd is None and result.cost_incomplete is True
    assert result.diagnostics["workspace_placement"] == "harness_mismatch"
    assert result.identity["harness_recorded"] == "local"


@pytest.mark.parametrize(("status", "expected"), [("timeout", "timeout"), ("crash", "process_failed")])
def test_isolated_failures_are_typed_revoked_and_cleaned(tmp_path, status, expected):
    """Crash and timeout revoke the credential and remove the remote directory."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path}
    host = _Host(state)
    authority = ScopedCredentialAuthority()
    runner = SshWorkspaceKernelSessionRunner(
        config, transport=host, invoke=_invoke(state, status=status), epic_root=tmp_path / "epics",
        authority=authority,
    )
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.diagnostics["failure_class"] == expected
    assert result.identity["credential"]["revoked"] is True
    assert result.identity["credential"]["seam_closed"] is True
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_a_missing_credential_seam_never_starts_the_agent(tmp_path):
    """Without a provisioned credential the session is a typed failure, not an unbilled run."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path}
    host = _Host(state)

    def broken_factory(cfg, *, authority):
        raise IsolationError("credential", "the helper refused")

    runner = SshWorkspaceKernelSessionRunner(
        config, transport=host, invoke=_invoke(state), epic_root=tmp_path / "epics", seam_factory=broken_factory,
    )
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.status == "failed"
    assert result.cost_usd is None and result.cost_incomplete is False
    assert result.diagnostics["failure_class"] == "credential_unavailable"
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_the_orchestrator_refuses_env_without_isolation_and_records_isolation_with_it(tmp_path):
    """A remote agent requires the explicit seam; the run record says which one."""
    from tests.support.orchestrator import code, llm_script, make_orch, run

    settings = (
        "runtimes:\n  ssh:\n    gpu-box:\n      host: 127.0.0.1\n      user: eval\n"
        "      harness: env\n"
    )
    with pytest.raises(WorkspaceError, match="credential seam"):
        RemoteWorkspaceConfig(
            runtime="gpu-box", root=str(tmp_path / "plain" / "srv"),
            harness="env", settings_file=tmp_path / "plain" / "s.yaml",
        )

    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path / "env", isolation=isolation)
    config.settings_file.parent.mkdir(parents=True, exist_ok=True)
    config.settings_file.write_text(settings, encoding="utf-8")
    state: dict = {"tmp_path": tmp_path / "env", "write_epic": True, "epic_harness": "env"}
    orch = make_orch(tmp_path / "env", llm_script(code("speed=1.2")), [], max_rounds=0, workspace_config=config)
    orch.agent_runner = None
    orch.session_runner = _runner(tmp_path / "env", _Host(state), state, config=config, invoke=_invoke(state))
    result = run(orch)
    assert result.best_speedup == 1.2
    execution = json.loads((orch.run_dir / "execution.json").read_text())
    resolved = execution["workspace"]["resolved"]
    assert resolved["harness"] == "env"
    assert resolved["isolation"]["credential_seam"] == "loopback-fake"
    assert resolved["isolation"]["simulated"] is True


# ----------------------------------------------------------------------- orphan reaping


def test_the_real_transport_reaps_only_this_runs_inactive_sessions(tmp_path):
    """Orphan cleanup is bounded to the run token; another run's directory and active labels stay."""
    import stat

    script = tmp_path / "fake-ssh"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess, sys\n"
        "raise SystemExit(subprocess.run(sys.argv[-1], shell=True).returncode)\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    config = RemoteWorkspaceConfig(
        runtime="gpu-box", root=str(tmp_path / "srv" / "taic"),
        settings_file=tmp_path / "s.yaml", ssh_binary=str(script),
    )
    runtime = SavedRuntime(name="gpu-box", harness="local", host="fake", user="eval")
    transport = SshWorkspaceTransport(config, runtime=runtime)
    work = tmp_path / "work"
    work.mkdir()
    (work / "TASK.md").write_text("task", encoding="utf-8")
    manifest = build_manifest(work, ["TASK.md"])
    from taic.workspace_env import run_token

    token = run_token(tmp_path / "run")
    root = Path(config.root) / token
    mine = WorkspaceReceipt.build(
        run_id="run", session_label="old", session_id="old", runtime="gpu-box",
        workdir=str(root / "old"), host_work=work, manifest=manifest, prompt_surface_sha256="a" * 64,
    )
    active = WorkspaceReceipt.build(
        run_id="run", session_label="live", session_id="live", runtime="gpu-box",
        workdir=str(root / "live"), host_work=work, manifest=manifest, prompt_surface_sha256="a" * 64,
    )
    other = WorkspaceReceipt.build(
        run_id="other-run", session_label="old", session_id="old", runtime="gpu-box",
        workdir=str(root / "other"), host_work=work, manifest=manifest, prompt_surface_sha256="a" * 64,
    )
    for receipt, label in ((mine, "old"), (active, "live"), (other, "other")):
        transport.stage(str(root / label), work, manifest, receipt)
    removed = transport.reap_orphans(token, "run", {"live"})
    assert removed == ["old"]
    assert not (root / "old").exists()
    assert (root / "live").exists() and (root / "other").exists()
    assert transport.reap_orphans(token, "run", {"live", "old", "other"}) == []


def test_the_runner_reap_reports_removed_labels(tmp_path):
    """The runner names what it reaped, using the transport's receipt check."""
    config = _config(tmp_path)
    removed: list[str] = []

    class Reaper:
        def reap_orphans(self, run_token, run_id, active_labels):
            removed.append(f"{run_token}:{run_id}:{','.join(sorted(active_labels))}")
            return ["init"]

        def stage(self, *a, **k):
            raise AssertionError("not used")

        def collect(self, *a, **k):
            raise AssertionError("not used")

        def remove(self, *a, **k):
            raise AssertionError("not used")

    runner = SshWorkspaceKernelSessionRunner(config, transport=Reaper(), epic_root=tmp_path / "epics")
    labels = runner.reap_orphans(tmp_path / "run", active_labels={"r1.p0"})
    assert labels == ["init"]
    assert removed and removed[0].startswith("run-") and removed[0].endswith(":run:r1.p0")


def test_an_unexpected_seam_failure_still_revokes_and_cleans(tmp_path):
    """Any exception between mint and invoke must not leave a live token or a remote directory."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path}
    host = _Host(state)
    authority = ScopedCredentialAuthority()

    def exploding_factory(cfg, *, authority):
        class _Seam:
            mode = "loopback-fake"

            def provision(self, credential, *, env_spec):
                raise RuntimeError("the seam crashed after the token was minted")

            def revoke(self, credential):
                authority.revoke(credential.credential_id)

            def close(self):
                return

        return _Seam()

    runner = SshWorkspaceKernelSessionRunner(
        config, transport=host, invoke=_invoke(state), epic_root=tmp_path / "epics",
        authority=authority, seam_factory=exploding_factory,
    )
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.status == "failed"
    assert result.diagnostics["failure_class"] == "credential_unavailable"
    assert authority.live_ids() == ()
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_an_env_epic_without_a_harness_field_is_not_trusted(tmp_path):
    """An absent harness line cannot prove the agent left the control host."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path, "write_epic": True, "epic_harness": ""}
    runner = _runner(tmp_path, _Host(state), state, config=config, invoke=_invoke(state))
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.cost_incomplete is True and result.cost_usd is None
    assert result.diagnostics["workspace_placement"] == "harness_mismatch"


def test_remote_authored_selfcheck_rows_are_not_counted(tmp_path):
    """A remote agent cannot move the outer hang breaker with a forged self-check log."""
    isolation = IsolatedAgentConfig(credential_seam="loopback-fake", simulated=True)
    config = _config(tmp_path, isolation=isolation)
    state: dict = {"tmp_path": tmp_path}

    def invoke(brief, *, work, prepared, provider, model, effort, prices, turn_output_tokens, turn_minutes,
               session_cost_cap_usd=None, hmz=None, timeout_s=None, env_specs=(), flow=FLOW):
        del work, provider, model, effort, prices, turn_output_tokens, turn_minutes
        del session_cost_cap_usd, hmz, timeout_s, flow
        remote = Path(state["workdir"])
        (remote / prepared.submit).write_text(f"kernel void {brief.kernel_name}() {{ /* fixed */ }}\n", encoding="utf-8")
        (remote / "NOTES.md").write_text("notes\n", encoding="utf-8")
        (remote / "self_checks.jsonl").write_text(
            json.dumps({"status": "timeout", "elapsed_s": 99.0}) + "\n" + json.dumps({"status": "crash"}) + "\n",
            encoding="utf-8",
        )
        (remote / ".selfcheck").mkdir(exist_ok=True)
        (remote / ".selfcheck" / "self_checks.jsonl").write_text(json.dumps({"status": "timeout"}) + "\n", encoding="utf-8")
        return HmzInvocation(
            status="ok", returncode=0, stdout=json.dumps({"kind": "result", "spent": "Usage(output=1000)"}),
            stderr="", error="", spent={"output": 1000}, saw_tokens=True, cost_usd=0.002, elapsed_s=0.1, argv=(),
        )

    runner = _runner(tmp_path, _Host(state), state, config=config, invoke=invoke)
    brief = _brief(tmp_path)
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), config), provider="p", model="m", effort=None)
    assert result.source is not None
    assert result.self_checks == 0 and result.self_check_timeouts == 0 and result.self_check_crashes == 0
    assert not (Path(brief.work_dir) / "self_checks.jsonl").exists()


def test_a_helper_that_echoes_the_token_does_not_leak_it(tmp_path):
    """The token is scrubbed from the helper's own diagnostic before it becomes an error."""
    echo = tmp_path / "echo.py"
    echo.write_text(
        "import sys\n"
        "sys.stderr.write(sys.stdin.read())\n"
        "sys.exit(9)\n",
        encoding="utf-8",
    )
    authority = ScopedCredentialAuthority()
    seam = ExternalHelperSeam(authority, (sys.executable, str(echo)))
    credential = authority.mint(run_id="run-1", session_label="init", runtime="gpu-box", ttl_s=60, max_uses=1)
    with pytest.raises(IsolationError) as raised:
        seam.provision(credential, env_spec="x")
    assert credential.token not in str(raised.value)
    assert "[redacted]" in str(raised.value)
