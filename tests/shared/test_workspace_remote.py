"""Offline tests for the optional remote-workspace runner and its local default.

The transport here is an in-process stand-in shaped exactly like
:class:`taic.workspace_env.SshWorkspaceTransport`, and the invocation seam writes a fixed
session output into the stand-in's remote directory.  No ssh, container, model, provider or
GPU is used.  The real byte-level transport is tested separately in
``test_workspace_env.py``.
"""

from __future__ import annotations

import json
import shutil
import sys
import stat
import time
from dataclasses import replace
from pathlib import Path

import pytest

from taic.kernel_agent import (
    FLOW,
    HmzInvocation,
    Prices,
    SessionBrief,
    render_contract,
    run_kernel_agent,
    submission_name,
)
from taic.session_runtime import (
    HmzSessionPlacement,
    LocalHmzKernelSessionRunner,
    SelfCheckBinding,
    SessionExecution,
    SessionPaths,
    SshWorkspaceKernelSessionRunner,
)
from taic.workspace_env import (
    RemoteWorkspaceConfig,
    WorkspaceError,
    WorkspaceReceipt,
    session_layout,
)

PRICES = Prices(input=0.3, output=1.2, cache_read=0.006, cache_write=0.0)
KERNEL = "kernel void conv2d_kernel() { /* fixed by the fake agent */ }\n"
NOTES = "one version tried\n"
FEEDBACK = "# 对系统的建议\nkeep it short\n"


def _brief(tmp_path: Path, *, seedless: bool = False, label: str = "init") -> SessionBrief:
    """A seeded conv2d brief whose work directory is the canonical host path."""
    work = tmp_path / "run" / "agent_work" / label
    return SessionBrief(
        repo=tmp_path, task_name="conv2d", dtype="fp16", size="paper", kernel_name="conv2d_kernel",
        triton_source="def conv():\n    pass\n", description="a convolution", contract_text="grid 1,1,1",
        abi_stub="kernel void conv2d_kernel() {}", rtol=1e-3, atol_rel=1e-3, seedless=seedless,
        label=label, work_dir=str(work.resolve()),
        seed_source="kernel void conv2d_kernel() { /* seed */ }\n",
    )


def _config(tmp_path: Path, *, harness: str = "local", keep_workspace: bool = False) -> RemoteWorkspaceConfig:
    return RemoteWorkspaceConfig(
        runtime="gpu-box",
        root=str(tmp_path / "srv" / "taic"),
        harness=harness,
        settings_file=tmp_path / "hmz-settings.yaml",
        keep_workspace=keep_workspace,
    )


def _execution(work: Path, config: RemoteWorkspaceConfig, *, visible: str | None = None) -> SessionExecution:
    host = work.resolve()
    run_dir = host.parent.parent
    if visible is None:
        visible = session_layout(config, run_dir=run_dir, host_work=host, session_label=host.name).workdir
    return SessionExecution(
        placement=HmzSessionPlacement(harness_placement=config.harness, workspace_env="ssh", endpoint=config.runtime),
        paths=SessionPaths(host_work=host, agent_work=visible),
        prices=PRICES,
        turn_output_tokens=1000,
        turn_minutes=1,
        hmz_binary="/fake/hmz",
        selfcheck=SelfCheckBinding(mode="local_cli"),
    )


class FakeHost:
    """The remote directory as a local mirror, with the transport's exact shape."""

    def __init__(self, state: dict, *, fail_stage: bool = False, fail_collect: bool = False, fail_remove: bool = False) -> None:
        self.state = state
        self.fail_stage = fail_stage
        self.fail_collect = fail_collect
        self.fail_remove = fail_remove
        self.staged: list[str] = []
        self.collected: list[str] = []
        self.removed: list[str] = []
        self.stage_calls = 0
        self.receipt: WorkspaceReceipt | None = None

    def stage(self, workdir, host_work, manifest, receipt) -> None:
        self.stage_calls += 1
        if self.fail_stage:
            raise WorkspaceError("unavailable", "ssh could not reach the host")
        self.state["workdir"] = workdir
        self.receipt = receipt
        target = Path(workdir)
        target.mkdir(parents=True, exist_ok=True)
        marker = target / ".taic" / "session.json"
        if marker.is_file() and json.loads(marker.read_text())["receipt_id"] != receipt.receipt_id:
            raise WorkspaceError("conflict", "the remote workspace belongs to another session receipt")
        marker.parent.mkdir(exist_ok=True)
        marker.write_text(json.dumps(receipt.document))
        for entry in manifest.entries:
            (target / entry.path).write_bytes((Path(host_work) / entry.path).read_bytes())
            self.staged.append(entry.path)

    def collect(self, workdir, paths):
        if self.fail_collect:
            raise WorkspaceError("disconnected", "the host closed the connection")
        found = {}
        for relative in paths:
            path = Path(workdir) / relative
            if path.is_file():
                found[relative] = path.read_bytes()
                self.collected.append(relative)
        return found

    def remove(self, workdir, receipt) -> bool:
        if self.fail_remove:
            raise WorkspaceError("cleanup", "the remote workspace survived removal")
        path = Path(workdir)
        marker = path / ".taic" / "session.json"
        if not marker.is_file():
            if path.exists():
                raise WorkspaceError("conflict", "the remote workspace has no receipt")
            return False
        if receipt is not None and json.loads(marker.read_text())["receipt_id"] != receipt.receipt_id:
            raise WorkspaceError("conflict", "the remote workspace belongs to another session receipt")
        shutil.rmtree(path)
        self.removed.append(workdir)
        return True


def _fake_invoke(state: dict, *, status: str = "ok", tokens: bool = True, cost: float = 0.0012, epic_root: Path | None = None):
    """Build an ``invoke_hmz``-shaped seam that writes into the fake remote directory."""

    def invoke(brief, *, work, prepared, provider, model, effort, prices, turn_output_tokens, turn_minutes,
               session_cost_cap_usd=None, hmz=None, timeout_s=None, env_specs=(), flow=FLOW):
        del work, provider, model, effort, prices, turn_output_tokens, turn_minutes
        del session_cost_cap_usd, hmz, timeout_s, flow
        assert len(env_specs) == 1
        remote = Path(state["workdir"])
        # A timeout may leave a checkpointed submission; a crash or a normal answer without a
        # submission leaves none.
        if status in ("ok", "timeout"):
            (remote / prepared.submit).write_text(f"kernel void {brief.kernel_name}() {{ /* fixed */ }}\n", encoding="utf-8")
        (remote / "NOTES.md").write_text(NOTES if status != "crash" else "crashed without a kernel\n", encoding="utf-8")
        (remote / "FEEDBACK.md").write_text(FEEDBACK, encoding="utf-8")
        if status == "timeout":
            invocation_status, returncode = "timeout", None
            stream = ""
        elif status == "crash":
            invocation_status, returncode = "failed", 1
            stream = json.dumps({"kind": "failed", "text": "the provider turn failed"})
        else:
            invocation_status, returncode = "ok", 0
            stream = json.dumps({"kind": "result", "text": "done", "spent": "Usage(output=1000)"})
        state["env_spec"] = env_specs[0]
        if epic_root is not None:
            _write_epic(epic_root, host_work=Path(brief.work_dir), env_spec=env_specs[0], cost=cost if tokens else 0.0)
        return HmzInvocation(
            status=invocation_status,
            returncode=returncode,
            stdout=stream,
            stderr="" if status != "crash" else "hmz crashed",
            spent={"output": 1000} if tokens else {},
            saw_tokens=tokens,
            cost_usd=cost if tokens else None,
            elapsed_s=0.25,
            argv=(),
        )

    return invoke


def _write_epic(root: Path, *, host_work: Path, env_spec: str, cost: float, label: str = "epic-1") -> Path:
    """Write the hmz epic the real CLI would leave for this session."""
    folder = root / label / f"{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}-deadbeef"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
    events = [
        {"event": "began", "at": stamp, "workspace": str(host_work.resolve()), "envs": [env_spec], "used": [env_spec]},
        {"event": "opened", "at": stamp, "session": "pi-session-1"},
        {"event": "usage", "at": stamp, "cost": cost},
        {"event": "ended", "at": stamp, "how": "done"},
    ]
    (folder / "epic.jsonl").write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
    return folder


def _runner(tmp_path: Path, host: FakeHost, *, config=None, invoke=None, epic_root: Path | None = None) -> SshWorkspaceKernelSessionRunner:
    return SshWorkspaceKernelSessionRunner(
        config or _config(tmp_path),
        transport=host,
        invoke=invoke,
        epic_root=epic_root if epic_root is not None else tmp_path / "epics",
    )


def _fake_hmz(path: Path, *, submit: str) -> Path:
    """A fake hmz that writes the same outputs into the canonical local directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib\n"
        f"pathlib.Path({submit!r}).write_text({KERNEL!r})\n"
        f"pathlib.Path('NOTES.md').write_text({NOTES!r})\n"
        f"pathlib.Path('FEEDBACK.md').write_text({FEEDBACK!r})\n"
        "print(json.dumps({'kind': 'result', 'spent': 'Usage(output=1000)'}))\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


# --------------------------------------------------------------------------- end to end


def test_the_remote_runner_returns_the_same_candidate_and_canonical_layout(tmp_path):
    """Remote and local composition yield one candidate and the same host artifact layout."""
    brief = _brief(tmp_path / "remote")
    state: dict = {}
    host = FakeHost(state)
    runner = _runner(tmp_path / "remote", host, invoke=_fake_invoke(state, epic_root=tmp_path / "remote" / "epics"))
    result = runner.execute(
        brief, execution=_execution(Path(brief.work_dir), runner.config), provider="deepseek", model="deepseek-flash", effort=None,
    )
    assert result.source and result.source.startswith("kernel void conv2d_kernel")
    assert result.notes == NOTES
    assert result.feedback.startswith("# 对系统的建议")
    assert result.cost_incomplete is False
    assert result.cost_usd == pytest.approx(0.0012)
    assert result.identity["cost_bound"] is True
    assert result.identity["workspace"]["backend"] == "ssh"
    assert result.identity["workspace"]["runtime"] == "gpu-box"
    assert result.identity["env_spec"] == state["env_spec"]
    assert result.identity["epic"]["cost_usd"] == pytest.approx(0.0012)
    assert result.identity["cleaned"] is True
    assert host.removed == [state["workdir"]]
    assert not Path(state["workdir"]).exists()

    # The same brief through the local runner leaves the same top-level artifact set in the
    # same canonical directory shape, and byte-identical staged inputs.
    local_brief = _brief(tmp_path / "local")
    submit = submission_name("conv2d", "fp16", "paper", "init")
    fake = _fake_hmz(tmp_path / "local" / "hmz", submit=submit)
    local = run_kernel_agent(
        local_brief, work=Path(local_brief.work_dir), provider="deepseek", model="deepseek-flash",
        effort=None, prices=PRICES, turn_output_tokens=1000, turn_minutes=1, hmz=str(fake),
    )
    remote_files = sorted(path.name for path in Path(brief.work_dir).iterdir() if path.name != "scratch")
    local_files = sorted(path.name for path in Path(local_brief.work_dir).iterdir() if path.name != "scratch")
    assert remote_files == local_files
    assert submit in remote_files
    assert local.source and local.source.startswith("kernel void conv2d_kernel")
    # The remote TASK is its own prompt surface (no control_host paths, no local self-check),
    # so it is hashed separately; the seed, guidance and harness template stay byte-identical.
    assert (Path(brief.work_dir) / "TASK.md").read_bytes() != (Path(local_brief.work_dir) / "TASK.md").read_bytes()
    assert result.identity["prompt"]["task_sha256"] != _local_task_sha(local_brief)
    for name in ("kernel.metal", "HARNESS.md", "device.md", "experience.md"):
        if (Path(brief.work_dir) / name).is_file():
            assert (Path(brief.work_dir) / name).read_bytes() == (Path(local_brief.work_dir) / name).read_bytes()


def _local_task_sha(brief: SessionBrief) -> str:
    import hashlib

    return hashlib.sha256((Path(brief.work_dir) / "TASK.md").read_bytes()).hexdigest()


def test_the_outer_loop_evaluates_the_remote_submission_through_the_executor(tmp_path):
    """The returned kernel is evaluated by the same EvaluationExecutor the local path uses."""
    from tcenv.evaluation import EvaluationRequest, EvaluationResult, local_provenance, receipt_id_for
    from tests.support.orchestrator import code, llm_script, make_orch, run

    class Evaluator:
        def __init__(self) -> None:
            self.requests: list[EvaluationRequest] = []

        def formal(self, request: EvaluationRequest) -> EvaluationResult:
            self.requests.append(request)
            return EvaluationResult(
                report=_report(request),
                provenance=local_provenance(request),
                evidence={"candidate_sha256": request.candidate_sha256},
                receipt={
                    "receipt_id": receipt_id_for(request),
                    "request_id": request.request_id,
                    "request_digest": request.request_digest,
                    "measurement_kind": "measured",
                },
            )

        def selfcheck(self, request):  # pragma: no cover - the search only evaluates formally
            raise AssertionError("the search must not use selfcheck")

    evaluator = Evaluator()
    epoch = local_provenance(
        EvaluationRequest(task="relu", dtype="fp16", size="small", source="kernel void demo() {}", target_id="apple-m4-max")
    ).epoch
    config = _config(tmp_path / "outer")
    _write_settings(config, harness="local")
    orch = make_orch(
        tmp_path / "outer", llm_script(code("speed=1.2")), [], max_rounds=0, evaluator=evaluator,
        workspace_config=config, publishable_site_ids={"local"}, publishable_epochs={epoch},
    )
    orch.agent_runner = None
    state: dict = {}
    orch.session_runner = _runner(
        tmp_path / "outer", FakeHost(state),
        config=config, invoke=_fake_invoke(state, epic_root=tmp_path / "outer" / "epics"),
    )
    result = run(orch)
    assert result.best_speedup == 1.2
    assert evaluator.requests
    assert evaluator.requests[0].source.startswith("kernel void relu_kernel")
    execution = json.loads((orch.run_dir / "execution.json").read_text())
    assert execution["workspace"]["backend"] == "ssh"
    assert execution["workspace"]["resolved"] == {"harness": "local", "runtime": "gpu-box", "root": str(tmp_path / "outer" / "srv" / "taic")}
    events = [json.loads(line) for line in (orch.run_dir / "transcript.jsonl").read_text().splitlines()]
    agent_events = [event for event in events if event.get("event") == "agent"]
    assert agent_events and agent_events[0]["identity"]["workspace"]["runtime"] == "gpu-box"


def _report(request):
    from tcenv.evaluate import EvalReport

    return EvalReport(
        task=request.task, dtype=request.dtype, size=request.size, mode=request.mode,
        seed=1, stage="done", status="ok", target_id=request.target_id,
        speedup=1.2, speedups={"mlx": 1.2}, candidate_us=100.0 / 1.2, baseline_us={"mlx": 100.0},
        benchmark={"candidate": {"median_us": 100.0 / 1.2, "cv": 0.01}, "baselines": {"mlx": {"median_us": 100.0}}, "best_baseline": "mlx"},
    )


# ------------------------------------------------------------------------- typed failures


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_class"),
    [
        ("timeout", "timeout", "timeout"),
        ("crash", "failed", "process_failed"),
        ("no_submission", "ok", "no_submission_after_result"),
    ],
)
def test_remote_timeout_crash_and_no_submission_are_typed_and_clean_up(tmp_path, status, expected_status, expected_class):
    """Every remote failure maps to the existing typed outcome and removes the workspace."""
    brief = _brief(tmp_path)
    state: dict = {}
    host = FakeHost(state)
    runner = _runner(tmp_path, host, invoke=_fake_invoke(state, status=status, epic_root=tmp_path / "epics"))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.status == expected_status
    assert result.diagnostics["failure_class"] == expected_class
    assert result.source is None if status != "timeout" else result.source is not None
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_an_unavailable_host_is_a_typed_failure_and_does_not_claim_a_bill(tmp_path):
    """A workspace that never started is not cost_incomplete and never falls back to local."""
    brief = _brief(tmp_path)
    state: dict = {}
    host = FakeHost(state, fail_stage=True)
    runner = _runner(tmp_path, host, invoke=_fake_invoke(state))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.status == "failed"
    assert result.cost_usd is None
    assert result.cost_incomplete is False
    assert result.diagnostics["failure_class"] == "workspace_unavailable"
    assert result.diagnostics["workspace_error"] == "unavailable"
    assert not Path(brief.work_dir, submission_name("conv2d", "fp16", "paper", "init")).exists()


def test_an_unbound_epic_is_cost_incomplete(tmp_path):
    """A session whose bill cannot be tied to its epic stops the ledger, not the candidate."""
    brief = _brief(tmp_path)
    state: dict = {}
    runner = _runner(tmp_path, FakeHost(state), invoke=_fake_invoke(state))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.source is not None
    assert result.cost_usd is None
    assert result.cost_incomplete is True
    assert result.diagnostics["cost_binding"] == "unbound"
    assert result.identity["cost_bound"] is False


def test_a_failed_collection_is_recorded_and_cleans_up(tmp_path):
    """A disconnect after the session is a typed artifact failure, not a hidden kernel."""
    brief = _brief(tmp_path)
    state: dict = {}
    host = FakeHost(state, fail_collect=True)
    runner = _runner(tmp_path, host, invoke=_fake_invoke(state, epic_root=tmp_path / "epics"))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.source is None
    assert result.identity["collection_error"] == "disconnected"
    assert result.identity["cleaned"] is True
    assert not Path(state["workdir"]).exists()


def test_a_cleanup_failure_is_reported_not_hidden(tmp_path):
    """A workspace that survives removal is named in the result, not silently dropped."""
    brief = _brief(tmp_path)
    state: dict = {}
    host = FakeHost(state, fail_remove=True)
    runner = _runner(tmp_path, host, invoke=_fake_invoke(state, epic_root=tmp_path / "epics"))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.identity["cleaned"] is False
    assert result.identity["cleanup_error"] == "cleanup"


def test_keep_workspace_retains_the_remote_directory(tmp_path):
    """An explicit keep flag is the only way the directory survives."""
    brief = _brief(tmp_path)
    state: dict = {}
    host = FakeHost(state)
    runner = _runner(tmp_path, host, config=_config(tmp_path, keep_workspace=True), invoke=_fake_invoke(state, epic_root=tmp_path / "epics"))
    result = runner.execute(brief, execution=_execution(Path(brief.work_dir), runner.config), provider="p", model="m", effort=None)
    assert result.identity["retained"] is True
    assert Path(state["workdir"]).is_dir()


def test_a_bridge_binding_is_refused_for_a_remote_workspace(tmp_path):
    """The transport does not forward host self-check, so it refuses to pretend it does."""
    brief = replace(_brief(tmp_path), selfcheck_socket="/host/s.sock", selfcheck_token_file="/host/s.token")
    state: dict = {}
    host = FakeHost(state)
    runner = _runner(tmp_path, host, invoke=_fake_invoke(state))
    execution = replace(
        _execution(Path(brief.work_dir), runner.config),
        selfcheck=SelfCheckBinding(mode="host_bridge", endpoint_id="/host/s.sock", token_ref="/host/s.token"),
    )
    with pytest.raises(ValueError, match="no self-check channel"):
        runner.execute(brief, execution=execution, provider="p", model="m", effort=None)
    assert host.stage_calls == 0


def test_a_mismatched_placement_is_refused(tmp_path):
    """A runner configured for a local harness cannot be asked to run the agent on the host."""
    brief = _brief(tmp_path)
    state: dict = {}
    runner = _runner(tmp_path, FakeHost(state), invoke=_fake_invoke(state))
    execution = replace(
        _execution(Path(brief.work_dir), runner.config),
        placement=HmzSessionPlacement(harness_placement="env", workspace_env="ssh"),
    )
    with pytest.raises(ValueError, match="harness"):
        runner.execute(brief, execution=execution, provider="p", model="m", effort=None)


# ----------------------------------------------------------------------- local default


def test_the_local_runner_still_owns_the_default(tmp_path):
    """Nothing in the optional module is reached without an explicit config."""
    brief = _brief(tmp_path)
    work = Path(brief.work_dir)
    with pytest.raises(ValueError, match="local harness and workspace"):
        LocalHmzKernelSessionRunner(run_fn=lambda *a, **k: pytest.fail("must not run")).execute(
            brief,
            execution=SessionExecution(
                placement=HmzSessionPlacement(workspace_env="ssh"), paths=SessionPaths.local(work), prices=PRICES,
            ),
            provider="p", model="m", effort=None,
        )


def test_the_local_hmz_argv_is_unchanged_and_remote_adds_exactly_one_env(tmp_path):
    """A local run passes no `-e`; a remote run passes one mapping and no credential."""
    from taic.kernel_agent import hmz_argv, prepare_session

    brief = _brief(tmp_path)
    prepared = prepare_session(brief, Path(brief.work_dir))
    local = hmz_argv(brief, prepared, binary="hmz", provider="p", model="m", effort=None, turn_output_tokens=10, turn_minutes=1, session_cost_cap_usd=None)
    assert "-e" not in local
    remote = hmz_argv(
        brief, prepared, binary="hmz", provider="p", model="m", effort=None, turn_output_tokens=10,
        turn_minutes=1, session_cost_cap_usd=None, env_specs=("remote_workspace=ssh@gpu-box/srv/taic/r/i",),
    )
    assert remote.count("-e") == 1
    assert remote[remote.index("-e") + 1] == "remote_workspace=ssh@gpu-box/srv/taic/r/i"
    assert remote[:remote.index("-e")] == local[:local.index("-a")]
    assert "sk-" not in " ".join(remote)


def test_the_local_task_golden_is_byte_identical():
    """The shipped local TASK bytes are pinned; the optional renderer cannot drift them."""
    import hashlib

    brief = SessionBrief(
        repo=Path("/repo"), task_name="relu", dtype="fp16", size="small", kernel_name="relu_kernel",
        triton_source="def relu(x): return x", description="relu 描述", contract_text="kernel relu_kernel",
        abi_stub="kernel void relu_kernel()", rtol=1e-3, atol_rel=1e-3,
        seed_source="kernel void relu_kernel() {}\n", label="init", work_dir="/work/agent_work/init",
        target_id="apple-m4-max", scorecard="speedup 1.0", history="h", notes_summary="n", peer_tips="p",
        evidence="e", failure="f", bound_line="b", seed_report_json='{"a":1}',
    )
    assert hashlib.sha256(render_contract(brief).encode()).hexdigest() == "9d03a8db63f341031c7ce646d63ab59e8754401a0027299b2e307ed404427018"


def test_the_flow_declares_an_optional_remote_role_and_a_runtime_local_role():
    """The flow keeps `workspace` automatic and adds `remote_workspace` only as optional."""
    pytest.importorskip("hmz")
    from hmz.runtime.flowing.loading import load

    flow = load(str(FLOW), {})
    roles = {role.name: role for role in flow.declared()[1]}
    assert roles["workspace"].auto is True and roles["workspace"].required is True
    assert roles["remote_workspace"].auto is False and roles["remote_workspace"].required is False
    assert {capability.__name__ for capability in roles["remote_workspace"].capabilities} >= {"BashEnvMixin", "FilesEnvMixin"}


def _write_settings(config: RemoteWorkspaceConfig, *, harness: str) -> None:
    config.settings_file.parent.mkdir(parents=True, exist_ok=True)
    config.settings_file.write_text(
        "runtimes:\n"
        "  ssh:\n"
        "    gpu-box:\n"
        "      host: 127.0.0.1\n"
        "      user: eval\n"
        "      port: 2222\n"
        "      identity_file: ''\n"
        "      proxy_jump: ''\n"
        "      options: {}\n"
        "      alias: ''\n"
        "      config: ''\n"
        "      workdir: /srv/taic\n"
        "      fallback: []\n"
        "      made: typed\n"
        f"      harness: {harness}\n",
        encoding="utf-8",
    )


def test_the_local_default_records_no_workspace_and_no_identity(tmp_path):
    """A default run's execution manifest and agent events keep their old shape."""
    from tests.support.orchestrator import code, llm_script, make_orch, run

    local = make_orch(tmp_path / "local", llm_script(code("speed=1.2")), [], max_rounds=0)
    run(local)
    execution = json.loads((local.run_dir / "execution.json").read_text())
    assert "workspace" not in execution
    events = [json.loads(line) for line in (local.run_dir / "transcript.jsonl").read_text().splitlines()]
    assert all("identity" not in event for event in events if event.get("event") == "agent")


def test_a_remote_config_needs_a_saved_runtime_with_the_same_harness(tmp_path):
    """The orchestrator refuses to start when the saved harness contradicts the config."""
    from taic.orchestrator import RunStartError
    from tests.support.orchestrator import code, llm_script, make_orch

    config = _config(tmp_path, harness="local")
    _write_settings(config, harness="env")
    with pytest.raises(RunStartError, match="harness"):
        make_orch(tmp_path / "bad", llm_script(code("speed=1.2")), [], workspace_config=config)


def test_a_remote_config_refuses_a_host_selfcheck_bridge(tmp_path):
    """Self-check stays on the Phase 2 executor; the workspace transport does not forward it."""
    from taic.orchestrator import RunStartError
    from tests.support.orchestrator import code, llm_script, make_orch

    config = _config(tmp_path)
    _write_settings(config, harness="local")
    with pytest.raises(RunStartError, match="self-check"):
        make_orch(tmp_path / "bad", llm_script(code("speed=1.2")), [], workspace_config=config, selfcheck_bridge_factory=lambda *a: pytest.fail("must not start"))


def test_real_hmz_accepts_the_remote_role_and_fails_closed_before_any_session(tmp_path, monkeypatch):
    """Real hmz loads the flow, takes the `-e` role, and refuses an unreachable host.

    The saved runtime points at a closed loopback port, so this is a real ssh attempt that
    ends in a connection refusal.  What it proves is the wiring and the failure mode: hmz
    settles the environment before the agent starts, exits non-zero, and leaves no local
    session or submission behind.  It is not a transport success and it never contacts a
    provider.
    """
    import shutil
    import subprocess

    hmz = shutil.which("hmz")
    if not hmz:
        pytest.skip("hmz is not installed")
    home = tmp_path / "hmz-home"
    home.mkdir()
    (home / "settings.yaml").write_text(
        "clis:\n"
        "  fake-acp:\n"
        f"    command: [{json.dumps(sys.executable)}, {json.dumps(str(Path(__file__).resolve().parents[1] / 'support' / 'hmz_selfcheck_agent.py'))}]\n"
        "    hosts: ['192.0.2.1']\n"
        f"    state: [{json.dumps(str(tmp_path / 'agent-state'))}]\n"
        "runtimes:\n"
        "  ssh:\n"
        "    gpu-box:\n"
        "      host: 127.0.0.1\n"
        "      user: eval\n"
        "      port: 9\n"
        "      identity_file: ''\n"
        "      proxy_jump: ''\n"
        "      options: {}\n"
        "      alias: ''\n"
        "      config: ''\n"
        "      workdir: /srv/taic\n"
        "      fallback: []\n"
        "      made: typed\n"
        "      harness: local\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HUMANIZE_HOME", str(home))
    work = tmp_path / "agent_work" / "init"
    work.mkdir(parents=True)
    completed = subprocess.run(
        [
            hmz, "exec", "-f", str(FLOW),
            "-e", "remote_workspace=ssh@gpu-box/srv/taic/probe/init",
            "-a", "assistant=fake-acp/fake-model",
            "-p", "turn_minutes=1", "-p", "wrap_minutes=1", "-p", "budget.duration=3m",
            "--json", "--", "read TASK.md",
        ],
        cwd=work, capture_output=True, text=True, timeout=120, check=False,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode != 0
    assert "could not reach gpu-box" in output
    # hmz refused before the flow: no session output, no submission, no provider state.
    assert sorted(path.name for path in work.iterdir()) == []
    assert not (tmp_path / "agent-state").exists()
