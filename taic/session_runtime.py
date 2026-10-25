"""Execution placement and local runner interfaces for one kernel-writing session.

Phase 0a keeps the existing hmz behaviour.  This module only names the execution
parameters that used to be passed directly to :func:`taic.kernel_agent.run_kernel_agent`
and provides a local adapter.  It does not start a remote workspace, a container or a
different agent.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from .kernel_agent import AgentResult, Prices, SessionBrief, run_kernel_agent


@dataclass(frozen=True)
class HmzSessionPlacement:
    """Where the local session is placed.

    Phase 0a only supports the current local placement.  The optional fields are kept
    explicit so later phases can add an ``env`` or ``remote`` placement without
    changing the session content model.
    """

    agent_kind: str = "pi"
    harness_placement: str = "local"
    workspace_env: str = "local"
    endpoint: str = ""
    image_digest: str = ""


@dataclass(frozen=True)
class SessionPaths:
    """Host paths and the path the agent sees for one session."""

    host_work: Path
    agent_work: str = ""

    def __post_init__(self) -> None:
        """Default the agent-visible path to the host canonical path.

        The default is the current local layout.  A caller that supplies an explicit
        path is left alone; a remote runner can later materialize its own path while
        still writing the same canonical host directory.
        """
        if not self.agent_work:
            object.__setattr__(self, "agent_work", str(self.host_work))

    @classmethod
    def local(cls, work: Path | str) -> "SessionPaths":
        """Create the current local path view for ``work``."""
        return cls(host_work=Path(work))


@dataclass(frozen=True)
class SelfCheckBinding:
    """How the runner should expose the self-check command to the agent."""

    mode: str = "local_cli"  # local_cli | host_bridge | direct_client
    endpoint_id: str = ""
    token_ref: str = ""


@dataclass(frozen=True)
class SessionExecution:
    """Execution parameters that are not part of the session's content.

    These fields mirror the arguments that :func:`run_kernel_agent` already needs.
    Keeping them here avoids inventing a second :class:`SessionBrief` while making
    the local adapter and a future remote runner share one call shape.
    """

    placement: HmzSessionPlacement
    paths: SessionPaths
    prices: Prices
    turn_output_tokens: int = 200_000
    turn_minutes: int = 25
    session_cost_cap_usd: float | None = None
    timeout_s: float | None = None
    hmz_binary: str | None = None
    selfcheck: SelfCheckBinding = field(default_factory=SelfCheckBinding)


class HmzKernelSessionRunner(Protocol):
    """Run one hmz session and return the existing session result."""

    def execute(
        self,
        brief: SessionBrief,
        *,
        execution: SessionExecution,
        provider: str,
        model: str,
        effort: str | None,
    ) -> AgentResult:
        """Run one session with ``brief`` and ``execution``."""
        ...


class LocalHmzKernelSessionRunner:
    """Phase 0a adapter around the existing local :func:`run_kernel_agent`."""

    def __init__(self, run_fn: Callable[..., AgentResult] | None = None) -> None:
        """Bind the runner function.

        ``run_fn`` is injectable for tests.  In production it is the existing
        ``run_kernel_agent`` function, so this adapter does not duplicate process,
        timeout, billing or artifact behaviour.
        """
        self._run_fn = run_fn or run_kernel_agent

    def execute(
        self,
        brief: SessionBrief,
        *,
        execution: SessionExecution,
        provider: str,
        model: str,
        effort: str | None,
    ) -> AgentResult:
        """Run a local hmz session through the existing implementation.

        Only the current local placement is accepted here.  A non-local placement
        belongs to the later remote-workspace phase and must not silently fall back
        to the local runner.
        """
        placement = execution.placement
        if placement.agent_kind != "pi":
            raise ValueError(f"LocalHmzKernelSessionRunner only supports pi, got {placement.agent_kind!r}")
        if placement.harness_placement != "local" or placement.workspace_env != "local":
            raise ValueError("LocalHmzKernelSessionRunner only supports local harness and workspace")
        host = execution.paths.host_work.resolve()
        visible = Path(execution.paths.agent_work).resolve()
        if visible != host:
            raise ValueError("local session paths must map the agent workspace to host_work")
        if not brief.work_dir:
            # run_kernel_agent falls back to cwd for an empty work_dir. Fill it
            # from the canonical session path before TASK.md is rendered.
            brief.work_dir = str(host)
        elif Path(brief.work_dir).resolve() != host:
            raise ValueError("brief.work_dir must equal SessionPaths.host_work")
        binding = execution.selfcheck
        if binding.mode == "local_cli":
            if brief.selfcheck_socket or brief.selfcheck_token_file:
                raise ValueError("local_cli selfcheck binding conflicts with the brief bridge fields")
        elif binding.mode == "host_bridge":
            if not binding.endpoint_id or not binding.token_ref:
                raise ValueError("host_bridge selfcheck binding needs endpoint_id and token_ref")
            if brief.selfcheck_socket != binding.endpoint_id or brief.selfcheck_token_file != binding.token_ref:
                raise ValueError("host_bridge selfcheck binding must match the brief bridge fields")
        else:
            raise ValueError(f"LocalHmzKernelSessionRunner does not support selfcheck mode {binding.mode!r}")
        return self._run_fn(
            brief,
            work=host,
            provider=provider,
            model=model,
            effort=effort,
            prices=execution.prices,
            turn_output_tokens=execution.turn_output_tokens,
            turn_minutes=execution.turn_minutes,
            session_cost_cap_usd=execution.session_cost_cap_usd,
            hmz=execution.hmz_binary,
            timeout_s=execution.timeout_s,
        )


class SshWorkspaceKernelSessionRunner:
    """Optional Phase 3 runner: the same session, with its workspace on an SSH host.

    The canonical host directory is still ``agent_work/<label>``.  This runner prepares the
    same files there, stages exactly that bounded manifest onto the same filesystem path on
    a pre-provisioned host, starts the unchanged ``hmz exec`` line with one ``-e`` mapping,
    reads the declared artifacts back into the canonical directory, and only then lets the
    outer loop read a submission.  It adds no evaluator, no prompt rewrite for local runs
    and no fallback: an unavailable host is a typed failure.

    Isolation is not claimed here.  The hmz fence and the host's own permissions decide what
    the inner loop can reach; this class records the transport evidence and the session
    identity the outer loop needs to reconcile cost.
    """

    def __init__(
        self,
        config,
        *,
        transport=None,
        prepare: Callable[..., object] | None = None,
        invoke: Callable[..., object] | None = None,
        now: Callable[[], float] | None = None,
        flow: Path | None = None,
        epic_root: Path | None = None,
    ) -> None:
        """Bind the config and the injectable transport/invocation seams.

        @param config a validated :class:`taic.workspace_env.RemoteWorkspaceConfig`.
        @param transport an :class:`SshWorkspaceTransport`-shaped object; tests inject one.
        @param prepare the canonical host preparation function.
        @param invoke the ``hmz exec`` invocation function.
        @param now wall-clock source, injectable for epic binding tests.
        @param flow the flow directory whose prompt surface is hashed.
        @param epic_root where hmz keeps epics; None reads ``~/.hmz/epics``.
        """
        from . import kernel_agent as agent_module
        from .workspace_env import SshWorkspaceTransport

        self.config = config
        self._transport = transport if transport is not None else SshWorkspaceTransport(config)
        self._prepare = prepare or agent_module.prepare_session
        self._invoke = invoke or agent_module.invoke_hmz
        self._now = now or time.time
        self._flow = Path(flow) if flow is not None else agent_module.FLOW
        self._epic_root = Path(epic_root) if epic_root is not None else None
        self._agent = agent_module

    @property
    def placement(self):
        """The placement this runner implies, for the execution manifest."""
        return HmzSessionPlacement(
            harness_placement=self.config.harness,
            workspace_env="ssh",
            endpoint=self.config.runtime,
        )

    def paths(self, host_work: Path, session_label: str) -> "SessionPaths":
        """The host canonical and agent-visible paths for one session.

        Both are derived from the same layout the transport uses, so the caller cannot name
        a remote directory this runner would not stage into.
        """
        from .workspace_env import session_layout

        host = Path(host_work).resolve()
        run_dir = host.parent.parent if host.parent.name == "agent_work" else host.parent
        layout = session_layout(self.config, run_dir=run_dir, host_work=host, session_label=session_label)
        return SessionPaths(host_work=host, agent_work=layout.workdir)

    def execute(
        self,
        brief: SessionBrief,
        *,
        execution: SessionExecution,
        provider: str,
        model: str,
        effort: str | None,
    ) -> AgentResult:
        """Run one session in the remote workspace and return the canonical result."""
        from .workspace_env import (
            WorkspaceError,
            WorkspaceReceipt,
            build_manifest,
            check_relative_destination,
            epic_for_session,
            hmz_env_spec,
            session_layout,
        )

        placement = execution.placement
        expected = self.placement
        if placement.agent_kind != "pi":
            raise ValueError(f"SshWorkspaceKernelSessionRunner only supports pi, got {placement.agent_kind!r}")
        if placement.harness_placement != expected.harness_placement or placement.workspace_env != "ssh":
            raise ValueError(
                "the remote workspace config asked for harness="
                f"{expected.harness_placement!r} and workspace=ssh; got "
                f"harness={placement.harness_placement!r}, workspace={placement.workspace_env!r}"
            )
        host = execution.paths.host_work.resolve()
        run_dir = host.parent.parent if host.parent.name == "agent_work" else host.parent
        layout = session_layout(
            self.config,
            run_dir=run_dir,
            host_work=host,
            session_label=brief.label,
        )
        visible = Path(execution.paths.agent_work)
        if str(visible) != layout.workdir:
            raise ValueError("a remote session must name its remote workdir as agent_work")
        if not brief.work_dir:
            brief.work_dir = str(host)
        elif Path(brief.work_dir).resolve() != host:
            raise ValueError("brief.work_dir must equal SessionPaths.host_work")
        binding = execution.selfcheck
        if binding.mode != "local_cli" or brief.selfcheck_socket or brief.selfcheck_token_file:
            raise ValueError(
                "a remote workspace session has no self-check channel; self-check stays on the "
                "Phase 2 EvaluationExecutor and this transport does not forward it"
            )

        prepared = self._prepare(brief, host, remote_workspace=True, flow=self._flow)
        manifest = build_manifest(
            host,
            prepared.files,
            max_file_bytes=self.config.max_file_bytes,
            max_total_bytes=self.config.max_total_bytes,
        )
        receipt = WorkspaceReceipt.build(
            run_id=run_dir.name,
            session_label=brief.label,
            session_id=brief.label,
            runtime=self.config.runtime,
            workdir=layout.workdir,
            host_work=host,
            manifest=manifest,
            prompt_surface_sha256=prepared.prompt_surface_sha256,
        )
        env_spec = hmz_env_spec(self.config, layout.workdir)
        identity: dict = {
            "schema_version": 1,
            "receipt_id": receipt.receipt_id,
            "run_id": run_dir.name,
            "session_label": brief.label,
            "workspace": {
                "backend": "ssh",
                "harness": self.config.harness,
                "runtime": self.config.runtime,
                "workdir": layout.workdir,
                "host_work": str(host),
                "manifest_digest": manifest.digest,
                "manifest_files": len(manifest.entries),
                "manifest_bytes": manifest.total_bytes,
            },
            "prompt": {
                "task_sha256": prepared.task_sha256,
                "skill_sha256": prepared.skill_sha256,
                "prompt_surface_sha256": prepared.prompt_surface_sha256,
            },
            "env_spec": env_spec,
            "epic": None,
            "cost_bound": False,
            "collected": [],
            "missing": [],
        }

        started = self._now()
        try:
            self._transport.stage(layout.workdir, host, manifest, receipt)
        except WorkspaceError as exc:
            # Nothing has been billed yet: this is an explicit environment failure, not an
            # unknown bill, so it must not masquerade as cost_incomplete. A partially
            # staged directory still carries our receipt, so a best-effort removal is safe.
            if not self.config.keep_workspace:
                try:
                    self._transport.remove(layout.workdir, receipt)
                    identity["cleaned"] = True
                except WorkspaceError as cleanup_exc:
                    identity["cleaned"] = False
                    identity["cleanup_error"] = cleanup_exc.kind
            return AgentResult(
                source=None,
                cost_usd=None,
                cost_incomplete=False,
                status="failed",
                error=f"remote workspace unavailable: {exc.message}",
                diagnostics={"failure_class": "workspace_unavailable", "workspace_error": exc.kind},
                identity=identity,
            )

        invocation = None
        collection_error: WorkspaceError | None = None
        try:
            invocation = self._invoke(
                brief,
                work=host,
                prepared=prepared,
                provider=provider,
                model=model,
                effort=effort,
                prices=execution.prices,
                turn_output_tokens=execution.turn_output_tokens,
                turn_minutes=execution.turn_minutes,
                session_cost_cap_usd=execution.session_cost_cap_usd,
                hmz=execution.hmz_binary,
                timeout_s=execution.timeout_s,
                env_specs=(env_spec,),
                flow=self._flow,
            )
            try:
                found, candidates, missing = self._collect(layout, prepared, host)
                identity["collected"] = sorted(found)
                identity["missing"] = sorted(missing)
                identity["evidence_candidates"] = candidates
            except WorkspaceError as exc:
                collection_error = exc
                identity["collection_error"] = exc.kind
        finally:
            if not self.config.keep_workspace:
                try:
                    self._transport.remove(layout.workdir, receipt)
                    identity["cleaned"] = True
                except WorkspaceError as exc:
                    identity["cleaned"] = False
                    identity["cleanup_error"] = exc.kind
            else:
                identity["cleaned"] = False
                identity["retained"] = True

        if invocation is None:  # pragma: no cover - the invoke seam always returns or raises
            raise AssertionError("a remote session returned without an invocation")
        result = self._agent.collect_session(brief, host, invocation)
        result.identity = identity
        if collection_error is not None:
            result.diagnostics["failure_class"] = result.diagnostics.get("failure_class") or "artifact_incomplete"
            result.diagnostics["artifact_error"] = collection_error.kind
            result.error = (result.error or f"artifact collection failed: {collection_error.message}")[:400]

        binding = None
        try:
            binding = epic_for_session(
                host_work=host, env_spec=env_spec, not_before=started, root=self._epic_root,
            )
        except OSError as exc:  # reading the epic root must not discard a billed session
            identity["epic_error"] = f"{type(exc).__name__}: {exc}"[:180]
        if binding is not None:
            identity["epic"] = binding.as_dict()
            identity["cost_bound"] = True
        else:
            result.cost_incomplete = True
            result.cost_usd = None
            result.diagnostics["cost_binding"] = "unbound"
            result.diagnostics["cost_binding_reason"] = "no hmz epic names this session and remote workspace"
        if result.diagnostics.get("failure_class") == "" and result.cost_incomplete and result.source is None:
            result.diagnostics["failure_class"] = "no_submission"
        return result

    def _collect(self, layout, prepared, host: Path):
        """Read the declared artifacts back into the canonical host directory."""
        basis = [
            prepared.submit,
            "NOTES.md",
            "FEEDBACK.md",
            "self_checks.jsonl",
            ".selfcheck/self_checks.jsonl",
        ]
        found = self._transport.collect(layout.workdir, basis)
        missing = [name for name in basis if name not in found]
        self._materialize(host, found)
        text = b"".join(found.get(name, b"") for name in ("NOTES.md", "FEEDBACK.md"))
        candidates = self._agent.evidence_candidates(text.decode("utf-8", "replace"))
        extra: dict[str, bytes] = {}
        if candidates:
            extra = self._transport.collect(layout.workdir, candidates)
            # A candidate that is a directory or a control_host path never comes back as a
            # regular file; writing only what was returned keeps collection bounded.
            extra = {name: data for name, data in extra.items() if name not in found}
            self._materialize(host, extra)
        found.update(extra)
        # A timeout may still leave a checkpointed submission; anything the transport did
        # not return is recorded as a missing artifact, never invented locally.
        return found, candidates, missing

    @staticmethod
    def _materialize(host: Path, files: dict[str, bytes]) -> None:
        """Write collected bytes atomically under the canonical host directory."""
        from .workspace_env import check_relative_destination

        for relative, data in files.items():
            target = check_relative_destination(host, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".taic.tmp")
            temp.write_bytes(data)
            os.replace(temp, target)
