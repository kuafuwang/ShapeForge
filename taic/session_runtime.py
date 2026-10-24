"""Execution placement and local runner interfaces for one kernel-writing session.

Phase 0a keeps the existing hmz behaviour.  This module only names the execution
parameters that used to be passed directly to :func:`taic.kernel_agent.run_kernel_agent`
and provides a local adapter.  It does not start a remote workspace, a container or a
different agent.
"""

from __future__ import annotations

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
