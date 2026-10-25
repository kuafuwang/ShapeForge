"""Test-only flow for the Docker isolated-agent simulation.

The body and parameters are the same as ``taic/flows/kernel_once``, with two deliberate
differences that belong only to this fixture:

- ``online=PermissionKind.ALL`` lets the fake agent reach the control-host fake provider
  over ``host.docker.internal``.  The production flow keeps ``online=PermissionKind.NONE``;
  this fixture must never be used as evidence about the production fence.
- no ``_skills``: the fixture does not need ``taic-tools`` and must not duplicate its text.

The optional ``remote_workspace`` role is what the simulation's ``-e`` mapping fills.
"""

from datetime import timedelta
from typing import NotRequired

from hmz.flows import (
    Agent,
    AgentCollection,
    BashEnvMixin,
    Budget,
    Env,
    EnvCollection,
    FilesEnvMixin,
    FlowContext,
    FlowParams,
    LocalEnv,
    Permission,
    PermissionKind,
    flow,
)
from hmz.flows.errors import BudgetExceeded


class Assistant(Agent):
    """The fake ACP agent, with network allowed for the control-host fake provider."""

    _permission = Permission(
        local=PermissionKind.ALL,
        user=PermissionKind.READ,
        system=PermissionKind.READ,
        online=PermissionKind.ALL,
    )


class Agents(AgentCollection):
    assistant: Assistant


class RemoteWorkspace(Env, BashEnvMixin, FilesEnvMixin):
    """The Docker host's workspace directory."""


class Envs(EnvCollection):
    workspace: LocalEnv
    remote_workspace: NotRequired[RemoteWorkspace]


class Params(FlowParams):
    turn_output_tokens: int = 200_000
    turn_minutes: int = 25
    wrap_output_tokens: int = 4000
    wrap_minutes: int = 3
    submit_name: str = "submit.metal"
    seed_name: str = "kernel.metal"


@flow(agents=Agents, envs=Envs, params=Params)
async def kernel_once(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    """Run the fake agent once, in the remote workspace when it is given."""
    del ctx
    assistant = agents["assistant"]
    session = await assistant.spawn()
    env = envs.get("remote_workspace") or envs["workspace"]
    budget = Budget(
        output_tokens=params.turn_output_tokens,
        duration=timedelta(minutes=params.turn_minutes),
        graceful=False,
    )
    try:
        await assistant.run(task, session=session, env=env, budget=budget)
    except BudgetExceeded:
        wrap = Budget(
            output_tokens=params.wrap_output_tokens,
            duration=timedelta(minutes=params.wrap_minutes),
            graceful=False,
        )
        await assistant.run("finish", session=session, env=env, budget=wrap)
