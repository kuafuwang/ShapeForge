"""One coding-agent turn inside an isolated worktree for a human-approved change.

The agent may edit the worktree and run tests, but the flow does not commit. The outer
command collects a patch for a person; it is never applied automatically.
"""

from datetime import timedelta

from hmz.flows import Agent, AgentCollection, Budget, EnvCollection, FlowContext, FlowParams, LocalEnv, Permission, PermissionKind, flow


class Coder(Agent):
    """A pi session with terminal access; no kernel-only skill is imposed."""

    _permission = Permission(
        local=PermissionKind.ALL,
        user=PermissionKind.READ,
        system=PermissionKind.READ,
        online=PermissionKind.NONE,
    )

class Agents(AgentCollection):
    assistant: Coder


class Envs(EnvCollection):
    workspace: LocalEnv


class Params(FlowParams):
    turn_output_tokens: int = 120_000
    turn_minutes: int = 40
    code_cost: float = 0.5


@flow(agents=Agents, envs=Envs, params=Params)
async def code_patch(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    del ctx
    session = await agents["assistant"].spawn()
    await agents["assistant"].run(
        task,
        session=session,
        env=envs["workspace"],
        budget=Budget(
            cost=params.code_cost,
            output_tokens=params.turn_output_tokens,
            duration=timedelta(minutes=params.turn_minutes),
            graceful=False,
        ),
    )
