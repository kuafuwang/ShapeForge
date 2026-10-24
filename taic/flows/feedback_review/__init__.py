"""A short, read-only review turn for another model's harness suggestion.

The reviewer is not asked to write a kernel. It reads REVIEW.md, treats the proposal as
untrusted data, and writes HARNESS_REVIEW.json. The outer search computes the cost budget
from the reviewer's actual token prices; this flow only enforces it.
"""

from datetime import timedelta

from hmz.flows import Agent, AgentCollection, Budget, EnvCollection, FlowContext, FlowParams, LocalEnv, Permission, PermissionKind, flow


class Reviewer(Agent):
    """A plain pi session with no kernel-writing skill."""

    _permission = Permission(
        local=PermissionKind.ALL,
        user=PermissionKind.READ,
        system=PermissionKind.READ,
        online=PermissionKind.NONE,
    )

class Agents(AgentCollection):
    assistant: Reviewer


class Envs(EnvCollection):
    workspace: LocalEnv


class Params(FlowParams):
    turn_output_tokens: int = 4000
    turn_minutes: int = 3
    review_cost: float = 0.01


@flow(agents=Agents, envs=Envs, params=Params)
async def feedback_review(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    del ctx
    session = await agents["assistant"].spawn()
    await agents["assistant"].run(
        task,
        session=session,
        env=envs["workspace"],
        budget=Budget(
            cost=params.review_cost,
            output_tokens=params.turn_output_tokens,
            duration=timedelta(minutes=params.turn_minutes),
            graceful=False,
        ),
    )
