"""One coding-agent session in the workspace it is given.

The search starts this with ``hmz exec``. The flow does not plan, time, or decide which kernel stays.
The main turn has its own token and time budget. When that budget is spent, one short turn in the
same session writes the best kernel to ``submit_name``, then writes NOTES.md, then a short FEEDBACK.md.
"""

from datetime import timedelta

from hmz.flows import Agent, AgentCollection, Budget, EnvCollection, FlowContext, FlowParams, LocalEnv, flow
from hmz.flows.errors import BudgetExceeded


class Assistant(Agent):
    """pi for one kernel. ``taic-tools`` is the skill in this flow's ``skills/``."""

    _skills = ("taic-tools",)


class Agents(AgentCollection):
    """The one coding agent that writes the kernel."""

    assistant: Assistant


class Envs(EnvCollection):
    """The directory that holds the seed, ``TASK.md``, and the submission file."""

    workspace: LocalEnv


class Params(FlowParams):
    """Per-turn limits. The dollar cap is applied by the search as ``turn_output_tokens``."""

    turn_output_tokens: int = 200_000
    turn_minutes: int = 25
    wrap_output_tokens: int = 4000
    wrap_minutes: int = 3
    submit_name: str = "submit.metal"


def _wrap(name: str) -> str:
    return (
        "Stop changing the kernel and do not run another check. "
        f"Write the best kernel you have to {name}. The search reads only that file. "
        "The file does not exist until you write it. If you do not write it, the search records that this session returned no kernel. "
        "kernel.metal is the seed when one was placed. Editing it does not submit. "
        "Then write NOTES.md: each version, what changed, the self-check microseconds and cv, and whether you kept it. "
        "Last, write a short FEEDBACK.md: pitfalls for the next session on this task, and anything the system should change. "
        "Do not copy an older kernel into the submission file to hide a failure."
    )


@flow(agents=Agents, envs=Envs, params=Params)
async def kernel_once(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    """Run ``task`` once. A fresh session; nothing is carried to the next kernel."""
    del ctx
    assistant = agents["assistant"]
    session = await assistant.spawn()
    env = envs["workspace"]
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
        await assistant.run(_wrap(params.submit_name), session=session, env=env, budget=wrap)
