"""One coding-agent turn in the workspace it is given.

The search starts this with ``hmz exec``. The flow does not plan, time, or decide which kernel stays.
``hmz exec -f <this directory> -a assistant=pi/<provider>/<model> -p budget.cost=<usd>``.
"""

from hmz.flows import Agent, AgentCollection, EnvCollection, FlowContext, FlowParams, LocalEnv, flow


class Assistant(Agent):
    """pi for one kernel. ``taic-tools`` is the skill in this flow's ``skills/``."""

    _skills = ("taic-tools",)


class Agents(AgentCollection):
    """The one coding agent that writes the kernel."""

    assistant: Assistant


class Envs(EnvCollection):
    """The directory that holds ``kernel.metal`` and ``TASK.md``."""

    workspace: LocalEnv


class Params(FlowParams):
    """Nothing of the flow's own. The dollar cap is ``-p budget.cost``."""


@flow(agents=Agents, envs=Envs, params=Params)
async def kernel_once(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    """Run ``task`` once. A fresh session; nothing is carried to the next kernel."""
    del params, ctx
    assistant = agents["assistant"]
    session = await assistant.spawn()
    await assistant.run(task, session=session, env=envs["workspace"])
