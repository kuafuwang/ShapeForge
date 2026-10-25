"""One coding agent session in the workspace it is given.

The search starts this with ``hmz exec``. The flow does not plan, time, or decide which kernel stays.
The main turn has its own token and time budget. The submission file, NOTES.md and FEEDBACK.md are
updated during that turn. When the budget is spent, one short turn copies the best kernel with
``cp`` if the submission file is still missing, then fills in any lines those two notes still lack.

``workspace`` is the local directory the run was started in and is what every default run uses.
``remote_workspace`` is an optional second role: when the search passes
``-e remote_workspace=ssh@<runtime>/<path>`` the agent works there instead. Only one of the two is
used, and a run that names neither adds no ``-e`` and keeps the current local behaviour exactly.
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
    """pi for one kernel. ``taic-tools`` is the skill in this flow's ``skills/``."""

    _permission = Permission(
        local=PermissionKind.ALL,
        user=PermissionKind.READ,
        system=PermissionKind.READ,
        online=PermissionKind.NONE,
    )

    _skills = ("taic-tools",)


class Agents(AgentCollection):
    """The one coding agent that writes the kernel."""

    assistant: Assistant


class RemoteWorkspace(Env, BashEnvMixin, FilesEnvMixin):
    """An optional workspace on a pre-provisioned host, reached over ssh.

    The mixins are what the coding agent needs of it: run programs and read and write the
    session's files. Nothing else is granted, and the role is absent unless the caller passes
    it with ``-e``.
    """


class Envs(EnvCollection):
    """The directory that holds the seed, ``TASK.md``, and the submission file."""

    workspace: LocalEnv
    remote_workspace: NotRequired[RemoteWorkspace]


class Params(FlowParams):
    """Per-turn limits. The dollar cap is applied by the search as ``turn_output_tokens``."""

    turn_output_tokens: int = 200_000
    turn_minutes: int = 25
    wrap_output_tokens: int = 4000
    wrap_minutes: int = 3
    submit_name: str = "submit.metal"
    seed_name: str = "kernel.metal"


def _wrap(name: str, seed_name: str = "kernel.metal") -> str:
    return (
        "Stop changing the kernel and do not run another check. "
        f"If {name} is already in this directory, do not rewrite it. "
        f"If it is missing, copy your best version onto {name}.tmp with cp, then mv {name}.tmp onto {name}. "
        "Do not use the write tool to retype the whole kernel. "
        "The search reads only that file. If it is still missing, the search records that this session returned no kernel. "
        f"{seed_name} is the seed when one was placed. Editing it does not submit. "
        "Then fill in any missing lines in FEEDBACK.md and NOTES.md. "
        "One line in NOTES.md per version you tried. One line in the matching FEEDBACK.md section per pitfall or rejected premise. "
        "Do not copy an older kernel into the submission file to hide a failure."
    )


@flow(agents=Agents, envs=Envs, params=Params)
async def kernel_once(task: str, *, agents: Agents, envs: Envs, params: Params, ctx: FlowContext) -> None:
    """Run ``task`` once. A fresh session; nothing is carried to the next kernel."""
    del ctx
    assistant = agents["assistant"]
    session = await assistant.spawn()
    # The remote role is filled only by an explicit `-e`; a local run still takes the
    # runtime's own workspace and never reaches for a host.
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
        await assistant.run(_wrap(params.submit_name, params.seed_name), session=session, env=env, budget=wrap)
