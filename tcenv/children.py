"""Child processes that must die with the parent.

A Metal worker holds the machine GPU lock and can keep driving the display.
SIGTERM delivered only to the parent does not reach that child. The parent
records each child, kills its process group on the way out, and does the same
from a SIGTERM/SIGINT handler.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess

_LIVE: list[subprocess.Popen] = []
# Pids of children this module started with ``start_new_session=True``. Their pid is the process
# group id, and the group can outlive the leader (a Metal compile child, an xctrace recorder, the
# pi session behind hmz). Membership is dropped by ``forget`` so a recycled pid is never signaled.
_GROUP_LEADERS: set[int] = set()
_INSTALLED = False
_SECRET = re.compile(r"(API_KEY|APIKEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)


def scrubbed_env(base: dict | None = None) -> dict[str, str]:
    """The parent environment without API keys and other secrets.

    The child still receives PATH, HOME and the Metal toolchain variables.
    Credentials for the coding agent live in the models file, not in the environment.
    """
    source = os.environ if base is None else base
    return {str(key): str(value) for key, value in source.items() if not _SECRET.search(str(key))}


def kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's process group.

    For a child started by :func:`popen_group`, the group is signalled even after the leader has
    exited: the leader exiting does not empty the group, and the survivor is the process still
    holding the GPU or calling the provider. A process this module did not start falls back to the
    conservative ``poll``/``kill`` pair, because its pid is not known to be a group id.
    """
    in_group = proc.pid in _GROUP_LEADERS
    if not in_group and proc.poll() is not None:
        return
    try:
        if in_group:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        return
    except OSError:
        try:
            proc.kill()
        except OSError:
            return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def register(proc: subprocess.Popen) -> None:
    _LIVE.append(proc)


def forget(proc: subprocess.Popen) -> None:
    _GROUP_LEADERS.discard(proc.pid)
    try:
        _LIVE.remove(proc)
    except ValueError:
        pass


def kill_registered() -> None:
    for proc in list(_LIVE):
        kill_group(proc)
        forget(proc)


def install_termination_handlers() -> None:
    """Kill registered children, then re-raise SIGTERM or SIGINT with the default action."""
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def handle(signum, _frame):
        kill_registered()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)


def popen_group(argv: list[str], **kwargs) -> subprocess.Popen:
    """Start ``argv`` in a new session and register it. ``env`` is scrubbed."""
    env = scrubbed_env(kwargs.pop("env", None))
    proc = subprocess.Popen(argv, start_new_session=True, env=env, **kwargs)
    _GROUP_LEADERS.add(proc.pid)
    register(proc)
    return proc
