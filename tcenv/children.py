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
    """SIGKILL the child's process group. A child that has already exited is left alone."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
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
    register(proc)
    return proc
