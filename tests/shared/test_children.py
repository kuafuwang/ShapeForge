"""Process-group cleanup: a child that outlives its leader must still be killed."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from tcenv.children import forget, kill_group, install_termination_handlers, popen_group


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie has already been killed; its parent (often launchd, after the leader exited) still
    # has to reap it. Counting it as alive would make this test wait on someone else's cleanup.
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_kill_group_reaches_a_child_that_outlived_its_leader(tmp_path):
    """`kill_group` used to return as soon as the leader had exited, leaving the survivor running.

    The survivor is what matters in production: a Metal compile child, an xctrace recorder or the
    pi session behind hmz is still holding the GPU or calling the provider.
    """
    script = tmp_path / "spawn.sh"
    marker = tmp_path / "child.pid"
    script.write_text(f"#!/bin/sh\nsleep 30 &\necho $! > {marker}\nexit 0\n")
    script.chmod(0o755)
    proc = popen_group([str(script)], cwd=str(tmp_path))
    assert proc.wait(timeout=10) == 0
    deadline = time.time() + 10
    child = 0
    while time.time() < deadline:
        if marker.is_file() and marker.read_text().strip():
            child = int(marker.read_text().strip())
            break
        time.sleep(0.02)
    assert child > 0 and _alive(child)
    try:
        kill_group(proc)
        deadline = time.time() + 10
        while time.time() < deadline and _alive(child):
            time.sleep(0.02)
        assert not _alive(child), "the child survived its leader"
    finally:
        forget(proc)
        try:
            os.kill(child, 9)
        except OSError:
            pass


def test_scrubbed_env_drops_every_key_shaped_variable():
    """A child process inherits the environment; *_KEY and *_TOKEN names are secrets too."""
    from tcenv.children import scrubbed_env

    env = scrubbed_env({
        "PATH": "/bin", "NORMAL_VAR": "v", "KEYBOARD_LAYOUT": "us", "TOKENIZERS_PARALLELISM": "1",
        "GEMINI_KEY": "k", "AWS_ACCESS_KEY_ID": "k", "GH_KEY": "k", "OPENAI_API_KEY": "k",
        "MY_SECRET": "k", "DB_PASSWORD": "k", "KEY": "k",
    })
    assert env == {"PATH": "/bin", "NORMAL_VAR": "v", "KEYBOARD_LAYOUT": "us", "TOKENIZERS_PARALLELISM": "1"}


def test_sighup_kills_a_registered_child(tmp_path):
    """Closing an IDE/terminal sends SIGHUP; the hmz session must not outlive the search."""
    import sys

    marker = tmp_path / "child.pid"
    script = tmp_path / "parent.py"
    script.write_text(
        "import sys, time\n"
        "from tcenv.children import install_termination_handlers, popen_group\n"
        "install_termination_handlers()\n"
        "child = popen_group([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(marker)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    import os
    import subprocess

    repo = Path(__file__).resolve().parents[2]
    parent = subprocess.Popen([sys.executable, str(script)], cwd=repo, env={**os.environ, "PYTHONPATH": str(repo)})
    try:
        deadline = time.time() + 15
        child = 0
        while time.time() < deadline:
            if marker.is_file() and marker.read_text().strip():
                child = int(marker.read_text().strip())
                break
            time.sleep(0.02)
        assert child > 0 and _alive(child)
        parent.send_signal(signal.SIGHUP)
        parent.wait(timeout=10)
        deadline = time.time() + 10
        while time.time() < deadline and _alive(child):
            time.sleep(0.02)
        assert not _alive(child), "the registered child survived SIGHUP"
    finally:
        try:
            parent.kill()
        except OSError:
            pass

