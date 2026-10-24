"""Process-group cleanup: a child that outlives its leader must still be killed."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from tcenv.children import forget, kill_group, install_termination_handlers, popen_group


def _alive(pid: int) -> bool:
    """@brief 判断进程是否还活着，已被杀但尚未回收的僵尸进程算作已死。

    @param pid 进程号。
    @return 进程仍在运行且不是僵尸时为 True。
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie has already been killed; its parent (often launchd, after the leader exited) still
    # has to reap it. Counting it as alive would make this test wait on someone else's cleanup.
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def test_kill_group_reaches_a_child_that_outlived_its_leader(tmp_path):
    """@brief 验证 kill_group 能杀掉比组长活得更久的子进程。

    kill_group 过去只要组长退出就返回，存活的子进程继续跑。生产里要紧的正是这个存活者：一个仍占着 GPU 的
    Metal 编译子进程、一个 xctrace 录制进程，或 hmz 背后还在调模型的 pi 会话。

    @param tmp_path pytest 提供的临时目录。
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
    """@brief 验证子进程环境里所有密钥形状的变量（*_KEY、*_TOKEN、*_SECRET、*_PASSWORD）都被剔除，而名字里只是含 KEY 的普通变量保留。

    子进程会继承环境，除 *_KEY 和 *_TOKEN 之外，其它名字里带密钥意味的变量同样是密钥。
    """
    from tcenv.children import scrubbed_env

    env = scrubbed_env({
        "PATH": "/bin", "NORMAL_VAR": "v", "KEYBOARD_LAYOUT": "us", "TOKENIZERS_PARALLELISM": "1",
        "GEMINI_KEY": "k", "AWS_ACCESS_KEY_ID": "k", "GH_KEY": "k", "OPENAI_API_KEY": "k",
        "MY_SECRET": "k", "DB_PASSWORD": "k", "KEY": "k",
    })
    assert env == {"PATH": "/bin", "NORMAL_VAR": "v", "KEYBOARD_LAYOUT": "us", "TOKENIZERS_PARALLELISM": "1"}


def test_scrubbed_env_can_preserve_one_named_capability():
    """@brief 只有调用方明确点名的短期能力变量能穿过敏感变量清洗。"""
    from tcenv.children import scrubbed_env

    env = scrubbed_env(
        {"PATH": "/bin", "TAIC_SELFCHECK_TOKEN": "scoped", "OTHER_TOKEN": "secret"},
        keep=frozenset({"TAIC_SELFCHECK_TOKEN"}),
    )
    assert env == {"PATH": "/bin", "TAIC_SELFCHECK_TOKEN": "scoped"}


def test_sighup_kills_a_registered_child(tmp_path):
    """@brief 验证父进程收到 SIGHUP 时，它注册过的子进程也被杀掉。

    关掉 IDE 或终端会发 SIGHUP，hmz 会话不应该活得比搜索更久。

    @param tmp_path pytest 提供的临时目录。
    """
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

