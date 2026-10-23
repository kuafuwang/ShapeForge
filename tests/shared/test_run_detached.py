"""scripts/run_detached.py: a command started through it must not be a descendant of whoever started it (no GPU, no network, a few seconds)."""
# 总契约：脱离后的命令不再是启动者的子孙。下面每条断言失败，都是这条契约的一个破口。

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_detached.py"


def _ps(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


def _alive(pid: int) -> bool:
    return bool(_ps(pid, "pid"))


def _launch(tmp_path: Path, *cmd: str, extra=(), pass_fds=()) -> tuple[subprocess.CompletedProcess, Path, Path]:
    log, pidfile = tmp_path / "out.log", tmp_path / "out.pid"
    r = subprocess.run([sys.executable, str(SCRIPT), "--log", str(log), "--pidfile", str(pidfile), *extra, "--", *cmd], capture_output=True, text=True, timeout=20, pass_fds=pass_fds)
    return r, log, pidfile


def _wait_for(predicate, timeout=8.0):
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def _stop(pid: int) -> None:
    """Kill the detached process group (never our own) and wait until the process is gone."""
    pgid = os.getpgid(pid) if _alive(pid) else None
    # 杀的是对方的进程组；pgid 与本测试相同则拒绝，避免把测试自己一起杀掉。
    if pgid is not None and pgid != os.getpgid(0):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    # 对方必须已经消失；还活着说明进程组信号没打到被脱离的那个 pid。
    assert _wait_for(lambda: not _alive(pid)), f"pid {pid} did not exit"


def test_the_command_is_detached_from_the_launcher(tmp_path):
    # 命令是 sleep 30，长于下面的 15 秒上限：启动器若同步等它结束，时间不变量必破。
    start = time.time()
    r, log, pidfile = _launch(tmp_path, "sleep", "30")
    pid = int(r.stdout.strip())
    try:
        # 脱离成功才是 0；非 0 表示还没交出一个已收养的进程就失败了。
        assert r.returncode == 0, r.stderr
        # 必须马上返回，不能等 sleep 30；超过 15 秒说明启动器仍挂在命令上。
        assert time.time() - start < 15, "the launcher must not wait for the command"
        # 印出的 pid 必须仍活着：启动器返回不等于命令已被杀掉或已经结束。
        assert _alive(pid)
        # pidfile 里必须是同一个最终 pid；对不上说明文件里不是印出来的那个进程。
        assert pidfile.read_text().strip() == str(pid)  # the pidfile holds the final pid, the one that was printed
        # ppid 必须是 1：已被 launchd 收养，不是启动器、终端或本测试进程的子进程。
        assert _ps(pid, "ppid") == "1", "re-parented to launchd: not a child of the launcher, the shell or the IDE"
        # 必须有自己的进程组；与本测试同组的话，关终端杀组会把它一起带走。
        assert os.getpgid(pid) != os.getpgid(0), "own process group"
        # 必须有自己的会话；与本测试同 sid 的话，仍挂在原来的终端会话上。
        assert os.getsid(pid) != os.getsid(0), "own session"
        # sid 不得等于 pid：不能是会话领袖。这是双重 fork 的结果，否则它还能再拿到控制终端。
        assert os.getsid(pid) != pid, "not a session leader (the double fork) so that it can never acquire a controlling terminal"
        # 控制终端只能是 ??、? 或空；出现具体 tty 说明没有脱离会话。
        assert _ps(pid, "tty") in ("??", "?", ""), "no controlling terminal"
        # 进程名必须是 sleep 30 本身：caffeinate 把 pid 让给了命令。仍显示 caffeinate 就不是最终 pid。
        assert _ps(pid, "command") == "sleep 30", "the detached pid is the command itself (caffeinate hands the pid over to it)"
        # 本机有 caffeinate 时，它必须是该 pid 的子进程，替命令持有防空闲睡眠断言；等不到说明没挂上。
        if shutil.which("caffeinate"):
            assert _wait_for(lambda: bool(subprocess.run(["pgrep", "-P", str(pid), "caffeinate"], capture_output=True).stdout.strip())), "caffeinate keeps the no-idle-sleep assertion on behalf of the command"
        # 日志里必须有 run_detached 横幅；没有说明启动记录没写进 --log。
        assert log.exists() and "run_detached" in log.read_text()
    finally:
        _stop(pid)


def test_output_goes_to_the_log_stdin_is_empty_and_no_descriptor_leaks(tmp_path):
    rd, wr = os.pipe()
    leak = os.dup(wr)
    # 故意留下可继承的 fd 40（命令自己不会打开这个号）：/dev/fd/40 还在就只能是继承描述符没关。
    os.dup2(leak, 40)  # a high descriptor number that nothing in the command opens by itself
    os.set_inheritable(40, True)
    os.close(leak)
    try:
        script = 'echo "out:$RD_TEST"; echo err >&2; if [ -z "$(cat)" ]; then echo stdin-empty; fi; if [ -e /dev/fd/40 ]; then echo leaked; else echo no-leak; fi'
        r, log, pidfile = _launch(tmp_path, "sh", "-c", script, extra=("--env", "RD_TEST=hello"), pass_fds=(40,))
        # 启动器必须成功返回；非 0 时日志还不能证明 stdio、环境和 fd。
        assert r.returncode == 0, r.stderr
        pid = int(r.stdout.strip())
        # 按整行匹配：横幅会复述命令文本，里面的 leaked 等词不算命令自己的输出。
        lines = lambda: log.read_text().splitlines()  # noqa: E731  (the first line, the banner, repeats the command line)
        # 必须等到 no-leak 或 leaked 这一行；超时说明探针没进日志，泄漏与否还不可判。
        assert _wait_for(lambda: "no-leak" in lines() or "leaked" in lines()), log.read_text()
        # out:hello：stdout 进了日志，且 --env 的 RD_TEST 传进了命令。err：stderr 进了同一份日志。
        # stdin-empty：stdin 为空（/dev/null）。no-leak 且没有整行 leaked：继承的 fd 40 已被关掉。
        assert "out:hello" in lines() and "err" in lines() and "stdin-empty" in lines() and "no-leak" in lines() and "leaked" not in lines()
        # 探针写完后进程必须退出；超时仍活着说明这条短命令脱离后没有正常结束。
        assert _wait_for(lambda: not _alive(pid))
    finally:
        os.close(40)
        os.close(rd)
        os.close(wr)


def test_a_command_that_does_not_exist_fails_cleanly(tmp_path):
    r, log, pidfile = _launch(tmp_path, "/nonexistent/definitely-not-a-command", extra=("--no-caffeinate",))
    # 命令不存在时启动器必须非 0 退出，且 stderr 带 not found；0 会让调用方以为已经在跑。
    assert r.returncode != 0 and "not found" in r.stderr
    # stdout 必须为空：失败路径不能交出 pid，否则调用方会去等一个没启动的进程。
    assert r.stdout.strip() == ""
    # 不得留下 pidfile：没有最终 pid 可公布。文件还在就表示失败仍声称有脱离进程。
    assert not pidfile.exists()


# 本机找不到 caffeinate 就跳过（标记为仅 macOS）。断言只看日志文件：事先写好的第一行还在文件头，说明是追加打开，不是截断重写。
@pytest.mark.skipif(shutil.which("caffeinate") is None, reason="caffeinate is macOS only")
def test_the_log_is_appended_not_truncated(tmp_path):
    log = tmp_path / "out.log"
    log.write_text("earlier output\n")
    r, _, _ = _launch(tmp_path, "true")
    # 包装路径上必须 0 退出；非 0 时还没有写日志这件事，追加与否无从谈起。
    assert r.returncode == 0, r.stderr
    # 先等到 run_detached 横幅落盘；横幅没出现就不能判断新内容是追加上去的。
    assert _wait_for(lambda: "run_detached" in log.read_text())
    # 事先的第一行必须还在文件头：日志是追加。若打开时被截断，开头会变成横幅。
    assert log.read_text().startswith("earlier output\n")
