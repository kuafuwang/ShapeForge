#!/usr/bin/env python3
"""Start a long command so that it survives the closing of the terminal / editor window that launched it (stdlib only, macOS and Linux).

    python3 scripts/run_detached.py --log /tmp/exp_x.log --pidfile /tmp/exp_x.pid [--cwd DIR] [--env K=V ...] -- <cmd> [args...]

The command is started as ``caffeinate -i <cmd> [args...]`` (``caffeinate -i`` keeps the machine from idle-sleeping while the command runs, and
exits with it; ``--no-caffeinate`` runs the command bare). The launcher itself exits at once and prints the pid of the detached process. That pid
is the command itself: caffeinate keeps the process it was exec'd as for the command and forks a small helper child that holds the sleep assertion
on behalf of that pid (``pmset -g assertions`` shows it).

What makes it survive an IDE window close (which kills the process tree / process group / pty session of its terminals):

* double fork: the first child calls ``os.setsid()`` (own session and process group, no controlling terminal) and exits, so the second child
  (the one that is exec'd) is re-parented to launchd (ppid 1) and is not a session leader, so it can never acquire a controlling terminal;
* every inherited descriptor is closed; stdin is /dev/null; stdout and stderr go to ``--log`` (appended; the file is created if needed);
* SIGHUP is ignored (like ``nohup``) and SIGPIPE is back to the default (Python ignores it, and exec would pass that on to the command).

``--pidfile`` is written (atomically) by the detached process itself before the exec, so it holds the final pid. If the exec fails, the launcher
prints the error, exits non-zero and leaves no pidfile. The environment is inherited; ``--env K=V`` (repeatable) adds to it.
"""

# 往下读的顺序：main 解析参数，launch 做双重 fork 并用管道取回 pid，_daemon_child 在第二个子进程里换成命令。
# 关掉 IDE 窗口会杀掉终端的进程组和会话。命令要活下来，得离开这两者，而且自己不能再当会话领袖。

from __future__ import annotations

import argparse
import os
import resource
import shutil
import signal
import sys
import time

# 与正常的一行 pid 区分开。exec 没换成命令时写它，启动器据此失败退出，并且不留下 pidfile。
_EXEC_FAILED = "EXEC-FAILED "  # marker the detached child writes into the pipe if it cannot exec


def _close_all_but(keep: set[int]) -> None:
    """Close every descriptor above stderr except those in ``keep``."""
    # 没有可移植的“列出所有打开的 fd”的办法（macOS 也没有 /proc/self/fd）。
    # 软限制是还能新分配的 fd 号上界；平时继承来的号都在它下面，所以拿它当扫描终点。
    # 从 3 起：0/1/2 是命令的标准输入输出，调用方已经接好，这里再关就把日志和 /dev/null 拆掉。
    try:
        soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # “无限”没有可扫的尽头，换成一个够用的有限上界。
        top = soft if soft != resource.RLIM_INFINITY else 65536
    except (ValueError, OSError):
        # 问不到限制时仍要扫一段，否则 IDE 传来的高位 fd 会漏进长跑的命令。
        top = 4096
    # 软限制可以事后调小，已经打开的更大号仍在。至少扫到 256，免得一个过小的报告值漏掉这段常见继承 fd。
    # 软限制也可能到一百万以上，封顶 65536，不按整个上界去关。再高的号不在这次清理里。
    top = max(256, min(top, 65536))
    # closerange(lo, hi) 关掉区间 [lo, hi)，空号的错误会被忽略，不必对每个整数 close 再吞 EBADF。
    # 按 keep 把区间切开，那些号留着。此刻唯一要留的是管道写端：它还得把 pid 送回启动器，
    # 提前关掉的话，对面立刻读到 EOF，分不清“写过 pid”和“什么都没写”。
    lo = 3
    for fd in sorted(keep) + [top]:
        if fd > lo:
            os.closerange(lo, fd)
        lo = max(lo, fd + 1)


def _write_pidfile(path: str, pid: int) -> None:
    # 由即将 exec 的这个进程来写：exec 不改变 pid，文件里就是最终那个进程号。
    # 临时名带上 pid，避免两次启动同时写同一个临时文件。
    # 同一目录里 os.replace 是原子的，读的人要么看到旧文件，要么看到完整的一行，不会读到半截。
    tmp = f"{path}.{pid}.tmp"
    with open(tmp, "w") as f:
        f.write(f"{pid}\n")
    os.replace(tmp, path)


def _daemon_child(argv: list[str], log: str, pidfile: str | None, cwd: str | None, env: dict[str, str], wfd: int) -> None:
    """Runs in the second child. Never returns: it execs the command or exits."""
    # 这里已经不是会话领袖，之后即使打开终端设备也拿不到控制终端。
    try:
        # 工作目录要在 exec 前定下来。日志和 pidfile 在 launch 里已变成绝对路径，不受这次切换影响。
        if cwd:
            os.chdir(cwd)
        # 脱离之后没有控制终端。stdin 若仍是 IDE 的 pty，进程还连着那扇要关的窗口。
        devnull = os.open(os.devnull, os.O_RDONLY)
        # 用追加，同一次实验里已经写下的内容才还在。stdout 和 stderr 会接到同一次打开上，共用文件偏移。
        logfd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        # 命令眼里的标准输入输出就是 fd 0/1/2，而且必须可继承才能穿过 exec。
        # os.open 得到的号默认 close-on-exec，exec 时会被丢掉；dup2 到 0/1/2 得到的新描述符默认可继承。
        os.dup2(devnull, 0)
        os.dup2(logfd, 1)
        os.dup2(logfd, 2)
        # 0/1/2 已经是 /dev/null 和日志。其余号是这两次 open 的原件，或 IDE 继承来的 pty：
        # 原件只是重复，pty 留着则进程还连着要关的窗口。管道写端还要回报 pid，所以留下。
        _close_all_but({wfd})
        # 终端挂断发 SIGHUP。必须用“忽略”而不是 Python 处理函数：忽略能穿过 exec，处理函数会被 exec 恢复成默认。
        # 这样和 nohup 一样，挂断信号不会杀掉已经启动的命令。
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        # Python 默认忽略 SIGPIPE，好让解释器自己写到断开的管道时抛异常而不是死掉。
        # 这个忽略会原样传给 exec 之后的程序。下游关了管道时，命令应按系统默认被 SIGPIPE 杀掉。
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        # --env 只盖列出的键。fork 已经复制了整份环境，没写进来的变量保持原样。
        os.environ.update(env)
        pid = os.getpid()
        # 旧的 stdout 按终端创建，行缓冲在那时就定了。换成包住 fd 2 的普通文件包装，横幅和 stderr 走同一条追加日志。
        # closefd=False：包装不拥有 fd 2。关掉包装不能把 stderr 一起关，否则 exec 之后命令没有标准错误。
        sys.stdout = sys.stderr = open(2, "w", closefd=False)
        # 普通文件是块缓冲，exec 不会冲洗 Python 的缓冲。不 flush，这行会在换成命令时丢掉。
        # 日志里的 sid 是新会话的 id，可用来核对它已经不在 IDE 终端那个会话里。
        print(f"=== run_detached {time.strftime('%Y-%m-%d %H:%M:%S')} pid {pid} sid {os.getsid(0)}: {' '.join(argv)}", flush=True)
        # 写在 exec 之前。随后 caffeinate 再把自己换成目标命令时 pid 也不变，所以这里就是最终 pid。
        if pidfile:
            _write_pidfile(pidfile, pid)
        # 直接写内核管道，不进 Python 的块缓冲，否则 exec 可能把 pid 弄丢。
        os.write(wfd, f"{pid}\n".encode())
        # 写完才带上 close-on-exec。成功的 exec 会关掉它，启动器把随后的 EOF 看成“已经换成命令”。
        # 若仍然可继承，命令会一直握着写端，对面永远等不到 EOF。
        os.set_inheritable(wfd, False)  # closed by a successful exec: the launcher sees EOF right after the pid
        os.execvp(argv[0], argv)
    except BaseException as e:  # noqa: BLE001  report to the launcher, whatever went wrong
        # 还没换成命令。pidfile 若已写上就删掉，避免留下一个并不会存在的进程号。删不掉也要继续把失败告诉启动器。
        try:
            if pidfile:
                os.unlink(pidfile)
        except OSError:
            pass
        try:
            # 可能 pid 那一行已经写过，失败标记跟在后面；也可能还没写 pid，标记自己就是第一行。
            os.write(wfd, f"{_EXEC_FAILED}{type(e).__name__}: {e}\n".encode())
        finally:
            # 127 是 exec 失败的惯例状态。必须在这里 _exit：若返回，会掉进 launch 里那个 finally 的 _exit(0)，失败被收成正常退出。
            # 启动器不收割这个进程（它已经要被 pid 1 收养），它认的是管道里的标记。
            os._exit(127)


def launch(argv: list[str], log: str, *, pidfile: str | None = None, cwd: str | None = None, env: dict[str, str] | None = None, caffeinate: bool = True) -> int:
    """Start ``argv`` detached (see the module docstring) and return the pid of the detached process. Raises ``RuntimeError`` if it cannot be started."""
    argv = list(argv)
    if not argv:
        raise RuntimeError("no command given")
    env = dict(env or {})
    # 在脱离之前确认命令找得到。PATH 含 --env 的覆盖，这样常见的“找不到”由启动器直接报错，不会先变成一个已脱离的失败进程。
    merged_path = env.get("PATH", os.environ.get("PATH", os.defpath))
    if shutil.which(argv[0], path=merged_path) is None:
        raise RuntimeError(f"command not found (or not executable): {argv[0]}")
    if caffeinate:
        cf = shutil.which("caffeinate", path=merged_path)
        if cf:
            # -i 防止机器空闲睡眠。caffeinate 会 exec 成目标命令，pid 保持不变，所以报上去的 pid 就是命令自己。
            # 它另外 fork 一个小帮手，替这个 pid 持有睡眠断言。
            argv = [cf, "-i", *argv]
        else:
            # 找不到就裸跑。这行打在启动器自己的 stderr 上：子进程里的 stderr 马上会改接到日志，敲命令的人只能在这里看见。
            print("run_detached: caffeinate not found; running the command without it", file=sys.stderr)
    # 子进程会 chdir。先收成绝对路径并建好目录：目录失败发生在脱离之前，启动器能直接报错；改目录之后仍然写到调用者指定的位置。
    log = os.path.abspath(log)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    if pidfile:
        pidfile = os.path.abspath(pidfile)
        os.makedirs(os.path.dirname(pidfile), exist_ok=True)
    if cwd:
        cwd = os.path.abspath(cwd)

    # 启动结果从这根管道回来。新管道默认 close-on-exec；fork 仍会把写端复制给子进程。
    # 先改成可继承，是把“exec 时关掉写端”推迟到第二个子进程写完 pid 之后。
    rfd, wfd = os.pipe()
    os.set_inheritable(wfd, True)
    # fork 会把还没写出去的用户态缓冲复制进子进程。先冲掉，避免同一段输出在子进程里再出现一次。
    sys.stdout.flush()
    sys.stderr.flush()
    # 启动器还要留下读管道、打印 pid，不能自己去当那个马上退出的会话领袖。
    # 从交互式终端启动时，它往往已经是进程组组长，自己 setsid 也会失败。所以另立会话放在第一个子进程里。
    pid1 = os.fork()
    if pid1 == 0:  # first child: new session, then fork again and leave
        try:
            # 读端只属于启动器。在分出第二个子进程之前关掉，那个进程从出生起就只有写端。
            os.close(rfd)
            # 成为新会话的领袖，同时成为新进程组的组长，并丢掉控制终端。
            # IDE 关窗杀掉的是原来那个终端会话和进程组；还留在里面，命令会一起死。
            os.setsid()
            pid2 = os.fork()
            if pid2 == 0:
                # 这个进程在新会话里，但不是领袖，因此不能再获得控制终端。真正的命令在 _daemon_child 里 exec。
                _daemon_child(argv, log, pidfile, cwd, env, wfd)
        finally:
            # 第一个子进程必须马上走：它一退出，第二个子进程才被 launchd 收养（ppid 1），它手里的写端副本也会关掉。
            # 第二个子进程若从 _daemon_child 返回，同样会掉到这里变成退出码 0，所以失败路径不能返回。
            os._exit(0)
    # 父进程也丢掉写端。否则命令即使已经 exec，这边仍握着写端，读永远等不到 EOF。
    os.close(wfd)
    os.waitpid(pid1, 0)  # the first child exits at once; the second one is already owned by launchd (or will be within microseconds)
    # 先收割第一个子进程，它才不会变成僵尸，写端也只剩第二个子进程那一份。
    # 然后再读：成功时 exec 关掉不可继承的写端，失败时 _exit 关掉它，这边都用 EOF 判断“话说完了”。
    # 一次 read 不一定拿全，所以一直读到空。
    data = b""
    while True:
        chunk = os.read(rfd, 4096)
        if not chunk:
            break
        data += chunk
    os.close(rfd)
    text = data.decode(errors="replace").strip()
    # 一个字都没有：第二个子进程没写成（例如 setsid 或第二次 fork 失败，第一个子进程已经 _exit，写端都关了）。
    if not text:
        raise RuntimeError("the detached process reported nothing")
    first, _, rest = text.partition("\n")
    # 失败标记可能是第一行（pid 还没写），也可能紧跟在 pid 后面（pid 已写入，随后 exec 失败）。两种都是启动失败。
    if first.startswith(_EXEC_FAILED):
        raise RuntimeError(first[len(_EXEC_FAILED):])
    if rest.startswith(_EXEC_FAILED):
        raise RuntimeError(rest[len(_EXEC_FAILED):])
    return int(first)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, usage="%(prog)s --log FILE [--pidfile FILE] [--cwd DIR] [--env K=V ...] [--no-caffeinate] -- CMD [ARGS...]")
    ap.add_argument("--log", required=True, help="file that receives the command's stdout and stderr (appended)")
    ap.add_argument("--pidfile", default=None, help="file that receives the pid of the detached process")
    ap.add_argument("--cwd", default=None, help="working directory of the command (default: the current one)")
    ap.add_argument("--env", action="append", default=[], metavar="K=V", help="extra environment variable (repeatable)")
    ap.add_argument("--no-caffeinate", action="store_true", help="do not wrap the command in 'caffeinate -i'")
    # `--` 之后不再解析启动器自己的选项，整段留给要跑的命令。
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="the command, after a literal --")
    a = ap.parse_args(argv)
    # remainder 里会带上那个 `--` 本身，命令从它后面开始。
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    # 只收集额外的键。完整环境由子进程 fork 时继承，再在 _daemon_child 里把这些键盖上去。
    env = {}
    for kv in a.env:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            ap.error(f"--env expects K=V, got {kv!r}")
        env[k] = v
    try:
        pid = launch(cmd, a.log, pidfile=a.pidfile, cwd=a.cwd, env=env, caffeinate=not a.no_caffeinate)
    except RuntimeError as e:
        # 脱离没完成。非零退出，且不打印 pid（pidfile 也不该存在）。
        print(f"run_detached: {e}", file=sys.stderr)
        return 1
    # 这一行是启动器唯一的成功输出：那个已经脱离、关窗之后还在的 pid。本进程随后就结束。
    print(pid)
    return 0


if __name__ == "__main__":
    # 退出码只表示脱离是否成功，不表示命令以后自己的成败。
    raise SystemExit(main())
