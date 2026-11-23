"""Child processes that must die with the parent.

A Metal worker holds the machine GPU lock and can keep driving the display.
SIGTERM delivered only to the parent does not reach that child. The parent
records each child, kills its process group on the way out, and does the same
from a SIGTERM/SIGINT handler.
"""

from __future__ import annotations

import io
import os
import re
import signal
import subprocess
import sys
import time

_LIVE: list[subprocess.Popen] = []
# Pids of children this module started with ``start_new_session=True``. Their pid is the process
# group id, and the group can outlive the leader (a Metal compile child, an xctrace recorder, the
# pi session behind hmz). Membership is dropped by ``forget`` so a recycled pid is never signaled.
_GROUP_LEADERS: set[int] = set()
_INSTALLED = False
_SECRET = re.compile(r"(^|_)(API_?KEY|APIKEY|KEYS?|TOKENS?|SECRETS?|PASSWORDS?|PASSWD|CREDENTIALS?)($|_)", re.I)


def scrubbed_env(base: dict | None = None, *, keep: frozenset[str] = frozenset()) -> dict[str, str]:
    """@brief 返回剔除了 API key 等敏感变量的环境。

    The parent environment without API keys and other secrets.

    The child still receives PATH, HOME and the Metal toolchain variables.
    Credentials for the coding agent live in the models file, not in the environment.
    ``keep`` is for a short-lived, explicitly scoped capability (the
    self-check bridge token) that the child must receive; callers must name it
    exactly rather than treating every token-shaped variable as safe.

    @param base 基础环境字典；为 None 时使用 ``os.environ``。
    @param keep 要保留的精确环境变量名。
    @return 仅含非敏感变量与 ``keep`` 中变量的新字典。
    """
    source = os.environ if base is None else base
    return {str(key): str(value) for key, value in source.items() if str(key) in keep or not _SECRET.search(str(key))}


def kill_group(proc: subprocess.Popen) -> None:
    """@brief 向子进程所在的进程组发送 SIGKILL。

    SIGKILL the child's process group.

    For a child started by :func:`popen_group`, the group is signalled even after the leader has
    exited: the leader exiting does not empty the group, and the survivor is the process still
    holding the GPU or calling the provider. A process this module did not start falls back to the
    conservative ``poll``/``kill`` pair, because its pid is not known to be a group id.

    @param proc 要终止的子进程。
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
    """@brief 登记一个子进程，使其在父进程退出时被终止。

    @param proc 已启动的子进程。
    """
    _LIVE.append(proc)


def forget(proc: subprocess.Popen) -> None:
    """@brief 注销子进程，并忘记它的进程组，避免复用 pid 被误杀。

    @param proc 要注销的子进程。
    """
    _GROUP_LEADERS.discard(proc.pid)
    try:
        _LIVE.remove(proc)
    except ValueError:
        pass


def kill_registered() -> None:
    """@brief 终止所有已登记的子进程并从表中移除。
    """
    for proc in list(_LIVE):
        kill_group(proc)
        forget(proc)


def install_termination_handlers() -> None:
    """@brief 安装 SIGTERM/SIGINT/SIGHUP/SIGQUIT 处理器，先杀掉子进程再按默认动作退出。

    Kill registered children, then re-raise the signal with its default action. SIGHUP is the one
    an IDE or terminal sends when its window closes; without it the hmz session outlives the search
    and keeps calling the provider with no ledger row.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def handle(signum, _frame):
        kill_registered()
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(signum, handle)


class _PosixSpawnProcess:
    """The small Popen surface used by the remote JobEngine and child reaper.

    This is intentionally not a general ``subprocess.Popen`` replacement. It supports binary
    pipes, poll/wait and signals; unsupported conveniences (for example ``communicate``) are
    left absent instead of being implemented with subtly different timeout semantics.
    """

    def __init__(self, pid: int, argv: list[str], stdin, stdout, stderr) -> None:
        self.pid = int(pid)
        self._argv = list(argv)
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        waited, status = os.waitpid(self.pid, os.WNOHANG)
        if waited == 0:
            return None
        if waited != self.pid:
            raise ChildProcessError(f"waitpid returned {waited} for child {self.pid}")
        self.returncode = int(os.waitstatus_to_exitcode(status))
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        deadline = None if timeout is None else time.monotonic() + max(0.0, float(timeout))
        while True:
            code = self.poll()
            if code is not None:
                return code
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(self._argv, float(timeout))
                time.sleep(min(0.02, remaining))
            else:
                time.sleep(0.02)

    def send_signal(self, sig: int) -> None:
        os.kill(self.pid, int(sig))

    def terminate(self) -> None:
        self.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        self.send_signal(signal.SIGKILL)


def _open_fds() -> set[int]:
    """@brief 枚举当前进程 FD；无法确认时失败而不是留下继承侧门。"""
    for directory in ("/dev/fd", "/proc/self/fd"):
        try:
            found = set()
            for name in os.listdir(directory):
                if not name.isdigit():
                    continue
                fd = int(name)
                try:
                    os.fstat(fd)
                except OSError:
                    # The directory stream's own descriptor may already be closed by the time
                    # listdir returns. Passing that stale entry to CLOSE aborts posix_spawn.
                    continue
                found.add(fd)
            return found
        except FileNotFoundError:
            continue
    raise OSError("cannot enumerate open file descriptors for a spawn-safe worker")


def _spawn_pipe(actions: list[tuple], target_fd: int) -> tuple[int, int]:
    """@brief 建一条 pipe，记录 dup2 目标并返回 (读 FD, 写 FD)。"""
    read_fd, write_fd = os.pipe()
    source = read_fd if target_fd == 0 else write_fd
    if source != target_fd:
        actions.append((os.POSIX_SPAWN_DUP2, source, target_fd))
    else:
        # os.pipe() starts non-inheritable. Without a dup2 action we must explicitly keep
        # this already-correct descriptor across exec.
        os.set_inheritable(source, True)
    # On macOS, DUP2(fd, fd) is not a harmless no-op: it can close the descriptor.
    # If the pipe already has the target number, leave it open and only close the others.
    return read_fd, write_fd


def popen_group_posix(argv: list[str], **kwargs) -> _PosixSpawnProcess:
    """@brief 用真正的 ``posix_spawn`` 启动 remote worker，不经过 gRPC 后的 fork-exec。

    ``subprocess.Popen(start_new_session=True)`` forces fork-exec on macOS. A gRPC server's poll
    state then leaks into the worker and aborts it before evaluation. ``os.posix_spawnp`` creates
    the new process image directly. This helper deliberately supports only the remote worker's
    binary PIPE setup and a new session; it never falls back to the unsafe fork path.
    """
    preserve = frozenset(kwargs.pop("preserve_env", ()))
    env = scrubbed_env(kwargs.pop("env", None), keep=preserve)
    cwd = kwargs.pop("cwd", None)
    stdin = kwargs.pop("stdin", None)
    stdout = kwargs.pop("stdout", None)
    stderr = kwargs.pop("stderr", None)
    if kwargs:
        raise TypeError(f"popen_group_posix does not support: {sorted(kwargs)}")
    if any(value not in (None, subprocess.PIPE) for value in (stdin, stdout, stderr)):
        raise ValueError("popen_group_posix supports only PIPE or inherited standard streams")
    if not argv or any(not isinstance(part, str) or not part or "\x00" in part for part in argv):
        raise ValueError("worker argv must be a non-empty list of non-empty strings")

    actions: list[tuple] = []
    pipes: list[int] = []
    try:
        if stdin is subprocess.PIPE:
            read_fd, write_fd = _spawn_pipe(actions, 0)
            pipes.extend((read_fd, write_fd))
            child_stdin, parent_stdin = read_fd, write_fd
        else:
            child_stdin = parent_stdin = None
        if stdout is subprocess.PIPE:
            read_fd, write_fd = _spawn_pipe(actions, 1)
            pipes.extend((read_fd, write_fd))
            child_stdout, parent_stdout = write_fd, read_fd
        else:
            child_stdout = parent_stdout = None
        if stderr is subprocess.PIPE:
            read_fd, write_fd = _spawn_pipe(actions, 2)
            pipes.extend((read_fd, write_fd))
            child_stderr, parent_stderr = write_fd, read_fd
        else:
            child_stderr = parent_stderr = None
        # All dup2 actions precede every close action. Close every unrelated parent FD,
        # including gRPC's inheritable poll pipes. A pipe endpoint at 0/1/2 is closed only
        # when it is not the target itself: DUP2(0, 0) followed by CLOSE(0) would otherwise
        # silently give the worker a closed stdin.
        target_fds = {
            0 if stdin is subprocess.PIPE else None,
            1 if stdout is subprocess.PIPE else None,
            2 if stderr is subprocess.PIPE else None,
        }
        for fd in pipes:
            if fd <= 2 and fd not in target_fds:
                actions.append((os.POSIX_SPAWN_CLOSE, fd))
        for fd in sorted(_open_fds()):
            if fd > 2:
                actions.append((os.POSIX_SPAWN_CLOSE, fd))
        if cwd is None:
            image, arguments = argv[0], list(argv)
        else:
            # posix_spawn has no portable chdir action. A tiny Python image with no gRPC imports
            # changes directory and execs the actual worker; argv remains data, not shell text.
            image = sys.executable
            arguments = [
                image, "-c",
                "import os,sys; os.chdir(sys.argv[1]); os.execvp(sys.argv[2], sys.argv[2:])",
                str(cwd), *argv,
            ]
        pid = os.posix_spawnp(image, arguments, env, file_actions=actions, setsid=True)
    except BaseException:
        for fd in pipes:
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    for fd in (child_stdin, child_stdout, child_stderr):
        if fd is not None:
            os.close(fd)
    proc = _PosixSpawnProcess(
        pid,
        list(argv),
        io.open(parent_stdin, "wb") if parent_stdin is not None else None,
        io.open(parent_stdout, "rb") if parent_stdout is not None else None,
        io.open(parent_stderr, "rb") if parent_stderr is not None else None,
    )
    _GROUP_LEADERS.add(proc.pid)
    register(proc)
    return proc


def popen_group(argv: list[str], **kwargs) -> subprocess.Popen:
    """@brief 在新会话中启动命令并登记进程组，环境已剔除敏感变量。

    Start ``argv`` in a new session and register it. ``env`` is scrubbed.
    ``preserve_env`` names exact, scoped capabilities that must survive the
    scrub; it is intentionally not a general "allow secrets" switch.

    @param argv 命令行参数列表。
    @param kwargs 其余传给 ``subprocess.Popen`` 的参数；``env`` 会被清洗，``preserve_env`` 不作为 Popen 参数。
    @return 已启动并登记的子进程。
    """
    preserve = frozenset(kwargs.pop("preserve_env", ()))
    env = scrubbed_env(kwargs.pop("env", None), keep=preserve)
    proc = subprocess.Popen(argv, start_new_session=True, env=env, **kwargs)
    _GROUP_LEADERS.add(proc.pid)
    register(proc)
    return proc
