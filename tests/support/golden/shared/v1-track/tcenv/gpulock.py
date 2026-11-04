"""Machine-wide GPU lock, so that no measurement overlaps other GPU work started by this project.

Timing (candidate benchmarks, baselines, triton-msl, calibration) holds the lock exclusively; other GPU work
(compilation, correctness tests, shader validation) holds it shared, so it can overlap other untimed work but
never a measurement. The lock is ``flock(2)`` on one file: the kernel releases it when the holder exits,
including after SIGKILL by the supervisor.

Re-entrant within a process: inside an exclusive hold every request is a no-op; inside a shared hold a shared
request is a no-op and an exclusive request upgrades for the duration of the block, then returns to shared.
"""

# 机器级 flock，协调本仓库发起的图形芯片工作，避免计时被别的任务打乱。
# 计时（候选、MLX / PyTorch 基线、triton-msl、带宽标定、Instruments 录制）取独占，测量窗口里没有另一份本仓库的芯片任务。
# 编译、正确性、着色器校验取共享，彼此可以重叠，同时仍会挡住独占计时。
# 等锁时 set_wait_hooks 让 worker 把心跳写成 W；evaluate.supervise 把 W 的时长从评测总超时里扣除，排队不算进时限。
# 锁文件默认在用户缓存目录，TCENV_GPU_LOCK_FILE 可以改到别的路径。

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

# 缺省路径。真正生效的是 lock_file()，环境变量可以把它换掉。
DEFAULT_LOCK_FILE = Path.home() / ".cache" / "ai_lowerling" / "gpu.lock"
DEFAULT_MAX_WAIT_S = 1800.0

_fd: int | None = None
_mode: str | None = None  # "sh" | "ex" while this process holds the lock
_what: str = ""
_writable: bool = True  # False when the lock file could only be opened read-only; flock still works, the holder text does not
_wait_hooks: tuple[Callable[[], None], Callable[[], None]] | None = None


# 等到调用方给的上限仍拿不到锁。这是锁自己的放弃时限，和评测总超时分开计。
class GpuLockTimeout(RuntimeError):
    """@brief 在调用方给定的等待上限内仍未拿到 GPU 锁。"""

    pass


# 读 TCENV_GPU_LOCK_FILE；变量未设置时用上面的默认文件。空字符串同样回落到默认路径。
def lock_file() -> Path:
    """@brief 取当前生效的锁文件路径。

    @return ``TCENV_GPU_LOCK_FILE`` 指定的路径；未设置或为空时用默认缓存路径。
    """
    return Path(os.environ.get("TCENV_GPU_LOCK_FILE") or DEFAULT_LOCK_FILE)


# 等锁开始时调用 on_wait，真的等到之后调用 on_acquired。worker 用它们写心跳 W 和 E。
# supervise 看到 W 就把这段时间从评测总超时里扣掉；单次启动超时看的是心跳 B，等锁写的是 W。
def set_wait_hooks(on_wait: Callable[[], None], on_acquired: Callable[[], None]) -> None:
    """@brief 安装等锁前后的回调。

    ``on_wait`` runs when the lock is found busy, ``on_acquired`` once it was obtained after waiting.

    @param on_wait 发现锁忙时调用的回调。
    @param on_acquired 等待后真正拿到锁时调用的回调。
    """
    global _wait_hooks
    _wait_hooks = (on_wait, on_acquired)


# 读锁文件里的说明。独占持有者才写它；只有共享持有者时文件是空的，等待日志会落到 untimed GPU work 那句。
def holder() -> str:
    """@brief 读取当前独占持有者写入的说明。

    Description written by the current exclusive holder (empty while only shared holders exist).

    @return 持有者说明文本；读不到时为空串。
    """
    try:
        return lock_file().read_text().strip()
    except OSError:
        return ""


# 把加锁事件送到观察页。观察写失败时直接返回，加锁本身继续。
def _gpu_note(kind: str, mode: str, what: str) -> None:
    """@brief 把一次加锁事件送到观察页，失败不影响加锁。

    @param kind 事件类型，如 wait / acquire / release / give_up。
    @param mode 锁模式，``"ex"`` 或 ``"sh"``。
    @param what 本次加锁用途的说明。
    """
    try:
        from tcenv.events import note_gpu

        note_gpu(kind, mode, what)
    except Exception:
        return


# 非阻塞轮询，才能在等待中打日志、触发 on_wait，并执行锁自己的放弃时限。返回值是排队秒数。
def _acquire(fd: int, flag: int, max_wait_s: float, log: Callable[[str], None] | None, what: str = "") -> float:
    """@brief 非阻塞轮询地等待 flock，直到拿到锁或超时。

    @param fd 锁文件描述符。
    @param flag ``fcntl.LOCK_EX`` 或 ``fcntl.LOCK_SH``。
    @param max_wait_s 本次排队的最长秒数。
    @param log 可选的日志函数，首次发现锁忙时调用一次。
    @param what 本次加锁用途的说明。
    @return 排队花费的秒数。
    @raises GpuLockTimeout 超过 ``max_wait_s`` 仍未拿到锁时抛出。
    """
    t0 = time.monotonic()
    waiting = False
    while True:
        try:
            fcntl.flock(fd, flag | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            # 只在第一次发现忙时记 wait。每 50ms 都打一条会把日志和心跳刷满。
            if not waiting:
                waiting = True
                _gpu_note("wait", "ex" if flag == fcntl.LOCK_EX else "sh", what)
                if log:
                    log(f"waiting for the GPU lock ({holder() or 'held by untimed GPU work'}; {lock_file()})")
                if _wait_hooks:
                    _wait_hooks[0]()
            # 锁自己的放弃时限。评测总超时是否把排队算进去，由心跳 W 在 supervise 里决定。
            if time.monotonic() - t0 > max_wait_s:
                _gpu_note("give_up", "ex" if flag == fcntl.LOCK_EX else "sh", what)
                raise GpuLockTimeout(f"the GPU lock was still busy after {max_wait_s:.0f}s ({holder() or 'shared holders'}; {lock_file()})")
            time.sleep(0.05)
    if waiting and _wait_hooks:
        _wait_hooks[1]()
    return time.monotonic() - t0


# 父进程干等子进程时调用。继续握着锁会死锁：共享挡住子进程的独占计时，独占挡住子进程的任何芯片工作。
@contextmanager
def released() -> Iterator[None]:
    """@brief 在代码块期间放开本进程的锁，结束后按原模式重新取回。

    Give up this process's hold (shared or exclusive) for the block and take the same kind again afterwards.

    A process that waits for a worker subprocess does not use the GPU; holding the lock meanwhile would deadlock with a worker
    that needs it (a shared holder blocks the worker's exclusive benchmark, an exclusive holder blocks any worker).

    @return 上下文管理器，无产出值。
    """
    global _fd, _mode
    mode, fd = _mode, _fd
    if mode is None or fd is None:
        # 本进程当前没握锁，例如父进程还没进入 gpu_lock。
        yield
        return
    if mode == "ex" and _writable:
        # 先清掉独占说明，松开后别人不会读到已经失效的持有者。只读打开的锁写不了说明。
        os.ftruncate(fd, 0)
    fcntl.flock(fd, fcntl.LOCK_UN)
    _mode = None
    _gpu_note("release", mode, _what)
    try:
        yield
    finally:
        _acquire(fd, fcntl.LOCK_EX if mode == "ex" else fcntl.LOCK_SH, DEFAULT_MAX_WAIT_S, None, _what)
        _mode = mode
        _gpu_note("acquire", mode, _what)


# 计时用默认独占；编译、正确性、着色器校验传入 exclusive=False 取共享。
# 产出值是这次排队的秒数。计时在拿到锁之后才开始，排队不进核的耗时。评测把它记进 benchmark 的 gpu_lock_wait_s。
@contextmanager
def gpu_lock(what: str, *, exclusive: bool = True, max_wait_s: float = DEFAULT_MAX_WAIT_S, log: Callable[[str], None] | None = None) -> Iterator[float]:
    """@brief 在代码块期间持有机器级 GPU 锁。

    Hold the GPU lock for the block; yields the seconds spent waiting for it.

    @param what 本次加锁用途的说明，会写进锁文件与观察事件。
    @param exclusive True 取独占锁（计时），False 取共享锁（不计时工作）。
    @param max_wait_s 排队的最长秒数。
    @param log 可选的日志函数。
    @return 上下文管理器，产出本次排队秒数。
    @raises GpuLockTimeout 超时仍未拿到锁时抛出。
    """
    global _fd, _mode, _what, _writable
    want = "ex" if exclusive else "sh"
    outer = _mode
    outer_what = _what
    if outer == "ex" or outer == want:
        # 已经独占，或已经处于同一种模式：重入不再排队，等锁时间记 0。
        yield 0.0
        return
    if _fd is None:
        path = lock_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            _fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
            _writable = True
        except OSError:
            # 编程代理的沙箱能读这台机器的家目录，不能写锁文件。flock 不需要写权限。
            # 搜索进程会先把锁文件建出来，所以这里只打开已经存在的文件。
            _fd = os.open(path, os.O_RDONLY)
            _writable = False
    if outer == "sh":
        # 先放开共享再抢独占。两个仍握着共享的升级者会互相等死。
        # Upgrades release first: two shared holders that both wait for exclusive while still holding shared would deadlock.
        fcntl.flock(_fd, fcntl.LOCK_UN)
        _mode = None
        _gpu_note("release", "sh", outer_what)
    try:
        waited = _acquire(_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH, max_wait_s, log, what)
    except BaseException:
        # 没抢到：外层原来没有锁就关掉 fd；原来是共享就把共享拿回来，避免把锁漏在半开状态。
        if outer is None:
            os.close(_fd)
            _fd = None
            _what = ""
        else:
            _acquire(_fd, fcntl.LOCK_SH, max_wait_s, log, outer_what)
            _mode = "sh"
            _what = outer_what
            _gpu_note("acquire", "sh", outer_what)
        raise
    _mode = want
    _what = what
    _gpu_note("acquire", want, what)
    if exclusive and _writable:
        # 独占者把 pid 和用途写进锁文件，等待方的 holder() 才能打出是谁占着。共享持有者留空。
        # 只读打开时跳过这两步，锁本身仍然是独占的。
        os.ftruncate(_fd, 0)
        os.pwrite(_fd, f"pid {os.getpid()}: {what}\n".encode(), 0)
    try:
        yield waited
    finally:
        if exclusive and _writable:
            os.ftruncate(_fd, 0)
        if outer == "sh":
            # 这段独占结束，回到外层的共享。先放开独占，并把状态改成没握锁。
            # 抢不回共享时不能仍标着独占，否则下一次 gpu_lock 会以为锁还在而直接放行。
            _gpu_note("release", want, what)
            fcntl.flock(_fd, fcntl.LOCK_UN)
            # 锁已经放开。抢不回共享时 _mode 保持空、_fd 仍开着，外层共享锁的 finally 负责关掉它。
            # 不能把 _mode 留在 "ex"，否则下一次 gpu_lock 会以为自己还握着独占锁。
            _mode = None
            _what = ""
            _acquire(_fd, fcntl.LOCK_SH, max_wait_s, log, outer_what)
            _mode = "sh"
            _what = outer_what
            _gpu_note("acquire", "sh", outer_what)
        else:
            _gpu_note("release", want, what)
            os.close(_fd)  # closing the descriptor releases the lock
            _fd, _mode, _what, _writable = None, None, "", True
