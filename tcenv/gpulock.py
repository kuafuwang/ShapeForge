"""Machine-wide GPU lock, so that no measurement overlaps other GPU work started by this project.

Timing (candidate benchmarks, baselines, triton-msl, calibration) holds the lock exclusively; other GPU work
(compilation, correctness tests, shader validation) holds it shared, so it can overlap other untimed work but
never a measurement. The lock is ``flock(2)`` on one file: the kernel releases it when the holder exits,
including after SIGKILL by the supervisor.

Re-entrant within a process: inside an exclusive hold every request is a no-op; inside a shared hold a shared
request is a no-op and an exclusive request upgrades for the duration of the block, then returns to shared.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

DEFAULT_LOCK_FILE = Path.home() / ".cache" / "ai_lowerling" / "gpu.lock"
DEFAULT_MAX_WAIT_S = 1800.0

_fd: int | None = None
_mode: str | None = None  # "sh" | "ex" while this process holds the lock
_what: str = ""
_wait_hooks: tuple[Callable[[], None], Callable[[], None]] | None = None


class GpuLockTimeout(RuntimeError):
    pass


def lock_file() -> Path:
    return Path(os.environ.get("TCENV_GPU_LOCK_FILE") or DEFAULT_LOCK_FILE)


def set_wait_hooks(on_wait: Callable[[], None], on_acquired: Callable[[], None]) -> None:
    """``on_wait`` runs when the lock is found busy, ``on_acquired`` once it was obtained after waiting."""
    global _wait_hooks
    _wait_hooks = (on_wait, on_acquired)


def holder() -> str:
    """Description written by the current exclusive holder (empty while only shared holders exist)."""
    try:
        return lock_file().read_text().strip()
    except OSError:
        return ""


def _gpu_note(kind: str, mode: str, what: str) -> None:
    try:
        from observe.publish import note_gpu

        note_gpu(kind, mode, what)
    except Exception:
        return


def _acquire(fd: int, flag: int, max_wait_s: float, log: Callable[[str], None] | None, what: str = "") -> float:
    t0 = time.monotonic()
    waiting = False
    while True:
        try:
            fcntl.flock(fd, flag | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if not waiting:
                waiting = True
                _gpu_note("wait", "ex" if flag == fcntl.LOCK_EX else "sh", what)
                if log:
                    log(f"waiting for the GPU lock ({holder() or 'held by untimed GPU work'}; {lock_file()})")
                if _wait_hooks:
                    _wait_hooks[0]()
            if time.monotonic() - t0 > max_wait_s:
                _gpu_note("give_up", "ex" if flag == fcntl.LOCK_EX else "sh", what)
                raise GpuLockTimeout(f"the GPU lock was still busy after {max_wait_s:.0f}s ({holder() or 'shared holders'}; {lock_file()})")
            time.sleep(0.05)
    if waiting and _wait_hooks:
        _wait_hooks[1]()
    return time.monotonic() - t0


@contextmanager
def released() -> Iterator[None]:
    """Give up this process's hold (shared or exclusive) for the block and take the same kind again afterwards.

    A process that waits for a worker subprocess does not use the GPU; holding the lock meanwhile would deadlock with a worker
    that needs it (a shared holder blocks the worker's exclusive benchmark, an exclusive holder blocks any worker)."""
    global _fd, _mode
    mode, fd = _mode, _fd
    if mode is None or fd is None:
        yield
        return
    if mode == "ex":
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


@contextmanager
def gpu_lock(what: str, *, exclusive: bool = True, max_wait_s: float = DEFAULT_MAX_WAIT_S, log: Callable[[str], None] | None = None) -> Iterator[float]:
    """Hold the GPU lock for the block; yields the seconds spent waiting for it."""
    global _fd, _mode, _what
    want = "ex" if exclusive else "sh"
    outer = _mode
    outer_what = _what
    if outer == "ex" or outer == want:
        yield 0.0
        return
    if _fd is None:
        path = lock_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        _fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    if outer == "sh":
        # Upgrades release first: two shared holders that both wait for exclusive while still holding shared would deadlock.
        fcntl.flock(_fd, fcntl.LOCK_UN)
        _mode = None
        _gpu_note("release", "sh", outer_what)
    try:
        waited = _acquire(_fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH, max_wait_s, log, what)
    except BaseException:
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
    if exclusive:
        os.ftruncate(_fd, 0)
        os.pwrite(_fd, f"pid {os.getpid()}: {what}\n".encode(), 0)
    try:
        yield waited
    finally:
        if exclusive:
            os.ftruncate(_fd, 0)
        if outer == "sh":
            fcntl.flock(_fd, fcntl.LOCK_UN)
            _acquire(_fd, fcntl.LOCK_SH, max_wait_s, log, outer_what)
            _mode = "sh"
            _what = outer_what
            _gpu_note("acquire", "sh", outer_what)
        else:
            _gpu_note("release", want, what)
            os.close(_fd)  # closing the descriptor releases the lock
            _fd, _mode, _what = None, None, ""
