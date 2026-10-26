"""Host process argv enumeration without executing ``ps``.

Fenced or hardened hosts deny executing ``ps`` even for the process's own
children, so the container supervisor cannot rely on it to prove that a
leftover ``docker start --attach`` client is gone.  The same kernel information
is read directly: ``/proc/<pid>/cmdline`` on Linux and ``sysctl(KERN_PROCARGS2)``
plus ``proc_listpids`` on macOS.  Reads are bounded.  A failure to enumerate
the process table (or to read a live process on Linux) raises
:class:`HostProcsError`: "cannot tell" is not "no attach client".  Only a
process that has really disappeared, or a Darwin process the kernel refuses to
inspect (``EINVAL``, which Darwin returns for zombies and other accounts),
may be skipped.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import sys
from typing import Iterator

_MAX_CMDLINE_BYTES = 4096
# Darwin's KERN_PROCARGS2 returns argc, the exec path, argv and then the whole
# environment.  The environment can be arbitrarily large, so only a bounded
# prefix is read; it must still contain every argv string (fail closed if not).
_MAX_ARGS_BYTES = 64 * 1024
_MAX_PROCS = 200_000
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_PROC_ALL_PIDS = 1

_LIBC = None
_LIBPROC = None


class HostProcsError(OSError):
    """The host process table could not be enumerated or read.

    Callers prove "no attach client survived" by scanning argv.  An
    enumeration or read failure is *unknown*, not an empty list: turning it
    into "nothing found" would let a surviving attach client pass the cleanup
    proof.  Only a process that has really disappeared may be skipped.
    """


def _linux_processes() -> Iterator[tuple[int, tuple[str, ...]]]:
    seen = 0
    try:
        entries = os.scandir("/proc")
    except OSError as exc:
        raise HostProcsError(f"could not enumerate host processes: {exc}") from exc
    with entries:
        for entry in entries:
            seen += 1
            if seen > _MAX_PROCS:
                raise HostProcsError("host process table is larger than the bounded scan limit")
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    raw = handle.read(_MAX_CMDLINE_BYTES + 1)
            except OSError as exc:
                # The process may have exited between listing and reading; that
                # is the only case in which an unreadable argv is not evidence
                # of an unknown survivor.
                if exc.errno in {errno.ENOENT, errno.ESRCH}:
                    continue
                raise HostProcsError(
                    f"could not read the command line of pid {pid}: {exc}"
                ) from exc
            if len(raw) > _MAX_CMDLINE_BYTES:
                # An over-long argv cannot be matched reliably; failing closed is
                # safer than silently dropping a possible attach client.
                raise HostProcsError(f"pid {pid} has an argv beyond the bounded read limit")
            argv = tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part)
            if argv:
                yield pid, argv


def _darwin_libraries():
    global _LIBC, _LIBPROC
    if _LIBC is None:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib", use_errno=True)
        # oldlenp is ``size_t *``: declaring it as c_int would let the kernel
        # write eight bytes into a four-byte object.
        libc.sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_size_t,
        ]
        libc.sysctl.restype = ctypes.c_int
        _LIBC = libc
    if _LIBPROC is None:
        library = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
        library.proc_listpids.argtypes = [
            ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_int,
        ]
        library.proc_listpids.restype = ctypes.c_int
        _LIBPROC = library
    return _LIBC, _LIBPROC


def _darwin_pids() -> list[int]:
    _libc, libproc = _darwin_libraries()
    needed = int(libproc.proc_listpids(_PROC_ALL_PIDS, 0, None, 0))
    if needed <= 0:
        raise HostProcsError(
            "could not enumerate host processes: "
            f"proc_listpids returned {needed} (errno {ctypes.get_errno()})"
        )
    if needed > _MAX_PROCS * 4:
        raise HostProcsError("host process table is larger than the bounded scan limit")
    # The table can grow between the size query and the read (normal on a busy
    # host).  A short table is retried with the larger size; a persistently
    # growing or unreadable table fails closed instead of reading as empty.
    for _attempt in range(4):
        if needed > _MAX_PROCS * 4:
            raise HostProcsError("host process table is larger than the bounded scan limit")
        buffer = ctypes.create_string_buffer(needed)
        used = int(libproc.proc_listpids(_PROC_ALL_PIDS, 0, buffer, needed))
        if used <= 0:
            raise HostProcsError(
                "could not enumerate host processes: "
                f"proc_listpids returned {used} (errno {ctypes.get_errno()})"
            )
        if used <= needed:
            raw = buffer.raw[:used]
            return [
                int.from_bytes(raw[index:index + 4], sys.byteorder)
                for index in range(0, len(raw) - 3, 4)
            ]
        needed = used
    raise HostProcsError("host process table kept growing during the bounded scan")


def _darwin_argv(pid: int) -> tuple[str, ...]:
    libc, _libproc = _darwin_libraries()
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return _darwin_unreadable(pid)
    if size.value <= 8:
        # The kernel succeeded for a live process but the record cannot even
        # hold argc + exec path: unknown is not "no arguments".
        raise HostProcsError(f"process {pid} returned a malformed argument record")
    read_size = min(size.value, _MAX_ARGS_BYTES)
    buffer = ctypes.create_string_buffer(read_size)
    sized = ctypes.c_size_t(read_size)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(sized), None, 0) != 0:
        return _darwin_unreadable(pid)
    raw = buffer.raw[: sized.value]
    if len(raw) < 4:
        raise HostProcsError(f"process {pid} returned a truncated argument record")
    # Layout: 4-byte argc, exec path, NUL padding, argv[0..argc-1], environment.
    argc = int.from_bytes(raw[:4], sys.byteorder)
    if argc <= 0 or argc > 4096:
        raise HostProcsError(f"process {pid} returned a malformed argument count ({argc})")
    # If the bounded read cut the record, drop the partial trailing string; the
    # argv completeness check below then decides whether the prefix was enough.
    truncated = sized.value >= read_size and not raw.endswith(b"\x00")
    parts = raw[4:].split(b"\x00")
    if truncated:
        parts = parts[:-1]
    # parts[0] is the exec path (its terminator was consumed by the split);
    # the kernel may then insert zero or more NUL padding bytes before argv.
    # Skipping *every* non-empty part here would swallow argv0 and argv1, so
    # only the padding NULs are skipped.
    index = 1
    while index < len(parts) and not parts[index]:
        index += 1
    argv = tuple(
        part.decode("utf-8", errors="replace") for part in parts[index:index + argc]
    )
    if len(argv) != argc:
        raise HostProcsError(
            f"process {pid} argument vector exceeds the bounded read limit"
        )
    return argv


_DARWIN_UNINSPECTABLE = frozenset({
    errno.ESRCH, errno.ENOENT, errno.EINVAL, errno.EIO,  # exited / zombie / racing exit
    errno.EPERM, errno.EACCES,  # another account: cannot be this frontend's attach client
})


def _darwin_unreadable(pid: int) -> tuple[str, ...]:
    """Darwin per-process ``KERN_PROCARGS2`` failures.

    Darwin returns ``EINVAL`` for zombie/kernel processes, ``EIO`` for a process
    that is exiting while being inspected, and ``EPERM`` for another account's
    processes; none of those can be an attach client spawned by this frontend.
    Any *other* errno is an unknown read failure and fails closed.  Enumeration
    failures are not handled here: they always raise.
    """

    error = ctypes.get_errno()
    if error in _DARWIN_UNINSPECTABLE:
        return ()
    raise HostProcsError(f"could not read the command line of pid {pid}: errno {error}")


def iter_processes() -> Iterator[tuple[int, tuple[str, ...]]]:
    """Yield ``(pid, argv)`` for every process whose argv can be read."""

    if sys.platform == "darwin":
        for pid in _darwin_pids():
            if pid <= 0 or pid == os.getpid():
                continue
            argv = _darwin_argv(pid)
            if argv:
                yield pid, argv
        return
    yield from _linux_processes()


def pids_mentioning(*needles: str) -> list[int]:
    """PIDs whose argv contains every needle as a full argument.

    Matching on whole arguments keeps an identifier (a 64-character container
    id) from matching inside an unrelated path or environment string.  Raises
    :class:`HostProcsError` when the process table cannot be read, so callers
    never mistake an unknown scan for an empty one.
    """

    wanted = [needle for needle in needles if needle]
    if not wanted:
        return []
    matches: list[int] = []
    for pid, argv in iter_processes():
        if all(any(needle == argument for argument in argv) for needle in wanted):
            matches.append(pid)
    return matches
