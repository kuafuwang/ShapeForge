"""Host process argv enumeration without executing ``ps``.

Fenced or hardened hosts deny executing ``ps`` even for the process's own
children, so the container supervisor cannot rely on it to prove that a
leftover ``docker start --attach`` client is gone.  The same kernel information
is read directly: ``/proc/<pid>/cmdline`` on Linux and ``sysctl(KERN_PROCARGS2)``
plus ``proc_listpids`` on macOS.  Reads are bounded; a process whose argv cannot
be read is simply skipped (it cannot be proven to be an attach client, and the
container itself is verified separately).
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
from typing import Iterator

_MAX_CMDLINE_BYTES = 4096
_MAX_PROCS = 200_000
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
_PROC_ALL_PIDS = 1

_LIBC = None
_LIBPROC = None


def _linux_processes() -> Iterator[tuple[int, tuple[str, ...]]]:
    seen = 0
    try:
        entries = os.scandir("/proc")
    except OSError:
        return
    with entries:
        for entry in entries:
            seen += 1
            if seen > _MAX_PROCS:
                return
            if not entry.name.isdigit():
                continue
            pid = int(entry.name)
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as handle:
                    raw = handle.read(_MAX_CMDLINE_BYTES + 1)
            except OSError:
                continue
            if len(raw) > _MAX_CMDLINE_BYTES:
                continue
            argv = tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part)
            if argv:
                yield pid, argv


def _darwin_libraries():
    global _LIBC, _LIBPROC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "/usr/lib/libSystem.B.dylib", use_errno=True)
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
    if needed <= 0 or needed > _MAX_PROCS * 4:
        return []
    buffer = ctypes.create_string_buffer(needed)
    used = int(libproc.proc_listpids(_PROC_ALL_PIDS, 0, buffer, needed))
    if used <= 0 or used > needed:
        return []
    raw = buffer.raw[:used]
    return [
        int.from_bytes(raw[index:index + 4], sys.byteorder)
        for index in range(0, len(raw) - 3, 4)
    ]


def _darwin_argv(pid: int) -> tuple[str, ...]:
    libc, _libproc = _darwin_libraries()
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_int(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return ()
    if size.value <= 8 or size.value > _MAX_CMDLINE_BYTES * 4:
        return ()
    buffer = ctypes.create_string_buffer(size.value)
    sized = ctypes.c_int(size.value)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(sized), None, 0) != 0:
        return ()
    raw = buffer.raw[: sized.value]
    # Layout: 4-byte argc, exec path, NUL padding, argv[0..argc-1], environment.
    argc = int.from_bytes(raw[:4], sys.byteorder)
    if argc <= 0 or argc > 4096:
        return ()
    parts = [part for part in raw[4:].split(b"\x00")]
    # Skip the exec path and the padding NULs that follow it.
    index = 0
    while index < len(parts) and parts[index]:
        index += 1
    while index < len(parts) and not parts[index]:
        index += 1
    argv = tuple(
        part.decode("utf-8", errors="replace") for part in parts[index:index + argc]
    )
    return argv if len(argv) == argc else ()


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
    id) from matching inside an unrelated path or environment string.
    """

    wanted = [needle for needle in needles if needle]
    if not wanted:
        return []
    matches: list[int] = []
    for pid, argv in iter_processes():
        if all(any(needle == argument for argument in argv) for needle in wanted):
            matches.append(pid)
    return matches
