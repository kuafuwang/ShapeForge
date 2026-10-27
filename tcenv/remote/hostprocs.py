"""Host process argv enumeration without executing ``ps``.

Fenced or hardened hosts deny executing ``ps`` even for the process's own
children, so the container supervisor cannot rely on it to prove that a
leftover ``docker start --attach`` client is gone.  The same kernel information
is read directly: ``/proc/<pid>/cmdline`` on Linux and ``sysctl(KERN_PROCARGS2)``
plus ``proc_listpids``/``proc_pidpath`` on macOS.  Reads are bounded.

**Unknown is not "no attach client".**  A process that cannot be inspected is
classified against the *candidate set* a caller can actually have spawned: an
attach client runs with this frontend's effective uid and its executable is the
configured Docker CLI (or the interpreter of its shebang).  Only a process that
is provably outside that set may be skipped: one that really disappeared, an
unreaped zombie (Darwin reports no bsd-info record and Linux ``Z``), a process
owned by another account, or one whose executable is readable and is not the
Docker CLI.  Anything else fails the scan with :class:`HostProcsError`, so a
caller never reads an unknown scan as an empty one and never reports verified
cleanup over a possibly surviving attach client.

Narrowing by executable and owner is what keeps the normal path working: a
hardened host always has processes nobody may inspect (``login``, other
accounts' daemons, kernel threads), and none of them can be a child this
frontend spawned.  When the platform cannot tell whether a process belongs to
the candidate set, the scan still fails closed with a readable error.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import sys
from typing import Iterable, Iterator

_MAX_CMDLINE_BYTES = 4096
# Darwin's KERN_PROCARGS2 returns argc, the exec path, argv and then the whole
# environment.  The environment can be arbitrarily large, so only a bounded
# prefix is read; it must still contain every argv string (fail closed if not).
_MAX_ARGS_BYTES = 1024 * 1024
_MAX_STAT_BYTES = 4096
_MAX_PROCS = 200_000
# ``/proc/<pid>/stat`` field 9: the kernel's ``task_struct.flags``.  Kernel
# threads carry ``PF_KTHREAD`` and have neither an executable nor a cmdline, so
# the kernel itself proves they cannot be a Docker CLI process.
_PF_KTHREAD = 0x00200000
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
    proof.  Only a process that is provably not a candidate may be skipped.
    """


class _Unreadable(Exception):
    """A live process whose argv the kernel refused to hand over."""


# -- argv readers ------------------------------------------------------------


def _linux_argv(pid: int) -> tuple[str, ...]:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read(_MAX_CMDLINE_BYTES + 1)
    except OSError as exc:
        # The process may have exited between the listing and this read; that is
        # the only case in which an unreadable argv is not evidence of a process
        # the scan has to account for.
        if exc.errno in {errno.ENOENT, errno.ESRCH}:
            return ()
        raise _Unreadable(
            f"could not read its command line ({exc.strerror or exc})"
        ) from exc
    if len(raw) > _MAX_CMDLINE_BYTES:
        raise _Unreadable("its command line exceeds the bounded read limit")
    argv = tuple(part.decode("utf-8", errors="replace") for part in raw.split(b"\x00") if part)
    if not argv:
        # A zombie, a kernel thread or a process that rewrote its argv.  An
        # empty cmdline is not evidence about which arguments are *absent*.
        raise _Unreadable("its command line is empty")
    return argv


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


def _darwin_unreadable() -> tuple[str, ...]:
    """Darwin per-process ``KERN_PROCARGS2`` failures.

    ``ESRCH``/``ENOENT`` prove the process is gone.  Everything else is
    *unknown*: whether it may be skipped is decided by
    :func:`_undecidable` from the executable path and the owner, never here.
    """

    error = ctypes.get_errno()
    if error in {errno.ESRCH, errno.ENOENT}:
        return ()
    raise _Unreadable(f"inspection failed with errno {error}")


def _darwin_argv(pid: int) -> tuple[str, ...]:
    libc, _libproc = _darwin_libraries()
    mib = (ctypes.c_int * 3)(_CTL_KERN, _KERN_PROCARGS2, pid)
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return _darwin_unreadable()
    if size.value <= 8:
        # The kernel succeeded for a live process but the record cannot even
        # hold argc + exec path: unknown is not "no arguments".
        raise _Unreadable("it returned a malformed argument record")
    read_size = min(size.value, _MAX_ARGS_BYTES)
    buffer = ctypes.create_string_buffer(read_size)
    sized = ctypes.c_size_t(read_size)
    if libc.sysctl(mib, 3, buffer, ctypes.byref(sized), None, 0) != 0:
        return _darwin_unreadable()
    raw = buffer.raw[: sized.value]
    if len(raw) < 4:
        raise _Unreadable("it returned a truncated argument record")
    # Layout: 4-byte argc, exec path, NUL padding, argv[0..argc-1], environment.
    argc = int.from_bytes(raw[:4], sys.byteorder)
    if argc <= 0 or argc > 4096:
        raise _Unreadable(f"it returned a malformed argument count ({argc})")
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
        raise _Unreadable("its argument vector exceeds the bounded read limit")
    if not argv:
        raise _Unreadable("its argument vector is empty")
    return argv


def _read_argv(pid: int) -> tuple[str, ...]:
    """``(pid)``'s argv, ``()`` when gone, :class:`_Unreadable` otherwise."""

    if sys.platform == "darwin":
        return _darwin_argv(pid)
    return _linux_argv(pid)


# -- executable path and ownership ------------------------------------------


def _darwin_pidpath(pid: int) -> str:
    """Executable path of ``pid``, or "" when the kernel cannot name it."""

    _libc, libproc = _darwin_libraries()
    if not hasattr(libproc, "proc_pidpath"):
        return ""
    libproc.proc_pidpath.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    libproc.proc_pidpath.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(4096)
    written = int(libproc.proc_pidpath(pid, buffer, 4096))
    if written <= 0:
        return ""
    return buffer.value.decode("utf-8", errors="replace")


def _linux_executable_path(pid: int) -> str:
    try:
        target = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        return ""
    suffix = " (deleted)"
    if target.endswith(suffix):
        target = target[: -len(suffix)]
    return target


def _executable_path(pid: int) -> str:
    """Executable path of ``pid``, or "" when it cannot be determined."""

    if sys.platform == "darwin":
        return _darwin_pidpath(pid)
    return _linux_executable_path(pid)


_PROC_PIDTBSDINFO = 3
# ``PROC_PIDT_SHORTBSDINFO``: the same owner/state/ppid record without the
# argv-adjacent fields.  Darwin answers it for processes the full record denies,
# including other accounts' processes, so it keeps the ownership proof available
# on a host where everything runs unprivileged.
_PROC_PIDT_SHORTBSDINFO = 13
_MACOS_STATES = {1: "S", 2: "R", 3: "S", 4: "T", 5: "Z"}


class _ProcBsdInfo(ctypes.Structure):
    _fields_ = [
        ("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32),
        ("pbi_xstatus", ctypes.c_uint32), ("pbi_pid", ctypes.c_uint32),
        ("pbi_ppid", ctypes.c_uint32), ("pbi_uid", ctypes.c_uint32),
        ("pbi_gid", ctypes.c_uint32), ("pbi_ruid", ctypes.c_uint32),
        ("pbi_rgid", ctypes.c_uint32), ("pbi_svuid", ctypes.c_uint32),
        ("pbi_svgid", ctypes.c_uint32), ("rfu_1", ctypes.c_uint32),
        ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
        ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32),
        ("pbi_pjobc", ctypes.c_uint32), ("e_tdev", ctypes.c_uint32),
        ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
        ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64),
    ]


class _ProcBsdShortInfo(ctypes.Structure):
    _fields_ = [
        ("pbsi_pid", ctypes.c_uint32), ("pbsi_ppid", ctypes.c_uint32),
        ("pbsi_pgid", ctypes.c_uint32), ("pbsi_status", ctypes.c_uint32),
        ("pbsi_comm", ctypes.c_char * 16), ("pbsi_flags", ctypes.c_uint32),
        ("pbsi_uid", ctypes.c_uint32), ("pbsi_gid", ctypes.c_uint32),
        ("pbsi_ruid", ctypes.c_uint32), ("pbsi_rgid", ctypes.c_uint32),
        ("pbsi_svuid", ctypes.c_uint32), ("pbsi_svgid", ctypes.c_uint32),
        ("pbsi_rfu", ctypes.c_uint32),
    ]


def _darwin_pidinfo(pid: int, kind: int, size: int) -> bytes:
    _libc, libproc = _darwin_libraries()
    if not hasattr(libproc, "proc_pidinfo"):
        return b""
    libproc.proc_pidinfo.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
    ]
    libproc.proc_pidinfo.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(size)
    written = int(libproc.proc_pidinfo(pid, kind, 0, buffer, size))
    if written <= 0:
        return b""
    return buffer.raw[:written]


def _darwin_bsdinfo(pid: int) -> tuple[int, str | None] | None:
    raw = _darwin_pidinfo(pid, _PROC_PIDTBSDINFO, ctypes.sizeof(_ProcBsdInfo))
    if len(raw) < ctypes.sizeof(_ProcBsdInfo):
        return None
    info = _ProcBsdInfo.from_buffer_copy(raw)
    return int(info.pbi_uid), _MACOS_STATES.get(int(info.pbi_status))


def _darwin_short_bsdinfo(pid: int) -> tuple[int, str | None] | None:
    raw = _darwin_pidinfo(pid, _PROC_PIDT_SHORTBSDINFO, ctypes.sizeof(_ProcBsdShortInfo))
    if len(raw) < ctypes.sizeof(_ProcBsdShortInfo):
        return None
    info = _ProcBsdShortInfo.from_buffer_copy(raw)
    return int(info.pbsi_uid), _MACOS_STATES.get(int(info.pbsi_status))


def _darwin_owner_and_state(pid: int) -> tuple[int | None, str | None]:
    """Owner uid and state of ``pid``, to prove exit/ownership when argv is unreadable."""

    return _darwin_bsdinfo(pid) or _darwin_short_bsdinfo(pid) or (None, None)



def _linux_stat(pid: int) -> tuple[str | None, int | None]:
    """State character and ``task_struct.flags`` of ``pid`` (``None`` if hidden)."""

    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            raw = handle.read(_MAX_STAT_BYTES)
    except OSError:
        return None, None
    if not raw:
        return None, None
    # ``comm`` may contain spaces and parentheses, so the state character is the
    # first field after the last ``)``; ``state ppid pgrp session tty_nr tpgid
    # flags`` follow it in that order.
    fields = raw.rsplit(b")", 1)[-1].split()
    state = fields[0].decode("ascii", errors="replace") if fields else None
    flags = int(fields[6]) if len(fields) > 6 and fields[6].isdigit() else None
    return state, flags


def _linux_owner_and_state(pid: int) -> tuple[int | None, str | None]:
    """Owner uid and state of ``pid`` from ``/proc`` (``None`` when hidden)."""

    state, _flags = _linux_stat(pid)
    try:
        owner = int(os.stat(f"/proc/{pid}").st_uid)
    except OSError:
        owner = None
    return owner, state


def _owner_and_state(pid: int) -> tuple[int | None, str | None]:
    if sys.platform == "darwin":
        return _darwin_owner_and_state(pid)
    return _linux_owner_and_state(pid)


def _kernel_thread(pid: int) -> bool:
    """Whether the kernel flags ``pid`` as a kernel thread (``PF_KTHREAD``).

    A kernel thread has an empty ``cmdline`` and no user-space executable, so no
    argv reader can ever inspect one: it is *unknowable*, not harmless.  The
    kernel's own flag is the proof that it cannot be a Docker CLI process, and
    that proof is what keeps the fail-closed rule usable on a root frontend that
    shares the host PID namespace (``--pid=host``) and therefore sees ``kthreadd``
    and friends as root-owned processes it must otherwise account for.
    """

    if sys.platform == "darwin":
        return False
    _state, flags = _linux_stat(pid)
    # An unreadable stat is *not* proof of a kernel thread: the caller keeps
    # failing closed through the owner/executable proof below.
    return flags is not None and bool(flags & _PF_KTHREAD)


def _no_live_record(pid: int) -> bool:
    """Whether the kernel holds no live bsd-info record for ``pid``.

    Darwin refuses both ``PROC_PIDTBSDINFO`` and ``PROC_PIDT_SHORTBSDINFO`` for
    a process with no task left (an unreaped zombie) or for a pid that vanished,
    while every live process answers at least one of them - including root-owned
    processes that a non-root caller may not read argv for.  A process the
    kernel cannot even describe is not executing ``docker start --attach``, so it
    may be skipped.  When the probes are unavailable this returns False and the
    caller stays fail-closed.
    """

    if sys.platform != "darwin":
        return False
    _libc, libproc = _darwin_libraries()
    if not hasattr(libproc, "proc_pidinfo"):
        return False
    return _darwin_bsdinfo(pid) is None and _darwin_short_bsdinfo(pid) is None


# -- classification ---------------------------------------------------------


def _undecidable(
    pid: int, executable_names: frozenset[str], detail: str,
) -> tuple[str, ...]:
    """Decide what an uninspectable process means for the caller's scan.

    Returns ``()`` only when the process is provably *not* a candidate attach
    client: its executable is readable and is not one this supervisor spawns, it
    is a zombie, it is owned by another account, or it is gone.  A live process
    owned by this account whose executable is the Docker CLI (or which cannot be
    named at all) could still be this frontend's ``docker start --attach``
    client, so it raises :class:`HostProcsError` instead of being skipped.
    """

    path = _executable_path(pid)
    if path and executable_names and os.path.basename(path) not in executable_names:
        # Decisive: attach clients are spawned from a fixed executable, so a
        # process running anything else cannot be one, whatever its argv is.
        return ()
    owner, state = _owner_and_state(pid)
    if state == "Z":
        # A zombie holds no descriptor and cannot be attaching to a container.
        return ()
    if _kernel_thread(pid):
        # ``PF_KTHREAD``: the kernel proves there is no user-space executable
        # (and thus no argv) that could be the Docker CLI.
        return ()
    if owner is None:
        if _no_live_record(pid):
            # The kernel has no task record for this pid: it is a zombie whose
            # parent has not reaped it, or it is already gone.
            return ()
        # Neither the path nor the owner is known.  ``kill(pid, 0)`` still
        # separates "gone" and "another account's process" from "ours".
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return ()
        except PermissionError:
            return ()
        raise HostProcsError(
            f"could not inspect live process {pid}, whose owner and executable cannot be read: {detail}"
        )
    if owner != os.geteuid():
        # Attach clients are children of this frontend, so they share its
        # effective uid; another account's process can never be one.
        return ()
    raise HostProcsError(
        "could not inspect live process "
        f"{pid} ({path or 'executable unknown'}) owned by this account, "
        f"which could still be an attach client of this frontend: {detail}"
    )


def _provably_gone(pid: int) -> bool:
    """Whether ``pid`` provably cannot be a live attach client any more."""

    _owner, state = _owner_and_state(pid)
    if state == "Z" or _no_live_record(pid):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        # Not signalable by this account: it is not a child of this frontend.
        return True
    return False


def _yield_inspectable(
    pids: Iterable[int], executable_names: frozenset[str],
) -> Iterator[tuple[int, tuple[str, ...]]]:
    for pid in pids:
        if pid <= 0 or pid == os.getpid():
            continue
        try:
            argv = _read_argv(pid)
        except _Unreadable as exc:
            argv = _undecidable(pid, executable_names, str(exc))
        if argv:
            yield pid, argv


# -- platform process tables ------------------------------------------------


def _linux_pids() -> Iterator[int]:
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
            yield int(entry.name)


def _linux_processes(
    executable_names: frozenset[str] = frozenset(),
) -> Iterator[tuple[int, tuple[str, ...]]]:
    yield from _yield_inspectable(_linux_pids(), executable_names)


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


def iter_processes(
    executable_names: frozenset[str] = frozenset(),
) -> Iterator[tuple[int, tuple[str, ...]]]:
    """Yield ``(pid, argv)`` for every process whose argv can be read.

    ``executable_names`` is the set of executable basenames a caller can spawn
    (the Docker CLI and its launcher).  A process outside that candidate set, or
    one owned by another account, is not an attach client and may be skipped
    when its argv cannot be read; an owned process that cannot be excluded fails
    the scan with :class:`HostProcsError`.  With an empty ``executable_names``
    the caller claims no candidate set at all, so every uninspectable process
    except a proven-gone/zombie/other-account one fails the scan.
    """

    if sys.platform == "darwin":
        yield from _yield_inspectable(_darwin_pids(), executable_names)
        return
    yield from _linux_processes(executable_names)


def pid_mentions(pid: int, *needles: str) -> bool:
    """Whether ``pid`` still has every needle in its argv (fresh read).

    Used immediately before signalling a survivor: a PID reused between the
    table scan and the signal must never be killed as if it were an attach
    client.  A vanished, zombied or other-account process returns False; an
    unreadable live one raises :class:`HostProcsError` so the caller fails
    closed instead of skipping the signal.
    """

    wanted = [needle for needle in needles if needle]
    if not wanted or pid <= 0:
        return False
    try:
        argv = _read_argv(pid)
    except _Unreadable as exc:
        if _provably_gone(pid):
            return False
        raise HostProcsError(
            f"could not read the arguments of live process {pid}: {exc}"
        ) from None
    return bool(argv) and all(any(needle == argument for argument in argv) for needle in wanted)


def pids_mentioning(
    *needles: str, executable_names: frozenset[str] = frozenset(),
) -> list[int]:
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
    for pid, argv in iter_processes(executable_names):
        if all(any(needle == argument for argument in argv) for needle in wanted):
            matches.append(pid)
    return matches
