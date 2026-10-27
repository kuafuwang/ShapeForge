"""Host process argv enumeration without executing ``ps``.

Fenced or hardened hosts deny executing ``ps`` even for the process's own
children, so the container supervisor cannot rely on it to prove that a
leftover ``docker start --attach`` client is gone.  The same kernel information
is read directly: ``/proc/<pid>/cmdline`` on Linux and ``sysctl(KERN_PROCARGS2)``
plus ``proc_listpids``/``proc_pidpath`` on macOS.  Reads are bounded.

**Unknown is not "no attach client".**  A process whose argv cannot be read is
skipped only when the host *proves* it cannot be a live ``docker start --attach``
client: the kernel reports a zombie (Linux ``Z``, Darwin ``SZOMB`` from
``sysctl(KERN_PROC, KERN_PROC_PID)``), a Linux ``PF_KTHREAD`` thread (no
user-space executable at all), the owner record names another account (an attach
client is a child of this frontend and shares its euid), or ``kill(pid, 0)``
reports ``ESRCH`` (the pid is gone).  Everything else fails the scan with
:class:`HostProcsError`, so a caller never reads an unknown scan as an empty one
and never reports verified cleanup over a possibly surviving attach client.

A refusal is retried for a bounded moment before it is decided: Darwin answers
``EIO`` for a live process while its address space changes (exec, fork, exit) and
reads fine again a few milliseconds later.  The retry makes a real survivor
visible (and matched) and lets an exiting process reach a positive zombie/gone
proof; a process that stays unknown still fails the scan.

The executable name is deliberately **not** used to narrow the scan.  A
surviving client's executable is whatever the configured Docker CLI ends up
execing: a script's shebang may go through ``/usr/bin/env``, an interpreter
symlink can land on ``python3.12``, and a macOS framework launcher reports a
completely different path (``.../Python.app/Contents/MacOS/Python``).  No
offline computation proves that set complete, so "the readable path is not the
Docker CLI" is not evidence, and :func:`iter_processes` keeps
``executable_names`` only to report which executables the caller spawns.
``EPERM`` from ``kill(pid, 0)`` is likewise not proof of another owner: a
hardened or sandboxed host denies signals to same-uid processes too, so an
unknown owner stays unknown and the scan fails closed.

What keeps the normal path usable is the kernel's own state plus the owner
record: a hardened host always carries uninspectable processes (``login``,
other accounts' daemons, kernel threads, unreaped zombies), and each of them is
proven non-candidate above without reading its argv.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import sys
import time
from typing import Iterable, Iterator

_MAX_CMDLINE_BYTES = 4096
# Darwin's KERN_PROCARGS2 returns argc, the exec path, argv and then the whole
# environment.  The environment can be arbitrarily large, so only a bounded
# prefix is read; the argv fields are parsed one by one out of that prefix, so a
# prefix that ends inside the environment is fine and one that ends inside argv
# raises instead of inventing the missing argument.
_MAX_ARGS_BYTES = 1024 * 1024
_MAX_STAT_BYTES = 4096
_MAX_PROCS = 200_000
# ``/proc/<pid>/stat`` field 9: the kernel's ``task_struct.flags``.  Kernel
# threads carry ``PF_KTHREAD`` and have neither an executable nor a cmdline, so
# the kernel itself proves they cannot be a Docker CLI process.
_PF_KTHREAD = 0x00200000
_CTL_KERN = 1
_KERN_PROCARGS2 = 49
# ``sysctl({CTL_KERN, KERN_PROC, KERN_PROC_PID, pid})`` (``KERN_PROC`` is 14 on
# Darwin, unlike the Linux ``CTL_KERN``/``KERN_PROC`` pair) returns a
# ``struct kinfo_proc`` even for an unreaped zombie, which ``proc_pidinfo``
# refuses.  ``p_stat``/``p_pid`` sit after ``p_un`` (16), ``p_vmspace`` (8),
# ``p_sigacts`` (8) and ``p_flag`` (4); ``p_pid`` is self-checked so a layout
# change fails closed instead of reporting another process's state.
_KERN_PROC = 14
_KERN_PROC_PID = 1
_KINFO_PROC_BYTES = 1024
_KINFO_PROC_STAT_OFFSET = 36
_KINFO_PROC_PID_OFFSET = 40
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
        # The process may have exited between the listing and this read, but
        # that is decided from positive evidence (:func:`_no_live_record`:
        # ``kill(pid, 0)`` answering ``ESRCH``), never from this errno: a hidden
        # or sandbox-denied ``/proc`` entry looks the same as a vanished one.
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
    """Darwin per-process ``KERN_PROCARGS2`` failure.

    No errno is special-cased here.  ``ESRCH`` looks like "gone", but the same
    answer is what a sandboxed or hardened kernel can give for a live process;
    :func:`_no_live_record` re-proves gone/zombie/owner independently.
    """

    raise _Unreadable(f"inspection failed with errno {ctypes.get_errno()}")


def _darwin_argv(pid: int) -> tuple[str, ...]:
    """``pid``'s argv from a bounded ``KERN_PROCARGS2`` prefix.

    The record ends with the whole environment, so a large one is normal and the
    bounded read may cut it.  Every argv entry is therefore parsed up to its own
    NUL terminator: counting ``split`` elements instead would turn the record's
    final stop (or the capped prefix's last NUL) into a phantom empty argument
    and could hide a missing trailing entry - and the last entry of
    ``docker start --attach --interactive <id>`` is the container id.
    """

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
    rest = raw[4:]
    terminator = rest.find(b"\x00")
    if terminator < 0:
        raise _Unreadable("its executable path is not terminated within the bounded read")
    rest = rest[terminator + 1:]
    # The kernel may pad with NULs before argv (after the exec path); they are
    # not arguments.  The padding is indistinguishable from an empty argv[0],
    # which no real CLI has.
    while rest.startswith(b"\x00"):
        rest = rest[1:]
    argv: list[str] = []
    for _index in range(argc):
        terminator = rest.find(b"\x00")
        if terminator < 0:
            # The bounded prefix ended before this entry's terminator: the argv
            # is not proven complete, so it is unknown (fail closed), never
            # shorter-and-therefore-not-matching.
            raise _Unreadable(
                "its argument vector is not complete within the bounded read limit"
            )
        argv.append(rest[:terminator].decode("utf-8", errors="replace"))
        rest = rest[terminator + 1:]
    if not argv:
        raise _Unreadable("its argument vector is empty")
    return tuple(argv)


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


def _darwin_pidinfo_raw(pid: int, kind: int, size: int) -> bytes | None:
    """Raw ``proc_pidinfo`` record, or ``None`` when the probe answered nothing.

    ``None`` means *unknown*: EPERM/EIO, a short record and a missing task all
    look like this, so the caller must never turn it into "the process is
    gone".  Only :func:`_darwin_proc_state` can prove a zombie, and only
    ``kill(pid, 0)`` can prove the pid disappeared.
    """

    try:
        _libc, libproc = _darwin_libraries()
        if not hasattr(libproc, "proc_pidinfo"):
            return None
        libproc.proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int,
        ]
        libproc.proc_pidinfo.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(size)
        written = int(libproc.proc_pidinfo(pid, kind, 0, buffer, size))
    except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError):
        return None
    if written <= 0 or written > size:
        return None
    return buffer.raw[:written]


def _darwin_owner(pid: int) -> int | None:
    """Owner uid of ``pid`` from the bsd-info records (``None`` when refused)."""

    for kind, structure in (
        (_PROC_PIDTBSDINFO, _ProcBsdInfo),
        (_PROC_PIDT_SHORTBSDINFO, _ProcBsdShortInfo),
    ):
        size = ctypes.sizeof(structure)
        raw = _darwin_pidinfo_raw(pid, kind, size)
        if raw is None or len(raw) < size:
            continue
        info = structure.from_buffer_copy(raw)
        return int(info.pbi_uid if kind == _PROC_PIDTBSDINFO else info.pbsi_uid)
    return None


def _darwin_proc_state(pid: int) -> tuple[bool, str | None, str]:
    """``(found, state, detail)`` from ``sysctl(KERN_PROC, KERN_PROC_PID)``.

    This is the one Darwin probe that describes an unreaped zombie (``SZOMB``)
    instead of refusing it, so a zombie can be skipped on positive evidence.  A
    zero-length answer is *no record*, not proof the pid is gone (``kernel_task``
    answers that way while it runs); a refused or short probe is unknown, and its
    state is not guessed from ``proc_pidinfo`` failures.  The layout is
    self-checked against ``p_pid``, so a kernel that moves ``p_stat`` makes the
    scan fail closed instead of reporting some other process's state.
    """

    try:
        libc, _libproc = _darwin_libraries()
        mib = (ctypes.c_int * 4)(_CTL_KERN, _KERN_PROC, _KERN_PROC_PID, pid)
        buffer = ctypes.create_string_buffer(_KINFO_PROC_BYTES)
        sized = ctypes.c_size_t(_KINFO_PROC_BYTES)
        ctypes.set_errno(0)
        result = int(libc.sysctl(mib, 4, buffer, ctypes.byref(sized), None, 0))
    except (AttributeError, OSError, TypeError, ValueError, ctypes.ArgumentError) as exc:
        return False, None, f"KERN_PROC_PID is unavailable ({exc})"
    if result != 0:
        return False, None, f"KERN_PROC_PID failed with errno {ctypes.get_errno()}"
    raw = buffer.raw[: sized.value]
    if not raw:
        return False, None, "KERN_PROC_PID has no record"
    if len(raw) < _KINFO_PROC_PID_OFFSET + 4:
        return False, None, "KERN_PROC_PID returned a short record"
    reported = int.from_bytes(
        raw[_KINFO_PROC_PID_OFFSET:_KINFO_PROC_PID_OFFSET + 4], sys.byteorder, signed=True,
    )
    if reported != pid:
        return False, None, "KERN_PROC_PID returned another process"
    return True, _MACOS_STATES.get(raw[_KINFO_PROC_STAT_OFFSET]), ""


def _darwin_owner_and_state(pid: int) -> tuple[int | None, str | None]:
    """Owner uid and kernel state of ``pid`` when argv cannot be read.

    The owner comes from the bsd-info records (the short one answers for other
    accounts' processes); the state comes from ``KERN_PROC_PID`` only.
    """

    found, state, _detail = _darwin_proc_state(pid)
    return _darwin_owner(pid), (state if found else None)



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
    """Whether the host *proves* ``pid`` is not a live attach client.

    Only positive evidence counts: the kernel state is a zombie (Linux ``Z``,
    Darwin ``SZOMB`` from ``KERN_PROC_PID``), the process is a Linux
    ``PF_KTHREAD`` thread, the owner record names another account (an attach
    client is a child of this frontend and shares its euid), or ``kill(pid, 0)``
    answers ``ESRCH``.  Nothing else does: ``EPERM`` is what a hardened or
    sandboxed host answers for same-uid processes too, and a failed
    ``proc_pidinfo``/``KERN_PROCARGS2``/``/proc`` read says nothing about the
    owner or the state.  Those cases stay unknown and the caller fails closed.
    """

    owner, state = _owner_and_state(pid)
    if state == "Z":
        # A zombie holds no descriptor and cannot be attaching to a container.
        return True
    if _kernel_thread(pid):
        # ``PF_KTHREAD``: no user-space executable and therefore no argv.
        return True
    if owner is not None and owner != os.geteuid():
        # Attach clients are children of this frontend, so they share its
        # effective uid; another account's process can never be one.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        # ``PermissionError`` included: a sandbox can deny signals to our own
        # children, so it is not proof of another owner or of a vanished pid.
        return False
    return False


# A live process can refuse ``KERN_PROCARGS2``/``/proc/<pid>/cmdline`` for a few
# milliseconds while its address space changes (exec, fork, exit on Darwin
# answers ``EIO`` and then reads fine again).  A refusal is therefore not yet a
# decision: re-read it while the answer is still unknown, so a real survivor is
# matched and an exiting one is skipped only once the kernel proves it.
_SCAN_SETTLE_ATTEMPTS = 7
_SCAN_SETTLE_S = 0.03


# -- classification ---------------------------------------------------------


def _inspect(pid: int) -> tuple[tuple[str, ...] | None, str]:
    """``(argv, detail)`` for ``pid``, settling a transient refusal.

    ``argv`` is the process's argument vector, ``()`` when the host proves the
    process cannot be a live attach client (:func:`_no_live_record`: zombie,
    kernel thread, another account, or ``ESRCH``), and ``None`` when it is still
    unknown after the bounded settle.  A refusal is retried rather than treated
    as a decision, so a live client that was momentarily unreadable is read (and
    matched) instead of being dropped, while an exiting process is only skipped
    after the kernel reports it as a zombie or gone.
    """

    detail = "the kernel refused the argument read"
    for attempt in range(_SCAN_SETTLE_ATTEMPTS):
        try:
            return _read_argv(pid), ""
        except _Unreadable as exc:
            detail = str(exc)
        if _no_live_record(pid):
            return (), ""
        if attempt + 1 < _SCAN_SETTLE_ATTEMPTS:
            time.sleep(_SCAN_SETTLE_S)
    return None, detail


def _inspection_error(pid: int, detail: str, executable_names: frozenset[str]) -> str:
    """Error text for a live process that is still unknown after settling."""

    owner, _state = _owner_and_state(pid)
    path = _executable_path(pid)
    hint = f"; this caller spawns {sorted(executable_names)}" if executable_names else ""
    return (
        f"could not inspect live process {pid} "
        f"({path or 'executable unknown'}, "
        f"uid {owner if owner is not None else 'unknown'}): {detail}{hint}"
    )


def _yield_inspectable(
    pids: Iterable[int], executable_names: frozenset[str] = frozenset(),
) -> Iterator[tuple[int, tuple[str, ...]]]:
    for pid in pids:
        if pid <= 0 or pid == os.getpid():
            continue
        argv, detail = _inspect(pid)
        if argv is None:
            # Unknown is not "no survivor": never narrow by executable name
            # (an interpreter or framework launcher has an arbitrary path) and
            # never treat ``EPERM`` as an owner proof.
            raise HostProcsError(_inspection_error(pid, detail, executable_names))
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

    An uninspectable process is skipped only when the host *proves* it cannot be
    a live attach client (zombie, kernel thread, another account, or ``ESRCH``);
    otherwise the scan fails with :class:`HostProcsError`.
    ``executable_names`` is **not** used to narrow the scan - a client's
    executable is whatever the configured CLI execs (script, ``env``,
    interpreter symlink, framework launcher), so that set cannot be proven
    complete.  It is accepted only so the failure can report which executables
    this caller spawns.
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
    unreadable live one - including one whose owner cannot be read and whose
    ``kill(pid, 0)`` answer is ``EPERM`` - raises :class:`HostProcsError` so the
    caller fails closed instead of skipping the signal.
    """

    wanted = [needle for needle in needles if needle]
    if not wanted or pid <= 0:
        return False
    argv, detail = _inspect(pid)
    if argv is None:
        raise HostProcsError(
            f"could not read the arguments of live process {pid}: {detail}"
        )
    return bool(argv) and all(any(needle == argument for argument in argv) for needle in wanted)


def pids_mentioning(
    *needles: str, executable_names: frozenset[str] = frozenset(),
) -> list[int]:
    """PIDs whose argv contains every needle as a full argument.

    Matching on whole arguments keeps an identifier (a 64-character container
    id) from matching inside an unrelated path or environment string.  Raises
    :class:`HostProcsError` when the process table cannot be read *or* when a
    process that could be a surviving attach client cannot be inspected, so
    callers never mistake an unknown scan for an empty one.
    """

    wanted = [needle for needle in needles if needle]
    if not wanted:
        return []
    matches: list[int] = []
    for pid, argv in iter_processes(executable_names):
        if all(any(needle == argument for argument in argv) for needle in wanted):
            matches.append(pid)
    return matches
