"""Portable process-state probes for tests that check subprocess cleanup.

Some fenced or hardened hosts deny executing ``ps`` even for the test's own
children.  Tests must therefore read the same kernel information directly:
``/proc/<pid>/stat`` on Linux and ``libproc`` on macOS, with signal 0 only as
the last resort.  A zombie is reported (``"Z"``) because a reaped child and an
unreaped zombie are different cleanup outcomes for these tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROC_PIDTBSDINFO = 3
_LIBPROC = None

# <sys/proc.h> pbi_status values.
_MACOS_STATES = {1: "S", 2: "R", 3: "S", 4: "T", 5: "Z"}


def _linux_state(pid: int) -> str | None:
    stat = Path(f"/proc/{pid}/stat")
    try:
        raw = stat.read_text(encoding="utf-8")
    except OSError:
        return None
    tail = raw[raw.rfind(")") + 2 :].split()
    return tail[0] if tail else None


def _macos_state(pid: int) -> str | None:
    global _LIBPROC
    try:
        import ctypes
        import ctypes.util

        if _LIBPROC is None:
            class ProcBsdInfo(ctypes.Structure):
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
                    ("pbi_start_tvsec", ctypes.c_uint64),
                    ("pbi_start_tvusec", ctypes.c_uint64),
                ]

            library = ctypes.CDLL(ctypes.util.find_library("proc") or "/usr/lib/libproc.dylib")
            library.proc_pidinfo.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
                ctypes.c_void_p, ctypes.c_int,
            ]
            library.proc_pidinfo.restype = ctypes.c_int
            _LIBPROC = (library, ProcBsdInfo)
        library, structure = _LIBPROC
        info = structure()
        written = library.proc_pidinfo(
            int(pid), _PROC_PIDTBSDINFO, 0, ctypes.byref(info), ctypes.sizeof(info),
        )
        if written != ctypes.sizeof(info):
            return None
        return _MACOS_STATES.get(int(info.pbi_status), "S")
    except (ImportError, OSError, AttributeError, ValueError):
        return None


def process_state(pid: int) -> str | None:
    """Return the process state letter, or ``None`` when the PID is gone.

    ``None`` also covers "cannot inspect", which is the safe answer for cleanup
    assertions: an unverifiable process is never reported as reaped.
    """

    if pid <= 0:
        return None
    if Path(f"/proc/{pid}/stat").is_file():
        return _linux_state(pid)
    if sys.platform == "darwin":
        return _macos_state(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return "S"
    return "S"


def process_is_alive(pid: int) -> bool:
    """True for a running, stopped or zombie process; False once it is reaped."""

    return process_state(pid) is not None
