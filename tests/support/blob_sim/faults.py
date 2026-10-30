"""Fault-injecting blob IO: real I/O errors, crashes, and buffer accounting.

The production store talks to :class:`~tcenv.remote.blob_channel.BlobIO`.  This
subclass lets a test force a specific filesystem operation to fail or to crash
the process exactly once, and records every read/write size so a test can prove
the channel never asked for a whole file at once.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Iterable

from tcenv.remote.blob_channel import LocalBlobIO


class FakeCrash(RuntimeError):
    """A simulated process crash at a specific I/O boundary."""


class FaultIO(LocalBlobIO):
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.ops: list[tuple[str, str]] = []
        self.read_sizes: list[int] = []
        self.write_sizes: list[int] = []
        self.written: dict[str, bytes] = {}
        self.write_log: list[tuple[str, int]] = []
        self.read_log: list[tuple[str, int]] = []
        self.crashes: list[str] = []
        self._failures: list[dict[str, object]] = []

    # -- programming ------------------------------------------------------
    def fail(
        self, operation: str, *, times: int = 1, match: str = "",
        error: Callable[[], BaseException] | None = None,
    ) -> None:
        """Make the next ``times`` matching ``operation`` calls raise."""

        self._failures.append({
            "operation": operation, "remaining": times, "match": match,
            "error": error or (lambda: OSError("injected I/O failure")),
            "crash": False,
        })

    def crash(self, operation: str, *, times: int = 1, match: str = "") -> None:
        """Simulate a crash: raise and leave whatever was already on disk."""

        self._failures.append({
            "operation": operation, "remaining": times, "match": match,
            "error": lambda: FakeCrash(f"injected crash during {operation}"),
            "crash": True,
        })

    def clear_faults(self) -> None:
        with self.lock:
            self._failures.clear()

    def _maybe_fail(self, operation: str, path: Path) -> None:
        with self.lock:
            for rule in self._failures:
                if rule["remaining"] <= 0 or rule["operation"] != operation:
                    continue
                if rule["match"] and str(rule["match"]) not in str(path):
                    continue
                rule["remaining"] = int(rule["remaining"]) - 1
                if rule["crash"]:
                    self.crashes.append(f"{operation}:{path.name}")
                raise rule["error"]()  # type: ignore[operator]

    # -- instrumented operations -----------------------------------------
    def read(self, path: Path, limit: int) -> bytes:
        self._maybe_fail("read", path)
        data = super().read(path, limit)
        with self.lock:
            self.ops.append(("read", path.name))
            self.read_sizes.append(len(data))
            self.read_log.append((path.name, len(data)))
        return data

    def read_opened(self, path: Path, descriptor: int, limit: int) -> bytes:
        self._maybe_fail("read", path)
        data = super().read_opened(path, descriptor, limit)
        with self.lock:
            self.ops.append(("read", path.name))
            self.read_sizes.append(len(data))
            self.read_log.append((path.name, len(data)))
        return data

    def unlink_at(self, parent_fd: int, name: str, path: Path) -> None:
        self._maybe_fail("remove", path)
        super().unlink_at(parent_fd, name, path)
        with self.lock:
            self.ops.append(("remove", path.name))

    def write_atomic(self, path: Path, data: bytes) -> None:
        self._maybe_fail("write_atomic", path)
        super().write_atomic(path, data)
        with self.lock:
            self.ops.append(("write_atomic", path.name))
            self.write_sizes.append(len(data))
            self.write_log.append((path.name, len(data)))
            self.written[path.name] = bytes(data)

    def remove(self, path: Path) -> None:
        self._maybe_fail("remove", path)
        super().remove(path)
        with self.lock:
            self.ops.append(("remove", path.name))

    def remove_tree(self, path: Path) -> None:
        self._maybe_fail("remove_tree", path)
        super().remove_tree(path)
        with self.lock:
            self.ops.append(("remove_tree", path.name))

    def listdir(self, path: Path) -> list[str]:
        self._maybe_fail("listdir", path)
        return super().listdir(path)

    # -- assertions helpers ----------------------------------------------
    def operations(self, operation: str | None = None) -> list[tuple[str, str]]:
        with self.lock:
            return [
                entry for entry in self.ops
                if operation is None or entry[0] == operation
            ]

    def chunk_files(self, root: Path) -> list[Path]:
        return sorted(path for path in root.rglob("*.chunk") if path.is_file())
