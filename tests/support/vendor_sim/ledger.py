"""Cross-process fake device resource ledger.

The worker (VendorDriver) and the worker-external DeviceController are separate
processes, so cleanup proof must be based on durable state rather than an
in-memory dict.  The ledger is deliberately test-only and stores no source
bytes, host paths, credentials or arbitrary vendor commands.
"""

from __future__ import annotations

import fcntl
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping


_BINDING_KEYS = (
    "device_group_id", "vendor_id", "profile_digest", "topology_digest", "resource_generation",
)


class LedgerError(RuntimeError):
    pass


def resource_matches_binding(resource: Mapping[str, Any], binding: Mapping[str, str]) -> bool:
    """True only when a ledger row belongs to exactly this device binding."""

    return all(resource.get(key) == binding.get(key) for key in _BINDING_KEYS)


class VendorLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.path.is_file()

    @contextmanager
    def _locked(self, *, create: bool = True) -> Iterator[dict[str, Any]]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            if not create and not self.path.is_file():
                # A deleted ledger is missing evidence, not an empty device.
                # Never recreate it from a read or a release.
                raise LedgerError("fake vendor ledger is missing")
            document = self._read_unlocked()
            yield document
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(
                json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {
                "schema_version": 1, "resources": {}, "cancelled": [],
                "device_running": [], "running_scope": {}, "probes": [], "blob_reads": [],
            }
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise LedgerError(f"fake vendor ledger is unreadable: {exc}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != 1:
            raise LedgerError("fake vendor ledger schema is unsupported")
        document.setdefault("blob_reads", [])
        for key in ("resources", "cancelled", "device_running", "probes", "blob_reads"):
            if not isinstance(document.get(key), (dict, list)):
                raise LedgerError("fake vendor ledger is corrupt")
        # A running marker is only meaningful with the binding it was taken
        # under; a marker without a scope can never be released.
        document.setdefault("running_scope", {})
        if not isinstance(document["running_scope"], dict):
            raise LedgerError("fake vendor ledger is corrupt")
        return document

    def read(self) -> dict[str, Any]:
        with self._locked(create=False) as document:
            return json.loads(json.dumps(document))

    def record_resource(
        self, *, attempt_id: str, lease_fence: int, device_group_id: str,
        resource_generation: str, kind: str, detail: Mapping[str, Any] | None = None,
        vendor_id: str = "", profile_digest: str = "", topology_digest: str = "",
    ) -> str:
        if kind not in {"module", "function", "stream", "event"}:
            raise ValueError("unknown fake device resource kind")
        resource_id = f"{attempt_id}:{lease_fence}:{kind}"
        with self._locked() as document:
            document["resources"][resource_id] = {
                "attempt_id": attempt_id,
                "lease_fence": lease_fence,
                "device_group_id": device_group_id,
                "vendor_id": vendor_id,
                "profile_digest": profile_digest,
                "topology_digest": topology_digest,
                "resource_generation": resource_generation,
                "kind": kind,
                "detail": dict(detail or {}),
            }
        return resource_id

    def mark_device_running(
        self, *, attempt_id: str, lease_fence: int, binding: Mapping[str, str],
    ) -> None:
        marker = f"{attempt_id}:{lease_fence}"
        with self._locked() as document:
            if marker not in document["device_running"]:
                document["device_running"].append(marker)
            document["running_scope"][marker] = {
                key: binding.get(key) for key in _BINDING_KEYS
            }

    def running_markers(self) -> list[dict[str, Any]]:
        """Every running marker with the binding it was taken under."""

        with self._locked(create=False) as document:
            return [
                {
                    "marker": marker,
                    "binding": (
                        None if document["running_scope"].get(marker) is None
                        else dict(document["running_scope"][marker])
                    ),
                }
                for marker in document["device_running"]
            ]

    def attempt_running(self, attempt_id: str, lease_fence: int) -> list[dict[str, Any]]:
        """Return this attempt/fence's running markers with their binding."""

        marker = f"{attempt_id}:{lease_fence}"
        with self._locked(create=False) as document:
            if marker not in document["device_running"]:
                return []
            scope = document["running_scope"].get(marker)
            return [{"marker": marker, "binding": None if scope is None else dict(scope)}]

    def mark_cancelled(self, *, attempt_id: str, lease_fence: int) -> None:
        with self._locked() as document:
            marker = f"{attempt_id}:{lease_fence}"
            if marker not in document["cancelled"]:
                document["cancelled"].append(marker)

    def record_blob_read(
        self, *, attempt_id: str, lease_fence: int, content_id: str,
        chunks: int, size_bytes: int, digest: str, max_chunk_bytes: int,
        binding: Mapping[str, str],
    ) -> None:
        """Record how the worker consumed a content-addressed artifact.

        The row proves the read was chunked and verifies the exact content id, so
        a test can assert both byte identity and the absence of a whole-file
        buffer without trusting a log line.
        """

        with self._locked() as document:
            reads = document.setdefault("blob_reads", [])
            if not isinstance(reads, list):
                raise LedgerError("fake vendor ledger is corrupt")
            reads.append({
                "attempt_id": attempt_id,
                "lease_fence": lease_fence,
                "content_id": content_id,
                "chunks": chunks,
                "size_bytes": size_bytes,
                "digest": digest,
                "max_chunk_bytes": max_chunk_bytes,
                "binding": {key: binding.get(key) for key in _BINDING_KEYS},
            })

    def blob_reads(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            reads = document.get("blob_reads", [])
            if not isinstance(reads, list):
                raise LedgerError("fake vendor ledger is corrupt")
            return json.loads(json.dumps(reads))

    def record_probe(self, *, vendor_version: str, resource_generation: str) -> None:
        with self._locked() as document:
            document["probes"].append({
                "vendor_version": vendor_version,
                "resource_generation": resource_generation,
            })

    def attempt_resources(self, attempt_id: str, lease_fence: int) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return [
                resource for resource in document["resources"].values()
                if resource["attempt_id"] == attempt_id and resource["lease_fence"] == lease_fence
            ]

    def live_resources(self) -> list[dict[str, Any]]:
        with self._locked(create=False) as document:
            return list(document["resources"].values())

    def release_attempt(
        self, attempt_id: str, lease_fence: int,
        *, binding: Mapping[str, str] | None = None,
    ) -> list[str]:
        """Delete this attempt/fence's rows, never another device binding's.

        ``binding`` scopes the deletion to the exact device group/profile/
        topology/resource generation, so a controller for one generation can
        never free (or claim to have freed) another generation's resources.
        """

        released: list[str] = []
        marker = f"{attempt_id}:{lease_fence}"
        with self._locked(create=False) as document:
            for resource_id, resource in list(document["resources"].items()):
                if resource["attempt_id"] != attempt_id or resource["lease_fence"] != lease_fence:
                    continue
                if binding is not None and not resource_matches_binding(resource, binding):
                    continue
                released.append(resource_id)
                del document["resources"][resource_id]
            # The running marker is part of the same lease: only the binding it
            # was taken under (or an explicit unscoped release) may clear it.
            scope = document["running_scope"].get(marker)
            if marker in document["device_running"]:
                if binding is None or (scope is not None and all(
                    scope.get(key) == binding.get(key) for key in _BINDING_KEYS
                )):
                    document["device_running"].remove(marker)
                    document["running_scope"].pop(marker, None)
        return sorted(released)

    def clear(self, *, operator: bool = False) -> None:
        """Erase the ledger.  Test setup only, never a vendor/worker action."""

        if operator is not True:
            raise TypeError("VendorLedger.clear() is a test-setup helper; pass operator=True")
        with self._locked() as document:
            document["resources"] = {}
            document["cancelled"] = []
            document["device_running"] = []
            document["running_scope"] = {}
            document["probes"] = []

    def is_cancelled(self, attempt_id: str, lease_fence: int) -> bool:
        with self._locked() as document:
            return f"{attempt_id}:{lease_fence}" in document["cancelled"]
