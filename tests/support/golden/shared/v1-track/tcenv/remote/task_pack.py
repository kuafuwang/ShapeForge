"""Read-only task pack manifest, digest and fingerprint verification.

A deployed evaluator trusts a versioned task pack.  Before serving any job it
recomputes a canonical manifest over the actual bytes, checks the declared
``task_pack_digest`` and verifies every declared task fingerprint with the
deployed task registry.  A mismatch is an evaluator/deployment failure, never a
candidate error.

This module uses only the standard library at import time.  The production
fingerprint provider imports the task registry lazily; test compositions inject
their own provider.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from . import contracts

SCHEMA_VERSION = 1
MAX_FILES = 4096
MAX_TOTAL_BYTES = 64 * 1024 * 1024
MAX_FINGERPRINTS = 4096
_FINGERPRINT_FILE = "fingerprints.json"
_MANIFEST_FILE = "manifest.json"
_HEX64 = re.compile(r"[0-9a-f]{64}")


class TaskPackError(ValueError):
    """The deployed read-only task pack cannot be trusted."""


@dataclass(frozen=True)
class TaskPack:
    root: Path
    manifest: dict[str, Any]
    digest: str
    fingerprints: frozenset[str]
    fingerprint_bindings: dict[str, tuple[str, str, str]]
    tasks: tuple[dict[str, str], ...]
    target_id: str
    backend: str
    reference_versions: dict[str, str]


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise TaskPackError("task pack file escapes its root") from exc
    text = relative.as_posix()
    if not text or text.startswith("/") or ".." in relative.parts:
        raise TaskPackError("task pack paths must be normalized relative POSIX paths")
    return text


def _walk_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in list(dirnames):
            candidate = base / name
            if candidate.is_symlink():
                raise TaskPackError("task pack must not contain symlinked directories")
        for name in filenames:
            path = base / name
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise TaskPackError("task pack must not contain symlinks")
            if not stat.S_ISREG(info.st_mode):
                raise TaskPackError("task pack may contain only regular files")
            files.append(path)
    return files


def _read_json(path: Path, name: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskPackError(f"cannot read {name}: {exc}") from exc


def _fingerprint_entries(document: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(document, Mapping) or set(document) != {"schema_version", "tasks"}:
        raise TaskPackError("fingerprints.json must contain schema_version and tasks")
    if document["schema_version"] != SCHEMA_VERSION or not isinstance(document["tasks"], list):
        raise TaskPackError("fingerprints.json has an unsupported schema")
    if len(document["tasks"]) > MAX_FINGERPRINTS:
        raise TaskPackError("fingerprints.json has too many entries")
    entries: list[dict[str, str]] = []
    for item in document["tasks"]:
        if not isinstance(item, Mapping) or set(item) != {"task", "dtype", "size", "fingerprint"}:
            raise TaskPackError("fingerprint entries must contain task/dtype/size/fingerprint")
        for key in ("task", "dtype", "size"):
            if not isinstance(item[key], str) or not item[key]:
                raise TaskPackError(f"fingerprint entry {key} must be non-empty text")
        fingerprint = item["fingerprint"]
        if not isinstance(fingerprint, str) or not _HEX64.fullmatch(fingerprint):
            raise TaskPackError("declared task fingerprint must be 64 lowercase hex characters")
        entries.append({key: str(item[key]) for key in ("task", "dtype", "size", "fingerprint")})
    if not entries:
        raise TaskPackError("task pack must declare at least one task fingerprint")
    return tuple(entries)


def local_pack_digest(pack_files, *, root: Path | str) -> str:
    """@brief 本地 task-pack 文件集的 manifest 摘要（与部署侧同一文档形状）。

    The local repository has no deployed pack directory, so the *registered* pack file set
    (from the track binding) is hashed with the same manifest document
    ``{schema_version, files:[{path,size,sha256}]}`` and the same canonical encoder that
    :func:`load_task_pack` uses.  Missing, duplicated or symlinked files and paths that are
    not repository-relative fail closed: a skipped file would silently change the identity.

    @param pack_files 仓库相对 POSIX 路径序列。
    @param root 解析这些路径的根目录。
    @return 十六进制 sha256。
    @raises TaskPackError 文件集非法。
    """
    root = Path(os.path.abspath(Path(root)))
    if root.is_symlink() or not root.is_dir():
        raise TaskPackError("local task pack root must be a real directory")
    values: list[str] = []
    folded: dict[str, str] = {}
    for raw in pack_files:
        if not isinstance(raw, str) or not raw or os.path.isabs(raw) or "\\" in raw:
            raise TaskPackError(f"local pack file {raw!r} must be a relative POSIX path")
        normalized = str(PurePosixPath(raw))
        if normalized != raw or normalized in (".", "..") or normalized.startswith("../"):
            raise TaskPackError(f"local pack file {raw!r} must be a normalized relative path")
        key = raw.casefold()
        if key in folded:
            raise TaskPackError(f"local pack file {raw!r} collides with {folded[key]!r}")
        folded[key] = raw
        values.append(raw)
    if not values:
        raise TaskPackError("the local task pack file set is empty")
    if len(values) > MAX_FILES:
        raise TaskPackError("local task pack has too many files")
    entries: list[dict[str, Any]] = []
    total = 0
    for relative in sorted(values):
        absolute = root / relative
        if absolute.is_symlink() or not absolute.is_file():
            raise TaskPackError(f"local pack file {relative!r} must be an existing regular file")
        data = absolute.read_bytes()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise TaskPackError("local task pack exceeds its byte budget")
        entries.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    manifest = {"schema_version": SCHEMA_VERSION, "files": entries}
    return hashlib.sha256(contracts.canonical_digest_document(manifest)).hexdigest()


def load_task_pack(
    root: Path | str,
    *,
    expected_digest: str = "",
    fingerprint_provider: Callable[[Mapping[str, str]], str] | None = None,
) -> TaskPack:
    """Recompute the pack manifest and verify digest plus fingerprints.

    ``expected_digest`` empty means the caller accepts the computed digest (the
    deployment still records it).  When ``fingerprint_provider`` is supplied,
    every declared fingerprint must equal the provider's recomputation.
    """

    if not isinstance(root, (str, os.PathLike)):
        raise TaskPackError("task pack root must be a path")
    root = Path(os.path.abspath(Path(root)))
    if not root.is_dir():
        raise TaskPackError("task pack root is not a directory")
    if root.is_symlink():
        raise TaskPackError("task pack root cannot be a symlink")
    if expected_digest and (not isinstance(expected_digest, str) or not _HEX64.fullmatch(expected_digest)):
        raise TaskPackError("expected task pack digest must be 64 lowercase hex characters")

    fingerprint_path = root / _FINGERPRINT_FILE
    tasks = _fingerprint_entries(_read_json(fingerprint_path, _FINGERPRINT_FILE))

    files = _walk_files(root)
    if len(files) > MAX_FILES:
        raise TaskPackError("task pack has too many files")
    entries: list[dict[str, Any]] = []
    total = 0
    for path in sorted(files, key=lambda item: _relative_posix(item, root)):
        relative = _relative_posix(path, root)
        if relative == _MANIFEST_FILE:
            raise TaskPackError("task pack must not ship a precomputed manifest.json")
        data = path.read_bytes()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise TaskPackError("task pack exceeds its byte budget")
        entries.append({
            "path": relative,
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": entries,
    }
    encoded = contracts.canonical_digest_document(manifest)
    digest = hashlib.sha256(encoded).hexdigest()
    if expected_digest and digest != expected_digest:
        raise TaskPackError("deployed task pack digest differs from the expected digest")

    fingerprints = {entry["fingerprint"] for entry in tasks}
    bindings: dict[str, tuple[str, str, str]] = {}
    bound_triples: dict[tuple[str, str, str], str] = {}
    for entry in tasks:
        fingerprint = entry["fingerprint"]
        triple = (entry["task"], entry["dtype"], entry["size"])
        if fingerprint in bindings and bindings[fingerprint] != triple:
            raise TaskPackError("the same fingerprint is bound to two different tasks")
        if triple in bound_triples and bound_triples[triple] != fingerprint:
            raise TaskPackError("the same task/dtype/size is bound to two different fingerprints")
        bindings[fingerprint] = triple
        bound_triples[triple] = fingerprint
    if fingerprint_provider is not None:
        for entry in tasks:
            recomputed = fingerprint_provider(entry)
            if not isinstance(recomputed, str) or not _HEX64.fullmatch(recomputed):
                raise TaskPackError("fingerprint provider returned an invalid digest")
            if recomputed != entry["fingerprint"]:
                raise TaskPackError(
                    f"task fingerprint for {entry['task']}/{entry['dtype']}/{entry['size']} "
                    "does not match the deployed task registry"
                )

    deployment = _read_json(root / "pack.json", "pack.json")
    if not isinstance(deployment, Mapping) or set(deployment) != {
        "schema_version", "backend", "target_id", "reference_versions",
    }:
        raise TaskPackError("pack.json must contain schema_version/backend/target_id/reference_versions")
    if deployment["schema_version"] != SCHEMA_VERSION:
        raise TaskPackError("pack.json has an unsupported schema")
    backend = deployment["backend"]
    target_id = deployment["target_id"]
    versions = deployment["reference_versions"]
    if not isinstance(backend, str) or not backend:
        raise TaskPackError("pack backend must be non-empty text")
    if not isinstance(target_id, str) or not target_id:
        raise TaskPackError("pack target_id must be non-empty text")
    if not isinstance(versions, Mapping) or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in versions.items()
    ):
        raise TaskPackError("reference_versions must map non-empty names to versions")

    return TaskPack(
        root=root,
        manifest=manifest,
        digest=digest,
        fingerprints=frozenset(fingerprints),
        fingerprint_bindings=bindings,
        tasks=tasks,
        target_id=target_id,
        backend=backend,
        reference_versions={str(key): str(value) for key, value in sorted(versions.items())},
    )


def production_fingerprint_provider(backend: str) -> Callable[[Mapping[str, str]], str]:
    """Return a provider backed by the deployed task registry.

    Importing ``tasks`` is deliberately lazy: manifest verification must work in
    a deployment check before any device runtime is loaded.
    """

    def provider(entry: Mapping[str, str]) -> str:
        from tasks import get_task

        from tcenv.task import task_fingerprint

        task = get_task(str(entry["task"]), str(entry["dtype"]), str(entry["size"]))
        return task_fingerprint(task, backend)

    return provider
