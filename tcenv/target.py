"""Device profiles and Metal backend identity. No import of ``taic``."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TARGETS = REPO / "targets"
BACKENDS = REPO / "backends"
DEFAULT_TARGET_ID = "apple-m4-max"


class TargetError(RuntimeError):
    """The requested target cannot be used."""


@dataclass(frozen=True)
class Limits:
    max_buffer_index: int
    max_threads_per_threadgroup: int
    max_threadgroup_memory_bytes: int
    counter_sample_us: int


@dataclass(frozen=True)
class TargetProfile:
    id: str
    backend: str
    status: str
    display_name: str
    match_name_contains: str
    match_family: str
    calibration_device_name: str
    limits: Limits
    root: Path
    language: str
    file_ext: str

    @property
    def device_doc(self) -> Path:
        return self.root / "device.md"

    @property
    def experience(self) -> Path:
        return self.root / "experience.md"

    @property
    def skills(self) -> Path:
        return BACKENDS / self.backend / "skills"

    @property
    def wiki(self) -> Path:
        return BACKENDS / self.backend / "wiki"


def load_target(target_id: str | None = None) -> TargetProfile:
    """Load one profile. ``reserved`` profiles load, but :func:`require_active` refuses to run them."""
    name = target_id or DEFAULT_TARGET_ID
    path = TARGETS / name / "target.toml"
    if not path.is_file():
        raise TargetError(f"target {name} has no target.toml")
    doc = tomllib.loads(path.read_text(encoding="utf-8"))
    if str(doc.get("id") or "") != name:
        raise TargetError(f"target.toml id {doc.get('id')!r} is not the directory {name}")
    backend = str(doc.get("backend") or "")
    from .backends.registry import load_backend

    backend_obj = load_backend(backend)
    backend_doc = {
        "language": backend_obj.language,
        "file_ext": backend_obj.file_ext,
    }
    device = doc.get("device") or {}
    limits = doc.get("limits") or {}
    root = path.parent
    for asset in ("device.md", "experience.md"):
        if not (root / asset).is_file():
            raise TargetError(f"target {name} is missing {asset}")
    return TargetProfile(
        id=str(doc["id"]),
        backend=backend,
        status=str(doc.get("status") or "reserved"),
        display_name=str(doc.get("display_name") or ""),
        match_name_contains=str(device.get("match_name_contains") or ""),
        match_family=str(device.get("match_family") or ""),
        calibration_device_name=str(device.get("calibration_device_name") or ""),
        limits=Limits(
            max_buffer_index=int(limits["max_buffer_index"]),
            max_threads_per_threadgroup=int(limits["max_threads_per_threadgroup"]),
            max_threadgroup_memory_bytes=int(limits["max_threadgroup_memory_bytes"]),
            counter_sample_us=int(limits["counter_sample_us"]),
        ),
        root=root,
        language=str(backend_doc.get("language") or ""),
        file_ext=str(backend_doc.get("file_ext") or ""),
    )


def require_active(target_id: str | None = None) -> TargetProfile:
    profile = load_target(target_id)
    if profile.status != "active":
        raise TargetError(f"target {profile.id} is {profile.status}; it cannot run")
    return profile


def device_matches(profile: TargetProfile, info: dict) -> bool:
    name = str(info.get("name") or "")
    family = str(info.get("gpu_family") or "")
    return profile.match_name_contains in name and family == profile.match_family


def limit_notes(profile: TargetProfile, info: dict) -> list[str]:
    """Warn when the device reports a thread or threadgroup-memory cap the profile does not record.

    Buffer index and the counter sample interval are not in ``info()``. The profile limit stays in force.
    """
    notes = []
    live_threads = info.get("max_threads_per_threadgroup")
    live_memory = info.get("max_threadgroup_memory_bytes")
    if live_threads is not None and int(live_threads) != profile.limits.max_threads_per_threadgroup:
        notes.append(
            f"target {profile.id} records {profile.limits.max_threads_per_threadgroup} threads per threadgroup; "
            f"this device reports {int(live_threads)}. The profile limit stays in force."
        )
    if live_memory is not None and int(live_memory) != profile.limits.max_threadgroup_memory_bytes:
        notes.append(
            f"target {profile.id} records {profile.limits.max_threadgroup_memory_bytes} B of threadgroup memory; "
            f"this device reports {int(live_memory)} B. The profile limit stays in force."
        )
    return notes


def calibration_cache_path(target_id: str | None = None) -> Path:
    """Where a new measurement is written. This is outside the source tree."""
    profile = load_target(target_id)
    return Path.home() / ".cache" / "ai_lowerling" / "calibration" / f"{profile.id}.json"


def calibration_file(target_id: str | None = None) -> tuple[Path, str]:
    """``(path, source)``. ``TCENV_CALIBRATION`` wins, then the cache, then a legacy file already in the target directory.

    A measurement is not written back into the target directory. ``get_calibration`` writes the cache path.
    """
    override = os.environ.get("TCENV_CALIBRATION")
    if override:
        return Path(override), "env"
    cache = calibration_cache_path(target_id)
    if cache.is_file():
        return cache, "cache"
    legacy = load_target(target_id).root / "calibration.json"
    if legacy.is_file():
        return legacy, "target"
    return cache, "cache"


def read_calibration(target_id: str | None = None) -> tuple[dict, Path, str]:
    path, source = calibration_file(target_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TargetError(f"calibration {path} is not an object")
    return data, path, source


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
