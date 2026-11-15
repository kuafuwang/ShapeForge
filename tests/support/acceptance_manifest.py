"""Read-only SHA-256 manifest parsing for closed acceptance packages.

The path rules reproduce the 2026-11-14 baseline: package-local files win,
missing ``docs/架构/`` paths are looked up at their migrated ``golden/shared``
location, and a literal ``../../../../`` prefix is stripped before resolving
from the repository root.  This module only reads bytes; it never imports or
executes an acceptance verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_SHARED = Path("tests/support/golden/shared")
OLD_DOCS_PREFIX = "docs/架构/"
FOUR_PARENT_PREFIX = "../../../../"
ROOT_PATH_PREFIXES = {
    "taic", "tcenv", "tests", "docs", "tasks", "experiments", "backends",
    "targets", "scripts", "observe", "kernels",
}


@dataclass(frozen=True)
class ManifestEntry:
    """One parsed manifest row and its read-only path classification."""

    manifest: Path
    package: Path
    name: str
    expected_sha256: str
    target: Path
    resolution: str

    @property
    def actual_sha256(self) -> str | None:
        if not self.target.is_file():
            return None
        return hashlib.sha256(self.target.read_bytes()).hexdigest()

    @property
    def matches(self) -> bool:
        return self.actual_sha256 == self.expected_sha256

    @property
    def inside_package(self) -> bool:
        return self.resolution == "package_internal"


def is_archived_fixture(path: Path) -> bool:
    """Byte copies under acceptance-epochs are evidence, not live packages."""
    return "acceptance-epochs" in path.parts


def manifest_paths(root: Path = ROOT) -> list[Path]:
    """Return all acceptance manifests without executing package code."""
    shared = [
        path for path in (root / GOLDEN_SHARED).glob("**/manifest.sha256")
        if not is_archived_fixture(path)
    ]
    return sorted(shared, key=lambda path: str(path.relative_to(root))) + sorted(
        (root / "docs/架构").glob("**/manifest.sha256"),
        key=lambda path: str(path.relative_to(root)),
    )


def resolve_manifest_target(manifest: Path, name: str, root: Path = ROOT) -> tuple[Path, str]:
    """Resolve one historical manifest path using the frozen baseline rules."""
    package = manifest.parent.resolve()
    direct = (package / name).resolve()
    if name.startswith(FOUR_PARENT_PREFIX):
        return root / name[len(FOUR_PARENT_PREFIX):], "legacy_four_parent"
    root_candidate = root / name
    if name.startswith(OLD_DOCS_PREFIX) and not root_candidate.exists():
        relocated = root / GOLDEN_SHARED / name[len(OLD_DOCS_PREFIX):]
        return relocated, "relocated_docs"
    if direct.exists() and direct.is_relative_to(package):
        return direct, "package_internal"
    if not direct.is_relative_to(package) and name.startswith("../"):
        resolution = "same_tree" if not name.startswith("../../") else "other_relative"
        return direct, resolution
    if direct.is_relative_to(package) and Path(name).parts[0] not in ROOT_PATH_PREFIXES:
        # Deleting package-local evidence must not turn it into an external
        # repo-root row (or select an unrelated root file with the same name).
        return direct, "package_internal"
    return root_candidate, "repo_root"


def read_manifest(manifest: Path, root: Path = ROOT) -> Iterator[ManifestEntry]:
    """Yield non-comment ``<sha256>  <path>`` rows from one manifest."""
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, name = line.split("  ", 1)
        target, resolution = resolve_manifest_target(manifest, name, root)
        yield ManifestEntry(
            manifest=manifest,
            package=manifest.parent,
            name=name,
            expected_sha256=expected,
            target=target,
            resolution=resolution,
        )


def scan_manifests(root: Path = ROOT) -> list[ManifestEntry]:
    """Read every acceptance manifest row; hashing remains an explicit caller choice."""
    return [entry for manifest in manifest_paths(root) for entry in read_manifest(manifest, root)]
