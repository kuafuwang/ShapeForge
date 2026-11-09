"""Verify the independent B-evidence observation/restoration epoch (read-only)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPECTED_STATUS = "closed_evidence_restored_and_raw_observations_preserved"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load() -> dict:
    return json.loads((HERE / "resolution.json").read_text(encoding="utf-8"))


def verify_resolution(document: dict) -> None:
    if document.get("status") != EXPECTED_STATUS:
        raise ValueError("the B-evidence resolution is not complete")
    if document.get("raw_observations_preserved") is not True or document.get("closed_evidence_restored") is not True:
        raise ValueError("both preservation and restoration must be true")
    if document.get("source_incident_record_modified") is not False:
        raise ValueError("the original incident record was modified")
    rows = document.get("files") or []
    if len(rows) != 9 or len({row["path"] for row in rows}) != 9:
        raise ValueError("exactly nine distinct B-evidence files must be resolved")
    for row in rows:
        relative = row["path"]
        raw = ROOT / row["raw_copy"]
        archived = ROOT / relative
        if not raw.is_file() or raw.is_symlink() or raw.stat().st_nlink != 1:
            raise ValueError(f"missing or aliased raw observation: {raw}")
        if sha(raw) != row["observed_sha256"]:
            raise ValueError(f"raw observation digest mismatch: {relative}")
        if sha(archived) != row["head_sha256"] or sha(archived) != row["restored_sha256"]:
            raise ValueError(f"closed evidence was not restored: {relative}")
        if json.loads(archived.read_text())["median_us"] != row["head_median_us"]:
            raise ValueError(f"restored median_us mismatch: {relative}")
        if json.loads(raw.read_text())["median_us"] != row["observed_median_us"]:
            raise ValueError(f"preserved median_us mismatch: {relative}")
    for version in range(1, 5):
        source = (ROOT / f"tests/backends/metal/test_task_migration_b{version}_gpu.py").read_text(encoding="utf-8")
        if "write_gpu_observation" not in source or "evidence.mkdir" in source:
            raise ValueError(f"B{version} still overwrites archived evidence")


def self_test(document: dict) -> None:
    changed = copy.deepcopy(document)
    changed["files"][0]["restored_sha256"] = "0" * 64
    try:
        verify_resolution(changed)
    except ValueError:
        pass
    else:
        raise AssertionError("a modified restored digest was accepted")


def verify_manifest() -> None:
    expected_files = {"README.md", "resolution.json", "verify.py", "manifest.sha256"}
    for line in (HERE / "manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        if sha(HERE / relative) != expected:
            raise ValueError(f"manifest mismatch: {relative}")
        expected_files.add(relative)
    raw_files = {str(path.relative_to(HERE)) for path in (HERE / "raw").rglob("gpu-*.json")}
    if len(raw_files) != 9 or raw_files != expected_files - {"README.md", "resolution.json", "verify.py", "manifest.sha256"}:
        raise ValueError("the raw evidence set must contain exactly nine files")


def main() -> int:
    document = load()
    verify_resolution(document)
    self_test(document)
    verify_manifest()
    print(json.dumps({
        "done": True,
        "scope": "independent B-evidence observation preservation and closed-archive restoration only",
        "files": len(document["files"]),
        "status": EXPECTED_STATUS,
        "gpu_or_model_executed": False,
        "performance_publication": False,
        "manifest": True,
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
