"""Read-only task pack manifest/digest/fingerprint recomputation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcenv.remote import contracts
from tcenv.remote.task_pack import TaskPackError, load_task_pack


FINGERPRINT = "a" * 64


def _pack(root: Path, *, fingerprint: str = FINGERPRINT, extra: bytes = b"reference") -> Path:
    (root / "sources").mkdir(parents=True)
    (root / "sources" / "relu.msl").write_bytes(b"kernel void relu() {}")
    (root / "sources" / "reference.bin").write_bytes(extra)
    (root / "pack.json").write_text(json.dumps({
        "schema_version": 1,
        "backend": "metal",
        "target_id": "apple-m4-max",
        "reference_versions": {"naive": "v1", "baseline": "v1"},
    }), encoding="utf-8")
    (root / "fingerprints.json").write_text(json.dumps({
        "schema_version": 1,
        "tasks": [{"task": "relu", "dtype": "fp16", "size": "small", "fingerprint": fingerprint}],
    }), encoding="utf-8")
    return root


def test_task_pack_recomputes_manifest_digest_and_fingerprints(tmp_path):
    root = _pack(tmp_path / "pack")
    pack = load_task_pack(root)
    assert pack.backend == "metal"
    assert pack.target_id == "apple-m4-max"
    assert pack.fingerprints == {FINGERPRINT}
    assert pack.reference_versions["naive"] == "v1"
    paths = [entry["path"] for entry in pack.manifest["files"]]
    assert paths == sorted(paths)
    assert "fingerprints.json" in paths and "pack.json" in paths
    # Recomputing with the expected digest and matching provider is stable.
    again = load_task_pack(root, expected_digest=pack.digest, fingerprint_provider=lambda _entry: FINGERPRINT)
    assert again.digest == pack.digest


def test_task_pack_detects_byte_and_fingerprint_drift(tmp_path):
    root = _pack(tmp_path / "pack")
    pack = load_task_pack(root)
    with pytest.raises(TaskPackError, match="expected digest"):
        load_task_pack(root, expected_digest="f" * 64)
    with pytest.raises(TaskPackError, match="task registry"):
        load_task_pack(root, fingerprint_provider=lambda _entry: "b" * 64)
    (root / "sources" / "reference.bin").write_bytes(b"tampered")
    with pytest.raises(TaskPackError, match="expected digest"):
        load_task_pack(root, expected_digest=pack.digest)


def test_task_pack_rejects_symlinks_precomputed_manifest_and_bad_shapes(tmp_path):
    root = _pack(tmp_path / "pack")
    (root / "sources" / "link").symlink_to(root / "sources" / "relu.msl")
    with pytest.raises(TaskPackError, match="symlink"):
        load_task_pack(root)

    clean = _pack(tmp_path / "clean")
    (clean / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(TaskPackError, match="precomputed manifest"):
        load_task_pack(clean)

    bad = _pack(tmp_path / "bad")
    (bad / "fingerprints.json").write_text(json.dumps({"schema_version": 1, "tasks": [
        {"task": "relu", "dtype": "fp16", "size": "small", "fingerprint": "not-hex"},
    ]}), encoding="utf-8")
    with pytest.raises(TaskPackError, match="64 lowercase hex"):
        load_task_pack(bad)

    missing = _pack(tmp_path / "missing")
    (missing / "fingerprints.json").write_text(json.dumps({"schema_version": 1, "tasks": []}), encoding="utf-8")
    with pytest.raises(TaskPackError, match="at least one"):
        load_task_pack(missing)


def test_task_pack_digest_is_canonical_across_roots(tmp_path):
    first = load_task_pack(_pack(tmp_path / "one"))
    second = load_task_pack(_pack(tmp_path / "two"))
    assert first.digest == second.digest
    assert first.manifest == second.manifest
