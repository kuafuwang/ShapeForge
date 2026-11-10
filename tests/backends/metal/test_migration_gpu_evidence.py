"""B GPU evidence outputs must never overwrite a closed phase archive (CPU only)."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

from tests.support.migration_gpu_evidence import RUN_DIR_ENV, write_gpu_observation


def _record() -> dict:
    return {"task": "gemv", "status": "ok", "median_us": 2.5}


def test_default_observation_is_fresh_and_does_not_touch_closed_evidence(tmp_path):
    phase = tmp_path / "B1"
    archived = phase / "evidence" / "gpu-gemv.json"
    archived.parent.mkdir(parents=True)
    original = b'{"task":"gemv","median_us":4.0}'
    archived.write_bytes(original)
    result = write_gpu_observation(phase, "gemv", _record())
    assert result.is_file()
    assert result != archived
    assert not result.is_relative_to(archived.parent)
    assert json.loads(result.read_text()) == _record()
    assert archived.read_bytes() == original


def test_explicit_epoch_directory_is_allowed_only_outside_the_closed_package(tmp_path, monkeypatch):
    phase = tmp_path / "B2"
    fresh = tmp_path / "new-epoch"
    monkeypatch.setenv(RUN_DIR_ENV, str(fresh))
    result = write_gpu_observation(phase, "lion", _record())
    assert result == fresh / "gpu-lion.json"
    assert result.is_file()
    monkeypatch.setenv(RUN_DIR_ENV, str(phase / "evidence"))
    with pytest.raises(ValueError, match="outside the closed evidence directory"):
        write_gpu_observation(phase, "lion", _record())


def _verifier():
    path = Path(__file__).resolve().parents[3] / "tests/support/golden/shared/多DSL-Track-阶段验收/MT1b/verify.py"
    spec = importlib.util.spec_from_file_location("mt1b_b_evidence_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from tests.support.boundaries import rebind_acceptance_root

    rebind_acceptance_root(module)
    return module


def test_b_evidence_resolution_restores_archive_and_preserves_raw_observations():
    verifier = _verifier()
    verifier.verify_external_b_evidence(verifier._load("identity.json"))
    resolution = verifier._load("b-evidence-resolution.json")
    incident = json.loads((verifier.ROOT / resolution["incident"]).read_text(encoding="utf-8"))
    assert len(incident["files"]) == 9
    for row in incident["files"]:
        assert (verifier.ROOT / row["raw_copy"]).is_file()
        assert (verifier.ROOT / row["path"]).read_bytes()


def test_b_evidence_resolution_rejects_pending_or_modified_source_record():
    verifier = _verifier()
    resolution = copy.deepcopy(verifier._load("b-evidence-resolution.json"))
    resolution["status"] = "pending_outer_probe"
    with pytest.raises(ValueError, match="restore closed evidence"):
        verifier.verify_external_b_evidence(verifier._load("identity.json"), resolution_document=resolution)
    resolution = copy.deepcopy(verifier._load("b-evidence-resolution.json"))
    resolution["source_incident_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="incident record was modified"):
        verifier.verify_external_b_evidence(verifier._load("identity.json"), resolution_document=resolution)


def test_all_migration_gpu_writers_use_the_non_destructive_observation_boundary():
    for version in range(1, 5):
        source = (Path(__file__).parent / f"test_task_migration_b{version}_gpu.py").read_text(encoding="utf-8")
        assert "write_gpu_observation" in source
        assert "evidence.mkdir" not in source
        assert '(evidence / f"gpu-{task}.json").write_text' not in source
