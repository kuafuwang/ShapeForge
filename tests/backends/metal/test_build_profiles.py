"""The diagnostic build-profile registry is one JSON file, not Python data (A2, CPU).

A2 moved the registered diagnostic profiles to ``backends/metal/build_profiles.json`` and deleted
the Python literal, so probes, calibration and the runtime's internal fill kernel read one
authoritative, schema-checked document.  No device is opened here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tcenv.backends.metal.source_builder import (
    BUILD_PROFILES_PATH,
    DIAGNOSTIC_BUILD_PROFILES,
    load_build_profiles,
)

ROOT = Path(__file__).resolve().parents[3]


def test_the_registered_profiles_come_from_the_json_document():
    """@brief 运行期读取的 profile 与 JSON 文档逐项一致。"""
    document = json.loads(BUILD_PROFILES_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert set(document["profiles"]) == {
        "metal.internal-fill",
        "metal.raw-source-calibration",
        "metal.raw-source-probe",
    }
    for name, entry in document["profiles"].items():
        assert DIAGNOSTIC_BUILD_PROFILES[name] == {"assemble": entry["assemble"], "math": entry["math"]}
        assert entry["purpose"].strip()


def test_the_python_profile_table_is_gone():
    """@brief source_builder.py 里不再有 profile 名/设置的字面量副本（单一权威）。"""
    source = (ROOT / "tcenv" / "backends" / "metal" / "source_builder.py").read_text(encoding="utf-8")
    for name in ("metal.raw-source-probe", "metal.raw-source-calibration", "metal.internal-fill"):
        assert name not in source, f"the profile {name} is still hard-coded in Python"
    assert '"raw_source": "safe"' not in source
    assert BUILD_PROFILES_PATH.name == "build_profiles.json"
    assert BUILD_PROFILES_PATH.parent.name == "metal"
    assert BUILD_PROFILES_PATH.parent.parent.name == "backends"


def test_the_profile_schema_is_closed_and_refuses_unknown_values(tmp_path):
    """@brief 未知 schema/字段/取值一律拒绝，profile 不能夹带构建选项。"""
    def write(document: dict) -> Path:
        path = tmp_path / "build_profiles.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def base() -> dict:
        return {
            "schema_version": 1,
            "profiles": {"metal.probe": {"assemble": "raw_source", "math": "safe", "purpose": "probe"}},
        }

    assert load_build_profiles(write(base())) == {"metal.probe": {"assemble": "raw_source", "math": "safe"}}

    for mutate, message in (
        (lambda d: d.update(schema_version=2), "schema_version"),
        (lambda d: d.update(extra=True), "exactly schema_version and profiles"),
        (lambda d: d["profiles"]["metal.probe"].update(math="turbo"), "math mode"),
        (lambda d: d["profiles"]["metal.probe"].update(assemble="fake_header"), "assemble mode"),
        (lambda d: d["profiles"]["metal.probe"].update(tile=[1, 2]), "exactly"),
        (lambda d: d["profiles"]["metal.probe"].update(purpose=""), "purpose"),
        (lambda d: d["profiles"].update({"metal.empty": {"assemble": "raw_source", "math": "safe"}}), "purpose"),
        (lambda d: d.update(profiles={}), "at least one profile"),
    ):
        document = base()
        mutate(document)
        with pytest.raises(ValueError, match=message):
            load_build_profiles(write(document))
    with pytest.raises(ValueError, match="cannot read"):
        load_build_profiles(tmp_path / "missing.json")
