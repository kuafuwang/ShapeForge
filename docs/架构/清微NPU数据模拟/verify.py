#!/usr/bin/env python3
"""Verify the archived offline Qingmicro NPU data/protocol simulation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.sha256"
DELIVERABLES = ("README.md", "simulate.py", "verify.py", "report.json")
FORBIDDEN_MEASUREMENT_KEYS = {
    "candidate_us",
    "confirmed_speedup",
    "speedup",
    "device_elapsed_ns",
    "host_api_elapsed_ns",
}


def _check_manifest() -> None:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(DELIVERABLES)
    seen = []
    for line in lines:
        digest, name = line.split("  ", 1)
        assert name in DELIVERABLES and name not in seen
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == digest
        seen.append(name)
    assert set(seen) == set(DELIVERABLES)
    print(f"OK: manifest.sha256 ({len(seen)} files)")


def _walk(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key, item
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _check_report_boundary() -> None:
    report = json.loads((HERE / "report.json").read_text(encoding="utf-8"))
    assert report["schema_id"] == "local.qingmicro-npu-data-simulation.1"
    assert report["simulated"] is True
    assert report["publication_allowed"] is False
    assert report["npu_p0"]["status"] == "blocked"
    assert report["npu_p0"]["real_device_available"] is False
    assert report["npu_p0"]["correctness_on_device_proven"] is False
    assert report["npu_p0"]["performance_on_device_proven"] is False
    assert report["remote"]["measurement_kind"] == "simulated"
    assert report["remote"]["publication_allowed"] is False
    assert report["remote"]["timing_numbers_included"] is False
    assert report["data_manifest"]["task"]["op_name"] == "sigmoid"
    assert report["remote"]["fixture_task"] == "rope"
    assert report["bundle"]["one_byte_tamper_rejected"] is True
    assert not FORBIDDEN_MEASUREMENT_KEYS.intersection(key for key, _ in _walk(report))
    print("OK: simulated-only boundary; no device correctness/timing/speedup/publication claim")


def main() -> None:
    _check_manifest()
    _check_report_boundary()
    subprocess.run([sys.executable, str(HERE / "simulate.py"), "--replay"], check=True)
    print("OK: 清微 NPU 离线数据与协议模拟验收通过（NPU-P0 仍 blocked）")


if __name__ == "__main__":
    main()
