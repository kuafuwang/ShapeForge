"""Confirmed kernels, one directory per target, task, dtype and size."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tcenv.target import REPO


def catalog_dir(target_id: str, task: str, dtype: str, size: str) -> Path:
    return REPO / "kernels" / target_id / task / dtype / size


def archive_confirmed(
    *,
    target_id: str,
    backend: str,
    task: str,
    dtype: str,
    size: str,
    source: str,
    speedup: float,
    run_id: str,
    label: str,
    device_name: str = "",
    gpu_family: str = "",
    file_ext: str = ".metal",
    confirmed_values: list | None = None,
    candidate_us: float | None = None,
    baseline_us: dict | None = None,
) -> Path:
    """Write the confirmed kernel. A higher speedup replaces ``best``. Every confirmation is appended to history."""
    import hashlib

    from tcenv.target import calibration_file, sha256_file
    from tcenv.task import task_fingerprint
    from tasks import get_task

    folder = catalog_dir(target_id, task, dtype, size)
    folder.mkdir(parents=True, exist_ok=True)
    meta_path = folder / "meta.json"
    previous = 0.0
    if meta_path.is_file():
        try:
            previous = float(json.loads(meta_path.read_text(encoding="utf-8")).get("confirmed_speedup") or 0)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            previous = 0.0
    calib_path, _source = calibration_file(target_id)
    record = {
        "target_id": target_id,
        "backend": backend,
        "device_name": device_name,
        "gpu_family": gpu_family,
        "task": task,
        "dtype": dtype,
        "size": size,
        "task_fingerprint": task_fingerprint(get_task(task, dtype, size)),
        "calibration_sha256": sha256_file(calib_path) if calib_path.is_file() else "",
        "run_id": run_id,
        "winner_label": label,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "confirmed_speedup": speedup,
        "confirmed_values": list(confirmed_values or []),
        "candidate_us": candidate_us,
        "baseline_us": baseline_us or {},
        "updated_at": time.time(),
    }
    with (folder / "history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    best = folder / f"best{file_ext}"
    if not best.is_file() or speedup >= previous:
        best.write_text(source)
        meta_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return best


def list_kernels(target_id: str | None = None) -> list[dict]:
    root = REPO / "kernels"
    rows = []
    if not root.is_dir():
        return rows
    targets = [root / target_id] if target_id else [path for path in root.iterdir() if path.is_dir()]
    for base in targets:
        if not base.is_dir():
            continue
        for meta in base.glob("*/*/*/meta.json"):
            try:
                doc = json.loads(meta.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(doc, dict):
                doc["path"] = str(meta.parent / "best.metal")
                rows.append(doc)
    return rows
