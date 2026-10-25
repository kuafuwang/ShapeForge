"""Confirmed kernels, one directory per target, task, dtype and size."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tcenv.target import REPO


def catalog_root() -> Path:
    """@brief 已确认内核的根目录。

    ``TAIC_CATALOG_ROOT`` 可以把测试或金样录制重定向到仓库之外。

    @return 存放已确认内核的根目录。
    """
    import os

    override = os.environ.get("TAIC_CATALOG_ROOT")
    return Path(override) if override else REPO / "kernels"


def catalog_dir(target_id: str, task: str, dtype: str, size: str) -> Path:
    """@brief 已确认内核的存放目录：目标/任务/数据类型/尺寸。

    @param target_id 目标标识。
    @param task 任务名。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @return 该组合对应的目录路径（不保证存在）。
    """
    return catalog_root() / target_id / task / dtype / size


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
    """@brief 写入已确认的内核。

    加速比更高时会替掉 ``best``；每一条确认都会追加到 ``history.jsonl``。

    @param target_id 目标标识。
    @param backend 后端标识，如 ``metal``。
    @param task 任务名。
    @param dtype 数据类型。
    @param size 尺寸标签。
    @param source 确认过的内核源码。
    @param speedup 确认后的加速比。
    @param run_id 来源运行标识。
    @param label 胜出候选的标签。
    @param device_name 设备名。
    @param gpu_family GPU 家族。
    @param file_ext 内核文件扩展名。
    @param confirmed_values 本次确认得到的测量值列表。
    @param candidate_us 候选内核的耗时（微秒）。
    @param baseline_us 各基线耗时（微秒）。
    @return ``best`` 内核文件的路径。
    """
    import hashlib

    from tcenv.target import calibration_file, sha256_file
    from tcenv.task import task_fingerprint
    from tasks import get_task

    from . import control

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
        "task_fingerprint": task_fingerprint(get_task(task, dtype, size), backend),
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
    if not best.is_file() or speedup > previous:
        # 源码和 meta 必须一起落地：先写 best 再写 meta 时崩在中间，会让下次确认读到
        # previous=0，用一颗更慢的核覆盖掉最快的核。两个文件各自原子替换。
        meta_text = json.dumps(record, indent=2)
        control.write_text_atomic(best, source)
        control.write_text_atomic(meta_path, meta_text)
    return best


def list_kernels(target_id: str | None = None) -> list[dict]:
    """@brief 列出已确认内核的 ``meta.json`` 内容，并补上内核文件路径。

    @param target_id 只看这个目标；``None`` 时扫描所有目标。
    @return 元数据字典列表；每个多一个 ``path`` 键指向 ``best.*``。
    """
    root = catalog_root()
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
                found = sorted(meta.parent.glob("best.*"))
                doc["path"] = str(found[0] if found else meta.parent / "best.metal")
                rows.append(doc)
    return rows
