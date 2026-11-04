"""Confirmed kernels, one directory per target, task, dtype and size."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from tcenv.target import REPO

LOG = logging.getLogger("taic.kernel_catalog")


def blob_root() -> Path:
    """@brief 内容寻址 blob 的根目录（确认写路径的持久证据，不是派生可删数据）。

    ``TAIC_BLOB_ROOT`` 可以把测试或只读自检重定向到仓库之外；observe 与 taic 共用这一份定义，
    否则两边的路径会悄悄分叉。

    @return 默认 ``data/blobs`` 或环境变量指定的目录。
    """
    override = os.environ.get("TAIC_BLOB_ROOT")
    return Path(override) if override else REPO / "data" / "blobs"


def blob_path(sha256: str) -> Path:
    """@brief 一个摘要对应的 blob 路径：``<root>/<sha[:2]>/<sha>``。

    @param sha256 内核源码的十六进制摘要。
    @return blob 文件路径（不保证存在）。
    """
    return blob_root() / sha256[:2] / sha256


def write_blob(source: str, sha256: str | None = None) -> Path | None:
    """@brief 把确认过的源码写成内容寻址 blob；已存在则跳过，失败只记一行日志。

    捕获点必须在确认写路径上（调用方手里已有源码字符串），不能等到周期对账：确认写盘到观察进程
    下一次 tick 之间，run 目录可能已经被删，而这是唯一副本。写 blob **不增加 I/O**，也**不允许**
    让确认失败——与 publish 的 never-raises 是同一条纪律。

    @param source 已确认的内核源码。
    @param sha256 源码摘要；``None`` 时现算（调用方通常已经有）。
    @return blob 路径；写失败时返回 None。
    """
    import hashlib

    from . import control

    digest = sha256 or hashlib.sha256(source.encode("utf-8")).hexdigest()
    try:
        path = blob_path(digest)
        if path.is_file():
            return path
        control.write_text_atomic(path, source)
        return path
    except OSError as exc:
        # 半截文件由 write_text_atomic 的 finally 清掉；正式名字只可能是完整内容。
        LOG.warning("blob write failed for %s: %s", digest, exc)
        return None


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
    task_fingerprint_hex: str | None = None,
    task_mode: str = "v1",
    track: dict | None = None,
    candidate_sha256: str | None = None,
    transport_sha256: str | None = None,
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
    @param track 运行记录里的 ``track`` 块（id/version/delivery），bundle 与 single 的读者靠它分辨交付机制。
    @param candidate_sha256 候选身份（bundle 是 manifest digest，single 与 transport 相同）；缺省按 ``source`` 推算。
    @param transport_sha256 提交文本的原始字节摘要；缺省等于 ``source`` 的 UTF-8 摘要（旧 blob 复用不受影响）。
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
        "task_mode": task_mode,
        # v2 由调用方给合同指纹（v2 目录没有 v1 装配，不能再 get_task 一次）；v1 维持原计算。
        "task_fingerprint": (
            task_fingerprint_hex if task_fingerprint_hex is not None
            else task_fingerprint(get_task(task, dtype, size), backend)
        ),
        "calibration_sha256": sha256_file(calib_path) if calib_path.is_file() else "",
        "run_id": run_id,
        "winner_label": label,
        # ``source_sha256`` stays the transport spelling the blob index is keyed by, so an existing
        # blob keeps its address.  ``transport_sha256`` and ``candidate_sha256`` make the two
        # identities explicit (they differ for a bundle, coincide for a single file).
        "source_sha256": transport_sha256 or hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "transport_sha256": transport_sha256 or hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "candidate_sha256": candidate_sha256 or hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "track": dict(track) if track else None,
        "confirmed_speedup": speedup,
        "confirmed_values": list(confirmed_values or []),
        "candidate_us": candidate_us,
        "baseline_us": baseline_us or {},
        "updated_at": time.time(),
    }
    # §9.1：blob 与目录史同一处写；失败吞掉，确认照常成功（索引侧会补登记已有的 blob）。
    write_blob(source, record["source_sha256"])
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


def list_kernels(target_id: str | None = None, *, limit: int | None = None, cursor: str | None = None) -> list[dict]:
    """@brief 列出已确认内核的 ``meta.json`` 内容，并补上内核文件路径。

    目录史会随发布增长，所以按 ``path`` 排序后支持 keyset 分页：``cursor`` 传上一页最后一行的
    ``path``，``limit`` 给本页行数。两参数都不给时返回全部（旧行为）。

    @param target_id 只看这个目标；``None`` 时扫描所有目标。
    @param limit 本页最多返回多少行；``None`` 表示不限。
    @param cursor 上一页最后一行的 ``path``；只返回排序在它之后的行。
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
    rows.sort(key=lambda row: str(row.get("path") or ""))
    if cursor:
        rows = [row for row in rows if str(row.get("path") or "") > str(cursor)]
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    return rows
