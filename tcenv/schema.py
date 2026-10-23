"""Version boundary for a worker request and an EvalReport.

Version 0 is a historical document with no ``schema_version`` key. A newer version than this process knows, or a field this version does not list, is refused. Dropping that field and still building a report is how an older parent mis-reads a newer file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPORT_SCHEMA_VERSION = 3
# Measurement protocol, separate from the file schema. Bump when the timing window,
# confirmation batches or target limits change, so aggregates can group by epoch.
PROTOCOL_VERSION = "3"
REQUEST_SCHEMA_VERSION = 4  # 4 requires target_id.

# ``eligible`` is computed from ``status`` and written for older readers. It is not a stored field.
REPORT_DERIVED = frozenset({"eligible"})
REQUEST_FIELDS = frozenset({
    "schema_version", "task", "dtype", "size", "kernel_path", "mode", "seed", "reps", "cases", "out", "heartbeat", "stage",
    "fast_bench", "require_contract_launch", "measure_serial", "target_id",
})
REQUEST_REQUIRED = frozenset({"task", "dtype", "size", "kernel_path", "mode", "seed", "reps", "out", "target_id"})


class SchemaError(ValueError):
    """@brief 本进程无法读取该文档，文件需要人工检查。

    This document cannot be read by this process. The file needs human inspection.
    """

    def __init__(self, message: str) -> None:
        """@brief 构造错误，并补上需要人工检查的标记。

        @param message 原始错误说明，末尾标点会被去掉。
        """
        text = message.rstrip(".")
        if "needs human inspection" not in text:
            text += "; needs human inspection"
        super().__init__(text)


def document_version(doc: object, *, max_version: int, what: str) -> int:
    """@brief 读取文档的版本号。

    The document's version. A missing ``schema_version`` is version 0. A newer version is refused.

    @param doc 待检查的文档，应为 JSON 对象。
    @param max_version 本进程能读的最高版本。
    @param what 文档的人类可读名称，用于错误信息。
    @return 文档版本号；缺少字段时为 0。
    @raises SchemaError 文档不是对象、版本号非法或高于 ``max_version`` 时抛出。
    """
    if not isinstance(doc, dict):
        raise SchemaError(f"{what} is not a JSON object")
    if "schema_version" not in doc:
        return 0
    version = doc["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 0:
        raise SchemaError(f"{what} schema_version {version!r} is not a version number")
    if version > max_version:
        raise SchemaError(f"{what} schema_version {version} is newer than this process ({max_version})")
    return version


def migrate_report(doc: dict, known_fields: set[str]) -> dict:
    """@brief 把版本 0/1 的报告迁移成构造函数字段。

    Copy a version 0 or 1 report into constructor fields.

    Version 0 and version 1 store the same fields. The only migration is to drop the derived ``eligible`` flag.
    Any other field is refused, including on a version 0 file, so it is not silently ignored.

    @param doc 原始报告字典。
    @param known_fields 本版本认识的字段集合。
    @return 可传给 ``EvalReport(**...)`` 的字段字典。
    @raises SchemaError 版本过新、含未知字段或缺少必要字段时抛出。
    """
    document_version(doc, max_version=REPORT_SCHEMA_VERSION, what="report")
    out = dict(doc)
    eligible = out.pop("eligible", None)
    unknown = set(out) - set(known_fields) - {"schema_version"}
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SchemaError(f"report has fields this process does not know ({names})")
    out.pop("schema_version", None)
    missing = [name for name in ("task", "dtype", "size", "mode") if name not in out]
    if missing:
        raise SchemaError(f"report is missing {missing}")
    # 老文件有时只写了 eligible，没有 status。eligible 为真就当成当时的 ok，否则读出来会变成 pending、不能当种子。
    if "status" not in out and eligible is True:
        out["status"] = "ok"
    return out


def read_request(doc: dict) -> dict:
    """@brief 校验并返回一个本进程可执行的 worker 请求。

    A worker request this process can run. Version 0 is the historical shape, without ``schema_version``.

    @param doc 请求字典。
    @return 原请求字典（字段已校验）。
    @raises SchemaError 版本过新、含未知字段或缺少必要字段时抛出。
    """
    document_version(doc, max_version=REQUEST_SCHEMA_VERSION, what="worker request")
    unknown = set(doc) - REQUEST_FIELDS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise SchemaError(f"worker request has fields this process does not know ({names})")
    missing = sorted(REQUEST_REQUIRED - set(doc))
    if missing:
        raise SchemaError(f"worker request is missing {missing}")
    return doc


def write_json_atomic(path: Path, doc) -> None:
    """@brief 原子地写入一个 JSON 文档。

    Write ``doc`` beside ``path`` and replace it into place. A reader sees the old file or the new one.

    @param path 目标文件路径，父目录会在需要时创建。
    @param doc 可 JSON 序列化的文档。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
