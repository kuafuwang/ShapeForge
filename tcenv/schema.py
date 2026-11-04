"""Version boundary for a worker request and an EvalReport.

Version 0 is a historical document with no ``schema_version`` key. A newer version than this process knows, or a field this version does not list, is refused. Dropping that field and still building a report is how an older parent mis-reads a newer file.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPORT_SCHEMA_VERSION = 4
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


def loads_strict(text: str, *, what: str = "document", reject_constants: bool = False) -> object:
    """@brief 解析 JSON，并拒绝重复键。

    Parse JSON text and refuse duplicate object keys.

    ``json.loads`` silently keeps the last value of a duplicated key, so a file can carry two
    different ``build`` objects and every later check sees only one of them.  Report readers must
    parse through this loader *before* handing the object to ``EvalReport.from_json``: a dict no
    longer remembers that a key was repeated.

    @param text JSON 文本。
    @param what 文档的人类可读名称，用于错误信息。
    @param reject_constants 是否拒绝 ``NaN``/``Infinity``/``-Infinity`` 字面量（身份文档用）。
    @return 解析出的 JSON 值。
    @raises SchemaError 重复键、非有限字面量或 JSON 语法错误时抛出。
    """

    def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
        out: dict = {}
        for key, value in pairs:
            if key in out:
                raise SchemaError(f"{what} repeats the key {key!r}")
            out[key] = value
        return out

    def _no_constant(name: str) -> object:
        raise SchemaError(f"{what} contains the non-finite literal {name}")

    if not isinstance(text, str):
        raise SchemaError(f"{what} is not JSON text")
    try:
        return json.loads(
            text,
            object_pairs_hook=_no_duplicates,
            **({"parse_constant": _no_constant} if reject_constants else {}),
        )
    except SchemaError:
        raise
    except ValueError as exc:
        raise SchemaError(f"{what} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        # A deeply nested document exhausts the parser's stack; ``json`` does not wrap it.
        # Every reader of this loader already maps ``SchemaError`` to its own typed error, so
        # normalizing here keeps envelope/geometry/host-tiling on one closed set.
        raise SchemaError(f"{what} is nested too deeply to parse: {exc}") from exc


def read_report_json(text: str, *, what: str = "report") -> dict:
    """@brief 严格解析一份报告 JSON 文本，返回原始对象（不构造报告）。

    Parse a report file strictly; the caller passes the object to ``EvalReport.from_json``.

    @param text 报告 JSON 文本。
    @param what 人类可读名称，用于错误信息。
    @return 顶层对象。
    @raises SchemaError 不是 JSON 对象或含重复键时抛出。
    """
    doc = loads_strict(text, what=what)
    if not isinstance(doc, dict):
        raise SchemaError(f"{what} is not a JSON object")
    return doc


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
