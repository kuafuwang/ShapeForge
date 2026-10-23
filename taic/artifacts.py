"""Versioned run files other than the transcript.

``transcript.jsonl`` is the billed-request record and is append-only. ``state.json``, ``result.json`` and ``ledger.json`` are checkpoints: they can speed up a resume, and they must not be written back over the transcript. A file that is present but damaged, or newer than this process, needs human inspection. It is not treated as work that was never done.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.schema import PROTOCOL_VERSION, SchemaError, document_version

RESULT_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
RUN_CONFIG_SCHEMA_VERSION = 2


class NeedsInspection(RuntimeError):
    """@brief 存储的报告或检查点虽在，但其内容不可信，需要人来看。"""


def read_versioned(path: Path, max_version: int) -> dict | None:
    """@brief 返回 ``path`` 处的 JSON 对象；文件不存在时返回 ``None``。

    文件存在但不是可读对象，或版本号不被认识时，抛出 ``NeedsInspection``。也就是说，
    「文件在但读不懂」不等于「这件事没做过」。

    @param path 版本化 JSON 文件的路径。
    @param max_version 调用方认识的最大文档版本号。
    @return 解析后的 JSON 对象；文件不存在时为 ``None``。
    @raises NeedsInspection 文件损坏、不可读，或文档版本超出 ``max_version`` 时。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    try:
        document_version(doc, max_version=max_version, what=str(path))
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    return doc
