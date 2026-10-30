"""Test-only v1 → v2 OpSpec mapping used to prove A1's contract equivalence.

This is not a production migration path and not a second authority: it takes the frozen v1
``opspec.json`` facts, removes the retired ``uses`` list, sets ``spec_version=2`` and adds the
document-level discovery default, then re-validates through the v2 loader.  A5 will perform
the per-task migration; A1 uses this mapping to compare every existing v1 semantic instance
with its v2 instantiation.
"""

from __future__ import annotations

import copy
from pathlib import Path

from tcenv.opspec import OpSpec, load_opspec
from tcenv.opspec.v2 import opspec_v2_from_document

ROOT = Path(__file__).resolve().parents[2]
TASKS = ROOT / "tasks"


def v2_document_from_v1(document: dict) -> dict:
    """@brief 把一份 v1 OpSpec 文档映射为等价 v2 文档（测试用）。

    @param document v1 文档。
    @return v2 文档。
    @raises ValueError 输入不是 v1 文档。
    """
    if int(document.get("spec_version", 0)) != 1:
        raise ValueError(f"not a v1 OpSpec document: spec_version={document.get('spec_version')!r}")
    mapped = copy.deepcopy(dict(document))
    mapped["spec_version"] = 2
    mapped.pop("uses", None)
    sizes = list(mapped["sizes"])
    default_size = "small" if "small" in sizes else sizes[0]
    mapped["discovery"] = {
        "phase": 2,
        "default_instance": {"dtype": mapped["type_vars"]["T"][0], "size": default_size},
    }
    return mapped


def mapped_opspec(path: str | Path) -> OpSpec:
    """@brief 读取 v1 OpSpec 并返回通过 v2 校验的映射结果。

    @param path opspec.json 路径。
    @return v2 :class:`OpSpec`。
    """
    return opspec_v2_from_document(v2_document_from_v1(load_opspec(path).document), source=f"mapped:{Path(path)}")


def opspec_paths() -> tuple[Path, ...]:
    """@brief 所有 shipped v1 ``opspec.json``（不含动态 sem_int 家族）。

    @return 路径元组。
    """
    return tuple(sorted(path for path in TASKS.rglob("opspec.json") if path.is_file()))


__all__ = ["TASKS", "mapped_opspec", "opspec_paths", "v2_document_from_v1"]
