"""Static task-pack closure for the v2 standard entry (A3).

The pack file set is derived, never hand-written:

* the task's own standard files come from the v2 registration (``opspec.json``,
  ``reference.py`` and the optional profile/baseline/display/known-good files);
* the code a worker needs to execute that task is the transitive **repo-local import closure**
  of the hardware-neutral contract entry points (``tcenv.taskcontract`` and ``tcenv.opspec``).

The closure deliberately stops at a forbidden root (``tcenv.backends``, ``taic``, ``tasks``,
``tests``, ``observe``, ``experiments``): those are not part of a task pack, and following one
would either drag a backend/toolchain into the pack or create a circular deployment dependency.
An import that is not repo-local (stdlib, torch, numpy, protobuf) is not a pack file.

Knowledge files (``guidance_catalog.md``/``agent_guidance``/``skills``) are never in this set;
they have their own ``run_config.agent_guidance`` identity.
"""

from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from ..identity import canonical_bytes

PACK_SCHEMA_ID = "tcenv.task.standard-pack.1"
PACK_SCHEMA_VERSION = 1
#: Hardware-neutral contract entry points whose import closure defines the code pack.
PACK_ENTRY_MODULES: tuple[str, ...] = ("tcenv.taskcontract", "tcenv.opspec")
#: Roots that are never part of a task pack, even if a module mentions them.
FORBIDDEN_PACK_ROOTS = frozenset({"backends", "taic", "tasks", "tests", "observe", "experiments"})
#: Standard task files, in registration order.  The same list A1's loader recognises.
STANDARD_TASK_FILES: tuple[str, ...] = (
    "opspec.json",
    "reference.py",
    "reference.json",
    "baselines.json",
    "baselines.py",
    "display.json",
    "known_good.metal",
)


class PackClosureError(ValueError):
    """@brief task pack 闭包不可确定。"""


@dataclass(frozen=True)
class TaskPackClosure:
    """@brief 一次 v2 任务的部署 pack（任务文件 + 中立代码闭包）。"""

    task_files: tuple[str, ...]
    code_files: tuple[str, ...]
    digest: str

    def document(self) -> dict:
        """@brief pack 的 JSON 文档（只列路径，不列字节）。

        @return 字段字典。
        """
        return {
            "schema_id": PACK_SCHEMA_ID,
            "task_files": list(self.task_files),
            "code_files": list(self.code_files),
            "digest": self.digest,
        }


def _module_parts(root: Path, path: Path) -> tuple[str, ...] | None:
    """@brief 把一个 ``tcenv`` Python 文件映射成模块名片段。

    @param root 仓库根。
    @param path 文件路径。
    @return 模块片段（不含 ``.py``）；不在 tcenv 下时为 None。
    """
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    if relative.suffix != ".py" or not relative.parts or relative.parts[0] != "tcenv":
        return None
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]
    return tuple(parts)


def _module_path(root: Path, parts: Iterable[str]) -> Path | None:
    """@brief 由模块片段找到仓库内的文件。

    @param root 仓库根。
    @param parts 模块片段。
    @return 文件路径或 None（外部/不存在的模块）。
    """
    base = root.joinpath(*parts)
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _imported_modules(tree: ast.AST, module_parts: tuple[str, ...], root: Path) -> set[tuple[str, ...]]:
    """@brief 一个模块的 repo-local import 目标。

    @param tree AST。
    @param module_parts 当前模块片段。
    @param root 仓库根。
    @return 目标模块片段集合。
    """
    package = module_parts if module_parts and (root.joinpath(*module_parts).is_dir()) else module_parts[:-1]
    found: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = tuple(alias.name.split("."))
                if parts and parts[0] == "tcenv":
                    found.add(parts)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = list(package)
                up = node.level - 1
                if up > len(base):
                    continue
                base = base[: len(base) - up] if up else base
                if node.module:
                    base.extend(node.module.split("."))
                parts = tuple(base)
            elif node.module and node.module.split(".")[0] == "tcenv":
                parts = tuple(node.module.split("."))
            else:
                continue
            if parts:
                found.add(parts)
                # ``from . import helper`` names modules in aliases; include them too.
                for alias in node.names:
                    if alias.name != "*":
                        found.add(parts + tuple(alias.name.split(".")))
    return {parts for parts in found if parts and parts[0] == "tcenv"}


def code_import_closure(
    root: str | Path,
    *,
    entries: Iterable[str] = PACK_ENTRY_MODULES,
) -> tuple[str, ...]:
    """@brief 计算入口模块的 repo-local 静态 import 闭包。

    @param root 仓库根。
    @param entries 入口模块名。
    @return 仓库相对 POSIX 路径（排序、去重）。
    @raises PackClosureError 入口模块不存在或闭包跨越禁止根。
    """
    root = Path(root).resolve()
    pending: list[tuple[str, ...]] = []
    for name in entries:
        parts = tuple(name.split("."))
        if _module_path(root, parts) is None:
            raise PackClosureError(f"pack entry module {name!r} does not exist under {root}")
        pending.append(parts)
    seen: set[tuple[str, ...]] = set()
    files: set[str] = set()
    while pending:
        parts = pending.pop()
        if parts in seen:
            continue
        seen.add(parts)
        path = _module_path(root, parts)
        if path is None:
            # A relative import that names a symbol rather than a submodule (``from . import x``
            # where x is not a module) is not a pack file.
            continue
        parts = _module_parts(root, path) or parts
        if parts[1:2] and parts[1] in FORBIDDEN_PACK_ROOTS:
            continue
        try:
            relative = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise PackClosureError(f"pack module {path} resolves outside the code root {root}") from exc
        files.add(relative)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for target in _imported_modules(tree, parts, root):
            if target[1:2] and target[1] in FORBIDDEN_PACK_ROOTS:
                continue
            pending.append(target)
    # Runtime JSON resources are part of the pack even though no import statement names them:
    # the A1/A2 loaders open their schema with ``Path(__file__).../schema/<name>.json``.  Include
    # every schema resource beside an included module automatically; a hand-written resource list
    # would drift exactly like a hand-written file list.  Only ``schema/*.json`` is considered, so
    # an unrelated data directory cannot silently enter a task pack.
    for relative in tuple(files):
        path = root / relative
        schema_dir = path.parent / "schema"
        if schema_dir.is_symlink():
            raise PackClosureError(
                f"{schema_dir}: a runtime schema directory must be a real directory, not a symlink; "
                "the loader would read bytes outside the deployment pack"
            )
        if schema_dir.exists() and not schema_dir.is_dir():
            raise PackClosureError(f"{schema_dir}: a runtime schema path must be a real directory")
        if not schema_dir.is_dir():
            continue
        # ``Path.rglob`` does not descend into directory symlinks, so it would silently omit a
        # nested ``schema/nested -> outside`` subtree while the loader could still read it.  Walk
        # explicitly with ``followlinks=False`` and reject every symlink or non-regular entry at
        # every level before adding any resource.  This checks entries that are not ``*.json`` too:
        # a symlinked directory is a pack boundary violation even if its current contents are not
        # currently named by the loader.
        for current, dirnames, filenames in os.walk(schema_dir, followlinks=False):
            current_path = Path(current)
            for name in sorted(dirnames):
                entry = current_path / name
                if entry.is_symlink():
                    raise PackClosureError(
                        f"{entry}: a runtime schema path traverses a symlinked directory; the loader could "
                        "read bytes outside the deployment pack"
                    )
                if not entry.is_dir():
                    raise PackClosureError(f"{entry}: a runtime schema directory entry must be a directory")
            for name in sorted(filenames):
                resource = current_path / name
                if resource.is_symlink():
                    raise PackClosureError(
                        f"{resource}: a runtime schema must be a regular file, not a symlink; the loader follows "
                        "the link and would read bytes outside the deployment pack"
                    )
                if not resource.is_file():
                    raise PackClosureError(f"{resource}: a runtime schema must be a regular file")
                if resource.suffix != ".json":
                    continue
                try:
                    resolved = resource.resolve().relative_to(root)
                except ValueError as exc:
                    raise PackClosureError(
                        f"{resource}: a runtime schema resolves outside the code root {root}"
                    ) from exc
                files.add(resolved.as_posix())
    return tuple(sorted(files))


def task_pack_files(task_dir: str | Path) -> tuple[str, ...]:
    """@brief 一个 v2 任务目录的注册文件集（不含知识文件）。

    @param task_dir 任务目录。
    @return 任务相对 POSIX 路径。
    @raises PackClosureError 目录不是 v2 标准入口。
    """
    root = Path(task_dir)
    if not (root / "opspec.json").is_file():
        raise PackClosureError(f"{root}: a v2 task pack needs opspec.json")
    if not (root / "reference.py").is_file():
        raise PackClosureError(f"{root}: a v2 task pack needs reference.py")
    files = tuple(name for name in STANDARD_TASK_FILES if (root / name).is_file())
    return files


def task_pack_closure(
    task_dir: str | Path,
    *,
    root: str | Path,
    code_files: Iterable[str] | None = None,
) -> TaskPackClosure:
    """@brief 任务文件 + 中立代码闭包 + 唯一的 pack 摘要。

    The digest is ``SHA256(RFC8785({schema_id, entries}))`` over the sorted task-file entries
    (task-relative path + bytes) and code entries (repo-relative path + bytes), using the same
    canonical manifest encoder as the artifact set.  The task files and the code files are
    different roots, so the manifest records which side a path belongs to.

    @param task_dir 任务目录。
    @param root 仓库根（代码文件解析）。
    @param code_files 覆盖代码闭包（测试/裁剪用）。
    @return 闭包与摘要。
    @raises PackClosureError 文件/路径非法。
    """
    root = Path(root).resolve()
    task_files = task_pack_files(task_dir)
    code = tuple(sorted(code_files)) if code_files is not None else code_import_closure(root)
    entries: list[dict] = []
    for side, paths, base in (("task", task_files, Path(task_dir)), ("code", code, root)):
        for relative in paths:
            text = str(PurePosixPath(relative))
            if text != relative or text.startswith("../") or text in (".", ".."):
                raise PackClosureError(f"pack path {relative!r} must be a normalized relative POSIX path")
            path = base / relative
            if path.is_symlink() or not path.is_file():
                raise PackClosureError(f"pack file {relative!r} must be an existing regular file")
            data = path.read_bytes()
            entries.append(
                {
                    "side": side,
                    "path": relative if side == "task" else text,
                    "size": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
    manifest = {
        "schema_id": PACK_SCHEMA_ID,
        "schema_version": PACK_SCHEMA_VERSION,
        "entries": sorted(entries, key=lambda item: (item["side"], item["path"])),
    }
    digest = hashlib.sha256(canonical_bytes(manifest)).hexdigest()
    return TaskPackClosure(task_files=task_files, code_files=code, digest=digest)


__all__ = [
    "FORBIDDEN_PACK_ROOTS",
    "PACK_ENTRY_MODULES",
    "PACK_SCHEMA_ID",
    "PACK_SCHEMA_VERSION",
    "STANDARD_TASK_FILES",
    "PackClosureError",
    "TaskPackClosure",
    "code_import_closure",
    "task_pack_closure",
    "task_pack_files",
]
