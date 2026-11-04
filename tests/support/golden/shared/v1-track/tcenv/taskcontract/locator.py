"""Task discovery, locator identity and the staged pack a local worker can rebuild (A3).

A v2 task is discovered from its standard directory, never from a central name list: a directory
with ``opspec.json`` declares ``spec_version`` and the loader decides v1/v2.  Only a v2 directory
becomes a :class:`TaskLocator`; a v1 directory stays on the v1 path and is not silently treated as
a v2 task.  The locator's identity is the task id/op/instance plus the two pack digests, never the
host path, so the same staged task has one fingerprint on the control host and on a local worker.

The staged pack carries the task-relative files and enough of the code closure to verify the
deployment pack digest.  ``rebuild`` writes a fresh directory, re-derives both digests and the
contract, and refuses any drift instead of "fixing up" the fingerprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping

from ..identity import fingerprint
from ..track import REPO_ROOT
from .loader import TaskContract, inspect_standard_entry, load_task_contract
from .pack import PackClosureError, task_pack_closure, task_pack_files

LOCATOR_SCHEMA_ID = "tcenv.task.locator.1"
LOCATOR_SCHEMA_VERSION = 1
STAGED_PACK_SCHEMA_ID = "tcenv.task.staged-pack.1"


class TaskDiscoveryError(ValueError):
    """@brief v2 任务发现/定位不可确定。"""


def _safe_relative_path(text: object, *, what: str) -> str:
    """@brief 校验/返回规范化的相对 POSIX 路径（任务 id 或 staged 文件名）。

    Rejecting absolute paths, backslashes, empty/``.``/``..`` components and non-canonical
    spellings is a write-safety requirement, not cosmetic: ``rebuild`` joins these values onto a
    worker root and writes them.  A path that passes here has no component that can leave the
    root through ``..`` or a drive/absolute prefix.

    @param text 待校验值。
    @param what 诊断名称。
    @return 原字符串。
    @raises TaskDiscoveryError 非法。
    """
    if not isinstance(text, str) or not text:
        raise TaskDiscoveryError(f"{what} must be a non-empty relative POSIX path")
    if "\\" in text or text.startswith("/") or (len(text) >= 2 and text[1] == ":" and text[0].isalpha()):
        raise TaskDiscoveryError(f"{what} {text!r} is absolute or contains a backslash")
    path = PurePosixPath(text)
    if str(path) != text or any(part in (".", "..") for part in path.parts):
        raise TaskDiscoveryError(f"{what} {text!r} is not a normalized relative path")
    return text


def _relative_task_id(root: Path, task_dir: Path) -> str:
    """@brief 由任务根与目录算出稳定 task id（相对 POSIX 路径）。

    @param root 任务根。
    @param task_dir 任务目录。
    @return task id。
    @raises TaskDiscoveryError 目录不在根下或路径非法。
    """
    try:
        relative = task_dir.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise TaskDiscoveryError(f"{task_dir}: task directory is outside the discovery root {root}") from exc
    text = relative.as_posix()
    return _safe_relative_path(text, what=f"{task_dir}: task id")


def _locator(task_dir: Path, task_id: str, contract: TaskContract, *, root: Path) -> "TaskLocator":
    """@brief 由已装配合同构造 locator。

    @param task_dir 任务目录。
    @param task_id 稳定 task id。
    @param contract v2 任务合同。
    @param root 代码闭包根。
    @return locator。
    @raises TaskDiscoveryError pack 闭包失败。
    """
    try:
        closure = task_pack_closure(task_dir, root=root)
    except PackClosureError as exc:
        raise TaskDiscoveryError(f"{task_dir}: task pack closure failed: {exc}") from exc
    return TaskLocator(
        task_id=task_id,
        op_name=contract.op_name,
        dtype=contract.dtype,
        size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=closure.digest,
        task_dir=task_dir,
        code_root=Path(root),
    )


@dataclass(frozen=True)
class TaskLocator:
    """@brief 一个已发现的 v2 任务（路径不进入身份）。"""

    task_id: str
    op_name: str
    dtype: str
    size: str
    pack_digest: str
    deployment_pack_digest: str
    task_dir: Path = field(compare=False, repr=False)
    #: The code closure root used to build this locator.  A worker may rebuild against a
    #: different checkout, so this host path is not identity; ``load`` recomputes the closure
    #: against this root and refuses a task/code drift instead of trusting the stored digest.
    code_root: Path = field(default=REPO_ROOT, compare=False, repr=False)

    def __post_init__(self) -> None:
        """@brief 校验标识与摘要形状。

        @raises TaskDiscoveryError 非法。
        """
        for name in ("op_name", "dtype", "size"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise TaskDiscoveryError(f"TaskLocator.{name} must be a non-empty string")
        _safe_relative_path(self.task_id, what="TaskLocator.task_id")
        for name in ("pack_digest", "deployment_pack_digest"):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise TaskDiscoveryError(f"TaskLocator.{name} must be a lowercase hexadecimal sha256")
        if not isinstance(self.task_dir, Path):
            object.__setattr__(self, "task_dir", Path(self.task_dir))
        if not isinstance(self.code_root, Path):
            object.__setattr__(self, "code_root", Path(self.code_root))

    def document(self) -> dict[str, Any]:
        """@brief locator 的身份文档（不含 host 路径与 root-dependent pack 拼写）。

        ``pack_digest`` is the A1 contract spelling, which prefixes repo-relative paths for a
        repository task and uses bare names for a staged task.  It is kept as a diagnostic field
        but is *not* the locator identity: ``deployment_pack_digest`` hashes the task-relative
        names + bytes, so the same staged task fingerprints identically on a worker.

        @return 字段字典。
        """
        return {
            "schema_id": LOCATOR_SCHEMA_ID,
            "schema_version": LOCATOR_SCHEMA_VERSION,
            "task_id": self.task_id,
            "op_name": self.op_name,
            "dtype": self.dtype,
            "size": self.size,
            "deployment_pack_digest": self.deployment_pack_digest,
        }

    @property
    def identity(self) -> str:
        """@brief locator 的 canonical 摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.document())

    def load(self) -> TaskContract:
        """@brief 重新装配合同，并证明任务/代码闭包自发现以来没有漂移。

        A locator is a fingerprint, not a cache: reading the current files and then reporting the
        old ``deployment_pack_digest`` would attach stale evidence to changed bytes.  ``load``
        therefore re-derives the contract and both pack digests against ``code_root`` and compares
        the complete locator document; any drift is a refusal, not a silent repair.

        @return 合同。
        @raises TaskDiscoveryError 任务文件或代码闭包已漂移。
        """
        contract = load_task_contract(self.task_dir, dtype=self.dtype, size=self.size)
        try:
            closure = task_pack_closure(self.task_dir, root=self.code_root)
        except PackClosureError as exc:
            raise TaskDiscoveryError(f"{self.task_id}: task pack closure failed: {exc}") from exc
        rebuilt = TaskLocator(
            task_id=self.task_id,
            op_name=contract.op_name,
            dtype=contract.dtype,
            size=contract.size,
            pack_digest=contract.pack_digest,
            deployment_pack_digest=closure.digest,
            task_dir=self.task_dir,
            code_root=self.code_root,
        )
        if rebuilt.document() != self.document():
            raise TaskDiscoveryError(
                f"{self.task_id}: the task or its code pack drifted since discovery; expected "
                f"{self.deployment_pack_digest}, got {rebuilt.deployment_pack_digest}"
            )
        return contract

    def read_pack(self) -> tuple[tuple[str, bytes], ...]:
        """@brief 读取任务 pack 文件的确切字节（任务相对路径，排序）。

        @return ``(path, bytes)`` 元组。
        @raises TaskDiscoveryError 文件缺失/逃逸。
        """
        try:
            names = task_pack_files(self.task_dir)
        except PackClosureError as exc:
            raise TaskDiscoveryError(f"{self.task_id}: {exc}") from exc
        files: list[tuple[str, bytes]] = []
        for name in names:
            path = self.task_dir / name
            if path.is_symlink() or not path.is_file():
                raise TaskDiscoveryError(f"{self.task_id}: pack file {name!r} is not a regular file")
            files.append((name, path.read_bytes()))
        return tuple(files)

    def stage(self) -> "StagedTaskPack":
        """@brief 生成可跨本地 worker 重建的 staging pack。

        @return staging pack。
        """
        return StagedTaskPack(locator=self.document(), files=self.read_pack())


@dataclass(frozen=True)
class StagedTaskPack:
    """@brief 任务文件 + locator 身份的可重建快照（不含 host 路径）。"""

    locator: Mapping[str, Any]
    files: tuple[tuple[str, bytes], ...]

    def __post_init__(self) -> None:
        """@brief 校验快照形状。

        @raises TaskDiscoveryError 非法。
        """
        if not isinstance(self.locator, Mapping) or self.locator.get("schema_id") != LOCATOR_SCHEMA_ID:
            raise TaskDiscoveryError("StagedTaskPack needs a locator document")
        _safe_relative_path(self.locator.get("task_id"), what="staged task id")
        if not self.files:
            raise TaskDiscoveryError("a staged pack needs at least one file")
        names: set[str] = set()
        for name, data in self.files:
            if not isinstance(name, str) or not name or not isinstance(data, (bytes, bytearray)):
                raise TaskDiscoveryError("a staged pack file needs a relative name and bytes")
            _safe_relative_path(name, what="staged pack file")
            if name in names:
                raise TaskDiscoveryError(f"staged pack repeats {name!r}")
            names.add(name)

    def rebuild(self, root: str | Path, *, code_root: str | Path | None = None) -> TaskLocator:
        """@brief 在本地工作根重建任务并校验指纹；任何漂移都拒绝。

        @param root 重建根目录（会创建 ``<task_id>/``）。
        @param code_root 代码闭包根；缺省仓库根。
        @return 重建后的 locator。
        @raises TaskDiscoveryError 摘要/任务身份不一致。
        """
        document = dict(self.locator)
        task_id = _safe_relative_path(document.get("task_id"), what="staged task id")
        safe_files = tuple(
            (_safe_relative_path(name, what="staged pack file"), bytes(data)) for name, data in self.files
        )
        if not safe_files:
            raise TaskDiscoveryError("a staged pack needs at least one file")
        root_path = Path(root).resolve()
        task_dir = root_path / task_id
        current = root_path
        for part in PurePosixPath(task_id).parts:
            current = current / part
            if current.is_symlink():
                raise TaskDiscoveryError(f"{current}: refusing to rebuild through a symlinked path component")
        if task_dir.exists():
            if any(task_dir.iterdir()):
                raise TaskDiscoveryError(f"{task_dir}: refusing to rebuild into a non-empty directory")
        task_dir.mkdir(parents=True, exist_ok=True)
        for name, data in safe_files:
            target = task_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        try:
            contract = load_task_contract(task_dir, dtype=str(document["dtype"]), size=str(document["size"]))
        except Exception as exc:  # noqa: BLE001 - report the contract failure
            raise TaskDiscoveryError(f"staged task does not assemble: {exc}") from exc
        rebuilt = _locator(task_dir, task_id, contract, root=Path(code_root) if code_root is not None else REPO_ROOT)
        if rebuilt.document() != document:
            raise TaskDiscoveryError(
                "the rebuilt task fingerprint differs from the staged locator; the staged task or the code pack drifted"
            )
        return rebuilt


def discover_standard_tasks(
    root: str | Path = REPO_ROOT / "tasks",
    *,
    code_root: str | Path | None = None,
) -> tuple[TaskLocator, ...]:
    """@brief 自动发现 v2 标准任务目录（v1 目录不进入 v2 发现）。

    @param root 发现根（默认仓库 ``tasks/``）。
    @param code_root 代码闭包根；缺省仓库根。
    @return 按 task_id 排序的 locator。
    @raises TaskDiscoveryError 重复 task/op 或混合目录。
    """
    root = Path(root)
    if not root.is_dir():
        raise TaskDiscoveryError(f"{root}: discovery root is not a directory")
    code_root = Path(code_root) if code_root is not None else REPO_ROOT
    found: list[TaskLocator] = []
    seen_ids: dict[str, str] = {}
    seen_ops: dict[str, TaskLocator] = {}
    for spec in sorted(root.rglob("opspec.json")):
        task_dir = spec.parent
        if "__pycache__" in task_dir.parts:
            continue
        entry = inspect_standard_entry(task_dir)
        if not entry.is_v2:
            continue
        task_id = _relative_task_id(root, task_dir)
        if task_id.casefold() in seen_ids:
            raise TaskDiscoveryError(f"duplicate task id {task_id!r} (also {seen_ids[task_id.casefold()]!r})")
        seen_ids[task_id.casefold()] = task_id
        contract = load_task_contract(task_dir)
        locator = _locator(task_dir, task_id, contract, root=code_root)
        previous = seen_ops.get(locator.op_name)
        if previous is not None and previous.deployment_pack_digest != locator.deployment_pack_digest:
            raise TaskDiscoveryError(
                f"op {locator.op_name!r} is declared by two tasks with different packs: "
                f"{previous.task_id!r} and {task_id!r}"
            )
        seen_ops[locator.op_name] = locator
        found.append(locator)
    return tuple(sorted(found, key=lambda item: item.task_id))


__all__ = [
    "LOCATOR_SCHEMA_ID",
    "LOCATOR_SCHEMA_VERSION",
    "STAGED_PACK_SCHEMA_ID",
    "StagedTaskPack",
    "TaskDiscoveryError",
    "TaskLocator",
    "discover_standard_tasks",
]
