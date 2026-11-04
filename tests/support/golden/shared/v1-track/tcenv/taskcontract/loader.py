"""The standard v2 task directory: explicit v1/v2 dispatch and one contract object.

The standard entry of a migrated operator is::

    tasks/<op>/
      opspec.json     required, spec_version 2
      reference.py    required, independent oracle
      reference.json  optional sampling/reference profile
      baselines.json  optional performance baselines (+ baselines.py runner)
      display.json    optional representation override
      known_good.metal optional harness self-check

A directory with a v1 marker (``track.json``/``task.py``/``kernel_triton.py``/``naive.metal``)
next to a v2 entry, or a v2 marker next to a v1 OpSpec, is an assembly error.  A v2 entry
without ``reference.py`` is an assembly error too: there is no fallback to the v1 path.

A1 freezes the semantic/reference/display/baseline identity produced here.  It deliberately
stops before the target ABI and candidate geometry: those are A2/A3, and a v2 contract is
therefore not turned into a fake ``Contract``/``Task``.  ``Task.baselines`` (the eventual
evaluation object) is sourced from ``TaskContract.baselines``; an empty tuple derives
``effective_correctness_only``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from ..identity import IdentityError, fingerprint, load_strict_json
from ..opspec.model import OpSpec, load_opspec
from ..opspec.v2 import SemanticInstanceV2, instantiate_v2
from ..task_presentation import render_statement
from ..track import REPO_ROOT, TrackBinding, load_track, pack_digest
from .baselines import BaselinePackage, BaselineRegistration, load_baselines
from .display import DisplayOverrides, load_display
from .errors import AssemblyError, TaskContractError, UnsupportedContractCapability
from .reference import ReferenceModule, load_reference

STANDARD_ENTRY_SCHEMA_VERSION = 1
OPSPEC_FILE = "opspec.json"
#: Files that belong to the retired v1 assembly.  Their presence in a v2 directory means the
#: migration was mixed, not that v2 may read them.
V1_MARKER_FILES: tuple[str, ...] = ("track.json", "task.py", "kernel_triton.py", "naive.metal")
#: Files that belong to the v2 standard entry.
V2_STANDARD_FILES: tuple[str, ...] = (
    "opspec.json",
    "reference.py",
    "reference.json",
    "baselines.json",
    "baselines.py",
    "display.json",
    "known_good.metal",
)
#: v2 marker subset used for mixed-directory detection.
V2_MARKER_FILES: tuple[str, ...] = (
    "reference.py",
    "reference.json",
    "baselines.json",
    "baselines.py",
    "display.json",
    "known_good.metal",
)


@dataclass(frozen=True)
class StandardEntry:
    """@brief 一个标准任务目录的显式分派结果（v1 或 v2）。"""

    schema_version: int
    task_dir: Path
    op_name: str
    spec_version: int
    mode: str
    files: tuple[str, ...]

    @property
    def is_v2(self) -> bool:
        """@brief 是否为 v2 目录。

        @return ``spec_version == 2`` 时为 True。
        """
        return self.spec_version == 2


def _present(root: Path, names: tuple[str, ...]) -> tuple[str, ...]:
    """@brief 目录中实际存在的文件名（固定顺序）。

    @param root 目录。
    @param names 候选文件名。
    @return 已存在文件元组。
    """
    return tuple(name for name in names if (root / name).is_file())


def inspect_standard_entry(task_dir: str | Path) -> StandardEntry:
    """@brief 按 ``spec_version`` 显式识别标准任务目录，并拒绝混合目录。

    @param task_dir 任务目录。
    @return 分派结果。
    @raises AssemblyError 缺 opspec、未知版本、混合目录或缺 v2 reference。
    """
    root = Path(task_dir)
    if not root.is_dir():
        raise AssemblyError(f"{root}: task directory does not exist")
    spec_path = root / OPSPEC_FILE
    if not spec_path.is_file():
        raise AssemblyError(f"{root}: a standard task directory requires {OPSPEC_FILE}")
    try:
        document = load_strict_json(spec_path)
    except IdentityError as exc:
        raise AssemblyError(f"{spec_path}: {exc}") from exc
    version = document.get("spec_version") if isinstance(document, Mapping) else None
    if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2):
        raise AssemblyError(
            f"{spec_path}: spec_version must be the integer 1 or 2, got {version!r}; no loader falls back"
        )
    op_name = str(document.get("op_name", ""))
    v1_present = _present(root, V1_MARKER_FILES)
    v2_present = _present(root, V2_MARKER_FILES)

    if version == 1:
        if v2_present:
            raise AssemblyError(
                f"{root}: mixed v1/v2 task directory; a spec_version 1 task has no v2 files, found {list(v2_present)}"
            )
        if "track.json" not in v1_present:
            raise AssemblyError(f"{root}: a spec_version 1 task requires track.json")
        return StandardEntry(
            schema_version=STANDARD_ENTRY_SCHEMA_VERSION,
            task_dir=root,
            op_name=op_name,
            spec_version=1,
            mode="v1",
            files=_present(root, ("opspec.json",) + V1_MARKER_FILES),
        )

    if v1_present:
        raise AssemblyError(
            f"{root}: mixed v1/v2 task directory; a spec_version 2 task must not carry {list(v1_present)} "
            "(A5 removes them atomically during migration)"
        )
    if "reference.py" not in v2_present:
        raise AssemblyError(
            f"{root}: a spec_version 2 task requires reference.py; a v2 task never falls back to the v1 track/task path"
        )
    if root.name != op_name:
        raise AssemblyError(
            f"{root}: directory name {root.name!r} differs from opspec op_name {op_name!r}; the standard entry "
            "is tasks/<op>/ and automatic discovery depends on that identity"
        )
    return StandardEntry(
        schema_version=STANDARD_ENTRY_SCHEMA_VERSION,
        task_dir=root,
        op_name=op_name,
        spec_version=2,
        mode="v2",
        files=_present(root, V2_STANDARD_FILES),
    )


@dataclass(frozen=True)
class LegacyStandardTask:
    """@brief 一个未迁移 v1 任务的显式分派结果（读旧路径，不做 v2 装配）。"""

    entry: StandardEntry
    track: TrackBinding
    spec: OpSpec

    @property
    def mode(self) -> str:
        """@brief 固定为 ``v1``。

        @return ``"v1"``。
        """
        return "v1"

    @property
    def op_name(self) -> str:
        """@brief 算子名。

        @return OpSpec 名字。
        """
        return self.spec.op_name


def _default_instance(document: Mapping[str, Any]) -> tuple[str, str]:
    """@brief 由 discovery 默认实例或声明顺序选一个 ``(dtype, size)``。

    @param document v2 文档。
    @return dtype/size。
    """
    discovery = document.get("discovery") or {}
    default = discovery.get("default_instance")
    if default:
        return str(default["dtype"]), str(default["size"])
    return str(document["type_vars"]["T"][0]), next(iter(document["sizes"]))


def _pack_identity(root: Path, names: tuple[str, ...]) -> tuple[tuple[str, ...], str]:
    """@brief 计算标准文件的 pack 清单与摘要（知识文件永远不在其中）。

    @param root 任务目录。
    @param names 已存在的标准文件名。
    @return ``(pack_files, digest)``。
    @raises TaskContractError pack 不可读。
    """
    try:
        relative = root.resolve().relative_to(REPO_ROOT.resolve())
        repo_relative = True
    except ValueError:
        repo_relative = False
    if repo_relative:
        files = tuple(f"{relative.as_posix()}/{name}" for name in names)
        return files, pack_digest(files)
    files = tuple(names)
    from ..remote.task_pack import TaskPackError, local_pack_digest

    try:
        return files, local_pack_digest(files, root=root)
    except TaskPackError as exc:
        raise TaskContractError(f"{root}: task pack is invalid: {exc}") from exc


@dataclass(frozen=True)
class TaskContract:
    """@brief A1 的 v2 任务合同（语义 + reference + 表示 + baseline 身份）。

    The contract intentionally has no ``Contract``/``Contract.grid`` and no target ABI.  A2/A3
    attach the derived ``TensorMemoryContract``/``ExecutionGeometry`` and build the evaluator
    ``Task``; this object is the single source those layers consume, so the v2 path cannot
    quietly fall back to a per-task LoweringProfile.
    """

    schema_version: int
    entry: StandardEntry
    spec: OpSpec
    instance: SemanticInstanceV2
    reference: ReferenceModule
    baselines: tuple[BaselineRegistration, ...]
    display: DisplayOverrides
    known_good: str
    statement: str
    pack_files: tuple[str, ...]
    pack_digest: str
    baseline_runners: Mapping[str, Callable[..., Any]] = field(default_factory=dict, repr=False)

    @property
    def op_name(self) -> str:
        """@brief 算子名。

        @return OpSpec 名。
        """
        return self.instance.op_name

    @property
    def dtype(self) -> str:
        """@brief 本次实例的 dtype。

        @return dtype。
        """
        return self.instance.dtype

    @property
    def size(self) -> str:
        """@brief 本次实例的 size 档位。

        @return size。
        """
        return self.instance.size

    @property
    def correctness_only(self) -> bool:
        """@brief 没有 baseline 时只发布 correctness。

        @return ``baselines`` 为空时为 True。
        """
        return not self.baselines

    @property
    def reference_inputs(self) -> tuple[str, ...]:
        """@brief reference 初值合同（input + 所有 read_write，含 output/read_write）。

        @return 张量名元组。
        """
        return self.instance.reference_inputs

    @property
    def reference_outputs(self) -> tuple[str, ...]:
        """@brief reference 差分合同（write 输出 + read_write 终值）。

        @return 张量名元组。
        """
        return self.instance.reference_outputs

    @property
    def reference_source(self) -> str:
        """@brief 只读 reference 展示文本。

        @return 源文本。
        """
        return self.reference.display_source()

    def task_baselines(self) -> tuple[Any, ...]:
        """@brief 投影为 evaluator ``Task.baselines``（A2/A3 接线用）。

        No baseline registration means an empty tuple, which is exactly the input from which
        the evaluation layer derives ``effective_correctness_only``.  The method does not run a
        baseline and does not invent one; it only maps the registered identity and runner.

        @return ``BaselineImpl`` 元组（无 baseline 时为空）。
        """
        if not self.baselines:
            return ()
        from ..task import BaselineImpl

        values: list[Any] = []
        for entry in self.baselines:
            runner = self.baseline_runners[entry.name]
            values.append(
                BaselineImpl(
                    entry.name,
                    entry.framework,
                    runner,
                    implementation_id=entry.implementation_id,
                    implementation_version=int(entry.implementation_version),
                    params=dict(entry.params),
                )
            )
        return tuple(values)

    def fingerprint_document(self) -> dict[str, Any]:
        """@brief 任务合同身份（合同 + reference + display + baseline + pack）。

        Knowledge/guidance is not a field here and is not in ``pack_digest``.

        @return 字段字典。
        """
        return {
            "fingerprint_kind": "task_contract",
            "fingerprint_version": 1,
            "op_name": self.op_name,
            "dtype": self.dtype,
            "size": self.size,
            "opspec_digest": self.spec.digest,
            "semantic_instance": self.instance.fingerprint_document(),
            "reference": self.reference.document(),
            "baselines": [entry.document() for entry in self.baselines],
            "display": self.display.document(),
            "known_good_sha256": hashlib.sha256(self.known_good.encode("utf-8")).hexdigest() if self.known_good else "",
            "statement_sha256": hashlib.sha256(self.statement.encode("utf-8")).hexdigest(),
            "pack_digest": self.pack_digest,
        }

    def fingerprint(self) -> str:
        """@brief 任务合同身份的 canonical 摘要。

        @return 十六进制 sha256。
        """
        return fingerprint(self.fingerprint_document())


def load_task_contract(
    task_dir: str | Path,
    *,
    dtype: str | None = None,
    size: str | None = None,
    variant: str | None = None,
) -> TaskContract:
    """@brief 装配一个 v2 标准任务合同。

    @param task_dir 任务目录。
    @param dtype 实例 dtype；缺省用 discovery 默认实例或声明顺序。
    @param size 实例 size；缺省同上。
    @param variant ``@variant`` 文本；v2 不支持，非 None 时拒绝。
    @return 合同对象。
    @raises UnsupportedContractCapability v2 ``@variant`` 或未实现能力。
    @raises AssemblyError 目录/文件结构错误。
    """
    entry = inspect_standard_entry(task_dir)
    if entry.spec_version == 1:
        raise AssemblyError(
            f"{entry.task_dir}: this is a v1 task; load it through the v1 path (tasks.get_task). "
            "The v2 loader never falls back to v1."
        )
    if variant is not None:
        raise UnsupportedContractCapability(
            f"{entry.task_dir}: v2 has no @variant assembly; a variant is a candidate-owned launch choice, "
            "not a second task contract"
        )
    spec = load_opspec(entry.task_dir / OPSPEC_FILE)
    discovery = spec.document.get("discovery") or {}
    if "variant_acceptance" in discovery:
        raise UnsupportedContractCapability(
            f"{entry.task_dir}: discovery.variant_acceptance is reserved but not implemented in the A1 contract; "
            "a v2 @variant must be refused at assembly time"
        )
    if dtype is None or size is None:
        default_dtype, default_size = _default_instance(spec.document)
        if dtype is None:
            dtype = default_dtype
        if size is None:
            size = default_size
    instance = instantiate_v2(spec, dtype, size)
    # A2 derives the workspace side of the ABI (`TensorMemoryContract` + `workspace_fills`), so a
    # workspace tensor is a normal declaration here.  It is scratch memory: the reference
    # projections exclude it, so it can neither supply an oracle input nor be diffed as a result.
    if not instance.reference_outputs:
        raise AssemblyError(
            f"{entry.task_dir}: a v2 task must declare at least one result tensor (an output or a read_write tensor)"
        )

    reference = load_reference(entry.task_dir, op_name=spec.op_name, instance=instance)
    display = load_display(entry.task_dir, op_name=spec.op_name, opspec_formula=instance.formula)
    baselines = load_baselines(entry.task_dir, op_name=spec.op_name)
    known_good_path = entry.task_dir / "known_good.metal"
    known_good = known_good_path.read_text(encoding="utf-8") if known_good_path.is_file() else ""
    statement = render_statement(instance, display)
    pack_files, digest = _pack_identity(
        entry.task_dir, tuple(name for name in V2_STANDARD_FILES if (entry.task_dir / name).is_file())
    )
    return TaskContract(
        schema_version=STANDARD_ENTRY_SCHEMA_VERSION,
        entry=entry,
        spec=spec,
        instance=instance,
        reference=reference,
        baselines=baselines.registrations,
        display=display,
        known_good=known_good,
        statement=statement,
        pack_files=pack_files,
        pack_digest=digest,
        baseline_runners=MappingProxyType(dict(baselines.runners)),
    )


def load_standard_task(task_dir: str | Path) -> TaskContract | LegacyStandardTask:
    """@brief 按 ``spec_version`` 显式分派：v1 读旧绑定，v2 装配合同。

    This is the one dispatch entry.  It never tries v2 first and falls back (or the reverse):
    the version decides which files are legal, and the wrong branch is an error.

    @param task_dir 任务目录。
    @return v1 的 :class:`LegacyStandardTask` 或 v2 的 :class:`TaskContract`。
    @raises AssemblyError 目录结构非法或混合。
    """
    entry = inspect_standard_entry(task_dir)
    if entry.spec_version == 1:
        spec = load_opspec(entry.task_dir / OPSPEC_FILE)
        track = load_track(entry.task_dir / "track.json")
        return LegacyStandardTask(entry=entry, track=track, spec=spec)
    return load_task_contract(entry.task_dir)


__all__ = [
    "OPSPEC_FILE",
    "STANDARD_ENTRY_SCHEMA_VERSION",
    "V1_MARKER_FILES",
    "V2_MARKER_FILES",
    "V2_STANDARD_FILES",
    "BaselinePackage",
    "LegacyStandardTask",
    "StandardEntry",
    "TaskContract",
    "inspect_standard_entry",
    "load_standard_task",
    "load_task_contract",
]
