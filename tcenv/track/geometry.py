"""Typed ``ExecutionGeometry`` resolution, fallback pairing and execution identity (A2).

A1 froze the geometry *fields* and deliberately did not parse them.  This module is the typed
layer:

* :func:`resolve_geometry` turns a candidate (or fallback) ``ExecutionGeometry`` into
  :class:`ResolvedGeometry` — concrete ``grid``/``threads``/``threadgroup_bytes``, active/inactive
  stages, workspace uses and named ``Derived Scalars``.  Every failure is a ``GeometryError``
  with the offending location; nothing is defaulted.
* :class:`GeometryPair` keeps the fallback **independent**: it is a separate typed geometry with
  its own ``source``, it is never merged into, embedded in or inferred from the candidate, and
  both raw documents plus both resolutions enter the execution identity.  A worker that silently
  uses the fallback instead of the candidate is therefore a mismatch, not an unobservable
  substitution.
* :func:`execution_identity` is that identity; :func:`dispatch_parity` compares the recorded
  stages with the per-stage dispatch a build actually produced.

What a candidate may write is *not* restricted here: there is no loop/goto/persistent review, no
SDK-feature whitelist and no per-op tiling rule.  The only rules are the ones the ABI and the
target impose: names resolve, expressions evaluate exactly, values are positive, resources fit,
workspace tensors exist, derived scalar names are unambiguous, and at least one stage runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..contract_errors import GeometryError
from ..identity import fingerprint
from ..opspec.v2 import SemanticInstanceV2
from .abi import TargetAbiLimits
from .expr import BOOL, INT, evaluate_term, kind_of, names_of, parse_term
from .plane import ExecutionGeometry, GeometryStage, TensorMemoryContract

GEOMETRY_RESOLUTION_VERSION = 1
GEOMETRY_SOURCES = frozenset({"candidate", "fallback"})
#: Kernel and derived-scalar identifiers: a Metal function name is an ASCII identifier.  The
#: derived-scalar rule is the plan's "by name" rule, not a style preference about the kernel.
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_DERIVED_NAME = re.compile(r"[A-Z][A-Z0-9]*(_[A-Z0-9]+)*")


@dataclass(frozen=True)
class WorkspaceUse:
    """@brief 一个 stage 对 workspace 张量的使用。"""

    name: str
    zero: bool

    def document(self) -> dict[str, Any]:
        """@brief 该使用的 JSON 文档。

        @return 字段字典。
        """
        return {"name": self.name, "zero": bool(self.zero)}


@dataclass(frozen=True)
class ResolvedStage:
    """@brief 已求值的一个 stage。"""

    index: int
    kernel: str
    grid: tuple[int, int, int]
    threads: tuple[int, int, int]
    threadgroup_bytes: int
    workspace: tuple[WorkspaceUse, ...]
    active: bool
    when: bool | int | None
    derived_scalars: tuple[tuple[str, int], ...]

    @property
    def threads_per_group(self) -> int:
        """@brief 每组线程数。

        @return 三个分量的乘积。
        """
        return self.threads[0] * self.threads[1] * self.threads[2]

    def derived(self, name: str) -> int:
        """@brief 按名字取派生标量的值。

        @param name 名字。
        @return 值。
        @raises KeyError 该 stage 没有这个派生标量。
        """
        for key, value in self.derived_scalars:
            if key == name:
                return value
        raise KeyError(name)

    def document(self) -> dict[str, Any]:
        """@brief 该 stage 的已求值投影（进入执行身份与 dispatch 证据）。

        @return 字段字典。
        """
        return {
            "index": self.index,
            "kernel": self.kernel,
            "grid": list(self.grid),
            "threads": list(self.threads),
            "threadgroup_bytes": int(self.threadgroup_bytes),
            "workspace": [use.document() for use in self.workspace],
            "active": bool(self.active),
            "when": self.when,
            "derived_scalars": [{"name": name, "value": int(value)} for name, value in self.derived_scalars],
        }

    def dispatch_signature(self) -> tuple[Any, ...]:
        """@brief 逐段实际 dispatch 的比较键。

        @return 可比较的元组。
        """
        return (
            self.kernel,
            self.grid,
            self.threads,
            int(self.threadgroup_bytes),
            tuple(use.name for use in self.workspace),
        )


@dataclass(frozen=True)
class ResolvedGeometry:
    """@brief 一整份已求值的几何（候选或 fallback）。"""

    source: str
    stages: tuple[ResolvedStage, ...]
    resolution_version: int = GEOMETRY_RESOLUTION_VERSION

    @property
    def active_stages(self) -> tuple[ResolvedStage, ...]:
        """@brief 该实例实际启动的 stage。

        @return 激活 stage 元组。
        """
        return tuple(stage for stage in self.stages if stage.active)

    @property
    def workspace_uses(self) -> tuple[WorkspaceUse, ...]:
        """@brief 所有激活 stage 声明的 workspace 使用（按首次出现顺序去重）。

        @return 使用元组。
        @raises GeometryError 同一 workspace 被声明成零与非零两种策略。
        """
        seen: dict[str, bool] = {}
        for stage in self.active_stages:
            for use in stage.workspace:
                if seen.get(use.name, use.zero) != use.zero:
                    raise GeometryError(
                        f"workspace `{use.name}` is declared both zero and not-zero by active stages; the "
                        "clearing policy of one workspace must be one value"
                    )
                seen.setdefault(use.name, use.zero)
        return tuple(WorkspaceUse(name=name, zero=zero) for name, zero in seen.items())

    def document(self) -> dict[str, Any]:
        """@brief 已求值几何的 JSON 文档。

        @return 字段字典。
        """
        return {
            "resolution_version": self.resolution_version,
            "source": self.source,
            "stages": [stage.document() for stage in self.stages],
        }


@dataclass(frozen=True)
class GeometryPair:
    """@brief 候选几何与其独立 fallback（两者都进入执行身份）。"""

    candidate_document: Mapping[str, Any]
    fallback_document: Mapping[str, Any] | None = None
    candidate: ResolvedGeometry | None = None
    fallback: ResolvedGeometry | None = None
    candidate_error: str | None = None
    fallback_error: str | None = None
    resolution_version: int = GEOMETRY_RESOLUTION_VERSION

    def __post_init__(self) -> None:
        """@brief 校验来源，禁止把 fallback 当候选或把候选当 fallback。

        @raises GeometryError 来源或状态不符。
        """
        if str(self.candidate_document.get("source")) != "candidate":
            raise GeometryError(
                f"GeometryPair candidate has source {self.candidate_document.get('source')!r}; a fallback is a "
                "separate, independent geometry and never becomes the candidate"
            )
        if self.fallback_document is not None and str(self.fallback_document.get("source")) != "fallback":
            raise GeometryError(
                f"GeometryPair fallback has source {self.fallback_document.get('source')!r}; the fallback must "
                "declare itself"
            )
        if self.candidate is None and self.candidate_error is None:
            raise GeometryError("GeometryPair has neither a resolved candidate nor the error that stopped it")
        if self.fallback_document is not None and self.fallback is None and self.fallback_error is None:
            raise GeometryError("GeometryPair has a fallback document but neither its resolution nor its error")

    def document(self) -> dict[str, Any]:
        """@brief 配对的 JSON 文档。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "resolution_version": self.resolution_version,
            "candidate": dict(self.candidate_document),
            "candidate_resolution": self.candidate.document() if self.candidate is not None else None,
            "candidate_error": self.candidate_error,
        }
        if self.fallback_document is not None:
            document["fallback"] = dict(self.fallback_document)
            document["fallback_resolution"] = self.fallback.document() if self.fallback is not None else None
            document["fallback_error"] = self.fallback_error
        return document

    @property
    def identity(self) -> str:
        """@brief 执行身份：候选与（可选）独立 fallback 的完整文档与求值结果。

        @return 十六进制 sha256。
        """
        return fingerprint(
            {
                "fingerprint_kind": "execution-geometry",
                "fingerprint_version": 1,
                "pair": self.document(),
            }
        )


@dataclass(frozen=True)
class GeometrySelection:
    """@brief 本次执行选用的几何及其身份。"""

    provider: str
    geometry: ResolvedGeometry
    identity: str
    used_fallback: bool
    candidate_error: str | None = None

    def document(self) -> dict[str, Any]:
        """@brief 选择的 JSON 文档（进入证据，不进入 EvalReport 自由字段）。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "provider": self.provider,
            "used_fallback": bool(self.used_fallback),
            "identity": self.identity,
            "geometry": self.geometry.document(),
        }
        if self.candidate_error is not None:
            document["candidate_error"] = self.candidate_error
        return document


def resolve_geometry(
    geometry: ExecutionGeometry,
    *,
    instance: SemanticInstanceV2,
    contract: TensorMemoryContract,
    limits: TargetAbiLimits,
    where: str = "execution_geometry",
) -> ResolvedGeometry:
    """@brief 求值一份几何，并校验它与实例/ABI/目标上限一致。

    @param geometry 候选或 fallback 几何。
    @param instance v2 语义实例。
    @param contract 已派生的 ABI 合同。
    @param limits 目标上限。
    @param where 诊断前缀。
    @return 已求值几何。
    @raises GeometryError 任何一项校验失败。
    """
    if not isinstance(geometry, ExecutionGeometry):
        raise GeometryError(f"{where}: expected an ExecutionGeometry")
    if geometry.source not in GEOMETRY_SOURCES:
        raise GeometryError(f"{where}.source: must be one of {sorted(GEOMETRY_SOURCES)}, got {geometry.source!r}")
    if not geometry.stages:
        raise GeometryError(f"{where}: a geometry needs at least one stage")
    names = _evaluation_names(instance, where)
    workspace_names = {tensor.name for tensor in contract.tensors if tensor.space == "workspace"}
    reserved = set(names) | {tensor.name for tensor in contract.tensors}
    stages = tuple(
        _resolve_stage(
            stage,
            index,
            names=names,
            workspace_names=workspace_names,
            reserved=reserved,
            limits=limits,
            where=f"{where}.stages[{index}]",
        )
        for index, stage in enumerate(geometry.stages)
    )
    resolved = ResolvedGeometry(source=geometry.source, stages=stages)
    if not resolved.active_stages:
        raise GeometryError(
            f"{where}: every stage is inactive for {instance.op_name}.{instance.dtype}.{instance.size}; a zero-stage "
            "launch is refused rather than turned into an empty dispatch"
        )
    # One workspace must have one clearing policy across the active stages; resolving the
    # geometry is the right place to refuse the conflict, so no later consumer can observe a
    # half-valid workspace declaration.
    _ = resolved.workspace_uses
    return resolved


def _resolve_stage(
    stage: GeometryStage,
    index: int,
    *,
    names: Mapping[str, int],
    workspace_names: set[str],
    reserved: set[str],
    limits: TargetAbiLimits,
    where: str,
) -> ResolvedStage:
    """@brief 求值一个 stage 的字段。

    @param stage 几何 stage。
    @param index 段序号。
    @param names 符号/属性名到整数的映射。
    @param workspace_names 合同 workspace 张量名。
    @param reserved 不可复用的名字（符号/属性/张量）。
    @param limits 目标上限。
    @param where 诊断位置。
    @return 已求值 stage。
    @raises GeometryError 校验失败。
    """
    if not isinstance(stage.kernel, str) or not _IDENTIFIER.fullmatch(stage.kernel):
        raise GeometryError(f"{where}.kernel: {stage.kernel!r} is not a kernel identifier")
    derived = _derived_scalars(stage, names=names, reserved=reserved, where=where)
    # A derived scalar is a named integer, not a second symbol space: the launch shape may use
    # it exactly as it may use a symbol or an attr.  (It may not *depend* on another derived
    # scalar; that rule lives in `_derived_scalars`.)
    stage_names = {**dict(names), **dict(derived)}
    grid = _triple(stage.grid, f"{where}.grid", stage_names)
    threads = _triple(stage.threads, f"{where}.threads", stage_names)
    if any(value < 1 for value in grid):
        raise GeometryError(f"{where}.grid: every component must be >= 1, got {grid}")
    if any(value < 1 for value in threads):
        raise GeometryError(f"{where}.threads: every component must be >= 1, got {threads}")
    product = threads[0] * threads[1] * threads[2]
    if product > limits.max_threads_per_threadgroup:
        raise GeometryError(
            f"{where}.threads: {product} threads per group exceeds the target limit "
            f"{limits.max_threads_per_threadgroup}"
        )
    threadgroup_bytes = _volume(stage.threadgroup_bytes, f"{where}.threadgroup_bytes", stage_names)
    if threadgroup_bytes < 0:
        raise GeometryError(f"{where}.threadgroup_bytes: must not be negative")
    if threadgroup_bytes > limits.max_threadgroup_memory_bytes:
        raise GeometryError(
            f"{where}.threadgroup_bytes: {threadgroup_bytes} B exceeds the target limit "
            f"{limits.max_threadgroup_memory_bytes} B"
        )
    workspace = _workspace_uses(stage.workspace, workspace_names=workspace_names, where=where)

    if stage.when is None:
        when: bool | int | None = None
        active = True
    else:
        node = parse_term(stage.when, f"{where}.when")
        if kind_of(node) == BOOL:
            value = evaluate_term(stage.when, stage_names, where=f"{where}.when", expect=BOOL)
            when, active = bool(value), bool(value)
        else:
            value = evaluate_term(stage.when, stage_names, where=f"{where}.when", expect=INT)
            when, active = int(value), int(value) != 0

    return ResolvedStage(
        index=index,
        kernel=stage.kernel,
        grid=grid,
        threads=threads,
        threadgroup_bytes=threadgroup_bytes,
        workspace=workspace,
        active=active,
        when=when,
        derived_scalars=derived,
    )


def _evaluation_names(instance: SemanticInstanceV2, where: str) -> dict[str, int]:
    """@brief 几何表达式可见的名字：符号与整数属性。

    @param instance 语义实例。
    @param where 诊断位置。
    @return 名字到整数。
    @raises GeometryError 符号与属性重名，或属性不是整数。
    """
    names: dict[str, int] = {}
    for name, value in instance.symbols.items():
        names[str(name)] = int(value)
    for name, value in instance.resolved_attrs.items():
        if name in names:
            raise GeometryError(
                f"{where}: name {name!r} is both a symbol and an attr; geometry cannot resolve it unambiguously"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            # A float/string attr is a semantic parameter, not a launch size: it is simply not part
            # of the expression name space, so a geometry that references it fails as an unknown
            # name instead of being silently coerced.
            continue
        names[str(name)] = int(value)
    if not names:
        raise GeometryError(
            f"{where}: the instance declares no symbol or integer attr, so no geometry can be sized"
        )
    return names


def _triple(term: Any, where: str, names: Mapping[str, int]) -> tuple[int, int, int]:
    """@brief 求值一个三元组。

    @param term 三元组术语。
    @param where 诊断位置。
    @param names 名字映射（symbols、整数 attrs 与本 stage 的派生标量）。
    @return 三个整数。
    @raises GeometryError 长度或取值非法。
    """
    if not isinstance(term, (list, tuple)):
        raise GeometryError(f"{where}: expected three comma-separated terms, got {type(term).__name__}")
    if len(term) != 3:
        raise GeometryError(f"{where}: expected exactly three terms, got {len(term)}")
    values = []
    for index, item in enumerate(term):
        values.append(_volume(item, f"{where}[{index}]", names))
    return tuple(values)  # type: ignore[return-value]


def _volume(term: Any, where: str, names: Mapping[str, int]) -> int:
    """@brief 求值一个整数术语。

    @param term 术语。
    @param where 诊断位置。
    @param names 名字映射。
    @return 整数。
    @raises GeometryError 类型或求值非法。
    """
    value = evaluate_term(term, names, where=where, expect=INT)
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _derived_scalars(
    stage: GeometryStage, *, names: Mapping[str, int], reserved: set[str], where: str
) -> tuple[tuple[str, int], ...]:
    """@brief 求值并校验派生标量。

    A derived scalar may depend only on symbols/attrs (the plan's rule), never on another derived
    scalar and never on a tensor.  The candidate writes the name; the harness binds it by name
    with ``setBytes``, so there is no hand-written ``binding_slot`` to disagree with.

    @param stage 几何 stage。
    @param names 符号/属性名映射。
    @param reserved 不可复用的名字。
    @param where 诊断位置。
    @return 名字与值的元组。
    @raises GeometryError 名字/依赖/重复非法。
    """
    seen: dict[str, int] = {}
    for entry in stage.derived_scalars:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise GeometryError(f"{where}.derived_scalars: each entry must be a (name, term) pair, got {entry!r}")
        name, term = entry
        if not isinstance(name, str) or not _DERIVED_NAME.fullmatch(name):
            raise GeometryError(f"{where}.derived_scalars: {name!r} is not a derived-scalar name")
        if name in seen:
            raise GeometryError(f"{where}.derived_scalars: duplicate derived scalar {name!r}")
        if name in reserved:
            raise GeometryError(
                f"{where}.derived_scalars: {name!r} shadows a symbol, attr or tensor name; a by-name binding "
                "must be unambiguous"
            )
        referenced = names_of(parse_term(term, f"{where}.derived_scalars[{name}]"))
        foreign = sorted(referenced - set(names))
        if foreign:
            raise GeometryError(
                f"{where}.derived_scalars[{name}]: uses {foreign}; a derived scalar may depend only on "
                f"symbols/attrs (declared: {sorted(names)})"
            )
        seen[name] = _volume(term, f"{where}.derived_scalars[{name}]", names)
    return tuple(seen.items())


def _workspace_uses(term: Any, *, workspace_names: set[str], where: str) -> tuple[WorkspaceUse, ...]:
    """@brief 求值一个 stage 的 workspace 声明。

    @param term stage.workspace 字段。
    @param workspace_names 合同 workspace 张量名。
    @param where 诊断位置。
    @return 使用元组。
    @raises GeometryError 未声明的 workspace、重复使用或字段非法。
    """
    if term is None:
        return ()
    if not isinstance(term, (list, tuple)):
        raise GeometryError(f"{where}.workspace: expected a list of workspace names, got {type(term).__name__}")
    uses: list[WorkspaceUse] = []
    seen: set[str] = set()
    for position, item in enumerate(term):
        if isinstance(item, str):
            name, zero = item, False
        elif isinstance(item, Mapping):
            if set(item) != {"name", "zero"}:
                raise GeometryError(
                    f"{where}.workspace[{position}]: a workspace declaration is a name or {{name, zero}}, got "
                    f"{sorted(item)}"
                )
            name, zero = str(item["name"]), bool(item["zero"])
        else:
            raise GeometryError(f"{where}.workspace[{position}]: unsupported declaration {item!r}")
        if name not in workspace_names:
            raise GeometryError(
                f"{where}.workspace[{position}]: {name!r} is not a workspace tensor of the ABI; declared workspace "
                f"tensors are {sorted(workspace_names) or 'none'}"
            )
        if name in seen:
            raise GeometryError(f"{where}.workspace: {name!r} is listed twice")
        seen.add(name)
        uses.append(WorkspaceUse(name=name, zero=zero))
    return tuple(uses)


def geometry_pair(
    candidate: ExecutionGeometry,
    fallback: ExecutionGeometry | None = None,
    *,
    instance: SemanticInstanceV2,
    contract: TensorMemoryContract,
    limits: TargetAbiLimits,
) -> GeometryPair:
    """@brief 独立求值候选与 fallback 并配对（候选失败也保留失败原因，不吞掉）。

    @param candidate 候选几何。
    @param fallback 独立 fallback 几何；没有时省略。
    @param instance 语义实例。
    @param contract ABI 合同。
    @param limits 目标上限。
    @return 配对。
    @raises GeometryError 传入的候选几何不是 ``source="candidate"``。
    """
    if not isinstance(candidate, ExecutionGeometry):
        raise GeometryError("geometry_pair needs a candidate ExecutionGeometry")
    if candidate.source != "candidate":
        raise GeometryError(
            f"the candidate geometry declares source {candidate.source!r}; pass the fallback with its own source "
            "and let the harness pair them"
        )
    if fallback is not None and not isinstance(fallback, ExecutionGeometry):
        raise GeometryError("geometry_pair needs an ExecutionGeometry for the fallback")
    if fallback is not None and fallback.source != "fallback":
        raise GeometryError(f"the fallback geometry declares source {fallback.source!r}; expected 'fallback'")

    resolved_candidate: ResolvedGeometry | None
    candidate_error: str | None
    try:
        resolved_candidate = resolve_geometry(
            candidate, instance=instance, contract=contract, limits=limits, where="candidate"
        )
        candidate_error = None
    except GeometryError as exc:
        resolved_candidate, candidate_error = None, str(exc)

    resolved_fallback: ResolvedGeometry | None = None
    fallback_error: str | None = None
    if fallback is not None:
        try:
            resolved_fallback = resolve_geometry(
                fallback, instance=instance, contract=contract, limits=limits, where="fallback"
            )
        except GeometryError as exc:
            resolved_fallback, fallback_error = None, str(exc)

    return GeometryPair(
        candidate_document=candidate.document(),
        fallback_document=None if fallback is None else fallback.document(),
        candidate=resolved_candidate,
        fallback=resolved_fallback,
        candidate_error=candidate_error,
        fallback_error=fallback_error,
    )


def select_geometry(pair: GeometryPair, *, allow_fallback: bool = False) -> GeometrySelection:
    """@brief 选择本次执行使用的几何（候选优先，fallback 低优先级且必须显式允许）。

    The fallback is not a silent rescue: without ``allow_fallback`` a call that needs it fails, and
    the selection records ``used_fallback`` so the worker's dispatch can be compared with what the
    evidence says ran.  The identity always covers both geometries.

    @param pair 几何配对。
    @param allow_fallback 是否允许退到独立 fallback。
    @return 选择结果。
    @raises GeometryError 候选不可用且不允许/没有 fallback。
    """
    if pair.candidate is not None and pair.candidate.active_stages:
        return GeometrySelection(
            provider="candidate",
            geometry=pair.candidate,
            identity=pair.identity,
            used_fallback=False,
        )
    reason = pair.candidate_error or "the candidate geometry has no active stage"
    if pair.fallback is None or pair.fallback_error is not None:
        suffix = f"; the fallback is also unusable ({pair.fallback_error})" if pair.fallback_error else ""
        raise GeometryError(
            f"the candidate geometry is unusable ({reason}) and no usable independent fallback is registered{suffix}"
        )
    if not allow_fallback:
        raise GeometryError(
            f"the candidate geometry is unusable ({reason}); using the independent fallback needs an explicit "
            "opt-in so the evidence records it"
        )
    if not pair.fallback.active_stages:
        raise GeometryError("the fallback geometry has no active stage; refusing to dispatch nothing")
    return GeometrySelection(
        provider="fallback",
        geometry=pair.fallback,
        identity=pair.identity,
        used_fallback=True,
        candidate_error=reason,
    )


def execution_identity(pair: GeometryPair) -> str:
    """@brief 几何配对的执行身份摘要。

    @param pair 配对。
    @return 十六进制 sha256。
    """
    return pair.identity


def dispatch_parity(
    expected: ResolvedGeometry,
    observed: Iterable[Sequence[Any]],
    *,
    where: str = "dispatch",
) -> list[str]:
    """@brief 比较记录下来的几何与构建实际报告的逐段 dispatch。

    @param expected 记录在证据里的几何。
    @param observed 实际 dispatch 的逐段 ``(kernel, grid, threads, threadgroup_bytes, workspace)``。
    @param where 诊断前缀。
    @return 人类可读的不匹配列表；空表示一致。
    """
    wanted = [stage.dispatch_signature() for stage in expected.active_stages]
    got: list[tuple[Any, ...]] = []
    for item in observed:
        values = tuple(item)
        if len(values) != 5:
            raise GeometryError(
                f"{where}: a dispatch record needs (kernel, grid, threads, threadgroup_bytes, workspace)"
            )
        kernel, grid, threads, threadgroup_bytes, workspace = values
        got.append(
            (
                str(kernel),
                tuple(int(v) for v in grid),
                tuple(int(v) for v in threads),
                int(threadgroup_bytes),
                tuple(str(w) for w in workspace),
            )
        )
    problems: list[str] = []
    if len(wanted) != len(got):
        problems.append(f"{where}: expected {len(wanted)} active stages, got {len(got)} dispatched")
    for index, (want, have) in enumerate(zip(wanted, got)):
        if want != have:
            problems.append(f"{where}: stage {index} dispatched {have!r}, the recorded geometry says {want!r}")
    return problems


def selection_parity(
    selection: GeometrySelection,
    *,
    provider: str,
    observed: Iterable[Sequence[Any]],
    where: str = "dispatch",
) -> list[str]:
    """@brief 比较“证据里选择的几何”与 worker 实际执行的提供者/逐段 dispatch。

    A worker that silently uses the independent fallback (or dispatches stages the recorded
    geometry does not have) must be a mismatch even when some stages happen to agree.  The
    provider is compared before the dispatch so "right stages, wrong geometry source" cannot
    pass as parity.

    @param selection 证据中的几何选择。
    @param provider worker 实际使用的几何提供者（``candidate``/``fallback``）。
    @param observed 实际 dispatch 的逐段记录。
    @param where 诊断前缀。
    @return 人类可读的不匹配列表；空表示一致。
    """
    problems: list[str] = []
    if provider != selection.provider:
        problems.append(
            f"{where}: the worker dispatched the {provider!r} geometry, but the evidence selected "
            f"{selection.provider!r} (used_fallback={selection.used_fallback})"
        )
    problems.extend(dispatch_parity(selection.geometry, observed, where=where))
    return problems


__all__ = [
    "GEOMETRY_RESOLUTION_VERSION",
    "GEOMETRY_SOURCES",
    "GeometryPair",
    "GeometrySelection",
    "ResolvedGeometry",
    "ResolvedStage",
    "WorkspaceUse",
    "dispatch_parity",
    "execution_identity",
    "geometry_pair",
    "resolve_geometry",
    "select_geometry",
    "selection_parity",
]
