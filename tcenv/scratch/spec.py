"""Hardware-neutral candidate scratch: declaration, quota and lifecycle.

Scratch is candidate-owned intermediate storage.  It is not an OpSpec workspace tensor and it
does not name a Metal or NPU type.  An empty declaration is omitted from every canonical
document so existing geometry, bundle and artifact bytes stay identical.  A non-empty
declaration is part of the execution identity: dtype, shape terms and init policy all move
the digest, and an old cache key must not hit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contract_errors import GeometryError
from ..dtypes import get_dtype
from ..track.expr import INT, evaluate_term

INIT_POLICIES = frozenset({"uninitialized_poison", "device_zero", "host_written"})
SCRATCH_ACCESS = frozenset({"read", "write", "read_write"})
DEFAULT_MAX_SCRATCH_BUFFERS = 8
DEFAULT_MAX_TOTAL_SCRATCH_BYTES = 67_108_864
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def bundle_delivery_allowed(*, workspace_declared: bool, scratch_declared: bool) -> bool:
    """@brief msl-bundle 是否有真实可写的中间存储。

    空壳（OpSpec 无 workspace，且候选几何也没有 scratch）继续拒绝。任一来源非空即允许进入
    投影；启动时还没有候选，``scratch_declared`` 为 False，无 workspace 的题仍在启动处拒绝。

    @param workspace_declared 权威 OpSpec 是否声明了 workspace 张量。
    @param scratch_declared 候选 ``ExecutionGeometry.scratch`` 是否非空。
    @return 允许交付时为 True。
    """
    return bool(workspace_declared or scratch_declared)


def _json_term(value: Any) -> Any:
    """@brief 把 size term 收成 JSON 值（tuple 变成 list）。"""
    if isinstance(value, tuple):
        return [_json_term(item) for item in value]
    if isinstance(value, list):
        return [_json_term(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _json_term(item) for key, item in value.items()}
    return value


def _identifier(name: Any, where: str) -> str:
    text = str(name)
    if not isinstance(name, str) or not _IDENTIFIER.fullmatch(text):
        raise GeometryError(f"{where}: {name!r} is not a scratch identifier")
    return text


@dataclass(frozen=True)
class ScratchSpec:
    """@brief 一份候选自有 scratch 的闭集声明（求值前）。"""

    name: str
    dtype: str
    size_terms: tuple[Any, ...]
    init_policy: str = "uninitialized_poison"

    def __post_init__(self) -> None:
        _identifier(self.name, "scratch.name")
        if self.init_policy not in INIT_POLICIES:
            raise GeometryError(
                f"scratch {self.name!r}: init_policy must be one of {sorted(INIT_POLICIES)}, got {self.init_policy!r}"
            )
        try:
            get_dtype(self.dtype)
        except KeyError as exc:
            raise GeometryError(f"scratch {self.name!r}: {exc}") from exc
        if not isinstance(self.size_terms, tuple) or not self.size_terms:
            raise GeometryError(f"scratch {self.name!r}: size_terms must be a non-empty product of integer terms")

    def document(self) -> dict[str, Any]:
        """@brief 进入几何文档的字段。空 scratch 列表的调用方不得写入该键。"""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "size_terms": [_json_term(term) for term in self.size_terms],
            "init_policy": self.init_policy,
        }

    @classmethod
    def from_document(cls, document: Any, *, where: str) -> "ScratchSpec":
        """@brief 从几何 JSON 还原一份声明。

        @param document 一个 scratch 对象。
        @param where 诊断位置。
        @return 声明。
        @raises GeometryError 字段不闭合或类型非法。
        """
        if not isinstance(document, Mapping):
            raise GeometryError(f"{where}: a scratch declaration must be an object")
        allowed = {"name", "dtype", "size_terms", "init_policy"}
        unknown = sorted(set(document) - allowed)
        if unknown or not {"name", "dtype", "size_terms"} <= set(document):
            raise GeometryError(
                f"{where}: scratch fields must be name, dtype, size_terms and optional init_policy, got {sorted(document)}"
            )
        terms = document["size_terms"]
        if not isinstance(terms, (list, tuple)) or not terms:
            raise GeometryError(f"{where}.size_terms: expected a non-empty list of integer terms")
        return cls(
            name=str(document["name"]),
            dtype=str(document["dtype"]),
            size_terms=tuple(terms),
            init_policy=str(document.get("init_policy", "uninitialized_poison")),
        )


@dataclass(frozen=True)
class ScratchUse:
    """@brief 一个 stage 对 scratch 的访问声明。"""

    name: str
    access: str

    def __post_init__(self) -> None:
        _identifier(self.name, "scratch use")
        if self.access not in SCRATCH_ACCESS:
            raise GeometryError(
                f"scratch {self.name!r}: access must be one of {sorted(SCRATCH_ACCESS)}, got {self.access!r}"
            )

    def document(self) -> dict[str, str]:
        return {"name": self.name, "access": self.access}

    @classmethod
    def from_document(cls, document: Any, *, where: str) -> "ScratchUse":
        if not isinstance(document, Mapping) or set(document) != {"name", "access"}:
            raise GeometryError(f"{where}: a scratch use is {{name, access}}, got {document!r}")
        return cls(name=str(document["name"]), access=str(document["access"]))


@dataclass(frozen=True)
class ResolvedScratch:
    """@brief 已按实例求值的 scratch（字节数进入执行身份）。"""

    name: str
    dtype: str
    size_terms: tuple[Any, ...]
    numel: int
    nbytes: int
    init_policy: str

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "size_terms": [_json_term(term) for term in self.size_terms],
            "numel": int(self.numel),
            "nbytes": int(self.nbytes),
            "init_policy": self.init_policy,
        }

    @classmethod
    def from_document(cls, document: Any, *, where: str) -> "ResolvedScratch":
        """@brief 从已求值几何文档还原一块 scratch。

        还原的是冻结文档里的字节数，不按 size term 重算。worker 拿这份文档和
        重新求值的几何比身份；重算会把一份被改过的冻结文档悄悄变成“一致”。

        @param document 一个 resolved scratch 对象。
        @param where 诊断位置。
        @return 已求值 scratch。
        @raises GeometryError 字段不闭合或类型非法。
        """
        if not isinstance(document, Mapping):
            raise GeometryError(f"{where}: a resolved scratch must be an object")
        allowed = {"name", "dtype", "size_terms", "numel", "nbytes", "init_policy"}
        if set(document) != allowed:
            raise GeometryError(
                f"{where}: resolved scratch fields must be {sorted(allowed)}, got {sorted(document)}"
            )
        terms = document["size_terms"]
        if not isinstance(terms, (list, tuple)) or not terms:
            raise GeometryError(f"{where}.size_terms: expected a non-empty list of integer terms")
        numel = document["numel"]
        nbytes = document["nbytes"]
        if isinstance(numel, bool) or not isinstance(numel, int) or numel < 1:
            raise GeometryError(f"{where}.numel: expected a positive integer, got {numel!r}")
        if isinstance(nbytes, bool) or not isinstance(nbytes, int) or nbytes < 1:
            raise GeometryError(f"{where}.nbytes: expected a positive integer, got {nbytes!r}")
        dtype = str(document["dtype"])
        try:
            get_dtype(dtype)
        except KeyError as exc:
            raise GeometryError(f"{where}: {exc}") from exc
        policy = str(document["init_policy"])
        if policy not in INIT_POLICIES:
            raise GeometryError(
                f"{where}: init_policy must be one of {sorted(INIT_POLICIES)}, got {policy!r}"
            )
        return cls(
            name=_identifier(document["name"], f"{where}.name"),
            dtype=dtype,
            size_terms=tuple(terms),
            numel=int(numel),
            nbytes=int(nbytes),
            init_policy=policy,
        )


def parse_scratch_list(document: Any, *, where: str) -> tuple[ScratchSpec, ...]:
    """@brief 解析几何顶层的可选 ``scratch`` 数组。缺省与空数组都是空声明。"""
    if document is None:
        return ()
    if not isinstance(document, (list, tuple)):
        raise GeometryError(f"{where}: scratch must be a list of declarations")
    specs = tuple(ScratchSpec.from_document(item, where=f"{where}[{index}]") for index, item in enumerate(document))
    return specs


def parse_scratch_uses(document: Any, *, where: str) -> tuple[ScratchUse, ...]:
    """@brief 解析一个 stage 的可选 scratch 使用列表。"""
    if document is None:
        return ()
    if not isinstance(document, (list, tuple)):
        raise GeometryError(f"{where}: scratch uses must be a list")
    uses: list[ScratchUse] = []
    seen: set[str] = set()
    for index, item in enumerate(document):
        if isinstance(item, ScratchUse):
            use = item
        else:
            use = ScratchUse.from_document(item, where=f"{where}[{index}]")
        if use.name in seen:
            raise GeometryError(f"{where}: scratch {use.name!r} is listed twice")
        seen.add(use.name)
        uses.append(use)
    return tuple(uses)


def scratch_nbytes(spec: ScratchSpec, names: Mapping[str, int], *, where: str) -> tuple[int, int]:
    """@brief 求值元素数与字节数。每一项必须是正整数，总字节进入配额。"""
    total = 1
    for index, term in enumerate(spec.size_terms):
        value = evaluate_term(term, names, where=f"{where}.size_terms[{index}]", expect=INT)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise GeometryError(f"{where}.size_terms[{index}]: expected a positive integer, got {value!r}")
        total *= int(value)
    nbytes = total * int(get_dtype(spec.dtype).nbytes)
    return total, nbytes


def check_scratch_lifecycle(
    specs: Sequence[ScratchSpec],
    stages: Sequence[tuple[bool, Sequence[ScratchUse]]],
    *,
    reserved_names: set[str],
    workspace_names: set[str],
    names: Mapping[str, int],
    max_scratch_buffers: int,
    max_total_scratch_bytes: int,
    host_program_present: bool,
    where: str = "scratch",
) -> tuple[ResolvedScratch, ...]:
    """@brief 编译前的配额、重名、未写先读与 init 互斥检查。

    ``uninitialized_poison`` 的第一处激活访问必须是只写。``device_zero`` 与 ``host_written``
    不能是同一块（同名即拒绝）。没有 Host Program / Host Plan 的候选不能声明 ``host_written``。
    scratch 名不得与 OpSpec 张量、符号、属性或 workspace 重复，避免同一块缓冲被计两次。

    @param specs 候选声明。
    @param stages ``(active, uses)``，按 stage 顺序。
    @param reserved_names 张量、符号、属性、派生标量。
    @param workspace_names ABI workspace 名（与 scratch 重复即双重计费）。
    @param names 实例符号，用于形状求值。
    @param max_scratch_buffers 块数上限。
    @param max_total_scratch_bytes 单份声明的总字节上限（不含轮转副本）。
    @param host_program_present 候选是否带 C99 Host Program 或 JSON Host Plan。
    @param where 诊断前缀。
    @return 已求值 scratch。
    @raises GeometryError 任一规则失败。
    """
    if len(specs) > int(max_scratch_buffers):
        raise GeometryError(
            f"{where}: {len(specs)} scratch buffers exceeds max_scratch_buffers {int(max_scratch_buffers)}"
        )
    by_name: dict[str, ScratchSpec] = {}
    resolved: list[ResolvedScratch] = []
    total_bytes = 0
    for index, spec in enumerate(specs):
        slot = f"{where}[{index}:{spec.name}]"
        if spec.name in by_name:
            previous = by_name[spec.name].init_policy
            raise GeometryError(
                f"{slot}: scratch name {spec.name!r} is declared twice ({previous} and {spec.init_policy}); "
                "device_zero and host_written are mutually exclusive and a name cannot be repeated"
            )
        if spec.name in reserved_names or spec.name in workspace_names:
            raise GeometryError(
                f"{slot}: scratch name {spec.name!r} collides with an OpSpec tensor, symbol, attr or workspace; "
                "borrowing an output such as y, or billing the same buffer as both workspace and scratch, is refused"
            )
        if spec.init_policy == "host_written" and not host_program_present:
            raise GeometryError(
                f"{slot}: init_policy host_written requires a C99 host/tiling.c or a JSON Host Plan; "
                "declaring the policy without executing a host program is refused"
            )
        numel, nbytes = scratch_nbytes(spec, names, where=slot)
        total_bytes += nbytes
        by_name[spec.name] = spec
        resolved.append(
            ResolvedScratch(
                name=spec.name,
                dtype=spec.dtype,
                size_terms=tuple(spec.size_terms),
                numel=numel,
                nbytes=nbytes,
                init_policy=spec.init_policy,
            )
        )
    if total_bytes > int(max_total_scratch_bytes):
        raise GeometryError(
            f"{where}: scratch uses {total_bytes} bytes, above max_total_scratch_bytes {int(max_total_scratch_bytes)}"
        )
    for _active, uses in stages:
        unknown = [use.name for use in uses if use.name not in by_name]
        if unknown:
            raise GeometryError(f"{where}: stage scratch {unknown} is not declared on the geometry")
    for spec in specs:
        seen_active = False
        for active, uses in stages:
            matched = [use for use in uses if use.name == spec.name]
            if not matched or not active:
                continue
            use = matched[0]
            if not seen_active and spec.init_policy == "uninitialized_poison" and use.access != "write":
                raise GeometryError(
                    f"{where}: uninitialized_poison scratch {spec.name!r} is first used as {use.access}; "
                    "the first active stage must bind it write-only so a later stage cannot read an unwritten value"
                )
            seen_active = True
        if not seen_active:
            raise GeometryError(
                f"{where}: scratch {spec.name!r} is not used by any active stage; an unused scratch is refused"
            )
    return tuple(resolved)


__all__ = [
    "DEFAULT_MAX_SCRATCH_BUFFERS",
    "DEFAULT_MAX_TOTAL_SCRATCH_BYTES",
    "INIT_POLICIES",
    "SCRATCH_ACCESS",
    "ResolvedScratch",
    "ScratchSpec",
    "ScratchUse",
    "bundle_delivery_allowed",
    "check_scratch_lifecycle",
    "parse_scratch_list",
    "parse_scratch_uses",
    "scratch_nbytes",
]
