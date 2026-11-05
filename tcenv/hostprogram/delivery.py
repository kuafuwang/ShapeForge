"""Candidate-delivery helpers for the two Host Tiling representations (V3).

A candidate may ship **one** ``host_tiling`` entry:

* ``host/tiling.json`` — the existing declarative linear writes (input-independent);
* ``host/tiling.c``    — restricted C99 compiled/loaded only in a native64 bounded worker.

Both are the same *role*; only the representation differs.  The suffix decides which parser runs,
so a ``.c`` file can never be silently interpreted as JSON (or the reverse).  This module is
backend-neutral: it imports no Metal and no task loader, so the portfolio plane can validate a
program entry without importing ``tcenv.backends.metal``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any, Sequence

from .contract import HostContractSpec, host_contract_spec, referenced_workspaces
from .errors import HostProgramError
from .precheck import static_precheck

#: Suffix -> representation.  ``.c`` is the restricted C99 Host Program; every other path keeps
#: the historical JSON linear parser (the A2/A4 plane never constrained the plan's file suffix).
HOST_TILING_SUFFIXES = {".c": "program"}


@dataclass(frozen=True)
class HostTilingEntry:
    """@brief 候选唯一的 Host Tiling entry（表示 + 源码）。"""

    path: str
    kind: str
    source: str
    sha256: str

    def document(self) -> dict[str, Any]:
        return {"path": self.path, "kind": self.kind, "sha256": self.sha256}


def host_tiling_entry(candidate: Any) -> HostTilingEntry | None:
    """@brief 取候选唯一的 ``host_tiling`` entry。

    @param candidate ``CandidateArtifact``。
    @return entry 或 None（候选没有 host 工作）。
    @raises HostProgramError 多条 entry、未知后缀或不是 UTF-8 文本。
    """
    entries = [entry for entry in getattr(candidate, "entries", ()) if entry.role == "host_tiling"]
    if not entries:
        return None
    if len(entries) != 1:
        raise HostProgramError(
            f"a candidate needs exactly one host_tiling entry, got {len(entries)}; Host Tiling has one "
            "representation (a JSON linear plan or a restricted C99 host/tiling.c), never both"
        )
    entry = entries[0]
    suffix = Path(entry.path).suffix.lower()
    # A ``.c`` entry is the restricted C99 Host Program.  Everything else remains the JSON linear
    # plan: the existing bundle fixtures name it ``host/tiling_a.metal`` and the closed JSON parser
    # is the authority for those bytes.
    kind = HOST_TILING_SUFFIXES.get(suffix, "linear")
    try:
        source = bytes(entry.data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HostProgramError(f"host_tiling entry {entry.path!r} is not UTF-8 text: {exc}") from exc
    return HostTilingEntry(
        path=str(entry.path), kind=kind, source=source, sha256=hashlib.sha256(bytes(entry.data)).hexdigest()
    )


def preflight_host_program(
    source: str,
    *,
    spec: HostContractSpec,
    declared_workspaces: Sequence[str],
    where: str = "host/tiling.c",
) -> tuple[str, ...]:
    """@brief 投影期的 Host Program 预检：危险语法 + workspace 名字必须在合同里。

    真正编译只在有界 worker 的 ``build()`` 里发生；这里不读 include、不调用编译器。

    @param source 候选 C 源码。
    @param spec 生成的合同。
    @param declared_workspaces 任务合同声明的 workspace 名。
    @param where 诊断位置。
    @return 问题列表；空表示通过。
    """
    problems = list(static_precheck(source))
    referenced = referenced_workspaces(source, spec)
    unknown = [name for name in referenced if name not in set(declared_workspaces)]
    if unknown:
        problems.append(f"{where}: references undeclared workspace(s) {unknown}")
    return tuple(problems)


def require_host_program_preflight(
    source: str,
    *,
    instance: Any,
    declared_workspaces: Sequence[str],
    where: str = "host/tiling.c",
) -> HostContractSpec:
    """@brief 预检并返回生成的合同（供 portfolio 预检复用）。

    @param source 候选 C 源码。
    @param instance 语义实例。
    @param declared_workspaces 任务合同声明的 workspace 名。
    @param where 诊断位置。
    @return 合同。
    @raises HostProgramError 预检失败。
    """
    spec = host_contract_spec(instance=instance)
    problems = preflight_host_program(
        source, spec=spec, declared_workspaces=declared_workspaces, where=where
    )
    if problems:
        raise HostProgramError("; ".join(problems[:6]))
    return spec


__all__ = [
    "HOST_TILING_SUFFIXES",
    "HostTilingEntry",
    "host_tiling_entry",
    "preflight_host_program",
    "require_host_program_preflight",
]
