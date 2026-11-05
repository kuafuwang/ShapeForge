"""Backend-neutral Host Program results shared by native execution and Metal evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class HostInputRead:
    name: str
    tensor_id: int
    dtype: str
    elem_off: int
    count: int
    nbytes: int
    sha256: str | None
    import_name: str | None = None

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tensor_id": int(self.tensor_id),
            "dtype": self.dtype,
            "import": self.import_name,
            "elem_off": int(self.elem_off),
            "count": int(self.count),
            "nbytes": int(self.nbytes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HostWorkspaceSpan:
    name: str
    offset: int
    count: int
    elem_nbytes: int
    data: bytes
    sha256: str | None

    @property
    def nbytes(self) -> int:
        return len(self.data)

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "offset": int(self.offset),
            "count": int(self.count),
            "elem_nbytes": int(self.elem_nbytes),
            "nbytes": int(self.nbytes),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class HostRun:
    """@brief 一次 native Host Program 调用的原子结果。"""

    status: int
    error_code: str | None
    error_detail: str
    host_wall_ns: int
    input_reads: tuple[HostInputRead, ...]
    spans: tuple[HostWorkspaceSpan, ...]
    resolved_scalars: Mapping[str, int]
    audit: bool

    @property
    def ok(self) -> bool:
        return self.status == 0

    def document(self) -> dict[str, Any]:
        return {
            "status": int(self.status),
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "host_tiling_wall_ns": int(self.host_wall_ns),
            "input_reads": [read.document() for read in self.input_reads],
            "workspace_writes": [span.document() for span in self.spans],
            "resolved_scalars": {key: int(value) for key, value in self.resolved_scalars.items()},
            "audit": bool(self.audit),
        }


__all__ = ["HostInputRead", "HostRun", "HostWorkspaceSpan"]
