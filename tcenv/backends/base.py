"""Backend protocol for the evaluation layer.

Phase 1 is intentionally small: the Metal adapter delegates to today's functions
and no component is moved.  The protocol draws the boundary that later phases can
migrate behind without changing the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


@dataclass(frozen=True)
class DeviceInfo:
    backend: str
    name: str
    family: str
    limits: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Capabilities:
    supports_launch_spec: bool = False
    supports_validation: bool = False
    supports_device_timestamps: bool = False
    supports_profiler: bool = False


class Backend(Protocol):
    id: str
    version: str
    language: str
    file_ext: str
    capabilities: Capabilities

    # Parent-process methods must be CPU/text only.  They may import the
    # backend module, but must not initialise the device runtime.
    def screen(self, source: str, task: Any = None) -> list[str]: ...
    def parse_launch(self, source: str, contract: Any) -> Any | None: ...
    def validation_env(self, base_env: Mapping[str, str]) -> dict[str, str]: ...
    def parse_validation_stderr(self, stderr: str) -> list[str]: ...
    def is_instrumented_size_only(self, violations: list[str]) -> bool: ...

    # Child-process methods.  The caller supplies the runtime context.
    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None: ...
    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo: ...
    def compile(self, ctx: Any, source: str, contract: Any, *, math: str = "fast"): ...
    def compile_all(self, ctx: Any, source: str, contract: Any, *, names: list[str] | None = None, math: str = "fast"): ...
    def check_abi(
        self,
        contract: Any,
        kernels: list[Any],
        *,
        max_tg_memory: int,
        max_buffer_index: int,
        target_id: str,
        spec: Any = None,
    ) -> list[str]: ...
    def pipeline_stats(self, kernel: Any, contract: Any, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict: ...
    def pipeline_stats_many(self, pairs: list, device_info: Mapping[str, Any], *, kernel_source: str | None = None) -> dict: ...
