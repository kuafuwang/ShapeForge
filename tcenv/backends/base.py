"""Backend protocol for the evaluation layer.

Phase 1 is intentionally small: the Metal adapter delegates to today's functions
and no component is moved.  The protocol draws the boundary that later phases can
migrate behind without changing the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol


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


@dataclass
class BenchHarness:
    rotation_sets: int
    dispatch_note: str
    protocol_note: str
    cand_run: Callable[[int], float]
    cand_serial_run: Callable[[int], float] | None
    baseline_runners: Mapping[str, Callable[[int], float]]
    gpu_samples: list[float]
    spot_check: Callable[[], list[str]]


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
    def validation_problems(self, violations: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]: ...

    # Child-process methods.  The caller supplies the runtime context.
    def set_heartbeat_hooks(self, *, on_compile=None, on_launch=None) -> None: ...
    def context(self): ...
    def compile_error(self): ...
    def runtime_error(self): ...
    def probe(self, ctx: Any, target: Any = None) -> DeviceInfo: ...
    def prepare_case(self, task: Any, seed: int): ...
    def baseline_runners(self, task: Any, data0: Any) -> Mapping[str, Any]: ...
    def calibration(self, target_id: str) -> dict: ...
    def roofline(self, task: Any, time_s: float, calibration: dict) -> dict: ...
    def assemble(self, contract: Any, source: str) -> tuple[str, int]: ...
    def rewrite_compiler_lines(self, contract: Any, message: str, header_lines: int) -> str: ...
    def describe(self, contract: Any) -> str: ...
    def abi_stub(self, contract: Any) -> str: ...
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
    def prepare_correctness(self, ctx: Any, contract: Any, *, tight: bool, spec: Any = None): ...
    def run_correctness(self, ctx: Any, kernel: Any, task: Any, argset: Any, seed: int, *, cases=None, baseline_fns=None, determinism: bool, launcher=None): ...
    def prepare_bench(
        self,
        ctx: Any,
        kernels: list[Any],
        contract: Any,
        task: Any,
        *,
        spec: Any,
        data0: Any,
        reps: int,
        measure_serial: bool,
        base_runners: Mapping[str, Any],
        selected: list[str],
    ) -> BenchHarness: ...
