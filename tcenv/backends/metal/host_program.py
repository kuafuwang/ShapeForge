"""Metal-side binding of the restricted C99 / Wasm Host Program (V3).

The neutral runtime (``tcenv.hostprogram.runtime``) owns the Wasm Module/Store; this module only
knows the Metal workspace buffers.  It is the seam where the plan's Host Program ABI meets the v2
runtime:

* the build descriptor ``NativeExecutable.descriptor["host_program"]`` binds the module digest,
  compiler/engine versions and fuel/memory bounds into ``native_digest``;
* the executor reuses the module compiled in this bounded worker, or (a different process) rebuilds
  it from the candidate source and refuses unless the bytes/assembled digest reproduce the
  descriptor exactly;
* a failed Wasm run produces **no** workspace bytes: :class:`HostProgramFailed` carries the typed
  run document and the caller turns it into a candidate failure;
* audit runs copy the valid span and hash it; timed runs only copy the span and add the host wall
  time to the measured window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import time
from typing import Any, Mapping, Sequence

from ...hostprogram.contract import (
    STORAGE_GUARD_BYTES,
    HostContractSpec,
    assembled_sha256,
    host_contract_spec,
    referenced_workspaces,
    storage_layout,
)
from ...hostprogram.delivery import HostTilingEntry, host_tiling_entry
from ...hostprogram.errors import HostProgramError
from ...hostprogram.runtime import HostRun, HostRuntime
from ...hostprogram.toolchain import (
    HostBuild,
    compile_host_program,
    discover_toolchain,
    fuel_budget,
    memory_budget_bytes,
    recall_build,
    remember_build,
)

HOST_PROGRAM_RUN_SCHEMA_ID = "tcenv.hostprogram.run.1"
HOST_PROGRAM_RUN_SCHEMA_VERSION = 1


class HostProgramFailed(RuntimeError):
    """@brief 一次 host run 返回非 0 / trap / fuel 耗尽 / 越界；没有任何部分写入。"""

    def __init__(self, run: HostRun) -> None:
        self.run = run
        super().__init__(run.error_detail or f"host program failed with status {run.status}")

    @property
    def code(self) -> str:
        return self.run.error_code or "host_program_invalid"

    def document(self) -> dict[str, Any]:
        return self.run.document()


@dataclass(frozen=True)
class HostProgramTiming:
    """@brief 计时路径的最小返回值（不算哈希、不建 JSON 文档）。"""

    host_wall_ns: int
    wasm_wall_ns: int
    fuel_used: int


def _target_values(target: Any) -> dict[str, int]:
    raw = dict(getattr(target, "raw_limits", {}) or {})
    return {
        "max_buffer_index": int(raw.get("max_buffer_index", 0)),
        "max_threads_per_threadgroup": int(raw.get("max_threads_per_threadgroup", 0)),
        "max_threadgroup_memory_bytes": int(raw.get("max_threadgroup_memory_bytes", 0)),
    }


@dataclass
class MetalHostProgram:
    """@brief 一个候选 Host Program 在 Metal 执行面的完整绑定。"""

    entry: HostTilingEntry
    build: HostBuild
    spec: HostContractSpec
    target_id: str
    target_values: Mapping[str, int]
    native_digest: str | None
    inside_timing: bool
    input_dependent: bool
    preprocesses_inputs: bool
    candidate_sha256: str
    zero_flags: Mapping[str, bool] = field(default_factory=dict)
    rebuilt: bool = False
    #: Process-local call counters for the timing evidence (never identity).
    calls: int = 0
    audit_calls: int = 0
    _runtime: HostRuntime | None = field(default=None, repr=False, compare=False)
    _zero_flags: Mapping[str, bool] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def source_entry(cls, candidate: Any) -> HostTilingEntry | None:
        """@brief 取候选的 ``.c`` Host Program entry；JSON linear 返回 None。"""
        entry = host_tiling_entry(candidate)
        if entry is None or entry.kind != "program":
            return None
        return entry

    @classmethod
    def build_for(
        cls,
        candidate: Any,
        *,
        instance: Any,
        tensors: Sequence[Any],
        target: Any,
        target_id: str,
        native_digest: str | None,
        inside_timing: bool,
        input_dependent: bool,
        preprocesses_inputs: bool,
        geometry: Any,
        toolchain: Any | None = None,
    ) -> "MetalHostProgram | None":
        """@brief 构建期编译一次 Host Program；没有 ``.c`` entry 时返回 None。

        @param candidate 候选。
        @param instance 语义实例。
        @param tensors ABI 张量（与题面使用同一份形状解析）。
        @param target 目标档案。
        @param target_id 目标 id。
        @param native_digest 已绑定的 native 摘要（descriptor 里还没写 host_program）。
        @param inside_timing lifecycle 的计时归属（必须为 True）。
        @param input_dependent lifecycle 的输入依赖（必须为 True）。
        @param preprocesses_inputs lifecycle 的预处理声明（必须为 True）。
        @param geometry 已求值几何（用于拒绝不可绑定的 workspace）。
        @param toolchain 已发现的工具链；None 时现场发现。
        @return 绑定或 None。
        @raises HostProgramError 交付/ABI/编译/生命周期不合法。
        """
        entry = cls.source_entry(candidate)
        if entry is None:
            return None
        spec = host_contract_spec(instance=instance, tensors=tensors)
        declared = {view.name for view in spec.workspaces}
        referenced = referenced_workspaces(entry.source, spec)
        unknown = sorted(set(referenced) - declared)
        if unknown:
            raise HostProgramError(f"{entry.path}: references undeclared workspace(s) {unknown}")
        possible = {use.name for stage in geometry.stages for use in stage.workspace}
        unbound = sorted(set(referenced) - possible)
        if unbound:
            raise HostProgramError(
                f"{entry.path}: writes workspace(s) {unbound} that no possible stage binds; refuse instead of "
                "allocating a block nothing can read (the possible-stage union is the allocation scope)"
            )
        if not input_dependent or not preprocesses_inputs or not inside_timing:
            raise HostProgramError(
                f"{entry.path}: a Host Program that can read input tensors must be declared "
                "input_dependent=true + preprocesses_inputs=true + inside_timing=true; the timed window may not "
                "be bypassed by declaring the program input-independent"
            )
        build = compile_host_program(entry.source, spec=spec, entry_path=entry.path, toolchain=toolchain)
        remember_build(build)
        return cls(
            entry=entry,
            build=build,
            spec=spec,
            target_id=str(target_id),
            target_values=_target_values(target),
            native_digest=native_digest,
            inside_timing=True,
            input_dependent=True,
            preprocesses_inputs=True,
            candidate_sha256=str(candidate.candidate_sha256),
        )

    @classmethod
    def from_descriptor(
        cls,
        candidate: Any,
        *,
        instance: Any,
        tensors: Sequence[Any],
        target: Any,
        target_id: str,
        descriptor: Mapping[str, Any],
        native_digest: str | None,
        inside_timing: bool,
        input_dependent: bool,
        preprocesses_inputs: bool,
    ) -> "MetalHostProgram | None":
        """@brief 执行期由 descriptor 重建绑定；同进程复用已编译 module，否则逐字节重编译校验。

        @param candidate 候选。
        @param instance 语义实例。
        @param tensors ABI 张量。
        @param target 目标档案。
        @param target_id 目标 id。
        @param descriptor ``native.descriptor["host_program"]``。
        @param native_digest native 摘要。
        @param inside_timing/input_dependent/preprocesses_inputs lifecycle 声明。
        @return 绑定或 None。
        @raises HostProgramError descriptor 与候选/源码/模块不一致。
        """
        entry = cls.source_entry(candidate)
        if entry is None:
            if descriptor:
                raise HostProgramError(
                    "the native descriptor records a Host Program but the candidate has no host/tiling.c entry"
                )
            return None
        if not descriptor:
            raise HostProgramError(
                f"{entry.path}: the candidate ships a C99 Host Program but the build descriptor has no "
                "host_program record; a module without build identity cannot be executed"
            )
        if str(descriptor.get("entry_path", "")) != entry.path:
            raise HostProgramError(
                f"the host_program descriptor names {descriptor.get('entry_path')!r} but the candidate entry is "
                f"{entry.path!r}"
            )
        if str(descriptor.get("program_sha256", "")) != entry.sha256:
            raise HostProgramError(
                f"{entry.path}: the candidate bytes do not match the program_sha256 recorded at build time"
            )
        if not inside_timing or not input_dependent or not preprocesses_inputs:
            raise HostProgramError(
                f"{entry.path}: a Host Program that can read input tensors must stay inside the timed window with "
                "input_dependent=true and preprocesses_inputs=true"
            )
        spec = host_contract_spec(instance=instance, tensors=tensors)
        used = tuple(str(name) for name in descriptor.get("used_workspaces", ()))
        expected_wasm = str(descriptor.get("wasm_sha256", ""))
        expected_assembled = str(descriptor.get("assembled_host_sha256", ""))
        build = recall_build(expected_wasm)
        rebuilt = False
        if build is None:
            build = compile_host_program(
                entry.source, spec=spec, entry_path=entry.path, used_workspaces=used
            )
            remember_build(build)
            rebuilt = True
            if build.wasm_sha256 != expected_wasm or build.assembled_sha256 != expected_assembled:
                raise HostProgramError(
                    f"{entry.path}: rebuilding the Host Program produced a different module "
                    f"({build.wasm_sha256[:12]} vs {expected_wasm[:12]}); the build identity is not reproducible "
                    "on this toolchain"
                )
        if build.wasm_sha256 != expected_wasm:
            raise HostProgramError(
                f"{entry.path}: the reused module's wasm_sha256 does not match the build descriptor"
            )
        if build.assembled_sha256 != expected_assembled:
            raise HostProgramError(
                f"{entry.path}: the reused module's assembled_host_sha256 does not match the build descriptor"
            )
        if int(descriptor.get("fuel", build.fuel)) != build.fuel:
            raise HostProgramError(f"{entry.path}: the descriptor's fuel bound does not match the rebuilt module")
        return cls(
            entry=entry,
            build=build,
            spec=spec,
            target_id=str(target_id),
            target_values=_target_values(target),
            native_digest=native_digest,
            inside_timing=True,
            input_dependent=True,
            preprocesses_inputs=True,
            candidate_sha256=str(candidate.candidate_sha256),
            rebuilt=rebuilt,
        )

    # --- execution ---------------------------------------------------------------------------

    def runtime(self) -> HostRuntime:
        """@brief 构造/复用 Wasm 执行器（Module 只编译一次）。

        @return 执行器。
        """
        if self._runtime is not None:
            return self._runtime
        runtime = HostRuntime(self.build, self.spec)
        runtime.set_target(self.target_values)
        self._runtime = runtime
        return runtime

    @property
    def written_workspaces(self) -> frozenset[str]:
        """@brief 可能被 host 写入的 workspace（源码引用的、且已分配的块）。"""
        return frozenset(self.build.used_workspaces)

    def _reader(self, argset: Any):
        buffers = argset.buffers

        def read(name: str, byte_off: int, nbytes: int) -> bytes:
            gb = buffers.get(name)
            if gb is None:
                raise HostProgramError(f"host program requested input {name!r} that the ABI does not expose")
            if byte_off < 0 or nbytes < 0 or byte_off + nbytes > gb.nbytes:
                raise HostProgramError(
                    f"host program requested bytes [{byte_off}, {byte_off + nbytes}) of input {name!r}, "
                    f"but the ABI span is {gb.nbytes} bytes"
                )
            start = gb.guard + byte_off
            return bytes(gb.mem[start : start + nbytes])

        return read

    def execute(self, argset: Any, *, audit: bool) -> dict[str, Any] | HostProgramTiming:
        """@brief 执行一次 host run；成功后才把有效 span 写进真实 workspace。

        @param argset ``V2ArgSet``（只读输入 + workspace 缓冲）。
        @param audit True 时记录完整 HostRunEvidence；计时路径只返回耗时。
        @return 审计文档或计时三元组。
        @raises HostProgramFailed 非 0/trap/fuel/越界；此时没有任何 workspace 字节被提交。
        """
        self._zero_flags = dict(getattr(argset, "workspace_zero", {}) or {})
        self.calls += 1
        if audit:
            self.audit_calls += 1
        started = time.perf_counter_ns()
        try:
            run = self.runtime().execute(self._reader(argset), audit=audit)
        except HostProgramError as exc:
            # A layout/span/guard/input violation is a candidate failure too: convert it into the
            # same typed failure document so the correctness runner reports `incorrect` instead of
            # crashing the evaluator.
            raise HostProgramFailed(
                HostRun(
                    status=-2,
                    error_code=exc.code,
                    error_detail=str(exc),
                    host_wall_ns=time.perf_counter_ns() - started,
                    fuel_used=0,
                    input_reads=(),
                    spans=(),
                    resolved_scalars={},
                    audit=bool(audit),
                )
            ) from exc
        if not run.ok:
            raise HostProgramFailed(run)
        # Runtime net for workspace ownership: even if the static name scan missed an alias and
        # the envelope could not refuse a host-write/device-zero conflict at build time, a non-empty
        # span for a device-zeroed block is refused here, before any byte reaches the device.
        zeroed = {name for name, zero in getattr(argset, "workspace_zero", {}).items() if zero}
        offending = sorted({span.name for span in run.spans if span.count and span.name in zeroed})
        if offending:
            raise HostProgramFailed(
                HostRun(
                    status=-2,
                    error_code="host_program_invalid",
                    error_detail=(
                        f"host program wrote device-zeroed workspace(s) {offending}; host-written and "
                        "device-zeroed initialization of one block are mutually exclusive"
                    ),
                    host_wall_ns=run.host_wall_ns,
                    fuel_used=run.fuel_used,
                    input_reads=run.input_reads,
                    spans=(),
                    resolved_scalars={},
                    audit=bool(audit),
                )
            )
        # Validate every span before copying any of them: a later bad span must not leave an
        # earlier span committed (the run's atomicity is all-or-nothing).
        placements: list[tuple[Any, int, int, Any]] = []
        for span in run.spans:
            gb = argset.workspace(span.name)
            elements = span.elem_nbytes
            start = gb.guard + span.offset * elements
            stop = start + len(span.data)
            if stop > gb.guard + gb.nbytes:
                raise HostProgramFailed(
                    HostRun(
                        status=-2,
                        error_code="host_program_invalid",
                        error_detail=f"workspace {span.name!r} span leaves the allocated Metal buffer",
                        host_wall_ns=run.host_wall_ns,
                        fuel_used=run.fuel_used,
                        input_reads=run.input_reads,
                        spans=(),
                        resolved_scalars={},
                        audit=bool(audit),
                    )
                )
            placements.append((gb, start, stop, span))
        writes: list[dict[str, Any]] = []
        for gb, start, stop, span in placements:
            gb.mem[start:stop] = span.data
            if audit:
                writes.append(span.document())
        wall = time.perf_counter_ns() - started
        if not audit:
            return HostProgramTiming(host_wall_ns=wall, wasm_wall_ns=run.host_wall_ns, fuel_used=run.fuel_used)
        policy: dict[str, str] = {}
        written = {span.name for span in run.spans if span.count > 0}
        for name, zero in getattr(argset, "workspace_zero", {}).items():
            if name in written:
                policy[name] = "host"
            else:
                policy[name] = "declared-zero" if zero else "poison-sentinel"
        return self._run_document(
            run,
            writes=writes,
            policy=policy,
            timing_mode="audit",
            typed_error=None,
            host_wall_ns=wall,
        )

    def failure_document(self, run: HostRun) -> dict[str, Any]:
        """@brief 失败 run 的完整 HostRunEvidence（无 workspace_writes、无 partial span）。

        @param run 失败结果。
        @return 与成功 audit 同一字段集，``typed_error`` 记录失败码。
        """
        policy: dict[str, str] = {}
        for name, zero in self._zero_flags.items():
            policy[name] = "declared-zero" if zero else "poison-sentinel"
        return self._run_document(
            run,
            writes=[],
            policy=policy,
            timing_mode="audit",
            typed_error=run.error_code or "host_program_invalid",
            host_wall_ns=run.host_wall_ns,
        )

    def _run_document(
        self,
        run: HostRun,
        *,
        writes: list[dict[str, Any]],
        policy: Mapping[str, str],
        timing_mode: str,
        typed_error: str | None,
        host_wall_ns: int,
    ) -> dict[str, Any]:
        """@brief 组装完整 HostRunEvidence（成功/失败共用同一字段集）。"""
        return {
            "schema_id": HOST_PROGRAM_RUN_SCHEMA_ID,
            "schema_version": HOST_PROGRAM_RUN_SCHEMA_VERSION,
            "candidate_sha256": self.candidate_sha256,
            "backend_id": "metal",
            "target_id": self.target_id,
            "native_digest": self.native_digest,
            "entry_path": self.entry.path,
            "program_sha256": self.build.program_sha256,
            "assembled_host_sha256": self.build.assembled_sha256,
            "wasm_module_digest": self.build.wasm_sha256,
            "engine": self.build.toolchain.engine,
            "engine_version": self.build.toolchain.engine_version,
            "compiler": self.build.toolchain.clang,
            "compiler_version": self.build.toolchain.clang_version,
            "input_reads": [item.document() for item in run.input_reads],
            "workspace_writes": writes,
            "unwritten_policy": dict(policy),
            "resolved_scalars": {key: int(value) for key, value in run.resolved_scalars.items()},
            # Dyn-0 has a frozen execution geometry and no runtime LaunchGeometry; the runtime
            # fills the frozen identity in after the call, and both Dyn-1 digests stay None.
            "geometry_digest": None,
            "resolved_envelope_digest": None,
            "launch_geometry_digest": None,
            "input_dependent": True,
            "preprocesses_inputs": True,
            "inside_timing": bool(self.inside_timing),
            "host_status": int(run.status),
            "typed_error": typed_error,
            "error_detail": "" if run.ok else str(run.error_detail),
            "host_tiling_wall_ns": int(host_wall_ns),
            "wasm_wall_ns": int(run.host_wall_ns),
            "fuel_used": int(run.fuel_used),
            "fuel_budget": int(self.build.fuel),
            "device_kernel_ns": None,
            "end_to_end_wall_ns": None,
            "timing_mode": timing_mode,
            "simulated": False,
            "publication_allowed": False,
            "publication_gate": "fresh_confirmation",
        }

    def document(self) -> dict[str, Any]:
        """@brief 构建文档（进入 ``NativeExecutable.descriptor["host_program"]``）。"""
        descriptor = dict(self.build.descriptor())
        descriptor["target_id"] = self.target_id
        descriptor["layout"] = {
            "abi_bytes": int(self.spec.layout().abi_bytes),
            "storage_bytes": int(
                max(
                    (r.guard_hi + STORAGE_GUARD_BYTES for r in storage_layout(self.spec, self.build.used_workspaces)),
                    default=0,
                )
            ),
            "mapped_inputs": sum(1 for view in self.spec.inputs if view.mapped),
            "metadata_inputs": sum(1 for view in self.spec.inputs if not view.mapped),
        }
        return descriptor


# Re-export the descriptor helper so callers do not import the toolchain module for one name.
def descriptor_digest(descriptor: Mapping[str, Any]) -> str:
    """@brief canonical descriptor 的 sha256（供 NPU parity / 审计复算）。

    @param descriptor host_program descriptor。
    @return 十六进制摘要。
    """
    raw = json.dumps(dict(descriptor), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


__all__ = [
    "HOST_PROGRAM_RUN_SCHEMA_ID",
    "HOST_PROGRAM_RUN_SCHEMA_VERSION",
    "HostProgramFailed",
    "HostProgramTiming",
    "MetalHostProgram",
    "assembled_sha256",
    "descriptor_digest",
    "discover_toolchain",
    "fuel_budget",
    "memory_budget_bytes",
]
