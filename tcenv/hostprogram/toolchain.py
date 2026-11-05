"""Standard Clang + LLVM ``lld`` → Wasm, with a frozen engine (V3).

There is deliberately no object loader here.  The Host Program is a normal C99 translation unit
compiled by the system compiler and linked by ``wasm-ld``; the resulting module is instantiated
by a maintained Wasm engine (``wasmtime``).  Nothing in this module runs at projection time: the
projection only reads the source and renders the contract header.  The real compiler call happens
in the bounded worker's ``build()`` exactly once per candidate.

The module is validated *after* linking and *before* it is handed to the executor:

* the export set must contain ``memory``, ``host_tiling`` and ``sf_host_abi_offset``;
* every import must be one of the generated typed read functions in module ``sf``.  A candidate
  that declares its own ``__attribute__((import_module(...)))`` extern cannot smuggle a host
  function into the instance: the import list is refused here and again by the executor;
* the module is instantiated once at build time with a bounded memory limit, so a program whose
  static storage exceeds the declared capacity budget fails as a typed candidate error instead of
  consuming the control host's memory at evaluation time.

Fuel and the linear-memory budget scale with the task's declared capacity:

``fuel = 2_000_000 + 512*chains + 64*total_tokens + 1024*max_items``

``memory = mapped_inputs + used_workspace_spans + guards + stack(64 KiB) + 256 KiB + 16*4*max_items``

The coefficients are frozen in V3 from the largest available KDA tier's planner and leave headroom
for the heap/LPT/range-cut planner shipped as the fixture.  They are an upper bound for the
planner's own work, not a performance target; the external watchdog is unchanged and remains the
only defence against a module that never returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

from .contract import (
    HOST_STACK_BYTES,
    INPUT_READ_IMPORTS,
    HostContractSpec,
    assembled_sha256,
    assembled_source,
    storage_layout,
)
from .errors import HostCapabilityUnavailable, HostProgramError
from .precheck import static_precheck

#: Frozen Wasm engine: the plan freezes the engine after the V3 probe.
ENGINE_NAME = "wasmtime"
COMPILER_TIMEOUT_S = 90
#: v8-flavoured clang binaries known to target wasm32 with a real linker.
_CLANG_CANDIDATES: tuple[str, ...] = (
    "/opt/homebrew/opt/llvm/bin/clang",
    "/usr/local/opt/llvm/bin/clang",
    "/opt/homebrew/bin/clang",
    "/usr/bin/clang",
)
_REQUIRED_EXPORTS = ("memory", "host_tiling", "sf_host_abi_offset")
_ALLOWED_IMPORTS = {name for name, _ in INPUT_READ_IMPORTS}


@dataclass(frozen=True)
class HostToolchain:
    """@brief 冻结的 Host Program 工具链（clang/lld + Wasm engine）。"""

    clang: str
    clang_version: str
    wasm_ld: str
    engine: str
    engine_version: str

    def document(self) -> dict[str, Any]:
        return {
            "clang": self.clang,
            "clang_version": self.clang_version,
            "wasm_ld": self.wasm_ld,
            "engine": self.engine,
            "engine_version": self.engine_version,
        }


def _first_line(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:200]
    return "unknown"


def _engine_version() -> str:
    try:
        return str(importlib.metadata.version(ENGINE_NAME))
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover - dependency is declared
        raise HostCapabilityUnavailable(
            f"the Wasm engine {ENGINE_NAME!r} is not installed; run `uv sync` (V3 requires {ENGINE_NAME})"
        ) from exc


def discover_toolchain() -> HostToolchain:
    """@brief 查找并校验 clang/lld/Wasm engine；缺依赖抛 typed ``backend_capability_unavailable``。

    @return 工具链身份。
    @raises HostCapabilityUnavailable 缺 clang 或 wasm-ld 或 engine。
    """
    clang = os.environ.get("SF_HOST_CLANG") or ""
    if clang:
        candidates: Sequence[str] = (clang,)
    else:
        candidates = _CLANG_CANDIDATES
    found = next((path for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)
    if found is None:
        which = shutil.which("clang")
        found = which if which else None
    if found is None:
        raise HostCapabilityUnavailable(
            "no clang found for the restricted C99 Host Program; install LLVM (brew install llvm) or set "
            "SF_HOST_CLANG. The JSON linear host/tiling.json path is unaffected."
        )
    try:
        version = subprocess.run(
            [found, "--version"], capture_output=True, text=True, timeout=20, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostCapabilityUnavailable(f"clang at {found!r} is not runnable: {exc}") from exc
    clang_version = _first_line(version.stdout + version.stderr) if version.returncode == 0 else "unknown"
    wasm_ld = shutil.which("wasm-ld")
    if wasm_ld is None:
        for guess in (
            "/opt/homebrew/opt/llvm/bin/wasm-ld",
            "/opt/homebrew/bin/wasm-ld",
            "/usr/local/opt/llvm/bin/wasm-ld",
            "/usr/local/bin/wasm-ld",
        ):
            if Path(guess).is_file():
                wasm_ld = guess
                break
    if wasm_ld is None:
        raise HostCapabilityUnavailable(
            "wasm-ld is missing (clang can emit a wasm32 object but cannot link the module); install "
            "LLVM/lld (brew install llvm lld). The JSON linear host/tiling.json path is unaffected."
        )
    return HostToolchain(
        clang=str(found),
        clang_version=clang_version,
        wasm_ld=str(wasm_ld),
        engine=ENGINE_NAME,
        engine_version=_engine_version(),
    )


def _shape_values(spec: HostContractSpec) -> dict[str, int]:
    return {field.name: int(field.value) for field in spec.shape_fields if isinstance(field.value, int)}


def max_items(spec: HostContractSpec) -> int:
    """@brief 合同允许的最大工作项数（work_items 双 word / item_counts 容量的上界）。

    @param spec 合同。
    @return 项数（没有该类 workspace 时为 1）。
    """
    best = 1
    for view in spec.workspaces:
        if view.name in ("work_items", "item_counts"):
            per_item = 2 if view.name == "work_items" and view.elem_nbytes == 4 else 1
            best = max(best, view.capacity // per_item)
    return best


def fuel_budget(spec: HostContractSpec) -> int:
    """@brief 按 capacity 缩放的 fuel 上限（V3 冻结系数，见模块文档）。

    @param spec 合同。
    @return fuel 整数。
    """
    values = _shape_values(spec)
    chains = max(1, int(values.get("num_seqs", 1)) * int(values.get("H", 1)))
    tokens = max(1, int(values.get("total_tokens", 1024)))
    return int(2_000_000 + 512 * chains + 64 * tokens + 1024 * max_items(spec))


def memory_budget_bytes(spec: HostContractSpec, used_workspaces: Sequence[str]) -> int:
    """@brief 线性内存上限：映射输入 + 有效 workspace span + 哨兵 + 栈 + planner 余量。

    @param spec 合同。
    @param used_workspaces 实际引用的 workspace。
    @return 字节数（16 字节对齐）。
    """
    regions = storage_layout(spec, used_workspaces)
    storage = max((region.guard_hi + 32 for region in regions), default=0)
    allowance = 256 * 1024 + 16 * 4 * max_items(spec)
    total = storage + HOST_STACK_BYTES + allowance
    page = 64 * 1024
    return (total + page - 1) // page * page


@dataclass(frozen=True)
class HostBuild:
    """@brief 一次 build 期 Host Program 编译产物（Wasm 只在本进程内存里，不进 Schema-4）。"""

    entry_path: str
    program_sha256: str
    source_sha256: str
    assembled_sha256: str
    wasm_sha256: str
    wasm: bytes
    header: str
    used_workspaces: tuple[str, ...]
    fuel: int
    memory_budget_bytes: int
    toolchain: HostToolchain
    #: Process-local Wasm objects created by the build-time validation.  They are *not* identity
    #: (the descriptor hashes the module bytes, not the live object); reusing them in the same
    #: process is what makes "compile once, one Store per call" true.  A different process rebuilds
    #: the module from the source and must reproduce ``wasm_sha256`` exactly.
    engine: Any = field(default=None, repr=False, compare=False)
    module: Any = field(default=None, repr=False, compare=False)

    def descriptor(self) -> dict[str, Any]:
        """@brief 进入 ``NativeExecutable.descriptor["host_program"]`` 的身份字段。

        只放摘要与版本，不放字节；``.wasm`` 不进入 Schema-4 ``ArtifactBundle.entries``，也不放进
        ``BuildOutcome.payloads``。
        """
        # Compact on purpose: the native descriptor is a closed, budgeted document and the
        # four-stage varlen fixture is already close to the bound.  Only identity facts go in;
        # the full header/layout is covered by ``assembled_host_sha256``.
        return {
            "schema_id": "tcenv.hostprogram.native.1",
            "entry_path": self.entry_path,
            "program_sha256": self.program_sha256,
            "assembled_host_sha256": self.assembled_sha256,
            "wasm_sha256": self.wasm_sha256,
            "entry": "host_tiling",
            "used_workspaces": list(self.used_workspaces),
            "fuel": int(self.fuel),
            "memory_bytes": int(self.memory_budget_bytes),
            "engine": self.toolchain.engine,
            "engine_version": self.toolchain.engine_version,
            "compiler": self.toolchain.clang_version,
            "linker": str(Path(self.toolchain.wasm_ld).name),
        }


def _validate_module(module: Any, *, where: str) -> None:
    """@brief 只允许生成的 export 与 ``sf`` 读 import。

    @param module 已解析的 Wasm module。
    @param where 诊断位置。
    @raises HostProgramError export/import 不合规。
    """
    exports = {export.name for export in module.exports}
    missing = [name for name in _REQUIRED_EXPORTS if name not in exports]
    if missing:
        raise HostProgramError(
            f"{where}: the Host Program module does not export {missing}; it must export its own linear "
            "memory, host_tiling and sf_host_abi_offset"
        )
    for item in module.imports:
        if item.module != "sf" or item.name not in _ALLOWED_IMPORTS:
            raise HostProgramError(
                f"{where}: the Host Program declares an unregistered import "
                f"{item.module!r}.{item.name!r}; only the generated sf_input_read_* functions are provided"
            )


def compile_host_program(
    source: str,
    *,
    spec: HostContractSpec,
    entry_path: str,
    toolchain: HostToolchain | None = None,
    used_workspaces: Sequence[str] | None = None,
    timeout_s: int = COMPILER_TIMEOUT_S,
) -> HostBuild:
    """@brief 编译受限 C99 → Wasm；只允许在 build 期的有界 worker 里调用。

    @param source 候选 C 源码。
    @param spec 生成的合同。
    @param entry_path 候选 entry 路径（身份）。
    @param toolchain 已发现的工具链；None 时现场发现。
    @param used_workspaces 需要分配存储的 workspace；None 表示全部（quick/probe 路径）。
    @param timeout_s 编译器超时（有界 worker 之外仍由 supervisor 兜底）。
    @return 构建产物。
    @raises HostProgramError 源码/编译/模块不合规。
    @raises HostCapabilityUnavailable 缺工具链或 engine。
    """
    problems = static_precheck(source)
    if problems:
        raise HostProgramError("host program refused by the static precheck: " + "; ".join(problems[:6]))
    from .contract import referenced_workspaces

    referenced = set(referenced_workspaces(source, spec))
    if used_workspaces is None:
        used = tuple(view.name for view in spec.workspaces if view.name in referenced)
    else:
        used = tuple(str(name) for name in used_workspaces)
        declared = {view.name for view in spec.workspaces}
        unknown = sorted(set(used) - declared)
        if unknown:
            raise HostProgramError(f"host program asked to allocate undeclared workspaces {unknown}")
    toolchain = toolchain or discover_toolchain()
    header = spec.header(used_workspaces=used)
    assembled = assembled_source(header, source)
    digest = assembled_sha256(header, source)
    budget = memory_budget_bytes(spec, used)
    fuel = fuel_budget(spec)
    with tempfile.TemporaryDirectory(prefix="sf-host-") as tmp:
        root = Path(tmp)
        src = root / "host_program.c"
        out = root / "host_program.wasm"
        src.write_text(assembled, encoding="utf-8")
        argv = [
            toolchain.clang,
            "--target=wasm32",
            "-std=c99",
            "-O2",
            "-nostdlib",
            "-fno-builtin",
            "-Wl,--no-entry",
            "-Wl,--export=host_tiling",
            "-Wl,--export=sf_host_abi_offset",
            "-Wl,--export-memory",
            f"-Wl,-z,stack-size={HOST_STACK_BYTES}",
            f"-Wl,--max-memory={budget}",
            "-o",
            str(out),
            str(src),
        ]
        search_path = os.pathsep.join(
            dict.fromkeys(
                [
                    str(Path(toolchain.clang).parent),
                    str(Path(toolchain.wasm_ld).parent),
                    "/opt/homebrew/bin",
                    "/usr/local/bin",
                    "/usr/bin",
                    "/bin",
                ]
            )
        )
        env = {
            "PATH": search_path,
            "TMPDIR": str(root),
            "HOME": str(root),
            "LANG": "C",
        }
        try:
            done = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s, cwd=str(root), env=env, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise HostProgramError(
                f"host program compilation exceeded {timeout_s}s and was stopped; a planner that does not "
                "terminate in the compiler is not a candidate"
            ) from exc
        except OSError as exc:
            raise HostCapabilityUnavailable(f"could not run clang for the Host Program: {exc}") from exc
        if done.returncode != 0:
            diagnostic = (done.stderr or done.stdout or "").strip()
            raise HostProgramError(
                "host program failed to compile:\n" + diagnostic[:4000]
            )
        if not out.is_file():
            raise HostProgramError("clang reported success but produced no .wasm module")
        wasm = out.read_bytes()
    if not wasm.startswith(b"\x00asm"):
        raise HostProgramError("the compiler output is not a Wasm module")
    import wasmtime

    try:
        config = wasmtime.Config()
        config.consume_fuel = True
        engine = wasmtime.Engine(config)
        module = wasmtime.Module(engine, wasm)
        _validate_module(module, where=entry_path)
        store = wasmtime.Store(engine)
        store.set_fuel(fuel)
        store.set_limits(memory_size=budget)
        linker = wasmtime.Linker(engine)
        for import_name, _ in INPUT_READ_IMPORTS:
            linker.define_func(
                "sf",
                import_name,
                wasmtime.FuncType(
                    [wasmtime.ValType.i32()] * 4, [wasmtime.ValType.i32()]
                ),
                lambda _id, _off, _dst, _count: 1,
            )
        linker.instantiate(store, module)
    except HostProgramError:
        raise
    except Exception as exc:  # noqa: BLE001 - engine/validation failure is a candidate build failure
        raise HostProgramError(
            f"the Wasm engine rejected the compiled Host Program ({type(exc).__name__}: {exc}); the module must "
            f"fit the declared capacity budget of {budget} bytes"
        ) from exc
    return HostBuild(
        engine=engine,
        module=module,
        entry_path=str(entry_path),
        program_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        assembled_sha256=digest,
        wasm_sha256=hashlib.sha256(wasm).hexdigest(),
        wasm=wasm,
        header=header,
        used_workspaces=used,
        fuel=fuel,
        memory_budget_bytes=budget,
        toolchain=toolchain,
    )


#: Process-local build cache.  The build compiles the module once; the executor (same bounded
#: worker process) reuses the bytes, and a different process must reproduce the exact module or be
#: refused.  The key is the *full build identity*, not only the compiled bytes: two sources that
#: differ only in a comment compile to the same Wasm but have different assembled/source identity.
#: A cache is not identity evidence: the descriptor digest is.
_BUILT: dict[tuple, HostBuild] = {}


def _build_key(
    wasm_sha256: str,
    assembled_sha256: str,
    program_sha256: str,
    fuel: int,
    memory_budget_bytes: int,
    used_workspaces: Sequence[str],
) -> tuple:
    return (
        str(wasm_sha256),
        str(assembled_sha256),
        str(program_sha256),
        int(fuel),
        int(memory_budget_bytes),
        tuple(str(name) for name in used_workspaces),
    )


def remember_build(build: HostBuild) -> HostBuild:
    """@brief 记住 build 产物（供同进程 executor 复用）。

    @param build 构建产物。
    @return 原对象。
    """
    _BUILT[
        _build_key(
            build.wasm_sha256,
            build.assembled_sha256,
            build.program_sha256,
            build.fuel,
            build.memory_budget_bytes,
            build.used_workspaces,
        )
    ] = build
    return build


def recall_build(
    wasm_sha256: str,
    *,
    assembled_sha256: str | None = None,
    program_sha256: str | None = None,
    fuel: int | None = None,
    memory_budget_bytes: int | None = None,
    used_workspaces: Sequence[str] | None = None,
) -> HostBuild | None:
    """@brief 按完整构建身份取同进程已编译的产物；不匹配时返回 None。

    只按 ``wasm_sha256`` 命中会把另一个 assembled/source 身份的 build 交给调用方；调用方
    随后会因为 assembled 不一致而拒绝一个合法候选。缺少任一身份字段时保守地视为 miss，让
    调用方重编译并用 descriptor 逐字节校验。

    @param wasm_sha256 编译后的 module 摘要。
    @param assembled_sha256 assembled 全文摘要。
    @param program_sha256 source 摘要。
    @param fuel fuel 上限。
    @param memory_budget_bytes 线性内存上限。
    @param used_workspaces 实际分配的 workspace。
    @return build 或 None。
    """
    if None in (assembled_sha256, program_sha256, fuel, memory_budget_bytes, used_workspaces):
        return None
    key = _build_key(
        wasm_sha256,
        assembled_sha256,
        program_sha256,
        int(fuel),
        int(memory_budget_bytes),
        used_workspaces,
    )
    return _BUILT.get(key)


__all__ = [
    "COMPILER_TIMEOUT_S",
    "ENGINE_NAME",
    "HostBuild",
    "HostToolchain",
    "compile_host_program",
    "discover_toolchain",
    "fuel_budget",
    "max_items",
    "memory_budget_bytes",
    "recall_build",
    "remember_build",
]
