"""Restricted C99 → worker-local native64 dylib.

No Wasm engine, linker, module or Store participates in this path.  Compilation and ``ctypes.CDLL``
are worker-only operations.  The control host receives the descriptor and typed execution results,
never a native address.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
from typing import Sequence

from .contract import HostContractSpec, storage_layout
from .errors import HostCapabilityUnavailable, HostProgramError
from .native_abi import (
    NATIVE_ABI_SCHEMA,
    assemble_native_source,
    native_assembled_sha256,
    render_native_support,
)
from .precheck import static_precheck

COMPILER_TIMEOUT_S = 90
_REQUIRED_EXPORTS = ("sf_native_invoke", "sf_native_before_call", "sf_native_guard_status")


@dataclass(frozen=True)
class NativeToolchain:
    clang: str
    clang_version: str
    execution_backend: str = "native-c99"
    platform: str = field(default_factory=lambda: f"{platform.system()}-{platform.machine()}")

    def document(self) -> dict[str, str]:
        return {
            "clang": self.clang,
            "clang_version": self.clang_version,
            "execution_backend": self.execution_backend,
            "platform": self.platform,
        }


def discover_native_toolchain() -> NativeToolchain:
    if platform.system() != "Darwin":
        raise HostCapabilityUnavailable(
            "the current native Host Program loader uses a Darwin dylib and `size -m`; "
            "this platform needs a separately validated native toolchain"
        )
    configured = os.environ.get("SF_HOST_CLANG", "")
    candidates = (configured,) if configured else ("/usr/bin/clang", "/opt/homebrew/opt/llvm/bin/clang", "/usr/local/opt/llvm/bin/clang")
    clang = next((path for path in candidates if path and Path(path).is_file() and os.access(path, os.X_OK)), None)
    if clang is None:
        clang = shutil.which("clang")
    if clang is None:
        raise HostCapabilityUnavailable(
            "no native C99 clang found; install Command Line Tools or LLVM, or set SF_HOST_CLANG"
        )
    try:
        done = subprocess.run([clang, "--version"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostCapabilityUnavailable(f"native clang at {clang!r} is not runnable: {exc}") from exc
    if done.returncode != 0:
        raise HostCapabilityUnavailable(f"native clang at {clang!r} failed --version: {done.stderr[:500]}")
    version = next((line.strip() for line in (done.stdout + done.stderr).splitlines() if line.strip()), "unknown")
    return NativeToolchain(clang=str(clang), clang_version=version[:200])


def native_memory_budget_bytes(spec: HostContractSpec, used_workspaces: Sequence[str]) -> int:
    """@brief 声明性 worker 存储预算；真正的 runaway 边界仍是外部进程 watchdog。"""
    regions = storage_layout(spec, used_workspaces)
    storage = max((region.guard_hi + 32 for region in regions), default=0)
    max_items = 1
    for view in spec.workspaces:
        if view.name in ("work_items", "item_counts"):
            per_item = 2 if view.name == "work_items" and view.elem_nbytes == 4 else 1
            max_items = max(max_items, view.capacity // per_item)
    total = storage + 64 * 1024 + 256 * 1024 + 16 * 4 * max_items
    page = 64 * 1024
    return (total + page - 1) // page * page


@dataclass(frozen=True)
class NativeHostBuild:
    entry_path: str
    program_sha256: str
    source_sha256: str
    assembled_sha256: str
    native_sha256: str
    native: bytes
    header: str
    library_path: str
    used_workspaces: tuple[str, ...]
    memory_budget_bytes: int
    toolchain: NativeToolchain
    library: object | None = field(default=None, repr=False, compare=False)

    def descriptor(self) -> dict[str, object]:
        return {
            "schema_id": NATIVE_ABI_SCHEMA,
            "entry_path": self.entry_path,
            "program_sha256": self.program_sha256,
            "assembled_host_sha256": self.assembled_sha256,
            "native_sha256": self.native_sha256,
            "entry": "sf_native_invoke",
            "used_workspaces": list(self.used_workspaces),
            "memory_bytes": int(self.memory_budget_bytes),
            "execution_backend": self.toolchain.execution_backend,
            "platform": self.toolchain.platform,
            "compiler": self.toolchain.clang_version,
        }


_BUILT: dict[tuple[object, ...], NativeHostBuild] = {}


def _cache_key(
    native_sha256: str,
    *,
    assembled_sha256: str | None = None,
    program_sha256: str | None = None,
    memory_budget_bytes: int | None = None,
    used_workspaces: tuple[str, ...] | None = None,
) -> tuple[object, ...]:
    return (
        str(native_sha256), assembled_sha256, program_sha256,
        None if memory_budget_bytes is None else int(memory_budget_bytes), used_workspaces,
    )


def remember_native_build(build: NativeHostBuild) -> NativeHostBuild:
    _BUILT[_cache_key(
        build.native_sha256,
        assembled_sha256=build.assembled_sha256,
        program_sha256=build.program_sha256,
        memory_budget_bytes=build.memory_budget_bytes,
        used_workspaces=build.used_workspaces,
    )] = build
    return build


def recall_native_build(
    native_sha256: str,
    *,
    assembled_sha256: str | None = None,
    program_sha256: str | None = None,
    memory_budget_bytes: int | None = None,
    used_workspaces: tuple[str, ...] | None = None,
) -> NativeHostBuild | None:
    if None in (assembled_sha256, program_sha256, memory_budget_bytes, used_workspaces):
        return None
    return _BUILT.get(_cache_key(
        native_sha256,
        assembled_sha256=assembled_sha256,
        program_sha256=program_sha256,
        memory_budget_bytes=memory_budget_bytes,
        used_workspaces=used_workspaces,
    ))


def _validate_link_boundary(library: Path, *, entry_path: str, memory_budget_bytes: int) -> None:
    """@brief 最终二进制不得有未解析宿主调用、候选自定义导出或超额静态存储。"""
    nm = shutil.which("nm")
    size = shutil.which("size")
    if nm is None or size is None:
        raise HostCapabilityUnavailable("native Host Program validation requires `nm` and `size` beside Clang")
    try:
        undefined = subprocess.run([nm, "-u", str(library)], capture_output=True, text=True, timeout=20, check=False)
        exported = subprocess.run([nm, "-gU", str(library)], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostProgramError(f"native Host Program symbol validation failed: {exc}") from exc
    if undefined.returncode != 0 or exported.returncode != 0:
        raise HostProgramError(
            f"{entry_path}: nm could not inspect the native dylib: {undefined.stderr or exported.stderr}"
        )
    unresolved = [line.split()[-1] for line in undefined.stdout.splitlines() if line.split()]
    if unresolved:
        raise HostProgramError(
            f"{entry_path}: native Host Program has unresolved host symbols {unresolved[:12]}; "
            "only the generated self-contained ABI and typed callbacks are allowed"
        )
    unexpected = []
    for line in exported.stdout.splitlines():
        fields = line.split()
        if fields and not fields[-1].startswith("_sf_native_"):
            unexpected.append(fields[-1])
    if unexpected:
        raise HostProgramError(
            f"{entry_path}: native Host Program exports non-harness symbols {unexpected[:12]}; "
            "candidate functions must remain worker-private"
        )
    try:
        measured = subprocess.run([size, "-m", str(library)], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostProgramError(f"native Host Program static-storage validation failed: {exc}") from exc
    if measured.returncode != 0:
        raise HostProgramError(f"{entry_path}: size could not inspect the native dylib: {measured.stderr}")
    in_data = False
    data_bytes: int | None = None
    for line in measured.stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("Segment "):
            in_data = stripped.startswith("Segment __DATA:")
        elif in_data and stripped.startswith("total "):
            data_bytes = int(stripped.split()[1])
            break
    if data_bytes is None or data_bytes > int(memory_budget_bytes):
        raise HostProgramError(
            f"{entry_path}: native Host Program static storage {data_bytes!r} exceeds the declared "
            f"memory budget {memory_budget_bytes} bytes"
        )


def compile_native_host_program(
    source: str,
    *,
    spec: HostContractSpec,
    entry_path: str,
    toolchain: NativeToolchain | None = None,
    used_workspaces: Sequence[str] | None = None,
    timeout_s: int = COMPILER_TIMEOUT_S,
) -> NativeHostBuild:
    """@brief 在 worker 内编译并验证 native Host Program；绝不返回可执行指针给控制机。"""
    problems = static_precheck(source)
    if problems:
        raise HostProgramError("host program refused by the static precheck: " + "; ".join(problems[:6]))
    from .contract import referenced_workspaces

    referenced = set(referenced_workspaces(source, spec))
    if used_workspaces is None:
        used = tuple(view.name for view in spec.workspaces if view.name in referenced)
    else:
        used = tuple(dict.fromkeys(str(name) for name in used_workspaces))
        unknown = sorted(set(used) - {view.name for view in spec.workspaces})
        if unknown:
            raise HostProgramError(f"host program asked to allocate undeclared workspaces {unknown}")
    toolchain = toolchain or discover_native_toolchain()
    header = render_native_support(spec, used)
    assembled = assemble_native_source(header, source)
    digest = native_assembled_sha256(header, source)
    budget = native_memory_budget_bytes(spec, used)
    # This directory intentionally outlives the build call: dlopen keeps the mapped file alive.
    # It is private to the worker and never serialized as an ArtifactBundle entry/payload.
    root = Path(tempfile.mkdtemp(prefix="sf-native-worker-"))
    src = root / "host_program.c"
    out = root / "host.dylib"
    src.write_text(assembled, encoding="utf-8")
    argv = [
        toolchain.clang, "-std=c99", "-O2", "-fPIC", "-shared", "-nostdlib",
        "-fno-builtin", "-fno-stack-protector", "-fvisibility=hidden",
        f"-ffile-prefix-map={root}=.", "-Wl,-undefined,error",
        "-Wl,-install_name,@rpath/shapeforge_host.dylib",
        str(src), "-o", str(out), "-lSystem",
    ]
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=int(timeout_s), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise HostProgramError(f"native Host Program compiler did not finish: {exc}") from exc
    if done.returncode != 0:
        raise HostProgramError(
            "native Host Program compilation failed: " + (done.stderr or done.stdout)[:8000]
        )
    native = out.read_bytes()
    if not native:
        raise HostProgramError("native Host Program produced an empty dylib")
    _validate_link_boundary(out, entry_path=entry_path, memory_budget_bytes=budget)
    try:
        library = ctypes.CDLL(str(out))
        missing = [name for name in _REQUIRED_EXPORTS if not hasattr(library, name)]
    except OSError as exc:
        raise HostProgramError(f"native Host Program dylib could not be loaded in the worker: {exc}") from exc
    if missing:
        raise HostProgramError(f"native Host Program is missing generated exports {missing}")
    build = NativeHostBuild(
        entry_path=str(entry_path),
        program_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        assembled_sha256=digest,
        native_sha256=hashlib.sha256(native).hexdigest(),
        native=native,
        header=header,
        library_path=str(out),
        used_workspaces=tuple(used),
        memory_budget_bytes=budget,
        toolchain=toolchain,
        library=library,
    )
    return remember_native_build(build)


__all__ = [
    "NativeHostBuild",
    "NativeToolchain",
    "compile_native_host_program",
    "discover_native_toolchain",
    "native_memory_budget_bytes",
    "recall_native_build",
    "remember_native_build",
]
