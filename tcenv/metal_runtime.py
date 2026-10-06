"""PyObjC wrapper over Metal: compile MSL, reflect bindings, allocate buffers, dispatch, time.

This is the analogue of ``ptxas`` + the CUDA driver in the paper: it turns candidate
source text into a loaded pipeline and launches it with a fixed contract.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import objc
import Metal

# PyObjC does not know that ``reflection`` and ``error`` of this selector are out
# parameters; without this registration the reflection object is silently dropped.
objc.registerMetaDataForSelector(
    b"NSObject",
    b"newComputePipelineStateWithFunction:options:reflection:error:",
    {
        "arguments": {
            4: {"type_modifier": objc._C_OUT, "null_accepted": True},
            5: {"type_modifier": objc._C_OUT, "null_accepted": True},
        }
    },
)


class MetalCompileError(Exception):
    """MSL failed to compile or the pipeline could not be created."""


class MetalRuntimeError(Exception):
    """The GPU reported an error while executing a command buffer."""


@dataclass
class Binding:
    name: str
    index: int
    kind: str  # "buffer" | "threadgroup" | "other"
    dtype_code: int | None  # MTLDataType of the pointee / scalar
    access: str  # "read" | "readwrite" | "write"
    elem_size: int | None
    used: bool
    tg_bytes: int | None = None


@dataclass
class CompiledKernel:
    name: str
    pso: object
    bindings: list[Binding]
    max_total_threads: int
    execution_width: int
    static_tg_bytes: int
    compile_s: float
    pipeline_s: float
    language_version: str = ""
    # keep the library alive as long as the pipeline is used
    library: object = None


@dataclass(frozen=True)
class BufferArg:
    buf: object  # MTLBuffer
    offset: int = 0


@dataclass(frozen=True)
class ScalarArg:
    data: bytes


@dataclass
class RunTiming:
    wall_s: float  # commit -> waitUntilCompleted
    gpu_s: float  # GPUEndTime - GPUStartTime for the whole command buffer
    iters: int

    @property
    def wall_per_iter_s(self) -> float:
        return self.wall_s / self.iters

    @property
    def gpu_per_iter_s(self) -> float:
        return self.gpu_s / self.iters


# Optional (begin, end) callbacks around every GPU launch. The worker uses them to publish a heartbeat so that the
# parent process can kill a runaway kernel after seconds instead of minutes (a stuck GPU also starves the display).
_LAUNCH_HOOKS: tuple | None = None


def set_launch_hooks(begin, end) -> None:
    global _LAUNCH_HOOKS
    _LAUNCH_HOOKS = (begin, end)


_ACCESS = {
    Metal.MTLBindingAccessReadOnly: "read",
    Metal.MTLBindingAccessReadWrite: "readwrite",
    Metal.MTLBindingAccessWriteOnly: "write",
}


def _language_version(name: str):
    attr = "MTLLanguageVersion" + name.replace(".", "_")
    return getattr(Metal, attr)


class MetalContext:
    def __init__(self) -> None:
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("no Metal device available")
        self.queue = self.device.newCommandQueue()
        self._lang_cache: str | None = None

    # -- device info ---------------------------------------------------------
    def info(self) -> dict:
        d = self.device
        fam = "unknown"
        for name in ("Apple9", "Apple8", "Apple7", "Apple6"):
            if d.supportsFamily_(getattr(Metal, f"MTLGPUFamily{name}")):
                fam = name
                break
        return {
            "name": str(d.name()),
            "gpu_family": fam,
            "max_threads_per_threadgroup": int(d.maxThreadsPerThreadgroup().width),
            "max_threadgroup_memory_bytes": int(d.maxThreadgroupMemoryLength()),
            "unified_memory": bool(d.hasUnifiedMemory()),
            "recommended_working_set_gb": float(d.recommendedMaxWorkingSetSize()) / 2**30,
            "language_version": self.language_version(),
        }

    def language_version(self) -> str:
        """Highest MSL version that actually compiles on this machine."""
        if self._lang_cache is None:
            probe = "#include <metal_stdlib>\nkernel void k(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 1.0f; }\n"
            for v in ("4.0", "3.2", "3.1", "3.0"):
                try:
                    opts = Metal.MTLCompileOptions.alloc().init()
                    opts.setLanguageVersion_(_language_version(v))
                except AttributeError:
                    continue
                lib, _err = self.device.newLibraryWithSource_options_error_(probe, opts, None)
                if lib is not None:
                    self._lang_cache = v
                    break
            else:
                self._lang_cache = "2.4"
        return self._lang_cache

    # -- compile -------------------------------------------------------------
    def compile(self, source: str, kernel_name: str, *, math: str = "fast", language: str | None = None) -> CompiledKernel:
        lang = language or self.language_version()
        opts = Metal.MTLCompileOptions.alloc().init()
        opts.setLanguageVersion_(_language_version(lang))
        if math == "safe":
            if hasattr(opts, "setMathMode_"):
                opts.setMathMode_(Metal.MTLMathModeSafe)
            else:  # pragma: no cover - old macOS
                opts.setFastMathEnabled_(False)
        elif math == "relaxed" and hasattr(opts, "setMathMode_"):
            opts.setMathMode_(Metal.MTLMathModeRelaxed)

        t0 = time.perf_counter()
        lib, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
        if lib is None:
            raise MetalCompileError(str(err.localizedDescription()) if err is not None else "unknown compile error")
        t1 = time.perf_counter()

        fn = lib.newFunctionWithName_(kernel_name)
        if fn is None:
            names = [str(n) for n in lib.functionNames()]
            raise MetalCompileError(
                f"kernel function '{kernel_name}' not found in the compiled library (functions found: {names}). "
                "It must be declared with the `kernel` qualifier and exactly this name."
            )
        pso, refl, perr = self.device.newComputePipelineStateWithFunction_options_reflection_error_(
            fn, Metal.MTLPipelineOptionArgumentInfo | Metal.MTLPipelineOptionBufferTypeInfo, None, None
        )
        if pso is None:
            raise MetalCompileError(
                "pipeline creation failed: " + (str(perr.localizedDescription()) if perr is not None else "unknown error")
            )
        t2 = time.perf_counter()

        bindings: list[Binding] = []
        if refl is not None:
            for b in refl.bindings():
                bt = b.type()
                access = _ACCESS.get(b.access(), "readwrite")
                if bt == Metal.MTLBindingTypeBuffer:
                    bindings.append(
                        Binding(str(b.name()), int(b.index()), "buffer", int(b.bufferDataType()), access, int(b.bufferDataSize()), bool(b.isUsed()))
                    )
                elif bt == Metal.MTLBindingTypeThreadgroupMemory:
                    bindings.append(
                        Binding(str(b.name()), int(b.index()), "threadgroup", None, access, None, bool(b.isUsed()), int(b.threadgroupMemoryDataSize()))
                    )
                else:
                    bindings.append(Binding(str(b.name()), int(b.index()), "other", None, access, None, bool(b.isUsed())))
        return CompiledKernel(
            name=kernel_name,
            pso=pso,
            bindings=bindings,
            max_total_threads=int(pso.maxTotalThreadsPerThreadgroup()),
            execution_width=int(pso.threadExecutionWidth()),
            static_tg_bytes=int(pso.staticThreadgroupMemoryLength()),
            compile_s=t1 - t0,
            pipeline_s=t2 - t1,
            language_version=lang,
            library=lib,
        )

    # -- buffers -------------------------------------------------------------
    def new_buffer(self, nbytes: int):
        buf = self.device.newBufferWithLength_options_(max(int(nbytes), 16), Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise MemoryError(f"could not allocate a {nbytes}-byte Metal buffer")
        return buf

    @staticmethod
    def view(buf) -> memoryview:
        """Writable byte view of the whole buffer (shared storage)."""
        return memoryview(buf.contents().as_buffer(int(buf.length())))

    # -- dispatch ------------------------------------------------------------
    def run(
        self,
        kernel: CompiledKernel,
        arg_sets: Sequence[Sequence[tuple[int, BufferArg | ScalarArg]]],
        grid: tuple[int, int, int],
        threads: tuple[int, int, int],
        iters: int = 1,
        dispatch: str = "serial",
    ) -> RunTiming:
        """Encode ``iters`` dispatches into one command buffer and wait for it.

        Iteration ``i`` uses ``arg_sets[i % len(arg_sets)]`` so callers can rotate
        through several buffer sets to defeat caches.

        ``dispatch="serial"``: every dispatch starts after the previous one finished (Metal's default encoder).
        ``dispatch="concurrent"``: dispatches on different buffer sets may overlap, as the independent kernels of an MLX
        graph do; a buffer barrier separates every pass over all the sets, so a dispatch never overlaps another one that
        uses the same buffers.
        """
        if iters < 1:
            raise ValueError("iters must be >= 1")
        if dispatch not in ("serial", "concurrent"):
            raise ValueError(f"dispatch must be 'serial' or 'concurrent', got {dispatch!r}")
        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoderWithDispatchType_(Metal.MTLDispatchTypeConcurrent) if dispatch == "concurrent" else cb.computeCommandEncoder()
        enc.setComputePipelineState_(kernel.pso)
        g = Metal.MTLSizeMake(*grid)
        t = Metal.MTLSizeMake(*threads)
        nsets = len(arg_sets)
        for i in range(iters):
            if dispatch == "concurrent" and i > 0 and i % nsets == 0:
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            for idx, a in arg_sets[i % nsets]:
                if isinstance(a, BufferArg):
                    enc.setBuffer_offset_atIndex_(a.buf, a.offset, idx)
                else:
                    enc.setBytes_length_atIndex_(a.data, len(a.data), idx)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(g, t)
        enc.endEncoding()
        if _LAUNCH_HOOKS:
            _LAUNCH_HOOKS[0]()
        t0 = time.perf_counter()
        try:
            cb.commit()
            cb.waitUntilCompleted()
        finally:
            if _LAUNCH_HOOKS:
                _LAUNCH_HOOKS[1]()
        wall = time.perf_counter() - t0
        if cb.status() != Metal.MTLCommandBufferStatusCompleted:
            err = cb.error()
            raise MetalRuntimeError(str(err.localizedDescription()) if err is not None else f"command buffer status {cb.status()}")
        return RunTiming(wall_s=wall, gpu_s=float(cb.GPUEndTime() - cb.GPUStartTime()), iters=iters)


_CTX: MetalContext | None = None


def get_context() -> MetalContext:
    global _CTX
    if _CTX is None:
        _CTX = MetalContext()
    return _CTX


_LOC_RE = re.compile(r"program_source:(\d+):(\d+)")


def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)
