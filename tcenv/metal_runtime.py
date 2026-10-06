"""PyObjC wrapper over Metal: compile MSL, reflect bindings, allocate buffers, dispatch, time.

This is the analogue of ``ptxas`` + the CUDA driver in the paper: it turns candidate
source text into a loaded pipeline and launches it with a fixed contract.
"""

# 真正把源码送上图形芯片的一层。compile 用 newLibraryWithSource 编 MSL，再用 newComputePipelineState 建成可启动的 pipeline，反射留给 ABI。
# run 把若干次 dispatch 编进同一个命令缓冲，commit 之后等到 waitUntilCompleted，这段墙钟才是候选的秒表。
# 着色器校验要单独再跑一遍：开关是进程级环境变量。full 模式先另起一个子进程，带上这些变量，用紧分配走 compile/run；通过后再起第二个子进程做完整评测（含计时），这次不再打开校验。
# 插桩会拉长每次访存，和计时挤在一个进程里会把秒表打乱；带保护带的分配又让小越界落在校验按分配长度看不见的余量里。
# 挂起体现在 waitUntilCompleted 会一直堵到核自己结束，正在跑的 Metal 核不能从外部抢占。commit 前的 launch hook 写心跳 B，父进程据此在单次启动超时后 SIGKILL worker。

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Sequence

import objc
import Metal

# PyObjC 默认不把 reflection 和 error 当 out 参数。漏注册的话反射对象会被丢掉，ABI 检查就没有绑定可对。
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


# 编译失败和设备执行失败分成两种异常，evaluate 才能记成 compile_error 或 runtime_error。
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


# wall_s 是和 MLX / PyTorch 对齐的墙钟。gpu_s 来自命令缓冲的 GPU 时间戳，加速比不用它。
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


# worker 在每次启动前后写心跳 B / E。waitUntilCompleted 若因死循环一直不返回，父进程靠 B 的年龄 SIGKILL。
# 等 GPU 锁用的是另一套 hook，写的是 W，排队不会被当成单次启动挂起。
def set_launch_hooks(begin, end) -> None:
    global _LAUNCH_HOOKS
    _LAUNCH_HOOKS = (begin, end)


_ACCESS = {
    Metal.MTLBindingAccessReadOnly: "read",
    Metal.MTLBindingAccessReadWrite: "readwrite",
    Metal.MTLBindingAccessWriteOnly: "write",
}


# 把 "3.2" 这种名字对上 MTLLanguageVersion3_2。绑定里没有这个常量时由调用方换下一个版本。
def _language_version(name: str):
    attr = "MTLLanguageVersion" + name.replace(".", "_")
    return getattr(Metal, attr)


class MetalContext:
    # 默认设备和命令队列。没有设备，后面的编译和启动都无从开始。
    def __init__(self) -> None:
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("no Metal device available")
        self.queue = self.device.newCommandQueue()
        self._lang_cache: str | None = None

    # -- device info ---------------------------------------------------------
    # 设备名、GPU 家族、每组线程上限和 threadgroup 内存上限。ABI 和标定读的是这里。
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

    # 从高到低试编译一小段核，选这台机器实际能编过的最高 MSL。
    def language_version(self) -> str:
        """Highest MSL version that actually compiles on this machine."""
        if self._lang_cache is None:
            probe = "#include <metal_stdlib>\nkernel void k(device float* x [[buffer(0)]], uint i [[thread_position_in_grid]]) { x[i] = 1.0f; }\n"
            for v in ("4.0", "3.2", "3.1", "3.0"):
                try:
                    opts = Metal.MTLCompileOptions.alloc().init()
                    opts.setLanguageVersion_(_language_version(v))
                except AttributeError:
                    # 这套 PyObjC 没有该版本常量，换下一个。
                    continue
                lib, _err = self.device.newLibraryWithSource_options_error_(probe, opts, None)
                if lib is not None:
                    self._lang_cache = v
                    break
            else:
                # 4.0 到 3.0 都不可用时记成 2.4，不再探测。
                self._lang_cache = "2.4"
        return self._lang_cache

    # -- compile -------------------------------------------------------------
    # 源码到可启动对象的路径。库编不过、找不到 kernel 名字、pipeline 建不起来，都是 MetalCompileError。
    # 反射（名字、槽位、读写、数据类型）跟着 pipeline 留下，契约判断在 check_abi。
    def compile(self, source: str, kernel_name: str, *, math: str = "fast", language: str | None = None) -> CompiledKernel:
        lang = language or self.language_version()
        opts = Metal.MTLCompileOptions.alloc().init()
        opts.setLanguageVersion_(_language_version(lang))
        if math == "safe":
            # 关掉 fast-math，浮点更接近 IEEE。评测入口使用默认的 fast，不传这个参数。
            if hasattr(opts, "setMathMode_"):
                opts.setMathMode_(Metal.MTLMathModeSafe)
            else:  # pragma: no cover - old macOS
                # 旧系统没有 setMathMode_，用关掉 fast-math 的开关代替。
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
        # ArgumentInfo 与 BufferTypeInfo 让反射带上名字、槽位、读写和标量类型。没有这两项，ABI 检查看不到绑定。
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
                # 反射分成普通缓冲、动态 threadgroup 长度，以及其他（纹理等）。元素类型能否向量化留给 check_abi。
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
    # 共享存储让 CPU 用 memoryview 填保护带、GPU 读同一块地址。分配长度有 16 字节下限。
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
    # 真正启动。墙钟从 commit 到 waitUntilCompleted，bench 拿 wall_s 与基线的 perf_counter 对齐；gpu_s 是设备时间戳，加速比用墙钟。
    # 钩子包住这次等待：开始写 B，结束写 E。核若死循环，wait 不会返回，父进程靠 B 的年龄 SIGKILL。
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
                # 每转完一整轮缓冲组才插 barrier。同一组缓冲因此不会自己重叠；不同组可以重叠，和 MLX 图里的独立调用一样。
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
            for idx, a in arg_sets[i % nsets]:
                if isinstance(a, BufferArg):
                    enc.setBuffer_offset_atIndex_(a.buf, a.offset, idx)
                else:
                    enc.setBytes_length_atIndex_(a.data, len(a.data), idx)
            enc.dispatchThreadgroups_threadsPerThreadgroup_(g, t)
        enc.endEncoding()
        if _LAUNCH_HOOKS:
            # B 写在 commit 之前。父进程用 B 的时间戳判断这一次启动已经挂了多久。
            _LAUNCH_HOOKS[0]()
        t0 = time.perf_counter()
        try:
            cb.commit()
            cb.waitUntilCompleted()
        finally:
            # 正常返回或 Python 异常时写 E。父进程 SIGKILL 不会执行到这里，它靠进程消失结束等待。
            if _LAUNCH_HOOKS:
                _LAUNCH_HOOKS[1]()
        wall = time.perf_counter() - t0
        if cb.status() != Metal.MTLCommandBufferStatusCompleted:
            # 状态不是 Completed 就抛 MetalRuntimeError，这一次不会被当成耗时样本。
            err = cb.error()
            raise MetalRuntimeError(str(err.localizedDescription()) if err is not None else f"command buffer status {cb.status()}")
        return RunTiming(wall_s=wall, gpu_s=float(cb.GPUEndTime() - cb.GPUStartTime()), iters=iters)


_CTX: MetalContext | None = None


# 一个进程一份设备和队列。着色器校验在另一个进程里，它的环境变量和这份上下文互不影响。
def get_context() -> MetalContext:
    global _CTX
    if _CTX is None:
        _CTX = MetalContext()
    return _CTX


# 编译诊断里的 program_source 行号。回指用户核的映射在 contract.rewrite_compiler_lines，用的是那里自己的正则。
_LOC_RE = re.compile(r"program_source:(\d+):(\d+)")


# 去掉诊断文本里的 ANSI 颜色码。本文件没有调用点。
def strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)
