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
    wall_s: float  # encode + commit -> waitUntilCompleted, same window as mx.eval / torch launch
    gpu_s: float  # GPUEndTime - GPUStartTime for the whole command buffer
    iters: int

    @property
    def wall_per_iter_s(self) -> float:
        """@brief 每次迭代的平均墙钟秒数。

        @return 墙钟时间除以迭代次数。
        """
        return self.wall_s / self.iters

    @property
    def gpu_per_iter_s(self) -> float:
        """@brief 每次迭代的平均 GPU 时间。

        @return 设备时间戳差值除以迭代次数。
        """
        return self.gpu_s / self.iters


# Optional (begin, end) callbacks around every GPU launch. The worker uses them to publish a heartbeat so that the
# parent process can kill a runaway kernel after seconds instead of minutes (a stuck GPU also starves the display).
_LAUNCH_HOOKS: tuple | None = None
_COMPILE_HOOKS: tuple | None = None


# worker 在每次启动前后写心跳 B / E。waitUntilCompleted 若因死循环一直不返回，父进程靠 B 的年龄 SIGKILL。
# 等 GPU 锁用的是另一套 hook，写的是 W，排队不会被当成单次启动挂起。
def set_compile_hooks(begin, end) -> None:
    """@brief 设置编译前后的回调，供 worker 写编译心跳。

    @param begin 编译开始前调用的函数。
    @param end 编译返回或抛错后调用的函数。

    Call ``begin`` before Metal compilation and ``end`` after it returns or raises.

    The worker writes heartbeat ``C`` so the parent can kill a stuck frontend without waiting out the whole worker timeout.
    """
    global _COMPILE_HOOKS
    _COMPILE_HOOKS = (begin, end)


def set_launch_hooks(begin, end) -> None:
    """@brief 设置 GPU 启动前后的回调，供 worker 写启动心跳。

    @param begin commit 之前调用的函数。
    @param end waitUntilCompleted 之后调用的函数。
    """
    global _LAUNCH_HOOKS
    _LAUNCH_HOOKS = (begin, end)


_ACCESS = {
    Metal.MTLBindingAccessReadOnly: "read",
    Metal.MTLBindingAccessReadWrite: "readwrite",
    Metal.MTLBindingAccessWriteOnly: "write",
}


def _bindings_of(refl) -> list[Binding]:
    """@brief 把 Metal 反射对象转成绑定列表。

    @param refl pipeline 反射对象，可为 None。
    @return Binding 列表，描述名字、槽位、种类、读写与元素大小。
    """
    bindings: list[Binding] = []
    if refl is None:
        return bindings
    for b in refl.bindings():
        bt = b.type()
        access = _ACCESS.get(b.access(), "readwrite")
        if bt == Metal.MTLBindingTypeBuffer:
            bindings.append(Binding(str(b.name()), int(b.index()), "buffer", int(b.bufferDataType()), access, int(b.bufferDataSize()), bool(b.isUsed())))
        elif bt == Metal.MTLBindingTypeThreadgroupMemory:
            bindings.append(Binding(str(b.name()), int(b.index()), "threadgroup", None, access, None, bool(b.isUsed()), int(b.threadgroupMemoryDataSize())))
        else:
            bindings.append(Binding(str(b.name()), int(b.index()), "other", None, access, None, bool(b.isUsed())))
    return bindings


# 把 "3.2" 这种名字对上 MTLLanguageVersion3_2。绑定里没有这个常量时由调用方换下一个版本。
def _language_version(name: str):
    """@brief 把 ``"3.2"`` 这样的版本名映射到 MTLLanguageVersion 常量。

    @param name 点分版本名。
    @return 对应的 Metal 常量。
    """
    attr = "MTLLanguageVersion" + name.replace(".", "_")
    return getattr(Metal, attr)


class MetalContext:
    # 默认设备和命令队列。没有设备，后面的编译和启动都无从开始。
    def __init__(self) -> None:
        """@brief 取得系统默认 Metal 设备并创建命令队列。

        @raises RuntimeError 当机器上没有可用的 Metal 设备。
        """
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("no Metal device available")
        self.queue = self.device.newCommandQueue()
        self._lang_cache: str | None = None

    # -- device info ---------------------------------------------------------
    # 设备名、GPU 家族、每组线程上限和 threadgroup 内存上限。ABI 和标定读的是这里。
    def info(self) -> dict:
        """@brief 读取设备名、GPU 家族与各项上限。

        @return 设备信息字典，供 ABI 检查与标定使用。
        """
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
            "max_threads_per_threadgroup_xyz": (
                int(d.maxThreadsPerThreadgroup().width),
                int(d.maxThreadsPerThreadgroup().height),
                int(d.maxThreadsPerThreadgroup().depth),
            ),
            "max_threadgroup_memory_bytes": int(d.maxThreadgroupMemoryLength()),
            "unified_memory": bool(d.hasUnifiedMemory()),
            "recommended_working_set_gb": float(d.recommendedMaxWorkingSetSize()) / 2**30,
            "recommended_working_set_bytes": int(d.recommendedMaxWorkingSetSize()),
            "max_buffer_length": int(d.maxBufferLength()),
            "language_version": self.language_version(),
        }

    # 从高到低试编译一小段核，选这台机器实际能编过的最高 MSL。
    def language_version(self) -> str:
        """@brief 探测这台机器实际能编译的最高 MSL 版本。

        @return 版本名字符串，全部失败时为 ``"2.4"``。

        Highest MSL version that actually compiles on this machine.
        """
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
        """@brief 把源码编成 library 与可启动的 compute pipeline，并保留反射。

        @param source 完整 MSL 源码。
        @param kernel_name 要取出的核函数名。
        @param math 数学模式，``fast``、``safe`` 或 ``relaxed``。
        @param language MSL 版本名；省略时用探测到的最新版本。
        @return 含 pipeline、反射绑定与构建耗时的 CompiledKernel。
        @raises MetalCompileError 当源码编不过、找不到核名或 pipeline 建不起来。
        """
        lang = language or self.language_version()
        opts = Metal.MTLCompileOptions.alloc().init()
        opts.setLanguageVersion_(_language_version(lang))
        if math == "safe":
            # 关掉 fast-math，浮点更接近 IEEE。没有 taic-launch 时评测不传 math，这里的默认是 fast。
            if hasattr(opts, "setMathMode_"):
                opts.setMathMode_(Metal.MTLMathModeSafe)
            else:  # pragma: no cover - old macOS
                # 旧系统没有 setMathMode_，用关掉 fast-math 的开关代替。
                opts.setFastMathEnabled_(False)
        elif math == "relaxed" and hasattr(opts, "setMathMode_"):
            opts.setMathMode_(Metal.MTLMathModeRelaxed)

        if _COMPILE_HOOKS:
            _COMPILE_HOOKS[0]()
        t0 = time.perf_counter()
        try:
            lib, err = self.device.newLibraryWithSource_options_error_(source, opts, None)
        finally:
            if _COMPILE_HOOKS:
                _COMPILE_HOOKS[1]()
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
        bindings = _bindings_of(refl)
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

    def compile_all(self, source: str, kernel_names: Sequence[str], *, math: str = "fast", language: str | None = None) -> list[CompiledKernel]:
        """@brief 一次编出一个 library 里的多个核，按给定顺序返回 pipeline。

        @param source 完整 MSL 源码。
        @param kernel_names 需要的核函数名序列。
        @param math 数学模式。
        @param language MSL 版本名；省略时用探测到的最新版本。
        @return 与 ``kernel_names`` 同序的 CompiledKernel 列表。
        @raises MetalCompileError 当源码编不过、找不到某个核名或 pipeline 建不起来。

        One library, one pipeline per name, in order. ``compile_s`` is on the first kernel only.
        """
        if not kernel_names:
            raise MetalCompileError("compile_all needs at least one kernel name")
        first = self.compile(source, kernel_names[0], math=math, language=language)
        lib = first.library
        out = [first]
        for name in kernel_names[1:]:
            t1 = time.perf_counter()
            fn = lib.newFunctionWithName_(name)
            if fn is None:
                names = [str(n) for n in lib.functionNames()]
                raise MetalCompileError(f"kernel function '{name}' not found in the compiled library (functions found: {names})")
            pso, refl, perr = self.device.newComputePipelineStateWithFunction_options_reflection_error_(
                fn, Metal.MTLPipelineOptionArgumentInfo | Metal.MTLPipelineOptionBufferTypeInfo, None, None
            )
            if pso is None:
                raise MetalCompileError("pipeline creation failed: " + (str(perr.localizedDescription()) if perr is not None else "unknown error"))
            t2 = time.perf_counter()
            bindings = _bindings_of(refl)
            out.append(CompiledKernel(
                name=name, pso=pso, bindings=bindings,
                max_total_threads=int(pso.maxTotalThreadsPerThreadgroup()),
                execution_width=int(pso.threadExecutionWidth()),
                static_tg_bytes=int(pso.staticThreadgroupMemoryLength()),
                compile_s=0.0, pipeline_s=t2 - t1, language_version=first.language_version, library=lib,
            ))
        return out

    # -- buffers -------------------------------------------------------------
    # 共享存储让 CPU 用 memoryview 填保护带、GPU 读同一块地址。分配长度有 16 字节下限。
    def new_buffer(self, nbytes: int):
        """@brief 分配一块共享存储的 MTLBuffer。

        @param nbytes 字节数，不足 16 时按 16 分配。
        @return 新的 MTLBuffer。
        @raises MemoryError 当设备分配失败。
        """
        buf = self.device.newBufferWithLength_options_(max(int(nbytes), 16), Metal.MTLResourceStorageModeShared)
        if buf is None:
            raise MemoryError(f"could not allocate a {nbytes}-byte Metal buffer")
        return buf

    @staticmethod
    def view(buf) -> memoryview:
        """@brief 返回整块缓冲的可写字节视图。

        @param buf MTLBuffer。
        @return 覆盖整块缓冲的 memoryview。

        Writable byte view of the whole buffer (shared storage).
        """
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
        """@brief 把 ``iters`` 次 dispatch 编进一个命令缓冲并等待完成。

        @param kernel 已编译的核。
        @param arg_sets 每次迭代使用的绑定组序列，按 ``i % len`` 轮转。
        @param grid 线程组数量 (x, y, z)。
        @param threads 每组线程数 (x, y, z)。
        @param iters dispatch 次数。
        @param dispatch ``serial`` 或 ``concurrent``。
        @return 墙钟与 GPU 时间的 RunTiming。
        @raises ValueError 当 ``iters`` 小于 1 或 ``dispatch`` 非法。
        @raises MetalRuntimeError 当命令缓冲没有正常完成。

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
        # 墙钟含编码。MLX 的 mx.eval 和 PyTorch 的启动循环都含这一段。
        t0 = time.perf_counter()
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

    #: The one internal kernel the runtime may need (zero-filling a workspace buffer).  It is a
    #: fixed raw-source kernel with no task contract; the composition root builds it through the
    #: backend factory's registered ``metal.internal-fill`` diagnostic profile and *injects* the
    #: pipeline here, so the runtime never compiles on its own (plan §5.2).
    FILL_KERNEL_SOURCE = (
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void taic_fill_zero(device uint* p [[buffer(0)]], constant uint& nwords [[buffer(1)]], "
        "uint gid [[thread_position_in_grid]]) { if (gid < nwords) p[gid] = 0u; }\n"
    )
    FILL_KERNEL_NAME = "taic_fill_zero"

    def inject_fill_kernel(self, kernel: Any) -> None:
        """@brief 注入由 backend 工厂构建的清零核（runtime 不再自己编译）。

        @param kernel 已构建的 ``CompiledKernel``。
        @raises ValueError 核名与登记的内部 fill 核不一致。
        """
        name = str(getattr(kernel, "name", ""))
        if name != self.FILL_KERNEL_NAME:
            raise ValueError(
                f"the injected fill kernel is {name!r}, expected {self.FILL_KERNEL_NAME!r}"
            )
        self._fill = kernel
        self._fill_outcome = getattr(kernel, "_build_outcome", None)

    def _fill_kernel(self) -> CompiledKernel:
        """@brief 返回已注入（并缓存）的清零核；未注入是内部执行错误。

        @return 清零核的 CompiledKernel。
        @raises MetalRuntimeError 清零核未被注入（组合根没有经 backend 工厂构建）。
        """
        cached = getattr(self, "_fill", None)
        if cached is None:
            raise MetalRuntimeError(
                "the internal zero-fill kernel was not injected: the composition root must build it "
                "through the backend factory (metal.internal-fill) before running a fill"
            )
        return cached

    def run_pipeline(
        self,
        stages: Sequence[CompiledKernel],
        grids: Sequence[tuple[int, int, int]],
        threads: Sequence[tuple[int, int, int]],
        threadgroup_bytes: Sequence[int],
        arg_sets: Sequence[Sequence[Sequence[tuple]]],
        fills: Sequence[Sequence[tuple]],
        iters: int = 1,
        dispatch: str = "serial",
    ) -> RunTiming:
        """@brief 把 ``iters`` 轮多段流水线编进一个命令缓冲并等待完成。

        @param stages 按顺序执行的已编译核。
        @param grids 各段的线程组数量。
        @param threads 各段的每组线程数。
        @param threadgroup_bytes 各段动态 threadgroup 内存字节数。
        @param arg_sets ``arg_sets[组][段]`` 给出该段的绑定。
        @param fills ``fills[组]`` 为计时窗内要清零的 (buf, offset, nbytes)。
        @param iters 流水线轮数。
        @param dispatch ``serial`` 或 ``concurrent``。
        @return 墙钟与 GPU 时间的 RunTiming。
        @raises ValueError 当迭代数、dispatch 模式或段列表非法。
        @raises MetalRuntimeError 当命令缓冲没有正常完成。

        ``arg_sets[set][stage]`` is that stage's bindings. ``fills[set]`` is ``(buf, offset, nbytes)``
        zeroed, inside the timed buffer, before the stages of that iteration. A buffer barrier sits
        between the fill and every pair of stages. Concurrent dispatch still barriers between stages
        of one iteration, and between passes over the rotating sets.
        """
        import struct

        if iters < 1:
            raise ValueError("iters must be >= 1")
        if dispatch not in ("serial", "concurrent"):
            raise ValueError(f"dispatch must be 'serial' or 'concurrent', got {dispatch!r}")
        if not stages:
            raise ValueError("run_pipeline needs at least one stage")
        t0 = time.perf_counter()
        nsets = len(arg_sets)
        filled = [bool(one) for one in fills]
        fill = self._fill_kernel() if any(filled) else None
        cb = self.queue.commandBuffer()
        enc = cb.computeCommandEncoderWithDispatchType_(Metal.MTLDispatchTypeConcurrent) if dispatch == "concurrent" else cb.computeCommandEncoder()
        for op in pipeline_ops(iters, nsets, len(stages), concurrent=dispatch == "concurrent", filled=filled):
            kind = op[0]
            if kind in ("rotate_barrier", "barrier"):
                enc.memoryBarrierWithScope_(Metal.MTLBarrierScopeBuffers)
                continue
            if kind == "fill":
                s = op[1]
                enc.setComputePipelineState_(fill.pso)
                for buf, offset, nbytes in fills[s]:
                    nwords = int(nbytes) // 4
                    groups = max(1, (nwords + 255) // 256)
                    enc.setBuffer_offset_atIndex_(buf, int(offset), 0)
                    enc.setBytes_length_atIndex_(struct.pack("<I", nwords), 4, 1)
                    enc.dispatchThreadgroups_threadsPerThreadgroup_(Metal.MTLSizeMake(groups, 1, 1), Metal.MTLSizeMake(256, 1, 1))
                continue
            s, si = op[1], op[2]
            stage = stages[si]
            enc.setComputePipelineState_(stage.pso)
            for idx, arg in arg_sets[s][si]:
                if isinstance(arg, BufferArg):
                    enc.setBuffer_offset_atIndex_(arg.buf, arg.offset, idx)
                else:
                    enc.setBytes_length_atIndex_(arg.data, len(arg.data), idx)
            dyn = next((b for b in stage.bindings if b.kind == "threadgroup"), None)
            if dyn is not None:
                enc.setThreadgroupMemoryLength_atIndex_(int(threadgroup_bytes[si]), int(dyn.index))
            enc.dispatchThreadgroups_threadsPerThreadgroup_(
                Metal.MTLSizeMake(*grids[si]), Metal.MTLSizeMake(*threads[si])
            )
        enc.endEncoding()
        if _LAUNCH_HOOKS:
            _LAUNCH_HOOKS[0]()
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


def pipeline_ops(iters: int, nsets: int, nstages: int, *, concurrent: bool, filled: Sequence[bool]) -> list[tuple]:
    """@brief 排出一个命令缓冲里各段的编码顺序。

    @param iters 流水线轮数。
    @param nsets 轮转缓冲组数。
    @param nstages 每轮的段数。
    @param concurrent 是否使用并发 encoder（需要轮转 barrier）。
    @param filled 每组是否有需要清零的 workspace。
    @return 操作元组列表：fill / barrier / rotate_barrier / dispatch。

    Encode order for one command buffer.

    Every iteration fills that rotation set before its stages, including iterations past the
    first pass. A kernel that cached a transformed tile in workspace on iteration 0 cannot
    reuse it on iteration ``nsets``.
    """
    ops: list[tuple] = []
    for i in range(iters):
        s = i % nsets
        if concurrent and i > 0 and i % nsets == 0:
            ops.append(("rotate_barrier", s))
        if filled[s]:
            ops.append(("fill", s))
            ops.append(("barrier", s))
        for si in range(nstages):
            if si > 0:
                ops.append(("barrier", s, si))
            ops.append(("dispatch", s, si))
    return ops


_CTX: MetalContext | None = None


# 一个进程一份设备和队列。着色器校验在另一个进程里，它的环境变量和这份上下文互不影响。
def get_context() -> MetalContext:
    """@brief 返回进程内共享的 Metal 上下文。

    @return MetalContext 单例。
    """
    global _CTX
    if _CTX is None:
        _CTX = MetalContext()
    return _CTX


# 编译诊断里的 program_source 行号。回指用户核的映射在 contract.rewrite_compiler_lines，用的是那里自己的正则。
_LOC_RE = re.compile(r"program_source:(\d+):(\d+)")


# 去掉诊断文本里的 ANSI 颜色码。本文件没有调用点。
def strip_ansi(s: str) -> str:
    """@brief 去掉诊断文本里的 ANSI 颜色码。

    @param s 原始文本。
    @return 去掉颜色码的文本。
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", s)
