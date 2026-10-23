"""ABI conformance check of a compiled candidate against the contract.

Metal pointers are typed in source but are just addresses at run time (like PTX
``.param .u64``), so buffer *pointee* types are free (``half4*`` over a half buffer
is a legitimate vectorization). What is fixed: buffer slot, name, access mode,
scalar types, no extra arguments, and a launch configuration the pipeline can run.
"""

# 编译成功之后、按契约启动之前。compile() 留下的参数反射在这里对上任务契约。
# 问题列表非空时，evaluate 记为 abi_error 并结束，正确性和计时都用不上这份核。

from __future__ import annotations

import re

from typing import TYPE_CHECKING

from .contract import Contract
from ...dtypes import get_dtype

if TYPE_CHECKING:
    from .runtime import CompiledKernel


def _threadgroup_limit() -> int:
    from ...target import DEFAULT_TARGET_ID, load_target

    return load_target(DEFAULT_TARGET_ID).limits.max_threadgroup_memory_bytes


def _buffer_index_limit() -> int:
    from ...target import DEFAULT_TARGET_ID, load_target

    return load_target(DEFAULT_TARGET_ID).limits.max_buffer_index


# 反射对的是槽位、名字、读写、标量类型和这份 pipeline 的启动上限。缓冲指向的元素类型可以向量化，这里放行。
# 返回人类可读的问题列表；空列表才算通过。
def check_abi(contract: Contract, kernel: "CompiledKernel", *, max_tg_memory: int | None = None, target_id: str | None = None) -> list[str]:
    """Return a list of human-readable violations (empty list means conformant)."""
    if max_tg_memory is None:
        from ...target import DEFAULT_TARGET_ID, load_target

        max_tg_memory = load_target(target_id or DEFAULT_TARGET_ID).limits.max_threadgroup_memory_bytes
    problems: list[str] = []
    by_index = {}
    for b in kernel.bindings:
        if b.kind == "buffer":
            by_index.setdefault(b.index, b)

    expected_indices = set()
    for a in contract.args:
        expected_indices.add(a.index)
        b = by_index.get(a.index)
        if b is None:
            problems.append(f"missing kernel parameter `{a.msl_declaration}`")
            continue
        if b.name != a.name:
            problems.append(f"[[buffer({a.index})]] must be named `{a.name}` (found `{b.name}`)")
        if a.role == "input" and b.access != "read":
            # 输入必须只读。运行时本来就有输入快照（改了就判 unsafe），这条静态检查的用处是**在评测前**就退掉：
            # 少一次编译和一次 GPU 启动，反馈也更快。放开它（换取就地算法）要等有人量过就地改写在本仓库
            # 的真实收益，见 docs/conv2d约束排查.md「留下的」。
            problems.append(f"input `{a.name}` must be declared `device const {get_dtype(a.dtype).msl}*` (it is writable; kernels must not modify inputs)")
        elif a.role == "output" and b.access == "read":
            problems.append(f"output `{a.name}` is declared read-only; it must be a non-const `device` pointer")
        elif a.role == "scalar":
            # 缓冲的元素类型允许向量化，标量不允许：宽度和 MTL 数据类型必须与契约一致，harness 按这个布局 setBytes。
            d = get_dtype(a.dtype)
            if b.dtype_code != d.mtl_code or b.elem_size != d.nbytes:
                problems.append(f"scalar `{a.name}` must be `constant {d.msl}& {a.name} [[buffer({a.index})]]`")
            if b.access != "read":
                problems.append(f"scalar `{a.name}` must be read-only (`constant`)")

    for b in kernel.bindings:
        if b.kind == "buffer" and b.index not in expected_indices:
            problems.append(f"unexpected extra buffer parameter `{b.name}` at [[buffer({b.index})]]; the harness binds only the contract arguments")
        elif b.kind == "threadgroup":
            # 动态 threadgroup 参数要在启动时另给长度，harness 不传。片上数组写成函数体内的静态 threadgroup T name[N]。
            problems.append(
                f"dynamic threadgroup-memory parameter `{b.name}` [[threadgroup({b.index})]] is not allowed; "
                "declare static `threadgroup T name[N];` inside the kernel body instead"
            )
        elif b.kind == "other":
            problems.append(f"unsupported parameter `{b.name}` (textures/samplers/etc. are not part of the ABI)")

    # pipeline 建成之后，maxTotalThreadsPerThreadgroup 才反映寄存器或 threadgroup 内存压力。契约要求的线程数超过它，这次启动不成立。
    if contract.threads_per_tg > kernel.max_total_threads:
        problems.append(
            f"launch infeasible: the contract launches {contract.threads_per_tg} threads per threadgroup but this pipeline supports at most "
            f"{kernel.max_total_threads} (register/threadgroup-memory pressure is too high; reduce live registers)"
        )
    # 调用方省略时用目标档案里的线程组内存上限。evaluate 传入的是设备报告的 maxThreadgroupMemoryLength。
    if kernel.static_tg_bytes > max_tg_memory:
        problems.append(f"static threadgroup memory {kernel.static_tg_bytes} B exceeds the device limit of {max_tg_memory} B")
    return problems


def check_pipeline(contract: Contract, kernels: dict, spec, *, max_tg_memory: int | None = None, max_buffer_index: int | None = None, target_id: str | None = None) -> list[str]:
    """Per-stage ABI for a ``taic-launch`` spec. ``kernels`` maps a kernel name to its compiled pipeline."""
    if max_tg_memory is None or max_buffer_index is None:
        from ...target import DEFAULT_TARGET_ID, load_target

        limits = load_target(target_id or DEFAULT_TARGET_ID).limits
        if max_tg_memory is None:
            max_tg_memory = limits.max_threadgroup_memory_bytes
        if max_buffer_index is None:
            max_buffer_index = limits.max_buffer_index
    problems: list[str] = []
    contract_index = {a.name: a.index for a in contract.args}
    writable_outputs: set[str] = set()
    for stage in spec.stages:
        kernel = kernels.get(stage.kernel)
        if kernel is None:
            problems.append(f"stage `{stage.kernel}` was not compiled")
            continue
        buffers = [b for b in kernel.bindings if b.kind == "buffer"]
        indexes = [b.index for b in buffers]
        names = [b.name for b in buffers]
        if len(indexes) != len(set(indexes)):
            problems.append(f"stage `{stage.kernel}` binds two parameters to the same buffer slot")
        if len(names) != len(set(names)):
            problems.append(f"stage `{stage.kernel}` binds the same buffer name twice")
        if any(b.kind == "other" for b in kernel.bindings):
            problems.append(f"stage `{stage.kernel}` uses a texture or sampler, which this evaluation does not bind")
        if any(i > max_buffer_index for i in indexes):
            problems.append(f"stage `{stage.kernel}` uses a buffer index above {max_buffer_index}")
        dynamic = [b for b in kernel.bindings if b.kind == "threadgroup"]
        if stage.threadgroup_bytes:
            if len(dynamic) != 1:
                problems.append(f"stage `{stage.kernel}` declares threadgroup bytes but does not have exactly one [[threadgroup(n)]] parameter")
            else:
                total = kernel.static_tg_bytes + stage.threadgroup_bytes
                if total > max_tg_memory:
                    problems.append(
                        f"stage `{stage.kernel}` combined threadgroup memory {total} B "
                        f"(static {kernel.static_tg_bytes} + dynamic {stage.threadgroup_bytes}) exceeds the device limit of {max_tg_memory} B"
                    )
        elif dynamic:
            problems.append(f"stage `{stage.kernel}` has a dynamic threadgroup parameter but the launch line does not declare threadgroup bytes")
        if kernel.static_tg_bytes > max_tg_memory:
            problems.append(f"static threadgroup memory {kernel.static_tg_bytes} B exceeds the device limit of {max_tg_memory} B")
        if stage.threads[0] * stage.threads[1] * stage.threads[2] > kernel.max_total_threads:
            problems.append(
                f"stage `{stage.kernel}` launches {stage.threads[0] * stage.threads[1] * stage.threads[2]} threads per group "
                f"but this pipeline supports at most {kernel.max_total_threads}"
            )
        listed = set(stage.workspaces)
        for b in buffers:
            if b.name in contract_index:
                if b.index != contract_index[b.name]:
                    problems.append(f"stage `{stage.kernel}` binds `{b.name}` at [[buffer({b.index})]]; the contract slot is {contract_index[b.name]}")
                arg = contract.arg(b.name)
                if arg.role == "input" and b.access != "read":
                    problems.append(f"input `{b.name}` must stay `device const` in stage `{stage.kernel}`")
                if arg.role == "output" and b.access != "read":
                    writable_outputs.add(b.name)
            elif b.name in listed:
                pass
            else:
                problems.append(f"stage `{stage.kernel}` buffer `{b.name}` is neither a contract argument nor a workspace listed for this stage")
        for name in listed:
            if name not in names:
                problems.append(f"stage `{stage.kernel}` lists workspace `{name}` but does not bind it")
    missing = [a.name for a in contract.outputs if a.name not in writable_outputs]
    if missing:
        problems.append(
            "pipeline does not bind these outputs as writable in any stage: " + ", ".join(missing)
            + ". A stage may read an output, but some stage must write it; an unwritten output still fails at runtime."
        )
    return problems


_STATIC_TG_LIMIT = re.compile(r"^static threadgroup memory \d+ B exceeds the device limit of \d+ B$")


def _is_threadgroup_limit_line(item: str) -> bool:
    return "threadgroup memory" in item and "exceeds the device limit" in item


def validation_problems(problems: list[str], static_bytes: list[int], max_tg_memory: int) -> list[str]:
    """What a shader-validation pass should report before it decides to launch.

    Instrumentation reports about twice the static threadgroup memory. When that figure
    is already over the device max, launching makes Metal abort the process inside
    ``validateComputeFunctionArguments``. Report the canonical static line and do not
    launch. The parent then rechecks the same kernel without instrumentation.
    A signature or binding failure is kept, and the size line is dropped, so the
    inflated number cannot hide it.
    """
    inflated = [n for n in static_bytes if n > max_tg_memory]
    if not inflated:
        return problems
    others = [item for item in problems if not _is_threadgroup_limit_line(item)]
    if others:
        return others
    biggest = max(inflated)
    return [f"static threadgroup memory {biggest} B exceeds the device limit of {max_tg_memory} B"]


def only_static_threadgroup_limit(violations: list[str]) -> bool:
    """True when every violation is the static threadgroup-size line.

    Shader validation compiles with threadgroup instrumentation, and that pipeline reports
    about twice the memory the kernel declared. The uninstrumented pipeline is the check
    against the device limit. A signature or launch failure is never this case.
    """
    return bool(violations) and all(_STATIC_TG_LIMIT.match(item) for item in violations)
