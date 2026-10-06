"""ABI conformance check of a compiled candidate against the contract.

Metal pointers are typed in source but are just addresses at run time (like PTX
``.param .u64``), so buffer *pointee* types are free (``half4*`` over a half buffer
is a legitimate vectorization). What is fixed: buffer slot, name, access mode,
scalar types, no extra arguments, and a launch configuration the pipeline can run.
"""

# 编译成功之后、按契约启动之前。compile() 留下的参数反射在这里对上任务契约。
# 问题列表非空时，evaluate 记为 abi_error 并结束，正确性和计时都用不上这份核。

from __future__ import annotations

from .contract import Contract
from .dtypes import get_dtype
from .metal_runtime import CompiledKernel


# 反射对的是槽位、名字、读写、标量类型和这份 pipeline 的启动上限。缓冲指向的元素类型可以向量化，这里放行。
# 返回人类可读的问题列表；空列表才算通过。
def check_abi(contract: Contract, kernel: CompiledKernel, *, max_tg_memory: int = 32768) -> list[str]:
    """Return a list of human-readable violations (empty list means conformant)."""
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
            # 输入必须只读。核若能写输入，后面的输入快照会失败，评测数据也会被改掉。
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
    # 调用方省略时用 32768。evaluate 传入的是设备报告的 maxThreadgroupMemoryLength。
    if kernel.static_tg_bytes > max_tg_memory:
        problems.append(f"static threadgroup memory {kernel.static_tg_bytes} B exceeds the device limit of {max_tg_memory} B")
    return problems
