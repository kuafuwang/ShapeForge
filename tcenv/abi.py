"""ABI conformance check of a compiled candidate against the contract.

Metal pointers are typed in source but are just addresses at run time (like PTX
``.param .u64``), so buffer *pointee* types are free (``half4*`` over a half buffer
is a legitimate vectorization). What is fixed: buffer slot, name, access mode,
scalar types, no extra arguments, and a launch configuration the pipeline can run.
"""

from __future__ import annotations

from .contract import Contract
from .dtypes import get_dtype
from .metal_runtime import CompiledKernel


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
            problems.append(f"input `{a.name}` must be declared `device const {get_dtype(a.dtype).msl}*` (it is writable; kernels must not modify inputs)")
        elif a.role == "output" and b.access == "read":
            problems.append(f"output `{a.name}` is declared read-only; it must be a non-const `device` pointer")
        elif a.role == "scalar":
            d = get_dtype(a.dtype)
            if b.dtype_code != d.mtl_code or b.elem_size != d.nbytes:
                problems.append(f"scalar `{a.name}` must be `constant {d.msl}& {a.name} [[buffer({a.index})]]`")
            if b.access != "read":
                problems.append(f"scalar `{a.name}` must be read-only (`constant`)")

    for b in kernel.bindings:
        if b.kind == "buffer" and b.index not in expected_indices:
            problems.append(f"unexpected extra buffer parameter `{b.name}` at [[buffer({b.index})]]; the harness binds only the contract arguments")
        elif b.kind == "threadgroup":
            problems.append(
                f"dynamic threadgroup-memory parameter `{b.name}` [[threadgroup({b.index})]] is not allowed; "
                "declare static `threadgroup T name[N];` inside the kernel body instead"
            )
        elif b.kind == "other":
            problems.append(f"unsupported parameter `{b.name}` (textures/samplers/etc. are not part of the ABI)")

    if contract.threads_per_tg > kernel.max_total_threads:
        problems.append(
            f"launch infeasible: the contract launches {contract.threads_per_tg} threads per threadgroup but this pipeline supports at most "
            f"{kernel.max_total_threads} (register/threadgroup-memory pressure is too high; reduce live registers)"
        )
    if kernel.static_tg_bytes > max_tg_memory:
        problems.append(f"static threadgroup memory {kernel.static_tg_bytes} B exceeds the device limit of {max_tg_memory} B")
    return problems
