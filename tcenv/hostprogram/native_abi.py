"""Native64 Host Program ABI generated from the authoritative HostContractSpec.

Unlike the historical Wasm32 renderer, this module has no Python ``sizeof``/``offsetof`` model.
The C compiler owns every structure layout.  Python interacts only with generated scalar/address
accessors and one no-argument ``sf_native_invoke()`` entry; it never passes or dereferences an ABI
structure pointer.

This is a worker-only execution ABI, not a sandbox claim.  The external v2 worker and its watchdog
remain the isolation boundary for native candidate code.
"""

from __future__ import annotations

from collections.abc import Iterable
import hashlib

from .contract import (
    HOST_STACK_BYTES,
    INPUT_READ_IMPORTS,
    STORAGE_GUARD_BYTES,
    STORAGE_GUARD_PATTERN,
    HostContractSpec,
)

NATIVE_ABI_SCHEMA = "tcenv.hostprogram.native-abi.1"
NATIVE_SOURCE_MARKER = "/* CANDIDATE_SOURCE */"


def assemble_native_source(header: str, candidate_source: str) -> str:
    """@brief 返回实际交给 Clang 的完整 C 文件（含 candidate 后的 worker 支持）。"""
    if header.count(NATIVE_SOURCE_MARKER) != 1:
        raise ValueError("native ABI header must contain exactly one candidate marker")
    return header.replace(NATIVE_SOURCE_MARKER, candidate_source)


def native_assembled_sha256(header: str, candidate_source: str) -> str:
    """@brief 哈希真实编译字节，而不是旧 Wasm 的 header+source 拼接近似。"""
    return hashlib.sha256(assemble_native_source(header, candidate_source).encode("utf-8")).hexdigest()


def _used(spec: HostContractSpec, used_workspaces: Iterable[str] | None) -> tuple[str, ...]:
    if used_workspaces is None:
        return tuple(view.name for view in spec.workspaces)
    used = tuple(dict.fromkeys(str(name) for name in used_workspaces))
    unknown = sorted(set(used) - {view.name for view in spec.workspaces})
    if unknown:
        raise ValueError(f"native Host Program asked for undeclared workspaces {unknown}")
    return used


def render_native_support(spec: HostContractSpec, used_workspaces: Iterable[str] | None = None) -> str:
    """@brief 生成 candidate 前的强类型 ABI，以及 candidate 后的 worker 支持函数。"""
    used = set(_used(spec, used_workspaces))
    lines: list[str] = []
    add = lines.append
    add("/* ShapeForge generated native64 ABI.  Compiler-owned layout; no Wasm offsets. */")
    add("#ifndef SF_HOST_CONTRACT_H")
    add("#define SF_HOST_CONTRACT_H")
    add("#include <stdint.h>")
    add("#include <stddef.h>")
    add("#define SF_HOST_ABI_VERSION 2")
    add("#define SF_HOST_OK 0")
    add("#define SF_HOST_REFUSED 1")
    add("#define SF_HOST_INVALID_BOUNDS 2")
    add("#define SF_HOST_CAPACITY_EXCEEDED 3")
    add("#define SF_HOST_IMPORT_REFUSED 4")
    add(f"#define SF_HOST_STACK_BYTES {HOST_STACK_BYTES}u")
    add(f"#define SF_HOST_GUARD_BYTES {STORAGE_GUARD_BYTES}u")
    add(f"#define SF_HOST_GUARD_PATTERN 0x{STORAGE_GUARD_PATTERN:02X}u")
    add("#define SF_NATIVE_EXPORT __attribute__((visibility(\"default\")))")
    add("typedef char sf_native_pointer_must_be_64bit[(sizeof(void *) == 8u) ? 1 : -1];")
    add("")
    for view in spec.workspaces:
        add(f"#define SF_HOST_USE_{view.name.upper()} {1 if view.name in used else 0}u")
        add(f"#define SF_WS_{view.name.upper()}_CAPACITY_ELEMS {view.capacity}u")
        add(f"#define SF_WS_{view.name.upper()}_ELEM_NBYTES {view.elem_nbytes}u")
    add("typedef struct { uint32_t id; uint32_t numel; uint32_t elem_nbytes; } SfTensorMeta;")
    add("typedef struct {")
    if not spec.inputs:
        add("    uint32_t reserved;")
    for view in spec.inputs:
        if view.mapped:
            add(f"    {view.ctype} *{view.name};")
            add(f"    uint32_t {view.name}_numel;")
        else:
            add(f"    SfTensorMeta {view.name};")
    add("} SfHostInputs;")
    add("typedef struct {")
    if not spec.shape_fields:
        add("    uint32_t reserved;")
    for field in spec.shape_fields:
        add(f"    {field.ctype} {field.name};")
    add("} SfHostShapes;")
    add("typedef struct {")
    if not spec.target_fields:
        add("    uint32_t reserved;")
    for field in spec.target_fields:
        add(f"    {field.ctype} {field.name};")
    add("} SfHostTarget;")
    add("typedef struct {")
    if not spec.workspaces:
        add("    uint32_t reserved;")
    for view in spec.workspaces:
        add(f"    {view.ctype} *{view.name};")
        add(f"    uint32_t {view.name}_capacity;")
        add(f"    uint32_t {view.name}_offset;")
        add(f"    uint32_t {view.name}_count;")
        add(f"    uint32_t {view.name}_elem_nbytes;")
    add("} SfHostWorkspaces;")
    add("typedef struct { uint32_t reserved; } SfHostResolved;")
    add("typedef struct {")
    add("    SfHostShapes shapes; SfHostInputs inputs; SfHostTarget target;")
    add("    SfHostWorkspaces workspaces; SfHostResolved resolved;")
    add("} SfHostAbi;")
    add("")

    allocated = [view for view in spec.inputs if view.mapped] + [
        view for view in spec.workspaces if view.name in used
    ]
    for view in allocated:
        count = view.numel if hasattr(view, "numel") else view.capacity
        add("typedef struct __attribute__((aligned(16))) {")
        add(f"    uint8_t lo[SF_HOST_GUARD_BYTES]; {view.ctype} data[{max(1, count)}u];")
        add("    uint8_t hi[SF_HOST_GUARD_BYTES];")
        add(f"}} SfNativeRegion_{view.name};")
        add(f"static SfNativeRegion_{view.name} sf_native_region_{view.name};")
    add("static SfHostAbi sf_host_abi = {")
    add("    .inputs = {")
    if not spec.inputs:
        add("        .reserved = 0u,")
    for view in spec.inputs:
        if view.mapped:
            add(f"        .{view.name} = sf_native_region_{view.name}.data,")
            add(f"        .{view.name}_numel = {view.numel}u,")
        else:
            add(f"        .{view.name} = {{ .id = {view.tensor_id}u, .numel = {view.numel}u, .elem_nbytes = {view.elem_nbytes}u }},")
    add("    },")
    add("    .workspaces = {")
    if not spec.workspaces:
        add("        .reserved = 0u,")
    for view in spec.workspaces:
        pointer = f"sf_native_region_{view.name}.data" if view.name in used else "0"
        add(f"        .{view.name} = {pointer},")
        add(f"        .{view.name}_capacity = {view.capacity}u,")
        add(f"        .{view.name}_offset = 0u, .{view.name}_count = 0u,")
        add(f"        .{view.name}_elem_nbytes = {view.elem_nbytes}u,")
    add("    }")
    add("};")
    add("")
    add("int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,")
    add("                    const SfHostTarget *target, SfHostWorkspaces *workspaces,")
    add("                    SfHostResolved *resolved);")
    for import_name, _ in INPUT_READ_IMPORTS:
        add(f"int32_t {import_name}(int32_t, uint32_t, uintptr_t, uint32_t);")
    add("")

    # All Python↔native interactions go through these functions.  They are definitions in the
    # same translation unit, so no libc, loader or ambient host symbol is needed by the candidate.
    for field in spec.shape_fields:
        add(f"SF_NATIVE_EXPORT void sf_native_set_shape_{field.name}(uint32_t value) {{")
        if field.ctype == "float":
            add("    union { uint32_t u; float f; } bits; bits.u = value;")
            add(f"    sf_host_abi.shapes.{field.name} = bits.f;")
        elif field.ctype == "int32_t":
            add(f"    sf_host_abi.shapes.{field.name} = (int32_t)value;")
        else:
            add(f"    sf_host_abi.shapes.{field.name} = value;")
        add("}")
        add(f"SF_NATIVE_EXPORT uint32_t sf_native_shape_{field.name}(void) {{")
        if field.ctype == "float":
            add(f"    union {{ uint32_t u; float f; }} bits; bits.f = sf_host_abi.shapes.{field.name};")
            add("    return bits.u;")
        elif field.ctype == "int32_t":
            add(f"    return (uint32_t)sf_host_abi.shapes.{field.name};")
        else:
            add(f"    return sf_host_abi.shapes.{field.name};")
        add("}")
    for field in spec.target_fields:
        add(f"SF_NATIVE_EXPORT void sf_native_set_target_{field.name}(uint32_t value) {{")
        add(f"    sf_host_abi.target.{field.name} = (uint32_t)value;")
        add("}")
        add(f"SF_NATIVE_EXPORT uint32_t sf_native_target_{field.name}(void) {{")
        add(f"    return (uint32_t)sf_host_abi.target.{field.name};")
        add("}")
    for view in spec.inputs:
        if not view.mapped:
            for suffix in ("id", "numel", "elem_nbytes"):
                add(f"SF_NATIVE_EXPORT uint32_t sf_native_input_{view.name}_{suffix}(void) {{")
                add(f"    return sf_host_abi.inputs.{view.name}.{suffix};")
                add("}")
            continue
        add(f"SF_NATIVE_EXPORT uintptr_t sf_native_input_{view.name}_address(void) {{")
        add(f"    return (uintptr_t)sf_native_region_{view.name}.data;")
        add("}")
        add(f"SF_NATIVE_EXPORT uintptr_t sf_native_input_{view.name}_abi_address(void) {{")
        add(f"    return (uintptr_t)sf_host_abi.inputs.{view.name};")
        add("}")
        add(f"SF_NATIVE_EXPORT uint32_t sf_native_input_{view.name}_numel(void) {{")
        add(f"    return sf_host_abi.inputs.{view.name}_numel;")
        add("}")
    for view in spec.workspaces:
        name = view.name
        add(f"SF_NATIVE_EXPORT uintptr_t sf_native_workspace_{name}_address(void) {{")
        if name in used:
            add(f"    return (uintptr_t)sf_native_region_{name}.data;")
        else:
            add("    return (uintptr_t)0;")
        add("}")
        add(f"SF_NATIVE_EXPORT uintptr_t sf_native_workspace_{name}_abi_address(void) {{")
        add(f"    return (uintptr_t)sf_host_abi.workspaces.{name};")
        add("}")
        for suffix, field in (("capacity", "capacity"), ("offset", "offset"), ("count", "count"), ("elem_nbytes", "elem_nbytes")):
            add(f"SF_NATIVE_EXPORT uint32_t sf_native_workspace_{name}_{suffix}(void) {{")
            add(f"    return sf_host_abi.workspaces.{name}_{field};")
            add("}")
    add("SF_NATIVE_EXPORT void sf_native_before_call(void) {")
    for view in spec.workspaces:
        add(f"    sf_host_abi.workspaces.{view.name}_offset = 0u;")
        add(f"    sf_host_abi.workspaces.{view.name}_count = 0u;")
    for view in allocated:
        add(f"    for (uint32_t i = 0; i < SF_HOST_GUARD_BYTES; ++i) {{")
        add(f"        sf_native_region_{view.name}.lo[i] = (uint8_t)SF_HOST_GUARD_PATTERN;")
        add(f"        sf_native_region_{view.name}.hi[i] = (uint8_t)SF_HOST_GUARD_PATTERN;")
        add("    }")
    add("}")
    add("SF_NATIVE_EXPORT int32_t sf_native_guard_status(void) {")
    for view in allocated:
        add(f"    for (uint32_t i = 0; i < SF_HOST_GUARD_BYTES; ++i) {{")
        add(f"        if (sf_native_region_{view.name}.lo[i] != (uint8_t)SF_HOST_GUARD_PATTERN ||")
        add(f"            sf_native_region_{view.name}.hi[i] != (uint8_t)SF_HOST_GUARD_PATTERN) return 1;")
        add("    }")
    add("    return 0;")
    add("}")
    add("")

    # Read callbacks are registered by the worker.  Their destination is a native address, not a
    # Wasm32 offset.  The Python callback still checks tensor id/dtype/range and records every read.
    for index, (import_name, dtype) in enumerate(INPUT_READ_IMPORTS):
        add(f"typedef int32_t (*SfNativeRead_{index})(int32_t, uint32_t, uintptr_t, uint32_t);")
        add(f"static SfNativeRead_{index} sf_native_read_{index};")
        add(f"SF_NATIVE_EXPORT void sf_native_set_read_{index}(SfNativeRead_{index} callback) {{")
        add(f"    sf_native_read_{index} = callback;")
        add("}")
    add(NATIVE_SOURCE_MARKER)
    for index, (import_name, dtype) in enumerate(INPUT_READ_IMPORTS):
        add(f"int32_t {import_name}(int32_t id, uint32_t off, uintptr_t dst, uint32_t count) {{")
        add(f"    return sf_native_read_{index} ? sf_native_read_{index}(id, off, dst, count) : SF_HOST_IMPORT_REFUSED;")
        add("}")
    add("SF_NATIVE_EXPORT int32_t sf_native_invoke(void) {")
    add("    return host_tiling(&sf_host_abi.inputs, &sf_host_abi.shapes, &sf_host_abi.target,")
    add("                       &sf_host_abi.workspaces, &sf_host_abi.resolved);")
    add("}")
    add("#endif")
    add("")
    return "\n".join(lines)


__all__ = [
    "NATIVE_ABI_SCHEMA",
    "NATIVE_SOURCE_MARKER",
    "assemble_native_source",
    "native_assembled_sha256",
    "render_native_support",
]
