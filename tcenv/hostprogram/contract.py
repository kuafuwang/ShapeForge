"""Generated, strongly typed ``host_contract.h`` for the restricted C99 Host Program (V3).

The header is **derived from the task ABI**, never hand-written by the candidate: field names,
dtypes, capacities, target knobs and Wasm32 offsets all come from the authoritative OpSpec /
``TensorMemoryContract``.  A candidate that uses an undeclared name fails at ``clang`` time, and a
candidate that never writes host addresses or Wasm offsets at all — it writes
``inputs->cu_seqlens`` and ``workspaces->work_items.count``.

Layout rules that the generated ``typedef``/``offsetof`` macros freeze:

* every field is 4 bytes wide (Wasm32 pointers are 4 bytes), so the struct offsets are a plain
  cumulative sum and a Python layout and the C ``sizeof`` cannot drift;
* the pointer fields point **into the module's own exported linear memory** (a static storage
  array in the generated header), never into a second ``Memory`` and never at host addresses;
* every allocated view has a 32-byte guard sentinel before and after it.  The accessors validate
  the declared span before copying; the sentinels catch a planner's out-of-bounds write across
  regions instead of letting it silently poison a neighbouring workspace;
* an input/output pair never shares storage: mapped inputs and workspace spans are distinct
  aligned regions, and ``validate_regions`` checks the disjointness at runtime;
* only workspaces the candidate source actually references get used.  A capacity the program never
  touches (``log_decay_tab`` is ~384 MiB at the V6 scale) must not be allocated in Wasm memory,
  so the header exposes the field but leaves the pointer null unless ``SF_HOST_USE_<NAME>`` is 1.

The header is prefixed to the candidate source.  ``assembled_host_sha256`` covers header + source,
so the exact declared layout is part of the build identity.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

from ..dtypes import get_dtype
from ..track.expr import INT, evaluate_term

CONTRACT_SCHEMA_ID = "tcenv.hostprogram.contract.1"
CONTRACT_SCHEMA_VERSION = 1
#: A tensor at or below this many bytes gets a typed pointer in the module's linear memory.
MAX_MAPPED_INPUT_BYTES = 64 * 1024
#: Aggregate cap for directly mapped inputs.  Beyond it the tensor becomes metadata-only, so the
#: Wasm memory budget cannot grow with the number of small-ish inputs.
MAX_TOTAL_MAPPED_BYTES = 1 << 20
#: Guard sentinel bytes before and after every allocated view.
STORAGE_GUARD_BYTES = 32
STORAGE_GUARD_PATTERN = 0xA5
#: Fixed stack reserved by the toolchain for the planner's own arrays/locals.
HOST_STACK_BYTES = 64 * 1024

#: Canonical dtype -> C type.  A dtype outside this map cannot be directly mapped *or* read
#: through an import in V3: it stays metadata-only (``id``/``numel``/``elem_nbytes``), which is
#: enough for a planner that only needs the declared algebra.  The KDA planners read
#: ``cu_seqlens`` (i32) and optionally ``scale`` (fp32).
_C_DTYPE: Mapping[str, str] = {
    "i32": "int32_t",
    "u32": "uint32_t",
    "fp32": "float",
    "i8": "int8_t",
    "u8": "uint8_t",
}

#: Import names the generated header declares and the executor is allowed to provide.
INPUT_READ_IMPORTS: tuple[tuple[str, str], ...] = (
    ("sf_input_read_i32", "i32"),
    ("sf_input_read_u32", "u32"),
    ("sf_input_read_f32", "fp32"),
)

_C_KEYWORDS = frozenset(
    """
    auto break case char const continue default do double else enum extern float for goto if
    inline int long register restrict return short signed sizeof static struct switch typedef
    union unsigned void volatile while _Bool _Complex _Imaginary
    """.split()
)
_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


@dataclass(frozen=True)
class HostField:
    """@brief shapes/target/resolved 里的一个具名字段（Wasm32 布局固定 4 字节）。"""

    name: str
    ctype: str
    value: int | float | None = None

    def packed(self) -> bytes:
        """@brief 把该字段的值打成 little-endian 4 字节。

        @return 字节串。
        @raises ValueError ``value`` 为 None（target 字段的值由执行期给出）。
        """
        import struct

        if self.value is None:
            raise ValueError(f"field {self.name!r} has no value to pack")
        if self.ctype == "float":
            return struct.pack("<f", float(self.value))
        if self.ctype == "int32_t":
            return struct.pack("<i", int(self.value))
        return struct.pack("<I", int(self.value))


@dataclass(frozen=True)
class HostInputView:
    """@brief 一个可读 global input 在 Host Program 中的视图。"""

    name: str
    dtype: str
    numel: int
    elem_nbytes: int
    tensor_id: int
    mapped: bool
    ctype: str | None

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "numel": int(self.numel),
            "elem_nbytes": int(self.elem_nbytes),
            "tensor_id": int(self.tensor_id),
            "mapped": bool(self.mapped),
            "ctype": self.ctype,
        }


@dataclass(frozen=True)
class HostWorkspaceView:
    """@brief 一个 workspace 在 Host Program 中的单连续有效 span。"""

    name: str
    dtype: str
    capacity: int
    elem_nbytes: int
    ctype: str

    def document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "capacity": int(self.capacity),
            "elem_nbytes": int(self.elem_nbytes),
            "ctype": self.ctype,
        }


def _field_name(name: object, where: str) -> str:
    """@brief 校验一个字段名能作为 C 标识符，且没有占用 harness 前缀。

    @param name 原始名字。
    @param where 诊断位置。
    @return 名字。
    @raises ValueError 非法。
    """
    text = str(name)
    if not _C_IDENTIFIER.match(text):
        raise ValueError(f"{where}: {text!r} is not a C identifier")
    if text in _C_KEYWORDS:
        raise ValueError(f"{where}: {text!r} is a C keyword")
    if text.startswith("sf_") or text.startswith("SF_"):
        raise ValueError(f"{where}: {text!r} uses the harness-reserved sf_/SF_ prefix")
    return text


def _resolve_numel(term: Any, instance: Any, where: str) -> int:
    """@brief 求值 ABI 形状里的一个维度。

    @param term 形状项。
    @param instance 语义实例。
    @param where 诊断位置。
    @return 整数维度。
    @raises ValueError 求值失败。
    """
    names = {str(k): int(v) for k, v in dict(instance.symbols).items()}
    names.update(
        {
            str(k): int(v)
            for k, v in dict(instance.resolved_attrs).items()
            if isinstance(v, int) and not isinstance(v, bool)
        }
    )
    try:
        return int(evaluate_term(term, names, where=where, expect=INT))
    except Exception as exc:  # noqa: BLE001 - normalized into the typed contract plane
        raise ValueError(f"{where}: cannot resolve shape term {term!r}: {exc}") from exc


def _numel(shape: Sequence[Any], instance: Any, where: str) -> int:
    total = 1
    for index, term in enumerate(shape):
        total *= _resolve_numel(term, instance, f"{where}.shape[{index}]")
    return int(total)


@dataclass(frozen=True)
class HostContractSpec:
    """@brief 一个任务的 Host Program 合同（ABI 名称/类型/容量，无任何调度策略）。"""

    op_name: str
    shape_fields: tuple[HostField, ...]
    inputs: tuple[HostInputView, ...]
    workspaces: tuple[HostWorkspaceView, ...]
    target_fields: tuple[HostField, ...]
    mapped_limit: int = MAX_MAPPED_INPUT_BYTES

    def input(self, name: str) -> HostInputView:
        for view in self.inputs:
            if view.name == name:
                return view
        raise KeyError(name)

    def workspace(self, name: str) -> HostWorkspaceView:
        for view in self.workspaces:
            if view.name == name:
                return view
        raise KeyError(name)

    @property
    def readable_inputs(self) -> tuple[HostInputView, ...]:
        return self.inputs

    def document(self) -> dict[str, Any]:
        """@brief 合同文档（进入 ``assembled_host_sha256`` 的输入侧身份）。"""
        return {
            "schema_id": CONTRACT_SCHEMA_ID,
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "op_name": self.op_name,
            "shapes": [{"name": f.name, "ctype": f.ctype, "value": f.value} for f in self.shape_fields],
            "inputs": [view.document() for view in self.inputs],
            "workspaces": [view.document() for view in self.workspaces],
            "target": [{"name": f.name, "ctype": f.ctype} for f in self.target_fields],
            "mapped_limit": int(self.mapped_limit),
        }

    @property
    def digest(self) -> str:
        """@brief 合同文档的规范 sha256。"""
        raw = json.dumps(self.document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def layout(self) -> "HostAbiLayout":
        """@brief 计算 Wasm32 结构体/字段偏移（由生成头里的 static assert 反向锁住）。

        @return 布局。
        """
        return _layout(self)

    def header(self, *, used_workspaces: Iterable[str] | None = None) -> str:
        """@brief 生成 ``host_contract.h`` 全文。

        @param used_workspaces 需要真正分配线性内存的 workspace 名；``None`` 表示全部
            （题面展示用完整 ABI）。
        @return C 头文本。
        @raises ValueError 未声明的 workspace 名。
        """
        used = {view.name for view in self.workspaces} if used_workspaces is None else set(used_workspaces)
        unknown = sorted(used - {view.name for view in self.workspaces})
        if unknown:
            raise ValueError(f"host contract header asked for unknown workspaces {unknown}")
        return _render_header(self, used)


def _target_fields() -> tuple[HostField, ...]:
    """@brief 执行期写入的 target 字段（只含 target profile 公开且进入身份的整数）。"""
    return (
        HostField("max_buffer_index", "uint32_t"),
        HostField("max_threads_per_threadgroup", "uint32_t"),
        HostField("max_threadgroup_memory_bytes", "uint32_t"),
    )


def host_contract_spec(
    *,
    instance: Any,
    tensors: Sequence[Any] | None = None,
    mapped_limit: int = MAX_MAPPED_INPUT_BYTES,
) -> HostContractSpec:
    """@brief 由 OpSpec 实例（可选 ABI 张量）派生 Host Program 合同。

    @param instance ``SemanticInstanceV2``。
    @param tensors 张量声明；默认 ``instance.tensors``（ABI 张量可传入以使用同一份解析后声明）。
    @param mapped_limit 单张量直接映射的字节上限。
    @return 合同。
    @raises ValueError 形状不可求值或名字不是合法 C 标识符。
    """
    if isinstance(mapped_limit, bool) or not isinstance(mapped_limit, int) or mapped_limit < 0:
        raise ValueError("mapped_limit must be an integer >= 0")
    shape_fields: list[HostField] = []
    for name, value in dict(instance.symbols).items():
        shape_fields.append(HostField(_field_name(name, "symbols"), "uint32_t", int(value)))
    for name, value in dict(instance.resolved_attrs).items():
        if isinstance(value, bool):
            raise ValueError(f"attrs.{name}: a boolean is not a Host Program scalar")
        cname = _field_name(name, "attrs")
        if isinstance(value, int):
            ctype = "uint32_t" if value >= 0 else "int32_t"
        elif isinstance(value, float):
            ctype = "float"
        else:
            raise ValueError(f"attrs.{name}: unsupported Host Program scalar {value!r}")
        shape_fields.append(HostField(cname, ctype, value))

    declared = tuple(tensors if tensors is not None else instance.tensors)
    inputs: list[HostInputView] = []
    workspaces: list[HostWorkspaceView] = []
    mapped_total = 0
    tensor_id = 0
    for tensor in declared:
        space = str(getattr(tensor, "space", "global"))
        if space == "workspace":
            dtype = str(tensor.dtype)
            nbytes = get_dtype(dtype).nbytes
            capacity = _numel(tuple(tensor.shape), instance, f"workspace {tensor.name!r}")
            workspaces.append(
                HostWorkspaceView(
                    name=_field_name(tensor.name, "workspace"),
                    dtype=dtype,
                    capacity=capacity,
                    elem_nbytes=nbytes,
                    ctype=_C_DTYPE.get(dtype, "uint8_t"),
                )
            )
            continue
        access = str(getattr(tensor, "access", "read"))
        if access not in ("read", "read_write"):
            continue
        dtype = str(tensor.dtype)
        nbytes = get_dtype(dtype).nbytes
        numel = _numel(tuple(tensor.shape), instance, f"input {tensor.name!r}")
        tensor_id += 1
        ctype = _C_DTYPE.get(dtype)
        mapped = (
            ctype is not None
            and numel * nbytes <= mapped_limit
            and mapped_total + numel * nbytes <= MAX_TOTAL_MAPPED_BYTES
        )
        if mapped:
            mapped_total += numel * nbytes
        inputs.append(
            HostInputView(
                name=_field_name(tensor.name, "input"),
                dtype=dtype,
                numel=numel,
                elem_nbytes=nbytes,
                tensor_id=tensor_id,
                mapped=mapped,
                ctype=ctype if mapped else None,
            )
        )
    return HostContractSpec(
        op_name=str(instance.op_name),
        shape_fields=tuple(shape_fields),
        inputs=tuple(inputs),
        workspaces=tuple(workspaces),
        target_fields=_target_fields(),
        mapped_limit=int(mapped_limit),
    )


# --- Wasm32 layout -----------------------------------------------------------------------------


@dataclass(frozen=True)
class HostFieldLayout:
    name: str
    ctype: str
    offset: int


@dataclass(frozen=True)
class HostInputLayout:
    name: str
    mapped: bool
    offset: int
    numel_offset: int | None
    numel: int
    elem_nbytes: int
    dtype: str


@dataclass(frozen=True)
class HostWorkspaceLayout:
    name: str
    ptr_offset: int
    capacity_offset: int
    offset_offset: int
    count_offset: int
    elem_nbytes_offset: int
    capacity: int
    elem_nbytes: int


@dataclass(frozen=True)
class HostAbiLayout:
    abi_bytes: int
    shapes_offset: int
    inputs_offset: int
    target_offset: int
    workspaces_offset: int
    resolved_offset: int
    shape_fields: tuple[HostFieldLayout, ...]
    input_fields: tuple[HostInputLayout, ...]
    target_fields: tuple[HostFieldLayout, ...]
    workspace_fields: tuple[HostWorkspaceLayout, ...]
    storage_bytes: int

    def shape_offset(self, name: str) -> int:
        for field in self.shape_fields:
            if field.name == name:
                return self.shapes_offset + field.offset
        raise KeyError(name)

    def input_field(self, name: str) -> HostInputLayout:
        for field in self.input_fields:
            if field.name == name:
                return field
        raise KeyError(name)

    def workspace_field(self, name: str) -> HostWorkspaceLayout:
        for field in self.workspace_fields:
            if field.name == name:
                return field
        raise KeyError(name)


@dataclass(frozen=True)
class HostStorageRegion:
    """@brief 一个已分配视图在 module 线性内存中的边界。"""

    name: str
    kind: str
    data_offset: int
    nbytes: int
    guard_lo: int
    guard_hi: int


def _struct(fields: Sequence[tuple[str, int]]) -> tuple[int, dict[str, int]]:
    offset = 0
    out: dict[str, int] = {}
    for name, size in fields:
        out[name] = offset
        offset += size
    return offset, out


def _layout(spec: HostContractSpec) -> HostAbiLayout:
    shapes_size, _ = _struct([(f.name, 4) for f in spec.shape_fields] or [("reserved", 4)])
    input_sizes = [
        (view.name, 8 if view.mapped else 12) for view in spec.inputs
    ] or [("reserved", 4)]
    inputs_size, _ = _struct(input_sizes)
    target_size, _ = _struct([(f.name, 4) for f in spec.target_fields] or [("reserved", 4)])
    workspaces_size, _ = _struct(
        [(f"{view.name}_{suffix}", 4) for view in spec.workspaces for suffix in ("ptr", "capacity", "offset", "count", "elem_nbytes")]
        or [("reserved", 4)]
    )
    resolved_size = 4
    order = ("shapes", "inputs", "target", "workspaces", "resolved")
    abi_size, offsets = _struct(
        [
            ("shapes", shapes_size),
            ("inputs", inputs_size),
            ("target", target_size),
            ("workspaces", workspaces_size),
            ("resolved", resolved_size),
        ]
    )
    assert tuple(offsets) == order

    shape_layout = tuple(
        HostFieldLayout(f.name, f.ctype, offsets["shapes"] + index * 4)
        for index, f in enumerate(spec.shape_fields)
    )
    input_layout: list[HostInputLayout] = []
    cursor = offsets["inputs"]
    for view in spec.inputs:
        if view.mapped:
            input_layout.append(
                HostInputLayout(view.name, True, cursor, cursor + 4, view.numel, view.elem_nbytes, view.dtype)
            )
            cursor += 8
        else:
            input_layout.append(
                HostInputLayout(view.name, False, cursor, None, view.numel, view.elem_nbytes, view.dtype)
            )
            cursor += 12
    target_layout = tuple(
        HostFieldLayout(f.name, f.ctype, offsets["target"] + index * 4)
        for index, f in enumerate(spec.target_fields)
    )
    workspace_layout: list[HostWorkspaceLayout] = []
    cursor = offsets["workspaces"]
    for view in spec.workspaces:
        workspace_layout.append(
            HostWorkspaceLayout(
                name=view.name,
                ptr_offset=cursor,
                capacity_offset=cursor + 4,
                offset_offset=cursor + 8,
                count_offset=cursor + 12,
                elem_nbytes_offset=cursor + 16,
                capacity=view.capacity,
                elem_nbytes=view.elem_nbytes,
            )
        )
        cursor += 20
    return HostAbiLayout(
        abi_bytes=abi_size,
        shapes_offset=offsets["shapes"],
        inputs_offset=offsets["inputs"],
        target_offset=offsets["target"],
        workspaces_offset=offsets["workspaces"],
        resolved_offset=offsets["resolved"],
        shape_fields=shape_layout,
        input_fields=tuple(input_layout),
        target_fields=target_layout,
        workspace_fields=tuple(workspace_layout),
        storage_bytes=0,
    )


def _align(value: int, alignment: int = 16) -> int:
    return (value + alignment - 1) // alignment * alignment


def storage_layout(spec: HostContractSpec, used_workspaces: Iterable[str]) -> tuple[HostStorageRegion, ...]:
    """@brief 计算已分配视图（映射输入 + 被引用的 workspace）的线性内存边界。

    @param spec 合同。
    @param used_workspaces 需要分配存储的 workspace 名。
    @return 区域元组；每块前后各有 ``STORAGE_GUARD_BYTES`` 字节哨兵。
    @raises ValueError 未声明的 workspace。
    """
    used = set(used_workspaces)
    known = {view.name for view in spec.workspaces}
    unknown = sorted(used - known)
    if unknown:
        raise ValueError(f"host storage asked for unknown workspaces {unknown}")
    regions: list[HostStorageRegion] = []
    cursor = 0

    def add(name: str, kind: str, nbytes: int) -> None:
        nonlocal cursor
        guard_lo = _align(cursor)
        cursor = guard_lo + STORAGE_GUARD_BYTES
        data = _align(cursor)
        cursor = data + int(nbytes)
        guard_hi = _align(cursor)
        cursor = guard_hi + STORAGE_GUARD_BYTES
        regions.append(HostStorageRegion(name, kind, data, int(nbytes), guard_lo, guard_hi))

    for view in spec.inputs:
        if view.mapped:
            add(view.name, "input", view.numel * view.elem_nbytes)
    for view in spec.workspaces:
        if view.name in used:
            add(view.name, "workspace", view.capacity * view.elem_nbytes)
    return tuple(regions)


# --- header rendering -------------------------------------------------------------------------


def _render_header(spec: HostContractSpec, used: set[str]) -> str:
    layout = spec.layout()
    regions = {region.name: region for region in storage_layout(spec, used)}
    total = _align(max((region.guard_hi + STORAGE_GUARD_BYTES for region in regions.values()), default=0))
    lines: list[str] = []
    add = lines.append
    add("/* Generated by tcenv.hostprogram — restricted C99 / Wasm Host Program ABI.")
    add(f" * task: {spec.op_name}")
    add(" * Fixed Wasm32 layout; the static assertions below prove the offsets. Do not edit.")
    add(" */")
    add("#ifndef SF_HOST_CONTRACT_H")
    add("#define SF_HOST_CONTRACT_H")
    add("#include <stdint.h>")
    add("#include <stddef.h>")
    add("")
    add("#define SF_HOST_ABI_VERSION 1")
    add("#define SF_HOST_OK 0")
    add("#define SF_HOST_REFUSED 1")
    add("#define SF_HOST_INVALID_BOUNDS 2")
    add("#define SF_HOST_CAPACITY_EXCEEDED 3")
    add("#define SF_HOST_IMPORT_REFUSED 4")
    add("")
    add("#define SF_HOST_STORAGE_BYTES %du" % total)
    add("#define SF_HOST_STACK_BYTES %du" % HOST_STACK_BYTES)
    add("#define SF_HOST_GUARD_BYTES %du" % STORAGE_GUARD_BYTES)
    add("#define SF_HOST_GUARD_PATTERN 0xA5u")
    add("")
    for view in spec.workspaces:
        add(f"#ifndef SF_HOST_USE_{view.name.upper()}")
        add(f"#define SF_HOST_USE_{view.name.upper()} {1 if view.name in used else 0}")
        add("#endif")
    add("")
    add("typedef struct { uint32_t id; uint32_t numel; uint32_t elem_nbytes; } SfTensorMeta;")
    add("")
    # Inputs
    add("typedef struct {")
    if not spec.inputs:
        add("    uint32_t reserved;")
    for view in spec.inputs:
        if view.mapped:
            add(f"    {view.ctype} *{view.name};  /* {view.dtype}, {view.numel} elems, read-only */")
            add(f"    uint32_t {view.name}_numel;")
        else:
            add(f"    SfTensorMeta {view.name};  /* {view.dtype}, {view.numel} elems, read via sf_input_read_* */")
    add("} SfHostInputs;")
    add("")
    # Shapes
    add("typedef struct {")
    if not spec.shape_fields:
        add("    uint32_t reserved;")
    for field in spec.shape_fields:
        add(f"    {field.ctype} {field.name};")
    add("} SfHostShapes;")
    add("")
    # Target
    add("typedef struct {")
    for field in spec.target_fields:
        add(f"    {field.ctype} {field.name};")
    add("} SfHostTarget;")
    add("")
    # Workspaces
    add("typedef struct {")
    if not spec.workspaces:
        add("    uint32_t reserved;")
    for view in spec.workspaces:
        add(f"    {view.ctype} *{view.name};  /* {view.dtype}, capacity {view.capacity} elems */")
        add(f"    uint32_t {view.name}_capacity;")
        add(f"    uint32_t {view.name}_offset;")
        add(f"    uint32_t {view.name}_count;")
        add(f"    uint32_t {view.name}_elem_nbytes;")
    add("} SfHostWorkspaces;")
    add("")
    # Resolved
    add("typedef struct {")
    add("    uint32_t reserved;  /* Dyn-0: the launch geometry is static; no runtime scalar is declared */")
    add("} SfHostResolved;")
    add("")
    add("typedef struct {")
    add("    SfHostShapes shapes;")
    add("    SfHostInputs inputs;")
    add("    SfHostTarget target;")
    add("    SfHostWorkspaces workspaces;")
    add("    SfHostResolved resolved;")
    add("} SfHostAbi;")
    add("")
    add("/* Strong typing is enforced by the compiler: an undeclared name does not exist. */")
    add("#define SF_ABI_SHAPES ((uint32_t)offsetof(SfHostAbi, shapes))")
    add("#define SF_ABI_INPUTS ((uint32_t)offsetof(SfHostAbi, inputs))")
    add("#define SF_ABI_TARGET ((uint32_t)offsetof(SfHostAbi, target))")
    add("#define SF_ABI_WORKSPACES ((uint32_t)offsetof(SfHostAbi, workspaces))")
    add("#define SF_ABI_RESOLVED ((uint32_t)offsetof(SfHostAbi, resolved))")
    for field in spec.shape_fields:
        add(f"#define SF_SHAPE_{field.name.upper()} ((uint32_t)offsetof(SfHostShapes, {field.name}))")
    for view in spec.inputs:
        if view.mapped:
            add(f"#define SF_INPUT_{view.name.upper()} ((uint32_t)offsetof(SfHostInputs, {view.name}))")
            add(f"#define SF_INPUT_{view.name.upper()}_NUMEL ((uint32_t)offsetof(SfHostInputs, {view.name}_numel))")
        else:
            add(f"#define SF_INPUT_{view.name.upper()} ((uint32_t)offsetof(SfHostInputs, {view.name}))")
    for view in spec.workspaces:
        upper = view.name.upper()
        add(f"#define SF_WS_{upper}_PTR ((uint32_t)offsetof(SfHostWorkspaces, {view.name}))")
        add(f"#define SF_WS_{upper}_CAPACITY ((uint32_t)offsetof(SfHostWorkspaces, {view.name}_capacity))")
        add(f"#define SF_WS_{upper}_CAPACITY_ELEMS {view.capacity}u")
        add(f"#define SF_WS_{upper}_ELEM_NBYTES_VALUE {view.elem_nbytes}u")
        add(f"#define SF_WS_{upper}_OFFSET ((uint32_t)offsetof(SfHostWorkspaces, {view.name}_offset))")
        add(f"#define SF_WS_{upper}_COUNT ((uint32_t)offsetof(SfHostWorkspaces, {view.name}_count))")
        add(f"#define SF_WS_{upper}_ELEM_NBYTES ((uint32_t)offsetof(SfHostWorkspaces, {view.name}_elem_nbytes))")
    add("")
    # Static assertions (C99 has no _Static_assert; a negative array size is the portable form).
    assertions: list[tuple[str, bool]] = [
        (f"sizeof(SfHostShapes) == {_struct([(f.name, 4) for f in spec.shape_fields] or [('reserved', 4)])[0]}u", True),
        (f"sizeof(SfHostInputs) == {_struct([(v.name, 8 if v.mapped else 12) for v in spec.inputs] or [('reserved', 4)])[0]}u", True),
        (f"sizeof(SfHostTarget) == {4 * max(1, len(spec.target_fields))}u", True),
        (f"sizeof(SfHostWorkspaces) == {20 * max(1, len(spec.workspaces))}u", True),
        (f"sizeof(SfHostResolved) == 4u", True),
        (f"sizeof(SfHostAbi) == {layout.abi_bytes}u", True),
        (f"offsetof(SfHostAbi, inputs) == {layout.inputs_offset}u", True),
        (f"offsetof(SfHostAbi, target) == {layout.target_offset}u", True),
        (f"offsetof(SfHostAbi, workspaces) == {layout.workspaces_offset}u", True),
        (f"offsetof(SfHostAbi, resolved) == {layout.resolved_offset}u", True),
        (f"sizeof(SfTensorMeta) == 12u", True),
    ]
    for index, (expression, _) in enumerate(assertions):
        add(f"typedef char sf_assert_{index}[( ({expression}) ) ? 1 : -1];")
    add("")
    # Storage + static ABI
    add(f"static uint8_t sf_host_storage[{max(total, 16)}u] __attribute__((aligned(16)));")
    add("")
    add("static SfHostAbi sf_host_abi = {")
    add("    .inputs = {")
    if not spec.inputs:
        add("        .reserved = 0u,")
    for view in spec.inputs:
        if view.mapped:
            region = regions[view.name]
            add(f"        .{view.name} = ({view.ctype} *)(sf_host_storage + {region.data_offset}u),")
            add(f"        .{view.name}_numel = {view.numel}u,")
        else:
            # Metadata-only inputs: the typed read imports validate this id/dtype/range.
            add(
                f"        .{view.name} = {{ .id = {view.tensor_id}u, .numel = {view.numel}u, "
                f".elem_nbytes = {view.elem_nbytes}u }},"
            )
    add("    },")
    add("    .workspaces = {")
    if not spec.workspaces:
        add("        .reserved = 0u,")
    for view in spec.workspaces:
        upper = view.name.upper()
        if view.name in used:
            region = regions[view.name]
            add(f"#if SF_HOST_USE_{upper}")
            add(f"        .{view.name} = ({view.ctype} *)(sf_host_storage + {region.data_offset}u),")
            add("#else")
            add(f"        .{view.name} = ({view.ctype} *)(uintptr_t)0xFFFFFFFFu, /* unallocated trap address */")
            add("#endif")
        else:
            add(f"#if SF_HOST_USE_{upper}")
            add(f'#error "workspace {view.name} is used but was not allocated in this build"')
            add("#else")
            add(f"        .{view.name} = ({view.ctype} *)(uintptr_t)0xFFFFFFFFu, /* unallocated trap address */")
            add("#endif")
        add(f"        .{view.name}_capacity = {view.capacity}u,")
        add(f"        .{view.name}_offset = 0u,")
        add(f"        .{view.name}_count = 0u,")
        add(f"        .{view.name}_elem_nbytes = {view.elem_nbytes}u,")
    add("    },")
    add("};")
    add("")
    add("static void sf_host_init_guards(void) {")
    add("    uint32_t i;")
    for region in regions.values():
        add(f"    for (i = 0u; i < SF_HOST_GUARD_BYTES; i++) {{")
        add(f"        sf_host_storage[{region.guard_lo}u + i] = (uint8_t)SF_HOST_GUARD_PATTERN;")
        add(f"        sf_host_storage[{region.guard_hi}u + i] = (uint8_t)SF_HOST_GUARD_PATTERN;")
        add("    }")
    add("}")
    add("")
    add("uint32_t sf_host_abi_offset(void) {")
    add("    sf_host_init_guards();")
    add("    return (uint32_t)(uintptr_t)&sf_host_abi;")
    add("}")
    add("")
    # Imports: whitelisted, typed range reads for inputs that are not directly mapped.
    for import_name, dtype in INPUT_READ_IMPORTS:
        ctype = _C_DTYPE[dtype]
        add(f'__attribute__((import_module("sf"), import_name("{import_name}")))')
        add(f"extern int32_t {import_name}(uint32_t id, uint32_t elem_off, uint32_t dst_off, uint32_t count);")
    add("")
    add("/* The candidate supplies this entry.  Return SF_HOST_OK (0) to commit every span below;")
    add(" * any non-zero value, trap or fuel exhaustion discards all spans and resolved scalars. */")
    add("int32_t host_tiling(const SfHostInputs *inputs,")
    add("                    const SfHostShapes *shapes,")
    add("                    const SfHostTarget *target,")
    add("                    SfHostWorkspaces *workspaces,")
    add("                    SfHostResolved *resolved);")
    add("")
    add("#endif /* SF_HOST_CONTRACT_H */")
    return "\n".join(lines) + "\n"


def assembled_source(header: str, candidate_source: str) -> str:
    """@brief 头文件 + 候选源码的单一编译单元。

    @param header 生成的 ``host_contract.h`` 文本。
    @param candidate_source 候选 C99 源码。
    @return 拼接文本。
    """
    return header + "\n/* --- candidate source --- */\n" + candidate_source


def assembled_sha256(header: str, candidate_source: str) -> str:
    """@brief ``assembled_host_sha256``：头 + 源码的完整摘要。

    @param header 生成的头。
    @param candidate_source 候选源码。
    @return 十六进制 sha256。
    """
    return hashlib.sha256(assembled_source(header, candidate_source).encode("utf-8")).hexdigest()


def referenced_workspaces(source: str, spec: HostContractSpec) -> tuple[str, ...]:
    """@brief 候选源码实际引用的 workspace（决定哪些块需要在 Wasm 内存里分配）。

    只做名字级静态扫描：``workspaces->name`` / ``workspaces.name`` / ``SF_WS_NAME_*``。 一个
    未引用的 workspace 不分配，``log_decay_tab`` 这类大表就不会因为声明存在而被无条件实例化。
    真正的写入范围仍由运行期的 span 校验和 envelope 的 host/device_zero 冲突检查兜住。

    @param source 候选源码。
    @param spec 合同。
    @return 引用的 workspace 名（合同声明顺序）。
    """
    if not spec.workspaces:
        return ()
    # Match the declared names as whole identifiers so ``work_items_count`` / ``work_items_ptr``
    # resolve to the workspace ``work_items`` without relying on the source indexing the pointer.
    names = sorted((view.name for view in spec.workspaces), key=len, reverse=True)
    pattern = re.compile(
        r"\bworkspaces\s*(?:->|\.)\s*(?:" + "|".join(re.escape(name) for name in names) + r")(?![A-Za-z0-9])"
    )
    found = {match.group(0).split("->")[-1].split(".")[-1].strip() for match in pattern.finditer(source)}
    for match in re.finditer(r"\bSF_WS_([A-Z0-9_]+?)_(?:PTR|CAPACITY|CAPACITY_ELEMS|OFFSET|COUNT|ELEM_NBYTES|ELEM_NBYTES_VALUE)\b", source):
        found.add(match.group(1).lower())
    return tuple(view.name for view in spec.workspaces if view.name in found)


#: Return codes the candidate entry may use.  Kept in one place so the statement, the header and
#: the runtime cannot describe different protocols.
HOST_RETURN_CODES: tuple[tuple[str, str], ...] = (
    ("SF_HOST_OK", "0: commit every recorded span; the caller copies exactly the valid spans"),
    ("SF_HOST_REFUSED", "1: the candidate refuses this input; the launch fails, nothing is committed"),
    ("SF_HOST_INVALID_BOUNDS", "2: cu_seqlens/geometry invariant violated; nothing is committed"),
    ("SF_HOST_CAPACITY_EXCEEDED", "3: the plan does not fit the declared workspace capacity"),
    ("SF_HOST_IMPORT_REFUSED", "4: a whitelisted read import refused the requested range"),
)


def host_program_statement_section(instance: Any) -> str:
    """@brief 题面里的 Host Program 合同（完整 header + C99 子集 + 返回码）。

    只在 varlen 实例上渲染：固定 shape 题的题面、父包字节与既有证据身份保持不变。
    ``host_contract.h`` 由 OpSpec/ABI 派生，不是候选手写；实际 build 时头文件按候选引用的
    workspace 分配存储（未引用的块指针为 0，不占用 Wasm 线性内存）。

    @param instance ``SemanticInstanceV2``。
    @return Markdown/文本小节。
    """
    spec = host_contract_spec(instance=instance)
    lines: list[str] = []
    add = lines.append
    add("## Host Program (restricted C99 -> Wasm, optional)")
    add(
        "Instead of the declarative `host/tiling.json`, a candidate may ship `host/tiling.c`. "
        "The harness prepends the generated `host_contract.h` below and compiles the source with the "
        "standard Clang + LLVM wasm-ld toolchain into one Wasm module. The module is built once and "
        "executed by a maintained Wasm engine; each call gets a fresh Store, a fuel budget scaled by "
        "the declared capacity and a bounded linear memory. There is no libc, no WASI, no native "
        "library and no host pointer."
    )
    add("")
    add("### Entry point")
    add("```c")
    add("int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,")
    add("                    const SfHostTarget *target, SfHostWorkspaces *workspaces,")
    add("                    SfHostResolved *resolved);")
    add("```")
    add("")
    add("### C99 subset")
    add(
        "- allowed: ordinary functions, loops, conditionals, integer/floating arithmetic, bit "
        "operations, fixed-size arrays/structs, `static` scratch sized from the contract macros, "
        "and candidate-owned heaps, binary searches, LPT/range-cut or any other planner;"
    )
    add(
        "- allowed headers: none. `stdint.h`/`stddef.h` are already included by the generated "
        "header; a candidate `#include`, `#pragma`, inline assembly, libc call, function pointer, "
        "varargs, file/network/process/environment access or self-declared host import is refused "
        "before compilation;"
    )
    add(
        "- inputs: small tensors are typed pointers in the module's own linear memory (`inputs->cu_seqlens`, "
        "`inputs->scale`, ...). Large tensors expose only `id`/`numel`/`elem_nbytes`; read them through "
        "`sf_input_read_i32/u32/f32(id, elem_off, dst_off, count)`, which validates the id, dtype and range "
        "and records the actual interval;"
    )
    add(
        "- workspaces: one contiguous valid span per workspace, `offset`/`count` in elements "
        "(`workspaces->work_items_count`), with `0 <= offset` and `offset + count <= capacity`. "
        "Per-core layout must be packed inside that span. The tail outside the span is never "
        "initialised by the harness in the timed path; a device rereads it at its own risk;"
    )
    add(
        "- initialization is exclusive: a workspace this program writes is host-initialized; the "
        "geometry may not also declare it `zero` (device-zeroed). The conflict is refused at "
        "envelope time; `poison` tails are still checked by the poison correctness trial;"
    )
    add(
        "- the entry must return 0. A non-zero return, a Wasm trap or fuel exhaustion discards every "
        "span and resolved scalar from that call: no partial write can reach the device."
    )
    add("")
    add("### Return codes")
    for name, meaning in HOST_RETURN_CODES:
        add(f"- `{name}` — {meaning}")
    add("")
    add(
        "Because the program can read input tensor values, the lifecycle is derived, not declared: "
        "`input_dependent=true`, `preprocesses_inputs=true`, `inside_timing=true`. Host work is "
        "always part of the measured end-to-end time; there is no prepare cache in V3."
    )
    add("")
    add("### Generated `host_contract.h` (full ABI)")
    add("```c")
    lines.extend(spec.header().rstrip("\n").split("\n"))
    add("```")
    return "\n".join(lines)


def resolve_numel(term: Any, instance: Any, where: str) -> int:
    """@brief 公开的形状求值入口（测试/工具复用，不复制 ABI 解析规则）。"""
    return _resolve_numel(term, instance, where)


def resolved_numel(shape: Sequence[Any], instance: Any, where: str) -> int:
    """@brief 公开的 shape→numel 入口。"""
    return _numel(shape, instance, where)


__all__ = [
    "CONTRACT_SCHEMA_ID",
    "CONTRACT_SCHEMA_VERSION",
    "HOST_STACK_BYTES",
    "HostAbiLayout",
    "HostContractSpec",
    "HostField",
    "HostFieldLayout",
    "HostInputLayout",
    "HostInputView",
    "HostStorageRegion",
    "HostWorkspaceLayout",
    "HostWorkspaceView",
    "INPUT_READ_IMPORTS",
    "MAX_MAPPED_INPUT_BYTES",
    "MAX_TOTAL_MAPPED_BYTES",
    "STORAGE_GUARD_BYTES",
    "STORAGE_GUARD_PATTERN",
    "assembled_sha256",
    "assembled_source",
    "HOST_RETURN_CODES",
    "host_contract_spec",
    "host_program_statement_section",
    "referenced_workspaces",
    "resolve_numel",
    "resolved_numel",
    "storage_layout",
]
