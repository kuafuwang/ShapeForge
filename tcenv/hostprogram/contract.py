"""Backend-neutral Host Program contract derived from OpSpec/TensorMemoryContract.

Names, dtypes, capacities, input mapping and target knobs live here.  The native64 C renderer is
in ``native_abi.py`` and owns all structure layout through the C compiler; this module contains no
Python pointer-offset model.  Storage budgeting is separate from native layout.  Candidate-owned
algorithms never enter this contract.
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
#: A tensor at or below this many bytes gets a typed pointer in native worker storage.
MAX_MAPPED_INPUT_BYTES = 64 * 1024
#: Aggregate cap for directly mapped inputs.  Beyond it the tensor becomes metadata-only, so the
#: native worker storage cannot grow with the number of small-ish inputs.
MAX_TOTAL_MAPPED_BYTES = 1 << 20
#: Guard sentinel bytes before and after every allocated view.
STORAGE_GUARD_BYTES = 32
STORAGE_GUARD_PATTERN = 0xA5
#: Fixed stack reserved by the toolchain for the planner's own arrays/locals.
HOST_STACK_BYTES = 64 * 1024

#: Canonical dtype -> C type.  A dtype outside this map cannot be directly mapped *or* read
#: through an import in V3: it stays metadata-only (``id``/``numel``/``elem_nbytes``), which is
#: enough for a planner that only needs the declared algebra.
#:
#: fp16/bf16 have no portable C arithmetic type, so they use an explicit **raw-bit** ABI:
#: ``uint16_t`` (one element is one IEEE-754 binary16 / bfloat16 bit pattern).  The candidate
#: decodes with its own integer/bit operations; the harness never silently converts.  The same
#: raw-bit rule applies to the whitelisted ``sf_input_read_u16`` import for large tensors.
_C_DTYPE: Mapping[str, str] = {
    "i32": "int32_t",
    "u32": "uint32_t",
    "fp32": "float",
    "fp16": "uint16_t",
    "bf16": "uint16_t",
    "i16": "int16_t",
    "u16": "uint16_t",
    "i8": "int8_t",
    "u8": "uint8_t",
}

#: Import names the generated header declares and the executor is allowed to provide.  ``u16``
#: is the raw-bit read for fp16/bf16 (and i16/u16); its dtype check is "elem_nbytes == 2", so the
#: candidate must consult the tensor metadata for the semantic dtype.
INPUT_READ_IMPORTS: tuple[tuple[str, str], ...] = (
    ("sf_input_read_i32", "i32"),
    ("sf_input_read_u32", "u32"),
    ("sf_input_read_f32", "fp32"),
    ("sf_input_read_u16", "u16"),
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
    """@brief shapes/target 里的一个具名强类型字段。"""

    name: str
    ctype: str
    value: int | float | None = None


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
    #: Per-call runtime scalars (``SfHostResolved``).  Empty for a fixed-shape or L1 instance: those
    #: already carry ``total_tokens``/``num_seqs`` as *shape* fields, so adding a second copy would be a
    #: second authority for the same number and would change their generated header bytes.  An L2
    #: capacity instance has a static capacity shape and a runtime resolved shape, so it declares
    #: ``total_tokens``/``num_seqs`` here instead.
    resolved_fields: tuple[HostField, ...] = ()

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
            "resolved": [{"name": f.name, "ctype": f.ctype} for f in self.resolved_fields],
            "mapped_limit": int(self.mapped_limit),
        }

    @property
    def digest(self) -> str:
        """@brief 合同文档的规范 sha256。"""
        raw = json.dumps(self.document(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _resolved_fields(instance: Any) -> tuple[HostField, ...]:
    """@brief 执行期写入的 runtime resolved 字段（只有 L2 capacity 实例才有）。

    ``total_tokens``/``num_seqs`` are the *resolved* shape of this call; the capacity shape stays in
    ``SfHostShapes``.  A fixed/L1 instance keeps an empty tuple so its ``SfHostResolved`` remains the
    historical ``{reserved}`` struct, its generated header stays byte-identical, and every recorded L1
    ``assembled_host_sha256``/descriptor size stays valid.

    @param instance ``SemanticInstanceV2``。
    @return 字段元组。
    """
    if str(getattr(instance, "variable_length_kind", "fixed")) != "l2":
        return ()
    return (HostField("total_tokens", "uint32_t"), HostField("num_seqs", "uint32_t"))


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
        # §3.2: only declared read-only inputs are exposed.  A read_write state is device-owned;
        # letting the host read/write it would silently add a second mutation authority.
        if access != "read":
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
        resolved_fields=_resolved_fields(instance),
    )


# --- Storage budgeting (not a native structure layout) --------------------------


@dataclass(frozen=True)
class HostStorageRegion:
    """@brief 一个已分配视图在 module 线性内存中的边界。"""

    name: str
    kind: str
    data_offset: int
    nbytes: int
    guard_lo: int
    guard_hi: int


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


def referenced_workspaces(source: str, spec: HostContractSpec) -> tuple[str, ...]:
    """@brief 候选源码实际引用的 workspace（决定哪些块需要在 native worker 里分配）。

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


def minimal_host_example(spec: HostContractSpec) -> str:
    """@brief 题面里最小、可编译/可执行的 host_tiling 函数体。

    只使用合同里真实存在的 ``cu_seqlens``/``work_items``/``item_counts``/``handoff_flags``；不读取
    device 输出，不做 KDA 计算，只示范“读输入 → 写一个连续 span → 返回 0”的完整机制。它是候选
    自己书写的代码，不是 harness 规则，也不进任何 timed/candidate path。

    @param spec 合同。
    @return 完整 C 函数文本；合同不满足示例前提时返回只返回 0 的最小函数。
    """
    needs = {"work_items", "item_counts", "handoff_flags"}
    available = {view.name for view in spec.workspaces}
    try:
        cu = spec.input("cu_seqlens")
    except KeyError:
        cu = None
    if not needs <= available or cu is None or not cu.mapped:
        return (
            "int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,\n"
            "                    const SfHostTarget *target, SfHostWorkspaces *workspaces,\n"
            "                    SfHostResolved *resolved) {\n"
            "    (void)inputs; (void)shapes; (void)target; (void)workspaces; (void)resolved;\n"
            "    return SF_HOST_OK;  /* this instance declares no work_items example workspace */\n"
            "}\n"
        )
    return """int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)shapes; (void)target; (void)resolved;
    /* One piece for chain 0: the whole first sequence.  Bounds are checked before any write. */
    uint32_t seq = 0u;
    int32_t begin = inputs->cu_seqlens[seq];
    int32_t end = inputs->cu_seqlens[seq + 1u];
    if (end < begin) {
        return SF_HOST_INVALID_BOUNDS;
    }
    uint32_t len = (uint32_t)(end - begin);
    workspaces->work_items[0] = (uint32_t)begin | (len << 16);
    workspaces->work_items[1] = 0u;  /* chain 0, piece 0, pieces 1 */
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u;
    workspaces->item_counts[0] = 1;
    workspaces->item_counts_offset = 0u;
    workspaces->item_counts_count = 1u;
    workspaces->handoff_flags[0] = 0;  /* FLAG_EMPTY: the device may publish a state */
    workspaces->handoff_flags_offset = 0u;
    workspaces->handoff_flags_count = 1u;
    return SF_HOST_OK;
}
"""


def minimal_device_example(spec: HostContractSpec) -> str:
    """@brief 与上面 host 示例配套的最小 device kernel（只示范读取 host span）。

    这是示意性的 MSL：它假设三个 workspace 按声明顺序绑定到 buffer(0..2)。真实任务仍按完整
    ABI 绑定所有声明张量；题面明确说明这点，不把它冒充 KDA 正确 kernel。

    @param spec 合同。
    @return MSL 文本。
    """
    needed = {"work_items", "item_counts", "handoff_flags"}
    if not needed <= {view.name for view in spec.workspaces}:
        return "/* this instance has no work_items/item_counts/handoff_flags example */\n"
    return """#include <metal_stdlib>
using namespace metal;

kernel void example(device const uint *items [[buffer(0)]],
                    device const int *counts [[buffer(1)]],
                    device int *flags [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]) {
    if (tid >= (uint)counts[0]) {
        return;
    }
    flags[tid] = (int)(items[2u * tid] & 0xFFFFu);
}
"""


def minimal_geometry_example(spec: HostContractSpec) -> str:
    """@brief 与 host+device 示例配套的执行几何（最小单 stage）。"""
    needed = {"work_items", "item_counts", "handoff_flags"}
    if not needed <= {view.name for view in spec.workspaces}:
        return "/* no example geometry for this instance */\n"
    symbols = {field.name for field in spec.shape_fields}
    grid = "num_seqs*H" if {"num_seqs", "H"} <= symbols else "1"
    document = {
        "schema_version": 1,
        "source": "candidate",
        "stages": [
            {
                "kernel": "example",
                "grid": [grid, 1, 1],
                "threads": [1, 1, 1],
                "threadgroup_bytes": 0,
                "workspace": [
                    {"name": "work_items", "zero": False},
                    {"name": "item_counts", "zero": False},
                    {"name": "handoff_flags", "zero": False},
                ],
                "when": None,
                "derived_scalars": [],
            }
        ],
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def host_program_statement_section(instance: Any) -> str:
    """@brief 题面里的 Host Program 合同（完整 header + C99 子集 + 返回码）。

    只在 varlen 实例上渲染：固定 shape 题的题面、父包字节与既有证据身份保持不变。
    ``host_contract.h`` 由 OpSpec/ABI 派生，不是候选手写；实际 build 时头文件按候选引用的
    workspace 分配存储（未引用的块指针为 0，不占用 native worker 存储）。

    @param instance ``SemanticInstanceV2``。
    @return Markdown/文本小节。
    """
    spec = host_contract_spec(instance=instance)
    lines: list[str] = []
    add = lines.append
    add("## Host Program (restricted C99 -> native64 worker, optional)")
    add(
        "Instead of the declarative `host/tiling.json`, a candidate may ship `host/tiling.c`. "
        "The harness inserts it into the generated `host_contract.h` below and compiles one native64 "
        "worker-local dylib with the system Clang. Compilation, loading and execution happen only in "
        "the bounded v2 worker; the control host receives typed JSON results, never a native pointer. "
        "There is no Wasm module, Store, engine or fuel counter. Candidate libc/process/file/network "
        "access is refused by the restricted-C99 precheck, and the external worker watchdog is the "
        "runaway boundary. Native execution is not claimed to be a memory sandbox."
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
        "- inputs: small tensors are typed pointers in disjoint native64 worker regions "
        "(`inputs->cu_seqlens`, `inputs->scale`, ...). Large tensors expose only "
        "`id`/`numel`/`elem_nbytes`; read them through "
        "`sf_input_read_i32/u32/f32/u16(id, elem_off, (uintptr_t)dst, count)`, which validates "
        "the id, dtype and range and records the actual interval. The destination is a native "
        "worker address, never a Wasm32 offset;"
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
        "- the entry must return 0. A non-zero return, native worker crash/timeout, pointer/metadata "
        "change, guard hit or invalid span discards every span and resolved scalar from that call: "
        "no partial write can reach the device."
    )
    add("")
    add("### Return codes")
    for name, meaning in HOST_RETURN_CODES:
        add(f"- `{name}` — {meaning}")
    add("")
    add(
        "Because the program can read input tensor values, the lifecycle is derived, not declared: "
        "`input_dependent=true`, `preprocesses_inputs=true`, `inside_timing=true`. Host work is "
        "always part of the measured end-to-end time; there is no input-value prepare cache."
    )
    add("")
    add("### Minimal complete host+device example")
    add(
        "This is a **mechanism** example, not a correct KDA kernel: it reads `cu_seqlens`, writes one "
        "work item and the device kernel consumes the host-written span. The real task binds all "
        "declared tensors; a candidate owns its actual algorithm and launch."
    )
    add("")
    add("**host/tiling.c**")
    add("```c")
    lines.extend(minimal_host_example(spec).rstrip("\n").split("\n"))
    add("```")
    add("")
    add("**device/example.metal**")
    add("```metal")
    lines.extend(minimal_device_example(spec).rstrip("\n").split("\n"))
    add("```")
    add("")
    add("**host/execution_geometry.json**")
    add("```json")
    lines.extend(minimal_geometry_example(spec).rstrip("\n").split("\n"))
    add("```")
    add("")
    add("### Generated `host_contract.h` (full ABI)")
    add("```c")
    from .native_abi import NATIVE_SOURCE_MARKER, render_native_support

    header = render_native_support(spec).replace(
        NATIVE_SOURCE_MARKER, "/* candidate host/tiling.c is inserted here at build */"
    )
    lines.extend(header.rstrip("\n").split("\n"))
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
    "HostContractSpec",
    "HostField",
    "HostInputView",
    "HostStorageRegion",
    "HostWorkspaceView",
    "INPUT_READ_IMPORTS",
    "MAX_MAPPED_INPUT_BYTES",
    "MAX_TOTAL_MAPPED_BYTES",
    "STORAGE_GUARD_BYTES",
    "STORAGE_GUARD_PATTERN",
    "HOST_RETURN_CODES",
    "host_contract_spec",
    "host_program_statement_section",
    "minimal_device_example",
    "minimal_geometry_example",
    "minimal_host_example",
    "referenced_workspaces",
    "resolve_numel",
    "resolved_numel",
    "storage_layout",
]
