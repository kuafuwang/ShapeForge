"""OpSpec: the single semantic fact source for one operator (plan §4.1/§4.2).

An ``OpSpec`` document says *what* is computed: formula, symbol/shape domain, dtype
coverage, tensors and layout, numerics and tolerances, attrs, semantic case domain,
sizes with a ``usage`` split, and the structured work-accounting expression.  It never
carries grid/threads, an ABI, a toolchain, track text, device facts or a candidate
source; those belong to the LoweringProfile, BuildProvenance, the track binding and
the candidate respectively.

``instantiate(opspec, dtype, size)`` is the only place that turns those semantics into
one concrete ``SemanticInstance``; ``lower(instance, profile)`` (``tcenv.lowering``)
is the only place that turns it into a backend ``Contract``.

Work accounting has exactly one source: the ``work_accounting`` block below, evaluated
by the registered rule version ``WORK_ACCOUNTING_VERSION``.  There is no second
hand-written ``flops``/``min_bytes`` pair to disagree with it (F-19).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..dtypes import DTYPES, get_dtype
from ..identity import IdentityError, fingerprint, load_strict_json
from ..terms import TermError, eval_term, normalize_term
from .validator import SchemaFailure, validate

#: Version of the structural schema in ``schema/opspec.v1.schema.json``.
OPSPEC_SCHEMA_VERSION = 1
#: Rules for defaults, dtype/shape acceptance, tolerance materialisation and instance fields.
#: ``-2`` (R2) adds the closed ``cases.inputs`` domain keys (``_domain_case``/``_base``/
#: ``_inject``/``_max``): the evaluation domain is now a validated part of the OpSpec instead
#: of an unvalidated ``Task.meta`` dict.  That widens the accepted document domain, so the
#: registered rule version changes and every ``opspec_digest`` is deliberately reset.
SEMANTIC_RULES_VERSION = "opspec-semantics-3"
#: Rules for turning ``work_accounting`` into ``flops``/``min_bytes`` (single JSON type: float).
WORK_ACCOUNTING_VERSION = "opspec-work-accounting-expr-1"

SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
OPSPEC_SCHEMA_PATH = SCHEMA_DIR / "opspec.v1.schema.json"

VALID_USAGE = frozenset({"golden", "benchmark"})
_DOMAIN_KEYS = frozenset({"gt", "ge", "lt", "le", "enum"})
#: Closed set of ``_``-prefixed evaluation-domain keys allowed inside ``cases.inputs``.
_CASE_DOMAIN_KEYS = frozenset({"_domain_case", "_base", "_inject", "_max"})

#: Registered generated-program kinds (R2): the program is *semantic data*, so it lives in the
#: OpSpec document (and therefore in ``opspec_digest``) instead of a Python dict or a seed.
PROGRAM_KINDS = frozenset({"int32-block-program"})
#: Version of the numeric rule table carried in ``program.rules`` (renderers/evaluator read it).
PROGRAM_RULE_VERSION = 1
_PROGRAM_BINARY_OPS = frozenset({"+", "-", "*", "^", "&", "|"})
_PROGRAM_SCALAR_OPS = frozenset({"+", "-", "*", "^"})
_PROGRAM_COMPARE_OPS = frozenset({"<", ">", "<=", ">=", "==", "!="})
#: Statement fields each kind requires and forbids; a field that is not listed is forbidden.
_PROGRAM_FIELDS: Mapping[str, tuple[frozenset[str], frozenset[str]]] = {
    "bin": (frozenset({"op"}), frozenset({"c", "d", "rounds"})),
    "sbin": (frozenset({"op", "b"}), frozenset({"c", "d", "const", "rounds"})),
    "shl": (frozenset({"const"}), frozenset({"b", "c", "d", "op", "rounds"})),
    "shr": (frozenset({"const"}), frozenset({"b", "c", "d", "op", "rounds"})),
    "mod": (frozenset({"const"}), frozenset({"b", "c", "d", "op", "rounds"})),
    "where": (frozenset({"b", "c", "d", "op"}), frozenset({"const", "rounds"})),
    "neg": (frozenset(), frozenset({"b", "c", "d", "op", "const", "rounds"})),
    "hash": (frozenset({"const"}), frozenset({"b", "c", "d", "op", "rounds"})),
    "loop": (frozenset({"b", "const", "rounds"}), frozenset({"c", "d", "op"})),
    "sum": (frozenset(), frozenset({"b", "c", "d", "op", "const", "rounds"})),
    "max": (frozenset(), frozenset({"b", "c", "d", "op", "const", "rounds"})),
    "min": (frozenset(), frozenset({"b", "c", "d", "op", "const", "rounds"})),
}
_PROGRAM_BLOCK_NAMES = frozenset({"xv", "yv", "idx"})
_INT32_MIN, _INT32_MAX = -(1 << 31), (1 << 31) - 1


class OpSpecError(ValueError):
    """@brief OpSpec 结构或语义非法，或请求的 dtype/size 不在声明域内。"""


@dataclass(frozen=True)
class TensorBinding:
    """@brief 一个张量在某个 ``(dtype, size)`` 实例里的具体绑定。"""

    name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    strides: tuple[int, ...]
    layout: str
    space: str
    access: str
    doc: str = ""
    #: 声明里的 dim 词（v2 填充；v1 为空元组）。L2 的 logical/padding 区域按它识别 resolved 轴：
    #: 只比对 ``total_tokens``/``num_seqs`` 这两个词，绝不用“值恰好相等”去猜维度。
    dim_terms: tuple[str, ...] = ()
    #: 声明里的轴角色（只有 L2 的 capacity 语义会用到）：``(axis, "parameter")`` 表示该轴是独立参数
    #: 范围，即使长度恰好等于容量也不是 packed 的 token/sequence 轴。它是对形状代数无法区分
    #: ``dt_bias`` 的 ``H*D`` 与误声明的 packed 轴的显式答案，进实例身份。
    axis_roles: tuple[tuple[int, str], ...] = ()

    def declared_axis_role(self, axis: int) -> str | None:
        """@brief 该轴的显式角色（没有声明时为 ``None``）。

        @param axis 轴下标。
        @return ``"parameter"`` 或 ``None``。
        """
        for index, role in self.axis_roles:
            if int(index) == int(axis):
                return str(role)
        return None

    @property
    def numel(self) -> int:
        """@brief 元素个数。

        @return 形状各维乘积。
        """
        count = 1
        for dim in self.shape:
            count *= int(dim)
        return count

    @property
    def nbytes(self) -> int:
        """@brief 字节数。

        @return 元素个数乘该 dtype 的单元素字节数。
        """
        return self.numel * get_dtype(self.dtype).nbytes


@dataclass(frozen=True)
class Tolerances:
    """@brief 某个实例物化后的容差与精确比较标记。"""

    rtol: float
    atol_rel: float
    exact: bool


@dataclass(frozen=True)
class WorkAccounting:
    """@brief 由语义实例唯一推导出的记账值（R1 统一物化为 float）。"""

    flops: float
    min_bytes: float


@dataclass(frozen=True)
class CaseDomain:
    """@brief 语义用例域：声明顺序的用例名与逐 case 的输入域参数。"""

    cases: tuple[str, ...]
    inputs: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class CapacityShape:
    """@brief 一个 L2 (capacity) 档的**容量**形状与两条声明用的 padding 常量。

    L2 keeps one compiled ABI for many runtime batches: buffers, ABI spans and the Host Program
    storage are sized by this *capacity* shape, while ``resolved_total_tokens``/``resolved_num_seqs``
    (this call's real batch) are runtime facts.  L1 is the degenerate case where the resolved shape
    always equals the capacity, which is why every L1 symbol keeps its old meaning.

    ``cu_seqlens_sentinel`` fills ``cu_seqlens[num_seqs + 1:]`` and must not be a legal offset (it is
    negative), so a pointer walking past the batch reads an obviously invalid boundary instead of a
    convenient ``0``.  ``padding_poison_byte`` fills every element outside the logical region of a
    capacity-sized tensor; the harness validates both the input and the output/final-state padding.
    """

    total_tokens: int
    num_seqs: int
    cu_seqlens_sentinel: int
    padding_poison_byte: int

    def document(self) -> dict[str, int]:
        """@brief 该容量形状的 JSON 文档（进入 ``semantic_instance_digest``）。

        @return 字段字典。
        """
        return {
            "total_tokens": int(self.total_tokens),
            "num_seqs": int(self.num_seqs),
            "cu_seqlens_sentinel": int(self.cu_seqlens_sentinel),
            "padding_poison_byte": int(self.padding_poison_byte),
        }


@dataclass(frozen=True)
class WorkloadProfile:
    """@brief L2 的一个已声明负载档（canonical ``seq_lens`` + 稳定 id）。

    A capacity tier does not bind one segmentation: it declares ``SemanticInstanceV2.cu_seqlens=None``
    and publishes *workload profiles* instead.  A profile is the declared total/count a case runs, so
    the harness can reconstruct the expected resolved shape independently of the task's
    ``gen_inputs``.  Which segmentation inside that total/count a case generates remains the task's own
    coverage plan.
    """

    id: str
    seq_lens: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        """@brief 该 profile 的总 token 数。

        @return ``sum(seq_lens)``。
        """
        return int(sum(self.seq_lens))

    @property
    def num_seqs(self) -> int:
        """@brief 该 profile 的序列条数。

        @return ``len(seq_lens)``。
        """
        return len(self.seq_lens)

    def document(self) -> dict[str, Any]:
        """@brief 该 profile 的 JSON 文档（进入 ``semantic_instance_digest``）。

        @return 字段字典。
        """
        return {"id": self.id, "seq_lens": [int(value) for value in self.seq_lens]}


@dataclass(frozen=True)
class SemanticInstance:
    """@brief ``(opspec, dtype, size)`` 的唯一语义实例（派生 DTO，不是权威）。"""

    opspec_digest: str
    op_name: str
    category: str
    dtype: str
    size: str
    symbols: Mapping[str, int]
    tensors: tuple[TensorBinding, ...]
    resolved_attrs: Mapping[str, Any]
    spec_params: Mapping[str, str]
    tolerances: Tolerances
    case_domain: CaseDomain
    work_accounting: WorkAccounting
    acc_dtype: str
    size_usage: tuple[str, ...]

    def evaluation_meta(self) -> dict[str, Any]:
        """@brief 由 ``cases.inputs`` 物化的只读评测域视图（旧 ``Task.meta`` 的派生替代）。

        Only keys the OpSpec actually declares appear, so an instance without an evaluation
        domain yields ``{}``.  ``domain_case`` is dtype-scoped: the registered list names the
        dtypes whose declared domain is swept.

        @return ``domain_case``/``ieee_base``/``ieee_inputs``/``ieee_max`` 组成的字典。
        """
        meta: dict[str, Any] = {}
        cases = self.case_domain.inputs
        for values in cases.values():
            dtypes = values.get("_domain_case")
            if isinstance(dtypes, (list, tuple)) and self.dtype in dtypes:
                meta["domain_case"] = True
        ieee = cases.get("ieee")
        if isinstance(ieee, Mapping):
            if "_base" in ieee:
                meta["ieee_base"] = str(ieee["_base"])
            if "_inject" in ieee:
                meta["ieee_inputs"] = tuple(str(name) for name in ieee["_inject"])
            if "_max" in ieee:
                meta["ieee_max"] = {
                    str(name): int(eval_term(value, self.symbols, where=f"cases.inputs.ieee._max.{name}"))
                    for name, value in dict(ieee["_max"]).items()
                }
        return meta

    def tensor(self, name: str) -> TensorBinding:
        """@brief 按名字取张量绑定。

        @param name 张量名。
        @return 对应绑定。
        @raises KeyError 没有该张量。
        """
        for binding in self.tensors:
            if binding.name == name:
                return binding
        raise KeyError(name)

    def fingerprint_document(self) -> dict[str, Any]:
        """@brief §3.1 冻结的 SemanticInstance 投影（闭集）。

        @return 只含 JSON 值、字段闭集的派生文档。
        """
        document = {
            "opspec_digest": self.opspec_digest,
            "dtype": self.dtype,
            "size": self.size,
            "shapes": {
                "symbols": {name: int(value) for name, value in self.symbols.items()},
                "tensors": {binding.name: [int(dim) for dim in binding.shape] for binding in self.tensors},
            },
            "resolved_attrs": dict(self.resolved_attrs),
            "spec_params": dict(self.spec_params),
            "tolerances": {
                "rtol": float(self.tolerances.rtol),
                "atol_rel": float(self.tolerances.atol_rel),
                "exact": bool(self.tolerances.exact),
            },
            "case_domain": {
                "cases": list(self.case_domain.cases),
                "inputs": {case: dict(values) for case, values in self.case_domain.inputs.items()},
            },
            "work_accounting": {
                "flops": float(self.work_accounting.flops),
                "min_bytes": float(self.work_accounting.min_bytes),
            },
        }
        axis_roles = {
            binding.name: {str(int(axis)): str(role) for axis, role in binding.axis_roles}
            for binding in self.tensors
            if binding.axis_roles
        }
        # A role is part of L2 logical/padding semantics, not presentation metadata.  Add it only
        # when declared so fixed-shape and existing L1 projections stay byte-identical.
        if axis_roles:
            document["axis_roles"] = axis_roles
        return document


@dataclass(frozen=True)
class OpSpec:
    """@brief 一份通过 JSON Schema 与语义校验的 OpSpec 文档。"""

    document: Mapping[str, Any]
    source: str
    digest: str

    @property
    def spec_version(self) -> int:
        """@brief 文档结构版本（v1 或 v2）。

        The version is read from the document itself, so v1 and v2 are dispatched explicitly
        and no loader falls back from one to the other.

        @return ``document["spec_version"]``。
        """
        return int(self.document["spec_version"])

    @property
    def op_name(self) -> str:
        """@brief 算子名。

        @return ``document["op_name"]``。
        """
        return str(self.document["op_name"])

    @property
    def category(self) -> str:
        """@brief 算子类别。

        @return ``document["category"]``。
        """
        return str(self.document["category"])

    @property
    def dtypes(self) -> tuple[str, ...]:
        """@brief dtype 覆盖列表（type var ``T``）。

        @return 声明顺序的 dtype 短名。
        """
        return tuple(self.document["type_vars"]["T"])

    @property
    def sizes(self) -> Mapping[str, Any]:
        """@brief 已声明 size 的名字到定义。

        @return size 字典。
        """
        return self.document["sizes"]

    def digest_document(self) -> dict[str, Any]:
        """@brief §3.1 ``opspec_digest`` canonical envelope（按 spec_version 分派）。

        The public projection must recompute to the same digest the loader stored; a v2
        document therefore returns the v2 envelope instead of the v1 rules/version.

        @return 含 kind/version/规则版本与整份 OpSpec 的文档。
        """
        if self.spec_version == 2:
            from . import v2

            return v2.digest_envelope(self.document)
        return {
            "fingerprint_kind": "opspec",
            "fingerprint_version": 1,
            "semantic_rules_version": SEMANTIC_RULES_VERSION,
            "work_accounting_version": WORK_ACCOUNTING_VERSION,
            "document": dict(self.document),
        }


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    """@brief 断言一个值是非空 JSON 对象。

    @param value 待检查的值。
    @param where 诊断位置。
    @return 该对象。
    @raises OpSpecError 不是对象。
    """
    if not isinstance(value, Mapping) or not value:
        raise OpSpecError(f"{where} must be a non-empty object")
    return value


def _check_symbol(name: str, symbols: Mapping[str, Any], where: str) -> None:
    """@brief 断言符号已声明。

    @param name 符号名。
    @param symbols 已声明符号表。
    @param where 诊断位置。
    @raises OpSpecError 未声明。
    """
    if name not in symbols:
        raise OpSpecError(f"{where}: symbol {name!r} is not declared in symbols")


def _resolve_term(text: Any, symbols: Mapping[str, int], where: str) -> int:
    """@brief 解析一个 dim/stride 词（整数、符号名或 ``*`` 连接的符号积）。

    @param text 整数、符号名或 ``A*B`` 形式的符号积。
    @param symbols 已解析符号值。
    @param where 诊断位置。
    @return 具体整数。
    @raises OpSpecError 类型或符号非法。
    """
    if isinstance(text, bool) or not isinstance(text, (int, str)):
        raise OpSpecError(f"{where}: expected an integer or a symbol name, got {text!r}")
    if isinstance(text, int):
        if text < 0:
            raise OpSpecError(f"{where}: {text!r} must not be negative")
        return text
    if text.isdigit():
        return int(text)
    if "*" in text:
        product = 1
        for part in text.split("*"):
            if part.isdigit():
                product *= int(part)
                continue
            if part not in symbols:
                raise OpSpecError(f"{where}: symbol {part!r} is not declared")
            product *= int(symbols[part])
        return product
    if text not in symbols:
        raise OpSpecError(f"{where}: symbol {text!r} is not declared")
    return int(symbols[text])


#: The two symbols that shrink between calls on an L2 (capacity) tier.  They are the only axes the
#: logical/padding decomposition may treat as resolved.
RESOLVED_AXIS_NAMES: tuple[str, ...] = ("total_tokens", "num_seqs")
#: The one varlen tensor whose capacity-sized axis is not a logical/padding axis: its unused tail holds
#: the declared sentinel and is validated from the bytes instead.
BOUNDARY_TENSOR: str = "cu_seqlens"


def resolve_axis_kind(
    dim: int,
    term: Any,
    symbols: Mapping[str, int],
    *,
    where: str,
) -> tuple[str, str | None]:
    """@brief 把一个声明 dim 轴分类为 resolved 轴或可证明的 invariant 轴，否则拒绝。

    This is the single rule behind the L2 logical/padding contract (mirrored at runtime by
    :func:`tcenv.taskcontract.runtime_shape.axis_kind`).  ``symbols`` must already contain the tier's
    resolved capacity values, so ``total_tokens``/``num_seqs`` are the *capacity*, not a batch.

    Accepted:

    ``("resolved", name)``
        the term is exactly ``total_tokens`` or ``num_seqs``; the axis shrinks to the resolved value.
    ``("invariant", None)``
        the term is a product of declared symbols, none of which is a resolved symbol and none of which
        has a capacity value equal to the capacity ``total_tokens``/``num_seqs``.  Each factor is an
        independently declared extent, so the axis cannot stand in for the resolved axis.

    Refused (instead of being silently treated as invariant):

    * an opaque symbol whose only declared value *is* the capacity (the ``T2 = total_tokens`` alias): it
      is indistinguishable from the resolved axis and would hide the unused region from both the
      padding check and the logical comparison;
    * a fused term (``total_tokens*H``): no single axis shrinks, so the logical region is undefined;
    * a term naming an undeclared symbol.

    A product such as the parameter extent ``H*D`` stays legal even when it numerically equals the
    capacity total: its factors are declared independently and size the rest of the ABI, and a numeric
    collision is not evidence that the axis is per-token.  The dangerous case is an opaque symbol whose
    declared value is exactly the batch bound, i.e. an alias.

    @param dim 该轴的声明（容量）长度。
    @param term 该轴的声明 dim 词。
    @param symbols 该档已解析符号（含容量派生的 resolved 值）。
    @param where 诊断位置。
    @return ``("resolved", name)`` 或 ``("invariant", None)``。
    @raises OpSpecError 无法证明该轴是 invariant 且不是 canonical resolved 轴。
    """
    if isinstance(term, int) and not isinstance(term, bool):
        parts = [str(int(term))]
    elif isinstance(term, str):
        parts = str(term).split("*")
    else:
        raise OpSpecError(f"{where}: expected an integer or a symbol name, got {term!r}")
    resolved = [part for part in parts if part in RESOLVED_AXIS_NAMES]
    if resolved:
        if len(parts) != 1:
            raise OpSpecError(
                f"{where}: the dim term {term!r} fuses the resolved symbol {resolved[0]} with another factor, but "
                "a capacity axis must be the standalone symbol so its logical region shrinks to the resolved value "
                "and the remaining rows are auditable padding"
            )
        name = resolved[0]
        if name not in symbols:
            raise OpSpecError(f"{where}: the resolved symbol {name!r} is not instantiated for this tier")
        if int(dim) != int(symbols[name]):
            raise OpSpecError(
                f"{where}: the resolved axis {name!r} is bound to length {int(dim)}, but the capacity symbol is "
                f"{int(symbols[name])}; the logical region would not be a prefix of the capacity"
            )
        return "resolved", name
    for part in parts:
        if not part.isdigit() and part not in symbols:
            raise OpSpecError(f"{where}: symbol {part!r} is not declared")
    bounds = {name: int(symbols[name]) for name in RESOLVED_AXIS_NAMES if name in symbols}
    colliding = [
        (part, int(symbols[part]))
        for part in parts
        if not part.isdigit() and int(symbols[part]) in tuple(bounds.values())
    ]
    if colliding:
        part, value = colliding[0]
        detail = ", ".join(f"{name}={bound}" for name, bound in bounds.items())
        raise OpSpecError(
            f"{where}: the dim term {term!r} is not a canonical resolved axis, but its factor {part!r} has the "
            f"capacity value {value}, which equals the capacity {detail}. The harness cannot prove this axis is "
            "not a resolved-axis alias, and an aliased row axis would hide the unused region from both the "
            "padding check and the logical comparison. Declare the axis as 'total_tokens'/'num_seqs', or express "
            f"the invariant extent through symbols that are not at the capacity value (got {term!r})"
        )
    return "invariant", None


def capacity_axis_role_problem(
    dim: int,
    term: Any,
    symbols: Mapping[str, int],
    *,
    declared_role: str | None,
    is_result: bool,
    where: str,
) -> str | None:
    """@brief 一个 capacity 尺寸轴的角色是否**可证明**（canonical resolved 或显式参数范围）。

    :func:`resolve_axis_kind` alone cannot close this class: a shape algebra has no way to tell a
    legitimate ``dt_bias`` (`H*D == total_tokens` as an independent parameter extent) from a packed
    axis/token axis that merely *evaluates* to the capacity (``"4"``, ``"2*2"``, ``"H*2"``, ``"H*D"``), so
    a numeric rule would either miss the aliases or reject the independent product.  The declaration has to
    settle it, and this function is the single rule both the document validator and the runtime use:

    * a **result** tensor's capacity-sized axis must name ``total_tokens``/``num_seqs``.  Otherwise the
      logical region becomes the whole capacity buffer: the error scale is dominated by the padding poison
      (a wrong logical result then passes) and the padding check has no region to verify;
    * a **non-result** tensor (an input) that carries a capacity-sized axis which is not canonical must
      declare that axis as an independent *parameter* extent (``axis_roles: {<axis>: "parameter"}`).  Without
      the declaration the harness would silently treat a packed input as a parameter array, so a custom
      ``error_scale`` hook would be handed the padding it must never see (a plausible
      ``sqrt(mean(inputs["q"]**2))`` then inflates the final-state denominator and a wrong logical result
      passes);
    * when the capacity declares ``total_tokens == num_seqs == N (>1)`` the two canonical names are
      numerically identical, so no axis of length ``N`` can be *proven* to be the one it names: the tier must
      declare distinct capacity bounds (a 1/1 capacity is exempt because both resolved values are then 1);
    * a declared role that contradicts a canonical term, or is placed on a result tensor, is refused.

    ``declared_role`` is the tensor's explicit role for this axis (``"parameter"``) or ``None``.  An
    unrecognised role string is refused instead of being ignored.

    @param dim 该轴的声明（容量）长度。
    @param term 该轴的声明 dim 词。
    @param symbols 已解析符号（含容量 resolved 值）。
    @param declared_role 该轴的显式角色（``"parameter"`` 或 ``None``）。
    @param is_result 该张量是否是 reference 结果（role=output 或 access=read_write）。
    @param where 诊断位置。
    @return 问题描述；轴不是 capacity 尺寸或角色可证明时为 ``None``。
    """
    resolved_values = {name: int(symbols[name]) for name in RESOLVED_AXIS_NAMES if name in symbols}
    bounds = set(resolved_values.values())
    if declared_role is not None and declared_role != "parameter":
        return f"{where}: unknown axis role {declared_role!r}; the only declarable role is 'parameter'"
    if int(dim) not in bounds:
        # The declaration is inert where the axis is not capacity-sized (an extent that does not collide
        # with the capacity is already provably invariant).  Refusing it would make one tensor's global
        # declaration fail a *different* tier, which is worse than ignoring a no-op.
        return None
    ambiguous = len(resolved_values) == len(RESOLVED_AXIS_NAMES) and len(bounds) == 1 and int(dim) > 1
    try:
        kind, _name = resolve_axis_kind(int(dim), term, symbols, where=where)
    except OpSpecError:
        # The regular per-axis classification already refused this axis (alias/fused/missing); adding a
        # second complaint would only bury the more precise one.
        return None
    canonical = kind == "resolved"
    if declared_role is not None:
        if canonical:
            return (
                f"{where}: axis role 'parameter' contradicts the canonical resolved term {term!r}; a "
                "total_tokens/num_seqs axis is the logical axis and cannot also be an independent parameter extent"
            )
        if is_result:
            return (
                f"{where}: a result tensor cannot declare its capacity-sized axis as a 'parameter' extent; "
                "the logical region would be the whole capacity buffer and the error scale would include the "
                "padding poison"
            )
        return None
    if is_result:
        if ambiguous:
            return (
                f"{where}: the capacity declares total_tokens == num_seqs == {int(dim)}, so a result axis of "
                f"length {int(dim)} cannot be proven to be the token axis or the sequence axis; declare distinct "
                "capacity bounds (a 1/1 capacity is exempt because both resolved values are then 1)"
            )
        if canonical:
            return None
        return (
            f"{where}: a result tensor axis of length {int(dim)} equals a capacity resolved bound, but its "
            f"declared term {term!r} is not 'total_tokens'/'num_seqs'; the logical region would be the whole "
            "capacity buffer, so the error scale would include the padding poison and the padding check would "
            "have no region to verify. Name the axis 'total_tokens'/'num_seqs' (a parameter extent of the same "
            "length belongs on a non-result tensor)"
        )
    if canonical:
        if ambiguous:
            return (
                f"{where}: the capacity declares total_tokens == num_seqs == {int(dim)}, so an axis of length "
                f"{int(dim)} cannot be proven to be the token axis or the sequence axis; declare distinct "
                "capacity bounds (a 1/1 capacity is exempt because both resolved values are then 1)"
            )
        return None
    return (
        f"{where}: a capacity-sized axis of length {int(dim)} with term {term!r} is neither a canonical "
        "resolved axis ('total_tokens'/'num_seqs') nor declared as an independent parameter extent; the "
        "logical/padding region cannot be proven, and a custom error_scale hook would be handed the padding "
        "poison. Name the packed axis 'total_tokens'/'num_seqs', or declare "
        "{'axis_roles': {'<axis>': 'parameter'}} if this axis really is a parameter extent"
    )


def _resolve_domain(attr: str, spec: Mapping[str, Any], value: Any, where: str) -> Any:
    """@brief 校验一个已物化 attr 值是否落在登记的 domain 文法内。

    @param attr attr 名。
    @param spec attr 定义（含可选 ``domain``）。
    @param value 已物化值。
    @param where 诊断位置。
    @return 原值。
    @raises OpSpecError 越域或 domain 文法是未知约束。
    """
    domain = spec.get("domain")
    if domain is None:
        return value
    unknown = set(domain) - _DOMAIN_KEYS
    if unknown:
        raise OpSpecError(f"{where}: attr {attr!r} has unsupported domain constraints {sorted(unknown)}")
    for key, bound in domain.items():
        if key == "enum":
            if value not in bound:
                raise OpSpecError(f"{where}: attr {attr!r} value {value!r} is outside the declared enum")
            continue
        if key == "gt" and not value > bound:
            raise OpSpecError(f"{where}: attr {attr!r} value {value!r} must be > {bound!r}")
        if key == "ge" and not value >= bound:
            raise OpSpecError(f"{where}: attr {attr!r} value {value!r} must be >= {bound!r}")
        if key == "lt" and not value < bound:
            raise OpSpecError(f"{where}: attr {attr!r} value {value!r} must be < {bound!r}")
        if key == "le" and not value <= bound:
            raise OpSpecError(f"{where}: attr {attr!r} value {value!r} must be <= {bound!r}")
    return value


def validate_opspec(document: Any, *, source: str = "<opspec>") -> None:
    """@brief 按 ``spec_version`` 显式分派的结构 schema + 语义规则校验。

    @param document 已由严格 JSON loader 解析的文档。
    @param source 诊断用的来源名。
    @raises OpSpecError 结构、版本或语义非法。
    """
    version = document.get("spec_version") if isinstance(document, Mapping) else None
    if isinstance(version, bool) or not isinstance(version, int):
        raise OpSpecError(f"{source}: spec_version must be the integer 1 or 2, got {version!r}")
    if version == 2:
        from . import v2

        v2.validate_opspec_v2(document, source=source)
        return
    if version != 1:
        raise OpSpecError(f"{source}: unsupported spec_version {version!r}; registered versions are 1 and 2")
    try:
        schema = load_strict_json(OPSPEC_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - the registered schema is shipped
        raise OpSpecError(f"registered OpSpec schema is unreadable: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise OpSpecError(f"{source}: {exc}") from exc
    if not isinstance(document, Mapping):  # pragma: no cover - schema already enforces it
        raise OpSpecError(f"{source}: an OpSpec must be a JSON object")

    symbols = document["symbols"]
    tensors = document["tensors"]
    attrs = document["attrs"]
    uses = list(document["uses"])

    if sorted(uses) != sorted(attrs):
        raise OpSpecError(
            f"{source}: attrs and uses must list the same names; attrs={sorted(attrs)} uses={sorted(uses)}"
        )
    for name in uses:
        _check_symbol_like(name, f"{source}.uses")
    for dtype in document["type_vars"]["T"]:
        if dtype not in DTYPES:
            raise OpSpecError(f"{source}: type var T lists unknown dtype {dtype!r}")

    seen_tensors: set[str] = set()
    for name, tensor in tensors.items():
        if name in seen_tensors:
            raise OpSpecError(f"{source}: duplicate tensor {name!r}")
        seen_tensors.add(name)
        dtype = tensor["dtype"]
        if dtype != "T" and dtype not in DTYPES:
            raise OpSpecError(f"{source}: tensor {name!r} dtype {dtype!r} is not a registered dtype")
        if len(tensor["strides"]) != len(tensor["dims"]):
            raise OpSpecError(f"{source}: tensor {name!r} must declare one stride per dim")
        role, access = tensor["role"], tensor["access"]
        if role == "input" and access not in ("read", "read_write"):
            raise OpSpecError(f"{source}: input tensor {name!r} must be read/read_write, got {access!r}")
        if role == "output" and access not in ("write", "read_write"):
            raise OpSpecError(f"{source}: output tensor {name!r} must be write/read_write, got {access!r}")
        for dim in tensor["dims"]:
            _check_symbol(dim, symbols, f"{source}.tensors.{name}.dims")

    for attr, spec in attrs.items():
        value = _resolve_weight(spec, str(spec["dtype"]), attr)
        _resolve_domain(attr, spec, value, f"{source}.attrs.{attr}")

    declared_params = list(document["spec_params"])
    if len(set(declared_params)) != len(declared_params):
        raise OpSpecError(f"{source}: spec_params must not repeat a symbol")
    for name in declared_params:
        _check_symbol(name, symbols, f"{source}.spec_params")

    cases = list(document["cases"]["names"])
    if len(set(cases)) != len(cases):
        raise OpSpecError(f"{source}: cases.names must not repeat a case")
    declared_inputs = document["cases"].get("inputs", {})
    for case, values in declared_inputs.items():
        if case not in cases:
            raise OpSpecError(f"{source}: cases.inputs names unknown case {case!r}")
        for key, value in values.items():
            if key.startswith("_"):
                _validate_case_domain(source, case, key, value, cases, tensors, document)
                continue
            if key not in attrs:
                raise OpSpecError(f"{source}: cases.inputs.{case} names unknown attr {key!r}")

    for size, definition in document["sizes"].items():
        _validate_size(source, size, definition, symbols, varlen=False)

    _validate_accounting(source, document["work_accounting"], symbols, tensors)

    if "program" in document:
        _validate_program(source, document["program"], tensors)


def _validate_case_domain(
    source: str,
    case: str,
    key: str,
    value: Any,
    cases: list[str],
    tensors: Mapping[str, Any],
    document: Mapping[str, Any],
) -> None:
    """@brief 校验一个 ``_`` 前缀的评测域键（闭集，不能靠自由文本决定采样域）。

    @param source 诊断来源。
    @param case 所在 case 名。
    @param key ``_`` 前缀键名。
    @param value 声明值。
    @param cases 已声明 case 列表。
    @param tensors 已声明张量。
    @param document 整份 OpSpec。
    @raises OpSpecError 键未知、类型不符或引用未声明名字。
    """
    where = f"{source}.cases.inputs.{case}.{key}"
    if key not in _CASE_DOMAIN_KEYS:
        raise OpSpecError(f"{where}: unknown evaluation-domain key; registered keys are {sorted(_CASE_DOMAIN_KEYS)}")
    if key == "_domain_case":
        if not isinstance(value, list) or not value:
            raise OpSpecError(f"{where}: must be a non-empty list of declared dtypes")
        declared = list(document["type_vars"]["T"])
        for dtype in value:
            if dtype not in declared:
                raise OpSpecError(f"{where}: {dtype!r} is not in type_vars.T {declared}")
        return
    if case != "ieee":
        raise OpSpecError(f"{where}: {key!r} is only valid on the 'ieee' case")
    if key == "_base":
        if not isinstance(value, str) or value not in cases:
            raise OpSpecError(f"{where}: must name a declared case, got {value!r}")
        if value == "ieee":
            raise OpSpecError(f"{where}: the ieee case cannot be its own base")
        return
    if key == "_inject":
        if not isinstance(value, list) or not value:
            raise OpSpecError(f"{where}: must be a non-empty list of tensor names")
        for name in value:
            if name not in tensors:
                raise OpSpecError(f"{where}: unknown tensor {name!r}")
        return
    if not isinstance(value, Mapping) or not value:
        raise OpSpecError(f"{where}: must be a non-empty object of tensor name -> magnitude")
    for name, magnitude in value.items():
        if name not in tensors:
            raise OpSpecError(f"{where}: unknown tensor {name!r}")
        if isinstance(magnitude, Mapping):
            try:
                normalize_term(magnitude, where=f"{where}.{name}")
            except TermError as exc:
                raise OpSpecError(str(exc)) from exc
            continue
        if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
            raise OpSpecError(f"{where}.{name}: magnitude must be numeric or a structured term, got {magnitude!r}")


def _check_symbol_like(name: str, where: str) -> None:
    """@brief 断言一个名字是合法的 attr 标识符。

    @param name 名字。
    @param where 诊断位置。
    @raises OpSpecError 名字非法。
    """
    if not isinstance(name, str) or not name.startswith(tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")):
        raise OpSpecError(f"{where}: {name!r} must start with an uppercase letter")


#: varlen 派生符号：由 ``seq_lens``/``capacity`` 算出，不写进档位。
_DERIVED_VARLEN_SYMBOLS = frozenset({"total_tokens", "num_seqs", "NSP1"})
#: 一个 capacity 档的 ``capacity`` 块允许且必需的键（闭集）。
_CAPACITY_FIELDS = frozenset({"total_tokens", "num_seqs", "cu_seqlens_sentinel", "padding_poison_byte"})
#: 一个 workload profile 的键（闭集）。
_PROFILE_FIELDS = frozenset({"id", "seq_lens"})
_PROFILE_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _require_int(value: Any, where: str) -> int:
    """@brief 要求一个 JSON 整数（拒绝 bool/浮点/字符串）。

    @param value 声明值。
    @param where 诊断位置。
    @return 整数值。
    @raises OpSpecError 不是整数。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpSpecError(f"{where}: must be an integer, got {value!r}")
    return int(value)


def _validate_capacity_tier(
    source: str,
    size: str,
    definition: Mapping[str, Any],
    symbols: Mapping[str, Any],
) -> None:
    """@brief 校验一个 L2 capacity 档：容量派生规则、两条 padding 常量与 workload profile 集合。

    A capacity tier is what lets one compiled ABI serve many runtime batches, so the document has to
    declare — not just imply — three different things:

    * the **capacity** shape (``total_tokens``/``num_seqs``) that sizes every buffer, ABI span and
      Host Program storage;
    * the two padding markers (``cu_seqlens_sentinel`` for the boundary tail, ``padding_poison_byte``
      for every element outside the logical region), because a tail that may legitimately be ``0`` or
      ``0xA5``-by-accident cannot be audited;
    * at least two **workload profiles** with at least two distinct totals *and* counts, so a task
      cannot declare an L2 tier and then only ever run its capacity point.  At least one profile must
      reach full capacity, which is the only case where the resolved shape equals the capacity — the
      L1-degenerate boundary.

    The derived symbols (``total_tokens``/``num_seqs``/``NSP1``) are computed from the capacity and
    must not also be written in the tier, exactly like the ``seq_lens`` rule.

    @param source 诊断来源。
    @param size 档位名。
    @param definition 档位定义。
    @param symbols 已声明符号。
    @raises OpSpecError 非法。
    """
    where = f"{source}.sizes.{size}"
    capacity = definition["capacity"]
    if not isinstance(capacity, Mapping):
        raise OpSpecError(f"{where}.capacity: must be an object")
    keys = set(capacity)
    if keys != _CAPACITY_FIELDS:
        raise OpSpecError(
            f"{where}.capacity: has fields {sorted(keys)}; a capacity block declares exactly {sorted(_CAPACITY_FIELDS)}"
        )
    total_tokens = _require_int(capacity["total_tokens"], f"{where}.capacity.total_tokens")
    num_seqs = _require_int(capacity["num_seqs"], f"{where}.capacity.num_seqs")
    sentinel = _require_int(capacity["cu_seqlens_sentinel"], f"{where}.capacity.cu_seqlens_sentinel")
    poison = _require_int(capacity["padding_poison_byte"], f"{where}.capacity.padding_poison_byte")
    if total_tokens < 1:
        raise OpSpecError(f"{where}.capacity.total_tokens: must be >= 1 (an empty batch is not a valid capacity)")
    if total_tokens > _INT32_MAX:
        raise OpSpecError(
            f"{where}.capacity.total_tokens: {total_tokens} exceeds the i32 range of the cu_seqlens ABI "
            f"({_INT32_MAX}); a wider total needs a separate int64 contract"
        )
    if num_seqs < 1:
        raise OpSpecError(f"{where}.capacity.num_seqs: must be >= 1")
    if not (_INT32_MIN <= sentinel < 0):
        raise OpSpecError(
            f"{where}.capacity.cu_seqlens_sentinel: must be a negative int32 so it cannot be mistaken for a "
            f"legal offset, got {sentinel}"
        )
    if not (1 <= poison <= 255):
        raise OpSpecError(
            f"{where}.capacity.padding_poison_byte: must be in 1..255 (a convenient 0 is not a poison marker), "
            f"got {poison}"
        )
    derived_values = {"total_tokens": total_tokens, "num_seqs": num_seqs, "NSP1": num_seqs + 1}
    repeated = sorted(set(derived_values) & set(definition))
    if repeated:
        raise OpSpecError(
            f"{where}: a capacity tier derives {repeated} from capacity and must not declare them"
        )
    undeclared = sorted(name for name in derived_values if name not in symbols)
    if undeclared:
        raise OpSpecError(
            f"{where}: a capacity tier derives {undeclared} from 'capacity', so they must be declared in "
            "symbols (with their min) even though the tier does not write them"
        )
    for name, value in derived_values.items():
        minimum = int(symbols[name].get("min", 0))
        if value < minimum:
            raise OpSpecError(
                f"{where}: the tier derives {name}={value} from capacity, below the declared min {minimum}"
            )
    profiles = definition["workload_profiles"]
    if not isinstance(profiles, list) or len(profiles) < 2:
        raise OpSpecError(
            f"{where}.workload_profiles: an L2 tier declares at least two workload profiles (one capacity point "
            "is an L1 tier, not continuous batching)"
        )
    seen_ids: set[str] = set()
    totals: set[int] = set()
    counts: set[int] = set()
    full = False
    for index, profile in enumerate(profiles):
        at = f"{where}.workload_profiles[{index}]"
        if not isinstance(profile, Mapping):
            raise OpSpecError(f"{at}: must be an object")
        fields = set(profile)
        if fields != _PROFILE_FIELDS:
            raise OpSpecError(f"{at}: has fields {sorted(fields)}; a workload profile declares exactly {sorted(_PROFILE_FIELDS)}")
        identifier = profile["id"]
        if not isinstance(identifier, str) or _PROFILE_ID_RE.fullmatch(identifier) is None:
            raise OpSpecError(f"{at}.id: must match {_PROFILE_ID_RE.pattern}, got {identifier!r}")
        if identifier in seen_ids:
            raise OpSpecError(f"{at}.id: duplicates the profile id {identifier!r}")
        seen_ids.add(identifier)
        seq_lens = profile["seq_lens"]
        if not isinstance(seq_lens, list) or not seq_lens:
            raise OpSpecError(f"{at}.seq_lens: must be a non-empty list of positive integers")
        for value in seq_lens:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OpSpecError(f"{at}.seq_lens: every segment length must be a positive integer")
        profile_total = sum(int(value) for value in seq_lens)
        if profile_total > total_tokens:
            raise OpSpecError(
                f"{at}.seq_lens: total_tokens {profile_total} exceeds the declared capacity {total_tokens}"
            )
        if len(seq_lens) > num_seqs:
            raise OpSpecError(
                f"{at}.seq_lens: {len(seq_lens)} sequences exceed the declared capacity {num_seqs}"
            )
        totals.add(profile_total)
        counts.add(len(seq_lens))
        if profile_total == total_tokens and len(seq_lens) == num_seqs:
            full = True
    if len(totals) < 2 or len(counts) < 2:
        raise OpSpecError(
            f"{where}.workload_profiles: needs at least two distinct resolved totals and two distinct resolved "
            f"counts, got totals {sorted(totals)} and counts {sorted(counts)}; a capacity tier that only ever "
            "runs one resolved shape is L1 under another name"
        )
    if not full:
        raise OpSpecError(
            f"{where}.workload_profiles: at least one profile must reach the full capacity "
            f"(total_tokens={total_tokens}, num_seqs={num_seqs}); the capacity point itself has to be covered"
        )


def _validate_size(
    source: str,
    size: str,
    definition: Mapping[str, Any],
    symbols: Mapping[str, Any],
    *,
    varlen: bool,
) -> None:
    """@brief 校验一个 size 档位：符号齐全、无未知键、usage 合法、varlen 分布合法。

    ``_validate_size`` is shared by v1 and v2, but ``seq_lens`` is a v2-only declaration: a v1
    document has fixed-shape tiers only, so passing ``varlen=False`` refuses the field instead of
    silently accepting a distribution v1 cannot instantiate.  The derived-symbol exemption is
    likewise scoped to a *legal varlen tier*: a fixed tier that omits ``total_tokens``/``num_seqs``/
    ``NSP1`` is refused here, at validation time, instead of raising ``KeyError`` during
    instantiation.

    @param source 诊断来源。
    @param size 档位名。
    @param definition 档位定义。
    @param symbols 已声明符号。
    @param varlen 本 spec 版本是否允许 varlen 档（只有 v2 允许）。
    @raises OpSpecError 非法。
    """
    if not isinstance(definition, Mapping):
        raise OpSpecError(f"{source}.sizes.{size}: must be an object")
    for field in ("seq_lens", "capacity", "workload_profiles"):
        if field in definition and not varlen:
            raise OpSpecError(
                f"{source}.sizes.{size}: declares {field!r}, but a varlen/capacity tier is an OpSpec v2 "
                "feature; a v1 document declares fixed-shape tiers only"
            )
    if "capacity" in definition:
        # L2：容量档用 ``capacity`` + ``workload_profiles`` 代替 ``seq_lens``；两种声明互斥，
        # 因为一个档不能既把某一套分段写进语义身份、又把总形状留给运行时。
        if "seq_lens" in definition:
            raise OpSpecError(
                f"{source}.sizes.{size}: declares both 'seq_lens' (L1 canonical segmentation) and 'capacity' "
                "(L2 capacity); a tier is exactly one of fixed-shape, L1 or L2"
            )
        extra = set(definition) - set(symbols) - {"usage", "capacity", "workload_profiles"}
        if extra:
            raise OpSpecError(f"{source}.sizes.{size}: unknown fields {sorted(extra)}")
        if "workload_profiles" not in definition:
            raise OpSpecError(
                f"{source}.sizes.{size}.workload_profiles: a capacity tier must declare its workload profiles; "
                "the capacity alone is not a benchmark distribution"
            )
        _validate_capacity_tier(source, size, definition, symbols)
        _validate_usage(source, size, definition)
        missing = set(symbols) - set(definition) - _DERIVED_VARLEN_SYMBOLS
        if missing:
            raise OpSpecError(f"{source}.sizes.{size}: missing symbols {sorted(missing)}")
        _validate_tier_symbols(source, size, definition, symbols, derived=_DERIVED_VARLEN_SYMBOLS)
        return
    # varlen：``seq_lens`` 是该档的可变长序列规约（每段 token 数），由它派生 total_tokens/num_seqs/NSP1。
    # 固定 shape 档不写它；写了它，该档就是一个 varlen 实例（长度分布）。
    extra = set(definition) - set(symbols) - {"usage", "seq_lens"}
    if extra:
        raise OpSpecError(f"{source}.sizes.{size}: unknown fields {sorted(extra)}")
    # 派生符号只在真正的 varlen 档里豁免：固定档缺符号必须在这里就报错，不能延后成 KeyError。
    derived = _DERIVED_VARLEN_SYMBOLS if "seq_lens" in definition else frozenset()
    missing = set(symbols) - set(definition) - derived
    if missing:
        raise OpSpecError(f"{source}.sizes.{size}: missing symbols {sorted(missing)}")
    if "seq_lens" in definition:
        seq_lens = definition["seq_lens"]
        # 派生符号由 seq_lens 算出；再写一遍是自相矛盾的重复声明，不静默覆盖。
        repeated = sorted(_DERIVED_VARLEN_SYMBOLS & set(definition))
        if repeated:
            raise OpSpecError(
                f"{source}.sizes.{size}: a varlen tier derives {repeated} from seq_lens and must not declare them"
            )
        undeclared = sorted(name for name in _DERIVED_VARLEN_SYMBOLS if name not in symbols)
        if undeclared:
            raise OpSpecError(
                f"{source}.sizes.{size}: a varlen tier derives {undeclared} from seq_lens, so they must be "
                "declared in symbols (with their min) even though the tier does not write them"
            )
        if not isinstance(seq_lens, list) or not seq_lens:
            raise OpSpecError(f"{source}.sizes.{size}.seq_lens: must be a non-empty list of positive integers")
        for value in seq_lens:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise OpSpecError(f"{source}.sizes.{size}.seq_lens: every segment length must be a positive integer")
        # 静态声明只允许正长度：零长度段是运行时 ``cu_seqlens`` 的事（V1 的 zero_length case），
        # 不由声明分布表达。总数还必须落在 ``cu_seqlens`` 的 i32 域内。
        total_tokens = sum(int(value) for value in seq_lens)
        if total_tokens > _INT32_MAX:
            raise OpSpecError(
                f"{source}.sizes.{size}.seq_lens: total_tokens {total_tokens} exceeds the i32 range of the "
                f"cu_seqlens ABI ({_INT32_MAX}); a wider total needs a separate int64 contract"
            )
        derived_values = {"total_tokens": total_tokens, "num_seqs": len(seq_lens), "NSP1": len(seq_lens) + 1}
        for name, value in derived_values.items():
            minimum = int(symbols[name].get("min", 0))
            if value < minimum:
                raise OpSpecError(
                    f"{source}.sizes.{size}: the tier derives {name}={value} from seq_lens, below the declared "
                    f"min {minimum}"
                )
    _validate_usage(source, size, definition)
    _validate_tier_symbols(
        source,
        size,
        definition,
        symbols,
        derived=_DERIVED_VARLEN_SYMBOLS if "seq_lens" in definition else frozenset(),
    )


def _validate_tier_symbols(
    source: str,
    size: str,
    definition: Mapping[str, Any],
    symbols: Mapping[str, Any],
    *,
    derived: frozenset[str],
) -> None:
    """@brief 校验档位里直接写出的符号值（类型/下限）；派生符号不在档位里写。

    @param source 诊断来源。
    @param size 档位名。
    @param definition 档位定义。
    @param symbols 已声明符号。
    @param derived 由 ``seq_lens``/``capacity`` 派生的符号（跳过）。
    @raises OpSpecError 非法。
    """
    for name in symbols:
        if name not in definition or name in derived:
            continue
        value = definition[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise OpSpecError(f"{source}.sizes.{size}.{name}: must be an integer")
        minimum = symbols[name].get("min", 0)
        if value < minimum:
            raise OpSpecError(f"{source}.sizes.{size}.{name}: {value} is below the declared min {minimum}")


def _validate_usage(source: str, size: str, definition: Mapping[str, Any]) -> None:
    """@brief 校验一个档位的 ``usage`` 列表（固定档、L1 档与容量档共用）。

    @param source 诊断来源。
    @param size 档位名。
    @param definition 档位定义。
    @raises OpSpecError 非法。
    """
    usage = definition.get("usage")
    if not isinstance(usage, list) or not usage:
        raise OpSpecError(f"{source}.sizes.{size}: usage must be a non-empty list")
    unknown = set(usage) - VALID_USAGE
    if unknown:
        raise OpSpecError(f"{source}.sizes.{size}: unknown usage {sorted(unknown)}")
    if len(set(usage)) != len(usage):
        raise OpSpecError(f"{source}.sizes.{size}: usage must not repeat")


def _validate_accounting(
    source: str,
    accounting: Mapping[str, Any],
    symbols: Mapping[str, Any],
    tensors: Mapping[str, Any],
) -> None:
    """@brief 校验记账表达式文法：每项恰好一种量，符号/张量都已声明。

    @param source 诊断来源。
    @param accounting ``work_accounting`` 块。
    @param symbols 已声明符号。
    @param tensors 已声明张量。
    @raises OpSpecError 文法非法。
    """
    for field in ("flops", "min_bytes"):
        for index, term in enumerate(accounting[field]["terms"]):
            where = f"{source}.work_accounting.{field}.terms[{index}]"
            kinds = [key for key in ("tensor", "literal", "symbols") if key in term]
            if len(kinds) != 1:
                raise OpSpecError(f"{where}: a term needs exactly one of tensor/literal/symbols")
            if "tensor" in term and term["tensor"] not in tensors:
                raise OpSpecError(f"{where}: unknown tensor {term['tensor']!r}")
            for name in term.get("symbols", ()):
                _check_symbol(name, symbols, where)


def _validate_program(source: str, program: Mapping[str, Any], tensors: Mapping[str, Any]) -> None:
    """@brief 校验一份生成程序（闭集语句表）：名字先定义后使用，字段按 kind 取用。

    The program is semantic data, so the same rules that protect hand-written OpSpecs apply
    here: an operand must be defined before it is read, a scalar reduction may only feed the
    scalar operand slot, unknown/irrelevant fields are refused, and every constant must stay in
    the int32 domain the reference evaluates.

    @param source 诊断来源。
    @param program ``document['program']``。
    @param tensors 已声明张量（``xv``/``yv`` 由它们派生）。
    @raises OpSpecError 程序非法。
    """
    where = f"{source}.program"
    if program["kind"] not in PROGRAM_KINDS:
        raise OpSpecError(f"{where}.kind: unregistered program kind {program['kind']!r}; known: {sorted(PROGRAM_KINDS)}")
    block = int(program["block"])
    if block < 1:
        raise OpSpecError(f"{where}.block: must be positive")
    rules = program["rules"]
    if int(rules["version"]) != PROGRAM_RULE_VERSION:
        raise OpSpecError(
            f"{where}.rules.version: unregistered rule version {rules['version']!r}; registered: {PROGRAM_RULE_VERSION}"
        )
    shift_ranges = {"shl": tuple(int(v) for v in rules["shl_range"]), "shr": tuple(int(v) for v in rules["shr_range"])}
    for kind, (low, high) in shift_ranges.items():
        if not (0 <= low <= high <= 31):
            raise OpSpecError(f"{where}.rules.{kind}_range: a defined int32 shift domain is 0..31, got {[low, high]}")
    if not (0 <= int(rules["mod_mask"]) < (1 << 31)):
        raise OpSpecError(f"{where}.rules.mod_mask: must be a non-negative int32 mask")
    for field in ("hash_shift", "loop_shift"):
        if not (0 <= int(rules[field]) <= 31):
            raise OpSpecError(f"{where}.rules.{field}: must be within 0..31 (defined int32 shift)")
    if int(rules["index_mask"]) < 0:
        raise OpSpecError(f"{where}.rules.index_mask: must be non-negative")
    for field in ("wrap_kinds", "plain_kinds"):
        declared = list(rules[field])
        if len(declared) != len(set(declared)):
            raise OpSpecError(f"{where}.rules.{field}: must not repeat a kind")
    wrap_kinds = set(rules["wrap_kinds"])
    plain_kinds = set(rules["plain_kinds"])
    if wrap_kinds & plain_kinds:
        raise OpSpecError(f"{where}.rules: a kind cannot be both wrapping and plain: {sorted(wrap_kinds & plain_kinds)}")
    if wrap_kinds | plain_kinds != set(_PROGRAM_FIELDS):
        unknown = sorted((wrap_kinds | plain_kinds) - set(_PROGRAM_FIELDS))
        missing = sorted(set(_PROGRAM_FIELDS) - (wrap_kinds | plain_kinds))
        raise OpSpecError(f"{where}.rules: the wrap/plain partition must cover every kind; unknown={unknown} missing={missing}")
    inputs = sorted(name for name, tensor in tensors.items() if tensor["role"] == "input")
    if inputs != ["x", "y"]:
        raise OpSpecError(
            f"{where}: an int32 block program reads the two input blocks 'x'/'y', got inputs {inputs}"
        )
    blocks = set(_PROGRAM_BLOCK_NAMES)
    scalars = {"K"}
    for index, statement in enumerate(program["statements"]):
        at = f"{where}.statements[{index}]"
        kind = statement["kind"]
        required, forbidden = _PROGRAM_FIELDS[kind]
        missing = sorted(required - set(statement))
        if missing:
            raise OpSpecError(f"{at}: {kind} statement is missing {missing}")
        extra = sorted(set(statement) & forbidden)
        if extra:
            raise OpSpecError(f"{at}: {kind} statement must not carry {extra}")
        if "const" in statement and not (_INT32_MIN <= int(statement["const"]) <= _INT32_MAX):
            raise OpSpecError(f"{at}: constant {statement['const']} is outside the int32 domain")
        if "rounds" in statement and not (1 <= int(statement["rounds"]) <= 64):
            raise OpSpecError(f"{at}: rounds must be within 1..64, got {statement['rounds']!r}")
        # Kind-specific numeric domains: the reference/renderers are only defined here, so the
        # document itself must be refused otherwise (the generator happening to be well behaved
        # is not a substitute for validating the document).
        const = int(statement.get("const", 0))
        if kind in ("shl", "shr"):
            low, high = shift_ranges[kind]
            if not (low <= const <= high):
                raise OpSpecError(f"{at}: {kind} amount {const} is outside the declared domain {[low, high]}")
        if kind == "mod" and not (1 <= const <= max(int(rules["mod_mask"]), 1)):
            raise OpSpecError(f"{at}: mod divisor {const} must be in 1..{max(int(rules['mod_mask']), 1)}")
        if kind == "loop" and statement.get("rounds") is None:
            raise OpSpecError(f"{at}: a loop statement needs rounds")
        if kind in ("bin", "where"):
            if statement["op"] not in (_PROGRAM_BINARY_OPS if kind == "bin" else _PROGRAM_COMPARE_OPS):
                raise OpSpecError(f"{at}: op {statement['op']!r} is not valid for {kind}")
        elif kind == "sbin":
            if statement["op"] not in _PROGRAM_SCALAR_OPS:
                raise OpSpecError(f"{at}: op {statement['op']!r} is not valid for sbin")
        for field in ("a", "b", "c", "d"):
            if field not in statement:
                continue
            name = statement[field]
            if kind == "sbin" and field == "b":
                if name not in scalars:
                    raise OpSpecError(f"{at}.b: {name!r} is not a scalar value (K or a reduction result)")
                continue
            if name not in blocks:
                raise OpSpecError(f"{at}.{field}: {name!r} is not a defined block value")
        if kind == "bin" and ("b" in statement) == ("const" in statement):
            raise OpSpecError(f"{at}: a bin statement needs exactly one of b/const")
        dst = str(statement["dst"])
        if dst in blocks or dst in scalars:
            raise OpSpecError(f"{at}: {dst!r} is defined twice (or shadows a reserved block/scalar name)")
        if kind in ("sum", "max", "min"):
            scalars.add(dst)
        else:
            blocks.add(dst)
    for name in program["final"]:
        if name not in blocks:
            raise OpSpecError(f"{where}.final: {name!r} is not a defined block value")


def _opspec(document: Mapping[str, Any], source: str, where: str) -> OpSpec:
    """@brief 由一份已校验文档构造 ``OpSpec``（含 §3.1 的唯一摘要）。

    @param document OpSpec 文档。
    @param source 来源名。
    @param where 诊断前缀。
    @return ``OpSpec``。
    @raises OpSpecError 身份不可 canonical 化。
    """
    envelope = {
        "fingerprint_kind": "opspec",
        "fingerprint_version": 1,
        "semantic_rules_version": SEMANTIC_RULES_VERSION,
        "work_accounting_version": WORK_ACCOUNTING_VERSION,
        "document": document,
    }
    try:
        digest = fingerprint(envelope)
    except IdentityError as exc:
        raise OpSpecError(f"{where}: identity is not canonical: {exc}") from exc
    return OpSpec(document=document, source=source, digest=digest)


def opspec_from_document(document: Mapping[str, Any], *, source: str) -> OpSpec:
    """@brief 校验并包装一份*物化在内存里*的 OpSpec（生成族用，例如 ``sem_int``）。

    The generated document goes through exactly the same schema, semantic validation and
    canonical digest as a file on disk; the source string identifies the registered generator
    invocation (never a host path), so the identity stays reproducible.

    @param document OpSpec 文档。
    @param source 来源名（如 ``generated:sem_int:7``）。
    @return 通过结构 + 语义校验的 :class:`OpSpec`。
    @raises OpSpecError 结构或语义非法。
    """
    validate_opspec(document, source=source)
    if document.get("spec_version") == 2:
        from . import v2

        return v2._opspec_v2(document, source=source, where=source)
    return _opspec(document, source=source, where=source)


def load_opspec(path: str | Path) -> OpSpec:
    """@brief 读取并校验一份 OpSpec 文件。

    @param path ``opspec.json`` 路径。
    @return 通过结构 + 语义校验的 :class:`OpSpec`。
    @raises OpSpecError 文件不可读、JSON 非法或语义非法。
    """
    resolved = Path(path)
    try:
        document = load_strict_json(resolved)
    except IdentityError as exc:
        raise OpSpecError(f"{resolved}: {exc}") from exc
    validate_opspec(document, source=str(resolved))
    if document.get("spec_version") == 2:
        from . import v2

        return v2._opspec_v2(document, source=str(resolved), where=str(resolved))
    return _opspec(document, source=str(resolved), where=str(resolved))


def _resolve_weight(spec: Mapping[str, Any], dtype: str, attr: str = "<attr>") -> Any:
    """@brief 物化一个 attr 的值（显式 default；没有则拒绝，绝不补值）。

    @param spec attr 定义。
    @param dtype attr 的语义类型（``float``/``int``）。
    @param attr attr 名，用于诊断。
    @return 物化值。
    @raises OpSpecError 缺少实例值与显式 default，或类型不符。
    """
    if "default" not in spec:
        raise OpSpecError(f"attr {attr!r} has no value and no explicit default; the framework must not invent one")
    raw = spec["default"]
    if dtype == "float":
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise OpSpecError(f"attr {attr!r}: float default must be numeric, got {raw!r}")
        return float(raw)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise OpSpecError(f"attr {attr!r}: int default must be an integer, got {raw!r}")
    return int(raw)


def _eval_accounting_terms(
    terms: list[Mapping[str, Any]],
    *,
    symbols: Mapping[str, int],
    tensors: Mapping[str, TensorBinding],
    per_element: bool,
) -> float:
    """@brief 求值一组记账项。

    @param terms 记账项列表。
    @param symbols 已解析符号。
    @param tensors 已解析张量绑定。
    @param per_element 为真时张量项乘以该张量的字节数（``min_bytes``），否则只乘元素数。
    @return 浮点总量（R1 统一物化为 float）。
    """
    total = 0.0
    for term in terms:
        coefficient = float(term.get("coefficient", 1))
        if "tensor" in term:
            binding = tensors[term["tensor"]]
            base = float(binding.nbytes if per_element else binding.numel)
            total += coefficient * base
        elif "literal" in term:
            total += coefficient * float(term["literal"])
        else:
            product = 1.0
            for name in term["symbols"]:
                product *= float(symbols[name])
            total += coefficient * product
    return total


def instantiate(spec: OpSpec, dtype: str, size: str) -> SemanticInstance:
    """@brief 由 ``(opspec, dtype, size)`` 唯一产生语义实例。

    @param spec 已校验的 OpSpec。
    @param dtype 请求 dtype。
    @param size 请求 size 档位。
    @return 唯一的 :class:`SemanticInstance`。
    @raises OpSpecError dtype/size 不在声明域内，或 attrs/shape 非法。
    """
    if spec.spec_version == 2:
        from . import v2

        return v2.instantiate_v2(spec, dtype, size)
    if dtype not in spec.dtypes:
        raise OpSpecError(f"{spec.op_name}: dtype {dtype!r} is not in the declared coverage {list(spec.dtypes)}")
    if size not in spec.sizes:
        raise OpSpecError(f"{spec.op_name}: size {size!r} is not declared; declared sizes are {sorted(spec.sizes)}")
    definition = spec.sizes[size]
    symbols = {name: int(definition[name]) for name in spec.document["symbols"]}

    tensors: list[TensorBinding] = []
    for name, tensor in spec.document["tensors"].items():
        bound_dtype = dtype if tensor["dtype"] == "T" else str(tensor["dtype"])
        shape = tuple(_resolve_term(dim, symbols, f"{spec.op_name}.tensors.{name}.dims") for dim in tensor["dims"])
        strides = tuple(
            _resolve_term(stride, symbols, f"{spec.op_name}.tensors.{name}.strides") for stride in tensor["strides"]
        )
        if tensor["layout"] in ("row_major", "contiguous"):
            expected = _row_major_strides(shape)
            if strides != expected:
                raise OpSpecError(
                    f"{spec.op_name}.tensors.{name}: {tensor['layout']} declares strides {strides}, "
                    f"but the shape {shape} implies {expected}"
                )
        doc = tensor.get("doc", "")
        tensors.append(
            TensorBinding(
                name=name,
                role=str(tensor["role"]),
                dtype=bound_dtype,
                shape=shape,
                strides=strides,
                layout=str(tensor["layout"]),
                space=str(tensor["space"]),
                access=str(tensor["access"]),
                doc=str(doc),
            )
        )

    resolved_attrs: dict[str, Any] = {}
    for name, attr in spec.document["attrs"].items():
        value = _resolve_weight(attr, str(attr["dtype"]), name)
        resolved_attrs[name] = _resolve_domain(name, attr, value, f"{spec.op_name}.attrs.{name}")

    tolerances = spec.document["numerics"]["tolerances"]
    if dtype not in tolerances:
        raise OpSpecError(f"{spec.op_name}: numerics.tolerances has no entry for dtype {dtype!r}")

    cases = tuple(spec.document["cases"]["names"])
    declared_inputs = spec.document["cases"].get("inputs", {})
    case_inputs: dict[str, dict[str, Any]] = {}
    for case, values in declared_inputs.items():
        case_inputs[case] = dict(values)

    by_name = {binding.name: binding for binding in tensors}
    accounting = spec.document["work_accounting"]
    flops = _eval_accounting_terms(
        list(accounting["flops"]["terms"]), symbols=symbols, tensors=by_name, per_element=False
    )
    min_bytes = _eval_accounting_terms(
        list(accounting["min_bytes"]["terms"]), symbols=symbols, tensors=by_name, per_element=True
    )

    return SemanticInstance(
        opspec_digest=spec.digest,
        op_name=spec.op_name,
        category=spec.category,
        dtype=dtype,
        size=size,
        symbols=symbols,
        tensors=tuple(tensors),
        resolved_attrs=resolved_attrs,
        spec_params={name: str(symbols[name]) for name in spec.document["spec_params"]},
        tolerances=Tolerances(
            rtol=float(tolerances[dtype]["rtol"]),
            atol_rel=float(tolerances[dtype]["atol_rel"]),
            exact=bool(spec.document["numerics"]["exact"]),
        ),
        case_domain=CaseDomain(cases=cases, inputs=case_inputs),
        work_accounting=WorkAccounting(flops=float(flops), min_bytes=float(min_bytes)),
        acc_dtype=str(spec.document["numerics"]["acc_dtype"]),
        size_usage=tuple(spec.document["sizes"][size]["usage"]),
    )


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    """@brief 计算行主序连续张量的 stride。

    @param shape 形状。
    @return 行主序 stride 元组。
    """
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * int(shape[index + 1])
    return tuple(strides)


__all__ = [
    "OPSPEC_SCHEMA_PATH",
    "OPSPEC_SCHEMA_VERSION",
    "SEMANTIC_RULES_VERSION",
    "WORK_ACCOUNTING_VERSION",
    "CaseDomain",
    "OpSpec",
    "OpSpecError",
    "SemanticInstance",
    "TensorBinding",
    "Tolerances",
    "WorkAccounting",
    "instantiate",
    "load_opspec",
    "validate_opspec",
]
