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
        return {
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


#: varlen 派生符号：由 ``seq_lens`` 算出，不写进档位。
_DERIVED_VARLEN_SYMBOLS = frozenset({"total_tokens", "num_seqs", "NSP1"})


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
    if "seq_lens" in definition and not varlen:
        raise OpSpecError(
            f"{source}.sizes.{size}: declares 'seq_lens', but a varlen (packed-sequence) tier is an OpSpec v2 "
            "feature; a v1 document declares fixed-shape tiers only"
        )
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
    usage = definition.get("usage")
    if not isinstance(usage, list) or not usage:
        raise OpSpecError(f"{source}.sizes.{size}: usage must be a non-empty list")
    unknown = set(usage) - VALID_USAGE
    if unknown:
        raise OpSpecError(f"{source}.sizes.{size}: unknown usage {sorted(unknown)}")
    if len(set(usage)) != len(usage):
        raise OpSpecError(f"{source}.sizes.{size}: usage must not repeat")
    for name in symbols:
        if name not in definition:
            continue  # varlen 派生符号由 seq_lens 算出，不在该档直接写（固定档已在上面报错）
        value = definition[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise OpSpecError(f"{source}.sizes.{size}.{name}: must be an integer")
        minimum = symbols[name].get("min", 0)
        if value < minimum:
            raise OpSpecError(f"{source}.sizes.{size}.{name}: {value} is below the declared min {minimum}")


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
