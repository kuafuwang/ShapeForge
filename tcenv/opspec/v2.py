"""OpSpec v2: the cross-hardware task contract (A1) and its only instantiation.

v2 is a *sibling* of v1, not a fallback for it.  The two versions are dispatched by
``spec_version``: a v1 document is validated by :mod:`tcenv.opspec.model`, a v2 document
by this module, and any other version is refused.  Nothing here changes a v1 digest: v2
has its own canonical envelope (``fingerprint_version: 2``), its own rule version and its
own closed ``SemanticInstance`` projection.

What v2 adds to the semantics:

* every declared attr is materialised into ``resolved_attrs`` — there is no ``uses`` list
  and no free-text formula parsing (``formula`` is a summary only);
* tensors keep their role, shape (symbolic dims/stride products), dtype, layout, stride,
  memory space and access; ``read_write`` tensors are part of both contracts: their initial
  value is a reference input (even when role is ``output``) and their final value is a result;
* document-level discovery metadata (``phase``/``default_instance``/``tags``/
  ``variant_acceptance``) is part of ``opspec_digest`` but never of typed semantics,
  ABI, launch selection or the verdict.

What v2 deliberately does **not** contain: a global IR, a pattern resolver, a
``StagePartitioner``, a ``LoweringRuleRegistry``, grid/threads/tile/num_warps or any
track/lowering profile.  Those are A2/A3 concerns and, for launch facts, candidate-owned.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..dtypes import DTYPES, get_dtype
from ..identity import IdentityError, fingerprint, load_strict_json
from .model import (
    WORK_ACCOUNTING_VERSION,
    CaseDomain,
    OpSpec,
    OpSpecError,
    TensorBinding,
    Tolerances,
    WorkAccounting,
    _eval_accounting_terms,
    _resolve_domain,
    _resolve_term,
    _resolve_weight,
    _row_major_strides,
    _validate_accounting,
    _validate_case_domain,
    _validate_size,
)
from .validator import SchemaFailure, validate

#: Structural schema version of a v2 document.
OPSPEC_V2_SCHEMA_VERSION = 2
#: Semantic rules for defaults, dtype/shape acceptance, attr materialisation and the
#: discovery block.  Bumping this deliberately resets every v2 ``opspec_digest``.
SEMANTIC_RULES_VERSION_V2 = "opspec-v2-semantics-1"
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"
OPSPEC_V2_SCHEMA_PATH = SCHEMA_DIR / "opspec.v2.schema.json"

#: Frozen v2 attr identifier syntax: one or more uppercase alphanumeric segments joined
#: by single underscores.  This makes ``Block_M``/``NumWarps``/``EPS-value`` structural
#: errors instead of accidental semantic attrs.
_ATTR_NAME_PATTERN = r"^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$"
_ATTR_NAME_RE = re.compile(_ATTR_NAME_PATTERN)
#: Attr names are semantic parameters, never launch/optimisation parameters.  The check is
#: a denylist of words that can only describe a candidate's geometry, so a v2 document
#: cannot smuggle ``BLOCK_M``/``NUM_WARPS``/``THREADS`` through ``attrs``.  The comparison
#: removes underscores and lowercases, so ``BLOCKM``/``NUMWARPS`` are rejected too.
_GEOMETRY_ATTR_TOKENS = frozenset(
    {
        "BLOCK",
        "BLOCKS",
        "GRID",
        "LANE",
        "LANES",
        "SIMD",
        "THREAD",
        "THREADS",
        "THREADGROUP",
        "TILE",
        "TILES",
        "UNROLL",
        "VECTOR",
        "WARP",
        "WARPS",
    }
)


#: The L1 correctness-case floor of the varlen plan (§5.1).  A packed-varlen task must declare
#: this domain (it may add more cases): ``canonical`` is the published benchmark/validate workload
#: and has to come first, because the runtime measures ``case_domain.cases[0]``; the others reshape
#: the *same* total/count into permuted, skewed, prime-length, worst-handoff, chunk-boundary,
#: zero-length, single-sequence and fresh segmentations.  The harness validates this floor — it
#: does not invent a task's coverage — so a varlen task cannot silently ship canonical-only
#: correctness.
VARLEN_CASE_DOMAIN: tuple[str, ...] = (
    "canonical",
    "permuted",
    "skewed",
    "all_unaligned_primes",
    "max_handoff_stress",
    "chunk_boundary",
    "zero_length",
    "single_sequence",
    "fresh_distribution",
)


@dataclass(frozen=True)
class SemanticInstanceV2:
    """@brief ``(opspec v2, dtype, size)`` 的唯一语义实例（派生 DTO，不是权威）。"""

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
    discovery: Mapping[str, Any]
    formula: str = ""
    #: varlen 实例的分段信息（``[0, *cumsum(seq_lens)]``），固定 shape 实例为 ``None``。
    #: reference 按它分段计算，gen_inputs 按它生成打包输入，正确性按它做 ragged 比较。
    cu_seqlens: tuple[int, ...] | None = None

    @property
    def spec_version(self) -> int:
        """@brief 该实例来自 OpSpec v2。

        @return ``2``。
        """
        return OPSPEC_V2_SCHEMA_VERSION

    @property
    def reference_outputs(self) -> tuple[str, ...]:
        """@brief reference 差分合同覆盖的张量名（声明顺序）。

        A reference result is not "the write outputs"; a ``read_write`` tensor's final value is
        a result too, whether the tensor is declared as input or output.  This is the single
        definition of that set; the reference boundary and the evaluator both read it.

        @return 张量名元组。
        """
        return tuple(
            binding.name
            for binding in self.tensors
            if binding.space != "workspace" and (binding.role == "output" or binding.access == "read_write")
        )

    @property
    def reference_inputs(self) -> tuple[str, ...]:
        """@brief reference 执行需要初值的张量名（声明顺序）。

        This is the input-side counterpart of :attr:`reference_outputs`: every ``input``
        tensor plus every ``read_write`` tensor.  A ``read_write`` tensor is read before it is
        written even when its role is ``output`` (for example an in-place output buffer), so
        its initial value belongs to the reference input contract, not only its final value to
        the result contract.  A tensor may appear in both sets (that is exactly read_write).

        A ``space: workspace`` tensor is a scratch buffer: it is neither a reference input nor a
        reference result, so it is excluded from both projections (A2 derives it into the
        ``TensorMemoryContract`` and clears it only where the candidate declared it zero).

        @return 张量名元组。
        """
        return tuple(
            binding.name
            for binding in self.tensors
            if binding.space != "workspace" and (binding.role == "input" or binding.access == "read_write")
        )

    @property
    def write_outputs(self) -> tuple[str, ...]:
        """@brief 纯 write 输出（不含 read_write 输入/输出）。

        @return 张量名元组。
        """
        return tuple(
            binding.name
            for binding in self.tensors
            if binding.space != "workspace" and binding.role == "output" and binding.access == "write"
        )

    @property
    def read_write_tensors(self) -> tuple[str, ...]:
        """@brief 所有 ``access: read_write`` 张量（初值读、终值写）。

        @return 张量名元组。
        """
        return tuple(
            binding.name for binding in self.tensors if binding.space != "workspace" and binding.access == "read_write"
        )

    @property
    def canonical_seq_lens(self) -> tuple[int, ...] | None:
        """@brief 该档声明的 canonical 长度分布（固定 shape 实例为 ``None``）。

        This is the *published* profile from the OpSpec tier, derived from the canonical
        ``cu_seqlens``.  It is what a candidate may use to choose a schedule; the correctness
        cases reshape exactly this total/count, so it is never a promise that the boundaries are
        the ones the candidate will see.

        @return 每段 token 数元组。
        """
        if self.cu_seqlens is None:
            return None
        return tuple(int(end) - int(begin) for begin, end in zip(self.cu_seqlens, self.cu_seqlens[1:]))

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

    def evaluation_meta(self) -> dict[str, Any]:
        """@brief 由 ``cases.inputs`` 物化的只读评测域视图。

        @return ``domain_case``/``ieee_*`` 等已声明键组成的字典。
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
                    str(name): _resolve_max_term(value, self.symbols, f"cases.inputs.ieee._max.{name}")
                    for name, value in dict(ieee["_max"]).items()
                }
        return meta

    def fingerprint_document(self) -> dict[str, Any]:
        """@brief v2 的冻结 ``SemanticInstance`` 投影（闭集）。

        Unlike the v1 projection this binds role/access/layout/space as well: a read_write
        final-value contract or an output/input role change is a different evaluation meaning
        and therefore a different task identity.

        @return 只含 JSON 值的派生文档。
        """
        document: dict[str, Any] = {
            "spec_version": OPSPEC_V2_SCHEMA_VERSION,
            "opspec_digest": self.opspec_digest,
            "dtype": self.dtype,
            "size": self.size,
            "symbols": {name: int(value) for name, value in self.symbols.items()},
            "tensors": [
                {
                    "name": binding.name,
                    "role": binding.role,
                    "dtype": binding.dtype,
                    "shape": [int(dim) for dim in binding.shape],
                    "strides": [int(stride) for stride in binding.strides],
                    "layout": binding.layout,
                    "space": binding.space,
                    "access": binding.access,
                }
                for binding in self.tensors
            ],
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
        # varlen 才有的键：分段边界进身份（total_tokens/num_seqs 相同、切分不同的实例是不同问题）。
        # 固定 shape 实例不出现这个键，它们的冻结投影与加 varlen 之前逐位相同。
        if self.cu_seqlens is not None:
            document["cu_seqlens"] = list(self.cu_seqlens)
        if self.discovery:
            document["discovery"] = dict(self.discovery)
        if self.formula:
            document["formula"] = self.formula
        return document


def _resolve_max_term(value: Any, symbols: Mapping[str, int], where: str) -> int:
    """@brief 求值一个 ``ieee_max`` 幅度（数值或已登记的 ``terms`` 结构化项）。

    @param value 数值或结构化项。
    @param symbols 已解析符号值。
    @param where 诊断位置。
    @return 整数结果。
    @raises OpSpecError 项非法。
    """
    from ..terms import TermError, eval_term

    try:
        return int(eval_term(value, symbols, where=where))
    except TermError as exc:
        raise OpSpecError(str(exc)) from exc


def _validate_v2_tensor(source: str, name: str, tensor: Mapping[str, Any], symbols: Mapping[str, Any]) -> None:
    """@brief 校验一个 v2 张量声明。

    @param source 诊断来源。
    @param name 张量名。
    @param tensor 张量声明。
    @param symbols 已声明符号。
    @raises OpSpecError 非法。
    """
    where = f"{source}.tensors.{name}"
    dtype = tensor["dtype"]
    if dtype != "T" and dtype not in DTYPES:
        raise OpSpecError(f"{where}.dtype: {dtype!r} is not a registered dtype")
    if len(tensor["strides"]) != len(tensor["dims"]):
        raise OpSpecError(f"{where}: must declare one stride per dim")
    role, access = tensor["role"], tensor["access"]
    if role == "input" and access not in ("read", "read_write"):
        raise OpSpecError(f"{where}: input tensor must be read/read_write, got {access!r}")
    if role == "output" and access not in ("write", "read_write"):
        raise OpSpecError(f"{where}: output tensor must be write/read_write, got {access!r}")
    for field in ("dims", "strides"):
        for term in tensor[field]:
            _validate_symbol_term(str(term), symbols, f"{where}.{field}")


def _validate_symbol_term(term: str, symbols: Mapping[str, Any], where: str) -> None:
    """@brief 校验 dim/stride 词只由整数和已声明符号组成。

    @param term 词文本。
    @param symbols 已声明符号。
    @param where 诊断位置。
    @raises OpSpecError 语法或符号非法。
    """
    if term.isdigit():
        return
    for part in term.split("*"):
        if part.isdigit():
            continue
        if part not in symbols:
            raise OpSpecError(f"{where}: symbol {part!r} is not declared in symbols")


def _validate_attr_name(name: str, where: str) -> None:
    """@brief 校验 attr 是纯语义参数，不是候选几何参数。

    @param name attr 名。
    @param where 诊断位置。
    @raises OpSpecError 名字不是合法语法量。
    """
    if _ATTR_NAME_RE.fullmatch(name) is None:
        raise OpSpecError(
            f"{where}: attr {name!r} must match {_ATTR_NAME_PATTERN} (uppercase semantic identifier, "
            "single underscores); launch/optimisation spellings carry no semantic meaning"
        )
    compact = name.replace("_", "").lower()
    forbidden = sorted(token for token in _GEOMETRY_ATTR_TOKENS if token.lower() in compact)
    if forbidden:
        raise OpSpecError(
            f"{where}: attr {name!r} names launch/optimisation geometry {forbidden}; attrs are "
            "semantic parameters and launch facts belong to the candidate's ExecutionGeometry"
        )


def _validate_discovery(source: str, discovery: Mapping[str, Any], document: Mapping[str, Any]) -> None:
    """@brief 校验文档级发现元数据（不参与 typed 语义或 launch）。

    @param source 诊断来源。
    @param discovery ``discovery`` 块。
    @param document 整份 OpSpec。
    @raises OpSpecError 引用未声明 dtype/size 或标签重复。
    """
    where = f"{source}.discovery"
    phase = discovery["phase"]
    if isinstance(phase, bool) or not isinstance(phase, int) or phase < 1:
        raise OpSpecError(f"{where}.phase: must be a positive integer")
    tags = list(discovery.get("tags", ()))
    if len(tags) != len(set(tags)):
        raise OpSpecError(f"{where}.tags: must not repeat a tag")
    default = discovery.get("default_instance")
    if default is None:
        return
    dtype, size = str(default["dtype"]), str(default["size"])
    if dtype not in document["type_vars"]["T"]:
        raise OpSpecError(f"{where}.default_instance.dtype: {dtype!r} is not in type_vars.T")
    if size not in document["sizes"]:
        raise OpSpecError(f"{where}.default_instance.size: {size!r} is not a declared size")


def validate_opspec_v2(document: Mapping[str, Any], *, source: str = "<opspec>") -> None:
    """@brief 校验一份 OpSpec v2（结构 schema + 语义规则）。

    @param document 已由严格 JSON loader 解析的文档。
    @param source 诊断来源。
    @raises OpSpecError 结构或语义非法。
    """
    try:
        schema = load_strict_json(OPSPEC_V2_SCHEMA_PATH)
    except IdentityError as exc:  # pragma: no cover - the registered schema is shipped
        raise OpSpecError(f"registered OpSpec v2 schema is unreadable: {exc}") from exc
    try:
        validate(document, schema)
    except SchemaFailure as exc:
        raise OpSpecError(f"{source}: {exc}") from exc
    if not isinstance(document, Mapping):  # pragma: no cover - schema already enforces it
        raise OpSpecError(f"{source}: an OpSpec must be a JSON object")
    if int(document["spec_version"]) != OPSPEC_V2_SCHEMA_VERSION:  # pragma: no cover - schema const
        raise OpSpecError(f"{source}: spec_version must be {OPSPEC_V2_SCHEMA_VERSION}")

    symbols = document["symbols"]
    tensors = document["tensors"]
    attrs = document["attrs"]
    for dtype in document["type_vars"]["T"]:
        if dtype not in DTYPES:
            raise OpSpecError(f"{source}: type var T lists unknown dtype {dtype!r}")

    for name, tensor in tensors.items():
        _validate_v2_tensor(source, name, tensor, symbols)

    for name, spec in attrs.items():
        _validate_attr_name(str(name), f"{source}.attrs")
        value = _resolve_weight(spec, str(spec["dtype"]), str(name))
        _resolve_domain(str(name), spec, value, f"{source}.attrs.{name}")

    declared_params = list(document["spec_params"])
    if len(set(declared_params)) != len(declared_params):
        raise OpSpecError(f"{source}: spec_params must not repeat a symbol")
    for name in declared_params:
        if name not in symbols:
            raise OpSpecError(f"{source}: spec_params names undeclared symbol {name!r}")

    cases = list(document["cases"]["names"])
    if len(set(cases)) != len(cases):
        raise OpSpecError(f"{source}: cases.names must not repeat a case")
    for case, values in document["cases"].get("inputs", {}).items():
        if case not in cases:
            raise OpSpecError(f"{source}: cases.inputs names unknown case {case!r}")
        for key, value in values.items():
            if key.startswith("_"):
                _validate_case_domain(source, case, key, value, cases, tensors, document)
                continue
            if key not in attrs:
                raise OpSpecError(f"{source}: cases.inputs.{case} names unknown attr {key!r}")

    for size, definition in document["sizes"].items():
        _validate_size(source, size, definition, symbols, varlen=True)
    _validate_varlen_tiers(source, document)

    _validate_accounting(source, document["work_accounting"], symbols, tensors)

    if "discovery" in document:
        _validate_discovery(source, document["discovery"], document)


def _validate_varlen_tiers(source: str, document: Mapping[str, Any]) -> None:
    """@brief 校验一题里 varlen 档与固定档不混用，且 varlen 题有合法的 ``cu_seqlens`` 输入。

    A task's tiers either all declare ``seq_lens`` or none does.  A mixed task would need a
    packed-batch ABI (``cu_seqlens``, integer boundaries, ``num_seqs + 1`` offsets) for some
    instances and a plain ABI for others, so the reference and the harness could not tell which
    layout an instance has without guessing.  Mixed tiers are therefore refused instead of being
    defined differently per instance.

    A varlen task must carry the boundaries as a real input tensor: it is an ``i32`` input of
    shape ``[NSP1]`` (``num_seqs + 1`` offsets).  The ABI is int32 in this phase, so the derived
    ``total_tokens`` is range-checked against ``INT32_MAX`` by :func:`_validate_size`.

    @param source 诊断来源。
    @param document v2 文档。
    @raises OpSpecError 档位混用或 ``cu_seqlens`` 不符合合同。
    """
    sizes = document["sizes"]
    varlen_sizes = sorted(size for size, definition in sizes.items() if "seq_lens" in definition)
    if not varlen_sizes:
        return
    fixed_sizes = sorted(size for size in sizes if size not in varlen_sizes)
    if fixed_sizes:
        raise OpSpecError(
            f"{source}.sizes: {fixed_sizes} declare fixed-shape symbols while {varlen_sizes} declare seq_lens; "
            "one task is either entirely varlen (every tier declares seq_lens) or entirely fixed, because the "
            "packed-batch cu_seqlens ABI must exist for every instance of the task"
        )
    symbols = document["symbols"]
    for name in ("total_tokens", "num_seqs", "NSP1"):
        if name not in symbols:
            raise OpSpecError(
                f"{source}.symbols: a varlen task derives {name!r} from seq_lens and must declare it "
                "(the packed tensor dims and the cu_seqlens length are expressed with it)"
            )
    cases = [str(name) for name in document["cases"]["names"]]
    if cases[0] != "canonical":
        raise OpSpecError(
            f"{source}.cases.names: a varlen task must declare 'canonical' first — it is the published "
            f"benchmark profile and the runtime measures the first case; got {cases[0]!r}"
        )
    missing = [name for name in VARLEN_CASE_DOMAIN if name not in cases]
    if missing:
        raise OpSpecError(
            f"{source}.cases.names: a varlen task must cover the L1 correctness cases {list(VARLEN_CASE_DOMAIN)}; "
            f"missing {missing}. Those cases reshape the same total/count, so a task cannot claim arbitrary "
            "segmentation support while only ever running its canonical profile (extra cases are allowed)"
        )
    tensor = document["tensors"].get("cu_seqlens")
    if tensor is None:
        raise OpSpecError(
            f"{source}.tensors: a varlen (seq_lens) task must declare the packed-batch boundaries as an "
            "i32 input tensor 'cu_seqlens' of shape [NSP1]"
        )
    problems: list[str] = []
    if tensor["role"] != "input":
        problems.append("role must be 'input' (the boundaries are runtime data)")
    if tensor["access"] != "read":
        problems.append(f"access must be 'read', got {tensor['access']!r}")
    if tensor["space"] != "global":
        problems.append(f"space must be 'global', got {tensor['space']!r}")
    if tensor["dtype"] != "i32":
        problems.append(f"dtype must be 'i32' (the L1 boundary ABI is int32), got {tensor['dtype']!r}")
    if [str(dim) for dim in tensor["dims"]] != ["NSP1"]:
        problems.append(f"dims must be ['NSP1'] (num_seqs + 1 offsets), got {list(tensor['dims'])}")
    if [str(stride) for stride in tensor["strides"]] != ["1"]:
        problems.append(f"strides must be ['1'], got {list(tensor['strides'])}")
    if problems:
        raise OpSpecError(f"{source}.tensors.cu_seqlens: illegal varlen ABI: {'; '.join(problems)}")


def digest_envelope(document: Mapping[str, Any]) -> dict[str, Any]:
    """@brief v2 ``opspec_digest`` 的唯一 canonical envelope。

    @param document v2 文档。
    @return envelope 文档。
    """
    return {
        "fingerprint_kind": "opspec",
        "fingerprint_version": 2,
        "semantic_rules_version": SEMANTIC_RULES_VERSION_V2,
        "work_accounting_version": WORK_ACCOUNTING_VERSION,
        "document": document,
    }


def _opspec_v2(document: Mapping[str, Any], source: str, where: str) -> OpSpec:
    """@brief 由已校验的 v2 文档构造 ``OpSpec``（唯一摘要）。

    @param document v2 文档。
    @param source 来源名。
    @param where 诊断前缀。
    @return ``OpSpec``（``spec_version`` 由文档给出）。
    @raises OpSpecError 身份不可 canonical 化。
    """
    envelope = digest_envelope(document)
    try:
        digest = fingerprint(envelope)
    except IdentityError as exc:
        raise OpSpecError(f"{where}: identity is not canonical: {exc}") from exc
    return OpSpec(document=document, source=source, digest=digest)


def opspec_v2_from_document(document: Mapping[str, Any], *, source: str) -> OpSpec:
    """@brief 校验并包装一份内存中的 v2 OpSpec（生成族/测试用）。

    @param document v2 文档。
    @param source 来源名。
    @return 通过校验的 :class:`OpSpec`。
    @raises OpSpecError 结构或语义非法。
    """
    validate_opspec_v2(document, source=source)
    return _opspec_v2(document, source=source, where=source)


def load_opspec_v2(path: str | Path) -> OpSpec:
    """@brief 读取并校验一份 v2 OpSpec 文件。

    @param path ``opspec.json`` 路径。
    @return 通过校验的 :class:`OpSpec`。
    @raises OpSpecError 文件不可读、JSON 非法或语义非法。
    """
    resolved = Path(path)
    try:
        document = load_strict_json(resolved)
    except IdentityError as exc:
        raise OpSpecError(f"{resolved}: {exc}") from exc
    validate_opspec_v2(document, source=str(resolved))
    return _opspec_v2(document, source=str(resolved), where=str(resolved))


def instantiate_v2(spec: OpSpec, dtype: str, size: str) -> SemanticInstanceV2:
    """@brief 由 ``(v2 OpSpec, dtype, size)`` 唯一产生语义实例。

    @param spec 已校验的 v2 OpSpec。
    @param dtype 请求 dtype。
    @param size 请求 size 档位。
    @return 唯一的 :class:`SemanticInstanceV2`。
    @raises OpSpecError dtype/size 不在声明域内，或 attrs/shape 非法。
    """
    if spec.spec_version != OPSPEC_V2_SCHEMA_VERSION:  # pragma: no cover - callers dispatch
        raise OpSpecError(f"{spec.op_name}: instantiate_v2 needs a spec_version 2 document")
    if dtype not in spec.dtypes:
        raise OpSpecError(f"{spec.op_name}: dtype {dtype!r} is not in the declared coverage {list(spec.dtypes)}")
    if size not in spec.sizes:
        raise OpSpecError(f"{spec.op_name}: size {size!r} is not declared; declared sizes are {sorted(spec.sizes)}")
    definition = spec.sizes[size]
    # 符号值来自该档：固定档直接写，varlen 档的 total_tokens/num_seqs/NSP1 由 seq_lens 派生（见下）。
    symbols = {name: int(definition[name]) for name in spec.document["symbols"] if name in definition}
    if "seq_lens" in definition and "cu_seqlens" not in spec.document["tensors"]:
        # ``validate_opspec_v2`` already refuses this; a hand-built ``OpSpec`` must not slip past it.
        raise OpSpecError(
            f"{spec.op_name}: varlen size {size!r} needs an i32 'cu_seqlens' input tensor in the document"
        )
    # varlen：该档声明了 ``seq_lens``（可变长序列规约）时，由它派生运行时维度并加入 symbols。
    # total_tokens = 所有段 token 总和；num_seqs = 段数；NSP1 = num_seqs + 1（cu_seqlens 的长度）。
    # 它们是该档的确定值（不同长度分布是不同档），张量 dims/strides 与几何照常按它们派生。
    cu_seqlens: tuple[int, ...] | None = None
    if "seq_lens" in definition:
        seq_lens = [int(value) for value in definition["seq_lens"]]
        symbols["total_tokens"] = sum(seq_lens)
        symbols["num_seqs"] = len(seq_lens)
        symbols["NSP1"] = len(seq_lens) + 1
        offsets = [0]
        for length in seq_lens:
            offsets.append(offsets[-1] + length)
        cu_seqlens = tuple(offsets)

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
                doc=str(tensor.get("doc", "")),
            )
        )

    resolved_attrs: dict[str, Any] = {}
    for name, attr in spec.document["attrs"].items():
        # No ``uses`` list exists in v2: *every* declared attr is materialised, and the only
        # source of a value is the explicit default (the framework never invents one).
        value = _resolve_weight(attr, str(attr["dtype"]), name)
        resolved_attrs[name] = _resolve_domain(name, attr, value, f"{spec.op_name}.attrs.{name}")

    tolerances = spec.document["numerics"]["tolerances"]
    if dtype not in tolerances:
        raise OpSpecError(f"{spec.op_name}: numerics.tolerances has no entry for dtype {dtype!r}")

    cases = tuple(spec.document["cases"]["names"])
    case_inputs = {case: dict(values) for case, values in spec.document["cases"].get("inputs", {}).items()}

    by_name = {binding.name: binding for binding in tensors}
    accounting = spec.document["work_accounting"]
    flops = _eval_accounting_terms(list(accounting["flops"]["terms"]), symbols=symbols, tensors=by_name, per_element=False)
    min_bytes = _eval_accounting_terms(
        list(accounting["min_bytes"]["terms"]), symbols=symbols, tensors=by_name, per_element=True
    )

    return SemanticInstanceV2(
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
        discovery=dict(spec.document.get("discovery", {})),
        formula=str(spec.document.get("formula", "")),
        cu_seqlens=cu_seqlens,
    )


#: Type alias documenting that v2 instantiation returns the v2 projection, not v1.
V2Instance = SemanticInstanceV2

__all__ = [
    "OPSPEC_V2_SCHEMA_PATH",
    "OPSPEC_V2_SCHEMA_VERSION",
    "SEMANTIC_RULES_VERSION_V2",
    "SemanticInstanceV2",
    "VARLEN_CASE_DOMAIN",
    "digest_envelope",
    "instantiate_v2",
    "load_opspec_v2",
    "opspec_v2_from_document",
    "validate_opspec_v2",
]
