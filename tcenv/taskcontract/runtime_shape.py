"""Runtime resolved / computed shape of one L2 (capacity) call, and its padding contract.

One compiled ABI, many runtime batches
--------------------------------------
An L2 tier fixes a **capacity** shape (``max_total_tokens``/``max_seqs``); the batch it actually runs
is a *resolved* shape that changes between calls.  Three shapes therefore have to stay apart
(``docs/架构/varlen支持计划.md`` §2.4):

``capacity``
    the OpSpec tier's declared maximum: what every buffer, ABI span, workspace and Host Program
    storage is sized by.  It is the only shape a build sees, so it is what dispatch keys on.
``resolved``
    this call's real ``total_tokens``/``num_seqs``: a runtime fact, carried by :class:`ResolvedShape`
    into the Host Program (``SfHostResolved``) and the evaluation/launch evidence.  It must never
    enter a finite-domain guard or a build digest.
``logical``
    the region correctness compares: ``output[:resolved_total]`` and
    ``final_state[:resolved_num_seqs]``.  Everything outside it is *padding* and must still hold the
    declared poison after the candidate ran.

L1 is the degenerate case (``resolved == capacity``); it keeps its old validation funnel and its
old identity, so nothing here changes an L1 result.  A fixed-shape instance has no resolved shape at
all and never reaches this module.

Who owns what
-------------
The **task** owns the case→segmentation plan (which legal split of a profile's total/count a case
generates) and fills the padding.  The **harness** owns the expected resolved shape (reconstructed
from the declared workload profiles, not from the generator's own claim), the legality of the actual
``cu_seqlens`` (int32, starts at 0, non-decreasing, prefix length ``num_seqs + 1``, last prefix entry
``== resolved total``, tail exactly the declared sentinel), and the poison check on both sides of the
call.  A task cannot certify its own boundaries and cannot shrink its batch to look like a different
case.

The case→profile assignment is deterministic and public: profiles are ordered by their declared id,
and case ``i`` of ``cases.names`` runs profile ``i % len(profiles)``.  That makes a sweep over the
declared cases a *continuous-batching* sequence of different totals/counts on one build instead of a
replay of one static shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..contract_errors import VarlenContractError
from ..opspec.model import (
    BOUNDARY_TENSOR,
    OpSpecError,
    capacity_axis_role_problem,
    resolve_axis_kind,
)
from ..opspec.v2 import SemanticInstanceV2
from .varlen import INT32_MAX, _dtype_name, offsets

#: Resolved-scalar names the Host Program ABI exposes for an L2 instance (``SfHostResolved``).
RESOLVED_SCALAR_NAMES: tuple[str, ...] = ("total_tokens", "num_seqs")


@dataclass(frozen=True)
class ResolvedShape:
    """@brief 一次 L2 调用的 resolved 形状（logical 比较与 padding 规则也从它派生）。"""

    kind: str
    total_tokens: int
    num_seqs: int
    capacity_total_tokens: int | None
    capacity_num_seqs: int | None
    cu_seqlens_sentinel: int | None
    padding_poison_byte: int | None
    workload_profile_id: str | None
    #: 本 case 所属 workload profile 的 canonical ``seq_lens``（L1 为该档声明分布）。任务的 case
    #: planner 用它作为“公开分布”，与 L1 完全同构；它不是 promise，实际分段仍可不同。
    canonical_seq_lens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        """@brief 校验 resolved 形状自身的合法性（与任何输入字节无关）。

        @raises VarlenContractError 空 batch、负数、超容量。
        """
        if self.kind not in ("l1", "l2"):
            raise VarlenContractError(f"resolved shape kind must be 'l1' or 'l2', got {self.kind!r}")
        if self.total_tokens < 1:
            raise VarlenContractError(
                f"resolved total_tokens must be >= 1; an empty batch is not a legal varlen call, got {self.total_tokens}"
            )
        if self.num_seqs < 1:
            raise VarlenContractError(f"resolved num_seqs must be >= 1, got {self.num_seqs}")
        if self.total_tokens > INT32_MAX:
            raise VarlenContractError(
                f"resolved total_tokens {self.total_tokens} exceeds the int32 boundary domain {INT32_MAX}"
            )
        if self.kind == "l2":
            if self.capacity_total_tokens is None or self.capacity_num_seqs is None:
                raise VarlenContractError("an L2 resolved shape needs the capacity it was resolved against")
            if self.cu_seqlens_sentinel is None or self.cu_seqlens_sentinel >= 0:
                raise VarlenContractError("an L2 resolved shape needs the declared negative cu_seqlens sentinel")
            if self.padding_poison_byte is None or not (1 <= int(self.padding_poison_byte) <= 255):
                raise VarlenContractError("an L2 resolved shape needs the declared padding poison byte")
            if self.total_tokens > self.capacity_total_tokens:
                raise VarlenContractError(
                    f"resolved total_tokens {self.total_tokens} exceeds the declared capacity "
                    f"{self.capacity_total_tokens}"
                )
            if self.num_seqs > self.capacity_num_seqs:
                raise VarlenContractError(
                    f"resolved num_seqs {self.num_seqs} exceeds the declared capacity {self.capacity_num_seqs}"
                )

    @property
    def is_full_capacity(self) -> bool:
        """@brief resolved 形状是否正好等于容量（L1 的退化点）。

        @return 布尔。
        """
        return self.total_tokens == self.capacity_total_tokens and self.num_seqs == self.capacity_num_seqs

    def resolved_scalars(self) -> dict[str, int]:
        """@brief 进入 ``SfHostResolved`` / HostRun / 评测证据的 runtime scalar。

        @return ``{"total_tokens": .., "num_seqs": ..}``。
        """
        return {"total_tokens": int(self.total_tokens), "num_seqs": int(self.num_seqs)}

    def document(self) -> dict[str, Any]:
        """@brief 该 resolved 形状的 JSON 文档（进 evaluation/launch 证据，不进 build 身份）。

        @return 字段字典。
        """
        document: dict[str, Any] = {
            "kind": self.kind,
            "total_tokens": int(self.total_tokens),
            "num_seqs": int(self.num_seqs),
            "workload_profile_id": self.workload_profile_id,
            "cu_seqlens_count": int(self.num_seqs) + 1,
            "canonical_seq_lens": [int(value) for value in self.canonical_seq_lens],
        }
        if self.kind == "l2":
            document["capacity"] = {
                "total_tokens": int(self.capacity_total_tokens or 0),
                "num_seqs": int(self.capacity_num_seqs or 0),
            }
            document["cu_seqlens_sentinel"] = int(self.cu_seqlens_sentinel or 0)
            document["padding_poison_byte"] = int(self.padding_poison_byte or 0)
            document["full_capacity"] = bool(self.is_full_capacity)
        return document


def workload_profile_for_case(instance: SemanticInstanceV2, case: str) -> Any:
    """@brief 该 case 在本 L2 实例上运行的 workload profile。

    Profiles are sorted by their declared (validated-unique) id and assigned to the declared case
    order cyclically, so ``gen_inputs``, the equivalence check and the harness all derive the same
    expectation without the generator claiming one.  Sorting by id (not by author order) means the
    assignment is reproducible from the canonical document alone.

    @param instance 语义实例。
    @param case 用例名。
    @return ``WorkloadProfile``；非 L2 实例为 ``None``。
    @raises VarlenContractError case 未声明。
    """
    if not instance.is_capacity:
        return None
    cases = tuple(instance.case_domain.cases)
    if case not in cases:
        raise VarlenContractError(
            f"case {case!r} is not declared; declared cases are {list(cases)}"
        )
    profiles = tuple(sorted(instance.workload_profiles, key=lambda item: item.id))
    if not profiles:  # pragma: no cover - instantiate_v2 refuses an L2 tier without profiles
        raise VarlenContractError(f"{instance.op_name}.{instance.size}: an L2 instance declares no workload profile")
    return profiles[cases.index(case) % len(profiles)]


def resolved_shape_for_case(instance: SemanticInstanceV2, case: str) -> ResolvedShape | None:
    """@brief 由已声明档位得到该 case 的**期望** resolved 形状。

    This is the harness's independent reconstruction: the expectation comes from the OpSpec tier and
    the case order, never from the task's ``gen_inputs``.  ``gen_inputs`` receives the same value (so a
    task can plan internally), but a produced input that disagrees with it is refused instead of
    silently accepted.

    @param instance 语义实例。
    @param case 用例名。
    @return ``ResolvedShape``；固定 shape 实例为 ``None``。
    @raises VarlenContractError case 未声明或 L2 档位自相矛盾。
    """
    kind = instance.variable_length_kind
    if kind == "fixed":
        return None
    if kind == "l1":
        return ResolvedShape(
            kind="l1",
            total_tokens=int(instance.symbols["total_tokens"]),
            num_seqs=int(instance.symbols["num_seqs"]),
            capacity_total_tokens=int(instance.symbols["total_tokens"]),
            capacity_num_seqs=int(instance.symbols["num_seqs"]),
            cu_seqlens_sentinel=None,
            padding_poison_byte=None,
            workload_profile_id=None,
            canonical_seq_lens=tuple(instance.canonical_seq_lens or ()),
        )
    capacity = instance.capacity
    profile = workload_profile_for_case(instance, case)
    return ResolvedShape(
        kind="l2",
        total_tokens=int(profile.total_tokens),
        num_seqs=int(profile.num_seqs),
        capacity_total_tokens=int(capacity.total_tokens),
        capacity_num_seqs=int(capacity.num_seqs),
        cu_seqlens_sentinel=int(capacity.cu_seqlens_sentinel),
        padding_poison_byte=int(capacity.padding_poison_byte),
        workload_profile_id=str(profile.id),
        canonical_seq_lens=tuple(int(value) for value in profile.seq_lens),
    )


def resolved_shape_from_inputs(instance: SemanticInstanceV2, inputs: Mapping[str, Any]) -> ResolvedShape:
    """@brief 从**实际输入字节**读出一次调用的 resolved 形状（不是从声明猜）。

    The L2 boundary tensor is capacity-sized: ``cu_seqlens[0:num_seqs+1]`` is this call's legal
    offset prefix and ``cu_seqlens[num_seqs+1:]`` is the declared negative sentinel.  ``num_seqs`` is
    therefore the index of the first sentinel entry minus one, and the resolved total is the last
    prefix entry.  This is the only place that derivation lives; a candidate never resolves it, and a
    runtime scalar it publishes is checked against it.

    @param instance 语义实例。
    @param inputs 输入映射（含 capacity 尺寸的 ``cu_seqlens``）。
    @return ``ResolvedShape``（L2 时按字节解析；L1 时就是档位值）。
    @raises VarlenContractError 边界不可读或不在任何合法形状内。
    """
    if instance.variable_length_kind == "l1":
        return resolved_shape_for_case(instance, instance.case_domain.cases[0])
    if not instance.is_capacity:  # pragma: no cover - callers gate on variable_length_kind
        raise VarlenContractError(f"{instance.op_name}.{instance.size}: a fixed-shape instance has no resolved shape")
    capacity = instance.capacity
    cu = inputs.get("cu_seqlens")
    if cu is None:
        raise VarlenContractError("the capacity ABI declares an i32 'cu_seqlens' input, but it is missing")
    values = offsets(cu)
    if values is None:
        raise VarlenContractError(
            f"cu_seqlens is not readable as integers (dtype={_dtype_name(cu)!r}); the boundary tensor must be a "
            "dense i32 tensor with readable storage"
        )
    tail = [index for index, value in enumerate(values) if value == capacity.cu_seqlens_sentinel]
    if tail:
        first = tail[0]
        if any(value != capacity.cu_seqlens_sentinel for value in values[first:]):
            raise VarlenContractError(
                f"cu_seqlens capacity tail must be entirely the declared sentinel "
                f"{capacity.cu_seqlens_sentinel}; entry {first} starts a prefix that is not contiguous poison"
            )
        if first < 2:
            raise VarlenContractError(
                "cu_seqlens must carry at least one sequence (offset 0 and its end) before the capacity tail"
            )
    else:
        # Full capacity: there is no unused tail, so every declared entry is a real offset.  This is the
        # L1-degenerate point of an L2 tier and must stay legal instead of being read as "no batch".
        first = len(values)
        if first < 2:
            raise VarlenContractError("cu_seqlens must carry at least two offsets (offset 0 and one end)")
    return ResolvedShape(
        kind="l2",
        total_tokens=int(values[first - 1]),
        num_seqs=int(first - 1),
        capacity_total_tokens=int(capacity.total_tokens),
        capacity_num_seqs=int(capacity.num_seqs),
        cu_seqlens_sentinel=int(capacity.cu_seqlens_sentinel),
        padding_poison_byte=int(capacity.padding_poison_byte),
        workload_profile_id=None,
        # Parsed from the bytes, this is the *actual* segmentation; the declared profile it belongs to is
        # not known here (``resolved_shape_for_case`` carries that), so it is left empty instead of
        # relabelling an actual split as a declared one.
        canonical_seq_lens=(),
    )


def cu_seqlens_problems(
    instance: SemanticInstanceV2,
    resolved: ResolvedShape,
    cu: Any,
    *,
    expected_shape: tuple[int, ...],
) -> tuple[str, ...]:
    """@brief 校验一次 L2 调用的 ``cu_seqlens``：前缀合法 + 尾部正好是声明的 sentinel。

    The prefix check is the L1 rule (``i32``, starts at 0, non-decreasing, at least one token, last
    entry ``== resolved total``) applied to ``num_seqs + 1`` entries.  The tail rule is what makes the
    capacity ABI auditable: every unused entry must be the declared negative sentinel, so a candidate
    (or a prefetching device stage) that walks past ``num_seqs + 1`` reads something that cannot be a
    legal offset instead of a convenient ``0``.  A wrong last prefix entry — the case where the
    generator quietly changed the total — is refused here, not absorbed by a "prefix compare".

    @param instance 语义实例。
    @param resolved 本次调用的期望 resolved 形状。
    @param cu ``cu_seqlens`` 张量。
    @param expected_shape capacity 尺寸的声明形状。
    @return 问题描述元组；为空表示合法。
    """
    capacity = instance.capacity
    assert capacity is not None  # callers gate on is_capacity
    problems: list[str] = []
    shape = tuple(int(dim) for dim in getattr(cu, "shape", ()))
    if shape != tuple(expected_shape):
        problems.append(f"cu_seqlens has shape {shape}, but the capacity ABI declares {tuple(expected_shape)}")
    dtype = _dtype_name(cu)
    if dtype != "int32":
        problems.append(f"cu_seqlens dtype is {dtype or type(cu).__name__!r}; the boundary ABI is i32")
    values = offsets(cu)
    if values is None:
        return (*problems, "cu_seqlens is not readable as integers")
    if len(values) != tuple(expected_shape)[0]:
        return (*problems, f"cu_seqlens carries {len(values)} offsets, but {tuple(expected_shape)[0]} are declared")
    prefix_len = int(resolved.num_seqs) + 1
    prefix = values[:prefix_len]
    tail = values[prefix_len:]
    if len(prefix) < 2:
        problems.append(f"cu_seqlens must carry at least two prefix offsets, got {len(prefix)}")
    if prefix and prefix[0] != 0:
        problems.append(f"cu_seqlens must start at 0, got {prefix[0]}")
    if prefix and prefix[-1] != int(resolved.total_tokens):
        problems.append(
            f"cu_seqlens prefix must end at the resolved total_tokens ({resolved.total_tokens}), got {prefix[-1]}"
        )
    for index, (begin, end) in enumerate(zip(prefix, prefix[1:])):
        if end < begin:
            problems.append(
                f"cu_seqlens must be non-decreasing; sequence {index} ends before it begins ({begin} > {end})"
            )
            break
    for index, value in enumerate(tail):
        if value != capacity.cu_seqlens_sentinel:
            problems.append(
                f"cu_seqlens capacity tail entry {prefix_len + index} is {value}, but the declared poison sentinel "
                f"is {capacity.cu_seqlens_sentinel}; the unused tail must stay poison"
            )
            break
    if any(value < 0 or value > INT32_MAX for value in prefix):
        problems.append("every cu_seqlens offset must stay inside the int32 domain")
    return tuple(problems)


def axis_kind(instance: SemanticInstanceV2, dim: Any, term: Any) -> str:
    """@brief 把一个声明 dim 轴分类为 resolved 轴、可证明的 invariant 轴，或拒绝。

    The logical/padding decomposition needs to know *which* axis of a capacity tensor is this call's
    ``total_tokens``/``num_seqs``.  Reading that from bare integers would guess, so only the declared
    dim term counts.  Two kinds of term are accepted:

    ``resolved:<name>``
        the term is exactly ``total_tokens`` or ``num_seqs`` and the bound length is that capacity
        symbol.  This axis shrinks to the resolved value.
    ``invariant``
        the term is a product of declared symbols, **none** of which is a resolved symbol and **none**
        of which has a capacity value equal to the capacity ``total_tokens``/``num_seqs``.  Each factor
        is then an independent parameter extent, and the axis cannot be a stand-in for the resolved
        axis.

    Anything else is refused instead of being silently read as invariant.  That covers the alias escape
    (a second symbol ``T2`` declared at the capacity value and used as the row axis: it is
    indistinguishable from ``total_tokens`` and would hide the whole unused region from both the padding
    check and the logical comparison), a fused term (``total_tokens*H``: no single axis shrinks) and a
    missing term (nothing to prove anything with).  Refusing is a *validation* failure for a document
    and a typed refusal at prepare/check time for a hand-built instance, never a full-buffer fallback.

    A product such as the parameter extent ``H*D`` is allowed even when ``H*D`` numerically equals the
    capacity total: its factors are independently declared and used to size the rest of the ABI, and the
    collision is not evidence that the axis is per-token.  The dangerous case is an *opaque* symbol whose
    only declared value is the capacity, which is exactly what an alias is.

    @param instance 语义实例。
    @param dim 该轴的声明（容量）长度。
    @param term 该轴的声明 dim 词；``None`` 表示绑定没有 dim 词。
    @return ``"resolved:<name>"`` 或 ``"invariant"``。
    @raises VarlenContractError 无法证明该轴是 invariant 且不是 canonical resolved 轴。
    """
    if not instance.is_capacity:
        return "invariant"
    if term is None:
        raise VarlenContractError(
            f"{instance.op_name}.{instance.size}: an L2 (capacity) tensor axis of length {int(dim)} has no declared "
            "dim term, so the harness cannot tell whether it is a resolved axis or an invariant extent; the "
            "logical/padding region would be undefined"
        )
    try:
        kind, name = resolve_axis_kind(
            int(dim), term, instance.symbols, where=f"{instance.op_name}.{instance.size}"
        )
    except OpSpecError as exc:
        # The document validator and the runtime share one rule; only the exception type differs
        # (a loaded instance is refused as a varlen contract violation, not as a schema error).
        raise VarlenContractError(str(exc)) from exc
    return "invariant" if kind == "invariant" else f"resolved:{name}"


def capacity_axis_problems(instance: SemanticInstanceV2) -> tuple[str, ...]:
    """@brief 一次 L2 实例的全部张量轴是否都能被分类（alias/fused/missing 全部拒绝）。

    The per-axis rule is :func:`axis_kind`; this function applies it to every non-workspace tensor so
    the check is instance-wide instead of "only the tensors the current call happens to slice".  The
    boundary tensor ``cu_seqlens`` is exempt: its capacity-sized axis is the sentinel-padded tail, which
    :func:`cu_seqlens_problems` validates from the bytes.  Workspaces are candidate-owned scratch and
    never carry a logical/padding contract.  A non-row-major layout is refused as well: the slabs index
    the contiguous byte image every serializer/poison check works on.

    @param instance 语义实例。
    @return 问题描述元组；为空表示每个轴都可分类。
    """
    if not instance.is_capacity:
        return ()
    problems: list[str] = []
    result_names = set(instance.reference_outputs)
    for binding in instance.tensors:
        if binding.space == "workspace" or binding.name == BOUNDARY_TENSOR:
            continue
        if binding.layout not in ("row_major", "contiguous"):
            # The logical/padding slabs index the *contiguous* byte image (`to_raw`/`from_raw` and the
            # poison check all do), so a genuinely strided layout would make the region unprovable — and a
            # mis-identified logical region can put poison back into the error scale.
            problems.append(
                f"{binding.name}: layout {binding.layout!r} is not row-major/contiguous, so the "
                "logical/padding byte region cannot be derived for a capacity tier"
            )
            continue
        for axis, dim in enumerate(binding.shape):
            term = binding.dim_terms[axis] if axis < len(binding.dim_terms) else None
            where = f"{binding.name}.dims[{axis}]"
            try:
                axis_kind(instance, dim, term)
            except VarlenContractError as exc:
                problems.append(f"{where}: {exc}")
                continue
            # The role-based closure (literally the same function as the document validator): a result axis at
            # the capacity bound must name ``total_tokens``/``num_seqs``, and a non-result packed axis must be
            # named or explicitly declared a parameter extent.  Without it ``"4"``/``"H*2"``/``"H*D"`` classify
            # invariant, the logical region becomes the whole capacity buffer, and a custom error_scale reading
            # ``inputs["q"]`` is handed the padding poison.
            problem = capacity_axis_role_problem(
                int(dim),
                term,
                instance.symbols,
                declared_role=binding.declared_axis_role(axis),
                is_result=binding.name in result_names,
                where=where,
            )
            if problem:
                problems.append(problem)
    return tuple(problems)


def require_capacity_axes(instance: SemanticInstanceV2) -> None:
    """@brief 断言每个容量张量轴都可分类，否则 typed 拒绝（不退回整缓冲区比较）。

    @param instance 语义实例。
    @raises VarlenContractError 存在 alias/fused/missing 轴。
    """
    problems = capacity_axis_problems(instance)
    if problems:
        raise VarlenContractError(
            "an L2 capacity instance has axes whose logical/padding role cannot be proven; refusing rather than "
            "comparing the whole capacity buffer: " + "; ".join(problems)
        )


def logical_region(instance: SemanticInstanceV2, resolved: ResolvedShape, name: str) -> tuple[slice, ...]:
    """@brief 一个（已声明）张量的 logical 区域切片元组。

    @param instance 语义实例。
    @param resolved 本次调用的 resolved 形状。
    @param name 张量名（必须是已声明张量）。
    @return 与声明维数等长的切片元组。
    @raises VarlenContractError 轴不可分类。
    """
    binding = instance.tensor(name)
    logical, _slabs = padding_slabs(instance, resolved, binding.shape, binding.dim_terms)
    return logical


def declared_dim_terms(instance: SemanticInstanceV2, name: str) -> tuple[str, ...]:
    """@brief 一个已绑定张量的**声明** dim 词（用于识别 resolved 轴）。

    The bound :class:`~tcenv.opspec.model.TensorBinding` keeps both the resolved capacity dimensions and
    the declared dim text (``"total_tokens"`` / ``"num_seqs"`` / ``"H"`` / ...).  Resolving a logical
    region from bare integers would guess: ``H * D`` can equal ``total_tokens``, and a tensor whose row
    axis merely *looks* like the capacity would then be mis-sliced.  Only the declared term counts.

    @param instance 语义实例。
    @param name 张量名。
    @return dim 词元组；没有该张量时为空元组。
    """
    for binding in instance.tensors:
        if binding.name == name:
            return tuple(str(term) for term in binding.dim_terms)
    return ()


def padding_slabs(
    instance: SemanticInstanceV2,
    resolved: ResolvedShape,
    shape: Sequence[int],
    terms: Sequence[str],
) -> tuple[tuple[slice, ...], tuple[tuple[slice, ...], ...]]:
    """@brief 一个 capacity 张量的 logical 区域与 padding 区域（不相交的 slab 列表）。

    ``shape`` is the *declared* (capacity) shape and ``terms`` its declared dim text.  An axis whose
    term is ``total_tokens``/``num_seqs`` shrinks to the resolved value; every other axis keeps its
    capacity length (``H``/``D``/``NSP1``/...).  The padding region is the box complement, decomposed
    into at most one slab per resolved axis, so no element mask is materialised (a real-size batch
    would be hundreds of megabytes).

    A tensor with no resolved axis (``A_log``, ``dt_bias``, ``scale``, ...) has an empty padding region
    and its logical region is the whole buffer.  The same decomposition is used for inputs *and*
    outputs, so "only the logical prefix is compared" and "the padding must stay poison" cannot
    disagree about where the boundary is.

    An axis whose declared term cannot be classified as either a resolved axis or a provably invariant
    extent is refused by :func:`axis_kind`; there is no "compare the whole buffer" fallback, because that
    would silently accept an aliased row axis as a logical region.

    @param instance 语义实例。
    @param resolved 本次调用的 resolved 形状。
    @param shape 声明的 capacity 形状。
    @param terms 声明的 dim 词。
    @return ``(logical_slices, padding_slab_list)``。
    @raises VarlenContractError 某个轴无法证明其 logical/padding 角色。
    """
    resolved_dims = {"total_tokens": int(resolved.total_tokens), "num_seqs": int(resolved.num_seqs)}
    logical: list[slice] = []
    resolved_axes: list[tuple[int, int, int]] = []
    for axis, dim in enumerate(shape):
        term = terms[axis] if axis < len(terms) else None
        kind = axis_kind(instance, dim, term)
        if kind.startswith("resolved:"):
            name = kind.split(":", 1)[1]
            logical.append(slice(0, resolved_dims[name]))
            resolved_axes.append((axis, resolved_dims[name], int(dim)))
        else:
            logical.append(slice(0, int(dim)))
    slabs: list[tuple[slice, ...]] = []
    for axis, live, bound in resolved_axes:
        if live >= bound:
            continue
        slab: list[slice] = []
        for other in range(len(shape)):
            if other < axis:
                slab.append(logical[other])
            elif other == axis:
                slab.append(slice(live, bound))
            else:
                slab.append(slice(0, int(shape[other])))
        slabs.append(tuple(slab))
    return tuple(logical), tuple(slabs)


def padding_byte_problems(
    instance: SemanticInstanceV2,
    resolved: ResolvedShape,
    raw: Mapping[str, bytes],
    shapes: Mapping[str, Sequence[int]],
    elem_nbytes: Mapping[str, int],
    *,
    where: str,
    declared_terms: Mapping[str, Sequence[str]] | None = None,
    padding_byte: int | None = None,
) -> tuple[str, ...]:
    """@brief 在**存储字节**上检查 logical 区域之外是否仍是声明的 padding poison。

    This is the authoritative form: the bytes are what the device reads and writes, so the check cannot
    be confused by a widened logical dtype.  ``round_to`` widens fp16/bf16 to float32 for the oracle, and
    ``203.25`` (the fp16 pattern ``0x5A5A``) has a *different* float32 encoding — a value-level check on
    the widened tensor would reject every honest generator for the wrong reason.

    @param instance 语义实例。
    @param resolved 本次调用的 resolved 形状。
    @param raw 张量名 → 存储字节。
    @param shapes 张量名 → 声明的 capacity 形状。
    @param elem_nbytes 张量名 → 单元素存储字节数。
    @param where 诊断位置。
    @param declared_terms 张量名 → 声明的 dim 词；省略时取自实例绑定。
    @param padding_byte 期望的 padding 字节；省略时用容量档声明值。
    @return 问题描述元组；为空表示合法。
    """
    if not instance.is_capacity:
        return ()
    import numpy as np

    capacity = instance.capacity
    expected = int(capacity.padding_poison_byte if padding_byte is None else padding_byte)
    problems: list[str] = []
    for name, data in raw.items():
        shape = tuple(int(dim) for dim in shapes[name])
        nbytes = int(elem_nbytes[name])
        terms = (
            tuple(str(term) for term in declared_terms[name])
            if declared_terms is not None and name in declared_terms
            else declared_dim_terms(instance, name)
        )
        _, slabs = padding_slabs(instance, resolved, shape, terms)
        if not slabs:
            continue
        array = np.frombuffer(data, dtype=np.uint8)
        if int(array.size) != int(np.prod(shape)) * nbytes:
            problems.append(
                f"{where}: {name} carries {int(array.size)} bytes, but the capacity shape {shape} at "
                f"{nbytes} byte(s) per element needs {int(np.prod(shape)) * nbytes}"
            )
            continue
        array = array.reshape(shape + (nbytes,))
        for slab in slabs:
            region = array[tuple(slab)]
            if int(region.size) and not bool((region == expected).all()):
                bad = int((region != expected).sum())
                problems.append(
                    f"{where}: {name} padding {list(slab)} is not the declared poison byte 0x{expected:02X}; "
                    f"{bad} byte(s) were changed outside the logical region"
                )
                break
    return tuple(problems)


def padding_problems(
    instance: SemanticInstanceV2,
    resolved: ResolvedShape,
    tensors: Mapping[str, Any],
    *,
    where: str,
    declared_terms: Mapping[str, Sequence[str]] | None = None,
    padding_byte: int | None = None,
) -> tuple[str, ...]:
    """@brief 检查 capacity 输入/输出里 logical 区域之外是否仍**逐字节**是声明的 poison。

    Byte comparison (not value comparison) is deliberate: fp16/bf16/fp32 padding must be detectable
    without a dtype-dependent NaN trick, and a candidate that "writes a zero" outside the logical
    region has to be refused.  ``padding_byte`` overrides the declared value only for a negative test.

    @param instance 语义实例。
    @param resolved 本次调用的 resolved 形状。
    @param tensors 张量名 → torch 张量。
    @param where 诊断位置。
    @param declared_terms 张量名 → 声明的 dim 词；省略时取自实例绑定。
    @param padding_byte 期望的 padding 字节；省略时用容量档声明值。
    @return 问题描述元组；为空表示合法。
    """
    if not instance.is_capacity:
        return ()
    # The torch form is a convenience for already-storage-dtype tensors; it converts each tensor to its
    # own storage bytes and reuses the byte check, so both entry points agree by construction.
    raw: dict[str, bytes] = {}
    shapes: dict[str, tuple[int, ...]] = {}
    elem_nbytes: dict[str, int] = {}
    for name, tensor in tensors.items():
        value = tensor.detach().to("cpu").contiguous()
        raw[name] = value.numpy().tobytes()
        shapes[name] = tuple(int(dim) for dim in value.shape)
        elem_nbytes[name] = int(value.element_size())
    return padding_byte_problems(
        instance,
        resolved,
        raw,
        shapes,
        elem_nbytes,
        where=where,
        declared_terms=declared_terms,
        padding_byte=padding_byte,
    )


__all__ = [
    "BOUNDARY_TENSOR",
    "RESOLVED_SCALAR_NAMES",
    "ResolvedShape",
    "axis_kind",
    "capacity_axis_problems",
    "cu_seqlens_problems",
    "declared_dim_terms",
    "logical_region",
    "padding_byte_problems",
    "padding_problems",
    "padding_slabs",
    "require_capacity_axes",
    "resolved_shape_for_case",
    "resolved_shape_from_inputs",
    "workload_profile_for_case",
]
