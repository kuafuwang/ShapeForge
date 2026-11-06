"""Runtime facts of a varlen (packed-sequence) instance: the boundaries the harness checks.

A varlen tier fixes ``total_tokens``/``num_seqs`` (so the packed tensor shapes, the buffer ABI and
every compile-time geometric constant stay constants) but **not** the segmentation:
``sizes.<tier>.seq_lens`` is only the *canonical* profile that is published for scheduling, exact
dispatch and benchmarking.  Correctness must therefore accept every legal ``cu_seqlens`` under the
same total/count, and the harness — not the task's ``gen_inputs`` — owns the invariant check:

* the tensor is the declared ``i32`` input of length ``num_seqs + 1``;
* it starts at 0, is monotonically non-decreasing, and ends at ``total_tokens``;
* the batch carries at least one token (an *empty sequence* is legal, an *empty batch* is not),
  and every offset stays inside the int32 domain of the boundary ABI.

Static declaration and runtime input are checked separately.  The static side (``sizes.<tier>.
seq_lens``) requires every declared segment to be >= 1 and is enforced by
:func:`tcenv.opspec.model._validate_size`; a zero-length segment may only come from the runtime
tensor, which is why :func:`runtime_segment_problems` checks ``cu_seqlens[s+1] - cu_seqlens[s] ==
0`` instead of re-reading the declaration.

Nothing here reads the task's generator, so a task cannot certify its own boundaries: the single
funnel is :class:`tcenv.taskcontract.reference.ReferenceModule`, which calls
:func:`require_runtime_segments` for both ``generate`` and ``execute``.

This module is deliberately independent of :mod:`tcenv.portfolio.domain`: the task-contract plane
must not import the portfolio plane, and a worker that only receives the task pack still has to
refuse an illegal boundary set.
"""

from __future__ import annotations

from typing import Any, Mapping

from ..contract_errors import VarlenContractError
from ..opspec.v2 import SemanticInstanceV2

#: Upper bound of the int32 boundary ABI (``i32`` offsets).
INT32_MAX = (1 << 31) - 1
#: Exceptions a boundary read can raise for an **unreadable layout** even though the tensor's
#: shape/dtype match: no storage (sparse COO), no data (meta), non-integer content, NaN/Inf.
#: Deliberately not ``BaseException``: ``KeyboardInterrupt``/``SystemExit`` are not input errors
#: and must keep propagating.
_UNREADABLE_BOUNDARY_ERRORS = (TypeError, ValueError, OverflowError, RuntimeError, NotImplementedError)


def is_varlen(instance: Any) -> bool:
    """@brief 该实例是否是变长（L1 packed-varlen 或 L2 capacity）实例。

    ``cu_seqlens is not None`` would be the wrong test: an L2 instance deliberately carries no canonical
    segmentation, and treating it as fixed-shape would skip the whole boundary contract.  The single
    discriminator is :attr:`SemanticInstanceV2.variable_length_kind`.

    @param instance 语义实例。
    @return 变长（L1/L2）时为真。
    """
    return isinstance(instance, SemanticInstanceV2) and instance.variable_length_kind in ("l1", "l2")


def is_capacity(instance: Any) -> bool:
    """@brief 该实例是否是 L2 capacity 实例。

    @param instance 语义实例。
    @return 布尔。
    """
    return isinstance(instance, SemanticInstanceV2) and instance.is_capacity


def offsets(cu_seqlens: Any) -> tuple[int, ...] | None:
    """@brief 把边界张量/序列读成整数元组（不可读时返回 ``None``，绝不抛裸异常）。

    A ``cu_seqlens`` tensor that reached this helper may still be wrong in every way an attacker
    or a buggy generator can arrange: two-dimensional, nested, floating with NaN/Inf, objects.
    The contract check must always end in a *typed* refusal, so conversion is total: any shape,
    dtype or value that cannot be read as a flat integer sequence returns ``None`` and the caller
    reports the problem instead of raising ``ValueError``/``OverflowError``/``TypeError``.

    @param cu_seqlens torch 张量、numpy 数组或整数序列。
    @return 偏移元组；不是可读的平铺整数序列（含 sparse/meta 等无存储布局）时为 ``None``。
    """
    try:
        raw = cu_seqlens.tolist() if hasattr(cu_seqlens, "tolist") else cu_seqlens
        if not isinstance(raw, (list, tuple)):
            return None
        values = [int(value) for value in raw]
    except _UNREADABLE_BOUNDARY_ERRORS:
        # NaN/Inf（float 边界的典型错法）、嵌套值（rank > 1）、无存储（sparse COO）与无数据
        # （meta）都落到这里，统一变成“不可读”而不是裸异常。
        return None
    return tuple(values)


def segment_lengths(cu_seqlens: Any) -> tuple[int, ...]:
    """@brief 由边界取每段长度（``cu_seqlens[s+1] - cu_seqlens[s]``）。

    @param cu_seqlens 边界（torch 张量或整数序列）。
    @return 每段 token 数（空序列为 0）。
    @raises VarlenContractError 边界不可读。
    """
    values = offsets(cu_seqlens)
    if values is None:
        raise VarlenContractError(f"cu_seqlens must be an integer tensor/sequence, got {type(cu_seqlens).__name__}")
    return tuple(end - begin for begin, end in zip(values, values[1:]))


def _dtype_name(value: Any) -> str:
    """@brief 张量 dtype 的短名（``torch.int32`` → ``int32``，numpy 保持 ``int32``）。

    @param value 张量/数组。
    @return dtype 短名；没有 dtype 时为空串。
    """
    name = str(getattr(value, "dtype", ""))
    return name.rsplit(".", 1)[-1] if name else ""


def runtime_segment_problems(
    instance: SemanticInstanceV2,
    inputs: Mapping[str, Any],
    *,
    resolved: Any = None,
    case: str | None = None,
) -> tuple[str, ...]:
    """@brief 校验本次实际 ``cu_seqlens`` 是否满足 L1（或 L2 capacity）合同。

    The check runs on the *actual* bytes handed to the oracle/device path.  It accepts an empty
    sequence (``cu_seqlens[s] == cu_seqlens[s+1]``) but never an empty batch, and it never
    compares the tensor with the instance's canonical copy: a permuted/skewed/fresh
    segmentation is legal precisely because it differs from that copy.

    An L2 instance is checked against its **resolved** shape instead of the static symbols: the prefix
    ``cu_seqlens[0:num_seqs+1]`` is the L1 rule applied to this call's total/count, and the rest of the
    capacity-sized tensor must be exactly the declared negative sentinel.  When the caller did not
    supply a resolved shape, the expectation is reconstructed from the declared workload profile for
    this case (the harness's own expectation, never the generator's claim).

    @param instance 语义实例。
    @param inputs ``reference_inputs`` 校正后的输入映射。
    @param resolved 本次调用的期望 resolved 形状（L2）；省略时由 ``case`` 的已声明 profile 重建。
    @param case 用例名（L2 重建 resolved 期望用）。
    @return 问题描述元组；为空表示合法。
    """
    if not is_varlen(instance):
        return ()
    if instance.is_capacity:
        return _capacity_segment_problems(instance, inputs, resolved=resolved, case=case)
    total_tokens = int(instance.symbols["total_tokens"])
    num_seqs = int(instance.symbols["num_seqs"])
    expected = num_seqs + 1
    problems: list[str] = []
    if num_seqs < 1:
        problems.append(f"num_seqs must be >= 1 for a packed batch, got {num_seqs}")
    if int(instance.symbols["NSP1"]) != expected:
        problems.append(
            f"the instance declares NSP1={instance.symbols['NSP1']}, but num_seqs + 1 = {expected} offsets are required"
        )
    if total_tokens < 1:
        problems.append(f"total_tokens must be >= 1 for a packed batch, got {total_tokens}")
    if total_tokens > INT32_MAX:
        problems.append(f"total_tokens {total_tokens} exceeds the int32 boundary domain {INT32_MAX}")
    cu = inputs.get("cu_seqlens")
    if cu is None:
        return (*problems, "the varlen ABI declares an i32 'cu_seqlens' input, but it is missing")
    shape = tuple(int(dim) for dim in getattr(cu, "shape", ()))
    if shape != (expected,):
        problems.append(f"cu_seqlens has shape {shape}, but num_seqs + 1 = {expected} offsets are required")
    dtype = _dtype_name(cu)
    if dtype != "int32":
        problems.append(f"cu_seqlens dtype is {dtype or type(cu).__name__!r}; the boundary ABI is i32")
    values = offsets(cu)
    if values is None:
        return (
            *problems,
            "cu_seqlens is not readable as integers "
            f"(layout={getattr(cu, 'layout', None)}, device={getattr(cu, 'device', None)}); the boundary "
            "tensor must be a dense i32 tensor with readable storage",
        )
    if len(values) != expected:
        return (*problems, f"cu_seqlens carries {len(values)} offsets, but {expected} are required")
    if any(value < 0 or value > INT32_MAX for value in values):
        problems.append("every cu_seqlens offset must stay inside the int32 domain")
    if values[0] != 0:
        problems.append(f"cu_seqlens must start at 0, got {values[0]}")
    if values[-1] != total_tokens:
        problems.append(f"cu_seqlens must end at total_tokens ({total_tokens}), got {values[-1]}")
    for index, (begin, end) in enumerate(zip(values, values[1:])):
        if end < begin:
            problems.append(
                f"cu_seqlens must be non-decreasing; sequence {index} ends before it begins ({begin} > {end})"
            )
            break
    return tuple(problems)


def _capacity_segment_problems(
    instance: SemanticInstanceV2,
    inputs: Mapping[str, Any],
    *,
    resolved: Any = None,
    case: str | None = None,
) -> tuple[str, ...]:
    """@brief L2 的 ``cu_seqlens`` 合同：resolved 前缀 + 容量尾部 sentinel。

    The expected resolved shape comes from the declared workload profile of this case.  When a caller has
    no case (the device-side revalidation path), the shape is read from the bytes themselves and only the
    *structural* rules are applied (prefix legality, ``total_tokens <= capacity``, tail sentinel); that is
    strictly weaker than the case-bound check, so it can never accept a wrong resolved total that the
    case-bound path would refuse.

    @param instance 语义实例。
    @param inputs 输入映射。
    @param resolved 期望的 resolved 形状。
    @param case 用例名。
    @return 问题描述元组。
    """
    from .runtime_shape import cu_seqlens_problems, resolved_shape_for_case, resolved_shape_from_inputs

    expected = resolved
    if expected is None and case is not None:
        expected = resolved_shape_for_case(instance, case)
    if expected is None:
        try:
            expected = resolved_shape_from_inputs(instance, inputs)
        except VarlenContractError as exc:
            return (str(exc),)
    capacity = instance.capacity
    assert capacity is not None  # callers gate on is_capacity
    bound = [binding for binding in instance.tensors if binding.name == "cu_seqlens"]
    if not bound:  # pragma: no cover - validate_opspec_v2 requires the tensor
        return ("the capacity ABI declares no cu_seqlens tensor",)
    expected_shape = tuple(int(dim) for dim in bound[0].shape)
    return cu_seqlens_problems(instance, expected, inputs.get("cu_seqlens"), expected_shape=expected_shape)


def require_runtime_segments(
    instance: SemanticInstanceV2,
    inputs: Mapping[str, Any],
    *,
    where: str,
    resolved: Any = None,
    case: str | None = None,
) -> None:
    """@brief 断言本次实际分段合法，否则抛 typed refusal。

    @param instance 语义实例。
    @param inputs 输入映射。
    @param where 诊断位置。
    @param resolved 期望的 resolved 形状（L2）。
    @param case 用例名（L2 重建期望用）。
    @raises VarlenContractError 边界不满足 L1/L2 合同。
    """
    problems = runtime_segment_problems(instance, inputs, resolved=resolved, case=case)
    if problems:
        raise VarlenContractError(f"{where}: illegal runtime cu_seqlens: {'; '.join(problems)}")


__all__ = [
    "INT32_MAX",
    "is_capacity",
    "is_varlen",
    "offsets",
    "require_runtime_segments",
    "runtime_segment_problems",
    "segment_lengths",
]
