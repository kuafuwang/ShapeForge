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


def is_varlen(instance: Any) -> bool:
    """@brief 该实例是否是 L1 packed-varlen 实例。

    @param instance 语义实例。
    @return 有 canonical ``cu_seqlens`` 时为真。
    """
    return isinstance(instance, SemanticInstanceV2) and instance.cu_seqlens is not None


def offsets(cu_seqlens: Any) -> tuple[int, ...] | None:
    """@brief 把边界张量/序列读成整数元组（不可读时返回 ``None``）。

    @param cu_seqlens torch 张量、numpy 数组或整数序列。
    @return 偏移元组；既不是张量也没有可迭代值时为 ``None``。
    """
    if hasattr(cu_seqlens, "tolist"):
        return tuple(int(value) for value in cu_seqlens.tolist())
    if isinstance(cu_seqlens, (list, tuple, range)):
        return tuple(int(value) for value in cu_seqlens)
    try:
        return tuple(int(value) for value in cu_seqlens)
    except TypeError:
        return None


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


def runtime_segment_problems(instance: SemanticInstanceV2, inputs: Mapping[str, Any]) -> tuple[str, ...]:
    """@brief 校验本次实际 ``cu_seqlens`` 是否满足 L1 合同（固定 shape 实例恒为空）。

    The check runs on the *actual* bytes handed to the oracle/device path.  It accepts an empty
    sequence (``cu_seqlens[s] == cu_seqlens[s+1]``) but never an empty batch, and it never
    compares the tensor with the instance's canonical copy: a permuted/skewed/fresh
    segmentation is legal precisely because it differs from that copy.

    @param instance 语义实例。
    @param inputs ``reference_inputs`` 校正后的输入映射。
    @return 问题描述元组；为空表示合法。
    """
    if not is_varlen(instance):
        return ()
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
        return (*problems, f"cu_seqlens is not readable as integers ({type(cu).__name__})")
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


def require_runtime_segments(instance: SemanticInstanceV2, inputs: Mapping[str, Any], *, where: str) -> None:
    """@brief 断言本次实际分段合法，否则抛 typed refusal。

    @param instance 语义实例。
    @param inputs 输入映射。
    @param where 诊断位置。
    @raises VarlenContractError 边界不满足 L1 合同。
    """
    problems = runtime_segment_problems(instance, inputs)
    if problems:
        raise VarlenContractError(f"{where}: illegal runtime cu_seqlens: {'; '.join(problems)}")


__all__ = [
    "INT32_MAX",
    "is_varlen",
    "offsets",
    "require_runtime_segments",
    "runtime_segment_problems",
    "segment_lengths",
]
