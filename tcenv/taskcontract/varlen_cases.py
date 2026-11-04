"""What an L1 correctness case actually realized on a tier (auditable coverage).

`VARLEN_CASE_DOMAIN` (in :mod:`tcenv.opspec.v2`) is the floor a varlen task must declare, and
:mod:`tcenv.taskcontract.varlen` refuses an illegal runtime boundary set.  Neither answers the
next question: *did the case named ``all_unaligned_primes`` really run prime-length segments on
this tier, or did the tier's ``total_tokens``/``num_seqs`` make that impossible and the planner
fell back to the only legal segmentation?*  A successful correctness report cannot tell the two
apart, so this module classifies every case from the **actual** boundaries and records what was
realized, what was impossible and why.

The classification is harness-owned and mechanical: it only inspects the boundary vector it is
handed, never the task's generator.  The kinds are closed:

``exact``
    the case's defining property holds on the actual segments;
``partial``
    the case holds for the facets the tier can host, and the recorded ``missing`` facets cannot
    be hosted (for example a two-sequence tier has no slot for the other ``chunk_boundary``
    targets, or a uniform canonical profile has no different permutation);
``not_constructible``
    the property cannot exist for this ``total_tokens``/``num_seqs`` (an all-prime layout that
    does not exist, an empty segment that would make the batch empty, a fresh draw or a
    reordering on a tier with exactly one legal segmentation);
``deviates``
    the planner *could* have realized the property but did not — a task-planner bug signal, not a
    candidate failure;
``unclassified``
    the task declared an extra case this harness has no L1 property for.

A case is never counted as full coverage when it is ``partial``/``not_constructible``/
``deviates``: the per-case record travels in the correctness report and the aggregate
``case_coverage`` lists the case names per kind.
"""

from __future__ import annotations

from typing import Any

#: Closed kind vocabulary (see the module docstring).
EXACT = "exact"
PARTIAL = "partial"
NOT_CONSTRUCTIBLE = "not_constructible"
DEVIATES = "deviates"
UNCLASSIFIED = "unclassified"
CASE_REALIZATION_KINDS: tuple[str, ...] = (EXACT, PARTIAL, NOT_CONSTRUCTIBLE, DEVIATES, UNCLASSIFIED)

#: Chunk size the ``chunk_boundary`` case plans around.  A candidate owns its real chunk/BT
#: choice; this only defines the sizes the *case* promises to approach.  The shipped tasks
#: duplicate the value because a reference module cannot import this package; the shared CPU
#: test asserts both copies equal this one.
VARLEN_CHUNK = 4
#: ``chunk_boundary`` targets: ``BT-1``, ``2BT-1``, ``BT``, ``BT+1``, ``1``, ``2BT+1``.
VARLEN_CHUNK_TARGETS: tuple[int, ...] = (
    VARLEN_CHUNK - 1,
    2 * VARLEN_CHUNK - 1,
    VARLEN_CHUNK,
    VARLEN_CHUNK + 1,
    1,
    2 * VARLEN_CHUNK + 1,
)


def _is_prime(value: int) -> bool:
    """@brief 素数判定（只用于独立核对 case 的可构造性）。

    @param value 整数。
    @return 是否素数。
    """
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def prime_partition(total_tokens: int, num_seqs: int) -> tuple[int, ...] | None:
    """@brief 独立搜索一个全素数分段（仅用于判断该 case 是否可构造）。

    @param total_tokens 总 token 数。
    @param num_seqs 段数。
    @return 非递增素数元组；不存在时为 ``None``。
    """
    largest = total_tokens - 2 * (num_seqs - 1)
    if num_seqs < 1 or largest < 2:
        return None
    primes = [value for value in range(2, largest + 1) if _is_prime(value)]

    def walk(remaining: int, slots: int, ceiling: int, chosen: tuple[int, ...]) -> tuple[int, ...] | None:
        if slots == 1:
            return chosen + (remaining,) if remaining <= ceiling and _is_prime(remaining) else None
        for prime in reversed(primes):
            if prime > ceiling or prime > remaining - 2 * (slots - 1):
                continue
            found = walk(remaining - prime, slots - 1, prime, chosen + (prime,))
            if found is not None:
                return found
        return None

    return walk(total_tokens, num_seqs, largest, ())


def _record(
    case: str,
    kind: str,
    lengths: tuple[int, ...],
    *,
    reason: str,
    realized: list[str] | None = None,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    """@brief 构造一条闭合的 realization 记录。

    @param case 用例名。
    @param kind 闭集种类。
    @param lengths 实际每段长度。
    @param reason 人可读的原因。
    @param realized 已实现的性质面。
    @param missing 未实现/不可实现的性质面。
    @return JSON 安全的记录。
    """
    return {
        "case": case,
        "kind": kind,
        "segments": [int(value) for value in lengths],
        "realized": list(realized or ()),
        "missing": list(missing or ()),
        "reason": reason,
    }


def case_realization(case: str, lengths: tuple[int, ...], canonical: tuple[int, ...]) -> dict[str, Any]:
    """@brief 由**实际**分段判断一个 L1 case 在该档上实现了什么。

    @param case 用例名（与 OpSpec ``cases.names`` 一致）。
    @param lengths 实际每段长度（``cu_seqlens[s+1] - cu_seqlens[s]``）。
    @param canonical 该档声明的 canonical 分段长度。
    @return 闭合记录：``case``/``kind``/``segments``/``realized``/``missing``/``reason``。
    @raises ValueError 传入的长度与 canonical 不一致（harness 调用错误）。
    """
    lengths = tuple(int(value) for value in lengths)
    canonical = tuple(int(value) for value in canonical)
    if len(lengths) != len(canonical) or sum(lengths) != sum(canonical) or not canonical:
        raise ValueError(f"case {case!r}: {lengths} is not a legal reshaping of canonical {canonical}")
    total, count = sum(lengths), len(lengths)

    if case == "canonical":
        if lengths == canonical:
            return _record(case, EXACT, lengths, reason="the declared canonical profile", realized=["canonical profile"])
        return _record(case, DEVIATES, lengths, reason="the case was declared canonical but ran another layout")

    if count == 1:
        if case == "single_sequence":
            return _record(
                case, EXACT, lengths, reason="the tier is itself the degenerate single sequence", realized=["one sequence"]
            )
        return _record(
            case,
            NOT_CONSTRUCTIBLE,
            lengths,
            reason="a num_seqs == 1 tier has exactly one legal segmentation, so this case's shape cannot be constructed",
            missing=["a multi-segment layout"],
        )

    if case == "permuted":
        if lengths == canonical:
            return _record(case, DEVIATES, lengths, reason="the case replayed the canonical profile")
        if sorted(lengths) == sorted(canonical):
            return _record(
                case, EXACT, lengths, reason="a reordering of the canonical multiset", realized=["canonical multiset"]
            )
        return _record(
            case,
            PARTIAL,
            lengths,
            reason="the canonical profile is uniform, so no different order exists; a different legal split was used",
            realized=["a different legal split"],
            missing=["a permutation of the canonical multiset"],
        )

    if case == "skewed":
        long = total - 2 * (count - 1)
        if long >= 2 and tuple(sorted(lengths, reverse=True)) == (long, *([2] * (count - 1))):
            return _record(case, EXACT, lengths, reason="one long sequence plus two-token fragments", realized=["one long sequence", "two-token fragments"])
        if long < 2 and tuple(sorted(lengths, reverse=True)) == (total - (count - 1), *([1] * (count - 1))):
            return _record(
                case,
                PARTIAL,
                lengths,
                reason="the tier cannot host two-token fragments; one-token fragments were used",
                realized=["one long sequence", "one-token fragments"],
                missing=["two-token fragments"],
            )
        return _record(case, DEVIATES, lengths, reason="the layout is neither one long sequence plus two-token fragments nor the one-token fallback")

    if case == "all_unaligned_primes":
        if all(_is_prime(value) for value in lengths):
            return _record(case, EXACT, lengths, reason="every segment length is prime", realized=["prime segments"])
        if prime_partition(total, count) is None:
            return _record(
                case,
                NOT_CONSTRUCTIBLE,
                lengths,
                reason=f"total_tokens={total} / num_seqs={count} has no all-prime segmentation",
                missing=["prime segments"],
            )
        return _record(
            case, DEVIATES, lengths, reason="an all-prime segmentation exists for this tier but was not used"
        )

    if case == "max_handoff_stress":
        expected = (total - (count - 1), *([1] * (count - 1)))
        if lengths == expected:
            return _record(
                case, EXACT, lengths, reason="one longest chain plus one-token fragments", realized=["longest chain", "one-token fragments"]
            )
        return _record(case, DEVIATES, lengths, reason=f"expected {list(expected)} for the documented formula")

    if case == "chunk_boundary":
        planned = tuple(VARLEN_CHUNK_TARGETS[index % len(VARLEN_CHUNK_TARGETS)] for index in range(count - 1))
        realized = [target for index, target in enumerate(planned) if lengths[index] == target]
        missing = [target for index, target in enumerate(planned) if lengths[index] != target]
        missing.extend(VARLEN_CHUNK_TARGETS[len(planned) :])
        if missing:
            return _record(
                case,
                PARTIAL,
                lengths,
                reason=(
                    f"the tier has {count - 1} planned chunk slots, so only {realized or 'no'} of the "
                    f"BT targets {list(VARLEN_CHUNK_TARGETS)} could be realized (the last segment is the remainder)"
                ),
                realized=[f"BT target {target}" for target in realized],
                missing=[f"BT target {target}" for target in missing],
            )
        return _record(
            case,
            EXACT,
            lengths,
            reason="every planned chunk target was realized",
            realized=[f"BT target {target}" for target in realized],
        )

    if case == "zero_length":
        if lengths.count(0) == 1 and any(value > 0 for value in lengths) and total >= 1:
            return _record(case, EXACT, lengths, reason="exactly one empty sequence in a non-empty batch", realized=["one empty sequence"])
        return _record(case, DEVIATES, lengths, reason="the case did not produce exactly one empty segment")

    if case == "single_sequence":
        non_empty = [value for value in lengths if value > 0]
        if len(non_empty) == 1:
            return _record(
                case,
                EXACT,
                lengths,
                reason="exactly one non-empty sequence carries the whole batch; the others are empty",
                realized=["one non-empty sequence", "empty remaining sequences"],
            )
        return _record(case, DEVIATES, lengths, reason="more than one sequence is non-empty")

    if case == "fresh_distribution":
        if lengths != canonical:
            return _record(case, EXACT, lengths, reason="a legal segmentation that differs from the published canonical profile", realized=["fresh segmentation"])
        return _record(case, DEVIATES, lengths, reason="the fresh draw replayed the canonical profile")

    return _record(
        case,
        UNCLASSIFIED,
        lengths,
        reason="this harness registers no mechanical L1 property for the case; its coverage is not counted",
    )


__all__ = [
    "CASE_REALIZATION_KINDS",
    "DEVIATES",
    "EXACT",
    "NOT_CONSTRUCTIBLE",
    "PARTIAL",
    "UNCLASSIFIED",
    "VARLEN_CHUNK",
    "VARLEN_CHUNK_TARGETS",
    "case_realization",
    "prime_partition",
]
