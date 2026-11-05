"""Independent Python oracle for the V3 Host Program planner fixtures.

This module is **test-only**: it never enters a candidate, a task pack or the timed path.  It is a
second implementation of the policy the C99 fixture encodes, written differently on purpose (Python
integer arithmetic + ``heapq`` instead of the fixture's fixed-point target and hand-written binary
heap) so a byte/behaviour comparison is a real differential, not the same function called twice.

The canonical mode reuses :func:`tests.support.npu_sim.varlen_worklist.plan_pieces`, which is
already an independent transliteration of the device plan stage; ``range_cut_lpt`` mirrors only the
candidate's mode-2 policy.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Sequence

from .npu_sim.varlen_worklist import Piece, plan_pieces

#: Candidate-owned fixed-point constants (mirror the C fixture's documented model).
COST_SHIFT = 8
START_COST = 230
HANDOFF_COST = 102
MAX_WORKERS = 8


@dataclass(frozen=True)
class OracledPiece:
    """@brief 一个 oracle piece：工作项 + 整数/定点成本。"""

    chain: int
    piece: int
    begin: int
    valid: int
    pieces: int
    cost: int

    @property
    def last(self) -> bool:
        return self.piece + 1 == self.pieces

    def as_piece(self) -> Piece:
        return Piece(chain=self.chain, piece=self.piece, begin=self.begin, valid=self.valid, pieces=self.pieces)


def canonical_pieces(segments: Sequence[int], heads: int, max_split: int) -> tuple[Piece, ...]:
    """@brief mode 1 的 oracle：均衡切分 + canonical 顺序（复用独立实现的 V2 planner）。"""
    return plan_pieces(segments, heads, max_split)


def range_cut_pieces(segments: Sequence[int], heads: int, max_split: int) -> tuple[OracledPiece, ...]:
    """@brief mode 2 的 range-cut oracle：固定点成本目标、每段至多 ``max_split`` 个非空 piece。"""
    out: list[OracledPiece] = []
    for chain in range(len(segments) * heads):
        seq, _head = divmod(chain, heads)
        length = int(segments[seq])
        if length == 0:
            continue
        pieces = min(max_split, length)
        target = (length << COST_SHIFT) // pieces + START_COST
        begin = sum(int(value) for value in segments[:seq])
        remaining = length
        cursor = begin
        for index in range(pieces):
            slots_left = pieces - index - 1
            if index + 1 == pieces:
                size = remaining
            else:
                want = max(1, (target - START_COST) >> COST_SHIFT)
                size = min(want, remaining - slots_left)
            last = index + 1 == pieces
            cost = (size << COST_SHIFT) + START_COST + (0 if last else HANDOFF_COST)
            out.append(OracledPiece(chain, index, cursor, size, pieces, cost))
            cursor += size
            remaining -= size
    return tuple(out)


def lpt_assignment(pieces: Sequence[OracledPiece], workers: int) -> tuple[dict[int, int], int]:
    """@brief 整链 LPT：``heapq`` 维护工人负载；返回 ``chain -> worker`` 与 makespan。

    链必须整体落在一个 worker 上（同一链的 piece 之间是 handoff 依赖）。
    """
    if workers < 1:
        raise ValueError("workers must be >= 1")
    chain_cost: dict[int, int] = {}
    for piece in pieces:
        chain_cost[piece.chain] = chain_cost.get(piece.chain, 0) + piece.cost
    order = sorted(chain_cost, key=lambda chain: (-chain_cost[chain], chain))
    heap = [(0, worker) for worker in range(workers)]
    heapq.heapify(heap)
    assignment: dict[int, int] = {}
    for chain in order:
        load, worker = heapq.heappop(heap)
        assignment[chain] = worker
        heapq.heappush(heap, (load + chain_cost[chain], worker))
    makespan = max((load for load, _worker in heap), default=0)
    return assignment, int(makespan)


def round_robin_assignment(pieces: Sequence[OracledPiece], workers: int) -> tuple[dict[int, int], int]:
    """@brief 朴素链轮转分配（LPT 的比较基线）。"""
    if workers < 1:
        raise ValueError("workers must be >= 1")
    chains = sorted({piece.chain for piece in pieces})
    assignment = {chain: index % workers for index, chain in enumerate(chains)}
    loads = [0] * workers
    for piece in pieces:
        loads[assignment[piece.chain]] += piece.cost
    return assignment, int(max(loads, default=0))


def emitted_items(pieces: Sequence[OracledPiece], assignment: dict[int, int], workers: int) -> tuple[Piece, ...]:
    """@brief 按 worker 分组后的发射顺序（与 C fixture 的 mode 2 相同）。"""
    out: list[Piece] = []
    for worker in range(workers):
        for piece in pieces:
            if assignment[piece.chain] == worker:
                out.append(piece.as_piece())
    return tuple(out)


def pieces_by_chain(pieces: Iterable[tuple[int, int, int, int, int]]) -> dict[int, list[tuple[int, int, int, int, int]]]:
    """@brief ``(chain, piece, begin, valid, pieces)`` 按链分组。"""
    out: dict[int, list[tuple[int, int, int, int, int]]] = {}
    for item in pieces:
        out.setdefault(item[0], []).append(item)
    return out


def coverage_problems(
    items: Sequence[tuple[int, int, int, int, int]],
    segments: Sequence[int],
    heads: int,
    *,
    max_split: int,
) -> list[str]:
    """@brief 工作项是否恰好覆盖每条非空链（连续、非空、piece 数合法、每项一次）。

    @param items ``(chain, piece, begin, valid, pieces)``。
    @param segments 实际分段。
    @param heads H。
    @param max_split 每链最大 piece 数。
    @return 问题列表；空表示覆盖正确。
    """
    problems: list[str] = []
    grouped = pieces_by_chain(items)
    expected_chains = {
        seq * heads + head
        for seq, length in enumerate(segments)
        if int(length) > 0
        for head in range(heads)
    }
    if set(grouped) != expected_chains:
        problems.append(f"chains {sorted(set(grouped) ^ expected_chains)} differ from the non-empty chains")
        return problems
    seen: set[tuple[int, int]] = set()
    for chain, chain_items in grouped.items():
        seq, _head = divmod(chain, heads)
        begin = sum(int(value) for value in segments[:seq])
        end = begin + int(segments[seq])
        ordered = sorted(chain_items, key=lambda item: item[1])
        pieces = ordered[0][4] if ordered else 0
        if pieces < 1 or pieces > max_split or pieces > int(segments[seq]):
            problems.append(f"chain {chain}: piece count {pieces} is outside [1, {min(max_split, segments[seq])}]")
        if [item[1] for item in ordered] != list(range(len(ordered))):
            problems.append(f"chain {chain}: piece indices are not 0..{len(ordered) - 1}")
        cursor = begin
        for _chain, index, item_begin, valid, item_pieces in ordered:
            if (chain, index) in seen:
                problems.append(f"chain {chain} piece {index} appears more than once")
            seen.add((chain, index))
            if item_pieces != pieces:
                problems.append(f"chain {chain} piece {index}: pieces word {item_pieces} != {pieces}")
            if item_begin != cursor:
                problems.append(f"chain {chain} piece {index}: begin {item_begin} != {cursor}")
            if valid < 1:
                problems.append(f"chain {chain} piece {index}: empty piece")
            cursor += valid
        if cursor != end:
            problems.append(f"chain {chain}: pieces end at {cursor}, segment ends at {end}")
    if len(seen) != len(items):
        problems.append(f"{len(items)} items but {len(seen)} unique (chain, piece) keys")
    return problems


__all__ = [
    "COST_SHIFT",
    "HANDOFF_COST",
    "MAX_WORKERS",
    "OracledPiece",
    "START_COST",
    "canonical_pieces",
    "coverage_problems",
    "emitted_items",
    "lpt_assignment",
    "pieces_by_chain",
    "range_cut_pieces",
    "round_robin_assignment",
]
