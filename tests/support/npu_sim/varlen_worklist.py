"""Test-only NPU simulator of the backend-neutral varlen work list and state protocol.

This module is **not** an NPU backend and never executes MSL.  It answers a narrower question:
does the candidate-owned work list / handoff protocol have the properties the plan demands when it
is mapped onto cores with real constraints?

* every work item is claimed by exactly one core;
* the dependency graph (piece p of a chain consumes the state produced by piece p-1) is the same
  graph the Metal fixture relies on;
* a core holds one item at a time, so a *consumer-first* launch order can deadlock when the free
  core slots are all held by consumers whose producers have not started.  This is a **host-side
  scheduling** argument about a finite pool of workers that spin on a producer; it is never offered
  as Metal safety evidence — the Metal Dyn-0 fixtures use serial stage encoders with a buffer
  barrier and never spin on another threadgroup's flag;
* a partial-tail writeback is simulated at the target's DMA alignment.  A granule that covers bytes
  of the neighbouring TND sequence is a clobber unless the candidate declares a masked write /
  read-modify-write policy;
* the per-core scratch (UB/L1) budget and the item-table location (small table cached in scratch,
  large table read from global) are modelled explicitly.

Every report carries ``simulated: true`` and ``publication_allowed: false``; a simulated report is
not a device measurement and can never enter confirmation, publication, catalog or brief.  That holds
for *every* document this module can hand out, not only for the ``simulate()`` wrapper: an
``AdapterReport.document()`` (including the Metal-side adapter, which is a transliteration of the
device plan stage and never a device run) carries the same fixed markers, so a caller cannot get an
unmarked work-list document by skipping the wrapper.

The work list itself is the *same* semantic plan the Metal Dyn-0 fixtures build on device: a chain
is cut into ``P = min(max_split, len)`` **non-empty** pieces (``base, rem = divmod(len, P)``; the
first ``rem`` pieces get ``base + 1`` tokens).  V2 has no host program yet, so parity is drawn between
two independent adapter paths rather than two Wasm modules:

* ``metal_adapter``  — a transliteration of the device plan stage (``piece_handoff.metal``) plus the
  Dyn-0 static launch model: one item per threadgroup slot, idle slots exit on the runtime count.
* ``npu_adapter``    — a host-neutral planner written differently (prefix sums + accumulated piece
  sizes, then canonical ``(chain, piece)`` order), mapped onto a finite number of cores.

``parity_problems`` compares work-list bytes and dependency edges and checks on **both** sides that
every item is claimed exactly once.  Core assignment and launch are target-specific and are never
compared.  The GPU test additionally compares the *real device* work list that
``piece_handoff.metal`` writes with ``npu_adapter``'s bytes, so the transliteration is not the only
witness of the device layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import accumulate
import hashlib
import struct
from typing import Iterable, Sequence

#: Fixed report flags.  A simulated report is never publication evidence.
SIMULATED = True
PUBLICATION_ALLOWED = False
MEASUREMENT_KIND = "simulated"
EVIDENCE_SCOPE = "host-plan/work-list/state-protocol only"

#: Scheduler orders an adversarial launch can use.
ORDERS = ("stage", "reverse_stage", "consumer_first")


def evidence_markers() -> dict:
    """@brief 任何模拟输出的固定标记（单一来源，避免多条报告路径漂移）。

    ``simulate()`` 与 ``AdapterReport.document()`` 都从这里取标记：一处漏写就会产出一份没有
    ``simulated``/``publication_allowed`` 的“看起来像证据”的文档，所以不允许两处各自拼一份。

    @return 固定标记字典。
    """
    return {
        "simulated": SIMULATED,
        "publication_allowed": PUBLICATION_ALLOWED,
        "measurement_kind": MEASUREMENT_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
    }


@dataclass(frozen=True)
class SimTarget:
    """@brief 一个模拟目标：core 数、每 core scratch、DMA 对齐与 item-table 容量。"""

    id: str
    kind: str
    cores: int
    scratch_bytes: int
    dma_alignment: int
    item_table_capacity: int
    masked_write: bool = False

    def __post_init__(self) -> None:
        """@brief 校验目标参数。

        @raises ValueError 非法。
        """
        if self.kind not in ("metal", "npu"):
            raise ValueError(f"target {self.id!r}: kind must be metal or npu, got {self.kind!r}")
        for name in ("cores", "scratch_bytes", "dma_alignment", "item_table_capacity"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"target {self.id!r}: {name} must be an integer >= 1, got {value!r}")
        if self.dma_alignment & (self.dma_alignment - 1):
            raise ValueError(f"target {self.id!r}: dma_alignment must be a power of two")

    def document(self) -> dict:
        """@brief 目标的 JSON 文档。

        @return 字段字典。
        """
        return {
            "id": self.id,
            "kind": self.kind,
            "cores": self.cores,
            "scratch_bytes": self.scratch_bytes,
            "dma_alignment": self.dma_alignment,
            "item_table_capacity": self.item_table_capacity,
            "masked_write": self.masked_write,
        }


@dataclass(frozen=True)
class Piece:
    """@brief 规划出的一个 piece（工作项）；与 Metal fixture 的两字打包一一对应。"""

    chain: int
    piece: int
    begin: int
    valid: int
    pieces: int

    @property
    def last(self) -> bool:
        """@brief 是否是该链的末 piece。

        @return 布尔值。
        """
        return self.piece + 1 == self.pieces

    def words(self) -> tuple[int, int]:
        """@brief 与 Metal fixture 相同的两字打包（候选自有布局）。

        @return ``(word0, word1)``。
        """
        if self.begin >= 1 << 16 or self.valid >= 1 << 16:
            raise ValueError(f"piece {self.chain}:{self.piece} does not fit the fixture's 16-bit token packing")
        if self.chain >= 1 << 8 or self.pieces >= 1 << 8:
            raise ValueError(f"chain {self.chain} does not fit the fixture's 8-bit chain/piece packing")
        return (self.begin | (self.valid << 16), self.chain | (self.piece << 8) | (self.pieces << 16))


def plan_pieces(segments: Sequence[int], heads: int, max_split: int) -> tuple[Piece, ...]:
    """@brief 由实际分段规划 piece（与 OpSpec/ fixture 相同的均衡切分规则）。

    A chain is one ``(sequence, head)``; a segment of length ``len`` is cut into
    ``P = min(max_split, len)`` **non-empty** pieces: ``base, rem = divmod(len, P)`` and the first
    ``rem`` pieces get ``base + 1`` tokens, the rest ``base``.  Because ``P <= len`` every piece has
    at least one token, so a piece never has ``valid == 0`` (an empty writeback has no DMA span and
    must not be judged as clobbering a neighbour).  An empty segment has no piece at all: its
    terminal state is its initial state, handled by the planner stage.

    Items are emitted in canonical ``(chain, piece)`` order, the order the device plan stage writes
    them in; a parity check compares the bytes, so the order is part of the contract of this helper.

    @param segments 每条序列的实际 token 数。
    @param heads H。
    @param max_split 每链最大 piece 数（OpSpec 的 ``MAX_SPLIT_PER_CHAIN``）。
    @return piece 元组（按 chain、piece 排序）。
    """
    if heads < 1 or max_split < 1:
        raise ValueError("heads and max_split must be >= 1")
    if any(int(value) < 0 for value in segments):
        raise ValueError("segment lengths must be >= 0")
    lengths = [int(value) for value in segments]
    out: list[Piece] = []
    for chain in range(len(lengths) * heads):
        seq, head = divmod(chain, heads)
        length = lengths[seq]
        if length == 0:
            continue
        pieces = min(max_split, length)
        base, rem = divmod(length, pieces)
        cursor = sum(lengths[:seq])
        for piece in range(pieces):
            size = base + (1 if piece < rem else 0)
            out.append(Piece(chain=chain, piece=piece, begin=cursor, valid=size, pieces=pieces))
            cursor += size
    return tuple(out)


def pack_items(items: Iterable[Piece]) -> bytes:
    """@brief 把工作项压成与设备一致的字节（parity 的比较对象）。

    @param items piece 序列。
    @return 小端两字/项的字节串。
    """
    return b"".join(struct.pack("<II", *item.words()) for item in items)


def work_list_digest(items: Iterable[Piece]) -> str:
    """@brief 工作项字节的 sha256。

    @param items piece 序列。
    @return 十六进制摘要。
    """
    return hashlib.sha256(pack_items(items)).hexdigest()


def dependency_edges(items: Sequence[Piece]) -> tuple[tuple[int, int], ...]:
    """@brief handoff 依赖边 ``(producer, consumer)``（同一链的相邻 piece）。

    @param items piece 序列。
    @return 边元组（按 (producer, consumer) 排序）。
    """
    index = {(item.chain, item.piece): position for position, item in enumerate(items)}
    edges = []
    for (chain, piece), producer in index.items():
        consumer = index.get((chain, piece + 1))
        if consumer is not None:
            edges.append((producer, consumer))
    return tuple(sorted(edges))


def state_sources(items: Sequence[Piece]) -> tuple[dict, ...]:
    """@brief 每个工作项的状态来源与去向（首/中/末 piece 的协议角色）。

    @param items piece 序列。
    @return 每个 item 的角色记录。
    """
    return tuple(
        {
            "chain": item.chain,
            "piece": item.piece,
            "state_source": "initial_state" if item.piece == 0 else "handoff",
            "state_sink": "final_state" if item.last else "handoff",
        }
        for item in items
    )


def empty_chains(segments: Sequence[int], heads: int) -> tuple[int, ...]:
    """@brief 没有 piece 的链（长度为 0 的序列）：终态必须等于初态，不产生 handoff。

    @param segments 实际分段。
    @param heads H。
    @return 链编号元组。
    """
    return tuple(
        seq * heads + head
        for seq, length in enumerate(segments)
        if int(length) == 0
        for head in range(heads)
    )


def claim_problems(items: Sequence[Piece], claims: Iterable[tuple[int, tuple[int, ...]]]) -> list[str]:
    """@brief 检查“每项恰好被一个 core/TG 槽位认领”（parity 的两侧都要过这一关）。

    Empty claim lists are allowed: an idle Metal threadgroup or an unused core claims nothing and
    exits on the runtime count.  Slot numbers must be unique so a schedule cannot hide a double
    booking behind two slots with the same id.

    @param items piece 序列。
    @param claims ``(槽位, 该槽位认领的 item 下标)`` 序列（可含空槽位）。
    @return 问题列表；空表示通过。
    """
    claims = tuple(claims)
    problems: list[str] = []
    slots = [slot for slot, _ in claims]
    if len(slots) != len(set(slots)):
        problems.append("a core slot is listed twice")
    claimed = [item for _, cases in claims for item in cases]
    if len(claimed) != len(set(claimed)):
        problems.append("an item was claimed by more than one core")
    missing = sorted(set(range(len(items))) - set(claimed))
    if missing:
        problems.append(f"items never claimed: {missing}")
    unknown = sorted(set(claimed) - set(range(len(items))))
    if unknown:
        problems.append(f"claims point outside the item list: {unknown}")
    return problems


@dataclass(frozen=True)
class Schedule:
    """@brief 一次模拟调度的结果（核心认领、事件、等待与死锁）。"""

    order: str
    cores: int
    claims: tuple[tuple[int, tuple[int, ...]], ...]
    events: tuple[dict, ...]
    waits: int
    deadlock: bool
    completed: tuple[int, ...]

    def document(self) -> dict:
        """@brief 调度结果的 JSON 文档（同样带固定模拟标记）。

        ``simulate_schedule()`` 返回的 ``Schedule`` 可以被调用者直接取走，所以这份文档本身就是
        一份可独立引用的模拟结果，不是“只能挂在 ``simulate()`` 里的小片段”：它同样从
        ``evidence_markers()`` 取 ``simulated``/``publication_allowed``。靠注释声明“请勿独立使用”
        拦不住调用方，标记得写在输出里。

        @return 字段字典。
        """
        return {
            **evidence_markers(),
            "order": self.order,
            "cores": self.cores,
            "claims": [{"core": core, "items": list(items)} for core, items in self.claims],
            "events": [dict(event) for event in self.events],
            "waits": self.waits,
            "deadlock": self.deadlock,
            "completed": list(self.completed),
        }


def _preference(items: Sequence[Piece], order: str) -> list[int]:
    """@brief 调度偏好顺序（返回 item 下标）。

    ``reverse_stage`` keeps every stage together and only reverses the chain order *inside* a
    stage (a producer-order injection that is still topological).  ``consumer_first`` puts the
    deepest pieces first globally, so with few core slots the consumers occupy every slot while
    their producers are still pending.

    @param items piece 序列。
    @param order ``stage``/``reverse_stage``/``consumer_first``。
    @return 下标列表。
    """
    indices = list(range(len(items)))
    if order == "stage":
        return sorted(indices, key=lambda i: (items[i].piece, items[i].chain))
    if order == "reverse_stage":
        return sorted(indices, key=lambda i: (items[i].piece, -items[i].chain))
    if order == "consumer_first":
        return sorted(indices, key=lambda i: (-items[i].piece, items[i].chain))
    raise ValueError(f"unknown schedule order {order!r}; registered orders are {list(ORDERS)}")


def simulate_schedule(items: Sequence[Piece], target: SimTarget, *, order: str) -> Schedule:
    """@brief 在有限 core 槽位上模拟一次调度，检测 consumer-first 等待环。

    A core holds at most one item.  A started item keeps its core until all of its producers have
    finished, exactly like a spinning consumer on a device where forward progress is not guaranteed
    across cores.  If a full round neither finishes an item nor starts one, the schedule
    deadlocked: every core slot is held by a consumer whose producer can never be scheduled.

    @param items piece 序列。
    @param target 模拟目标（core 数）。
    @param order 调度顺序。
    @return 调度结果。
    """
    deps: dict[int, set[int]] = {index: set() for index in range(len(items))}
    for producer, consumer in dependency_edges(items):
        deps[consumer].add(producer)
    pending = list(_preference(items, order))
    started: set[int] = set()
    done: set[int] = set()
    occupied: dict[int, int] = {}
    claims: dict[int, list[int]] = {core: [] for core in range(target.cores)}
    events: list[dict] = []
    waits = 0
    deadlock = False
    while len(done) < len(items):
        progressed = False
        for core in sorted(occupied):
            item = occupied[core]
            if deps[item] <= done:
                del occupied[core]
                done.add(item)
                events.append({"kind": "finish", "core": core, "item": item})
                progressed = True
        for core in range(target.cores):
            if core in occupied or not pending:
                continue
            item = pending.pop(0)
            started.add(item)
            claims[core].append(item)
            occupied[core] = item
            ready = deps[item] <= done
            events.append({"kind": "start", "core": core, "item": item, "target": items[item].chain * 100 + items[item].piece})
            if not ready:
                waits += 1
                events.append({"kind": "wait", "core": core, "item": item})
            progressed = True
        if not progressed:
            deadlock = True
            break
    return Schedule(
        order=order,
        cores=target.cores,
        claims=tuple((core, tuple(items_list)) for core, items_list in claims.items()),
        events=tuple(events),
        waits=waits,
        deadlock=deadlock,
        completed=tuple(sorted(done)),
    )


def scratch_plan(items: Sequence[Piece], target: SimTarget, *, dim: int, elem_bytes: int = 4) -> dict:
    """@brief 每 core scratch 预算与 item-table 位置（UB/L1 缓存 vs global fallback）。

    @param items piece 序列。
    @param target 模拟目标。
    @param dim D（piece 局部 ``D*D`` 状态）。
    @param elem_bytes 状态元素字节数。
    @return scratch 计划与缺陷。
    """
    state_bytes = dim * dim * elem_bytes
    table_bytes = len(items) * 8  # two u32 words per item
    cached = len(items) <= target.item_table_capacity
    peak = state_bytes + (table_bytes if cached else 0)
    return {
        "state_bytes": state_bytes,
        "item_table_bytes": table_bytes,
        "item_table_location": "scratch" if cached else "global",
        "peak_bytes": peak,
        "capacity_bytes": target.scratch_bytes,
        "fits": peak <= target.scratch_bytes,
    }


def tnd_ranges(segments: Sequence[int], *, heads: int, dim: int, elem_bytes: int) -> tuple[tuple[int, int], ...]:
    """@brief 输出缓冲里每条序列的字节区间（TND 打包布局）。

    @param segments 每条序列 token 数。
    @param heads H。
    @param dim D。
    @param elem_bytes 输出元素字节数。
    @return 每条序列的 ``(start, end)`` 字节区间。
    """
    row = heads * dim * elem_bytes
    out = []
    begin = 0
    for length in segments:
        out.append((begin * row, (begin + int(length)) * row))
        begin += int(length)
    return tuple(out)


def alignment_clobbers(
    items: Sequence[Piece],
    segments: Sequence[int],
    *,
    heads: int,
    dim: int,
    elem_bytes: int,
    alignment: int,
    masked_write: bool,
) -> list[dict]:
    """@brief 按 DMA 对齐粒度检查 partial tail 写回是否踩到下一条 TND 序列。

    A DMA writeback covers a whole aligned granule; a piece that ends mid-sequence therefore writes
    bytes belonging to the next TND sequence unless the candidate declares a masked write or a
    read-modify-write strategy.  GPU-style per-thread predication is not assumed here.

    A piece with ``valid == 0`` performs **no writeback at all**: it has no DMA span, so an aligned
    granule cannot be attributed to it.  The planner never emits such a piece (it cuts a chain into
    non-empty pieces), but a candidate/planner bug that does must not be misreported as a clobber.

    @param items piece 序列。
    @param segments 实际分段。
    @param heads H。
    @param dim D。
    @param elem_bytes 输出元素字节数。
    @param alignment DMA 对齐粒度（字节）。
    @param masked_write 候选是否声明 masked write/RMW。
    @return 踩踏记录；空表示没有跨序列字节被覆盖。
    """
    ranges = tnd_ranges(segments, heads=heads, dim=dim, elem_bytes=elem_bytes)
    out: list[dict] = []
    for item in items:
        if item.valid == 0:
            # Nothing is written, so nothing can be clobbered.
            continue
        chain_seq = item.chain // heads
        row = heads * dim * elem_bytes
        start = item.begin * row
        end = (item.begin + item.valid) * row
        span_start = (start // alignment) * alignment
        span_end = -(-end // alignment) * alignment
        for other, (other_start, other_end) in enumerate(ranges):
            if other == chain_seq or other_start == other_end:
                continue
            overlap = max(0, min(span_end, other_end) - max(span_start, other_start))
            if overlap and not masked_write:
                out.append(
                    {
                        "chain": item.chain,
                        "sequence": chain_seq,
                        "piece": item.piece,
                        "dma_span": [span_start, span_end],
                        "clobbers_sequence": other,
                        "clobber_bytes": overlap,
                    }
                )
    return out


@dataclass(frozen=True)
class AdapterReport:
    """@brief 一个后端风格 adapter 的 work list、认领与 target 特有约束结果。

    Parity only looks at ``words``/``items``/``edges`` and at each side's claims.  ``launch`` (grid,
    cores, waves) and core assignment are target-specific: they are reported for evidence but are
    never required to match.
    """

    name: str
    target: SimTarget
    items: tuple[Piece, ...]
    words: bytes
    edges: tuple[tuple[int, int], ...]
    claims: tuple[tuple[int, tuple[int, ...]], ...]
    launch: dict
    defects: tuple[str, ...] = ()
    schedule: Schedule | None = None
    scratch: dict | None = None
    clobbers: tuple[dict, ...] = ()
    empty_writebacks: tuple[dict, ...] = ()

    def claim_problems(self) -> list[str]:
        """@brief 本 adapter 侧“每项恰好一次认领”的问题。

        @return 问题列表；空表示通过。
        """
        return claim_problems(self.items, self.claims)

    def document(self) -> dict:
        """@brief adapter 结果的 JSON 文档（带固定模拟标记）。

        ``metal_adapter`` 也只是设备行为的 Python 转写，不是设备执行，所以两条 adapter 报告都从
        ``evidence_markers()`` 取同一组固定标记。缺这两个键的输出形态不能进入证据链。

        @return 字段字典。
        """
        return {
            **evidence_markers(),
            "adapter": self.name,
            "target": self.target.document(),
            "launch": dict(self.launch),
            "work_list": {
                "digest": work_list_digest(self.items),
                "count": len(self.items),
                "edges": [list(edge) for edge in self.edges],
                "claims": [{"slot": slot, "items": list(cases)} for slot, cases in self.claims],
            },
            "defects": list(self.defects),
        }


def metal_device_plan(segments: Sequence[int], heads: int, max_split: int) -> tuple[Piece, ...]:
    """@brief ``piece_handoff.metal`` 规划 stage 的逐行转写（Metal 侧 adapter 的 work list 来源）。

    The device builds the list with one running cursor per chain; the head does not participate in
    the split and is implied by ``chain``.  ``GPU`` 侧的真实字节由 ``test_the_device_work_list...``
    与 ``npu_adapter`` 差分，所以这份转写不是设备布局的唯一证据。

    @param segments 实际分段。
    @param heads H。
    @param max_split 每链最大 piece 数。
    @return piece 元组（按 chain、piece 排序）。
    """
    lengths = [int(value) for value in segments]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    out: list[Piece] = []
    for chain in range(len(lengths) * heads):
        seq = chain // heads
        length = lengths[seq]
        if length == 0:
            continue
        pieces = min(max_split, length)
        base, rem = divmod(length, pieces)
        cursor = prefix[seq]
        for piece in range(pieces):
            size = base + (1 if piece < rem else 0)
            out.append(Piece(chain=chain, piece=piece, begin=cursor, valid=size, pieces=pieces))
            cursor += size
    return tuple(out)


def npu_host_plan(segments: Sequence[int], heads: int, max_split: int) -> tuple[Piece, ...]:
    """@brief host-neutral 规划：前缀和 + 累加 piece 大小，按 (seq, piece) 展开后归位排序。

    与 ``metal_device_plan``/``plan_pieces`` 是同一规则的另一份实现（不同的算术路径与展开顺序），
    这样 parity 比较的是两个 adapter，而不是同一个函数换标签。

    @param segments 实际分段。
    @param heads H。
    @param max_split 每链最大 piece 数。
    @return piece 元组（按 chain、piece 排序）。
    """
    lengths = [int(value) for value in segments]
    prefix = [0]
    for length in lengths:
        prefix.append(prefix[-1] + length)
    out: list[Piece] = []
    for seq, length in enumerate(lengths):
        if length == 0:
            continue
        pieces = min(max_split, length)
        base, rem = divmod(length, pieces)
        sizes = [base + (1 if piece < rem else 0) for piece in range(pieces)]
        starts = list(accumulate(sizes, initial=prefix[seq]))[:-1]
        for piece, start in enumerate(starts):
            for head in range(heads):
                out.append(
                    Piece(
                        chain=seq * heads + head,
                        piece=piece,
                        begin=start,
                        valid=sizes[piece],
                        pieces=pieces,
                    )
                )
    out.sort(key=lambda item: (item.chain, item.piece))
    return tuple(out)


def metal_adapter(segments: Sequence[int], *, heads: int, max_split: int, target: SimTarget) -> AdapterReport:
    """@brief Metal 风格 adapter：device 规划 + Dyn-0 静态 grid，一项一个 TG，空闲槽位读 count 退出。

    @param segments 实际分段。
    @param heads H。
    @param max_split 每链最大 piece 数。
    @param target 目标约束（这里只用于记录；Metal 认领不依赖 core 数）。
    @return adapter 结果。
    """
    items = metal_device_plan(segments, heads, max_split)
    # Dyn-0 geometry: the grid is the *ABI capacity* (one threadgroup per possible piece of every
    # chain), not the runtime item count.  Extra threadgroups read item_counts[0] and exit, so a
    # chain with fewer than max_split pieces leaves its later slots idle.
    grid = len(segments) * heads * max_split
    if len(items) > grid:
        raise ValueError(f"{len(items)} items do not fit the static grid of {grid} threadgroups")
    claims = tuple((slot, (slot,) if slot < len(items) else ()) for slot in range(grid))
    return AdapterReport(
        name="metal",
        target=target,
        items=items,
        words=pack_items(items),
        edges=dependency_edges(items),
        claims=claims,
        launch={
            "kind": "metal",
            # Which fixture's launch geometry this document describes.  ``piece_handoff.metal`` declares
            # all four stages as ``threads=1,1,1`` (plan + piece0/1/2); the 32-thread launch belongs to
            # fixture A's ``varlen_chain`` and must not leak into this adapter.
            "fixture": "piece_handoff.metal",
            "grid": grid,
            "threads_per_threadgroup": 1,
            "runtime_items": len(items),
            "idle_slots": grid - len(items),
            "stages": 4,
        },
    )


def npu_adapter(
    segments: Sequence[int],
    *,
    heads: int,
    max_split: int,
    dim: int,
    target: SimTarget,
    order: str = "stage",
    elem_bytes: int = 2,
) -> AdapterReport:
    """@brief NPU 风格 adapter：host-neutral 规划 + 有限 core 槽位调度 + UB/L1 与 DMA 约束。

    @param segments 实际分段。
    @param heads H。
    @param max_split 每链最大 piece 数。
    @param dim D。
    @param target 模拟目标。
    @param order 调度顺序。
    @param elem_bytes 输出元素字节数（fp16 为 2）。
    @return adapter 结果。
    """
    items = npu_host_plan(segments, heads, max_split)
    schedule = simulate_schedule(items, target, order=order)
    scratch = scratch_plan(items, target, dim=dim)
    clobbers = tuple(
        alignment_clobbers(
            items,
            segments,
            heads=heads,
            dim=dim,
            elem_bytes=elem_bytes,
            alignment=target.dma_alignment,
            masked_write=target.masked_write,
        )
    )
    empty_writebacks = tuple({"chain": item.chain, "piece": item.piece} for item in items if item.valid == 0)
    defects: list[str] = []
    if schedule.deadlock:
        defects.append(f"schedule order {order!r} deadlocked with {target.cores} core slots")
    if not scratch["fits"]:
        defects.append(
            f"per-core scratch needs {scratch['peak_bytes']} B, target {target.id} has {scratch['capacity_bytes']} B"
        )
    if clobbers:
        defects.append(
            f"{len(clobbers)} partial-tail writeback(s) cross a TND sequence boundary at "
            f"{target.dma_alignment} B DMA alignment and no masked write is declared"
        )
    return AdapterReport(
        name="npu",
        target=target,
        items=items,
        words=pack_items(items),
        edges=dependency_edges(items),
        claims=schedule.claims,
        launch={
            "kind": "npu",
            "cores": target.cores,
            "waves": -(-len(items) // target.cores) if items else 0,
        },
        defects=tuple(defects),
        schedule=schedule,
        scratch=scratch,
        clobbers=clobbers,
        empty_writebacks=empty_writebacks,
    )


def parity_problems(left: AdapterReport, right: AdapterReport) -> list[str]:
    """@brief 两个 adapter 的 parity 检查（work list/依赖边/两侧唯一认领）。

    Compared: work-list **bytes**, dependency edges, and "every item claimed exactly once" on each
    side.  Not compared: core assignment, ``launch``/grid, target constraints and their defect
    profiles — a coarser DMA granule may legitimately add defects on one side only.

    @param left 一侧 adapter。
    @param right 另一侧 adapter。
    @return 问题列表；空表示通过。
    """
    problems: list[str] = []
    if left.words != right.words:
        problems.append(f"work list bytes differ ({left.name} vs {right.name})")
    if left.edges != right.edges:
        problems.append(f"dependency edges differ ({left.name} vs {right.name})")
    problems.extend(f"{left.name}: {problem}" for problem in left.claim_problems())
    problems.extend(f"{right.name}: {problem}" for problem in right.claim_problems())
    return problems


def unpack_items(raw: bytes, count: int) -> tuple[Piece, ...]:
    """@brief 把设备写出的两字工作项解码回 ``Piece``（GPU 差分测试的 Metal 侧输入）。

    @param raw 工作项字节（小端）。
    @param count 有效项数（``item_counts[0]``）。
    @return piece 元组。
    """
    if count < 0 or len(raw) < 8 * count:
        raise ValueError(f"{len(raw)} bytes cannot hold {count} work items")
    out: list[Piece] = []
    for index in range(count):
        word0, word1 = struct.unpack_from("<II", raw, 8 * index)
        out.append(
            Piece(
                chain=word1 & 0xFF,
                piece=(word1 >> 8) & 0xFF,
                pieces=(word1 >> 16) & 0xFF,
                begin=word0 & 0xFFFF,
                valid=word0 >> 16,
            )
        )
    return tuple(out)


def simulate(
    segments: Sequence[int],
    *,
    heads: int,
    dim: int,
    max_split: int,
    target: SimTarget,
    order: str = "stage",
    elem_bytes: int = 2,
) -> dict:
    """@brief 一次完整的 NPU 侧模拟：工作项、调度、scratch、对齐踩踏与报告标记。

    @param segments 实际分段（来自同一 OpSpec case 的 ``cu_seqlens``）。
    @param heads H。
    @param dim D。
    @param max_split 每链最大 piece 数。
    @param target 模拟目标。
    @param order 调度顺序。
    @param elem_bytes 输出元素字节数（fp16 为 2）。
    @return 报告文档（``simulated=true``/``publication_allowed=false``）。
    """
    adapter = npu_adapter(
        segments,
        heads=heads,
        max_split=max_split,
        dim=dim,
        target=target,
        order=order,
        elem_bytes=elem_bytes,
    )
    items = adapter.items
    schedule = adapter.schedule
    scratch = adapter.scratch
    clobbers = list(adapter.clobbers)
    empty_writebacks = [dict(entry) for entry in adapter.empty_writebacks]
    defects = list(adapter.defects)
    assert schedule is not None and scratch is not None
    return {
        **evidence_markers(),
        "target": target.document(),
        "launch": dict(adapter.launch),
        "work_list": {
            "digest": work_list_digest(items),
            "count": len(items),
            "claims": [{"slot": slot, "items": list(cases)} for slot, cases in adapter.claims],
            "empty_chains": list(empty_chains(segments, heads)),
            "items": [
                {
                    "chain": item.chain,
                    "piece": item.piece,
                    "begin": item.begin,
                    "valid": item.valid,
                    "pieces": item.pieces,
                }
                for item in items
            ],
            "edges": [list(edge) for edge in dependency_edges(items)],
            "state_roles": [dict(role) for role in state_sources(items)],
        },
        "schedule": schedule.document(),
        "scratch": scratch,
        "alignment": {
            "dma_alignment": target.dma_alignment,
            "masked_write": target.masked_write,
            "empty_writebacks": empty_writebacks,
            "clobbers": clobbers,
        },
        "defects": defects,
        "ok": not defects,
    }


#: Metal-like stub used for parity: one item per "threadgroup", a small threadgroup table cache and
#: byte-addressable (predicated) stores, so DMA alignment never clobbers a neighbour.
METAL_STUB = SimTarget(
    id="sim-metal-stub",
    kind="metal",
    cores=64,
    scratch_bytes=32768,
    dma_alignment=4,
    item_table_capacity=64,
)

#: NPU-like stub: fewer cores, an L1-sized scratch and a coarse DMA writeback granule.
NPU_STUB = SimTarget(
    id="sim-npu-stub",
    kind="npu",
    cores=20,
    scratch_bytes=65536,
    dma_alignment=512,
    item_table_capacity=64,
)


__all__ = [
    "EVIDENCE_SCOPE",
    "MEASUREMENT_KIND",
    "METAL_STUB",
    "NPU_STUB",
    "ORDERS",
    "PUBLICATION_ALLOWED",
    "SIMULATED",
    "AdapterReport",
    "Piece",
    "Schedule",
    "SimTarget",
    "alignment_clobbers",
    "claim_problems",
    "dependency_edges",
    "empty_chains",
    "evidence_markers",
    "metal_adapter",
    "metal_device_plan",
    "npu_adapter",
    "npu_host_plan",
    "pack_items",
    "parity_problems",
    "plan_pieces",
    "simulate",
    "simulate_schedule",
    "scratch_plan",
    "state_sources",
    "tnd_ranges",
    "unpack_items",
    "work_list_digest",
]
