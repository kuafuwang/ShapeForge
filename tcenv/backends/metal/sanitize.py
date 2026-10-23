"""Safety checks (the compute-sanitizer analogue).

Three independent mechanisms:
  1. Guard bands + NaN-poisoned outputs: every buffer lives in the middle of a larger
     allocation whose margins are filled with a known pattern; outputs are pre-filled
     with all-ones bits (NaN for every float type), so out-of-bounds writes and
     unwritten outputs are both detected.
  2. Input immutability: inputs must be bit-identical after the launch.
  3. Metal shader validation (``MTL_SHADER_VALIDATION=1``) in a dedicated subprocess,
     which instruments device/threadgroup memory accesses and barrier usage.
"""

# 正确性数值比较之前的安全层，也负责把着色器校验的环境变量送进单独的子进程。
# 保护带：缓冲放在更大分配的中间，两侧填已知图案；图案被改就是越界写。输入两侧用 0xFF，越界读会把 NaN 灌进结果；输出两侧用 0xCD，专门留给出界的写。
# NaN 投毒：输出有效载荷先填满 0xFF。浮点读出来是 NaN，整数是 -1。比较时整元素仍等于该图案，就是没被写过。
# 输入不可变：load_inputs 留下字节快照，启动后 safety_violations 发现字节变了，就是核写了输入。
# 确定性由 testing.run_correctness 比较多次启动：结果正确且保护带完好时，全零初值与毒化初值不同，说明核依赖输出里的旧内容；
# 两者相同而再次毒化仍不同，说明有数据竞争或无序原子。poison() 负责毒化初值。
# 着色器校验按整块 MTLBuffer 的长度算界，保护带和 256 字节对齐都在这块里面，余量中的小越界只有图案被改才抓得到。
# 校验子进程改用 tight 分配（没有这圈余量），边界才和逻辑长度重合。

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .contract import AbiArg, Contract
from ...dtypes import get_dtype

if TYPE_CHECKING:
    from .runtime import BufferArg, MetalContext, ScalarArg

GUARD_BYTES = 16384
GUARD_PATTERN = 0xCD
POISON_PATTERN = 0xFF  # NaN for fp16/bf16/fp32; -1 for ints

# 校验会插桩 device / threadgroup 访存和 barrier。REPORT_TO_STDERR 让父进程从 stderr 看见违例。
VALIDATION_ENV = {
    "MTL_DEBUG_LAYER": "1",
    "MTL_SHADER_VALIDATION": "1",
    "MTL_SHADER_VALIDATION_GLOBAL_MEMORY": "1",
    "MTL_SHADER_VALIDATION_THREADGROUP_MEMORY": "1",
    "MTL_SHADER_VALIDATION_THREADGROUP_SYNC": "1",
    "MTL_SHADER_VALIDATION_TEXTURE_USAGE": "1",
    # without this, violations are not printed anywhere we can see them
    "MTL_SHADER_VALIDATION_REPORT_TO_STDERR": "1",
}

_VALIDATION_MARKERS = re.compile(
    r"(Invalid (device|threadgroup|constant|texture)[a-z ]* (load|store|access|read|write)|executing kernel function|"
    r"failed assertion|Execution of the command buffer was aborted|GPU (Soft|Hard) Fault|Page Fault|Insufficient Permission|"
    r"validation (error|failure)|\brace\b|uninitialized threadgroup)",
    re.IGNORECASE,
)


# 从着色器校验子进程的 stderr 里抽出失败行。校验跑在另一个进程，违例只出现在那里。
def validation_problems(stderr_text: str) -> list[str]:
    """@brief 从校验子进程的 stderr 里抽出失败行。

    @param stderr_text 校验子进程的标准错误文本。
    @return 匹配到的失败行（每条截到 400 字符，最多 12 条）。

    Extract lines of stderr that indicate a validation failure.
    """
    out = []
    for line in stderr_text.splitlines():
        if _VALIDATION_MARKERS.search(line):
            out.append(line.strip()[:400])
    # 一份核最多留 12 条，避免把整份评测报告刷满。
    return out[:12]


def _round_up(n: int, m: int) -> int:
    """@brief 把 ``n`` 向上取整到 ``m`` 的倍数。

    @param n 原始长度。
    @param m 对齐单位。
    @return 对齐后的长度。
    """
    return (n + m - 1) // m * m


# 逻辑缓冲放在更大分配的中间，两侧图案被改写就是越界写。
# tight 时不要两侧：着色器校验按 MTLBuffer 的长度算界，收成正好 nbytes 才能看见越界。
class GuardedBuffer:
    """A buffer embedded in a larger allocation. ``tight=True`` allocates exactly ``nbytes`` with no
    margins so that Metal's shader validation sees every out-of-bounds access."""

    def __init__(self, ctx: MetalContext, nbytes: int, *, guard: int = GUARD_BYTES, tight: bool = False, pattern: int = GUARD_PATTERN) -> None:
        """@brief 在一块带两侧保护带的分配中间放下逻辑缓冲。

        @param ctx Metal 上下文。
        @param nbytes 逻辑有效载荷字节数。
        @param guard 单侧保护带字节数。
        @param tight 是否省掉保护带（着色器校验按分配长度判界）。
        @param pattern 保护带填充图案。
        """
        self.ctx = ctx
        self.nbytes = nbytes
        self.tight = tight
        self.pattern = pattern
        self.guard = 0 if tight else guard
        guard = self.guard
        # 有效长度向上对齐到 256。多出来的尾巴仍算界外，写进对齐空隙也是越界。
        self.padded = nbytes if tight else _round_up(nbytes, 256)
        self.total = guard + self.padded + guard
        self.buf = ctx.new_buffer(self.total)
        self.mem = ctx.view(self.buf)
        self.reset_guards()

    @property
    def arg(self) -> "BufferArg":
        """@brief 返回指向逻辑缓冲起点的 BufferArg。

        @return 带保护带偏移的 BufferArg。
        """
        from .runtime import BufferArg

        return BufferArg(self.buf, self.guard)

    def reset_guards(self) -> None:
        """@brief 把两侧保护带重新填成图案。"""
        if self.guard:
            self.mem[: self.guard] = bytes([self.pattern]) * self.guard
        tail = self.total - self.guard - self.nbytes
        if tail > 0:
            self.mem[self.guard + self.nbytes : self.total] = bytes([self.pattern]) * tail

    def write(self, raw: bytes) -> None:
        """@brief 把输入字节写进逻辑缓冲的有效载荷区。

        @param raw 已序列化的输入字节。
        @raises ValueError 当字节数与声明的有效载荷不符。
        """
        if len(raw) != self.nbytes:
            raise ValueError(f"expected {self.nbytes} bytes, got {len(raw)}")
        self.mem[self.guard : self.guard + self.nbytes] = raw

    # 把输出有效载荷填成全 1。浮点上是 NaN，整数上是 -1，没写过的元素会保持这个图案。
    # 确定性复跑另用全零初值对照；两次不同说明结果依赖输出里的旧内容。比较在 testing.run_correctness。
    def poison(self) -> None:
        """@brief 用全 1 图案毒化输出有效载荷。"""
        self.mem[self.guard : self.guard + self.nbytes] = bytes([POISON_PATTERN]) * self.nbytes

    def read(self) -> bytes:
        """@brief 读出逻辑缓冲的有效载荷字节。

        @return 有效载荷的字节串。
        """
        return bytes(self.mem[self.guard : self.guard + self.nbytes])

    # 头尾图案被改就是越界写。尾段含 256 字节对齐空隙，写进空隙同样算越界。
    def guard_violations(self) -> list[str]:
        """@brief 检查头尾保护带是否被改写。

        @return 越界写问题列表，空列表表示保护带完好。
        """
        out = []
        head = bytes(self.mem[: self.guard])
        tail = bytes(self.mem[self.guard + self.nbytes : self.total])
        if head.count(self.pattern) != len(head):
            first = next(i for i, b in enumerate(head) if b != self.pattern)
            out.append(f"{self.guard - first} bytes before the buffer start were overwritten")
        if tail.count(self.pattern) != len(tail):
            first = next(i for i, b in enumerate(tail) if b != self.pattern)
            out.append(f"write past the end of the buffer (first corrupted byte at +{first} bytes after the end)")
        return out


# 一次启动用的整组参数。输入保护带用 0xFF，越界读会把结果污染成 NaN；输出保护带用 0xCD，越界写和「读到的毒」才能分开。
class ArgSet:
    """A full set of bound arguments for one launch of a contract."""

    def __init__(self, ctx: MetalContext, contract: Contract, *, guard: int = GUARD_BYTES, tight: bool = False, workspaces: tuple = ()) -> None:
        """@brief 为一次启动分配全部契约参数与 workspace 的受保护缓冲。

        @param ctx Metal 上下文。
        @param contract 任务契约。
        @param guard 单侧保护带字节数。
        @param tight 是否使用紧分配。
        @param workspaces 需要额外分配的 workspace 声明。
        """
        self.ctx = ctx
        self.contract = contract
        # Input margins are NaN so that out-of-bounds reads poison the result; output margins use a
        # distinct pattern so that out-of-bounds writes are detected.
        self.buffers: dict[str, GuardedBuffer] = {
            a.name: GuardedBuffer(ctx, a.nbytes, guard=guard, tight=tight, pattern=POISON_PATTERN if a.role == "input" else GUARD_PATTERN)
            for a in contract.buffers
        }
        self.workspace_bufs: dict[str, GuardedBuffer] = {
            w.name: GuardedBuffer(ctx, w.nbytes, guard=guard, tight=tight, pattern=GUARD_PATTERN)
            for w in workspaces
        }
        self.workspace_zero = {w.name: w.zero for w in workspaces}
        self.scalar_bytes: dict[str, bytes] = {a.name: scalar_bytes(a) for a in contract.scalars}
        self._input_snapshot: dict[str, bytes] = {}

    def reset_guards_all(self) -> None:
        """@brief 重置全部契约缓冲与 workspace 的保护带。"""
        for gb in (*self.buffers.values(), *self.workspace_bufs.values()):
            gb.reset_guards()

    def prepare_workspaces(self, *, poison: bool, fill_zero: bool) -> None:
        """@brief 设置 workspace 在正确性启动前的宿主端初值。

        @param poison 是否允许用毒化图案填充。
        @param fill_zero 是否强制清零。

        Host-side initial contents for a correctness launch. Bench zeros inside the timed buffer instead.
        """
        for name, gb in self.workspace_bufs.items():
            if fill_zero or self.workspace_zero.get(name) or not poison:
                gb.mem[gb.guard : gb.guard + gb.nbytes] = bytes(gb.nbytes)
            else:
                gb.poison()

    def stage_args(self, kernel) -> list[tuple[int, "BufferArg | ScalarArg"]]:
        """@brief 生成某一段的绑定列表。

        @param kernel 该段已编译的核。
        @return 按槽位排列的 (索引, BufferArg 或 ScalarArg) 列表。

        Bindings for one compiled stage. Names select contract arguments or this set's workspaces.
        """
        from .runtime import BufferArg, ScalarArg

        out: list[tuple[int, BufferArg | ScalarArg]] = []
        for b in kernel.bindings:
            if b.kind != "buffer":
                continue
            arg = next((a for a in self.contract.args if a.name == b.name), None)
            if arg is not None and arg.role == "scalar":
                out.append((b.index, ScalarArg(self.scalar_bytes[b.name])))
            elif b.name in self.buffers:
                out.append((b.index, self.buffers[b.name].arg))
            elif b.name in self.workspace_bufs:
                out.append((b.index, self.workspace_bufs[b.name].arg))
        return out

    def workspace_fills(self) -> list[tuple]:
        """@brief 列出计时窗内需要清零的 workspace 区域。

        @return (buf, offset, nbytes) 元组列表。
        """
        return [(gb.buf, gb.guard, gb.nbytes) for gb in self.workspace_bufs.values()]

    # 留下本次写入的原始字节。启动后 safety_violations 用这份快照判断输入有没有被核改掉。
    def load_inputs(self, raw_by_name: dict[str, bytes]) -> None:
        """@brief 写入输入字节并留下快照，供启动后判断输入是否被改写。

        @param raw_by_name 输入名到原始字节的映射。
        """
        for a in self.contract.inputs:
            self.buffers[a.name].write(raw_by_name[a.name])
        self._input_snapshot = dict(raw_by_name)

    def poison_outputs(self) -> None:
        """@brief 毒化全部输出有效载荷。"""
        for a in self.contract.outputs:
            self.buffers[a.name].poison()

    def args(self) -> list[tuple[int, "BufferArg | ScalarArg"]]:
        """@brief 按契约顺序生成完整绑定列表。

        @return (索引, BufferArg 或 ScalarArg) 列表。
        """
        from .runtime import BufferArg, ScalarArg

        out: list[tuple[int, BufferArg | ScalarArg]] = []
        for a in self.contract.args:
            if a.role == "scalar":
                out.append((a.index, ScalarArg(self.scalar_bytes[a.name])))
            else:
                out.append((a.index, self.buffers[a.name].arg))
        return out

    def read_outputs(self) -> dict[str, bytes]:
        """@brief 读出全部输出的有效载荷字节。

        @return 输出名到字节串的映射。
        """
        return {a.name: self.buffers[a.name].read() for a in self.contract.outputs}

    # 两类错误：保护带图案破损是越界写；输入字节和快照不同是核改了输入。
    def safety_violations(self) -> list[str]:
        """@brief 汇总越界写与输入被改写两类安全问题。

        @return 问题列表，空列表表示安全层通过。
        """
        out: list[str] = []
        for name, gb in (*self.buffers.items(), *self.workspace_bufs.items()):
            for v in gb.guard_violations():
                out.append(f"buffer `{name}`: {v}")
        for a in self.contract.inputs:
            if self._input_snapshot and self.buffers[a.name].read() != self._input_snapshot[a.name]:
                out.append(f"input `{a.name}` was modified by the kernel")
        return out


# 标量按契约宽度收成小端字节，GPU 上的 constant 引用按这个布局读。
def scalar_bytes(a: AbiArg) -> bytes:
    """@brief 按契约宽度把标量收成小端字节。

    @param a 契约里的标量参数。
    @return 供 setBytes 使用的字节串。
    """
    import struct

    d = get_dtype(a.dtype)
    if d.name == "fp32":
        return struct.pack("<f", float(a.value))
    if d.name == "fp16":
        return struct.pack("<e", float(a.value))
    if d.name == "bf16":
        # struct 没有 bfloat16，用 torch 取出同样的 16 位再交给 setBytes。
        import torch

        return torch.tensor([float(a.value)], dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    fmt = {"i32": "<i", "u32": "<I", "i8": "<b", "u8": "<B"}[d.name]
    return struct.pack(fmt, int(a.value))


# 复制一份环境再叠上校验开关。这些变量是进程级的，只交给单独再起的校验子进程。
def env_with_validation(base: dict[str, str] | None = None) -> dict[str, str]:
    """@brief 复制一份环境变量并叠加着色器校验开关。

    @param base 基础环境；省略时用当前进程环境。
    @return 打开校验开关后的新字典。
    """
    env = dict(os.environ if base is None else base)
    env.update(VALIDATION_ENV)
    return env
