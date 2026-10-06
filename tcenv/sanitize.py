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

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .contract import AbiArg, Contract
from .metal_runtime import BufferArg, MetalContext, ScalarArg
from .dtypes import get_dtype

GUARD_BYTES = 16384
GUARD_PATTERN = 0xCD
POISON_PATTERN = 0xFF  # NaN for fp16/bf16/fp32; -1 for ints

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
    r"validation (error|failure)|race|uninitialized threadgroup)",
    re.IGNORECASE,
)


def validation_problems(stderr_text: str) -> list[str]:
    """Extract lines of stderr that indicate a validation failure."""
    out = []
    for line in stderr_text.splitlines():
        if _VALIDATION_MARKERS.search(line):
            out.append(line.strip()[:400])
    return out[:12]


def _round_up(n: int, m: int) -> int:
    return (n + m - 1) // m * m


class GuardedBuffer:
    """A buffer embedded in a larger allocation. ``tight=True`` allocates exactly ``nbytes`` with no
    margins so that Metal's shader validation sees every out-of-bounds access."""

    def __init__(self, ctx: MetalContext, nbytes: int, *, guard: int = GUARD_BYTES, tight: bool = False, pattern: int = GUARD_PATTERN) -> None:
        self.ctx = ctx
        self.nbytes = nbytes
        self.tight = tight
        self.pattern = pattern
        self.guard = 0 if tight else guard
        guard = self.guard
        self.padded = nbytes if tight else _round_up(nbytes, 256)
        self.total = guard + self.padded + guard
        self.buf = ctx.new_buffer(self.total)
        self.mem = ctx.view(self.buf)
        self.reset_guards()

    @property
    def arg(self) -> BufferArg:
        return BufferArg(self.buf, self.guard)

    def reset_guards(self) -> None:
        if self.guard:
            self.mem[: self.guard] = bytes([self.pattern]) * self.guard
        tail = self.total - self.guard - self.nbytes
        if tail > 0:
            self.mem[self.guard + self.nbytes : self.total] = bytes([self.pattern]) * tail

    def write(self, raw: bytes) -> None:
        if len(raw) != self.nbytes:
            raise ValueError(f"expected {self.nbytes} bytes, got {len(raw)}")
        self.mem[self.guard : self.guard + self.nbytes] = raw

    def poison(self) -> None:
        self.mem[self.guard : self.guard + self.nbytes] = bytes([POISON_PATTERN]) * self.nbytes

    def read(self) -> bytes:
        return bytes(self.mem[self.guard : self.guard + self.nbytes])

    def guard_violations(self) -> list[str]:
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


class ArgSet:
    """A full set of bound arguments for one launch of a contract."""

    def __init__(self, ctx: MetalContext, contract: Contract, *, guard: int = GUARD_BYTES, tight: bool = False) -> None:
        self.ctx = ctx
        self.contract = contract
        # Input margins are NaN so that out-of-bounds reads poison the result; output margins use a
        # distinct pattern so that out-of-bounds writes are detected.
        self.buffers: dict[str, GuardedBuffer] = {
            a.name: GuardedBuffer(ctx, a.nbytes, guard=guard, tight=tight, pattern=POISON_PATTERN if a.role == "input" else GUARD_PATTERN)
            for a in contract.buffers
        }
        self.scalar_bytes: dict[str, bytes] = {a.name: scalar_bytes(a) for a in contract.scalars}
        self._input_snapshot: dict[str, bytes] = {}

    def reset_guards_all(self) -> None:
        for gb in self.buffers.values():
            gb.reset_guards()

    def load_inputs(self, raw_by_name: dict[str, bytes]) -> None:
        for a in self.contract.inputs:
            self.buffers[a.name].write(raw_by_name[a.name])
        self._input_snapshot = dict(raw_by_name)

    def poison_outputs(self) -> None:
        for a in self.contract.outputs:
            self.buffers[a.name].poison()

    def args(self) -> list[tuple[int, BufferArg | ScalarArg]]:
        out: list[tuple[int, BufferArg | ScalarArg]] = []
        for a in self.contract.args:
            if a.role == "scalar":
                out.append((a.index, ScalarArg(self.scalar_bytes[a.name])))
            else:
                out.append((a.index, self.buffers[a.name].arg))
        return out

    def read_outputs(self) -> dict[str, bytes]:
        return {a.name: self.buffers[a.name].read() for a in self.contract.outputs}

    def safety_violations(self) -> list[str]:
        out: list[str] = []
        for name, gb in self.buffers.items():
            for v in gb.guard_violations():
                out.append(f"buffer `{name}`: {v}")
        for a in self.contract.inputs:
            if self._input_snapshot and self.buffers[a.name].read() != self._input_snapshot[a.name]:
                out.append(f"input `{a.name}` was modified by the kernel")
        return out


def scalar_bytes(a: AbiArg) -> bytes:
    import struct

    d = get_dtype(a.dtype)
    if d.name == "fp32":
        return struct.pack("<f", float(a.value))
    if d.name == "fp16":
        return struct.pack("<e", float(a.value))
    if d.name == "bf16":
        import torch

        return torch.tensor([float(a.value)], dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
    fmt = {"i32": "<i", "u32": "<I", "i8": "<b", "u8": "<B"}[d.name]
    return struct.pack(fmt, int(a.value))


def env_with_validation(base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base is None else base)
    env.update(VALIDATION_ENV)
    return env
