"""Framework baselines (MLX, PyTorch-MPS) driven under the same timing protocol as candidates.

Paper counterpart: the autotuned Triton kernel. Here the strongest locally available
implementations play that role; the faster of the two is the comparison point.

Timing protocol for frameworks: ``k`` iterations are enqueued over rotating input sets,
then the queue is synchronized. For MLX the lazy graph is built *before* the timer starts
so that only encode + GPU time is measured.
"""

from __future__ import annotations

import time
from typing import Callable

import torch

from ...task import BaselineImpl, Task
from ...tensors import round_to

RING_TARGET_BYTES = 192 * 2**20
RING_MAX = 16


def rotation_count(task: Task) -> int:
    """How many distinct buffer sets to cycle through so that the working set exceeds the caches."""
    per = max(task.total_buffer_bytes, 1)
    return max(1, min(RING_MAX, -(-RING_TARGET_BYTES // per)))


def _mx():
    import mlx.core as mx

    return mx


_MX_DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def to_framework(t: torch.Tensor, dtype: str, framework: str):
    if dtype in ("i32", "u32"):  # integer inputs (e.g. packed bit words): keep the 32 bits, no float round trip
        if framework == "mlx":
            import numpy as np

            mx = _mx()
            return mx.array(t.detach().cpu().contiguous().to(torch.int32).numpy().astype(np.uint32 if dtype == "u32" else np.int32))
        return t.detach().to(device="mps", dtype=torch.int32).contiguous()
    if framework == "mlx":
        mx = _mx()
        arr = mx.array(t.detach().cpu().contiguous().to(torch.float32).numpy())
        return arr.astype(getattr(mx, _MX_DTYPES[dtype]))
    if framework == "torch":
        td = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype]
        return t.detach().to(device="mps", dtype=td).contiguous()
    raise ValueError(framework)


def _flatten(x):
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten(y)
    else:
        yield x


class BaselineRunner:
    def __init__(self, impl: BaselineImpl, task: Task, rounded_inputs: dict[str, torch.Tensor], rotate: int | None = None) -> None:
        self.impl = impl
        self.task = task
        self.rotate = rotate if rotate is not None else rotation_count(task)
        args = task.contract.inputs
        self.sets = [[to_framework(rounded_inputs[a.name], a.dtype, impl.framework) for a in args] for _ in range(self.rotate)]
        self._sync_all()

    # -- helpers -------------------------------------------------------------
    def _sync_all(self) -> None:
        if self.impl.framework == "mlx":
            _mx().eval(*[a for s in self.sets for a in s])
        else:
            torch.mps.synchronize()

    def _to_torch(self, x) -> torch.Tensor:
        if self.impl.framework == "mlx":
            import numpy as np

            mx = _mx()
            return torch.from_numpy(np.array(x.astype(mx.float32)))
        return x.detach().to(torch.float32).cpu()

    # -- correctness output ----------------------------------------------------
    def outputs(self) -> dict[str, torch.Tensor]:
        res = self.impl.fn(*self.sets[0])
        res = res if isinstance(res, (tuple, list)) else (res,)
        if self.impl.framework == "mlx":
            _mx().eval(*res)
        else:
            torch.mps.synchronize()
        outs = self.task.contract.outputs
        if len(res) != len(outs):
            raise RuntimeError(f"baseline {self.impl.name} returned {len(res)} outputs, task has {len(outs)}")
        return {a.name: self._to_torch(r).reshape(a.shape) for a, r in zip(outs, res)}

    # -- timing ----------------------------------------------------------------
    def run_k(self, k: int) -> float:
        fn, sets, n = self.impl.fn, self.sets, len(self.sets)
        if self.impl.framework == "mlx":
            mx = _mx()
            outs = [fn(*sets[i % n]) for i in range(k)]  # lazy graph construction, untimed
            flat = list(_flatten(outs))
            t0 = time.perf_counter()
            mx.eval(*flat)
            return time.perf_counter() - t0
        t0 = time.perf_counter()
        keep = None
        for i in range(k):
            keep = fn(*sets[i % n])
        torch.mps.synchronize()
        dt = time.perf_counter() - t0
        del keep
        return dt


def build_runners(task: Task, rounded_inputs: dict[str, torch.Tensor], names: list[str] | None = None) -> dict[str, BaselineRunner]:
    runners: dict[str, BaselineRunner] = {}
    for impl in task.baselines:
        if names is not None and impl.name not in names:
            continue
        try:
            runners[impl.name] = BaselineRunner(impl, task, rounded_inputs)
        except Exception as e:  # a baseline that cannot run on this machine is skipped, not fatal
            runners[impl.name] = None  # type: ignore[assignment]
            print(f"[tcenv] baseline {impl.name} unavailable: {type(e).__name__}: {e}")
    return {k: v for k, v in runners.items() if v is not None}
