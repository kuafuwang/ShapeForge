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
    """@brief 计算让工作集超过缓存所需的轮转缓冲组数。

    @param task 任务对象，用其总缓冲字节数估算。
    @return 轮转组数，介于 1 到 RING_MAX 之间。

    How many distinct buffer sets to cycle through so that the working set exceeds the caches.
    """
    per = max(task.total_buffer_bytes, 1)
    return max(1, min(RING_MAX, -(-RING_TARGET_BYTES // per)))


def _mx():
    """@brief 惰性导入并返回 mlx.core 模块。

    @return mlx.core 模块对象。
    """
    import mlx.core as mx

    return mx


_MX_DTYPES = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def to_framework(t: torch.Tensor, dtype: str, framework: str):
    """@brief 把 PyTorch 张量搬到指定框架，并保持契约要求的位宽。

    @param t 源张量。
    @param dtype 契约数据类型名。
    @param framework ``mlx`` 或 ``torch``。
    @return 该框架下的数组或张量。
    @raises ValueError 当 ``framework`` 不是 ``mlx`` 或 ``torch``。
    """
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
    """@brief 递归展开嵌套的 list/tuple，产出叶子元素。

    @param x 任意元素或嵌套序列。
    @return 叶子元素的生成器。
    """
    if isinstance(x, (list, tuple)):
        for y in x:
            yield from _flatten(y)
    else:
        yield x


class BaselineRunner:
    def __init__(self, impl: BaselineImpl, task: Task, rounded_inputs: dict[str, torch.Tensor], rotate: int | None = None) -> None:
        """@brief 为一次评测准备该基线的轮转输入组。

        @param impl 基线实现声明。
        @param task 任务对象。
        @param rounded_inputs 已按契约舍入的输入张量。
        @param rotate 轮转组数；省略时按任务自动计算。
        """
        self.impl = impl
        self.task = task
        self.rotate = rotate if rotate is not None else rotation_count(task)
        args = task.contract.inputs
        self.sets = [[to_framework(rounded_inputs[a.name], a.dtype, impl.framework) for a in args] for _ in range(self.rotate)]
        self._sync_all()

    # -- helpers -------------------------------------------------------------
    def _sync_all(self) -> None:
        """@brief 同步该基线已排队的全部工作。"""
        if self.impl.framework == "mlx":
            _mx().eval(*[a for s in self.sets for a in s])
        else:
            torch.mps.synchronize()

    def _to_torch(self, x) -> torch.Tensor:
        """@brief 把基线输出转成 float32 的 CPU 张量。

        @param x 框架原生数组或张量。
        @return CPU 上的 float32 张量。
        """
        if self.impl.framework == "mlx":
            import numpy as np

            mx = _mx()
            return torch.from_numpy(np.array(x.astype(mx.float32)))
        return x.detach().to(torch.float32).cpu()

    # -- correctness output ----------------------------------------------------
    def outputs(self) -> dict[str, torch.Tensor]:
        """@brief 在第一个输入组上求基线输出，并转成契约形状。

        @return 输出名到 CPU float32 张量的映射。
        @raises RuntimeError 当基线返回的输出个数与契约不符。
        """
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
        """@brief 计时 k 次基线调用，MLX 只计 encode 与 GPU 时间。

        @param k 调用次数。
        @return 该批调用的墙钟秒数。
        """
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
    """@brief 为任务里能跑的基线各建一个计时器。

    @param task 任务对象。
    @param rounded_inputs 已舍入的输入张量。
    @param names 只构造这些基线；省略时全部尝试。
    @return 基线名到 BaselineRunner 的映射，跑不起来的基线被跳过。
    """
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
