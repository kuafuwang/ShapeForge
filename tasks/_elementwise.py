"""Builder shared by sigmoid / relu / silu / swiglu."""

# sigmoid / relu / silu / swiglu 只换公式、输入名和基线，启动方式相同，所以共用 build_elementwise。
# 契约在这里钉死：kernel_name 默认 "<名字>_kernel"，grid=(n/BLOCK_SIZE, 1, 1)，threads=(NUM_WARPS*32, 1, 1)。
# args 按 input_names 排输入缓冲，再追加输出 y；constants 是 N 和 BLOCK_SIZE。模型不能改核名、槽位、grid、threads 和这两项常量。
# reference 先把每个输入转成 float64，再调用该题自己的 ref_fn，这就是标准答案。题面默认读 tasks/<名字>/kernel_triton.py，朴素 Metal 读 naive.metal。
from __future__ import annotations

from typing import Callable

import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances

from ._common import check_dtype, load_text

BLOCK_SIZE = 1024
NUM_WARPS = 4
# 四题共用：每个 threadgroup 处理 1024 个元素（4×32 线程）。paper 是 2^24 个元素，small 是 2^16。
SIZES = {"paper": 1 << 24, "small": 1 << 16}


# 各题传入 ref_fn、baselines、flops_per_elem、description。语义题用 triton_source / naive_msl / kernel_name / task_name 盖住默认值。
def build_elementwise(
    name: str,
    dtype: str,
    size: str,
    *,
    input_names: tuple[str, ...],
    ref_fn: Callable[..., torch.Tensor],
    baselines: tuple[BaselineImpl, ...],
    flops_per_elem: float,
    description: str,
    large: float = 1.0e4,
    normal_mag: float = 3.0,
    kernel_name: str | None = None,
    triton_source: str | None = None,
    naive_msl: str | None = None,
    task_name: str | None = None,
) -> Task:
    check_dtype(dtype)
    n = SIZES[size]
    args = [AbiArg(nm, i, "input", dtype, (n,)) for i, nm in enumerate(input_names)]
    args.append(AbiArg("y", len(input_names), "output", dtype, (n,)))
    contract = Contract(
        kernel_name=kernel_name or f"{name}_kernel",
        grid=(n // BLOCK_SIZE, 1, 1),
        threads=(NUM_WARPS * 32, 1, 1),
        args=tuple(args),
        constants=(("N", "uint", n), ("BLOCK_SIZE", "uint", BLOCK_SIZE)),
    )

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        mag = normal_mag if case == "normal" else 1.0
        return {nm: sample(case, (n,), seed + 1_000_003 * i, dtype, mag=mag, large=large) for i, nm in enumerate(input_names)}

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"y": ref_fn(*[inp[nm].to(torch.float64) for nm in input_names])}

    rtol, atol_rel = default_tolerances(dtype)
    nbuf = len(input_names) + 1
    src_dir = name  # directory holding kernel_triton.py / naive.metal
    return Task(
        name=task_name or name,
        dtype=dtype,
        size=size,
        contract=contract,
        triton_source=triton_source if triton_source is not None else load_text(src_dir, "kernel_triton.py", dtype, {"N": str(n), "BLOCK_SIZE": str(BLOCK_SIZE), "NUM_WARPS": str(NUM_WARPS)}),
        description=description,
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs,
        reference=reference,
        baselines=baselines,
        flops=flops_per_elem * n,
        min_bytes=float(nbuf * n * (2 if dtype in ("fp16", "bf16") else 4)),
        rtol=rtol,
        atol_rel=atol_rel,
        naive_msl=naive_msl if naive_msl is not None else load_text(src_dir, "naive.metal", dtype),
    )
