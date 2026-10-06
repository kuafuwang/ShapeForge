# Lion 优化器的一步。description：c = b1*m + (1-b1)*g，p_out = p*(1 - lr*wd) - lr*sign(c)，m_out = b2*m + (1-b2)*g。系数是模块上的 LR、WD、BETA1、BETA2。
# 标准答案是 reference：p、g、m 转成 double 后按同一式子写出 p_out 和 m_out。error_scale 在 sign 的间断点放宽 p_out 的容差。
# 基线有三条。mlx 是 mx.compile 的同一步；torch_mps 是未融合的逐算子 _torch；torch_compile 用 torch.compile 把这一步收成一个 Metal 核。
# 模型不能改 contract：核名 lion_kernel，grid=(n/BLOCK_SIZE, 1, 1)，threads=(128, 1, 1)，缓冲 p/g/m/p_out/m_out，常量 N/BLOCK_SIZE/LR/WD/BETA1/BETA2。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text, mlx_compiled, torch_compiled

LR, WD, BETA1, BETA2 = 1e-3, 0.1, 0.9, 0.99
BLOCK_SIZE = 1024
SIZES = {"paper": 1 << 23, "small": 1 << 16}


def _mlx(p, g, m):
    def build(mx):
        def step(p, g, m):
            u = mx.sign(BETA1 * m + (1 - BETA1) * g)
            return p * (1 - LR * WD) - LR * u, BETA2 * m + (1 - BETA2) * g

        return step

    return mlx_compiled("lion", build)(p, g, m)


def _torch(p, g, m):  # unfused eager ops: there is no fused Lion kernel on MPS
    u = torch.sign(BETA1 * m + (1 - BETA1) * g)
    return p * (1 - LR * WD) - LR * u, BETA2 * m + (1 - BETA2) * g


def _torch_compile(p, g, m):  # Inductor (Metal backend) fuses the whole step into one kernel
    return torch_compiled("lion", _torch)(p, g, m)


def build(dtype: str, size: str) -> Task:
    check_dtype(dtype)
    n = SIZES[size]
    args = (
        AbiArg("p", 0, "input", dtype, (n,), doc="parameter"),
        AbiArg("g", 1, "input", dtype, (n,), doc="gradient"),
        AbiArg("m", 2, "input", dtype, (n,), doc="momentum"),
        AbiArg("p_out", 3, "output", dtype, (n,)),
        AbiArg("m_out", 4, "output", dtype, (n,)),
    )
    contract = Contract(
        kernel_name="lion_kernel", grid=(n // BLOCK_SIZE, 1, 1), threads=(128, 1, 1), args=args,
        constants=(("N", "uint", n), ("BLOCK_SIZE", "uint", BLOCK_SIZE), ("LR", "float", LR), ("WD", "float", WD), ("BETA1", "float", BETA1), ("BETA2", "float", BETA2)),
    )

    rtol, atol_rel = default_tolerances(dtype)

    def gen_inputs(case, seed):
        out = {nm: sample(case, (n,), seed + 7 * i, dtype, mag=1.0, large=1.0e3) for i, nm in enumerate(("p", "g", "m"))}
        if case == "tiny" and dtype != "fp16":
            # tiny 且不是 fp16 时把 p、g、m 乘 1e3，让 (1-beta)*x 留在正规数范围，避免 sign 两侧对调。
            # fp32 subnormals are flushed by the GPU: keep the products (1-beta)*x of the step formula in the normal range,
            # otherwise sign(beta1*m + (1-beta1)*g) legitimately flips for every implementation.
            out = {k: v * 1.0e3 for k, v in out.items()}
        return out

    def reference(inp):
        p, g, m = (inp[k].double() for k in ("p", "g", "m"))
        u = torch.sign(BETA1 * m + (1 - BETA1) * g)
        return {"p_out": p * (1 - LR * WD) - LR * u, "m_out": BETA2 * m + (1 - BETA2) * g}

    def error_scale(inp, ref):
        # sign() is discontinuous: where beta1*m + (1-beta1)*g cancels to ~0 a correct kernel may legitimately round to the
        # other side. Allow the full +-2*lr step difference exactly there (and nowhere else).
        p, g, m = (inp[k].double() for k in ("p", "g", "m"))
        a, b = BETA1 * m, (1 - BETA1) * g
        amb = (a + b).abs() <= 1e-5 * (a.abs() + b.abs())
        base = torch.sqrt(torch.mean(ref["p_out"] ** 2)).expand_as(ref["p_out"]).clone()
        base[amb] = 2.5 * LR / atol_rel  # atol_rel of *this* dtype (a module-level global leaked between dtypes in one process)
        return {"p_out": base, "m_out": torch.sqrt(torch.mean(ref["m_out"] ** 2))}

    return Task(
        name="lion", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("lion", "kernel_triton.py", dtype, {"N": str(n)}),
        description="One Lion optimizer step: c = b1*m + (1-b1)*g; p_out = p*(1 - lr*wd) - lr*sign(c); m_out = b2*m + (1-b2)*g (fp32 math, each output rounded once).",
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference, error_scale=error_scale,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch), BaselineImpl("torch_compile", "torch", _torch_compile)),
        flops=10.0 * n, min_bytes=float(5 * n * 2), rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("lion", "naive.metal", dtype),
    )

