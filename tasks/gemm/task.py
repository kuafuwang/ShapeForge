# 行主序矩阵乘。build 的 description：C = A @ B，A 是 [M,K]，B 是 [K,N]，C 是 [M,N]，fp32 累加，结果舍入一次。普通 gemm 的 epilogue_ref 是恒等。
# 标准答案在 reference：A、B 用 float 做 @（CPU 上按 fp32 累加），再转 double 交给 epilogue_ref。
# 速度基线是 baselines 里的 mlx（a @ b）和 torch_mps（torch.matmul）。
# 模型不能改 contract：kernel_name、grid=(N/BLOCK_N, M/BLOCK_M, 1)、threads、缓冲 A/B/C 的槽位和形状，以及常量 M/N/K/BLOCK_M/BLOCK_N/BLOCK_K。
# 题面由 load_text 读取 kernel_triton.py。build_variant 只换 tile 和线程数，算子和 reference 保持这一套。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text

import re

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
THREADS = 128
DEFAULT_VARIANT = f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K}t{THREADS}"


# 把 "<BM>x<BN>x<BK>t<threads>" 拆成 tile 和线程数。tile 必须整除 M、N、K，线程数必须是 32 的倍数。
def parse_variant(variant: str, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    """@brief 把 ``"<BM>x<BN>x<BK>t<threads>"`` 拆成 tile 和线程数。

    ``"<BM>x<BN>x<BK>t<threads>"`` -> (BM, BN, BK, threads); the tile must divide the problem and the threads must be whole SIMD groups.

    @param variant 启动契约变体文本。
    @param shape 问题的 ``(M, N, K)``。
    @return ``(BM, BN, BK, threads)``。
    @raises ValueError 文本格式不对、tile 不整除问题形状，或线程数不合规时抛出。
    """
    m = re.fullmatch(r"(\d+)x(\d+)x(\d+)t(\d+)", variant)
    if not m:
        raise ValueError(f"bad GEMM contract variant {variant!r}; expected <BM>x<BN>x<BK>t<threads>, e.g. {DEFAULT_VARIANT}")
    bm, bn, bk, th = map(int, m.groups())
    M, N, K = shape
    if M % bm or N % bn or K % bk:
        raise ValueError(f"tile {bm}x{bn}x{bk} does not divide {M}x{N}x{K}")
    if th % 32 or th > 1024 or (bm * bn) % (th * 8) or bn % 8:
        raise ValueError(f"threads {th} must be a multiple of 32, at most 1024, and BM*BN must be a multiple of 8*threads (the naive kernel gives each thread a TMx8 tile)")
    return bm, bn, bk, th
SHAPES = {"paper": (2048, 2048, 2048), "small": (256, 256, 128)}
MAGS = {"normal": 1.0, "wide_range": 0.1}


def _mlx(a, b):
    """@brief mlx 侧的矩阵乘基线。

    @param a 左矩阵，mlx 数组。
    @param b 右矩阵，mlx 数组。
    @return ``a @ b``。
    """
    return a @ b


def _torch(a, b):
    """@brief torch 侧的矩阵乘基线。

    @param a 左矩阵。
    @param b 右矩阵。
    @return ``torch.matmul(a, b)``。
    """
    return torch.matmul(a, b)


# gemm 与 gemm_silu 共用。epilogue_ref 接在 fp32 乘积之后；epilogue_msl 和 epilogue_def 写进朴素 Metal。
def build_gemm(name: str, dtype: str, size: str, *, epilogue_ref, epilogue_msl: str, epilogue_def: str, baselines, description: str, extra_flops: float, variant: str = "") -> Task:
    """@brief 装配 GEMM 类题，gemm 与 gemm_silu 共用。

    ``epilogue_ref`` 接在 fp32 乘积之后；``epilogue_msl`` 和 ``epilogue_def`` 写进朴素 Metal。

    @param name 算子名，用来拼核名和题面目录。
    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @param epilogue_ref 接在 fp32 乘积之后的收尾函数。
    @param epilogue_msl 写进朴素 Metal 的收尾调用名。
    @param epilogue_def 写进朴素 Metal 的收尾函数定义。
    @param baselines 基线实现序列。
    @param description 题面描述。
    @param extra_flops 收尾步骤额外的每输出元素浮点运算量。
    @param variant 启动契约变体，空串表示默认。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    M, N, K = SHAPES[size]
    BLOCK_M, BLOCK_N, BLOCK_K, THREADS = parse_variant(variant or DEFAULT_VARIANT, (M, N, K))
    full_name = f"{name}@{variant}" if variant else name
    contract = Contract(
        kernel_name=f"{name}_kernel",
        grid=(N // BLOCK_N, M // BLOCK_M, 1),
        threads=(THREADS, 1, 1),
        args=(
            AbiArg("A", 0, "input", dtype, (M, K), doc="row-major [M, K]"),
            AbiArg("B", 1, "input", dtype, (K, N), doc="row-major [K, N]"),
            AbiArg("C", 2, "output", dtype, (M, N), doc="row-major [M, N]"),
        ),
        constants=(
            ("M", "uint", M), ("N", "uint", N), ("K", "uint", K),
            ("BLOCK_M", "uint", BLOCK_M), ("BLOCK_N", "uint", BLOCK_N), ("BLOCK_K", "uint", BLOCK_K),
        ),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        return {
            "A": sample(case, (M, K), seed, dtype, mag=mag, large=20.0, spike=5.0),
            "B": sample(case, (K, N), seed + 31, dtype, mag=mag, large=20.0, spike=5.0),
        }

    def reference(inp):
        c = inp["A"].float() @ inp["B"].float()  # fp32 accumulate on CPU (exact products of rounded inputs)
        return {"C": epilogue_ref(c.double())}

    rtol, atol_rel = default_tolerances(dtype)
    subs = {"M": str(M), "N": str(N), "K": str(K), "BM": str(BLOCK_M), "BN": str(BLOCK_N), "BK": str(BLOCK_K), "NW": str(THREADS // 32)}
    naive = load_text("gemm", "naive.metal", dtype, {"KERNEL": f"{name}_kernel", "EPILOGUE": epilogue_msl, "EPILOGUE_DEF": epilogue_def,
                                                    "THREADS": str(THREADS), "TM": str(BLOCK_M * BLOCK_N // (THREADS * 8)), "TPR": str(BLOCK_N // 8)})
    # meta 里另放一份 Triton→MSL 编译器接受的题面。和库比速度看 baselines。
    meta = {}
    if name == "gemm":  # the only variant triton-msl accepts for this operator (see tcenv.triton_runner)
        meta["triton_baseline_variant"] = {"source": load_text("gemm", "kernel_triton_msl.py", dtype, subs), "note": "program-id roles swapped, M/N/K as runtime scalars"}
    elif name == "gemm_silu":  # the backend refuses a fused epilogue; it asks for a separate epilogue kernel
        meta["triton_baseline_variant"] = {"source": load_text("gemm_silu", "kernel_triton_msl.py", dtype, subs), "note": "GEMM storing the fp32 accumulator + a separate SiLU kernel (the backend refuses the fused epilogue)"}
    return Task(
        name=full_name, dtype=dtype, size=size, contract=contract,
        triton_source=load_text(name, "kernel_triton.py", dtype, subs),
        description=description,
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference, baselines=baselines,
        flops=2.0 * M * N * K + extra_flops * M * N, min_bytes=float((M * K + K * N + M * N) * 2),
        rtol=rtol, atol_rel=atol_rel, naive_msl=naive, meta=meta,
    )


def build(dtype: str, size: str) -> Task:
    """@brief 装配矩阵乘题，收尾是恒等。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_gemm(
        "gemm", dtype, size,
        epilogue_ref=lambda c: c,
        epilogue_msl="", epilogue_def="",
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        description="C = A @ B with row-major A[M,K], B[K,N], C[M,N]; fp32 accumulation, result rounded once.",
        extra_flops=0.0,
    )


# 同一算子、同一 reference 和 baselines，只把 variant 交给 parse_variant，换掉 grid 用的 tile 和 threads。
def build_variant(dtype: str, size: str, variant: str) -> Task:
    """@brief 同一算子、同一 reference 与 baselines，只换掉 grid 用的 tile 和 threads。

    The same operator under another launch contract (tile and threads); used by the contract-sensitivity study.

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @param variant 启动契约变体文本，交给 ``parse_variant``。
    @return 装配好的 ``Task``，名字带 ``@<variant>`` 后缀。
    """
    return build_gemm(
        "gemm", dtype, size,
        epilogue_ref=lambda c: c,
        epilogue_msl="", epilogue_def="",
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("torch_mps", "torch", _torch)),
        description="C = A @ B with row-major A[M,K], B[K,N], C[M,N]; fp32 accumulation, result rounded once.",
        extra_flops=0.0, variant=variant,
    )
