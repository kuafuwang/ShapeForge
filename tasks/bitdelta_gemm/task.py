# BitDelta 矩阵乘。description：y = scale * (x @ W)，W 是打包在 bits 里的 ±1 矩阵，要在核内解码，乘积按 fp32 累加后舍入一次。SCALE = 0.02。
# 标准答案是 reference：unpack_pm1 把 int32 字展开成 ±1 的 float32，再计算 (x.float() @ w).double() * SCALE。
# 四条基线都是先解包再做稠密乘。mlx 与 torch_mps 把解包和乘法分成两步；mlx_compile 与 torch_compile 把解码收进一次编译，矩阵乘保持单独一次调用。
# 模型不能改 contract：核名 bitdelta_gemm_kernel，grid=(N/BLOCK_N, M/BLOCK_M, 1)，threads=(128, 1, 1)，缓冲 x、u32 的 bits、标量 scale、y，常量 M/N/K/BLOCK_*。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text, mlx_compiled, torch_compiled

BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
SCALE = 0.02
SHAPES = {"paper": (512, 4096, 4096), "small": (128, 256, 256)}  # (M, N, K)
EDGE_WORDS = [0, -1, 0x55555555, -0x55555556, 1, -(1 << 31), 0x0F0F0F0F, -0x0F0F0F10]


# 第 kb 个字的第 j 位是行 kb*32+j 的符号：1 映成 +1，0 映成 -1。reference 用这份展开当 W。
def unpack_pm1(words: torch.Tensor) -> torch.Tensor:
    """@brief 把打包的符号字展开成 ±1 的 float32 矩阵。

    [K//32, N] int32 words -> [K, N] float32 of +-1 (bit j of word kb is row kb*32 + j).

    @param words 打包的符号字，形状 ``[K//32, N]``。
    @return ``[K, N]`` 的 ±1 矩阵。
    """
    w = words.to(torch.int64)
    j = torch.arange(32, dtype=torch.int64).reshape(1, 32, 1)
    bit = (w[:, None, :] >> j) & 1
    return (bit.to(torch.float32) * 2 - 1).reshape(-1, words.shape[-1])


def build(dtype: str, size: str) -> Task:
    """@brief 装配 BitDelta 矩阵乘题。

    核名 ``bitdelta_gemm_kernel``，``grid=(N/BLOCK_N, M/BLOCK_M, 1)``，``threads=(128, 1, 1)``，
    缓冲 x、u32 的 bits、标量 scale、y，常量 M/N/K/BLOCK_*。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    M, N, K = SHAPES[size]
    contract = Contract(
        kernel_name="bitdelta_gemm_kernel",
        grid=(N // BLOCK_N, M // BLOCK_M, 1),
        threads=(128, 1, 1),
        args=(
            AbiArg("x", 0, "input", dtype, (M, K), doc="activations, row-major [M, K]"),
            AbiArg("bits", 1, "input", "u32", (K // 32, N), doc="packed signs: bit j of bits[kb, n] is the sign of W[kb*32 + j, n] (1 -> +1, 0 -> -1)"),
            AbiArg("scale", 2, "scalar", "fp32", value=SCALE, doc="per-matrix scale"),
            AbiArg("y", 3, "output", dtype, (M, N)),
        ),
        constants=(("M", "uint", M), ("N", "uint", N), ("K", "uint", K), ("BLOCK_M", "uint", BLOCK_M), ("BLOCK_N", "uint", BLOCK_N), ("BLOCK_K", "uint", BLOCK_K)),
    )

    def gen_inputs(case, seed):
        g = torch.Generator().manual_seed(seed & 0x7FFFFFFF)
        if case in ("edge_bits",):
            words = torch.tensor(EDGE_WORDS, dtype=torch.int64)[torch.randint(0, len(EDGE_WORDS), (K // 32, N), generator=g)]
        else:
            words = torch.randint(-(1 << 31), 1 << 31, (K // 32, N), generator=g, dtype=torch.int64)
        xcase = "normal" if case in ("random", "edge_bits") else case
        return {"x": sample(xcase, (M, K), seed + 5, dtype, mag=1.0, large=20.0, spike=5.0), "bits": words}

    def reference(inp):
        w = unpack_pm1(inp["bits"])
        return {"y": (inp["x"].float() @ w).double() * SCALE}

    rtol, atol_rel = default_tolerances(dtype)
    return Task(
        name="bitdelta_gemm", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("bitdelta_gemm", "kernel_triton.py", dtype, {"M": str(M), "N": str(N), "K": str(K)}),
        description=(
            "y = scale * (x @ W) where W is a +-1 matrix stored as packed sign bits (BitDelta). The bits must be decoded inside the kernel; products accumulate in fp32 and the result is rounded once. "
            "Note: no fused library kernel exists for this operator. The baselines unpack the bits into an fp16 matrix and call a dense matmul; the strongest ones (mx.compile, torch.compile = Inductor on Metal) fuse the decode into one elementwise kernel, so the cost to beat is that decode pass plus the matmul."
        ),
        cases=("random", "edge_bits", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(
            BaselineImpl("mlx", "mlx", _mlx_factory(K, N)),
            BaselineImpl("mlx_compile", "mlx", _mlx_compiled_factory(K, N)),
            BaselineImpl("torch_mps", "torch", _torch_factory(K, N)),
            BaselineImpl("torch_compile", "torch", _torch_compiled_factory(K, N)),
        ),
        flops=2.0 * M * N * K, min_bytes=float(M * K * 2 + (K // 32) * N * 4 + M * N * 2),
        rtol=rtol, atol_rel=atol_rel, naive_msl=load_text("bitdelta_gemm", "naive.metal", dtype),
        meta={"baseline_note": "unfused unpack + dense matmul (no fused library kernel exists); torch_compile (Inductor on Metal) fuses the decode and is the strong baseline", "ieee_base": "random"},
    )


def _mlx_factory(K, N):
    """@brief 造一个 mlx 侧的“先解包再稠密乘”基线。

    @param K 矩阵乘的缩减维长度。
    @param N 输出列数。
    @return 签名如 ``f(x, bits)`` 的基线函数。
    """
    def f(x, bits):
        import mlx.core as mx

        j = mx.arange(32, dtype=mx.uint32).reshape(1, 32, 1)
        b = (bits.reshape(K // 32, 1, N) >> j) & 1
        w = (b.astype(x.dtype) * 2 - 1).reshape(K, N)
        return (x @ w) * SCALE

    return f


def _mlx_compiled_factory(K, N):  # the decode of the bits is fused into one kernel; the matmul stays a separate primitive
    """@brief 造一个 mlx 基线，解包收进一次编译，矩阵乘仍是单独一次调用。

    @param K 矩阵乘的缩减维长度。
    @param N 输出列数。
    @return 签名如 ``f(x, bits)`` 的基线函数。
    """
    def f(x, bits):
        return mlx_compiled(f"bitdelta_{K}_{N}", lambda mx: _mlx_factory(K, N))(x, bits)

    return f


def _torch_compiled_factory(K, N):
    """@brief 造一个 torch 基线，解包由 Inductor 融进一个核，矩阵乘仍是单独一次调用。

    @param K 矩阵乘的缩减维长度。
    @param N 输出列数。
    @return 签名如 ``f(x, bits)`` 的基线函数。
    """
    def f(x, bits):
        return torch_compiled(f"bitdelta_{K}_{N}", _torch_factory(K, N))(x, bits)

    return f


def _torch_factory(K, N):
    """@brief 造一个 torch 侧的“先解包再稠密乘”基线。

    @param K 矩阵乘的缩减维长度。
    @param N 输出列数。
    @return 签名如 ``f(x, bits)`` 的基线函数。
    """
    def f(x, bits):
        j = torch.arange(32, dtype=torch.int32, device=x.device).reshape(1, 32, 1)
        b = (bits.reshape(K // 32, 1, N) >> j) & 1
        w = (b.to(x.dtype) * 2 - 1).reshape(K, N)
        return (x @ w) * SCALE

    return f
