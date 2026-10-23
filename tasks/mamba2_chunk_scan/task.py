# Mamba-2 的 chunk scan，一个 B/C 组。description 里的 y[t] 由三部分相加：块内 s<=t 的因果和、exp(cs[t]) 乘 C[t] 与 state 的点积、D*x[t]。CHUNK = 64。
# 标准答案是 reference：用内嵌的 _RefOps 调用 ssd_chunk_scan。_RefOps.f32 把张量转成 double，式子和下面的库基线相同。
# 基线都走 ssd_chunk_scan。mlx、mlx_compile、torch_mps 把乘加操作数升到 fp32；mlx_half_compile 和 torch_half 传 half=True，乘加保持输入 dtype。
# 模型不能改 contract：核名 mamba2_chunk_scan_kernel，grid=(NC, B*H, 1)，threads=(128, 1, 1)，从 x 到 y 八个参数，常量 BATCH/SEQLEN/NHEADS/HEADDIM/DSTATE/CHUNK/NCHUNKS。
import torch

from tcenv.contract import AbiArg, Contract
from tcenv.distributions import sample
from tcenv.tensors import round_to
from tcenv.task import BaselineImpl, Task, default_tolerances
from tasks._common import check_dtype, load_text, mlx_compiled

CHUNK = 64
# (BATCH, SEQLEN, NHEADS, HEADDIM, DSTATE)
SHAPES = {"paper": (2, 2048, 8, 64, 64), "small": (1, 256, 2, 64, 64)}
MAGS = {"normal": 1.0, "wide_range": 0.1}
DT_LO, DT_HI = 1.0e-3, 5.0e-2  # step sizes
A_LO, A_HI = 1.0, 8.0  # |A| per head (A is negative): the decay of a chunk, exp(cs), stays a normal fp32 number (>= e^-26)


# 库基线和标准答案共用这一式。half=False 时乘加操作数升到 fp32；half=True 时乘加保持输入 dtype。
def ssd_chunk_scan(x, dt, cs, bm, cm, d, st, *, ops, batch, seq, heads, headdim, dstate, half=False):
    """@brief 用框架算子写出 Mamba-2 的 chunk scan，库基线与标准答案共用这一式。

    The operator in framework ops (the formulation of the Mamba-2 paper, listing 1, for one group): this is what the vendor baselines run.
    ``ops`` supplies einsum / exp / where / tril so that the same code runs on MLX and torch. The decay factors and the sums are fp32.
    ``half=False`` upcasts every matmul operand to fp32; ``half=True`` keeps the matmul operands in the input dtype (what the production Triton
    kernels do: half-precision matmuls with fp32 accumulation inside the library, results rounded to the element type per product).

    @param x 输入，形状 ``[BATCH, SEQLEN, NHEADS, HEADDIM]``。
    @param dt 步长，``> 0``。
    @param cs 块内 dt*A 的含前缀和，``[BATCH, NHEADS, NCHUNKS, CHUNK]``。
    @param bm B 矩阵，``[BATCH, SEQLEN, DSTATE]``。
    @param cm C 矩阵，``[BATCH, SEQLEN, DSTATE]``。
    @param d 每 head 的 skip 连接，``[NHEADS]``。
    @param st 每个块开始时的 state，``[BATCH, NCHUNKS, NHEADS, HEADDIM, DSTATE]``。
    @param ops 算子集，提供 f32 / permute / like / einsum / exp / where / causal。
    @param batch 批大小。
    @param seq 序列长度。
    @param heads head 数。
    @param headdim head 维。
    @param dstate state 维。
    @param half 矩阵乘操作数是否保持输入 dtype。
    @return 输出，形状 ``[BATCH, SEQLEN, NHEADS, HEADDIM]``。
    """
    nc = seq // CHUNK
    f = ops.f32
    mm = (lambda a: a) if half else f  # operand of a matmul
    x5 = x.reshape(batch, nc, CHUNK, heads, headdim)
    dt4 = ops.permute(f(dt).reshape(batch, nc, CHUNK, heads), (0, 3, 1, 2))  # [B, H, NC, L]
    cs4 = f(cs)  # [B, H, NC, L]
    br = bm.reshape(batch, nc, CHUNK, dstate)
    cr = cm.reshape(batch, nc, CHUNK, dstate)
    cb = f(ops.einsum("bcln,bcsn->bcls", mm(cr), mm(br)))  # C[t] . B[s]
    decay = ops.exp(cs4[..., :, None] - cs4[..., None, :])  # exp(cs[t] - cs[s])
    w = ops.where(ops.causal(CHUNK), cb[:, None] * decay * dt4[..., None, :], 0.0)  # [B, H, NC, L, L]
    w = ops.like(w, x) if half else w
    y = f(ops.einsum("bhcls,bcshp->bclhp", w, mm(x5)))
    y = y + f(ops.einsum("bcln,bchpn->bclhp", mm(cr), mm(st))) * ops.permute(ops.exp(cs4), (0, 2, 3, 1))[..., None]
    y = y + f(d)[:, None] * f(x5)
    return y.reshape(batch, seq, heads, headdim)


class _MlxOps:
    """@brief ``ssd_chunk_scan`` 在 mlx 上的算子集。"""

    @staticmethod
    def f32(a):
        """@brief 转成 float32。

        @param a mlx 数组。
        @return float32 数组。
        """
        import mlx.core as mx

        return a.astype(mx.float32)

    @staticmethod
    def permute(a, axes):
        """@brief 按给定轴顺序重排。

        @param a mlx 数组。
        @param axes 新的轴顺序。
        @return 重排后的数组。
        """
        return a.transpose(*axes)

    @staticmethod
    def like(a, ref):
        """@brief 把 a 转成 ref 的 dtype。

        @param a mlx 数组。
        @param ref 提供目标 dtype 的数组。
        @return 转好的数组。
        """
        return a.astype(ref.dtype)

    @staticmethod
    def einsum(spec, *a):
        """@brief 按爱因斯坦求和约定做张量缩并。

        @param spec 缩并描述字符串。
        @param a 参与缩并的数组，按 spec 顺序。
        @return 缩并结果。
        """
        import mlx.core as mx

        return mx.einsum(spec, *a)

    @staticmethod
    def exp(a):
        """@brief 逐元素指数。

        @param a mlx 数组。
        @return ``mx.exp(a)``。
        """
        import mlx.core as mx

        return mx.exp(a)

    @staticmethod
    def where(m, a, b):
        """@brief 按掩码二选一。

        @param m 布尔掩码，mlx 数组。
        @param a 掩码为真时取的值。
        @param b 掩码为假时取的值。
        @return 选择结果。
        """
        import mlx.core as mx

        return mx.where(m, a, b)

    @staticmethod
    def causal(n):
        """@brief 造 ``n x n`` 的下三角因果掩码。

        @param n 方阵边长。
        @return 布尔掩码数组。
        """
        import mlx.core as mx

        return mx.tril(mx.ones((n, n))) > 0


class _TorchOps:
    """@brief ``ssd_chunk_scan`` 在 torch 上的算子集。"""

    @staticmethod
    def f32(a):
        """@brief 转成 float32。

        @param a 输入张量。
        @return float32 张量。
        """
        return a.float()

    @staticmethod
    def permute(a, axes):
        """@brief 按给定轴顺序重排。

        @param a 输入张量。
        @param axes 新的轴顺序。
        @return 重排后的张量。
        """
        return a.permute(*axes)

    @staticmethod
    def like(a, ref):
        """@brief 把 a 转成 ref 的 dtype。

        @param a 输入张量。
        @param ref 提供目标 dtype 的张量。
        @return 转好的张量。
        """
        return a.to(ref.dtype)

    einsum = staticmethod(torch.einsum)
    exp = staticmethod(torch.exp)

    @staticmethod
    def where(m, a, b):
        """@brief 按掩码二选一，假分支按 a 的 dtype 造零。

        @param m 布尔掩码张量。
        @param a 掩码为真时取的值。
        @param b 掩码为假时用到的标量值。
        @return 选择结果。
        """
        return torch.where(m, a, torch.as_tensor(b, dtype=a.dtype, device=a.device))

    @staticmethod
    def causal(n):
        """@brief 在 MPS 上造 ``n x n`` 的下三角因果掩码。

        @param n 方阵边长。
        @return 布尔掩码张量。
        """
        return torch.tril(torch.ones((n, n), dtype=torch.bool, device="mps"))


def build(dtype: str, size: str) -> Task:
    """@brief 装配 Mamba-2 chunk scan 题。

    核名 ``mamba2_chunk_scan_kernel``，``grid=(NC, B*H, 1)``，``threads=(128, 1, 1)``，从 x 到 y 八个参数，
    常量 BATCH/SEQLEN/NHEADS/HEADDIM/DSTATE/CHUNK/NCHUNKS。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    B, S, H, P, N = SHAPES[size]
    assert S % CHUNK == 0
    NC = S // CHUNK
    args = (
        AbiArg("x", 0, "input", dtype, (B, S, H, P), doc="[BATCH, SEQLEN, NHEADS, HEADDIM]"),
        AbiArg("dt", 1, "input", dtype, (B, S, H), doc="step size, > 0"),
        AbiArg("dA_cumsum", 2, "input", dtype, (B, H, NC, CHUNK), doc="inclusive cumulative sum of dt*A inside each chunk (non-increasing)"),
        AbiArg("Bmat", 3, "input", dtype, (B, S, N), doc="[BATCH, SEQLEN, DSTATE], shared by all heads"),
        AbiArg("Cmat", 4, "input", dtype, (B, S, N), doc="[BATCH, SEQLEN, DSTATE], shared by all heads"),
        AbiArg("Dskip", 5, "input", dtype, (H,), doc="skip connection per head"),
        AbiArg("states", 6, "input", dtype, (B, NC, H, P, N), doc="state entering each chunk [BATCH, NCHUNKS, NHEADS, HEADDIM, DSTATE]"),
        AbiArg("y", 7, "output", dtype, (B, S, H, P), doc="[BATCH, SEQLEN, NHEADS, HEADDIM]"),
    )
    contract = Contract(
        kernel_name="mamba2_chunk_scan_kernel", grid=(NC, B * H, 1), threads=(128, 1, 1), args=args,
        constants=(("BATCH", "uint", B), ("SEQLEN", "uint", S), ("NHEADS", "uint", H), ("HEADDIM", "uint", P), ("DSTATE", "uint", N), ("CHUNK", "uint", CHUNK), ("NCHUNKS", "uint", NC)),
    )

    def gen_inputs(case, seed):
        mag = MAGS.get(case, 1.0)
        mk = lambda shape, s: sample(case, shape, seed + s, dtype, mag=mag, large=4.0, spike=5.0)
        dt = round_to(DT_LO + (DT_HI - DT_LO) * torch.sigmoid(sample("normal", (B, S, H), seed + 155, dtype)), dtype)
        a = -(A_LO + (A_HI - A_LO) * torch.sigmoid(sample("normal", (H,), seed + 186, dtype)))
        cs = (dt.float().reshape(B, NC, CHUNK, H) * a).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()
        return {"x": mk((B, S, H, P), 0), "dt": dt, "dA_cumsum": cs, "Bmat": mk((B, S, N), 31), "Cmat": mk((B, S, N), 62),
                "Dskip": mk((H,), 124), "states": mk((B, NC, H, P, N), 93)}

    # 标准答案的 ops：f32 改成 double，因果掩码建在 CPU 上。
    class _RefOps(_TorchOps):
        f32 = staticmethod(lambda a: a.double())
        where = staticmethod(lambda m, a, b: torch.where(m, a, torch.zeros((), dtype=a.dtype)))
        causal = staticmethod(lambda n: torch.tril(torch.ones((n, n), dtype=torch.bool)))

    def reference(inp):
        y = ssd_chunk_scan(*(inp[k] for k in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip", "states")), ops=_RefOps, batch=B, seq=S, heads=H, headdim=P, dstate=N)
        return {"y": y}

    kw = dict(batch=B, seq=S, heads=H, headdim=P, dstate=N)

    def _mlx(*a):
        return ssd_chunk_scan(*a, ops=_MlxOps, **kw).astype(a[0].dtype)

    def _mlx_compile(*a):
        return mlx_compiled(f"mamba2_{size}", lambda mx: (lambda *t: ssd_chunk_scan(*t, ops=_MlxOps, **kw)))(*a).astype(a[0].dtype)

    def _mlx_half_compile(*a):  # matmul operands in the element type: the strongest framework-level formulation
        return mlx_compiled(f"mamba2_half_{size}", lambda mx: (lambda *t: ssd_chunk_scan(*t, ops=_MlxOps, half=True, **kw)))(*a).astype(a[0].dtype)

    def _torch(*a):
        return ssd_chunk_scan(*a, ops=_TorchOps, **kw).to(a[0].dtype)

    def _torch_half(*a):
        return ssd_chunk_scan(*a, ops=_TorchOps, half=True, **kw).to(a[0].dtype)

    rtol, atol_rel = default_tolerances(dtype)
    return Task(
        name="mamba2_chunk_scan", dtype=dtype, size=size, contract=contract,
        triton_source=load_text("mamba2_chunk_scan", "kernel_triton.py", dtype, {"BATCH": str(B), "SEQLEN": str(S), "NHEADS": str(H), "HEADDIM": str(P), "DSTATE": str(N), "CHUNK": str(CHUNK), "NCHUNKS": str(NC)}),
        description=(
            "Mamba-2 (SSD) chunk scan for one B/C group: per (batch, head, chunk) of CHUNK time steps, "
            "y[t] = sum_{s<=t} (C[t].B[s]) exp(cs[t]-cs[s]) dt[s] x[s] + exp(cs[t]) (C[t] . state[p]) + D x[t] (fp32 math, result rounded once); "
            "only the causal terms s <= t of the chunk exist (nothing is multiplied by a masked zero)."
        ),
        cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        gen_inputs=gen_inputs, reference=reference,
        baselines=(BaselineImpl("mlx", "mlx", _mlx), BaselineImpl("mlx_compile", "mlx", _mlx_compile), BaselineImpl("mlx_half_compile", "mlx", _mlx_half_compile),
                   BaselineImpl("torch_mps", "torch", _torch), BaselineImpl("torch_half", "torch", _torch_half)),
        # C·B 的 einsum 没有 head 轴，B/C 全 head 共享，这一项只乘 B·NC。对 x 和对 state 的两项按 head 算。
        flops=float(B * NC * (2 * CHUNK * CHUNK * N) + B * H * NC * (2 * CHUNK * CHUNK * P + 2 * CHUNK * P * N)),
        min_bytes=float(2 * (2 * B * S * H * P + B * S * H + B * H * S + 2 * B * S * N + H + B * NC * H * P * N)),
        rtol=rtol, atol_rel=atol_rel,
        naive_msl=load_text("mamba2_chunk_scan", "naive.metal", dtype),
        meta={"ieee_inputs": ("states",)},  # a special in x would make masked zeros (0 * Inf) legitimate-looking NaNs for tiled kernels; the states enter through a full dot
    )
