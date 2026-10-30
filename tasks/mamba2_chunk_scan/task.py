# Mamba-2 的 chunk scan，一个 B/C 组。语义在 opspec.json（含 CHUNK 符号与七个形状符号）；
# 契约与启动事实（核名、grid=(NCHUNKS, BATCH*NHEADS, 1)、threads、七个常量）来自登记的
# metal-mamba2-chunk-scan profile。这里只登记：输入生成参数、标准答案与五条基线的实现身份。
import torch

from tcenv.distributions import sample
from tcenv.tensors import round_to
from tasks._assembly import build_task, derive
from tasks._common import check_dtype, mlx_compiled

TASK_DIR = "mamba2_chunk_scan"
PROFILE_NAME = "metal-mamba2-chunk-scan"

def ssd_chunk_scan(x, dt, cs, bm, cm, d, st, *, ops, batch, seq, heads, headdim, dstate, chunk, half=False):
    """@brief 用框架算子写出 Mamba-2 的 chunk scan，库基线与标准答案共用这一式。

    The operator in framework ops (the formulation of the Mamba-2 paper, listing 1, for one group): this is what the vendor baselines run.
    ``ops`` supplies einsum / exp / where / tril so that the same code runs on MLX and torch. The decay factors and the sums are fp32.
    ``half=False`` upcasts every matmul operand to fp32; ``half=True`` keeps the matmul operands in the input dtype (what the production Triton
    kernels do: half-precision matmuls with fp32 accumulation inside the library, results rounded to the element type per product).

    @param x 输入，形状 ``[BATCH, SEQLEN, NHEADS, HEADDIM]``。
    @param dt 步长，``> 0``。
    @param cs 块内 dt*A 的含前缀和，``[BATCH, NHEADS, NchunkS, chunk]``。
    @param bm B 矩阵，``[BATCH, SEQLEN, DSTATE]``。
    @param cm C 矩阵，``[BATCH, SEQLEN, DSTATE]``。
    @param d 每 head 的 skip 连接，``[NHEADS]``。
    @param st 每个块开始时的 state，``[BATCH, NchunkS, NHEADS, HEADDIM, DSTATE]``。
    @param ops 算子集，提供 f32 / permute / like / einsum / exp / where / causal。
    @param batch 批大小。
    @param seq 序列长度。
    @param heads head 数。
    @param headdim head 维。
    @param dstate state 维。
    @param chunk 每个 chunk 的时间步数（来自 OpSpec 的 ``chunk`` 符号）。
    @param half 矩阵乘操作数是否保持输入 dtype。
    @return 输出，形状 ``[BATCH, SEQLEN, NHEADS, HEADDIM]``。
    """
    nc = seq // chunk
    f = ops.f32
    mm = (lambda a: a) if half else f  # operand of a matmul
    x5 = x.reshape(batch, nc, chunk, heads, headdim)
    dt4 = ops.permute(f(dt).reshape(batch, nc, chunk, heads), (0, 3, 1, 2))  # [B, H, NC, L]
    cs4 = f(cs)  # [B, H, NC, L]
    br = bm.reshape(batch, nc, chunk, dstate)
    cr = cm.reshape(batch, nc, chunk, dstate)
    cb = f(ops.einsum("bcln,bcsn->bcls", mm(cr), mm(br)))  # C[t] . B[s]
    decay = ops.exp(cs4[..., :, None] - cs4[..., None, :])  # exp(cs[t] - cs[s])
    w = ops.where(ops.causal(chunk), cb[:, None] * decay * dt4[..., None, :], 0.0)  # [B, H, NC, L, L]
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


def _gen_inputs(params: dict, shape: tuple[int, ...], dtype: str):
    """@brief 由 track 登记参数构造输入生成闭包（dt 与衰减按登记的区间构造）。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(BATCH, SEQLEN, NHEADS, HEADDIM, DSTATE)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> 输入字典``。
    """
    B, S, H, P, N = shape
    mags = {str(key): float(value) for key, value in params["mags"].items()}
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    spike = float(params["spike"])
    offsets = {str(key): int(value) for key, value in params["seed_offsets"].items()}
    dt_lo, dt_hi = (float(value) for value in params["dt_range"])
    decay_lo, decay_hi = (float(value) for value in params["decay_range"])
    chunk = int(params["chunk"])

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        mag = mags.get(case, default_mag)

        def make(shape: tuple[int, ...], offset: int):
            return sample(case, shape, seed + offset, dtype, mag=mag, large=large, spike=spike)

        nc = S // chunk
        dt = round_to(dt_lo + (dt_hi - dt_lo) * torch.sigmoid(sample("normal", (B, S, H), seed + offsets["dt"], dtype)), dtype)
        a = -(decay_lo + (decay_hi - decay_lo) * torch.sigmoid(sample("normal", (H,), seed + offsets["decay"], dtype)))
        cs = (dt.float().reshape(B, nc, chunk, H) * a).cumsum(dim=2).permute(0, 3, 1, 2).contiguous()
        return {
            "x": make((B, S, H, P), offsets["x"]),
            "dt": dt,
            "dA_cumsum": cs,
            "Bmat": make((B, S, N), offsets["Bmat"]),
            "Cmat": make((B, S, N), offsets["Cmat"]),
            "Dskip": make((H,), offsets["Dskip"]),
            "states": make((B, nc, H, P, N), offsets["states"]),
        }

    return gen_inputs


class _RefOps(_TorchOps):
    """@brief 标准答案的算子集：``f32`` 升到 double，因果掩码建在 CPU 上。"""

    f32 = staticmethod(lambda a: a.double())
    where = staticmethod(lambda m, a, b: torch.where(m, a, torch.zeros((), dtype=a.dtype)))
    causal = staticmethod(lambda n: torch.tril(torch.ones((n, n), dtype=torch.bool)))


def build(dtype: str, size: str):
    """@brief 装配 Mamba-2 chunk scan 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    B, S, H, P, N, chunk = (
        int(instance.symbols[name])
        for name in ("BATCH", "SEQLEN", "NHEADS", "HEADDIM", "DSTATE", "CHUNK")
    )
    nc = int(instance.symbols["NCHUNKS"])
    if S % chunk or nc != S // chunk:
        raise ValueError(f"{TASK_DIR}: CHUNK={chunk}/SEQLEN={S}/NCHUNKS={nc} is not a valid chunking")
    shape = (B, S, H, P, N)
    params = dict(track.implementations["gen_inputs"]["params"])
    params["chunk"] = chunk
    kw = dict(batch=B, seq=S, heads=H, headdim=P, dstate=N, chunk=chunk)

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        y = ssd_chunk_scan(
            *(inp[key] for key in ("x", "dt", "dA_cumsum", "Bmat", "Cmat", "Dskip", "states")),
            ops=_RefOps,
            **kw,
        )
        return {"y": y}

    def baseline_fns() -> dict:
        out = {}
        for entry in track.baselines:
            name = str(entry["name"])
            params_ = dict(entry["params"])
            framework = str(entry["framework"])
            half = bool(params_["half"])
            compile_mode = params_["compile"]
            if str(params_["ops"]) != "ssd_chunk_scan":
                raise ValueError(f"{TASK_DIR}: unregistered baseline formulation {params_['ops']!r}")
            if compile_mode not in (None, "mlx"):
                raise ValueError(f"{TASK_DIR}: unregistered compile mode {compile_mode!r}")
            if framework == "mlx":
                if compile_mode == "mlx":
                    out[name] = _mlx_compiled_baseline(kw, size, half)
                else:
                    def f(*a, _half=half):
                        return ssd_chunk_scan(*a, ops=_MlxOps, half=_half, **kw).astype(a[0].dtype)
                    out[name] = f
            elif framework == "torch":
                def g(*a, _half=half):
                    return ssd_chunk_scan(*a, ops=_TorchOps, half=_half, **kw).to(a[0].dtype)
                out[name] = g
            else:
                raise ValueError(f"{TASK_DIR}: unregistered baseline framework {framework!r}")
        return out

    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(params, shape, dtype),
            "reference": reference,
            "error_scale": None,
        },
        baseline_fns=baseline_fns(),
        derived=derived,
    )


def _mlx_compiled_baseline(kw: dict, size: str, half: bool):
    """@brief 造一条 ``mx.compile`` 基线（解码式与矩阵乘仍是同一条式子）。

    @param kw ``ssd_chunk_scan`` 的形状关键字。
    @param size 尺寸档位（编译缓存键）。
    @param half 矩阵乘操作数是否保持输入 dtype。
    @return 基线闭包。
    """

    def f(*a):
        return mlx_compiled(
            f"mamba2_{'half_' if half else ''}{size}",
            lambda mx: (lambda *t: ssd_chunk_scan(*t, ops=_MlxOps, half=half, **kw)),
        )(*a).astype(a[0].dtype)

    return f


