# BitDelta 矩阵乘。语义（含 SCALE attr）在 opspec.json；契约与启动事实（核名、grid、threads、
# 六个常量、位于 2 号槽的 fp32 标量 scale）来自登记的 metal-bitdelta-gemm profile。
# 这里只登记：输入生成参数、标准答案（解包 + fp64 累加）与四条基线的实现身份。
import torch

from tcenv.distributions import sample
from tcenv.track import REPO_ROOT
from tasks._assembly import build_task, require_attr
from tasks._common import check_dtype, mlx_compiled, torch_compiled

TASK_DIR = "bitdelta_gemm"
#: 诊断键（不是语义/启动事实）：基线与标准答案的差别在哪。
BASELINE_NOTE = (
    "unfused unpack + dense matmul (no fused library kernel exists); "
    "torch_compile (Inductor on Metal) fuses the decode and is the strong baseline"
)
PROFILE_NAME = "metal-bitdelta-gemm"
TRACK_PATH = REPO_ROOT / "tasks" / TASK_DIR / "track.json"


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


def _gen_inputs(params: dict, shape: tuple[int, int, int, int], dtype: str):
    """@brief 由 track 登记参数构造输入生成闭包（bits 用打包符号字，x 按 case 采样）。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(M, N, K, KB)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"x", "bits"}``。
    """
    M, N, K, KB = shape
    seed_mask = int(params["seed_mask"])
    edge_values = torch.tensor([int(value) for value in params["edge_words"]], dtype=torch.int64)
    bits_case = str(params["bits_case"])
    case_map = {str(key): str(value) for key, value in params["x_case_map"].items()}
    x_offset = int(params["x_seed_offset"])
    mag = float(params["x_mag"])
    large = float(params["x_large"])
    spike = float(params["x_spike"])

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed & seed_mask)
        if case == bits_case:
            words = edge_values[torch.randint(0, len(edge_values), (KB, N), generator=generator)]
        else:
            words = torch.randint(-(1 << 31), 1 << 31, (KB, N), generator=generator, dtype=torch.int64)
        xcase = case_map.get(case, case)
        return {
            "x": sample(xcase, (M, K), seed + x_offset, dtype, mag=mag, large=large, spike=spike),
            "bits": words,
        }

    return gen_inputs


def build(dtype: str, size: str):
    """@brief 装配 BitDelta 矩阵乘题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    check_dtype(dtype)
    from tasks._assembly import derive

    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    M, N, K, KB = (int(instance.symbols[name]) for name in ("M", "N", "K", "KB"))
    if KB * 32 != K:
        raise ValueError(f"{TASK_DIR}: the packed-sign tensor needs KB*32 == K, got KB={KB} K={K}")
    scale = require_attr(instance, "SCALE")

    implementations = {
        "gen_inputs": _gen_inputs(
            dict(track.implementations["gen_inputs"]["params"]), (M, N, K, KB), dtype
        ),
        "reference": None,
        "error_scale": None,
    }

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        w = unpack_pm1(inp["bits"])
        return {"y": (inp["x"].float() @ w).double() * scale}

    implementations["reference"] = reference

    baseline_fns = {}
    for entry in track.baselines:
        name = str(entry["name"])
        params = dict(entry["params"])
        decode = str(params["decode"])
        if decode not in ("separate", "compiled"):
            raise ValueError(f"{TASK_DIR}: unregistered decode mode {decode!r} for baseline {name}")
        if name in ("mlx", "mlx_compile"):
            framework = "mlx"
        elif name in ("torch_mps", "torch_compile"):
            framework = "torch"
        else:
            raise ValueError(f"{TASK_DIR}: unregistered baseline {name!r}")
        if framework != str(entry["framework"]):
            raise ValueError(f"{TASK_DIR}: baseline {name} declares framework {entry['framework']!r}")
        baseline_fns[name] = _baseline(framework, decode == "compiled", K, N, scale)

    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations=implementations,
        baseline_fns=baseline_fns,
        derived=derived,
        meta_extras={"baseline_note": BASELINE_NOTE},
    )


def _baseline(framework: str, compile_decode: bool, K: int, N: int, scale: float):
    """@brief 造一条登记的 BitDelta 基线（解码 + 稠密乘，可选把解码收进一次编译）。

    @param framework ``mlx`` 或 ``torch``。
    @param compile_decode 解码是否编译。
    @param K 缩减维长度。
    @param N 输出列数。
    @param scale 语义 attr ``SCALE``。
    @return ``(x, bits)`` 基线闭包。
    """
    if framework == "mlx":
        def mlx_decode(x, bits):
            import mlx.core as mx

            j = mx.arange(32, dtype=mx.uint32).reshape(1, 32, 1)
            b = (bits.reshape(K // 32, 1, N) >> j) & 1
            w = (b.astype(x.dtype) * 2 - 1).reshape(K, N)
            return (x @ w) * scale

        if not compile_decode:
            return mlx_decode

        def mlx_compiled_decode(x, bits):
            return mlx_compiled(f"bitdelta_{K}_{N}", lambda mx: mlx_decode)(x, bits)

        return mlx_compiled_decode

    def torch_decode(x, bits):
        j = torch.arange(32, dtype=torch.int32, device=x.device).reshape(1, 32, 1)
        b = (bits.reshape(K // 32, 1, N) >> j) & 1
        w = (b.to(x.dtype) * 2 - 1).reshape(K, N)
        return (x @ w) * scale

    if not compile_decode:
        return torch_decode

    def torch_compiled_decode(x, bits):
        return torch_compiled(f"bitdelta_{K}_{N}", torch_decode)(x, bits)

    return torch_compiled_decode


__all__ = ["TASK_DIR", "PROFILE_NAME", "TRACK_PATH", "build", "unpack_pm1"]
