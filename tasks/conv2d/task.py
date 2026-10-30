# 3x3 卷积。语义只写在 opspec.json；契约与启动事实来自登记的 metal-conv2d profile。
import torch
import torch.nn.functional as F

from tcenv.distributions import sample
from tasks._assembly import build_task, derive

TASK_DIR = "conv2d"


def _gen_inputs(params: dict, shapes: dict, dtype: str):
    """@brief 按 track 登记参数构造 x/w 生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param shapes 输入名 → 形状。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"x", "w"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    spike = float(params["spike"])
    offsets = dict(params["seed_offsets"])

    def gen_inputs(case, seed):
        mag = mags.get(case, default_mag)
        return {
            name: sample(case, shapes[name], seed + int(offsets[name]), dtype, mag=mag, large=large, spike=spike)
            for name in ("x", "w")
        }

    return gen_inputs


def _reference(params: dict, dims: dict):
    """@brief 卷积参考实现：有限输入走 ``F.conv2d``，含 NaN/Inf 时逐 tap 累加。

    @param params ``implementations.reference.params``。
    @param dims ``N/OH/OW/OC/KH/KW``。
    @return ``inp -> {"y"}``。
    """
    if params.get("accumulate") != "double_output" or params.get("finite") != "conv2d" or params.get("non_finite") != "tap_sum":
        raise ValueError(f"conv2d reference registration changed: {params}")
    n, oh, ow, oc = dims["N"], dims["OH"], dims["OW"], dims["OC"]
    kh, kw, kc = dims["KH"], dims["KW"], dims["C"]

    def reference(inp):
        if not (torch.isfinite(inp["x"]).all() and torch.isfinite(inp["w"]).all()):
            # CPU conv2d may use a Winograd transform, which smears one Inf over a whole tile as NaN;
            # IEEE semantics are those of the sum of products, so take the same fp32 sum tap by tap.
            x, w = inp["x"].float(), inp["w"].float()
            y = torch.zeros((n, oh, ow, oc), dtype=torch.float32)
            for i in range(kh):
                for j in range(kw):
                    y += x[:, i:i + oh, j:j + ow, :] @ w[:, i, j, :].T
            return {"y": y.double()}
        x = inp["x"].permute(0, 3, 1, 2).contiguous().float()
        w = inp["w"].permute(0, 3, 1, 2).contiguous().float()
        return {"y": F.conv2d(x, w).permute(0, 2, 3, 1).contiguous().double()}

    return reference


def _mlx(params: dict):
    """@brief mlx 侧卷积基线（``mx.conv2d``）。

    @param params track 登记参数。
    @return ``(x, w) -> mx.conv2d(x, w)``。
    """
    if params.get("kernel") != "mx.conv2d":
        raise ValueError(f"conv2d mlx baseline registration changed: {params}")

    def _mlx(x, w):
        import mlx.core as mx

        return mx.conv2d(x, w)

    return _mlx


def _torch(params: dict):
    """@brief torch 侧卷积基线：NHWC→NCHW→conv2d→NHWC。

    @param params track 登记参数。
    @return ``(x, w) -> NHWC``。
    """
    if params.get("kernel") != "torch.nn.functional.conv2d" or params.get("layout") != "nchw_roundtrip":
        raise ValueError(f"conv2d torch baseline registration changed: {params}")
    return lambda x, w: F.conv2d(x.permute(0, 3, 1, 2), w.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def build(dtype: str, size: str):
    """@brief 装配 3x3 卷积题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    shapes = {"x": instance.tensor("x").shape, "w": instance.tensor("w").shape}
    dims = {name: int(instance.symbols[name]) for name in ("N", "H", "W", "C", "OC", "KH", "KW", "OH", "OW")}
    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), shapes, dtype),
            "reference": _reference(dict(track.implementations["reference"]["params"]), dims),
            "error_scale": None,
        },
        baseline_fns={
            "mlx": _mlx(dict(track.baselines[0]["params"])),
            "torch_mps": _torch(dict(track.baselines[1]["params"])),
        },
        derived=derived,
    )
