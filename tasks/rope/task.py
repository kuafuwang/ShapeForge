# 旋转位置编码。语义（BASE、公式、shape/容差/记账）只写在 opspec.json；
# 契约与启动事实来自登记的 metal-rope profile；这里只装配表示与 oracle 闭包。
import torch

from tcenv.distributions import sample
from tasks._assembly import build_task, derive

TASK_DIR = "rope"


def _gen_inputs(params: dict, shape: tuple[int, int, int, int], dtype: str):
    """@brief 按 track 登记参数构造输入生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(B, H, S, D)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"x"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])

    def gen_inputs(case, seed):
        return {"x": sample(case, shape, seed, dtype, mag=mags.get(case, default_mag), large=large)}

    return gen_inputs


def _reference(params: dict, base: float, seq: int, head_dim: int):
    """@brief 半分割旋转参考实现（float64）。

    @param params ``implementations.reference.params``。
    @param base 已物化的 ``BASE`` attr。
    @param seq 序列长度符号 ``S``。
    @param head_dim 头维度符号 ``D``。
    @return ``inp -> {"y"}``。
    """
    if params.get("accumulate") != "float64" or params.get("base_attr") != "BASE" or params.get("layout") != "half_split":
        raise ValueError(f"rope reference registration changed: {params}")

    def reference(inp):
        x = inp["x"].double()
        half = head_dim // 2
        inv_freq = base ** (-torch.arange(half, dtype=torch.float64) / half)
        ang = torch.arange(seq, dtype=torch.float64)[:, None] * inv_freq[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        x1, x2 = x[..., :half], x[..., half:]
        return {"y": torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)}

    return reference


def _mlx(params: dict, base: float):
    """@brief mlx 侧旋转位置编码基线（``mx.fast.rope``）。

    @param params ``baselines[0].params``。
    @param base 已物化的 ``BASE`` attr。
    @return ``x -> mx.fast.rope(...)``。
    """
    if (
        params.get("kernel") != "mx.fast.rope"
        or params.get("traditional") is not False
        or params.get("base_attr") != "BASE"
        or float(params.get("scale", 0.0)) != 1.0
        or int(params.get("offset", -1)) != 0
    ):
        raise ValueError(f"rope mlx baseline registration changed: {params}")

    def _mlx_rope(x):
        import mlx.core as mx

        return mx.fast.rope(x, x.shape[-1], traditional=False, base=base, scale=1.0, offset=0)

    return _mlx_rope


def build(dtype: str, size: str):
    """@brief 装配旋转位置编码题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    shape = tuple(int(instance.symbols[name]) for name in ("B", "H", "S", "D"))
    base = float(instance.resolved_attrs["BASE"])
    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), shape, dtype),
            "reference": _reference(
                dict(track.implementations["reference"]["params"]),
                base,
                int(instance.symbols["S"]),
                int(instance.symbols["D"]),
            ),
            "error_scale": None,
        },
        baseline_fns={"mlx": _mlx(dict(track.baselines[0]["params"]), base)},
        derived=derived,
    )
