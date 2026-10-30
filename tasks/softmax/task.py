# 按行 softmax。语义只写在 opspec.json；契约由登记的 Metal LoweringProfile 派生。
import torch

from tcenv.distributions import sample
from tasks._assembly import build_task, derive

TASK_DIR = "softmax"


def _gen_inputs(params: dict, shape: tuple[int, int], dtype: str):
    """@brief 按登记参数构造输入生成闭包。

    @param params track 里 ``implementations.gen_inputs.params``。
    @param shape ``(M, N)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"x"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])

    def gen_inputs(case, seed):
        return {"x": sample(case, shape, seed, dtype, mag=mags.get(case, default_mag), large=large)}

    return gen_inputs


def _reference(params: dict):
    """@brief 按登记参数构造参考实现。

    @param params track 里 ``implementations.reference.params``。
    @return ``inp -> {"y"}``。
    """
    if params["accumulate"] != "float64" or params["dim"] != "last":
        raise ValueError(f"softmax reference only implements float64 softmax on the last dim, got {params}")

    def reference(inp):
        return {"y": torch.softmax(inp["x"].double(), dim=-1)}

    return reference


def _mlx(params: dict):
    """@brief mlx 侧 softmax 基线（``precise=True``）。

    @param params 登记参数。
    @return ``x -> mx.softmax(x, axis=-1, precise=True)``。
    """
    if params["kernel"] != "mx.softmax" or params["axis"] != "last" or params["precise"] is not True:
        raise ValueError(f"softmax mlx baseline registration changed: {params}")

    def _mlx_softmax(x):
        import mlx.core as mx

        return mx.softmax(x, axis=-1, precise=True)

    return _mlx_softmax


def _torch(params: dict):
    """@brief torch 侧 softmax 基线。

    @param params 登记参数。
    @return ``x -> torch.softmax(x, dim=-1)``。
    """
    if params["kernel"] != "torch.softmax" or params["dim"] != "last":
        raise ValueError(f"softmax torch baseline registration changed: {params}")

    def _torch_softmax(x):
        return torch.softmax(x, dim=-1)

    return _torch_softmax


def build(dtype: str, size: str):
    """@brief 装配按行 softmax 题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    shape = (int(instance.symbols["M"]), int(instance.symbols["N"]))
    impls = track.implementations
    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(impls["gen_inputs"]["params"]), shape, dtype),
            "reference": _reference(dict(impls["reference"]["params"])),
            "error_scale": None,
        },
        baseline_fns={
            "mlx": _mlx(dict(track.baselines[0]["params"])),
            "torch_mps": _torch(dict(track.baselines[1]["params"])),
        },
        derived=derived,
    )
