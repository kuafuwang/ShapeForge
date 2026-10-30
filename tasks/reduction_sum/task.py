# 按行求和。语义只写在 opspec.json；契约由登记的 Metal LoweringProfile 派生。
import torch

from tcenv.distributions import sample
from tasks._assembly import build_task, derive

TASK_DIR = "reduction_sum"


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
    if params["accumulate"] != "float64" or params["axis"] != "last":
        raise ValueError(f"reduction_sum reference only implements float64 sum on the last axis, got {params}")

    def reference(inp):
        return {"y": inp["x"].double().sum(-1)}

    return reference


def _error_scale(params: dict):
    """@brief 按登记参数构造 error scale（每行 ``sum |x|``）。

    @param params 登记参数。
    @return ``(inp, ref) -> {"y"}``。
    """
    if params["scale"] != "sum_abs_rows":
        raise ValueError(f"reduction_sum error scale registration changed: {params}")

    def error_scale(inp, ref):
        # 正确的 fp32 求和，误差跟着 sum(|x|) 走，而不是 |sum x|（抵消）。
        return {"y": inp["x"].double().abs().sum(-1)}

    return error_scale


def _mlx(params: dict):
    """@brief mlx 侧按行求和基线。

    @param params 登记参数。
    @return ``x -> mx.sum(x, axis=-1)``。
    """
    if params["kernel"] != "mx.sum" or params["axis"] != "last":
        raise ValueError(f"reduction_sum mlx baseline registration changed: {params}")

    def _mlx_sum(x):
        import mlx.core as mx

        return mx.sum(x, axis=-1)

    return _mlx_sum


def _torch(params: dict):
    """@brief torch 侧按行求和基线。

    @param params 登记参数。
    @return ``x -> torch.sum(x, dim=-1)``。
    """
    if params["kernel"] != "torch.sum" or params["dim"] != "last":
        raise ValueError(f"reduction_sum torch baseline registration changed: {params}")

    def _torch_sum(x):
        return torch.sum(x, dim=-1)

    return _torch_sum


def build(dtype: str, size: str):
    """@brief 装配按行求和题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper``、``small`` 或 ``plan``。
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
            "error_scale": _error_scale(dict(impls["error_scale"]["params"])),
        },
        baseline_fns={
            "mlx": _mlx(dict(track.baselines[0]["params"])),
            "torch_mps": _torch(dict(track.baselines[1]["params"])),
        },
        derived=derived,
    )
