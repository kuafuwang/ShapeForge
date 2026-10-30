# 矩阵乘向量。语义只写在 opspec.json；契约与启动事实来自登记的 metal-gemv profile。
import torch

from tcenv.distributions import sample
from tasks._assembly import build_task, derive

TASK_DIR = "gemv"


def _gen_inputs(params: dict, shape: tuple[int, int], dtype: str):
    """@brief 按 track 登记参数构造输入生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param shape ``(M, K)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"A", "x"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    spike = float(params["spike"])
    offsets = dict(params["seed_offsets"])
    m, k = shape

    def gen_inputs(case, seed):
        mag = mags.get(case, default_mag)
        return {
            "A": sample(case, (m, k), seed + int(offsets["A"]), dtype, mag=mag, large=large, spike=spike),
            "x": sample(case, (k,), seed + int(offsets["x"]), dtype, mag=mag, large=large, spike=spike),
        }

    return gen_inputs


def _reference(params: dict):
    """@brief ``A @ x`` 的 float64 参考实现。

    @param params ``implementations.reference.params``。
    @return ``inp -> {"y"}``。
    """
    if params.get("accumulate") != "float64" or params.get("operation") != "matmul_vector":
        raise ValueError(f"gemv reference registration changed: {params}")

    def reference(inp):
        return {"y": inp["A"].double() @ inp["x"].double()}

    return reference


def _mlx(params: dict):
    """@brief mlx 侧矩阵乘向量基线（``a @ x``）。

    @param params ``baselines[*].params``。
    @return ``(a, x) -> a @ x``。
    """
    if params.get("kernel") != "matmul":
        raise ValueError(f"gemv mlx baseline registration changed: {params}")
    return lambda a, x: a @ x


def _torch(params: dict):
    """@brief torch 侧矩阵乘向量基线（``torch.mv``）。

    @param params ``baselines[*].params``。
    @return ``(a, x) -> torch.mv(a, x)``。
    """
    if params.get("kernel") != "torch.mv":
        raise ValueError(f"gemv torch baseline registration changed: {params}")
    return lambda a, x: torch.mv(a, x)


def build(dtype: str, size: str):
    """@brief 装配矩阵乘向量题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    shape = (int(instance.symbols["M"]), int(instance.symbols["K"]))
    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), shape, dtype),
            "reference": _reference(dict(track.implementations["reference"]["params"])),
            "error_scale": None,
        },
        baseline_fns={
            "mlx": _mlx(dict(track.baselines[0]["params"])),
            "torch_mps": _torch(dict(track.baselines[1]["params"])),
        },
        derived=derived,
    )
