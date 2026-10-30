# Lion 优化器一步。语义（LR/WD/BETA1/BETA2、公式、容差、记账）只写在 opspec.json；
# 契约与启动事实来自登记的 metal-lion profile；这里只装配表示、oracle 与 error scale。
import torch

from tcenv.distributions import sample
from tasks._assembly import build_task, derive, require_attr

TASK_DIR = "lion"


def _gen_inputs(params: dict, n: int, dtype: str):
    """@brief 按 track 登记参数构造输入生成闭包。

    @param params ``implementations.gen_inputs.params``。
    @param n 元素个数。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"p", "g", "m"}``。
    """
    names = tuple(params["inputs"])
    mag = float(params["mag"])
    large = float(params["large"])
    seed_offset = int(params["seed_offset"])
    tiny_scale = float(params["tiny_non_fp16_scale"])

    def gen_inputs(case, seed):
        out = {name: sample(case, (n,), seed + seed_offset * index, dtype, mag=mag, large=large)
               for index, name in enumerate(names)}
        if case == "tiny" and dtype != "fp16":
            # fp32 subnormals are flushed by the GPU: keep (1-beta)*x in the normal range, otherwise
            # sign(beta1*m + (1-beta1)*g) legitimately flips for every implementation.
            out = {key: value * tiny_scale for key, value in out.items()}
        return out

    return gen_inputs


def _reference(params: dict, attrs: dict):
    """@brief Lion 一步的 float64 参考实现。

    @param params ``implementations.reference.params``。
    @param attrs 已物化的 ``LR``/``WD``/``BETA1``/``BETA2``。
    @return ``inp -> {"p_out", "m_out"}``。
    """
    if params.get("accumulate") != "float64" or list(params.get("attrs", ())) != ["LR", "WD", "BETA1", "BETA2"]:
        raise ValueError(f"lion reference registration changed: {params}")
    lr, wd, beta1, beta2 = attrs["LR"], attrs["WD"], attrs["BETA1"], attrs["BETA2"]

    def reference(inp):
        p, g, m = (inp[k].double() for k in ("p", "g", "m"))
        u = torch.sign(beta1 * m + (1 - beta1) * g)
        return {"p_out": p * (1 - lr * wd) - lr * u, "m_out": beta2 * m + (1 - beta2) * g}

    return reference


def _error_scale(params: dict, attrs: dict, atol_rel: float):
    """@brief ``sign`` 间断点放宽的误差尺度（与冻结实现逐字节一致）。

    @param params ``implementations.error_scale.params``。
    @param attrs 已物化的 Lion attrs。
    @param atol_rel 本实例物化的 ``atol_rel``。
    @return ``(inp, ref) -> {"p_out", "m_out"}``。
    """
    rel = float(params["discontinuity_rel"])
    steps = float(params["atol_steps"])
    lr, beta1, beta2 = attrs["LR"], attrs["BETA1"], attrs["BETA2"]

    def error_scale(inp, ref):
        p, g, m = (inp[k].double() for k in ("p", "g", "m"))
        a, b = beta1 * m, (1 - beta1) * g
        amb = (a + b).abs() <= rel * (a.abs() + b.abs())
        base = torch.sqrt(torch.mean(ref["p_out"] ** 2)).expand_as(ref["p_out"]).clone()
        base[amb] = steps * lr / atol_rel
        return {"p_out": base, "m_out": torch.sqrt(torch.mean(ref["m_out"] ** 2))}

    return error_scale


def _mlx(params: dict, attrs: dict):
    """@brief mlx 侧 Lion 一步（``mx.compile`` 整步融合）。

    @param params ``baselines[*].params``。
    @param attrs 已物化的 Lion attrs。
    @return ``(p, g, m) -> (p_out, m_out)``。
    """
    from tasks._common import mlx_compiled

    if params.get("kernel") != "mx.compile" or params.get("expression") != "simplified_lion_step":
        raise ValueError(f"lion mlx baseline registration changed: {params}")
    lr, wd, beta1, beta2 = attrs["LR"], attrs["WD"], attrs["BETA1"], attrs["BETA2"]

    def build(mx):
        def step(p, g, m):
            u = mx.sign(beta1 * m + (1 - beta1) * g)
            return p * (1 - lr * wd) - lr * u, beta2 * m + (1 - beta2) * g

        return step

    return lambda p, g, m: mlx_compiled("lion", build)(p, g, m)


def _torch_eager(attrs: dict):
    """@brief 未融合的 torch eager Lion 一步。

    @param attrs 已物化的 Lion attrs。
    @return ``(p, g, m) -> (p_out, m_out)``。
    """
    lr, wd, beta1, beta2 = attrs["LR"], attrs["WD"], attrs["BETA1"], attrs["BETA2"]

    def step(p, g, m):
        u = torch.sign(beta1 * m + (1 - beta1) * g)
        return p * (1 - lr * wd) - lr * u, beta2 * m + (1 - beta2) * g

    return step


def _torch_mps(params: dict, attrs: dict):
    """@brief torch eager 基线。

    @param params ``baselines[*].params``。
    @param attrs 已物化的 Lion attrs。
    @return 实现闭包。
    """
    if params.get("kernel") != "eager_ops":
        raise ValueError(f"lion torch baseline registration changed: {params}")
    return _torch_eager(attrs)


def _torch_compile(params: dict, attrs: dict):
    """@brief ``torch.compile`` 融合基线（Inductor 的 Metal 后端把整步收成一个核）。

    @param params ``baselines[*].params``。
    @param attrs 已物化的 Lion attrs。
    @return 实现闭包。
    """
    from tasks._common import torch_compiled

    if params.get("kernel") != "torch.compile" or params.get("expression") != "eager_ops":
        raise ValueError(f"lion torch_compile baseline registration changed: {params}")
    eager = _torch_eager(attrs)
    return lambda p, g, m: torch_compiled("lion", eager)(p, g, m)


def build(dtype: str, size: str):
    """@brief 装配 Lion 优化器一步题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    n = int(instance.symbols["N"])
    attrs = {name: require_attr(instance, name) for name in ("LR", "WD", "BETA1", "BETA2")}
    impls = track.implementations
    builders = {"mlx": _mlx, "torch_mps": _torch_mps, "torch_compile": _torch_compile}
    baseline_fns = {
        str(entry["name"]): builders[str(entry["name"])](dict(entry["params"]), attrs) for entry in track.baselines
    }
    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(impls["gen_inputs"]["params"]), n, dtype),
            "reference": _reference(dict(impls["reference"]["params"]), attrs),
            "error_scale": _error_scale(
                dict(impls["error_scale"]["params"]), attrs, float(instance.tolerances.atol_rel)
            ),
        },
        baseline_fns=baseline_fns,
        derived=derived,
    )
