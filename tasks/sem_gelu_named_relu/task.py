# 语义恢复题：注释写成 ReLU、函数名是 relu_kernel，正文真正执行的是 tanh 近似 GELU。
# 语义（GELU 的常数与公式、shape/容差/记账）只写在 opspec.json；契约与启动事实来自登记的
# metal-relu profile；题面与 known-good 是本目录的误导性文本。
import torch

from tasks._assembly import build_task, derive, require_attr

TASK_DIR = "sem_gelu_named_relu"


def build(dtype: str, size: str):
    """@brief 装配名字/注释误导的语义恢复题（实际语义是 tanh 近似 GELU）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    from tasks._pointwise import gen_inputs_from_params

    derived = derive(TASK_DIR, dtype, size)
    track, instance, _specialization = derived
    c = require_attr(instance, "GELU_C")
    k = require_attr(instance, "GELU_K")
    names = tuple(binding.name for binding in instance.tensors if binding.role == "input")
    shapes = {name: instance.tensor(name).shape for name in names}

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = inp["x"].to(torch.float64)
        return {"y": 0.5 * x * (1.0 + torch.tanh(c * (x + k * x**3)))}

    return build_task(
        TASK_DIR,
        dtype,
        size,
        implementations={
            "gen_inputs": gen_inputs_from_params(dict(track.implementations["gen_inputs"]["params"]), names, shapes, dtype),
            "reference": reference,
            "error_scale": None,
        },
        baseline_fns={},
        derived=derived,
    )
