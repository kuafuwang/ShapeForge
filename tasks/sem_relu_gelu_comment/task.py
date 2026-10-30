# 语义恢复题：注释/文档字符串写成 GELU，函数名是 relu_kernel，正文真正执行的是 y = x > 0 ? x : 0。
# 语义（实际执行的 ReLU 写法、shape/容差/记账）只写在 opspec.json；契约与启动事实来自登记的
# metal-relu profile；题面与 known-good 是本目录的误导性文本，这里只装配表示与 oracle 闭包。
import torch

from tasks._pointwise import build_pointwise

TASK_DIR = "sem_relu_gelu_comment"


def _relu_as_written(x: torch.Tensor) -> torch.Tensor:
    """@brief 按题面写的那样算 ``x > 0 ? x : 0``（NaN 与负零都变成 +0）。

    ``torch.relu`` keeps NaN and negative zero, so it is not this specification.

    @param x 输入张量，已转成 float64。
    @return 比较写法的结果。
    """
    return torch.where(x > 0, x, torch.zeros_like(x))


def build(dtype: str, size: str):
    """@brief 装配名字/注释误导的语义恢复题（实际语义是 ReLU）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_pointwise(
        TASK_DIR,
        dtype,
        size,
        references={"relu_as_written": _relu_as_written},
        baseline_builders={},
    )
