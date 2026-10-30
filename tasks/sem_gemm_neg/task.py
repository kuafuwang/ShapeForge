# 语义恢复题：题面说标准 GEMM，正文把累加写成 acc -= tl.dot(a, b)，即 C = -(A @ B)。
# 语义（负累加公式、shape/容差/记账）只写在 opspec.json；契约与启动事实来自登记的 metal-gemm
# profile；known-good 是本目录的负累加 Metal 文本。
from tasks._matmul import build_matmul

TASK_DIR = "sem_gemm_neg"


def build(dtype: str, size: str):
    """@brief 装配负累加的语义恢复题（C = -(A @ B)）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_matmul(
        TASK_DIR,
        dtype,
        size,
        epilogues={"negate": lambda c: -c},
        baseline_builders={},
    )
