# 生成式逐元素题 ``gen_silu``。语义只在 opspec.json（公式＝闭集表达式），
# 题面/known-good 由登记的生成器从该文档渲染；启动事实来自登记的 metal-silu profile。
from tasks.generated.elementwise import build_generated

TASK_DIR = "generated/silu"


def build(dtype: str, size: str):
    """@brief 装配 ``gen_silu``。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_generated(TASK_DIR, dtype, size)
