# 语义恢复题：FlashAttention 的 qk 标度写成 SM_SCALE + log2(e)（和，不是积）。
# 语义（有效标度公式、shape/容差/记账）只写在 opspec.json；契约与启动事实来自登记的
# metal-flash-attention profile；题面与 known-good 是本目录的误导性文本。
from tasks._attention import build_attention

TASK_DIR = "sem_attention_odd_scale"


def build(dtype: str, size: str):
    """@brief 装配标度写成和的注意力语义恢复题。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_attention(TASK_DIR, dtype, size)
