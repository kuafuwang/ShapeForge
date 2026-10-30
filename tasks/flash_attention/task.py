# 非因果注意力。语义只写在 opspec.json；默认 tile 与启动事实来自登记的 metal-flash-attention profile；
# 候选的 @<BLOCK_M>x<BLOCK_N>t<threads> 由登记的 variant factory 解析，只换 tile/threads。
from tasks._attention import build_attention

TASK_DIR = "flash_attention"


def build(dtype: str, size: str):
    """@brief 装配 FlashAttention 题（默认 tile）。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    """
    return build_attention(TASK_DIR, dtype, size)


def build_variant(dtype: str, size: str, variant: str):
    """@brief 同一注意力、同一 scale，只换 query/key tile 与 threads。

    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant ``<BLOCK_M>x<BLOCK_N>t<threads>``。
    @return 装配好的 ``Task``，名字带 ``@<variant>`` 后缀。
    """
    return build_attention(TASK_DIR, dtype, size, variant=variant)
