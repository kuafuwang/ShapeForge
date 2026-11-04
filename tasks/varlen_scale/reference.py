"""varlen_scale: the independent oracle of the packed-varlen elementwise scale.

``y[t] = SCALE * x[t]`` over every token of the packed batch.  The operator is elementwise, so the
per-sequence boundaries carried by ``cu_seqlens`` do not change the math: every token is scaled on
its own.  The oracle stays in fp64 and never reads the workspace or a symbol.
"""

import torch


def reference(inputs, attrs):
    """@brief fp64 oracle of the packed-varlen elementwise scale.

    @param inputs 合同输入（``x`` 已按声明 dtype 舍入并加宽成 float32；``cu_seqlens`` 是分段偏移）。
    @param attrs 已解析的整数 attr（``SCALE``）。
    @return ``{"y": ...}``（fp64）。
    """
    x = inputs["x"].to(torch.float64)
    scale = float(attrs["SCALE"])
    return {"y": scale * x}


def gen_inputs(case, seed, instance):
    """@brief 生成打包 varlen 输入：cu_seqlens 张量 + 打包 x。

    cu_seqlens 由实例的 varlen 规约派生（首项 0、单调、末项 = total_tokens），x 按 total_tokens 采样。

    @param case 算例名（未用，varlen 只有 normal）。
    @param seed 随机种子。
    @param instance 语义实例（含 cu_seqlens 与 total_tokens/H/D symbols）。
    @return 输入字典。
    """
    generator = torch.Generator().manual_seed(int(seed))
    total_tokens = int(instance.symbols["total_tokens"])
    heads = int(instance.symbols["H"])
    dim = int(instance.symbols["D"])
    cu = torch.tensor(list(instance.cu_seqlens), dtype=torch.int32)
    x = torch.randn((total_tokens, heads, dim), generator=generator)
    return {"x": x, "cu_seqlens": cu}
