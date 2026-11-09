# 目标
把算子 `gemm` 写成 Metal，dtype `fp16`，规模 `small`。
C = A @ B with row-major A[M,K], B[K,N], C[M,N]; fp32 accumulation, result rounded once.

# Task: gemm (linear_algebra)
# Instance: dtype=fp16 size=small
# Summary: C[m,n] = sum_k A[m,k] * B[k,n]

## Tensors (declaration order)
- A: role=input dtype=fp16 shape=[256, 128] layout=row_major strides=[128, 1] space=global access=read
- B: role=input dtype=fp16 shape=[128, 256] layout=row_major strides=[256, 1] space=global access=read
- C: role=output dtype=fp16 shape=[256, 256] layout=row_major strides=[256, 1] space=global access=write

## Attrs (resolved)
(none)

## Cases: normal, wide_range, large, tiny, outlier, ieee
## Tolerances: rtol=0.002 atol_rel=0.001 exact=False
## Work accounting: flops=16777216.0 min_bytes=262144.0
## Discovery: {'phase': 1, 'default_instance': {'dtype': 'fp16', 'size': 'small'}}

规格是上面的 OpSpec 题面与派生 ABI；没有 Triton 源。下面的只读参考实现是判分 oracle，不许 import，也不许抄进候选：
```python
"""gemm: independent fp64 reference and the frozen G0 input stream.

The module is the whole oracle of the task.  It may not import the harness (``tasks``/
``tcenv``) or any candidate, so the v1 sampler and the harness IEEE injection are carried
as the frozen block below; A5 verify checks that this block is identical in every
migrated reference.
"""

import torch


# ---- frozen G0 sampler block (byte-identical in every A5-migrated reference) ----------------
import math

_FLOAT_MAX = {"fp16": 65504.0, "bf16": 3.3895313892515355e38, "fp32": 3.4028234663852886e38}
_TINY_NORMAL = {"fp16": 2.0**-14, "bf16": 2.0**-126, "fp32": 2.0**-126}
_TINY_SUBNORMAL = {"fp16": 2.0**-24, "bf16": 2.0**-133, "fp32": 2.0**-149}
_SPECIALS = (float("nan"), float("inf"), float("-inf"), -0.0)
_IEEE_SEED_STRIDE = 104_729


def _generator(seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)
    return g


def _sample(case, shape, seed, dtype, *, mag=1.0, large=None, spike=100.0):
    """The frozen v1 adversarial sampler (normal/wide_range/large/tiny/outlier/cancel/const)."""
    g = _generator(seed)
    shape = tuple(int(s) for s in shape)
    dmax = _FLOAT_MAX.get(dtype, 3.0e38)
    if case == "normal":
        return torch.randn(shape, generator=g) * mag
    if case == "wide_range":
        e = torch.rand(shape, generator=g) * 5 - 3
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e) * mag
    if case == "large":
        top = large if large is not None else min(dmax * 0.25, 1.0e4)
        return (torch.rand(shape, generator=g) * 2 - 1) * top
    if case == "tiny":
        if dtype == "fp16":
            lo, hi = _TINY_SUBNORMAL["fp16"], _TINY_NORMAL["fp16"] * 4
        else:
            lo = _TINY_NORMAL.get(dtype, 1.2e-38) * 2.0
            hi = lo * 4096.0
        e = torch.rand(shape, generator=g) * (math.log10(hi) - math.log10(lo)) + math.log10(lo)
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        return s * torch.pow(10.0, e)
    if case == "outlier":
        x = torch.randn(shape, generator=g) * mag
        mask = torch.rand(shape, generator=g) < 1e-3
        return torch.where(mask, x * spike, x)
    if case == "cancel":
        x = torch.randn(shape, generator=g) * mag
        big = (torch.rand(shape, generator=g) * 0.5 + 0.5) * 512.0
        s = torch.where(torch.rand(shape, generator=g) < 0.5, -1.0, 1.0)
        v = big * s
        if shape and shape[-1] % 2 == 0:
            vv = v.reshape(*shape[:-1], shape[-1] // 2, 2)
            vv[..., 1] = -vv[..., 0]
            v = vv.reshape(shape)
        return x + v
    if case == "const":
        return torch.full(shape, 1.0 * mag if mag != 1.0 else 0.75)
    raise ValueError("unknown case " + repr(case))


def _inject(t, seed, max_n=None):
    """The frozen v1 IEEE injection (``prepare_case`` applied it after sampling)."""
    flat = t.reshape(-1).clone()
    numel = flat.numel()
    last = t.shape[-1] if t.dim() else 1
    rows = max(1, numel // max(1, last))
    n = int(min(256, max(8, 0.1 * rows, numel / 65536), max(1, numel // 4)))
    if max_n is not None:
        n = max(2, min(n, max_n))
    perm = torch.randperm(numel, generator=_generator(seed))
    perm = perm[(perm != 0) & (perm != numel - 1)]
    pos = torch.cat([torch.tensor([0, numel - 1]), perm[: n - 2]])[:n]
    kinds = torch.randperm(n, generator=_generator(seed + 1)) % len(_SPECIALS)
    flat[pos] = torch.tensor([_SPECIALS[int(k)] for k in kinds], dtype=flat.dtype)
    return flat.reshape(t.shape)
# ---- end frozen block ----------------------------------------------------------------------


_MAGS = {"normal": 1.0, "wide_range": 0.1}
_DEFAULT_MAG = 1.0
_LARGE = 20.0
_SPIKE = 5.0
_SEED_OFFSET_A = 0
_SEED_OFFSET_B = 31


def gen_inputs(case, seed, instance):
    """The G0 case/seed stream (the ``ieee`` case is the normal stream plus specials)."""
    base = "normal" if case == "ieee" else case
    mag = _MAGS.get(base, _DEFAULT_MAG)
    a_shape = tuple(instance.tensor("A").shape)
    b_shape = tuple(instance.tensor("B").shape)
    A = _sample(base, a_shape, seed + _SEED_OFFSET_A, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)
    B = _sample(base, b_shape, seed + _SEED_OFFSET_B, instance.dtype, mag=mag, large=_LARGE, spike=_SPIKE)
    if case == "ieee":
        A = _inject(A, seed + _IEEE_SEED_STRIDE)
        B = _inject(B, seed + 2 * _IEEE_SEED_STRIDE)
    return {"A": A, "B": B}


def reference(inputs, attrs):
    """The independent fp64 oracle (one rounding at the output)."""
    c = inputs["A"].float() @ inputs["B"].float()
    d = c.double()
    return {"C": d}
```

# 正确性
容差 |out - ref| <= 0.002 * |ref| + 0.001 * scale。
NaN、Inf 必须和参考一致，符号也要一致。输入保持 `device const`，不许改。
声明 `zero` 的工作区在每次计时迭代里由评测填零，这次填零算进你的时间；没声明 `zero` 的工作区保留你的 Host Tiling 计划写进去的内容。

# 约束
{
  "abi_id": "abi.gemm.fp16.small.apple-m4-max",
  "acc_dtype": "fp32",
  "backend_id": "metal",
  "schema_version": 1,
  "target_id": "apple-m4-max",
  "tensors": [
    {
      "access": "read",
      "doc": "row-major [M, K]",
      "dtype": "fp16",
      "name": "A",
      "role": "input",
      "shape": [
        256,
        128
      ],
      "space": "global",
      "strides": [
        128,
        1
      ]
    },
    {
      "access": "read",
      "doc": "row-major [K, N]",
      "dtype": "fp16",
      "name": "B",
      "role": "input",
      "shape": [
        128,
        256
      ],
      "space": "global",
      "strides": [
        256,
        1
      ]
    },
    {
      "access": "write",
      "doc": "row-major [M, N]",
      "dtype": "fp16",
      "name": "C",
      "role": "output",
      "shape": [
        256,
        256
      ],
      "space": "global",
      "strides": [
        256,
        1
      ]
    }
  ],
  "tolerances": {
    "atol_rel": 0.001,
    "exact": false,
    "rtol": 0.002
  }
}

签名骨架：
```metal
kernel void gemm_kernel(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tid [[thread_position_in_threadgroup]]
)
```

必须自带且只能带一个 `/* taic-launch */` 块：外层不会替你选 grid/threads。写成 `launch NAME grid=16,1,1 threads=128,1,1`，NAME 与你的 kernel 函数同名。`grid=(16, 1, 1)` 和 `grid=16, 1, 1` 是同一个网格。除法必须整除。不是合同常量的名字，编译前就会被拒绝。`math fast` 或 `math safe` 只能有一行。
最多 32 段。不许纹理。不许把任何旧核原文贴回来。
着色器校验在你退出之后由外层做。插桩会把线程组内存报大，边界写保守些。
本场要交的文件是 `gemm-fp16-small-init.metal`。外层只读这个文件。开场时它不存在。
你写了它，外层就把里面的内容当作你交回的核。你不写它，外层记这场没有交回内核。
`kernel.metal` 在有种子时已经放在本目录，只供阅读。外层不读它。改它不算交卷。
某版一过 `--quick` 就用 `cp` 复制成 `gemm-fp16-small-init.metal.tmp`，再用 `mv` 替换成 `gemm-fp16-small-init.metal`，别等结束；更快且合格也这样替换。无合格版时放最近一版，不抄种子。checkpoint 不代替外层判卷。

`experience.md` 是这个项目已经测过的东西：哪些启动契约与 tile 划算、哪些算子已经贴着带宽屋顶、哪个库换了算法。它是线索不是规则：你自己测出来的数比它高。

# 前提要自己测
成绩单、上一轮笔记和参考资料里关于这台机器的断言（某种操作数类型的矩阵乘加多快、某条指令存不存在、某个上限是多少）都是假设，不是事实。成绩单印了按操作数类型实测的矩阵峰值；再往细的速率没有人替你测过。若一句断言决定某个算法能不能做（例如「这种数据类型进矩阵单元就慢一倍」），先写一个几十行的小探针把它测出来，再决定放弃。只凭一句没测过的断言放弃的方向，不算试过；放弃时把这句断言原样写进 NOTES 并标「未验证」，下一轮可以重开。探针的写法在技能 `taic-tools` 里。

# 验证命令
只对答案，不计时：
```
uv run --directory /repo python -m tcenv.cli eval-v2 gemm /work/init/gemm-fp16-small-init.metal --dtype fp16 --size small --quick --target apple-m4-max
```

# 评测命令
对答案并计时。协议和外层相同。自测数字只供你判断，不进种子，也不进对外的加速比。
```
uv run --directory /repo python -m tcenv.cli eval-v2 gemm /work/init/gemm-fp16-small-init.metal --dtype fp16 --size small --timed --target apple-m4-max --self-check-dir /work/init/.selfcheck --reps 15 --min-improvement 1.005 --noise-z 2.0
```
上面两条命令里的交卷文件和 report.json 是绝对路径。不要改成相对路径。
命令会把一行结果追加到本目录的 `self_checks.jsonl`。出现过 timeout 或 crash 之后，再跑 `--timed` 会被拒绝。
交卷前请在 `--timed` 后面加 `--validate` 跑最后一次：那一遍会开插桩校验（抓越界和竞争），外层也会做同一件事，早发现能省一场会话。
这个目录最多记 20 次 `--timed`，累计 120 秒。超过之后 `--timed` 会拒绝，你仍可交卷，外层照常测。
在本目录运行。GPU 锁是机器上那一把，不要改锁路径，也不要自己另写一把。
一个不返回的核会冻住显示器。

# 晋升标准
外层换种子要同时快过 1.005 倍，并且超过 2.0 个标准误。
这个门槛要用你这一版自己的 cv，所以这里不算一个目标微秒数。`--timed` 每次会印「够换种子」或「不够」。那一行注明自测不算数。

现在没有种子，也没有 `kernel.metal`。先把一颗核写进 `gemm-fp16-small-init.metal`。外层评测合格，它就成为种子。

# 这个系统怎么工作
你在一场接力里：两个委员会轮流，一轮一场新会话，对话不带过去。外层判卷是机械的。你自报的数字不进种子，也不进对外的加速比。
固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档和队友留下的前提。你可以读整个仓库，只能写本目录。
本场的主产出始终是 `gemm-fp16-small-init.metal`；先写一版可交的核，再实验。`NOTES.md`/`FEEDBACK.md` 当场追加；每版记一行，每个坑或否决前提也记一行。
给系统的建议只给人看，有人在看板复核后改仓库。给下一棒的避坑提示只在本运行内往下传，不跨题目。放弃某个方向所依据的断言必须写进「被否决的前提」节，不要夹在避坑里。
**没有内容就不要写。** 你的输出按 token 计费：某一节没话说就整节省略，不要写「无」「（空）」「none」这类占位符——它们是纯成本，看板还要逐条清理。整份 `FEEDBACK.md` 都没有内容时，不要创建这个文件。
文件开头一行 `kind: prompt|skill|tool|rule|decision`，可以写多个，用逗号分开；没有系统层面的建议或求助时，这一行也不要写。
有内容时按这五个标题写，只写有话的那几节：`# 需要帮助`、`# 对系统的建议`、`# 卡住的原因`、`# 给下一棒的避坑提示`、`# 被否决的前提`。缺权限、sudo、许可证、凭据：当场写 `# 需要帮助`，能绕开也写。建议附 ```harness-proposal 块（见 HARNESS.md）。
被否决的前提只写这两种：
`[已实测: 文件名 | 一句命题 | 数字和单位]`
`[未验证 | 一句命题 | 为什么没测]`
探针格只写一个文件名。参数写进命题格或数字格。引用同运行别的会话，写 `r9.p0/sim_w63.py` 这种以 `agent_work` 为根的路径。
「已实测」要写清测的是哪条命题。「会不会算错」和「误差多大」是两条。
写完跑一次：
```
uv run --directory /repo python -m taic.agent_feedback check /work/init/FEEDBACK.md --work /work/init
```
比较两版快慢或做消融之前，先读技能 `taic-tools` 的「测量」节。动手前先读本目录的 `device.md` 和 `experience.md`。再查仓库 `docs/`、本运行 `agent_work/*/NOTES.md`。

# 证据
在本目录维护 `NOTES.md`：每一版改了什么、自测微秒和 cv、留下还是放弃、理由。
涉及硬件上限的断言（速率、指令、容量）必须附探针文件名和实测数字；没有实测的写「未验证」。被否决的方向连同否决它的数字一起记，方便下一位复核。
这是交给下一位的记录。下一位看不到这次对话。

交卷文件 `gemm-fp16-small-init.metal` 里一直放目前最好的一版。外层只读这个文件。
