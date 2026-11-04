"""Specimen tasks for v1 *mechanism* tests whose original task has since migrated to v2.

The hardware-feature tests are about the *mechanism* — static source analysis, compile evidence,
roofline attribution, prompt blocks — and they were written against an elementwise specimen
(``lion``) and a compute specimen (``conv2d``).  B2 migrated both, so ``tasks.get_task`` can no
longer build them, and no other preserved v1 task belongs to those families.

Re-pointing those tests at an unrelated preserved task would silently change what they assert
(the family decides the verdict), so instead this module rebuilds the *shape* of the v1 task
from the migration evidence: the contract constants the v1 LoweringProfile declared, the kernel
name, and the archived naive candidate text.  ``hw.build_report`` needs exactly
``contract.constants`` / ``contract.kernel_name`` / ``contract.render_header()`` plus
``name``/``dtype``/``size`` — nothing here is a second task registry, and nothing evaluates.

When a test can use a task that is still on the v1 path it must do so (``v1_tasks.v1_task``);
this module is only for specimens that no longer exist as live tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收"


@dataclass(frozen=True)
class SpecimenArg:
    """@brief 合同参数的最小形状（名字与元素类型）。"""

    name: str
    dtype: str


@dataclass(frozen=True)
class SpecimenContract:
    """@brief v1 合同的最小形状：核名、常量表与常量头渲染。"""

    kernel_name: str
    constants: tuple[tuple[str, str, str], ...]  # (名字, MSL 类型, 字面量)
    args: tuple[SpecimenArg, ...] = ()
    grid: tuple[int, int, int] = (1, 1, 1)
    threads: tuple[int, int, int] = (128, 1, 1)
    describe_text: str = ""
    abi_stub_text: str = ""

    def abi_stub(self) -> str:
        """@brief 系统提示里的签名骨架（机制测试只要求它存在且指向同一个核名）。

        @return MSL 签名占位文本。
        """
        if self.abi_stub_text:
            return self.abi_stub_text
        return f"kernel void {self.kernel_name}(...);"

    def describe(self) -> str:
        """@brief 供系统提示渲染的合同描述（核名 + 常量表）。

        @return 多行描述文本。
        """
        if self.describe_text:
            return self.describe_text
        lines = [f"kernel {self.kernel_name}", "constants:"]
        lines.extend(f"  {name} ({msl_type}) = {literal}" for name, msl_type, literal in self.constants)
        return "\n".join(lines)

    def render_header(self) -> str:
        """@brief 渲染常量声明头（与 v1 ``Contract.render_header`` 同一形式）。

        @return MSL 常量声明文本。
        """
        # constants 里存的是**数值本身**（int/float），与 v1 ``Contract.constants`` 一致——
        # 硬件特征分析直接用这些数值；MSL 的 u/f 后缀只在渲染常量头时补上。
        # v1 的合同头同样以 metal_stdlib 与 using namespace metal 开头；GEMM 参考核里的
        # simdgroup_matrix 只有在这个命名空间下才解析得到。
        prefix = "#include <metal_stdlib>\nusing namespace metal;\n"
        suffix = {"uint": "u", "float": "f"}
        return prefix + "".join(
            f"constant constexpr {msl_type} {name} = {literal}{suffix.get(msl_type, '')};\n"
            for name, msl_type, literal in self.constants
        )


@dataclass(frozen=True)
class SpecimenTask:
    """@brief 供硬件特征机制测试使用的任务形状（不是可评测任务）。"""

    name: str
    dtype: str
    size: str
    contract: SpecimenContract
    naive_msl: str = field(default="")
    description: str = ""
    flops: float = 0.0
    min_bytes: float = 0.0
    rtol: float = 0.002
    atol_rel: float = 0.001
    cases: tuple[str, ...] = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
    triton_source: str = ""


def _uint(name: str, value: int) -> tuple[str, str, str]:
    """@brief 一个 uint 常量。

    @param name 常量名。
    @param value 数值。
    @return 常量三元组。
    """
    return (name, "uint", int(value))


def _float(name: str, literal: str) -> tuple[str, str, str]:
    """@brief 一个 float 常量。

    @param name 常量名。
    @param literal 字面量文本。
    @return 常量三元组。
    """
    return (name, "float", float(literal))


def _candidate(batch: str, task: str, dtype: str) -> str:
    """@brief 读迁移证据里归档的 naive 候选，并渲染 dtype 占位符。

    @param batch 批次名（B1/B2）。
    @param task 任务名。
    @param dtype 元素类型。
    @return MSL 源文本。
    """
    path = ACCEPTANCE / batch / "evidence" / "candidates" / f"{task}-naive.msl"
    spelling = "half" if dtype == "fp16" else "bfloat"
    return path.read_text(encoding="utf-8").replace("{{T}}", spelling)


def _lion(dtype: str, size: str) -> SpecimenTask:
    """@brief elementwise 样本：lion 的 v1 合同形状与标量候选。

    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 样本任务。
    """
    n = 8388608 if size == "paper" else 65536
    msl = "half" if dtype == "fp16" else "bfloat"
    contract = SpecimenContract(
        kernel_name="lion_kernel",
        args=tuple(SpecimenArg(name, msl) for name in ("p", "g", "m", "p_out", "m_out")),
        grid=(n // 1024, 1, 1),
        constants=(
            _uint("N", n),
            _uint("BLOCK_SIZE", 1024),
            _float("LR", "0.001"),
            _float("WD", "0.1"),
            _float("BETA1", "0.9"),
            _float("BETA2", "0.99"),
        ),
    )
    nbytes = 2 if dtype in ("fp16", "bf16") else 4
    return SpecimenTask(
        "lion", dtype, size, contract, _candidate("B2", "lion", dtype),
        description="One Lion optimizer step (fp32 math, each output rounded once).",
        rtol=0.002 if dtype == "fp16" else 0.016,
        atol_rel=0.001 if dtype == "fp16" else 0.008,
        flops=float(10 * n),
        min_bytes=float(5 * n * nbytes),  # p/g/m 三个输入 + p_out/m_out 两个输出
    )


def _conv2d(dtype: str, size: str) -> SpecimenTask:
    """@brief compute 样本：conv2d 的 v1 合同形状与候选。

    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 样本任务。
    """
    paper = size == "paper"
    symbols = {
        "N": 32 if paper else 4,
        "H": 56 if paper else 18,
        "W": 56 if paper else 18,
        "C": 64 if paper else 32,
        "OC": 128 if paper else 64,
        "KH": 3,
        "KW": 3,
        "OH": 54 if paper else 16,
        "OW": 54 if paper else 16,
    }
    m_total = symbols["N"] * symbols["OH"] * symbols["OW"]
    k_total = symbols["KH"] * symbols["KW"] * symbols["C"]
    constants = tuple(_uint(name, value) for name, value in symbols.items()) + (
        _uint("M_TOTAL", m_total),
        _uint("K_TOTAL", k_total),
        _uint("BLOCK_M", 64),
        _uint("BLOCK_N", 64),
        _uint("BLOCK_K", 32),
    )
    msl = "half" if dtype == "fp16" else "bfloat"
    contract = SpecimenContract(
        kernel_name="conv2d_kernel",
        constants=constants,
        args=tuple(SpecimenArg(name, msl) for name in ("x", "w", "y")),
        grid=(symbols["OC"] // 64, (symbols["N"] * symbols["OH"] * symbols["OW"] + 63) // 64, 1),
    )
    nbytes = 2 if dtype in ("fp16", "bf16") else 4
    elements = symbols["N"] * symbols["H"] * symbols["W"] * symbols["C"] + symbols["OC"] * k_total + m_total * symbols["OC"]
    text = _contract_text("conv2d")
    contract = SpecimenContract(
        kernel_name=contract.kernel_name,
        constants=contract.constants,
        args=contract.args,
        describe_text=text["describe"],
        abi_stub_text=text["abi_stub"],
    )
    return SpecimenTask(
        "conv2d", dtype, size, contract, _candidate("B2", "conv2d", dtype),
        description="3x3 convolution, stride 1, no padding, no bias, NHWC activations; fp32 accumulation.",
        rtol=0.002 if dtype == "fp16" else 0.016,
        atol_rel=0.001 if dtype == "fp16" else 0.008,
        flops=float(2 * m_total * symbols["OC"] * k_total),
        min_bytes=float(elements * nbytes),
    )


def _contract_text(task: str) -> dict:
    """@brief 读取旧提交里真实 v1 合同的 describe/abi_stub 文本。

    预算类测试对契约长度敏感（overlay 是否保留、哪些段落被裁剪），简写的描述会改变被测
    行为，所以这里用从 v1 装配直接取出的原文，而不是重新拼一份近似文本。

    @param task 任务名。
    @return 含 describe/abi_stub 的字典。
    """
    import json

    path = ROOT / "tests" / "support" / "golden" / "shared" / "v1-contract-text.json"
    return json.loads(path.read_text(encoding="utf-8"))[task]


def _bitdelta(dtype: str, size: str) -> SpecimenTask:
    """@brief compute/GEMM 样本：bitdelta_gemm 的 v1 合同形状与候选。

    SCALE 在 v1 里是标量 buffer（2 号槽），不是编译常量，所以常量表与 v1 一样不含它。
    """
    paper = size == "paper"
    m, n, k, kb = (512, 4096, 4096, 128) if paper else (128, 256, 256, 8)
    msl = "half" if dtype == "fp16" else "bfloat"
    contract = SpecimenContract(
        kernel_name="bitdelta_gemm_kernel",
        constants=(
            _uint("M", m), _uint("N", n), _uint("K", k), _uint("KB", kb),
            _uint("BLOCK_M", 64), _uint("BLOCK_N", 64), _uint("BLOCK_K", 32),
        ),
        args=(SpecimenArg("x", msl), SpecimenArg("bits", "uint"), SpecimenArg("y", msl)),
        grid=(n // 64, m // 64, 1),
    )
    return SpecimenTask(
        "bitdelta_gemm", dtype, size, contract, _candidate("B3", "bitdelta_gemm", dtype),
        description="y = scale * (x @ W) with W a +-1 matrix stored as packed sign bits.",
        flops=float(2 * m * n * k),
        min_bytes=float((m * k + k * n // 32 + m * n) * (2 if dtype in ("fp16", "bf16") else 4)),
    )


def _flash_attention(dtype: str, size: str) -> SpecimenTask:
    """@brief attention 样本：flash_attention 的 v1 合同形状与候选（含 TPR/SHUFFLES 渲染）。

    TPR = threads/BLOCK_M = 128/32 = 4；SHUFFLES 是 profile 按 log2(TPR) 生成的归约序列。
    """
    paper = size == "paper"
    b, h, s, d = (2, 8, 1024, 64) if paper else (1, 2, 128, 64)
    msl = "half" if dtype == "fp16" else "bfloat"
    text = _contract_text("flash_attention")
    contract = SpecimenContract(
        kernel_name="flash_attention_kernel",
        constants=(
            _uint("B", b), _uint("H", h), _uint("S", s), _uint("D", d),
            _uint("BLOCK_M", 32), _uint("BLOCK_N", 32), _float("SM_SCALE", "0.125"),
        ),
        args=tuple(SpecimenArg(name, msl) for name in ("q", "k", "v", "o")),
        grid=(s // 32, b * h, 1),
        describe_text=text["describe"],
        abi_stub_text=text["abi_stub"],
    )
    shuffles = "        s += simd_shuffle_xor(s, 1);\n        s += simd_shuffle_xor(s, 2);\n"
    naive = _candidate("B3", "flash_attention", dtype).replace("{{TPR}}", "4").replace("{{SHUFFLES}}", shuffles)
    nbytes = 2 if dtype in ("fp16", "bf16") else 4
    return SpecimenTask(
        "flash_attention", dtype, size, contract, naive,
        description="Non-causal scaled dot-product attention with the online-softmax recurrence.",
        flops=float(4 * b * h * s * s * d),
        min_bytes=float(4 * b * h * s * d * nbytes),
    )


_BUILDERS = {"lion": _lion, "conv2d": _conv2d, "bitdelta_gemm": _bitdelta, "flash_attention": _flash_attention}


def specimen_task(name: str, dtype: str = "fp16", size: str = "paper") -> SpecimenTask:
    """@brief 构造一个已迁移样本题的任务形状，供 v1 机制测试使用。

    @param name 样本题名（目前支持 lion/conv2d）。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 样本任务。
    @raises KeyError 未登记的样本题（保留题应走 ``v1_tasks.v1_task``）。
    """
    if name not in _BUILDERS:
        raise KeyError(f"{name!r} is not a recorded specimen; live v1 tasks belong to tests.support.v1_tasks")
    return _BUILDERS[name](dtype, size)
