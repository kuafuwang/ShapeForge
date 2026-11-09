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

import functools
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACCEPTANCE = ROOT / "tests" / "support" / "golden" / "shared" / "跨硬件算子生成分工-阶段验收"


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
    #: v1 的语义/降档身份（opspec_digest / lowering_profile_digest / 渲染参数 / baseline 登记）。
    #: 这些数字来自 v1 装配的冻结记录，不由 v2 重新推导——resume、发布门、远端评测的身份
    #: 必须还是迁移前的那一个，否则历史 run 会被误判成另一个问题。
    identity: dict = field(default_factory=dict)
    exact: bool = False
    meta: dict = field(default_factory=dict)


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


def _contract_text(task: str, dtype: str = "fp16", size: str = "paper") -> dict:
    """@brief 读取旧提交里真实 v1 合同的 describe/abi_stub（以及 Triton 题面）文本。

    fp16 的条目按题名登记，其它 dtype 用 ``题名|dtype`` 追加键（dtype 会出现在 describe 的
    存储类型与 abi_stub 的 MSL 类型里，不能共用一份）。

    预算类测试对契约长度敏感（overlay 是否保留、哪些段落被裁剪），简写的描述会改变被测
    行为，所以这里用从 v1 装配直接取出的原文，而不是重新拼一份近似文本。

    @param task 任务名。
    @return 含 describe/abi_stub 的字典。
    """
    import json

    path = ROOT / "tests" / "support" / "golden" / "shared" / "v1-contract-text.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get(f"{task}|{dtype}|{size}", data.get(f"{task}|{dtype}", data[task]))


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


def _sem_int_5(dtype: str, size: str) -> SpecimenTask:
    """@brief sem_int_5 的 v1 形状（sem_int profile：BLOCK=256、threads=64、per_thread=4）。"""
    return _sem_int("sem_int_5", dtype, size)


def _sem_int(seed_name: str, dtype: str, size: str) -> SpecimenTask:
    """@brief sem_int 动态族的一个 seed：归档的 v1 题面、合同与身份。

    族里每个 seed 的 N 与 BLOCK/threads 相同，但 Triton 题面、opspec_digest 与 ``program_lines``
    不同，所以合同与身份按 seed 各取一份，不用 sem_int_5 的去代替别的 seed。
    """
    n = 65536 if size == "paper" else 4096
    ident = _identity(seed_name, dtype, size)
    contract = SpecimenContract(
        kernel_name="sem_int_kernel",
        constants=(_uint("N", n), _uint("BLOCK", 256)),
        args=(SpecimenArg("x", "int"), SpecimenArg("y", "int"), SpecimenArg("out", "int")),
        grid=(n // 256, 1, 1),
        threads=(64, 1, 1),
        describe_text=ident.get("describe", ""),
        abi_stub_text=ident.get("abi_stub", ""),
    )
    return SpecimenTask(
        seed_name, "i32", size, contract, ident.get("naive", "") or _candidate("B4", "sem_int_5", "fp16"),
        description=ident.get("description", ""),
        flops=ident.get("flops", 0.0), min_bytes=ident.get("min_bytes", 0.0),
        rtol=ident.get("rtol", 0.0), atol_rel=ident.get("atol_rel", 0.0),
        exact=ident.get("exact", True), meta=ident.get("meta", {}),
        cases=tuple(ident.get("cases", ("random", "small", "edge"))),
        triton_source=ident.get("triton_source", ""),
        identity=ident,
    )


def _render_triton(template: str, values: dict[str, object]) -> str:
    """@brief 按 v1 的 ``load_text`` 规则渲染 Triton 题面（``{{键}}`` → 具体写法）。

    @param template 含占位符的模板文本。
    @param values 占位符到取值的映射。
    @return 渲染后的文本。
    @raises AssertionError 渲染后仍有未替换的占位符。
    """
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", str(value))
    leftover = re.findall(r"\{\{[A-Z0-9_]+\}\}", rendered)
    assert not leftover, f"unrendered placeholders: {sorted(set(leftover))}"
    return rendered


@functools.lru_cache(maxsize=1)
def _records() -> dict:
    """@brief 读一次归档的 v1 身份记录（约 0.6 MB，逐字节内容在测试期间不变）。

    每个用例都要装配 v1 Task；每次重读整份 JSON 会把 fast suite 拖慢一倍以上。

    @return 记录字典（只读使用）。
    """
    path = ROOT / "tests" / "support" / "golden" / "shared" / "v1-task-records.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(name: str, dtype: str = "fp16", size: str = "paper") -> dict:
    """@brief 读取 v1 装配的身份记录（digest、渲染参数、baseline 登记、题面）。

    @param name 题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 身份字典；没有登记时为空字典。
    """
    data = _records()
    record = data.get(f"{name}|{dtype}|{size}")
    if record is None and name.startswith("sem_int_"):
        # v1 的 fp16/bf16 都是 i32 别名，记录按别名各存一份，内容相同。
        record = data.get(f"{name}|fp16|{size}")
    return record if record is not None else {}


def _sem_attention(dtype: str, size: str) -> SpecimenTask:
    """@brief sem_attention_odd_scale 的 v1 形状（与 flash_attention 共用 FA profile 的几何）。

    qk scale 的模式不同（``sum_with_log2e``），但合同几何、常量表与参数槽位相同；describe/abi
    取归档的 v1 原文，所以提示词与反馈文本逐字节等于迁移前。
    """
    paper = size == "paper"
    b, h, s, d = (2, 8, 1024, 64) if paper else (1, 2, 128, 64)
    msl = "half" if dtype == "fp16" else "bfloat"
    text = _contract_text("sem_attention_odd_scale", dtype, size)
    ident = _identity("sem_attention_odd_scale", dtype, size)
    # 归档记录里带的就是这次 (dtype, size) 的 v1 合同原文，直接覆盖尺寸相关的文本。
    text = {"describe": ident.get("describe", text.get("describe", "")),
            "abi_stub": ident.get("abi_stub", text.get("abi_stub", "")),
            "triton_template": text.get("triton_template", "")}
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
    nbytes = 2 if dtype in ("fp16", "bf16") else 4
    return SpecimenTask(
        "sem_attention_odd_scale", dtype, size, contract, _candidate("B4", "sem_attention_odd_scale", dtype),
        description=("FlashAttention forward as written in the specification (read how qk_scale is formed "
                     "before using the standard 1/sqrt(D) scale)."),
        flops=float(4 * b * h * s * s * d),
        min_bytes=float(4 * b * h * s * d * nbytes),
        rtol=0.002 if dtype == "fp16" else 0.016,
        atol_rel=0.001 if dtype == "fp16" else 0.008,
        identity=ident,
        triton_source=_render_triton(
            text["triton_template"],
            {"TL": "tl.float16" if dtype == "fp16" else "tl.bfloat16",
             "B": b, "H": h, "S": s, "D": d, "BM": 32, "BN": 32, "NW": 4},
        ),
    )


_BUILDERS = {"lion": _lion, "conv2d": _conv2d, "bitdelta_gemm": _bitdelta, "flash_attention": _flash_attention,
             "sem_attention_odd_scale": _sem_attention, "sem_int_5": _sem_int_5,
             "sem_int_3": lambda dtype, size: _sem_int("sem_int_3", dtype, size),
             "sem_int_007": lambda dtype, size: _sem_int("sem_int_007", dtype, size)}


def specimen_task(name: str, dtype: str = "fp16", size: str = "paper") -> SpecimenTask:
    """@brief 委托给 ``v1_tasks.v1_task``：返回与 v1 完全兼容的完整 Task。

    本模块的 ``_BUILDERS`` 提供归档的启动几何（kernel/grid/threads/constants/naive 文本），
    ``v1_task`` 再用 v2 的合同把 v1 的 Contract、reference 与 baselines 补齐。直接使用 Task
    的测试（hw_features/build_report/domain_sweep/...）因此不需要改写。

    @param name 样本题名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return 与 v1 装配一致的 ``tcenv.task.Task``。
    """
    from tests.support.v1_tasks import v1_task as _v1_task

    return _v1_task(name, dtype, size)
