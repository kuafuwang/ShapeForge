"""Formal equivalence checking: the interface only.

The paper's eligibility rule has a third condition besides *safe* and *numerically correct*: a candidate is verified to be
equivalent to the specification by a formal tool for NVIDIA PTX. There is no such tool for Metal Shading Language / AIR on Apple GPUs,
so this first version defines the interface and always answers ``unsupported``. Eligibility in this project is therefore
"safe and numerically correct on randomized adversarial inputs" (``EvalReport.eligible``), which is evidence, not proof.

A real checker plugs in by replacing ``verify`` (same signature) and returning ``verified`` or ``refuted`` with a counterexample.
"""
# 形式等价的插槽。evaluate 在碰 GPU 之前调用它，结果只写入报告，不参与判生死。
# 本文件不和标准答案比任何输出。逐元素比较发生在 correctness 关：同一组输入上，候选写出的结果对任务的 CPU reference（float64/float32），判据是 |out - ref| <= rtol*|ref| + atol_rel*scale。
# 差分测试防的是安全关已经通过、但函数算错的候选。它在随机对抗输入上找反例，找不到也不证明所有输入都等价。
# 这里固定回答 unsupported，是因为没有面向 Metal / AIR 的形式工具，不能把「没证明」写成 refuted。

from __future__ import annotations

from dataclasses import asdict, dataclass

UNSUPPORTED = "unsupported"
VERIFIED = "verified"
REFUTED = "refuted"
UNKNOWN = "unknown"  # the checker ran but could neither prove nor refute within its limits
STATUSES = (UNSUPPORTED, VERIFIED, REFUTED, UNKNOWN)


@dataclass(frozen=True)
class VerifyResult:
    status: str
    message: str
    counterexample: dict | None = None

    # 只有 verified 才算证明了和规格等价。unsupported 和 unknown 都不算。
    @property
    def proven(self) -> bool:
        return self.status == VERIFIED

    def to_json(self) -> dict:
        return asdict(self)


# 签名留给将来的证明器。现在固定答 unsupported：没有面向 Metal / AIR 的形式工具时，不能把「没证明」写成 refuted。
# 和标准答案比数值、用来抓住算错实现的，是 correct 关的随机差分测试，不在本函数里。
def verify(task, kernel_source: str, *, timeout_s: float = 60.0) -> VerifyResult:
    """Check that ``kernel_source`` computes the same function as the task's specification for all inputs.

    ``task`` is a ``tcenv.task.Task``; ``kernel_source`` the candidate's MSL. Returns a ``VerifyResult``; this version never proves
    or refutes anything.
    """
    return VerifyResult(
        UNSUPPORTED,
        "no formal equivalence checker exists for Metal Shading Language / AIR; eligibility rests on safety checks and randomized "
        "differential tests, which can find bugs but cannot prove their absence",
    )
