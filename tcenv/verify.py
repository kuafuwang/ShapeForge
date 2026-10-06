"""Formal equivalence checking: the interface only.

The paper's eligibility rule has a third condition besides *safe* and *numerically correct*: a candidate is verified to be
equivalent to the specification by a formal tool for NVIDIA PTX. There is no such tool for Metal Shading Language / AIR on Apple GPUs,
so this first version defines the interface and always answers ``unsupported``. Eligibility in this project is therefore
"safe and numerically correct on randomized adversarial inputs" (``EvalReport.eligible``), which is evidence, not proof.

A real checker plugs in by replacing ``verify`` (same signature) and returning ``verified`` or ``refuted`` with a counterexample.
"""

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

    @property
    def proven(self) -> bool:
        return self.status == VERIFIED

    def to_json(self) -> dict:
        return asdict(self)


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
