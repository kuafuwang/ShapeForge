"""Task = everything the environment knows about one kernel to be lowered."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .contract import Contract


@dataclass(frozen=True)
class BaselineImpl:
    """@brief 同一计算的一个厂商调优实现，作为计时的基线。

    A strong, vendor-tuned implementation of the same computation.

    ``fn`` receives framework arrays in the order of the contract's input arguments and
    returns one array (or a tuple, in the order of the contract's output arguments).
    """

    name: str  # e.g. "mlx", "torch_mps"
    framework: str  # "mlx" | "torch"
    fn: Callable


@dataclass
class Task:
    """@brief 一个待 lowering 内核的全部已知信息：契约、规格、参考与基线。"""

    name: str
    dtype: str
    size: str
    contract: Contract
    triton_source: str  # the *specification* shown to the model; never executed
    description: str
    cases: tuple[str, ...]
    gen_inputs: Callable[[str, int], dict[str, torch.Tensor]]
    reference: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]
    baselines: tuple[BaselineImpl, ...]
    flops: float
    min_bytes: float
    rtol: float
    atol_rel: float
    # Optional per-output error scale (e.g. sum |x| for cancellation-sensitive reductions).
    error_scale: Callable[[dict[str, torch.Tensor], dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None = None
    # A known-good kernel (not necessarily fast) that validates the harness itself.
    # ``reference_sources`` is keyed by backend id; ``naive_msl`` remains the Metal
    # compatibility spelling until Phase 3 callers migrate.
    naive_msl: str = ""
    reference_sources: dict[str, str] = field(default_factory=dict)
    exact: bool = False  # integer tasks compare bit-exactly
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """@brief 让 ``naive_msl`` 与 ``reference_sources`` 的 Metal 条目保持同步。
        """
        if self.naive_msl and "metal" not in self.reference_sources:
            self.reference_sources["metal"] = self.naive_msl
        elif "metal" in self.reference_sources and not self.naive_msl:
            self.naive_msl = self.reference_sources["metal"]

    def reference_for(self, backend_id: str) -> str:
        """@brief 取该任务在指定后端上的已知正确源码。

        The task's known-good source for one backend; empty when it has none.

        @param backend_id 后端 id，如 ``"metal"``。
        @return 参考内核源码；没有时返回空串。
        """
        if backend_id == "metal" and self.naive_msl:
            return self.naive_msl
        return self.reference_sources.get(backend_id) or ""

    @property
    def key(self) -> str:
        """@brief 任务的三段式键。

        @return ``name_dtype_size`` 形式的字符串。
        """
        return f"{self.name}_{self.dtype}_{self.size}"

    @property
    def total_buffer_bytes(self) -> int:
        """@brief 契约中所有 buffer 参数的总字节数。

        @return 总字节数。
        """
        return sum(a.nbytes for a in self.contract.buffers)


def task_fingerprint(task: Task) -> str:
    """@brief 计算被测问题的指纹。

    Identity of the measured problem, separate from ``tcenv.triton_runner.spec_fingerprint``.

    ``spec_fingerprint`` is the triton-msl baseline cache key. This one also covers the naive kernel, the tolerances,
    the flop and byte counts, and whether a full-domain sweep is declared. The reference function's code is not hashed;
    ``run_config.json`` records ``git_commit`` and ``git_dirty`` for that.

    @param task 要取指纹的任务。
    @return 十六进制 sha256 指纹。
    """
    import hashlib
    import json

    meta = task.meta or {}

    def plain(value):
        if isinstance(value, tuple):
            return [plain(v) for v in value]
        if isinstance(value, dict):
            return {str(k): plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
        return value

    recorded = {
        "ieee_base": plain(meta.get("ieee_base")),
        "ieee_inputs": plain(meta.get("ieee_inputs")),
        "ieee_max": plain(meta.get("ieee_max")),
        "domain_case": bool(meta.get("domain_case")),
        "source": meta.get("source") or "",
    }
    parts = (
        task.contract.describe(),
        task.triton_source,
        task.reference_for("metal"),
        repr(task.cases),
        repr((task.rtol, task.atol_rel, task.exact)),
        repr((task.flops, task.min_bytes)),
        task.dtype,
        task.size,
        json.dumps(recorded, sort_keys=True, ensure_ascii=False),
    )
    h = hashlib.sha256()
    for part in parts:
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def default_tolerances(dtype: str) -> tuple[float, float]:
    """@brief 按元素类型给出默认容差。

    (rtol, atol_rel). Tolerance is rtol*|ref| + atol_rel*scale per element.

    @param dtype 元素类型短名。
    @return ``(rtol, atol_rel)`` 二元组；整数类型为 ``(0.0, 0.0)``。
    """
    if dtype == "fp16":
        return 2.0e-3, 1.0e-3
    if dtype == "bf16":
        return 1.6e-2, 8.0e-3
    if dtype == "fp32":
        return 1.0e-5, 1.0e-5
    return 0.0, 0.0
