"""Task = everything the environment knows about one kernel to be lowered."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

from .contract import Contract


@dataclass(frozen=True)
class BaselineImpl:
    """A strong, vendor-tuned implementation of the same computation.

    ``fn`` receives framework arrays in the order of the contract's input arguments and
    returns one array (or a tuple, in the order of the contract's output arguments).
    """

    name: str  # e.g. "mlx", "torch_mps"
    framework: str  # "mlx" | "torch"
    fn: Callable


@dataclass
class Task:
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
    naive_msl: str = ""
    exact: bool = False  # integer tasks compare bit-exactly
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.name}_{self.dtype}_{self.size}"

    @property
    def total_buffer_bytes(self) -> int:
        return sum(a.nbytes for a in self.contract.buffers)


def task_fingerprint(task: Task) -> str:
    """Identity of the measured problem, separate from ``tcenv.triton_runner.spec_fingerprint``.

    ``spec_fingerprint`` is the triton-msl baseline cache key. This one also covers the naive kernel, the tolerances,
    the flop and byte counts, and whether a full-domain sweep is declared. The reference function's code is not hashed;
    ``run_config.json`` records ``git_commit`` and ``git_dirty`` for that.
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
        task.naive_msl,
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
    """(rtol, atol_rel). Tolerance is rtol*|ref| + atol_rel*scale per element."""
    if dtype == "fp16":
        return 2.0e-3, 1.0e-3
    if dtype == "bf16":
        return 1.6e-2, 8.0e-3
    if dtype == "fp32":
        return 1.0e-5, 1.0e-5
    return 0.0, 0.0
