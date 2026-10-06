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


def default_tolerances(dtype: str) -> tuple[float, float]:
    """(rtol, atol_rel). Tolerance is rtol*|ref| + atol_rel*scale per element."""
    if dtype == "fp16":
        return 2.0e-3, 1.0e-3
    if dtype == "bf16":
        return 1.6e-2, 8.0e-3
    if dtype == "fp32":
        return 1.0e-5, 1.0e-5
    return 0.0, 0.0
