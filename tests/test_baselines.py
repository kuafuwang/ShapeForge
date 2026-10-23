"""The vendor baselines themselves: every declared baseline runs and matches the reference, and weak unfused ones have a compiled counterpart."""

from __future__ import annotations

import pytest

from tasks import get_task, list_tasks
from tcenv.backends.metal.baselines import BaselineRunner
from tcenv.backends.metal.testing import _floor, prepare_case, scaled_error

pytestmark = pytest.mark.gpu

ALL = list(list_tasks(include_phase2=True))


def test_composite_operators_have_compiled_baselines():
    names = {t: {b.name for b in get_task(t, "fp16", "small").baselines} for t in ("lion", "bitdelta_gemm", "silu", "swiglu", "gemm_silu")}
    assert {"torch_compile"} <= names["lion"]
    assert {"mlx_compile", "torch_compile"} <= names["bitdelta_gemm"]
    assert all({"mlx", "torch_mps"} <= n for n in names.values())  # the original baselines stay


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_baselines_match_the_reference_on_ordinary_inputs(name, dtype):
    """The most accurate baseline defines what is achievable (that is how the correctness tolerance is relaxed), but none may be garbage:
    MLX's fp16 reduction, for instance, is measurably less accurate than torch's."""
    task = get_task(name, dtype, "small")
    d = prepare_case(task, task.cases[0], 321)
    errs = {}
    for impl in task.baselines:
        outs = BaselineRunner(impl, task, d.rounded, rotate=1).outputs()
        errs[impl.name] = max(
            float(scaled_error(outs[a.name], d.ref[a.name], d.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype)).max()) for a in task.contract.outputs
        )
    assert min(errs.values()) <= 2.0 and max(errs.values()) <= 16.0, errs
