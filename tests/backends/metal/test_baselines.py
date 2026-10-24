"""The vendor baselines themselves: every declared baseline runs and matches the reference, and weak unfused ones have a compiled counterpart."""

from __future__ import annotations

import pytest

from tasks import get_task, list_tasks
from tcenv.backends.metal.baselines import BaselineRunner
from tcenv.backends.metal.testing import _floor, prepare_case, scaled_error

pytestmark = pytest.mark.gpu

ALL = list(list_tasks(include_phase2=True))


def test_composite_operators_have_compiled_baselines():
    """@brief 检查组合算子的 task 在原有 baseline 之外还声明了编译式 baseline。

    lion 需含 ``torch_compile``；bitdelta_gemm 需同时含 ``mlx_compile`` 与 ``torch_compile``；
    各 task 都保留原有 ``mlx`` 与 ``torch_mps``。
    """
    names = {t: {b.name for b in get_task(t, "fp16", "small").baselines} for t in ("lion", "bitdelta_gemm", "silu", "swiglu", "gemm_silu")}
    assert {"torch_compile"} <= names["lion"]
    assert {"mlx_compile", "torch_compile"} <= names["bitdelta_gemm"]
    assert all({"mlx", "torch_mps"} <= n for n in names.values())  # the original baselines stay


@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_baselines_match_the_reference_on_ordinary_inputs(name, dtype):
    """@brief 每份 baseline 在小尺寸普通输入上都要贴近参考结果，且最准的一份不能被放宽到失效。

    最准的 baseline 定义了可达上限（正确性容差就是据此放宽的），但任何一份都不能离谱，
    例如 MLX 的 fp16 归约在可测程度上比 torch 更不准。断言最小误差不超过 2.0、最大误差不超过 16.0。

    @param name 被参数化的 task 名。
    @param dtype 被参数化的数据类型（fp16 或 bf16）。
    """
    task = get_task(name, dtype, "small")
    d = prepare_case(task, task.cases[0], 321)
    errs = {}
    for impl in task.baselines:
        outs = BaselineRunner(impl, task, d.rounded, rotate=1).outputs()
        errs[impl.name] = max(
            float(scaled_error(outs[a.name], d.ref[a.name], d.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype)).max()) for a in task.contract.outputs
        )
    assert min(errs.values()) <= 2.0 and max(errs.values()) <= 16.0, errs
