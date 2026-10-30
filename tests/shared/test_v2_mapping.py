"""A1: every shipped v1 OpSpec maps to a v2 instance with the same typed semantics.

This is a comparison, not a migration: the shipped tasks and their identity stay v1.  The
test-only mapping removes ``uses`` and adds the v2 discovery default, then instantiates both
loaders and compares every semantic field (tensors, attrs, tolerances, cases, accounting,
evaluation metadata).
"""

from __future__ import annotations

import pytest

from tcenv.opspec import instantiate, load_opspec

from ..support.v2_mapping import TASKS, mapped_opspec, opspec_paths

OPSPECS = opspec_paths()
IDS = [str(path.relative_to(TASKS)) for path in OPSPECS]


def _view(instance) -> dict:
    """@brief v1/v2 共有的语义视图（不含各自新增的 spec_version/discovery）。

    @param instance 语义实例。
    @return 可比较字典。
    """
    return {
        "op_name": instance.op_name,
        "category": instance.category,
        "dtype": instance.dtype,
        "size": instance.size,
        "symbols": dict(instance.symbols),
        "tensors": [
            (
                binding.name,
                binding.role,
                binding.dtype,
                tuple(int(dim) for dim in binding.shape),
                tuple(int(stride) for stride in binding.strides),
                binding.layout,
                binding.space,
                binding.access,
            )
            for binding in instance.tensors
        ],
        "resolved_attrs": dict(instance.resolved_attrs),
        "spec_params": dict(instance.spec_params),
        "tolerances": (
            float(instance.tolerances.rtol),
            float(instance.tolerances.atol_rel),
            bool(instance.tolerances.exact),
        ),
        "cases": tuple(instance.case_domain.cases),
        "case_inputs": {case: dict(values) for case, values in instance.case_domain.inputs.items()},
        "flops": float(instance.work_accounting.flops),
        "min_bytes": float(instance.work_accounting.min_bytes),
        "acc_dtype": instance.acc_dtype,
        "size_usage": tuple(instance.size_usage),
        "evaluation_meta": instance.evaluation_meta(),
    }


@pytest.mark.parametrize("path", OPSPECS, ids=IDS)
def test_every_v1_opspec_maps_to_the_same_v2_semantic_instance(path):
    """@brief 所有 shipped v1 文档在 v2 下实例化出相同语义（逐 dtype × size 比较）。"""
    v1_spec = load_opspec(path)
    v2_spec = mapped_opspec(path)
    assert v1_spec.spec_version == 1 and v2_spec.spec_version == 2
    assert v1_spec.dtypes == v2_spec.dtypes
    combinations = 0
    for dtype in v1_spec.dtypes:
        for size in v1_spec.sizes:
            first = instantiate(v1_spec, dtype, size)
            second = instantiate(v2_spec, dtype, size)
            assert _view(first) == _view(second), f"{path}: {dtype}/{size} changed semantic meaning"
            combinations += 1
    assert combinations > 0


def test_the_mapping_covers_the_frozen_g0_domain():
    """@brief 映射不能只覆盖一个玩具题；至少覆盖 G0 的 82 个组合。"""
    combinations = 0
    tasks: set[str] = set()
    for path in OPSPECS:
        spec = load_opspec(path)
        tasks.add(spec.op_name)
        combinations += len(spec.dtypes) * len(spec.sizes)
    assert combinations >= 82, f"mapping covers only {combinations} (dtype,size) combinations"
    assert len(tasks) >= 16, f"mapping covers only {len(tasks)} operators"
