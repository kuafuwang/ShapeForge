"""H-19: the IEEE case path itself, compared between the frozen R0 task and the migrated task.

The equivalence harness deliberately maps ``ieee`` to its base distribution, because the special
values are injected by the *evaluation* layer, not by the generator.  That left the whole IEEE
path unverified: ``prepare_case`` → ``inject_specials`` → raw/rounded bytes → reference → error
scale → NaN/Inf judgement.  This module exercises that path for both sides.

Rules pinned here:

* the inputs, reference and scale **expected** values come from the frozen task;
* each side's default RMS scale is computed from its **own finite** projection
  (``_finite``), never from a reference containing NaN/Inf and never borrowed across sides;
* special values really are injected (the case is not a plain ``normal`` sample);
* the NaN/Inf agreement rule is actually exercised.

CPU only: ``prepare_case`` allocates host tensors and opens no device.
"""

from __future__ import annotations

import pytest
import torch

from tasks import get_task
from tcenv.backends.metal.testing import IEEE_CASE, prepare_case, scaled_error, _floor
from tests.support.frozen_tasks import SEEDS, extract_snapshot, frozen_tasks, old_build, raw_bytes
from tests.support.frozen_tasks import REMAINING_GEN, is_remaining


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """@brief 导出 R0 登记提交的 ``tasks/`` 快照。

    @return :class:`tests.support.frozen_tasks.FrozenSnapshot`。
    """
    return extract_snapshot(tmp_path_factory.mktemp("r0_tasks"))


def test_the_ieee_case_is_injected_and_the_default_scale_uses_finite_values(frozen):
    """@brief 冻结旧 Task 与新 Task 的 IEEE ``prepare_case`` 逐字段一致，且特殊值确实被注入。"""
    from tcenv.backends.metal.testing import _finite

    checked = 0
    for task_name, dtype, size in frozen.migrated_cases():
        with frozen_tasks(frozen.root):
            old = old_build(task_name, dtype, size)
        new = get_task(task_name, dtype, size)
        if IEEE_CASE not in old.cases:
            continue
        for seed in SEEDS:
            old_case = prepare_case(old, IEEE_CASE, seed)
            new_case = prepare_case(new, IEEE_CASE, seed)
            where = f"{task_name}/{dtype}/{size}/{seed}"
            assert set(new_case.rounded) == set(old_case.rounded), f"{where}: input names"
            for name, expected in old_case.rounded.items():
                actual = new_case.rounded[name]
                assert actual.dtype == expected.dtype, f"{where}/{name}: dtype"
                assert tuple(actual.shape) == tuple(expected.shape), f"{where}/{name}: shape"
                assert raw_bytes(actual) == raw_bytes(expected), f"{where}/{name}: rounded bytes"
                assert new_case.raw[name] == old_case.raw[name], f"{where}/{name}: raw bytes"
            for name, expected in old_case.ref.items():
                actual = new_case.ref[name]
                assert actual.dtype == expected.dtype, f"{where}/{name}: ref dtype"
                assert tuple(actual.shape) == tuple(expected.shape), f"{where}/{name}: ref shape"
                assert raw_bytes(actual) == raw_bytes(expected), f"{where}/{name}: ref bytes"
            # Specials must actually be injected on BOTH sides: compare each side's IEEE inputs with
            # its own base distribution (``ieee_base``) and require each side to differ (asserted
            # separately below), not merely to agree with the other side.
            base_case = old.meta.get("ieee_base", "normal")
            old_base = old.gen_inputs(base_case, seed)
            new_base = new.gen_inputs(base_case, seed)
            old_differs = any(
                raw_bytes(old_case.rounded[name])
                != raw_bytes(old_base[name].reshape(old_case.rounded[name].shape))
                for name in old_case.rounded
            )
            new_differs = any(
                raw_bytes(new_case.rounded[name])
                != raw_bytes(new_base[name].reshape(new_case.rounded[name].shape))
                for name in new_case.rounded
            )
            # Both sides must actually inject specials for a declared ieee case: equality alone
            # would let "neither side injected" (False == False) pass silently.
            assert old_differs, f"{where}: the pinned implementation injected no special values"
            assert new_differs, f"{where}: the migrated implementation injected no special values"
            # Scale: dtype/shape/bytes, from each side's own finite projection.
            assert set(new_case.scale) == set(old_case.scale), f"{where}: scale names"
            for name, expected in old_case.scale.items():
                actual = new_case.scale[name]
                assert actual.dtype == expected.dtype, f"{where}/{name}: scale dtype"
                assert tuple(actual.shape) == tuple(expected.shape), f"{where}/{name}: scale shape"
                assert raw_bytes(actual) == raw_bytes(expected), f"{where}/{name}: scale bytes"
            # The default scale must come from each side's own FINITE reference; a registered
            # error_scale (e.g. reduction_sum's sum|row|) replaces the default and was already
            # compared byte-for-byte above.
            if old.error_scale is None:
                for name, value in old_case.ref.items():
                    finite_rms = torch.sqrt(torch.mean(_finite(value).to(torch.float64) ** 2))
                    assert torch.equal(
                        torch.as_tensor(old_case.scale[name]), finite_rms
                    ), f"{where}/{name}: default scale must be the finite-only RMS"
                    assert torch.equal(
                        torch.as_tensor(new_case.scale[name]), finite_rms
                    ), f"{where}/{name}: new default scale must be the finite-only RMS"
            checked += 1
    assert checked, "no migrated task declares an ieee case yet"


def test_the_special_values_are_not_the_plain_distribution(frozen):
    """@brief 快速冒烟：至少一个已迁移组合的 IEEE 输入确实不同于其 base 分布。

    主 sweep 已对每个组合逐侧断言，这里只保留一个有名字的冒烟用例。
    """
    import importlib

    with frozen_tasks(frozen.root):
        old = old_build("rmsnorm", "fp16", "small")
        plain = importlib.import_module("tasks.rmsnorm.task")
    case = prepare_case(old, IEEE_CASE, 0)
    base = plain.build("fp16", "small").gen_inputs("normal", 0)
    differs = any(not torch.equal(case.rounded[name], base[name].reshape(case.rounded[name].shape)) for name in case.rounded)
    assert differs, "the ieee case must differ from the plain base distribution"


def test_the_nan_and_inf_agreement_rule_is_executed(frozen):
    """@brief NaN 只在参考为 NaN 处一致、Inf 同号一致：这条特殊值判决必须真实参与。"""
    with frozen_tasks(frozen.root):
        old = old_build("rmsnorm", "fp16", "small")
    case = prepare_case(old, IEEE_CASE, 0)
    name = next(iter(case.ref))
    ref = case.ref[name]
    finite_ref = torch.isfinite(ref).reshape(-1)
    # A candidate that turns a FINITE reference element into NaN must be rejected.
    if finite_ref.any():
        index = int(finite_ref.nonzero()[0])
        assert torch.isfinite(ref.reshape(-1)[index])
        broken = {key: value.clone() for key, value in case.ref.items()}
        flat = broken[name].reshape(-1)
        flat[index] = torch.tensor(float("nan"), dtype=flat.dtype)
        error = scaled_error(broken[name], ref, case.scale[name], 0.002, 0.001, False, _floor("fp16"))
        assert not bool((error <= 1.0).all()), "a NaN where the reference is finite must fail"
    # Same-sign infinities agree; opposite signs do not.
    plus = torch.tensor([float("inf")], dtype=torch.float32)
    minus = torch.tensor([float("-inf")], dtype=torch.float32)
    same = scaled_error(plus, plus, torch.ones(()), 0.002, 0.001, False, 0.0)
    opposite = scaled_error(minus, plus, torch.ones(()), 0.002, 0.001, False, 0.0)
    assert bool((same <= 1.0).all()) and not bool((opposite <= 1.0).all())
