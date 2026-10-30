"""Tolerance / exact verdicts compared between the frozen R0 task and the migrated task.

This is the §4.4-3 verdict check: for every migrated ``(task, dtype, size)`` and every declared
logical case/seed, the **frozen** task produces the expected inputs, reference and error scale,
and both sides then decide pass/fail with *their own* ``(rtol, atol_rel, exact)`` and *their own*
per-output dtype floor (``_floor(arg.dtype)``, exactly as ``tcenv/backends/metal/testing.py``
and ``tcenv/triton_runner.py`` call it).

CPU only: no device is opened.
"""

from __future__ import annotations

import pytest
import torch

from tasks import get_task
from tcenv.backends.metal.testing import scaled_error, _floor
import tests.backends.metal.test_tolerance_verdict as _self
from tcenv.backends.metal import testing as metal_testing
from tests.support.frozen_tasks import (
    REMAINING_GEN,
    REMAINING_STATIC,
    SEEDS,
    extract_snapshot,
    frozen_tasks,
    is_remaining,
    old_build,
)

@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    """@brief 导出 R0 登记提交的 ``tasks/`` 快照。

    @return :class:`tests.support.frozen_tasks.FrozenSnapshot`。
    """
    return extract_snapshot(tmp_path_factory.mktemp("r0_tasks"))


def _logical_cases(task) -> list[str]:
    """@brief 生成器接管的逻辑 case（``ieee`` 由 harness 经 ``ieee_base`` 物化）。

    @param task 任务。
    @return case 名列表。
    """
    base = (task.meta or {}).get("ieee_base", "normal")
    return [base if name == "ieee" else name for name in task.cases]


def _verdict(task, inputs, reference, output) -> dict[str, bool]:
    """@brief 用该 task 自己的容差与**逐输出 dtype floor** 判决。

    @param task 任务。
    @param inputs 该 task 的输入。
    @param reference 该 task 的参考输出。
    @param output 候选输出。
    @return 输出名 → 是否在容差内。
    """
    if task.error_scale is not None:
        scale = task.error_scale(inputs, reference)
    else:
        scale = {name: torch.sqrt(torch.mean(value.double() ** 2)) for name, value in reference.items()}
    verdict = {}
    for arg in task.contract.outputs:
        name = arg.name
        error = scaled_error(
            output[name],
            reference[name],
            scale[name],
            task.rtol,
            task.atol_rel,
            task.exact,
            _floor(arg.dtype),
        )
        verdict[name] = bool((error <= 1.0).all())
    return verdict


def _perturb(reference: dict, factor: float, offset: float) -> dict:
    """@brief 对**每个**输出制造一个同形状扰动候选。

    @param reference 参考输出。
    @param factor 乘性系数。
    @param offset 加性偏移（用于保证 0 元素也被扰动）。
    @return 扰动后的输出字典。
    """
    out = {}
    for name, value in reference.items():
        out[name] = (value.double() * factor + offset).to(value.dtype)
    return out


def test_the_old_and_new_verdicts_agree_on_every_migrated_combination(frozen):
    """@brief H-15：冻结旧 Task 与迁移后 Task 的完整逐输出判决必须一致。

    候选覆盖三档：参考自身、容差内的近边界扰动、越界扰动；旧/新两侧各自用自己的
    ``(rtol, atol_rel, exact)`` 与**逐输出 dtype floor**，比较**完整 verdict dict**。
    case 域从冻结旧 Task 枚举，并断言旧/新 case 集一致（防止新实现缩窄扫描域）。
    """
    cases = frozen.migrated_cases()
    assert cases, "at least one static task must be migrated at this point in R2"
    for task_name, dtype, size in cases:
        with frozen_tasks(frozen.root):
            old = old_build(task_name, dtype, size)
        new = get_task(task_name, dtype, size)
        assert (old.exact, old.rtol, old.atol_rel) == (new.exact, new.rtol, new.atol_rel)
        assert set(old.cases) == set(new.cases), f"{task_name}: case domain shrank or grew"
        # Enumerate from the FROZEN task, not from the migrated one.
        base = (old.meta or {}).get("ieee_base", "normal")
        logical = [base if name == "ieee" else name for name in old.cases]
        for case in logical:
            for seed in SEEDS:
                expected = old.gen_inputs(case, seed)
                actual = new.gen_inputs(case, seed)
                old_ref, new_ref = old.reference(expected), new.reference(actual)
                for factor, offset, label in ((1.0, 0.0, "reference"), (1.0 + old.rtol / 2.0, 0.0, "inside"), (1.5, 1.0, "outside")):
                    old_verdict = _verdict(old, expected, old_ref, _perturb(old_ref, factor, offset))
                    new_verdict = _verdict(new, actual, new_ref, _perturb(new_ref, factor, offset))
                    where = f"{task_name}/{dtype}/{size}/{case}/{seed}/{label}"
                    assert set(old_verdict) == set(new_verdict) == {arg.name for arg in old.contract.outputs}
                    assert old_verdict == new_verdict, f"{where}: verdicts disagree {old_verdict} vs {new_verdict}"
                    if label == "reference":
                        assert all(old_verdict.values()), f"{where}: the reference must pass"
                    if label == "outside":
                        assert not any(old_verdict.values()), f"{where}: a 50% error must be rejected everywhere"
                # The near-boundary candidate must stay inside for at least one output.
                inside = _verdict(old, expected, old_ref, _perturb(old_ref, 1.0 + old.rtol / 2.0, 0.0))
                assert any(inside.values()), f"{task_name}/{case}/{seed}: inside-tolerance candidate rejected"


def _bump_first(output: dict, delta: int) -> dict:
    """@brief 只扰动第一个输出的第一个元素（用于 exact 判决的拒绝面）。

    @param output 输出字典。
    @param delta 加性偏移。
    @return 扰动后的新字典。
    """
    bumped = {name: value.clone() for name, value in output.items()}
    name = next(iter(bumped))
    flat = bumped[name].reshape(-1)
    flat[0] = flat[0] + delta
    return bumped


def test_the_old_and_new_exact_verdicts_agree_on_the_dynamic_integer_family(frozen):
    """@brief R2-4b：动态 ``exact=true`` 族进入同一旧/新判决 sweep。

    冻结旧 Task 产生 case/输入/参考与 scale；旧/新两侧各用自己的 ``(rtol, atol_rel, exact)``
    与逐输出 dtype floor 判决：参考输出必须通过，**单元素 +1/-1 扰动必须被拒绝**，两侧 verdict
    完全一致。这是 §4.4-3 对“首个真实 exact=true task 自动进入 sweep”的落实。
    """
    for seed in (0, 1, 2, 7, 42):
        name = f"sem_int_{seed}"
        for size in ("paper", "small"):
            with frozen_tasks(frozen.root):
                old = old_build(name, "fp16", size)
            new = get_task(name, "fp16", size)
            assert (old.exact, old.rtol, old.atol_rel) == (new.exact, new.rtol, new.atol_rel) == (True, 0.0, 0.0)
            for case in _logical_cases(old):
                for value in SEEDS:
                    expected, actual = old.gen_inputs(case, value), new.gen_inputs(case, value)
                    old_ref, new_ref = old.reference(expected), new.reference(actual)
                    candidates = (
                        ("reference", old_ref, new_ref),
                        ("single+1", _bump_first(old_ref, 1), _bump_first(new_ref, 1)),
                        ("single-1", _bump_first(old_ref, -1), _bump_first(new_ref, -1)),
                    )
                    for label, old_candidate, new_candidate in candidates:
                        old_verdict = _verdict(old, expected, old_ref, old_candidate)
                        new_verdict = _verdict(new, actual, new_ref, new_candidate)
                        where = f"{name}/{size}/{case}/{value}/{label}"
                        assert set(old_verdict) == set(new_verdict) == {arg.name for arg in old.contract.outputs}
                        assert old_verdict == new_verdict, f"{where}: verdicts disagree {old_verdict} vs {new_verdict}"
                        if label == "reference":
                            assert all(old_verdict.values()), f"{where}: the exact reference must pass"
                        else:
                            assert not any(old_verdict.values()), f"{where}: a one-element exact error must be rejected"


def test_scaled_error_is_exactly_one_at_the_inclusive_boundary():
    """@brief H-16：float64 精确构造 ``scaled_error`` 为 1−ε / 1 / 1+ε，钉住 ``<= 1``。

    ``ref = 1.0``、``scale = 0``、``rtol = 0.5``、``atol_rel = 0`` 使 ``tol = 0.5`` 二进制精确；
    ``out = 1.5`` 时归一化误差恰为 1，再用 ``nextafter`` 取两侧。
    """
    ref = torch.tensor([1.0, 1.0], dtype=torch.float64)
    scale = torch.zeros((), dtype=torch.float64)
    rtol, atol_rel = 0.5, 0.0

    def error(out: torch.Tensor) -> float:
        return float(scaled_error(out, ref, scale, rtol, atol_rel, False).max())

    at_boundary = torch.tensor([1.5, 1.0], dtype=torch.float64)
    assert error(at_boundary) == 1.0

    inside = at_boundary.clone()
    inside[0] = torch.nextafter(inside[0], torch.tensor(0.0, dtype=torch.float64))
    outside = at_boundary.clone()
    outside[0] = torch.nextafter(outside[0], torch.tensor(float("inf"), dtype=torch.float64))
    assert error(inside) < 1.0 < error(outside)

    assert bool((scaled_error(at_boundary, ref, scale, rtol, atol_rel, False) <= 1.0).all())
    assert not bool((scaled_error(outside, ref, scale, rtol, atol_rel, False) <= 1.0).all())


def test_the_floor_argument_comes_from_each_output_dtype(monkeypatch):
    """@brief floor 必须逐输出取 ``_floor(arg.dtype)``，不能拿 ``task.dtype`` 顶替。

    用一个输出为 bf16、task dtype 为 fp16 的合成 Contract，并 spy ``scaled_error`` 捕获实际参数；
    若实现误用 ``task.dtype``，捕获到的 floor 会是 fp16 的值而不是 bf16 的值。
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from tcenv.contract import AbiArg, Contract

    assert metal_testing._floor("bf16") != metal_testing._floor("fp16")
    contract = Contract(
        kernel_name="synthetic",
        grid=(1, 1, 1),
        threads=(32, 1, 1),
        args=(AbiArg("x", 0, "input", "fp16", (4,)), AbiArg("y", 1, "output", "bf16", (4,))),
    )
    task = SimpleNamespace(
        contract=contract, dtype="fp16", rtol=0.01, atol_rel=0.0, exact=False, error_scale=None
    )
    captured: list[float] = []

    def spy(output, reference, scale, rtol, atol_rel, exact, floor=0.0):
        captured.append(floor)
        return torch.zeros_like(reference)

    monkeypatch.setattr(_self, "scaled_error", spy)
    reference = {"y": torch.zeros(4, dtype=torch.bfloat16)}
    _verdict(task, {}, reference, reference)
    assert captured == [metal_testing._floor("bf16")]


def test_the_migration_inventory_matches_the_code(frozen):
    """@brief Metal 侧也检查“已迁移 ⇔ 不在 remaining”，单独跑本文件也不会静默漏扫。

    判据与 shared harness 共用 `tests.support.frozen_tasks.is_remaining`，没有第三份清单。
    """
    for task in frozen.tasks + frozen.phase2 + frozen.semantic + frozen.generated:
        built = get_task(task, "fp16", "paper")
        migrated = built.semantic_instance is not None and built.specialization is not None
        assert migrated != is_remaining(task), (
            f"{task}: migrated={migrated} but inventory says remaining={is_remaining(task)}; "
            "update tests/support/frozen_tasks.py in the same change"
        )
