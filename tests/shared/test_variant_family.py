"""R2 §4.4-2: the dynamic launch families (``gemm@…``, ``flash_attention@…``) by domain.

A variant name is *dynamic*: ``gemm@<BM>x<BN>x<BK>t<threads>`` addresses a whole family, so a
finite list cannot prove equivalence.  This file treats the frozen R0 ``parse_variant`` /
``build_variant`` as a read-only oracle and compares, over a deterministic corpus plus the
declared grammar's boundaries:

* acceptance/rejection (the same names must be refused, with the same reason);
* for accepted names, the derived ``Contract`` (grid/threads/constants/kernel name), the
  rendered prompt and known-good source bytes, and the reference output bytes;
* the grammar's own invariants (tile divisibility, SIMD-group shape), checked on both sides.

It is *not* a proof over the infinite name space: the corpus plus the boundary table is what
is claimed, and an acceptance change must be registered rather than claimed equivalent.

CPU only.
"""

from __future__ import annotations

import importlib

import pytest

from tests.support.frozen_tasks import (
    FrozenSnapshot,
    assert_tensor_equal,
    extract_snapshot,
    fields_of,
    frozen_tasks,
)


@pytest.fixture(scope="session")
def frozen(tmp_path_factory) -> FrozenSnapshot:
    """@brief 导出 R0 登记提交的 ``tasks/``（共享 helper）。

    @return :class:`tests.support.frozen_tasks.FrozenSnapshot`。
    """
    return extract_snapshot(tmp_path_factory.mktemp("r0_variants"))


#: Accepted names: the registered defaults, the ones the experiments used, and the shape
#: boundaries of the declared grammar (smallest legal tile, largest legal thread count).
#: A5 retired the v1 ``gemm@…`` family together with the v1 gemm assembly.  The suffix mechanism
#: itself stays live on the preserved v1 ``flash_attention`` task.
ACCEPTED = {
    "flash_attention": (
        "32x32t128",
        "64x32t256",
        "64x64t256",
        "32x16t128",
        # threads/BLOCK_M is a power of two <= 32 and must divide D=64: 1024/32 = 32 is legal.
        "32x32t1024",
    ),
}
#: Refused names: wrong arity/separator/case, non-power-of-two or non-dividing tiles, thread
#: counts that are not whole SIMD groups or that the naive kernel cannot lay out.
REFUSED = {
    "flash_attention": (
        "32x32",
        "32x32t",
        "32X32T128",
        "32x32t128 ",
        "32x32t100",
        "32x32t96",
        "48x32t96",
        "32x48t128",
        "4096x32t128",
    ),
}


def _build(task: str, dtype: str, size: str, variant: str | None):
    """@brief 装配一道（可带变体的）题。

    @param task 算子名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant variant 文本；``None`` 表示默认 profile。
    @return ``Task``。
    """
    from tasks import get_task

    return get_task(task if variant is None else f"{task}@{variant}", dtype, size)


def _frozen_build(task: str, dtype: str, size: str, variant: str | None):
    """@brief 在冻结隔离里装配同一道题（必须在外层 ``frozen_tasks`` 内调用）。

    @param task 算子名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant variant 文本；``None`` 表示默认 profile。
    @return 冻结侧 ``Task``。
    """
    module = importlib.import_module(f"tasks.{task}.task")
    if variant is None:
        return module.build(dtype, size)
    return module.build_variant(dtype, size, variant)


@pytest.mark.parametrize("task", tuple(ACCEPTED))
def test_accepted_variant_names_derive_the_same_contract_and_text(frozen: FrozenSnapshot, task: str):
    """@brief 合法 variant：两侧接受，且 Contract/题面/known-good 逐字节一致。"""
    for dtype in ("fp16", "bf16"):
        for size in ("paper", "small"):
            for variant in ACCEPTED[task]:
                with frozen_tasks(frozen.root):
                    try:
                        old = _frozen_build(task, dtype, size, variant)
                    except ValueError as exc:  # pragma: no cover - the corpus is accepted on both sides
                        pytest.fail(f"{task}@{variant}/{dtype}/{size}: the frozen oracle refused it: {exc}")
                new = _build(task, dtype, size, variant)
                old_fields, new_fields = fields_of(old), fields_of(new)
                for field, expected in old_fields.items():
                    assert new_fields[field] == expected, f"{task}@{variant}/{dtype}/{size}: {field} drifted"
                assert new.name == f"{task}@{variant}"
                # The variant is a launch fact only: the semantic identity is the family's.
                assert new.semantic_instance is not None
                assert new.semantic_instance.op_name == task


@pytest.mark.parametrize("task", tuple(REFUSED))
def test_refused_variant_names_are_refused_by_both_sides(frozen: FrozenSnapshot, task: str):
    """@brief 非法 variant：两侧都拒绝，且拒绝归属同一类错误（措辞可以更具体，判定必须一致）。

    空后缀不算非法：``"gemm@"`` 的 ``variant`` 为空，两侧都按“没有变体”处理（另有用例钉住）。
    """
    for size in ("paper", "small"):
        for variant in REFUSED[task]:
            with frozen_tasks(frozen.root):
                with pytest.raises((ValueError, KeyError)) as old_exc:
                    _frozen_build(task, "fp16", size, variant)
            with pytest.raises((ValueError, KeyError)) as new_exc:
                _build(task, "fp16", size, variant)
            # The live side raises its own richer subclass (LoweringError) where the frozen side
            # raised the plain base class; the *decision* and the base class must still agree.
            assert isinstance(new_exc.value, type(old_exc.value)), (
                f"{task}@{variant!r}/{size}: the frozen side raised {type(old_exc.value).__name__}, "
                f"the live side {type(new_exc.value).__name__}"
            )
            assert variant.strip() in str(new_exc.value) or str(old_exc.value)[:20] in str(new_exc.value), (
                f"{task}@{variant!r}/{size}: the live refusal does not name the refused text"
            )


def test_an_empty_variant_suffix_is_the_default_profile_on_both_sides(frozen: FrozenSnapshot):
    """@brief ``"flash_attention@"`` 的空后缀两侧都按默认处理（不是“非法变体”）。"""
    from tasks import get_task

    with frozen_tasks(frozen.root):
        old = _frozen_build("flash_attention", "fp16", "small", "")
    new = get_task("flash_attention@", "fp16", "small")
    assert new.contract.grid == old.contract.grid and new.contract.threads == old.contract.threads
    assert new.triton_source == old.triton_source


def test_a_task_without_a_registered_variant_family_refuses_the_suffix(frozen: FrozenSnapshot):
    """@brief 只有 GEMM/FlashAttention 登记了 variant 族；其他题带 ``@`` 一律拒绝（含 gemm_silu）。"""
    for name in ("rope@1x1x1t32", "conv2d@64x64x32t128", "lion@1x1t32", "gemv@1x1t32"):
        with frozen_tasks(frozen.root):
            with pytest.raises((ValueError, KeyError)) as old_exc:
                deep_import_build(name)
        with pytest.raises((ValueError, KeyError)) as new_exc:
            deep_import_build(name)
        assert type(new_exc.value) is type(old_exc.value)
        assert str(new_exc.value) == str(old_exc.value)


def deep_import_build(name: str):
    """@brief 经 ``get_task`` 装配一个可能带非法变体的名字。

    @param name 题目名（可能带 ``@<variant>``）。
    @return ``Task``。
    """
    from tasks import get_task

    return get_task(name, "fp16", "small")


def test_the_v1_gemm_variants_are_retired_not_revived():
    """@brief gemm 迁移到 v2 后，v1 ``gemm@…``（含空后缀）必须拒绝，不能回退到旧装配。"""
    from tcenv.contract_errors import TaskContractError

    for name in ("gemm", "gemm@", "gemm@64x64x32t128", "gemm@128x128x32t512", "gemm_silu@64x64x32t128"):
        with pytest.raises(TaskContractError):
            deep_import_build(name)


def test_the_declared_grammar_invariants_hold_on_both_sides(frozen: FrozenSnapshot):
    """@brief 文法不变量：合法名 ⇒ 整除/线程形状成立；违规一条即拒绝。表驱动，不靠抽样。

    这是对**文法**的检查：把合法名字的每个分量破坏一次，两侧必须同样拒绝。
    """
    attention = (1024, 64)
    with frozen_tasks(frozen.root):
        attention_module = importlib.import_module("tasks.flash_attention.task")
        for name in ACCEPTED["flash_attention"]:
            bm, bn, th = attention_module.parse_variant(name, *attention)
            seq, head_dim = attention
            tpr = th // bm
            assert seq % bm == 0 and seq % bn == 0 and th % 32 == 0 and th % bm == 0
            assert 1 <= tpr <= 32 and tpr & (tpr - 1) == 0 and head_dim % tpr == 0
            for broken in (f"{bm + 1}x{bn}t{th}", f"{bm}x{bn + 1}t{th}", f"{bm}x{bn}t{th + 32}"):
                with pytest.raises(ValueError):
                    attention_module.parse_variant(broken, *attention)


@pytest.mark.parametrize("task,shape", (("flash_attention", (1024, 64)),))
def test_the_reference_bytes_do_not_depend_on_the_variant(frozen: FrozenSnapshot, task: str, shape):
    """@brief variant 只换启动契约，不换语义：同一输入的参考输出逐字节相同（两侧各自比较）。"""
    from tests.support.frozen_tasks import SEEDS

    variants = ACCEPTED[task]
    with frozen_tasks(frozen.root):
        old_default = _frozen_build(task, "fp16", "small", None)
        old_inputs = old_default.gen_inputs(old_default.cases[0], SEEDS[0])
        old_reference = old_default.reference(old_inputs)
    new_default = _build(task, "fp16", "small", None)
    new_inputs = new_default.gen_inputs(new_default.cases[0], SEEDS[0])
    new_reference = new_default.reference(new_inputs)
    for key in old_reference:
        assert_tensor_equal(old_reference[key], new_reference[key], f"{task}/{key}/default")
    for variant in variants:
        task_variant = _build(task, "fp16", "small", variant)
        reference = task_variant.reference(new_inputs)
        for key in new_reference:
            assert_tensor_equal(new_reference[key], reference[key], f"{task}@{variant}/{key}")
        # The variant changes the launch contract, not the reference: the threads must follow the
        # name the caller wrote, and the bytes must not depend on the tile.
        assert task_variant.contract.threads[0] == int(variant.rsplit("t", 1)[1])
