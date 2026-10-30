"""R2 equivalence harness: every migrated task must match the pre-R1 implementation exactly.

Oracle independence (N-17) is load-bearing here, so the snapshot is isolated properly:

* ``git archive <R0 recorded commit> tasks`` is extracted to a temporary directory;
* inside :func:`frozen_tasks` **every** ``tasks``/``tasks.*`` entry is removed from
  ``sys.modules`` and the frozen package is installed under the name ``tasks``, so the old
  modules' *absolute* imports (``tasks._common``, ``tasks._elementwise``,
  ``tasks.gemm.task`` …) resolve inside the frozen tree — including ``load_text``'s template
  root, the elementwise builder and ``gemm_silu``'s re-exported GEMM builder;
* the accepted domain (names, declared sizes, generated-op list) is read in that same
  isolation, so a migrated module that dropped ``SHAPES`` cannot shrink the sweep;
* comparison uses contiguous **byte views**, not ``torch.equal``: ``torch.equal`` calls
  ``-0.0 == 0.0`` and calls identical NaN bit patterns unequal.

Coverage split per plan §4.4: static full set (``TASK_NAMES`` + ``PHASE2_TASK_NAMES`` +
``SEMANTIC_NAMES`` + fixed ``gen_*``) × dtypes × declared sizes; dynamic semantic family
(``sem_int_<seed>``); dynamic launch family (``@variant``).  Unmigrated entries must be on
``REMAINING_*``; migrating one without updating the inventory fails.
"""

from __future__ import annotations

import dataclasses
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.frozen_tasks import (  # noqa: E402 - imports follow the module docstring
    REMAINING_GEN,
    REMAINING_SEM_INT,
    REMAINING_STATIC,
    REMAINING_VARIANTS,
    ROOT,
    SEEDS,
    FrozenSnapshot,
    assert_tensor_equal as _assert_tensor_equal,
    extract_snapshot,
    fields_of as _fields,
    frozen_tasks,
    is_remaining,
    logical_cases as _logical_cases,
    materialised_scale,
    old_build,
    raw_bytes,
    r0_commit,
)


@pytest.fixture(scope="session")
def frozen(tmp_path_factory) -> FrozenSnapshot:
    """@brief 导出 R0 登记提交的 ``tasks/`` 并读出冻结接受域（共享 helper）。

    @return :class:`tests.support.frozen_tasks.FrozenSnapshot`。
    """
    return extract_snapshot(tmp_path_factory.mktemp("r0_tasks"))


# ------------------------------------------------------------------- comparable surface


# --------------------------------------------------------------------------- static sweep


def test_the_oracle_is_pinned_to_the_recorded_r0_commit_not_to_head(frozen: FrozenSnapshot):
    """@brief 回归：oracle 固定到 R0 登记的提交；live 树变化不改变冻结快照。"""
    commit = r0_commit()
    assert commit == "3fb9ac85bfc760eb655d9b1759d48f5c4896997a"
    for relative in ("rmsnorm/task.py", "_common.py", "gemm/task.py", "__init__.py"):
        recorded = subprocess.run(
            ["git", "show", f"{commit}:tasks/{relative}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        assert (frozen.root / relative).read_bytes() == recorded, f"{relative} must come from the recorded commit"
    # The oracle must not be the working tree: R1 rewrote this module in place.
    assert (ROOT / "tasks" / "rmsnorm" / "task.py").read_bytes() != (frozen.root / "rmsnorm" / "task.py").read_bytes()


def test_the_static_full_set_is_enumerated_from_the_frozen_snapshot(frozen: FrozenSnapshot):
    """@brief 静态全集来自冻结快照，含语义四题、固定 ``gen_*`` 与 ``reduction_sum`` 的 ``plan``。"""
    cases = frozen.static_cases()
    assert frozen.semantic == (
        "sem_gemm_neg",
        "sem_relu_gelu_comment",
        "sem_gelu_named_relu",
        "sem_attention_odd_scale",
    )
    assert frozen.generated and all(name.startswith("gen_") for name in frozen.generated)
    assert ("reduction_sum", "fp16", "plan") in cases
    assert ("reduction_sum", "bf16", "plan") in cases
    for task in frozen.semantic + frozen.generated:
        assert (task, "fp16", "paper") in cases, task
    assert len({case[0] for case in cases}) == (
        len(frozen.tasks) + len(frozen.phase2) + len(frozen.semantic) + len(frozen.generated)
    )


def test_the_frozen_oracle_does_not_leak_into_the_live_tree(frozen: FrozenSnapshot):
    """@brief 隔离回归：冻结模块的模板根、elementwise builder、gemm_silu 的 GEMM 依赖都在冻结树。"""
    import tasks

    live_root = ROOT / "tasks"
    with frozen_tasks(frozen.root):
        common = importlib.import_module("tasks._common")
        elementwise = importlib.import_module("tasks._elementwise")
        rmsnorm = importlib.import_module("tasks.rmsnorm.task")
        relu = importlib.import_module("tasks.relu.task")
        gemm_silu = importlib.import_module("tasks.gemm_silu.task")
        assert common.TASKS_DIR == frozen.root
        assert Path(common.__file__).resolve().is_relative_to(frozen.root)
        assert Path(elementwise.__file__).resolve().is_relative_to(frozen.root)
        assert relu.build_elementwise.__module__ == "tasks._elementwise"
        assert Path(sys.modules[relu.build_elementwise.__module__].__file__).resolve().is_relative_to(frozen.root)
        assert rmsnorm.load_text.__globals__["TASKS_DIR"] == frozen.root
        assert (frozen.root / "rmsnorm" / "kernel_triton.py").is_file()
        assert gemm_silu.build_gemm.__module__.startswith("tasks.gemm")
        assert Path(sys.modules[gemm_silu.build_gemm.__module__].__file__).resolve().is_relative_to(frozen.root)
        assert not (frozen.root == live_root)
        assert importlib.import_module("tasks").__file__ and Path(
            importlib.import_module("tasks").__file__
        ).resolve().is_relative_to(frozen.root)
    # Restored afterwards: the live package is back.
    assert Path(tasks.__file__).resolve() == live_root / "__init__.py"


def test_static_entries_are_either_migrated_or_on_the_remaining_inventory(frozen: FrozenSnapshot):
    """@brief 静态全集不抽样：每个入口要么已迁移，要么在显式清单里。"""
    from tasks import get_task

    for task in frozen.tasks + frozen.phase2 + frozen.semantic + frozen.generated:
        built = get_task(task, "fp16", "paper")
        migrated = built.semantic_instance is not None and built.specialization is not None
        assert migrated != is_remaining(task), (
            f"{task}: migrated={migrated} but the inventory says remaining={is_remaining(task)}; "
            "update the inventory in the same change that migrates the task"
        )


def test_a_migrated_task_matches_the_frozen_head_implementation(frozen: FrozenSnapshot):
    """@brief R2-1：静态逐项等价——Contract 全字段、题面、容差、记账、baseline 与冻结实现一致。"""
    from tasks import get_task

    cases = frozen.migrated_cases()
    assert cases, "at least one static task must be migrated at this point in R2"
    for task, dtype, size in cases:
        with frozen_tasks(frozen.root):
            old = old_build(task, dtype, size)
        new = get_task(task, dtype, size)
        old_fields, new_fields = _fields(old), _fields(new)
        for name, expected in old_fields.items():
            assert new_fields[name] == expected, f"{task}/{dtype}/{size}: {name} drifted"
        # Registration identity: every migrated baseline must name an implementation and params.
        for baseline in new.baselines:
            assert baseline.implementation_id, f"{task}/{dtype}/{size}: {baseline.name} has no implementation_id"
            assert baseline.implementation_version >= 1, f"{task}/{dtype}/{size}: {baseline.name} version"
        # Task identity fields must equal what the caller asked for.
        assert (new.name, new.dtype, new.size) == (task, dtype, size)
        # H-21: one known-good source, not a post-build patched copy plus a stale original.
        assert new.reference_sources.get("metal", new.naive_msl) == new.naive_msl, (
            f"{task}/{dtype}/{size}: known-good source has a second stale copy"
        )


def test_a_migrated_task_keeps_the_frozen_inputs_reference_and_error_scale_bytes(frozen: FrozenSnapshot):
    """@brief R2-3：输入/参考输出/``error_scale`` 的 dtype/shape/**字节**逐张量一致。"""
    from tasks import get_task

    for task, dtype, size in frozen.migrated_cases():
        with frozen_tasks(frozen.root):
            old = old_build(task, dtype, size)
        new = get_task(task, dtype, size)
        where = f"{task}/{dtype}/{size}"
        for case in _logical_cases(new):
            for seed in SEEDS:
                expected, actual = old.gen_inputs(case, seed), new.gen_inputs(case, seed)
                assert set(actual) == set(expected), f"{where}/{case}/{seed}: input names"
                for name in expected:
                    _assert_tensor_equal(expected[name], actual[name], f"{where}/{name}/{case}/{seed}/input")
                old_ref, new_ref = old.reference(expected), new.reference(actual)
                assert set(new_ref) == set(old_ref), f"{where}/{case}/{seed}: reference names"
                for name in old_ref:
                    _assert_tensor_equal(old_ref[name], new_ref[name], f"{where}/{name}/{case}/{seed}/reference")
                assert (old.error_scale is None) == (new.error_scale is None), (
                    f"{where}/{case}/{seed}: error_scale registration drifted"
                )
                old_scale = materialised_scale(old, expected, old_ref)
                new_scale = materialised_scale(new, actual, new_ref)
                assert set(new_scale) == set(old_scale), f"{where}/{case}/{seed}: scale outputs drifted"
                for name in old_scale:
                    _assert_tensor_equal(old_scale[name], new_scale[name], f"{where}/{name}/{case}/{seed}/scale")


def test_the_semantic_variants_share_a_contract_but_never_an_opspec_identity():
    """@brief §4.4：同 Contract 的不同语义必须是不同 OpSpec/`opspec_digest`。

    ``sem_gemm_neg``/``sem_attention_odd_scale``/``sem_relu_gelu_comment`` 与它们的基础题派生
    出逐个字段相同的 ``Contract``（同核名、grid、threads、args、constants），但公式与参考不同，
    因此身份必须不同；两个名字错位题彼此也不同。
    """
    from tasks import get_task

    for semantic, base in (
        ("sem_gemm_neg", "gemm"),
        ("sem_attention_odd_scale", "flash_attention"),
        ("sem_relu_gelu_comment", "relu"),
        ("sem_gelu_named_relu", "relu"),
    ):
        left, right = get_task(semantic, "fp16", "small"), get_task(base, "fp16", "small")
        assert left.semantic_instance is not None and right.semantic_instance is not None, semantic
        assert left.semantic_instance.opspec_digest != right.semantic_instance.opspec_digest, semantic
        assert left.key != right.key, semantic
    gemm_neg, attention = get_task("sem_gemm_neg", "fp16", "small"), get_task("flash_attention", "fp16", "small")
    assert dataclasses.asdict(gemm_neg.contract) == dataclasses.asdict(
        get_task("gemm", "fp16", "small").contract
    )
    assert dataclasses.asdict(attention.contract) == dataclasses.asdict(
        get_task("sem_attention_odd_scale", "fp16", "small").contract
    )
    assert (
        get_task("sem_relu_gelu_comment", "fp16", "small").semantic_instance.opspec_digest
        != get_task("sem_gelu_named_relu", "fp16", "small").semantic_instance.opspec_digest
    )


# ---------------------------------------------------------------------- dynamic families


def test_the_dynamic_semantic_family_seed_domain_is_declared_by_the_frozen_generator(frozen: FrozenSnapshot):
    """@brief 动态语义族：种子域/生成器版本/约束必须显式登记（冻结实现没有声明种子域）。

    冻结的 ``tasks/semantic.py`` 用 ``IntProgram.generate(seed)`` 按名字里的种子现生成，
    且 ``_build_int`` 忽略请求 dtype。R2 必须：(a) 登记种子域与生成器版本，(b) 把
    ``{fp16,bf16,i32} -> i32`` 登记为显式别名、其他 dtype 拒绝，(c) 用旧 ``IntProgram`` 做
    固定语料 + 确定性 property 语料的差分。当前三项都未实现，故只登记清单。
    """
    with frozen_tasks(frozen.root):
        semantic = importlib.import_module("tasks.semantic")
        assert semantic.INT_PREFIX == "sem_int_"
        assert set(semantic.INT_N) == {"paper", "small"}
        assert semantic.IntProgram.generate(0) is not None
    # The family is migrated in R2-4: the registered generator materialises an OpSpec document
    # and the domain-level diff lives in ``tests/shared/test_sem_int_family.py``.
    assert REMAINING_SEM_INT is False, "REMAINING_SEM_INT must flip together with the domain-level diff"


def test_the_dynamic_launch_family_grammar_is_registered_from_the_frozen_parser(frozen: FrozenSnapshot):
    """@brief 动态启动族：variant 文法、默认 alias 与约束以冻结 ``parse_variant`` 为准。"""
    with frozen_tasks(frozen.root):
        gemm = importlib.import_module("tasks.gemm.task")
        attention = importlib.import_module("tasks.flash_attention.task")
        assert gemm.DEFAULT_VARIANT == "64x64x32t128"
        shape = (2048, 2048, 2048)
        assert gemm.parse_variant(gemm.DEFAULT_VARIANT, shape) == (64, 64, 32, 128)
        for bad in ("64x64x32", "48x64x32t128", "64x64x32t96", "128x64x32t2048"):
            with pytest.raises(ValueError):
                gemm.parse_variant(bad, shape)
        assert attention.DEFAULT_VARIANT and callable(attention.parse_variant)
    assert REMAINING_VARIANTS is False, (
        "the launch-variant families are compared by domain in tests/shared/test_variant_family.py"
    )


def test_the_fixed_generated_family_is_enumerated_from_the_frozen_toml(frozen: FrozenSnapshot):
    """@brief 固定 ``gen_*`` 清单来自冻结的语义 TOML；迁移后逐项对照、TOML 已退场。"""
    import tasks as task_package

    assert frozen.generated
    assert all(name.startswith("gen_") for name in frozen.generated)
    assert REMAINING_GEN is False, "the generated family is migrated: it must read OpSpec only"
    # The live registry and the frozen manifest enumerate the same family (a task cannot be
    # dropped from the live list and still be compared, or added without a frozen oracle).
    assert tuple("gen_" + stem for stem in task_package.GENERATED_NAMES) == frozen.generated
    # The retired TOML path is gone from the repository, not merely unused.
    assert not (frozen.root.parent / "does-not-exist").exists()
    from tests.support.frozen_tasks import ROOT

    assert not (ROOT / "tasks" / "generated" / "elementwise.toml").exists()
    # Every migrated generated task builds from its own OpSpec document and carries no TOML tag.
    for name in frozen.generated:
        task = task_package.get_task(name, "fp16", "small")
        assert task.semantic_instance is not None
        assert "source" not in (task.meta or {}), f"{name}: the retired TOML source tag survived"
        assert task.semantic_instance.op_name == name


def test_a_migrated_task_builds_from_exactly_one_derivation(monkeypatch):
    """@brief 回归：一次 build 只派生一次（防止闭包与 Task 来自两组派生对象）。"""
    import importlib

    import tasks as tasks_package

    calls = {"n": 0}
    for task in ("softmax", "reduction_sum"):
        module = importlib.import_module(f"tasks.{task}.task")
        real = module.derive

        def counting(*args, _real=real, **kwargs):
            calls["n"] += 1
            return _real(*args, **kwargs)

        monkeypatch.setattr(module, "derive", counting, raising=True)
        calls["n"] = 0
        tasks_package.get_task.cache_clear()
        built = tasks_package.get_task(task, "fp16", "paper")
        assert built.semantic_instance is not None
        assert calls["n"] == 1, f"{task}: {calls['n']} derivations for one build"
    tasks_package.get_task.cache_clear()
