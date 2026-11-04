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
``REMAINING_*``; migrating one without updating the inventory fails.  The live registry only
enumerates preserved v1 names (A6 removed the migrated name entries); the R0 snapshot below is
the frozen oracle for both sides.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.contract_errors import TaskContractError
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.pack import task_pack_closure
from tcenv.track import REPO_ROOT as TRACK_ROOT

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


def _v2(task: str, dtype: str = "fp16", size: str = "paper"):
    """@brief 读取已迁移题的标准 v2 合同（不经过 v1 registry）。

    @param task 已迁移题名。
    @param dtype 元素类型。
    @param size 规模档位。
    @return ``TaskContract``。
    """
    return load_task_contract(ROOT / "tasks" / task, dtype=dtype, size=size)


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
    # The oracle must not be the working tree: A5 deleted these v1 assemblies and replaced the
    # directory with the v2 standard entry (the frozen copy still has the old task.py).
    for task in ("rmsnorm", "gemm"):
        assert not (ROOT / "tasks" / task / "task.py").exists(), task
        assert json.loads((ROOT / "tasks" / task / "opspec.json").read_text())["spec_version"] == 2, task


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
    # 拒绝路径必须走生产注册表：测试会把 ``tasks.get_task`` 指向归档 specimen。
    from tests.support.v1_tasks import production_get_task as get_task

    for task in frozen.tasks + frozen.phase2 + frozen.semantic + frozen.generated:
        migrated = not is_remaining(task)
        if migrated:
            with pytest.raises(TaskContractError):
                get_task(task, "fp16", "paper")
            assert _v2(task).op_name == task
        else:
            built = get_task(task, "fp16", "paper")
            assert built.semantic_instance is not None
        assert migrated != is_remaining(task), (
            f"{task}: migrated={migrated} but the inventory says remaining={is_remaining(task)}; "
            "update the inventory in the same change that migrates the task"
        )


def test_a_migrated_task_matches_the_frozen_head_implementation(frozen: FrozenSnapshot):
    """@brief A5：迁移题的语义字段与冻结 v1 实现一致，且 v1 registry 不再装配它。

    A5 retired the v1 ABI/launch contract (the candidate owns geometry), so this comparison is
    the semantic surface the standard entry still owns: cases, tolerances, work accounting and
    baseline identity.  The byte-level input/reference/error-scale comparison follows below; the
    full 82-combination G0 comparison lives in ``tests/backends/metal/test_task_migration_a5.py``.
    """
    from tests.support.v1_tasks import production_get_task as get_task

    cases = frozen.migrated_cases()
    assert cases, "A5 must leave migrated static tasks for this harness"
    for task, dtype, size in cases:
        with frozen_tasks(frozen.root):
            old = old_build(task, dtype, size)
        contract = _v2(task, dtype, size)
        where = f"{task}/{dtype}/{size}"
        assert (contract.op_name, contract.dtype, contract.size) == (task, dtype, size), where
        assert tuple(contract.instance.case_domain.cases) == tuple(old.cases), where
        tolerances = contract.instance.tolerances
        assert (tolerances.rtol, tolerances.atol_rel, tolerances.exact) == (old.rtol, old.atol_rel, old.exact), where
        work = contract.instance.work_accounting
        assert (work.flops, work.min_bytes) == (float(old.flops), float(old.min_bytes)), where
        assert [(entry.name, entry.framework) for entry in contract.baselines] == [
            (entry.name, entry.framework) for entry in old.baselines
        ], where
        with pytest.raises(TaskContractError):
            get_task(task, dtype, size)


def test_a_migrated_task_keeps_the_frozen_inputs_reference_and_error_scale_bytes(frozen: FrozenSnapshot):
    """@brief A5：输入/参考输出/``error_scale`` 的 dtype/shape/**字节**与冻结 v1 逐张量一致。"""
    from tests.support.frozen_tasks import materialised_scale as _scale

    for task, dtype, size in frozen.migrated_cases():
        with frozen_tasks(frozen.root):
            old = old_build(task, dtype, size)
        contract = _v2(task, dtype, size)
        where = f"{task}/{dtype}/{size}"
        for case in old.cases:
            logical = old.meta.get("ieee_base", "normal") if case == "ieee" else case
            for seed in SEEDS:
                # The v1 ``gen_inputs`` never injected IEEE; the harness did it separately.  The
                # A5 G0 comparison covers that injection; here both sides generate the logical
                # base case so the sampler bytes are comparable.
                expected = old.gen_inputs(logical, seed)
                actual = contract.reference.generate(contract.instance, logical, seed)
                assert set(actual) == set(expected), f"{where}/{case}/{seed}: input names"
                for name in expected:
                    _assert_tensor_equal(expected[name], actual[name], f"{where}/{name}/{case}/{seed}/input")
                old_ref = old.reference(expected)
                new_ref = contract.reference.execute(contract.instance, actual)
                assert set(new_ref) == set(old_ref), f"{where}/{case}/{seed}: reference names"
                for name in old_ref:
                    _assert_tensor_equal(old_ref[name], new_ref[name], f"{where}/{name}/{case}/{seed}/reference")
                old_scale = _scale(old, expected, old_ref)
                new_scale = contract.reference.error_scale(contract.instance, actual, new_ref)
                if new_scale is None:
                    import torch

                    new_scale = {name: torch.sqrt(torch.mean(value.double() ** 2)) for name, value in new_ref.items()}
                assert set(new_scale) == set(old_scale), f"{where}/{case}/{seed}: scale outputs drifted"
                for name in old_scale:
                    _assert_tensor_equal(old_scale[name], new_scale[name], f"{where}/{name}/{case}/{seed}/scale")


def test_the_semantic_variants_share_a_contract_but_never_an_opspec_identity():
    """@brief §4.4：同 Contract 的不同语义必须是不同 OpSpec/`opspec_digest`。

    ``sem_attention_odd_scale`` stays v1, so it is compared through ``get_task``.  The other
    semantic variants and their bases migrated in A5; their identity is the v2 OpSpec digest.
    """
    from tasks import get_task

    pairs = (
        ("sem_gemm_neg", "gemm", True),
        ("sem_attention_odd_scale", "flash_attention", False),
        ("sem_relu_gelu_comment", "relu", True),
        ("sem_gelu_named_relu", "relu", True),
    )
    digests: dict[str, str] = {}
    for semantic, base, migrated in pairs:
        if semantic == "sem_attention_odd_scale":
            # B3：sem_attention_odd_scale 仍在 v1，而它的基础题 flash_attention 已迁 v2。
            # 测试意图不变——同一 Contract 形状下，语义变体与基础题的 OpSpec 身份必须不同。
            left = get_task(semantic, "fp16", "small")
            right = _v2(base, "fp16", "small")
            digests[semantic] = left.semantic_instance.opspec_digest
            assert left.semantic_instance.opspec_digest != right.instance.opspec_digest, semantic
            from tcenv.task import task_fingerprint

            assert task_fingerprint(left) != right.fingerprint(), semantic
            continue
        if migrated:
            left = _v2(semantic, "fp16", "small")
            right = _v2(base, "fp16", "small")
            digests[semantic] = left.instance.opspec_digest
            assert left.instance.opspec_digest != right.instance.opspec_digest, semantic
            assert left.fingerprint() != right.fingerprint(), semantic
        else:
            left = get_task(semantic, "fp16", "small")
            right = get_task(base, "fp16", "small")
            digests[semantic] = left.semantic_instance.opspec_digest
            assert left.semantic_instance.opspec_digest != right.semantic_instance.opspec_digest, semantic
            assert left.key != right.key, semantic
    assert digests["sem_relu_gelu_comment"] != digests["sem_gelu_named_relu"]


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
    # ``sem_int`` has no standard directory; A5 kept it on v1, and the domain-level diff lives
    # in ``tests/shared/test_sem_int_family.py``.
    assert REMAINING_SEM_INT is True, "the dynamic sem_int family stays on the v1 path in A5"


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
    # ``flash_attention@…`` stays v1; the gemm variants retired with the v1 gemm assembly.
    assert REMAINING_VARIANTS is True, (
        "A5 retires gemm@… but keeps flash_attention@…; see tests/shared/test_variant_family.py"
    )


def test_the_fixed_generated_family_is_the_v2_standard_entry_only(frozen: FrozenSnapshot):
    """@brief 固定 ``gen_*`` 清单来自冻结的语义 TOML；A6 后只有 v2 标准目录读它。

    R0 的 TOML 清单仍是冻结 oracle（它决定要对照哪四道题），但活的仓库里不再有第二份
    ``gen_*`` 名字清单：A6 删除了 ``tasks/generated/`` 与 ``tasks.GENERATED_NAMES``，
    迁移题只能从自己的 ``opspec.json`` + ``reference.py`` 装配。
    """
    import tasks as task_package

    assert frozen.generated
    assert all(name.startswith("gen_") for name in frozen.generated)
    assert REMAINING_GEN is False, "the generated family is migrated: it must read OpSpec only"
    # The v1 registry no longer carries a generated-family name list, and the retired parser
    # directory is gone from the repository (not merely unused).
    assert not hasattr(task_package, "GENERATED_NAMES")
    from tests.support.frozen_tasks import ROOT

    assert not (ROOT / "tasks" / "generated").exists()
    assert not (ROOT / "tasks" / "_pointwise.py").exists()
    assert not (ROOT / "tasks" / "_matmul.py").exists()
    # Every migrated generated task reads its own v2 standard entry; the v1 registry refuses it
    # and the retired TOML tag cannot survive because the v2 document has no such field.
    from tests.support.v1_tasks import production_get_task as get_task

    for name in frozen.generated:
        with pytest.raises(TaskContractError):
            get_task(name, "fp16", "small")
        contract = _v2(name, "fp16", "small")
        assert contract.instance.op_name == name
        assert "source" not in contract.spec.document


def test_a_migrated_task_builds_from_exactly_one_derivation():
    """@brief A5：v2 标准入口只读自己的 OpSpec/reference/pack，不 import 退役的 v1 task 模块。"""
    import sys

    for task in ("softmax", "reduction_sum"):
        assert not (ROOT / "tasks" / task / "task.py").exists(), task
        before = {name for name in sys.modules if name == "tasks" or name.startswith("tasks.")}
        contract = _v2(task, "fp16", "paper")
        after = {name for name in sys.modules if name == "tasks" or name.startswith("tasks.")}
        assert after == before, f"{task}: the v2 loader imported a v1 task module"
        assert contract.instance.op_name == task
        closure = task_pack_closure(TRACK_ROOT / "tasks" / task, root=TRACK_ROOT)
        assert "opspec.json" in closure.task_files and "reference.py" in closure.task_files
        from tests.support.pack_identity import assert_task_pack_identity

        # The contract pack digest and the deployment closure digest are different domains;
        # each is recomputed from its own file set instead of a 64-character shape check.
        assert_task_pack_identity(
            contract, closure, task_dir=TRACK_ROOT / "tasks" / task, repo_root=TRACK_ROOT,
        )
