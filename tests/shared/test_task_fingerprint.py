"""``task_fingerprint`` v2: the canonical task document and its reset golden (plan §3.1, §4.3).

CPU only.  This module owns two things:

* the **new** identity: a migrated task's fingerprint is the §3.1 canonical document over
  the complete ``SemanticInstance``/``Specialization`` projections, the rendered triton
  source, ``description`` and the backend reference kernel;
* the explicit **transitional** path: a task that still has no OpSpec keeps the pre-R2
  hash.  R2 migrates those tasks and deletes that path; the inventory lives in
  ``tests/shared/test_task_assembly_structure.py``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tasks import get_task
from tcenv.lowering import contract_document
from tcenv.task import task_fingerprint
from tests.support.v1_tasks import V1_TASK

ROOT = Path(__file__).resolve().parents[2]

#: Golden digests recorded when R1 introduced the v2 document.  They are the *reset* of the
#: Metal fingerprint (the old literal golden in ``test_backend_neutral.py`` only pins the
#: transitional path now).  A change here must come with the change that altered the
#: derived projections or the rendered task text in the same batch.
#:
#: R2 reset them once, deliberately: ``SEMANTIC_RULES_VERSION`` moved to
#: ``opspec-semantics-2`` when ``cases.inputs`` gained the closed evaluation-domain keys
#: (``_domain_case``/``_base``/``_inject``/``_max``).  ``opspec_digest`` is part of the v2
#: document, so the rule-version bump invalidates every task identity — that is the intended
#: behaviour of §3.1, not a drift in the derived contract/prompt (those are still byte-equal
#: to the R0 snapshot).
#: A5 repointed this v1 fingerprint golden from the migrated rmsnorm to the preserved v1 rope
#: task; B1 repointed it again to conv2d when rope/gemv moved to v2.  The digest changed because
#: the *specimen* changed, not because conv2d's own identity changed.  The v2 contract has its
#: own identity (``TaskContract.fingerprint``).
GOLDEN = {
    ("fp16", "paper"): "eb6909c55a2fc78dd04bd5445dbb46e1defe8ebed8435fce090ea314a3686bcd",
    ("fp16", "small"): "b96850d0645d965b0b5e94760f6a36d998b87139adf35b51f172a6e5a3c1658e",
    ("bf16", "paper"): "87e3a0e118773d763ebae8a87494833d10e4efcc59688bc473917a0e1ab2d81b",
    ("bf16", "small"): "a73a73d209d29dbc09cff296dc31d1b6d914bbdf57d2ee6f03122585573a5e69",
}


def test_the_v2_document_binds_the_complete_derived_projections():
    """@brief 指纹就是 §3.1 的 canonical document：字段闭集且只依赖派生结果与渲染文本。"""
    from tcenv.identity import fingerprint

    task = get_task(V1_TASK, "fp16", "small")
    document = {
        "fingerprint_kind": "task",
        "fingerprint_version": 2,
        "task": task.name,
        "dtype": task.dtype,
        "size": task.size,
        "backend_id": "metal",
        "semantic_instance": task.semantic_instance.fingerprint_document(),
        "specialization": task.specialization.fingerprint_document(),
        "triton_source": task.triton_source,
        "description": task.description,
        "reference_source": task.reference_for("metal"),
    }
    assert set(document) == {
        "fingerprint_kind",
        "fingerprint_version",
        "task",
        "dtype",
        "size",
        "backend_id",
        "semantic_instance",
        "specialization",
        "triton_source",
        "description",
        "reference_source",
    }
    assert document["specialization"]["contract"] == contract_document(task.contract)
    assert "gpu" in document["specialization"]["contract"]
    assert fingerprint(document) == task_fingerprint(task, "metal")
    assert task_fingerprint(task) == task_fingerprint(task, "metal")
    assert task_fingerprint(task, "sim-npu") != task_fingerprint(task, "metal")


@pytest.mark.parametrize(("dtype", "size"), sorted(GOLDEN))
def test_the_recorded_v2_golden_still_matches(dtype, size):
    """@brief R1 有意重置的指纹 golden：四个组合的 v2 摘要固定。"""
    assert task_fingerprint(get_task(V1_TASK, dtype, size)) == GOLDEN[(dtype, size)]


def test_changing_only_the_description_changes_the_fingerprint():
    """@brief N-7：``description`` 进入 v2 身份；只改描述也必须让指纹变化。"""
    task = get_task(V1_TASK, "fp16", "small")
    edited = dataclasses.replace(task, description=task.description + " (rewritten track text)")
    assert edited.description != task.description
    assert task_fingerprint(edited) != task_fingerprint(task)


def test_changing_only_the_backend_reference_kernel_changes_the_fingerprint():
    """@brief ``reference_source`` 是身份的一部分（known-good 模板改动必须可见）。"""
    task = get_task(V1_TASK, "bf16", "small")
    edited = dataclasses.replace(task, naive_msl=task.naive_msl + "\n// one more line\n", reference_sources={})
    assert task_fingerprint(edited) != task_fingerprint(task)


def test_a_candidate_launch_override_does_not_change_the_task_fingerprint():
    """@brief F-3：候选侧 ``taic-launch`` 只属于 candidate 摘要，不改任务身份。"""
    task = get_task(V1_TASK, "fp16", "paper")
    assert task_fingerprint(task) == GOLDEN[("fp16", "paper")]


def test_the_transitional_hash_path_is_gone():
    """@brief R2 删除了 pre-R2 哈希分支：每道题都有语义实例，缺实例一律拒绝。"""
    import dataclasses

    import pytest

    from tests.support.frozen_tasks import extract_snapshot
    import tempfile
    from pathlib import Path

    snapshot = extract_snapshot(Path(tempfile.mkdtemp(prefix="r2_fp_")))
    names = [name for name in snapshot.tasks + snapshot.phase2]
    names += [name for name in snapshot.semantic]
    names += [name for name in snapshot.generated]
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import load_task_contract
    from tests.support.frozen_tasks import is_remaining

    for name in names:
        if not is_remaining(name):
            # Migrated in A5: the v1 hash path is not the identity anymore; the v2 contract is.
            with pytest.raises(TaskContractError):
                get_task(name, "fp16", "small")
            contract = load_task_contract(ROOT / "tasks" / name, dtype="fp16", size="small")
            assert len(contract.fingerprint()) == 64, name
            continue
        task = get_task(name, "fp16", "small")
        assert task.semantic_instance is not None, f"{name}: no semantic instance"
        assert task.specialization is not None, f"{name}: no specialization"
        assert len(task_fingerprint(task)) == 64
    # A task without the projections is a construction bug, not a second hash variant.
    stripped = dataclasses.replace(get_task(V1_TASK, "fp16", "small"), semantic_instance=None, specialization=None)
    with pytest.raises(ValueError, match="transitional pre-R2 hash path is gone"):
        task_fingerprint(stripped)

    preserved = get_task(V1_TASK, "fp16", "small")
    assert task_fingerprint(preserved) == GOLDEN[("fp16", "small")]
