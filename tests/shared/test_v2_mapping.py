"""A1: every shipped v1 OpSpec maps to a v2 instance with the same typed semantics.

This is a comparison, not a migration: the shipped tasks and their identity stay v1.  The
test-only mapping removes ``uses`` and adds the v2 discovery default, then instantiates both
loaders and compares every semantic field (tensors, attrs, tolerances, cases, accounting,
evaluation metadata).
"""

from __future__ import annotations

import pytest

from tcenv.opspec import instantiate, load_opspec

from ..support.v2_mapping import ROOT, TASKS, mapped_opspec, opspec_paths

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


def test_the_mapping_and_the_a5_migration_together_cover_the_frozen_g0_domain():
    """@brief 覆盖冻结域：8 个保留 v1 题/32 组合走映射，16 个迁移题/82 组合走 v2 标准入口。

    Before A5 this test required the *mapping alone* to span the 82 G0 combinations, because
    every G0 task was still v1.  A5 migrated those 16 tasks, so their v1 documents no longer
    exist and the mapping can only cover the tasks the boundary keeps on v1.  The frozen domain
    is still covered, but by the two authorities that now own it: the mapping for the preserved
    v1 tasks, and the v2 standard entry (checked field by field against the G0 corpus by
    ``tests/backends/metal/test_task_migration_a5.py``) for the migrated ones.

    Both sides are asserted exactly.  The preserved task set comes from the frozen
    ``preserved_v1_out_of_scope`` list, and its (task, dtype, size) combinations come from the
    v1 OpSpec documents — a count of whatever happens to be on disk would pass with one toy
    mapping.  The migrated side comes from the frozen ``migration_candidates`` list and the G0
    corpus, and must be exactly the 16 tasks / 82 combinations.
    """
    import json

    from tcenv.taskcontract import inspect_standard_entry

    from tests.support.boundaries import migrated_tasks, preserved_v1

    # B4 把最后两道 v1 题（sem_attention_odd_scale 与 sem_int 动态族）迁走，边界里的保留集为空。
    preserved = tuple(name for name in preserved_v1() if not name.startswith("sem_int "))
    assert preserved == (), "B4 之后 v1 已没有保留题；新增保留项必须在边界文件里登记"
    mapped: set[tuple[str, str, str]] = set()
    for path in OPSPECS:
        spec = load_opspec(path)
        for dtype in spec.dtypes:
            for size in spec.sizes:
                mapped.add((spec.op_name, str(dtype), str(size)))
    assert {name for name, _dtype, _size in mapped} == set(preserved), (
        "the mapping covers a different task set than the current preserved inventory"
    )
    for name in preserved:
        assert {dtype for task, dtype, _size in mapped if task == name} == {"fp16", "bf16"}, name
        assert {size for task, _dtype, size in mapped if task == name} == {"paper", "small"}, name

    # G0 的冻结域是 8 保留题/32 组合 + 16 迁移题/82 组合。B1–B4 逐批迁走这 8 题（2+2+3+1），
    # 所以「当前 v1 文档的映射组合」（B4 之后为 0）+「各批次冻结证据里的组合」必须仍恰好是 32 个。
    # sem_int 动态族另册（``sem_int-frozen.json``）：G0 把它记成一族，不在这 32 个并列组合里。
    batch_combos: set[tuple[str, str, str]] = set()
    for path in sorted((ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收").glob("B*/evidence/v1-frozen.json")):
        document = json.loads(path.read_text(encoding="utf-8"))
        batch_combos |= {(combo["task"], combo["dtype"], combo["size"]) for combo in document["combos"]}
    assert {task for task, _dtype, _size in batch_combos} == {
        "gemv", "rope", "conv2d", "lion", "flash_attention", "bitdelta_gemm",
        "mamba2_chunk_scan", "sem_attention_odd_scale",
    }
    assert len(mapped) + len(batch_combos) == 32, (
        f"the original preserved domain is 32 combinations; mapping has {len(mapped)}, batches have {len(batch_combos)}"
    )
    assert not (mapped & batch_combos), "a migrated task is still offered as a v1 document"

    boundary = json.loads(
        (ROOT / "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json").read_text(encoding="utf-8")
    )
    migrated = {str(entry["task"]) for entry in boundary["migration_candidates"]}
    assert len(migrated) == 16, f"the frozen migration inventory has {len(migrated)} tasks"  # G0 文档本身不变
    corpus = json.loads(
        (ROOT / "docs/架构/跨硬件算子生成分工-G0/corpus/index.json").read_text(encoding="utf-8")
    )
    migrated_combos = [combo for combo in corpus["combos"] if combo["task"] in migrated]
    assert {combo["task"] for combo in migrated_combos} == migrated
    assert len(migrated_combos) == 82, f"the frozen corpus lost combinations: {len(migrated_combos)}"
    for name in sorted(set(migrated_tasks())):
        entry = inspect_standard_entry(ROOT / "tasks" / name)
        assert entry.is_v2, f"{name}: the migrated task is not on the v2 standard entry"
