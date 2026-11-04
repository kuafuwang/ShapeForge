"""B4: sem_attention_odd_scale and the sem_int family on the v2 standard entry, with byte-exact equivalence evidence.

The G0 corpus has no tensors for these two tasks (they were ``preserved_v1_out_of_scope``), so
B4 froze its own oracle *before* deleting the v1 assembly: 8 ``(task, dtype, size)`` combinations
× 18 ``(case, seed)`` records of input/reference/error-scale bytes, in
``docs/架构/跨硬件算子生成分工-阶段验收/B4/evidence/v1-frozen.json``.  ``compare.py`` replays the
v2 contract against those hashes; this module runs that comparison as a fixed CPU test and pins
the retirement boundary that goes with it:

* the v1 registry refuses both names by ``spec_version`` (typed refusal, never a fallback);
* their directories are v2-only and carry no launch facts (geometry is candidate-owned);
* the retired ``metal-conv2d``/``metal-lion`` profiles are gone from the Metal registry;
* the *current* inventory comes from the G0 boundary plus B4's own ``boundary.json`` — G0/A5/A6
  documents are not edited.

No GPU and no model call: the GPU slice uses the preserved naive candidates and lives separately.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

# 生产装配（拒绝已迁任务）必须绕过测试的 specimen 夹具。
from tests.support.v1_tasks import production_get_task as get_task
from tcenv.contract_errors import TaskContractError
from tcenv.taskcontract import discover_standard_tasks, inspect_standard_entry
from tests.support.boundaries import migrated_tasks, preserved_v1

ROOT = Path(__file__).resolve().parents[3]
B4 = ROOT / "docs" / "架构" / "跨硬件算子生成分工-阶段验收" / "B4"
TASKS = ("sem_attention_odd_scale", "sem_int_5", "sem_int_3", "sem_int_7", "sem_int_1", "sem_int_2", "sem_int_4")


def _load(name: str):
    """@brief 载入 B4 的证据工具模块（不是生产模块）。

    @param name 模块文件基名。
    @return 模块对象。
    """
    import sys

    path = B4 / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_b2_{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("task", TASKS)
def test_b4_every_frozen_v1_record_is_reproduced_byte_for_byte(task):
    """@brief 8 组合 × 18 算例：输入、fp64 参考与误差尺度全部逐字节一致。

    这不是容差比较：同一份数学 oracle、同一条 seed 流，任何一字节差异都是迁移缺陷。
    """
    compare = _load("compare_sem_int" if task.startswith("sem_int_") else "compare")
    evidence = "sem_int-frozen.json" if task.startswith("sem_int_") else "v1-frozen.json"
    frozen = json.loads((B4 / "evidence" / evidence).read_text(encoding="utf-8"))
    # sem_int_007 的 op_name 是 sem_int_7：冻结证据按 v1 题名（sem_int_007）记录。
    lookup = "sem_int_007" if task == "sem_int_7" else task
    combos = [combo for combo in frozen["combos"] if combo["task"] == lookup]
    assert len(combos) == 4, f"{task}: expected 2 dtypes × paper/small"
    for combo in combos:
        result = compare.compare_combo(combo)
        expected_records = 9 if task.startswith("sem_int_") else 18
        assert result["records"] == expected_records
        expected_inputs = (18 if task.startswith("sem_int_") else 54)
        assert result["exact_input_bytes"] == expected_inputs
        assert result["exact_reference_bytes"] == expected_records
        assert result["exact_scale_bytes"] == expected_records


@pytest.mark.parametrize("task", TASKS)
def test_b4_the_v1_registry_refuses_the_migrated_name(task):
    """@brief v1 registry 按 ``spec_version`` 拒绝，不给未知名、不给回退。"""
    for spelling in (task, f"{task}@128x64x32t256"):
        with pytest.raises(TaskContractError) as refused:
            get_task(spelling, "fp16", "small")
        assert "v2 standard entry" in str(refused.value), spelling


@pytest.mark.parametrize("task", TASKS)
def test_b4_the_standard_directory_is_v2_only_and_has_no_launch_facts(task):
    """@brief 目录只有 v2 入口：无 v1 装配标记，文档里没有任何 grid/threads/tile 事实。"""
    directory = ROOT / "tasks" / task
    entry = inspect_standard_entry(directory)
    assert entry.is_v2 and entry.files[:2] == ("opspec.json", "reference.py")
    for name in ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py"):
        assert not (directory / name).exists(), (task, name)
    document = json.loads((directory / "opspec.json").read_text(encoding="utf-8"))
    assert document["spec_version"] == 2 and document["op_name"] == task
    forbidden = {"grid", "threads", "threadgroup", "num_warps", "tile", "block_m", "block_s", "lowering_profile"}
    text = json.dumps(document).lower()
    for key in forbidden:
        assert f'"{key}"' not in text, f"{task}: launch fact {key} re-entered the task document"


def test_b4_the_retired_lowering_profiles_are_gone():
    """@brief B4 后 Metal 的 v1 LoweringProfile 全部退役（在干净子进程里核对，避免同进程注册缓存）。

    v1 已无任何任务，profile 注册表应为空；sem_int 动态族与 FA 的 profile 也在其内。
    """
    import subprocess
    import sys

    script = (
        "from tcenv import lowering\n"
        "lowering.ensure_lowerings('metal')\n"
        "print([p.name for p in lowering.registered_profiles('metal')])"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "[]", result.stdout


def test_b4_the_current_inventory_comes_from_the_boundary_files():
    """@brief 本批两题进入已迁移集并离开保留集；G0 的冻结边界不被后续批次改写。

    全局数量与完整保留集由 ``tests.support.boundaries`` 和各批次边界文件决定，本测试只锁
    本批事实，这样后续批次不必返工旧批次的断言。
    """
    migrated = set(migrated_tasks())
    assert {"sem_attention_odd_scale", "sem_int_5", "sem_int_3", "sem_int_7", "sem_int_1", "sem_int_2", "sem_int_4"} <= migrated
    preserved = set(preserved_v1())
    assert "sem_attention_odd_scale" not in preserved and "sem_int_5" not in preserved and "sem_int dynamic family" not in preserved
    assert preserved == set(), "B4 后 v1 保留集应为空（sem_int 动态族也已冻结为固定 v2 题）"
    g0 = json.loads(
        (ROOT / "docs" / "架构" / "跨硬件算子生成分工-G0" / "migration-boundary.json").read_text(encoding="utf-8")
    )
    assert len(g0["migration_candidates"]) == 16, "G0 的冻结清单不应变化"
    assert "sem_attention_odd_scale" in g0["preserved_v1_out_of_scope"], "G0 的冻结边界不应被后续批次改写"


def test_b4_discovery_finds_both_tasks_and_the_naive_candidates_are_preserved():
    """@brief v2 自动发现包含两题；naive 候选文本作为证据保留，供 GPU 切片使用。"""
    names = {locator.task_id for locator in discover_standard_tasks()}
    assert {"sem_attention_odd_scale", "sem_int_5", "sem_int_7"} <= names
    for task in TASKS:
        # sem_int_007 的候选按 v1 题名保存；sem_int_7 是它 op_name 对应的目录名。
        candidate = B4 / "evidence" / "candidates" / f"{'sem_int_007' if task == 'sem_int_7' else task}-naive.msl"
        assert candidate.is_file() and candidate.read_text(encoding="utf-8").strip(), task
