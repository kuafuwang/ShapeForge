"""RMSNorm migrated to the v2 standard entry matches the frozen G0 corpus.

A5 deleted the v1 ``tasks/rmsnorm/task.py`` lowering this file used to compare byte-for-byte
against the R0 draft.  The independent oracle is now the frozen G0 corpus itself: every
rmsnorm ``(dtype, size)`` combination must reproduce its input/reference/scale bytes, tensors,
tolerances, work accounting and baseline identity.  No GPU.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
A5 = ROOT / "docs/架构/跨硬件算子生成分工-阶段验收/A5"


def _load_g0():
    """@brief 载入 A5 的 G0 对照工具（不是生产模块）。

    @return 模块对象。
    """
    path = A5 / "g0_compare.py"
    name = "_a5_g0_rmsnorm"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


G0 = _load_g0()
INDEX = G0.index()
RMSNORM_COMBOS = [combo for combo in G0.combos(INDEX) if combo["task"] == "rmsnorm"]
RMSNORM_IDS = [G0.combo_id(combo) for combo in RMSNORM_COMBOS]


@pytest.mark.parametrize("combo", RMSNORM_COMBOS, ids=RMSNORM_IDS)
def test_rmsnorm_matches_the_frozen_g0_corpus(combo):
    """@brief 每个 rmsnorm 组合的输入/reference/scale/tensor/容差/记账/baseline 与 G0 一致。"""
    result = G0.compare_combo(combo)
    assert result.records == len(combo["case_records"])
    assert result.exact_input_bytes >= result.records
    assert result.exact_reference_bytes == result.records
    assert result.exact_scale_bytes == result.records


def test_rmsnorm_is_a_v2_standard_entry_without_the_retired_assembly():
    """@brief 目录只有标准 entry；v1 registry 拒绝它，旧 task.py/track.json 不再存在。"""
    # 生产注册表（拒绝已迁题）必须绕过测试的 specimen 装配。
    from tests.support.v1_tasks import production_get_task as get_task
    from tcenv.contract_errors import TaskContractError
    from tcenv.taskcontract import load_task_contract

    directory = ROOT / "tasks" / "rmsnorm"
    for name in ("opspec.json", "reference.py", "reference.json", "baselines.py", "baselines.json", "display.json"):
        assert (directory / name).is_file(), name
    for retired in ("task.py", "track.json", "kernel_triton.py", "kernel_triton_msl.py", "naive.metal", "__init__.py"):
        assert not (directory / retired).exists(), retired
    contract = load_task_contract(directory, dtype="fp16", size="small")
    assert contract.instance.op_name == "rmsnorm"
    assert contract.reference.error_scale_kind == "none"
    assert [entry.name for entry in contract.baselines] == ["mlx", "torch_mps"]
    with pytest.raises(TaskContractError) as error:
        get_task("rmsnorm", "fp16", "small")
    assert "v2 standard entry" in str(error.value)
