"""varlen 评测报告里的实际输入身份（CPU；只跑 reference/投影，不编译、不启动 kernel）。

L1 的 correctness 必须能回答“这次到底跑了哪组分段”。`prepare_v2_case` 是 Metal v2 runtime 里
把 reference 输入变成设备初值和比较参考的那一步，所以它的 `input_digest` 必须在每道 varlen 题、
每个 case 上都等于 `portfolio.domain.evaluation_input_digest`（case + seed + 实际 cu 字节 + 输入
张量身份），并最终进入 ``CaseResult.input_sha256``。固定 shape 实例保持空串：它们的输入身份已经
由实例指纹和既有证据字段表达，不新增键。
"""

from __future__ import annotations

import pytest

from tcenv.backends.metal.v2_runtime import prepare_v2_case
from tcenv.opspec.v2 import VARLEN_CASE_DOMAIN
from tcenv.portfolio.domain import evaluation_input_digest
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.locator import REPO_ROOT

#: ``op_name -> (task directory, declared sizes)``；只列 varlen 题。
TASKS = {
    "varlen_kda": (REPO_ROOT / "tasks" / "varlen_kda", ("fixed", "mixed", "uniform")),
    "varlen_scale": (REPO_ROOT / "tasks" / "varlen_scale", ("mixed", "uniform", "single")),
}
COMBOS = tuple((op, size) for op, (_, sizes) in TASKS.items() for size in sizes)
SEED = 7


@pytest.mark.parametrize("op,size", COMBOS, ids=lambda value: str(value))
@pytest.mark.parametrize("case", VARLEN_CASE_DOMAIN)
def test_every_case_reports_the_digest_of_the_inputs_it_actually_prepared(op, size, case):
    """@brief ``prepare_v2_case`` 记录的输入身份与实际生成的 cu 字节一致（失败也能重放）。"""
    directory, _ = TASKS[op]
    contract = load_task_contract(directory, dtype="fp16", size=size)
    data = prepare_v2_case(contract, case, SEED)

    expected = evaluation_input_digest(contract.instance, case, SEED, data.rounded)
    assert data.input_digest == expected
    assert len(data.input_digest) == 64
    # 同一 (instance, case, seed) 重放得到同一身份；换 case 得到不同身份。
    assert prepare_v2_case(contract, case, SEED).input_digest == data.input_digest


def test_a_fixed_shape_instance_keeps_the_pre_varlen_report_shape():
    """@brief 固定 shape 实例不新增输入身份键（空串），varlen 之外证据形状不变。"""
    contract = load_task_contract(REPO_ROOT / "tasks" / "lion", dtype="fp16", size="small")
    assert contract.instance.cu_seqlens is None
    data = prepare_v2_case(contract, "tiny", SEED)
    assert data.input_digest == ""
