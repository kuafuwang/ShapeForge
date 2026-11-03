"""The migrated ``gen_*`` family equals its hand-written twin on the v2 standard entry.

A5 moved ``gen_relu``/``gen_sigmoid``/``gen_silu``/``gen_swiglu`` from the
``tasks/generated/<op>/task.py`` renderer to ``tasks/gen_<op>/`` standard directories.  A6
deleted the old parser (``tasks/generated/elementwise.py``), its helper
(``tasks/_pointwise.py``) and the registry's ``GENERATED_NAMES`` list, so this file no longer
imports a retired module: the four tasks are enumerated from the frozen G0 migration boundary
(the same oracle the migration used), and only their v2 behaviour is compared.

What survives from the old family is the property that mattered -- the generated task IS the
hand-written task on tensors, cases, tolerances, work accounting and reference bytes -- plus
the retirement boundary: the v1 registry must refuse a ``gen_*`` name and the parser module must
be gone.  The retired expression grammar has no v2 equivalent to test: the v2 reference is
plain Python and ``opspec.json.formula`` is a summary only, so there is no second code generator
in the task tree.  No GPU.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from tasks import get_task
from tcenv.contract_errors import TaskContractError
from tcenv.opspec.model import OpSpecError, load_opspec
from tcenv.taskcontract import load_task_contract
from tcenv.track import REPO_ROOT

ROOT = Path(__file__).resolve().parents[3]
BOUNDARY = ROOT / "docs/架构/跨硬件算子生成分工-G0/migration-boundary.json"


def _generated_pairs() -> tuple[tuple[str, str], ...]:
    """@brief 从冻结的 G0 迁移边界取四道生成族题目（不是新的名字清单）。

    @return ``(gen_<op>, <op>)`` 元组。
    """
    document = json.loads(BOUNDARY.read_text(encoding="utf-8"))
    names = tuple(str(item["task"]) for item in document["migration_candidates"])
    generated = tuple(name for name in names if name.startswith("gen_"))
    assert len(generated) == 4, generated
    return tuple((name, name[len("gen_") :]) for name in generated)


PAIRS = _generated_pairs()


def _sha(t: torch.Tensor) -> str:
    """@brief 计算张量原始字节摘要（逐位比较）。

    @param t 待摘要的张量。
    @return 十六进制 SHA-256。
    """
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _tensor_shape(tensor) -> tuple:
    """@brief v2 张量声明的可比投影。

    @param tensor 张量声明。
    @return ``(name, shape, dtype, access)``。
    """
    return (tensor.name, tuple(int(v) for v in tensor.shape), tensor.dtype, tensor.access)


def _leaves(document) -> list:
    """@brief 递归取出文档里的全部标量/键，供“没有启动事实”的闭集检查。

    @param document JSON 文档。
    @return 叶子列表。
    """
    if isinstance(document, dict):
        out = list(document)
        for value in document.values():
            out.extend(_leaves(value))
        return out
    if isinstance(document, list):
        out = []
        for value in document:
            out.extend(_leaves(value))
        return out
    return [document]


@pytest.mark.parametrize("gen_name,hand_name", PAIRS)
def test_generated_task_matches_the_hand_written_twin(gen_name, hand_name):
    """@brief 生成任务与手写任务在张量、用例、容差、记账和 reference 字节上一致。"""
    hand = load_task_contract(REPO_ROOT / "tasks" / hand_name, dtype="fp16", size="small")
    gen = load_task_contract(REPO_ROOT / "tasks" / gen_name, dtype="fp16", size="small")
    assert [_tensor_shape(t) for t in hand.instance.tensors] == [_tensor_shape(t) for t in gen.instance.tensors]
    assert hand.instance.symbols == gen.instance.symbols
    assert hand.instance.case_domain.cases == gen.instance.case_domain.cases
    assert hand.instance.tolerances == gen.instance.tolerances
    assert dataclasses.asdict(hand.instance.work_accounting) == dataclasses.asdict(gen.instance.work_accounting)
    for case in ("normal", "wide_range", "outlier"):
        h = hand.reference.generate(hand.instance, case, 12345)
        g = gen.reference.generate(gen.instance, case, 12345)
        assert set(h) == set(g)
        for name in h:
            assert _sha(h[name]) == _sha(g[name]), (gen_name, case, name)
        hr = hand.reference.execute(hand.instance, h)
        gr = gen.reference.execute(gen.instance, g)
        assert set(hr) == set(gr)
        for name in hr:
            assert _sha(hr[name]) == _sha(gr[name]), (gen_name, case, name)


@pytest.mark.parametrize("gen_name,hand_name", PAIRS)
def test_every_generated_task_is_one_v2_standard_entry(gen_name, hand_name):
    """@brief 生成题目只有标准 entry；v1 registry 拒绝它，旧 task.py/track.json 不再存在。"""
    directory = REPO_ROOT / "tasks" / gen_name
    document = load_opspec(directory / "opspec.json").document
    assert document["spec_version"] == 2 and document["op_name"] == gen_name
    for name in ("opspec.json", "reference.py", "reference.json", "baselines.py", "baselines.json", "display.json"):
        assert (directory / name).is_file(), (gen_name, name)
    for retired in ("task.py", "track.json", "kernel_triton.py", "naive.metal", "__init__.py"):
        assert not (directory / retired).exists(), (gen_name, retired)
    with pytest.raises(TaskContractError) as error:
        get_task(gen_name, "fp16", "small")
    assert "v2 standard entry" in str(error.value)
    # The document owns no launch fact: grid/threads/tile are candidate-owned (A5 retired the
    # registered N/BLOCK_SIZE constants together with the v1 generator).
    keys = {str(key).lower() for key in _leaves(document)}
    assert not ({"grid", "threads", "num_warps", "tile", "block_size"} & keys), sorted(keys)
    # The v2 schema is closed: an extra unknown field is refused instead of becoming a second
    # place to configure the task.
    broken = copy.deepcopy(document)
    broken["tile"] = [64, 64]
    with pytest.raises(OpSpecError):
        from tcenv.opspec.model import validate_opspec

        validate_opspec(broken, source=gen_name)


def test_the_retired_generated_parser_and_its_name_list_are_gone():
    """@brief 旧解析器、旧 helper 与登记里的生成族名字条目都已删除（不是未使用）。"""
    import tasks as task_package

    assert not (REPO_ROOT / "tasks" / "generated").exists()
    assert not (REPO_ROOT / "tasks" / "_pointwise.py").exists()
    assert not (REPO_ROOT / "tasks" / "_matmul.py").exists()
    assert not hasattr(task_package, "GENERATED_NAMES")
    assert importlib.util.find_spec("tasks.generated") is None
    for gen_name, _hand in PAIRS:
        assert (REPO_ROOT / "tasks" / gen_name / "reference.py").is_file(), gen_name


def test_a_v2_contract_is_sealed_against_a_post_hoc_tolerance_change():
    """@brief 容差是 v2 语义实例的派生字段：合同对象不可事后改容差。"""
    gen = load_task_contract(REPO_ROOT / "tasks" / "gen_relu", dtype="fp16", size="small")
    with pytest.raises(dataclasses.FrozenInstanceError):
        gen.instance.tolerances.rtol = gen.instance.tolerances.rtol + 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        gen.instance = copy.deepcopy(gen.instance)
