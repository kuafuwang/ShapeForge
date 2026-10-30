"""G0 comparison: loading/assembling v2 leaves every frozen v1 task identity untouched.

A1 must not drift the v1 contract, the Contract serialisation projection, the case list,
tolerances, work accounting, prompt text or the registered implementation identities.  The
G0 corpus recorded all of them for the 82 in-scope ``(task, dtype, size, variant)`` combos;
this test recomputes the same projection from the current v1 path and compares it field by
field.  It is CPU-only and never runs a kernel.

The G0 corpus is a historical archive: its ``source_head`` is the old checkout, so this test
compares *content*, not the recorded HEAD (``gen_corpus.py --replay`` owns the HEAD check).
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from tasks import get_task

ROOT = Path(__file__).resolve().parents[2]
G0_INDEX = ROOT / "docs" / "架构" / "跨硬件算子生成分工-G0" / "corpus" / "index.json"

INDEX = json.loads(G0_INDEX.read_text(encoding="utf-8"))
COMBOS = INDEX["combos"]


def _json_value(value):
    """@brief 与 G0 corpus 相同的 JSON 投影（dataclass → dict，tuple → list）。

    @param value 任意可投影值。
    @return JSON 值。
    """
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"not JSON canonical: {type(value).__name__}")


def _combo_id(combo: dict) -> str:
    """@brief 稳定的 pytest 节点 id。

    @param combo G0 组合。
    @return id 文本。
    """
    variant = combo.get("variant") or "base"
    return f"{combo['task']}-{combo['dtype']}-{combo['size']}-{variant}"


def _projections(task) -> dict:
    """@brief 需要与 G0 对照的 v1 投影。

    @param task 当前 v1 任务。
    @return 投影字典。
    """
    implementations = _json_value(task.evaluation_implementations)
    implementations.pop("task_pack", None)
    return {
        "contract": _json_value(task.contract),
        "contract_text": {
            "describe": task.contract.describe(),
            "header": task.contract.render_header(),
            "abi": task.contract.abi_stub(),
        },
        "specialization": _json_value(task.specialization),
        "triton_source": task.triton_source,
        "naive_msl": task.naive_msl,
        "reference_sources": _json_value(task.reference_sources),
        "description": task.description,
        "cases": list(task.cases),
        "tolerances": {"rtol": task.rtol, "atol_rel": task.atol_rel, "exact": task.exact},
        "work": {"flops": float(task.flops), "min_bytes": float(task.min_bytes)},
        "meta": _json_value(task.meta),
        "implementation_identities": {
            key: implementations.get(key)
            for key in ("gen_inputs", "reference", "error_scale", "baselines")
        },
        "generated_blocks": {
            key: implementations.get(key) for key in ("spec_text", "known_good") if key in implementations
        },
    }


@pytest.mark.parametrize("combo", COMBOS, ids=[_combo_id(combo) for combo in COMBOS])
def test_the_g0_v1_combo_keeps_its_contract_identity_and_seed_contract(combo):
    """@brief 每个 G0 v1 组合的合同/题面/cases/容差/记账/实现身份与冻结语料完全一致。"""
    task = get_task(combo["name"], combo["dtype"], combo["size"])
    actual = _projections(task)
    expected = {
        "contract": combo["contract"],
        "contract_text": combo["contract_text"],
        "specialization": combo["specialization"],
        "triton_source": combo["triton_source"],
        "naive_msl": combo["naive_msl"],
        "reference_sources": combo["reference_sources"],
        "description": combo["description"],
        "cases": combo["cases"],
        "tolerances": combo["tolerances"],
        "work": combo["work"],
        "meta": combo["meta"],
        "implementation_identities": {
            key: combo["evaluation_implementations"].get(key)
            for key in ("gen_inputs", "reference", "error_scale", "baselines")
        },
        "generated_blocks": {
            key: combo["evaluation_implementations"].get(key)
            for key in ("spec_text", "known_good")
            if key in combo["evaluation_implementations"]
        },
    }
    for key, value in expected.items():
        assert actual[key] == value, f"v1 identity drifted for {_combo_id(combo)}: {key}"
    assert actual["specialization"]["opspec_digest"] == combo["specialization"]["opspec_digest"]
    assert actual["specialization"]["lowering_profile_digest"] == combo["specialization"]["lowering_profile_digest"]


def test_the_g0_combo_inventory_is_unchanged():
    """@brief 语料范围仍是 16 题 / 82 组合；A1 不得偷偷迁移或删除组合。"""
    assert len(COMBOS) == 82
    assert len({combo["task"] for combo in COMBOS}) == 16
    assert INDEX["schema_version"] == 1
    assert INDEX["seed_formula"] == "base_seed + 7919 * (case_index + 1)"
    assert INDEX["seeds"] == [0, 7, 1234]
