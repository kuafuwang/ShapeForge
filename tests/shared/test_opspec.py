"""OpSpec: structural schema, semantic rules, ``instantiate`` and its negatives (plan §4.1/§4.2).

CPU only, backend free: these tests load ``tasks/rmsnorm/opspec.json`` directly and call
``tcenv.opspec`` + ``tcenv.identity``; they never import a backend package.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tcenv import identity
from tcenv.opspec import model
from tcenv.opspec.model import OpSpecError, instantiate, load_opspec, validate_opspec

ROOT = Path(__file__).resolve().parents[2]
OPSPEC = ROOT / "tasks" / "rmsnorm" / "opspec.json"


def _document() -> dict:
    """@brief 读取真实的 RMSNorm OpSpec 作为可变异副本。

    @return 深拷贝的文档字典。
    """
    return json.loads(OPSPEC.read_text(encoding="utf-8"))


def _rejects(mutate) -> None:
    """@brief 断言一次变异被结构/语义校验拒绝。

    @param mutate 接收可变文档的变异函数。
    """
    document = _document()
    mutate(document)
    with pytest.raises(OpSpecError):
        validate_opspec(document, source="<test>")


def _digest(document: dict, monkeypatch=None) -> str:
    """@brief 用同一规则版本重算一份文档的 ``opspec_digest``。

    @param document OpSpec 文档。
    @return 摘要。
    """
    envelope = {
        "fingerprint_kind": "opspec",
        "fingerprint_version": 1,
        "semantic_rules_version": model.SEMANTIC_RULES_VERSION,
        "work_accounting_version": model.WORK_ACCOUNTING_VERSION,
        "document": document,
    }
    return identity.fingerprint(envelope)


def test_the_shipped_rmsnorm_opspec_is_valid_and_declares_the_frozen_domain():
    """@brief 真实 OpSpec 通过校验，且只声明 R0 已有的两个 size（F-5：不得偷偷加 tail_odd）。"""
    spec = load_opspec(OPSPEC)
    assert spec.op_name == "rmsnorm"
    assert spec.dtypes == ("fp16", "bf16")
    assert sorted(spec.sizes) == ["paper", "small"]
    assert spec.sizes["paper"]["M"] == 4096 and spec.sizes["paper"]["H"] == 4096
    assert spec.sizes["small"]["M"] == 64 and spec.sizes["small"]["H"] == 512
    assert spec.sizes["paper"]["usage"] == ["golden", "benchmark"]
    assert len(spec.digest) == 64


@pytest.mark.parametrize("field", ["spec_version", "op_name", "formula", "uses", "tensors", "work_accounting"])
def test_every_required_field_is_required(field):
    """@brief 缺任一必填字段都被拒绝。"""
    _rejects(lambda document: document.pop(field))


def test_unknown_fields_are_refused_at_every_level():
    """@brief 顶层、张量、numerics、attrs、sizes 里的未知字段都 fail closed。"""
    _rejects(lambda document: document.__setitem__("grid", [1, 1, 1]))
    _rejects(lambda document: document["tensors"]["x"].__setitem__("threads", 256))
    _rejects(lambda document: document["numerics"].__setitem__("tolerance_hint", "loose"))
    _rejects(lambda document: document["attrs"]["EPS"].__setitem__("value", 1e-6))
    _rejects(lambda document: document["sizes"]["paper"].__setitem__("tiles", 8))
    _rejects(lambda document: document["work_accounting"]["flops"].__setitem__("expr", "4*M*H"))


def test_a_size_outside_the_declared_symbol_domain_is_refused():
    """@brief H 的 min 是 1：声明 H=0 的档位必须被拒绝（越域 shape）。"""
    _rejects(lambda document: document["sizes"]["small"].__setitem__("H", 0))
    _rejects(lambda document: document["sizes"]["paper"].pop("H"))
    _rejects(lambda document: document["sizes"]["paper"].__setitem__("usage", ["golden", "portfolio"]))


def test_an_attr_used_by_the_formula_needs_a_value_and_a_default_is_the_only_source():
    """@brief F-4：``uses`` 里有 EPS 而属性既无实例值也无显式 default 时拒绝，绝不补 0。"""
    _rejects(lambda document: document["attrs"]["EPS"].pop("default"))
    _rejects(lambda document: document["uses"].append("TEMPERATURE"))
    _rejects(lambda document: document["attrs"].__setitem__("TEMPERATURE", {"dtype": "float", "default": 1.0}))


def test_an_attr_outside_its_registered_domain_is_refused():
    """@brief ``{"gt": 0}`` 的 EPS 声明为 0 时拒绝。"""
    _rejects(lambda document: document["attrs"]["EPS"].__setitem__("default", 0.0))
    _rejects(lambda document: document["attrs"]["EPS"].__setitem__("domain", {"near": 1.0}))


def test_an_undeclared_dtype_or_size_is_refused_at_instantiation():
    """@brief 请求 fp32 或 plan 档位都不在声明域内。"""
    spec = load_opspec(OPSPEC)
    with pytest.raises(OpSpecError):
        instantiate(spec, "fp32", "paper")
    with pytest.raises(OpSpecError):
        instantiate(spec, "fp16", "plan")
    with pytest.raises(OpSpecError):
        instantiate(spec, "i32", "paper")


def test_duplicate_json_keys_are_refused_by_the_loader(tmp_path):
    """@brief 重复键由严格 loader 拒绝，不靠“最后一个值生效”决定身份。"""
    text = OPSPEC.read_text(encoding="utf-8").replace('"op_name": "rmsnorm"', '"op_name": "rmsnorm", "op_name": "relu"')
    path = tmp_path / "opspec.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(OpSpecError):
        load_opspec(path)


def test_changing_only_the_shape_domain_changes_the_digest_even_when_the_instance_is_identical():
    """@brief F-1：两个 OpSpec 在 paper 上实例相同，但域不同 → ``opspec_digest`` 不同。"""
    base = _document()
    widened = copy.deepcopy(base)
    widened["symbols"]["H"] = {"min": 2, "note": "same declarable sizes, tighter domain"}
    spec_base = load_opspec(OPSPEC)
    proposal = _explicit_opspec(widened)
    left = instantiate(spec_base, "fp16", "paper").fingerprint_document()
    right = instantiate(proposal, "fp16", "paper").fingerprint_document()
    assert {k: v for k, v in left.items() if k != "opspec_digest"} == {
        k: v for k, v in right.items() if k != "opspec_digest"
    }
    assert left["opspec_digest"] != right["opspec_digest"]


def test_changing_only_a_rule_version_changes_the_digest(monkeypatch):
    """@brief F-24：只改 ``semantic_rules_version``/``work_accounting_version`` 也改变 ``opspec_digest``。"""
    document = _document()
    baseline = _digest(document)
    assert baseline == load_opspec(OPSPEC).digest
    monkeypatch.setattr(model, "SEMANTIC_RULES_VERSION", "opspec-semantics-2")
    monkeypatch.setattr(model, "WORK_ACCOUNTING_VERSION", "opspec-work-accounting-expr-2")
    assert _digest(document) != baseline
    assert load_opspec(OPSPEC).digest != baseline


def test_work_accounting_is_one_source_and_one_json_type():
    """@brief F-19：``flops``/``min_bytes`` 只来自语义记账表达式，且物化为 float。"""
    spec = load_opspec(OPSPEC)
    for dtype in ("fp16", "bf16"):
        for size, (m, h) in (("paper", (4096, 4096)), ("small", (64, 512))):
            instance = instantiate(spec, dtype, size)
            assert instance.work_accounting.flops == pytest.approx(4.0 * m * h)
            assert instance.work_accounting.min_bytes == pytest.approx(float((2 * m * h + h) * 2))
            assert type(instance.work_accounting.flops) is float
            assert type(instance.work_accounting.min_bytes) is float


def test_a_tail_shape_instantiates_without_widening_the_shipped_domain():
    """@brief F-6：``H=3`` 的 tail 实例能正常实例化；F-5：它不能进入发布的 OpSpec 域。"""
    document = _document()
    document["sizes"]["tail_odd"] = {"M": 1, "H": 3, "usage": ["golden"]}
    spec = _explicit_opspec(document)
    instance = instantiate(spec, "fp16", "tail_odd")
    assert instance.tensor("x").shape == (1, 3)
    assert instance.tensor("w").shape == (3,)
    assert instance.size_usage == ("golden",)
    assert instance.symbols == {"M": 1, "H": 3}
    assert sorted(load_opspec(OPSPEC).sizes) == ["paper", "small"]


def test_the_semantic_instance_projection_is_a_closed_set():
    """@brief §3.1 的 SemanticInstance 投影字段闭合，buffer 不含 value，无声明 case 不写空对象。"""
    document = instantiate(load_opspec(OPSPEC), "bf16", "small").fingerprint_document()
    assert set(document) == {
        "opspec_digest",
        "dtype",
        "size",
        "shapes",
        "resolved_attrs",
        "spec_params",
        "tolerances",
        "case_domain",
        "work_accounting",
    }
    assert set(document["shapes"]) == {"symbols", "tensors"}
    assert document["shapes"]["tensors"] == {"x": [64, 512], "w": [512], "y": [64, 512]}
    assert set(document["tolerances"]) == {"rtol", "atol_rel", "exact"}
    assert set(document["case_domain"]) == {"cases", "inputs"}
    assert document["case_domain"]["inputs"] == {}
    assert set(document["work_accounting"]) == {"flops", "min_bytes"}
    assert document["spec_params"] == {"M": "64", "H": "512"}
    assert document["resolved_attrs"] == {"EPS": 1e-6}


def test_cases_keep_the_declared_order_and_dtype_tolerances_are_materialised():
    """@brief cases 顺序来自 OpSpec；容差按 dtype 物化为数值。"""
    spec = load_opspec(OPSPEC)
    fp16 = instantiate(spec, "fp16", "paper")
    bf16 = instantiate(spec, "bf16", "paper")
    assert fp16.case_domain.cases == ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
    assert (fp16.tolerances.rtol, fp16.tolerances.atol_rel, fp16.tolerances.exact) == (0.002, 0.001, False)
    assert (bf16.tolerances.rtol, bf16.tolerances.atol_rel, bf16.tolerances.exact) == (0.016, 0.008, False)


def test_instantiation_does_not_mutate_the_loaded_document():
    """@brief ``instantiate`` 只读文档：两次实例化的输入文档字节不变。"""
    spec = load_opspec(OPSPEC)
    before = copy.deepcopy(spec.document)
    instantiate(spec, "fp16", "paper")
    instantiate(spec, "bf16", "small")
    assert spec.document == before
    assert _digest(spec.document) == spec.digest


def _explicit_opspec(document: dict) -> model.OpSpec:
    """@brief 由一份已校验文档构造显式 OpSpec（测试内需要两份不同域的文档）。

    @param document 已通过校验的 OpSpec 文档。
    @return :class:`OpSpec`。
    """
    validate_opspec(document, source="<test>")
    return model.OpSpec(document=document, source="<test>", digest=_digest(document))
