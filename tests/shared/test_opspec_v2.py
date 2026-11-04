"""OpSpec v2: closed schema, explicit dispatch, instantiation and its negatives (A1).

CPU only, backend free.  The v2 document is exercised through the public loader
(``load_opspec``) so the tests also prove that ``spec_version`` dispatch is real and that v1
remains untouched.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tcenv.opspec import OpSpecError, SemanticInstance, SemanticInstanceV2, instantiate, load_opspec
from tcenv.identity import fingerprint
from tcenv.opspec.v2 import (
    OPSPEC_V2_SCHEMA_VERSION,
    SEMANTIC_RULES_VERSION_V2,
    opspec_v2_from_document,
    validate_opspec_v2,
)

from ..support.v2_tasks import rmsnorm_opspec, rmsnorm_read_write_opspec

ROOT = Path(__file__).resolve().parents[2]
# B4 之后仓库里没有 v1 文档了：v1 的 ``sem_attention_odd_scale`` opspec 归档在
# ``tests/support/golden/shared/v1-track``，本文件用它验证 spec_version 分派。
V1_OPSPEC = (
    Path(__file__).resolve().parents[1] / "support" / "golden" / "shared" / "v1-track"
    / "tasks" / "sem_attention_odd_scale" / "opspec.json"
)


def _write(tmp_path: Path, document: dict, name: str = "rmsnorm") -> Path:
    """@brief 把一份 v2 文档写成标准目录里的 opspec.json。

    @param tmp_path 临时目录。
    @param document v2 文档。
    @param name 目录名（必须等于 op_name）。
    @return opspec.json 路径。
    """
    task = tmp_path / name
    task.mkdir(parents=True, exist_ok=True)
    path = task / "opspec.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _rejects(document: dict) -> None:
    """@brief 断言一份 v2 文档被结构/语义校验拒绝。

    @param document 文档。
    """
    with pytest.raises(OpSpecError):
        validate_opspec_v2(document, source="<test>")


def test_the_v2_fixture_passes_and_instantiates_the_declared_domain(tmp_path):
    """@brief v2 fixture 通过 loader，实例带确定 dtype/shape/attrs/容差/记账。"""
    document = rmsnorm_opspec()
    spec = load_opspec(_write(tmp_path, document))
    assert spec.spec_version == OPSPEC_V2_SCHEMA_VERSION == 2
    assert spec.op_name == "rmsnorm"
    assert spec.dtypes == ("fp16", "bf16")
    assert sorted(spec.sizes) == ["paper", "small"]
    assert len(spec.digest) == 64

    instance = instantiate(spec, "fp16", "small")
    assert isinstance(instance, SemanticInstanceV2)
    assert (instance.dtype, instance.size, instance.symbols) == ("fp16", "small", {"M": 64, "H": 512})
    assert instance.tensor("x").shape == (64, 512)
    assert instance.tensor("w").strides == (1,)
    assert instance.resolved_attrs == {"EPS": 1e-06}
    assert instance.reference_outputs == ("y",)
    assert (instance.tolerances.rtol, instance.tolerances.atol_rel, instance.tolerances.exact) == (0.002, 0.001, False)
    assert instance.work_accounting.flops == pytest.approx(4.0 * 64 * 512)
    assert instance.work_accounting.min_bytes == pytest.approx(float((2 * 64 * 512 + 512) * 2))
    assert instance.case_domain.cases == ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
    assert instance.discovery["default_instance"] == {"dtype": "fp16", "size": "small"}


def test_v1_and_v2_are_dispatched_by_spec_version_without_a_shared_loader(tmp_path):
    """@brief v1 仍走 v1（返回 v1 SemanticInstance），v2 走 v2；未知版本拒绝。"""
    v1 = load_opspec(V1_OPSPEC)
    assert v1.spec_version == 1
    assert isinstance(instantiate(v1, "fp16", "small"), SemanticInstance)
    assert not isinstance(instantiate(v1, "fp16", "small"), SemanticInstanceV2)

    v2 = load_opspec(_write(tmp_path, rmsnorm_opspec()))
    assert v2.spec_version == 2
    assert isinstance(instantiate(v2, "fp16", "small"), SemanticInstanceV2)

    document = rmsnorm_opspec()
    document["spec_version"] = 3
    with pytest.raises(OpSpecError):
        load_opspec(_write(tmp_path, document, name="rmsnorm"))

    # A boolean is not an integer version: Python's ``True == 1`` must not select v1.
    document = rmsnorm_opspec()
    document["spec_version"] = True
    with pytest.raises(OpSpecError):
        load_opspec(_write(tmp_path, document, name="rmsnorm"))


def test_v2_has_no_uses_list_and_materialises_every_declared_attr(tmp_path):
    """@brief v2 删除 uses：所有声明 attrs 都物化；无 default 的属性仍拒绝。"""
    document = rmsnorm_opspec()
    document["attrs"]["TEMPERATURE"] = {"dtype": "float", "default": 0.5}
    spec = opspec_v2_from_document(document, source="<test>")
    instance = instantiate(spec, "fp16", "small")
    assert instance.resolved_attrs == {"EPS": 1e-06, "TEMPERATURE": 0.5}

    with_uses = rmsnorm_opspec()
    with_uses["uses"] = ["EPS"]
    _rejects(with_uses)

    without_default = rmsnorm_opspec()
    without_default["attrs"]["EPS"].pop("default")
    _rejects(without_default)


def test_attrs_may_not_smuggle_launch_or_tiling_parameters():
    """@brief attrs 是纯语义参数；几何名的大小写/连接变体一律拒绝。"""
    for name in (
        "BLOCK_M",
        "NUM_WARPS",
        "THREADS",
        "GRID",
        "TILE_K",
        "VECTOR_WIDTH",
        "BLOCKM",
        "NUMWARPS",
        "BLOCK_SIZE",
        "WARP_COUNT",
        "THREADGROUP_X",
    ):
        document = rmsnorm_opspec()
        document["attrs"][name] = {"dtype": "int", "default": 4}
        with pytest.raises(OpSpecError, match="launch/optimisation geometry"):
            validate_opspec_v2(document, source="<test>")

    # Attr identifiers are frozen uppercase semantic identifiers, not arbitrary strings or
    # camelCase/compound launch spellings.
    for name in ("Block_M", "NumWarps", "EPS-value", "EPS_", "A__B", "_A", "block"):
        document = rmsnorm_opspec()
        document["attrs"][name] = {"dtype": "int", "default": 4}
        with pytest.raises(OpSpecError, match="must match"):
            validate_opspec_v2(document, source="<test>")

    # Existing semantic names remain legal.
    for name in ("EPS", "SM_SCALE", "GELU_C", "A1_B2"):
        document = rmsnorm_opspec()
        document["attrs"][name] = {"dtype": "int", "default": 4}
        validate_opspec_v2(document, source="<test>")


def test_the_tensor_contract_keeps_role_access_layout_and_stride():
    """@brief role/access/stride 闭合；非法读写组合与未声明符号被拒绝。"""
    document = rmsnorm_opspec()
    document["tensors"]["x"]["access"] = "write"
    _rejects(document)

    document = rmsnorm_opspec()
    document["tensors"]["y"]["access"] = "read"
    _rejects(document)

    document = rmsnorm_opspec()
    document["tensors"]["x"]["strides"] = ["H"]
    _rejects(document)

    document = rmsnorm_opspec()
    document["tensors"]["x"]["dims"] = ["M", "Z"]
    _rejects(document)

    document = rmsnorm_opspec()
    document["tensors"]["x"]["strides"] = ["M", "1"]
    spec = opspec_v2_from_document(document, source="<test>")
    with pytest.raises(OpSpecError, match="declares strides"):
        instantiate(spec, "fp16", "small")


def test_read_write_final_values_are_the_reference_diff_contract():
    """@brief ``read_write`` 输入/输出的终值属于结果集合；投影绑定 access。"""
    spec = opspec_v2_from_document(rmsnorm_read_write_opspec(), source="<rw>")
    instance = instantiate(spec, "fp16", "small")
    assert instance.reference_outputs == ("y", "acc")
    assert instance.write_outputs == ()
    assert instance.read_write_tensors == ("y", "acc")
    projection = instance.fingerprint_document()
    tensors = {entry["name"]: entry for entry in projection["tensors"]}
    assert tensors["y"]["access"] == "read_write"
    assert tensors["acc"]["role"] == "input" and tensors["acc"]["access"] == "read_write"
    assert projection["spec_version"] == 2


def test_work_accounting_is_still_one_expression_source():
    """@brief 记账仍只由 work_accounting 求值（float），read_write 张量可显式计入字节。"""
    spec = opspec_v2_from_document(rmsnorm_read_write_opspec(), source="<rw>")
    instance = instantiate(spec, "bf16", "paper")
    assert instance.work_accounting.flops == pytest.approx(4.0 * 4096 * 4096)
    assert instance.work_accounting.min_bytes == pytest.approx(float((3 * 4096 * 4096 + 4096) * 2))
    assert type(instance.work_accounting.flops) is float


def test_discovery_metadata_is_validated_but_not_typed_semantics():
    """@brief discovery 默认实例必须引用声明域；variant_acceptance 可被结构校验但装配期拒绝。"""
    document = rmsnorm_opspec()
    document["discovery"]["default_instance"] = {"dtype": "fp32", "size": "small"}
    _rejects(document)

    document = rmsnorm_opspec()
    document["discovery"]["default_instance"] = {"dtype": "fp16", "size": "plan"}
    _rejects(document)

    document = rmsnorm_opspec()
    document["discovery"]["tags"] = ["normalization", "normalization"]
    _rejects(document)

    document = rmsnorm_opspec()
    document["discovery"]["variant_acceptance"] = {"factory_id": "factory.x", "implementation_version": 1}
    spec = opspec_v2_from_document(document, source="<test>")
    assert instantiate(spec, "fp16", "small").discovery["variant_acceptance"]["factory_id"] == "factory.x"
    assert SEMANTIC_RULES_VERSION_V2.startswith("opspec-v2-")


def test_no_global_ir_pattern_resolver_or_stage_partitioner_is_part_of_v2():
    """@brief v2 合同没有 IR/pattern/partition/lowering 规则字段：未知字段一律拒绝。"""
    for field in ("ir", "patterns", "stage_partition", "lowering_rules", "grid", "threads", "num_warps", "track"):
        document = rmsnorm_opspec()
        document[field] = {}
        _rejects(document)


def test_an_undeclared_dtype_or_size_is_refused_at_instantiation(tmp_path):
    """@brief 请求未声明 dtype/size 仍拒绝（v2 不扩大域）。"""
    spec = load_opspec(_write(tmp_path, rmsnorm_opspec()))
    with pytest.raises(OpSpecError):
        instantiate(spec, "fp32", "small")
    with pytest.raises(OpSpecError):
        instantiate(spec, "fp16", "plan")


def test_digest_document_recomputes_the_loaded_digest_for_v1_and_v2(tmp_path):
    """@brief 公共 digest_document() 必须按 spec_version 返回能复算出同一 digest 的 envelope。"""
    v1 = load_opspec(V1_OPSPEC)
    assert v1.digest_document()["fingerprint_version"] == 1
    assert fingerprint(v1.digest_document()) == v1.digest

    v2 = load_opspec(_write(tmp_path, rmsnorm_opspec()))
    assert v2.spec_version == 2
    assert v2.digest_document()["fingerprint_version"] == 2
    assert v2.digest_document()["semantic_rules_version"] == SEMANTIC_RULES_VERSION_V2
    assert fingerprint(v2.digest_document()) == v2.digest


def test_the_v2_document_is_not_mutated_by_instantiation(tmp_path):
    """@brief instantiate 只读；重复实例化的 digest 与文档都不变。"""
    path = _write(tmp_path, rmsnorm_opspec())
    spec = load_opspec(path)
    before = copy.deepcopy(spec.document)
    instantiate(spec, "fp16", "small")
    instantiate(spec, "bf16", "paper")
    assert spec.document == before
    assert load_opspec(path).digest == spec.digest


def test_unknown_v2_fields_are_refused_at_every_level():
    """@brief 顶层/张量/numerics/attrs/sizes/discovery 的未知字段都 fail closed。"""
    document = rmsnorm_opspec()
    document["tensors"]["x"]["tile"] = 64
    _rejects(document)

    document = rmsnorm_opspec()
    document["numerics"]["ulp_tolerance"] = 2
    _rejects(document)

    document = rmsnorm_opspec()
    document["attrs"]["EPS"]["unit"] = "1"
    _rejects(document)

    document = rmsnorm_opspec()
    document["sizes"]["small"]["tiles"] = 8
    _rejects(document)

    document = rmsnorm_opspec()
    document["discovery"]["shape_family"] = "x"
    _rejects(document)
