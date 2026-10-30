"""A1: the op_desc summary is deterministic, closed and non-authoritative."""

from __future__ import annotations

from tcenv.opspec import (
    OP_DESC_TEMPLATE_KEYS,
    instantiate,
    opspec_v2_from_document,
    render_op_desc,
)

from ..support.v2_tasks import rmsnorm_opspec, rmsnorm_read_write_opspec, sem_opspec

TEMPLATE = {
    "op_name": "rmsnorm",
    "category": "normalization",
    "description": "y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2, axis=H) + EPS) * w",
    "shape_info": {
        "input_shapes": [
            {"name": "x", "shape": [64, 512], "dtype": "fp16"},
            {"name": "w", "shape": [512], "dtype": "fp16"},
        ],
        "output_shapes": [{"name": "y", "shape": [64, 512], "dtype": "fp16"}],
    },
    "attributes": {"EPS": 1e-06},
    "test_cases": [
        {"name": "normal", "input_shape": [64, 512], "dtype": "fp16"},
        {"name": "wide_range", "input_shape": [64, 512], "dtype": "fp16"},
        {"name": "large", "input_shape": [64, 512], "dtype": "fp16"},
        {"name": "tiny", "input_shape": [64, 512], "dtype": "fp16"},
        {"name": "outlier", "input_shape": [64, 512], "dtype": "fp16"},
        {"name": "ieee", "input_shape": [64, 512], "dtype": "fp16"},
    ],
}


def _instance(document: dict, dtype: str = "fp16", size: str = "small"):
    return instantiate(opspec_v2_from_document(document, source="<op-desc-test>"), dtype, size)


def test_op_desc_matches_the_upstream_exemplar_shape_and_the_mapped_rmsnorm():
    """@brief 摘要的闭集键/值与 CAKE2 模板形状及映射后的 RMSNorm 语义一致。"""
    instance = _instance(rmsnorm_opspec())
    rendered = render_op_desc(instance)
    assert tuple(rendered) == OP_DESC_TEMPLATE_KEYS
    assert rendered == TEMPLATE
    assert tuple(rendered["shape_info"]) == ("input_shapes", "output_shapes")
    for entry in rendered["shape_info"]["input_shapes"] + rendered["shape_info"]["output_shapes"]:
        assert tuple(entry) == ("name", "shape", "dtype")
    for case in rendered["test_cases"]:
        assert tuple(case) == ("name", "input_shape", "dtype")
    assert render_op_desc(instance) == rendered


def test_op_desc_lists_read_write_final_values_as_outputs():
    """@brief read_write 终值属于结果集合，因此也进入 output_shapes。"""
    instance = _instance(rmsnorm_read_write_opspec())
    rendered = render_op_desc(instance)
    assert [entry["name"] for entry in rendered["shape_info"]["input_shapes"]] == ["x", "w", "acc"]
    assert [entry["name"] for entry in rendered["shape_info"]["output_shapes"]] == ["y", "acc"]
    assert tuple(rendered) == OP_DESC_TEMPLATE_KEYS


def test_op_desc_is_a_task_agnostic_projection_and_never_enters_the_digest():
    """@brief 摘要是纯投影：另一题只是另一份字段，且 digest 文档没有摘要字段。"""
    instance = _instance(sem_opspec("sem_gemm_neg"))
    rendered = render_op_desc(instance)
    assert rendered["op_name"] == "sem_gemm_neg"
    assert rendered["description"] == "y = -x"
    assert rendered["attributes"] == {}
    assert rendered["shape_info"]["input_shapes"] == [{"name": "x", "shape": [8], "dtype": "fp16"}]
    assert "op_desc" not in instance.fingerprint_document()
    assert render_op_desc(instance) == rendered
