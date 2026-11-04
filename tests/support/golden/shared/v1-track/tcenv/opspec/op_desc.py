"""Read-only op_desc interop summary (CAKE2-shaped, non-authoritative).

The summary is a projection of the instantiated v2 semantics for external op-description
consumers.  It is never read back by the loader, never part of ``opspec_digest`` and never a
source of tensors, attrs, tolerances, launch geometry or a verdict.  A2/A3 own any actual
interop adapter; A1 only freezes the deterministic renderer and its closed key set.
"""

from __future__ import annotations

from typing import Any

#: The upstream op_desc exemplar's closed top-level/nested keys (A1 comparison contract).
OP_DESC_TEMPLATE_KEYS = ("op_name", "category", "description", "shape_info", "attributes", "test_cases")
OP_DESC_SHAPE_KEYS = ("input_shapes", "output_shapes")
OP_DESC_TENSOR_KEYS = ("name", "shape", "dtype")
OP_DESC_CASE_KEYS = ("name", "input_shape", "dtype")


def render_op_desc(instance: Any) -> dict[str, Any]:
    """@brief 由已实例化 v2 语义渲染只读 op_desc 摘要。

    ``test_cases[].input_shape`` is the first declared input's shape, matching the upstream
    exemplar's single-input field.  Multi-input details remain in ``shape_info.input_shapes``;
    the summary is descriptive only and cannot override the typed OpSpec.

    @param instance v2 语义实例。
    @return 闭集摘要字典。
    """
    inputs = [binding for binding in instance.tensors if binding.role == "input"]
    outputs = [instance.tensor(name) for name in instance.reference_outputs]
    first_shape = [int(dim) for dim in inputs[0].shape] if inputs else []
    return {
        "op_name": instance.op_name,
        "category": instance.category,
        "description": instance.formula,
        "shape_info": {
            "input_shapes": [
                {"name": binding.name, "shape": [int(dim) for dim in binding.shape], "dtype": binding.dtype}
                for binding in inputs
            ],
            "output_shapes": [
                {"name": binding.name, "shape": [int(dim) for dim in binding.shape], "dtype": binding.dtype}
                for binding in outputs
            ],
        },
        "attributes": dict(instance.resolved_attrs),
        "test_cases": [
            {"name": case, "input_shape": list(first_shape), "dtype": instance.dtype}
            for case in instance.case_domain.cases
        ],
    }


__all__ = [
    "OP_DESC_CASE_KEYS",
    "OP_DESC_SHAPE_KEYS",
    "OP_DESC_TEMPLATE_KEYS",
    "OP_DESC_TENSOR_KEYS",
    "render_op_desc",
]
