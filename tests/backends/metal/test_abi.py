"""Metal shader-validation limit handling."""

from __future__ import annotations

from tcenv.backends.metal.abi import only_static_threadgroup_limit


def test_shader_validation_threadgroup_size_is_not_treated_as_the_device_limit():
    from tcenv.backends.metal.abi import only_static_threadgroup_limit

    inflated = "static threadgroup memory 40960 B exceeds the device limit of 32768 B"
    assert only_static_threadgroup_limit([inflated])
    assert not only_static_threadgroup_limit([])
    assert not only_static_threadgroup_limit([inflated, "missing kernel parameter `x`"])
    assert not only_static_threadgroup_limit(["launch infeasible: the contract launches 128 threads"])
