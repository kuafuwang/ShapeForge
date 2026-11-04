"""搜索装题分派：v1 保留题与 v2 标准目录各自走自己的路，互不 fallback。"""

from __future__ import annotations

import pytest

from taic.task_binding import TaskBindingError, bind_task


def test_a_v2_standard_entry_binds_a_locator_and_contract_not_a_v1_task():
    binding = bind_task("gemm", "fp16", "small")
    assert binding.is_v2 and binding.mode == "v2"
    assert binding.task is None
    assert binding.locator is not None and binding.contract is not None
    assert (binding.contract.dtype, binding.contract.size) == ("fp16", "small")
    assert len(binding.task_pack_digest) == 64
    assert binding.kernel_name == ""
    assert binding.contract_fingerprint == binding.contract.fingerprint()


def test_a_non_default_instance_rebuilds_the_locator_for_that_instance():
    # worker 按 locator 重建任务：请求的 (dtype,size) 必须与合同一致，不能用默认实例冒充。
    binding = bind_task("gemm", "bf16", "paper")
    assert binding.contract.dtype == "bf16" and binding.contract.size == "paper"
    assert binding.locator.dtype == "bf16" and binding.locator.size == "paper"
    assert binding.contract.instance.symbols["M"] == 2048


def test_a_preserved_v1_task_still_binds_through_the_v1_registry():
    binding = bind_task("sem_attention_odd_scale", "fp16", "small")
    assert binding.mode == "v1" and not binding.is_v2
    assert binding.task is not None and binding.contract is None
    assert binding.kernel_name == binding.task.contract.kernel_name
    assert binding.task_pack_digest is None


def test_a_v2_task_refuses_a_launch_variant_instead_of_assembling_a_second_contract():
    with pytest.raises(TaskBindingError, match="no @variant"):
        bind_task("gemm@128x64x32t256", "fp16", "small")


def test_v2_tolerances_come_from_the_semantic_instance():
    binding = bind_task("gemm", "fp16", "small")
    tolerances = binding.tolerances()
    assert tolerances["rtol"] == pytest.approx(0.002)
    assert tolerances["atol_rel"] == pytest.approx(0.001)
    assert tolerances["exact"] is False


def test_the_v2_presentation_and_signature_follow_the_derived_abi():
    binding = bind_task("gemm", "fp16", "small")
    presentation = binding.presentation()
    assert presentation.statement == binding.contract.statement
    stub = binding.signature_stub()
    # ABI 声明顺序 A,B,C；没有 workspace 参数；核名只是建议（候选可改名，launch 必须指到它）。
    assert stub.index("A [[buffer(0)]]") < stub.index("B [[buffer(1)]]") < stub.index("C [[buffer(2)]]")
    assert "workspace" not in stub
    assert len(binding.presentation_sha256()) == 64
