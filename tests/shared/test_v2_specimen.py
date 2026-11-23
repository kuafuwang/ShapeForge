"""The shared mechanism-test fixture is a real v2 contract, not a projected v1 Task."""

from __future__ import annotations

import pytest

from tcenv.taskcontract import TaskContract
from tests.support.v2_specimen import V2_TASK, bind_v2, v2_contract, v2_pack_digest


def test_default_fixture_uses_the_shipped_v2_contract_and_binding():
    contract = v2_contract(size="small")
    binding = bind_v2(size="small")

    assert isinstance(contract, TaskContract)
    assert contract.op_name == V2_TASK
    assert contract.instance.op_name == V2_TASK
    assert binding.mode == "v2"
    assert binding.contract is not None
    assert binding.contract.fingerprint() == contract.fingerprint()
    assert binding.task is None
    assert binding.locator is not None
    assert binding.locator.pack_digest == contract.pack_digest == v2_pack_digest(size="small")
    assert binding.task_pack_digest == binding.locator.deployment_pack_digest
    assert binding.kernel_name == ""


def test_integer_fixture_uses_the_dtype_actually_declared_by_v2():
    contract = v2_contract("sem_int_5", "i32", "small")
    assert contract.dtype == "i32"
    assert contract.instance.op_name == "sem_int_5"


def test_v1_dtype_aliases_are_not_silently_reintroduced():
    with pytest.raises(Exception):
        v2_contract("sem_int_5", "fp16", "small")
