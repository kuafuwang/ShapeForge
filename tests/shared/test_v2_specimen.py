"""The shared mechanism-test fixture is a real v2 contract, not a projected v1 Task."""

from __future__ import annotations

import pytest

from tcenv.taskcontract import TaskContract
from tests.support.v2_specimen import V2_TASK, bind_v2, v2_contract, v2_evaluation_fixture, v2_pack_digest


def test_default_fixture_uses_the_shipped_v2_contract_and_binding():
    contract = v2_contract(size="small")
    binding = bind_v2(size="small")

    assert isinstance(contract, TaskContract)
    assert contract.op_name == V2_TASK
    assert contract.instance.op_name == V2_TASK
    assert binding.mode == "v2"
    assert binding.contract is not None
    assert binding.contract.fingerprint() == contract.fingerprint()
    assert not hasattr(binding, "task")
    assert binding.locator is not None
    assert binding.locator.pack_digest == contract.pack_digest == v2_pack_digest(size="small")
    assert binding.task_pack_digest == binding.locator.deployment_pack_digest
    assert binding.kernel_name == ""


def test_evaluation_fixture_uses_candidate_geometry_and_the_current_v2_identity():
    fixture = v2_evaluation_fixture()
    task = fixture.task

    assert fixture.binding.is_v2
    assert task.name == V2_TASK
    assert task.contract.kernel_name == "flash_attention_kernel"
    assert task.specialization.opspec_digest == fixture.contract.spec.digest
    assert task.specialization.lowering_profile_digest == fixture.contract.fingerprint()
    assert task.evaluation_implementations["task_pack"]["digest"] == fixture.binding.locator.deployment_pack_digest
    assert task.triton_source == fixture.contract.statement


def test_subject_confirmation_fixture_passes_the_independent_verifier():
    from taic.portfolio_confirmation import verify_subject_document
    from tests.support.report_build import v2_subject_confirmation

    source = b"kernel void candidate() {}\n"
    document = v2_subject_confirmation(V2_TASK, "fp16", "small", source=source, speedup=1.25)
    verified = verify_subject_document(document)
    assert verified["status"] == "confirmed"
    assert verified["candidate_sha256"] == __import__("hashlib").sha256(source).hexdigest()
    assert verified["confirmation_seed"] != verified["search_seed"]
    assert verified["identity"]["kind"] == "match"


def test_integer_fixture_uses_the_dtype_actually_declared_by_v2():
    contract = v2_contract("sem_int_5", "i32", "small")
    assert contract.dtype == "i32"
    assert contract.instance.op_name == "sem_int_5"


def test_v1_dtype_aliases_are_not_silently_reintroduced():
    with pytest.raises(Exception):
        v2_contract("sem_int_5", "fp16", "small")
