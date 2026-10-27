"""P0 NPU cleanup proof, quarantine and Host/Device timing separation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from tcenv.remote import contracts
from tcenv.remote.drivers.base import CleanupProof
from tcenv.remote.job_types import JobError
from tests.support.job_engine import (
    NPU_TARGET,
    NPU_TOKEN,
    npu_engine,
    npu_get,
    npu_request,
    npu_retry,
    npu_submit,
)
from tests.support.vendor_sim.ledger import VendorLedger
from tests.support.vendor_sim.vendor_a import (
    FakeVendorAController,
    build_execution,
)

SOURCE = b"npu-kernel relu_kernel\n"


def _wait_engine(engine, request, states: set[str], timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    snapshot = npu_get(engine, request)
    while snapshot.state not in states and time.monotonic() < deadline:
        time.sleep(0.02)
        snapshot = npu_get(engine, request)
    assert snapshot.state in states, f"timed out waiting for {states}: {snapshot.state}"
    return snapshot


def test_controller_probe_capabilities_health_cancel_and_reconcile(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    profile = build_execution().device_group.to_dict()
    facts = controller.probe({"device_group_id": profile["device_group_id"]})
    assert facts.vendor_id == "vendor-a"
    assert facts.logical_devices == 1
    capabilities = controller.capabilities(facts)
    assert capabilities["artifact_kinds"] == ["source"]
    assert capabilities["launch_kinds"] == ["KERNEL"]
    assert capabilities["timing_sources"] == ["DEVICE_EVENT"]
    assert controller.health_check({"device_group_id": profile["device_group_id"]}).ok

    ledger = VendorLedger(ledger_path)
    ledger.record_resource(
        attempt_id="attempt-probe", lease_fence=7,
        device_group_id=profile["device_group_id"],
        resource_generation=profile["resource_generation"],
        kind="module", detail={"format": "ELF/SIMT"},
    )
    controller.cancel("attempt-probe", 7)
    assert ledger.is_cancelled("attempt-probe", 7)
    proof = controller.reconcile("attempt-probe", 7)
    assert isinstance(proof, CleanupProof)
    assert proof.complete is True
    assert proof.attempt_id == "attempt-probe"
    assert proof.lease_fence == 7
    assert proof.released_resources


def test_reconcile_refuses_resources_bound_to_another_fence(tmp_path):
    ledger_path = tmp_path / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    ledger = VendorLedger(ledger_path)
    group = build_execution().device_group
    ledger.record_resource(
        attempt_id="attempt-stale", lease_fence=1,
        device_group_id=group.device_group_id,
        resource_generation=group.resource_generation,
        kind="stream", detail={},
    )
    proof = controller.reconcile("attempt-stale", 2)
    assert proof.complete is False
    assert "stale fence" in proof.detail
    assert ledger.live_resources()


def test_unproven_device_cleanup_quarantines_and_refuses_retry(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path, release_supported=False)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-quarantine-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert "device cleanup was not proven" in snapshot.recovery_error
        assert snapshot.result is None
        assert engine.close_error is not None
        live = VendorLedger(ledger_path).live_resources()
        assert live, "failed cleanup must retain observable live resources"
        with pytest.raises(JobError) as excinfo:
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
        assert excinfo.value.code == "failed_precondition"
    finally:
        engine.close(wait=5)


def test_a_missing_device_controller_cannot_prove_cleanup(tmp_path):
    root = tmp_path / "run"
    engine = npu_engine(
        root / "state", ledger_path=root / "ledger.json", controller=None,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-no-controller-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        # A cooperative worker-local close is not proof of device release.
        assert snapshot.quarantined is True
        assert "no device controller" in snapshot.recovery_error
        assert snapshot.result is None
        assert engine.close_error is not None
    finally:
        engine.close(wait=5)


def test_worker_killed_after_launch_reconciles_cleanly(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-killed-1", source=SOURCE)
        npu_submit(engine, request)
        snapshot = _wait_engine(engine, request, {"interrupted"})
        assert snapshot.quarantined is False
        assert snapshot.failure is not None and snapshot.failure.retryable is True
        assert VendorLedger(ledger_path).live_resources() == []
        proof = controller.reconcile(snapshot.current_attempt.attempt_id, snapshot.current_attempt.lease_fence)
        assert proof.complete is True
    finally:
        engine.close(wait=5)


def test_host_api_elapsed_is_not_written_as_device_elapsed(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_HOST_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-timing-1", source=SOURCE)
        npu_submit(engine, request)
        snapshot = _wait_engine(engine, request, {"completed"})
        evidence = snapshot.result.evidence_json()
        assert evidence is not None
        timing = evidence["timing"]
        assert timing["source"] == "DEVICE_EVENT"
        assert timing["clock_domain"] == "vendor-a-device-event-clock-v1"
        assert timing["synchronization_completed"] is True
        assert timing["host_api_elapsed_ns"] >= 50_000_000
        assert timing["device_elapsed_ns"] != timing["host_api_elapsed_ns"]
        # The report's candidate time comes from the device event, not the host
        # API span; a mutation that swaps them must fail this assertion.
        assert snapshot.result.report_json()["candidate_us"] == timing["device_elapsed_ns"] / 1000.0
        assert snapshot.result.provenance["measurement_kind"] == "simulated"
        assert snapshot.result.provenance["publication_allowed"] is False
        assert snapshot.result.receipt["candidate_digest"] == request.candidate_digest
        assert "candidate_sha256" not in snapshot.result.receipt
    finally:
        engine.close(wait=5)


def test_fake_vendor_is_not_imported_by_production_entry_points():
    root = Path(__file__).resolve().parents[2]
    for name in (
        "tcenv/remote/production.py",
        "tcenv/remote/grpc_service.py",
        "tcenv/remote/job_engine.py",
        "tcenv/remote/job_store.py",
        "tcenv/remote/grpc_server.py",
        "tcenv/remote/grpc_client.py",
        "tcenv/remote/drivers/base.py",
    ):
        text = (root / name).read_text(encoding="utf-8")
        assert "tests.support.vendor_sim" not in text, name
        assert "FakeVendorA" not in text, name
        assert "TCENV_VENDOR_LEDGER" not in text, name
    assert contracts.NPU_SOURCE_API_CONTRACT_ID not in (
        root / "tcenv/remote/production.py"
    ).read_text(encoding="utf-8")


def test_production_entry_refuses_the_npu_contract_and_a_non_metal_legacy_target():
    import argparse

    from tcenv.remote.grpc_service import verify_deployment

    base = {"worker_module": "tcenv.remote.worker", "target_id": NPU_TARGET, "backend": "vendor-a"}
    npu = argparse.Namespace(
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID, **base,
    )
    with pytest.raises(ValueError, match="test-only"):
        verify_deployment(npu)

    legacy_off_metal = argparse.Namespace(
        api_contract_id=contracts.API_CONTRACT_ID,
        evidence_contract_id=contracts.EVIDENCE_CONTRACT_ID, **base,
    )
    with pytest.raises(ValueError, match="Metal target"):
        verify_deployment(legacy_off_metal)

    mixed = argparse.Namespace(
        api_contract_id=contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=contracts.EVIDENCE_CONTRACT_ID, **base,
    )
    with pytest.raises(ValueError, match="unsupported API/evidence contract pair"):
        verify_deployment(mixed)
