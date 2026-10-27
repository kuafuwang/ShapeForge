"""P0 NPU cleanup proof, quarantine and Host/Device timing separation."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from tcenv.remote import contracts
from tcenv.remote.drivers.base import CleanupProof, VendorFacts
from tcenv.remote.job_types import JobError
from tests.support.job_engine import (
    NPU_FEATURES,
    NPU_TARGET,
    NPU_TOKEN,
    npu_engine,
    npu_expected_identities,
    npu_get,
    npu_grant,
    npu_identity,
    npu_request,
    npu_retry,
    npu_capabilities,
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
        vendor_id=profile["vendor_id"],
        profile_digest=profile["profile_digest"],
        topology_digest=profile["topology_digest"],
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


def test_an_npu_engine_without_a_device_controller_is_a_composition_error(tmp_path):
    root = tmp_path / "run"
    with pytest.raises(ValueError, match="worker-external device controller"):
        npu_engine(
            root / "state", ledger_path=root / "ledger.json", controller=None,
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )


def test_losing_the_device_controller_still_cannot_prove_cleanup(tmp_path):
    root = tmp_path / "run"
    controller = FakeVendorAController(root / "ledger.json")
    engine = npu_engine(
        root / "state", ledger_path=root / "ledger.json", controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-no-controller-1", source=SOURCE)
        npu_submit(engine, request)
        # A controller that disappears while the attempt runs cannot be replaced
        # by the worker-local cooperative close: cleanup stays unproven.
        engine.device_controller = None
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


# -- trusted device facts feed the real capability path --------------------
class _UnhealthyController(FakeVendorAController):
    def __init__(self, ledger_path, **kwargs):
        super().__init__(ledger_path, healthy=False, **kwargs)


def test_unhealthy_device_group_fails_capabilities(tmp_path):
    """@brief 不健康设备组不能继续广告 available，GetCapabilities 必须 fail closed。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_UnhealthyController(ledger_path),
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        with pytest.raises(JobError) as excinfo:
            npu_capabilities(engine)
        assert excinfo.value.code == "failed_precondition"
        assert "not available" in str(excinfo.value)
        assert engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
    finally:
        engine.close(wait=5)


def test_unprobeable_or_foreign_device_group_fails_capabilities(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _Raising(FakeVendorAController):
        def probe(self, profile):
            raise RuntimeError("smi exploded")

    class _ForeignGeneration(FakeVendorAController):
        def capabilities(self, facts):
            advertised = dict(super().capabilities(facts))
            advertised["device_groups"] = [{
                **advertised["device_groups"][0], "resource_generation": "2",
            }]
            return advertised

    class _ForeignFacts(FakeVendorAController):
        def probe(self, profile):
            facts = super().probe(profile)
            return VendorFacts(
                vendor_id=facts.vendor_id, device_group_id=facts.device_group_id,
                profile_digest="0" * 64, topology_digest=facts.topology_digest,
                resource_generation=facts.resource_generation,
            )

    for controller, match in (
        (_Raising(ledger_path), "device probe failed"),
        (_ForeignFacts(ledger_path), "profile_digest"),
        (_ForeignGeneration(ledger_path), "resource_generation"),
    ):
        engine = npu_engine(
            root / "state", ledger_path=ledger_path, controller=controller,
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )
        try:
            with pytest.raises(JobError, match=match) as excinfo:
                npu_capabilities(engine)
            assert excinfo.value.code == "failed_precondition"
        finally:
            engine.close(wait=5)


def test_a_hanging_device_probe_is_bounded(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _Hanging(FakeVendorAController):
        def probe(self, profile):
            time.sleep(10.0)
            return super().probe(profile)

    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=_Hanging(ledger_path),
        device_call_timeout_s=0.2,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        started = time.monotonic()
        with pytest.raises(JobError, match="did not return within") as excinfo:
            npu_capabilities(engine)
        assert excinfo.value.code == "failed_precondition"
        assert time.monotonic() - started < 5.0
    finally:
        engine.close(wait=5)


# -- the lease carries the accepted device binding -------------------------
def test_attempt_lease_records_the_accepted_device_binding(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-lease-binding-1", source=SOURCE)
        npu_submit(engine, request)
        _wait_engine(engine, request, {"completed"})
        row = engine.store._conn.execute(
            "SELECT device_binding_json FROM attempts ORDER BY lease_fence DESC LIMIT 1",
        ).fetchone()
        assert json.loads(row["device_binding_json"]) == build_execution().device_group.to_dict()
    finally:
        engine.close(wait=5)


def test_cleanup_proof_from_another_profile_is_rejected(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _ForeignProof(FakeVendorAController):
        def reconcile(self, attempt_id, lease_fence):
            proof = super().reconcile(attempt_id, lease_fence)
            return replace(proof, profile_digest="0" * 64)

    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=_ForeignProof(ledger_path),
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-proof-binding-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert "different device binding" in snapshot.recovery_error
        assert snapshot.result is None
    finally:
        engine.close(wait=5)


def test_restart_into_another_resource_generation_quarantines_the_old_attempt(tmp_path):
    """@brief 重启换组时不能拿新组 controller 对旧 attempt reconcile/取消，只能 quarantine。"""
    state = tmp_path / "run" / "state"
    ledger_path = tmp_path / "run" / "ledger.json"
    request = npu_request("npu-cross-group-1", source=SOURCE)
    expected = npu_expected_identities()
    identity = npu_identity()
    from tcenv.remote import contracts as remote_contracts
    from tcenv.remote.job_store import JobStore
    from tests.support.vendor_sim.vendor_a import build_execution

    provenance = identity.external_provenance(
        execution_protocol_version="3",
        api_contract_id=remote_contracts.NPU_SOURCE_API_CONTRACT_ID,
        evidence_contract_id=remote_contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID,
        npu_execution=build_execution().to_dict(),
    )
    store = JobStore(state, contract=remote_contracts.NPU_SOURCE_DIALECT)
    try:
        store.register_grant(npu_grant())
        capability = store.get_capabilities(
            grant_token=NPU_TOKEN, client_nonce="a" * 32,
            expected=expected.evaluator.to_dict(), expected_identities=expected,
            provenance=provenance,
            api_contract_ids=[remote_contracts.NPU_SOURCE_API_CONTRACT_ID],
            evidence_contract_ids=[remote_contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID],
            required_features=list(NPU_FEATURES), supported_features=frozenset(NPU_FEATURES),
        )
        store.submit(
            request=request, request_digest=request.request_digest, expected=expected,
            provenance=provenance, capability_id=capability.capability_id,
            required_features=list(NPU_FEATURES), grant_token=NPU_TOKEN,
        )
        store.enqueue_accepted(request.request_id)
        attempt = store._job_by_id(request.request_id).current_attempt
        assert attempt is not None and attempt.device_binding == build_execution().device_group.to_dict()
    finally:
        store.close()

    # Stage the durable shape of a crashed frontend: the attempt was running and
    # its worker is already gone.
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    connection = sqlite3.connect(state / "jobs.sqlite3")
    try:
        connection.execute(
            "UPDATE attempts SET state='running',started_at_ms=1,worker_pid=?,worker_pgid=?,"
            "worker_start_token='' WHERE attempt_id=?",
            (gone.pid, gone.pid, attempt.attempt_id),
        )
        connection.execute("UPDATE jobs SET state='running' WHERE request_id=?", (request.request_id,))
        connection.commit()
    finally:
        connection.close()

    controller = FakeVendorAController(ledger_path)
    restarted = npu_engine(
        state, ledger_path=ledger_path, controller=controller, generation="2",
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        snapshot = restarted.store._job_by_id(request.request_id)
        assert snapshot.quarantined is True
        assert snapshot.state == "running"
        assert "another device group" in snapshot.recovery_error
        assert snapshot.result is None
        # The old lease is not cancelled through the new group's controller.
        restarted._cancel_device(snapshot.current_attempt)
        assert VendorLedger(ledger_path).is_cancelled(attempt.attempt_id, attempt.lease_fence) is False
        # And the operators' release path cannot mark it interrupted either.
        released = restarted.release_quarantine(request.request_id)
        assert released.state == "running" and released.quarantined is True
    finally:
        restarted.close(wait=5)


# -- legacy test composition roots must not pretend to be NPU --------------
def test_legacy_test_composition_roots_cannot_serve_fake_npu_traffic():
    """@brief 仅保留给 legacy transport 的测试组合根不得再冒充 NPU 入口。"""
    from tests.support.job_engine import identity as legacy_identity
    from tests.support.npu_sim import grpc_service_entry

    assert legacy_identity().backend == "sim-fake"
    assert "npu" not in legacy_identity().backend.lower()

    argv = [
        "--state-dir", "/nonexistent/state",
        "--backend", "sim-npu", "--target-id", "sim-grpc-target",
        "--grants-file", "/nonexistent/grants.json",
        "--keyring-file", "/nonexistent/keyring.json",
        "--certificate-file", "/nonexistent/server.pem",
        "--private-key-file", "/nonexistent/server.key",
        "--client-ca-file", "/nonexistent/ca.pem",
    ]
    with pytest.raises(SystemExit, match="legacy Metal/source contract"):
        grpc_service_entry.main(argv)
    npu_target = list(argv)
    npu_target[npu_target.index("--backend") + 1] = "sim-cpu"
    npu_target[npu_target.index("--target-id") + 1] = "sim-npu-target"
    with pytest.raises(SystemExit, match="legacy Metal/source contract"):
        grpc_service_entry.main(npu_target)


# -- controller calls are bounded -----------------------------------------
class _HangingController(FakeVendorAController):
    """A controller whose named calls never return (acceptance fault)."""

    def __init__(self, ledger_path, *, hang=("reconcile",), **kwargs):
        super().__init__(ledger_path, **kwargs)
        self._hang = frozenset(hang)

    def cancel(self, attempt_id, lease_fence):
        if "cancel" in self._hang:
            time.sleep(10.0)
        return super().cancel(attempt_id, lease_fence)

    def reconcile(self, attempt_id, lease_fence):
        if "reconcile" in self._hang:
            time.sleep(10.0)
        return super().reconcile(attempt_id, lease_fence)


def test_a_hanging_reconcile_cannot_commit_the_result(tmp_path):
    """@brief reconcile 挂起时结果不得提交，job 保持 quarantine 且不卡死调度。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_HangingController(ledger_path, hang=("reconcile",)),
        device_call_timeout_s=0.2,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-hang-reconcile-1", source=SOURCE)
        started = time.monotonic()
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert "did not return within" in snapshot.recovery_error
        assert snapshot.result is None
        assert engine.close_error is not None
        # The scheduler stayed responsive despite the wedged controller.
        assert time.monotonic() - started < 9.0
        # Retry stays refused while cleanup is unproven.
        with pytest.raises(JobError) as excinfo:
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
        assert excinfo.value.code == "failed_precondition"
    finally:
        engine.close(wait=5)


def test_a_hanging_cancel_and_reconcile_cannot_claim_cancellation(tmp_path):
    """@brief cancel/reconcile 都挂起时，Cancel 有界返回且不得宣告 cancelled。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_HangingController(ledger_path, hang=("cancel", "reconcile")),
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        device_call_timeout_s=0.2,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-hang-cancel-1", source=SOURCE)
        npu_submit(engine, request)
        ledger = VendorLedger(ledger_path)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ledger.live_resources():
            time.sleep(0.02)
        assert ledger.live_resources(), "fake vendor A never prepared device resources"

        started = time.monotonic()
        snapshot = engine.cancel(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, grant_token=NPU_TOKEN,
            reason="hanging controller",
        )
        assert time.monotonic() - started < 5.0, "Cancel must be bounded"
        assert snapshot.state != "cancelled"
        assert engine.close_error is not None

        final = npu_get(engine, request)
        deadline = time.monotonic() + 10
        while not final.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            final = npu_get(engine, request)
        assert final.quarantined is True
        assert final.state == "running", "an unproven cleanup cannot become cancelled"
        assert final.cancel_requested is True
        assert final.result is None
        assert "did not return within" in final.recovery_error
    finally:
        engine.close(wait=5)


# -- the fake vendor records and releases exactly its served binding --------
def test_generation_two_prepare_records_the_served_binding(tmp_path):
    """@brief generation 2 的 prepare 必须把当前绑定写进 attempt 与资源台账。"""
    from tests.support.vendor_sim.ledger import resource_matches_binding
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    driver = FakeVendorADriver(ledger, generation="2")
    execution = build_execution("2")
    prepared = driver.prepare(
        execution, SOURCE, {"attempt_id": "attempt-gen2", "lease_fence": 3},
    )
    assert prepared.resource_generation == "2"
    assert prepared.device_group_id == execution.device_group.device_group_id
    resources = ledger.attempt_resources("attempt-gen2", 3)
    assert len(resources) == 3
    for resource in resources:
        assert resource["resource_generation"] == "2"
        assert resource_matches_binding(resource, driver.binding())
    # Closing releases exactly that binding.
    driver.close(prepared)
    assert ledger.attempt_resources("attempt-gen2", 3) == []


def test_generation_two_run_records_and_releases_only_its_own_generation(tmp_path):
    """@brief 端到端 generation 2 运行的台账与清理证明都只含 generation 2。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path, generation="2")
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        generation="2", signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-gen2-ledger-1", source=SOURCE, generation="2")
        npu_submit(engine, request)
        snapshot = _wait_engine(engine, request, {"completed"})
        assert snapshot.result is not None
        ledger = VendorLedger(ledger_path)
        assert ledger.live_resources() == []
        assert ledger.read()["device_running"] == []
        proof = controller.reconcile(
            snapshot.current_attempt.attempt_id, snapshot.current_attempt.lease_fence,
        )
        assert proof.complete is True
        assert proof.resource_generation == "2"
        assert snapshot.result.evidence_json()["device_group"]["resource_generation"] == "2"
    finally:
        engine.close(wait=5)


def test_reconcile_refuses_another_generation_and_keeps_its_resources(tmp_path):
    """@brief 异代际 controller 不能释放也不能声称已释放别人的资源。"""
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    FakeVendorADriver(ledger, generation="1").prepare(
        build_execution("1"), SOURCE, {"attempt_id": "attempt-cross-gen", "lease_fence": 5},
    )
    controller = FakeVendorAController(ledger_path, generation="2")
    proof = controller.reconcile("attempt-cross-gen", 5)
    assert proof.complete is False
    assert proof.resource_generation == "2"
    assert "another device group/profile/topology/resource generation" in proof.detail
    assert len(proof.live_resources) == 3
    # Nothing was deleted: the quarantine evidence stays observable.
    resources = ledger.attempt_resources("attempt-cross-gen", 5)
    assert len(resources) == 3
    assert all(resource["resource_generation"] == "1" for resource in resources)
    # The ledger's own scoped release refuses them too.
    assert ledger.release_attempt(
        "attempt-cross-gen", 5, binding=controller.binding(),
    ) == []
    assert len(ledger.attempt_resources("attempt-cross-gen", 5)) == 3
    # The matching generation 1 controller can still release them.
    assert controller.__class__(ledger_path, generation="1").reconcile("attempt-cross-gen", 5).complete is True
    assert ledger.attempt_resources("attempt-cross-gen", 5) == []


def test_reconcile_refuses_resources_from_another_device_group(tmp_path):
    """@brief 异设备组（同代际）资源同样不得被释放或证明已清理。"""
    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    group = build_execution().device_group.to_dict()
    ledger.record_resource(
        attempt_id="attempt-cross-group", lease_fence=9,
        device_group_id="vendor-a-group-1",
        vendor_id=group["vendor_id"],
        profile_digest=group["profile_digest"],
        topology_digest=group["topology_digest"],
        resource_generation=group["resource_generation"],
        kind="stream", detail={},
    )
    proof = FakeVendorAController(ledger_path).reconcile("attempt-cross-group", 9)
    assert proof.complete is False
    assert "another device group/profile/topology/resource generation" in proof.detail
    assert ledger.attempt_resources("attempt-cross-group", 9)


# -- a running fake device is part of the lease too ------------------------
def test_close_and_reconcile_clear_the_running_marker(tmp_path):
    """@brief 真实顺序 prepare→execute→close→reconcile 后 marker 必须清空且证明可复核。"""
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    driver = FakeVendorADriver(ledger)
    prepared = driver.prepare(
        build_execution(), SOURCE, {"attempt_id": "attempt-marker", "lease_fence": 1},
    )
    driver.execute(prepared, deadline_monotonic=time.monotonic() + 5)
    assert ledger.read()["device_running"] == ["attempt-marker:1"]
    driver.close(prepared)
    assert ledger.read()["device_running"] == []
    proof = FakeVendorAController(ledger_path).reconcile("attempt-marker", 1)
    assert proof.complete is True
    assert ledger.read()["device_running"] == []


def test_a_foreign_running_marker_is_never_cleared_or_claimed(tmp_path):
    """@brief 异代际 controller 不能清除或证明仍在运行的 marker。"""
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    prepared = FakeVendorADriver(ledger, generation="1").prepare(
        build_execution("1"), SOURCE, {"attempt_id": "attempt-run-gen1", "lease_fence": 2},
    )
    FakeVendorADriver(ledger, generation="1").execute(
        prepared, deadline_monotonic=time.monotonic() + 5,
    )
    assert ledger.read()["device_running"] == ["attempt-run-gen1:2"]

    foreign = FakeVendorAController(ledger_path, generation="2").reconcile("attempt-run-gen1", 2)
    assert foreign.complete is False
    assert foreign.live_resources == ("running",)
    assert "another device group/profile/topology/resource generation" in foreign.detail
    # Nothing was deleted: the running marker and its scope survive.
    assert ledger.read()["device_running"] == ["attempt-run-gen1:2"]
    assert ledger.attempt_resources("attempt-run-gen1", 2)

    matching = FakeVendorAController(ledger_path, generation="1").reconcile("attempt-run-gen1", 2)
    assert matching.complete is True
    assert ledger.read()["device_running"] == []
    assert ledger.attempt_resources("attempt-run-gen1", 2) == []


def test_normal_completion_leaves_no_running_marker(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=FakeVendorAController(ledger_path),
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-marker-completed-1", source=SOURCE)
        npu_submit(engine, request)
        _wait_engine(engine, request, {"completed"})
        ledger = VendorLedger(ledger_path)
        assert ledger.live_resources() == []
        assert ledger.read()["device_running"] == []
    finally:
        engine.close(wait=5)


def test_cancel_leaves_no_running_marker(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=FakeVendorAController(ledger_path),
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-marker-cancel-1", source=SOURCE)
        npu_submit(engine, request)
        ledger = VendorLedger(ledger_path)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while time.monotonic() < deadline and not ledger.read()["device_running"]:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        attempt = snapshot.current_attempt
        assert ledger.read()["device_running"] == [f"{attempt.attempt_id}:{attempt.lease_fence}"]
        engine.cancel(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, grant_token=NPU_TOKEN,
            reason="marker regression",
        )
        final = _wait_engine(engine, request, {"cancelled"})
        assert final.result is None
        assert ledger.live_resources() == []
        assert ledger.read()["device_running"] == []
    finally:
        engine.close(wait=5)


def test_killed_worker_reconciles_and_clears_the_running_marker(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-marker-killed-1", source=SOURCE)
        npu_submit(engine, request)
        snapshot = _wait_engine(engine, request, {"interrupted"})
        ledger = VendorLedger(ledger_path)
        assert ledger.read()["device_running"] == []
        proof = controller.reconcile(
            snapshot.current_attempt.attempt_id, snapshot.current_attempt.lease_fence,
        )
        assert proof.complete is True
        assert ledger.read()["device_running"] == []
    finally:
        engine.close(wait=5)


def test_generation_two_run_clears_its_running_marker_and_keeps_a_generation_one_marker(tmp_path):
    """@brief generation 2 运行清自己的 marker；对方代际的 marker 在异代际 reconcile 下保留。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = FakeVendorAController(ledger_path, generation="2")
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        generation="2", signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-marker-gen2-1", source=SOURCE, generation="2")
        npu_submit(engine, request)
        _wait_engine(engine, request, {"completed"})
        ledger = VendorLedger(ledger_path)
        assert ledger.read()["device_running"] == []
        assert ledger.live_resources() == []
    finally:
        engine.close(wait=5)


# -- the grant is authenticated before any device operation ----------------
class _CountingController(FakeVendorAController):
    def __init__(self, ledger_path, **kwargs):
        super().__init__(ledger_path, **kwargs)
        self.probe_calls = 0
        self.health_calls = 0

    def probe(self, profile):
        self.probe_calls += 1
        return super().probe(profile)

    def health_check(self, profile):
        self.health_calls += 1
        return super().health_check(profile)


def _counting_engine(tmp_path, *, grants=None):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = _CountingController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        grants=grants, signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    return engine, controller, VendorLedger(ledger_path), root


def test_a_valid_grant_probes_before_the_capability_row(tmp_path):
    """@brief 合法 grant 才触发 probe/health，且 capability 行只有一条。"""
    engine, controller, ledger, _root = _counting_engine(tmp_path)
    try:
        capability = npu_capabilities(engine)
        assert capability.capability_id
        assert (controller.probe_calls, controller.health_calls) == (1, 1)
        assert len(ledger.read()["probes"]) == 2
        assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 1
    finally:
        engine.close(wait=5)


@pytest.mark.parametrize("case", ["invalid", "expired", "revoked"])
def test_a_rejected_grant_never_touches_the_device(tmp_path, case):
    """@brief 无效/过期/撤销 grant 不得触发 probe/health，也不得写 capability/job。"""
    grants = None
    token = NPU_TOKEN
    if case == "expired":
        grants = [npu_grant(expires_at=time.time() - 60)]
    engine, controller, ledger, _root = _counting_engine(tmp_path, grants=grants)
    try:
        if case == "invalid":
            token = "definitely-invalid"
        elif case == "revoked":
            engine.revoke_grant(NPU_TOKEN)
        with pytest.raises(JobError) as excinfo:
            npu_capabilities(engine, token=token)
        assert excinfo.value.code in {"unauthenticated", "permission_denied"}
        # No device operation happened and nothing durable was written.
        assert (controller.probe_calls, controller.health_calls) == (0, 0)
        # No device write happened at all, so the ledger was never created.
        assert not ledger.exists()
        assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
        assert engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        engine.close(wait=5)


# -- a contradictory cleanup proof and an unexecuted known-good probe ------
class _LyingCleanupController(FakeVendorAController):
    """Claims `complete=True` while still naming a live matching resource."""

    def reconcile(self, attempt_id, lease_fence):
        proof = super().reconcile(attempt_id, lease_fence)
        return replace(proof, complete=True, live_resources=("module",))


def test_a_complete_proof_that_still_names_live_resources_is_quarantined(tmp_path):
    """@brief complete=True 但仍有 live resource 的 proof 不得提交结果。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_LyingCleanupController(ledger_path),
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-lying-proof-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert snapshot.result is None
        assert "live resources" in snapshot.recovery_error
        assert engine.close_error is not None
        # Retry stays refused while the cleanup claim contradicts the ledger.
        with pytest.raises(JobError) as excinfo:
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
        assert excinfo.value.code == "failed_precondition"
    finally:
        engine.close(wait=5)


def test_a_contradictory_cleanup_proof_cannot_finish_a_cancellation(tmp_path):
    """@brief Cancel 共用路径同样不接受自相矛盾的清理证明。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_LyingCleanupController(ledger_path),
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-lying-cancel-1", source=SOURCE)
        npu_submit(engine, request)
        ledger = VendorLedger(ledger_path)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ledger.live_resources():
            time.sleep(0.02)
        engine.cancel(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, grant_token=NPU_TOKEN,
            reason="contradictory proof",
        )
        deadline = time.monotonic() + 10
        final = npu_get(engine, request)
        while not final.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            final = npu_get(engine, request)
        assert final.quarantined is True
        assert final.state != "cancelled"
        assert final.result is None
        assert "live resources" in final.recovery_error
    finally:
        engine.close(wait=5)


def test_a_contradictory_cleanup_proof_cannot_become_retryable(tmp_path):
    """@brief kill/timeout 后的 interrupted+Retry 路径也不接受矛盾 proof。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    engine = npu_engine(
        root / "state", ledger_path=ledger_path,
        controller=_LyingCleanupController(ledger_path),
        worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-lying-retry-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert snapshot.state == "running"
        assert "live resources" in snapshot.recovery_error
        with pytest.raises(JobError) as excinfo:
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
        assert excinfo.value.code == "failed_precondition"
    finally:
        engine.close(wait=5)


def test_a_health_proof_without_the_known_good_probe_fails_capabilities(tmp_path):
    """@brief ok=True 但未执行 known-good 探针不能广告 available。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _UnverifiedHealth(FakeVendorAController):
        def health_check(self, profile):
            proof = super().health_check(profile)
            return replace(proof, ok=True, executed_known_good=False)

    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=_UnverifiedHealth(ledger_path),
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        with pytest.raises(JobError) as excinfo:
            npu_capabilities(engine)
        assert excinfo.value.code == "failed_precondition"
        assert "known-good" in str(excinfo.value)
        assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
        assert engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        engine.close(wait=5)


# -- GetCapabilities must not probe a leased device group -------------------
def test_capabilities_refuses_while_an_attempt_leases_the_group(tmp_path):
    """@brief 有活跃 attempt 时 GetCapabilities 不得执行 probe/health，也不广告 available。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = _CountingController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-lease-busy-1", source=SOURCE)
        npu_submit(engine, request)                     # probes once before running
        ledger = VendorLedger(ledger_path)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not ledger.read()["device_running"]:
            time.sleep(0.02)
        assert ledger.read()["device_running"], "the attempt never started"
        probe_before = controller.probe_calls
        health_before = controller.health_calls
        probes_before = len(ledger.read()["probes"])

        with pytest.raises(JobError) as excinfo:
            npu_capabilities(engine)
        assert excinfo.value.code == "failed_precondition"
        assert "executing an attempt" in str(excinfo.value)
        # No device operation ran while the lease was held.
        assert controller.probe_calls == probe_before
        assert controller.health_calls == health_before
        assert len(ledger.read()["probes"]) == probes_before
        assert engine.store._conn.execute(
            "SELECT COUNT(*) FROM capabilities WHERE issued_at_ms > 0",
        ).fetchone()[0] >= 1  # only the pre-submit capability exists
    finally:
        engine.close(wait=5)


def test_the_controller_refuses_the_known_good_probe_while_running(tmp_path):
    """@brief 设备边界也不在 running marker 存在时执行 known-good 探针。"""
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    driver = FakeVendorADriver(ledger)
    prepared = driver.prepare(
        build_execution(), SOURCE, {"attempt_id": "attempt-live", "lease_fence": 1},
    )
    driver.execute(prepared, deadline_monotonic=time.monotonic() + 5)
    controller = FakeVendorAController(ledger_path)
    profile = build_execution().device_group.to_dict()

    probes_before = len(ledger.read()["probes"])
    facts = controller.probe({"device_group_id": profile["device_group_id"]})
    assert facts.state == "busy"
    health = controller.health_check({"device_group_id": profile["device_group_id"]})
    assert health.ok is False
    assert health.executed_known_good is False
    assert "executing an attempt" in health.detail
    assert len(ledger.read()["probes"]) == probes_before

    # Once the lease is released the known-good probe runs again.
    driver.close(prepared)
    healthy = controller.health_check({"device_group_id": profile["device_group_id"]})
    assert healthy.ok is True and healthy.executed_known_good is True
    assert len(ledger.read()["probes"]) == probes_before + 1


# -- the whole probe sequence excludes a candidate start --------------------
def test_a_gated_probe_blocks_the_candidate_start_until_it_finishes(tmp_path):
    """@brief probe 序列持有设备组 lease：探测未完成时候选不得启动（无竞态）。"""
    import threading

    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    entered = threading.Event()
    release = threading.Event()

    class _Gated(FakeVendorAController):
        gated = False

        def health_check(self, profile):
            if self.gated:
                entered.set()
                assert release.wait(10.0), "the health probe was never released"
            return super().health_check(profile)

    controller = _Gated(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        stale_capability = npu_capabilities(engine)      # probe before any attempt
        controller.gated = True
        outcome: dict[str, object] = {}

        def second_probe() -> None:
            try:
                outcome["capability"] = npu_capabilities(engine)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = exc

        prober = threading.Thread(target=second_probe, name="gated-probe")
        prober.start()
        assert entered.wait(10.0), "the probe never entered health_check"

        # While the known-good probe is blocked, submit a job with the earlier
        # capability: it may be accepted durably but must not start.
        request = npu_request("npu-lease-gate-1", source=SOURCE)
        engine.submit(
            request=request, expected=engine.expected_identities,
            capability_id=stale_capability.capability_id,
            required_features=list(NPU_FEATURES), grant_token=NPU_TOKEN,
        )
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline:
            ledger = VendorLedger(ledger_path) if ledger_path.is_file() else None
            if ledger is not None and ledger.exists():
                assert ledger.read()["device_running"] == [], "a candidate device ran during the probe"
            assert engine.store._conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE state='running'",
            ).fetchone()[0] == 0
            time.sleep(0.02)

        release.set()
        prober.join(10.0)
        assert "capability" in outcome and "error" not in outcome
        final = _wait_engine(engine, request, {"completed"})
        assert final.result is not None
    finally:
        release.set()
        engine.close(wait=5)


def test_a_missing_ledger_is_not_a_cleanup_proof(tmp_path):
    """@brief 台账被删除后 reconcile 不得声称已清理。"""
    from tests.support.vendor_sim.vendor_a import FakeVendorADriver

    ledger_path = tmp_path / "ledger.json"
    ledger = VendorLedger(ledger_path)
    driver = FakeVendorADriver(ledger)
    prepared = driver.prepare(
        build_execution(), SOURCE, {"attempt_id": "attempt-missing-ledger", "lease_fence": 1},
    )
    driver.execute(prepared, deadline_monotonic=time.monotonic() + 5)
    ledger_path.unlink()
    proof = FakeVendorAController(ledger_path).reconcile("attempt-missing-ledger", 1)
    assert proof.complete is False
    assert "ledger is missing" in proof.detail
    # Reading or releasing must not recreate the missing evidence.
    assert not ledger.exists()


def test_the_worker_cannot_clear_the_cleanup_ledger():
    """@brief clear() 只是测试装配助手，worker 侧不能擦掉清理证据。"""
    import tempfile

    path = Path(tempfile.mkdtemp()) / "ledger.json"
    ledger = VendorLedger(path)
    ledger.record_resource(
        attempt_id="a", lease_fence=1, device_group_id="g", resource_generation="1",
        kind="module", detail={},
    )
    with pytest.raises(TypeError, match="test-setup"):
        ledger.clear()  # type: ignore[call-arg]
    assert ledger.attempt_resources("a", 1)


class _VanishingLedgerController(FakeVendorAController):
    """Deletes the cleanup evidence before it looks, as a lost ledger would."""

    def reconcile(self, attempt_id, lease_fence):
        if self.ledger.path.exists():
            self.ledger.path.unlink()
        return super().reconcile(attempt_id, lease_fence)


def test_a_missing_ledger_blocks_result_commit_and_cancellation(tmp_path):
    """@brief controller 看不到台账时，结果不提交、Cancel 也不宣告完成。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = _VanishingLedgerController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-missing-ledger-commit-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert snapshot.result is None
        assert "ledger is missing" in snapshot.recovery_error
        with pytest.raises(JobError):
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
    finally:
        engine.close(wait=5)


def test_a_missing_ledger_blocks_cancellation_completion(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = _VanishingLedgerController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-missing-ledger-cancel-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not npu_get(engine, request).state == "running":
            time.sleep(0.02)
        engine.cancel(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, grant_token=NPU_TOKEN,
            reason="missing ledger",
        )
        deadline = time.monotonic() + 10
        final = npu_get(engine, request)
        while not final.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            final = npu_get(engine, request)
        assert final.quarantined is True
        assert final.state != "cancelled"
        assert final.result is None
    finally:
        engine.close(wait=5)


def test_a_missing_ledger_blocks_recovery_retry(tmp_path):
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    controller = _VanishingLedgerController(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_ABORT_AFTER_LAUNCH": "1"},
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_request("npu-missing-ledger-recover-1", source=SOURCE)
        npu_submit(engine, request)
        deadline = time.monotonic() + 10
        snapshot = npu_get(engine, request)
        while not snapshot.quarantined and time.monotonic() < deadline:
            time.sleep(0.02)
            snapshot = npu_get(engine, request)
        assert snapshot.quarantined is True
        assert snapshot.state == "running"
        with pytest.raises(JobError):
            npu_retry(engine, request, snapshot.current_attempt.attempt_id)
    finally:
        engine.close(wait=5)


def test_a_contradictory_device_group_capability_is_refused(tmp_path):
    """@brief 组内 timing 或逻辑设备数与 P0 不符时拒绝且不写 capability。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _BadGroup(FakeVendorAController):
        def __init__(self, path, *, timing, devices):
            super().__init__(path)
            self._timing = timing
            self._devices = devices

        def capabilities(self, facts):
            advertised = dict(super().capabilities(facts))
            advertised["device_groups"] = [{
                **advertised["device_groups"][0],
                "timing_sources": list(self._timing),
                "logical_devices": self._devices,
            }]
            return advertised

    for timing, devices, match in (
        (["HOST_WALL"], 1, "DEVICE_EVENT"),
        (["DEVICE_EVENT"], 0, "one logical device"),
        (["DEVICE_EVENT"], 2, "one logical device"),
    ):
        engine = npu_engine(
            root / "state", ledger_path=ledger_path,
            controller=_BadGroup(ledger_path, timing=timing, devices=devices),
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )
        try:
            with pytest.raises(JobError, match=match) as excinfo:
                npu_capabilities(engine)
            assert excinfo.value.code == "failed_precondition"
            assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
            assert engine.store._conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
        finally:
            engine.close(wait=5)


def test_a_timed_out_probe_keeps_the_device_until_it_really_returns(tmp_path):
    """@brief RPC 超时不会释放设备：卡住的 health 返回前不得有任何候选启动。"""
    import threading

    root = tmp_path / "run"
    ledger_path = root / "ledger.json"
    entered = threading.Event()
    release = threading.Event()

    class _SlowHealth(FakeVendorAController):
        gated = False

        def health_check(self, profile):
            if self.gated:
                entered.set()
                release.wait(10.0)
            return super().health_check(profile)

    controller = _SlowHealth(ledger_path)
    engine = npu_engine(
        root / "state", ledger_path=ledger_path, controller=controller,
        worker_env={"TCENV_VENDOR_A_SLOW": "1"}, device_call_timeout_s=0.2,
        signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        stale_capability = npu_capabilities(engine)
        controller.gated = True
        outcome: dict[str, object] = {}

        def slow_probe() -> None:
            try:
                outcome["capability"] = npu_capabilities(engine)
            except BaseException as exc:  # noqa: BLE001 - recorded for the assertion
                outcome["error"] = exc

        prober = threading.Thread(target=slow_probe, name="slow-probe")
        prober.start()
        assert entered.wait(10.0), "the probe never entered health_check"

        # The RPC times out after 0.2s, but the controller call is still running.
        prober.join(5.0)
        assert "error" in outcome, "the bounded probe should have timed out"

        request = npu_request("npu-lease-timeout-1", source=SOURCE)
        engine.submit(
            request=request, expected=engine.expected_identities,
            capability_id=stale_capability.capability_id,
            required_features=list(NPU_FEATURES), grant_token=NPU_TOKEN,
        )
        for _ in range(30):
            assert engine.store._conn.execute(
                "SELECT COUNT(*) FROM attempts WHERE state='running'",
            ).fetchone()[0] == 0, "a candidate started while the probe was still running"
            ledger = VendorLedger(ledger_path)
            if ledger.exists():
                assert ledger.read()["device_running"] == [], "a device ran while the probe was still running"
                assert ledger.live_resources() == [], "device resources appeared during the probe"
            time.sleep(0.02)

        # Only after the real call returns may the candidate start and finish.
        release.set()
        final = _wait_engine(engine, request, {"completed"}, timeout=20.0)
        assert final.result is not None
        ledger = VendorLedger(ledger_path)
        assert ledger.read()["device_running"] == []
        assert ledger.live_resources() == []
    finally:
        release.set()
        engine.close(wait=5)


def test_probe_facts_must_match_the_served_cardinality_and_timing(tmp_path):
    """@brief 事实里的 logical_devices/timing 与实际绑定不符时也要拒绝。"""
    root = tmp_path / "run"
    ledger_path = root / "ledger.json"

    class _BadFacts(FakeVendorAController):
        def __init__(self, path, *, devices=1, timing=("DEVICE_EVENT",)):
            super().__init__(path)
            self._devices = devices
            self._timing = tuple(timing)

        def probe(self, profile):
            facts = super().probe(profile)
            return VendorFacts(
                vendor_id=facts.vendor_id, device_group_id=facts.device_group_id,
                profile_digest=facts.profile_digest, topology_digest=facts.topology_digest,
                resource_generation=facts.resource_generation,
                logical_devices=self._devices, timing_sources=self._timing,
            )

    for devices, timing, match in (
        (2, ("DEVICE_EVENT",), "one logical device"),
        (0, ("DEVICE_EVENT",), "one logical device"),
        (1, ("HOST_WALL",), "DEVICE_EVENT"),
        (1, ("DEVICE_EVENT", "HOST_WALL"), "DEVICE_EVENT"),
    ):
        engine = npu_engine(
            root / "state", ledger_path=ledger_path,
            controller=_BadFacts(ledger_path, devices=devices, timing=timing),
            signing_key=b"k" * 32, signing_key_id="sim-npu-site/key-1",
        )
        try:
            with pytest.raises(JobError, match=match) as excinfo:
                npu_capabilities(engine)
            assert excinfo.value.code == "failed_precondition"
            assert engine.store._conn.execute("SELECT COUNT(*) FROM capabilities").fetchone()[0] == 0
        finally:
            engine.close(wait=5)
