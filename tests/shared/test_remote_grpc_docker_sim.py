"""Phase 2D2 simulated gRPC/TLS integration test (Docker Linux VM).

This test is opt-in because it builds a container image and needs Docker
Desktop.  The runner itself writes ``runs/grpc-sim-<date>/report.json`` when
invoked directly; the test writes a temporary report so CI does not pollute the
acceptance artifact.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TAIC_DOCKER_GRPC_SIM") != "1",
    reason="set TAIC_DOCKER_GRPC_SIM=1 to run the simulated Docker acceptance",
)


def test_simulated_grpc_tls_docker_acceptance(tmp_path):
    from tests.support.npu_sim.grpc_sim_runner import build_and_run

    report = build_and_run(report_path=tmp_path / "report.json", fresh=True)
    assert report["simulated"] is True
    assert report["publication_allowed"] is False
    assert report["status"] == "passed", report.get("failure") or report.get("failed_checks")
    results = {entry["name"]: entry for entry in report["results"]}
    for name in (
        "tls_mtls_capabilities",
        "simulated_result_and_publication_gate",
        "retry_only_interrupted",
        "cancel_running_job",
        "revoked_grant_rejected",
        "wrong_hmac_key_rejected",
        "revoked_hmac_key_rejected",
        "connection_reset_reconciles_via_getjob",
        "half_open_deadline_then_getjob",
        "container_restart_explicit_retry",
        "stale_worker_fail_closed",
        "backup_restore_quarantine",
        "rollback_restart_compat",
        "cleanup",
    ):
        assert results[name]["ok"], results[name]
    assert "real NPU" in report["not_run"]
    assert not (tmp_path / "run_config.json").exists()
