"""Phase 2D2 simulated gRPC/TLS integration test (Docker Linux VM).

This test is opt-in because it builds a container image and needs Docker
Desktop.  The runner itself writes ``runs/grpc-sim-<date>/report.json`` when
invoked directly; the test writes a temporary report so CI does not pollute the
acceptance artifact.

``status=passed`` is only accepted here when every required Phase 2D check ran
and passed, the optional ``not_run`` list contains no required item, and the
post-cleanup evidence is final (no acceptance containers, worker process groups
or temp directories left behind).
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
    from tests.support.npu_sim.grpc_sim_runner import REQUIRED_CHECKS, build_and_run

    report = build_and_run(report_path=tmp_path / "report.json", fresh=True)
    assert report["simulated"] is True
    assert report["publication_allowed"] is False
    assert report["status"] == "passed", report.get("failure") or report.get("failed_checks") or report.get("blockers")
    assert report["blockers"] == []
    assert report["failed_checks"] == []
    assert report["missing_required"] == []
    assert report["required_checks"] == list(REQUIRED_CHECKS)

    results = {entry["name"]: entry for entry in report["results"]}
    for name in REQUIRED_CHECKS:
        assert name in results, f"required check did not run: {name}"
        assert results[name]["ok"], results[name]
    assert results["cleanup"]["ok"], results["cleanup"]
    cleanup = report["cleanup"]
    assert cleanup["final"] is True
    assert cleanup["containers_left"] == []
    assert cleanup["kept_for_inspection"] is False
    assert set(cleanup["volumes_retained"]) == {
        "taic-sim-state", "taic-sim-limits", "taic-sim-backup", "taic-sim-restored",
    }
    assert not any(cleanup["temp_paths"].values()), cleanup["temp_paths"]
    assert not any(cleanup["process_groups"].values()), cleanup["process_groups"]

    # Evidence for the previously missing acceptance items is present.
    unsigned = results["unsigned_result_rejected"]["evidence"]
    assert unsigned["durable_result_signed"] is True
    revoked_cert = results["revoked_client_cert_rejected"]["evidence"]
    assert len(revoked_cert["revoked_fingerprint_sha256"]) == 64
    rollback = results["rollback_previous_image"]["evidence"]
    assert rollback["rolled_back_from_image_id"] != rollback["rolled_back_to_image_id"]
    assert rollback["rolled_back_from_revision"] != rollback["rolled_back_to_revision"]
    assert rollback["previous_commit"]
    assert rollback["product_diff_files"], rollback
    assert rollback["same_persistent_schema"] is True
    rotation = results["hmac_key_rotation_historical_verification"]["evidence"]
    assert rotation["historical_key_id"] != rotation["new_key_id"]
    assert results["hmac_key_revocation_quarantines_history"]["evidence"]["quarantined"] is True
    assert results["grant_renewal_reattach"]["evidence"]["attempt_id"]
    assert results["active_attempt_cancelled_on_revocation"]["evidence"]["state"] == "cancelled"
    assert results["secret_redaction"]["evidence"]["error_echoed_token"] is False
    assert report["environment"]["image_id"] == rollback["rolled_back_from_image_id"]
    assert report["previous_source"]["commit"] == rollback["previous_commit"]

    # The report is diagnostic, not a search/publication artifact.
    not_run = set(report["not_run"])
    assert "real NPU" in not_run
    assert not_run.isdisjoint(REQUIRED_CHECKS)
    for forbidden in ("run_config.json", "confirmed", "catalog", "brief"):
        assert not (tmp_path / forbidden).exists()
    assert json.loads((tmp_path / "report.json").read_text())["status"] == "passed"
