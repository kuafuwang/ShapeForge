"""Opt-in Docker isolated-agent simulation (Phase 4, simulated only).

This is a **real** isolated execution: the fake agent runs inside a pre-provisioned Ubuntu
container over SSH, receives a short-lived scoped token through a host-side helper, and calls
the control-host fake provider.  It is still a simulation: the agent CLI is a deterministic
fixture, the provider is fake, and the fixture flow relaxes ``online`` to ``ALL`` so the fake
agent can reach the provider.  The production flow keeps ``online=NONE``; this run does not
prove the production network fence.

It is skipped unless ``TAIC_DOCKER_AGENT_SIM=1`` is set, because it builds a container image
and needs Docker Desktop.  The runner writes ``runs/remote-agent-sim-<date>/report.json`` when
invoked directly; the test writes a temporary report so the safe suite does not pollute the
acceptance artifact.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("TAIC_DOCKER_AGENT_SIM") != "1",
    reason="set TAIC_DOCKER_AGENT_SIM=1 to run the isolated-agent Docker simulation",
)


def test_the_isolated_fake_agent_runs_in_a_docker_host_and_cleans_up(tmp_path):
    from tests.support.agent_sim.runner import REQUIRED_CHECKS, run_simulation

    report = run_simulation(report_path=tmp_path / "report.json")
    assert report["simulated"] is True
    assert report["publication_allowed"] is False
    assert report["status"] == "passed", report.get("failure") or report.get("failed_checks")
    assert report["failed_checks"] == []
    assert report["missing_required"] == []
    assert report["required_checks"] == list(REQUIRED_CHECKS)

    results = {entry["name"]: entry for entry in report["results"]}
    for name in REQUIRED_CHECKS:
        assert results[name]["status"] == "passed", (name, results[name])
    assert any("real pi CLI" in item for item in report["not_run"])
    assert any("production flow with online=NONE" in item for item in report["not_run"])

    # The submission and the fake agent's provider evidence came back from the container
    # through the real SSH transport into the canonical agent_work layout.
    notes = json.loads(report["session"]["notes"])
    assert notes["credential_present"] is True
    assert notes["provider_status"] == 200
    assert notes["provider_body"] == "fake"
    credential = report["session"]["identity"]["credential"]
    assert credential["provisioned_to_agent"] is True
    assert credential["revoked"] is True
    assert "live_after_cleanup" not in credential
    assert report["session"]["identity"]["harness_recorded"] == "env"
    assert report["session"]["identity"]["cleaned"] is True
    assert report["session"]["status"] == "timeout" or report["session"]["status"] == "ok"
    assert report["session"]["diagnostics"]["session_ids"] == ["fake-acp-session"]
