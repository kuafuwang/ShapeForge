"""Route-M offline review never turns a recheck into activation or a grant."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from taic import trust
from taic.route_m_recheck_audit import SCHEMA, audit, render_summary

A = "kernel void a(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 1.0f; }\n"
B = "kernel void b(device float *y [[buffer(0)]], uint i [[thread_position_in_grid]]) { y[i] = 2.0f; }\n"


def _case(runs: Path, *, conclusion: str = "no_diff") -> tuple[Path, Path]:
    run = runs / "apple-m4-max" / "run-a"
    folder = run / "candidates" / "001_r1.p0"
    folder.mkdir(parents=True)
    source_dir = run / "exact-sources"
    source_dir.mkdir()
    left, right = source_dir / "a.metal", source_dir / "b.metal"
    left.write_text(A, encoding="utf-8")
    right.write_text(B, encoding="utf-8")
    sha_a, sha_b = hashlib.sha256(A.encode()).hexdigest(), hashlib.sha256(B.encode()).hexdigest()
    document = {
        "schema": "taic.paired_comparison.1",
        "applied": True,
        "gpu": True,
        "timed_by": "gpu_timestamp",
        "backend": "metal",
        "sha_a": sha_a,
        "sha_b": sha_b,
        "conclusion": conclusion,
        "direction": "a_faster" if conclusion == "diff" else None,
    }
    (folder / "paired_comparison_recheck.json").write_text(json.dumps({
        "command": ["python", "-m", "tcenv.cli", "paired-compare", str(left), str(right), "--task", "conv2d"],
        "document": document,
    }), encoding="utf-8")
    (folder / "report.json").write_text(json.dumps({"task": "conv2d"}), encoding="utf-8")
    (run / "run_config.json").write_text(json.dumps({
        "task": "conv2d", "target": {"id": "apple-m4-max", "backend": "metal"},
    }), encoding="utf-8")
    (run / "feedback.jsonl").write_text(json.dumps({
        "label": "r1.p0", "seq": 1, "provider": "deepseek", "model": "deepseek-flash",
        "effort": "high", "task": "conv2d", "target_id": "apple-m4-max", "backend": "metal",
        "measurement_kind": "measured",
    }) + "\n", encoding="utf-8")
    return run, runs


def _snapshot(root: Path) -> dict[str, int]:
    return {str(path): path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()}


def test_real_trust_failure_is_explicit_and_writes_nothing(tmp_path: Path):
    runs = tmp_path / "runs"
    run, _ = _case(runs)
    targets = tmp_path / "targets"
    before = _snapshot(tmp_path)
    document = audit(runs, trust_root=targets)
    assert document["schema"] == SCHEMA
    assert document["actual_route_m"]["coverage"] == "unavailable"
    case = document["offline_replay"]["cases"][0]
    assert case["blocker"] == "gate:overlay_tip_trust"
    assert [gate["status"] for gate in case["gates"][:4]] == ["pass"] * 4
    assert case["gates"][4]["reason"] == "no active grant"
    assert case["shadow_eligible"] is False
    assert case["delivery"]["injected_bytes"] == 0
    assert document["activation_authorized"] is False
    assert _snapshot(tmp_path) == before
    assert not (run / "harness_route_m.jsonl").exists()
    assert not (run / "harness_overlay.jsonl").exists()


def test_a_synthetic_grant_can_make_shadow_eligible_but_never_active(tmp_path: Path):
    runs = tmp_path / "runs"
    run, _ = _case(runs)
    targets = tmp_path / "targets"
    trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high",
        scopes=["overlay:tip"], root=targets,
    )
    before = _snapshot(run)
    document = audit(runs, trust_root=targets)
    case = document["offline_replay"]["cases"][0]
    assert case["shadow_eligible"] is True
    assert all(gate["status"] == "pass" for gate in case["gates"])
    assert case["delivery"]["delivered"] is False
    assert case["delivery"]["injected_bytes"] == 0
    assert document["activation_authorized"] is False
    assert document["changes_activation"] is False
    assert _snapshot(run) == before
    assert not (run / "harness_route_m.jsonl").exists()
    assert not (run / "harness_overlay.jsonl").exists()


def test_too_noisy_is_not_a_deliverable(tmp_path: Path):
    runs = tmp_path / "runs"
    _case(runs, conclusion="too_noisy")
    document = audit(runs, trust_root=tmp_path / "targets")
    case = document["offline_replay"]["cases"][0]
    assert case["blocker"] == "not_run:too_noisy is not a deliverable conclusion"
    assert all(gate["status"] == "not_run" for gate in case["gates"])
    assert "offline" in render_summary(document).lower()


def test_real_replay_is_read_only_and_never_counts_missing_traces_as_zero_success():
    root = Path(__file__).resolve().parents[2]
    runs = root / "runs"
    if not runs.is_dir():
        pytest.skip("requires the repository's real runs/ corpus")
    before = _snapshot(runs)
    document = audit(runs, trust_root=root / "targets")
    after = _snapshot(runs)
    assert after == before
    actual = document["actual_route_m"]
    if not actual["trace_files"]:
        assert actual["coverage"] == "unavailable"
        assert actual["traces"] == 0
    for case in document["offline_replay"]["cases"]:
        assert case["delivery"]["delivered"] is False
        assert case["delivery"]["injected_bytes"] == 0
    assert document["activation_authorized"] is False
    assert document["changes_activation"] is False
