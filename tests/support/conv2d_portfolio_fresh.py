"""Read-only replay checks for the P2-K portfolio-level fresh suite."""

from __future__ import annotations

import json
from pathlib import Path

from taic.portfolio_confirmation import verify_subject_document
from taic.portfolio_manifest import DispatchManifest, ManifestSubject

REPO_ROOT = Path(__file__).resolve().parents[2]
P2K = REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2k"
FRESH_SUITE = P2K / "evidence" / "fresh-suite.json"
VALIDATED_MANIFEST = P2K / "evidence" / "validated-manifest.json"
PACKED_PROBE = P2K / "evidence" / "packed-half-probe.json"
FREEZE_DIGEST = "db264da7e7a4c389e120f740ea1e71806c7655e0d70c820453afe0d1cc2f2bb0"
VALIDATION_DIGEST = "8ceace0e53013cdf7d2c06b32db1225e3f766dfa7a565cd9bea854d5d8ea6d64"
EXPECTED_KEYS = {
    "fp16/small/benchmark",
    "fp16/paper/benchmark",
    "fp16/small/golden",
    "fp16/paper/golden",
}
EXPECTED_SHA = {
    "small": "50ac29d1bc5c0ebcb983f16d5faedeaa0869324b10a979613900b29a640486f2",
    "paper": "5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2",
}


def load_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def assert_fresh_suite() -> None:
    payload = load_json(FRESH_SUITE)
    if payload["schema_id"] != "shapeforge.conv2d-portfolio-fresh-suite.1":
        raise AssertionError("unknown P2-K suite schema")
    if payload["publication_allowed"] is not False:
        raise AssertionError("P2-K was marked as a publication")
    if payload["freeze_digest"] != FREEZE_DIGEST or payload["validation_digest"] != VALIDATION_DIGEST:
        raise AssertionError("P2-K freeze/validation identity drifted")
    subjects = payload["subjects"]
    if {item["instance_key"] for item in subjects} != EXPECTED_KEYS or len(subjects) != 4:
        raise AssertionError("P2-K does not cover exactly the four domain points")
    for subject in subjects:
        verified = verify_subject_document(subject)
        key = verified["instance_key"]
        size = key.split("/")[1]
        usage = verified["usage"]
        if verified["status"] != "confirmed" or verified["identity"]["kind"] != "match":
            raise AssertionError(f"{key}: fresh confirmation failed")
        if verified["provider"] != "candidate" or verified["candidate_sha256"] != EXPECTED_SHA[size]:
            raise AssertionError(f"{key}: wrong provider or candidate")
        if verified["search_seed"] == verified["confirmation_seed"]:
            raise AssertionError(f"{key}: fresh seed collapsed to search")
        if usage == "benchmark":
            if verified["coverage_kind"] != "tune_confirmation":
                raise AssertionError(f"{key}: wrong coverage kind")
            report = verified["confirmation_report"]
            if report["candidate_us"] > 1.10 * min(report["baseline_us"].values()):
                raise AssertionError(f"{key}: fresh run exceeds the 1.10 gate")
        elif verified["coverage_kind"] != "sampled_unseen":
            raise AssertionError(f"{key}: wrong coverage kind")
        cases = verified["confirmation_report"]["correctness"]["cases"]
        if [case["case"] for case in cases] != ["normal", "wide_range", "large", "tiny", "outlier", "ieee"] or not all(case["ok"] for case in cases):
            raise AssertionError(f"{key}: six-case correctness failed")
    manifest = DispatchManifest.from_document(load_json(VALIDATED_MANIFEST))
    if manifest.status != "validated" or manifest.activation_generation != 0:
        raise AssertionError("P2-K manifest was activated")
    if manifest.portfolio_freeze_digest != FREEZE_DIGEST or manifest.validation_plan_digest != VALIDATION_DIGEST:
        raise AssertionError("P2-K manifest belongs to another freeze/plan")
    if {item.instance_key for item in manifest.subjects} != EXPECTED_KEYS:
        raise AssertionError("P2-K manifest coverage drifted")
    stable = {item.instance_key: item.stable_confirmation_digest for item in manifest.subjects}
    for subject in subjects:
        projected = ManifestSubject.from_confirmation(subject)
        if stable[projected.instance_key] != projected.stable_confirmation_digest:
            raise AssertionError(f"{projected.instance_key}: manifest stable identity drifted")


def assert_packed_probe_stays_unadopted() -> None:
    payload = load_json(PACKED_PROBE)
    if payload["schema_id"] != "shapeforge.conv2d-small-packed-half-probe.1":
        raise AssertionError("unknown packed-half probe schema")
    if payload["publication_allowed"] is not False or payload["adopt_candidate"] is not False:
        raise AssertionError("packed-half probe was incorrectly adopted or published")
    if payload["baseline_candidate_sha256"] != EXPECTED_SHA["small"]:
        raise AssertionError("packed-half probe names another baseline")
    if payload["packed_candidate_sha256"] == payload["baseline_candidate_sha256"]:
        raise AssertionError("packed-half probe did not change the source")
    summary = payload["summary"]
    for label in ("old", "packed"):
        if summary[label]["all_mlx_above_one"] is not False:
            raise AssertionError(f"{label}: probe history was rewritten as a stable MLX win")
        if summary[label]["all_pass_latency_gate"] is not True:
            raise AssertionError(f"{label}: probe no longer passes the original latency gate")
    if not summary["packed"]["mlx_speedup_median"] < 1.0:
        raise AssertionError("packed-half probe is being presented as a stable >1.0 result")
