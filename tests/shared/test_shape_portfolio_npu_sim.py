"""P4: finite-domain shape portfolio over the existing fake-NPU source and bundle protocols.

The portfolio is the P0 finite ``y = -x`` task.  Dispatch uses the P0 resolver.  Validation and
manifest negatives use the P2 types.  Protocol traffic uses the unchanged fake vendor A source
contract and fake vendor B bundle contract.  The archived report is a replay of that simulation.
It is not NPU-P0, not device correctness, and not a speedup.
"""

from __future__ import annotations

import json

import pytest

from tcenv.remote import contracts
from tests.support.npu_sim.shape_portfolio import (
    BUNDLE_API,
    BUNDLE_EVIDENCE,
    EVIDENCE_PATH,
    SOURCE_API,
    SOURCE_EVIDENCE,
    encode_report,
    simulate,
)

EXPECTED = {
    "tail": ("candidate", "variant-c-tail", True),
    "below": ("fallback", None, False),
    "threshold": ("candidate", "variant-a-bundle", True),
    "above": ("candidate", "variant-a-bundle", True),
    "large": ("candidate", "variant-b-single", True),
}


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    return simulate(tmp_path_factory.mktemp("p4"))


def _by_instance(report):
    rows = {}
    for subject in report["protocol"]["subjects"]:
        for row in subject["dispatches"]:
            rows[row["instance_key"]] = row
    return rows


def test_p4_exact_dispatch_covers_the_finite_domain_and_keeps_fallback_uncredited(report):
    rows = _by_instance(report)
    assert set(rows) == {f"fp16/{size}/{usage}" for size in EXPECTED for usage in ("golden", "benchmark")}
    for key, row in rows.items():
        size = key.split("/")[1]
        provider, variant_key, credits = EXPECTED[size]
        assert row["provider"] == provider
        assert row["variant_key"] == variant_key
        assert row["credits_variant"] is credits
        if provider == "fallback":
            assert row["outcome"] == "fallback"
            assert row["variant_id"] is None
            assert row["fallback_reason"] == "outside_candidate_coverage"
        else:
            assert row["outcome"] == "resolved"
            assert row["fallback_id"] is None
    fallback_subjects = [
        subject for subject in report["protocol"]["subjects"] if subject["provider"] == "fallback"
    ]
    assert len(fallback_subjects) == 1
    assert fallback_subjects[0]["credits_variant"] is False
    assert fallback_subjects[0]["variant_key"] is None


def test_p4_source_and_bundle_receipts_bind_the_selected_projection(report):
    labels = {subject["label"] for subject in report["protocol"]["subjects"]}
    assert labels == {"variant-a-bundle", "variant-b-single", "variant-c-tail", "domain-fallback"}
    assert report["protocol"]["source_api_contract_id"] == SOURCE_API == contracts.NPU_SOURCE_API_CONTRACT_ID
    assert report["protocol"]["source_evidence_contract_id"] == SOURCE_EVIDENCE == contracts.NPU_SOURCE_EVIDENCE_CONTRACT_ID
    assert report["protocol"]["bundle_api_contract_id"] == BUNDLE_API == contracts.NPU_BUNDLE_API_CONTRACT_ID
    assert report["protocol"]["bundle_evidence_contract_id"] == BUNDLE_EVIDENCE == contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID
    seen = set()
    for subject in report["protocol"]["subjects"]:
        source = subject["source"]
        bundle = subject["bundle"]
        assert source["candidate_digest"] == {"algorithm": "sha256", "value": subject["submitted_sha256"]}
        assert bundle["candidate_digest"]["value"] == bundle["artifact_set_digest"]
        assert bundle["candidate_digest"]["algorithm"] == contracts.BUNDLE_DIGEST_ALGORITHM
        assert source["measurement_kind"] == bundle["measurement_kind"] == "simulated"
        assert source["publication_allowed"] is bundle["publication_allowed"] is False
        assert source["timing_numbers_included"] is bundle["timing_numbers_included"] is False
        assert source["fixture_task"] == bundle["fixture_task"] == report["protocol"]["fixture_task"]
        for row in subject["dispatches"]:
            assert row["candidate_sha256"] == subject["candidate_sha256"]
        seen.add(subject["candidate_sha256"])
    assert len(seen) == 4


def test_p4_evidence_stays_simulated_and_drops_timing_speedup_and_publication(report):
    assert report["schema_id"] == "local.shape-portfolio-npu-simulation.1"
    assert report["simulated"] is True
    assert report["publication_allowed"] is False
    assert report["npu_p0"] == {
        "status": "blocked",
        "real_device_available": False,
        "real_toolchain_validated": False,
        "correctness_on_device_proven": False,
        "performance_on_device_proven": False,
        "reason": "offline shape-portfolio protocol simulation cannot resolve NPU-P0 or publish NPU evidence",
    }
    assert "report.speedup" in report["omitted_protocol_fields"]
    assert "evidence.timing" in report["omitted_protocol_fields"]
    encoded = encode_report(report)
    for banned in ("\"speedup\"", "\"candidate_us\"", "\"timing\"", "\"publishable\"", "\"confirmed\""):
        assert banned not in encoded
    assert report["portfolio"]["op_name"] == "portfolio_pointwise"
    assert report["portfolio"]["target_id"] != report["protocol"]["subjects"][0]["source"]["device_group_id"]


def test_p4_bundle_tamper_and_forged_manifest_digest_are_rejected(report):
    negatives = report["negatives"]
    assert negatives["bundle_tamper_rejected"] is True
    assert "not its canonical projection" in negatives["forged_manifest_digest_rejected"]
    assert "must not credit" in negatives["fallback_credit_rejected"]
    assert "not in the frozen portfolio" in negatives["identity_swap_rejected"]


def test_p4_outside_domain_is_not_submitted_and_does_not_fall_back(report):
    outside = report["negatives"]["outside_domain"]
    contradictory = report["negatives"]["contradictory_request"]
    assert outside["outcome"] == "unsupported_contract_capability"
    assert contradictory["outcome"] == "semantic_unresolved"
    assert outside["provider"] is contradictory["provider"] is None
    assert outside["fallback_id"] is contradictory["fallback_id"] is None
    assert outside["protocol_submitted"] is contradictory["protocol_submitted"] is False


def test_p4_candidate_view_hides_unseen_keys_and_fresh_seeds(report):
    portfolio = report["portfolio"]
    view = portfolio["candidate_view"]
    assert view["tune_shards"] == ["fp16/large/benchmark", "fp16/large/golden"]
    assert portfolio["tune_shards"] == view["tune_shards"]
    assert portfolio["unseen_count"] == 8
    assert portfolio["required_unseen_count"] == 8
    assert set(view) == {"schema_id", "portfolio_freeze_digest", "tune_shards"}
    blob = json.dumps(view, sort_keys=True)
    assert "fresh_seed" not in blob
    assert "master" not in blob
    assert "unseen" not in blob
    for size in ("tail", "below", "threshold", "above"):
        assert f"fp16/{size}/" not in blob


def test_p4_document_only_manifest_cannot_be_activated(report):
    manifest = report["manifest"]
    assert manifest["status"] == "validated"
    assert manifest["activation_generation"] == 0
    assert manifest["derived_from_confirmations"] is False
    assert manifest["stable_confirmation_digest_is_fresh_confirmation"] is False
    assert "without confirmed sidecars" in report["negatives"]["activation_rejected"]
    document = manifest["document"]
    assert document["status"] == "validated"
    assert document["activation_generation"] == 0
    assert document["dispatch_manifest_digest"] == manifest["dispatch_manifest_digest"]
    credits = [subject["credits_variant"] for subject in document["subjects"]]
    assert credits.count(False) == 2
    assert credits.count(True) == 8
    assert all(subject["provider"] != "fallback" or subject["variant_id"] is None for subject in document["subjects"])


def test_p4_replay_matches_the_archived_evidence(report):
    assert EVIDENCE_PATH.is_file()
    assert encode_report(report) == EVIDENCE_PATH.read_text(encoding="utf-8")
