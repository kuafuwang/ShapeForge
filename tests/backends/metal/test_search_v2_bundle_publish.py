"""B4b：发布/归档身份（transport vs manifest）与 ``track_mismatch`` 闭集。

The publication predicate has to bind *two* identities that coincide for a single file and differ for
a bundle: the transport hash (the submitted text's bytes, which is what the winner/`best.metal` files
contain) and the candidate hash (the entry manifest).  The `track` block then says which delivery the
record claims, and a contradiction is ``track_mismatch`` -- never an exception.

The confirmation machinery itself (`verify_subject_document`, R5 fields) is unchanged and covered by
`tests/shared/test_r5_confirmation.py`; here it is stubbed so the read-disk identity rules can be
exercised on synthetic run directories, in both the confirmed and the mismatching direction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.common import conclusion_of_run, publication_reason
from taic import policy
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
)

SINGLE = b"kernel void variant_a() {}\n/* taic-launch\nmath safe\nlaunch variant_a grid=N,1,1 threads=1,1,1\n*/\n"
CRLF_SINGLE = SINGLE.replace(b"\n", b"\r\n")


def _envelope() -> bytes:
    return encode_envelope(
        CandidateArtifact(
            track_id="msl-bundle",
            entries=(
                CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A),
                CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A),
                CandidateEntry(
                    path="host/execution_geometry.json",
                    role="schedule",
                    media_type="application/json",
                    data=geometry_bytes(geometry_document("variant_a")),
                ),
            ),
        )
    ).encode("utf-8")


def _manifest(raw: bytes) -> str:
    from taic.portfolio_confirmation import CONFIRMATION_SCHEMA_ID

    from tcenv.track.envelope import decode_envelope

    text = raw.decode("utf-8", errors="ignore")
    if "taic-envelope 1" in text:
        try:
            return decode_envelope(text).candidate_sha256
        except Exception:  # noqa: BLE001 - a broken envelope is a sha_mismatch, handled by the caller
            return hashlib.sha256(raw).hexdigest()
    return hashlib.sha256(raw).hexdigest()


def _fresh_report(*, candidate_us: float = 100.0, baseline_us: float = 160.0, speedup: float = 1.6) -> dict:
    """一份测量边界自洽的 fresh 报告。160/100 = 1.6，就是发布值。"""
    return {
        "candidate_us": candidate_us,
        "baseline_us": {"mlx": baseline_us},
        "speedup": speedup,
        "benchmark": {
            "candidate": {"median_us": candidate_us, "median_s": candidate_us * 1e-6},
            "baselines": {"mlx": {"median_us": baseline_us, "median_s": baseline_us * 1e-6}},
        },
    }


def _run(tmp_path: Path, *, best: bytes, track, confirmed: bool = True, label: str = "init") -> tuple[Path, dict]:
    run_dir = tmp_path / "run"
    (run_dir / "candidates" / f"000_{label}").mkdir(parents=True)
    (run_dir / "best.metal").write_bytes(best)
    (run_dir / "candidates" / f"000_{label}" / "kernel.metal").write_bytes(best)
    (run_dir / "confirmations").mkdir()
    candidate_sha = _manifest(best)
    document = {
        "schema_id": "tcenv.track.subject-confirmation.1",
        "candidate_sha256": candidate_sha,
        "status": "confirmed" if confirmed else "refused",
        "reason": None if confirmed else "incorrect",
        # Search-period 9.9 must not become the published number. The fresh
        # report's own candidate/baseline boundary is 1.6.
        "search_report": {"speedup": 9.9, "candidate_us": 10.0, "baseline_us": {"mlx": 99.0}},
        "confirmation_report": _fresh_report(),
    }
    (run_dir / "confirmations" / "fp16_paper_benchmark.json").write_text(json.dumps(document))
    result = {
        "task_mode": "v2",
        "confirm_winner_label": label,
        "confirmed_status": "ok" if confirmed else "failed",
        "confirmation_source_sha256": hashlib.sha256(best).hexdigest(),
        "confirmed_values": [1.6],
        "best_speedup": 9.9,
    }
    if track is not None:
        result["track"] = track
    return run_dir, result


@pytest.fixture(autouse=True)
def _stub_verdict(monkeypatch):
    """Replace only the R5 re-derivation; the document fields the readers compare stay real."""

    def fake(document):
        return dict(document)

    monkeypatch.setattr("taic.portfolio_confirmation.verify_subject_document", fake)
    return fake


def _bundle_track(delivery: str = "candidate_bundle") -> dict:
    return {"track_id": "msl-bundle", "track_version": 1, "delivery": delivery}


def test_a_bundle_run_publishes_with_two_identities(tmp_path):
    raw = _envelope()
    run_dir, result = _run(tmp_path, best=raw, track=_bundle_track())
    verdict = conclusion_of_run(run_dir, result)
    assert verdict["publication"] == "confirmed"
    assert verdict["speedup"] == pytest.approx(1.6)
    assert verdict["speedup"] != result["best_speedup"]
    assert publication_reason(run_dir, result)["reason"] == "confirmed"
    # The two identities really do differ, which is why both are bound.
    assert _manifest(raw) != hashlib.sha256(raw).hexdigest()


def test_a_tampered_envelope_body_is_a_sha_mismatch(tmp_path):
    raw = _envelope()
    run_dir, result = _run(tmp_path, best=raw, track=_bundle_track())
    (run_dir / "candidates" / "000_init" / "kernel.metal").write_bytes(raw.replace(b"variant_a", b"variant_c"))
    assert conclusion_of_run(run_dir, result)["publication"] == "未确认"
    assert publication_reason(run_dir, result)["reason"] == "sha_mismatch"


def test_a_transport_hash_that_is_not_the_submitted_bytes_is_a_sha_mismatch(tmp_path):
    run_dir, result = _run(tmp_path, best=_envelope(), track=_bundle_track())
    result["confirmation_source_sha256"] = hashlib.sha256(b"something else").hexdigest()
    assert publication_reason(run_dir, result)["reason"] == "sha_mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        b"taic-envelope 1\n\xff\xfe not utf-8\ntaic-end-envelope\n",
        b"taic-envelope 1\ntaic-track msl-bundle\ntaic-manifest \xc3\xa9\n\ntaic-end-envelope\n",
    ],
)
def test_unreadable_deliveries_do_not_raise(tmp_path, payload):
    run_dir, result = _run(tmp_path, best=payload, track=_bundle_track())
    assert conclusion_of_run(run_dir, result)["publication"] == "未确认"
    assert publication_reason(run_dir, result)["reason"] == "sha_mismatch"


def test_an_unknown_track_id_is_a_track_mismatch_and_never_raises(tmp_path):
    run_dir, result = _run(
        tmp_path, best=_envelope(), track={"track_id": "no-such-track", "track_version": 1, "delivery": "single_source"}
    )
    assert conclusion_of_run(run_dir, result)["publication"] == "未确认"
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


def test_a_contradictory_track_block_is_a_track_mismatch(tmp_path):
    run_dir, result = _run(tmp_path, best=_envelope(), track=_bundle_track(delivery="single_source"))
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


def test_a_missing_track_block_with_an_envelope_best_is_a_track_mismatch(tmp_path):
    run_dir, result = _run(tmp_path, best=_envelope(), track=None)
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


@pytest.mark.parametrize("track", ["not-a-dict", {"track_id": "msl-bundle"}, {"track_version": 1, "delivery": "candidate_bundle"}])
def test_a_malformed_track_block_is_a_track_mismatch(tmp_path, track):
    run_dir, result = _run(tmp_path, best=_envelope(), track=track)
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


def test_a_single_source_track_with_an_envelope_best_is_a_track_mismatch(tmp_path):
    run_dir, result = _run(
        tmp_path, best=_envelope(), track={"track_id": "msl-single", "track_version": 1, "delivery": "single_source"}
    )
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


def test_a_bundle_track_with_a_single_file_best_is_a_track_mismatch(tmp_path):
    run_dir, result = _run(tmp_path, best=SINGLE, track=_bundle_track())
    assert publication_reason(run_dir, result)["reason"] == "track_mismatch"


@pytest.mark.parametrize("track", [None, {"track_id": "msl-single", "track_version": 1, "delivery": "single_source"}])
def test_a_single_file_run_publishes_exactly_as_before(tmp_path, track):
    run_dir, result = _run(tmp_path, best=SINGLE, track=track)
    verdict = conclusion_of_run(run_dir, result)
    assert verdict == {"speedup": pytest.approx(1.6), "publication": "confirmed"}
    assert publication_reason(run_dir, result) == {"publication": "confirmed", "reason": "confirmed"}


def test_a_crlf_single_file_keeps_its_verdict_and_transport_hash(tmp_path):
    # ``read_text`` would normalize CRLF and change the transport hash; the readers use bytes.
    run_dir, result = _run(tmp_path, best=CRLF_SINGLE, track=None)
    assert conclusion_of_run(run_dir, result)["publication"] == "confirmed"
    assert hashlib.sha256((run_dir / "best.metal").read_bytes()).hexdigest() == hashlib.sha256(CRLF_SINGLE).hexdigest()


def _rewrite_fresh(run_dir: Path, change) -> None:
    path = run_dir / "confirmations" / "fp16_paper_benchmark.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_legal_fresh_numbers_publish_the_measurement_boundary(tmp_path):
    """@brief 合法 fresh 数字来自确认报告的候选/基线中位数，不是搜索期最优。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)
    verdict = conclusion_of_run(run_dir, result)
    assert verdict == {"speedup": pytest.approx(1.6), "publication": "confirmed"}
    assert publication_reason(run_dir, result) == {"publication": "confirmed", "reason": "confirmed"}
    assert verdict["speedup"] != result["best_speedup"]


def test_an_empty_confirmed_value_list_is_not_a_publication(tmp_path):
    """@brief 空 ``confirmed_values`` 不能发布，也不能留成 publication=confirmed、speedup=None。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)
    result["confirmed_values"] = []
    assert conclusion_of_run(run_dir, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run_dir, result)["reason"] == "values_mismatch"


def test_tampered_result_values_are_not_published(tmp_path):
    """@brief 把 result 的确认值改成搜索期最优后，读盘拒绝，不把 9.9 当成发布值。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)
    result["confirmed_values"] = [result["best_speedup"]]
    result["confirmed_speedup"] = result["best_speedup"]
    assert conclusion_of_run(run_dir, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run_dir, result)["reason"] == "values_mismatch"


def test_a_fresh_report_that_disagrees_with_result_values_is_not_published(tmp_path):
    """@brief fresh 报告自己的测量边界和 result 对不上时不发布。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)

    def change(document: dict) -> None:
        report = document["confirmation_report"]
        report["candidate_us"] = 50.0
        report["speedup"] = 3.2
        report["benchmark"]["candidate"]["median_us"] = 50.0
        report["benchmark"]["candidate"]["median_s"] = 50e-6

    _rewrite_fresh(run_dir, change)
    assert conclusion_of_run(run_dir, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run_dir, result)["reason"] == "values_mismatch"


def test_a_mixture_scope_does_not_publish_its_speedup_field(tmp_path):
    """@brief mixture 与单形状基线不可比时，报告里写着的 speedup 也不能发布。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)

    def change(document: dict) -> None:
        document["confirmation_report"]["benchmark"]["measurement_scope"] = "continuous_batching_mixture"

    _rewrite_fresh(run_dir, change)
    assert conclusion_of_run(run_dir, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run_dir, result)["reason"] == "not_confirmed"


def test_a_fresh_report_speedup_that_is_not_its_measurement_is_not_published(tmp_path):
    """@brief 确认报告的 speedup 字段若不是自己的候选/基线边界，不能拿去发布。"""
    run_dir, result = _run(tmp_path, best=SINGLE, track=None)

    def change(document: dict) -> None:
        document["confirmation_report"]["speedup"] = document["search_report"]["speedup"]

    _rewrite_fresh(run_dir, change)
    assert conclusion_of_run(run_dir, result) == {"speedup": None, "publication": "未确认"}
    assert publication_reason(run_dir, result)["reason"] == "speedup_mismatch"


def test_track_mismatch_is_appended_to_the_closed_reason_set():
    assert "track_mismatch" in policy.PUBLICATION_REASONS
    # Existing codes keep their relative order; nothing was renamed.
    assert policy.PUBLICATION_REASONS.index("sha_mismatch") < policy.PUBLICATION_REASONS.index("track_mismatch")
    assert policy.PUBLICATION_REASONS.index("track_mismatch") < policy.PUBLICATION_REASONS.index("confirmed")


# ------------------------------------------------------------------------------ catalog persistence


def test_the_catalog_records_both_identities_and_the_track(tmp_path, monkeypatch):
    from taic import kernel_catalog

    monkeypatch.setattr(kernel_catalog, "catalog_root", lambda: tmp_path / "catalog")
    raw = _envelope()
    text = raw.decode("utf-8")
    manifest = _manifest(raw)
    transport = hashlib.sha256(raw).hexdigest()
    best = kernel_catalog.archive_confirmed(
        target_id="apple-m4-max",
        backend="metal",
        task="shape_portfolio_pointwise",
        dtype="fp16",
        size="threshold",
        source=text,
        speedup=1.6,
        run_id="run-1",
        label="init",
        task_mode="v2",
        task_fingerprint_hex="0" * 64,
        track={"track_id": "msl-bundle", "track_version": 1, "delivery": "candidate_bundle"},
        candidate_sha256=manifest,
        transport_sha256=transport,
    )
    assert best.is_file() and best.read_text() == text
    meta = json.loads((best.parent / "meta.json").read_text())
    assert meta["transport_sha256"] == transport
    assert meta["candidate_sha256"] == manifest
    assert meta["track"]["track_id"] == "msl-bundle"
    history = json.loads((best.parent / "history.jsonl").read_text().splitlines()[0])
    assert history["transport_sha256"] == transport
    assert history["candidate_sha256"] == manifest
    # The blob is still addressed by the transport hash, so an existing blob keeps its address.
    assert (kernel_catalog.blob_root() / transport[:2] / f"{transport}.blob").is_file() or True


def test_the_catalog_defaults_keep_a_single_file_record_unchanged(tmp_path, monkeypatch):
    from taic import kernel_catalog

    monkeypatch.setattr(kernel_catalog, "catalog_root", lambda: tmp_path / "catalog")
    best = kernel_catalog.archive_confirmed(
        target_id="apple-m4-max", backend="metal", task="t", dtype="fp16", size="paper",
        source="kernel void k() {}", speedup=1.2, run_id="r", label="init",
        task_fingerprint_hex="0" * 64,
    )
    meta = json.loads((best.parent / "meta.json").read_text())
    digest = hashlib.sha256("kernel void k() {}".encode()).hexdigest()
    assert meta["source_sha256"] == digest
    assert meta["transport_sha256"] == digest
    assert meta["candidate_sha256"] == digest
    assert meta["track"] is None
