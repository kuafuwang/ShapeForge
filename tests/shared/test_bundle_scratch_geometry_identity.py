"""Worker geometry identity keeps non-empty scratch and stays byte-compatible when empty.

The pre-fix reconstruction lives in ``tests/support/bundle_scratch_identity.py``.  It drops
``scratch`` on purpose, so this file can show the P2-Q failure without deleting the comparison
in ``tcenv/taskcontract/runner.py``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError
from tcenv.taskcontract.pack import code_import_closure
from tcenv.v2_worker import _lifecycle_from_document

from tests.support.bundle_scratch_identity import (
    historical_empty_round_trip,
    identity_delta,
    kda_contract,
    pre_fix_record,
    production_lifecycle,
    resolve_source,
    round_trip,
)
from tests.support.kda_fixed_bundle import bundle_source
from tests.support.seq_handoff import kda_source, pingpong_geometry, pingpong_source, scratch_sources

ROOT = Path(__file__).resolve().parents[2]
OLD_PACK = "5512395852ea7608a6075e2b6b45dca9bd85f112210af336cf6f8f77f5fd811f"
NEW_PACK = "b5b74185710e5c64cf55ed20cc472f3258e0fe2b1c631bb7f850eeb8dcbb069e"


def _pack_digest() -> tuple[tuple[str, ...], str]:
    files = code_import_closure(ROOT)
    digest = hashlib.sha256(b"".join((ROOT / name).read_bytes() for name in files)).hexdigest()
    return files, digest


def test_pre_fix_worker_drops_scratch_and_the_current_worker_keeps_it():
    """The P2-Q ping-pong and device_zero documents disagree before the fix and agree after it."""
    record = pre_fix_record()
    assert record["publication_allowed"] is False
    assert record["pre_fix"] == {
        "pingpong_documents_equal": False,
        "device_zero_documents_equal": False,
        "empty_documents_equal": True,
        "pingpong_dropped_scratch": True,
        "device_zero_dropped_scratch": True,
    }
    assert record["post_fix"]["pingpong_documents_equal"] is True
    assert record["post_fix"]["device_zero_documents_equal"] is True
    assert record["post_fix"]["pingpong_lifecycle_equal"] is True
    assert record["post_fix"]["device_zero_lifecycle_equal"] is True
    assert record["post_fix"]["empty_documents_equal"] is True
    assert record["post_fix"]["empty_scratch_key_absent"] is True
    assert record["post_fix"]["historical_empty_equal"] is True
    assert record["post_fix"]["historical_scratch_key_absent"] is True


def test_historical_empty_resolution_omits_scratch_and_matches_the_old_bytes():
    """A document that never had ``scratch`` round-trips without gaining the key."""
    report = historical_empty_round_trip()
    assert report["current_equal"] is True
    assert report["legacy_equal"] is True
    assert report["scratch_key_absent"] is True
    assert report["canonical_equal"] is True
    empty = round_trip(kda_source("seq_three_piece"))
    assert empty["current_equal"] is True
    assert empty["lifecycle_equal"] is True
    assert empty["frozen_has_scratch"] is False
    assert empty["legacy_equal"] is True


def test_a_geometry_object_without_scratch_still_reads_as_empty():
    """Old objects keep the ``getattr(..., "scratch", ())`` compatibility used by the fill plan."""
    from tcenv.track.geometry import ResolvedGeometry

    class OldGeometry:
        source = "candidate"

    assert getattr(OldGeometry(), "scratch", ()) == ()
    field = next(item for item in dataclasses.fields(ResolvedGeometry) if item.name == "scratch")
    assert field.default == ()
    source = (ROOT / "tcenv/track/protocol.py").read_text(encoding="utf-8")
    assert 'getattr(geometry, "scratch", ())' in source


def test_an_empty_scratch_list_is_omitted_from_the_rebuilt_document():
    """``[]`` and a missing key are the same empty declaration, so the key stays absent."""
    _resolved, lifecycle = production_lifecycle(kda_source("seq_three_piece"))
    document = json.loads(json.dumps(lifecycle.document()))
    assert "scratch" not in document["geometry"]
    document["geometry"]["scratch"] = []
    document["geometry"]["stages"][0]["scratch"] = []
    rebuilt = _lifecycle_from_document(document)
    assert rebuilt.geometry.document() == lifecycle.geometry.document()
    assert "scratch" not in rebuilt.geometry.document()


def test_a_non_list_scratch_field_is_rejected():
    _resolved, lifecycle = production_lifecycle(kda_source("seq_three_piece"))
    document = json.loads(json.dumps(lifecycle.document()))
    document["geometry"]["scratch"] = {"name": "state_a"}
    with pytest.raises(GeometryError, match="must be a list"):
        _lifecycle_from_document(document)


def test_tampered_frozen_nbytes_stay_tampered_and_fail_the_identity_comparison():
    """Rebuilding must not recompute nbytes, or a forged document would silently match."""
    resolved, lifecycle = production_lifecycle(pingpong_source())
    document = json.loads(json.dumps(lifecycle.document()))
    document["geometry"]["scratch"][0]["nbytes"] += 4
    rebuilt = _lifecycle_from_document(document)
    assert rebuilt.geometry.scratch[0].nbytes == resolved.scratch[0].nbytes + 4
    assert rebuilt.geometry.document() != resolved.document()
    runner = (ROOT / "tcenv/taskcontract/runner.py").read_text(encoding="utf-8")
    assert "lifecycle.geometry.document() != chosen.geometry.document()" in runner
    assert "the lifecycle was registered for a different resolved geometry" in runner


def test_single_read_write_poison_and_a_first_read_are_still_refused():
    with pytest.raises(GeometryError, match="first used as read_write"):
        resolve_source(scratch_sources()["uninitialized_poison"])
    document = pingpong_geometry()
    document["stages"][0]["scratch"][0]["access"] = "read"
    with pytest.raises(GeometryError, match="first used as read"):
        resolve_source(bundle_source(geometry=document))


def test_device_zero_and_host_written_lifecycles_round_trip():
    zero = round_trip(scratch_sources()["device_zero"])
    assert zero["current_equal"] is True and zero["lifecycle_equal"] is True
    assert zero["frozen_scratch"][0]["init_policy"] == "device_zero"
    document = pingpong_geometry()
    for spec in document["scratch"]:
        spec["init_policy"] = "host_written"
    written = round_trip(bundle_source(geometry=document))
    assert written["current_equal"] is True and written["lifecycle_equal"] is True
    assert {item["init_policy"] for item in written["frozen_scratch"]} == {"host_written"}


def test_nonempty_scratch_changes_execution_lifecycle_artifact_native_and_evaluation_identity():
    """Same candidate bytes, two resolved geometries.  Scratch moves every record identity, including evaluation."""
    delta = identity_delta(kda_source("seq_three_piece"), pingpong_source())
    assert delta["empty_resolved_has_scratch"] is False
    assert delta["scratch_resolved_has_scratch"] is True
    assert delta["execution_changed"] is True
    assert delta["lifecycle_changed"] is True
    assert delta["artifact_changed"] is True
    assert delta["native_changed"] is True
    assert delta["empty_execution"] != delta["scratch_execution"]
    assert delta["empty_artifact"] != delta["scratch_artifact"]
    assert delta["empty_native"] != delta["scratch_native"]
    assert delta["evaluation_identity"] == "TaskEvaluationResult.identity"
    assert delta["evaluation_device_verdict"] is False
    assert delta["empty_report_status"] == delta["scratch_report_status"] == "rejected"
    assert delta["evaluation_changed"] is True
    assert delta["empty_evaluation"] != delta["scratch_evaluation"]
    assert len(delta["empty_evaluation"]) == 64
    assert len(delta["scratch_evaluation"]) == 64
    assert delta["empty_candidate_sha256"] == delta["scratch_candidate_sha256"]
    assert delta["empty_task_pack_digest"] == delta["scratch_task_pack_digest"]
    assert "task_pack_digest" in delta["evaluation_unchanged_fields"]
    assert "candidate_sha256" in delta["evaluation_unchanged_fields"]
    changed = set(delta["evaluation_changed_fields"])
    assert {"geometry_identity", "bundle_digest", "artifact_set_digest", "native_digest", "report"} <= changed
    paths = {item["path"] for item in delta["evaluation_differences"]}
    assert "geometry_identity" in paths
    assert "artifact_set_digest" in paths
    assert "native_digest" in paths
    assert any(path.startswith("report.") for path in paths)
    assert "task_pack_digest" not in paths
    geometry_rows = [item for item in delta["evaluation_differences"] if "execution_geometry.json" in item["path"]]
    assert geometry_rows
    assert any(item["path"].endswith("size_bytes") and item["scratch"] > item["empty"] for item in geometry_rows)


def test_kda_handoff_states_remain_an_opspec_workspace():
    """The worker-geometry fix does not move the SEQ state onto scratch."""
    tensor = kda_contract().instance.tensor("handoff_states")
    assert tensor.space == "workspace"
    opspec = json.loads((ROOT / "tasks/kda/opspec.json").read_text(encoding="utf-8"))
    names = [name for name, item in opspec["tensors"].items() if item.get("space") == "workspace"]
    assert "handoff_states" in names
    assert "work_items" in names
    assert "item_counts" in names


def test_pack_digest_moves_with_the_scratch_parser_and_keeps_79_files():
    """``ResolvedScratch.from_document`` is on the packed type.  ``v2_worker.py`` is the call site and is outside."""
    files, digest = _pack_digest()
    from tests.support.live_anchors import load_live_anchors
    anchors = load_live_anchors(ROOT)
    assert list(files) == anchors["pack_closure"]["files"]
    assert len(files) == 79
    assert digest == anchors["pack_closure"]["sha256"]
    p2r = json.loads(
        (ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2r/pack-closure.json").read_text()
    )
    assert p2r["sha256"] == NEW_PACK
    assert NEW_PACK != OLD_PACK
    assert "tcenv/scratch/spec.py" in files
    assert "tcenv/v2_worker.py" not in files
    assert "def from_document" in (ROOT / "tcenv/scratch/spec.py").read_text(encoding="utf-8")
