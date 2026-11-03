"""P1 bundle contract: artifact set, oneof, digest chain, storage and evidence.

The tests here are deliberately structural: a bundle's identity is the
canonical manifest digest, so a rewritten path, a normalized byte, a swapped
transport form or a rewritten attempt must all fail closed rather than quietly
re-identify the artifact.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest

from taic.evaluation_outbox import EvaluationOutbox, bind_wire_digest
from tcenv.evaluation import EvaluationRequest, EvaluationUnavailable
from tcenv.remote import contracts
from tcenv.remote.artifact_bundle import (
    BUNDLE_SCHEMA_ID,
    MAX_BUNDLE_INLINE_ENTRY_BYTES,
    ArtifactBundle,
    BundleEntry,
    BundleError,
    artifact_set_digest,
    bundle_document,
    entry_from_file,
    payload_bundle,
    payload_bytes,
    validate_relative_path,
)
from tests.support.v1_tasks import V1_TASK
from tcenv.remote.job_types import JobError, StateCorrupt
from tests.support.job_engine import (
    NPU_BUNDLE_FEATURES,
    NPU_TASK_FINGERPRINT,
    NPU_TOKEN,
    npu_bundle_engine,
    npu_bundle_get,
    npu_bundle_request,
    npu_bundle_submit,
)
from tests.support.vendor_sim.harness import (
    NPU_HOST_SIGNATURE_KEY,
    GraphGrpcHarness,
)
from tests.support.vendor_sim.vendor_b import (
    FakeVendorBController,
    build_graph_bundle,
    build_graph_execution,
)

GRAPH_ENTRIES = (
    ("graph/main.json", "GRAPH_DEFINITION", "application/json", b'{"nodes":[1,2]}\n'),
    ("graph/params.bin", "PARAMETER_TABLE", "application/octet-stream", b"\x00\x01\x02\xfe\xff"),
)


def _bundle(entries=GRAPH_ENTRIES) -> ArtifactBundle:
    return ArtifactBundle(entries=tuple(
        BundleEntry(
            path=path, role=role, media_type=media_type, size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(), data=data,
        )
        for path, role, media_type, data in entries
    ))


def _wait(engine, request, states, timeout: float = 20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshot = npu_bundle_get(engine, request)
        if snapshot.state in states:
            return snapshot
        time.sleep(0.02)
    raise AssertionError(f"job did not reach {states}; last state {snapshot.state}")


# -- artifact set identity -------------------------------------------------
def test_the_artifact_set_digest_covers_the_path_sorted_manifest_only():
    forward = _bundle()
    reverse = _bundle(tuple(reversed(GRAPH_ENTRIES)))
    assert forward.artifact_set_digest == reverse.artifact_set_digest
    manifest = forward.manifest
    assert [entry["path"] for entry in manifest["entries"]] == ["graph/main.json", "graph/params.bin"]
    # The manifest never contains bytes, transport form or a host path.
    encoded = json.dumps(manifest, sort_keys=True)
    assert "data" not in manifest["entries"][0]
    assert "blob_ref" not in encoded
    assert str(Path.home()) not in encoded


@pytest.mark.parametrize("mutation", ["path", "role", "media_type", "size_bytes", "sha256"])
def test_any_manifest_field_change_changes_the_artifact_set_digest(mutation):
    bundle = _bundle()
    changed = {
        "path": "graph/other.json", "role": "AUXILIARY", "media_type": "text/plain",
        "size_bytes": bundle.entries[0].size_bytes + 1, "sha256": "0" * 64,
    }[mutation]
    documents = [item.manifest() for item in bundle.entries]
    documents[0][mutation] = changed
    assert artifact_set_digest(BUNDLE_SCHEMA_ID, documents) != bundle.artifact_set_digest


def test_exact_bytes_are_never_normalized():
    """A CRLF rewrite or a BOM is a different artifact, not the same one."""

    lf = b"graph { }\n"
    crlf = b"graph { }\r\n"
    bom = b"\xef\xbb\xbfgraph { }\n"
    digests = {
        artifact_set_digest(BUNDLE_SCHEMA_ID, [{
            "path": "g/x", "role": "GRAPH_DEFINITION", "media_type": "text/plain",
            "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
        }])
        for data in (lf, crlf, bom)
    }
    assert len(digests) == 3
    entry = BundleEntry(path="g/x", role="GRAPH_DEFINITION", media_type="text/plain",
                        size_bytes=len(lf), sha256=hashlib.sha256(lf).hexdigest(), data=crlf)
    with pytest.raises(BundleError):
        ArtifactBundle(entries=(entry,))


@pytest.mark.parametrize("path", [
    "/etc/passwd", "graph/../../etc/passwd", "./graph/main.json", "graph/./main.json",
    "graph//main.json", "graph/", "", "graph\\main.json", "graph/\x00main.json",
    "graph/main.json ", " graph/main.json",
])
def test_unsafe_bundle_paths_are_rejected(path):
    with pytest.raises(BundleError):
        validate_relative_path(path)


def test_duplicate_and_case_colliding_paths_are_rejected():
    first = BundleEntry(path="graph/Main.json", role="AUXILIARY", media_type="text/plain",
                        size_bytes=1, sha256=hashlib.sha256(b"a").hexdigest(), data=b"a")
    duplicate = BundleEntry(path="graph/Main.json", role="AUXILIARY", media_type="text/plain",
                            size_bytes=1, sha256=hashlib.sha256(b"b").hexdigest(), data=b"b")
    colliding = BundleEntry(path="graph/main.json", role="AUXILIARY", media_type="text/plain",
                            size_bytes=1, sha256=hashlib.sha256(b"c").hexdigest(), data=b"c")
    for second in (duplicate, colliding):
        with pytest.raises(BundleError, match="duplicated"):
            ArtifactBundle(entries=(first, second))


def test_a_symlink_or_device_node_never_becomes_a_bundle_entry(tmp_path):
    source = tmp_path / "artifacts"
    source.mkdir()
    (source / "main.json").write_bytes(b"{}")
    (source / "link.json").symlink_to(source / "main.json")
    with pytest.raises(BundleError, match="symbolic link"):
        entry_from_file(source, "link.json", role="AUXILIARY", media_type="application/json")
    with pytest.raises(BundleError, match="regular file"):
        entry_from_file("/dev", "null", role="AUXILIARY", media_type="application/octet-stream")
    with pytest.raises(BundleError, match="segments"):
        entry_from_file(source, "../artifacts/main.json", role="AUXILIARY", media_type="application/json")
    # A symbolic-linked *parent* would move the read outside the source tree.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.json").write_bytes(b"{}")
    (source / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="escapes its root|symbolic link"):
        entry_from_file(source, "linked/main.json", role="AUXILIARY", media_type="application/json")


def test_inline_size_limits_and_the_required_blob_form():
    big = b"x" * (MAX_BUNDLE_INLINE_ENTRY_BYTES + 1)
    entry = BundleEntry(path="model/weights.bin", role="MODEL_DEFINITION",
                        media_type="application/octet-stream", size_bytes=len(big),
                        sha256=hashlib.sha256(big).hexdigest(), data=big)
    with pytest.raises(BundleError, match="inline byte limit"):
        ArtifactBundle(entries=(entry,))
    # A large entry must reference the content-addressed channel; a small one
    # must not (the transport form is decided by size, not by the caller).
    small = b"small"
    descriptor = {
        "content_id": contracts.blob_content_id(hashlib.sha256(small).hexdigest()),
        "sha256": hashlib.sha256(small).hexdigest(),
        "size_bytes": len(small), "media_type": "application/octet-stream", "chunk_size": 4,
    }
    with pytest.raises(BundleError, match="must be inlined"):
        ArtifactBundle(entries=(BundleEntry(
            path="p.bin", role="PARAMETER_TABLE", media_type="application/octet-stream",
            size_bytes=len(small), sha256=descriptor["sha256"], blob_ref=descriptor,
        ),))
    # And a blob reference whose identity disagrees with the manifest is refused.
    with pytest.raises(BundleError, match="does not match its manifest identity"):
        ArtifactBundle(entries=(BundleEntry(
            path="p.bin", role="PARAMETER_TABLE", media_type="application/octet-stream",
            size_bytes=len(big), sha256=hashlib.sha256(big).hexdigest(),
            blob_ref=dict(descriptor, size_bytes=len(big)),
        ),))


def test_a_rewritten_entry_list_cannot_keep_the_old_artifact_identity():
    bundle = _bundle()
    document = copy.deepcopy(bundle.to_dict())
    document["entries"][0]["sha256"] = "0" * 64
    with pytest.raises(BundleError, match="artifact_set_digest"):
        bundle_document(document)
    document = copy.deepcopy(bundle.to_dict())
    document["artifact_set_digest"] = "1" * 64
    with pytest.raises(BundleError, match="artifact_set_digest"):
        bundle_document(document)


def test_the_bundle_payload_round_trips_and_refuses_a_swapped_payload():
    bundle = build_graph_bundle()
    payload = payload_bytes(bundle)
    restored = payload_bundle(payload)
    assert restored.to_dict() == bundle.to_dict()
    tampered = payload.replace(b"GRAPH", b"graph")
    with pytest.raises(BundleError):
        payload_bundle(tampered)


def test_materialize_writes_every_inline_entry_under_the_attempt_root(tmp_path):
    bundle = _bundle()
    written = bundle.materialize(tmp_path / "attempt", anchor=tmp_path)
    assert sorted(written) == ["graph/main.json", "graph/params.bin"]
    for path, role, media_type, data in GRAPH_ENTRIES:
        assert written[path].read_bytes() == data


def test_materialize_cannot_be_raced_into_writing_through_a_swapped_parent(tmp_path, monkeypatch):
    """The parent is resolved once, through a descriptor -- not by path name.

    The race is deterministic instead of timing based: the checked directory's
    name is replaced by a symbolic link *after* it was opened and before the
    entry is written, which is exactly the window an ``lstat`` + ``open(path)``
    implementation loses.  With ``openat`` the descriptor still names the
    checked directory, so nothing may appear outside the attempt root.
    """

    from tcenv.remote import artifact_bundle

    root = tmp_path / "attempt"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    real = artifact_bundle._descend_without_symlinks
    swapped = {"done": False}

    def raced(root_fd, segments, path):
        directory_fd = real(root_fd, segments, path)
        if not swapped["done"]:
            swapped["done"] = True
            os.rename(root / "graph", root / "graph-real")
            (root / "graph").symlink_to(outside, target_is_directory=True)
        return directory_fd

    monkeypatch.setattr(artifact_bundle, "_descend_without_symlinks", raced)
    # The first entry goes to the directory that was really opened (it is now
    # named ``graph-real``); the second entry is refused because the swapped name
    # is a symbolic link.
    with pytest.raises(BundleError, match="symbolic link"):
        _bundle().materialize(root, anchor=tmp_path)
    assert swapped["done"] is True
    # The escape the old implementation allowed: a file under the swapped target.
    assert list(outside.iterdir()) == []
    # The bytes landed in the directory that was really checked.
    assert (root / "graph-real" / "main.json").read_bytes() == GRAPH_ENTRIES[0][3]


def test_materialize_handles_cannot_be_talked_into_reading_a_swapped_path(tmp_path, monkeypatch):
    """The consumer gets a verified handle, not a path name to re-resolve.

    One entry makes the write itself succeed (unlike the two-entry race above,
    where the second entry always fails), so the returned value is the only thing
    that could hand the caller the bytes behind the swapped name.
    """

    from tcenv.remote import artifact_bundle

    single = _bundle((GRAPH_ENTRIES[0],))
    root = tmp_path / "attempt"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "main.json").write_bytes(b"attacker bytes")
    root.mkdir()
    real = artifact_bundle._descend_without_symlinks
    swapped = {"done": False}

    def raced(root_fd, segments, path):
        directory_fd = real(root_fd, segments, path)
        if not swapped["done"]:
            swapped["done"] = True
            os.rename(root / "graph", root / "graph-real")
            (root / "graph").symlink_to(outside, target_is_directory=True)
        return directory_fd

    monkeypatch.setattr(artifact_bundle, "_descend_without_symlinks", raced)
    written = single.materialize(root, anchor=tmp_path)
    assert swapped["done"] is True
    handle = written["graph/main.json"]
    # The handle re-walks with O_NOFOLLOW: the swapped name is refused instead
    # of being read.  A bare Path here would return the attacker's bytes.
    with pytest.raises(BundleError, match="symbolic link"):
        handle.read_bytes()
    with pytest.raises(BundleError, match="symbolic link"):
        handle.verify()
    # And the write itself never left the root.
    assert (root / "graph-real" / "main.json").read_bytes() == GRAPH_ENTRIES[0][3]


def test_materialize_handles_verify_the_bytes_they_return(tmp_path):
    """The replacement here is a *plain* directory and a *plain* file.

    No symbolic link is involved, so refusing the name is not enough: a handle
    that re-resolves ``root/path`` and returns whatever it finds would hand back
    the attacker's bytes.  Reading and verifying have to be one operation.
    """

    root = tmp_path / "attempt"
    written = _bundle().materialize(root, anchor=tmp_path)
    handle = written["graph/main.json"]
    original = GRAPH_ENTRIES[0][3]
    assert handle.read_bytes() == original
    # Move the real tree away and put a plain directory with a plain file there.
    os.rename(root, tmp_path / "attempt-moved")
    (root / "graph").mkdir(parents=True)
    (root / "graph" / "main.json").write_bytes(b"attacker-bytes")
    with pytest.raises(BundleError, match="differs from its declared size"):
        handle.read_bytes()
    with pytest.raises(BundleError):
        handle.verify()
    # Same size, different content: the digest check, not just the length.
    (root / "graph" / "main.json").write_bytes(b"X" * len(original))
    with pytest.raises(BundleError, match="no longer matches its digest"):
        handle.read_bytes()
    with pytest.raises(BundleError, match="no longer matches its digest"):
        handle.verify()
    # The bytes that were really written are still available from the moved tree.
    assert (tmp_path / "attempt-moved" / "graph" / "main.json").read_bytes() == original


def test_materialize_handles_read_and_verify_the_exact_bytes(tmp_path):
    written = _bundle().materialize(tmp_path / "attempt", anchor=tmp_path)
    for path, _role, _media_type, data in GRAPH_ENTRIES:
        handle = written[path]
        assert handle.path == path
        assert handle.read_bytes() == data
        handle.verify()
    # The handle refuses to invent bytes for an entry that is gone.
    (tmp_path / "attempt" / "graph" / "params.bin").unlink()
    with pytest.raises(BundleError):
        written["graph/params.bin"].read_bytes()


def test_entry_from_file_reads_through_checked_descriptors(tmp_path, monkeypatch):
    """A swap between the check and the read must never yield outside bytes."""

    from tcenv.remote import artifact_bundle

    source = tmp_path / "artifacts"
    (source / "sub").mkdir(parents=True)
    (source / "sub" / "a.bin").write_bytes(b"trusted")
    (source / "flat.bin").write_bytes(b"trusted-flat")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "a.bin").write_bytes(b"attacker")
    (outside / "flat.bin").write_bytes(b"attacker-flat")

    real = artifact_bundle._descend_without_symlinks
    mode = {"value": "parent"}

    def raced(root_fd, segments, path):
        directory_fd = real(root_fd, segments, path)
        if mode["value"] == "parent" and segments:
            mode["value"] = "parent-done"
            os.rename(source / "sub", source / "sub-real")
            (source / "sub").symlink_to(outside, target_is_directory=True)
        return directory_fd

    monkeypatch.setattr(artifact_bundle, "_descend_without_symlinks", raced)
    entry = entry_from_file(
        source, "sub/a.bin", role="AUXILIARY", media_type="application/octet-stream",
    )
    assert entry.data == b"trusted", entry.data

    # Final-component swap: the name becomes a symlink after the directory was
    # opened, so the descriptor-based read has to refuse it.
    def raced_final(root_fd, segments, path):
        directory_fd = real(root_fd, segments, path)
        if mode["value"] == "parent-done" and not segments:
            mode["value"] = "final-done"
            os.rename(source / "flat.bin", source / "flat-real.bin")
            (source / "flat.bin").symlink_to(outside / "flat.bin")
        return directory_fd

    monkeypatch.setattr(artifact_bundle, "_descend_without_symlinks", raced_final)
    with pytest.raises(BundleError, match="symbolic link"):
        entry_from_file(
            source, "flat.bin", role="AUXILIARY", media_type="application/octet-stream",
        )


def test_alias_collisions_between_an_ancestor_and_a_path_are_refused(tmp_path):
    """One alias key decides full-path and ancestor conflicts alike.

    ``Graph`` and ``graph/child`` are the same directory on a case-insensitive
    filesystem, and composed/decomposed ``café`` are the same name on APFS/HFS+.
    Both have to fail at manifest time, before ``materialize`` writes anything.
    """

    def entry(path: str) -> BundleEntry:
        data = path.encode("utf-8")
        return BundleEntry(path=path, role="AUXILIARY", media_type="text/plain",
                           size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                           data=data)

    with pytest.raises(BundleError, match="parent of another entry"):
        ArtifactBundle(entries=(entry("Graph"), entry("graph/child")))
    # A file whose name is the composed form of a decomposed parent would make
    # the filesystem create a directory where the file already is (EEXIST after
    # the first entry was written).
    composed = "café"
    decomposed = "café/other.bin"
    with pytest.raises(BundleError, match="parent of another entry"):
        ArtifactBundle(entries=(entry(composed), entry(decomposed)))
    # A full-path alias is the same conflict, not a second rule.
    with pytest.raises(BundleError, match="duplicated"):
        ArtifactBundle(entries=(entry("café/main.bin"), entry("café/main.bin")))
    # The rejection happens before any output exists: the target root was never
    # created because no bundle survived construction.
    assert not (tmp_path / "attempt").exists()
    # A legal bundle still keeps its exact (non-normalized) path in the manifest.
    legal = _bundle((("café/main.bin", "AUXILIARY", "text/plain", b"x"),))
    assert legal.manifest["entries"][0]["path"] == "café/main.bin"
    assert legal.entries[0].sha256 == hashlib.sha256(b"x").hexdigest()


def test_materialize_refuses_a_symlinked_parent_instead_of_writing_through_it(tmp_path):
    """A planted ``graph -> elsewhere`` must not become an out-of-root write.

    Checking only the final target is not enough: ``mkdir(parents=True)``
    happily accepts an existing symlinked directory and the ``open`` that
    follows would land outside the attempt scope.  The assertion is on the real
    escape (bytes appearing outside the root), not on a message.
    """

    root = tmp_path / "attempt"
    outside = tmp_path / "outside"
    outside.mkdir()
    root.mkdir()
    (root / "graph").symlink_to(outside, target_is_directory=True)
    bundle = _bundle()
    with pytest.raises(BundleError, match="symbolic link"):
        bundle.materialize(root, anchor=tmp_path)
    assert list(outside.iterdir()) == []


def test_materialize_refuses_a_symlink_at_the_target_name(tmp_path):
    outside_file = tmp_path / "victim.txt"
    outside_file.write_bytes(b"original")
    root = tmp_path / "attempt"
    (root / "graph").mkdir(parents=True)
    (root / "graph" / "main.json").symlink_to(outside_file)
    with pytest.raises(BundleError, match="symbolic link"):
        _bundle().materialize(root, anchor=tmp_path)
    assert outside_file.read_bytes() == b"original"


def test_materialize_refuses_a_symlinked_parent_above_the_attempt_root(tmp_path):
    """A link at ``<state>/build`` must not redirect the write outside state.

    Checking only the final attempt directory is not enough: ``build`` is a
    worker-writable component under the trusted state anchor.  The attempt root
    and every component below it are walked from that anchor, so the planted
    link fails before the first byte is written, and a handle returned by a
    normal materialize refuses to read through it later.
    """

    state = tmp_path / "state"
    state.mkdir()
    outside = tmp_path / "outside"
    (outside / "attempt").mkdir(parents=True)
    state.joinpath("build").symlink_to(outside, target_is_directory=True)
    root = state / "build" / "attempt"

    with pytest.raises(BundleError, match="symbolic link"):
        _bundle().materialize(root, anchor=state)
    assert list((outside / "attempt").iterdir()) == []

    # Normal materialize from the same anchor works; then the parent is swapped
    # for a link and read_bytes must re-check every component, not just the
    # final attempt name.
    # Remove the planted link so the normal write can create the real build dir.
    state.joinpath("build").unlink()
    written = _bundle().materialize(state / "build" / "attempt", anchor=state)
    assert written["graph/main.json"].read_bytes() == GRAPH_ENTRIES[0][3]
    os.rename(state / "build", state / "build-original")
    state.joinpath("build").symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="symbolic link"):
        written["graph/main.json"].read_bytes()
    assert list((outside / "attempt").iterdir()) == []


def test_materialize_refuses_a_symlinked_root_or_a_file_in_the_way(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(outside, target_is_directory=True)
    with pytest.raises(BundleError, match="must not be a symbolic link"):
        _bundle().materialize(linked_root, anchor=tmp_path)
    assert list(outside.iterdir()) == []
    # A regular file where a directory component belongs is refused, not replaced.
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "graph").write_bytes(b"not a directory")
    with pytest.raises(BundleError, match="not a directory"):
        _bundle().materialize(root, anchor=tmp_path)
    assert (root / "graph").read_bytes() == b"not a directory"
    assert sorted(item.name for item in root.iterdir()) == ["graph"]
    # An existing target is never overwritten by a second attempt.
    root = tmp_path / "again"
    _bundle().materialize(root, anchor=tmp_path)
    with pytest.raises(BundleError, match="already exists"):
        _bundle().materialize(root, anchor=tmp_path)


def test_an_entry_cannot_be_the_parent_of_another_entry():
    parent = BundleEntry(path="graph", role="AUXILIARY", media_type="text/plain",
                         size_bytes=1, sha256=hashlib.sha256(b"a").hexdigest(), data=b"a")
    child = BundleEntry(path="graph/main.json", role="AUXILIARY", media_type="text/plain",
                        size_bytes=1, sha256=hashlib.sha256(b"b").hexdigest(), data=b"b")
    with pytest.raises(BundleError, match="parent of another entry"):
        ArtifactBundle(entries=(parent, child))


# -- version 4 request identity -------------------------------------------
def test_the_source_bundle_oneof_is_exactly_one_branch():
    bundle = build_graph_bundle()
    execution = build_graph_execution()
    digest = {"algorithm": contracts.BUNDLE_DIGEST_ALGORITHM, "value": bundle.artifact_set_digest}
    common = dict(
        kind="formal_eval", run_id="r", session_id="s", task=V1_TASK, dtype="fp16",
        size="small", target_id="sim-npu-graph", task_fingerprint="d" * 64, mode="quick",
        seed="3", reps=2, queue_timeout_ms=3000, candidate_timeout_ms=3000, cases=None,
        fast_bench=False, require_contract_launch=False, validate=False,
        measure_serial=False, cache_policy="reuse",
        candidate_digest=digest, execution=execution.to_dict(),
    )
    both = {**common, "artifact": {
        "bundle": bundle.to_dict(), "source": {"size_bytes": 1, "sha256": "0" * 64},
    }}
    with pytest.raises(contracts.ContractError, match="exactly one"):
        contracts.request_digest(both, source_bytes=b"", dialect=contracts.NPU_BUNDLE_DIALECT)
    neither = {**common, "artifact": {}}
    with pytest.raises(contracts.ContractError, match="exactly one"):
        contracts.request_digest(neither, source_bytes=b"", dialect=contracts.NPU_BUNDLE_DIALECT)
    # A bundle branch that is offered the source contract, and a source branch
    # offered the blob contract, are both refused: the contract version decides
    # the admissible form.
    with pytest.raises(contracts.ContractError):
        contracts.request_digest(
            {**common, "artifact": {"bundle": bundle.to_dict()}}, source_bytes=b"",
            dialect=contracts.NPU_BLOB_DIALECT,
        )
    source = b"npu-kernel relu\n"
    source_request = {
        **common,
        "candidate_digest": {"algorithm": "sha256", "value": hashlib.sha256(source).hexdigest()},
        "execution": replace(execution, artifact_kind="source", launch_kind="KERNEL",
                             launch_subtype="SIMT", synchronization="stream-event",
                             task_family="kernel",
                             source_media_type="text/plain; charset=utf-8").to_dict(),
        "artifact": {"source": {"size_bytes": len(source),
                                "sha256": hashlib.sha256(source).hexdigest()}},
    }
    source_digest = contracts.request_digest(
        source_request, source_bytes=source, dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    assert len(source_digest) == 64
    stored = contracts.npu_stored_request_digest(
        {**source_request, "request_id": "req-1"}, dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    assert stored == source_digest
    with pytest.raises(contracts.ContractError):
        contracts.request_digest(
            {**source_request, "blob_ref": {}}, source_bytes=source,
            dialect=contracts.NPU_BUNDLE_DIALECT,
        )


def test_the_execution_spec_states_the_family_and_refuses_a_borrowed_shape():
    execution = build_graph_execution().to_dict()
    assert execution["resources"] == {
        "logical_devices": 1, "required_rank_spans": None, "required_die_spans": None,
    }
    # A graph task cannot borrow stream synchronization or the SIMT subtype.
    broken = dict(execution, synchronization="stream-event")
    with pytest.raises(contracts.ContractError, match="requires"):
        contracts.npu_bundle_execution_document(broken)
    broken = dict(execution, launch_subtype="SIMT")
    with pytest.raises(contracts.ContractError, match="requires"):
        contracts.npu_bundle_execution_document(broken)
    # The cache policy stays a request-level field, never part of the spec.
    with pytest.raises(contracts.ContractError, match="unknown"):
        contracts.npu_bundle_execution_document(dict(execution, cache_policy="refresh"))
    # A physical placement hint has no representation at all.
    with pytest.raises(contracts.ContractError, match="unknown"):
        contracts.npu_bundle_execution_document(dict(execution, bdf="0000:01:00.0"))


def test_an_old_contract_version_cannot_carry_the_version_4_shape():
    request = npu_bundle_request()
    fields = request.digest_fields()
    with pytest.raises(contracts.ContractError, match="unknown request fields"):
        contracts.request_digest(fields, source_bytes=b"", dialect=contracts.NPU_SOURCE_DIALECT)
    with pytest.raises(contracts.ContractError, match="unknown request fields"):
        contracts.request_digest(fields, source_bytes=b"", dialect=contracts.NPU_BLOB_DIALECT)


# -- timing evidence version 2 --------------------------------------------
def _evidence(**overrides):
    bundle = build_graph_bundle()
    document = {
        "schema_id": contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
        "candidate_digest": {"algorithm": contracts.BUNDLE_DIGEST_ALGORITHM,
                             "value": bundle.artifact_set_digest},
        "artifact_set_digest": bundle.artifact_set_digest,
        "device_group": build_graph_execution().device_group.to_dict(),
        "topology": {
            "logical_devices": 1, "declared_devices": ["device-0"],
            "required_rank_spans": None, "required_die_spans": None,
        },
        "spans": [{"device_id": "device-0", "rank": None, "die": None,
                   "clock_domain": "svc-clock", "elapsed_ns": 1_250_000}],
        "timing": {
            "source": "DEVICE_EVENT", "clock_domain": "svc-clock",
            "synchronization_completed": True, "device_elapsed_ns": 1_250_000,
            "host_api_elapsed_ns": 2_500_000, "launches": 1,
        },
        "buffer": None,
        "measurement_kind": "simulated",
        "publication_allowed": False,
        "complete": True,
    }
    document.update(overrides)
    return document


def test_version_two_evidence_separates_host_and_device_time():
    document = contracts.npu_timing_evidence_document_v2(_evidence())
    assert document["timing"]["host_api_elapsed_ns"] == 2_500_000
    assert document["timing"]["device_elapsed_ns"] == 1_250_000
    report = {"status": "ok", "candidate_us": 1250.0}
    contracts.npu_report_timing_check(document, report)
    # A report that claims the host time as the candidate time is refused.
    with pytest.raises(contracts.ContractError, match="device-event elapsed"):
        contracts.npu_report_timing_check(document, {"status": "ok", "candidate_us": 2500.0})


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["spans"].clear(), "missing a declared device span"),
    (lambda d: d["timing"].__setitem__("device_elapsed_ns", 999), "aggregate"),
    (lambda d: d["timing"].__setitem__("clock_domain", "unobserved"), "observed clock domain"),
    (lambda d: d["spans"][0].__setitem__("rank", 0), "must not report a rank"),
    (lambda d: d.update(buffer={"flushed": True, "records_written": 4,
                                "records_delivered": 3, "dropped": 1, "overwritten": 0}),
     "intact buffer"),
    (lambda d: d.update(buffer={"flushed": False, "records_written": 4,
                                "records_delivered": 4, "dropped": 0, "overwritten": 0}),
     "flushed buffer"),
])
def test_incomplete_or_contradictory_evidence_is_never_complete(mutation, match):
    document = _evidence()
    mutation(document)
    with pytest.raises(contracts.ContractError, match=match):
        contracts.npu_timing_evidence_document_v2(document)


def _span(device: str, *, rank=None, die=None, elapsed: int = 1_250_000) -> dict:
    return {"device_id": device, "rank": rank, "die": die,
            "clock_domain": "svc-clock", "elapsed_ns": elapsed}


def test_a_device_with_two_ranks_reports_one_span_per_rank():
    """One device with two ranks is two spans, not a duplicated device.

    The old rule keyed uniqueness on ``device_id`` alone, so a topology with
    ``required_rank_spans=2`` could never be complete: the second rank of the
    same device was refused as a duplicate span.
    """

    from tcenv.remote.job_types import NpuResources

    resources = NpuResources(required_rank_spans=2)
    top = _evidence()["topology"]
    top["required_rank_spans"] = 2
    document = _evidence(
        topology=top, spans=[_span("device-0", rank=0, elapsed=1_250_000),
                             _span("device-0", rank=1, elapsed=1_300_000)],
        timing={"source": "DEVICE_EVENT", "clock_domain": "svc-clock",
                "synchronization_completed": True, "device_elapsed_ns": 1_300_000,
                "host_api_elapsed_ns": 2_500_000, "launches": 2},
    )
    checked = contracts.npu_timing_evidence_document_v2(
        document, expected_resources=resources.to_dict(),
    )
    assert checked["complete"] is True
    assert [(span["device_id"], span["rank"]) for span in checked["spans"]] == [
        ("device-0", 0), ("device-0", 1),
    ]
    # The aggregate is still the slowest span, not the average or the first.
    assert checked["timing"]["device_elapsed_ns"] == 1_300_000


def test_completeness_is_the_cartesian_product_of_the_declared_topology():
    """Every declared ``device x rank x die`` coordinate needs exactly one span."""

    base = _evidence()
    topology = dict(base["topology"], logical_devices=2,
                    declared_devices=["device-0", "device-1"], required_rank_spans=2)
    complete = [
        _span("device-0", rank=0), _span("device-0", rank=1, ),
        _span("device-1", rank=0), _span("device-1", rank=1),
    ]
    timing = {"source": "DEVICE_EVENT", "clock_domain": "svc-clock",
              "synchronization_completed": True, "device_elapsed_ns": 1_250_000,
              "host_api_elapsed_ns": 2_500_000, "launches": 1}
    checked = contracts.npu_timing_evidence_document_v2(
        _evidence(topology=topology, spans=complete, timing=timing),
    )
    assert checked["complete"] is True
    # A missing device is named as a missing device.
    with pytest.raises(contracts.ContractError, match="missing a declared device span: device-1"):
        contracts.npu_timing_evidence_document_v2(_evidence(
            topology=topology, timing=timing,
            spans=[_span("device-0", rank=0), _span("device-0", rank=1)],
        ))
    # A device that kept its rank 0 span but lost rank 1 is a missing rank, and
    # it is missing for *that* device -- not for the group as a whole.
    with pytest.raises(contracts.ContractError, match="missing a declared rank span: device-1/rank 1"):
        contracts.npu_timing_evidence_document_v2(_evidence(
            topology=topology, timing=timing,
            spans=complete[:3],
        ))
    # The same coordinate twice is refused even when the count looks right.
    with pytest.raises(contracts.ContractError, match="coordinate twice"):
        contracts.npu_timing_evidence_document_v2(_evidence(
            topology=topology, timing=timing,
            spans=[complete[0], complete[0], complete[2], complete[3]],
        ))
    # A rank the topology does not declare is refused instead of widening it.
    with pytest.raises(contracts.ContractError, match="does not cover"):
        contracts.npu_timing_evidence_document_v2(_evidence(
            topology=topology, timing=timing,
            spans=complete[:3] + [_span("device-1", rank=2)],
        ))
    # A span for a device the topology never declared is refused.
    with pytest.raises(contracts.ContractError, match="declared topology does not have"):
        contracts.npu_timing_evidence_document_v2(_evidence(
            topology=topology, timing=timing,
            spans=complete + [_span("device-2", rank=0)],
        ))


def test_a_graph_only_vendor_is_not_penalised_for_absent_rank_or_die_spans():
    checked = contracts.npu_timing_evidence_document_v2(_evidence())
    assert checked["complete"] is True
    assert checked["spans"] == [{
        "device_id": "device-0", "rank": None, "die": None,
        "clock_domain": "svc-clock", "elapsed_ns": 1_250_000,
    }]
    assert checked["topology"]["required_rank_spans"] is None
    assert checked["topology"]["required_die_spans"] is None


def test_a_topology_no_evidence_row_could_cover_is_refused_up_front():
    from tcenv.remote.job_types import NpuExecutionSpec, NpuResources

    # The execution spec cannot declare 8 devices x 64 ranks: no bounded row
    # could ever report those 512 coordinates, so the request is refused before
    # a device is touched instead of failing every result afterwards.
    with pytest.raises(contracts.ContractError, match="device x rank x die spans"):
        contracts.device_group_requirements(
            {"logical_devices": 8, "required_rank_spans": 64, "required_die_spans": 2},
        )
    with pytest.raises(contracts.ContractError, match="device x rank x die spans"):
        contracts.npu_bundle_execution_document(dict(
            build_graph_execution().to_dict(),
            resources={"logical_devices": 8, "required_rank_spans": 64, "required_die_spans": 2},
        ))


def test_a_declared_rank_shape_requires_every_rank_span():
    from tcenv.remote.job_types import NpuResources

    resources = NpuResources(required_rank_spans=2)
    document = _evidence()
    document["topology"]["required_rank_spans"] = 2
    document["spans"] = [
        {"device_id": "device-0", "rank": 0, "die": None,
         "clock_domain": "svc-clock", "elapsed_ns": 1_250_000},
    ]
    with pytest.raises(contracts.ContractError, match="missing a declared rank span"):
        contracts.npu_timing_evidence_document_v2(document, expected_resources=resources.to_dict())
    # The declared device count is part of the topology contract too.
    with pytest.raises(contracts.ContractError, match="differs from the declared execution"):
        contracts.npu_timing_evidence_document_v2(
            _evidence(), expected_resources=NpuResources(required_die_spans=2).to_dict(),
        )
    # A diagnostic row is admissible but never publishable and never "ok".
    diagnostic = _evidence(
        complete=False, spans=[],
        timing={"source": "DEVICE_EVENT", "clock_domain": "unobserved",
                "synchronization_completed": False, "device_elapsed_ns": 0,
                "host_api_elapsed_ns": 1_000, "launches": 0},
    )
    document = contracts.npu_timing_evidence_document_v2(diagnostic)
    assert document["complete"] is False and document["publication_allowed"] is False
    with pytest.raises(contracts.ContractError, match="ok report"):
        contracts.npu_timing_evidence_document_v2(diagnostic, report_status="ok")
    with pytest.raises(contracts.ContractError, match="cannot be publishable"):
        contracts.npu_timing_evidence_document_v2(
            _evidence(complete=False, spans=[],
                      timing={"source": "DEVICE_EVENT", "clock_domain": "unobserved",
                              "synchronization_completed": False, "device_elapsed_ns": 0,
                              "host_api_elapsed_ns": 1_000, "launches": 0},
                      publication_allowed=True),
        )


# -- submission, storage and cleanup --------------------------------------
def test_a_bundle_job_persists_verifies_and_cleans_up_its_artifact_bytes(tmp_path):
    root = tmp_path / "run"
    service = root / "graph-service.json"
    controller = FakeVendorBController(service)
    engine = npu_bundle_engine(
        root / "state", service_path=service, controller=controller,
        signing_key=b"s" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_bundle_request("bundle-store-1")
        npu_bundle_submit(engine, request)
        final = _wait(engine, request, {"completed", "failed", "interrupted"})
        assert final.state == "completed", final.failure
        evidence = json.loads(final.result.evidence)
        assert evidence["artifact_set_digest"] == request.npu.bundle.artifact_set_digest
        row = engine.store._conn.execute(
            "SELECT source FROM jobs WHERE request_id=?", (request.request_id,)
        ).fetchone()
        # Terminal cleanup removes the artifact payload with the job.
        assert row["source"] is None
        stored = engine.store._conn.execute(
            "SELECT request_json, request_digest FROM jobs WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
        document = json.loads(stored["request_json"])
        assert "bundle" in document["artifact"]
        assert contracts.npu_stored_request_digest(
            document, dialect=contracts.NPU_BUNDLE_DIALECT,
        ) == stored["request_digest"]
    finally:
        engine.close(wait=5)


def test_a_rewritten_stored_request_or_payload_fails_closed(tmp_path):
    root = tmp_path / "run"
    service = root / "graph-service.json"
    controller = FakeVendorBController(service)
    state = root / "state"
    engine = npu_bundle_engine(
        state, service_path=service, controller=controller,
        signing_key=b"s" * 32, signing_key_id="sim-npu-site/key-1",
    )
    try:
        request = npu_bundle_request("bundle-tamper-1")
        npu_bundle_submit(engine, request)
        _wait(engine, request, {"completed", "failed", "interrupted"})
        # Rewriting the artifact document (a different graph path) changes the
        # recomputed digest, so the durable read refuses the row.
        connection = engine.store._conn
        row = connection.execute(
            "SELECT request_json FROM jobs WHERE request_id=?", (request.request_id,)
        ).fetchone()
        document = json.loads(row["request_json"])
        document["artifact"]["bundle"]["entries"][0]["path"] = "graph/evil.json"
        connection.execute(
            "UPDATE jobs SET request_json=? WHERE request_id=?",
            (json.dumps(document), request.request_id),
        )
        connection.commit()
        with pytest.raises((StateCorrupt, JobError)):
            engine.store._job_by_id(request.request_id)
    finally:
        engine.close(wait=5)

    # Reopening the store re-validates every durable row: a rewritten request
    # document must fail closed instead of loading a different artifact set.
    with pytest.raises((StateCorrupt, JobError)):
        reopened = npu_bundle_engine(
            state, service_path=service, controller=controller,
            signing_key=b"s" * 32, signing_key_id="sim-npu-site/key-1",
        )
        try:
            reopened.store._job_by_id("bundle-tamper-1")
        finally:
            reopened.close(wait=5)


def test_a_tampered_bundle_payload_is_refused_on_reload(tmp_path):
    request = npu_bundle_request("bundle-payload-1")
    document = request.without_source()
    payload = request.artifact_bytes
    from tcenv.remote.job_types import ExternalEvaluationRequest

    restored = ExternalEvaluationRequest.from_stored(document, payload)
    assert restored.request_digest == request.request_digest
    with pytest.raises(JobError, match="payload"):
        ExternalEvaluationRequest.from_stored(document, payload.replace(b"PARAMETER_TABLE", b"AUXILIARY    "))


def test_the_bundle_candidate_digest_binds_the_manifest_not_the_transport():
    from tcenv.remote.job_types import CandidateDigest, NpuSourceRequest

    bundle = build_graph_bundle()
    request = npu_bundle_request("bundle-digest-1", bundle=bundle)
    assert request.candidate_digest == {
        "algorithm": contracts.BUNDLE_DIGEST_ALGORITHM,
        "value": bundle.artifact_set_digest,
    }
    # A candidate digest that does not describe this artifact set is refused
    # before a digest is ever computed.
    other = _bundle()
    with pytest.raises(JobError, match="candidate digest must equal artifact_set_digest"):
        NpuSourceRequest(
            candidate_digest=CandidateDigest(
                contracts.BUNDLE_DIGEST_ALGORITHM, other.artifact_set_digest,
                allowed_algorithms=(contracts.BUNDLE_DIGEST_ALGORITHM,),
            ),
            execution=build_graph_execution(),
            bundle=bundle,
        )
    # The same bytes reach the same identity whether the caller lists the
    # entries forward or backward.
    reversed_bundle = _bundle(tuple(reversed(GRAPH_ENTRIES)))
    assert npu_bundle_request("bundle-digest-2", bundle=reversed_bundle).request_digest == (
        npu_bundle_request("bundle-digest-2", bundle=_bundle()).request_digest
    )


# -- host outbox chain ----------------------------------------------------
def _outbox(root: Path) -> EvaluationOutbox:
    return EvaluationOutbox(
        root, signature_key=NPU_HOST_SIGNATURE_KEY, transport="grpc",
        api_contract_id=contracts.NPU_BUNDLE_API_CONTRACT_ID,
        evidence_contract_id=contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID,
    )


def _host_request(tmp_path: Path, harness: GraphGrpcHarness, request_id: str) -> EvaluationRequest:
    inner = EvaluationRequest(
        kind="formal_eval", request_id=request_id, run_id="npu-run-1",
        session_id="npu-session-1", task=V1_TASK, dtype="fp16", size="small",
        target_id=harness.engine.identity.target_id,
        task_fingerprint=NPU_TASK_FINGERPRINT, source="", mode="quick", seed=3, reps=2,
        queue_timeout_s=3.0, candidate_timeout_s=3.0,
    )
    return bind_wire_digest(inner, harness.factory, npu_contract=True)


def test_the_outbox_binds_the_artifact_set_digest_and_algorithm(tmp_path):
    root = tmp_path / "run"
    with GraphGrpcHarness(tmp_path) as harness:
        request = _host_request(tmp_path, harness, "bundle-outbox-1")
        assert request.wire_candidate_algorithm == contracts.BUNDLE_DIGEST_ALGORITHM
        assert request.wire_candidate_digest == harness.bundle.artifact_set_digest
        outbox = _outbox(root)
        assert outbox.begin(request, epoch=harness.factory.epoch) is None
        result = harness.factory.formal(request)
        outbox.complete(request, result)
        row = json.loads(outbox._state_path(request.request_id).read_text(encoding="utf-8"))
        assert row["schema_version"] == 2
        assert row["candidate_digest"] == {
            "algorithm": contracts.BUNDLE_DIGEST_ALGORITHM,
            "value": harness.bundle.artifact_set_digest,
        }
        replay = _outbox(root).begin(request, epoch=harness.factory.epoch)
        assert replay is not None
        assert replay.receipt["candidate_digest"] == row["candidate_digest"]

        # A rewritten row or a rewritten retained envelope must not replay.
        row["candidate_digest"]["value"] = "4" * 64
        outbox._state_path(request.request_id).write_text(json.dumps(row), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _outbox(root).begin(request, epoch=harness.factory.epoch)


def test_the_outbox_refuses_a_bundle_row_without_the_wire_identity(tmp_path):
    root = tmp_path / "run"
    with GraphGrpcHarness(tmp_path) as harness:
        request = _host_request(tmp_path, harness, "bundle-outbox-2")
        unbound = replace(request, wire_candidate_algorithm="")
        with pytest.raises(EvaluationUnavailable, match="algorithm"):
            _outbox(root).begin(unbound, epoch=harness.factory.epoch)
        unbound = replace(request, wire_candidate_digest="")
        with pytest.raises(EvaluationUnavailable, match="CandidateDigest"):
            _outbox(root).begin(unbound, epoch=harness.factory.epoch)
        # A candidate value the evaluator did not bind is refused at the next
        # wire binding: the host cannot recompute it, so the evaluator is the
        # authority and the durable row must agree with it.
        other = replace(request, wire_candidate_digest="5" * 64)
        with pytest.raises(EvaluationUnavailable, match="differs"):
            bind_wire_digest(other, harness.factory, npu_contract=True)


def test_a_tampered_bundle_receipt_or_evidence_breaks_the_signed_chain(tmp_path):
    root = tmp_path / "run"
    with GraphGrpcHarness(tmp_path) as harness:
        request = _host_request(tmp_path, harness, "bundle-outbox-3")
        outbox = _outbox(root)
        outbox.begin(request, epoch=harness.factory.epoch)
        result = harness.factory.formal(request)
        outbox.complete(request, result)

        payload = json.loads(outbox._result_path(request.request_id).read_text(encoding="utf-8"))
        payload["evidence"]["artifact_set_digest"] = "6" * 64
        outbox._result_path(request.request_id).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _outbox(root).begin(request, epoch=harness.factory.epoch)

        payload["evidence"]["artifact_set_digest"] = harness.bundle.artifact_set_digest
        payload["receipt"]["candidate_digest"]["value"] = "7" * 64
        outbox._result_path(request.request_id).write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(EvaluationUnavailable):
            _outbox(root).begin(request, epoch=harness.factory.epoch)


def test_the_signed_evidence_descriptor_is_the_version_two_identity():
    """A version-2 document must not be labelled with the version-1 descriptor.

    The descriptor is what a consumer reads *before* parsing the bytes, and it
    is part of the signed envelope.  Labelling a version-2 TimingEvidence with
    the P0/P0b schema id (and ``application/json``) makes a version-1 consumer
    accept a document it does not understand -- the old frontend and the old
    client agreed on the wrong value, so neither end noticed.
    """

    bundle = build_graph_bundle()
    group = build_graph_execution().device_group.to_dict()
    candidate = {"algorithm": contracts.BUNDLE_DIGEST_ALGORITHM,
                 "value": bundle.artifact_set_digest}
    evidence = json.dumps(
        contracts.npu_timing_evidence_document_v2(_evidence()), sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    envelope = {
        "provenance": {
            "schema_version": 1, "evaluator_epoch": "e" * 64, "site_id": "sim-npu-site",
            "device_id": "sim-npu-graph-device", "backend": "vendor-b",
            "backend_version": "1", "target_id": "sim-npu-graph",
            "worker_build": "fake-vendor-b-worker-1", "task_pack_digest": "f" * 64,
            "calibration_digest": "c" * 64, "protocol_version": "3",
            "api_contract_id": contracts.NPU_BUNDLE_API_CONTRACT_ID,
            "evidence_contract_id": contracts.NPU_BUNDLE_EVIDENCE_CONTRACT_ID,
            "measurement_kind": "simulated", "publication_allowed": False,
            "device_group": group,
            "artifact_kind": "bundle", "launch_kind": "GRAPH",
            "launch_subtype": "SERVICE", "timing_sources": ["DEVICE_EVENT"],
            "attempt_id": "attempt-1", "request_digest": "0" * 64,
        },
        "receipt": {
            "receipt_id": contracts.attempt_receipt_id(
                request_id="bundle-descriptor-1", attempt_id="attempt-1",
                request_digest="0" * 64, candidate_digest=candidate, device_group=group,
                dialect=contracts.NPU_BUNDLE_DIALECT,
            ),
            "request_id": "bundle-descriptor-1",
            "attempt_id": "attempt-1", "request_digest": "0" * 64,
            "lease_fence": "1", "session_sequence": "1",
            "issued_at": "2026-10-29T00:00:00Z",
            "candidate_digest": candidate,
            "device_group": group,
        },
        "report": {"schema_id": contracts.REPORT_SCHEMA_ID, "media_type": "application/json",
                   "size_bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
        "evidence": {
            "schema_id": contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID,
            "media_type": contracts.NPU_BUNDLE_EVIDENCE_MEDIA_TYPE,
            "size_bytes": len(evidence), "sha256": hashlib.sha256(evidence).hexdigest(),
        },
    }
    contracts.result_signature_document(
        envelope, key_id="sim-npu-site/key-1", dialect=contracts.NPU_BUNDLE_DIALECT,
    )
    for schema_id, media_type in (
        (contracts.NPU_TIMING_EVIDENCE_SCHEMA_ID, "application/json"),
        (contracts.NPU_BUNDLE_TIMING_EVIDENCE_SCHEMA_ID, "application/json"),
        ("tcenv.remote.v1.selfcheck-evidence", contracts.NPU_BUNDLE_EVIDENCE_MEDIA_TYPE),
    ):
        mislabelled = copy.deepcopy(envelope)
        mislabelled["evidence"]["schema_id"] = schema_id
        mislabelled["evidence"]["media_type"] = media_type
        with pytest.raises(contracts.ContractError, match="not the TimingEvidence identity"):
            contracts.result_signature_document(
                mislabelled, key_id="sim-npu-site/key-1", dialect=contracts.NPU_BUNDLE_DIALECT,
            )
        with pytest.raises(contracts.ContractError, match="not the TimingEvidence identity"):
            contracts.sign_result(
                mislabelled, key_id="sim-npu-site/key-1", key=b"k" * 32,
                dialect=contracts.NPU_BUNDLE_DIALECT,
            )


# -- fence and retry ------------------------------------------------------
def test_an_interrupted_bundle_attempt_retries_with_a_new_fence(tmp_path):
    root = tmp_path / "run"
    service = root / "graph-service.json"
    controller = FakeVendorBController(service)
    engine = npu_bundle_engine(
        root / "state", service_path=service, controller=controller,
        signing_key=b"s" * 32, signing_key_id="sim-npu-site/key-1",
        worker_env={"TCENV_VENDOR_B_FAULT": "lost_token"},
    )
    try:
        request = npu_bundle_request("bundle-retry-1", candidate_timeout_ms=4000)
        first = npu_bundle_submit(engine, request)
        attempt = first.current_attempt
        interrupted = _wait(engine, request, {"interrupted", "failed", "completed"})
        assert interrupted.state == "interrupted", interrupted.state
        engine.worker_env["TCENV_VENDOR_B_FAULT"] = ""
        # The old attempt cannot be retried with the old capability id, and the
        # new attempt gets a strictly newer fence.
        from tests.support.job_engine import npu_bundle_capabilities

        capability = npu_bundle_capabilities(engine)
        retried = engine.retry(
            request_id=request.request_id, request_digest=request.request_digest,
            expected=engine.expected_identities, interrupted_attempt_id=attempt.attempt_id,
            capability_id=capability.capability_id, required_features=list(NPU_BUNDLE_FEATURES),
            grant_token=NPU_TOKEN,
        )
        assert retried.current_attempt.attempt_id != attempt.attempt_id
        assert retried.current_attempt.lease_fence > attempt.lease_fence
        with pytest.raises(JobError):
            engine.retry(
                request_id=request.request_id, request_digest=request.request_digest,
                expected=engine.expected_identities, interrupted_attempt_id=attempt.attempt_id,
                capability_id=capability.capability_id,
                required_features=list(NPU_BUNDLE_FEATURES), grant_token=NPU_TOKEN,
            )
    finally:
        engine.close(wait=5)
