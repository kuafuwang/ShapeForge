#!/usr/bin/env python3
"""Verify the R0 baseline package without importing the project.

Checks, in order:

1. ``manifest.json`` parses and its file tables are internally consistent
   (no path appears twice; every archived file is listed; every listed archive
   file exists).
2. Every archived snapshot file matches the manifest's size and SHA-256.
3. ``vectors/digests.json`` is recomputed from the archived canonical documents
   with a standalone RFC 8785-subset serializer (``json.dumps`` with sorted keys
   and no floats) and compared with the recorded digests.
4. ``vectors/encoding_examples.json`` reproduces the plan §3.1 number encoding:
   shortest round-trip ``repr``, ``float(s) == value``, ``repr(float(s)) == s``, and the
   three distinct-value document pairs (``0.0``/``-0.0``, ``1``/``"1.0"``,
   ``16777216``/``"16777216.0"``) produce different canonical bytes *and* different
   SHA-256 digests — not a literal Python comparison of the fixture labels.
5. With ``--replay``: a pristine ``HEAD`` export plus
   ``drafts/tracked-worktree.patch`` plus ``drafts/untracked-draft.patch``
   reproduces every archived snapshot file byte for byte; and
   ``drafts/adopted-r3-input.patch`` reproduces the adopted R3 subset recorded in
   ``drafts/adopted-r3-input.files.json`` while leaving the discarded files alone.

Usage (from the repository root):

    python3 docs/架构/构建执行与证据语义升级-R0/verify.py
    python3 docs/架构/构建执行与证据语义升级-R0/verify.py --replay
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def fail(message: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"FAIL: {message}")
    raise SystemExit(1)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _reject_unnormalized_values(value) -> None:
    """Refuse values outside the frozen canonical subset before serializing."""
    if isinstance(value, float):
        # This covers finite floats as well as NaN/Inf: §3.1 requires the
        # normalization pass to turn every float into its round-trip string.
        raise ValueError("a bare float reached the canonical serializer")
    if isinstance(value, int) and not isinstance(value, bool):
        if not -(1 << 53) + 1 <= value <= (1 << 53) - 1:
            raise ValueError("canonical integers must be IEEE-754 safe")
        return
    if value is None or isinstance(value, (bool, str)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _reject_unnormalized_values(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str) or not key or not key.isascii():
                raise ValueError("canonical field names must be non-empty ASCII")
            _reject_unnormalized_values(item)
        return
    raise ValueError(f"unsupported canonical value: {type(value).__name__}")


def canonical_bytes(document: dict) -> bytes:
    """RFC 8785 bytes for the identity subset: ASCII keys, no floats."""
    _reject_unnormalized_values(document)
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def load_manifest() -> dict:
    if not MANIFEST.is_file():
        fail("manifest.json is missing")
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"manifest.json is not valid JSON: {exc}")
    if manifest.get("baseline_id") != "r0-build-execution-evidence":
        fail("manifest baseline_id is not the R0 baseline")
    return manifest


def check_snapshot(manifest: dict) -> None:
    """Check the archive against the manifest, and vice versa."""

    entries = manifest["files"]["snapshot"]
    paths = [entry["path"] for entry in entries]
    if len(paths) != len(set(paths)):
        fail("a path appears twice in files.snapshot")

    listed = {entry["archive"] for entry in entries}
    on_disk = {
        path.relative_to(HERE).as_posix()
        for path in (HERE / "draft-files").rglob("*")
        if path.is_file()
    }
    if listed != on_disk:
        missing = sorted(listed - on_disk)
        extra = sorted(on_disk - listed)
        fail(f"archive/manifest mismatch; missing={missing} unlisted={extra}")

    for entry in entries:
        archive = HERE / entry["archive"]
        size = archive.stat().st_size
        if size != entry["size"]:
            fail(f"{entry['archive']}: size {size} != manifest {entry['size']}")
        if sha256(archive) != entry["sha256"]:
            fail(f"{entry['archive']}: sha256 does not match the manifest")


def check_excluded(manifest: dict) -> None:
    """Excluded files are listed with a reason and never also archived."""

    excluded = manifest["files"]["excluded"]
    changed_snapshot = [
        entry for entry in manifest["files"]["snapshot"]
        if entry["status"] in ("modified", "untracked")
    ]
    expected = (
        manifest["worktree"]["untracked_files"]
        + manifest["worktree"]["tracked_changed"]
        - len(changed_snapshot)
    )
    if len(excluded) != expected:
        fail(f"files.excluded has {len(excluded)} entries, expected {expected}")
    for entry in excluded:
        if not entry.get("reason") or not entry.get("workstream"):
            fail(f"excluded entry without reason/workstream: {entry.get('path')}")
    archived = {entry["path"] for entry in manifest["files"]["snapshot"]}
    both = sorted(archived & {entry["path"] for entry in excluded})
    if both:
        fail(f"a path is both archived and excluded: {both}")

    frozen = (HERE / "git" / "untracked-files.txt").read_text(encoding="utf-8").splitlines()
    frozen = [path for path in frozen if path]
    if len(frozen) != manifest["worktree"]["untracked_files"]:
        fail("git/untracked-files.txt does not match worktree.untracked_files")
    classified = {entry["path"] for entry in excluded if entry["status"] == "untracked"}
    classified |= {
        entry["path"] for entry in manifest["files"]["snapshot"] if entry["status"] == "untracked"
    }
    if set(frozen) != classified:
        missing = sorted(set(frozen) - classified)
        extra = sorted(classified - set(frozen))
        fail(f"untracked classification mismatch; missing={missing} extra={extra}")


def check_package(manifest: dict) -> None:
    """Every file in the package is listed, and every listed file matches."""

    listed = {entry["path"] for entry in manifest["files"]["package"]}
    for entry in manifest["files"]["package"]:
        path = HERE / entry["path"]
        if not path.is_file():
            fail(f"package file is missing: {entry['path']}")
        if path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
            fail(f"package file changed since the manifest was frozen: {entry['path']}")

    on_disk = set()
    for path in HERE.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(HERE).as_posix()
        on_disk.add(relative)
    allowed = listed | {entry["archive"] for entry in manifest["files"]["snapshot"]}
    allowed |= {"manifest.json", "manifest.sha256"}
    unexplained = sorted(on_disk - allowed)
    if unexplained:
        fail(f"files in the package that the manifest does not explain: {unexplained}")


def check_vectors() -> None:
    """Recompute every digest and the §3.1 number encoding."""

    digests = json.loads((HERE / "vectors" / "digests.json").read_text(encoding="utf-8"))
    if not digests:
        fail("vectors/digests.json is empty")
    for name, item in sorted(digests.items()):
        document = json.loads((HERE / item["document"]).read_text(encoding="utf-8"))
        recomputed = hashlib.sha256(canonical_bytes(document)).hexdigest()
        if recomputed != item["digest"]:
            fail(f"vector {name}: digest {item['digest']} != recomputed {recomputed}")
        if len(item["digest"]) != 64 or item["digest"] != item["digest"].lower():
            fail(f"vector {name}: digest is not 64 lowercase hex characters")

    examples = json.loads((HERE / "vectors" / "encoding_examples.json").read_text(encoding="utf-8"))
    for item in examples["float_examples"]:
        value = float(item["literal"])
        text = item["canonical"]
        if repr(value) != text:
            fail(f"encoding: repr(float({item['literal']!r})) is {repr(value)!r}, not {text!r}")
        if float(text) != value or repr(float(text)) != text:
            fail(f"encoding: {text!r} does not round-trip to its own repr")
    for item in examples["integer_examples"]:
        if int(item["literal"]) != item["canonical_value"]:
            fail(f"encoding: integer example {item['literal']} is not stable")
    pairs = examples["distinct_canonical_documents"]
    if {item["label"] for item in pairs} != {
        "signed_zero", "int_vs_float_string", "int_vs_integral_float",
    }:
        fail("encoding: the distinct-value document set changed")
    for item in pairs:
        left = canonical_bytes(item["left"])
        right = canonical_bytes(item["right"])
        if left == right:
            fail(f"encoding: {item['label']} collapses to the same canonical bytes {left!r}")
        if hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest():
            fail(f"encoding: {item['label']} collapses to the same digest")
    # Every rejected_literals row is an executable negative, including a
    # finite bare float: normalization must happen before this serializer.
    rejected = examples.get("rejected_literals")
    if not rejected:
        fail("encoding: rejected_literals is missing")
    for item in rejected:
        literal = item["literal"]
        if not isinstance(literal, str):
            fail("encoding: a rejected literal is not text")
        try:
            value = float(literal)
        except ValueError:
            fail(f"encoding: rejected literal {literal!r} is not numeric")
        try:
            canonical_bytes({"v": value})
        except ValueError:
            continue
        fail(f"encoding: the canonical serializer accepted rejected literal {literal!r}")

    # A normalized decimal string remains a legal canonical value.
    canonical_bytes({"v": "1.0"})


def apply_patch(work: Path, patch: Path) -> None:
    """Apply one frozen patch, turning a git failure into a check failure."""

    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn", str(patch)],
        cwd=work, capture_output=True, text=True,
    )
    if result.returncode != 0:
        fail(f"{patch.name} does not apply to a pristine HEAD export: {result.stderr.strip()[:400]}")


def export_head(head: str, work: Path) -> None:
    """Unpack ``git archive HEAD`` into ``work`` and make it a git worktree."""

    archive = subprocess.run(
        ["git", "-C", str(HERE.parents[2]), "archive", head],
        capture_output=True,
    )
    if archive.returncode != 0:
        fail(f"cannot export HEAD {head}: git archive failed")
    subprocess.run(["tar", "-x", "-C", str(work)], input=archive.stdout, check=True)
    subprocess.run(["git", "init", "-q"], cwd=work, check=True)


def check_replay(manifest: dict) -> None:
    """Apply the frozen patches to a pristine HEAD export and compare bytes."""

    head = manifest["head"]["commit"]
    patches = [HERE / "drafts" / "tracked-worktree.patch", HERE / "drafts" / "untracked-draft.patch"]
    for patch in patches:
        if not patch.is_file():
            fail(f"missing patch {patch.name}")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "head"
        work.mkdir()
        export_head(head, work)
        for patch in patches:
            apply_patch(work, patch)
        for entry in manifest["files"]["snapshot"]:
            replayed = work / entry["path"]
            if not replayed.is_file():
                fail(f"replay did not produce {entry['path']}")
            got = sha256(replayed)
            if got != entry["sha256"]:
                fail(f"replay of {entry['path']} hashes {got}, manifest says {entry['sha256']}")

    print("replay: pristine HEAD + frozen patches reproduces every archived file")
    check_adopted_replay(head)


def check_adopted_replay(head: str) -> None:
    """The adopted subset is a separate, reproducible patch (not the evidence patch)."""

    patch = HERE / "drafts" / "adopted-r3-input.patch"
    expected_path = HERE / "drafts" / "adopted-r3-input.files.json"
    if not patch.is_file() or not expected_path.is_file():
        fail("the adopted-r3-input patch or its expected-hash table is missing")
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if expected["base_head"] != head:
        fail("adopted-r3-input.files.json was frozen against another HEAD")

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "head"
        work.mkdir()
        export_head(head, work)
        apply_patch(work, patch)
        for path, item in expected["files"].items():
            data = (work / path).read_bytes()
            if len(data) != item["size"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                fail(f"adopted subset replay of {path} does not match adopted-r3-input.files.json")
        for path, unchanged in expected["unchanged_from_head"].items():
            if not unchanged:
                fail(f"adopted subset is not expected to keep {path}")
            head_blob = subprocess.run(
                ["git", "-C", str(HERE.parents[2]), "show", f"{head}:{path}"],
                check=True, capture_output=True,
            ).stdout
            if (work / path).read_bytes() != head_blob:
                fail(f"adopted subset patch must not touch {path}")
        if (work / "tests/shared/test_single_build_path.py").exists():
            fail("adopted subset patch must not create the discarded structural test")

    print("replay: drafts/adopted-r3-input.patch reproduces the adopted subset (8 files, discarded files untouched)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="also replay the frozen patches")
    args = parser.parse_args()

    manifest = load_manifest()
    check_snapshot(manifest)
    check_excluded(manifest)
    check_package(manifest)
    check_vectors()
    print(
        f"manifest: {len(manifest['files']['snapshot'])} archived files verified; "
        f"{manifest['worktree']['untracked_files']} untracked files classified"
    )
    if args.replay:
        check_replay(manifest)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
