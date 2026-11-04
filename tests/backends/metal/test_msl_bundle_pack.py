"""``scripts/msl_bundle_pack.py``：固定目录布局 → 唯一生产者 ``encode_envelope`` → 原子替换。

The packer is the agent's only way to produce a legal ``msl-bundle`` submission, so its contract is
tested as a user-facing tool: the exact role/media-type mapping, the authoring errors (non-zero exit,
nothing written), the byte identity with the encoder, and the atomic replacement of the submission
file the session brief names.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.track.envelope import decode_envelope, encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry
from tests.support.shape_portfolio_metal import (
    DEVICE_A,
    DEVICE_B,
    HOST_TILING_A,
    geometry_bytes,
    geometry_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "msl_bundle_pack.py"
GEOMETRY = geometry_bytes(geometry_document("variant_a"))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )


def _tree(root: Path, *, devices: int = 1) -> Path:
    src = root / "src"
    (src / "device").mkdir(parents=True)
    (src / "host").mkdir(parents=True)
    (src / "device" / "variant_a.metal").write_bytes(DEVICE_A)
    if devices == 2:
        (src / "device" / "variant_b.metal").write_bytes(DEVICE_B)
    (src / "host" / "tiling.json").write_bytes(HOST_TILING_A)
    (src / "host" / "execution_geometry.json").write_bytes(GEOMETRY)
    return src


def _expected(devices: int) -> CandidateArtifact:
    entries = [
        CandidateEntry(path="device/variant_a.metal", role="device_kernel", media_type="text/plain", data=DEVICE_A)
    ]
    if devices == 2:
        entries.append(
            CandidateEntry(path="device/variant_b.metal", role="device_kernel", media_type="text/plain", data=DEVICE_B)
        )
    entries.append(
        CandidateEntry(path="host/tiling.json", role="host_tiling", media_type="application/json", data=HOST_TILING_A)
    )
    entries.append(
        CandidateEntry(
            path="host/execution_geometry.json", role="schedule", media_type="application/json", data=GEOMETRY
        )
    )
    return CandidateArtifact(track_id="msl-bundle", entries=tuple(entries))


@pytest.mark.parametrize("devices", [1, 2])
def test_the_packer_produces_exactly_what_the_encoder_produces(tmp_path, devices):
    src = _tree(tmp_path, devices=devices)
    out = tmp_path / "submission.metal"
    result = _run("--src", str(src), "--out", str(out))
    assert result.returncode == 0, result.stderr
    assert out.read_text(encoding="utf-8") == encode_envelope(_expected(devices))
    # The body travels in plaintext: an agent can read its own submission back.
    assert "variant_a" in out.read_text(encoding="utf-8")


def test_the_packed_envelope_decodes_to_the_same_entries(tmp_path):
    src = _tree(tmp_path, devices=2)
    out = tmp_path / "submission.metal"
    assert _run("--src", str(src), "--out", str(out)).returncode == 0
    candidate = decode_envelope(out.read_text(encoding="utf-8"))
    assert candidate.track_id == "msl-bundle"
    by_path = {entry.path: entry for entry in candidate.entries}
    assert by_path["device/variant_a.metal"].role == "device_kernel"
    assert by_path["device/variant_b.metal"].data == DEVICE_B
    assert by_path["host/tiling.json"].role == "host_tiling"
    assert by_path["host/execution_geometry.json"].role == "schedule"
    assert by_path["host/execution_geometry.json"].media_type == "application/json"


def test_an_existing_submission_is_replaced_atomically(tmp_path):
    src = _tree(tmp_path)
    out = tmp_path / "nested" / "submission.metal"
    assert _run("--src", str(src), "--out", str(out)).returncode == 0
    first = out.read_bytes()
    # A second run with a changed device source must fully replace the old bytes.
    (src / "device" / "variant_a.metal").write_bytes(DEVICE_A + b"// tweak\n")
    assert _run("--src", str(src), "--out", str(out)).returncode == 0
    assert out.read_bytes() != first
    assert decode_envelope(out.read_text(encoding="utf-8")).entries[0].path
    leftovers = [item.name for item in out.parent.iterdir() if item.name != out.name]
    assert leftovers == [], f"the atomic write left temporaries behind: {leftovers}"


@pytest.mark.parametrize(
    "mutate,expected",
    [
        (lambda src: (src / "host" / "tiling.json").unlink(), "host/tiling.json is missing"),
        (lambda src: (src / "host" / "execution_geometry.json").unlink(), "host/execution_geometry.json is missing"),
        (lambda src: shutil.rmtree(src / "device"), "holds no *.metal device source"),
        (lambda src: (src / "host" / "notes.md").write_text("hi"), "no bundle role"),
    ],
)
def test_an_authoring_error_exits_non_zero_and_writes_nothing(tmp_path, mutate, expected):
    src = _tree(tmp_path)
    (src / "device").mkdir(exist_ok=True)
    mutate(src)
    out = tmp_path / "submission.metal"
    result = _run("--src", str(src), "--out", str(out))
    assert result.returncode == 2, result.stdout + result.stderr
    assert expected in result.stderr
    assert not out.exists(), "a refused pack must not touch the submission file"


def test_an_oversized_bundle_is_refused_by_the_encoder(tmp_path):
    src = _tree(tmp_path)
    # One device body over the 1,000,000-byte transport limit.
    (src / "device" / "variant_a.metal").write_bytes(DEVICE_A + b"//" + b"x" * 1_000_000)
    out = tmp_path / "submission.metal"
    result = _run("--src", str(src), "--out", str(out))
    assert result.returncode == 1, result.stdout + result.stderr
    assert "transport limit" in result.stderr
    assert not out.exists()


def test_a_missing_src_directory_is_an_argument_error(tmp_path):
    result = _run("--src", str(tmp_path / "nope"), "--out", str(tmp_path / "out.metal"))
    assert result.returncode == 2
    assert "is not a directory" in result.stderr


def test_the_packer_is_the_only_manifest_producer(tmp_path):
    # No shell pipeline or hand-written base64: the first line is the registered magic and the second
    # is the track, both written by ``encode_envelope``.
    src = _tree(tmp_path)
    out = tmp_path / "submission.metal"
    assert _run("--src", str(src), "--out", str(out)).returncode == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "taic-envelope 1"
    assert lines[1] == "taic-track msl-bundle"
    assert os.access(out, os.R_OK)
