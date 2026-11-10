#!/usr/bin/env python3
"""Package an ``msl-bundle`` working directory into the one submission text.

    uv run python scripts/msl_bundle_pack.py --src <dir> --out <submission.metal>

An ``msl-bundle`` candidate is several files (device sources, its Host Tiling plan and its
execution geometry).  The search submits exactly one text, so the files travel inside the
registered candidate envelope.  This script is the *only* producer: it builds a
``CandidateArtifact`` and calls ``encode_envelope``, so it can never invent a manifest, a base64
segment or a marker that the decoder would not accept.

``--src`` layout (fixed; anything else is an authoring error, not a candidate verdict)::

    <src>/device/**/*.metal              -> role device_kernel, text/plain   (>= 1, several are legal)
    <src>/host/tiling.json               -> role host_tiling,  application/json
    <src>/host/execution_geometry.json   -> role schedule,     application/json

``--out`` is the file the session brief names as the submission, so the write is atomic: a reader
never sees a half-written envelope, and a failed run leaves the previous submission intact.

Exit status: 0 packed, 2 authoring/argument error (nothing written), 1 the encoder refused the
bundle (size limit, non-UTF-8 entry) -- also nothing written.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # a `python scripts/...` run must import the checkout it lives in
    sys.path.insert(0, str(REPO_ROOT))

from tcenv.contract_errors import TaskContractError  # noqa: E402
from tcenv.track.envelope import EnvelopeError, encode_envelope  # noqa: E402
from tcenv.track.plane import CandidateArtifact, CandidateEntry  # noqa: E402

TRACK_ID = "msl-bundle"
DEVICE_DIR = "device"
HOST_TILING_JSON = "host/tiling.json"
HOST_TILING_C = "host/tiling.c"
HOST_TILING = HOST_TILING_JSON
HOST_GEOMETRY = "host/execution_geometry.json"
DEVICE_MEDIA_TYPE = "text/plain"
JSON_MEDIA_TYPE = "application/json"
C_MEDIA_TYPE = "text/x-c"


class PackError(Exception):
    """An authoring error: the source tree cannot become a bundle (exit status 2)."""


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def collect_entries(src: Path) -> tuple[CandidateEntry, ...]:
    """@brief Map the fixed ``--src`` layout to candidate entries.

    @param src bundle working directory.
    @return entries (device sources first, then the Host Tiling plan, then the geometry).
    @raises PackError the layout is incomplete or carries an unmapped path.
    """
    if not src.is_dir():
        raise PackError(f"--src {src} is not a directory")
    device_dir = src / DEVICE_DIR
    device_files = sorted(p for p in device_dir.rglob("*.metal")) if device_dir.is_dir() else []
    if not device_files:
        raise PackError(f"{DEVICE_DIR}/ holds no *.metal device source; an msl-bundle needs at least one")
    mapped: dict[str, tuple[str, str]] = {}
    entries: list[CandidateEntry] = []
    for path in device_files:
        relative = _relative(path, src)
        mapped[relative] = ("device_kernel", DEVICE_MEDIA_TYPE)
        entries.append(
            CandidateEntry(
                path=relative, role="device_kernel", media_type=DEVICE_MEDIA_TYPE, data=path.read_bytes()
            )
        )
    has_json = (src / HOST_TILING_JSON).is_file()
    has_c = (src / HOST_TILING_C).is_file()
    if has_json and has_c:
        raise PackError(
            f"both {HOST_TILING_JSON} and {HOST_TILING_C} are present; an msl-bundle ships exactly one Host Tiling "
            "representation (declarative JSON or restricted C99 Host Program), never both"
        )
    if not has_json and not has_c:
        raise PackError(
            f"{HOST_TILING_JSON} is missing (and no {HOST_TILING_C} Host Program was found); an msl-bundle ships its "
            "Host Tiling plan and its geometry"
        )
    tiling_rel = HOST_TILING_C if has_c else HOST_TILING_JSON
    tiling_media = C_MEDIA_TYPE if has_c else JSON_MEDIA_TYPE
    mapped[tiling_rel] = ("host_tiling", tiling_media)
    entries.append(
        CandidateEntry(
            path=tiling_rel, role="host_tiling", media_type=tiling_media, data=(src / tiling_rel).read_bytes()
        )
    )
    if not (src / HOST_GEOMETRY).is_file():
        raise PackError(f"{HOST_GEOMETRY} is missing; an msl-bundle ships its Host Tiling plan and its geometry")
    mapped[HOST_GEOMETRY] = ("schedule", JSON_MEDIA_TYPE)
    entries.append(
        CandidateEntry(
            path=HOST_GEOMETRY, role="schedule", media_type=JSON_MEDIA_TYPE, data=(src / HOST_GEOMETRY).read_bytes()
        )
    )
    unmapped = [
        _relative(path, src)
        for path in sorted(src.rglob("*"))
        if path.is_file() and _relative(path, src) not in mapped
    ]
    if unmapped:
        raise PackError(
            f"{src} holds path(s) with no bundle role: {unmapped}; only {DEVICE_DIR}/**/*.metal, "
            f"{HOST_TILING_JSON} (or {HOST_TILING_C}) and {HOST_GEOMETRY} are packed"
        )
    return tuple(entries)


def pack(src: Path) -> str:
    """@brief Build the envelope text for one bundle working directory.

    @param src bundle working directory.
    @return envelope text.
    @raises PackError the layout is wrong.
    @raises EnvelopeError the encoder refused the bundle (size/UTF-8/transport role).
    """
    entries = collect_entries(src)
    return encode_envelope(CandidateArtifact(track_id=TRACK_ID, entries=entries))


def write_atomically(out: Path, text: str) -> None:
    """@brief Replace ``out`` in one step.

    ``--out`` is the submission file the session brief names, and the harness reads it while the
    agent may still be running, so a partial write must never be observable.

    @param out destination path.
    @param text envelope text.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{out.name}.", dir=str(out.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, out)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    """@brief CLI entry point.

    @param argv 参数列表；None 时取 ``sys.argv[1:]``。
    @return 退出码。
    """
    parser = argparse.ArgumentParser(
        prog="msl_bundle_pack",
        description="package device/**/*.metal + host/tiling.json + host/execution_geometry.json into one submission",
    )
    parser.add_argument("--src", required=True, type=Path, help="the bundle working directory")
    parser.add_argument("--out", required=True, type=Path, help="the submission file to replace atomically")
    args = parser.parse_args(argv)
    try:
        text = pack(args.src)
    except PackError as exc:
        print(f"msl_bundle_pack: {exc}", file=sys.stderr)
        return 2
    except EnvelopeError as exc:
        print(f"msl_bundle_pack: the candidate envelope was refused: {exc}", file=sys.stderr)
        return 1
    except TaskContractError as exc:  # a raw plane error: still an authoring problem, never a traceback
        print(f"msl_bundle_pack: {exc}", file=sys.stderr)
        return 2
    try:
        write_atomically(args.out, text)
    except OSError as exc:
        print(f"msl_bundle_pack: cannot write {args.out}: {exc}", file=sys.stderr)
        return 2
    print(f"packed {len(text.encode('utf-8'))} bytes into {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
