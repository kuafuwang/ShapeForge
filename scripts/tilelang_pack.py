#!/usr/bin/env python3
"""Package a TileLang /2 candidate without importing or executing candidate code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tcenv.track import CandidateArtifact, CandidateEntry, encode_envelope, load_track_registry
from tcenv.track import tilelang as contract
from tcenv.track.delivery import candidate_geometry


def pack(src: Path, out: Path) -> str:
    source_path = src / contract.SOURCE_PATH
    geometry_path = src / contract.GEOMETRY_PATH
    if not source_path.is_file() or not geometry_path.is_file():
        raise ValueError(f"both {source_path} and {geometry_path} must be regular files")
    candidate = CandidateArtifact(
        track_id=contract.TILELANG_TRACK_ID,
        entries=(
            CandidateEntry(
                path=contract.SOURCE_PATH,
                role=contract.SOURCE_ROLE,
                media_type=contract.SOURCE_MEDIA_TYPE,
                data=source_path.read_bytes(),
            ),
            CandidateEntry(
                path=contract.GEOMETRY_PATH,
                role=contract.GEOMETRY_ROLE,
                media_type=contract.GEOMETRY_MEDIA_TYPE,
                data=geometry_path.read_bytes(),
            ),
        ),
    )
    spec = load_track_registry().get(contract.TILELANG_TRACK_ID, version=contract.TILELANG_METAL_TRACK_VERSION)
    contract.validate_candidate(candidate, spec)
    candidate_geometry(candidate)  # closed JSON/geometry validation; never executes the DSL source
    text = encode_envelope(candidate)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, out)
    return candidate.candidate_sha256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    digest = pack(args.src, args.out)
    print(json.dumps({"candidate_sha256": digest, "out": str(args.out), "packer_executed_candidate": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
