#!/usr/bin/env python3
"""Re-record P2-K fresh subjects and the validated, not-active manifest."""

from __future__ import annotations

import argparse
import json

from tests.backends.metal.conv2d_portfolio_fresh_record import (
    confirm_portfolio_subjects,
    validated_manifest,
    write_documents,
)
from tests.support.conv2d_portfolio_fresh import FRESH_SUITE, VALIDATED_MANIFEST


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        subjects = json.loads(FRESH_SUITE.read_text(encoding="utf-8"))["subjects"]
        manifest = validated_manifest(subjects)
        VALIDATED_MANIFEST.write_text(json.dumps(manifest.document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        write_documents(confirm_portfolio_subjects())
    print("OK: recorded P2-K fresh suite and validated (not active) manifest")


if __name__ == "__main__":
    main()
