#!/usr/bin/env python3
"""Negative fixture: write an undeclared file into the artifact directory.

Runs only after the hmz Profile is applied.  The trusted collector must reject
the whole artifact set because the file was never declared.  Exit 0 is expected;
the contract violation is detected outside the fence, not by this script.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--artifacts", type=Path, required=True)
    args = parser.parse_args()
    (args.artifacts / "artifact.txt").write_bytes(b"undeclared\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
