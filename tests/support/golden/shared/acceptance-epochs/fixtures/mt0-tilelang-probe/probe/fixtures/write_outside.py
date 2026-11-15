#!/usr/bin/env python3
"""Negative fixture: write an artificial outside-attempt canary.

Runs only after the hmz Profile is applied.  Exit 13 means the write was denied
(required); exit 14 means the Profile let the write through (a real failure).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--outside-path", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    try:
        args.outside_path.write_bytes(b"tampered\n")
    except PermissionError:
        return 13
    except OSError:
        return 13
    return 14


if __name__ == "__main__":
    raise SystemExit(main())
