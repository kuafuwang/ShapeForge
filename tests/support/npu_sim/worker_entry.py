"""Test-only composition root for the real one-job remote worker."""

from __future__ import annotations

import sys

from tcenv.remote.worker import run_once

from .fake_backend import FakeNpuBackend


if __name__ == "__main__":
    raise SystemExit(run_once(FakeNpuBackend(), sys.stdin.buffer, sys.stdout.buffer))
