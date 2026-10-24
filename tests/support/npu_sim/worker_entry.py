"""Test-only composition root for the real one-job remote worker."""

from __future__ import annotations

import os
import sys

from tcenv.remote.worker import run_once

from .fake_backend import FakeNpuBackend


if __name__ == "__main__":
    exit_code = os.environ.get("NPU_SIM_WORKER_EXIT")
    if exit_code is not None:
        raise SystemExit(int(exit_code))
    raise SystemExit(run_once(FakeNpuBackend(), sys.stdin.buffer, sys.stdout.buffer))
