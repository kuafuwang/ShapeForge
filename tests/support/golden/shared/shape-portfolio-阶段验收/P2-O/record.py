#!/usr/bin/env python3
"""Re-record the P2-O REG confirmations, plain-GEMM controls, probe and decision."""

from __future__ import annotations

from tests.backends.metal.gemm_silu_scratch_record import main

if __name__ == "__main__":
    main()
