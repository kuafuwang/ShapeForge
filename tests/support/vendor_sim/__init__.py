"""Test-only fake vendor A for the P0 source-only NPU contract.

This package is never imported by a production composition root.  It provides a
deterministic source -> fake ELF/SIMT adapter, a cross-process resource ledger,
and a worker-external controller so cleanup proofs and quarantine can be
exercised without a real NPU, SDK or driver.
"""

from .vendor_a import (
    FakeVendorAController,
    FakeVendorADriver,
    VENDOR_A_GROUP,
    build_execution,
)

__all__ = [
    "FakeVendorAController",
    "FakeVendorADriver",
    "VENDOR_A_GROUP",
    "build_execution",
]
