"""Vendor device-adapter boundary.

Production code depends only on the typed protocols in :mod:`base`; concrete
vendor SDKs never enter the gRPC frontend, local evaluator or supervisor.  The
P0 test-only fake vendor lives under ``tests/support/vendor_sim``.
"""

from .base import (
    CleanupProof,
    DeviceController,
    HealthProof,
    PreparedAttempt,
    VendorDriver,
    VendorFacts,
    VendorOutcome,
)

__all__ = [
    "CleanupProof",
    "DeviceController",
    "HealthProof",
    "PreparedAttempt",
    "VendorDriver",
    "VendorFacts",
    "VendorOutcome",
]
