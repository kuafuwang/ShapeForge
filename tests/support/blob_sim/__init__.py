"""Test-only blob channel support: fault-injecting IO and a chunked client.

Neither module is imported by a production composition root.  The store, the
gRPC adapter and the contract live in ``tcenv.remote``; this package only makes
their failure modes observable (real I/O errors, crashes, slow consumers) and
gives the tests a chunk-at-a-time gRPC client.
"""

from .client import BlobClient, descriptor_for
from .faults import FakeCrash, FaultIO

__all__ = ["BlobClient", "descriptor_for", "FakeCrash", "FaultIO"]
