"""Shape-portfolio assembly: pack a source tree into a parent envelope and check it.

Importing this package does not import ``tcenv.backends`` or ``taic`` by its own source.
The task-pack closure must not import it.
"""

from __future__ import annotations

from .assembly import (
    COVERAGE_SCHEMA_ID,
    MAX_VARIANTS,
    PARENT_TRANSPORT_ID,
    AssemblyDomain,
    CheckReport,
    CheckRow,
    PortfolioAssemblyError,
    benchmark_sizes,
    check_parent,
    decode_parent,
    domain_for_sizes,
    encode_parent,
    parent_entries_from_tree,
    write_text_atomic,
)

__all__ = [
    "COVERAGE_SCHEMA_ID",
    "MAX_VARIANTS",
    "PARENT_TRANSPORT_ID",
    "AssemblyDomain",
    "CheckReport",
    "CheckRow",
    "PortfolioAssemblyError",
    "benchmark_sizes",
    "check_parent",
    "decode_parent",
    "domain_for_sizes",
    "encode_parent",
    "parent_entries_from_tree",
    "write_text_atomic",
]
