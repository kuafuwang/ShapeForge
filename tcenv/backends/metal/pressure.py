"""Source-level accumulator pressure, not compiler register usage.

Metal exposes ``maxTotalThreadsPerThreadgroup`` but no public per-kernel register
count or spill flag.  On this machine that field stayed 1024 for both a 16-slot
and a 32-slot ``simdgroup_float8x8`` accumulator array, so it does not separate
the two.  This module reports a narrow, checkable proxy from the candidate
source: how many ``simdgroup_float8x8`` slots are declared and the largest
literal array.  It is an upper bound on declared storage, not evidence that the
compiler spilled; callers must label it as such.
"""

from __future__ import annotations

import re

_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DECL = re.compile(r"\bsimdgroup_float8x8\s+([^;]+);")
_DECLARATOR = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[\s*([^\]]+?)\s*\])?")


def _literal_size(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if text.endswith(("u", "U", "l", "L")):
        text = text[:-1]
    return int(text) if text.isdigit() else None


def source_pressure(source: str | None) -> dict:
    """Return declared fragment slots and arrays, or ``{}`` when none are visible.

    Array extents must be integer literals.  A symbolic extent is counted as one
    slot and omitted from ``arrays`` so the caller does not pretend to know it.
    """
    if not source:
        return {}
    text = _COMMENT.sub("", source)
    arrays: list[dict] = []
    slots = 0
    for match in _DECL.finditer(text):
        for declarator in match.group(1).split(","):
            item = _DECLARATOR.match(declarator.strip())
            if not item:
                continue
            size = _literal_size(item.group(2))
            count = 1 if size is None else size
            slots += count
            if size is not None:
                arrays.append({"name": item.group(1), "count": size})
    if not slots:
        return {}
    return {
        "fragment_slots": slots,
        "largest_array": max((a["count"] for a in arrays), default=0),
        "arrays": arrays,
        "basis": "declared simdgroup_float8x8 slots; not compiler register or spill data",
    }


def pressure_hint(pressure: dict) -> str:
    """Advisory hint only when the largest declared array exceeds the observed fast point."""
    largest = int(pressure.get("largest_array") or 0)
    if largest <= 16:
        return ""
    return (
        f"source declares a simdgroup_float8x8 array of {largest} slots; "
        "in conv2d 16 slots took 690 us and 32 took 1069 us. This is a source "
        "upper bound, not compiler register usage or a spill report; use "
        "`tcenv profile` spill events if the cause matters."
    )
