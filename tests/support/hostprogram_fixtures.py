"""Test-only fixtures for the restricted C99 / native64 Host Program.

Two fixed candidates live here:

* the planner source ``golden/shared/hostprogram/kda_planner.c`` (candidate-owned C99, both a
  canonical mode and an LPT/range-cut mode);
* the device fixture ``golden/metal/varlen/host_planned_handoff.metal`` (the Dyn-0 piece kernels
  with the device plan stage replaced by the host planner and one empty-sequence device stage).

``kda_bundle_source`` packages them as an ordinary ``msl-bundle`` envelope so the same submission
path used by search/quick can be exercised end to end.  Nothing here runs a GPU or a compiler; the
caller decides which authorized test does that.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry

GOLDEN = Path(__file__).resolve().parent / "golden"
PLANNER_PATH = GOLDEN / "shared" / "hostprogram" / "kda_planner.c"
DEVICE_PATH = GOLDEN / "metal" / "varlen" / "host_planned_handoff.metal"
GEOMETRY_PATH = "host/execution_geometry.json"
HOST_PATH = "host/tiling.c"
DEVICE_ENTRY_PATH = "device/host_planned_handoff.metal"

PLANNER_SOURCE = PLANNER_PATH.read_text(encoding="utf-8")

#: A **capacity-tier** (L2) test candidate: one work item per *resolved* sequence, packed as
#: ``[begin | len << 16, chain]`` in ``work_items`` and the item count in ``item_counts``.  It reads the
#: runtime resolved shape from ``SfHostResolved`` (``resolved->num_seqs``/``resolved->total_tokens``) and
#: checks the batch against the *capacity* in ``SfHostShapes``, which is exactly the L1-vs-L2 difference:
#: the static shape is the maximum, the resolved shape is this call.  It is a test fixture, not a harness
#: planner and not a KDA implementation.
CAPACITY_PLANNER_SOURCE = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)target;
    uint32_t total = resolved->total_tokens;
    uint32_t seqs = resolved->num_seqs;
    /* The capacity is a bound, never the batch itself. */
    if (total < 1u || seqs < 1u) return SF_HOST_INVALID_BOUNDS;
    if (total > shapes->total_tokens || seqs > shapes->num_seqs) return SF_HOST_CAPACITY_EXCEEDED;
    if (seqs + 1u > inputs->cu_seqlens_numel) return SF_HOST_INVALID_BOUNDS;
    uint32_t items = 0u;
    for (uint32_t s = 0u; s < seqs; ++s) {
        int32_t begin = inputs->cu_seqlens[s];
        int32_t end = inputs->cu_seqlens[s + 1u];
        if (begin < 0 || end < begin) return SF_HOST_INVALID_BOUNDS;
        if (2u * s + 1u >= SF_WS_WORK_ITEMS_CAPACITY_ELEMS) return SF_HOST_CAPACITY_EXCEEDED;
        workspaces->work_items[2u * s] = (uint32_t)begin | ((uint32_t)(end - begin) << 16);
        workspaces->work_items[2u * s + 1u] = s;
        ++items;
    }
    if (inputs->cu_seqlens[seqs] != (int32_t)total) return SF_HOST_INVALID_BOUNDS;
    workspaces->work_items_offset = 0u;
    workspaces->work_items_count = 2u * items;
    workspaces->item_counts[0] = (int32_t)items;
    workspaces->item_counts_offset = 0u;
    workspaces->item_counts_count = 1u;
    return SF_HOST_OK;
}
"""

#: The same planner but it *rewrites* a resolved scalar; used to prove the C contract check refuses a
#: forged runtime total instead of silently reporting the value the harness supplied.
CAPACITY_PLANNER_FORGES_RESOLVED = r"""
int32_t host_tiling(const SfHostInputs *inputs, const SfHostShapes *shapes,
                    const SfHostTarget *target, SfHostWorkspaces *workspaces,
                    SfHostResolved *resolved) {
    (void)inputs; (void)shapes; (void)target; (void)workspaces;
    resolved->total_tokens = resolved->total_tokens + 1u;
    resolved->num_seqs = 1u;
    return SF_HOST_OK;
}
"""
DEVICE_SOURCE = DEVICE_PATH.read_text(encoding="utf-8")

#: Workspaces the host planner writes; they are exactly the ones its source references.
HOST_WRITTEN = ("work_items", "item_counts", "handoff_flags")
#: Workspaces the piece stages bind; ``handoff_states`` is device-written only.
PIECE_WORKSPACES = ("work_items", "item_counts", "handoff_flags", "handoff_states")


def host_planned_geometry_document() -> dict:
    """@brief 与 device fixture 的 ``taic-launch`` 一致的几何文档（全部 workspace 非 device-zero）。"""
    stages = [
        {
            "kernel": "varlen_empty",
            "grid": [1, 1, 1],
            "threads": [1, 1, 1],
            "threadgroup_bytes": 0,
            "workspace": [],
            "when": None,
            "derived_scalars": [],
        }
    ]
    for name in ("varlen_piece0", "varlen_piece1", "varlen_piece2"):
        stages.append(
            {
                "kernel": name,
                "grid": ["num_seqs*H*3", 1, 1],
                "threads": [1, 1, 1],
                "threadgroup_bytes": 0,
                "workspace": [{"name": workspace, "zero": False} for workspace in PIECE_WORKSPACES],
                "when": None,
                "derived_scalars": [],
            }
        )
    return {"schema_version": 1, "source": "candidate", "stages": stages}


def geometry_bytes(document: dict | None = None) -> bytes:
    """@brief 规范化几何字节（与 bundle 的固定 geometry entry 一致）。"""
    payload = host_planned_geometry_document() if document is None else dict(document)
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"


def kda_bundle_entries(
    *,
    planner_source: str | None = None,
    device_source: str | None = None,
    geometry: dict | None = None,
) -> tuple[CandidateEntry, ...]:
    """@brief 构造 bundle 的三个文件（device / host/tiling.c / schedule geometry）。"""
    return (
        CandidateEntry(
            path=DEVICE_ENTRY_PATH,
            role="device_kernel",
            media_type="text/x-metal",
            data=(DEVICE_SOURCE if device_source is None else device_source).encode("utf-8"),
        ),
        CandidateEntry(
            path=HOST_PATH,
            role="host_tiling",
            media_type="text/x-c",
            data=(PLANNER_SOURCE if planner_source is None else planner_source).encode("utf-8"),
        ),
        CandidateEntry(
            path=GEOMETRY_PATH,
            role="schedule",
            media_type="application/json",
            data=geometry_bytes(geometry),
        ),
    )


def kda_bundle_source(**overrides) -> str:
    """@brief 把 fixture 打成可提交的 ``msl-bundle`` envelope 文本。"""
    candidate = CandidateArtifact(track_id="msl-bundle", entries=kda_bundle_entries(**overrides))
    return encode_envelope(candidate)


__all__ = [
    "DEVICE_ENTRY_PATH",
    "DEVICE_PATH",
    "DEVICE_SOURCE",
    "GEOMETRY_PATH",
    "HOST_PATH",
    "HOST_WRITTEN",
    "PIECE_WORKSPACES",
    "PLANNER_PATH",
    "PLANNER_SOURCE",
    "geometry_bytes",
    "host_planned_geometry_document",
    "kda_bundle_entries",
    "kda_bundle_source",
]
