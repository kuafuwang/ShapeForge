"""Fixed, hand-written shape-portfolio P1 slice (real Mac vertical slice; no planner, no model).

One finite elementwise task (``y = -x``) with an explicit ``workspace`` scratch tensor, two
specialised variants with different implementations **and** different geometries, and one
independent generic fallback:

* ``variant-a-bundle`` — candidate Host Tiling (``host/host_tiling_a.json``) fills ``scratch`` with
  ``-1`` for every index, and its device source multiplies ``x`` by that workspace value.  The
  Host Tiling result is load-bearing: if the plan does not run, the kernel reads guard/poison
  bytes and the correctness gate fails.  It is a ``msl-bundle`` because it ships host work and its
  declared geometry, and the test OpSpec declares a workspace.
* ``variant-b-single`` — a self-contained device source with its own ``taic-launch`` block,
  ``grid=N,1,1 threads=32,1,1`` and a bounds check; it becomes ``msl-single``.
* ``fallback`` — an independent generic implementation (one threadgroup of 1024, grid-stride
  loop) that handles any ``N`` without candidate Host Tiling; it becomes ``msl-single`` too.

The finite named instance pool contains every dispatch boundary the guards have:

| size        | N    | why                                                | provider      |
| ----------- | ---- | -------------------------------------------------- | ------------- |
| ``tail``    | 7    | odd tail, below the first boundary, ``rem``-free   | variant-b     |
| ``below``   | 63   | threshold − 1                                      | variant-b     |
| ``threshold`` | 64 | first dispatch boundary (variant-b → variant-a)    | variant-a     |
| ``above``   | 65   | threshold + 1                                      | variant-a     |
| ``hole``    | 100  | candidate coverage hole (no variant guard matches) | fallback      |
| ``boundary`` | 128 | second dispatch boundary (hole → variant-b)        | variant-b     |
| ``large``   | 1024 | the far side of the same boundary                  | variant-b     |

Nothing here runs a planner: the documents are literal candidate output, exactly the input the
portfolio parser reads in production.  The MSL sources only declare kernels, launchers and Host
Tiling plans; the harness still owns every semantic domain, ABI, guard and measurement fact.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tcenv.opspec.v2 import SemanticInstanceV2, instantiate_v2
from tcenv.portfolio.candidate import CandidateEntry
from tcenv.portfolio.domain import guard_digest
from tcenv.taskcontract import TaskContract, load_task_contract
from tcenv.track.plane import CANDIDATE_ROLES

from .v2_tasks import write_task

P1_OP = "shape_portfolio_pointwise"

#: The finite, explicitly named pool: two dispatch boundaries, an odd tail and a coverage hole.
P1_SIZES: dict[str, dict[str, Any]] = {
    "tail": {"N": 7, "usage": ["golden", "benchmark"]},
    "below": {"N": 63, "usage": ["golden", "benchmark"]},
    "threshold": {"N": 64, "usage": ["golden", "benchmark"]},
    "above": {"N": 65, "usage": ["golden", "benchmark"]},
    "hole": {"N": 100, "usage": ["golden", "benchmark"]},
    "boundary": {"N": 128, "usage": ["golden", "benchmark"]},
    "large": {"N": 1024, "usage": ["golden", "benchmark"]},
}

#: Which provider must win for each named size (the guards are declared below).
EXPECTED_PROVIDER: dict[str, str] = {
    "tail": "variant-b-single",
    "below": "variant-b-single",
    "threshold": "variant-a-bundle",
    "above": "variant-a-bundle",
    "hole": "fallback",
    "boundary": "variant-b-single",
    "large": "variant-b-single",
}

#: Candidate Host Tiling: scratch[i] = -1 for every i in [0, N), as ``fp16``.
HOST_TILING_A = (
    json.dumps(
        {
            "schema_id": "tcenv.metal.host-tiling.1",
            "schema_version": 1,
            "writes": [
                {
                    "workspace": "scratch",
                    "kind": "linear",
                    "start": -1,
                    "step": 0,
                    "count": "N",
                    "stride": 1,
                    "offset": 0,
                    "dtype": "fp32",
                }
            ],
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    + b"\n"
)

#: variant A: bundled, reads the Host Tiling workspace, one threadgroup of one thread per index.
DEVICE_A = b"""\
#include <metal_stdlib>
using namespace metal;
kernel void variant_a(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      device float* scratch [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
  y[i] = half(scratch[i] * float(x[i]));
}
/* taic-launch
math safe
workspace scratch N*4
launch variant_a grid=N,1,1 threads=1,1,1 workspace=scratch
*/
"""

#: variant B: self-contained, 32 threads per group, explicit bounds check.
DEVICE_B = b"""\
#include <metal_stdlib>
using namespace metal;
kernel void variant_b(device const half* x [[buffer(0)]],
                      device half* y [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
  if (i < N) {
    y[i] = half(-float(x[i]));
  }
}
/* taic-launch
math safe
launch variant_b grid=N,1,1 threads=32,1,1
*/
"""

#: The independent generic fallback: one threadgroup, grid-stride loop, any N, no Host Tiling.
DEVICE_GENERIC = b"""\
#include <metal_stdlib>
using namespace metal;
kernel void generic(device const half* x [[buffer(0)]],
                    device half* y [[buffer(1)]],
                    uint gid [[thread_position_in_grid]]) {
  for (uint i = gid; i < N; i += 1024) {
    y[i] = half(-float(x[i]));
  }
}
/* taic-launch
math safe
launch generic grid=1,1,1 threads=1024,1,1
*/
"""


def opspec(op_name: str = P1_OP, *, workspace: bool = True) -> dict[str, Any]:
    """@brief 固定有限 P1 题（1-D ``y=-x``，带可选 workspace scratch）。

    @param op_name 算子名（等于目录名）。
    @param workspace 是否声明 ``space == "workspace"`` 的 scratch 张量。
    @return v2 OpSpec 文档。
    """
    document: dict[str, Any] = {
        "spec_version": 2,
        "op_name": op_name,
        "category": "elementwise",
        "formula": "y = -x",
        "type_vars": {"T": ["fp16"]},
        "symbols": {"N": {"min": 1}},
        "tensors": {
            "x": {
                "role": "input",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "read",
            },
            "y": {
                "role": "output",
                "dims": ["N"],
                "dtype": "T",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "global",
                "access": "write",
            },
            "scratch": {
                "role": "output",
                "dims": ["N"],
                # A 4-byte element keeps the workspace span a multiple of 4 for every odd ``N``
                # (the launch grammar requires it); the candidate owns the value written here.
                "dtype": "fp32",
                "layout": "contiguous",
                "strides": ["1"],
                "space": "workspace",
                "access": "write",
            },
        },
        "numerics": {
            "acc_dtype": "fp32",
            "rounding": "single_round_at_output",
            "tolerances": {"fp16": {"rtol": 0.002, "atol_rel": 0.001}},
            "exact": False,
        },
        "attrs": {},
        "cases": {"names": ["normal"]},
        "sizes": copy.deepcopy(P1_SIZES),
        "spec_params": ["N"],
        "work_accounting": {
            "flops": {"terms": [{"coefficient": 1, "symbols": ["N"]}]},
            "min_bytes": {"terms": [{"tensor": "x"}, {"tensor": "y"}]},
        },
        "discovery": {"phase": 1, "default_instance": {"dtype": "fp16", "size": "threshold"}},
    }
    if not workspace:
        del document["tensors"]["scratch"]
    return document


def write_p1_task(root: Path, *, op_name: str = P1_OP, workspace: bool = True) -> Path:
    """@brief 写一个标准的有限 P1 测试题目录。

    @param root 父目录。
    @param op_name 算子名。
    @param workspace 是否声明 workspace 张量。
    @return 任务目录。
    """
    from .v2_tasks import SEM_REFERENCE

    return write_task(root, op_name=op_name, opspec=opspec(op_name, workspace=workspace), reference=SEM_REFERENCE)


def load_p1_task(root: Path, *, op_name: str = P1_OP, workspace: bool = True) -> TaskContract:
    """@brief 装配测试题合同。

    @param root 父目录。
    @param op_name 算子名。
    @param workspace 是否声明 workspace 张量。
    @return 任务合同。
    """
    return load_task_contract(write_p1_task(root, op_name=op_name, workspace=workspace))


def p1_instances(contract: TaskContract) -> tuple[SemanticInstanceV2, ...]:
    """@brief 该合同的全部显式命名实例（按 ``sizes`` 声明顺序）。

    @param contract 任务合同。
    @return 语义实例元组。
    """
    return tuple(instantiate_v2(contract.spec, contract.dtype, size) for size in P1_SIZES)


def geometry_document(
    kernel: str,
    *,
    threads: int = 1,
    grid: Any = None,
    workspace_zero: bool = False,
    workspace: bool = True,
    threadgroup_bytes: int = 0,
) -> dict[str, Any]:
    """@brief 一个候选几何文档（variant A 的 ``geometry_entry`` 内容）。

    @param kernel 内核名。
    @param threads 每组线程数。
    @param grid 首个 grid 分量；缺省 ``N``（一个线程组处理一个元素）。
    @param workspace_zero 是否声明 workspace 在计时窗内清零。
    @param workspace 是否声明 scratch workspace。
    @param threadgroup_bytes 线程组内存。
    @return 几何文档。
    """
    stage: dict[str, Any] = {
        "kernel": kernel,
        "grid": [grid if grid is not None else "N", 1, 1],
        "threads": [threads, 1, 1],
        "threadgroup_bytes": threadgroup_bytes,
        "workspace": [{"name": "scratch", "zero": bool(workspace_zero)}] if workspace else [],
        "when": None,
        "derived_scalars": [],
    }
    return {"schema_version": 1, "source": "candidate", "stages": [stage]}


def geometry_bytes(document: Mapping[str, Any]) -> bytes:
    """@brief 几何文档的规范 JSON 字节。

    @param document 文档。
    @return UTF-8 JSON 字节。
    """
    return json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def _bucket(key: str, expr: Any) -> dict[str, Any]:
    """@brief 一个 bucket + guard 文档。

    @param key bucket 名。
    @param expr guard 表达式。
    @return bucket 文档。
    """
    return {"bucket_key": key, "guard_digest": guard_digest(expr, where=key)}


#: First boundary at N=64: variant-b hands over to variant-a; the window closes at N=96.
GUARD_A: Any = ["and", ["ge", "N", 64], ["lt", "N", 96]]
#: The single-source variant keeps the tail/below side and the far side from the second boundary.
GUARD_B: Any = ["or", ["lt", "N", 64], ["ge", "N", 128]]


def _variant(key: str, bucket: str, expr: Any, entries: Iterable[str], geometry: str | None) -> dict[str, Any]:
    """@brief 一个 variant 文档。

    @param key variant 诊断名。
    @param bucket 所属 bucket。
    @param expr guard 表达式。
    @param entries 自己的 entries。
    @param geometry 几何声明路径或 None。
    @return variant 文档。
    """
    return {
        "variant_key": key,
        "bucket_key": bucket,
        "guard": {"expr": expr, "digest": guard_digest(expr, where=key)},
        "entries": list(entries),
        "geometry_entry": geometry,
    }


def portfolio_document(domain_plan_digest: str, *, planner: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """@brief P1 envelope：variant A ``msl-bundle`` / variant B 与 generic fallback ``msl-single``。

    @param domain_plan_digest 绑定的语义域摘要。
    @param planner planner 身份覆盖。
    @return envelope 文档。
    """
    return {
        "schema_id": "tcenv.candidate-portfolio.1",
        "domain_plan_digest": domain_plan_digest,
        "planner": dict(planner or {"id": "candidate-owned-planner", "version": 1}),
        "coverage_plan": {
            "buckets": [
                _bucket("threshold_window", GUARD_A),
                _bucket("tail_and_far_side", GUARD_B),
            ]
        },
        "variants": [
            _variant(
                "variant-a-bundle",
                "threshold_window",
                GUARD_A,
                ["device/variant_a.metal", "host/host_tiling_a.json"],
                "host/geometry_a.json",
            ),
            _variant("variant-b-single", "tail_and_far_side", GUARD_B, ["device/variant_b.metal"], None),
        ],
        "fallback": {
            "kind": "registered_implementation",
            "implementation_id": "portfolio.p1.generic",
            "implementation_version": 1,
            "entries": ["device/generic.metal"],
            "geometry_entry": None,
            "reasons": ["outside_candidate_coverage", "resource_unavailable"],
        },
    }


def bundle_fallback_document(domain_plan_digest: str) -> dict[str, Any]:
    """@brief 一个走 ``msl-bundle`` 的 fallback（自己有 Host Tiling + 几何），用于资源负例。

    The fallback never runs in the positive slice (it is projected as ``msl-single`` there); this
    variant exists so "the fallback itself is over the target limits" can be constructed with a
    declared geometry, exactly like the P0 dispatch contract allows.

    @param domain_plan_digest 绑定的语义域摘要。
    @return envelope 文档。
    """
    document = portfolio_document(domain_plan_digest)
    document["fallback"] = {
        "kind": "registered_implementation",
        "implementation_id": "portfolio.p1.generic.bundle",
        "implementation_version": 2,
        "entries": ["device/generic.metal", "host/host_tiling_a.json"],
        "geometry_entry": "host/geometry_generic.json",
        "reasons": ["outside_candidate_coverage", "resource_unavailable"],
    }
    return document


def source_entries() -> dict[str, tuple[str, str, bytes]]:
    """@brief 父 bundle 的全部可用源（路径 → role/media_type/bytes）。

    @return 路径到三元组的映射。
    """
    return {
        "device/variant_a.metal": ("device_kernel", "text/plain", DEVICE_A),
        "device/variant_b.metal": ("device_kernel", "text/plain", DEVICE_B),
        "device/generic.metal": ("device_kernel", "text/plain", DEVICE_GENERIC),
        "host/host_tiling_a.json": ("host_tiling", "application/json", HOST_TILING_A),
        "host/geometry_a.json": ("host_tiling", "application/json", geometry_bytes(geometry_document("variant_a"))),
        "host/geometry_generic.json": (
            "host_tiling",
            "application/json",
            geometry_bytes(geometry_document("generic", threads=1024)),
        ),
    }


def manifest_entry(document: Mapping[str, Any]) -> CandidateEntry:
    """@brief 把 envelope 文档包成固定的父 entry。

    @param document envelope 文档。
    @return ``coverage_plan/portfolio.json`` entry。
    """
    data = json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    # ``launch`` is the parent envelope role: a registered candidate-role name that no TrackSpec
    # accepts for a source, so the parent can never pass CandidateArtifact.validate_for.
    return CandidateEntry(path="coverage_plan/portfolio.json", role="launch", media_type="application/json", data=data)


#: A thread count above ``max_total_threads`` for the target's ABI limits.
OVER_LIMIT_THREADS = 2048


def over_limit_source(threads: int = OVER_LIMIT_THREADS) -> bytes:
    """@brief variant B 的源，但 launch 线程数超过目标上限（源内几何超限负例）。

    @param threads 超限线程数。
    @return MSL 源字节。
    """
    source = DEVICE_B.replace(b"threads=32,1,1", f"threads={threads},1,1".encode())
    if source == DEVICE_B:
        raise AssertionError("the fixture's variant B launch block changed; the over-limit source cannot be derived")
    return source


def over_limit_entries(document: Mapping[str, Any]) -> tuple[CandidateEntry, ...]:
    """@brief 用超限 variant B 源构造父 entries（其余与固定切片相同）。

    @param document envelope 文档。
    @return 父 entries。
    """
    return parent_entries(
        document,
        extra_sources={"device/variant_b.metal": ("device_kernel", "text/plain", over_limit_source())},
    )


def parent_entries(
    document: Mapping[str, Any],
    *,
    extra_sources: Mapping[str, tuple[str, str, bytes]] | None = None,
    drop: Iterable[str] = (),
    role_overrides: Mapping[str, str] | None = None,
) -> tuple[CandidateEntry, ...]:
    """@brief 由 envelope + 固定源构造父 bundle entries。

    @param document envelope 文档。
    @param extra_sources 额外路径 → (role, media_type, bytes)，用于负例替换某个源/几何。
    @param drop 不写入的源路径（即使 document 引用了它）。
    @param role_overrides 覆盖某些源的角色。
    @return 父 entries（manifest 在最后）。
    """
    sources = source_entries()
    sources.update(dict(extra_sources or {}))
    referenced: set[str] = set()
    for variant in document["variants"]:
        referenced.update(str(item) for item in variant["entries"])
        if variant.get("geometry_entry") is not None:
            referenced.add(str(variant["geometry_entry"]))
    fallback = document.get("fallback")
    if fallback is not None:
        referenced.update(str(item) for item in fallback["entries"])
        if fallback.get("geometry_entry") is not None:
            referenced.add(str(fallback["geometry_entry"]))
    unknown = sorted(path for path in referenced if path not in sources)
    if unknown:
        raise AssertionError(f"the fixture document references unknown sources {unknown}")
    dropped = set(drop)
    emitted = (referenced | set(extra_sources or {})) - dropped
    entries = []
    for path in sorted(emitted):
        role, media_type, data = sources[path]
        declared_role = (role_overrides or {}).get(path, role)
        if declared_role not in CANDIDATE_ROLES:
            raise AssertionError(f"fixture role {declared_role!r} is not a candidate role")
        entries.append(CandidateEntry(path=path, role=declared_role, media_type=media_type, data=data))
    entries.sort(key=lambda entry: entry.path)
    entries.append(manifest_entry(document))
    return tuple(entries)


def evidence_all_valid(portfolio) -> dict[str, str]:
    """@brief 全部 provider 都标记为 ``validated`` 的证据映射。

    @param portfolio 已解析 portfolio。
    @return 证据映射。
    """
    evidence = {variant_id: "validated" for variant_id in portfolio.variant_ids}
    if portfolio.fallback_id is not None:
        evidence[portfolio.fallback_id] = "validated"
    return evidence


__all__ = [
    "DEVICE_A",
    "DEVICE_B",
    "DEVICE_GENERIC",
    "OVER_LIMIT_THREADS",
    "over_limit_entries",
    "over_limit_source",
    "EXPECTED_PROVIDER",
    "GUARD_A",
    "GUARD_B",
    "HOST_TILING_A",
    "P1_OP",
    "P1_SIZES",
    "bundle_fallback_document",
    "evidence_all_valid",
    "geometry_bytes",
    "geometry_document",
    "load_p1_task",
    "manifest_entry",
    "opspec",
    "p1_instances",
    "parent_entries",
    "portfolio_document",
    "source_entries",
    "write_p1_task",
]
