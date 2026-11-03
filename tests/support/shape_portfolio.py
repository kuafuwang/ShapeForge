"""Fixed, hand-written shape-portfolio fixtures for P0 (no planner, no model, no GPU).

The fixtures are deliberately tiny and deterministic:

* one finite ``y = -x`` OpSpec with five explicit sizes — an odd tail (``N=7``), a point just
  below the dispatch threshold (``N=63``), the threshold (``N=64``) and +1 (``N=65``), and a large
  point (``N=1024``).  A ``workspace`` tensor is declared so an msl-bundle projection is legal;
* variant A is a candidate Host-Tiling bundle (device source + host plan + declared geometry),
  variant B is a self-contained ``taic-launch`` device source, variant C is an odd-tail source
  whose guard uses ``rem``;
* one generic fallback source (``msl-single``) declaring both execution-derived reasons;
* shared sources are declared through ``public_entries`` so every projection must keep them.

Nothing here runs a planner: the documents are literal candidate output, exactly the input the
portfolio parser reads in production.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from tcenv.opspec.v2 import SemanticInstanceV2
from tcenv.portfolio.candidate import CandidateEntry
from tcenv.portfolio.domain import guard_digest
from tcenv.taskcontract import TaskContract, load_task_contract
from tcenv.track.plane import CANDIDATE_ROLES

from .v2_tasks import SEM_REFERENCE, write_task

PORTFOLIO_OP = "portfolio_pointwise"

#: ``y = -x`` with one workspace scratch tensor and five explicit finite points.
PORTFOLIO_SIZES: dict[str, dict[str, Any]] = {
    "tail": {"N": 7, "usage": ["golden", "benchmark"]},
    "below": {"N": 63, "usage": ["golden", "benchmark"]},
    "threshold": {"N": 64, "usage": ["golden", "benchmark"]},
    "above": {"N": 65, "usage": ["golden", "benchmark"]},
    "large": {"N": 1024, "usage": ["golden", "benchmark"]},
}

DEVICE_A = b"""\
/* taic-launch
math safe
launch variant_a grid=ceil_div(N,256),1,1 threads=256,1,1 threadgroup=0
*/
kernel void variant_a(device half *x, device half *y, device half *scratch) {
  uint i = get_global_id(0);
  if (i < N) { scratch[i] = -x[i]; y[i] = scratch[i]; }
}
"""
DEVICE_B = b"""\
/* taic-launch
math safe
launch variant_b grid=ceil_div(N,128),1,1 threads=128,1,1 threadgroup=0
*/
kernel void variant_b(device half *x, device half *y) {
  uint i = get_global_id(0);
  if (i < N) { y[i] = -x[i]; }
}
"""
DEVICE_C = b"""\
/* taic-launch
math safe
launch variant_c grid=1,1,1 threads=32,1,1 threadgroup=0
*/
kernel void variant_c(device half *x, device half *y) {
  // odd tail: one threadgroup masks the remainder
  if (get_global_id(0) < N) { y[get_global_id(0)] = -x[get_global_id(0)]; }
}
"""
DEVICE_GENERIC = b"""\
/* taic-launch
math safe
launch generic grid=ceil_div(N,64),1,1 threads=64,1,1 threadgroup=0
*/
kernel void generic(device half *x, device half *y) {
  uint i = get_global_id(0);
  if (i < N) { y[i] = -x[i]; }
}
"""
COMMON = b"#define TAIC_PORTFOLIO_COMMON 1\n"
#: A real ``tcenv.metal.host-tiling.1`` plan (the existing closed parser validates it).
HOST_TILING = (
    json.dumps(
        {
            "schema_id": "tcenv.metal.host-tiling.1",
            "schema_version": 1,
            "writes": [
                {
                    "workspace": "scratch",
                    "kind": "linear",
                    "start": 0,
                    "step": 1,
                    "count": "N",
                    "stride": 1,
                    "offset": 0,
                    "dtype": "fp16",
                }
            ],
        },
        indent=2,
        sort_keys=True,
    ).encode("utf-8")
    + b"\n"
)


def portfolio_opspec(op_name: str = PORTFOLIO_OP, *, workspace: bool = True) -> dict[str, Any]:
    """@brief 固定有限 portfolio 题（1-D ``y=-x``，可选 workspace scratch）。

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
                "dtype": "T",
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
        "sizes": copy.deepcopy(PORTFOLIO_SIZES),
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


def write_portfolio_task(
    root: Path, *, op_name: str = PORTFOLIO_OP, workspace: bool = True
) -> Path:
    """@brief 写一个标准的有限 portfolio 测试题目录。

    @param root 父目录。
    @param op_name 算子名。
    @param workspace 是否声明 workspace 张量。
    @return 任务目录。
    """
    return write_task(
        root,
        op_name=op_name,
        opspec=portfolio_opspec(op_name, workspace=workspace),
        reference=SEM_REFERENCE,
    )


def load_portfolio_task(
    root: Path, *, op_name: str = PORTFOLIO_OP, workspace: bool = True
) -> TaskContract:
    """@brief 装配测试题合同。

    @param root 父目录。
    @param op_name 算子名。
    @param workspace 是否声明 workspace 张量。
    @return 任务合同。
    """
    return load_task_contract(write_portfolio_task(root, op_name=op_name, workspace=workspace))


def portfolio_instances(contract: TaskContract) -> tuple[SemanticInstanceV2, ...]:
    """@brief 该合同的全部显式命名实例（按 sizes 声明顺序）。

    @param contract 任务合同。
    @return 语义实例元组。
    """
    from tcenv.opspec.v2 import instantiate_v2

    return tuple(
        instantiate_v2(contract.spec, contract.dtype, size) for size in PORTFOLIO_SIZES
    )


def geometry_document(
    kernel: str,
    *,
    threads: int = 256,
    threadgroup_bytes: int = 0,
    workspace: bool = True,
    grid: Any = None,
    source: str = "candidate",
) -> dict[str, Any]:
    """@brief 一个候选几何文档（variant/fallback 的 ``geometry_entry`` 内容）。

    @param kernel 内核名。
    @param threads 每组线程数。
    @param threadgroup_bytes 线程组内存。
    @param workspace 是否声明 scratch workspace。
    @param grid 首个 grid 分量；缺省 ``ceil_div(N, threads)``。
    @param source 几何来源（候选声明必须是 ``candidate``）。
    @return 几何文档。
    """
    stage: dict[str, Any] = {
        "kernel": kernel,
        "grid": [grid if grid is not None else {"ceil_div": ["N", max(1, threads)]}, 1, 1],
        "threads": [threads, 1, 1],
        "threadgroup_bytes": threadgroup_bytes,
        "workspace": [{"name": "scratch", "zero": True}] if workspace else [],
        "when": None,
        "derived_scalars": [],
    }
    return {"schema_version": 1, "source": source, "stages": [stage]}


def geometry_bytes(document: Mapping[str, Any]) -> bytes:
    """@brief 几何文档的规范 JSON 字节。

    @param document 文档。
    @return UTF-8 JSON 字节。
    """
    return json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8") + b"\n"


def source_entries() -> dict[str, tuple[str, str, bytes]]:
    """@brief 父 bundle 的全部可用源（路径 → role/media_type/bytes）。

    @return 路径到三元组的映射。
    """
    return {
        "device/common.metal": ("device_kernel", "text/plain", COMMON),
        "device/variant_a.metal": ("device_kernel", "text/plain", DEVICE_A),
        "device/variant_b.metal": ("device_kernel", "text/plain", DEVICE_B),
        "device/variant_c.metal": ("device_kernel", "text/plain", DEVICE_C),
        "device/generic.metal": ("device_kernel", "text/plain", DEVICE_GENERIC),
        "host/tiling_a.metal": ("host_tiling", "application/json", HOST_TILING),
        "host/geometry_a.json": ("host_tiling", "application/json", geometry_bytes(geometry_document("variant_a"))),
        "host/geometry_generic.json": (
            "host_tiling",
            "application/json",
            geometry_bytes(geometry_document("generic")),
        ),
    }


def _bucket(key: str, expr: Any) -> dict[str, Any]:
    """@brief 一个 bucket + guard 文档。

    @param key bucket 名。
    @param expr 该 bucket 的 guard 表达式。
    @return bucket 文档。
    """
    return {"bucket_key": key, "guard_digest": guard_digest(expr, where=key)}


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


#: The three bucket guards: threshold window, large points, odd tail below 32.
GUARD_A: Any = ["and", ["ge", "N", 64], ["lt", "N", 128]]
GUARD_B: Any = ["ge", "N", 1024]
GUARD_C: Any = ["and", ["lt", "N", 32], ["eq", ["rem", "N", 2], 1]]


def portfolio_document(domain_plan_digest: str, *, planner: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """@brief 基础 portfolio envelope（variant A msL-bundle / B、C msl-single / generic fallback）。

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
                _bucket("large_points", GUARD_B),
                _bucket("odd_tail", GUARD_C),
            ]
        },
        "variants": [
            _variant(
                "variant-a-bundle",
                "threshold_window",
                GUARD_A,
                ["device/common.metal", "device/variant_a.metal", "host/tiling_a.metal"],
                "host/geometry_a.json",
            ),
            _variant("variant-b-single", "large_points", GUARD_B, ["device/variant_b.metal"], None),
            _variant("variant-c-tail", "odd_tail", GUARD_C, ["device/variant_c.metal"], None),
        ],
        "fallback": {
            "kind": "registered_implementation",
            "implementation_id": "portfolio.generic",
            "implementation_version": 1,
            "entries": ["device/generic.metal"],
            "geometry_entry": None,
            "reasons": ["outside_candidate_coverage", "resource_unavailable"],
        },
    }


def bundle_fallback_document(domain_plan_digest: str) -> dict[str, Any]:
    """@brief 一个走 msl-bundle 的通用 fallback fixture（自己 Host Tiling + 几何 + 共享源）。

    @param domain_plan_digest 绑定的语义域摘要。
    @return envelope 文档。
    """
    document = portfolio_document(domain_plan_digest)
    document["fallback"] = {
        "kind": "registered_implementation",
        "implementation_id": "portfolio.generic.bundle",
        "implementation_version": 2,
        "entries": ["device/common.metal", "device/generic.metal", "host/tiling_a.metal"],
        "geometry_entry": "host/geometry_generic.json",
        "reasons": ["outside_candidate_coverage", "resource_unavailable"],
    }
    return document


def manifest_entry(document: Mapping[str, Any]) -> CandidateEntry:
    """@brief 把 envelope 文档包成固定的父 entry。

    @param document envelope 文档。
    @return ``coverage_plan/portfolio.json`` entry。
    """
    data = json.dumps(dict(document), indent=2, sort_keys=True).encode("utf-8") + b"\n"
    # ``launch`` is the parent envelope role: a registered candidate-role name that no TrackSpec
    # accepts for a source, so the parent can never pass CandidateArtifact.validate_for.
    return CandidateEntry(path="coverage_plan/portfolio.json", role="launch", media_type="application/json", data=data)


def parent_entries(
    document: Mapping[str, Any],
    *,
    extra_sources: Mapping[str, tuple[str, str, bytes]] | None = None,
    drop: Iterable[str] = (),
    role_overrides: Mapping[str, str] | None = None,
    package: str | None = None,
) -> tuple[CandidateEntry, ...]:
    """@brief 由 envelope + 固定源构造父 bundle entries。

    @param document envelope 文档。
    The bundle carries exactly the sources the document references (plus any ``extra_sources``
    bytes a negative test wants to hide in the envelope); ``drop`` removes a referenced source so
    a "declared but absent dependency" can be constructed.

    @param extra_sources 额外路径 → (role, media_type, bytes)。
    @param drop 不写入的源路径（即使 document 引用了它）。
    @param role_overrides 覆盖某些源的角色。
    @param package 自定义几何内容（写入 ``host/geometry_a.json``）。
    @return 父 entries（manifest 在最后）。
    """
    sources = source_entries()
    sources.update(dict(extra_sources or {}))
    if package is not None:
        sources["host/geometry_a.json"] = ("host_tiling", "application/json", package.encode("utf-8") + b"\n")
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


def rename_variant(document: Mapping[str, Any], variant_key: str, new_key: str) -> dict[str, Any]:
    """@brief 复制并重命名一个 variant 的诊断名（用于 author-neutral 断言）。

    @param document envelope 文档。
    @param variant_key 原诊断名。
    @param new_key 新诊断名。
    @return 新文档。
    """
    copied = copy.deepcopy(dict(document))
    for variant in copied["variants"]:
        if variant["variant_key"] == variant_key:
            variant["variant_key"] = new_key
            return copied
    raise AssertionError(f"no variant {variant_key!r}")


__all__ = [
    "COMMON",
    "DEVICE_A",
    "DEVICE_B",
    "DEVICE_C",
    "DEVICE_GENERIC",
    "GUARD_A",
    "GUARD_B",
    "GUARD_C",
    "HOST_TILING",
    "PORTFOLIO_OP",
    "PORTFOLIO_SIZES",
    "bundle_fallback_document",
    "evidence_all_valid",
    "geometry_bytes",
    "geometry_document",
    "load_portfolio_task",
    "manifest_entry",
    "parent_entries",
    "portfolio_document",
    "portfolio_instances",
    "portfolio_opspec",
    "rename_variant",
    "source_entries",
    "write_portfolio_task",
]
