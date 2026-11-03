"""Shape portfolio P0: outer envelope parsing and mechanical candidate projection (CPU).

The outer ``coverage_plan/portfolio.json`` envelope is parsed *before* any Track sees it, then
every variant/fallback is projected into an ordinary ``CandidateArtifact``:

* Host Tiling + a declared workspace + a declared geometry → ``msl-bundle`` with the geometry at
  the fixed ``host/execution_geometry.json`` ``schedule`` entry;
* exactly one self-contained device source (own ``taic-launch``) and no Host Tiling →
  ``msl-single``;
* anything else is refused — no invented host plan, no invented workspace, no second entry in a
  single-source delivery.

Every non-geometry entry keeps its path/role/media_type/bytes, shared sources are copied into
each projection, the parent digest is never reused as a child digest, and the original geometry
path and content SHA-256 enter the ``EvaluationBinding``.  ``backends/metal/tracks.json`` and the
Track/registry digests are pinned and unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError, UnsupportedContractCapability
from tcenv.identity import fingerprint
from tcenv.portfolio.domain import DomainAuthority
from tcenv.portfolio.candidate import (
    DOMAIN_FALLBACK_ROLE,
    PORTFOLIO_ENTRY_PATH,
    PORTFOLIO_SCHEMA_ID,
    PROJECTED_GEOMETRY_PATH,
    TIMING_BASELINE_ROLE,
    CandidateProjection,
    DomainFallback,
    EvaluationBinding,
    PortfolioCandidate,
    PortfolioVariant,
    parse_portfolio,
    project_provider,
)
from tcenv.portfolio.domain import guard_digest, normalize_semantic_domain_plan
from tcenv.portfolio.dispatch import ExactDispatchResolver
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track import CandidateArtifact, CandidateEntry, candidate_geometry
from tcenv.track.registry import load_track_registry

from ..support.shape_portfolio import (
    GUARD_C,
    bundle_fallback_document,
    geometry_bytes,
    geometry_document,
    load_portfolio_task,
    manifest_entry,
    parent_entries,
    portfolio_document,
    portfolio_instances,
    rename_variant,
    write_portfolio_task,
)

ROOT = Path(__file__).resolve().parents[2]
PROFILE = load_target(DEFAULT_TARGET_ID)
#: ``backends/metal/tracks.json`` and its Track/registry digests as frozen by A4.  P0 must not
#: add a portfolio Track or otherwise change the delivery mechanisms.
TRACKS_SHA256 = "00c1f53c0a51f2a910c490519362e52330fabe9f68e4240b5b33c9490e567f93"
REGISTRY_DIGEST = "9d568eec90bab265cdc5d498a7f31baeb0f1406240617e028f9a5da4dd6044cb"
TRACK_DIGESTS = {
    "msl-single": "7d19a788cb2df0b25816b9f4200d9676f8d7eec317d2411ef0a09c1b8c703af8",
    "msl-bundle": "31b10ee5b33bc6d418106cb85154f6d14d04e08fb92d5382462dbe212610b9ba",
}
FORBIDDEN_ARCHITECTURE_NAMES = (
    "PatternResolver",
    "LoweringRuleRegistry",
    "StagePartitioner",
    "TileConnector",
    "LoweringProfile",
    "@variant",
    "msl-portfolio",
)


def _domain(tmp_path):
    """@brief 固定题 + 有限域计划。

    @param tmp_path 临时目录。
    @return ``(contract, instances, plan)``。
    """
    contract = load_portfolio_task(tmp_path)
    instances = portfolio_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE)
    return contract, instances, plan


def _portfolio(tmp_path, *, document=None, **entry_kwargs):
    """@brief 解析基础 portfolio（可选 envelope/entry 覆盖）。

    @param tmp_path 临时目录。
    @param document envelope 覆盖。
    @param entry_kwargs ``parent_entries`` 参数。
    @return ``(portfolio, plan)``。
    """
    contract, instances, plan = _domain(tmp_path)
    body = document if document is not None else portfolio_document(plan.digest)
    entries = parent_entries(body, **entry_kwargs)
    return parse_portfolio(entries, domain_plan=plan), plan


def _authority(tmp_path, plan):
    """@brief 该固定题与 plan 绑定的权威来源。

    @param tmp_path 临时目录。
    @param plan 域计划。
    @return DomainAuthority。
    """
    contract, instances, _ = _domain(tmp_path)
    return DomainAuthority(
        plan=plan, spec=contract.spec, target=PROFILE, instances=instances, contract=contract
    )


def _with(document, mutate):
    """@brief 深拷贝后修改一份 envelope。

    @param document envelope。
    @param mutate 修改函数。
    @return 新文档。
    """
    copied = copy.deepcopy(dict(document))
    mutate(copied)
    return copied


def _replace_entry(entries, path, **fields):
    """@brief 替换一个父 entry 的字段（保持其它 entry 字节不变）。

    @param entries 父 entries。
    @param path 目标路径。
    @param fields 覆盖字段。
    @return 新 entries。
    """
    out = []
    for entry in entries:
        if entry.path != path:
            out.append(entry)
            continue
        values = {"path": entry.path, "role": entry.role, "media_type": entry.media_type, "data": bytes(entry.data)}
        values.update(fields)
        out.append(CandidateEntry(**values))
    return tuple(out)


def _variant(document, key):
    """@brief 按诊断名取 variant 文档。

    @param document envelope。
    @param key 诊断名。
    @return variant 文档。
    """
    for item in document["variants"]:
        if item["variant_key"] == key:
            return item
    raise AssertionError(f"no variant {key!r}")


def test_p0_the_envelope_is_parsed_before_any_track_sees_it(tmp_path):
    """@brief 母包由 portfolio parser 解包；它本身不是 Track 可消费的候选。"""
    portfolio, plan = _portfolio(tmp_path)
    assert portfolio.schema_id == PORTFOLIO_SCHEMA_ID
    assert portfolio.domain_plan_digest == plan.digest
    assert portfolio.planner.planner_id == "candidate-owned-planner"
    assert portfolio.planner.planner_version == 1
    assert [bucket.bucket_key for bucket in portfolio.buckets] == [
        "threshold_window",
        "large_points",
        "odd_tail",
    ]
    assert [variant.variant_key for variant in portfolio.variants] == [
        "variant-a-bundle",
        "variant-b-single",
        "variant-c-tail",
    ]
    assert len(set(portfolio.variant_ids)) == 3
    for identity in portfolio.variant_ids + (portfolio.fallback_id,):
        assert isinstance(identity, str) and len(identity) == 64
    assert portfolio.fallback is not None and portfolio.fallback.role == DOMAIN_FALLBACK_ROLE
    assert portfolio.parent_sha256 != portfolio.digest
    assert portfolio.manifest_document["schema_id"] == PORTFOLIO_SCHEMA_ID
    assert portfolio.manifest_document["domain_plan_digest"] == plan.digest
    registry = load_track_registry()
    # The parent is not a Track deliverable: it has no candidate geometry entry and a single
    # variant bundle still has more than one Host Tiling entry.
    parent_as_bundle = CandidateArtifact(track_id="msl-bundle", entries=portfolio.entries)
    with pytest.raises(GeometryError, match="ships no schedule entry"):
        candidate_geometry(parent_as_bundle)
    # A parent that only carries single-source variants is not even a legal msl-bundle; the
    # portfolio parser still accepts it, which proves the parent never goes through Track
    # validation.  (The projected children are the ordinary candidates.)
    single_document = portfolio_document(plan.digest)
    single_document["variants"] = [single_document["variants"][1]]
    single = parse_portfolio(parent_entries(single_document), domain_plan=plan)
    assert [variant.variant_key for variant in single.variants] == ["variant-b-single"]
    # The outer envelope itself is not a Track deliverable: its fixed manifest role is one no
    # registered Track accepts, so ``validate_for`` refuses the parent even before delivery shape.
    for track_id in ("msl-single", "msl-bundle"):
        with pytest.raises(TaskContractError, match="not registered"):
            CandidateArtifact(track_id=track_id, entries=single.entries).validate_for(registry.get(track_id))


def test_p0_each_variant_and_fallback_projects_to_an_ordinary_candidate(tmp_path):
    """@brief 机械投影：bundle/single 分轨、几何映射、字节与绑定完整。"""
    portfolio, plan = _portfolio(tmp_path)
    parent_documents = {entry.path: entry.document() for entry in portfolio.entries}
    registry = load_track_registry()
    projections: dict[str, CandidateProjection] = {}
    for provider in tuple(portfolio.variants) + ((portfolio.fallback,) if portfolio.fallback else ()):
        projections[provider.variant_key if hasattr(provider, "variant_key") else "fallback"] = project_provider(
            portfolio, provider, authority=_authority(tmp_path, plan)
        )
    bundle = projections["variant-a-bundle"]
    assert bundle.track_id == "msl-bundle"
    assert bundle.provider == "candidate"
    assert [entry.path for entry in bundle.candidate.entries] == [
        "device/common.metal",
        "device/variant_a.metal",
        "host/execution_geometry.json",
        "host/tiling_a.metal",
    ]
    geometry_entry = next(entry for entry in bundle.candidate.entries if entry.path == PROJECTED_GEOMETRY_PATH)
    assert geometry_entry.role == "schedule"
    assert geometry_entry.media_type == "application/json"
    assert geometry_entry.data == portfolio.entry("host/geometry_a.json").data
    assert bundle.candidate.validate_for(registry.get("msl-bundle")) == "accepted"
    assert bundle.binding.geometry_origin_path == "host/geometry_a.json"
    assert bundle.binding.geometry_origin_sha256 == hashlib.sha256(portfolio.entry("host/geometry_a.json").data).hexdigest()
    assert bundle.binding.geometry_path == PROJECTED_GEOMETRY_PATH
    assert bundle.binding.parent_portfolio_sha256 == portfolio.parent_sha256
    assert bundle.binding.candidate_sha256 == bundle.candidate.candidate_sha256

    for key in ("variant-b-single", "variant-c-tail"):
        projection = projections[key]
        assert projection.track_id == "msl-single"
        assert projection.provider == "candidate"
        assert len(projection.candidate.entries) == 1
        assert projection.candidate.entries[0].role == "device_kernel"
        assert projection.candidate.validate_for(registry.get("msl-single")) == "accepted"
        assert projection.binding.geometry_origin_path is None
        assert projection.binding.geometry_path is None
        assert b"taic-launch" in projection.candidate.entries[0].data

    fallback = projections["fallback"]
    assert fallback.track_id == "msl-single"
    assert fallback.provider == "fallback"
    assert fallback.binding.variant_id is None
    assert fallback.binding.fallback_id == portfolio.fallback_id
    assert fallback.identity == portfolio.fallback_id
    assert fallback.candidate.candidate_sha256 != portfolio.parent_sha256

    for projection in projections.values():
        # Every non-geometry entry is the parent's byte-identical entry; the child digest is new.
        for entry in projection.candidate.entries:
            if entry.path == PROJECTED_GEOMETRY_PATH:
                continue
            assert entry.document() == parent_documents[entry.path]
        assert projection.candidate.candidate_sha256 != portfolio.parent_sha256
        assert len(projection.binding.identity) == 64
    # The parent bytes are never mutated by a projection.
    assert {entry.path: entry.document() for entry in portfolio.entries} == parent_documents
    # A provider may not be projected through a different portfolio.
    other_document = portfolio_document(plan.digest, planner={"id": "another-planner", "version": 1})
    other_document["fallback"]["implementation_version"] = 2
    other_portfolio = parse_portfolio(parent_entries(other_document), domain_plan=plan)
    with pytest.raises(TaskContractError, match="does not belong to this portfolio"):
        project_provider(other_portfolio, portfolio.variants[0], authority=_authority(tmp_path, plan))
    with pytest.raises(TaskContractError, match="does not belong to this portfolio"):
        project_provider(other_portfolio, portfolio.fallback, authority=_authority(tmp_path, plan))


def test_p0_a_shared_source_is_copied_into_every_projection(tmp_path):
    """@brief 公共依赖不会在投影里被缩成"只拷主源"：两个 bundle provider 各自保留同一字节。"""
    _, _, plan = _domain(tmp_path)
    document = bundle_fallback_document(plan.digest)
    portfolio = parse_portfolio(parent_entries(document), domain_plan=plan)
    variant = portfolio.variants[0]
    fallback = portfolio.fallback
    assert fallback is not None and fallback.kind == "registered_implementation"
    variant_common = portfolio.entry("device/common.metal")
    assert "device/common.metal" in variant.entries and "device/common.metal" in fallback.entries
    variant_projection = project_provider(portfolio, variant, authority=_authority(tmp_path, plan))
    fallback_projection = project_provider(portfolio, fallback, authority=_authority(tmp_path, plan))
    for projection in (variant_projection, fallback_projection):
        common = next(entry for entry in projection.candidate.entries if entry.path == "device/common.metal")
        assert common.document() == variant_common.document()
        assert common.data == variant_common.data
        assert projection.track_id == "msl-bundle"
        assert len(projection.candidate.entries) == 4
    # Two providers, two independent identities and two independent geometry declarations.
    assert variant_projection.candidate.candidate_sha256 != fallback_projection.candidate.candidate_sha256
    assert variant_projection.binding.geometry_origin_path == "host/geometry_a.json"
    assert fallback_projection.binding.geometry_origin_path == "host/geometry_generic.json"
    assert variant_projection.identity != fallback_projection.identity


def test_p0_msl_single_requires_exactly_one_self_contained_device_source(tmp_path):
    """@brief 无 Host Tiling 时只允许一个自带 launch 的 device entry；多源/缺 launch 拒绝。"""
    contract, instances, plan = _domain(tmp_path)
    document = portfolio_document(plan.digest)
    good = parse_portfolio(parent_entries(document), domain_plan=plan)
    assert project_provider(good, good.variants[1], authority=_authority(tmp_path, plan)).track_id == "msl-single"
    # Two device entries without Host Tiling is not a legal bundle and not a single source.
    two_device = _with(document, lambda doc: _variant(doc, "variant-b-single").update(
        entries=["device/variant_b.metal", "device/variant_c.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(two_device), domain_plan=plan)
    with pytest.raises(TaskContractError, match="exactly one self-contained device entry"):
        project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))
    # Host Tiling without a declared geometry is still not a bundle.
    host_without_geometry = _with(document, lambda doc: _variant(doc, "variant-b-single").update(
        entries=["device/variant_b.metal", "host/tiling_a.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(host_without_geometry), domain_plan=plan)
    with pytest.raises(TaskContractError, match="exactly one self-contained device entry"):
        project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))
    # A single source must keep its own launch block.
    extra = {"device/no_launch.metal": ("device_kernel", "text/plain", b"kernel void none() {}\n")}
    no_launch = _with(document, lambda doc: _variant(doc, "variant-b-single").update(entries=["device/no_launch.metal"]))
    portfolio = parse_portfolio(parent_entries(no_launch, extra_sources=extra), domain_plan=plan)
    with pytest.raises(TaskContractError, match="no 'taic-launch' block"):
        project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))
    # A fake marker is not a launch block: the projection reuses the backend's launch scan, so a
    # line comment, a string literal or a second block is refused exactly like no marker at all.
    for label, source in (
        ("line comment", b"// taic-launch\nkernel void fake(device half *y) {}\n"),
        ("string literal", b'const char *marker = "taic-launch";\nkernel void fake(device half *y) {}\n'),
        (
            "second block",
            b"/* taic-launch */\n/* taic-launch */\nkernel void fake(device half *y) {}\n",
        ),
        (
            "marker after token",
            b"/* not-a-launch taic-launch */\nkernel void fake(device half *y) {}\n",
        ),
        ("prefix fake", b"/* taic-launch-fake */\nkernel void fake(device half *y) {}\n"),
        (
            "real plus misplaced",
            b"/* taic-launch */\n/* not-a-launch taic-launch */\nkernel void fake(device half *y) {}\n",
        ),
    ):
        name = f"device/fake_{label.replace(' ', '_')}.metal"
        extra = {name: ("device_kernel", "text/plain", source)}
        fake = _with(document, lambda doc: _variant(doc, "variant-b-single").update(entries=[name]))
        portfolio = parse_portfolio(parent_entries(fake, extra_sources=extra), domain_plan=plan)
        with pytest.raises(TaskContractError, match="invalid 'taic-launch' marker|no 'taic-launch' block"):
            project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))
    # A lone Host Tiling entry is not a device source at all.
    host_only = _with(document, lambda doc: _variant(doc, "variant-b-single").update(entries=["host/tiling_a.metal"]))
    portfolio = parse_portfolio(parent_entries(host_only), domain_plan=plan)
    with pytest.raises(TaskContractError, match="exactly one self-contained device entry"):
        project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))


def test_p0_msl_bundle_requires_host_tiling_and_a_declared_workspace(tmp_path):
    """@brief bundle 条件：一个 host entry、至少一个 device、权威 OpSpec 显式 workspace。"""
    contract, instances, plan = _domain(tmp_path)
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    variant = portfolio.variants[0]
    authority = DomainAuthority(
        plan=plan, spec=contract.spec, target=PROFILE, instances=instances, contract=contract
    )
    bundle = project_provider(portfolio, variant, authority=authority)
    assert bundle.track_id == "msl-bundle"
    # The projection itself runs the existing closed HostTilingPlan parser; a merely structural
    # host_tiling role is not enough (the negative plans below are refused by that parser).
    for label, data, fragment in (
        ("not json", b"// not a plan\n", "is not JSON"),
        (
            "unknown field",
            json.dumps(
                {
                    "schema_id": "tcenv.metal.host-tiling.1",
                    "schema_version": 1,
                    "writes": [],
                    "extra": 1,
                }
            ).encode(),
            "must have exactly",
        ),
        (
            "unknown workspace",
            json.dumps(
                {
                    "schema_id": "tcenv.metal.host-tiling.1",
                    "schema_version": 1,
                    "writes": [
                        {
                            "workspace": "nope",
                            "kind": "linear",
                            "start": 0,
                            "step": 1,
                            "count": "N",
                            "stride": 1,
                            "offset": 0,
                            "dtype": "fp16",
                        }
                    ],
                }
            ).encode(),
            "unknown workspace",
        ),
        (
            "dtype mismatch",
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
                            "dtype": "fp32",
                        }
                    ],
                }
            ).encode(),
            "does not match workspace",
        ),
    ):
        name = f"host/tiling_{label.replace(' ', '_')}.json"
        extra = {name: ("host_tiling", "application/json", data)}
        document = _with(
            portfolio_document(plan.digest),
            lambda doc: _variant(doc, "variant-a-bundle").update(
                entries=["device/variant_a.metal", name]
            ),
        )
        bad = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
        with pytest.raises(TaskContractError, match=fragment):
            project_provider(bad, bad.variants[0], authority=authority)
    # The workspace fact is derived from the authoritative OpSpec, not from a caller's boolean:
    # a real workspace-less OpSpec refuses msl-bundle even when the candidate geometry declares
    # scratch.  There is no flag left to lie with.
    bare = load_portfolio_task(tmp_path / "bare", workspace=False)
    bare_instances = portfolio_instances(bare)
    bare_plan = normalize_semantic_domain_plan(bare.spec, bare_instances, target=PROFILE)
    bare_authority = DomainAuthority(
        plan=bare_plan, spec=bare.spec, target=PROFILE, instances=bare_instances, contract=bare
    )
    assert not bare_authority.workspace_declared
    bare_portfolio = parse_portfolio(
        parent_entries(portfolio_document(bare_plan.digest)), domain_plan=bare_plan
    )
    assert bare_portfolio.variants[0].geometry is not None
    with pytest.raises(TaskContractError, match="declares no workspace tensor"):
        project_provider(bare_portfolio, bare_portfolio.variants[0], authority=bare_authority)
    # Nor may a projection borrow another domain's workspace authority.
    with pytest.raises(TaskContractError, match="different semantic domain plan"):
        project_provider(bare_portfolio, bare_portfolio.variants[0], authority=authority)
    # A bundle whose device sources only carry a fake marker (line comment) has no launch block.
    fake = b"// taic-launch\nkernel void fake(device half *y) {}\n"
    extra = {"device/fake_launch.metal": ("device_kernel", "text/plain", fake)}
    document = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-a-bundle").update(
        entries=["device/fake_launch.metal", "host/tiling_a.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
    with pytest.raises(TaskContractError, match="invalid 'taic-launch' marker"):
        project_provider(portfolio, portfolio.variants[0], authority=authority)
    # Two Host Tiling entries are refused (Host Tiling is one candidate plan, not a second file).
    extra = {"host/tiling_b.metal": ("host_tiling", "text/plain", b"// second host plan\n")}
    document = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-a-bundle").update(
        entries=["device/variant_a.metal", "host/tiling_a.metal", "host/tiling_b.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
    with pytest.raises(TaskContractError, match="exactly one host_tiling entry"):
        project_provider(portfolio, portfolio.variants[0], authority=authority)
    # No device source at all.
    document = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-a-bundle").update(
        entries=["host/tiling_a.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(document), domain_plan=plan)
    with pytest.raises(TaskContractError, match="at least one device"):
        project_provider(portfolio, portfolio.variants[0], authority=authority)


def test_p0_includes_match_the_existing_v2_screen(tmp_path):
    """@brief 投影 include 边界与既有 v2 screen 一致：只允许 <metal_...> 标准头。"""
    contract, instances, plan = _domain(tmp_path)
    authority = _authority(tmp_path, plan)
    # The ordinary bundle/single projections succeed; the acceptance harness additionally runs
    # the existing screen on their bytes (a shared test may not import a backend module).
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    project_provider(portfolio, portfolio.variants[0], authority=authority)
    project_provider(portfolio, portfolio.variants[1], authority=authority)
    # A quoted include is refused here exactly because the existing screen refuses it, whether
    # the named source is delivered, missing, or escaping the bundle.
    for label, source in (
        ("delivered", b'#include "device/common.metal"\n/* taic-launch */\nkernel void k(device half *y) {}\n'),
        ("missing", b'#include "device/nope.metal"\n/* taic-launch */\nkernel void k(device half *y) {}\n'),
        ("escape", b'#include "../evil.metal"\n/* taic-launch */\nkernel void k(device half *y) {}\n'),
    ):
        name = f"device/quoted_{label}.metal"
        extra = {name: ("device_kernel", "text/plain", source)}
        document = _with(
            portfolio_document(plan.digest),
            lambda doc: _variant(doc, "variant-b-single").update(entries=[name]),
        )
        portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
        with pytest.raises(TaskContractError, match="quoted #include"):
            project_provider(portfolio, portfolio.variants[1], authority=authority)
    # Angle-bracket host paths / non-Metal headers are refused too.
    for label, source in (
        ("host path", b"#include </etc/passwd>\n/* taic-launch */\nkernel void k(device half *y) {}\n"),
        ("bare host path", b"#include /etc/passwd\n/* taic-launch */\nkernel void k(device half *y) {}\n"),
        ("non-Metal header", b"#include <stdio.h>\n/* taic-launch */\nkernel void k(device half *y) {}\n"),
    ):
        name = f"device/angle_{label.replace(' ', '_')}.metal"
        extra = {name: ("device_kernel", "text/plain", source)}
        document = _with(
            portfolio_document(plan.digest),
            lambda doc: _variant(doc, "variant-b-single").update(entries=[name]),
        )
        portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
        with pytest.raises(TaskContractError, match="Metal standard header"):
            project_provider(portfolio, portfolio.variants[1], authority=authority)
    # A real Metal standard header passes, and so does the existing screen on the same bytes.
    metal = b"#include <metal_stdlib>\n/* taic-launch */\nkernel void k(device half *y) {}\n"
    extra = {"device/angle_metal.metal": ("device_kernel", "text/plain", metal)}
    document = _with(
        portfolio_document(plan.digest),
        lambda doc: _variant(doc, "variant-b-single").update(entries=["device/angle_metal.metal"]),
    )
    portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
    assert project_provider(portfolio, portfolio.variants[1], authority=authority).track_id == "msl-single"


def test_p0_a_legacy_bundle_without_a_coverage_manifest_is_refused(tmp_path):
    """@brief 旧 bundle（无 coverage_plan/portfolio.json）不得进入 portfolio 路径或投影。"""
    contract, instances, plan = _domain(tmp_path)
    entries = parent_entries(portfolio_document(plan.digest))
    legacy = tuple(entry for entry in entries if entry.path != PORTFOLIO_ENTRY_PATH)
    assert legacy
    assert all(entry.path != PORTFOLIO_ENTRY_PATH for entry in legacy)
    assert {entry.path for entry in legacy} >= {"device/variant_b.metal", "device/common.metal"}
    # An ordinary candidate bundle is not a portfolio: the envelope is required, not defaulted
    # from the presence of device sources.
    for bundle in (legacy, tuple(entry for entry in legacy if entry.path == "device/variant_b.metal")):
        with pytest.raises(TaskContractError, match="requires the fixed entry"):
            parse_portfolio(bundle)
        with pytest.raises(TaskContractError, match="requires the fixed entry"):
            parse_portfolio(bundle, domain_plan=plan)
    # The refusal happens before any Track parse/projection: there is no PortfolioCandidate, so
    # project_provider() has no variant/fallback to project and the resolver has no provider to
    # dispatch.  A legacy bundle can therefore neither be projected nor fall back through here.
    with pytest.raises(TaskContractError, match="requires the fixed entry"):
        parse_portfolio(legacy, where="a legacy candidate bundle")


def test_p0_malformed_envelopes_are_refused_closed(tmp_path):
    """@brief 未知字段/伪造 digest/未知 bucket/缺 entry/非法角色/重复几何全部拒绝。"""
    contract, instances, plan = _domain(tmp_path)
    base = portfolio_document(plan.digest)
    registry = load_track_registry()

    def reject(build, expected, message):
        """@brief 断言某个构造必定失败。

        @param build 无参构造。
        @param expected 期望异常类型。
        @param message 期望消息片段。
        """
        with pytest.raises(expected, match=message):
            build()

    def parse_doc(document, **kwargs):
        """@brief 解析一份文档。

        @param document envelope。
        @param kwargs parent_entries 参数。
        @return portfolio。
        """
        return parse_portfolio(parent_entries(document, **kwargs), domain_plan=plan)

    def parse_with(document):
        """@brief 用同一份父字节解析一份被修改的 envelope（只替换 manifest）。

        @param document envelope。
        @return portfolio。
        """
        body = tuple(entry for entry in parent_entries(base) if entry.path != PORTFOLIO_ENTRY_PATH)
        return parse_portfolio(body + (manifest_entry(document),), domain_plan=plan)

    reject(
        lambda: parse_doc(_with(base, lambda doc: doc.update(priority=1))),
        TaskContractError,
        "unknown field",
    )
    reject(
        lambda: parse_doc(_with(base, lambda doc: doc.update(schema_id="tcenv.candidate-portfolio.2"))),
        TaskContractError,
        "expected the constant",
    )
    duplicate = _with(base, lambda doc: _variant(doc, "variant-b-single").update(variant_key="variant-a-bundle"))
    reject(lambda: parse_doc(duplicate), TaskContractError, "repeats the diagnostic variant_key")
    unknown_bucket = _with(base, lambda doc: _variant(doc, "variant-b-single").update(bucket_key="nope"))
    reject(lambda: parse_doc(unknown_bucket), TaskContractError, "unknown bucket")
    bad_bucket = _with(
        base,
        lambda doc: doc["coverage_plan"]["buckets"][1].update(guard_digest="a" * 64),
    )
    reject(lambda: parse_doc(bad_bucket), TaskContractError, "canonical guard digest")
    reject(
        lambda: parse_doc(base, drop={"device/variant_b.metal"}),
        TaskContractError,
        "does not exist in the parent bundle",
    )
    escaped = _with(base, lambda doc: _variant(doc, "variant-b-single").update(entries=["../evil.metal"]))
    reject(
        lambda: parse_portfolio(parent_entries(base)[:-1] + (manifest_entry(escaped),), domain_plan=plan),
        TaskContractError,
        "normalized relative path",
    )
    launch_role = {"device/variant_b.metal": "launch"}
    reject(
        lambda: parse_doc(base, role_overrides=launch_role),
        TaskContractError,
        "never a second schedule/launch declaration",
    )
    schedule_role = {"notes/extra.metal": "schedule"}
    extra_bytes = {"notes/extra.metal": ("device_kernel", "text/plain", b"notes\n")}
    reject(
        lambda: parse_doc(base, extra_sources=extra_bytes, role_overrides=schedule_role),
        TaskContractError,
        "may carry only",
    )
    # A fallback may not deliver its own declared geometry as an ordinary source entry.
    fallback_leak = bundle_fallback_document(plan.digest)
    fallback_leak["fallback"]["entries"].append("host/geometry_generic.json")
    reject(lambda: parse_doc(fallback_leak), TaskContractError, "must not also be listed in entries")
    # Nor may another provider smuggle that geometry into its own entries (it would be delivered
    # as an ordinary Host Tiling next to the variant's mapped execution geometry).
    cross_leak = bundle_fallback_document(plan.digest)
    _variant(cross_leak, "variant-a-bundle")["entries"].append("host/geometry_generic.json")
    reject(lambda: parse_doc(cross_leak), TaskContractError, "also listed as source entries")
    geometry_role = {"host/geometry_a.json": "device_kernel"}
    reject(lambda: parse_doc(base, role_overrides=geometry_role), TaskContractError, "must use 'host_tiling'")
    geometry_media = _replace_entry(parent_entries(base), "host/geometry_a.json", media_type="text/plain")
    reject(
        lambda: parse_portfolio(geometry_media, domain_plan=plan),
        TaskContractError,
        "expected 'application/json'",
    )
    duplicate_geometry = _with(
        base, lambda doc: doc["fallback"].update(geometry_entry="host/geometry_a.json")
    )
    reject(lambda: parse_with(duplicate_geometry), TaskContractError, "duplicate geometry declaration")
    geometry_also_entry = _with(
        base,
        lambda doc: _variant(doc, "variant-a-bundle").update(
            entries=["device/variant_a.metal", "host/tiling_a.metal", "host/geometry_a.json"]
        ),
    )
    reject(lambda: parse_with(geometry_also_entry), TaskContractError, "must not also be listed in entries")
    missing_geometry = _with(
        base, lambda doc: _variant(doc, "variant-a-bundle").update(geometry_entry="host/absent.json")
    )
    reject(lambda: parse_with(missing_geometry), TaskContractError, "does not exist")
    fallback_source = geometry_bytes(geometry_document("variant_a", source="fallback")).decode("utf-8")
    reject(
        lambda: parse_doc(base, package=fallback_source),
        GeometryError,
        "must declare source 'candidate'",
    )
    empty_stages = geometry_document("variant_a")
    empty_stages["stages"] = []
    reject(
        lambda: parse_doc(base, package=json.dumps(empty_stages)),
        GeometryError,
        "at least one stage",
    )
    refuse_with_entries = _with(
        base,
        lambda doc: doc.update(
            fallback={
                "kind": "refuse",
                "implementation_id": None,
                "implementation_version": None,
                "entries": ["device/generic.metal"],
                "geometry_entry": None,
                "reasons": ["outside_candidate_coverage"],
            }
        ),
    )
    reject(lambda: parse_with(refuse_with_entries), TaskContractError, "must not carry entries")
    unknown_reason = _with(
        base, lambda doc: doc["fallback"].update(reasons=["because_the_candidate_says_so"])
    )
    reject(lambda: parse_with(unknown_reason), TaskContractError, "not one of")
    forged = _with(base, lambda doc: _variant(doc, "variant-b-single")["guard"].update(digest="f" * 64))
    reject(lambda: parse_doc(forged), TaskContractError, "canonical AST digest")
    raw_digest = hashlib.sha256(json.dumps(GUARD_C).encode("utf-8")).hexdigest()
    raw_spelling = _with(base, lambda doc: _variant(doc, "variant-c-tail")["guard"].update(digest=raw_digest))
    reject(lambda: parse_doc(raw_spelling), TaskContractError, "canonical AST digest")
    manifest_role = _replace_entry(parent_entries(base), PORTFOLIO_ENTRY_PATH, role="device_kernel")
    reject(
        lambda: parse_portfolio(manifest_role, domain_plan=plan),
        TaskContractError,
        "must declare 'launch'",
    )
    baseline_role = _with(base, lambda doc: doc["fallback"].update(role=TIMING_BASELINE_ROLE))
    reject(lambda: parse_with(baseline_role), TaskContractError, "unknown field")
    reject(
        lambda: parse_doc(_with(base, lambda doc: doc.update(variants=[]))),
        TaskContractError,
        "fewer than 1 items",
    )
    reject(
        lambda: parse_doc(_with(base, lambda doc: _variant(doc, "variant-b-single").update(entries=[]))),
        TaskContractError,
        "fewer than 1 items",
    )
    reject(
        lambda: parse_doc(_with(base, lambda doc: doc["planner"].update(version=0))),
        TaskContractError,
        "below the minimum",
    )
    # A hand-built variant whose delivered manifest omits a declared entry must not get a
    # narrower variant_id than the entries it actually delivers.
    portfolio = parse_doc(base)
    variant = portfolio.variants[0]
    narrowed = PortfolioVariant(
        variant_key=variant.variant_key,
        bucket_key=variant.bucket_key,
        guard=variant.guard,
        entries=variant.entries,
        geometry_entry=variant.geometry_entry,
        geometry=variant.geometry,
        geometry_sha256=variant.geometry_sha256,
        delivered=variant.delivered[1:],
        domain_plan_digest=variant.domain_plan_digest,
        planner=variant.planner,
    )
    reject(
        lambda: PortfolioCandidate(
            entries=portfolio.entries,
            manifest_entry=portfolio.manifest_entry,
            manifest=portfolio.manifest,
            domain_plan_digest=portfolio.domain_plan_digest,
            planner=portfolio.planner,
            buckets=portfolio.buckets,
            variants=(narrowed,),
            fallback=portfolio.fallback,
        ),
        TaskContractError,
        "cover exactly its declared entries",
    )
    # A different domain plan is not interchangeable with this portfolio.
    other = load_task_contract(write_portfolio_task(tmp_path / "other", op_name="portfolio_other"))
    other_plan = normalize_semantic_domain_plan(other.spec, portfolio_instances(other), target=PROFILE)
    reject(
        lambda: parse_portfolio(parent_entries(base), domain_plan=other_plan),
        TaskContractError,
        "not the supplied",
    )


def test_p0_a_projection_path_collision_and_parent_sha_reuse_are_refused(tmp_path):
    """@brief 映射路径冲突与父 SHA 复用都拒绝；真实投影永远不会复用父身份。"""
    contract, instances, plan = _domain(tmp_path)
    extra = {
        PROJECTED_GEOMETRY_PATH: ("host_tiling", "application/json", b'{"declared": "by the candidate"}\n')
    }
    document = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-a-bundle").update(
        entries=["device/variant_a.metal", "host/tiling_a.metal", PROJECTED_GEOMETRY_PATH]
    ))
    portfolio = parse_portfolio(parent_entries(document, extra_sources=extra), domain_plan=plan)
    with pytest.raises(TaskContractError, match="would overwrite"):
        project_provider(portfolio, portfolio.variants[0], authority=_authority(tmp_path, plan))
    # The child identity can never be the parent identity.
    portfolio = parse_portfolio(parent_entries(portfolio_document(plan.digest)), domain_plan=plan)
    projection = project_provider(portfolio, portfolio.variants[1], authority=_authority(tmp_path, plan))
    binding = projection.binding.document()
    binding["candidate_sha256"] = portfolio.parent_sha256
    with pytest.raises(TaskContractError, match="must not reuse the parent SHA"):
        EvaluationBinding(**binding)


def test_p0_variant_identity_is_stable_author_neutral_and_geometry_free(tmp_path):
    """@brief variant_id 只含声明内容：重命名/作者不变，guard/entries 变则变，且不含实例几何。"""
    contract, instances, plan = _domain(tmp_path)
    entries = parent_entries(portfolio_document(plan.digest))
    first = parse_portfolio(entries, domain_plan=plan)
    second = parse_portfolio(entries, domain_plan=plan)
    assert first.variant_ids == second.variant_ids
    # A renamed diagnostic key (and a different planner session/author) does not change identity.
    renamed = parse_portfolio(parent_entries(rename_variant(portfolio_document(plan.digest), "variant-a-bundle", "renamed")), domain_plan=plan)
    by_key_first = {variant.variant_key: variant.variant_id for variant in first.variants}
    by_key_renamed = {variant.variant_key: variant.variant_id for variant in renamed.variants}
    assert by_key_renamed["renamed"] == by_key_first["variant-a-bundle"]
    # A different guard changes the identity (and its bucket digest is updated consistently).
    other_guard = ["and", ["ge", "N", 65], ["lt", "N", 128]]
    mutated = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-a-bundle").update(
        guard={"expr": other_guard, "digest": guard_digest(other_guard, where="variant-a-bundle")},
    ))
    mutated["coverage_plan"]["buckets"][0]["guard_digest"] = guard_digest(other_guard, where="threshold_window")
    changed_guard = parse_portfolio(parent_entries(mutated), domain_plan=plan)
    assert changed_guard.variants[0].variant_id != first.variants[0].variant_id
    # A different entry set changes the identity.
    extra = {"device/extra.metal": ("device_kernel", "text/plain", b"kernel void extra() {}\n")}
    changed_entries = _with(portfolio_document(plan.digest), lambda doc: _variant(doc, "variant-b-single").update(
        entries=["device/variant_b.metal", "device/extra.metal"]
    ))
    portfolio = parse_portfolio(parent_entries(changed_entries, extra_sources=extra), domain_plan=plan)
    assert portfolio.variants[1].variant_id != first.variants[1].variant_id
    # The identity document contains declarations only — no measurement, no resolved grid/threads.
    identity = first.variants[0].identity_document()
    assert set(identity) == {
        "fingerprint_kind",
        "fingerprint_version",
        "domain_plan_digest",
        "planner",
        "guard",
        "entries",
        "geometry",
    }
    for forbidden in ("median_us", "speedup", "author", "session", "committee", "resolution", "grid"):
        assert forbidden not in json.dumps(identity)
    # The instance-resolved execution identity is a different thing and cannot stand in for it:
    # neither portfolio lookup nor the dispatch evidence accepts it as a stable variant key.
    variant = first.variants[0]
    assert variant.geometry is not None
    from tcenv.track.abi import abi_limits_from_target, derive_tensor_memory_contract
    from tcenv.track.geometry import geometry_pair

    abi = derive_tensor_memory_contract(instances[0], target=PROFILE)
    limits = abi_limits_from_target(PROFILE)
    pair = geometry_pair(variant.geometry, instance=instances[0], contract=abi, limits=limits)
    assert pair.identity != variant.variant_id
    assert first.variant_for_provider(pair.identity) is None
    with pytest.raises(TaskContractError, match="not a variant or fallback"):
        ExactDispatchResolver(
            domain_plan=plan,
            portfolio=first,
            evidence={pair.identity: "validated"},
            spec=contract.spec,
            target=PROFILE,
            instances=instances,
        )


def test_p0_the_track_registry_and_tracks_json_are_unchanged():
    """@brief tracks.json / Track / registry digest 不变，且没有新增 portfolio Track。"""
    path = ROOT / "backends/metal/tracks.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == TRACKS_SHA256
    registry = load_track_registry()
    assert fingerprint(registry.document()) == REGISTRY_DIGEST
    assert {track.track_id for track in registry.tracks} == {"msl-single", "msl-bundle"}
    for track in registry.tracks:
        assert fingerprint(track.document()) == TRACK_DIGESTS[track.track_id]
    with pytest.raises(UnsupportedContractCapability, match="unknown track"):
        registry.get("msl-portfolio")


def test_p0_the_domain_fallback_is_not_a_timing_baseline(tmp_path):
    """@brief DomainFallback 是正确性 provider，角色/身份/记功都与计时 baseline 分开。"""
    assert DOMAIN_FALLBACK_ROLE == "domain_fallback"
    assert TIMING_BASELINE_ROLE == "timing_baseline"
    assert DOMAIN_FALLBACK_ROLE != TIMING_BASELINE_ROLE
    portfolio, plan = _portfolio(tmp_path)
    fallback = portfolio.fallback
    assert isinstance(fallback, DomainFallback)
    assert fallback.role == DOMAIN_FALLBACK_ROLE
    assert isinstance(fallback.fallback_id, str) and len(fallback.fallback_id) == 64
    assert fallback.fallback_id not in portfolio.variant_ids
    document = fallback.document()
    assert document["role"] == DOMAIN_FALLBACK_ROLE
    for baseline_only in ("baseline", "median_us", "speedup", "timing", "benchmark"):
        assert baseline_only not in document
    projection = project_provider(portfolio, fallback, authority=_authority(tmp_path, plan))
    assert projection.provider == "fallback"
    assert projection.binding.provider == "fallback"
    assert projection.binding.variant_id is None
    assert projection.binding.fallback_id == fallback.fallback_id
    assert projection.identity not in portfolio.variant_ids
    # A refuse fallback is a declaration, not a projectable implementation.
    refuse = _with(portfolio_document(plan.digest), lambda doc: doc.update(
        fallback={
            "kind": "refuse",
            "implementation_id": None,
            "implementation_version": None,
            "entries": [],
            "geometry_entry": None,
            "reasons": ["outside_candidate_coverage"],
        }
    ))
    refused = parse_portfolio(parent_entries(refuse), domain_plan=plan)
    with pytest.raises(TaskContractError, match="not projected"):
        project_provider(refused, refused.fallback, authority=_authority(tmp_path, plan))
    assert refused.fallback.fallback_id != fallback.fallback_id


def test_p0_the_portfolio_package_revives_no_forbidden_architecture(tmp_path):
    """@brief 不复活 PatternResolver/LoweringProfile/@variant 等计划明确禁止的架构名。"""
    package = ROOT / "tcenv" / "portfolio"
    sources = sorted(package.glob("*.py"))
    assert [path.name for path in sources] == ["__init__.py", "candidate.py", "dispatch.py", "domain.py"]
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_ARCHITECTURE_NAMES:
            if token in text:
                offenders.append(f"{path.name}: {token}")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import tcenv.backends", "from tcenv.backends", "import taic", "from taic")):
                offenders.append(f"{path.name}: {stripped}")
    assert offenders == []
    import tcenv.portfolio.candidate as candidate_module
    import tcenv.portfolio.dispatch as dispatch_module
    import tcenv.portfolio.domain as domain_module

    for module in (candidate_module, dispatch_module, domain_module):
        for name in ("PatternResolver", "LoweringRuleRegistry", "StagePartitioner", "TileConnector", "LoweringProfile"):
            assert not hasattr(module, name)
    # The composition record has its own name; the runtime's single-stage record is untouched.
    assert not hasattr(dispatch_module, "DispatchRecord")
    assert dispatch_module.PortfolioDispatchRecord.__name__ == "PortfolioDispatchRecord"
