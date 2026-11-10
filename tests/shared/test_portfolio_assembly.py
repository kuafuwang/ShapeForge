"""M0 probes for shape-portfolio assembly: prefixed bundle paths and varlen domains.

These tests call only existing portfolio APIs. They pin the premises the assembly
commands rely on: a bundle variant may live under a ``variants/<key>/`` prefix, two
such variants can share one parent, and a varlen domain either normalizes or is
recorded as unsupported.
"""

from __future__ import annotations

import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import pytest

from taic.task_binding import bind_task
from tcenv.cli import main
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio.domain import DomainAuthority, guard_digest, normalize_semantic_domain_plan
from tcenv.portfolio_assembly import (
    PortfolioAssemblyError,
    benchmark_sizes,
    check_parent,
    decode_parent,
    domain_for_sizes,
    encode_parent,
    parent_entries_from_tree,
)
from tcenv.target import load_target
from tcenv.track.plane import CandidateEntry
from tests.support.kda_fixed_bundle import (
    DEVICE_ENTRY_PATH,
    GEOMETRY_ENTRY_PATH,
    HOST_ENTRY_PATH,
    bundle_entries,
)
from tests.support.portfolio_assembly import (
    PAPER_GUARD,
    SMALL_GUARD,
    gemm_variant_source,
    write_gemm_tree,
    write_kda_tree,
)

KDA_DTYPE = "fp16"
KDA_TARGET = "apple-m4-max"
KDA_SIZES = ("small", "tail", "paper")


def _kda_domain(sizes: tuple[str, ...]):
    """@brief 按计划的三行造 kda 域权威。

    @param sizes 纳入域的 size。
    @return ``(contract, authority)``。
    """
    contract = bind_task("kda", KDA_DTYPE, "small", target_id=KDA_TARGET).locator.load()
    target = load_target(KDA_TARGET)
    instances = tuple(instantiate_v2(contract.spec, contract.dtype, size) for size in sizes)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(
        plan=plan, spec=contract.spec, target=target, instances=instances, contract=contract
    )
    return contract, authority


def _prefixed_bundle(prefix: str) -> tuple[CandidateEntry, ...]:
    """@brief 把 kda bundle 的三个文件放到 ``variants/<key>/`` 下。

    几何那一项的角色从 ``schedule`` 改成 ``host_tiling``，投影时才会被映射成固定路径上的
    ``schedule``。

    @param prefix 形如 ``variants/a`` 的前缀。
    @return 三条改路径后的 entry。
    """
    placed = []
    for entry in bundle_entries():
        role = "host_tiling" if entry.path == GEOMETRY_ENTRY_PATH else entry.role
        placed.append(
            CandidateEntry(
                path=f"{prefix}/{entry.path}",
                role=role,
                media_type=entry.media_type,
                data=entry.data,
            )
        )
    return tuple(placed)


def _portfolio_document(plan_digest: str, variants: tuple[tuple[str, object, tuple[CandidateEntry, ...]], ...]) -> dict:
    """@brief 手写一份只含这些变体的 portfolio 文档。

    @param plan_digest 域计划摘要。
    @param variants ``(variant_key, guard, entries)``。
    @return 文档。
    """
    buckets = []
    declared = []
    for key, guard, entries in variants:
        digest = guard_digest(guard, where=key)
        geometry = next(entry.path for entry in entries if entry.path.endswith(GEOMETRY_ENTRY_PATH))
        own = [entry.path for entry in entries if entry.path != geometry]
        buckets.append({"bucket_key": key, "guard_digest": digest})
        declared.append(
            {
                "variant_key": key,
                "bucket_key": key,
                "guard": {"expr": guard, "digest": digest},
                "entries": own,
                "geometry_entry": geometry,
            }
        )
    return {
        "schema_id": "tcenv.candidate-portfolio.1",
        "domain_plan_digest": plan_digest,
        "planner": {"id": "m0-probe", "version": 1},
        "coverage_plan": {"buckets": buckets},
        "variants": declared,
        "fallback": {
            "kind": "refuse",
            "implementation_id": None,
            "implementation_version": None,
            "entries": [],
            "geometry_entry": None,
            "reasons": ["outside_candidate_coverage"],
        },
    }


def _parent_entries(document: dict, groups: tuple[tuple[CandidateEntry, ...], ...]) -> tuple[CandidateEntry, ...]:
    """@brief 父包 entries：各变体源文件加 ``coverage_plan/portfolio.json``。

    @param document portfolio 文档。
    @param groups 每个变体的源 entries。
    @return 父 entries。
    """
    sources = [entry for group in groups for entry in group]
    manifest = CandidateEntry(
        path="coverage_plan/portfolio.json",
        role="launch",
        media_type="application/json",
        data=json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    return tuple(sorted(sources, key=lambda entry: entry.path) + [manifest])


def test_probe_a_bundle_variant_under_a_prefixed_path_projects():
    """@brief 带前缀的 bundle 变体能投影成 ``msl-bundle``，几何落到固定 schedule 路径。
    """
    _contract, authority = _kda_domain(("small",))
    entries = _prefixed_bundle("variants/a")
    guard = ["ge", "N", 1]
    document = _portfolio_document(authority.plan.digest, (("a", guard, entries),))
    portfolio = parse_portfolio(_parent_entries(document, (entries,)), domain_plan=authority.plan)
    projection = project_provider(portfolio, portfolio.variants[0], authority=authority)
    assert projection.track_id == "msl-bundle"
    geometry = next(entry for entry in projection.candidate.entries if entry.path == "host/execution_geometry.json")
    assert geometry.role == "schedule"
    assert {entry.path for entry in entries} == {
        f"variants/a/{DEVICE_ENTRY_PATH}",
        f"variants/a/{HOST_ENTRY_PATH}",
        f"variants/a/{GEOMETRY_ENTRY_PATH}",
    }


def test_probe_two_bundle_variants_share_one_parent():
    """@brief 两个前缀 bundle 变体共用一个父包，guard 不重叠且都命中域点。
    """
    _contract, authority = _kda_domain(KDA_SIZES)
    left = _prefixed_bundle("variants/a")
    right = _prefixed_bundle("variants/b")
    document = _portfolio_document(
        authority.plan.digest,
        (
            ("a", ["lt", "N", 100], left),
            ("b", ["ge", "N", 100], right),
        ),
    )
    portfolio = parse_portfolio(_parent_entries(document, (left, right)), domain_plan=authority.plan)
    report = portfolio.registration_report(
        authority.plan, authority.instances, spec=authority.spec, target=authority.target
    )
    assert report.overlaps == ()
    assert report.empty_variants == ()
    symbols = {instance.size: int(instance.symbols["N"]) for instance in authority.instances}
    assert symbols == {"small": 14, "tail": 17, "paper": 256}


def test_probe_varlen_sizes_form_a_domain():
    """@brief ``varlen_kda`` 的两个 v6 fixed 档能组成有限域，且正好 4 个点。

    两个 size 各有 golden 与 benchmark，所以域里是 4 个点。若规范化抛异常，本测试应改成
    断言该异常类型，并在 M5 文档写明变长题暂不支持组合；这里先走成功分支。
    """
    contract = bind_task("varlen_kda", "fp16", "v6_h64_fixed", target_id=KDA_TARGET).locator.load()
    target = load_target(KDA_TARGET)
    sizes = ("v6_h64_fixed", "v6_h96_fixed")
    instances = tuple(instantiate_v2(contract.spec, contract.dtype, size) for size in sizes)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    assert len(plan.instances) == 4
    assert {point.size for point in plan.instances} == set(sizes)
    assert {point.usage for point in plan.instances} == {"golden", "benchmark"}


_TARGET = "apple-m4-max"
_SINGLE_KERNEL = """\
#include <metal_stdlib>
using namespace metal;
kernel void placeholder(device half* out [[buffer(0)]], uint i [[thread_position_in_grid]]) {}
/* taic-launch
math safe
launch placeholder grid=1,1,1 threads=1,1,1
*/
"""
_CLOSURE_SHA256 = "4732e2a5aa1010d212c17cbdac03bf0b3f6eed9e35fdf123bcc1712c64b57e14"


@lru_cache(maxsize=None)
def _contract(task: str, size: str):
    """@brief 缓存一次 bind，避免每条测试都重新发现任务。

    @param task 题名。
    @param size 合同实例的 size。
    @return 任务合同。
    """
    return bind_task(task, "fp16", size, target_id=_TARGET).locator.load()


def _domain(task: str, sizes: tuple[str, ...] | None = None):
    """@brief 造组装域。合同实例取 size 列表的第一档，保证它落在域里。

    @param task 题名。
    @param sizes 显式 size；缺省用带 benchmark 的档。
    @return ``(contract, domain)``。
    """
    probe = _contract(task, "small")
    chosen = sizes if sizes is not None else benchmark_sizes(probe.spec, probe.dtype)
    contract = probe if probe.size in chosen else _contract(task, chosen[0])
    domain = domain_for_sizes(contract, load_target(_TARGET), chosen)
    return contract, domain


def _coverage(root: Path, variants: list[dict], fallback: dict, *, planner_id: str = "test") -> None:
    """@brief 写一份 coverage.json。

    @param root 源目录。
    @param variants variant 条目。
    @param fallback fallback 对象。
    @param planner_id planner 名字。
    """
    document = {
        "schema_id": "tcenv.portfolio-assembly.coverage.1",
        "planner": {"id": planner_id, "version": 1},
        "variants": variants,
        "fallback": fallback,
    }
    (root / "coverage.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _write_kernel(root: Path, key: str, source: str) -> None:
    """@brief 写一个 single 变体。

    @param root 源目录。
    @param key variant_key。
    @param source MSL 源码。
    """
    path = root / "variants" / key / "kernel.metal"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_gemm_benchmark_sizes_follow_the_spec_order():
    """@brief gemm fp16 的 benchmark 档按 OpSpec 顺序是 paper、small。"""
    contract = _contract("gemm", "small")
    assert benchmark_sizes(contract.spec, "fp16") == ("paper", "small")


def test_task_pack_closure_is_unchanged_by_assembly():
    """@brief 组装模块不进任务包闭包，闭包哈希仍是 M0 记录的值。"""
    from tcenv.taskcontract.pack import code_import_closure

    root = Path(__file__).resolve().parents[2]
    files = code_import_closure(root)
    digest = hashlib.sha256(b"".join((root / name).read_bytes() for name in files)).hexdigest()
    assert len(files) == 74
    assert digest == _CLOSURE_SHA256
    assert all("portfolio_assembly" not in name for name in files)


def test_assembly_sources_do_not_import_backends_or_taic():
    """@brief 组装包自己的源码不导入后端或 taic。"""
    package = Path(__file__).resolve().parents[2] / "tcenv" / "portfolio_assembly"
    offenders = []
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name == "taic" or name.startswith("taic.") or name == "tcenv.backends" or name.startswith("tcenv.backends."):
                    offenders.append(f"{path.name}:{node.lineno}:{name}")
    assert offenders == []


def test_gemm_single_variants_pack_decode_and_check(tmp_path):
    """@brief gemm 两个 single 变体打包后逐点命中，投影都是 msl-single。"""
    contract, domain = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    entries = parent_entries_from_tree(root, domain)
    assert entries[-1].path == "coverage_plan/portfolio.json"
    assert [entry.path for entry in entries[:-1]] == sorted(entry.path for entry in entries[:-1])
    text = encode_parent(entries)
    again = encode_parent(parent_entries_from_tree(root, domain))
    assert text == again
    report = check_parent(decode_parent(text), domain)
    assert report.ok, report.problems
    assert len(report.rows) == 4
    assert {row.track_id for row in report.rows} == {"msl-single"}
    for row in report.rows:
        assert row.provider == "candidate"
        assert row.variant_key == row.size


def test_kda_bundle_variants_project_to_msl_bundle(tmp_path):
    """@brief kda 两个 bundle 变体检查通过，Host 程序是 tiling.c。"""
    _contract_kda, domain = _domain("kda")
    root = tmp_path / "src"
    write_kda_tree(root)
    report = check_parent(decode_parent(encode_parent(parent_entries_from_tree(root, domain))), domain)
    assert report.ok, report.problems
    assert report.rows
    assert {row.track_id for row in report.rows} == {"msl-bundle"}
    portfolio = parse_portfolio(decode_parent(encode_parent(parent_entries_from_tree(root, domain))), domain_plan=domain.plan)
    for variant in portfolio.variants:
        assert any(path.endswith("host/tiling.c") for path in variant.entries)
        assert variant.geometry_entry is not None
        assert variant.geometry_entry.endswith("host/execution_geometry.json")
        assert variant.geometry_entry not in variant.entries


def test_one_parent_can_mix_a_single_and_a_bundle(tmp_path):
    """@brief 同一个父包里可以同时有 single 和 bundle，两种 track 都出现。"""
    _contract_kda, domain = _domain("kda")
    root = tmp_path / "src"
    base = root / "variants" / "a"
    for entry in bundle_entries():
        destination = base / entry.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(entry.data)
    _write_kernel(root, "b", _SINGLE_KERNEL)
    _coverage(
        root,
        [
            {"variant_key": "a", "guard": ["lt", "N", 100]},
            {"variant_key": "b", "guard": ["ge", "N", 100]},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    report = check_parent(decode_parent(encode_parent(parent_entries_from_tree(root, domain))), domain)
    assert report.ok, report.problems
    assert {row.track_id for row in report.rows} == {"msl-single", "msl-bundle"}


def test_a_gemm_bundle_variant_is_rejected_for_missing_workspace(tmp_path):
    """@brief gemm 没有 workspace，bundle 变体在检查里失败，问题提到 workspace。"""
    _contract_gemm, domain = _domain("gemm", ("small",))
    root = tmp_path / "src"
    device = root / "variants" / "small" / "device" / "k.metal"
    device.parent.mkdir(parents=True)
    device.write_text(_SINGLE_KERNEL, encoding="utf-8")
    host = root / "variants" / "small" / "host"
    host.mkdir()
    (host / "tiling.c").write_text("int placeholder(void) { return 0; }\n", encoding="utf-8")
    (host / "execution_geometry.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "candidate",
                "stages": [
                    {
                        "kernel": "placeholder",
                        "grid": [1, 1, 1],
                        "threads": [1, 1, 1],
                        "threadgroup_bytes": 0,
                        "workspace": [],
                        "when": None,
                        "derived_scalars": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _coverage(
        root,
        [{"variant_key": "small", "guard": ["ge", "M", 1]}],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    report = check_parent(parent_entries_from_tree(root, domain), domain)
    assert not report.ok
    assert any("workspace" in problem for problem in report.problems)


def test_overlapping_guards_name_the_point(tmp_path):
    """@brief 两个变体同时接住同一个点时，问题里点名那个点。"""
    _contract_gemm, domain = _domain("gemm")
    root = tmp_path / "src"
    _write_kernel(root, "small", _SINGLE_KERNEL)
    _write_kernel(root, "paper", _SINGLE_KERNEL)
    _coverage(
        root,
        [
            {"variant_key": "small", "guard": ["ge", "M", 1]},
            {"variant_key": "paper", "guard": ["ge", "M", 1]},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    report = check_parent(parent_entries_from_tree(root, domain), domain)
    assert not report.ok
    assert any("fp16/small/benchmark" in problem and "两个以上变体同时命中" in problem for problem in report.problems)


def test_an_uncovered_point_needs_an_implementation_fallback(tmp_path):
    """@brief refuse 接不住的点失败；换成实现型 fallback 后该点 provider 是 fallback。"""
    contract, domain = _domain("gemm")
    root = tmp_path / "src"
    _write_kernel(root, "small", gemm_variant_source(contract, "small"))
    _coverage(
        root,
        [{"variant_key": "small", "guard": SMALL_GUARD}],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    refused = check_parent(parent_entries_from_tree(root, domain), domain)
    assert not refused.ok
    assert any("fp16/paper/benchmark" in problem and "没人接" in problem for problem in refused.problems)

    fallback = root / "fallback" / "kernel.metal"
    fallback.parent.mkdir()
    fallback.write_text(gemm_variant_source(contract, "paper"), encoding="utf-8")
    _coverage(
        root,
        [{"variant_key": "small", "guard": SMALL_GUARD}],
        {
            "kind": "registered_implementation",
            "implementation_id": "gemm.generic",
            "implementation_version": 1,
            "reasons": ["outside_candidate_coverage"],
        },
    )
    accepted = check_parent(parent_entries_from_tree(root, domain), domain)
    assert accepted.ok, accepted.problems
    paper = [row for row in accepted.rows if row.size == "paper"]
    assert paper
    assert {row.provider for row in paper} == {"fallback"}


def test_a_variant_that_matches_nothing_is_rejected(tmp_path):
    """@brief 一个变体没有命中任何点时检查失败。"""
    contract, domain = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    _write_kernel(root, "idle", _SINGLE_KERNEL)
    _coverage(
        root,
        [
            {"variant_key": "small", "guard": SMALL_GUARD},
            {"variant_key": "paper", "guard": PAPER_GUARD},
            {"variant_key": "idle", "guard": ["eq", "M", 3]},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    report = check_parent(parent_entries_from_tree(root, domain), domain)
    assert not report.ok
    assert any("idle" in problem and "没有命中任何点" in problem for problem in report.problems)


def test_an_unknown_guard_symbol_is_a_problem_not_an_exception(tmp_path):
    """@brief guard 用了不存在的符号时检查失败，不把异常抛出。"""
    _contract_gemm, domain = _domain("gemm")
    root = tmp_path / "src"
    _write_kernel(root, "small", _SINGLE_KERNEL)
    _coverage(
        root,
        [{"variant_key": "small", "guard": ["eq", "NOPE", 1]}],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    report = check_parent(parent_entries_from_tree(root, domain), domain)
    assert not report.ok
    assert any("NOPE" in problem for problem in report.problems)


def test_coverage_and_tree_layout_errors_are_assembly_errors(tmp_path):
    """@brief coverage 多键、目录对不上、single/bundle 形状冲突都抛组装错误，并带上路径。"""
    contract, domain = _domain("gemm")
    root = tmp_path / "extra"
    write_gemm_tree(root, contract)
    document = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    document["note"] = "no"
    (root / "coverage.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PortfolioAssemblyError, match="coverage.json"):
        parent_entries_from_tree(root, domain)

    mismatched = tmp_path / "names"
    write_gemm_tree(mismatched, contract)
    document = json.loads((mismatched / "coverage.json").read_text(encoding="utf-8"))
    document["variants"][0]["variant_key"] = "other"
    (mismatched / "coverage.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PortfolioAssemblyError, match="variants"):
        parent_entries_from_tree(mismatched, domain)

    mixed = tmp_path / "mixed"
    write_gemm_tree(mixed, contract)
    device = mixed / "variants" / "small" / "device" / "extra.metal"
    device.parent.mkdir()
    device.write_text("nope\n", encoding="utf-8")
    with pytest.raises(PortfolioAssemblyError, match="kernel.metal"):
        parent_entries_from_tree(mixed, domain)

    both = tmp_path / "both-host"
    _bundle_body(both / "variants" / "small", geometry=True, tiling_json=True)
    _write_kernel(both, "paper", _SINGLE_KERNEL)
    _coverage(
        both,
        [
            {"variant_key": "small", "guard": SMALL_GUARD},
            {"variant_key": "paper", "guard": PAPER_GUARD},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    with pytest.raises(PortfolioAssemblyError, match="tiling.c"):
        parent_entries_from_tree(both, domain)

    missing = tmp_path / "no-geometry"
    _bundle_body(missing / "variants" / "small", geometry=False, tiling_json=False)
    _write_kernel(missing, "paper", _SINGLE_KERNEL)
    _coverage(
        missing,
        [
            {"variant_key": "small", "guard": SMALL_GUARD},
            {"variant_key": "paper", "guard": PAPER_GUARD},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    with pytest.raises(PortfolioAssemblyError, match="execution_geometry.json"):
        parent_entries_from_tree(missing, domain)


@pytest.mark.parametrize(
    "reasons",
    [[[]], ["outside_candidate_coverage", 7]],
    ids=["nested-list", "non-string-item"],
)
def test_fallback_reason_items_are_validated_before_hashing(tmp_path, reasons):
    """@brief 非字符串 reasons（含不可哈希嵌套列表）都转成 typed 组装错误。"""
    contract, domain = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    document = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    document["fallback"]["reasons"] = reasons
    (root / "coverage.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(PortfolioAssemblyError, match="reasons"):
        parent_entries_from_tree(root, domain)


def _bundle_body(directory: Path, *, geometry: bool, tiling_json: bool) -> None:
    """@brief 写一个最小 bundle 目录，几何和 tiling.json 可选。

    @param directory 变体目录。
    @param geometry 是否写几何文件。
    @param tiling_json 是否同时写 tiling.json。
    """
    (directory / "device").mkdir(parents=True)
    (directory / "device" / "k.metal").write_text("k\n", encoding="utf-8")
    host = directory / "host"
    host.mkdir()
    (host / "tiling.c").write_text("int placeholder(void) { return 0; }\n", encoding="utf-8")
    if tiling_json:
        (host / "tiling.json").write_text("{}\n", encoding="utf-8")
    if geometry:
        (host / "execution_geometry.json").write_text("{}\n", encoding="utf-8")


def test_portfolio_pack_reports_exit_codes(tmp_path, capsys):
    """@brief portfolio-pack 成功为 0，编码器拒绝为 1，布局错误为 2 且不写文件。"""
    contract, _domain_gemm = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    out = tmp_path / "parent.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(root), "--out", str(out), "--dtype", "fp16"]) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "sizes: paper,small"
    assert out.is_file()

    missing_out = tmp_path / "missing.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(tmp_path / "absent"), "--out", str(missing_out), "--dtype", "fp16"]) == 2
    assert not missing_out.exists()

    huge = tmp_path / "huge"
    _write_kernel(huge, "small", "a" * 1_100_000)
    _coverage(
        huge,
        [{"variant_key": "small", "guard": ["ge", "M", 1]}],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    huge_out = tmp_path / "huge.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(huge), "--out", str(huge_out), "--dtype", "fp16", "--sizes", "small"]) == 1
    assert not huge_out.exists()


def test_portfolio_check_reports_exit_codes(tmp_path, capsys):
    """@brief portfolio-check 通过为 0，有问题为 1，文件不存在为 2。"""
    contract, _domain_gemm = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    out = tmp_path / "parent.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(root), "--out", str(out), "--dtype", "fp16"]) == 0
    capsys.readouterr()
    assert main(["portfolio-check", "gemm", str(out), "--dtype", "fp16"]) == 0
    checked = capsys.readouterr().out
    assert checked.splitlines()[0] == "sizes: paper,small"
    assert "RESULT: ok" in checked

    overlap = tmp_path / "overlap"
    _write_kernel(overlap, "small", _SINGLE_KERNEL)
    _write_kernel(overlap, "paper", _SINGLE_KERNEL)
    _coverage(
        overlap,
        [
            {"variant_key": "small", "guard": ["ge", "M", 1]},
            {"variant_key": "paper", "guard": ["ge", "M", 1]},
        ],
        {"kind": "refuse", "reasons": ["outside_candidate_coverage"]},
    )
    overlap_out = tmp_path / "overlap.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(overlap), "--out", str(overlap_out), "--dtype", "fp16"]) == 0
    capsys.readouterr()
    assert main(["portfolio-check", "gemm", str(overlap_out), "--dtype", "fp16"]) == 1
    assert "RESULT: rejected" in capsys.readouterr().out

    assert main(["portfolio-check", "gemm", str(tmp_path / "absent.envelope"), "--dtype", "fp16"]) == 2


def test_check_with_different_sizes_names_the_mismatch(tmp_path, capsys):
    """@brief 打包和检查的 --sizes 不一致时，失败消息要求两边相同。"""
    contract, _domain_gemm = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    out = tmp_path / "parent.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(root), "--out", str(out), "--dtype", "fp16", "--sizes", "small"]) == 0
    capsys.readouterr()
    assert main(["portfolio-check", "gemm", str(out), "--dtype", "fp16"]) == 1
    assert "相同的 --sizes" in capsys.readouterr().out


def test_portfolio_check_json_is_only_the_document(tmp_path, capsys):
    """@brief ``--json`` 只打印报告文档。"""
    contract, _domain_gemm = _domain("gemm")
    root = tmp_path / "src"
    write_gemm_tree(root, contract)
    out = tmp_path / "parent.envelope"
    assert main(["portfolio-pack", "gemm", "--src", str(root), "--out", str(out), "--dtype", "fp16"]) == 0
    capsys.readouterr()
    assert main(["portfolio-check", "gemm", str(out), "--dtype", "fp16", "--json"]) == 0
    text = capsys.readouterr().out
    document = json.loads(text)
    assert document["ok"] is True
    assert document["rows"]
    assert "sizes:" not in text
