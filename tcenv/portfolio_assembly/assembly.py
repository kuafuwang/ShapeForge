"""Assemble a shape-portfolio parent envelope from a source tree.

The module is backend-neutral: it reads a coverage document and variant files, builds the
parent entries, and checks guards plus projection.  It does not build, time, or activate
anything.  Callers in ``tcenv.cli``, the Metal evaluator, and the search driver import it.
It must not import ``tcenv.backends`` or ``taic``, and the task-pack closure must not
import this package.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from tcenv.contract_errors import GeometryError, TaskContractError
from tcenv.hostprogram.errors import HostProgramError
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import (
    FALLBACK_REASONS,
    PORTFOLIO_ENTRY_MEDIA_TYPE,
    PORTFOLIO_ENTRY_PATH,
    PORTFOLIO_ENTRY_ROLE,
    PORTFOLIO_SCHEMA_ID,
    parse_portfolio,
    project_provider,
)
from tcenv.portfolio.domain import DomainAuthority, SemanticDomainPlan, guard_digest, normalize_semantic_domain_plan
from tcenv.track.envelope import decode_envelope, encode_envelope
from tcenv.track.plane import CandidateArtifact, CandidateEntry

PARENT_TRANSPORT_ID = "shape-portfolio"
COVERAGE_SCHEMA_ID = "tcenv.portfolio-assembly.coverage.1"
MAX_VARIANTS = 16
_VARIANT_KEY = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")
_COVERAGE_KEYS = frozenset({"schema_id", "planner", "variants", "fallback"})
_VARIANT_ITEM_KEYS = frozenset({"variant_key", "guard", "bucket_key"})
_REFUSE_KEYS = frozenset({"kind", "reasons"})
_IMPLEMENTATION_KEYS = frozenset({"kind", "reasons", "implementation_id", "implementation_version"})
_GEOMETRY_NAME = "execution_geometry.json"


class PortfolioAssemblyError(TaskContractError):
    """@brief 源目录或 ``coverage.json`` 不符合组装约定。

    不另设 ``code``。沿用基类的 ``task_contract_invalid``，失败码表不会多一行。
    """


@dataclass(frozen=True)
class AssemblyDomain:
    """@brief 一次打包/检查使用的有限域。"""

    plan: SemanticDomainPlan
    authority: DomainAuthority
    instances: tuple[Any, ...]
    sizes: tuple[str, ...]


@dataclass(frozen=True)
class CheckRow:
    """@brief 一个域点的检查结果。"""

    instance_key: str
    size: str
    usage: str
    provider: str
    variant_key: str | None
    track_id: str | None


@dataclass(frozen=True)
class CheckReport:
    """@brief ``check_parent`` 的全部问题、逐点行和身份摘要。"""

    ok: bool
    problems: tuple[str, ...]
    rows: tuple[CheckRow, ...]
    domain_plan_digest: str
    portfolio_digest: str | None
    parent_sha256: str | None

    def document(self) -> dict[str, Any]:
        """@brief 报告的 JSON 文档。

        @return 可序列化字段。
        """
        return {
            "ok": self.ok,
            "problems": list(self.problems),
            "rows": [
                {
                    "instance_key": row.instance_key,
                    "size": row.size,
                    "usage": row.usage,
                    "provider": row.provider,
                    "variant_key": row.variant_key,
                    "track_id": row.track_id,
                }
                for row in self.rows
            ],
            "domain_plan_digest": self.domain_plan_digest,
            "portfolio_digest": self.portfolio_digest,
            "parent_sha256": self.parent_sha256,
        }


def benchmark_sizes(spec, dtype: str) -> tuple[str, ...]:
    """@brief 按 OpSpec 声明顺序取出带 benchmark 用途的 size。

    实例化抛 :class:`TaskContractError` 的档跳过。

    @param spec OpSpec。
    @param dtype 数据类型。
    @return size 名字。
    """
    found: list[str] = []
    for size in list(spec.sizes):
        try:
            instance = instantiate_v2(spec, dtype, size)
        except TaskContractError:
            continue
        if "benchmark" in instance.size_usage:
            found.append(str(size))
    return tuple(found)


def domain_for_sizes(contract, target, sizes: Sequence[str]) -> AssemblyDomain:
    """@brief 用给定 size 造域计划、权威对象和实例。

    @param contract 任务合同。其默认实例必须是其中一个 size。
    @param target 目标档案。
    @param sizes 域里的 size，不能空、不能重复。
    @return 组装域。
    @raises PortfolioAssemblyError size 为空或重复。
    """
    chosen = tuple(sizes)
    if not chosen:
        raise PortfolioAssemblyError("sizes must name at least one size")
    if len(set(chosen)) != len(chosen):
        raise PortfolioAssemblyError(f"sizes {chosen} repeat a name")
    instances = tuple(instantiate_v2(contract.spec, contract.dtype, size) for size in chosen)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=target)
    authority = DomainAuthority(
        plan=plan, spec=contract.spec, target=target, instances=instances, contract=contract
    )
    return AssemblyDomain(plan=authority.plan, authority=authority, instances=instances, sizes=chosen)


def parent_entries_from_tree(src: Path, domain: AssemblyDomain) -> tuple[CandidateEntry, ...]:
    """@brief 读源目录，生成父包 entries。

    返回值按路径排序，``coverage_plan/portfolio.json`` 放在最后。布局错误抛
    :class:`PortfolioAssemblyError`，消息里带出错路径。

    @param src 含 ``coverage.json`` 和 ``variants/`` 的目录。
    @param domain 这次打包绑定的域。
    @return 父 entries。
    @raises PortfolioAssemblyError 目录、coverage 或文件字节不合法。
    """
    root = Path(src)
    coverage_path = root / "coverage.json"
    coverage = _load_coverage(coverage_path)
    directories = _variant_directories(root / "variants")
    declared = _coverage_variants(coverage, coverage_path)
    if set(declared) != set(directories):
        raise PortfolioAssemblyError(
            f"{coverage_path}: variant_key set {sorted(declared)} does not match {root / 'variants'} "
            f"directories {sorted(directories)}"
        )
    shapes = {key: _classify_directory(root / "variants" / key, f"variants/{key}") for key in declared}
    fallback_doc, fallback_shape = _fallback_shape(root, coverage["fallback"], coverage_path)
    document = _portfolio_document(coverage, coverage_path, domain.plan.digest, shapes, fallback_doc, fallback_shape)
    entries = []
    for key in declared:
        entries.extend(_materialize(root, shapes[key]))
    if fallback_shape is not None:
        entries.extend(_materialize(root, fallback_shape))
    entries.sort(key=lambda entry: entry.path)
    entries.append(_manifest_entry(document))
    return tuple(entries)


def encode_parent(entries: Sequence[CandidateEntry]) -> str:
    """@brief 把父 entries 编成 ``shape-portfolio`` envelope。

    @param entries 父 entries。
    @return envelope 文本。
    @raises EnvelopeError 编码器拒绝（过大、过多或单文件）。
    """
    return encode_envelope(CandidateArtifact(track_id=PARENT_TRANSPORT_ID, entries=tuple(entries)))


def decode_parent(text: str) -> tuple[CandidateEntry, ...]:
    """@brief 解开父包。track 必须是 ``shape-portfolio``。

    @param text envelope 文本。
    @return entries。
    @raises EnvelopeError 不是一份 envelope。
    @raises PortfolioAssemblyError track 不是父包传输标记。
    """
    candidate = decode_envelope(text)
    if candidate.track_id != PARENT_TRANSPORT_ID:
        raise PortfolioAssemblyError(
            f"parent envelope track is {candidate.track_id!r}, expected {PARENT_TRANSPORT_ID!r}"
        )
    return candidate.entries


def write_text_atomic(out: Path, text: str) -> None:
    """@brief 先写同目录临时文件，再 ``os.replace`` 到目标路径。

    @param out 目标文件。
    @param text 全文。
    @raises OSError 目录不存在或替换失败。
    """
    destination = Path(out)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as temporary:
            temporary.write(text)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise


def check_parent(entries, domain: AssemblyDomain) -> CheckReport:
    """@brief 在 CPU 上检查 guard 覆盖和每个变体的投影。不抛检查失败。

    解析失败时 ``ok`` 为假、``rows`` 为空。域摘要对不上时，问题后面补一句打包和检查要传
    相同的 ``--sizes``。

    @param entries 父 entries。
    @param domain 检查使用的域，必须和打包时同一组 size。
    @return 检查报告。
    """
    digest = domain.plan.digest
    try:
        portfolio = parse_portfolio(entries, domain_plan=domain.plan)
    except (TaskContractError, GeometryError) as exc:
        message = str(exc)
        if "domain plan" in message and "not the supplied" in message:
            message = f"{message} 打包和检查要传相同的 --sizes"
        return CheckReport(
            ok=False,
            problems=(message,),
            rows=(),
            domain_plan_digest=digest,
            portfolio_digest=None,
            parent_sha256=None,
        )
    report = portfolio.registration_report(
        domain.plan, domain.instances, spec=domain.authority.spec, target=domain.authority.target
    )
    names = {variant.variant_id: variant.variant_key for variant in portfolio.variants}
    problems: list[str] = []
    for key in report.overlaps:
        problems.append(f"{key}: 两个以上变体同时命中")
    problems.extend(report.unknown_guards)
    for variant_id in report.empty_variants:
        problems.append(f"{names[variant_id]}: 没有命中任何点")

    tracks: dict[str, str] = {}
    projection_problems: list[str] = []
    for variant in portfolio.variants:
        try:
            projection = project_provider(portfolio, variant, authority=domain.authority)
        except (TaskContractError, HostProgramError) as exc:
            projection_problems.append(f"{variant.variant_key}: {exc}")
        else:
            tracks[variant.variant_id] = projection.track_id
    fallback = portfolio.fallback
    fallback_impl = fallback is not None and fallback.kind == "registered_implementation"
    fallback_track: str | None = None
    if fallback_impl:
        try:
            projection = project_provider(portfolio, fallback, authority=domain.authority)
        except (TaskContractError, HostProgramError) as exc:
            projection_problems.append(str(exc))
        else:
            fallback_track = projection.track_id

    points = {point.instance_key: point for point in domain.plan.instances}
    rows: list[CheckRow] = []
    for item in report.points:
        point = points[item.instance_key]
        matched = item.matched
        if len(matched) == 1:
            variant_id = matched[0]
            rows.append(
                CheckRow(
                    instance_key=item.instance_key,
                    size=point.size,
                    usage=point.usage,
                    provider="candidate",
                    variant_key=names[variant_id],
                    track_id=tracks.get(variant_id),
                )
            )
            continue
        if len(matched) == 0 and fallback_impl:
            rows.append(
                CheckRow(
                    instance_key=item.instance_key,
                    size=point.size,
                    usage=point.usage,
                    provider="fallback",
                    variant_key=None,
                    track_id=fallback_track,
                )
            )
            continue
        if len(matched) == 0:
            problems.append(f"{item.instance_key}: 这个点没人接")
        rows.append(
            CheckRow(
                instance_key=item.instance_key,
                size=point.size,
                usage=point.usage,
                provider="none",
                variant_key=None,
                track_id=None,
            )
        )
    problems.extend(projection_problems)
    return CheckReport(
        ok=not problems,
        problems=tuple(problems),
        rows=tuple(rows),
        domain_plan_digest=digest,
        portfolio_digest=portfolio.digest,
        parent_sha256=portfolio.parent_sha256,
    )


@dataclass(frozen=True)
class _Shape:
    """@brief 一个变体或 fallback 目录的文件清单。"""

    prefix: str
    kind: str
    sources: tuple[str, ...]
    geometry: str | None


def _load_coverage(path: Path) -> dict[str, Any]:
    """@brief 读取并校验 ``coverage.json`` 的闭集键。

    @param path 文件路径。
    @return 文档。
    @raises PortfolioAssemblyError 读不到、不是对象或键不对。
    """
    text = _read_utf8(path)
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PortfolioAssemblyError(f"{path}: coverage.json is not JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise PortfolioAssemblyError(f"{path}: coverage.json must be an object")
    extra = set(document) - _COVERAGE_KEYS
    missing = _COVERAGE_KEYS - set(document)
    if extra or missing:
        raise PortfolioAssemblyError(
            f"{path}: coverage.json keys must be exactly {sorted(_COVERAGE_KEYS)}, "
            f"extra={sorted(extra)} missing={sorted(missing)}"
        )
    if document["schema_id"] != COVERAGE_SCHEMA_ID:
        raise PortfolioAssemblyError(
            f"{path}: schema_id must be {COVERAGE_SCHEMA_ID!r}, got {document['schema_id']!r}"
        )
    _require_planner(document["planner"], path)
    if not isinstance(document["variants"], list) or not document["variants"]:
        raise PortfolioAssemblyError(f"{path}: variants must be a non-empty list")
    if len(document["variants"]) > MAX_VARIANTS:
        raise PortfolioAssemblyError(f"{path}: a portfolio has at most {MAX_VARIANTS} variants")
    return document


def _require_planner(planner: Any, path: Path) -> None:
    """@brief 校验 planner 只有 id 和正整数 version。

    @param planner coverage 里的 planner。
    @param path 诊断路径。
    @raises PortfolioAssemblyError 形状不对。
    """
    if not isinstance(planner, dict) or set(planner) != {"id", "version"}:
        raise PortfolioAssemblyError(f"{path}: planner must contain exactly id and version")
    if not isinstance(planner["id"], str) or not planner["id"].strip() or len(planner["id"]) > 128:
        raise PortfolioAssemblyError(f"{path}: planner.id must be a non-empty string of at most 128 characters")
    version = planner["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PortfolioAssemblyError(f"{path}: planner.version must be a positive integer")


def _variant_directories(root: Path) -> dict[str, Path]:
    """@brief ``variants/`` 下的一层目录，名字就是 variant_key。

    @param root ``variants`` 目录。
    @return 名字到路径。
    @raises PortfolioAssemblyError 缺目录、名字非法或超过 16 个。
    """
    if not root.is_dir() or root.is_symlink():
        raise PortfolioAssemblyError(f"{root}: expected a variants directory")
    found: dict[str, Path] = {}
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if child.name.startswith("."):
            continue
        if not child.is_dir() or child.is_symlink():
            raise PortfolioAssemblyError(f"{child}: a variant must be a directory")
        if not _VARIANT_KEY.fullmatch(child.name):
            raise PortfolioAssemblyError(
                f"{child}: variant_key must match [a-z0-9][a-z0-9_-]{{0,31}}"
            )
        found[child.name] = child
    if not found:
        raise PortfolioAssemblyError(f"{root}: a portfolio needs at least one variant directory")
    if len(found) > MAX_VARIANTS:
        raise PortfolioAssemblyError(f"{root}: a portfolio has at most {MAX_VARIANTS} variants")
    return found


def _coverage_variants(document: dict[str, Any], path: Path) -> dict[str, dict[str, Any]]:
    """@brief 校验 variants 数组并按 variant_key 索引。

    @param document coverage 文档。
    @param path 诊断路径。
    @return variant_key 到条目。
    @raises PortfolioAssemblyError 条目形状或名字非法。
    """
    declared: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(document["variants"]):
        where = f"{path}: variants[{index}]"
        if not isinstance(item, dict):
            raise PortfolioAssemblyError(f"{where} must be an object")
        extra = set(item) - _VARIANT_ITEM_KEYS
        if extra or "variant_key" not in item or "guard" not in item:
            raise PortfolioAssemblyError(
                f"{where} allows only variant_key, guard, and optional bucket_key; got {sorted(item)}"
            )
        key = item["variant_key"]
        if not isinstance(key, str) or _VARIANT_KEY.fullmatch(key) is None:
            raise PortfolioAssemblyError(f"{where}: variant_key {key!r} must match [a-z0-9][a-z0-9_-]{{0,31}}")
        if key in declared:
            raise PortfolioAssemblyError(f"{path}: variant_key {key!r} is repeated")
        bucket = item.get("bucket_key", key)
        if not isinstance(bucket, str) or not bucket.strip() or len(bucket) > 128:
            raise PortfolioAssemblyError(f"{where}: bucket_key must be a non-empty string of at most 128 characters")
        declared[key] = item
    return declared


def _fallback_shape(root: Path, fallback: Any, coverage_path: Path):
    """@brief 校验 fallback 对象，并在实现型时分类 ``fallback/`` 目录。

    @param root 源目录。
    @param fallback coverage 里的 fallback。
    @param coverage_path 诊断路径。
    @return ``(fallback 文档, shape 或 None)``。
    @raises PortfolioAssemblyError 种类、目录或原因非法。
    """
    where = f"{coverage_path}: fallback"
    if not isinstance(fallback, dict) or "kind" not in fallback:
        raise PortfolioAssemblyError(f"{where} must be an object with kind")
    kind = fallback["kind"]
    reasons = _reasons(fallback.get("reasons"), where)
    directory = root / "fallback"
    if kind == "refuse":
        if set(fallback) != _REFUSE_KEYS:
            raise PortfolioAssemblyError(f"{where} refuse allows only kind and reasons, got {sorted(fallback)}")
        if directory.exists():
            raise PortfolioAssemblyError(f"{directory}: refuse fallback must not have a fallback directory")
        document = {
            "kind": "refuse",
            "implementation_id": None,
            "implementation_version": None,
            "entries": [],
            "geometry_entry": None,
            "reasons": reasons,
        }
        return document, None
    if kind != "registered_implementation":
        raise PortfolioAssemblyError(f"{where}: unknown fallback kind {kind!r}")
    if set(fallback) != _IMPLEMENTATION_KEYS:
        raise PortfolioAssemblyError(
            f"{where} registered_implementation requires kind, reasons, implementation_id, "
            f"and implementation_version; got {sorted(fallback)}"
        )
    implementation_id = fallback["implementation_id"]
    version = fallback["implementation_version"]
    if not isinstance(implementation_id, str) or not implementation_id.strip() or len(implementation_id) > 128:
        raise PortfolioAssemblyError(f"{where}: implementation_id must be a non-empty string")
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise PortfolioAssemblyError(f"{where}: implementation_version must be a positive integer")
    if not directory.is_dir():
        raise PortfolioAssemblyError(f"{directory}: registered_implementation fallback needs this directory")
    shape = _classify_directory(directory, "fallback")
    return fallback, shape


def _reasons(value: Any, where: str) -> list[str]:
    """@brief 校验 fallback 原因是登记过的非空列表。

    @param value coverage 里的 reasons。
    @param where 诊断位置。
    @return 原因列表。
    @raises PortfolioAssemblyError 非法。
    """
    if not isinstance(value, list) or not value:
        raise PortfolioAssemblyError(f"{where}: reasons must be a non-empty list")
    if any(not isinstance(item, str) for item in value):
        raise PortfolioAssemblyError(f"{where}: reasons items must all be strings, got {value!r}")
    if len(set(value)) != len(value):
        raise PortfolioAssemblyError(f"{where}: reasons repeats an item")
    unknown = [item for item in value if item not in FALLBACK_REASONS]
    if unknown:
        raise PortfolioAssemblyError(
            f"{where}: reasons must be chosen from {sorted(FALLBACK_REASONS)}, got {value!r}"
        )
    return list(value)


def _classify_directory(directory: Path, prefix: str) -> _Shape:
    """@brief 把一个变体或 fallback 目录分成 single 或 bundle。

    @param directory 目录。
    @param prefix 父包里的相对前缀。
    @return 形状。
    @raises PortfolioAssemblyError 两种形状都不满足，或文件多出来。
    """
    files = _files_under(directory)
    relative = [path.relative_to(directory).as_posix() for path in files]
    has_kernel = "kernel.metal" in relative
    has_device = any(name == "device" or name.startswith("device/") for name in relative) or (directory / "device").exists()
    has_host = any(name == "host" or name.startswith("host/") for name in relative) or (directory / "host").exists()
    if has_kernel:
        if has_device or has_host or relative != ["kernel.metal"]:
            raise PortfolioAssemblyError(
                f"{directory / 'kernel.metal'}: a single variant is only kernel.metal and cannot also contain "
                f"{directory / 'device'} or {directory / 'host'}"
            )
        return _Shape(prefix=prefix, kind="single", sources=(f"{prefix}/kernel.metal",), geometry=None)
    device = sorted(name for name in relative if name.startswith("device/") and name.endswith(".metal"))
    host_c = "host/tiling.c" in relative
    host_json = "host/tiling.json" in relative
    geometry = "host/execution_geometry.json" in relative
    expected = set(device)
    if host_c:
        expected.add("host/tiling.c")
    if host_json:
        expected.add("host/tiling.json")
    if geometry:
        expected.add("host/execution_geometry.json")
    unexpected = sorted(set(relative) - expected)
    if unexpected:
        raise PortfolioAssemblyError(f"{directory / unexpected[0]}: unexpected file in a bundle variant")
    if not device:
        raise PortfolioAssemblyError(f"{directory}: a bundle variant needs at least one device/**/*.metal file")
    if host_c and host_json:
        raise PortfolioAssemblyError(
            f"{directory / 'host' / 'tiling.c'}: a bundle variant has both tiling.c and "
            f"{directory / 'host' / 'tiling.json'}; it needs exactly one Host program"
        )
    if not host_c and not host_json:
        raise PortfolioAssemblyError(f"{directory / 'host'}: a bundle variant needs host/tiling.c or host/tiling.json")
    if not geometry:
        raise PortfolioAssemblyError(
            f"{directory / 'host' / _GEOMETRY_NAME}: a bundle variant needs host/execution_geometry.json"
        )
    host_name = "host/tiling.c" if host_c else "host/tiling.json"
    sources = tuple(sorted(f"{prefix}/{name}" for name in (*device, host_name)))
    return _Shape(prefix=prefix, kind="bundle", sources=sources, geometry=f"{prefix}/host/{_GEOMETRY_NAME}")


def _files_under(directory: Path) -> list[Path]:
    """@brief 目录下的普通文件，拒绝符号链接。

    @param directory 根。
    @return 排序后的文件。
    @raises PortfolioAssemblyError 遇到符号链接。
    """
    found: list[Path] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise PortfolioAssemblyError(f"{path}: symlinks are not portfolio sources")
        if path.is_file():
            found.append(path)
    return found


def _portfolio_document(
    coverage: dict[str, Any],
    coverage_path: Path,
    plan_digest: str,
    shapes: dict[str, _Shape],
    fallback: dict[str, Any],
    fallback_shape: _Shape | None,
) -> dict[str, Any]:
    """@brief 由 coverage 和目录形状生成 ``portfolio.json`` 文档。

    @param coverage coverage 文档。
    @param coverage_path 诊断路径。
    @param plan_digest 域计划摘要。
    @param shapes 每个 variant_key 的形状。
    @param fallback 已规范化或原始的 fallback 对象。
    @param fallback_shape 实现型 fallback 的形状。
    @return portfolio 文档。
    @raises PortfolioAssemblyError guard 无法解析，或同一个 bucket 的 guard 不一致。
    """
    buckets: list[dict[str, Any]] = []
    bucket_digest: dict[str, str] = {}
    variants: list[dict[str, Any]] = []
    for item in coverage["variants"]:
        key = str(item["variant_key"])
        bucket = str(item.get("bucket_key", key))
        guard = item["guard"]
        try:
            own = guard_digest(guard, where=key)
            shared = guard_digest(guard, where=bucket)
        except GeometryError as exc:
            raise PortfolioAssemblyError(f"{coverage_path}: {exc}") from exc
        previous = bucket_digest.get(bucket)
        if previous is None:
            bucket_digest[bucket] = shared
            buckets.append({"bucket_key": bucket, "guard_digest": shared})
        elif previous != shared:
            raise PortfolioAssemblyError(f"{coverage_path}: bucket {bucket!r} is used with two different guards")
        shape = shapes[key]
        variants.append(
            {
                "variant_key": key,
                "bucket_key": bucket,
                "guard": {"expr": guard, "digest": own},
                "entries": list(shape.sources),
                "geometry_entry": shape.geometry,
            }
        )
    if fallback_shape is not None:
        fallback = {
            "kind": "registered_implementation",
            "implementation_id": fallback["implementation_id"],
            "implementation_version": fallback["implementation_version"],
            "entries": list(fallback_shape.sources),
            "geometry_entry": fallback_shape.geometry,
            "reasons": list(fallback["reasons"]),
        }
    planner = coverage["planner"]
    return {
        "schema_id": PORTFOLIO_SCHEMA_ID,
        "domain_plan_digest": plan_digest,
        "planner": {"id": planner["id"], "version": planner["version"]},
        "coverage_plan": {"buckets": buckets},
        "variants": variants,
        "fallback": fallback,
    }


def _materialize(root: Path, shape: _Shape) -> list[CandidateEntry]:
    """@brief 把形状里的路径读成 entries。几何文件角色是 ``host_tiling``。

    @param root 源目录。
    @param shape 目录形状。
    @return entries，不含 manifest。
    """
    produced: list[CandidateEntry] = []
    paths = list(shape.sources)
    if shape.geometry is not None:
        paths.append(shape.geometry)
    for relative in paths:
        path = root.joinpath(*relative.split("/"))
        data = _read_bytes(path)
        if relative.endswith(".metal"):
            role, media = "device_kernel", "text/plain"
        elif relative.endswith("/tiling.c") or relative.endswith("tiling.c"):
            role, media = "host_tiling", "text/x-c"
        elif relative.endswith("/tiling.json") or relative.endswith("tiling.json"):
            role, media = "host_tiling", "application/json"
        elif relative.endswith(f"/{_GEOMETRY_NAME}") or relative.endswith(_GEOMETRY_NAME):
            role, media = "host_tiling", "application/json"
        else:  # pragma: no cover - classifier only emits these suffixes
            raise PortfolioAssemblyError(f"{path}: no portfolio role for this file")
        produced.append(CandidateEntry(path=relative, role=role, media_type=media, data=data))
    return produced


def _manifest_entry(document: dict[str, Any]) -> CandidateEntry:
    """@brief 把 portfolio 文档写成固定路径的 launch entry。

    @param document portfolio 文档。
    @return manifest entry。
    """
    data = json.dumps(document, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    return CandidateEntry(
        path=PORTFOLIO_ENTRY_PATH,
        role=PORTFOLIO_ENTRY_ROLE,
        media_type=PORTFOLIO_ENTRY_MEDIA_TYPE,
        data=data,
    )


def _read_utf8(path: Path) -> str:
    """@brief 把文件读成 UTF-8 文本。

    @param path 文件。
    @return 文本。
    @raises PortfolioAssemblyError 读不到或不是 UTF-8。
    """
    return _read_bytes(path).decode("utf-8")


def _read_bytes(path: Path) -> bytes:
    """@brief 读取非空 UTF-8 文件字节。

    @param path 文件。
    @return 字节。
    @raises PortfolioAssemblyError 读不到、为空或不是 UTF-8。
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise PortfolioAssemblyError(f"{path}: {exc}") from exc
    if not data:
        raise PortfolioAssemblyError(f"{path}: file is empty")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PortfolioAssemblyError(f"{path}: not UTF-8 text: {exc}") from exc
    return data

