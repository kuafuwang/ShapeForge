"""Versioned run files other than the transcript.

``transcript.jsonl`` is the billed-request record and is append-only. ``state.json``, ``result.json`` and ``ledger.json`` are checkpoints: they can speed up a resume, and they must not be written back over the transcript. A file that is present but damaged, or newer than this process, needs human inspection. It is not treated as work that was never done.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.schema import PROTOCOL_VERSION, SchemaError, document_version, loads_strict

RESULT_SCHEMA_VERSION = 1
STATE_SCHEMA_VERSION = 1
LEDGER_SCHEMA_VERSION = 1
RUN_CONFIG_SCHEMA_VERSION = 3

#: Fixed schema of ``candidates/<idx>_<label>/evaluation_context.json`` (plan §7.2, D43).
#:
#: The file is a frozen *projection* of one remote evaluation, written next to the candidate that
#: was evaluated remotely: the pointer (request id and digest) plus the identity fields the outbox
#: row persists.  It is never the request itself, so a resume must not rebuild one from it, and it
#: never carries a handle, a host path, a credential or an exception repr.
EVALUATION_CONTEXT_SCHEMA_VERSION = 1
EVALUATION_CONTEXT_FILE_NAME = "evaluation_context.json"
#: Every key of the projection, in the order :func:`evaluation_context_document` writes them.
EVALUATION_CONTEXT_FIELDS = (
    "schema_version",
    "request_id",
    "request_digest",
    "candidate",
    "evaluator_epoch",
    "device_group",
    "build_provenance",
    "baseline_environment_digest",
)


class NeedsInspection(RuntimeError):
    """@brief 存储的报告或检查点虽在，但其内容不可信，需要人来看。"""


#: The four states a candidate/confirm report can be in.  Schema 4 is the only *current* evidence
#: format; an older report is undecidable, not "a slightly older current one".
REPORT_STATES = ("current", "old_format", "unreadable", "missing")


def document_state(report: object) -> str:
    """@brief 一份已解析报告的当前性：``current``（schema 4）/``old_format``/``unreadable``。

    The schema test itself, without a path: callers that hold only the document (a feedback block's
    report dict, an in-memory summary) still get the one definition.

    @param report 已解析的报告文档。
    @return ``current`` / ``old_format`` / ``unreadable``。
    """
    from tcenv.schema import REPORT_SCHEMA_VERSION

    if not isinstance(report, dict):
        return "unreadable"
    recorded = report.get("schema_version")
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded != REPORT_SCHEMA_VERSION:
        return "old_format"
    return "current"


def report_state(report_path: Path, report: object) -> str:
    """@brief 一份报告的当前性：``current``（schema 4）/``old_format``/``unreadable``/``missing``。

    Schema 4 is the only current evidence format: an older report has no build identity and no
    evaluation identity, so every reader marks it instead of filling those fields with a guess.  A
    file that exists but does not parse is ``unreadable``, which is not ``missing`` — "nobody wrote
    it" and "somebody broke it" are different facts.

    This is the one definition; the index, the detail page and the brief all call it.

    @param report_path ``report.json`` 路径。
    @param report 已解析的报告（``None`` 表示读不到）。
    @return ``REPORT_STATES`` 之一。
    """
    if not isinstance(report, dict):
        return "unreadable" if report_path.exists() else "missing"
    return document_state(report)


def read_versioned(path: Path, max_version: int) -> dict | None:
    """@brief 返回 ``path`` 处的 JSON 对象；文件不存在时返回 ``None``。

    文件存在但不是可读对象，或版本号不被认识时，抛出 ``NeedsInspection``。也就是说，
    「文件在但读不懂」不等于「这件事没做过」。

    @param path 版本化 JSON 文件的路径。
    @param max_version 调用方认识的最大文档版本号。
    @return 解析后的 JSON 对象；文件不存在时为 ``None``。
    @raises NeedsInspection 文件损坏、不可读，或文档版本超出 ``max_version`` 时。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = loads_strict(path.read_text(encoding="utf-8"), what=str(path))
    except (OSError, UnicodeDecodeError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    try:
        document_version(doc, max_version=max_version, what=str(path))
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    return doc


def read_versioned_exact(path: Path, *, version: int) -> dict | None:
    """@brief 读一份**恰好**是 ``version`` 的 JSON 文档；文件不存在时返回 ``None``。

    ``read_versioned`` accepts anything up to a maximum, which is right for checkpoints and wrong for
    evidence: a schema-3 report is not "a slightly older current report", it is a different format
    whose identity fields do not exist.  Plan §8 requires the report readers to refuse it outright
    (or mark it undecidable) instead of guessing the missing fields.

    @param path 版本化 JSON 文件的路径。
    @param version 唯一可接受的版本号。
    @return 解析后的 JSON 对象；文件不存在时为 ``None``。
    @raises NeedsInspection 文件损坏、重复键、版本不等于 ``version`` 或不是对象时。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        doc = loads_strict(path.read_text(encoding="utf-8"), what=str(path))
    except (OSError, UnicodeDecodeError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    if not isinstance(doc, dict):
        raise NeedsInspection(f"{path} is not a JSON object and needs human inspection")
    recorded = doc.get("schema_version")
    if isinstance(recorded, bool) or not isinstance(recorded, int) or recorded != version:
        raise NeedsInspection(
            f"{path} has report schema {recorded!r}, not {version}: an old-format report is not "
            "decidable evidence"
        )
    return doc


def _non_empty_text(value: object, *, what: str) -> str:
    """@brief 校验一个必填的非空字符串字段。

    @param value 待校验的值。
    @param what 字段路径，用于错误信息。
    @return 原字符串。
    @raises SchemaError 不是非空字符串（含 ``bool`` 与纯空白）时。
    """
    if not isinstance(value, str) or not value or value.strip() != value:
        raise SchemaError(f"{what} is not a non-empty string")
    return value


def _hex_digest(value: object, *, what: str) -> str:
    """@brief 校验一个必填的 sha256 摘要字段（64 位小写十六进制）。

    The projection's digests are the same sha256 hex the outbox row and the reports carry; a
    shorter or uppercase string would compare unequal to them anyway, so it is refused as damage
    at the schema boundary instead of surfacing later as a mysterious mismatch.

    @param value 待校验的值。
    @param what 字段路径，用于错误信息。
    @return 原字符串。
    @raises SchemaError 不是 64 位小写十六进制字符串时。
    """
    if (
        not isinstance(value, str) or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise SchemaError(f"{what} is not a sha256 hex digest")
    return value


def evaluation_context_candidate(value: object) -> object:
    """@brief 校验投影里的 ``candidate`` 字段：通用 dialect 是源码摘要，NPU dialect 是 CandidateDigest 文档。

    NPU 的 ``candidate`` 只允许 ``{algorithm, value}`` 两个键；多一个键就是被改写的投影，不是
    “更新的版本”。通用 dialect 下它是 ``candidate_sha256`` 字符串。两者都是非空字符串。

    @param value 投影里的 ``candidate``。
    @return 规范化后的值（字符串或二键字典）。
    @raises SchemaError 形状、键集合或类型不符合闭集时。
    """
    if isinstance(value, str):
        return _non_empty_text(value, what="evaluation_context.candidate")
    if not isinstance(value, dict):
        raise SchemaError("evaluation_context.candidate is neither a digest string nor a CandidateDigest object")
    if set(value) != {"algorithm", "value"}:
        raise SchemaError("evaluation_context.candidate keys are missing or unknown")
    return {
        "algorithm": _non_empty_text(value["algorithm"], what="evaluation_context.candidate.algorithm"),
        "value": _non_empty_text(value["value"], what="evaluation_context.candidate.value"),
    }


def evaluation_context_document(
    *,
    request_id: str,
    request_digest: str,
    candidate: object,
    evaluator_epoch: str,
    device_group: dict | None = None,
    build_provenance: dict | None = None,
    baseline_environment_digest: str | None = None,
) -> dict:
    """@brief 组装 ``evaluation_context.json`` 的固定 schema v1 文档。

    只做形状校验与键排序，不读文件、不探测设备、不补字段：缺的字段写 ``None`` 而不是猜。
    ``device_group`` 只在 NPU dialect 下非空，``build_provenance`` 只在 build dialect 下非空，
    ``baseline_environment_digest`` 取自已接受报告的顶层同名字段（报告为 ``null`` 时这里也是
    ``null``）。嵌套的 dialect 文档由 ``load_completed_context`` 与 outbox 行/result 逐项对账，
    这里只要求它是对象，不重复实现第二套 dialect 校验。

    @param request_id outbox 行的 ``request_id``（行文件名是它的 sha256）。
    @param request_digest 与 outbox 行相同的请求摘要（NPU 下即 wire digest）。
    @param candidate 通用 dialect 的源码摘要，或 NPU 的 CandidateDigest 文档。
    @param evaluator_epoch 本次评测的 evaluator epoch。
    @param device_group NPU 的 ``{device_group_id, vendor_id, profile_digest, topology_digest,
        resource_generation}``；通用 dialect 为 ``None``。
    @param build_provenance build dialect 的请求级 build 身份文档；其他 dialect 为 ``None``。
    @param baseline_environment_digest 已接受报告的 baseline 环境摘要，可为 ``None``。
    @return 固定 8 键的投影文档。
    @raises SchemaError 任一必填字段形状非法、或可空字段给了非法类型时。
    """
    document: dict = {
        "schema_version": EVALUATION_CONTEXT_SCHEMA_VERSION,
        "request_id": _non_empty_text(request_id, what="evaluation_context.request_id"),
        "request_digest": _hex_digest(request_digest, what="evaluation_context.request_digest"),
        "candidate": evaluation_context_candidate(candidate),
        "evaluator_epoch": _non_empty_text(evaluator_epoch, what="evaluation_context.evaluator_epoch"),
        "device_group": None,
        "build_provenance": None,
        "baseline_environment_digest": None,
    }
    if device_group is not None:
        if not isinstance(device_group, dict):
            raise SchemaError("evaluation_context.device_group is not an object")
        document["device_group"] = dict(device_group)
    if build_provenance is not None:
        if not isinstance(build_provenance, dict):
            raise SchemaError("evaluation_context.build_provenance is not an object")
        document["build_provenance"] = dict(build_provenance)
    if baseline_environment_digest is not None:
        document["baseline_environment_digest"] = _hex_digest(
            baseline_environment_digest, what="evaluation_context.baseline_environment_digest",
        )
    return document


def validate_evaluation_context(document: object, *, what: str = EVALUATION_CONTEXT_FILE_NAME) -> dict:
    """@brief 按固定 schema v1 校验一份 ``evaluation_context.json``。

    未知键、缺失键、版本不是 1、类型非法都是**损坏**，不是 ``unknown``：一份读不懂的投影
    说明有人改过它，不能当成“这次没做远端评测”。可空字段允许 ``null``。

    @param document 解析后的投影文档。
    @param what 文档的人类可读名称，用于错误信息。
    @return 逐字段校验后的投影（键集合与写入侧一致）。
    @raises SchemaError 形状不是固定 schema v1 时。
    """
    if not isinstance(document, dict):
        raise SchemaError(f"{what} is not a JSON object")
    if set(document) != set(EVALUATION_CONTEXT_FIELDS):
        missing = sorted(set(EVALUATION_CONTEXT_FIELDS) - set(document))
        unknown = sorted(set(document) - set(EVALUATION_CONTEXT_FIELDS))
        raise SchemaError(f"{what} keys are not the fixed schema (missing={missing}, unknown={unknown})")
    version = document["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != EVALUATION_CONTEXT_SCHEMA_VERSION:
        raise SchemaError(
            f"{what} schema_version {version!r} is not {EVALUATION_CONTEXT_SCHEMA_VERSION}"
        )
    out = {
        "schema_version": EVALUATION_CONTEXT_SCHEMA_VERSION,
        "request_id": _non_empty_text(document["request_id"], what=f"{what}.request_id"),
        "request_digest": _hex_digest(document["request_digest"], what=f"{what}.request_digest"),
        "candidate": evaluation_context_candidate(document["candidate"]),
        "evaluator_epoch": _non_empty_text(document["evaluator_epoch"], what=f"{what}.evaluator_epoch"),
        "device_group": document["device_group"],
        "build_provenance": document["build_provenance"],
        "baseline_environment_digest": document["baseline_environment_digest"],
    }
    for key in ("device_group", "build_provenance"):
        if out[key] is not None and not isinstance(out[key], dict):
            raise SchemaError(f"{what}.{key} is neither null nor an object")
    if out["baseline_environment_digest"] is not None:
        out["baseline_environment_digest"] = _hex_digest(
            out["baseline_environment_digest"], what=f"{what}.baseline_environment_digest",
        )
    return out


def reconcile_evaluation_context(
    projection: object, *, row: object, report: object, what: str = EVALUATION_CONTEXT_FILE_NAME,
) -> dict:
    """@brief 把一份投影与它声称的那一行/那份报告逐项对账（F-36）。

    A projection is only evidence if it agrees with the row it points at and with the accepted
    report.  Every identity field the outbox row persists is compared point-wise; the baseline
    environment digest has no column in the row, so it is compared with the report's own top-level
    field.  A field the row does not persist (the generic dialect stores no device group or
    request-level build document) must be ``None`` on both sides — a projection that invents one
    is damage, not a richer context.

    This is a *reader*: it fills nothing and rewrites nothing.  ``unknown`` (a missing row) is
    decided by the caller before this runs; anything that reaches here and disagrees is damage.

    @param projection 读回的（或本进程刚组装的）投影文档。
    @param row outbox 行文档。
    @param report 已接受报告的文档（``report.json`` 或 result 信封里的报告）。
    @param what 投影的人类可读名称，用于错误信息。
    @return 校验过的投影。
    @raises SchemaError 投影本身不合法，或与行/报告不一致时。
    """
    document = validate_evaluation_context(projection, what=what)
    if not isinstance(row, dict):
        raise SchemaError(f"{what} has no evaluation outbox row to reconcile with")
    expected_request = row.get("request_id")
    if document["request_id"] != expected_request:
        raise SchemaError(
            f"{what}.request_id names {document['request_id']!r}, not the row's {expected_request!r}"
        )
    expected_digest = row.get("request_digest")
    if document["request_digest"] != expected_digest:
        raise SchemaError(f"{what}.request_digest does not match the outbox row")
    # The row calls it ``candidate_digest`` for the NPU contract and ``candidate_sha256`` for the
    # generic dialect; a row that carries neither has no candidate identity to reconcile with.
    row_candidate = row.get("candidate_digest")
    if row_candidate is None:
        row_candidate = row.get("candidate_sha256")
    if document["candidate"] != row_candidate:
        raise SchemaError(f"{what}.candidate does not match the outbox row")
    if document["evaluator_epoch"] != row.get("evaluator_epoch"):
        raise SchemaError(f"{what}.evaluator_epoch does not match the outbox row")
    for field in ("device_group", "build_provenance"):
        if document[field] != row.get(field):
            raise SchemaError(f"{what}.{field} does not match the outbox row")
    report_document = report if isinstance(report, dict) else {}
    frozen_environment = report_document.get("baseline_environment_digest")
    if document["baseline_environment_digest"] != frozen_environment:
        raise SchemaError(f"{what}.baseline_environment_digest does not match the accepted report")
    return document


def read_evaluation_context(path: Path) -> dict | None:
    """@brief 读回 ``evaluation_context.json``；文件不存在时返回 ``None``。

    严格解析（拒绝重复键）再按固定 schema v1 校验。文件在但读不懂是 ``NeedsInspection``：
    远端上下文在这里是 ``unknown`` 还是 ``损坏`` 不是读盘实现可以含糊的事，损坏必须让人看。

    @param path 投影文件路径。
    @return 校验后的投影；文件不存在时 ``None``。
    @raises NeedsInspection 文件存在但不可读、JSON 非法、重复键或 schema 不符时。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        document = loads_strict(path.read_text(encoding="utf-8"), what=str(path))
    except (OSError, UnicodeDecodeError) as e:
        raise NeedsInspection(f"{path} is damaged and needs human inspection") from e
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e
    try:
        return validate_evaluation_context(document, what=str(path))
    except SchemaError as e:
        raise NeedsInspection(str(e)) from e


def validate_run_config_identity(document: dict) -> dict:
    """@brief 校验 schema-3 run_config 的 v1/v2 身份字段并返回原文档。

    Schema 3 adds the v2 formal track, the deployment pack digest and the publication mode.
    A v1 run is exactly ``track=None`` plus ``task_pack_digest=None`` (its schema-2
    ``task_fingerprint`` is not a pack digest).  A v2 run must name both a formal track and the
    staged pack digest; a half-filled pair is refused instead of guessed.  Pre-3 documents have
    none of these fields and are left alone for resume compatibility.

    @param document 已解析的 run_config 对象。
    @return 原文档。
    @raises NeedsInspection 身份字段非法。
    """
    if not isinstance(document, dict):
        raise NeedsInspection("run_config.json is not an object")
    version = document.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version < 3:
        return document
    if "effective_correctness_only" not in document or not isinstance(
        document["effective_correctness_only"], bool
    ):
        raise NeedsInspection("run_config.json schema 3 is missing a boolean effective_correctness_only")
    track = document.get("track")
    pack = document.get("task_pack_digest")
    if track is None:
        if pack is not None:
            raise NeedsInspection("run_config.json has a task_pack_digest without a formal track")
        return document
    if not isinstance(track, dict):
        raise NeedsInspection("run_config.json track must be null or an object")
    track_id = track.get("track_id")
    track_version = track.get("track_version")
    if not isinstance(track_id, str) or not track_id:
        raise NeedsInspection("run_config.json track has no formal track_id")
    if isinstance(track_version, bool) or not isinstance(track_version, int) or track_version < 1:
        raise NeedsInspection("run_config.json track has no integer track_version >= 1")
    if not isinstance(pack, str) or len(pack) != 64 or any(char not in "0123456789abcdef" for char in pack):
        raise NeedsInspection("run_config.json schema 3 with a formal track needs a lowercase sha256 task_pack_digest")
    return document
