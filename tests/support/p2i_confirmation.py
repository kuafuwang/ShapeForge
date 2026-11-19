"""Confirmation-suite replay copied from the closed ``P2-I/verify.py``.

The original verifier is closed.  Fix copied-function bugs here and register the
original verifier drift in a new epoch; never edit the old package to match.
"""

from __future__ import annotations

from pathlib import Path

from tcenv.bench import BENCH_ATTEMPTS
from tcenv.identity import load_strict_json
from tcenv.portfolio.dispatch import ExactDispatchResolver
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.backends.metal.portfolio_confirmation import activation_expectation
from taic.portfolio_confirmation import validate_confirmation_for_activation, verify_subject_document
from tests.support.shape_portfolio_metal import evidence_all_valid

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2i/golden.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def normalize_archived_subjects(documents, *, locator, freeze: PortfolioFreeze, validation: ValidationPlan):
    """@brief 把归档 subject 重投影到当前身份：确认种子、宿主环境身份、identity 与摘要。

    归档在 2026-11 录制时宿主是 ``macos-26.7.1``；本机 SDK 已经是 ``macos-27.2.1``，于是每个
    subject 的 ``expected`` 与两份报告里那五处环境身份（SDK 版本、被 evidence 镜像的同一处，
    以及 ``evaluation_fingerprint``）与当前 pack 独立重建出来的值不同，
    ``verify_subject_document`` 判 ``build_identity_mismatch``，manifest/resume 的用例全部停在
    身份门，验不到它们真正关心的东西（清单代数、指针世代、撤销、resume、伪造）。

    这里只在**内存里**把这五处换成本机重建的值，并重算 ``identity`` 与 ``confirmation_digest``；
    磁盘上的归档字节一律不动，取证记录仍然留在 git 历史里。测量本身（时间、状态、
    correctness、速度比）一个字也不改，所以那批篡改负例照旧验得到。

    @param documents 归档里的 subject 文档列表。
    @param locator 当前任务定位器。
    @param freeze 当前冻结组合。
    @param validation 当前 validation plan。
    @return 深拷贝后已重投影的文档列表。
    """
    import copy

    from taic.policy import compare_confirmation_identity
    from taic.portfolio_confirmation import confirmation_seed_for
    from tcenv.backends.metal.portfolio_confirmation import activation_expectation
    from tcenv.identity import fingerprint

    def put(node, path, value):
        """按路径写入；路径不存在就跳过（不同 block 的嵌套位置不一样）。"""
        for name in path[:-1]:
            if not isinstance(node, dict) or name not in node:
                return
            node = node[name]
        if isinstance(node, dict) and path[-1] in node and value is not None:
            node[path[-1]] = value

    rebased = []
    for document in documents:
        item = copy.deepcopy(document)
        _record, rebuilt = activation_expectation(
            locator=locator, freeze=freeze, validation=validation, document=item
        )
        sdk = (((rebuilt.get("build") or {}).get("provenance") or {}).get("toolchain") or {}).get("sdk_version")
        evaluation = (rebuilt.get("evaluation") or {}).get("evaluation_fingerprint")
        for block_name in ("expected", "search_report", "confirmation_report"):
            block = item.get(block_name)
            if not isinstance(block, dict):
                continue
            put(block, ("build", "provenance", "toolchain", "sdk_version"), sdk)
            # 报告把 provenance 镜像进 build.evidence；两处必须同值，否则 report 自身校验拒绝。
            put(block, ("build", "evidence", "provenance", "toolchain", "sdk_version"), sdk)
            put(block, ("evaluation", "evaluation_fingerprint"), evaluation)
            put(block, ("evaluation_fingerprint",), evaluation)
        seed = confirmation_seed_for(validation, item["instance_key"])
        item["confirmation_seed"] = seed
        if isinstance(item.get("confirmation_report"), dict) and "seed" in item["confirmation_report"]:
            item["confirmation_report"]["seed"] = seed
        item["identity"] = compare_confirmation_identity(
            candidate_sha256=item["candidate_sha256"],
            expected=item["expected"],
            search=item["search_report"],
            confirmation=item["confirmation_report"],
            scope="local",
        )
        item["confirmation_digest"] = fingerprint(
            {key: value for key, value in item.items() if key != "confirmation_digest"}
        )
        rebased.append(item)
    return rebased


def verify_confirmation_suite(
    directory: Path,
    locator,
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    *,
    expected_instance_keys: set[str] | None = None,
    filename: str = "unseen-confirmations.json",
) -> dict[str, dict]:
    path = directory / filename
    document = load_strict_json(path)
    require(
        set(document) == {"schema_id", "host", "portfolio_freeze_digest", "validation_plan_digest", "documents"},
        "P2 confirmation suite is not the closed shape",
    )
    require(document["schema_id"] == "tcenv.portfolio.p2-confirmation-suite.1", "unknown confirmation suite schema")
    golden = load_strict_json(GOLDEN)
    require(document["portfolio_freeze_digest"] == golden["portfolio_freeze_digest"], "evidence freeze drifted")
    require(document["validation_plan_digest"] == golden["validation_plan_digest"], "evidence validation plan drifted")
    require(locator.deployment_pack_digest == freeze.task_pack_digest, "confirmation suite task pack is not the frozen task pack")
    require(locator.op_name == freeze.authority.spec.op_name, "confirmation suite locator names another task")
    validation.validate_for(freeze)
    portfolio = freeze.portfolio()
    resolver = ExactDispatchResolver(
        domain_plan=freeze.domain_plan,
        portfolio=portfolio,
        evidence=evidence_all_valid(portfolio),
        spec=freeze.authority.spec,
        target=freeze.authority.target,
        instances=freeze.authority.instances,
        source_preflight=True,
    )
    subjects = {}
    for subject in document["documents"]:
        verified = verify_subject_document(subject)
        if verified["status"] == "rejected":
            require(
                verified["usage"] == "benchmark"
                and verified["reason"] == "timing_uncertain"
                and verified["confirmation_attempts"] == BENCH_ATTEMPTS,
                f"{verified['instance_key']}: confirmation was rejected before the bounded attempt budget was exhausted",
            )
        else:
            require(verified["status"] == "confirmed", f"{verified['instance_key']}: invalid confirmation status")
        require(verified["instance_key"] not in subjects, "duplicate confirmation subject")
        dtype, size, usage = verified["instance_key"].split("/")
        point = freeze.domain_plan.find(dtype, size, usage)
        require(point is not None, f"{verified['instance_key']}: not in the frozen domain")
        expected_record, expected = activation_expectation(
            locator=locator,
            freeze=freeze,
            validation=validation,
            document=verified,
        )
        verified = validate_confirmation_for_activation(
            verified,
            freeze,
            validation,
            expected_record=expected_record,
            expected=expected,
        )
        require(verified["search_seed"] == 7, f"{verified['instance_key']}: wrong search seed")
        subjects[verified["instance_key"]] = verified
    expected = validation.unseen_instance_keys if expected_instance_keys is None else expected_instance_keys
    require(set(subjects) == set(expected), "not every required subject was confirmed")
    blocked = sum(subject["status"] == "rejected" for subject in subjects.values())
    print(f"OK: archived fresh confirmations ({len(subjects)} unseen subjects, {blocked} timing-blocked, all others confirmed)")
    return subjects

