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

