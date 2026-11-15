#!/usr/bin/env python3
"""Record and independently replay the P2-K small+paper portfolio fresh suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from taic.portfolio_manifest import DispatchManifest
from tcenv.backends.metal.portfolio_confirmation import activation_expectation, confirm_subject
from tcenv.backends.metal.search_candidate import PROTOCOL_EPOCH, evidence_for, guidance_fingerprint
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio_assembly import domain_for_sizes, parent_entries_from_tree
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from tcenv.target import load_target
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from tests.backends.metal.conv2d_portfolio_record import conv2d_locator
from tests.support.conv2d_portfolio_fresh import FREEZE_DIGEST, FRESH_SUITE, VALIDATED_MANIFEST, VALIDATION_DIGEST
from tests.support.conv2d_small_mma import TREE

ROOT = Path(__file__).resolve().parents[3]
TUNE = ("fp16/paper/benchmark", "fp16/small/benchmark")
REQUIRED = ("fp16/paper/golden", "fp16/small/golden")


def master_key() -> bytes:
    from tests.support.acceptance_keys import P2H_MASTER_KEY
    return P2H_MASTER_KEY


def build_frozen():
    locator = conv2d_locator()
    contract = locator.load()
    domain = domain_for_sizes(contract, load_target("apple-m4-max"), ("small", "paper"))
    entries = parent_entries_from_tree(TREE, domain)
    portfolio = parse_portfolio(entries, domain_plan=domain.plan)
    parts = [
        "conv2d small+paper portfolio-level fresh confirmation P2-K",
        (TREE / "coverage.json").read_text(encoding="utf-8"),
        (TREE / "variants" / "small" / "kernel.metal").read_text(encoding="utf-8"),
        (TREE / "variants" / "paper" / "kernel.metal").read_text(encoding="utf-8"),
        (TREE / "fallback" / "kernel.metal").read_text(encoding="utf-8"),
    ]
    freeze = PortfolioFreeze(
        protocol_epoch=PROTOCOL_EPOCH,
        authority=domain.authority,
        candidate_entries=entries,
        task_pack_digest=locator.deployment_pack_digest,
        guidance_digest=guidance_fingerprint(*parts),
    )
    validation = ValidationPlan.derive(
        freeze,
        tune_shards=TUNE,
        unseen_count=len(REQUIRED),
        master_seed=master_key(),
        required_instance_keys=REQUIRED,
    )
    if freeze.digest != FREEZE_DIGEST or validation.digest != VALIDATION_DIGEST:
        raise AssertionError("P2-K freeze/validation no longer matches the archived evidence")
    return locator, contract, domain, portfolio, freeze, validation


def confirm_portfolio_subjects():
    locator, contract, domain, portfolio, freeze, validation = build_frozen()
    subjects = []
    for size, usage in (("small", "benchmark"), ("paper", "benchmark"), ("small", "golden"), ("paper", "golden")):
        instance = instantiate_v2(contract.spec, "fp16", size)
        point = domain.plan.find("fp16", size, usage)
        request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)
        document = confirm_subject(
            locator=locator_for_instance(locator, "fp16", size),
            domain_plan=domain.plan,
            portfolio=portfolio,
            request=request,
            evidence=evidence_for(portfolio),
            validation=validation,
            freeze=freeze,
            track_id="msl-single",
        )
        if document["status"] != "confirmed" or document["identity"]["kind"] != "match":
            raise AssertionError(f"{size}/{usage}: {document['status']} {document.get('reason')}")
        subjects.append(document)
    return subjects


def validated_manifest(subjects):
    locator, _contract, _domain, _portfolio, freeze, validation = build_frozen()
    expectations = {
        document["instance_key"]: activation_expectation(
            locator=locator, freeze=freeze, validation=validation, document=document, track_id="msl-single"
        )
        for document in subjects
    }
    return DispatchManifest.from_confirmations(
        freeze, validation, subjects, expectations, locator=locator
    )


def write_documents(subjects) -> None:
    _locator, _contract, _domain, _portfolio, freeze, validation = build_frozen()
    payload = {
        "schema_id": "shapeforge.conv2d-portfolio-fresh-suite.1",
        "publication_allowed": False,
        "freeze_digest": freeze.digest,
        "validation_digest": validation.digest,
        "tune_shards": list(TUNE),
        "unseen_instance_keys": list(validation.unseen_instance_keys),
        "subjects": subjects,
    }
    FRESH_SUITE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = validated_manifest(subjects)
    if manifest.status != "validated" or manifest.activation_generation != 0:
        raise AssertionError("P2-K must remain validated, not active")
    VALIDATED_MANIFEST.write_text(json.dumps(manifest.document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        subjects = json.loads(FRESH_SUITE.read_text(encoding="utf-8"))["subjects"]
        manifest = validated_manifest(subjects)
        VALIDATED_MANIFEST.write_text(json.dumps(manifest.document(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        write_documents(confirm_portfolio_subjects())
    print("OK: recorded P2-K fresh suite and validated (not active) manifest")


if __name__ == "__main__":
    main()
