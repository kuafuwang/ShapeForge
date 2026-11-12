#!/usr/bin/env python3
"""Re-record P2-J search-seed runs and, when requested, the small fresh subject."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from tcenv.backends.metal.search_candidate import confirm_candidate, guidance_fingerprint
from tcenv.target import load_target
from tests.backends.metal.conv2d_portfolio_record import conv2d_locator
from tests.backends.metal.conv2d_small_mma_record import record_search_seed_runs
from tests.support.conv2d_small_mma import SMALL_CONFIRMATION, TREE

ROOT = Path(__file__).resolve().parents[6]


def master_key() -> bytes:
    path = ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/verify.py"
    spec = importlib.util.spec_from_file_location("p2j_public_key", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.P2H_MASTER_KEY


def record_confirmation() -> None:
    from tcenv.taskcontract.portfolio_runner import locator_for_instance

    source = (TREE / "variants" / "small" / "kernel.metal").read_text(encoding="utf-8")
    document = confirm_candidate(
        locator_for_instance(conv2d_locator(), "fp16", "small"),
        source,
        target=load_target("apple-m4-max"),
        guidance_digest=guidance_fingerprint("conv2d small single-stage F4 simdgroup MMA recompute-U", source),
        master_seed=master_key(),
    )
    SMALL_CONFIRMATION.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"OK: small fresh {document['status']} seed={document['confirmation_seed']} sha={document['candidate_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    args = parser.parse_args()
    if args.confirmation:
        record_confirmation()
    else:
        record_search_seed_runs()
        print("OK: recorded P2-J small+paper search-seed runs")


if __name__ == "__main__":
    main()
