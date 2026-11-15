"""Real-Metal recorder for P2-P: the bitdelta_gemm REG fresh subjects, probe and decision."""

from __future__ import annotations


from tcenv.backends.metal.search_candidate import (
    confirm_candidate,
    guidance_fingerprint,
)
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from experiments.bitdelta_unpack_probe import run_probe
from tests.support.bitdelta_unpack import (
    DECISION,
    PAPER_CONFIRMATION,
    PAPER_SHA256,
    PROBE,
    SMALL_CONFIRMATION,
    SMALL_SHA256,
    assert_decision,
    decision_document,
    dump_json,
    load_json,
    stored_source,
)

GUIDANCE = (
    "P2-P bitdelta_gemm REG: decode the packed sign words inside each output tile, one stage, "
    "no global expanded matrix"
)


def locator(size: str):
    found = next(
        item
        for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
        if item.op_name == "bitdelta_gemm"
    )
    return locator_for_instance(found, "fp16", size)


def master_key() -> bytes:
    from tests.support.acceptance_keys import P2H_MASTER_KEY
    return P2H_MASTER_KEY


def record_confirmation(size: str) -> dict:
    source = stored_source(size)
    subject = confirm_candidate(
        locator(size),
        source,
        target=load_target("apple-m4-max"),
        guidance_digest=guidance_fingerprint(GUIDANCE, source),
        master_seed=master_key(),
    )
    expected_sha = SMALL_SHA256 if size == "small" else PAPER_SHA256
    if subject["candidate_sha256"] != expected_sha:
        raise AssertionError(f"{size} fresh subject named candidate {subject['candidate_sha256']}")
    path = SMALL_CONFIRMATION if size == "small" else PAPER_CONFIRMATION
    dump_json(path, subject)
    report = subject["confirmation_report"]
    print(
        f"OK: {size} fresh {subject['status']} seed={subject['confirmation_seed']} "
        f"candidate={report['candidate_us']:.4f} us gpu={report['benchmark']['candidate_gpu_us_median']:.4f} "
        f"cv={report['benchmark']['noise_cv']:.4f} baseline={report['baseline_us']}",
        flush=True,
    )
    return subject


def record_probe_evidence() -> dict:
    document = run_probe(PROBE)
    print(f"OK: wrote {PROBE.relative_to(REPO_ROOT)}", flush=True)
    return document


def record_decision_evidence() -> dict:
    subjects = {
        "small": load_json(SMALL_CONFIRMATION),
        "paper": load_json(PAPER_CONFIRMATION),
    }
    probe = load_json(PROBE)
    document = decision_document(subjects, probe)
    assert_decision(document, subjects, probe)
    dump_json(DECISION, document)
    for size, one in document["sizes"].items():
        print(
            f"OK: {size} REG {one['reg_candidate_us']:.4f} us vs roofline "
            f"{one['roofline_lower_bound_us']:.4f} us; REG decodes "
            f"{one['reg_decoded_elements']} elements ({one['reg_decode_repeats']}x, "
            f"{one['reg_tiling']['tile_m']}x{one['reg_tiling']['tile_n']} tiles); best legal SHARED "
            f"{one['best_legal_shared']} at {one['best_legal_shared_ratio_over_reg']:.4f}x REG "
            f"(adoption bar is <1.0x REG; registered-baseline gate would pass: "
            f"{one['best_legal_shared_would_pass_registered_baseline_gate']})",
            flush=True,
        )
    print("OK: adopted REG; no SHARED candidate source", flush=True)
    return document


def record_all() -> None:
    for size in ("small", "paper"):
        record_confirmation(size)
    record_probe_evidence()
    record_decision_evidence()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--decision", action="store_true")
    args = parser.parse_args()
    if not any((args.confirmation, args.probe, args.decision)):
        record_all()
        return
    if args.confirmation:
        for size in ("small", "paper"):
            record_confirmation(size)
    if args.probe:
        record_probe_evidence()
    if args.decision:
        record_decision_evidence()


if __name__ == "__main__":
    main()
