"""Real-Metal recorder for the gemm_silu REG candidates, plain-GEMM control, probe and decision."""

from __future__ import annotations


from tcenv.backends.metal.search_candidate import (
    confirm_candidate,
    evaluate_candidate,
    guidance_fingerprint,
)
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import locator_for_instance
from experiments.gemm_silu_scratch_probe import run_probe
from tests.support.gemm_silu_scratch import (
    CONTROL_SCHEMA,
    DECISION,
    PAPER_CONFIRMATION,
    PAPER_CONTROL,
    PROBE,
    SMALL_CONFIRMATION,
    SMALL_CONTROL,
    assert_decision,
    decision_document,
    dump_json,
    load_json,
    plain_gemm_source,
    sha256_bytes,
    stored_source,
)

GUIDANCE = (
    REPO_ROOT.joinpath("tests/support/golden/shared/acceptance-epochs/fixtures/p2o/guidance.txt")
    .read_text(encoding="utf-8")
    .rstrip("\n")
)


def locator(op: str, size: str):
    found = next(
        item
        for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
        if item.op_name == op
    )
    return locator_for_instance(found, "fp16", size)


def master_key() -> bytes:
    from tests.support.acceptance_keys import P2H_MASTER_KEY
    return P2H_MASTER_KEY


def record_confirmation(size: str) -> dict:
    source = stored_source(size)
    subject = confirm_candidate(
        locator("gemm_silu", size),
        source,
        target=load_target("apple-m4-max"),
        guidance_digest=guidance_fingerprint(GUIDANCE, source),
        master_seed=master_key(),
    )
    path = SMALL_CONFIRMATION if size == "small" else PAPER_CONFIRMATION
    dump_json(path, subject)
    print(
        f"OK: {size} fresh {subject['status']} seed={subject['confirmation_seed']} "
        f"sha={subject['candidate_sha256']}",
        flush=True,
    )
    return subject


def run_control(size: str) -> dict:
    source = plain_gemm_source(size)
    run = evaluate_candidate(
        locator("gemm", size),
        source,
        target=load_target("apple-m4-max"),
        mode="full",
        seed=7,
        reps=15,
    )
    report = run.result.report
    document = {
        "schema_id": CONTROL_SCHEMA,
        "control": "the registered plain GEMM candidate, same algorithm, no activation",
        "fresh_confirmation": False,
        "note": (
            "The control is measured on tasks/gemm because tasks/gemm_silu requires SiLU for "
            "correctness. Only its candidate median is used: it isolates the fused-epilogue cost."
        ),
        "publication_allowed": False,
        "report": report,
        "identities": run.document()["identities"],
        "source_origin": "tests/support/golden/metal/fixtures/"
        + ("gemm_small_steel.metal" if size == "small" else "gemm_paper_best.metal"),
        "source_sha256": sha256_bytes(source.encode("utf-8")),
        "size": size,
        "task": "gemm",
        "usage": "benchmark",
        "dtype": "fp16",
        "seed": 7,
        "reps": 15,
    }
    return document


def record_control(size: str) -> dict:
    document = run_control(size)
    dump_json(SMALL_CONTROL if size == "small" else PAPER_CONTROL, document)
    print(f"OK: {size} plain-GEMM control {document['report'].get('candidate_us')} us", flush=True)
    return document


def record_probe_evidence() -> dict:
    document = run_probe(PROBE)
    print(f"OK: wrote {PROBE.relative_to(REPO_ROOT)}", flush=True)
    return document


def record_decision_evidence() -> dict:
    subjects = {
        "small": load_json(SMALL_CONFIRMATION),
        "paper": load_json(PAPER_CONFIRMATION),
    }
    controls = {
        "small": load_json(SMALL_CONTROL),
        "paper": load_json(PAPER_CONTROL),
    }
    probe = load_json(PROBE)
    document = decision_document(subjects, controls, probe)
    assert_decision(document, subjects, controls, probe)
    dump_json(DECISION, document)
    for size, one in document["sizes"].items():
        print(
            f"OK: {size} REG {one['reg_candidate_us']:.4f} us vs hard lower bound "
            f"{one['hard_lower_bound_us']:.4f} us ({one['hard_lower_bound_over_reg']:.3f}x), "
            f"best probe extrapolation {one['best_probe_global_estimate_us']:.4f} us",
            flush=True,
        )
    print("OK: adopted REG; no GLOBAL candidate source", flush=True)
    return document


def record_all() -> None:
    for size in ("small", "paper"):
        record_confirmation(size)
    for size in ("small", "paper"):
        record_control(size)
    record_probe_evidence()
    record_decision_evidence()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", action="store_true")
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--decision", action="store_true")
    args = parser.parse_args()
    if not any((args.confirmation, args.control, args.probe, args.decision)):
        record_all()
        return
    if args.confirmation:
        for size in ("small", "paper"):
            record_confirmation(size)
    if args.control:
        for size in ("small", "paper"):
            record_control(size)
    if args.probe:
        record_probe_evidence()
    if args.decision:
        record_decision_evidence()


if __name__ == "__main__":
    main()
