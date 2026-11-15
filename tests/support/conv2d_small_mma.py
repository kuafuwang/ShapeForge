"""Replay helpers for the P2-J small+paper specialized-MMA portfolio."""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio_assembly import domain_for_sizes, parent_entries_from_tree
from tcenv.portfolio_performance.performance import MAX_REGRESSION_RATIO
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract
from tests.support.conv2d_portfolio import DECLARED_CASES, PAPER_SYMBOLS, SMALL_SYMBOLS, _shard, stable_identity

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE = Path(__file__).resolve().parent / "golden" / "metal" / "conv2d_small_mma"
P2J = REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2j"
# full-eval.json records the path the runs had when that summary was written.
RECORDED_RUNS = (
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/small-run.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-J/evidence/paper-run.json",
)
EVIDENCE = P2J / "evidence"
SMALL_RUN = EVIDENCE / "small-run.json"
PAPER_RUN = EVIDENCE / "paper-run.json"
FULL_EVAL = TREE / "full-eval.json"
SMALL_CONFIRMATION = EVIDENCE / "small-confirmation.json"
PAPER_SHA256 = "5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2"
SMALL_SHA256 = "50ac29d1bc5c0ebcb983f16d5faedeaa0869324b10a979613900b29a640486f2"
RUN_SCHEMA = "shapeforge.conv2d-portfolio-run.1"
FULL_EVAL_SCHEMA = "shapeforge.conv2d-small-mma-full-eval.1"


def load_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def dump_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def assembled():
    contract = load_task_contract(REPO_ROOT / "tasks" / "conv2d", dtype="fp16", size="small")
    domain = domain_for_sizes(contract, load_target("apple-m4-max"), ("small", "paper"))
    portfolio = parse_portfolio(parent_entries_from_tree(TREE, domain), domain_plan=domain.plan)
    return domain, portfolio


def assert_candidate_gate(document: dict) -> None:
    timing = document["timing"]
    if timing["noisy"] is not False:
        raise AssertionError(f"{document['size']} timing is noisy")
    if not timing["candidate_us"] > 0 or not timing["candidate_gpu_us_median"] > 0:
        raise AssertionError(f"{document['size']} has no positive timing")
    for name in ("torch_mps", "mlx"):
        ratio = float(timing["candidate_us"]) / float(timing["baseline_us"][name])
        if ratio > MAX_REGRESSION_RATIO:
            raise AssertionError(f"{document['size']} latency ratio versus {name} is {ratio}, above {MAX_REGRESSION_RATIO}")


def assert_run_identity(document: dict) -> None:
    domain, portfolio = assembled()
    stable_identity(document)
    if document["schema_id"] != RUN_SCHEMA:
        raise AssertionError("unknown P2-J run schema")
    if document["publication_allowed"] is not False or document["fresh_confirmation"] is not False:
        raise AssertionError("a search-seed portfolio run was marked published or fresh")
    if document["performance_claim"] is not None or document["active_manifest"] is not None:
        raise AssertionError("a portfolio run carried activation or a performance claim")
    if document["task"] != "conv2d" or document["dtype"] != "fp16" or document["usage"] != "benchmark":
        raise AssertionError("wrong task instance")
    if document["mode"] != "full" or document["seed"] != 7 or document["reps"] != 15:
        raise AssertionError("not the search-seed full evaluation")
    if document["status"] != "ok" or document["safety_ok"] is not True or document["nondeterminism"] != []:
        raise AssertionError("portfolio run is not safe and correct")
    cases = document["cases"]
    if [item["case"] for item in cases] != list(DECLARED_CASES) or not all(item["ok"] is True for item in cases):
        raise AssertionError("portfolio run did not pass all six cases")
    if document["portfolio_digest"] != portfolio.digest or document["domain_plan_digest"] != portfolio.domain_plan_digest:
        raise AssertionError("portfolio identity drifted")
    size = document["size"]
    if size not in ("small", "paper"):
        raise AssertionError(f"unexpected size {size}")
    variant = next(item for item in portfolio.variants if item.variant_key == size)
    projected = project_provider(portfolio, variant, authority=domain.authority)
    expected_sha = SMALL_SHA256 if size == "small" else PAPER_SHA256
    if document["provider"] != "candidate" or document["outcome"] != "resolved":
        raise AssertionError(f"{size} did not resolve to its specialized candidate")
    if document["variant_id"] != variant.variant_id or document["fallback_id"] is not None:
        raise AssertionError(f"{size} variant identity drifted")
    if document["fallback_reason"] is not None or document["credits_variant"] is not True:
        raise AssertionError(f"{size} variant credit drifted")
    if document["fallback_credits_variant"] is not False:
        raise AssertionError("fallback time was credited to a variant")
    if document["candidate_sha256"] != expected_sha or document["candidate_sha256"] != projected.candidate.candidate_sha256:
        raise AssertionError(f"{size} candidate sha drifted")
    assert_candidate_gate(document)


def full_eval_document(small: dict, paper: dict) -> dict:
    if (small["size"], paper["size"]) != ("small", "paper"):
        raise AssertionError("full-eval sources are not small and paper")
    return {
        "fresh_confirmation": False,
        "mode": "full",
        "note": "Derived from two real search-seed portfolio runs. Small also has a separate fresh confirmation; this summary is not a second fresh measurement.",
        "paper_fixture_sha256": PAPER_SHA256,
        "small_fixture_sha256": SMALL_SHA256,
        "publication_allowed": False,
        "reps": 15,
        "runs": list(RECORDED_RUNS),
        "schema_id": FULL_EVAL_SCHEMA,
        "seed": 7,
        "shards": [_shard(small), _shard(paper)],
    }


def assert_recorded_portfolio() -> None:
    small = load_json(SMALL_RUN)
    paper = load_json(PAPER_RUN)
    assert_run_identity(small)
    assert_run_identity(paper)
    if load_json(FULL_EVAL) != full_eval_document(small, paper):
        raise AssertionError("full-eval.json is not the projection of the two P2-J runs")
    if small["candidate_sha256"] == paper["candidate_sha256"]:
        raise AssertionError("small and paper resolved to the same candidate")
