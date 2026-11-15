"""Paths and replay checks for the candidate-owned conv2d small+paper portfolio.

The tree is what ``portfolio-pack`` reads.  The paper variant file is a byte copy of the
confirmed F(4×4) MMA fixture.  small has no specialized variant; ``fallback/kernel.metal``
is the independent general implementation.

The archived run documents keep one search-seed recording.  A later run must match their
stable identity and the 1.10 gate direction.  Wall-clock microseconds are not required to
come back equal.
"""

from __future__ import annotations

import json
from pathlib import Path

from tcenv.portfolio.candidate import parse_portfolio, project_provider
from tcenv.portfolio_assembly import domain_for_sizes, parent_entries_from_tree
from tcenv.portfolio_performance.performance import MAX_REGRESSION_RATIO
from tcenv.target import load_target
from tcenv.taskcontract import load_task_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
TREE = Path(__file__).resolve().parent / "golden" / "metal" / "conv2d_portfolio"
PAPER_FIXTURE = REPO_ROOT / "tests/support/golden/metal/fixtures/conv2d_f4_mma_scratch.metal"
CONFIRMED_PAPER_SHA256 = "5fbfea6c501f6439e5abc69a18633d1d1f6c8205759d4add39ce723c749b5be2"
P2I_EVIDENCE = REPO_ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2i/evidence"
# full-eval.json records the path the runs had when that summary was written.
RECORDED_RUNS = (
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-small.json",
    "tests/support/golden/shared/shape-portfolio-阶段验收/P2-I/evidence/conv2d-portfolio-paper.json",
)
SMALL_RUN = P2I_EVIDENCE / "conv2d-portfolio-small.json"
PAPER_RUN = P2I_EVIDENCE / "conv2d-portfolio-paper.json"
FULL_EVAL = TREE / "full-eval.json"
RUN_SCHEMA = "shapeforge.conv2d-portfolio-run.1"
FULL_EVAL_SCHEMA = "shapeforge.conv2d-portfolio-full-eval.1"
DECLARED_CASES = ("normal", "wide_range", "large", "tiny", "outlier", "ieee")
STABLE_KEYS = (
    "active_manifest",
    "candidate_sha256",
    "cases",
    "credits_variant",
    "domain_plan_digest",
    "dtype",
    "fallback_credits_variant",
    "fallback_id",
    "fallback_reason",
    "fresh_confirmation",
    "message",
    "mode",
    "nondeterminism",
    "outcome",
    "performance_claim",
    "portfolio_digest",
    "provider",
    "publication_allowed",
    "reps",
    "safety_ok",
    "schema_id",
    "seed",
    "size",
    "status",
    "task",
    "usage",
    "variant_id",
)
TIMING_KEYS = (
    "baseline_us",
    "best_baseline",
    "candidate_gpu_us_median",
    "candidate_us",
    "noise_cv",
    "noisy",
    "speedups",
)
PAPER_SYMBOLS = {
    "N": 32,
    "H": 56,
    "W": 56,
    "C": 64,
    "OC": 128,
    "KH": 3,
    "KW": 3,
    "OH": 54,
    "OW": 54,
}
SMALL_SYMBOLS = {
    "N": 4,
    "H": 18,
    "W": 18,
    "C": 32,
    "OC": 64,
    "KH": 3,
    "KW": 3,
    "OH": 16,
    "OW": 16,
}


def load_json(path: Path) -> dict:
    """@brief 读取一份排序后的 JSON 对象。

    @param path 文件路径。
    @return 对象。
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def dump_json(path: Path, document: dict) -> None:
    """@brief 把对象写成稳定的 JSON。

    @param path 文件路径。
    @param document 对象。
    """
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_identity(document: dict) -> dict:
    """@brief 去掉计时，留下一次运行里必须重放的身份。

    @param document portfolio run 文档。
    @return 稳定字段。
    """
    unknown = set(document) - set(STABLE_KEYS) - {"timing"}
    if unknown:
        raise AssertionError(f"unexpected portfolio run fields: {sorted(unknown)}")
    timing = document.get("timing")
    if not isinstance(timing, dict) or set(timing) != set(TIMING_KEYS):
        raise AssertionError(f"portfolio run timing fields are {sorted(timing) if isinstance(timing, dict) else timing}")
    if "candidate_us" in STABLE_KEYS:
        raise AssertionError("wall-clock time must stay outside the replayed identity")
    return {key: document[key] for key in STABLE_KEYS}


def assert_timing_gate(document: dict) -> None:
    """@brief 用墙钟和基线的比值检查 1.10 门，不比较两次运行的微秒数。

    paper 的候选变体对 Torch-MPS 和 MLX 的延迟比都必须不超过 1.10。small 的 fallback
    对这两条基线都必须超过 1.10，而且这次时间不算变体加速。

    @param document portfolio run 文档。
    """
    timing = document["timing"]
    if timing["noisy"] is not False:
        raise AssertionError(f"{document['size']} timing is noisy")
    if not timing["candidate_us"] > 0 or not timing["candidate_gpu_us_median"] > 0:
        raise AssertionError(f"{document['size']} has no positive timing")
    ratios = {
        name: float(timing["candidate_us"]) / float(timing["baseline_us"][name])
        for name in ("torch_mps", "mlx")
    }
    if document["size"] == "paper" and document["provider"] == "candidate":
        for name, ratio in ratios.items():
            if ratio > MAX_REGRESSION_RATIO:
                raise AssertionError(f"paper latency ratio versus {name} is {ratio}, above {MAX_REGRESSION_RATIO}")
        return
    if document["size"] == "small" and document["provider"] == "fallback":
        if document["credits_variant"] is not False or document["fallback_credits_variant"] is not False:
            raise AssertionError("small fallback time was credited as a variant")
        for name, ratio in ratios.items():
            if ratio <= MAX_REGRESSION_RATIO:
                raise AssertionError(f"small fallback latency ratio versus {name} is {ratio}, inside {MAX_REGRESSION_RATIO}")
        return
    raise AssertionError(f"no timing gate for provider {document['provider']!r} at size {document['size']!r}")


def _assembled():
    """@brief 组装仓库里的 conv2d small+paper 候选树。

    @return ``(domain, portfolio)``。
    """
    contract = load_task_contract(REPO_ROOT / "tasks" / "conv2d", dtype="fp16", size="small")
    target = load_target("apple-m4-max")
    domain = domain_for_sizes(contract, target, ("small", "paper"))
    portfolio = parse_portfolio(parent_entries_from_tree(TREE, domain), domain_plan=domain.plan)
    return domain, portfolio


def assert_run_identity(document: dict, *, portfolio, domain) -> None:
    """@brief 核对一份归档 run 的身份、六案和安全字段。

    @param document portfolio run 文档。
    @param portfolio 已解析的父包。
    @param domain 组装域。
    """
    stable_identity(document)
    if document["schema_id"] != RUN_SCHEMA:
        raise AssertionError(f"unknown portfolio run schema {document['schema_id']}")
    if document["publication_allowed"] is not False or document["fresh_confirmation"] is not False:
        raise AssertionError("a portfolio search-seed run was marked published or fresh")
    if document["performance_claim"] is not None or document["active_manifest"] is not None:
        raise AssertionError("a portfolio run carried an activation or a performance claim")
    if document["fallback_credits_variant"] is not False:
        raise AssertionError("fallback time was credited to a variant")
    if document["task"] != "conv2d" or document["dtype"] != "fp16" or document["usage"] != "benchmark":
        raise AssertionError("portfolio run is not the conv2d fp16 benchmark point")
    if document["mode"] != "full" or document["seed"] != 7 or document["reps"] != 15:
        raise AssertionError("portfolio run is not the search-seed full evaluation")
    if document["status"] != "ok" or document["safety_ok"] is not True or document["nondeterminism"] != []:
        raise AssertionError("portfolio run is not safe and correct")
    cases = document["cases"]
    if [item["case"] for item in cases] != list(DECLARED_CASES) or not all(item["ok"] is True for item in cases):
        raise AssertionError("portfolio run did not pass the six declared cases")
    projected_paper = project_provider(portfolio, portfolio.variants[0], authority=domain.authority)
    projected_fallback = project_provider(portfolio, portfolio.fallback, authority=domain.authority)
    if document["portfolio_digest"] != portfolio.digest or document["domain_plan_digest"] != portfolio.domain_plan_digest:
        raise AssertionError("archived portfolio identity does not match the current candidate tree")
    if document["size"] == "paper":
        if document["provider"] != "candidate" or document["outcome"] != "resolved":
            raise AssertionError("paper did not resolve to the candidate variant")
        if document["variant_id"] != portfolio.variants[0].variant_id or document["fallback_id"] is not None:
            raise AssertionError("paper variant identity drifted")
        if document["fallback_reason"] is not None or document["credits_variant"] is not True:
            raise AssertionError("paper variant credit drifted")
        if document["candidate_sha256"] != CONFIRMED_PAPER_SHA256:
            raise AssertionError("paper run is not the confirmed MMA fixture")
        if document["candidate_sha256"] != projected_paper.candidate.candidate_sha256:
            raise AssertionError("paper run sha does not match the projected variant")
        assert_timing_gate(document)
        return
    if document["size"] == "small":
        if document["provider"] != "fallback" or document["outcome"] != "fallback":
            raise AssertionError("small did not take the independent fallback")
        if document["fallback_reason"] != "outside_candidate_coverage" or document["variant_id"] is not None:
            raise AssertionError("small fallback reason drifted")
        if document["fallback_id"] != portfolio.fallback.fallback_id:
            raise AssertionError("small fallback identity drifted")
        if document["candidate_sha256"] != projected_fallback.candidate.candidate_sha256:
            raise AssertionError("small run sha does not match the projected fallback")
        if document["candidate_sha256"] == CONFIRMED_PAPER_SHA256 or document["credits_variant"] is not False:
            raise AssertionError("small reused the paper MMA identity")
        assert_timing_gate(document)
        return
    raise AssertionError(f"unexpected portfolio size {document['size']!r}")


def full_eval_document(small: dict, paper: dict) -> dict:
    """@brief 从两份 run 派生摘要。摘要不是另一份独立测量。

    @param small small run。
    @param paper paper run。
    @return full-eval 文档。
    """
    if (small["size"], paper["size"]) != ("small", "paper"):
        raise AssertionError("full-eval sources are not the small and paper runs")
    return {
        "fresh_confirmation": False,
        "mode": "full",
        "note": (
            "Derived from the two archived portfolio runs. Replay checks stable identity and the "
            "1.10 gate, not equal wall-clock microseconds. This is not a fresh confirmation."
        ),
        "paper_fixture_sha256": CONFIRMED_PAPER_SHA256,
        "publication_allowed": False,
        "reps": 15,
        "runs": list(RECORDED_RUNS),
        "schema_id": FULL_EVAL_SCHEMA,
        "seed": 7,
        "shards": [_shard(small), _shard(paper)],
    }


def _shard(document: dict) -> dict:
    """@brief 把一份 run 收成摘要里的一行。

    @param document portfolio run。
    @return 摘要行。
    """
    timing = document["timing"]
    return {
        "baseline_us": timing["baseline_us"],
        "best_baseline": timing["best_baseline"],
        "candidate_gpu_us_median": timing["candidate_gpu_us_median"],
        "candidate_sha256": document["candidate_sha256"],
        "candidate_us": timing["candidate_us"],
        "cases": [item["case"] for item in document["cases"]],
        "fallback_credits_variant": document["fallback_credits_variant"],
        "fallback_id": document["fallback_id"],
        "fallback_reason": document["fallback_reason"],
        "message": document["message"],
        "noise_cv": timing["noise_cv"],
        "noisy": timing["noisy"],
        "outcome": document["outcome"],
        "performance_claim": document["performance_claim"],
        "provider": document["provider"],
        "safety_ok": document["safety_ok"],
        "size": document["size"],
        "speedups": timing["speedups"],
        "status": document["status"],
        "usage": document["usage"],
        "variant_id": document["variant_id"],
    }


def assert_recorded_portfolio() -> None:
    """@brief 归档的两份 run 与派生摘要一致，并各自落在 1.10 门的正确一侧。"""
    domain, portfolio = _assembled()
    small = load_json(SMALL_RUN)
    paper = load_json(PAPER_RUN)
    assert_run_identity(small, portfolio=portfolio, domain=domain)
    assert_run_identity(paper, portfolio=portfolio, domain=domain)
    recorded = load_json(FULL_EVAL)
    derived = full_eval_document(small, paper)
    if recorded != derived:
        raise AssertionError("full-eval.json is not the projection of the two archived portfolio runs")
    if small["candidate_sha256"] == paper["candidate_sha256"]:
        raise AssertionError("small and paper resolved to the same candidate")
