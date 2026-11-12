"""Project one real conv2d portfolio run into the archived evidence document.

The Metal runner lives here so the shared replay checks can stay free of this backend.
A later measurement compares the stable projection and the 1.10 gate.  It does not have
to reproduce the recorded microseconds.
"""

from __future__ import annotations

from tcenv.backends.metal.search_candidate import evidence_for
from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio_assembly import domain_for_sizes, parent_entries_from_tree
from tcenv.target import load_target
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.taskcontract.portfolio_runner import run_portfolio_task
from tests.support.conv2d_portfolio import (
    FULL_EVAL,
    PAPER_RUN,
    SMALL_RUN,
    TREE,
    assert_recorded_portfolio,
    assert_timing_gate,
    dump_json,
    full_eval_document,
    load_json,
)


def conv2d_locator():
    """@brief 仓库里的 conv2d locator。

    @return 默认档 locator。
    """
    return next(
        item
        for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)
        if item.op_name == "conv2d"
    )


def load_conv2d_portfolio():
    """@brief 打包 small+paper 候选树。

    @return ``(domain, portfolio)``。
    """
    contract = conv2d_locator().load()
    target = load_target("apple-m4-max")
    domain = domain_for_sizes(contract, target, ("small", "paper"))
    portfolio = parse_portfolio(parent_entries_from_tree(TREE, domain), domain_plan=domain.plan)
    return domain, portfolio


def run_conv2d_portfolio(size: str, *, usage: str = "golden", mode: str = "quick", reps: int = 1):
    """@brief 在真实 Metal 上跑一个 conv2d 域点。

    @param size ``small`` 或 ``paper``。
    @param usage ``golden`` 或 ``benchmark``。
    @param mode ``quick`` 或 ``full``。
    @param reps 计时重复次数。
    @return portfolio 运行结果。
    """
    domain, portfolio = load_conv2d_portfolio()
    contract = conv2d_locator().load()
    instance = instantiate_v2(contract.spec, "fp16", size)
    point = domain.plan.find("fp16", size, usage)
    request = DispatchRequest.from_instance(instance, target_abi_digest=point.target_abi_digest, usage=usage)
    return run_portfolio_task(
        locator=conv2d_locator(),
        domain_plan=domain.plan,
        portfolio=portfolio,
        request=request,
        evidence=evidence_for(portfolio),
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(mode=mode, seed=7, reps=reps, cases=None),
        geometry_resolver_factory=msl_single_geometry_resolver,
        target_id="apple-m4-max",
    )


def project_portfolio_run(run, *, reps: int, usage: str) -> dict:
    """@brief 从一次真实运行取出身份、六案、安全和计时。

    @param run ``run_conv2d_portfolio`` 的结果。
    @param reps 这次运行使用的重复次数。
    @param usage 这次运行的 ``golden`` 或 ``benchmark``。
    @return 可归档的 run 文档。
    """
    report = run.result.report
    benchmark = report["benchmark"]
    document = run.document()
    identities = document["identities"]
    return {
        "active_manifest": document["active_manifest"],
        "candidate_sha256": identities["candidate_sha256"],
        "cases": [{"case": item["case"], "ok": bool(item["ok"])} for item in report["correctness"]["cases"]],
        "credits_variant": document["evaluation"]["credits_variant"],
        "domain_plan_digest": identities["domain_plan_digest"],
        "dtype": report["dtype"],
        "fallback_credits_variant": document["fallback_credits_variant"],
        "fallback_id": run.record.fallback_id,
        "fallback_reason": run.record.fallback_reason,
        "fresh_confirmation": False,
        "message": report["message"],
        "mode": report["mode"],
        "nondeterminism": list(report["safety"]["nondeterminism"]),
        "outcome": run.record.outcome,
        "performance_claim": document["performance_claim"],
        "portfolio_digest": identities["portfolio_digest"],
        "provider": run.record.provider,
        "publication_allowed": False,
        "reps": int(reps),
        "safety_ok": report["safety"]["ok"],
        "schema_id": "shapeforge.conv2d-portfolio-run.1",
        "seed": report["seed"],
        "size": report["size"],
        "status": report["status"],
        "task": report["task"],
        "timing": {
            "baseline_us": {name: float(value) for name, value in report["baseline_us"].items()},
            "best_baseline": benchmark.get("best_baseline"),
            "candidate_gpu_us_median": float(benchmark["candidate_gpu_us_median"]),
            "candidate_us": float(report["candidate_us"]),
            "noise_cv": float(benchmark["noise_cv"]),
            "noisy": bool(benchmark["noisy"]),
            "speedups": {name: float(value) for name, value in report["speedups"].items()},
        },
        "usage": usage,
        "variant_id": run.record.variant_id,
    }


def record_search_seed_runs() -> None:
    """@brief 重跑 small 与 paper 的搜索种子正式计时，并写成两份 run 和派生摘要。"""
    small = project_portfolio_run(
        run_conv2d_portfolio("small", usage="benchmark", mode="full", reps=15),
        reps=15,
        usage="benchmark",
    )
    paper = project_portfolio_run(
        run_conv2d_portfolio("paper", usage="benchmark", mode="full", reps=15),
        reps=15,
        usage="benchmark",
    )
    assert_timing_gate(small)
    assert_timing_gate(paper)
    dump_json(SMALL_RUN, small)
    dump_json(PAPER_RUN, paper)
    dump_json(FULL_EVAL, full_eval_document(load_json(SMALL_RUN), load_json(PAPER_RUN)))
    assert_recorded_portfolio()
