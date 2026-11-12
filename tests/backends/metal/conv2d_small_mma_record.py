"""Real-Metal runner and recorder for the P2-J conv2d portfolio."""

from __future__ import annotations

from tcenv.backends.metal.search_candidate import evidence_for
from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.candidate import parse_portfolio
from tcenv.portfolio.dispatch import DispatchRequest
from tcenv.portfolio_assembly import domain_for_sizes, parent_entries_from_tree
from tcenv.target import load_target
from tcenv.taskcontract.portfolio_runner import run_portfolio_task
from tests.backends.metal.conv2d_portfolio_record import conv2d_locator, project_portfolio_run
from tests.support.conv2d_small_mma import (
    FULL_EVAL,
    PAPER_RUN,
    SMALL_RUN,
    TREE,
    assert_recorded_portfolio,
    dump_json,
    full_eval_document,
    load_json,
)


def load_conv2d_small_mma_portfolio():
    contract = conv2d_locator().load()
    domain = domain_for_sizes(contract, load_target("apple-m4-max"), ("small", "paper"))
    portfolio = parse_portfolio(parent_entries_from_tree(TREE, domain), domain_plan=domain.plan)
    return domain, portfolio


def run_conv2d_small_mma(size: str, *, usage: str = "golden", mode: str = "quick", reps: int = 1):
    domain, portfolio = load_conv2d_small_mma_portfolio()
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


def record_search_seed_runs() -> None:
    small = project_portfolio_run(
        run_conv2d_small_mma("small", usage="benchmark", mode="full", reps=15), reps=15, usage="benchmark"
    )
    paper = project_portfolio_run(
        run_conv2d_small_mma("paper", usage="benchmark", mode="full", reps=15), reps=15, usage="benchmark"
    )
    dump_json(SMALL_RUN, small)
    dump_json(PAPER_RUN, paper)
    dump_json(FULL_EVAL, full_eval_document(load_json(SMALL_RUN), load_json(PAPER_RUN)))
    assert_recorded_portfolio()
