"""单候选 portfolio 的构造：恒真 guard、tune/unseen 纪律与 fresh seed。"""

from __future__ import annotations

import pytest

from taic.portfolio_confirmation import SEARCH_SEED, confirmation_seed_for
from taic.task_binding import bind_task
from tcenv.backends.metal.search_candidate import (
    confirmation_input,
    domain_for,
    envelope_entries,
    evidence_for,
    mode_for,
    portfolio_for,
    request_for,
    usage_for,
)
from tcenv.portfolio.dispatch import ExactDispatchResolver
from tcenv.target import load_target

SOURCE = "/* taic-launch\nlaunch gemm_kernel grid=1,1,1 threads=1,1,1\n*/\nkernel void gemm_kernel() {}\n"


def _binding():
    return bind_task("gemm", "fp16", "small")


def test_the_single_candidate_portfolio_has_one_always_true_variant():
    binding = _binding()
    target = load_target(binding.target_id)
    plan, authority = domain_for(binding.contract, target)
    portfolio = portfolio_for(SOURCE, plan)
    assert len(portfolio.variant_ids) == 1 and portfolio.fallback_id is None
    evidence = evidence_for(portfolio)
    assert set(evidence) == set(portfolio.variant_ids)
    resolver = ExactDispatchResolver(
        domain_plan=plan, portfolio=portfolio, evidence=evidence,
        spec=binding.contract.spec, target=target, instances=authority.instances,
        source_preflight=False,
    )
    record = resolver.resolve(request_for(binding.contract, plan))
    # 恒真 guard 不引用任何 symbol：dispatch 永远 resolved，不存在「搜索候选落在域外」。
    assert record.outcome == "resolved" and record.provider == "candidate"


def test_the_search_mode_follows_the_declared_usage():
    binding = _binding()
    assert usage_for(binding.contract) == "benchmark"
    assert mode_for(binding.contract) == "full"


def test_the_confirmation_input_freezes_the_tuned_instance_and_a_required_unseen_point():
    binding = _binding()
    target = load_target(binding.target_id)
    frozen = confirmation_input(
        binding.locator, SOURCE, target=target, guidance_digest="0" * 64, master_seed=12345,
    )
    tuned = "fp16/small/benchmark"
    golden = "fp16/small/golden"
    assert frozen.validation.tune_shards == (tuned,)
    assert frozen.validation.unseen_instance_keys == (golden,)
    assert frozen.validation.required_instance_keys == (golden,)
    # fresh 只属于确认：确认 seed 由 plan 派生，且绝不等于搜索期的 7。
    seed = confirmation_seed_for(frozen.validation, tuned)
    assert seed != SEARCH_SEED
    assert frozen.freeze.task_pack_digest == binding.locator.deployment_pack_digest
    assert frozen.request.instance_key == tuned
    # 同一候选/同一 run_seed 重建完全一致（resume 的可重放性）。
    again = confirmation_input(
        binding.locator, SOURCE, target=target, guidance_digest="0" * 64, master_seed=12345,
    )
    assert again.freeze.digest == frozen.freeze.digest
    assert again.validation.digest == frozen.validation.digest


def test_the_envelope_is_byte_deterministic():
    binding = _binding()
    target = load_target(binding.target_id)
    plan, _ = domain_for(binding.contract, target)
    assert envelope_entries(SOURCE, plan) == envelope_entries(SOURCE, plan)
