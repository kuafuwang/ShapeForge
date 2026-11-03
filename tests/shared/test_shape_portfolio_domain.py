"""Shape portfolio P0: semantic domain plan, canonical guards, trunc_div/rem (CPU).

These are the closed-contract rules of the shape-portfolio plan:

* a semantic domain is a **finite, explicitly named** OpSpec instance pool; ``non_finite`` is
  registered vocabulary but not activated, and a foreign dtype/size point is refused instead of
  being absorbed by numeric proximity;
* a guard is the existing ``tcenv.track.expr`` functional/JSON AST, canonicalised only
  structurally (``gt``/``ge`` re-orient, ``and``/``or`` flatten+sort, commutative ``eq``/``ne``
  sort) and hashed as that canonical AST — never as the author's string;
* ``trunc_div``/``rem`` truncate toward zero with the remainder's sign following the dividend and
  refuse a zero divisor, while ``/`` keeps its A2 exact-division semantics;
* a v2 ``@variant`` stays refused, so a portfolio cannot revive a second task contract.

No GPU, no model, no search.
"""

from __future__ import annotations

import copy
import json

import pytest

from tcenv.contract_errors import GeometryError, TaskContractError, UnsupportedContractCapability
from tcenv.opspec.v2 import instantiate_v2
from tcenv.portfolio.domain import (
    SEMANTIC_DOMAIN_PLAN_SCHEMA_ID,
    canonical_ast,
    canonical_guard,
    guard_ast_document,
    guard_digest,
    load_semantic_domain_plan,
    normalize_semantic_domain_plan,
    parse_guard,
    parse_semantic_domain_plan,
    semantic_instance_digest,
    target_generation_digest,
)
from tcenv.target import DEFAULT_TARGET_ID, load_target
from tcenv.taskcontract import load_task_contract
from tcenv.track.expr import evaluate_term, parse_term

from ..support.shape_portfolio import (
    PORTFOLIO_OP,
    load_portfolio_task,
    portfolio_instances,
    portfolio_opspec,
    write_portfolio_task,
)
from ..support.v2_tasks import sem_opspec, write_task

PROFILE = load_target(DEFAULT_TARGET_ID)
SIZES = {"tail", "below", "threshold", "above", "large"}
TARGET_DIGEST = target_generation_digest(PROFILE)

#: Golden digests of the canonical guard AST.  They pin the algorithm (not a self-referential
#: fixture): a different canonical form or a digest of the raw spelling changes these values.
GT_LT_DIGEST = "d01c11ab2cab72b80a7bc49344a0322273c163a18a502a65de08d23fbb65e891"
AND_DIGEST = "3392438f51ac37993df43e217d5bafbda14ff095e03282fbc4e273f2e4399992"
OR_DIGEST = "f9eacfdc56608972950e1640d9fff497d0943eeea7125e32b447701fb5f45de6"
SUB_LT_DIGEST = "af8707cdf8369f4980d659ae263f745c5bb240c47755a5457c5b516c5688348d"
REM_DIGEST = "9043507433934df5f01ec73d2feca30267f44b3f13e930bd804d805f1be986b2"
ADD_LT_DIGEST = "4407fca9dc83e2ba24976dd7aedd2e85d43395a01e503bdf33c52878bc54c2df"
MUL_LT_DIGEST = "d7b3cfff2778b6eac82e22348af8e7608d1483166174a320805a38c1a1fd4091"
MIN_LT_DIGEST = "445e972cbc0275d0526c9f12b5ab88fb72fb2f731e40864f529aefd2ba0fe8a9"
MAX_LT_DIGEST = "5d97bd96c098764fac004d8a207090790998643e203c280effc3f92f289b275b"
SUB_ORDER_DIGEST = "cb68bf135c1d32b26d2d24f85cd1e9429d567951975c04380616a20678ec95aa"


def _fixture(tmp_path):
    """@brief 固定有限 portfolio 题 + 规范化的域计划。

    @param tmp_path 临时目录。
    @return ``(contract, instances, plan)``。
    """
    contract = load_portfolio_task(tmp_path)
    instances = portfolio_instances(contract)
    plan = normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE)
    return contract, instances, plan


def test_p0_a_finite_domain_plan_normalizes_explicit_named_instances(tmp_path):
    """@brief 有限域规范化：显式点集、规范顺序、完整语义/ABI 身份、可重放摘要。"""
    contract, instances, plan = _fixture(tmp_path)
    assert plan.schema_id == SEMANTIC_DOMAIN_PLAN_SCHEMA_ID
    assert plan.domain_kind == "finite"
    assert plan.opspec_digest == contract.spec.digest
    assert plan.target_generation_digest == TARGET_DIGEST
    assert {item.size for item in plan.instances} == SIZES
    assert {item.usage for item in plan.instances} == {"golden", "benchmark"}
    assert plan.keys == tuple(sorted(plan.keys))
    assert len(plan.instances) == len(SIZES) * 2
    for item in plan.instances:
        assert len(item.semantic_instance_digest) == 64 and len(item.target_abi_digest) == 64
        assert item.instance_key == f"fp16/{item.size}/{item.usage}"
    # Golden (correctness) and benchmark are separate finite points; a request names one.
    assert plan.find("fp16", "threshold", "golden") is not None
    assert plan.find("fp16", "threshold", "benchmark") is not None
    assert plan.find("fp16", "threshold", "golden").semantic_instance_digest == plan.find(
        "fp16", "threshold", "benchmark"
    ).semantic_instance_digest
    # The semantic digest is the instance identity helper, not a source path or author field.
    by_size = {instance.size: instance for instance in instances}
    assert plan.find("fp16", "tail", "golden").semantic_instance_digest == semantic_instance_digest(
        by_size["tail"]
    )
    # Normalization is order-independent and the document round-trips to the same plan.
    reversed_plan = normalize_semantic_domain_plan(contract.spec, tuple(reversed(instances)), target=PROFILE)
    assert reversed_plan.digest == plan.digest
    assert reversed_plan.document() == plan.document()
    reparsed = parse_semantic_domain_plan(plan.document())
    assert reparsed.digest == plan.digest
    assert reparsed.document() == plan.document()
    # A plan is identity-bound to its OpSpec: a renamed task is a different domain.
    other = load_task_contract(write_portfolio_task(tmp_path / "other", op_name="portfolio_other"))
    other_plan = normalize_semantic_domain_plan(other.spec, portfolio_instances(other), target=PROFILE)
    assert other_plan.digest != plan.digest


def test_p0_a_non_finite_domain_is_refused_instead_of_treated_as_sampled_finite(tmp_path):
    """@brief ``non_finite`` 不激活：规范化与解析都拒绝，不能借有限点集宣称全域覆盖。"""
    contract, instances, plan = _fixture(tmp_path)
    with pytest.raises(UnsupportedContractCapability, match="not activated"):
        normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE, domain_kind="non_finite")
    with pytest.raises(TaskContractError, match="unknown domain_kind"):
        normalize_semantic_domain_plan(contract.spec, instances, target=PROFILE, domain_kind="sampled")
    document = plan.document()
    document["domain_kind"] = "non_finite"
    with pytest.raises(UnsupportedContractCapability, match="not activated"):
        parse_semantic_domain_plan(document)
    document = plan.document()
    document["domain_kind"] = "sampled"
    with pytest.raises(TaskContractError, match="unknown domain_kind|not one of|domain_kind"):
        parse_semantic_domain_plan(document)
    truncated = plan.document()
    truncated["instances"] = []
    with pytest.raises(TaskContractError, match="fewer than 1 items"):
        parse_semantic_domain_plan(truncated)


def test_p0_a_dtype_or_size_outside_the_opspec_is_refused(tmp_path):
    """@brief 域外 dtype/size 拒绝：OpSpec 不接受、外来 OpSpec 的实例也不能混入本域。"""
    contract, _, plan = _fixture(tmp_path)
    with pytest.raises(Exception):
        # The OpSpec itself refuses an undeclared dtype; the domain never invents the point.
        instantiate_v2(contract.spec, "fp32", "threshold")
    with pytest.raises(Exception):
        instantiate_v2(contract.spec, "fp16", "not_a_declared_size")
    # An instance assembled from another OpSpec is not a point of this finite domain, even when
    # its dtype/size spellings happen to exist.
    other_spec = portfolio_opspec("portfolio_other")
    other_spec["type_vars"] = {"T": ["fp16", "bf16"]}
    other_spec["numerics"]["tolerances"]["bf16"] = {"rtol": 0.01, "atol_rel": 0.01}
    other = load_task_contract(write_task(tmp_path / "foreign", op_name="portfolio_other", opspec=other_spec))
    foreign = instantiate_v2(other.spec, "bf16", "threshold")
    with pytest.raises(UnsupportedContractCapability, match="different OpSpec"):
        normalize_semantic_domain_plan(contract.spec, (foreign,), target=PROFILE)
    # A same-spelling instance from a different task is refused for the same reason.
    sem = load_task_contract(write_task(tmp_path / "sem", op_name="sem_gemm_neg", opspec=sem_opspec()))
    with pytest.raises(UnsupportedContractCapability, match="different OpSpec"):
        normalize_semantic_domain_plan(contract.spec, (sem.instance,), target=PROFILE)
    # Reading the plan from disk keeps the same closed schema.
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.document()), encoding="utf-8")
    assert load_semantic_domain_plan(path).digest == plan.digest


def test_p0_a_guard_digest_hashes_the_canonical_ast_not_the_spelling():
    """@brief 句法等价的 guard 得到同一 digest；语义不同的 guard 得到不同 digest。"""
    assert guard_digest(["gt", "N", 63]) == GT_LT_DIGEST
    assert guard_digest(["lt", 63, "N"]) == GT_LT_DIGEST
    assert guard_digest(["ge", "N", 64]) == guard_digest(["le", 64, "N"])
    assert guard_digest(["and", ["ge", "N", 64], ["lt", "N", 128]]) == AND_DIGEST
    assert guard_digest(["and", ["lt", "N", 128], ["le", 64, "N"]]) == AND_DIGEST
    assert guard_digest(["or", ["eq", "N", 7], ["and", ["lt", "N", 32], ["ne", "M", 0]]]) == OR_DIGEST
    assert (
        guard_digest(["or", ["and", ["ne", "M", 0], ["lt", "N", 32]], ["eq", 7, "N"]])
        == OR_DIGEST
    )
    assert guard_digest(["eq", ["rem", "N", 2], 1]) == REM_DIGEST
    assert guard_digest(["eq", 1, ["rem", "N", 2]]) == REM_DIGEST
    # Different semantics stay different: a changed operator, operand or threshold.
    assert guard_digest(["lt", "N", 64]) != guard_digest(["lt", "N", 65])
    assert guard_digest(["lt", "N", 64]) != guard_digest(["le", "N", 64])
    assert guard_digest(["eq", ["rem", "N", 2], 1]) != guard_digest(["eq", ["rem", "N", 4], 1])
    # A digest never covers the raw string: two spellings share one digest, one spelling does not
    # make a second identity available.
    assert guard_digest('lt(N, 64)') == guard_digest(["lt", "N", 64])


def test_p0_a_guard_canonicalization_is_pure_structure_and_idempotent():
    """@brief 只做结构重排：不改写算术、不消项、不引入优先级。"""
    # ``a-b<0`` must not be rewritten to ``a<b`` (that would be algebraic simplification).
    sub = canonical_guard(["lt", ["-", "N", 63], 0])
    assert guard_ast_document(sub.node) == {
        "call": "lt",
        "args": [
            {"call": "-", "args": [{"var": "N"}, {"const": 63}]},
            {"const": 0},
        ],
    }
    assert sub.digest == SUB_LT_DIGEST
    assert sub.digest != GT_LT_DIGEST
    # Flattening is observable in the AST and keeps one operator level.
    nested = canonical_guard(["and", ["lt", "N", 8], ["and", ["gt", "N", 0], ["lt", "N", 4]]])
    assert guard_ast_document(nested.node)["call"] == "and"
    assert len(guard_ast_document(nested.node)["args"]) == 3
    # Idempotence: canonicalising again changes nothing (same AST, same digest).
    assert canonical_ast(nested.node) == nested.node
    once = canonical_ast(parse_term(["gt", "N", 63], "t"))
    assert canonical_ast(once) == once
    assert guard_digest(["lt", 63, "N"]) == GT_LT_DIGEST
    # Every registered commutative function is sorted structurally, not just eq/ne: operands are
    # reordered by canonical bytes, never folded or cancelled.
    commutative = (
        ("add", ["lt", ["+", "N", 1], 5], ["lt", ["+", 1, "N"], 5], ADD_LT_DIGEST),
        ("mul", ["lt", ["*", "N", 7], 5], ["lt", ["*", 7, "N"], 5], MUL_LT_DIGEST),
        ("min", ["lt", ["min", "N", 7], 5], ["lt", ["min", 7, "N"], 5], MIN_LT_DIGEST),
        ("max", ["lt", ["max", "N", 7], 5], ["lt", ["max", 7, "N"], 5], MAX_LT_DIGEST),
    )
    for label, forward, backward, digest in commutative:
        assert guard_digest(forward) == digest, label
        assert guard_digest(backward) == digest, label
        assert canonical_guard(forward).digest == canonical_guard(backward).digest, label
        assert canonical_ast(parse_term(backward, label)) == canonical_ast(parse_term(forward, label)), label
        assert canonical_ast(canonical_ast(parse_term(backward, label))) == canonical_ast(
            parse_term(backward, label)
        ), label
    # Non-commutative subtraction keeps the author's order: sorting it would change semantics.
    sub_forward = canonical_guard(["lt", ["-", "N", 1], 5])
    sub_backward = canonical_guard(["lt", ["-", 1, "N"], 5])
    assert sub_forward.digest == SUB_ORDER_DIGEST
    assert sub_forward.digest != sub_backward.digest
    assert guard_ast_document(sub_forward.node)["args"][0]["args"] == [{"var": "N"}, {"const": 1}]
    assert guard_ast_document(sub_backward.node)["args"][0]["args"] == [{"const": 1}, {"var": "N"}]
    # A name is not a predicate and a predicate is not an integer: the closed type rule holds.
    with pytest.raises(GeometryError, match="predicate expression"):
        canonical_guard({"min": ["N", 4]})
    with pytest.raises(GeometryError, match="must be a predicate"):
        guard_digest(["not", "N"])


def test_p0_parse_guard_rejects_forged_and_non_canonical_digests():
    """@brief 声明 digest 必须就是规范 AST 摘要；伪造/原始字符串摘要/开放字段都拒绝。"""
    import hashlib

    expr = ["and", ["ge", "N", 64], ["lt", "N", 128]]
    good = {"expr": expr, "digest": AND_DIGEST}
    assert parse_guard(good).digest == AND_DIGEST
    raw_spelling = hashlib.sha256(b'["and", ["ge", "N", 64], ["lt", "N", 128]]').hexdigest()
    with pytest.raises(TaskContractError, match="canonical AST digest"):
        parse_guard({"expr": expr, "digest": raw_spelling})
    with pytest.raises(TaskContractError, match="canonical AST digest"):
        parse_guard({"expr": ["lt", 63, "N"], "digest": AND_DIGEST})
    with pytest.raises(TaskContractError, match="64-character lowercase hex"):
        parse_guard({"expr": expr, "digest": "not-a-digest"})
    with pytest.raises(TaskContractError, match="expected exactly"):
        parse_guard({"expr": expr})
    with pytest.raises(TaskContractError, match="expected exactly"):
        parse_guard({"expr": expr, "digest": AND_DIGEST, "priority": 1})
    with pytest.raises(GeometryError, match="predicate"):
        parse_guard({"expr": ["rem", "N", 2], "digest": "0" * 64})
    with pytest.raises(GeometryError, match="unknown geometry function"):
        parse_guard({"expr": ["smt_forall", "N"], "digest": "0" * 64})


def test_p0_trunc_div_and_rem_truncate_toward_zero_with_the_dividend_sign():
    """@brief 负数除法向零截断，余数符号跟被除数；奇数 tail 用 rem；除零是 GeometryError。"""
    names = {"N": 7, "M": 63}
    cases = (
        (["trunc_div", "-7", "2"], -3),
        (["trunc_div", "7", "-2"], -3),
        (["trunc_div", "-7", "-2"], 3),
        (["trunc_div", "7", "2"], 3),
        (["trunc_div", "-1", "2"], 0),
        (["trunc_div", "1", "-2"], 0),
        (["rem", "-7", "2"], -1),
        (["rem", "7", "-2"], 1),
        (["rem", "-7", "-2"], -1),
        (["rem", "7", "2"], 1),
        (["rem", "-8", "2"], 0),
        (["rem", "-1", "2"], -1),
    )
    for term, expected in cases:
        assert evaluate_term(term, {}, where="p0", expect="int") == expected, term
    # The identity a-b == trunc_div(a,b)*b + rem(a,b) holds for every sign combination.
    for numerator in (-9, -8, -1, 0, 1, 8, 9):
        for denominator in (-3, -2, 2, 3):
            quotient = evaluate_term(["trunc_div", numerator, denominator], {}, where="p0", expect="int")
            remainder = evaluate_term(["rem", numerator, denominator], {}, where="p0", expect="int")
            assert quotient * denominator + remainder == numerator
            assert (remainder == 0) or (remainder > 0) == (numerator > 0)
    for term in (["trunc_div", "1", "0"], ["rem", "1", "0"]):
        with pytest.raises(GeometryError, match="divisor is zero"):
            evaluate_term(term, {}, where="p0", expect="int")
    # Odd tail: 7/63/65 are odd, 64/1024 are not.  A "tail" predicate must use rem, not ``/``.
    assert evaluate_term(["eq", ["rem", "N", 2], 1], names, where="p0", expect="bool") is True
    for odd in (7, 63, 65):
        assert evaluate_term(["eq", ["rem", "N", 2], 1], {"N": odd}, where="p0", expect="bool") is True
    for even in (64, 1024):
        assert evaluate_term(["eq", ["rem", "N", 2], 1], {"N": even}, where="p0", expect="bool") is False


def test_p0_exact_division_keeps_the_a2_semantics():
    """@brief ``/`` 仍是精确整除：不得为 portfolio 改成截断除法。"""
    assert evaluate_term("M*H/256", {"M": 64, "H": 512}, where="p0", expect="int") == 128
    assert evaluate_term(["eq", ["/", "N", 2], 32], {"N": 64}, where="p0", expect="bool") is True
    with pytest.raises(GeometryError, match="divide evenly"):
        evaluate_term(["/", "7", "2"], {}, where="p0", expect="int")
    with pytest.raises(GeometryError, match="divide evenly"):
        evaluate_term(["/", "7", "0"], {}, where="p0", expect="int")
    # Negative division is still exact; trunc_div is the explicit opt-in for truncation.
    assert evaluate_term(["/", "-8", "2"], {}, where="p0", expect="int") == -4
    with pytest.raises(GeometryError, match="divide evenly"):
        evaluate_term(["/", "-7", "2"], {}, where="p0", expect="int")


def test_p0_a_v2_variant_is_still_refused(tmp_path):
    """@brief v2 ``@variant`` 与 discovery.variant_acceptance 仍在装配期拒绝。"""
    task_dir = write_portfolio_task(tmp_path)
    with pytest.raises(UnsupportedContractCapability, match="@variant"):
        load_task_contract(task_dir, variant="64x32t256")
    document = copy.deepcopy(portfolio_opspec())
    document["discovery"]["variant_acceptance"] = {
        "factory_id": "reserved.variant.factory",
        "implementation_version": 1,
    }
    document["discovery"]["default_instance"] = {"dtype": "fp16", "size": "tail"}
    variant_dir = write_portfolio_task(tmp_path / "variant", op_name=PORTFOLIO_OP)
    (variant_dir / "opspec.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UnsupportedContractCapability, match="variant_acceptance"):
        load_task_contract(variant_dir)


def test_p0_the_guard_name_space_is_symbols_and_integer_attrs_only(tmp_path):
    """@brief guard 只看符号与整数 attr；张量名/浮点 attr/未知名字是未知字段。"""
    contract = load_portfolio_task(tmp_path)
    names = {name: int(value) for name, value in contract.instance.symbols.items()}
    assert evaluate_term(["le", "N", 64], names, where="p0", expect="bool") is True
    for unknown in ("x", "y", "scratch", "NO_SUCH_SYMBOL"):
        with pytest.raises(GeometryError, match="undeclared name"):
            evaluate_term(["lt", unknown, 64], names, where="p0", expect="bool")
