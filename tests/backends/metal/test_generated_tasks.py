"""The ``gen_*`` family: OpSpec-driven generator, no TOML, same task as the hand-written twin.

The generated family used to be rendered from ``tasks/generated/elementwise.toml``.  R2 retires
that manifest: each ``gen_<op>`` task owns ``tasks/generated/<op>/opspec.json`` and the closed
expression generator renders the prompt/known-good from it.  This file keeps the two properties
the TOML path used to guarantee -- the generated task IS the hand-written task on the contract,
the accounting, the tolerances and the prepared bytes -- and adds the ones the new path owes:
the document is the authority, and the retired entry points are gone.  No GPU.
"""

from __future__ import annotations

import copy
import hashlib
import dataclasses

import pytest
import torch

from tasks import GENERATED_NAMES, get_task, list_tasks
from tasks.generated import elementwise
from tasks.generated.elementwise import parse_expr
from tasks.relu.task import build as build_relu
from tasks.sigmoid.task import build as build_sigmoid
from tasks.silu.task import build as build_silu
from tasks.swiglu.task import build as build_swiglu
from tcenv.backends.metal.testing import prepare_case
from tcenv.opspec.model import load_opspec
from tcenv.track import REPO_ROOT, load_track
from tcenv.triton_runner import parse_launch_header

from ...support.triton_specs import jit_params

PAIRS = (
    ("relu", build_relu, "gen_relu"),
    ("sigmoid", build_sigmoid, "gen_sigmoid"),
    ("silu", build_silu, "gen_silu"),
    ("swiglu", build_swiglu, "gen_swiglu"),
)


def _sha(t: torch.Tensor) -> str:
    """@brief 计算张量连续化后的原始字节摘要，用于逐位比较两份数据。

    @param t 待摘要的张量。
    @return 十六进制 SHA-256。
    """
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def _field_differences(hand, generated) -> list[str]:
    """@brief 比较生成题与手写题的契约、记账、容差与用例（两份独立 Task）。

    @param hand 手写题目的 ``Task``。
    @param generated 生成式题目的 ``Task``。
    @return 不一致的字段名列表，完全一致时为空。
    """
    bad = []
    for field in ("kernel_name", "grid", "threads", "constants"):
        if getattr(hand.contract, field) != getattr(generated.contract, field):
            bad.append(field)
    if len(hand.contract.args) != len(generated.contract.args):
        bad.append("args")
    else:
        for a, b in zip(hand.contract.args, generated.contract.args):
            if (a.name, a.index, a.role, a.dtype, a.shape) != (b.name, b.index, b.role, b.dtype, b.shape):
                bad.append(f"arg {a.name}")
    for field in ("cases", "rtol", "atol_rel", "exact", "flops", "min_bytes", "description"):
        if getattr(hand, field) != getattr(generated, field):
            bad.append(field)
    for key in ("domain_case",):
        if (hand.meta or {}).get(key) != (generated.meta or {}).get(key):
            bad.append(f"meta.{key}")
    return bad


@pytest.mark.parametrize("hand_name,hand_build,gen_name", PAIRS)
def test_generated_task_matches_the_hand_written_one(hand_name, hand_build, gen_name):
    """@brief 生成任务与手写任务在契约、记账、容差、用例和逐位参考数据上一致。

    @param hand_name 手写任务名。
    @param hand_build 手写任务的构造函数。
    @param gen_name 生成任务名。
    """
    hand = hand_build("fp16", "small")
    gen = get_task(gen_name, "fp16", "small")
    assert _field_differences(hand, gen) == []
    assert gen.name == gen_name and hand.name == hand_name
    assert gen.semantic_instance is not None and gen.semantic_instance.op_name == gen_name
    for case in hand.cases:
        a, b = prepare_case(hand, case, 12345), prepare_case(gen, case, 12345)
        for name in a.rounded:
            assert _sha(a.rounded[name]) == _sha(b.rounded[name])
        for name in a.ref:
            assert _sha(a.ref[name]) == _sha(b.ref[name])
    if hand.meta.get("domain_case"):
        a, b = prepare_case(hand, "domain", 3), prepare_case(gen, "domain", 3)
        assert _sha(a.rounded["x"]) == _sha(b.rounded["x"])
        assert _sha(a.ref["y"]) == _sha(b.ref["y"])
    params = jit_params(gen.triton_source, gen.contract.kernel_name)
    header, nw = parse_launch_header(gen.triton_source)
    assert nw == gen.contract.num_warps
    rest = [p for p in params if not p.endswith("_ptr")]
    assert [p for p in rest if p not in header] == []


def test_the_retired_toml_path_is_gone():
    """@brief TOML 清单与读写它的入口都已删除（不只是未使用）。"""
    assert not (REPO_ROOT / "tasks" / "generated" / "elementwise.toml").exists()
    for retired in ("build", "load_manifest", "mismatches", "_validate", "_triton", "_metal", "MANIFEST"):
        assert not hasattr(elementwise, retired), f"tasks.generated.elementwise.{retired} still exists"
    # The live registry and the on-disk documents enumerate the same family.
    for stem in GENERATED_NAMES:
        directory = REPO_ROOT / "tasks" / "generated" / stem
        assert (directory / "opspec.json").is_file(), stem
        assert (directory / "track.json").is_file(), stem
        assert (directory / "task.py").is_file(), stem


@pytest.mark.parametrize("gen_name", tuple("gen_" + stem for stem in GENERATED_NAMES))
def test_every_generated_task_renders_from_its_own_opspec_document(gen_name):
    """@brief 题面/known-good 来自该题的 OpSpec 文档，track 只登记生成器身份。"""
    stem = gen_name[len("gen_") :]
    track = load_track(REPO_ROOT / "tasks" / "generated" / stem / "track.json")
    assert track.document["opspec"] is not None, "a generated pointwise task has one semantic file"
    assert str(track.opspec_path).endswith(f"tasks/generated/{stem}/opspec.json")
    assert track.spec_text_generator is not None and track.known_good_generator is not None
    document = load_opspec(track.opspec_path).document
    task = get_task(gen_name, "fp16", "small")
    # The rendered prompt carries this document's expression, not a hand-written template.
    expression = elementwise.document_expression(document)[0]
    rendered = elementwise.render(expression, "tl", [b.name for b in task.semantic_instance.tensors if b.role == "input"])
    assert f"y = {rendered}" in task.triton_source
    assert task.reference_sources["metal"] == task.naive_msl


def test_the_document_is_the_authority_for_the_generated_text():
    """@brief 改文档公式 ⇒ 题面、known-good、参考三者同时改变；文档里的未知算子被拒绝。"""
    task = get_task("gen_relu", "fp16", "small")
    document = copy.deepcopy(load_opspec(REPO_ROOT / "tasks/generated/relu/opspec.json").document)
    block = int(next(value for name, _type, value in task.contract.constants if name == "BLOCK_SIZE"))
    n = int(task.semantic_instance.symbols["N"])
    spelling = task.specialization.contract
    before = (
        elementwise.triton_source(
            document,
            kernel_name=task.contract.kernel_name,
            n=n,
            num_warps=task.contract.num_warps,
            block_size=block,
            tl_dtype="tl.float16",
        ),
        elementwise.msl(
            document,
            kernel_name=task.contract.kernel_name,
            constants=task.contract.constants,
            dtype_msl="half",
            threads=task.contract.threads[0],
        ),
    )
    assert before[0] == task.triton_source and before[1] == task.naive_msl
    edited = copy.deepcopy(document)
    edited["formula"] = "relu(x) * exp(x)"
    after = (
        elementwise.triton_source(
            edited,
            kernel_name=spelling.kernel_name,
            n=n,
            num_warps=task.contract.num_warps,
            block_size=block,
            tl_dtype="tl.float16",
        ),
        elementwise.msl(
            edited,
            kernel_name=spelling.kernel_name,
            constants=task.contract.constants,
            dtype_msl="half",
            threads=task.contract.threads[0],
        ),
    )
    assert after[0] != before[0] and after[1] != before[1]
    x = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    old = elementwise.reference_for(document)(x)
    new = elementwise.reference_for(edited)(x)
    assert not torch.equal(old, new)
    # The expression grammar is closed: an operator outside the table is refused, not evaluated.
    broken = copy.deepcopy(document)
    broken["formula"] = "tanh(x)"
    with pytest.raises(ValueError):
        elementwise.document_expression(broken)


def test_the_expression_grammar_rejects_escape_hatches():
    """@brief 表达式拒绝 import、属性访问与未知调用；比较只允许单次比较。"""
    with pytest.raises(ValueError):
        parse_expr("__import__('os')", ["x"])
    with pytest.raises(ValueError):
        parse_expr("x.real", ["x"])
    with pytest.raises(ValueError):
        parse_expr("unknown(x)", ["x"])
    with pytest.raises(ValueError):
        parse_expr("0 < x < 1", ["x"])
    with pytest.raises(ValueError):
        parse_expr("x + y", ["x"])


def test_the_generator_requires_the_registered_launch_constants():
    """@brief 生成器只接受登记 profile 的 ``N``/``BLOCK_SIZE`` 常量，别的常量组合直接拒绝。"""
    document = load_opspec(REPO_ROOT / "tasks/generated" / "relu" / "opspec.json").document
    with pytest.raises(ValueError, match="N/BLOCK_SIZE"):
        elementwise.msl(
            document,
            kernel_name="relu_kernel",
            constants=(("M", "uint", 16), ("BLOCK", "uint", 1024)),
            dtype_msl="half",
            threads=128,
        )
    with pytest.raises(ValueError, match="uint"):
        elementwise.msl(
            document,
            kernel_name="relu_kernel",
            constants=(("N", "int", 16), ("BLOCK_SIZE", "uint", 1024)),
            dtype_msl="half",
            threads=128,
        )


def test_generated_names_are_not_in_the_hand_written_list():
    """@brief 生成任务名不出现在手写题列表里，但仍可按名取到；未知名抛 ``KeyError``。"""
    assert "gen_relu" not in list_tasks(include_phase2=True)
    assert get_task("gen_relu", "fp16", "small").name == "gen_relu"
    with pytest.raises(KeyError):
        get_task("gen_no_such", "fp16", "small")


def test_a_tolerance_change_is_visible_in_the_task_identity():
    """@brief 容差是派生字段：迁移后的 Task 不能事后改容差而保留同一身份。"""
    task = get_task("gen_relu", "fp16", "small")
    with pytest.raises(AttributeError):
        task.rtol = task.rtol + 1
    with pytest.raises(ValueError):
        dataclasses.replace(task, rtol=task.rtol + 1)
