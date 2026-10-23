"""Generated pointwise tasks match the hand-written ones on the contract and on the reference. No GPU."""

from __future__ import annotations

import hashlib

import pytest
import torch

from tasks import get_task, list_tasks
from tasks.generated.elementwise import build as build_generated
from tasks.generated.elementwise import mismatches, parse_expr
from tasks.relu.task import build as build_relu
from tasks.sigmoid.task import build as build_sigmoid
from tasks.silu.task import build as build_silu
from tasks.swiglu.task import build as build_swiglu
from tcenv.backends.metal.testing import prepare_case
from tcenv.triton_runner import parse_launch_header

from ...support.triton_specs import jit_params

PAIRS = (
    ("relu", build_relu, "gen_relu"),
    ("sigmoid", build_sigmoid, "gen_sigmoid"),
    ("silu", build_silu, "gen_silu"),
    ("swiglu", build_swiglu, "gen_swiglu"),
)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("hand_name,hand_build,gen_name", PAIRS)
def test_generated_task_matches_the_hand_written_one(hand_name, hand_build, gen_name):
    hand = hand_build("fp16", "small")
    gen = build_generated(gen_name, "fp16", "small")
    assert mismatches(hand, gen) == []
    assert gen.meta["source"].startswith("tasks/generated/elementwise.toml#")
    assert gen.name == gen_name and hand.name == hand_name
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


def test_a_tolerance_change_fails_the_comparison():
    hand = build_relu("fp16", "small")
    gen = build_generated("gen_relu", "fp16", "small")
    assert mismatches(hand, gen) == []
    gen.rtol = hand.rtol + 1
    assert "rtol" in mismatches(hand, gen)


def test_the_manifest_rejects_escape_hatches_and_a_binary_domain_sweep():
    with pytest.raises(ValueError):
        parse_expr("__import__('os')", ["x"])
    with pytest.raises(ValueError):
        parse_expr("x.real", ["x"])
    with pytest.raises(ValueError):
        parse_expr("unknown(x)", ["x"])
    from tasks.generated.elementwise import _validate

    with pytest.raises(ValueError, match="domain_case"):
        _validate({"task": "gen_pair", "inputs": ["a", "b"], "expr": "a * b", "domain_case": True})


def test_generated_names_are_not_in_the_hand_written_list():
    assert "gen_relu" not in list_tasks(include_phase2=True)
    assert get_task("gen_relu", "fp16", "small").name == "gen_relu"
    with pytest.raises(KeyError):
        get_task("gen_no_such", "fp16", "small")
