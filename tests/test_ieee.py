"""IEEE-754 special values (NaN, +-Inf, -0) in the correctness tests, and the determinism check on every case."""

from __future__ import annotations

import math

import pytest
import torch

from tasks import get_task, list_tasks
from tcenv.backends.metal.testing import IEEE_CASE, inject_specials, prepare_case, scaled_error

ALL_TASKS = list(list_tasks(include_phase2=True))
INF, NAN = float("inf"), float("nan")


# --------------------------------------------------------------------------- injection


@pytest.mark.parametrize("shape", [(4096,), (64, 128), (3, 5, 7), (1 << 16,)])
def test_inject_specials_plants_all_kinds_and_leaves_the_input_alone(shape):
    t = torch.randn(shape)
    before = t.clone()
    u = inject_specials(t, 17)
    assert torch.equal(t, before) and u.shape == t.shape
    flat = u.reshape(-1)
    assert int(torch.isnan(u).sum()) >= 1 and int((u == INF).sum()) >= 1 and int((u == -INF).sum()) >= 1
    assert int(((u == 0) & torch.signbit(u)).sum()) >= 1  # -0.0
    assert not torch.isfinite(flat[0]) or flat[0] == 0  # first and last element are always special
    assert not torch.isfinite(flat[-1]) or flat[-1] == 0
    changed = int((~torch.isfinite(u)).sum()) + int(((u == 0) & torch.signbit(u)).sum())
    assert 8 <= changed <= 256 + 1  # bounded, and most of the tensor stays untouched
    again = inject_specials(t, 17)  # reproducible from the seed (NaN != NaN, so compare with NaN replaced)
    assert torch.equal(torch.nan_to_num(again, nan=7.0), torch.nan_to_num(u, nan=7.0))


# --------------------------------------------------------------------------- comparison semantics


def err(out, ref, scale=1.0):
    return scaled_error(torch.tensor(out), torch.tensor(ref), scale, 1e-3, 1e-3, False)


def test_matching_special_values_are_exact_matches():
    assert err([NAN, INF, -INF, 1.0], [NAN, INF, -INF, 1.0]).max() == 0


@pytest.mark.parametrize(
    "out,ref",
    [
        (1.0, NAN),  # a NaN that was dropped
        (NAN, 1.0),  # a NaN that appeared
        (INF, -INF),  # wrong sign
        (65504.0, INF),  # overflow handled as a finite number
        (INF, 1.0),
        (NAN, INF),
        (INF, NAN),
    ],
)
def test_mismatching_special_values_fail(out, ref):
    e = err([out], [ref])
    assert math.isinf(float(e[0])) or float(e[0]) > 1, e


def test_finite_tolerance_is_unchanged():
    assert err([1.0005], [1.0], scale=1.0)[0] < 1
    assert err([1.5], [1.0], scale=1.0)[0] > 1


# --------------------------------------------------------------------------- the case on every task (CPU only)


@pytest.mark.parametrize("name", ALL_TASKS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_every_task_has_a_meaningful_ieee_case(name, dtype):
    task = get_task(name, dtype, "small")
    assert IEEE_CASE in task.cases
    d = prepare_case(task, IEEE_CASE, 99)
    assert any(int((~torch.isfinite(v)).sum()) for v in d.ref.values()), "the reference sees no special value at all"
    assert any(int(torch.isfinite(v).sum()) for v in d.ref.values()), "the whole reference output is non-finite: nothing is left to check numerically"
    for k, s in d.scale.items():
        assert bool(torch.isfinite(torch.as_tensor(s)).all()), f"non-finite error scale for {k}"
    n_special = sum(int(((~torch.isfinite(v.float())) | ((v.float() == 0) & torch.signbit(v.float()))).sum()) for v in d.rounded.values() if v.is_floating_point())
    assert n_special >= 8
    again = prepare_case(task, IEEE_CASE, 99)
    assert d.raw == again.raw  # reproducible from the seed
    assert d.raw != prepare_case(task, IEEE_CASE, 100).raw


# --------------------------------------------------------------------------- end to end (GPU)

gpu = pytest.mark.gpu

RELU_OK = "if (j < N) { float v = float(x[j]); y[j] = half(v < 0.0f ? 0.0f : v); }"
RELU_DROPS_NAN = "if (j < N) { float v = float(x[j]); y[j] = half(v > 0.0f ? v : 0.0f); }"
# correct on every case when the output starts out NaN-filled, but writes 0 instead of the result for NaN inputs when it does not
RELU_NAN_DEPENDS_ON_OUTPUT = "if (j < N) { float v = float(x[j]); half r = half(v < 0.0f ? 0.0f : v); y[j] = (isnan(v) && !isnan(float(y[j]))) ? half(0.0f) : r; }"


def relu_with(body: str) -> str:
    src = get_task("relu", "fp16", "small").naive_msl
    assert RELU_OK in src
    return src.replace(RELU_OK, body)


@gpu
def test_a_kernel_that_drops_nan_is_rejected_by_the_ieee_case_only():
    from tcenv.evaluate import evaluate

    rep = evaluate("relu", "fp16", "small", relu_with(RELU_DROPS_NAN), mode="quick", seed=5)
    assert rep.status == "incorrect"
    assert [c["case"] for c in rep.correctness["cases"] if not c["ok"]] == ["ieee"]
    assert "want nan" in rep.message and "got 0" in rep.message


@gpu
def test_determinism_is_checked_on_every_case_not_only_the_first():
    from tcenv.evaluate import evaluate

    rep = evaluate("relu", "fp16", "small", relu_with(RELU_NAN_DEPENDS_ON_OUTPUT), mode="quick", seed=5)
    assert rep.status == "unsafe" and "initial contents" in rep.message
    # all the earlier cases passed, the problem only shows on the last one
    assert [c["case"] for c in rep.correctness["cases"]][-1] == "ieee"
    assert all(c["ok"] for c in rep.correctness["cases"])


@gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_naive_kernels_of_all_tasks_pass_the_full_case_list_including_ieee(dtype):
    from tcenv.evaluate import evaluate

    for name in ALL_TASKS:
        t = get_task(name, dtype, "small")
        rep = evaluate(name, dtype, "small", t.naive_msl, mode="quick", seed=3)
        assert rep.status == "ok", f"{name} {dtype}: {rep.message}"
        assert IEEE_CASE in [c["case"] for c in rep.correctness["cases"]]


@gpu
@pytest.mark.parametrize("name", ALL_TASKS)
def test_at_least_one_vendor_baseline_reproduces_the_reference_on_special_values(name):
    """The case is only a fair requirement where IEEE semantics are achievable with a library implementation."""
    from tcenv.backends.metal.baselines import BaselineRunner
    from tcenv.backends.metal.testing import _floor

    task = get_task(name, "fp16", "small")
    d = prepare_case(task, IEEE_CASE, 1234)
    worst: dict[str, float] = {}
    for impl in task.baselines:
        outs = BaselineRunner(impl, task, d.rounded, rotate=1).outputs()
        worst[impl.name] = max(
            float(scaled_error(outs[a.name], d.ref[a.name], d.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype)).max())
            for a in task.contract.outputs
        )
    assert min(worst.values()) <= 1.0, worst


def test_building_a_task_for_one_dtype_does_not_change_another_dtypes_tolerances():
    """Regression: the Lion task used a module-level global for atol_rel, so the last dtype built won."""
    def scale(dtype):
        t = get_task("lion", dtype, "small")
        d = prepare_case(t, "tiny", 3)
        return float(t.error_scale(d.rounded, d.ref)["p_out"].max())

    first = scale("fp16")
    scale("bf16")
    assert scale("fp16") == first


def test_inject_specials_respects_a_cap_but_keeps_first_and_last():
    import torch

    from tcenv.backends.metal.testing import inject_specials

    t = torch.ones(128, 3, 3, 64)
    out = inject_specials(t, 5, max_n=4)
    special = ~torch.isfinite(out) | (out == 0)  # NaN, +-Inf, -0.0; the ones are never 0
    assert int(special.sum()) == 4
    assert bool(special.reshape(-1)[0]) and bool(special.reshape(-1)[-1])


def test_conv2d_ieee_reference_keeps_most_outputs_finite_and_matches_conv2d_on_finite_inputs():
    """CPU conv2d may smear one Inf over a Winograd tile as NaN; the IEEE reference must not, or the case checks nothing."""
    import torch
    import torch.nn.functional as F

    from tasks import get_task
    from tcenv.backends.metal.testing import prepare_case

    t = get_task("conv2d", "fp16", "small")
    d = prepare_case(t, "ieee", 11)
    y = d.ref["y"]
    assert torch.isfinite(y).double().mean() > 0.75
    assert torch.isinf(y).any() and torch.isnan(y).any()
    # finite inputs take the F.conv2d path; the tap-wise path must agree with it
    n = prepare_case(t, "normal", 12)
    ref = t.reference(n.rounded)["y"]
    tap = torch.zeros_like(ref, dtype=torch.float32)
    x, w = n.rounded["x"].float(), n.rounded["w"].float()
    OH, OW = ref.shape[1], ref.shape[2]
    for kh in range(3):
        for kw in range(3):
            tap += x[:, kh:kh + OH, kw:kw + OW, :] @ w[:, kh, kw, :].T
    assert torch.allclose(tap.double(), ref, rtol=1e-4, atol=1e-3 * float(ref.abs().max()))
