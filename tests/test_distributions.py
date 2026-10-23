"""Property tests for the adversarial generators in tcenv.distributions.

``sample`` draws a float32 CPU tensor. The ``dtype`` argument only picks magnitude
limits (the default ``large`` cap, and the ``tiny`` window). Callers round to the
storage type later.

IEEE-754 specials are not produced here. ``tcenv.backends.metal.testing.inject_specials`` plants
+Inf, -Inf, NaN and -0.0 when a task runs the ``ieee`` case. ``sample("ieee")``
is rejected.
"""

from __future__ import annotations

import pytest
import torch

from tcenv.distributions import GENERIC_CASES, generator, sample
from tcenv.dtypes import FLOAT_MAX, FLOAT_TINY_NORMAL, FLOAT_TINY_SUBNORMAL

CASES = ("normal", "uniform", "wide_range", "large", "tiny", "outlier", "cancel", "const", "sparse")
SEEDED = ("normal", "uniform", "wide_range", "large", "tiny", "outlier", "cancel", "sparse")
SHAPES = [(8,), (4, 5), (2, 3, 4), (), (0, 3)]


def _default_large_top(dtype: str) -> float:
    return min(FLOAT_MAX[dtype] * 0.25, 1.0e4)


def test_generic_cases_names_the_seven_task_distributions():
    assert GENERIC_CASES == ("normal", "wide_range", "large", "tiny", "outlier", "cancel", "const")


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("shape", SHAPES)
def test_sample_returns_float32_cpu_of_the_requested_shape(case, shape):
    t = sample(case, shape, 1, "fp16")
    assert t.dtype == torch.float32
    assert t.device.type == "cpu"
    assert tuple(t.shape) == tuple(shape)
    assert torch.isfinite(t).all()
    assert not torch.isnan(t).any()
    assert not torch.isinf(t).any()


def test_a_list_shape_is_accepted():
    t = sample("normal", [2, 3], 1)
    assert tuple(t.shape) == (2, 3) and t.dtype == torch.float32


@pytest.mark.parametrize("case", SEEDED)
def test_the_same_seed_reproduces_the_same_tensor(case):
    assert torch.equal(sample(case, (32, 16), 2026, "bf16"), sample(case, (32, 16), 2026, "bf16"))


@pytest.mark.parametrize("case", SEEDED)
def test_a_different_seed_changes_the_tensor(case):
    assert not torch.equal(sample(case, (32, 16), 2026), sample(case, (32, 16), 2027))


def test_sample_folds_the_seed_down_to_63_bits():
    assert torch.equal(sample("normal", (24,), (1 << 63) + 5), sample("normal", (24,), 5))
    assert torch.equal(sample("wide_range", (24,), -1), sample("wide_range", (24,), (1 << 63) - 1))


def test_const_ignores_the_seed_and_fills_three_quarters_unless_mag_scales_it():
    filled = sample("const", (2, 3), 1)
    assert torch.equal(filled, sample("const", (2, 3), 99))
    assert torch.equal(filled, torch.full((2, 3), 0.75))
    assert torch.equal(sample("const", (2, 3), 1, mag=2.5), torch.full((2, 3), 2.5))
    assert filled.dtype == torch.float32 and torch.isfinite(filled).all()


def test_normal_has_no_nan_or_inf_and_scales_with_mag():
    mag = 2.0
    t = sample("normal", (16384,), 19, mag=mag)
    assert t.dtype == torch.float32 and tuple(t.shape) == (16384,)
    assert not torch.isnan(t).any() and not torch.isinf(t).any()
    assert abs(float(t.mean())) < 0.15 * mag
    assert abs(float(t.std()) - mag) < 0.1 * mag
    base = sample("normal", (64,), 19, mag=1.0)
    assert torch.equal(sample("normal", (64,), 19, mag=4.0), base * 4)


def test_uniform_stays_inside_a_symmetric_interval():
    mag = 3.0
    t = sample("uniform", (8192,), 6, mag=mag)
    assert torch.isfinite(t).all()
    assert torch.all(t.abs() <= torch.tensor(mag, dtype=torch.float32))
    assert float(t.min()) < -0.9 * mag and float(t.max()) > 0.9 * mag
    base = sample("uniform", (64,), 6, mag=1.0)
    assert torch.equal(sample("uniform", (64,), 6, mag=4.0), base * 4)


def test_wide_range_spans_several_decades_and_both_signs():
    t = sample("wide_range", (8192,), 4, mag=1.0)
    assert torch.isfinite(t).all() and not torch.isnan(t).any()
    log_abs = torch.log10(t.abs())
    lo, hi = float(log_abs.min()), float(log_abs.max())
    assert -3.02 < lo < -2.5
    assert 1.5 < hi < 2.02
    assert hi - lo > 4
    assert (t < 0).any() and (t > 0).any()
    base = sample("wide_range", (64,), 4, mag=1.0)
    assert torch.equal(sample("wide_range", (64,), 4, mag=4.0), base * 4)


@pytest.mark.parametrize(
    "dtype,large",
    [
        ("fp16", None),
        ("bf16", None),
        ("fp32", None),
        ("fp16", 2.0e4),
        ("bf16", 1.0e20),
        ("fp32", 1.0e20),
    ],
)
def test_large_reaches_large_finite_values_the_dtype_can_hold(dtype, large):
    top = _default_large_top(dtype) if large is None else large
    assert top <= FLOAT_MAX[dtype]
    t = sample("large", (8192,), 11, dtype, large=large)
    assert t.dtype == torch.float32 and tuple(t.shape) == (8192,)
    assert torch.isfinite(t).all()
    limit = torch.tensor(top, dtype=torch.float32)
    assert torch.all(t.abs() <= limit)
    assert float(t.min()) < -0.5 * top and float(t.max()) > 0.5 * top
    assert float(t.abs().max()) <= FLOAT_MAX[dtype]


def test_tiny_fp16_reaches_subnormal_magnitudes_and_stays_finite():
    t = sample("tiny", (8192,), 5, "fp16")
    assert t.dtype == torch.float32 and torch.isfinite(t).all()
    mag = t.abs()
    normal = FLOAT_TINY_NORMAL["fp16"]
    sub = FLOAT_TINY_SUBNORMAL["fp16"]
    assert mag.min() > sub * 0.5
    assert mag.max() < normal * 8
    assert (mag < normal).any() and (mag >= normal).any()
    assert (t < 0).any() and (t > 0).any()
    assert torch.equal(t, sample("tiny", (8192,), 5, "fp16"))
    wider = sample("tiny", (128,), 7, "fp32")
    assert wider.dtype == t.dtype == torch.float32
    assert not torch.equal(wider, sample("tiny", (128,), 7, "fp16"))


@pytest.mark.parametrize("dtype", ["bf16", "fp32"])
def test_tiny_wider_dtypes_stay_above_the_smallest_normal(dtype):
    t = sample("tiny", (8192,), 5, dtype)
    assert t.dtype == torch.float32 and torch.isfinite(t).all()
    normal = FLOAT_TINY_NORMAL[dtype]
    lo, hi = normal * 2.0, normal * 2.0 * 4096.0
    mag = t.abs()
    assert float(mag.min()) > normal
    assert float(mag.max()) < hi * 1.01
    assert float(mag.min()) < lo * 10 and float(mag.max()) > hi / 10
    assert (t < 0).any() and (t > 0).any()


def test_outlier_spikes_a_rare_subset_and_stays_finite():
    n = 100_000
    base = sample("outlier", (n,), 23, mag=1.0, spike=1.0)
    spiked = sample("outlier", (n,), 23, mag=1.0, spike=80.0)
    assert torch.isfinite(base).all() and torch.isfinite(spiked).all()
    assert float(base.abs().median()) < 2
    changed = spiked != base
    assert 20 < int(changed.sum()) < 500
    assert torch.equal(spiked, torch.where(changed, base * 80, base))


def test_cancel_pairs_opposite_large_values_on_an_even_last_axis():
    t = sample("cancel", (128, 64), 9, mag=1.0)
    assert t.dtype == torch.float32 and torch.isfinite(t).all()
    left, right = t[:, 0::2], t[:, 1::2]
    pair = left + right
    assert float(pair.abs().max()) < 30
    assert float(t.abs().median()) > 100
    assert float(pair.abs().max()) < float(left.abs().median()) * 0.25


def test_cancel_keeps_large_values_when_the_last_axis_is_odd():
    t = sample("cancel", (64, 31), 9, mag=1.0)
    assert torch.isfinite(t).all()
    assert float(t.abs().min()) > 200
    assert float(t.abs().median()) > 100
    assert (t < 0).any() and (t > 0).any()


def test_sparse_is_mostly_exact_zeros_and_scales_the_rest():
    n = 20_000
    t = sample("sparse", (n,), 4, mag=2.0)
    assert t.dtype == torch.float32 and torch.isfinite(t).all()
    zeros = int((t == 0).sum())
    assert 0.85 * n < zeros < 0.95 * n
    assert int(((t == 0) & torch.signbit(t)).sum()) == 0
    base = sample("sparse", (512,), 4, mag=1.0)
    assert torch.equal(sample("sparse", (512,), 4, mag=4.0), base * 4)


def test_generator_is_a_cpu_generator_and_keeps_63_bits_of_seed():
    g = generator(5)
    assert isinstance(g, torch.Generator) and g.device.type == "cpu"
    first = torch.rand(5, generator=generator(5))
    assert torch.equal(first, torch.rand(5, generator=generator(5)))
    assert not torch.equal(first, torch.rand(5, generator=generator(6)))
    assert torch.equal(first, torch.rand(5, generator=generator((1 << 63) + 5)))
    assert torch.equal(torch.rand(4, generator=generator(-1)), torch.rand(4, generator=generator((1 << 63) - 1)))


def test_ieee_is_not_a_distribution_case():
    with pytest.raises(ValueError, match="unknown case 'ieee'"):
        sample("ieee", (8,), 1)
