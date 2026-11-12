"""CPU replay of the rejected GEMM Split-K analysis."""

from tests.support.gemm_splitk import assert_negative


def test_splitk_is_rejected_by_measured_floor_and_traffic_not_by_preference():
    assert_negative()
