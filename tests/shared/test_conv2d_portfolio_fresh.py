"""CPU replay of P2-K portfolio-level fresh evidence and the rejected micro-optimization."""

from tests.support.conv2d_portfolio_fresh import assert_fresh_suite, assert_packed_probe_stays_unadopted


def test_four_point_portfolio_fresh_suite_and_validated_manifest_replay():
    assert_fresh_suite()


def test_packed_half_probe_is_preserved_as_a_negative_result():
    assert_packed_probe_stays_unadopted()
