"""Static pre-screen: unbounded loops and the include policy. No GPU, no model."""

from __future__ import annotations

import pytest

from tcenv.backends.metal.screen import screen_source


@pytest.mark.parametrize("cond", ["0 == 0", "1 >= 1", "!false", "2", "(true)", "1u", "3 % 2 == 1", "1 / 1", "2 ** 0", "2 ** 3 - 7"])
def test_constant_true_conditions_are_refused(cond):
    """A constant-folded true is the same hang as literal `true`; the old scan only saw true/1."""
    assert screen_source(f"kernel void k() {{ while ({cond}) {{}} }}")


@pytest.mark.parametrize("cond", ["0", "false", "i < n", "n > 0", "x != 0", "sizeof(int) > 0", "0 / 1"])
def test_constant_false_or_non_constant_conditions_are_not_refused(cond):
    # `do {} while (0)` is a common macro idiom and terminates; sizeof() is not constant-folded here.
    assert screen_source(f"kernel void k() {{ while ({cond}) {{}} }}") == []


def test_goto_and_include_inside_comments_or_strings_are_data():
    assert screen_source("kernel void k() { /* goto again; */ }") == []
    assert screen_source('kernel void k() { const char *s = "#include \\"/etc/passwd\\""; }') == []
    assert screen_source("kernel void k() { again: ; goto again; }")


def test_a_real_non_metal_include_is_still_refused():
    assert screen_source('#include "/etc/passwd"\nkernel void k() {}')
    assert screen_source("#include <metal_stdlib>\nkernel void k() {}") == []
