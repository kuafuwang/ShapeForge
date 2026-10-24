"""Static pre-screen: unbounded loops and the include policy. No GPU, no model."""

from __future__ import annotations

import pytest

from tcenv.backends.metal.screen import screen_source


@pytest.mark.parametrize("cond", ["0 == 0", "1 >= 1", "!false", "2", "(true)", "1u", "3 % 2 == 1", "1 / 1", "2 ** 0", "2 ** 3 - 7"])
def test_constant_true_conditions_are_refused(cond):
    """@brief 验证常量折叠为真的循环条件与字面量 true 一样被拒绝。

    旧实现只识别 true/1，这里逐个断言各种等价真值表达式也会被拦下。

    @param cond 被包进 while 的条件表达式，由 parametrize 提供。
    """
    assert screen_source(f"kernel void k() {{ while ({cond}) {{}} }}")


@pytest.mark.parametrize("cond", ["0", "false", "i < n", "n > 0", "x != 0", "sizeof(int) > 0", "0 / 1"])
def test_constant_false_or_non_constant_conditions_are_not_refused(cond):
    """@brief 验证恒假或非常量的循环条件不会被误判为空循环拒绝。

    `do {} while (0)` 是常见宏写法且会终止；sizeof() 在这里不做常量折叠。

    @param cond 被包进 while 的条件表达式，由 parametrize 提供。
    """
    assert screen_source(f"kernel void k() {{ while ({cond}) {{}} }}") == []


def test_goto_and_include_inside_comments_or_strings_are_data():
    """@brief 验证注释与字符串里的 goto/include 被当作数据，不触发预筛；真实 goto 仍被拒绝。"""
    assert screen_source("kernel void k() { /* goto again; */ }") == []
    assert screen_source('kernel void k() { const char *s = "#include \\"/etc/passwd\\""; }') == []
    assert screen_source("kernel void k() { again: ; goto again; }")


def test_a_real_non_metal_include_is_still_refused():
    """@brief 验证真实的非 Metal 头文件包含仍被拒绝，而 metal_stdlib 被放行。"""
    assert screen_source('#include "/etc/passwd"\nkernel void k() {}')
    assert screen_source("#include <metal_stdlib>\nkernel void k() {}") == []
