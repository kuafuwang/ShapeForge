"""Shared helpers for Triton spec tests."""

from __future__ import annotations

import ast


def jit_params(src: str, name: str) -> list[str]:
    """@brief 用 AST 取出 Triton 源码里某个 jit 函数的形参名。

    @param src Triton 源码文本。
    @param name 目标函数名。
    @return 形参名列表，顺序与源码一致。
    @raises AssertionError 源码里找不到该函数时抛出。
    """
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"{name} not found")
