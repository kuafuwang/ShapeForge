"""Shared helpers for Triton spec tests."""

from __future__ import annotations

import ast


def jit_params(src: str, name: str) -> list[str]:
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args]
    raise AssertionError(f"{name} not found")
