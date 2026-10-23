"""The skill documents are part of the model's prompt, so every complete kernel in them must really compile."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tcenv.backends.metal.runtime import MetalCompileError, get_context

SKILLS = Path(__file__).resolve().parent.parent / "backends" / "metal" / "skills"
HDR = "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
BLOCK = re.compile(r"```metal\n(.*?)```", re.DOTALL)


def _blocks():
    for f in sorted(SKILLS.glob("*.md")):
        for i, m in enumerate(BLOCK.finditer(f.read_text())):
            yield pytest.param(m.group(1), id=f"{f.stem}#{i}")


@pytest.mark.gpu
@pytest.mark.parametrize("code", list(_blocks()))
def test_every_metal_block_compiles(code):
    names = re.findall(r"kernel\s+void\s+(\w+)", code)
    assert names, "a ```metal block must be a complete kernel (use ```metal-snippet for fragments)"
    try:
        get_context().compile(HDR + code, names[0])
    except MetalCompileError as e:
        pytest.fail(str(e)[:800])


def test_three_skill_documents_exist_and_are_sectioned():
    for name in ("metal_profiling", "msl_reference", "apple_gpu_optimization"):
        text = (SKILLS / f"{name}.md").read_text()
        assert text.count("\n## ") >= 5, name
        assert text.count("Keywords:") >= 5, name
