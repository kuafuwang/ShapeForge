"""The skill documents are part of the model's prompt, so every complete kernel in them must really compile."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tcenv.backends.metal.runtime import MetalCompileError, get_context

SKILLS = Path(__file__).resolve().parents[3] / "backends" / "metal" / "skills"
HDR = "#include <metal_stdlib>\n#include <metal_simdgroup>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
BLOCK = re.compile(r"```metal\n(.*?)```", re.DOTALL)


def _blocks():
    """@brief 遍历所有技能文档里的 ```metal 完整代码块并生成 pytest 参数。

    @return pytest.param 迭代器，id 形如文件名#序号。
    """
    for f in sorted(SKILLS.glob("*.md")):
        for i, m in enumerate(BLOCK.finditer(f.read_text())):
            yield pytest.param(m.group(1), id=f"{f.stem}#{i}")


@pytest.mark.gpu
@pytest.mark.parametrize("code", list(_blocks()))
def test_every_metal_block_compiles(code):
    """@brief 技能文档里的每个完整 kernel 代码块都能真正编译。

    @param code 从文档里抽出的 ```metal 代码块。
    """
    names = re.findall(r"kernel\s+void\s+(\w+)", code)
    assert names, "a ```metal block must be a complete kernel (use ```metal-snippet for fragments)"
    try:
        get_context().compile(HDR + code, names[0])
    except MetalCompileError as e:
        pytest.fail(str(e)[:800])


def test_three_skill_documents_exist_and_are_sectioned():
    """@brief 三份技能文档都存在，且至少有 5 个章节和 5 个 Keywords 行。
    """
    for name in ("metal_profiling", "msl_reference", "apple_gpu_optimization"):
        text = (SKILLS / f"{name}.md").read_text()
        assert text.count("\n## ") >= 5, name
        assert text.count("Keywords:") >= 5, name
