"""Cross-hardware boundary of the A1 production surface.

A1 only travels on the Mac/清微-neutral path.  This check is intentionally executable rather
than a doc sentence: the A1 modules, the Metal track registry and the task-contract pack set
must not mention a rejected hardware vendor's code/toolchain knowledge, must not import a
backend/host harness, and must not carry a knowledge/guidance body into ``task_pack_digest``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from tcenv.taskcontract import load_task_contract

from ..support.v2_tasks import write_task

ROOT = Path(__file__).resolve().parents[2]
A1_SOURCES = (
    ROOT / "tcenv" / "contract_errors.py",
    ROOT / "tcenv" / "task_presentation.py",
    ROOT / "tcenv" / "opspec" / "v2.py",
    ROOT / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json",
    ROOT / "tcenv" / "track" / "registry.py",
    ROOT / "tcenv" / "track" / "plane.py",
    ROOT / "tcenv" / "track" / "schema" / "tracks.v1.schema.json",
    ROOT / "backends" / "metal" / "tracks.json",
)
A1_DIRECTORY = ROOT / "tcenv" / "taskcontract"
#: Rejected platform names, matched as words so ``input`` cannot trip the check.
REJECTED_PLATFORM = re.compile(
    r"\b(cuda|cann|ascend[a-z]*|msopgen|msprof|easyasc|acl|npu|torch_npu|america|nvidia)\b", re.IGNORECASE
)


def _files() -> list[Path]:
    """@brief A1 生产文件（目录递归，排除缓存）。

    @return 文件列表。
    """
    files = list(A1_SOURCES) + [
        path for path in A1_DIRECTORY.rglob("*") if path.is_file() and "__pycache__" not in path.parts
    ]
    return sorted({path for path in files if path.is_file()})


def test_a1_production_mentions_no_rejected_hardware_platform():
    """@brief A1 生产代码/配置不携带 CUDA/CANN/Ascend 平台名称或专用知识。"""
    offenders: list[str] = []
    for path in _files():
        text = path.read_text(encoding="utf-8")
        for match in REJECTED_PLATFORM.finditer(text):
            offenders.append(f"{path.relative_to(ROOT)}: {match.group(0)}")
    assert offenders == [], "\n".join(offenders)


def test_a1_modules_do_not_import_a_backend_or_the_host_harness():
    """@brief A1 合同/track 平面只依赖通用层，不 import backend/taic/候选。"""
    import ast

    forbidden_roots = {"backends", "taic", "tasks", "tests", "torch_npu", "acl"}
    offenders: list[str] = []
    for path in _files():
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                if root in forbidden_roots:
                    offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}: {name}")
    assert offenders == [], "\n".join(offenders)


def test_the_pack_set_is_the_standard_entry_only_and_excludes_guidance(tmp_path):
    """@brief pack 只含标准合同文件；知识文件不进入 task_pack_digest。"""
    task_dir = write_task(tmp_path)
    (task_dir / "guidance_catalog.md").write_text("a wiki page\n", encoding="utf-8")
    (task_dir / "agent_guidance.md").write_text("a hint\n", encoding="utf-8")
    (task_dir / "skills").mkdir()
    (task_dir / "skills" / "author.md").write_text("secret\n", encoding="utf-8")
    contract = load_task_contract(task_dir)
    assert contract.pack_files == ("opspec.json", "reference.py")
    assert contract.pack_files == tuple(name for name in contract.pack_files if (task_dir / name).is_file())
    for forbidden in ("guidance_catalog.md", "agent_guidance.md", "skills/author.md"):
        assert forbidden not in contract.pack_files
        assert not any(forbidden in name for name in contract.pack_files)


def test_the_contract_and_track_documents_have_no_guidance_field(tmp_path):
    """@brief 知识只允许进 guidance_catalog/agent_guidance，不是合同/track 字段。"""
    from tcenv.track import load_track_registry

    track_document = json.dumps(load_track_registry().document())
    assert "guidance" not in track_document.lower()
    document = json.dumps(load_task_contract(write_task(tmp_path)).fingerprint_document()).lower()
    assert "guidance" not in document
