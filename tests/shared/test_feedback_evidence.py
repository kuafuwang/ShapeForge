"""A probe named by feedback can be restored from the scratch archive."""

from __future__ import annotations

import tarfile
from pathlib import Path

from experiments.restore_feedback_evidence import _restore


def test_a_named_probe_is_restored_from_scratch(tmp_path):
    """@brief 验证 FEEDBACK.md 中实测过的探针文件能从 scratch 归档里恢复回 agent 工作目录。

    @param tmp_path pytest 提供的临时目录，用来搭建 run 目录、agent_work 与 scratch 归档。
    """
    run = tmp_path / "run"
    work = run / "agent_work" / "r1"
    work.mkdir(parents=True)
    (work / "FEEDBACK.md").write_text("[已实测: probe.py | x | 1.0 us]\\n")
    scratch = run / "scratch"
    scratch.mkdir()
    with tarfile.open(scratch / "r1.tar.gz", "w:gz") as tar:
        tar.add(work / "FEEDBACK.md", arcname="probe.py")
    assert not (work / "probe.py").exists()
    row = _restore(run, work, "probe.py", True)
    assert row is not None and (work / "probe.py").is_file()
