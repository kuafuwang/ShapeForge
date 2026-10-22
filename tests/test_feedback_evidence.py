"""A probe named by feedback can be restored from the scratch archive."""

from __future__ import annotations

import tarfile
from pathlib import Path

from experiments.restore_feedback_evidence import _restore


def test_a_named_probe_is_restored_from_scratch(tmp_path):
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
