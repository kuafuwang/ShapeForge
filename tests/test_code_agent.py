"""The code-agent P2 path is patch-only; no model or GPU is needed for these tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from taic import code_agent
from taic.kernel_agent import Prices


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("before\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)
    return repo


def test_a_code_patch_is_collected_and_applied_only_with_confirmation(tmp_path):
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    (worktree / "new.txt").write_text("new\n")
    patch = code_agent.collect_patch(worktree)
    assert "a.txt" in patch and "new.txt" in patch
    run = tmp_path / "run"
    meta = code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m", "target_id": "apple-m4-max"})
    checked = code_agent.apply_code_patch(run, proposal_key="key", root=repo, check_only=True)
    assert checked["check"] is True and (repo / "a.txt").read_text() == "before\n"
    applied = code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    assert applied["applied"] is True and applied["approved_by"] == "alice"
    assert (repo / "a.txt").read_text() == "after\n" and (repo / "new.txt").read_text() == "new\n"
    decision = json.loads((run / "harness_decisions.jsonl").read_text().splitlines()[-1])
    assert decision["event"] == "code_promotion" and decision["approved_by"] == "alice"
    assert decision["change_ref"] == "" and decision["status"] == "local" and decision["implemented_by"] == "p/m"


def test_code_agent_budget_scales_with_the_implementer_price(tmp_path):
    cheap = code_agent.estimate_code_budget(Prices(0.3, 1.2, 0.006, 0), "x" * 3000)
    expensive = code_agent.estimate_code_budget(Prices(3, 15, 0.3, 3), "x" * 3000)
    assert cheap["budget_usd"] < expensive["budget_usd"]
    assert expensive["budget_usd"] > 0.2
