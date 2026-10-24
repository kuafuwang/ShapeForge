"""Opt-in real Docker acceptance for the isolated patch runner.

Set TAIC_TEST_DOCKER_IMAGE to an image containing the repository's Python/test
requirements. Without that explicit image, this test skips and never starts
Docker. No GPU or model is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from taic import code_agent


@pytest.mark.slow
@pytest.mark.skipif(not shutil.which("docker"), reason="Docker is unavailable")
@pytest.mark.skipif(not os.environ.get("TAIC_TEST_DOCKER_IMAGE"), reason="set TAIC_TEST_DOCKER_IMAGE to run real isolated acceptance")
def test_real_docker_patch_runner_applies_patch_and_runs_registered_check(tmp_path):
    """@brief 显式提供镜像时，真实容器应能应用补丁并通过注册的离线检查。"""
    repo = Path(__file__).resolve().parents[2]
    worker = subprocess.run(["git", "cat-file", "-e", "HEAD:taic/isolated_patch_worker.py"], cwd=repo, capture_output=True)
    if worker.returncode != 0:
        pytest.skip("the checked-out base commit does not contain the isolated worker; commit it before real Docker acceptance")
    worktree = code_agent.prepare_worktree(repo, tmp_path, "docker-integration")
    try:
        target = worktree / "tcenv" / "report.py"
        target.write_text(target.read_text(encoding="utf-8") + "\n# isolated Docker integration marker\n", encoding="utf-8")
        patch = code_agent.collect_patch(worktree)
        patch_path = tmp_path / "candidate.patch"
        patch_path.write_text(patch, encoding="utf-8")
        base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
        runner = code_agent.DockerPatchTestRunner(os.environ["TAIC_TEST_DOCKER_IMAGE"], timeout_s=900)
        results = runner.run(repo, patch_path, "docker-integration", check_items=[{"check_id": "reporting.offline_contract"}], base_commit=base)
        assert results and all(item.get("passed") is True for item in results)
        assert runner.evidence()["harness_base_commit"] == base
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo, check=False, capture_output=True)
