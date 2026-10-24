"""The code-agent P2 path is patch-only; no model or GPU is needed for these tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from taic import code_agent, harness
from taic.kernel_agent import Prices


def _repo(tmp_path: Path) -> Path:
    """@brief 在临时目录里建一个已初始化并提过基线提交的 git 仓库，供补丁流程测试使用。

    @param tmp_path pytest 提供的临时目录。
    @return 新建仓库路径（含一个内容为 ``before`` 的 ``a.txt``）。
    """
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
    """@brief 验证代码补丁先采集到 run 目录，只有带确认人时才会真的写回仓库并记决策事件。

    依次覆盖：采集的补丁含修改与新增文件；``check_only`` 只试应用而不改仓库；带 ``approved_by``
    时落地改动并追加一条 ``code_promotion`` 决策（change_ref 为空、status 为 local、implemented_by 为 provider/model）。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    """
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
    """@brief 验证代码 agent 的预算随实现者模型定价上升，且昂贵模型下预算仍有下限量级。

    @param tmp_path pytest 提供的临时目录（本用例未使用其内容，仅为夹具一致性）。
    """
    cheap = code_agent.estimate_code_budget(Prices(0.3, 1.2, 0.006, 0), "x" * 3000)
    expensive = code_agent.estimate_code_budget(Prices(3, 15, 0.3, 3), "x" * 3000)
    assert cheap["budget_usd"] < expensive["budget_usd"]
    assert expensive["budget_usd"] > 0.2


def test_a_patch_touching_a_forbidden_path_is_refused(tmp_path):
    """@brief 验证触及禁止路径的补丁被拒，包括路径穿越、八进制转义写成的大路径与 rename/copy。

    同时确认 ``patch_paths`` 能把 git 的八进制转义反解为 ``tasks/x.py``。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    @raises code_agent.CodeAgentError 补丁路径落在禁区（``may not change``）或含 rename/copy 时由 ``write_patch`` 抛出。
    """
    patch = """diff --git a/tasks/secret.md b/tasks/secret.md
new file mode 100644
--- /dev/null
+++ b/tasks/secret.md
@@ -0,0 +1 @@
+x
"""
    with pytest.raises(code_agent.CodeAgentError, match="may not change"):
        code_agent.write_patch(tmp_path / "run", "key", {"patch": patch})
    escaped = patch.replace("a/tasks/secret.md", "a/targets/../taic/trust.py").replace(
        "b/tasks/secret.md", "b/targets/../taic/trust.py"
    )
    with pytest.raises(code_agent.CodeAgentError, match="may not change"):
        code_agent.write_patch(tmp_path / "run", "escaped", {"patch": escaped})
    octal = (
        'diff --git "a/t\\141sks/x.py" "b/t\\141sks/x.py"\n'
        'index 3367afd..3e75765 100644\n'
        '--- "a/t\\141sks/x.py"\n+++ "b/t\\141sks/x.py"\n'
        '@@ -1 +1 @@\n-old\n+new\n'
    )
    assert code_agent.patch_paths(octal) == ["tasks/x.py"]
    with pytest.raises(code_agent.CodeAgentError, match="may not change"):
        code_agent.write_patch(tmp_path / "run", "octal", {"patch": octal})
    rename = "diff --git a/tasks/x.py b/safe.py\nsimilarity index 100%\nrename from tasks/x.py\nrename to safe.py\n"
    with pytest.raises(code_agent.CodeAgentError, match="rename/copy"):
        code_agent.write_patch(tmp_path / "run", "rename", {"patch": rename})


def test_code_tests_run_before_the_live_checkout_is_touched(tmp_path, monkeypatch):
    """@brief 验证跑测试时仓库尚未被改动：测试桩看到的是改动前的内容，测试结束后才真正写入。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @param monkeypatch pytest 的替换夹具，用于把 ``_test_patch_in_worktree`` 换成记录调用时刻的桩。
    """
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    seen = []

    def fake_test(test_repo, patch_path, proposal_key):
        seen.append((repo / "a.txt").read_text())

    monkeypatch.setattr(code_agent, "_test_patch_in_worktree", fake_test)
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, run_tests=True, approved_by="alice")
    assert seen == ["before\n"]
    assert (repo / "a.txt").read_text() == "after\n"


def test_a_code_promotion_is_bound_to_the_commit_and_cannot_be_committed_twice(tmp_path):
    """@brief 验证代码提升绑定了具体提交：一次提交后再次提交会因找不到待提交的本地提升而被拒。

    校验提交记录里的 paths 与 ``promotion_commit`` 事件一致。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @raises harness.ProposalError 已无待提交的本地提升时再次提交（``no locally applied promotion``）。
    """
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    (worktree / "new.txt").write_text("new\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    committed = harness.commit_local_promotions(run, message="promote code patch", by="alice", root=repo)
    assert committed["paths"] == ["a.txt", "new.txt"]
    event = json.loads((run / "harness_decisions.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "promotion_commit" and event["paths"] == ["a.txt", "new.txt"]
    with pytest.raises(harness.ProposalError, match="no locally applied promotion"):
        harness.commit_local_promotions(run, message="again", by="alice", root=repo)


def test_a_promotion_keeps_unrelated_staged_paths_out_of_its_commit(tmp_path):
    """@brief 验证提升提交只包含补丁自己的文件，已暂存的无关文件继续留在暂存区。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    """
    repo = _repo(tmp_path)
    (repo / "unrelated.txt").write_text("base\n")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "more"], cwd=repo, check=True)
    (repo / "unrelated.txt").write_text("staged\n")
    subprocess.run(["git", "add", "unrelated.txt"], cwd=repo, check=True)
    tree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (tree / "a.txt").write_text("after\n")
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": code_agent.collect_patch(tree)})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo)
    result = harness.commit_local_promotions(run, message="promote", root=repo)
    assert result["paths"] == ["a.txt"]
    changed = subprocess.run(["git", "show", "--format=", "--name-only", "HEAD"], cwd=repo, check=True, capture_output=True, text=True)
    assert changed.stdout.strip() == "a.txt"
    staged = subprocess.run(["git", "diff", "--cached", "--name-only"], cwd=repo, check=True, capture_output=True, text=True)
    assert staged.stdout.strip() == "unrelated.txt"


def test_a_code_promotion_refuses_a_changed_patch_file(tmp_path):
    """@brief 验证落地后补丁文件被改动（哈希变化）时拒绝提交。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @raises harness.ProposalError 补丁文件内容与记录的哈希不一致时抛出。
    """
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    meta = code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    patch_path = Path(meta["patch"])
    patch_path.write_text(patch_path.read_text().replace("index ", "index  ", 1))
    with pytest.raises(harness.ProposalError, match="patch hash does not match"):
        harness.commit_local_promotions(run, message="promote", by="alice", root=repo)


def test_a_code_promotion_refuses_to_commit_after_the_worktree_diverged(tmp_path):
    """@brief 验证已应用的补丁被手工改过后，提交阶段能发现工作树已偏离并拒绝提交。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @raises harness.ProposalError 仓库文件不再与应用时一致（``no longer match``）时抛出。
    """
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    (repo / "a.txt").write_text("manual edit\n")
    with pytest.raises(harness.ProposalError, match="no longer match"):
        harness.commit_local_promotions(run, message="promote", by="alice", root=repo)


def test_a_code_promotion_refuses_an_unrecorded_executable_bit(tmp_path):
    """@brief 验证未被记录的可执行位变更也算偏离，提交阶段会拒绝。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @raises harness.ProposalError 文件权限与应用时不一致（``no longer match``）时抛出。
    """
    repo = _repo(tmp_path)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "a.txt").write_text("after\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    (repo / "a.txt").chmod(0o755)
    with pytest.raises(harness.ProposalError, match="no longer match"):
        harness.commit_local_promotions(run, message="promote", by="alice", root=repo)


def test_a_code_promotion_refuses_unrelated_edits_in_the_same_file(tmp_path):
    """@brief 验证同一文件里出现补丁之外的额外改动时，即使反向应用仍能成功也拒绝提交。

    @param tmp_path pytest 提供的临时目录，用来放仓库、worktree 与 run 目录。
    @raises harness.ProposalError 文件内容不再是“基线 + 这份补丁”时抛出。
    """
    repo = _repo(tmp_path)
    (repo / "b.txt").write_text("one\ntwo\nthree\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "more"], cwd=repo, check=True)
    worktree = code_agent.prepare_worktree(repo, tmp_path, "key")
    (worktree / "b.txt").write_text("ONE\ntwo\nthree\n")
    patch = code_agent.collect_patch(worktree)
    run = tmp_path / "run"
    code_agent.write_patch(run, "key", {"patch": patch, "provider": "p", "model": "m"})
    code_agent.apply_code_patch(run, proposal_key="key", root=repo, approved_by="alice")
    # The reverse apply still succeeds, but the file now has text the patch did not add.
    (repo / "b.txt").write_text("ONE\ntwo\nthree\nextra\n")
    with pytest.raises(harness.ProposalError, match="no longer match"):
        harness.commit_local_promotions(run, message="promote", by="alice", root=repo)
