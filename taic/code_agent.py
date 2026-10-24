"""Prepare an isolated worktree, run one code agent, and collect a human-reviewable patch.

This is the P2 implementation path. The code agent is the default implementer, but it has
no merge/approval authority: the worktree is detached, the patch is inspected by a person,
and only ``apply_code_patch(..., approved_by=...)`` writes into the main repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .kernel_agent import Prices, dollars_for, spend_from_stream
from .llm import LLMError

FLOW = Path(__file__).resolve().parent / "flows" / "code_patch"
PATCH_DIR = "code_promotions"
REVIEW_LOG = "harness_reviews.jsonl"
MARGIN = 4.0
MIN_BUDGET_USD = 0.05
CACHE_READ_ALLOWANCE = 200_000
CACHE_WRITE_ALLOWANCE = 20_000

# Paths the P2 code agent may never touch. These are not normal feature areas:
# changing them can make a later check pass without making the kernel or harness
# better. They get a separate human security review, not an auto-applied patch.
SECURITY_CRITICAL = (
    "tests/",
    "tcenv/evaluate.py",
    "tcenv/schema.py",
    "tcenv/bench.py",
    "tcenv/backends/",
    "tcenv/gpulock.py",
    "taic/prompts.py",
    "taic/kernel_agent.py",
    "taic/harness.py",
    "taic/review.py",
    "taic/trust.py",
    "taic/code_agent.py",
    "taic/acceptance.py",
    "taic/acceptance_checks.py",
    "taic/orchestrator.py",
    "taic/policy.py",
    "taic/control.py",
    "taic/llm.py",
    "taic/artifacts.py",
    "taic/harness_experiment.py",
    "taic/resume.py",
    "taic/committees.py",
    "taic/cli.py",
    "taic/flows/",
    "experiments/common.py",
    "observe/",
    "targets/",
    "backends/",
    ".pi/",
    ".git/",
    ".github/",
    "control.json",
    "taic.toml",
    "taic.paper.toml",
    "pyproject.toml",
    "uv.lock",
)


class CodeAgentError(LLMError):
    """@brief 代码 agent 或工作树没能交出一个可用的补丁。"""


def _normalize_patch_path(value: str) -> str:
    """@brief 把补丁里的路径归一成仓库相对形式。

    反斜杠当正斜杠，去掉开头的 ``./`` 与多余的 ``..``/``.`` 片段。

    @param value 补丁里的原始路径。
    @return 归一后的路径；空输入返回空字符串。
    """
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    text = posixpath.normpath(raw)
    while text.startswith("./"):
        text = text[2:]
    return text


def patch_paths(patch: str) -> list[str]:
    """@brief 让 Git 自己解码补丁里的路径，包括它的八进制转义引用。

    改名/复制的补丁只报告目的路径，所以这两类特性在源路径与目的路径都被显式检查之前一律拒绝。

    @param patch 补丁文本。
    @return 去重后的路径列表，顺序与补丁一致。
    @raises CodeAgentError 补丁无法解析、路径认不全，或含未审查的改名/复制路径时。
    """
    proc = subprocess.run(
        ["git", "apply", "--numstat", "-z"], input=str(patch or "").encode("utf-8"), capture_output=True,
    )
    if proc.returncode != 0:
        raise CodeAgentError((proc.stderr or b"cannot parse patch").decode("utf-8", errors="replace").strip())
    paths: list[str] = []
    for entry in proc.stdout.split(b"\x00"):
        if not entry:
            continue
        parts = entry.split(b"\t", 2)
        if len(parts) != 3:
            raise CodeAgentError("cannot identify all patch paths")
        raw = parts[2]
        # A rename may use Git's NUL-separated old/new path representation.
        if not raw:
            raise CodeAgentError("unsupported rename path in patch")
        value = _normalize_patch_path(raw.decode("utf-8", errors="surrogateescape"))
        if value and value not in paths:
            paths.append(value)
    # --numstat reports only the destination of a rename/copy. Reject those patch
    # features until both source and destination are explicitly checked.
    if re.search(r"(?m)^(?:rename|copy) (?:from|to) ", str(patch or "")):
        raise CodeAgentError("rename/copy patches need explicit path review")
    return paths


def forbidden_path(path: str) -> bool:
    """@brief 判断一条路径是否属于 P2 代码路径永远不得修改的范围。

    包括仓库外路径、``tasks/``、``tcenv/gpulock.py``、``taic/trust.py``、``control.json``、
    目标的 ``trust.toml``，以及各类凭据文件。

    @param path 待检查的路径。
    @return 被禁止时为 ``True``。
    """
    value = _normalize_patch_path(path)
    if not value:
        return False
    if value == ".." or value.startswith("../"):
        return True
    value = value.lstrip("/")
    if value == "tasks" or value.startswith("tasks/"):
        return True
    if any(value == prefix.rstrip("/") or value.startswith(prefix) for prefix in SECURITY_CRITICAL):
        return True
    if value.startswith("targets/") and value.endswith("/trust.toml"):
        return True
    name = value.rsplit("/", 1)[-1].lower()
    if name in {".env", "credentials", "credentials.json"} or name.startswith((".env.", "credentials.", "id_rsa", "id_ed25519")):
        return True
    if "credential" in name or name.endswith(".pem") or name.endswith(".key"):
        return True
    return False


def forbidden_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if forbidden_path(path)]


def _validate_patch(patch: str) -> list[str]:
    """@brief 校验补丁里的路径都能被这条代码路径接受。

    @param patch 补丁文本。
    @return 补丁涉及的路径列表。
    @raises CodeAgentError 补丁没点名任何文件，或触及被禁止的路径时。
    """
    paths = patch_paths(patch)
    if not paths:
        raise CodeAgentError("the patch does not name any files")
    blocked = forbidden_paths(paths)
    if blocked:
        raise CodeAgentError("patch touches a path the P2 code path may not change: " + ", ".join(sorted(blocked)))
    return paths


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


def prepare_worktree(repo: Path, run_dir: Path, proposal_key: str, *, base_commit: str | None = None) -> Path:
    """@brief 在运行目录下从指定基线建一个游离工作树。

    同名目录已存在时先删掉再建，使重试不会叠加旧改动。

    @param repo 主仓库目录。
    @param run_dir 运行目录，工作树放在它的 ``code_worktrees/`` 下。
    @param proposal_key 提案键，同时用作工作树目录名。
    @return 新建工作树的路径。
    """
    path = Path(run_dir) / "code_worktrees" / proposal_key
    if path.exists():
        remove_worktree(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    base = base_commit or _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "worktree", "add", "--detach", str(path), base)
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    """@brief 强制删掉一个工作树，并兜底清掉残留目录。

    @param repo 主仓库目录。
    @param path 要删除的工作树路径。
    """
    _git(repo, "worktree", "remove", "--force", str(path), check=False)
    shutil.rmtree(path, ignore_errors=True)


def estimate_code_budget(prices: Prices, prompt: str, *, output_tokens: int = 8000) -> dict:
    """@brief 估算一次代码 agent 回合的预算，含上下文/缓存流量。

    预算由价格和提示长度推出，不是固定金额。

    @param prices 输入/输出/缓存读写单价。
    @param prompt 将要发出去的提示文本。
    @param output_tokens 期望的输出 token 数（会被夹到 512..32000）。
    @return 含 ``input_tokens``/``output_tokens``/``expected_usd``/``budget_usd`` 的字典。
    """
    text = str(prompt or "")
    input_tokens = max(2000, int(math.ceil(len(text.encode("utf-8")) / 3.0)))
    output_tokens = max(512, min(int(output_tokens), 32000))
    expected = (
        input_tokens / 1_000_000 * float(prices.input)
        + output_tokens / 1_000_000 * float(prices.output)
        + CACHE_READ_ALLOWANCE / 1_000_000 * float(prices.cache_read)
        + CACHE_WRITE_ALLOWANCE / 1_000_000 * float(prices.cache_write)
    )
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "expected_usd": round(expected, 6),
        "budget_usd": round(max(MIN_BUDGET_USD, expected * MARGIN), 6),
    }


def build_prompt(proposal: dict, *, run_id: str, acceptance_spec: dict | None = None) -> str:
    """@brief 拼出一次实现回合的 IMPLEMENT.md 提示。

    提示把提案当数据嵌入，并要求 agent 只在游离工作树里改文件、写完 IMPLEMENT_REPORT.json。

    @param proposal 待实现的提案字典。
    @param run_id 提案所属运行标识。
    @return 提示文本。
    """
    return f"""# Implement one approved harness proposal

You are implementing a proposal that has already been reviewed. Work only in this
isolated worktree. Do not work in the live run directory, do not change trust/budget/
control files, and do not commit or push.

--- BEGIN PROPOSAL ---
run: {run_id}
surface: {proposal.get('surface', '')}
scope: {json.dumps(proposal.get('scope') or {}, ensure_ascii=False)}
action: {proposal.get('action', '')}
evidence: {json.dumps(proposal.get('evidence') or {}, ensure_ascii=False)}
rationale: {proposal.get('rationale', '')}
--- END PROPOSAL ---

--- BEGIN ACCEPTANCE CONTRACT (untrusted specification data) ---
{json.dumps(acceptance_spec or {}, ensure_ascii=False, indent=2)}
--- END ACCEPTANCE CONTRACT ---

Implement the smallest coherent change, add or update tests, and run the relevant offline
tests. The acceptance contract carries the independent review's recommended test. Respect
its allowed_paths when non-empty, including tests. Never change evaluator semantics, correctness, tolerance,
budget, trust or approval code to make a check pass. Do not modify `tasks/`,
`tcenv/gpulock.py`, `taic/trust.py`, `targets/*/trust.toml`, `control.json`, or credentials.
Do not commit. At the end write `IMPLEMENT_REPORT.json`:

{{
  "summary": "what changed",
  "tests": ["commands and results"],
  "acceptance": [{{"check": "exact acceptance_checks entry", "passed": true, "evidence": "test or host-checkable output"}}],
  "unmet": ["acceptance items not demonstrated"],
  "risks": "remaining risks",
  "files": ["paths changed"]
}}

These are implementation claims for a person to verify, not proof of search improvement.
The outer harness will collect `git diff` and a person will review it. If you cannot
implement it safely, stop and write the report explaining why; do not make unrelated edits.
"""


def run_code_agent(
    worktree: Path,
    *,
    prompt: str,
    provider: str,
    model: str,
    effort: str | None,
    prices: Prices,
    budget_usd: float,
    hmz: str | None = None,
    timeout_s: float = 3600.0,
    turn_minutes: int = 40,
) -> dict:
    """@brief 跑一次 hmz 代码 agent 回合并收集它的 diff；自己从不 commit 也不 apply。

    超时时整组子进程被杀，已经花掉的费用挂在抛出的错误上，供调用方计账。

    @param worktree 游离工作树目录，agent 只能在这里动手。
    @param prompt 实现提示（会写成 IMPLEMENT.md）。
    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度；空值表示不指定。
    @param prices 计费单价。
    @param budget_usd 这一回合的费用上限。
    @param hmz hmz 可执行文件路径；``None`` 时从 PATH 找。
    @param timeout_s 子进程超时秒数。
    @param turn_minutes 回合时长上限（分钟）。
    @return 含 ``provider``/``model``/``cost_usd``/``tokens``/``patch``/``report`` 等字段的字典。
    @raises CodeAgentError 找不到 hmz、启动失败、超时，或 agent 没留下 diff 时。
    """
    binary = hmz or shutil.which("hmz")
    if not binary or not Path(binary).is_file():
        raise CodeAgentError("hmz is not on PATH; install it with `uv tool install hmz`")
    worktree = Path(worktree)
    worktree.mkdir(parents=True, exist_ok=True)
    (worktree / "IMPLEMENT.md").write_text(prompt, encoding="utf-8")
    plan = estimate_code_budget(prices, prompt)
    turn_tokens = plan["output_tokens"]
    argv = [
        binary, "exec", "-f", str(FLOW),
        "-a", f"assistant=pi/{provider}/{model}" + (f":{effort}" if effort else ""),
        "-p", f"turn_output_tokens={turn_tokens}",
        "-p", f"turn_minutes={int(turn_minutes)}",
        "-p", f"code_cost={budget_usd}",
        "-p", f"budget.cost={budget_usd}",
        "-p", f"budget.duration={int(turn_minutes) + 5}m",
        "--json", "--", "Read IMPLEMENT.md and implement it in this worktree. Do not commit.",
    ]
    started = time.time()
    from tcenv.children import forget, kill_group, popen_group

    try:
        # A group of its own: a timeout must reach the tools pi started, and the child must not
        # inherit API keys. The cost of the turn is kept on the error so the caller can bill it.
        proc = popen_group(argv, cwd=worktree, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        raise CodeAgentError(f"could not start hmz: {exc}") from exc
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                kill_group(proc)
                stdout, stderr = "", ""
            stream = (stdout or "") + "\n" + (stderr or "")
            spent, saw = spend_from_stream(stream)
            err = CodeAgentError(f"code agent timed out after {timeout_s:.0f}s")
            err.cost_usd = dollars_for(spent, prices) if saw else None
            err.tokens = spent
            raise err from None
    finally:
        kill_group(proc)
        forget(proc)
    stream = (stdout or "") + "\n" + (stderr or "")
    spent, saw = spend_from_stream(stream)
    cost = dollars_for(spent, prices) if saw else None
    report = {}
    try:
        report = json.loads((worktree / "IMPLEMENT_REPORT.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    for name in ("IMPLEMENT.md", "IMPLEMENT_REPORT.json"):
        (worktree / name).unlink(missing_ok=True)
    patch = collect_patch(worktree)
    if not patch.strip():
        tail = stream[-600:].strip()
        err = CodeAgentError(f"code agent left no diff; hmz tail: {tail}")
        err.cost_usd = cost
        err.tokens = spent
        raise err
    return {
        "provider": provider, "model": model, "effort": effort or "",
        "budget_usd": budget_usd, "cost_usd": cost, "elapsed_s": round(time.time() - started, 3),
        "tokens": spent, "patch": patch, "report": report,
        "worktree": str(worktree), "base_commit": _git(worktree, "rev-parse", "HEAD").stdout.strip(),
    }


def collect_patch(worktree: Path) -> str:
    """@brief 收集一份二进制安全的补丁，含新文件，且不在索引里留下暂存。

    先 ``git add -A`` 再取 ``git diff --cached --binary``，最后用 ``git reset`` 把索引恢复。

    @param worktree 工作树目录。
    @return 补丁文本。
    """
    _git(worktree, "add", "-A")
    diff = _git(worktree, "diff", "--cached", "--binary", "HEAD")
    _git(worktree, "reset", "--quiet")
    return diff.stdout


def write_patch(run_dir: Path, proposal_key: str, result: dict, *, target_id: str = "", acceptance_spec: dict | None = None) -> dict:
    """@brief 把生成的补丁与其元数据写到运行目录的 ``code_promotions/`` 下。

    写之前先做路径校验，使被禁止的补丁不会被记录下来。

    @param run_dir 运行目录。
    @param proposal_key 提案键，同时用作文件名。
    @param result ``run_code_agent`` 的返回值。
    @param target_id 目标标识。
    @return 写下的元数据（含补丁 sha256 与路径列表）。
    @raises CodeAgentError 补丁没有点名文件，或触及被禁止的路径时。
    """
    paths = _validate_patch(str(result.get("patch") or ""))
    folder = Path(run_dir) / PATCH_DIR
    folder.mkdir(parents=True, exist_ok=True)
    patch = folder / f"{proposal_key}.patch"
    patch.write_text(str(result.get("patch") or ""), encoding="utf-8")
    meta = {
        "schema_version": 1, "proposal_key": proposal_key,
        "target_id": target_id,
        "provider": result.get("provider"), "model": result.get("model"),
        "effort": result.get("effort"), "budget_usd": result.get("budget_usd"),
        "cost_usd": result.get("cost_usd"), "elapsed_s": result.get("elapsed_s"),
        "patch": str(patch), "sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "paths": paths,
        "base_commit": result.get("base_commit") or _git(Path(result.get("worktree") or Path(__file__).resolve().parent.parent), "rev-parse", "HEAD").stdout.strip(),
        "report": result.get("report") or {},
        "acceptance_spec": acceptance_spec or {},
    }
    if acceptance_spec:
        from .acceptance import implementation_report
        meta["implementation"] = implementation_report(result.get("report") or {}, acceptance_spec, paths, meta["sha256"])
        (folder / f"{proposal_key}.implementation.json").write_text(json.dumps(meta["implementation"], ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / f"{proposal_key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _proposal_row(run_dir: Path, proposal_key: str) -> dict:
    """@brief 取某个提案键对应的最后一条有效提案行。

    @param run_dir 运行目录。
    @param proposal_key 提案键。
    @return 提案行；找不到时为 ``{}``。
    """
    path = Path(run_dir) / "harness_proposals.jsonl"
    if not path.is_file():
        return {}
    found = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and row.get("status") == "valid" and str(row.get("proposal_key") or "") == proposal_key:
            found = row
    return found


def worktree_matches_patches(repo: Path, patches: list[Path], paths: list[str], *, staged_env: dict | None = None) -> tuple[bool, str]:
    """@brief 判断活工作树是否恰好等于 HEAD 加上全部补丁（按顺序）。

    反向 apply 只能证明一个补丁能被撤销，抓不到同一文件里别处的无关改动；所以要用一份全新的工作树
    比最终文件，而不是只看补丁能不能撤。

    @param repo 主仓库目录。
    @param patches 要依次应用的补丁路径。
    @param paths 需要比较的路径。
    @param staged_env 给出时改用 ``git ls-files --stage`` 比较暂存内容与文件模式。
    @return ``(是否一致, 原因或差异路径)``；一致时原因为空字符串。
    """
    parent = Path(tempfile.mkdtemp(prefix="taic-exact-"))
    temp = parent / "tree"
    added = False
    try:
        try:
            _git(repo, "worktree", "add", "--detach", str(temp), "HEAD")
        except subprocess.CalledProcessError as exc:
            return False, (exc.stderr or exc.stdout or "cannot create a test worktree").strip()
        added = True
        for patch in patches:
            applied = subprocess.run(["git", "apply", str(patch)], cwd=temp, capture_output=True, text=True)
            if applied.returncode != 0:
                return False, (applied.stderr or applied.stdout or f"patch does not apply to HEAD: {patch}").strip()
        if staged_env is not None:
            staged = _git(temp, "add", "--", *paths, check=False)
            if staged.returncode != 0:
                return False, (staged.stderr or staged.stdout).strip()
            expected_state = _git(temp, "ls-files", "--stage", "-z", "--", *paths).stdout
            actual_state = subprocess.run(
                ["git", "ls-files", "--stage", "-z", "--", *paths], cwd=repo,
                env=staged_env, capture_output=True, text=True, check=True,
            ).stdout
            return (True, "") if expected_state == actual_state else (False, "staged contents or file modes")
        for rel in paths:
            expected = temp / rel
            actual = Path(repo) / rel
            if expected.is_symlink() != actual.is_symlink():
                return False, rel
            if expected.is_symlink():
                if os.readlink(expected) != os.readlink(actual):
                    return False, rel
                continue
            if expected.is_file() != actual.is_file():
                return False, rel
            if expected.is_file():
                if expected.read_bytes() != actual.read_bytes():
                    return False, rel
                if bool(expected.stat().st_mode & 0o111) != bool(actual.stat().st_mode & 0o111):
                    return False, rel
        return True, ""
    finally:
        if added:
            _git(repo, "worktree", "remove", "--force", str(temp), check=False)
        shutil.rmtree(parent, ignore_errors=True)


def _test_patch_in_worktree(repo: Path, patch: Path, proposal_key: str, *, check_items: list[dict] | None = None, base_commit: str | None = None) -> list[dict]:
    """@brief 在游离工作树里应用并测试生成的补丁，绝不在活工作树里做。

    先确认 `import taic` 真的来自打好补丁的工作树，再跑离线测试。

    @param repo 主仓库目录。
    @param patch 补丁路径。
    @param proposal_key 提案键，用于临时目录名。
    @raises CodeAgentError 建不出工作树、打不上补丁、导入的不是打好补丁的树，或离线测试未过时。
    """
    parent = Path(tempfile.mkdtemp(prefix=f"taic-{proposal_key}-test-"))
    temp = parent / "tree"
    added = False
    try:
        try:
            _git(repo, "worktree", "add", "--detach", str(temp), base_commit or _git(repo, "rev-parse", "HEAD").stdout.strip())
        except subprocess.CalledProcessError as exc:
            raise CodeAgentError((exc.stderr or exc.stdout or "cannot create a test worktree").strip()) from exc
        added = True
        applied = subprocess.run(["git", "apply", str(patch)], cwd=temp, capture_output=True, text=True)
        if applied.returncode != 0:
            raise CodeAgentError((applied.stderr or applied.stdout or "git apply failed in the test worktree").strip())
        local = subprocess.run(
            ["uv", "run", "--project", str(repo), "python", "-c",
             "import pathlib, taic; assert pathlib.Path(taic.__file__).resolve().is_relative_to(pathlib.Path.cwd().resolve())"],
            cwd=temp, env={**os.environ, "PYTHONPATH": str(temp)}, capture_output=True, text=True,
        )
        if local.returncode != 0:
            raise CodeAgentError("test environment did not import the patched worktree: " + (local.stderr or local.stdout)[-500:])
        tests = subprocess.run(
            ["uv", "run", "--project", str(repo), "python", "-m", "pytest", "-q", "-m", "not gpu and not llm"],
            cwd=temp, env={**os.environ, "PYTHONPATH": str(temp)}, capture_output=True, text=True,
        )
        if tests.returncode != 0:
            raise CodeAgentError((tests.stdout + tests.stderr)[-2000:].strip() or "offline tests failed in the test worktree")
        if check_items is not None:
            from .acceptance_checks import run_checks
            return run_checks(check_items, worktree=temp, project=repo)
        return []
    finally:
        if added:
            _git(repo, "worktree", "remove", "--force", str(temp), check=False)
        shutil.rmtree(parent, ignore_errors=True)


def verify_code_patch(run_dir: Path, proposal_key: str, *, root: Path | None = None, approved_by: str = "", legacy_text: str = "", check_ids: list[str] | None = None) -> dict:
    """Run implementation acceptance in isolation, before an effect experiment or local apply."""
    run_dir = Path(run_dir)
    repo = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    folder = run_dir / PATCH_DIR
    patch = folder / f"{proposal_key}.patch"
    meta_path = folder / f"{proposal_key}.json"
    if not patch.is_file() or not meta_path.is_file():
        raise CodeAgentError("no generated patch and acceptance metadata")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("acceptance_spec") and not legacy_text:
        raise CodeAgentError("an old patch needs a new reviewed implementation or an explicit legacy acceptance upgrade")
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if meta.get("sha256") != digest:
        raise CodeAgentError("the patch file changed after it was recorded")
    from .acceptance import AcceptanceError, check_paths, read_spec
    from .harness_experiment import _review_for_spec
    if legacy_text:
        if not approved_by.strip() or not check_ids:
            raise CodeAgentError("legacy acceptance needs --by and at least one registered --check-id")
        from .acceptance import make_spec
        from .acceptance_checks import validate_items
        proposal = _proposal_row(run_dir, proposal_key)
        if not proposal:
            raise CodeAgentError("legacy acceptance needs its source proposal")
        checks = validate_items([{"check_id": check_id} for check_id in check_ids])
        evidence = dict((proposal.get("proposal") or {}).get("evidence") or {})
        # Old proposals do not get to choose the low-risk lane merely because a human
        # supplied legacy text; the normal risk classifier still decides.
        evidence.update({"host_checks": checks, "acceptance_checks": [legacy_text]})
        spec = make_spec({**proposal, "proposal": {**(proposal.get("proposal") or {}), "evidence": evidence}})
    else:
        spec = read_spec(meta["acceptance_spec"])
    if not spec["allowed_paths"]:
        raise CodeAgentError("acceptance contract needs non-empty allowed_paths")
    paths = _validate_patch(patch.read_text(encoding="utf-8", errors="replace"))
    _require_allowed_paths(run_dir, proposal_key, meta, paths)
    outside = check_paths(paths, spec["allowed_paths"])
    if outside:
        raise CodeAgentError("patch exceeds the acceptance contract: " + ", ".join(outside))
    from .acceptance import classify_change
    from .acceptance_checks import applicable_paths
    risk = classify_change(spec, patch.read_text(encoding="utf-8", errors="replace"))
    items = spec.get("host_checks")
    if not items:
        raise CodeAgentError("acceptance requires registered host check_ids; use verify-legacy to map old text explicitly")
    if check_ids is not None and sorted(check_ids) != sorted({item["check_id"] for item in items}):
        raise CodeAgentError("named check_ids differ from the acceptance contract")
    results = _test_patch_in_worktree(repo, patch, proposal_key, check_items=items, base_commit=meta.get("base_commit"))
    if not all(result["passed"] for result in results):
        raise CodeAgentError("one or more host acceptance checks failed")
    if risk["risk_tier"] == "offline_only" and not set(paths) <= set(applicable_paths([item["check_id"] for item in items])):
        raise CodeAgentError("the patch is not covered by its offline acceptance checks")
    record = {
        "schema_version": 1, "event": "implementation", "proposal_key": proposal_key,
        "spec_sha256": spec["spec_sha256"], "patch_sha256": digest, "risk_tier": risk["risk_tier"], "risk_reason": risk["reason"],
        "accepted": True, "host_verified": True, "host_checks": results,
        "tests_run": "registered checks and offline pytest in an isolated worktree", "t": time.time(),
    }
    if legacy_text:
        record.update({"legacy_upgrade": {"approved_by": approved_by.strip(), "original_text": legacy_text, "check_ids": sorted(check_ids or [])}, "accepted_by_human": approved_by.strip()})
    meta["acceptance_spec"] = spec
    meta["implementation"] = record
    (folder / f"{proposal_key}.implementation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _require_allowed_paths(run_dir: Path, proposal_key: str, meta: dict, paths: list[str]) -> list[str]:
    """@brief 要求补丁有非空、不可越界的 allowed_paths。

    新的 P2 提案必须把验收契约里的 ``allowed_paths`` 列出来；旧补丁没有契约时，
    从它对应的提案行读取。空列表不再表示“不限制”。

    @param run_dir 运行目录。
    @param proposal_key 提案键。
    @param meta 补丁元数据，可能含 ``acceptance_spec``。
    @param paths 补丁实际触及的路径。
    @return 归一化后的允许路径列表。
    @raises CodeAgentError 没有非空允许列表、允许路径本身越界，或补丁超出允许列表时。
    """
    spec = meta.get("acceptance_spec") if isinstance(meta.get("acceptance_spec"), dict) else None
    allowed = spec.get("allowed_paths") if spec is not None else None
    if allowed is None:
        proposal = _proposal_row(run_dir, proposal_key)
        evidence = (proposal.get("proposal") or {}).get("evidence") if isinstance(proposal.get("proposal"), dict) else None
        allowed = evidence.get("allowed_paths") if isinstance(evidence, dict) else None
    if not isinstance(allowed, list) or not allowed:
        raise CodeAgentError("the patch has no non-empty allowed_paths; refusing to apply")
    normalized = []
    for raw in allowed:
        path = _normalize_patch_path(str(raw))
        if not path or path == ".." or path.startswith("../") or str(raw).startswith("/"):
            raise CodeAgentError("allowed_paths must stay inside the repository")
        if forbidden_path(path):
            raise CodeAgentError(f"allowed_paths names a forbidden path: {path}")
        normalized.append(path)
    outside = [
        path for path in paths
        if not any(path == root or path.startswith(root.rstrip("/") + "/") for root in normalized)
    ]
    if outside:
        raise CodeAgentError("patch exceeds allowed_paths: " + ", ".join(sorted(outside)))
    return normalized


def apply_code_patch(
    run_dir: Path,
    *,
    proposal_key: str,
    root: Path | None = None,
    check_only: bool = False,
    run_tests: bool = False,
    approved_by: str = "cli",
    require_effect_validation: bool = False,
) -> dict:
    """@brief 检查通过后把生成的代码补丁应用到主仓库，从不 commit。

    测试在隔离工作树里对打好补丁的树跑。只有这一步通过后才会碰活工作树，而且应用后的字节必须仍与补丁一致。

    @param run_dir 运行目录，补丁与元数据放在它的 ``code_promotions/`` 下。
    @param proposal_key 提案键。
    @param root 主仓库目录；``None`` 时用本文件所在仓库。
    @param check_only 只做检查，不写入。
    @param run_tests 应用前先在隔离工作树里跑离线测试。
    @param approved_by 批准人，写入决策记录。
    @return 元数据加应用结果（含 ``applied``/``commit_pending`` 等）。
    @raises CodeAgentError 没有补丁、补丁被改过、路径被禁、无法 apply、目标文件有未提交改动，或测试未过时。
    """
    run_dir = Path(run_dir)
    repo = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    folder = run_dir / PATCH_DIR
    patch = folder / f"{proposal_key}.patch"
    meta_path = folder / f"{proposal_key}.json"
    if not patch.is_file():
        raise CodeAgentError("no generated patch; run `taic harness implement` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    digest = hashlib.sha256(patch.read_bytes()).hexdigest()
    if meta.get("sha256") and str(meta["sha256"]) != digest:
        raise CodeAgentError("the patch file changed after it was recorded")
    paths = _validate_patch(patch.read_text(encoding="utf-8", errors="replace"))
    _require_allowed_paths(run_dir, proposal_key, meta, paths)
    spec = meta.get("acceptance_spec")
    if spec:
        from .acceptance import AcceptanceError, check_paths, classify_change, read_spec
        try:
            spec = read_spec(spec)
            outside = check_paths(paths, spec["allowed_paths"])
        except AcceptanceError as exc:
            raise CodeAgentError(str(exc)) from exc
        if outside:
            raise CodeAgentError("patch exceeds the acceptance contract's allowed paths: " + ", ".join(outside))
        if not (meta.get("implementation") or {}).get("accepted"):
            raise CodeAgentError("implementation has unmet acceptance checks; inspect the report before applying")
        implementation = meta["implementation"]
        if implementation.get("spec_sha256") != spec["spec_sha256"] or implementation.get("patch_sha256") != digest or implementation.get("host_verified") is not True or not implementation.get("host_checks"):
            raise CodeAgentError("implementation has no host-verified checks bound to this patch")
        risk = classify_change(spec, patch.read_text(encoding="utf-8", errors="replace"))
        if risk["risk_tier"] == "human_gated" and not approved_by.strip():
            raise CodeAgentError("a security- or evaluator-critical patch needs a named human approver")
        if require_effect_validation and risk["risk_tier"] != "offline_only":
            from .harness_experiment import read_plan, summarize_result
            validated = False
            decisions_path = run_dir / "harness_decisions.jsonl"
            if decisions_path.is_file():
                for line in decisions_path.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") != "effect_validation" or event.get("validated") is not True or not event.get("approved_by"):
                        continue
                    if event.get("proposal_key") != proposal_key or event.get("spec_sha256") != spec["spec_sha256"] or event.get("change_sha256") != digest:
                        continue
                    # Read the host-stored result rather than trusting a copied verdict.
                    evidence_path = Path(str(event.get("evidence_path") or run_dir / "harness_experiments" / f"{event.get('experiment_id')}.result.json"))
                    try:
                        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                        if evidence_path.resolve() != (run_dir / "harness_experiments" / f"{event.get('experiment_id')}.result.json").resolve() and evidence_path.resolve().parent.name != event.get("experiment_id"):
                            continue
                    except (OSError, json.JSONDecodeError):
                        continue
                    try:
                        summary = summarize_result(evidence)
                    except ValueError:
                        continue
                    if summary["experiment_id"] == event["experiment_id"]:
                        validated = True
                        break
            if not validated:
                raise CodeAgentError("no effect validation binds this acceptance spec and patch")
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, capture_output=True, text=True)
    if check.returncode != 0:
        raise CodeAgentError((check.stderr or check.stdout or "git apply --check failed").strip())
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=repo, capture_output=True, text=True)
    if dirty.stdout.strip():
        raise CodeAgentError("the patch touches files with uncommitted changes; commit or stash them first")
    if check_only:
        return {**meta, "paths": paths, "applied": False, "check": True}
    if run_tests:
        _test_patch_in_worktree(repo, patch, proposal_key)
    apply = subprocess.run(["git", "apply", str(patch)], cwd=repo, capture_output=True, text=True)
    if apply.returncode != 0:
        raise CodeAgentError((apply.stderr or apply.stdout or "git apply failed").strip())
    reverse = subprocess.run(["git", "apply", "--reverse", "--check", str(patch)], cwd=repo, capture_output=True, text=True)
    if reverse.returncode != 0:
        raise CodeAgentError((reverse.stderr or reverse.stdout or "the applied patch no longer matches the working tree").strip())
    proposal = _proposal_row(run_dir, proposal_key)
    row = {
        "schema_version": 1, "event": "code_promotion", "status": "local", "t": time.time(),
        "proposal_key": proposal_key, "target_id": meta.get("target_id") or "",
        "change_ref": "", "patch_sha256": digest,
        "local_patch_sha256": digest, "patch": str(patch), "paths": paths,
        "run_id": proposal.get("run_id") or "",
        "claim_digest": proposal.get("claim_digest") or "",
        "source_seq": proposal.get("seq"),
        "implemented_by": f"{meta.get('provider')}/{meta.get('model')}" if meta.get("model") else "unknown",
        "approved_by": str(approved_by or "cli"),
        "tests_run": "offline pytest -m 'not gpu and not llm' in an isolated worktree" if run_tests else "",
        "spec_sha256": str((meta.get("acceptance_spec") or {}).get("spec_sha256") or ""),
        "implementation_accepted": bool((meta.get("implementation") or {}).get("accepted")),
        "implementation_missing": (meta.get("implementation") or {}).get("missing") or [],
        "risk_tier": (meta.get("implementation") or {}).get("risk_tier") or "",
        "host_checks": (meta.get("implementation") or {}).get("host_checks") or [],
        "legacy_upgrade": (meta.get("implementation") or {}).get("legacy_upgrade"),
    }
    with (run_dir / "harness_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        if meta.get("implementation"):
            implementation = {**meta["implementation"], "event": "implementation", "t": time.time(), "host_verified": bool(run_tests and meta["implementation"]["accepted"] and meta["implementation"].get("host_verified")), "tests_run": row["tests_run"], "approved_by": str(approved_by or "cli")}
            handle.write(json.dumps(implementation, ensure_ascii=False, sort_keys=True) + "\n")
    return {**meta, "applied": True, "local": True, "commit_pending": True, **row}
