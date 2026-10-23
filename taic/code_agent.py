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


class CodeAgentError(LLMError):
    """The code agent or worktree could not produce a usable patch."""


def _normalize_patch_path(value: str) -> str:
    raw = str(value or "").replace("\\", "/").strip()
    if not raw:
        return ""
    text = posixpath.normpath(raw)
    while text.startswith("./"):
        text = text[2:]
    return text


def patch_paths(patch: str) -> list[str]:
    """Ask Git to decode paths, including its quoted octal escape sequences."""
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
    """Paths the P2 code path may never change, even if the proposal asks for them."""
    value = _normalize_patch_path(path)
    if not value:
        return False
    if value == ".." or value.startswith("../"):
        return True
    value = value.lstrip("/")
    if value == "tasks" or value.startswith("tasks/"):
        return True
    if value in {"tcenv/gpulock.py", "taic/trust.py", "control.json"}:
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
    paths = patch_paths(patch)
    if not paths:
        raise CodeAgentError("the patch does not name any files")
    blocked = forbidden_paths(paths)
    if blocked:
        raise CodeAgentError("patch touches a path the P2 code path may not change: " + ", ".join(sorted(blocked)))
    return paths


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=check)


def prepare_worktree(repo: Path, run_dir: Path, proposal_key: str) -> Path:
    """Create a detached worktree from HEAD under the run directory."""
    path = Path(run_dir) / "code_worktrees" / proposal_key
    if path.exists():
        remove_worktree(repo, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "--detach", str(path), "HEAD")
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    _git(repo, "worktree", "remove", "--force", str(path), check=False)
    shutil.rmtree(path, ignore_errors=True)


def estimate_code_budget(prices: Prices, prompt: str, *, output_tokens: int = 8000) -> dict:
    """A price-derived allowance for one code-agent turn, including context/cache traffic."""
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


def build_prompt(proposal: dict, *, run_id: str) -> str:
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

Implement the smallest coherent change, add or update tests, and run the relevant offline
tests. Do not modify `tasks/`, `tcenv/gpulock.py`, `taic/trust.py`, `targets/*/trust.toml`,
`control.json`, or credentials. Do not commit. At the end write `IMPLEMENT_REPORT.json`:

{{
  "summary": "what changed",
  "tests": ["commands and results"],
  "risks": "remaining risks",
  "files": ["paths changed"]
}}

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
    """Run one hmz code-agent turn and collect its diff. It never commits or applies."""
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
    }


def collect_patch(worktree: Path) -> str:
    """A binary-safe patch including new files, without committing or leaving the index staged."""
    _git(worktree, "add", "-A")
    diff = _git(worktree, "diff", "--cached", "--binary", "HEAD")
    _git(worktree, "reset", "--quiet")
    return diff.stdout


def write_patch(run_dir: Path, proposal_key: str, result: dict, *, target_id: str = "") -> dict:
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
        "report": result.get("report") or {},
    }
    (folder / f"{proposal_key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _proposal_row(run_dir: Path, proposal_key: str) -> dict:
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
    """Whether the live checkout contains exactly HEAD plus every patch, in order.

    A reverse apply only proves one patch can be undone; it does not catch unrelated edits
    elsewhere in the same file. Compare the final files against a fresh worktree instead.
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


def _test_patch_in_worktree(repo: Path, patch: Path, proposal_key: str) -> None:
    """Apply and test the generated patch in a detached worktree, never in the live checkout."""
    parent = Path(tempfile.mkdtemp(prefix=f"taic-{proposal_key}-test-"))
    temp = parent / "tree"
    added = False
    try:
        try:
            _git(repo, "worktree", "add", "--detach", str(temp), "HEAD")
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
    finally:
        if added:
            _git(repo, "worktree", "remove", "--force", str(temp), check=False)
        shutil.rmtree(parent, ignore_errors=True)


def apply_code_patch(
    run_dir: Path,
    *,
    proposal_key: str,
    root: Path | None = None,
    check_only: bool = False,
    run_tests: bool = False,
    approved_by: str = "cli",
) -> dict:
    """Apply a generated code patch to the main repo after checks, never commit.

    Tests run against the patched tree in an isolated worktree. The live checkout is only
    touched after that check passes, and the applied bytes must still match the patch.
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
    }
    with (run_dir / "harness_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {**meta, "applied": True, "local": True, "commit_pending": True, **row}
