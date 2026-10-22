"""Prepare an isolated worktree, run one code agent, and collect a human-reviewable patch.

This is the P2 implementation path. The code agent is the default implementer, but it has
no merge/approval authority: the worktree is detached, the patch is inspected by a person,
and only ``apply_code_patch(..., approved_by=...)`` writes into the main repository.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
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
    try:
        proc = subprocess.run(argv, cwd=worktree, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        raise CodeAgentError(f"code agent timed out after {timeout_s:.0f}s") from exc
    stream = (proc.stdout or "") + "\n" + (proc.stderr or "")
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
        raise CodeAgentError(f"code agent left no diff; hmz tail: {tail}")
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
        "report": result.get("report") or {},
    }
    (folder / f"{proposal_key}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def apply_code_patch(
    run_dir: Path,
    *,
    proposal_key: str,
    root: Path | None = None,
    check_only: bool = False,
    run_tests: bool = False,
    approved_by: str = "cli",
) -> dict:
    """Apply a generated code patch to the main repo after git apply --check, never commit."""
    run_dir = Path(run_dir)
    repo = Path(root) if root is not None else Path(__file__).resolve().parent.parent
    folder = run_dir / PATCH_DIR
    patch = folder / f"{proposal_key}.patch"
    meta_path = folder / f"{proposal_key}.json"
    if not patch.is_file():
        raise CodeAgentError("no generated patch; run `taic harness implement` first")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    check = subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, capture_output=True, text=True)
    if check.returncode != 0:
        raise CodeAgentError((check.stderr or check.stdout or "git apply --check failed").strip())
    if check_only:
        return {**meta, "applied": False, "check": True}
    if run_tests:
        tests = subprocess.run(
            ["uv", "run", "pytest", "-q", "-m", "not gpu and not llm"],
            cwd=repo, capture_output=True, text=True,
        )
        if tests.returncode != 0:
            raise CodeAgentError((tests.stdout + tests.stderr)[-2000:].strip() or "offline tests failed")
    apply = subprocess.run(["git", "apply", str(patch)], cwd=repo, capture_output=True, text=True)
    if apply.returncode != 0:
        raise CodeAgentError((apply.stderr or apply.stdout or "git apply failed").strip())
    row = {
        "schema_version": 1, "event": "code_promotion", "status": "local", "t": time.time(),
        "proposal_key": proposal_key, "target_id": meta.get("target_id") or "",
        "change_ref": "", "patch_sha256": meta.get("sha256"),
        "local_patch_sha256": meta.get("sha256"),
        "implemented_by": f"{meta.get('provider')}/{meta.get('model')}" if meta.get("model") else "unknown",
        "approved_by": str(approved_by or "cli"),
        "tests_run": "offline pytest -m 'not gpu and not llm'" if run_tests else "",
    }
    with (run_dir / "harness_decisions.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return {**meta, "applied": True, "local": True, "commit_pending": True, **row}
