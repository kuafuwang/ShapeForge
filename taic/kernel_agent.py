"""Write one kernel by asking the pi coding agent, through hmz, to edit a file.

The agent is not given ``lookup_msl`` or ``quick_check``. It reads the MSL notes placed in its
directory and, when it wants a check, runs ``tcenv eval --quick``. That command does not time
the kernel. The search times the file the agent leaves behind and decides whether it stays.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from .llm import LLMError
from .parse import extract_metal

FLOW = Path(__file__).resolve().parent / "flows" / "kernel_once"
SKILL = Path(__file__).resolve().parent / "skills" / "msl_reference.md"
_KERNEL = re.compile(r"\bkernel\s+void\s+(\w+)\b")


class AgentError(LLMError):
    """hmz or pi could not take this turn. One branch is skipped; every branch failing ends the search."""


def kernel_text(text: str, kernel_name: str) -> str | None:
    """The Metal source in ``text``: a fenced block, or the whole file when it declares the kernel."""
    fenced = extract_metal(text, kernel_name)
    if fenced:
        return fenced
    if _KERNEL.search(text or "") and re.search(r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b", text or ""):
        body = text.strip("\n")
        return body + "\n"
    return None


def agent_spec(provider: str, model: str) -> str:
    """The ``-a assistant=`` value. pi names models as ``provider/id``."""
    return f"pi/{provider}/{model}"


def task_markdown(messages: list[dict], *, repo: Path, kernel_path: Path, task_name: str, dtype: str, size: str, kernel_name: str) -> str:
    """What the agent reads. The chat prompts are included, then replaced where they ask for a fence."""
    pasted = []
    for message in messages:
        if message.get("role") in ("system", "user") and isinstance(message.get("content"), str):
            pasted.append(message["content"])
    check = (
        "TCENV_GPU_LOCK_FILE=/tmp/ai_lowerling-gpu.lock "
        f"uv run --directory {repo} python -m tcenv.cli eval {task_name} {kernel_path} "
        f"--dtype {dtype} --size {size} --quick"
    )
    rules = (
        "You are a coding agent. The messages above were written for a chat model that answers "
        "with one ```metal fence. Ignore that reply format.\n"
        f"Edit only `{kernel_path.name}` in this directory. It must contain the complete kernel "
        f"`kernel void {kernel_name}`. Do not change the kernel name, the launch grid, or the thread count.\n"
        "A skill named taic-tools is loaded with this session. It defines two tools:\n"
        "lookup_msl: read `msl_reference.md` in this directory. That file is the Metal reference.\n"
        f"quick_check: bash this command and read its output:\n{check}\n"
        "That command does not time the kernel. A time you measure yourself is not the search result. "
        "Do not run a benchmark. The search times this file after you exit and decides whether it stays.\n"
        "A kernel that does not return can freeze the display."
    )
    return "\n\n".join([*pasted, rules]) + "\n"


def cost_from_stream(text: str) -> float | None:
    """The largest finite ``cost`` hmz printed. Usage costs are cumulative, so the max is the spend."""
    best: float | None = None
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        found = _costs(obj)
        if not found:
            continue
        top = max(found)
        best = top if best is None else max(best, top)
    return best


def _costs(obj: dict) -> list[float]:
    found = []
    for key in ("cost", "cost_usd"):
        value = obj.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0:
            found.append(float(value))
    usage = obj.get("usage")
    if isinstance(usage, dict):
        found.extend(_costs(usage))
    return found


def run_kernel_agent(
    messages: list[dict],
    *,
    work: Path,
    repo: Path,
    task_name: str,
    dtype: str,
    size: str,
    kernel_name: str,
    provider: str,
    model: str,
    budget_usd: float,
    hmz: str | None = None,
    timeout_s: float = 3600,
) -> tuple[str | None, float | None, str]:
    """Run one hmz turn. Returns ``(kernel source or None, cost or None, status tail)``.

    Raises:
      AgentError: hmz is not on PATH, or the process cannot be started.
    """
    binary = hmz or shutil.which("hmz")
    if not binary:
        raise AgentError("hmz is not on PATH; install it with `uv tool install hmz` or pass --kernel-agent off")
    work.mkdir(parents=True, exist_ok=True)
    kernel_path = work / "kernel.metal"
    seeded = ""
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            got = kernel_text(content, kernel_name)
            if got:
                seeded = got
    kernel_path.write_text(seeded)
    if SKILL.exists():
        shutil.copyfile(SKILL, work / "msl_reference.md")
    (work / "TASK.md").write_text(task_markdown(
        messages, repo=repo, kernel_path=kernel_path, task_name=task_name, dtype=dtype, size=size, kernel_name=kernel_name,
    ))
    cap = max(float(budget_usd), 0.01)
    argv = [
        binary, "exec", "-f", str(FLOW),
        "-a", f"assistant={agent_spec(provider, model)}",
        "-p", f"budget.cost={cap:.4f}",
        "--json",
        "--", "Read TASK.md in this directory and do it. Leave the finished kernel in kernel.metal.",
    ]
    try:
        proc = subprocess.run(argv, cwd=work, capture_output=True, text=True, timeout=timeout_s, env=os.environ.copy())
    except subprocess.TimeoutExpired as exc:
        raise AgentError(f"hmz did not finish within {timeout_s:.0f}s") from exc
    except OSError as exc:
        raise AgentError(f"could not start hmz: {exc}") from exc
    stream = (proc.stdout or "") + "\n" + (proc.stderr or "")
    cost = cost_from_stream(stream)
    written = kernel_path.read_text() if kernel_path.exists() else ""
    source = kernel_text(written, kernel_name)
    tail = (proc.stderr or proc.stdout or "").strip()[-400:]
    if proc.returncode != 0 and source is None:
        raise AgentError(f"hmz exited {proc.returncode}: {tail}")
    return source, cost, tail
