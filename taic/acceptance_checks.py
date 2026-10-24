"""Host-owned implementation checks. Proposals name IDs, never commands or imports."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


class CheckError(ValueError):
    """An unknown, unsafe or failed host acceptance check."""


@dataclass(frozen=True)
class CheckDefinition:
    check_id: str
    description: str
    pytest_nodes: tuple[str, ...]
    paths: tuple[str, ...]
    capabilities: tuple[str, ...] = ()


def _definitions() -> dict[str, CheckDefinition]:
    rows = (
        CheckDefinition("reporting.offline_contract", "A read-only reporting change has an offline regression check.", ("tests/shared/test_acceptance.py",), ("tcenv/report.py",)),
        CheckDefinition("tools.offline_protocol", "A host-mediated offline tool obeys its typed protocol.", ("tests/backends/metal/test_tool_protocol.py",), ("tcenv/tools/",)),
        CheckDefinition("harness.offline_experiment", "Experiment planning and evidence preserve their contracts without a model or GPU.", ("tests/shared/test_harness_experiment.py", "tests/shared/test_harness_runner.py"), ("experiments/",)),
        CheckDefinition("proposal.offline_contract", "A proposal conversion or acceptance handoff retains its offline contract.", ("tests/shared/test_acceptance.py",), ("taic/",)),
    )
    return {row.check_id: row for row in rows}


def check_definition(check_id: str) -> CheckDefinition:
    try:
        return _definitions()[check_id]
    except (KeyError, TypeError) as exc:
        raise CheckError(f"unknown host acceptance check_id: {check_id!r}") from exc


def validate_items(items) -> list[dict]:
    if not isinstance(items, list) or len(items) > 16:
        raise CheckError("host_checks must be a list of at most 16 objects")
    out = []
    for item in items:
        if not isinstance(item, dict) or set(item) - {"check_id", "description"}:
            raise CheckError("a host check needs only check_id and description")
        definition = check_definition(item.get("check_id"))
        description = item.get("description") or definition.description
        if not isinstance(description, str) or not description.strip():
            raise CheckError("a host check description is required")
        out.append({"check_id": definition.check_id, "description": description.strip()[:500]})
    return out


def applicable_paths(check_ids: list[str]) -> list[str]:
    return sorted({path for check_id in check_ids for path in check_definition(check_id).paths})


def run_checks(items: list[dict], *, worktree: Path, project: Path) -> list[dict]:
    """Run fixed pytest nodes in a patched tree. A skip is not an acceptance pass."""
    results = []
    for item in validate_items(items):
        definition = check_definition(item["check_id"])
        missing = [node for node in definition.pytest_nodes if not (worktree / node).is_file()]
        if definition.capabilities:
            raise CheckError("this host check requires capabilities not granted to the offline runner")
        record = {"check_id": definition.check_id, "passed": False, "exit_code": None, "output_excerpt": "", "reason": ""}
        if missing:
            record["reason"] = "missing registered tests: " + ", ".join(missing)
            results.append(record)
            continue
        command = [sys.executable, "-m", "pytest", "-q", "-n", "0", "-m", "not gpu and not llm and not slow", *definition.pytest_nodes]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
        completed = subprocess.run(command, cwd=worktree, env=env, capture_output=True, text=True, timeout=120)
        output = (completed.stdout + "\n" + completed.stderr)[-4000:]
        record["exit_code"] = completed.returncode
        record["output_excerpt"] = output
        record["passed"] = completed.returncode == 0 and not re.search(r"\b[1-9]\d* skipped\b", output)
        if not record["passed"]:
            record["reason"] = "registered check failed or was skipped"
        results.append(record)
    return results
