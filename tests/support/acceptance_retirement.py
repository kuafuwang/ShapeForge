"""Retirement registry validation and read-only precondition reports.

No function in this module deletes a package.  A human must approve D1 and then
approve each package separately before removing it from the working tree.
"""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from .acceptance_git import GitObjects
from .acceptance_inventory import build_inventory

ROOT = Path(__file__).resolve().parents[2]
RETIRED = ROOT / "tests/support/golden/shared/retired.json"
REQUIRED_FIELDS = {
    "path", "retired_at", "reason", "successor", "approved_by", "last_commit",
    "tree", "files", "bytes", "restore", "change",
}
PACKAGE_PREFIXES = ("tests/support/golden/shared/", "docs/架构/")
PROTECTED = (
    {f"R{i}" for i in range(7)} | {"G0"} | {f"A{i}" for i in range(1, 7)}
    | {"P0", "P1", "P2", "P2-0"}
)


def _is_protected(path: str) -> bool:
    short = path.rsplit("/", 1)[-1]
    return short in PROTECTED or short.endswith("-G0")


def _package_aliases(path: str) -> set[str]:
    short = path.rsplit("/", 1)[-1]
    aliases = {short}
    suffix = short.rsplit("-", 1)[-1]
    if re.fullmatch(r"[A-Z]+[0-9]+[a-z]?", suffix):
        aliases.add(suffix)
    return aliases


def load_retired(root: Path = ROOT) -> dict[str, Any]:
    return json.loads((root / "tests/support/golden/shared/retired.json").read_text(encoding="utf-8"))


def _tree_size(git: GitObjects, oid: str) -> tuple[int, int]:
    files = 0
    size = 0
    for _name, (mode, child) in git.tree_entries(oid).items():
        if mode == "40000" or mode.startswith("04"):
            child_files, child_bytes = _tree_size(git, child)
            files += child_files
            size += child_bytes
        else:
            kind, body = git.read(child)
            if kind != "blob":
                raise ValueError("retirement tree contains a non-blob/submodule entry")
            files += 1
            size += len(body)
    return files, size


def validate_retired(document: dict[str, Any], root: Path | None = None) -> None:
    """Validate schema, frequency limits and (when rooted) Git recoverability."""
    if document.get("schema_version") != 1 or not isinstance(document.get("packages"), list):
        raise ValueError("retired.json must have schema_version=1 and a packages list")
    limits = document.get("limits")
    if limits != {"max_per_system_change": 1, "max_per_quarter": 3}:
        raise ValueError("retirement limits must be one per system change and three per quarter")
    seen: set[str] = set()
    changes: dict[str, int] = {}
    quarters: dict[str, int] = {}
    git = GitObjects(root) if root is not None else None
    for row in document["packages"]:
        if not isinstance(row, dict) or not REQUIRED_FIELDS <= set(row):
            raise ValueError("retirement row is missing required fields")
        path = row["path"]
        if not isinstance(path, str) or not path or any(char.isspace() for char in path):
            raise ValueError(f"invalid retired path: {path!r}")
        pure = PurePosixPath(path)
        if (
            pure.is_absolute() or path == "." or pure.as_posix() != path
            or any(part in {".", ".."} for part in pure.parts)
            or not path.startswith(PACKAGE_PREFIXES) or path in seen
        ):
            raise ValueError(f"invalid, non-canonical or duplicate retired path: {path!r}")
        if _is_protected(path):
            raise ValueError(f"{path}: protected acceptance history cannot be retired")
        seen.add(path)
        for field in ("reason", "approved_by", "change"):
            if not isinstance(row[field], str) or not row[field].strip() or row[field] != row[field].strip():
                raise ValueError(f"{path}: {field} needs a normalized non-empty value")
        if not re.fullmatch(r"\d{8}-[a-z0-9-]+", row["change"]):
            raise ValueError(f"{path}: change must be the normalized retirement epoch id")
        if row["successor"] is not None and not isinstance(row["successor"], str):
            raise ValueError(f"{path}: successor must be a path or null")
        parsed = date.fromisoformat(row["retired_at"])
        change = row["change"]
        quarter = f"{parsed.year}-Q{(parsed.month - 1) // 3 + 1}"
        changes[change] = changes.get(change, 0) + 1
        quarters[quarter] = quarters.get(quarter, 0) + 1
        if changes[change] > 1:
            raise ValueError(f"more than one package retired by system change {change}")
        if quarters[quarter] > 3:
            raise ValueError(f"more than three packages retired in {quarter}")
        if not re.fullmatch(r"[0-9a-f]{40}", row["last_commit"]) or not re.fullmatch(r"[0-9a-f]{40}", row["tree"]):
            raise ValueError(f"{path}: last_commit and tree must be full lowercase object ids")
        if row["restore"] != f"git checkout {row['last_commit']} -- {path}":
            raise ValueError(f"{path}: restore command does not reproduce the recorded commit/path")
        if type(row["files"]) is not int or row["files"] <= 0 or type(row["bytes"]) is not int or row["bytes"] < 0:
            raise ValueError(f"{path}: a package needs positive files and non-negative integer bytes")
        if git is not None:
            from .acceptance_status import load_epochs
            baseline = next(
                (epoch for epoch in load_epochs(root) if epoch["epoch"] == "20261115-baseline"),
                None,
            )
            package_roots = set(baseline.get("package_roots", [])) if baseline else set()
            if path not in package_roots:
                raise ValueError(f"{path}: retirement path is not an exact inventoried package root")
            kind, _body = git.read(row["tree"])
            if kind != "tree":
                raise ValueError(f"{path}: recorded object is not a tree")
            if git.path_oid(row["last_commit"], path) != row["tree"]:
                raise ValueError(f"{path}: tree is not <last_commit>:<path>")
            if _tree_size(git, row["tree"]) != (row["files"], row["bytes"]):
                raise ValueError(f"{path}: recorded file/byte counts differ from the Git tree")


def _check(name: str, ok: bool, detail: object) -> dict[str, Any]:
    return {"name": name, "status": "pass" if ok else "blocked", "detail": detail}


def retirement_check(root: Path, package: str) -> dict[str, Any]:
    """Return every known blocker; the AGENTS.md uniqueness item stays human."""
    inventory = build_inventory(root)
    records = {row["path"]: row for row in inventory["packages"]}
    if package not in records:
        raise ValueError(f"not an inventoried acceptance package: {package}")
    record = records[package]
    checks: list[dict[str, Any]] = []

    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    rule_approved = (
        "Do not rewrite closed acceptance packages." in agents
        and "tests/support/golden/shared/retired.json" in agents
        and '"Retirement" means removing the package from the working tree' in agents
    )
    checks.append(_check(
        "D1_history_rule", rule_approved,
        "approved retirement wording is present" if rule_approved else "D1 has not approved an AGENTS.md rule change",
    ))

    active = sorted(set(record["inbound"]["active_tests"] + record["inbound"]["active_code"]))
    checks.append(_check("active_references", not active, active))

    checks.append(_check("protected_package", not _is_protected(package), sorted(PROTECTED)))

    from .acceptance_status import load_epochs
    retirement_epochs = [
        document for document in load_epochs(root) if document.get("change_kind") == "retirement"
    ]
    pins = []
    for other in inventory["packages"]:
        for pin in other["pins_other_packages"]:
            if pin["package"] != package:
                continue
            registered = any(
                row["package"] == other["path"] and row["check"] == "retirement_pin"
                and row["detail"].get("retired_package") == package
                and row["detail"].get("source_files") == pin["files"]
                and row["detail"].get("pinned_paths") == pin["pinned_paths"]
                for epoch in retirement_epochs for row in epoch["expected_drift"]
            )
            pins.append({"package": other["path"], "pinned_paths": pin["pinned_paths"], "registered": registered})
    checks.append(_check("closed_package_pins", all(pin["registered"] for pin in pins), pins))
    closed_references = []
    for reference in record["inbound"]["closed_packages"]:
        registered = any(
            row["package"] == reference["package"] and row["check"] == "retirement_reference"
            and row["detail"].get("retired_package") == package
            and row["detail"].get("source_files") == reference["files"]
            for epoch in retirement_epochs for row in epoch["expected_drift"]
        )
        closed_references.append({**reference, "registered": registered})
    checks.append(_check(
        "closed_package_references", all(item["registered"] for item in closed_references),
        closed_references,
    ))

    documents = record["inbound"]["docs"]
    checks.append(_check("documentation_references", not documents, documents))

    structure = (root / "tests/shared/test_structure.py").read_text(encoding="utf-8")
    archive_roots = re.findall(r'"([a-zA-Z0-9_./\-\u4e00-\u9fff]+)"', structure.split("ARCHIVE_ROOTS = (", 1)[1].split(")", 1)[0])
    affected_roots = [name for name in archive_roots if package == f"tests/support/{name}"]
    checks.append(_check("archive_roots", not affected_roots, affected_roots))

    short = record["path"].split("/")[-1]
    aliases = _package_aliases(package)

    def mentions_agents(line: str) -> bool:
        if package in line or f"docs/架构/{package.removeprefix('tests/support/golden/shared/')}" in line:
            return True
        if any(re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", line) for alias in aliases):
            return True
        # Expand compact phase lists such as P2-H/G/F before deciding that
        # AGENTS.md does not mention the candidate.
        return any(
            f"{prefix}-{letter}" == short
            for prefix, letters in re.findall(r"\b(P\d+)-([A-Z](?:/[A-Z])*)", line)
            for letter in letters.split("/")
        )

    agent_lines = [
        {"line": number, "text": line.strip()}
        for number, line in enumerate(agents.splitlines(), 1) if mentions_agents(line)
    ]
    checks.append({
        "name": "AGENTS_unique_source_human_review",
        "status": "pass" if not agent_lines else "human_required",
        "detail": agent_lines,
    })
    retired = load_retired(root)
    try:
        validate_retired(retired, root)
    except (KeyError, ValueError) as exc:
        checks.append(_check("retirement_registry", False, str(exc)))
    else:
        checks.append(_check("retirement_registry", True, retired["limits"]))
    eligible = all(item["status"] == "pass" for item in checks)
    return {
        "package": package,
        "kind": record["kind"],
        "eligible": eligible,
        "deletion_executed": False,
        "checks": checks,
        "note": "A report is not approval. D1, AGENTS.md uniqueness and every package need separate human decisions.",
    }
