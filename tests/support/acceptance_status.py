"""Classify acceptance-package drift without executing any acceptance verifier.

Manifest rows are matched to exact epoch entries, including the observed hash.
An old registration therefore cannot authorize a later, third version of a file.
Package-internal evidence mismatches are always closed-package rewrites.  A
Markdown update is accepted only with named human approval, an explicit epoch,
and a manifest row that already names the new bytes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from .acceptance_git import GitObjects
from .acceptance_inventory import _depends, _package_paths, _relative, _verifier_for
from .acceptance_manifest import GOLDEN_SHARED, ManifestEntry, manifest_paths, read_manifest, scan_manifests

ROOT = Path(__file__).resolve().parents[2]
EPOCHS = GOLDEN_SHARED / "acceptance-epochs"
EPOCH_NAME = re.compile(r"^(\d{8})-([a-z0-9-]+)\.json$")
CHANGE_KINDS = {"prompt", "closure", "worktree", "registry", "evidence_restore", "retirement"}
ENVIRONMENT_CHECKS = {"prompt", "closure", "worktree_head", "track_registry"}
BASELINE_EPOCH = "20261115-baseline"


def load_epochs(root: Path = ROOT) -> list[dict[str, Any]]:
    """Load and validate append-only epoch records; no latest-file alias exists."""
    directory = root / EPOCHS
    records: list[dict[str, Any]] = []
    epochs: set[str] = set()
    introduced_manifests: set[str] = set()
    for path in sorted(directory.glob("*.json")):
        match = EPOCH_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid acceptance epoch filename: {path.name}")
        document = json.loads(path.read_text(encoding="utf-8"))
        required = {"epoch", "date", "reason", "commit", "approved_by", "change_kind", "before", "after", "expected_drift"}
        missing = required - set(document)
        if missing:
            raise ValueError(f"{path}: missing epoch fields {sorted(missing)}")
        epoch = document["epoch"]
        if epoch != path.stem or epoch in epochs:
            raise ValueError(f"{path}: epoch must be unique and equal to its filename stem")
        if document["date"] != f"{match.group(1)[:4]}-{match.group(1)[4:6]}-{match.group(1)[6:]}":
            raise ValueError(f"{path}: date does not match the filename")
        if document["change_kind"] not in CHANGE_KINDS:
            raise ValueError(f"{path}: unknown change_kind {document['change_kind']!r}")
        if not isinstance(document["commit"], str) or not re.fullmatch(r"[0-9a-f]{40}", document["commit"]):
            raise ValueError(f"{path}: commit must be a full lowercase Git commit id")
        if not isinstance(document["approved_by"], str) or not document["approved_by"].strip():
            raise ValueError(f"{path}: named human approval is required")
        if not isinstance(document["expected_drift"], list):
            raise ValueError(f"{path}: expected_drift must be a list")
        if "manifest_pins" in document:
            manifest_pins = document["manifest_pins"]
            if epoch != BASELINE_EPOCH or not isinstance(manifest_pins, dict):
                raise ValueError(f"{path}: only the N0 baseline may pin manifest bytes")
            if any(
                not isinstance(name, str) or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in manifest_pins.items()
            ):
                raise ValueError(f"{path}: malformed manifest_pins")
            introduced_manifests.update(manifest_pins)
        if "new_manifest_pins" in document:
            new_pins = document["new_manifest_pins"]
            if epoch == BASELINE_EPOCH or not isinstance(new_pins, dict):
                raise ValueError(f"{path}: new_manifest_pins belongs to a non-baseline introduction")
            if any(
                not isinstance(name, str) or not name.endswith("/manifest.sha256")
                or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                for name, digest in new_pins.items()
            ):
                raise ValueError(f"{path}: malformed new_manifest_pins")
            repeated = sorted(introduced_manifests.intersection(new_pins))
            if repeated:
                raise ValueError(f"{path}: manifest bytes were already pinned at introduction: {repeated}")
            introduced_manifests.update(new_pins)
        if "manifest_rows" in document:
            rows = document["manifest_rows"]
            if epoch != BASELINE_EPOCH or not isinstance(rows, dict) or set(rows) != set(document.get("manifest_pins", {})):
                raise ValueError(f"{path}: manifest_rows must accompany exactly the baseline manifest_pins")
            if any(
                not isinstance(mapping, dict) or any(
                    not isinstance(name, str) or not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    for name, digest in mapping.items()
                )
                for mapping in rows.values()
            ):
                raise ValueError(f"{path}: malformed manifest_rows")
        for row in document["expected_drift"]:
            if not isinstance(row, dict) or not {"package", "check", "detail"} <= set(row):
                raise ValueError(f"{path}: malformed expected_drift row: {row!r}")
            if not isinstance(row["package"], str) or not isinstance(row["check"], str):
                raise ValueError(f"{path}: expected_drift package/check must be strings")
            if not isinstance(row["detail"], dict):
                raise ValueError(f"{path}: detail must bind paths and observed values")
        epochs.add(epoch)
        records.append(document)
    return records


def current_environment(root: Path = ROOT) -> dict[str, str]:
    """Return the live values to which environment drift registrations are bound."""
    from taic.trust import current_prompt_sha1
    from tcenv.taskcontract.pack import code_import_closure

    files = code_import_closure(root)
    closure = hashlib.sha256(b"".join((root / name).read_bytes() for name in files)).hexdigest()
    return {
        "prompt": current_prompt_sha1(),
        "closure": closure,
        "worktree_head": GitObjects(root).head(),
        "track_registry": hashlib.sha256((root / "backends/metal/tracks.json").read_bytes()).hexdigest(),
    }


def package_dependencies(root: Path = ROOT) -> dict[str, list[str]]:
    """Use the static N1 scanner without importing a verifier or reading Git history."""
    result: dict[str, list[str]] = {}
    for package in _package_paths(root):
        depends = _depends(_verifier_for(package, root), package, root)
        if depends:
            result[_relative(package, root)] = depends
    return result


def _actual(entry: ManifestEntry) -> str | None:
    return entry.actual_sha256


def _manifest_registration(
    epochs: list[dict[str, Any]], package: str, entry: ManifestEntry, actual: str | None,
) -> dict[str, Any] | None:
    """Match the original row, expected bytes and observed bytes, not just a package name."""
    for epoch in epochs:
        for row in epoch["expected_drift"]:
            if row["package"] != package or row["check"] != "manifest_sha256":
                continue
            detail = row["detail"]
            if (
                detail.get("path") == entry.name
                and detail.get("before") == entry.expected_sha256
                and detail.get("after") == actual
                and detail.get("resolution") == entry.resolution
            ):
                return {"epoch": epoch["epoch"], "approved_by": epoch["approved_by"], "detail": detail}
    return None


def _environment_registration(
    epochs: list[dict[str, Any]], package: str, check: str, current: str,
) -> dict[str, Any] | None:
    for epoch in epochs:
        for row in epoch["expected_drift"]:
            if row["package"] == package and row["check"] == check and row["detail"].get("after") == current:
                return {"epoch": epoch["epoch"], "approved_by": epoch["approved_by"], "detail": row["detail"]}
    return None


def _document_updates(
    root: Path, epochs: list[dict[str, Any]], entries: list[ManifestEntry],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate the three D3 requirements independently of hash-bucket classification."""
    updates: list[dict[str, Any]] = []
    problems: list[str] = []
    by_row = {(entry.package, entry.name): entry for entry in entries}
    retired = _retired_paths(root)
    for epoch in epochs:
        for row in epoch["expected_drift"]:
            if row["check"] != "document_update":
                continue
            if row["package"] in retired and not (root / row["package"]).exists():
                # A retired package keeps its historical D3 record in Git, but no live manifest
                # row remains to satisfy it.  The retirement registry, not a missing README,
                # is now the authority for those bytes.
                continue
            package = root / row["package"]
            detail = row["detail"]
            name = detail.get("path")
            entry = by_row.get((package, name))
            label = f"{row['package']}:{name}"
            if not isinstance(name, str) or not name.endswith(".md"):
                problems.append(f"document_update is not Markdown: {label}")
                continue
            if entry is None or entry.resolution != "package_internal":
                problems.append(f"document_update lacks a package-internal manifest row: {label}")
                continue
            actual = entry.actual_sha256
            if detail.get("classification") != "文档更新，不是证据篡改":
                problems.append(f"document_update lacks the explicit non-tampering classification: {label}")
            if detail.get("after") != actual or entry.expected_sha256 != actual:
                problems.append(f"document_update is not bound to both current bytes and the manifest: {label}")
                continue
            updates.append({
                "package": row["package"], "path": name, "epoch": epoch["epoch"],
                "approved_by": epoch["approved_by"], "before": detail.get("before"), "after": actual,
            })
    return updates, problems


def _retired_paths(root: Path) -> set[str]:
    """Validated retired package roots. An invalid record does not exempt a missing manifest."""
    from .acceptance_retirement import retired_package_roots

    return retired_package_roots(root)


def _baseline_manifests(epochs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for epoch in epochs:
        if epoch["epoch"] == BASELINE_EPOCH:
            return epoch
    return None


def _manifest_bytes_are_pinned(
    root: Path, epochs: list[dict[str, Any]], manifest: Path, current: str,
) -> bool:
    """Trust baseline bytes, or a document update changing exactly one manifest row."""
    relative = _relative(manifest, root)
    baseline = _baseline_manifests(epochs)
    if baseline is None:
        return False
    baseline_pins = baseline.get("manifest_pins", {})
    if baseline_pins.get(relative) == current:
        return True
    if any(epoch.get("new_manifest_pins", {}).get(relative) == current for epoch in epochs):
        # A new package path may be introduced once by a named epoch.  load_epochs rejects any
        # later introduction of the same path, so this cannot become a standing rewrite alias.
        return True
    baseline_rows = baseline.get("manifest_rows", {}).get(relative)
    if baseline_rows is None:
        return False
    current_rows = {entry.name: entry.expected_sha256 for entry in read_manifest(manifest, root)}
    package = _relative(manifest.parent, root)
    for epoch in epochs:
        for row in epoch["expected_drift"]:
            detail = row["detail"]
            document_path = detail.get("path")
            if not (
                row["package"] == package and row["check"] == "document_update"
                and detail.get("manifest_before") == baseline_pins.get(relative)
                and detail.get("manifest_after") == current
                and detail.get("before") == baseline_rows.get(document_path)
                and set(current_rows) == set(baseline_rows)
            ):
                continue
            if all(
                name == document_path or digest == baseline_rows[name]
                for name, digest in current_rows.items()
            ):
                return True
    return False


def build_status(root: Path = ROOT) -> dict[str, Any]:
    """Hash manifests and classify every mismatch; never execute package code."""
    epochs = load_epochs(root)
    manifests = manifest_paths(root)
    entries = [entry for manifest in manifests for entry in read_manifest(manifest, root)]
    updates, document_problems = _document_updates(root, epochs, entries)
    packages: dict[str, dict[str, Any]] = {}

    def record(package: str) -> dict[str, Any]:
        return packages.setdefault(package, {
            "path": package,
            "manifest": Counter({
                key: 0 for key in ("total", "matched", "registered", "rewritten", "unregistered")
            }),
            "environment": [],
            "rewritten": [],
            "unregistered": [],
            "registered": [],
        })

    for manifest in manifests:
        package = _relative(manifest.parent, root)
        actual = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if not _manifest_bytes_are_pinned(root, epochs, manifest, actual):
            item = {"path": "manifest.sha256", "expected": "baseline or approved document update", "actual": actual}
            row = record(package)
            row["manifest"]["rewritten"] += 1
            row["rewritten"].append(item)

    baseline = _baseline_manifests(epochs)
    pinned_manifests = set(baseline.get("manifest_pins", {})) if baseline else set()
    retired = _retired_paths(root)
    for relative in sorted(pinned_manifests - {_relative(path, root) for path in manifests}):
        package = Path(relative).parent.as_posix()
        if package in retired and not (root / package).exists():
            continue
        row = record(package)
        row["manifest"]["rewritten"] += 1
        row["rewritten"].append({
            "path": "manifest.sha256", "expected": baseline["manifest_pins"][relative], "actual": None,
        })

    for entry in entries:
        package = _relative(entry.package, root)
        row = record(package)
        row["manifest"]["total"] += 1
        actual = _actual(entry)
        if actual == entry.expected_sha256:
            row["manifest"]["matched"] += 1
            continue
        item = {
            "path": entry.name, "expected": entry.expected_sha256, "actual": actual,
            "resolution": entry.resolution,
        }
        if entry.inside_package:
            # JSON/NDJSON/golden/verify.py can never be excused by an epoch.  A
            # Markdown row with an updated manifest would have matched above;
            # if it still differs, the documented update is incomplete or stale.
            row["manifest"]["rewritten"] += 1
            row["rewritten"].append(item)
            continue
        registration = _manifest_registration(epochs, package, entry, actual)
        if registration is None:
            row["manifest"]["unregistered"] += 1
            row["unregistered"].append(item)
        else:
            row["manifest"]["registered"] += 1
            row["registered"].append({**item, "epoch": registration["epoch"]})

    if (root / "taic/trust.py").is_file() and (root / "tcenv/taskcontract/pack.py").is_file():
        environment = current_environment(root)
        for package, depends in package_dependencies(root).items():
            row = record(package)
            for check in ENVIRONMENT_CHECKS & set(depends):
                current = environment[check]
                registration = _environment_registration(epochs, package, check, current)
                item = {"check": check, "current": current}
                if registration is None:
                    row["environment"].append({**item, "status": "unregistered"})
                    row["unregistered"].append(item)
                else:
                    row["environment"].append({**item, "status": "registered", "epoch": registration["epoch"]})

    # Include evidence-only packages in the report even though they have no manifest.
    for package in _package_paths(root):
        record(_relative(package, root))

    summary = Counter({
        key: 0 for key in (
            "total", "matched", "registered", "rewritten", "unregistered",
            "environment_registered", "environment_unregistered",
        )
    })
    for row in packages.values():
        summary.update(row["manifest"])
        for item in row["environment"]:
            summary[f"environment_{item['status']}"] += 1
    summary["document_updates"] = len(updates)
    summary["invalid_document_updates"] = len(document_problems)
    summary["packages"] = len(packages)
    summary["unregistered"] += summary["environment_unregistered"] + len(document_problems)
    return {
        "schema_version": 1,
        "summary": dict(sorted(summary.items())),
        "packages": [
            {
                **row,
                "manifest": dict(sorted(row["manifest"].items())),
                "environment": sorted(row["environment"], key=lambda item: item["check"]),
            }
            for _, row in sorted(packages.items())
        ],
        "document_updates": updates,
        "document_update_problems": document_problems,
    }


def retirement_check(root: Path, package: str) -> dict[str, Any]:
    from .acceptance_retirement import retirement_check as check
    return check(root, package)


def print_status(document: dict[str, Any]) -> None:
    summary = document["summary"]
    print("Acceptance status (read-only; no verifier executed)")
    print(
        f"manifest={summary.get('total', 0)} matched={summary.get('matched', 0)} "
        f"registered_drift={summary.get('registered', 0)}"
    )
    print(
        f"closed_package_rewrites={summary.get('rewritten', 0)} "
        f"unregistered={summary.get('unregistered', 0)} "
        f"document_updates={summary.get('document_updates', 0)}"
    )
    print(
        f"environment_registered={summary.get('environment_registered', 0)} "
        f"environment_unregistered={summary.get('environment_unregistered', 0)}"
    )
    for row in document["packages"]:
        if row["rewritten"] or row["unregistered"] or row["registered"] or row["environment"]:
            manifest = row["manifest"]
            print(
                f"{row['path']}: matched={manifest.get('matched', 0)} "
                f"registered={manifest.get('registered', 0)} rewritten={manifest.get('rewritten', 0)} "
                f"unregistered={manifest.get('unregistered', 0)}; "
                f"environment={','.join(item['check'] + ':' + item['status'] for item in row['environment']) or '-'}"
            )
    for update in document["document_updates"]:
        print(
            f"DOCUMENT UPDATE {update['package']}/{update['path']}: "
            f"{update['before']} -> {update['after']} ({update['epoch']}; approved by {update['approved_by']})"
        )
    for row in document["packages"]:
        for item in row["rewritten"]:
            print(f"CLOSED PACKAGE REWRITTEN {row['path']}: {item['path']}")
        for item in row["unregistered"]:
            print(f"UNREGISTERED {row['path']}: {item}")
    for problem in document["document_update_problems"]:
        print(f"UNREGISTERED DOCUMENT UPDATE: {problem}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the complete status as JSON")
    parser.add_argument("--retirement-check", metavar="PACKAGE", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.retirement_check:
        result = retirement_check(ROOT, args.retirement_check)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    document = build_status(ROOT)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_status(document)
    if document["summary"].get("rewritten", 0) or document["summary"].get("unregistered", 0):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
