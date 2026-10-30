#!/usr/bin/env python3
"""Verify the upstream selective-port ledger without network or device access."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MAP_PATH = ROOT / "upstream-map.json"
MANIFEST_PATH = ROOT / "manifest.sha256"
ACTIONS = {"reference", "reuse", "port", "adapt", "local"}
REFERENCE_ONLY_COMPONENTS = {
    "candidate_scheduler_state_handoff_idioms",
    "dual_reference_baseline_and_antihacking_cases",
    "host_device_project_layout",
    "npu_msopgen_scaffold",
    "npu_build_install_and_packaging",
    "python_binding_and_acl_bridge",
    "easyasc_adapters",
    "device_timing_and_msprof",
    "evaluation_reporting_and_dashboard",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_map(document: dict) -> tuple[dict[str, dict], list[dict], list[dict]]:
    require(document.get("schema_version") == 1, "schema_version must be 1")
    require(isinstance(document.get("g0_complete"), bool), "g0_complete must be bool")
    require(document.get("status") == "source_audit_complete_integration_pending", "unexpected audit status")

    policy = document.get("selection_policy") or {}
    require(policy.get("rule") == "one_authority_per_responsibility_plus_complementary_composition", "bad selection rule")
    require(policy.get("runtime_dual_read") is False, "runtime dual read must be disabled")
    targets = policy.get("hardware_targets") or []
    require("Mac GPU / Metal / MSL" in targets, "Mac target missing")
    require("清微 NPU / locally frozen toolchain" in targets, "清微 target missing")
    rejected = (policy.get("hardware_filter") or {}).get("rejected_from_production_pack_and_guidance") or []
    require(any("CUDA" in item for item in rejected), "CUDA filter missing")
    require(any("AscendC" in item for item in rejected), "CANN filter missing")

    raw_sources = document.get("sources")
    require(isinstance(raw_sources, list) and raw_sources, "sources must be non-empty")
    sources: dict[str, dict] = {}
    for source in raw_sources:
        sid = source.get("id")
        require(isinstance(sid, str) and sid and sid not in sources, f"duplicate/missing source id: {sid!r}")
        require(source.get("repo", "").startswith("https://"), f"{sid}: repo must be HTTPS")
        require(re.fullmatch(r"[0-9a-f]{40}", source.get("commit", "")), f"{sid}: bad commit")
        require(bool(source.get("license")), f"{sid}: license missing")
        root = Path(source.get("root", ""))
        require(root.is_dir(), f"{sid}: root missing: {root}")
        require(head(root) == source["commit"], f"{sid}: checkout HEAD differs from recorded commit")
        source["_root_path"] = root
        sources[sid] = source

    raw_groups = document.get("selection_groups")
    require(isinstance(raw_groups, list) and raw_groups, "selection_groups must be non-empty")
    groups: list[dict] = []
    group_names: set[str] = set()
    for group in raw_groups:
        name = group.get("selection_group")
        require(isinstance(name, str) and name and name not in group_names, f"bad group: {name!r}")
        require(bool(group.get("selected_source")), f"{name}: selected_source missing")
        require(isinstance(group.get("alternatives_rejected"), list), f"{name}: alternatives must be a list")
        group_names.add(name)
        groups.append(group)
    npu = next((g for g in groups if g["selection_group"] == "npu_build_execution_and_profiling"), None)
    require(npu is not None and "清微" in npu["selected_source"], "NPU implementation must be local 清微")
    require("cake2" not in npu["selected_source"], "CAKE2 must not own the NPU build path")

    entries = document.get("entries")
    require(isinstance(entries, list) and entries, "entries must be non-empty")
    components: set[str] = set()
    for entry in entries:
        component = entry.get("component")
        require(isinstance(component, str) and component and component not in components, f"bad component: {component!r}")
        components.add(component)
        action = entry.get("action")
        require(action in ACTIONS, f"{component}: bad action {action!r}")
        for field in ("source", "upstream_path", "local_path", "reason", "verification"):
            require(isinstance(entry.get(field), str) and entry[field], f"{component}: {field} missing")
        source_ids = entry["source"].split(" + ")
        require(all(sid in sources for sid in source_ids), f"{component}: unknown source {source_ids}")
        if component in REFERENCE_ONLY_COMPONENTS:
            require(action == "reference", f"{component}: hardware-specific component must be reference-only")
        files = entry.get("upstream_files", [])
        require(isinstance(files, list), f"{component}: upstream_files must be a list")
        if action in {"reuse", "port", "adapt"}:
            require(bool(files), f"{component}: code port requires upstream_files")
            require(len(source_ids) == 1, f"{component}: code port needs one source root")
        if files:
            root = sources[source_ids[0]]["_root_path"]
            for raw in files:
                relative = PurePosixPath(raw)
                require(not relative.is_absolute() and ".." not in relative.parts, f"{component}: unsafe path {raw}")
                require((root / Path(*relative.parts)).is_file(), f"{component}: upstream file missing: {raw}")
        if "selection_group" in entry:
            require(entry["selection_group"] in group_names, f"{component}: unknown selection_group")
            require(bool(entry.get("composition_role")), f"{component}: composition_role missing")
    return sources, groups, entries


def verify_manifest() -> None:
    lines = MANIFEST_PATH.read_text().splitlines()
    require(len(lines) == 2, "manifest must contain two files")
    seen: set[str] = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None, f"bad digest for {name}")
        require(name not in seen, f"duplicate manifest entry: {name}")
        seen.add(name)
        require(sha256(ROOT / name) == digest, f"manifest mismatch: {name}")
    require(seen == {"upstream-map.json", "verify.py"}, "manifest file set is wrong")


def self_test(document: dict) -> None:
    """Prove that the verifier rejects hardware, identity and provenance drift."""
    def dual_read(doc: dict) -> None:
        doc["selection_policy"]["runtime_dual_read"] = True

    def cuda_port(doc: dict) -> None:
        next(e for e in doc["entries"] if e["component"] == "candidate_scheduler_state_handoff_idioms")["action"] = "adapt"

    def cann_builder(doc: dict) -> None:
        next(g for g in doc["selection_groups"] if g["selection_group"] == "npu_build_execution_and_profiling")["selected_source"] = "cake2 build_and_install.py"

    def missing_file(doc: dict) -> None:
        next(e for e in doc["entries"] if e["component"] == "snapshot_hash_and_python_packaging")["upstream_files"][0] = "missing.py"

    def bad_group(doc: dict) -> None:
        next(e for e in doc["entries"] if e["component"] == "reference_examples")["selection_group"] = "unknown"

    def bad_head(doc: dict) -> None:
        doc["sources"][0]["commit"] = "0" * 40

    cases = [dual_read, cuda_port, cann_builder, missing_file, bad_group, bad_head]
    for mutate in cases:
        candidate = copy.deepcopy(document)
        mutate(candidate)
        try:
            verify_map(candidate)
        except (ValueError, subprocess.CalledProcessError):
            continue
        raise AssertionError(f"negative fixture was accepted: {mutate.__name__}")
    print(f"OK: {len(cases)} negative fixtures rejected")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    document = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    sources, groups, entries = verify_map(document)
    if args.write_manifest:
        lines = [f"{sha256(ROOT / name)}  {name}" for name in ("upstream-map.json", "verify.py")]
        MANIFEST_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    verify_manifest()
    if args.self_test:
        self_test(document)
    print(f"OK: {len(sources)} sources, {len(entries)} entries, {len(groups)} selection groups")
    print("OK: source HEADs, upstream files, hardware filters and composition links verified")
    print("OK: manifest.sha256")


if __name__ == "__main__":
    main()
