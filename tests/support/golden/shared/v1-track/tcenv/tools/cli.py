"""Offline schema/manifest inspection and explicit host-side tool invocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tcenv.tools.cli")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    refresh = sub.add_parser("refresh-binding", help="recompute digests for an existing reviewed tool; never registers one")
    refresh.add_argument("--registry", type=Path, required=True)
    scaffold = sub.add_parser("scaffold-proposal", help="write inert drafts; no registration or execution")
    scaffold.add_argument("proposal")
    scaffold.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("check", help="validate only; no GPU is used")
    check.add_argument("request")
    check.add_argument("--session-role", default="", help="host-assigned role; empty has no capabilities")
    call = sub.add_parser("invoke", help="run a host-approved tool; may use the GPU")
    call.add_argument("request")
    call.add_argument("--session-role", required=True, choices=["kernel_once", "offline_test"])
    call.add_argument("--run-dir", required=True)
    call.add_argument("--allow-gpu", action="store_true")
    call.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    from .broker import invoke, validate_request, write_evidence
    from .manifest import ToolError
    from .registry import get_tool, load_registry

    try:
        if args.action == "list":
            print(json.dumps(load_registry(), ensure_ascii=False, indent=2))
            return 0
        if args.action == "refresh-binding":
            from .scaffold import refresh_binding
            print(json.dumps(refresh_binding(args.registry), ensure_ascii=False, indent=2))
            return 0
        if args.action == "scaffold-proposal":
            from .scaffold import scaffold_proposal
            proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
            print(json.dumps(scaffold_proposal(proposal, args.out), ensure_ascii=False, indent=2))
            return 0
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        manifest = get_tool(str(request.get("tool_id") or ""))
        validate_request(manifest, request, session_role=args.session_role)
        if args.action == "check":
            print(json.dumps({"valid": True, "tool_id": manifest.tool_id, "manifest_sha256": manifest.digest, "gpu_used": False}, ensure_ascii=False))
            return 0
        if not args.allow_gpu:
            print("refusing to invoke a GPU tool without --allow-gpu; use `check` for validation only", file=sys.stderr)
            return 5
        result = invoke(request, session_role=args.session_role, timeout_s=args.timeout)
        path = write_evidence(Path(args.run_dir), request, result)
        print(json.dumps({"evidence": str(path), "result": result}, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ok" else 1
    except (ToolError, OSError, json.JSONDecodeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 5


if __name__ == "__main__":
    raise SystemExit(main())
