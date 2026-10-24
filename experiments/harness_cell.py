"""Run one experiment cell with this worktree's own Python import path."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("request")
    args = parser.parse_args()
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    from taic.committees import load_committees
    from taic.llm import load_config, load_raw_config
    from taic.orchestrator import Orchestrator

    config_path = request["config"]
    cfg, search = load_config(config_path, preset=None, **request.get("overrides", {}))
    raw = load_raw_config(config_path, None)
    committees = load_committees(raw, active=cfg)
    orch = Orchestrator(request["task"], request["dtype"], request["size"], cfg, search, run_dir=Path(request["run_dir"]), committees=committees, target_id=request["target_id"], seed=request["seed"])
    result = asyncio.run(orch.run())
    (Path(request["run_dir"]) / "cell_result.json").write_text(json.dumps({"spent_usd": result.spent_usd, "stop_reason": result.stop_reason, "run_seed": result.run_seed}), encoding="utf-8")
    print(json.dumps({"spent_usd": result.spent_usd, "stop_reason": result.stop_reason, "run_seed": result.run_seed}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
