"""Explicit portfolio-run sidecar reader for observe (no SQLite migration)."""

from __future__ import annotations

from pathlib import Path

from tcenv.contract_errors import TaskContractError
from tcenv.identity import IdentityError, load_strict_json
from taic.portfolio_store import PortfolioRun


def read_shape_portfolio(run_dir: str | Path) -> dict:
    """@brief 读取 per-shard provider/credit/coverage/invalidation，不读 run_config 猜状态。"""
    marker = PortfolioRun.read_revocation_marker(run_dir)
    if marker is not None:
        run = marker
    else:
        try:
            run = PortfolioRun.from_document(load_strict_json(Path(run_dir) / "portfolio_run.json"))
        except (IdentityError, OSError) as exc:
            raise TaskContractError(f"cannot observe portfolio_run.json: {exc}") from exc
    return {
        "schema_id": "observe.shape-portfolio.1",
        "portfolio_freeze_digest": run.portfolio_freeze_digest,
        "validation_plan_digest": run.validation_plan_digest,
        "dispatch_manifest_digest": run.dispatch_manifest_digest,
        "activation_generation": run.activation_generation,
        "status": run.status,
        "coverage_kind": "full_finite",
        "subjects": [
            {
                "instance_key": subject.instance_key,
                "provider": subject.provider,
                "variant_id": subject.variant_id,
                "fallback_id": subject.fallback_id,
                "credits_variant": subject.credits_variant,
                "coverage_kind": subject.coverage_kind,
            }
            for subject in run.subjects
        ],
        "invalidated_artifacts": [dict(record) for record in run.invalidated_artifacts],
    }


__all__ = ["read_shape_portfolio"]
