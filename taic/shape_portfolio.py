"""P2-D: portfolio-specific checkpoint/resume entry (never the legacy taic resume path)."""

from __future__ import annotations

from pathlib import Path

from tcenv.contract_errors import TaskContractError
from taic.portfolio_store import PortfolioRun
from tcenv.portfolio_validation.validation import PortfolioFreeze, ValidationPlan
from .portfolio_manifest import ActivePortfolioPointer, DispatchManifest


def write_checkpoint(
    run_dir: str | Path,
    *,
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    manifest: DispatchManifest,
    run_config_digest: str,
) -> PortfolioRun:
    """@brief 原子写 sidecar 与 active pointer；已确认 batches 不会在 resume 重跑。"""
    if PortfolioRun.read_revocation_marker(run_dir) is not None:
        raise TaskContractError("cannot checkpoint over a revoked portfolio generation")
    run = PortfolioRun.from_manifest(freeze, validation, manifest, run_config_digest=run_config_digest)
    sidecar = Path(run_dir) / "portfolio_run.json"
    if sidecar.exists():
        from tcenv.identity import IdentityError, load_strict_json

        try:
            existing = PortfolioRun.from_document(load_strict_json(sidecar))
        except IdentityError as exc:
            raise TaskContractError(f"cannot read existing portfolio_run.json: {exc}") from exc
        if existing.status == "revoked":
            raise TaskContractError("cannot checkpoint over a revoked portfolio generation")
        if manifest.activation_generation < existing.activation_generation:
            raise TaskContractError("cannot roll a portfolio checkpoint back to an older generation")
        if (
            existing.run_config_digest != run_config_digest
            or existing.dispatch_manifest_digest != manifest.digest
        ):
            raise TaskContractError("cannot change an existing active checkpoint's run_config or manifest")
    run.write_atomic(run_dir)
    ActivePortfolioPointer(manifest.digest, manifest.activation_generation).write_atomic(
        Path(run_dir) / "active_portfolio_pointer.json"
    )
    return run


def resume_checkpoint(
    run_dir: str | Path,
    *,
    freeze: PortfolioFreeze,
    validation: ValidationPlan,
    manifest: DispatchManifest,
    run_config_digest: str,
) -> tuple[str, ...]:
    """@brief 读取严格 checkpoint，返回已完成 instance_keys（调用方不得重复确认它们）。"""
    if PortfolioRun.read_revocation_marker(run_dir) is not None:
        raise TaskContractError("a revoked portfolio_run cannot resume execution")
    run = PortfolioRun.read(
        run_dir,
        freeze=freeze,
        validation=validation,
        manifest=manifest,
        minimum_generation=manifest.activation_generation,
        run_config_digest=run_config_digest,
    )
    ActivePortfolioPointer.read(Path(run_dir) / "active_portfolio_pointer.json", manifest)
    return tuple(sorted(subject.instance_key for subject in run.subjects))


def revoke_checkpoint(run_dir: str | Path, *, reason: str) -> PortfolioRun:
    """@brief 写入带 reason 的失效账本；旧 pointer 留作 stale 证据，不能恢复执行。"""
    from tcenv.identity import load_strict_json

    if PortfolioRun.read_revocation_marker(run_dir) is not None:
        raise TaskContractError("portfolio generation is already revoked")
    path = Path(run_dir) / "portfolio_run.json"
    document = load_strict_json(path)
    run = PortfolioRun.from_document(document).revoke(reason=reason)
    run.write_revocation_marker(run_dir)
    run.write_atomic(run_dir)
    return run


__all__ = ["resume_checkpoint", "revoke_checkpoint", "write_checkpoint"]
