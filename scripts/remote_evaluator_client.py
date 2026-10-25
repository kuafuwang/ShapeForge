#!/usr/bin/env python3
"""External Python client for the ``tcenv.remote.v1`` evaluator.

The client holds only host-scoped credentials taken from files: a shared mTLS
identity, a short-lived session grant token and the HMAC verifier keyring.  It
never prints the token or key material.

Examples::

    python scripts/remote_evaluator_client.py capabilities \
        --endpoint evaluator.example.internal:7443 --ca ca.pem \
        --client-cert host.pem --client-key host.key --server-name evaluator.example.internal \
        --token-file run-1.token --keyring keyring.json --target-id npu-node-01

    python scripts/remote_evaluator_client.py conformance \
        --endpoint ... --source-file candidate.metal --task relu --dtype fp16 \
        --size small --fingerprint <64 hex> --run-id run-1 --session-id s1 \
        --request-id conf-1 --cancel-source-file sleeper.metal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcenv.remote import contracts  # noqa: E402
from tcenv.remote.conformance import conformance_request, result_summary, run_conformance  # noqa: E402
from tcenv.remote.grpc_client import (  # noqa: E402
    DEFAULT_REQUIRED_FEATURES,
    GrpcEvaluatorFactory,
    GrpcExpected,
    grpc_evaluator_factory_from_config,
)


def _token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise SystemExit(f"cannot read token file: {exc}") from exc
    if not token:
        raise SystemExit("token file is empty")
    return token


def _request(args: argparse.Namespace, *, request_id: str):
    try:
        source = Path(args.source_file).read_bytes()
    except OSError as exc:
        raise SystemExit(f"cannot read source file: {exc}") from exc
    if not source:
        raise SystemExit("source file is empty")
    try:
        return conformance_request(
            request_id=request_id,
            run_id=args.run_id,
            session_id=args.session_id,
            task=args.task,
            dtype=args.dtype,
            size=args.size,
            target_id=args.target_id,
            task_fingerprint=args.fingerprint,
            source=source,
            mode=args.mode,
            kind=args.kind,
            seed=args.seed,
            reps=args.reps,
            queue_timeout_ms=args.queue_timeout_ms,
            candidate_timeout_ms=args.candidate_timeout_ms,
        )
    except (ValueError, UnicodeDecodeError) as exc:
        raise SystemExit(f"invalid request: {exc}") from exc


def _factory(args: argparse.Namespace) -> GrpcEvaluatorFactory:
    return grpc_evaluator_factory_from_config(
        endpoint=args.endpoint,
        token_provider=lambda _run, _session: _token(args.token_file),
        run_id=args.run_id,
        target_id=args.target_id,
        ca_file=args.ca,
        certificate_file=args.client_cert,
        private_key_file=args.client_key,
        keyring_file=args.keyring,
        expected=GrpcExpected(
            site_id=args.expected_site_id,
            device_id=args.expected_device_id,
            target_id=args.target_id,
            worker_build=args.expected_worker_build,
            task_pack_digest=args.expected_task_pack_digest,
            evaluator_epoch=args.expected_epoch,
        ),
        api_contract_ids=(contracts.API_CONTRACT_ID,),
        evidence_contract_ids=(contracts.EVIDENCE_CONTRACT_ID,),
        required_features=tuple(args.required_features),
        server_name_override=args.server_name,
        rpc_timeout_s=args.timeout,
        watch_slice_s=args.watch_slice,
    )


def _add_transport(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--ca", required=True, type=Path)
    parser.add_argument("--client-cert", required=True, type=Path)
    parser.add_argument("--client-key", required=True, type=Path)
    parser.add_argument("--server-name", default="")
    parser.add_argument("--token-file", required=True, type=Path)
    parser.add_argument("--keyring", required=True, type=Path)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--expected-site-id", required=True)
    parser.add_argument("--expected-device-id", required=True)
    parser.add_argument("--expected-worker-build", default="")
    parser.add_argument("--expected-task-pack-digest", default="")
    parser.add_argument("--expected-epoch", default="")
    parser.add_argument("--required-features", nargs="*", default=list(DEFAULT_REQUIRED_FEATURES))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--watch-slice", type=float, default=30.0)


def _add_request(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-file", required=True, type=Path)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--size", default="small")
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--kind", choices=["formal_eval", "selfcheck"], default="formal_eval")
    parser.add_argument("--mode", default="quick")
    parser.add_argument("--seed", default="3")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--queue-timeout-ms", type=int, default=30_000)
    parser.add_argument("--candidate-timeout-ms", type=int, default=120_000)
    parser.add_argument("--request-id", default="conformance-1")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remote_evaluator_client")
    sub = parser.add_subparsers(dest="command", required=True)

    capabilities = sub.add_parser("capabilities", help="authenticated GetCapabilities only")
    _add_transport(capabilities)

    submit = sub.add_parser("submit", help="submit, watch and verify one job")
    _add_transport(submit)
    _add_request(submit)

    conformance = sub.add_parser("conformance", help="run the wire conformance flow")
    _add_transport(conformance)
    _add_request(conformance)
    conformance.add_argument(
        "--retry-source-file", required=True, type=Path,
        help="candidate bytes whose first attempt the evaluator interrupts",
    )
    conformance.add_argument("--retry-request-id", default="conformance-retry-1")
    conformance.add_argument("--cancel-source-file", type=Path)
    conformance.add_argument("--cancel-request-id", default="conformance-cancel-1")

    reattach = sub.add_parser(
        "reattach", help="replay an accepted job (grant renewal / key rotation phase)",
    )
    _add_transport(reattach)
    _add_request(reattach)
    reattach.add_argument("--expect-attempt-id", default="")
    reattach.add_argument("--expect-key-id", default="")
    reattach.add_argument(
        "--revoke-key", action="append", default=[],
        help="revoke this key ID in the host keyring before verification",
    )
    return parser


def _request_from_paths(args: argparse.Namespace, request_id: str, source_file: Path):
    saved = args.source_file
    args.source_file = source_file
    try:
        return _request(args, request_id=request_id)
    finally:
        args.source_file = saved


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    factory = None
    try:
        factory = _factory(args)
        if args.command == "capabilities":
            epoch = factory.ensure_connected()
            metadata = factory.execution_metadata()
            print(json.dumps({
                "ok": True,
                "evaluator_epoch": epoch,
                "endpoint": args.endpoint,
                **metadata,
            }, indent=2, sort_keys=True))
            return 0
        request = _request(args, request_id=args.request_id)
        if args.command == "submit":
            result = factory.formal(request) if request.kind == "formal_eval" else factory.selfcheck(request)
            print(json.dumps({"ok": True, "result": result_summary(result)}, indent=2, sort_keys=True))
            return 0
        if args.command == "reattach":
            for key_id in args.revoke_key:
                factory.keyring.revoke(key_id)
            result = factory.formal(request) if request.kind == "formal_eval" else factory.selfcheck(request)
            summary = result_summary(result)
            failures = []
            if args.expect_attempt_id and summary["attempt_id"] != args.expect_attempt_id:
                failures.append("attempt id differs from the interrupted/replayed attempt")
            if args.expect_key_id and summary["signature_key_id"] != args.expect_key_id:
                failures.append("result signature key id differs from the expected key")
            print(json.dumps({"ok": not failures, "result": summary, "failures": failures}, indent=2, sort_keys=True))
            return 0 if not failures else 1
        retry_request = _request_from_paths(args, args.retry_request_id, args.retry_source_file)
        second = None
        if args.cancel_source_file is not None:
            second = _request_from_paths(args, args.cancel_request_id, args.cancel_source_file)
        token = _token(args.token_file)
        report = run_conformance(
            factory=factory,
            request=request,
            second_request=second,
            retry_request=retry_request,
            token=token,
            required_features=tuple(args.required_features),
            timeout_s=args.timeout,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["ok"] else 1
    except Exception as exc:  # noqa: BLE001 - the CLI reports a sanitized failure
        print(json.dumps({
            "ok": False,
            "error": type(exc).__name__,
            "code": getattr(exc, "code", ""),
            "message": str(exc)[:1000],
        }, indent=2))
        return 2
    finally:
        if factory is not None:
            factory.close()


if __name__ == "__main__":
    raise SystemExit(main())
