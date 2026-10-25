#!/usr/bin/env python3
"""Provision and rotate external-evaluator certificates, grants and HMAC keys.

Every generated secret file is written with mode 0600 and no command prints raw
tokens or key material.  Certificate commands use the system ``openssl`` CLI;
the keyring and grant commands use only the standard library.

Examples::

    python scripts/remote_evaluator_admin.py keyring-init \
        --out /etc/tcenv/keyring.json --key-id npu-node-01/key-1
    python scripts/remote_evaluator_admin.py keyring-rotate \
        --file /etc/tcenv/keyring.json --key-id npu-node-01/key-2
    python scripts/remote_evaluator_admin.py grant-mint \
        --out /etc/tcenv/grants.json --append \
        --token-out /etc/tcenv/grants/run-1-session-1.token \
        --run-id run-1 --session-id session-1 --target-id npu-node-01 \
        --subject host-evaluator --kinds formal_eval,selfcheck --expires-in 3600
    python scripts/remote_evaluator_admin.py ca-init --out /etc/tcenv/pki
    python scripts/remote_evaluator_admin.py cert-issue --ca /etc/tcenv/pki \
        --name evaluator --kind server --san DNS:evaluator.example.internal
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tcenv.remote.grants import load_grants_file, write_grants_file  # noqa: E402
from tcenv.remote.identity import SessionGrant  # noqa: E402
from tcenv.remote.keyring import (  # noqa: E402
    KeyringError,
    KeyringFile,
    load_keyring_file,
    new_key,
    write_keyring_file,
)


def _load_raw_keyring(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read keyring: {exc}") from exc


def _write_private(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o600)


def _openssl() -> str:
    binary = shutil.which("openssl")
    if not binary:
        raise SystemExit("openssl is required for certificate commands")
    return binary


def _run_openssl(arguments: list[str]) -> None:
    completed = subprocess.run([_openssl(), *arguments], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"openssl failed: {completed.stdout}{completed.stderr}")


def _ca_config(path: Path, common_name: str, extension: str, body: str) -> Path:
    _write_private(path, f"""[req]
distinguished_name = dn
prompt = no
x509_extensions = {extension}
[dn]
CN = {common_name}
[{extension}]
{body}
""")
    return path


def command_keyring_init(args: argparse.Namespace) -> int:
    out = Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out} without --force")
    keyring = KeyringFile({args.key_id: new_key()}, frozenset(), args.key_id)
    write_keyring_file(out, keyring)
    print(f"wrote keyring {out} key_id={args.key_id}")
    return 0


def command_keyring_rotate(args: argparse.Namespace) -> int:
    path = Path(args.file)
    document = _load_raw_keyring(path)
    keys = {key_id: bytes.fromhex(value) for key_id, value in document["keys"].items()}
    revoked = frozenset(document.get("revoked", []))
    if args.key_id in keys:
        raise SystemExit(f"key id {args.key_id!r} already exists")
    keys[args.key_id] = new_key()
    write_keyring_file(path, KeyringFile(keys, revoked, args.key_id))
    print(f"rotated keyring {path} active_key_id={args.key_id}")
    return 0


def command_keyring_revoke(args: argparse.Namespace) -> int:
    path = Path(args.file)
    keyring = load_keyring_file(path)
    if args.key_id not in keyring.keys:
        raise SystemExit(f"key id {args.key_id!r} is not in the keyring")
    revoked = KeyringFile(
        {key_id: key for key_id, key in keyring.keys.items() if key_id != args.key_id},
        keyring.revoked | frozenset({args.key_id}),
        keyring.active_key_id if keyring.active_key_id != args.key_id else "",
    )
    active = args.new_active or revoked.active_key_id
    if not active:
        raise SystemExit("revoking the active key requires --new-active with another trusted key")
    if active not in revoked.trusted:
        raise SystemExit("--new-active must name a trusted, non-revoked key")
    write_keyring_file(path, KeyringFile(revoked.keys, revoked.revoked, active))
    print(f"revoked key_id={args.key_id} keyring={path} active_key_id={active}")
    return 0


def command_keyring_show(args: argparse.Namespace) -> int:
    keyring = load_keyring_file(args.file)
    print(json.dumps({
        "schema_version": 1,
        "key_ids": sorted(keyring.keys),
        "revoked": sorted(keyring.revoked),
        "active_key_id": keyring.active_key_id,
    }, indent=2, sort_keys=True))
    return 0


def command_grant_mint(args: argparse.Namespace) -> int:
    expires_at = None
    if args.expires_in:
        expires_at = datetime.now(timezone.utc).timestamp() + args.expires_in
    kinds = frozenset(part.strip() for part in args.kinds.split(",") if part.strip())
    modes = frozenset(part.strip() for part in (args.modes or "").split(",") if part.strip())
    grant = SessionGrant(
        token=secrets.token_urlsafe(32),
        run_id=args.run_id,
        session_id=args.session_id,
        target_id=args.target_id,
        allowed_kinds=kinds,
        allowed_modes=modes,
        subject=args.subject,
        max_timed=args.max_timed,
        max_gpu_s=args.max_gpu_s,
        expires_at=expires_at,
    )
    out = Path(args.out)
    if out.exists() and not args.append:
        raise SystemExit(f"refusing to overwrite {out} without --append")
    grants = load_grants_file(out) if out.exists() and args.append else []
    if any(existing.run_id == grant.run_id and existing.session_id == grant.session_id and existing.subject == grant.subject for existing in grants):
        raise SystemExit("a grant for this run/session/subject already exists; revoke it explicitly first")
    grants.append(grant)
    write_grants_file(out, grants)
    token_path = Path(args.token_out) if args.token_out else out.with_suffix(".token")
    _write_private(token_path, grant.token)
    print(
        f"minted grant run_id={grant.run_id} session_id={grant.session_id} "
        f"subject={grant.subject or '<derived>'} token_file={token_path} grants_file={out}"
    )
    return 0


def command_grants_show(args: argparse.Namespace) -> int:
    grants = load_grants_file(args.file)
    rows = [
        {
            "run_id": grant.run_id, "session_id": grant.session_id, "target_id": grant.target_id,
            "kinds": sorted(grant.allowed_kinds), "modes": sorted(grant.allowed_modes),
            "subject": grant.subject, "max_timed": grant.max_timed, "max_gpu_s": grant.max_gpu_s,
            "expires_at": None if grant.expires_at is None else datetime.fromtimestamp(grant.expires_at, timezone.utc).isoformat(),
        }
        for grant in grants
    ]
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


def command_ca_init(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    cert, key = out / "ca.pem", out / "ca.key"
    if cert.exists() or key.exists():
        raise SystemExit(f"{out} already contains a CA")
    config = _ca_config(
        out / "ca.cnf", args.common_name, "v3_ca",
        "basicConstraints = critical,CA:TRUE\n"
        "keyUsage = critical,keyCertSign,cRLSign\n"
        "subjectKeyIdentifier = hash\n",
    )
    _run_openssl([
        "req", "-x509", "-newkey", "rsa:3072", "-nodes", "-days", str(args.days),
        "-config", str(config), "-keyout", str(key), "-out", str(cert),
    ])
    key.chmod(0o600)
    cert.chmod(0o600)
    print(f"created CA {cert} (private key {key})")
    return 0


def command_cert_issue(args: argparse.Namespace) -> int:
    ca = Path(args.ca)
    ca_cert, ca_key = ca / "ca.pem", ca / "ca.key"
    if not ca_cert.is_file() or not ca_key.is_file():
        raise SystemExit(f"{ca} does not contain ca.pem and ca.key")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    key, csr, cert = out / f"{args.name}.key", out / f"{args.name}.csr", out / f"{args.name}.pem"
    config = _ca_config(out / f"{args.name}.cnf", args.name, "v3_ca", "basicConstraints = critical,CA:TRUE\n")
    _run_openssl([
        "req", "-newkey", "rsa:2048", "-nodes", "-config", str(config),
        "-keyout", str(key), "-out", str(csr),
    ])
    if args.kind == "server":
        body = (
            "basicConstraints = CA:FALSE\n"
            "keyUsage = critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage = serverAuth\n"
            f"subjectAltName = {args.san}\n"
            "subjectKeyIdentifier = hash\n"
        )
    else:
        body = (
            "basicConstraints = CA:FALSE\n"
            "keyUsage = critical,digitalSignature\n"
            "extendedKeyUsage = clientAuth\n"
            "subjectKeyIdentifier = hash\n"
        )
    extension = out / f"{args.name}.ext"
    _write_private(extension, body)
    _run_openssl([
        "x509", "-req", "-in", str(csr), "-CA", str(ca_cert), "-CAkey", str(ca_key),
        "-CAcreateserial", "-days", str(args.days), "-extfile", str(extension), "-out", str(cert),
    ])
    key.chmod(0o600)
    cert.chmod(0o600)
    csr.unlink(missing_ok=True)
    config.unlink(missing_ok=True)
    print(f"issued {args.kind} certificate {cert} (private key {key})")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="remote_evaluator_admin")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("keyring-init", help="create a signing/verifying keyring")
    init.add_argument("--out", required=True, type=Path)
    init.add_argument("--key-id", required=True)
    init.add_argument("--force", action="store_true")
    init.set_defaults(handler=command_keyring_init)

    rotate = sub.add_parser("keyring-rotate", help="add a new active key, keeping old keys")
    rotate.add_argument("--file", required=True, type=Path)
    rotate.add_argument("--key-id", required=True)
    rotate.set_defaults(handler=command_keyring_rotate)

    revoke = sub.add_parser("keyring-revoke", help="remove a key and record the revocation")
    revoke.add_argument("--file", required=True, type=Path)
    revoke.add_argument("--key-id", required=True)
    revoke.add_argument("--new-active")
    revoke.set_defaults(handler=command_keyring_revoke)

    show = sub.add_parser("keyring-show", help="print key ids and revocation state")
    show.add_argument("--file", required=True, type=Path)
    show.set_defaults(handler=command_keyring_show)

    grant = sub.add_parser("grant-mint", help="mint a session grant and its host token file")
    grant.add_argument("--out", required=True, type=Path, help="server grants file")
    grant.add_argument("--append", action="store_true")
    grant.add_argument("--token-out", type=Path)
    grant.add_argument("--run-id", required=True)
    grant.add_argument("--session-id", required=True)
    grant.add_argument("--target-id", required=True)
    grant.add_argument("--subject", default="host-evaluator")
    grant.add_argument("--kinds", default="formal_eval,selfcheck")
    grant.add_argument("--modes", default="")
    grant.add_argument("--max-timed", type=int, default=20)
    grant.add_argument("--max-gpu-s", type=float, default=120.0)
    grant.add_argument("--expires-in", type=int, default=3600, help="seconds; 0 means no expiry")
    grant.set_defaults(handler=command_grant_mint)

    grants = sub.add_parser("grants-show", help="print grants without tokens")
    grants.add_argument("--file", required=True, type=Path)
    grants.set_defaults(handler=command_grants_show)

    ca = sub.add_parser("ca-init", help="create a private test/deployment CA with openssl")
    ca.add_argument("--out", required=True, type=Path)
    ca.add_argument("--common-name", default="tcenv evaluator CA")
    ca.add_argument("--days", type=int, default=30)
    ca.set_defaults(handler=command_ca_init)

    issue = sub.add_parser("cert-issue", help="issue a server or client certificate")
    issue.add_argument("--ca", required=True, type=Path)
    issue.add_argument("--out", required=True, type=Path)
    issue.add_argument("--name", required=True)
    issue.add_argument("--kind", choices=["server", "client"], required=True)
    issue.add_argument("--san", default="DNS:localhost,IP:127.0.0.1")
    issue.add_argument("--days", type=int, default=30)
    issue.set_defaults(handler=command_cert_issue)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyringError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
