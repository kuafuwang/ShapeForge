"""Offline durable-state backup, restore, inspection and quarantine release.

Run these commands while the gRPC frontend is stopped: opening the store takes
the single-writer owner lock and protects the WAL snapshot.  The restored
directory carries ``restored.marker``; every nonterminal job stays quarantined
until an operator inspects it and releases it explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .job_store import JobStore
from .keyring import load_keyring_file


def _store(path: Path, keyring_path: Path) -> JobStore:
    keyring = load_keyring_file(keyring_path)
    active_key_id, active_key = keyring.require_active()
    return JobStore(
        path, signing_key=active_key, signing_key_id=active_key_id,
        signing_keyring=keyring.trusted,
    )


def command_backup(args: argparse.Namespace) -> int:
    store = _store(args.state_dir, args.keyring_file)
    try:
        destination = store.backup(args.out)
    finally:
        store.close()
    print(f"backup written to {destination}")
    return 0


def command_restore(args: argparse.Namespace) -> int:
    target = JobStore.restore_backup(args.backup, args.state_dir)
    print(f"restored backup into {target.parent}; nonterminal jobs are quarantined")
    return 0


def command_inspect(args: argparse.Namespace) -> int:
    store = _store(args.state_dir, args.keyring_file)
    try:
        rows = store._conn.execute(
            "SELECT state, COUNT(*) FROM jobs GROUP BY state ORDER BY state"
        ).fetchall()
        quarantined = store._conn.execute(
            "SELECT request_id,state,recovery_error FROM jobs WHERE quarantined=1 ORDER BY request_id"
        ).fetchall()
        document = {
            "schema_version": 1,
            "frontend_instance": store.frontend_instance,
            "event_window": store.event_window,
            "retention_ms": store.retention_ms,
            "states": {row[0]: row[1] for row in rows},
            "quarantined": [
                {"request_id": row[0], "state": row[1], "recovery_error": row[2]}
                for row in quarantined
            ],
        }
    finally:
        store.close()
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


def command_release(args: argparse.Namespace) -> int:
    store = _store(args.state_dir, args.keyring_file)
    try:
        snapshot = store.release_quarantine(args.request_id)
    finally:
        store.close()
    print(f"released {args.request_id} state={snapshot.state} quarantined={snapshot.quarantined}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m tcenv.remote.state_admin")
    sub = parser.add_subparsers(dest="command", required=True)

    backup = sub.add_parser("backup", help="consistent offline SQLite backup")
    backup.add_argument("--state-dir", required=True, type=Path)
    backup.add_argument("--out", required=True, type=Path)
    backup.add_argument("--keyring-file", required=True, type=Path)
    backup.set_defaults(handler=command_backup)

    restore = sub.add_parser("restore", help="restore a backup into an empty state dir")
    restore.add_argument("--backup", required=True, type=Path)
    restore.add_argument("--state-dir", required=True, type=Path)
    restore.set_defaults(handler=command_restore)

    inspect = sub.add_parser("inspect", help="print job counts and quarantined jobs")
    inspect.add_argument("--state-dir", required=True, type=Path)
    inspect.add_argument("--keyring-file", required=True, type=Path)
    inspect.set_defaults(handler=command_inspect)

    release = sub.add_parser("release-quarantine", help="explicitly release one restored job")
    release.add_argument("--state-dir", required=True, type=Path)
    release.add_argument("--keyring-file", required=True, type=Path)
    release.add_argument("--request-id", required=True)
    release.set_defaults(handler=command_release)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except Exception as exc:  # noqa: BLE001 - operator CLI reports a bounded error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
