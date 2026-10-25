"""Host-controlled HMAC verifier/signer keyring files.

The same document feeds the host verifier and the evaluator signer:

```json
{"schema_version": 1, "keys": {"site/key-1": "<64 hex chars>"},
 "revoked": ["site/key-0"], "active_key_id": "site/key-1"}
```

Normal rotation adds a key and moves ``active_key_id``; old keys stay for
historical results.  Revocation removes the key from ``keys`` and records it in
``revoked`` so neither side can silently trust it again.  Rotating keys never
changes the measurement epoch.
"""

from __future__ import annotations

import json
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from . import contracts

SCHEMA_VERSION = 1


class KeyringError(ValueError):
    """A keyring file is malformed, unreadable or missing its active key."""


@dataclass(frozen=True)
class KeyringFile:
    keys: dict[str, bytes]
    revoked: frozenset[str]
    active_key_id: str

    @property
    def trusted(self) -> dict[str, bytes]:
        return {key_id: key for key_id, key in self.keys.items() if key_id not in self.revoked}

    def require_active(self) -> tuple[str, bytes]:
        if self.active_key_id not in self.trusted:
            raise KeyringError("active signing key is missing or revoked")
        return self.active_key_id, self.trusted[self.active_key_id]

    def without(self, *key_ids: str) -> "KeyringFile":
        revoked = self.revoked | frozenset(key_ids)
        keys = {key_id: key for key_id, key in self.keys.items() if key_id not in revoked}
        active = self.active_key_id if self.active_key_id in keys else ""
        return KeyringFile(keys, revoked, active)


def _key_bytes(value: object, key_id: str) -> bytes:
    if isinstance(value, str):
        try:
            raw = bytes.fromhex(value)
        except ValueError as exc:
            raise KeyringError(f"key {key_id!r} must be lowercase hex text") from exc
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
    else:
        raise KeyringError("keyring values must be bytes or hex text")
    if len(raw) < 32:
        raise KeyringError("HMAC-SHA256 keys must contain at least 32 bytes")
    return raw


def _validate_key_id(key_id: object) -> str:
    if not isinstance(key_id, str):
        raise KeyringError("keyring key ids must be strings")
    try:
        return contracts.key_id(key_id)
    except contracts.ContractError as exc:
        raise KeyringError("keyring key id is invalid") from exc


def parse_keyring(document: object) -> KeyringFile:
    if not isinstance(document, dict):
        raise KeyringError("keyring must be an object")
    required = {"schema_version", "keys", "revoked"}
    unknown = set(document) - required - {"active_key_id"}
    if not required <= set(document) or unknown:
        raise KeyringError("keyring must contain schema_version, keys, revoked and optional active_key_id")
    if document["schema_version"] != SCHEMA_VERSION:
        raise KeyringError("keyring has an unsupported schema")
    keys_value = document["keys"]
    revoked_value = document["revoked"]
    if not isinstance(keys_value, dict) or not isinstance(revoked_value, list):
        raise KeyringError("keyring keys/revoked have the wrong shape")
    keys = {}
    for key_id, value in keys_value.items():
        validated = _validate_key_id(key_id)
        keys[validated] = _key_bytes(value, validated)
    revoked = frozenset(_validate_key_id(key_id) for key_id in revoked_value)
    active = document.get("active_key_id")
    if active is not None:
        active = _validate_key_id(active)
        if active not in keys or active in revoked:
            raise KeyringError("active_key_id must name a trusted, non-revoked key")
    else:
        active = ""
    if not keys:
        raise KeyringError("keyring has no keys")
    return KeyringFile(keys, revoked, active)


def load_keyring_file(path: Path | str) -> KeyringFile:
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise KeyringError(f"cannot read keyring: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise KeyringError("keyring must be a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise KeyringError("keyring must not be readable by group or others (mode 0600)")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise KeyringError(f"cannot read keyring: {exc}") from exc
    return parse_keyring(document)


def dump_keyring(keyring: KeyringFile, *, active_key_id: str | None = None) -> dict[str, object]:
    active = keyring.active_key_id if active_key_id is None else active_key_id
    if active:
        _validate_key_id(active)
        if active not in keyring.trusted:
            raise KeyringError("active_key_id must name a trusted, non-revoked key")
    return {
        "schema_version": SCHEMA_VERSION,
        "keys": {key_id: key.hex() for key_id, key in sorted(keyring.keys.items()) if key_id not in keyring.revoked},
        "revoked": sorted(keyring.revoked),
        "active_key_id": active,
    }


def write_keyring_file(path: Path | str, keyring: KeyringFile) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    document = dump_keyring(keyring)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def new_key() -> bytes:
    return secrets.token_bytes(32)
