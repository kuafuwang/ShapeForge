"""Per-certificate client revocation for the external evaluator frontend.

The Python gRPC TLS stack authenticates a client certificate chain against the
configured client CA bundle, but it neither fetches nor evaluates a CRL/OCSP
response.  A deployment therefore publishes a small, static revocation list of
**leaf-certificate** SHA-256 fingerprints: the standard "deny by certificate
identity" fallback for a stack without CRL/OCSP support.  It does not invent new
cryptography; the fingerprint is the SHA-256 of the DER certificate that the TLS
handshake already authenticated.

The frontend loads the list at startup (0600 regular file) and every RPC
compares the presented leaf certificate before the session grant is authorized:

```json
{"schema_version": 1,
 "revoked_client_certificates_sha256": ["<64 lowercase hex>", "..."]}
```

A malformed or group-readable file prevents the frontend from starting (fail
closed).  Rotation is a restart-loaded edit, like grant files and the HMAC
keyring, and never changes the measurement epoch.  The list is a *revocation*
mechanism only: an expired or not-yet-valid certificate is still the TLS
stack's decision.
"""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any, Mapping

import grpc

SCHEMA_VERSION = 1
_HEX = set("0123456789abcdef")
_PEM_MARKER = b"-----BEGIN CERTIFICATE-----"
_PEM_END = b"-----END CERTIFICATE-----"


class RevocationError(ValueError):
    """A revocation list is malformed, unreadable or uses the wrong shape."""


def normalize_fingerprint(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(character not in _HEX for character in value):
        raise RevocationError("a revoked certificate fingerprint must be 64 lowercase hex characters")
    return value


def normalize_fingerprints(values: Any) -> frozenset[str]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (set, frozenset, list, tuple)):
        raise RevocationError("revoked client certificates must be a collection of fingerprints")
    return frozenset(normalize_fingerprint(value) for value in values)


def _fingerprint(value: str) -> str:
    return normalize_fingerprint(value)


def parse_revocation_list(document: object) -> frozenset[str]:
    if not isinstance(document, Mapping) or set(document) != {"schema_version", "revoked_client_certificates_sha256"}:
        raise RevocationError(
            "revocation list must contain exactly schema_version and revoked_client_certificates_sha256"
        )
    if document["schema_version"] != SCHEMA_VERSION:
        raise RevocationError("revocation list has an unsupported schema")
    values = document["revoked_client_certificates_sha256"]
    if not isinstance(values, list):
        raise RevocationError("revoked_client_certificates_sha256 must be a list")
    return frozenset(_fingerprint(value) for value in values)


def load_revocation_file(path: Path | str) -> frozenset[str]:
    path = Path(path)
    try:
        info = path.stat()
    except OSError as exc:
        raise RevocationError(f"cannot read revocation list: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise RevocationError("revocation list must be a regular file")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RevocationError("revocation list must not be readable by group or others (mode 0600)")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RevocationError(f"cannot read revocation list: {exc}") from exc
    return parse_revocation_list(document)


def write_revocation_file(path: Path | str, fingerprints: frozenset[str]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    document = {
        "schema_version": SCHEMA_VERSION,
        "revoked_client_certificates_sha256": sorted(_fingerprint(value) for value in fingerprints),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def certificate_sha256(pem: bytes | bytearray | memoryview | str) -> str:
    """Return the SHA-256 of the DER bytes of one PEM certificate."""

    if isinstance(pem, str):
        data = pem.encode("ascii", "strict")
    elif isinstance(pem, (bytes, bytearray, memoryview)):
        data = bytes(pem)
    else:
        raise RevocationError("certificate must be PEM bytes or text")
    start = data.find(_PEM_MARKER)
    end = data.find(_PEM_END, start + len(_PEM_MARKER)) if start >= 0 else -1
    if start < 0 or end < 0:
        raise RevocationError("peer identity is not a PEM certificate")
    block = data[start:end + len(_PEM_END)] + b"\n"
    try:
        import ssl

        der = ssl.PEM_cert_to_DER_cert(block.decode("ascii"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise RevocationError("peer certificate is not valid PEM") from exc
    return hashlib.sha256(der).hexdigest()


def peer_certificate_sha256(context: grpc.ServicerContext) -> str | None:
    """Return the authenticated peer leaf fingerprint, or ``None`` without a peer cert."""

    try:
        auth = context.auth_context()
    except Exception:  # pragma: no cover - defensive for non-TLS contexts
        return None
    if not isinstance(auth, Mapping):
        return None
    for key in ("x509_pem_cert", "x509_pem_cert_chain"):
        values = auth.get(key)
        if not values:
            continue
        value: Any = values[0]
        if isinstance(value, str):
            value = value.encode("utf-8")
        try:
            return certificate_sha256(value)
        except RevocationError:
            return None
    return None


def peer_certificate_present(context: grpc.ServicerContext) -> bool:
    try:
        auth = context.auth_context()
    except Exception:  # pragma: no cover - defensive for non-TLS contexts
        return False
    if not isinstance(auth, Mapping):
        return False
    return bool(auth.get("x509_pem_cert") or auth.get("x509_pem_cert_chain"))
