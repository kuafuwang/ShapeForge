"""Disposable test PKI generated with the system ``openssl`` CLI.

Only tests use this helper.  Production certificates and grants are provisioned
by ``scripts/remote_evaluator_admin.py`` and the runbook; no private key is
committed.  Everything is created under a pytest temporary directory with
0600 key files.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class PkiUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class Pki:
    root: Path
    ca_cert: Path
    ca_key: Path
    server_cert: Path
    server_key: Path
    client_cert: Path
    client_key: Path
    other_ca_cert: Path
    other_client_cert: Path
    other_client_key: Path

    def keyring_file(self, *, key_id: str, key: bytes, revoked: tuple[str, ...] = ()) -> Path:
        path = self.root / "keyring.json"
        path.write_text(json.dumps({
            "schema_version": 1,
            "keys": {key_id: key.hex()},
            "revoked": list(revoked),
            "active_key_id": key_id,
        }), encoding="utf-8")
        path.chmod(0o600)
        return path


def openssl() -> str:
    path = shutil.which("openssl")
    if not path:
        raise PkiUnavailable("openssl is required to generate disposable test certificates")
    return path


def _run(*arguments: str) -> None:
    completed = subprocess.run(
        [openssl(), *arguments], text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        raise PkiUnavailable(
            f"openssl {arguments[0]} failed: {completed.stdout}{completed.stderr}"
        )


def _write(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(mode)


def _config(path: Path, common_name: str, extension: str, body: str) -> Path:
    _write(path, f"""[req]
distinguished_name = dn
prompt = no
x509_extensions = {extension}
[dn]
CN = {common_name}
[{extension}]
{body}
""")
    return path


def make_pki(directory: Path, *, san: str = "DNS:localhost,IP:127.0.0.1") -> Pki:
    """Create a CA, a server certificate and trusted/untrusted client certificates."""

    directory.mkdir(parents=True, exist_ok=True)
    root = directory
    ca_cert, ca_key = root / "ca.pem", root / "ca.key"
    ca_config = _config(
        root / "ca.cnf", "tcenv test CA", "v3_ca",
        "basicConstraints = critical,CA:TRUE\n"
        "keyUsage = critical,keyCertSign,cRLSign\n"
        "subjectKeyIdentifier = hash\n",
    )
    _run(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
        "-config", str(ca_config), "-keyout", str(ca_key), "-out", str(ca_cert),
    )

    def issued(name: str, *, client: bool, ca: Path = ca_cert, signing_key: Path = ca_key) -> tuple[Path, Path]:
        key, csr, cert = root / f"{name}.key", root / f"{name}.csr", root / f"{name}.pem"
        _run("req", "-newkey", "rsa:2048", "-nodes", "-config", str(ca_config), "-keyout", str(key), "-out", str(csr))
        usage = (
            "basicConstraints = CA:FALSE\n"
            "keyUsage = critical,digitalSignature\n"
            "extendedKeyUsage = clientAuth\n"
            "subjectKeyIdentifier = hash\n"
        ) if client else (
            "basicConstraints = CA:FALSE\n"
            "keyUsage = critical,digitalSignature,keyEncipherment\n"
            "extendedKeyUsage = serverAuth\n"
            f"subjectAltName = {san}\n"
            "subjectKeyIdentifier = hash\n"
        )
        extension = root / f"{name}.ext"
        _write(extension, usage)
        _run(
            "x509", "-req", "-in", str(csr), "-CA", str(ca),
            "-CAkey", str(signing_key), "-CAcreateserial", "-days", "2",
            "-extfile", str(extension), "-out", str(cert),
        )
        for path in (key, csr):
            path.chmod(0o600)
        return cert, key

    server_cert, server_key = issued("server", client=False)
    client_cert, client_key = issued("client", client=True)

    other_ca_cert, other_ca_key = root / "other-ca.pem", root / "other-ca.key"
    other_config = _config(
        root / "other-ca.cnf", "tcenv other CA", "v3_ca",
        "basicConstraints = critical,CA:TRUE\n"
        "keyUsage = critical,keyCertSign,cRLSign\n"
        "subjectKeyIdentifier = hash\n",
    )
    _run(
        "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "2",
        "-config", str(other_config), "-keyout", str(other_ca_key), "-out", str(other_ca_cert),
    )
    other_client_cert, other_client_key = issued(
        "other-client", client=True, ca=other_ca_cert, signing_key=other_ca_key,
    )

    for path in (ca_key, ca_cert, other_ca_key, other_ca_cert, server_cert, client_cert,
                 other_client_cert, server_key, client_key, other_client_key):
        path.chmod(0o600)
    return Pki(
        root=root,
        ca_cert=ca_cert,
        ca_key=ca_key,
        server_cert=server_cert,
        server_key=server_key,
        client_cert=client_cert,
        client_key=client_key,
        other_ca_cert=other_ca_cert,
        other_client_cert=other_client_cert,
        other_client_key=other_client_key,
    )
