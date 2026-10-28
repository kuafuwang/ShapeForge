"""Versioned multi-artifact bundle (NPU P1 contract, document version 4).

A bundle is a manifest of named, typed artifacts plus their exact bytes.  The
bytes never enter the canonical digest document: an entry carries a *relative
POSIX path*, a role, a media type, the exact byte count and the SHA-256 of the
exact bytes.  ``artifact_set_digest`` is
``SHA256(RFC8785(canonical_manifest))`` over the schema id and the
path-sorted entry list, so the same artifact set has one stable identity
regardless of upload order, transport form or host locale.

Two things this module deliberately refuses:

* any path that is not a plain relative POSIX path (absolute, ``..``, ``.``,
  empty, backslash, control characters, duplicate or case-colliding paths);
* any byte-level normalization: the digest is over the exact bytes, so a CRLF
  rewrite, a Unicode normalization or a BOM insertion changes the identity
  instead of silently matching.

Small entries are inlined in the message envelope; entries above
``MAX_BUNDLE_INLINE_ENTRY_BYTES`` must use the independent content-addressed
blob channel (P0b) and carry only a descriptor.  Both forms share one manifest,
so a bundle never depends on a host path, a shared directory or a URL.
"""

from __future__ import annotations

import base64
import errno
import hashlib
import json
import os
import re
import stat
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Bundle schema identity.  It is part of the canonical manifest and therefore
#: part of ``artifact_set_digest``.
BUNDLE_SCHEMA_ID = "tcenv.remote.npu-bundle.artifact-bundle.1"
BUNDLE_SCHEMA_VERSION = 1
#: CandidateDigest algorithm for a bundle artifact.
BUNDLE_DIGEST_ALGORITHM = "sha256-bundle-v1"

BUNDLE_ROLES = frozenset({
    "SOURCE",
    "EXECUTABLE_MODULE",
    "GRAPH_DEFINITION",
    "MODEL_DEFINITION",
    "PARAMETER_TABLE",
    "AUXILIARY",
})

MAX_BUNDLE_ENTRIES = 64
MAX_BUNDLE_PATH_BYTES = 240
MAX_BUNDLE_PATH_SEGMENTS = 16
#: Inline bytes are bounded so a bundle stays a control-plane message.  Bigger
#: artifacts must use the content-addressed blob channel.
MAX_BUNDLE_INLINE_ENTRY_BYTES = 262_144
MAX_BUNDLE_INLINE_BYTES = 1_000_000
#: Declared total, inline plus blob-backed entries.
MAX_BUNDLE_TOTAL_BYTES = 16_000_000

_HEX64 = re.compile(r"[0-9a-f]{64}")
_MEDIA_TYPE = re.compile(r"[A-Za-z0-9!#$&^_.+*/;= -]{1,160}")
_MANIFEST_FIELDS = ("path", "role", "media_type", "size_bytes", "sha256")


class BundleError(ValueError):
    """A bundle manifest, entry or exact bytes are not admissible."""


@dataclass(frozen=True)
class BundleEntry:
    """One manifest entry plus, for inline entries, its exact bytes."""

    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str
    #: Exact inline bytes.  ``None`` means the bytes live in the blob channel,
    #: or (with ``described=True``) that the bytes were deleted after a terminal
    #: job and only the persisted identity is left.
    data: bytes | None = None
    #: Content-addressed descriptor document for a blob-backed entry.
    blob_ref: Mapping[str, Any] | None = None
    #: True for a bundle rebuilt from storage after the inline bytes were
    #: cleaned up: the manifest is authoritative, the bytes are gone.
    described: bool = False

    @property
    def inline(self) -> bool:
        return self.data is not None or self.described

    def manifest(self) -> dict[str, Any]:
        """The manifest entry: no bytes, no transport form, no host path."""

        return {
            "path": self.path,
            "role": self.role,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }

    def document(self) -> dict[str, Any]:
        """The request-document entry: manifest plus its content *reference*.

        Inline bytes never appear here; they travel in the message envelope and
        are verified against ``size_bytes``/``sha256`` before admission.  The
        ``content`` discriminator keeps the transport form reconstructible from
        storage after the bytes are gone.
        """

        document = self.manifest()
        if self.blob_ref is not None:
            document["content"] = "blob"
            document["blob_ref"] = _blob_ref_dict(self.blob_ref)
        else:
            document["content"] = "inline"
        return document


@dataclass(frozen=True)
class MaterializedEntry:
    """One entry written and verified under an attempt root.

    It is a *handle*, not a path: reading through it re-walks the attempt
    directory with ``O_DIRECTORY|O_NOFOLLOW`` per component and re-verifies the
    exact size/digest, so a name that changed after ``materialize`` is refused
    instead of followed.
    """

    root: Path
    path: str
    size_bytes: int
    sha256: str

    def read_bytes(self) -> bytes:
        root_fd = _open_attempt_root(self.root, create=False)
        try:
            *parents, name = self.path.split("/")
            directory_fd = _descend_without_symlinks(root_fd, parents, self.path)
            try:
                return _read_entry_at(directory_fd, name, self.path)
            finally:
                os.close(directory_fd)
        finally:
            os.close(root_fd)

    def verify(self) -> None:
        data = self.read_bytes()
        if len(data) != self.size_bytes:
            raise BundleError(
                f"materialized bundle entry {self.path!r} size {len(data)} differs "
                f"from its declared size {self.size_bytes}"
            )
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise BundleError(
                f"materialized bundle entry {self.path!r} no longer matches its digest"
            )


@dataclass(frozen=True)
class ArtifactBundle:
    """A validated bundle; ``entries`` are kept in canonical (path) order."""

    entries: tuple[BundleEntry, ...] = field(default_factory=tuple)
    schema_id: str = BUNDLE_SCHEMA_ID

    def __post_init__(self) -> None:
        if self.schema_id != BUNDLE_SCHEMA_ID:
            raise BundleError("bundle schema id is not supported")
        entries = tuple(self.entries)
        for entry in entries:
            if not isinstance(entry, BundleEntry):
                raise BundleError("bundle entries must be typed BundleEntry values")
        normalized = normalize_entries(entries)
        if len(normalized["entries"]) != len(entries):
            raise BundleError("bundle entries collapsed while normalizing")
        # One canonical order (by path) so the request document and the digest
        # manifest are the same list in every language.
        object.__setattr__(self, "entries", tuple(normalized["entries"]))

    # -- identity ----------------------------------------------------------
    @property
    def manifest(self) -> dict[str, Any]:
        return canonical_manifest(self.schema_id, [entry.manifest() for entry in self.entries])

    @property
    def artifact_set_digest(self) -> str:
        return artifact_set_digest(self.schema_id, [entry.manifest() for entry in self.entries])

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def inline_bytes(self) -> int:
        return sum(len(entry.data or b"") for entry in self.entries)

    @property
    def blob_entries(self) -> tuple[BundleEntry, ...]:
        return tuple(entry for entry in self.entries if entry.data is None)

    def entry(self, path: str) -> BundleEntry:
        for entry in self.entries:
            if entry.path == path:
                return entry
        raise BundleError(f"bundle has no entry {path!r}")

    def roles(self) -> tuple[str, ...]:
        return tuple(sorted({entry.role for entry in self.entries}))

    # -- documents ---------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """The canonical request/receipt document (no bytes anywhere)."""

        return {
            "schema_id": self.schema_id,
            "artifact_set_digest": self.artifact_set_digest,
            "entries": [entry.document() for entry in self.entries],
        }

    def to_worker_dict(self) -> dict[str, Any]:
        """Private worker form: base64 inline bytes plus blob references."""

        document = self.to_dict()
        entries = []
        for entry, rendered in zip(self.entries, document["entries"]):
            if entry.data is None:
                if entry.described:
                    raise BundleError("a described bundle has no bytes to dispatch")
                # Blob-backed entries travel as a content reference only: the
                # worker reads the bytes from the trusted channel by content id.
                entries.append(dict(rendered))
                continue
            worker_entry = dict(rendered)
            worker_entry["data_base64"] = base64.b64encode(entry.data).decode("ascii")
            entries.append(worker_entry)
        document["entries"] = entries
        return document

    def verify_bytes(self) -> None:
        """Re-verify every inline entry against its declared size and digest."""

        for entry in self.entries:
            if entry.data is None:
                # A blob-backed entry's bytes live in the content-addressed
                # channel and are verified there; a described entry has none left.
                if entry.described or entry.blob_ref is not None:
                    continue
                raise BundleError(f"bundle entry {entry.path!r} has no bytes to verify")
            verify_entry_bytes(entry, entry.data)
            if entry.blob_ref is not None:
                raise BundleError(f"bundle entry {entry.path!r} must not be inline and blob-backed")

    def materialize(self, root: Path) -> dict[str, "MaterializedEntry"]:
        """Write the inline entries under ``root`` and return verified handles.

        The handles are deliberately not bare paths: a caller that re-reads
        ``root/graph/main.json`` resolves whatever that name points at *now*, so a
        parent renamed to a symbolic link after the write would hand the caller
        the attacker's bytes.  ``MaterializedEntry.read_bytes``/``verify`` walk the
        attempt directory again with ``O_DIRECTORY|O_NOFOLLOW`` per component.

        The directory belongs to one attempt.  Everything below it is traversed
        through *directory file descriptors*: every segment is opened with
        ``O_DIRECTORY|O_NOFOLLOW`` relative to its already-open parent and is
        created with ``mkdirat``, the file is created with ``openat`` plus
        ``O_CREAT|O_EXCL|O_NOFOLLOW`` and the bytes are read back from the same
        descriptor.  A path-name based check would still race: ``lstat`` a
        directory, get the name swapped for a symbolic link, then resolve the
        whole path again in ``open`` and write outside the bundle root.  With
        descriptors there is no second name resolution, so a swapped parent
        cannot redirect the write.
        """

        written: dict[str, MaterializedEntry] = {}
        root = Path(root)
        root_fd = _open_attempt_root(root)
        try:
            for entry in self.entries:
                if entry.data is None:
                    if entry.described:
                        raise BundleError("a described bundle has no inline bytes to materialize")
                    continue
                *parents, name = entry.path.split("/")
                directory_fd = _descend_without_symlinks(root_fd, parents, entry.path)
                try:
                    _write_entry_at(directory_fd, name, entry)
                finally:
                    os.close(directory_fd)
                written[entry.path] = MaterializedEntry(
                    root=root, path=entry.path,
                    size_bytes=entry.size_bytes, sha256=entry.sha256,
                )
        finally:
            os.close(root_fd)
        return written

    @classmethod
    def from_document(cls, value: Any) -> "ArtifactBundle":
        """Rebuild a *described* bundle from a content-free document.

        A terminal job deletes the inline bytes, but the manifest is still the
        authoritative identity; the described entries keep the digest and refuse
        to pretend the bytes are available.  Blob-backed entries keep their
        descriptor because the blob channel owns those bytes.
        """

        normalized = bundle_document(value)
        entries = [
            BundleEntry(
                path=document["path"], role=document["role"],
                media_type=document["media_type"], size_bytes=document["size_bytes"],
                sha256=document["sha256"], blob_ref=document.get("blob_ref"),
                described=document["content"] == "inline",
            )
            for document in normalized["entries"]
        ]
        return cls(entries=tuple(entries))

    @classmethod
    def from_worker_dict(cls, value: Mapping[str, Any]) -> "ArtifactBundle":
        """Rebuild the typed bundle from the private worker frame."""

        if not isinstance(value, Mapping):
            raise BundleError("worker bundle must be an object")
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, (list, tuple)):
            raise BundleError("worker bundle entries must be an array")
        entries: list[BundleEntry] = []
        for document in raw_entries:
            if not isinstance(document, Mapping):
                raise BundleError("worker bundle entry must be an object")
            if document.get("content") == "inline":
                if not isinstance(document.get("data_base64"), str):
                    raise BundleError("worker inline bundle entry has no exact bytes")
                try:
                    data = base64.b64decode(document["data_base64"], validate=True)
                except (ValueError, TypeError) as exc:
                    raise BundleError("worker bundle data_base64 is invalid") from exc
                blob_ref = None
            else:
                data = None
                blob_ref = document.get("blob_ref")
            entries.append(BundleEntry(
                path=document["path"], role=document["role"],
                media_type=document["media_type"], size_bytes=document["size_bytes"],
                sha256=document["sha256"], data=data, blob_ref=blob_ref,
            ))
        if value.get("schema_id") != BUNDLE_SCHEMA_ID:
            raise BundleError("worker bundle schema id is not supported")
        bundle = cls(entries=tuple(entries))
        if value.get("artifact_set_digest") != bundle.artifact_set_digest:
            raise BundleError("worker bundle artifact_set_digest does not bind the manifest")
        stripped = [
            {key: item for key, item in document.items() if key != "data_base64"}
            for document in raw_entries
        ]
        if stripped != bundle.to_dict()["entries"]:
            raise BundleError("worker bundle entries are not the canonical content-free document")
        return bundle


def validate_relative_path(value: Any) -> str:
    """Return a safe relative POSIX path or raise.

    Accepted: ``graph/main.json``, ``a.b/c-d_e.bin``.  Rejected: absolute paths,
    ``.``/``..`` segments, empty segments, trailing slashes, backslashes,
    control characters and non-UTF-8 text.  Nothing is normalized: a path is
    compared exactly as it will be materialized.
    """

    if not isinstance(value, str) or not value:
        raise BundleError("bundle entry path must be a non-empty string")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise BundleError("bundle entry path must contain valid Unicode scalar values") from exc
    if len(encoded) > MAX_BUNDLE_PATH_BYTES:
        raise BundleError("bundle entry path exceeds its UTF-8 byte limit")
    if value.startswith("/"):
        raise BundleError("bundle entry path must be relative")
    if "\\" in value:
        raise BundleError("bundle entry path must use POSIX separators")
    if value.endswith("/"):
        raise BundleError("bundle entry path must name a file")
    if any(character < " " or character == "\x7f" for character in value):
        raise BundleError("bundle entry path must not contain control characters")
    segments = value.split("/")
    if len(segments) > MAX_BUNDLE_PATH_SEGMENTS:
        raise BundleError("bundle entry path is too deep")
    for segment in segments:
        if segment in ("", ".", ".."):
            raise BundleError("bundle entry path must not contain empty, '.' or '..' segments")
        if segment != segment.strip():
            raise BundleError("bundle entry path segment must not have surrounding whitespace")
    return value


def validate_role(value: Any) -> str:
    if not isinstance(value, str) or value not in BUNDLE_ROLES:
        raise BundleError("bundle entry role is not a supported public role")
    return value


def validate_media_type(value: Any) -> str:
    if not isinstance(value, str) or not _MEDIA_TYPE.fullmatch(value):
        raise BundleError("bundle entry media type is not a bounded ASCII media type")
    return value


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise BundleError(f"{name} must be 64 lowercase hexadecimal characters")
    return value


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise BundleError(f"{name} is outside its integer bounds")
    return value


def canonical_manifest(schema_id: str, entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return the path-sorted manifest that ``artifact_set_digest`` covers."""

    if schema_id != BUNDLE_SCHEMA_ID:
        raise BundleError("bundle schema id is not supported")
    manifest_entries = []
    for entry in entries:
        manifest_entries.append({
            "path": validate_relative_path(entry["path"]),
            "role": validate_role(entry["role"]),
            "media_type": validate_media_type(entry["media_type"]),
            "size_bytes": _integer(entry["size_bytes"], "bundle entry size", 0, MAX_BUNDLE_TOTAL_BYTES),
            "sha256": _sha256(entry["sha256"], "bundle entry sha256"),
        })
    manifest_entries.sort(key=lambda item: item["path"])
    return {
        "schema_id": schema_id,
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "entries": manifest_entries,
    }


def artifact_set_digest(schema_id: str, entries: Iterable[Mapping[str, Any]]) -> str:
    """``SHA256(RFC8785(canonical_manifest))``; never over Protobuf bytes."""

    from .contracts import canonical_digest_document

    manifest = canonical_manifest(schema_id, entries)
    return hashlib.sha256(canonical_digest_document(manifest)).hexdigest()


def verify_entry_bytes(entry: BundleEntry, data: bytes) -> None:
    """Verify exact bytes: size, then SHA-256.  No normalization of any kind."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise BundleError("bundle entry bytes must be bytes-like")
    raw = bytes(data)
    if len(raw) != entry.size_bytes:
        raise BundleError(
            f"bundle entry {entry.path!r} size {len(raw)} differs from its declared size {entry.size_bytes}"
        )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry.sha256:
        raise BundleError(f"bundle entry {entry.path!r} SHA-256 does not match the exact bytes")


def _blob_ref_dict(value: Any) -> dict[str, Any]:
    """Accept a mapping or the typed descriptor without copying bytes."""

    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        document = to_dict()
        if isinstance(document, Mapping):
            return dict(document)
    raise BundleError("a bundle entry blob reference must be a descriptor or mapping")


def _blob_descriptor(value: Any) -> dict[str, Any]:
    from .contracts import ContractError, blob_descriptor_document

    try:
        return blob_descriptor_document(_blob_ref_dict(value))
    except (ContractError, BundleError) as exc:
        raise BundleError(f"bundle entry blob reference is invalid: {exc}") from exc


def _alias_key(path: str) -> str:
    """The filesystem-alias key of a path: Unicode composition plus case fold.

    Two paths with the same key would be one file on a case-insensitive or
    normalization-insensitive filesystem, so a bundle may not contain both.  The
    key is used *only* for conflict detection: the manifest and the digest keep
    the exact original path and no byte or path is ever normalized.
    """

    return unicodedata.normalize("NFC", path).casefold()


def _check_path_prefixes(paths: Iterable[str]) -> None:
    """Refuse a bundle where one entry's path is another entry's parent.

    ``graph`` and ``graph/main.json`` cannot both be materialized: one of them
    would have to be a file and a directory at the same time.  The comparison
    uses the same alias key as the full-path check, so ``Graph`` and
    ``graph/main.json`` (or a composed and a decomposed ``café``) are refused on
    every platform instead of failing half way through an attempt directory on a
    case-insensitive or normalization-insensitive filesystem.
    """

    paths = list(paths)
    alias_by_key = {_alias_key(path): path for path in paths}
    for path in paths:
        segments = path.split("/")
        for index in range(1, len(segments)):
            prefix = "/".join(segments[:index])
            clash = alias_by_key.get(_alias_key(prefix))
            if clash is not None:
                raise BundleError(
                    f"bundle entry path is the parent of another entry: {path!r} "
                    f"(aliases {clash!r})"
                )


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)


def _open_directory_at(parent_fd: int, name: str) -> int:
    """Open one child directory by name, refusing a symbolic link."""

    try:
        return os.open(
            name, _directory_flags() | getattr(os, "O_NOFOLLOW", 0), dir_fd=parent_fd,
        )
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            # macOS reports ENOTDIR for a symlink-to-directory and Linux reports
            # ELOOP, so the message is derived from the entry itself.  The stat is
            # relative to the verified descriptor, and the refusal already
            # happened in ``open``: this only names what was refused.
            if _is_symlink_at(parent_fd, name):
                raise BundleError(
                    f"bundle target parent must not be a symbolic link: {name!r}"
                ) from exc
            raise BundleError(
                f"bundle target parent is not a directory: {name!r}"
            ) from exc
        raise


def _is_symlink_at(directory_fd: int, name: str) -> bool:
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except OSError:
        return False
    return stat.S_ISLNK(info.st_mode)


def _open_attempt_root(root: Path, *, create: bool = True) -> int:
    """Open (creating if asked) the attempt directory without following links.

    The parent chain above the attempt directory is engine-owned state, so it is
    created by path; the attempt directory itself is opened relative to that
    parent with ``O_DIRECTORY|O_NOFOLLOW`` and can therefore never be a planted
    symbolic link.
    """

    if not root.name or root.name in (".", ".."):
        raise BundleError("the bundle target must name a directory")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_fd = os.open(parent, _directory_flags())
    try:
        try:
            return _open_directory_at(parent_fd, root.name)
        except FileNotFoundError:
            if not create:
                raise BundleError("the bundle target directory does not exist")
            try:
                os.mkdir(root.name, 0o700, dir_fd=parent_fd)
            except FileExistsError:
                pass
            return _open_directory_at(parent_fd, root.name)
    finally:
        os.close(parent_fd)


def _descend_without_symlinks(root_fd: int, segments: Iterable[str], path: str) -> int:
    """Return an fd for ``root/segments...``, creating plain directories only.

    Each step is resolved once, relative to the descriptor of the directory that
    was already verified, so swapping a parent name for a symbolic link cannot
    move the write: the descriptor still names the checked directory.
    """

    current_fd = os.dup(root_fd)
    for name in segments:
        try:
            next_fd = _open_directory_at(current_fd, name)
        except FileNotFoundError:
            try:
                os.mkdir(name, 0o700, dir_fd=current_fd)
            except FileExistsError:
                pass
            next_fd = _open_directory_at(current_fd, name)
        os.close(current_fd)
        current_fd = next_fd
    return current_fd


def _write_entry_at(directory_fd: int, name: str, entry: BundleEntry) -> None:
    """Create one entry with openat and verify the bytes through the same fd."""

    flags = (
        os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR, errno.EISDIR):
            raise BundleError(
                f"bundle target must not be a symbolic link or directory: {entry.path!r}"
            ) from exc
        if exc.errno == errno.EEXIST:
            # ``O_EXCL`` already refused to follow whatever is there; classify
            # it relative to the same descriptor so the message is accurate
            # without a second, racy path-name lookup.
            try:
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError:
                info = None
            if info is not None and stat.S_ISLNK(info.st_mode):
                raise BundleError(
                    f"bundle target must not be a symbolic link: {entry.path!r}"
                ) from exc
            raise BundleError(f"bundle target already exists: {entry.path!r}") from exc
        raise
    with os.fdopen(descriptor, "w+b") as handle:
        handle.write(entry.data)
        handle.flush()
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"materialized bundle entry is not a regular file: {entry.path!r}")
        handle.seek(0)
        verify_entry_bytes(entry, handle.read())


def _read_entry_at(directory_fd: int, name: str, path: str) -> bytes:
    """Open one entry with ``O_NOFOLLOW`` under a verified directory fd and read it."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BundleError(f"bundle entry must not be a symbolic link: {path!r}") from exc
        raise BundleError(f"bundle entry is not readable: {path!r}") from exc
    with os.fdopen(descriptor, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise BundleError(f"bundle entry is not a regular file: {path!r}")
        return handle.read()


def _check_entry_rules(
    *, path: str, size: int, digest: str, media_type: str, content: str,
    blob_ref: Mapping[str, Any] | None, seen: dict[str, str],
) -> None:
    alias = _alias_key(path)
    if alias in seen:
        # Two entries a case-insensitive or normalization-insensitive filesystem
        # would collapse into one file must never both exist.
        raise BundleError(
            f"bundle entry path is duplicated: {path!r} (aliases {seen[alias]!r})"
        )
    seen[alias] = path
    if content == "inline":
        if blob_ref is not None:
            raise BundleError("an inline bundle entry must not carry a blob reference")
        if size > MAX_BUNDLE_INLINE_ENTRY_BYTES:
            raise BundleError("an inline bundle entry exceeds the inline byte limit")
    elif content == "blob":
        if blob_ref is None:
            raise BundleError("a blob-backed bundle entry requires a blob reference")
        descriptor = _blob_descriptor(blob_ref)
        if descriptor["sha256"] != digest or descriptor["size_bytes"] != size:
            raise BundleError("bundle entry blob reference does not match its manifest identity")
        if descriptor["media_type"] != media_type:
            raise BundleError("bundle entry blob media type differs from the manifest media type")
        if size <= MAX_BUNDLE_INLINE_ENTRY_BYTES:
            # A small artifact has no reason to skip the inline path; this keeps
            # the "large artifact" channel honest instead of turning every entry
            # into a blob read.
            raise BundleError(
                "a bundle entry within the inline limit must be inlined, not blob-referenced"
            )
    else:
        raise BundleError("bundle entry content must be inline or blob")


def normalize_entries(entries: Iterable[BundleEntry]) -> dict[str, Any]:
    """Validate typed entries (with bytes) and return the normalized bundle."""

    kept: list[BundleEntry] = []
    seen: dict[str, str] = {}
    total = 0
    inline_total = 0
    for entry in entries:
        if not isinstance(entry, BundleEntry):
            raise BundleError("bundle entries must be typed BundleEntry values")
        path = validate_relative_path(entry.path)
        role = validate_role(entry.role)
        media_type = validate_media_type(entry.media_type)
        size = _integer(entry.size_bytes, "bundle entry size", 0, MAX_BUNDLE_TOTAL_BYTES)
        digest = _sha256(entry.sha256, "bundle entry sha256")
        content = "blob" if entry.blob_ref is not None else "inline"
        if entry.described and (entry.data is not None or entry.blob_ref is not None):
            raise BundleError("a described bundle entry carries no bytes and no blob reference")
        if entry.data is not None and not isinstance(entry.data, (bytes, bytearray, memoryview)):
            raise BundleError("bundle entry data must be bytes-like")
        if entry.data is None and entry.blob_ref is None and not entry.described:
            raise BundleError("a bundle entry needs exact bytes or a blob reference")
        _check_entry_rules(
            path=path, size=size, digest=digest, media_type=media_type,
            content=content, blob_ref=entry.blob_ref, seen=seen,
        )
        if entry.described:
            inline_total += size
            kept.append(BundleEntry(
                path=path, role=role, media_type=media_type, size_bytes=size,
                sha256=digest, described=True,
            ))
        elif entry.data is not None:
            raw = bytes(entry.data)
            verify_entry_bytes(
                BundleEntry(
                    path=path, role=role, media_type=media_type, size_bytes=size,
                    sha256=digest, data=raw,
                ),
                raw,
            )
            inline_total += size
            kept.append(BundleEntry(
                path=path, role=role, media_type=media_type, size_bytes=size,
                sha256=digest, data=raw,
            ))
        else:
            kept.append(BundleEntry(
                path=path, role=role, media_type=media_type, size_bytes=size,
                sha256=digest, blob_ref=_blob_descriptor(entry.blob_ref),
            ))
        total += size
    if not kept:
        raise BundleError("a bundle needs at least one entry")
    if len(kept) > MAX_BUNDLE_ENTRIES:
        raise BundleError("a bundle exceeds the entry-count limit")
    if inline_total > MAX_BUNDLE_INLINE_BYTES:
        raise BundleError("the inline bundle exceeds the total inline byte limit")
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise BundleError("a bundle exceeds the declared total byte limit")
    _check_path_prefixes(entry.path for entry in kept)
    kept.sort(key=lambda item: item.path)
    return {
        "schema_id": BUNDLE_SCHEMA_ID,
        "artifact_set_digest": artifact_set_digest(
            BUNDLE_SCHEMA_ID, [entry.manifest() for entry in kept],
        ),
        "entries": kept,
    }


def bundle_document(value: Any) -> dict[str, Any]:
    """Validate a content-free bundle document (request/receipt/storage form).

    Returns ``{"schema_id", "artifact_set_digest", "entries": [...]}`` where the
    entries carry the manifest fields plus ``content`` (and ``blob_ref`` for a
    blob-backed entry).  The declared ``artifact_set_digest`` must be exactly the
    digest of the manifest, so a rewritten entry list can never keep the old
    identity, and the inline/blob form is still reconstructible after the bytes
    are deleted.
    """

    if not isinstance(value, Mapping):
        raise BundleError("bundle must be an object")
    if set(value) != {"schema_id", "artifact_set_digest", "entries"}:
        raise BundleError("bundle fields are missing or unknown")
    if value["schema_id"] != BUNDLE_SCHEMA_ID:
        raise BundleError("bundle schema id is not supported by this contract")
    raw_entries = value["entries"]
    if not isinstance(raw_entries, (list, tuple)):
        raise BundleError("bundle entries must be an array")
    documents: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    total = 0
    inline_total = 0
    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            raise BundleError("bundle entry must be an object")
        allowed = set(_MANIFEST_FIELDS) | {"content", "blob_ref"}
        if set(entry) - allowed:
            raise BundleError("bundle entry has unknown fields")
        for name in _MANIFEST_FIELDS:
            if name not in entry:
                raise BundleError(f"bundle entry field {name} is missing")
        path = validate_relative_path(entry["path"])
        role = validate_role(entry["role"])
        media_type = validate_media_type(entry["media_type"])
        size = _integer(entry["size_bytes"], "bundle entry size", 0, MAX_BUNDLE_TOTAL_BYTES)
        digest = _sha256(entry["sha256"], "bundle entry sha256")
        content = entry.get("content")
        blob_ref = entry.get("blob_ref")
        _check_entry_rules(
            path=path, size=size, digest=digest, media_type=media_type,
            content=content, blob_ref=blob_ref, seen=seen,
        )
        document: dict[str, Any] = {
            "path": path, "role": role, "media_type": media_type,
            "size_bytes": size, "sha256": digest, "content": content,
        }
        if content == "blob":
            document["blob_ref"] = _blob_descriptor(blob_ref)
        total += size
        if content == "inline":
            inline_total += size
        documents.append(document)
    if not documents:
        raise BundleError("a bundle needs at least one entry")
    if len(documents) > MAX_BUNDLE_ENTRIES:
        raise BundleError("a bundle exceeds the entry-count limit")
    if inline_total > MAX_BUNDLE_INLINE_BYTES:
        raise BundleError("the inline bundle exceeds the total inline byte limit")
    if total > MAX_BUNDLE_TOTAL_BYTES:
        raise BundleError("a bundle exceeds the declared total byte limit")
    _check_path_prefixes(document["path"] for document in documents)
    documents.sort(key=lambda item: item["path"])
    manifest = canonical_manifest(BUNDLE_SCHEMA_ID, documents)
    computed = artifact_set_digest(BUNDLE_SCHEMA_ID, documents)
    declared = _sha256(value["artifact_set_digest"], "bundle artifact_set_digest")
    if declared != computed:
        raise BundleError("bundle artifact_set_digest does not bind the manifest")
    return {
        "schema_id": BUNDLE_SCHEMA_ID,
        "artifact_set_digest": computed,
        "entries": documents,
        "manifest": manifest,
    }


def payload_bytes(bundle: ArtifactBundle) -> bytes:
    """The exact bytes a job persists for its bundle artifact.

    Inline entries travel as base64 inside one canonical JSON document so the
    bytes have one deterministic container; blob-backed entries keep only their
    descriptor (the blob channel owns those bytes and its own lifecycle).
    """

    if not isinstance(bundle, ArtifactBundle):
        raise BundleError("bundle payload requires a typed ArtifactBundle")
    document = bundle.to_worker_dict()
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")


def payload_bundle(data: bytes) -> ArtifactBundle:
    """Rebuild the typed bundle from its persisted payload and verify the bytes."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise BundleError("bundle payload must be bytes-like")
    try:
        document = json.loads(bytes(data).decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise BundleError(f"bundle payload is not canonical JSON: {exc}") from exc
    return ArtifactBundle.from_worker_dict(document)


# -- host-side bundle building (test/tooling support) ----------------------
def entry_from_file(
    root: Path | str, relative: str, *, role: str, media_type: str,
) -> BundleEntry:
    """Read one artifact from a trusted host directory, through descriptors.

    This is the only place a host path is ever used.  The base directory and
    every component below it is opened with ``O_DIRECTORY|O_NOFOLLOW`` relative
    to its already-verified parent and the file itself with ``O_NOFOLLOW``; the
    bytes and the ``fstat`` come from the *same* descriptor.  A version that
    checked with ``lstat`` and then read by path would hand back the bytes of a
    symbolic link that appeared in between.  Nothing is normalized; bytes are
    hashed exactly.
    """

    base = Path(root)
    path = validate_relative_path(relative)
    if base.is_symlink():
        raise BundleError("bundle source directory must not be a symbolic link")
    try:
        base_fd = os.open(base, _directory_flags() | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise BundleError("bundle source directory does not exist") from exc
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BundleError("bundle source directory must not be a symbolic link") from exc
        raise BundleError(f"bundle source directory is not readable: {exc}") from exc
    *parents, name = path.split("/")
    try:
        directory_fd = _descend_without_symlinks(base_fd, parents, path)
    finally:
        os.close(base_fd)
    try:
        data = _read_entry_at(directory_fd, name, path)
    finally:
        os.close(directory_fd)
    return BundleEntry(
        path=path, role=validate_role(role), media_type=validate_media_type(media_type),
        size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), data=data,
    )


def bundle_from_directory(
    root: Path | str, entries: Iterable[tuple[str, str, str]],
) -> ArtifactBundle:
    """Build a bundle from ``(relative_path, role, media_type)`` declarations."""

    built = [entry_from_file(root, path, role=role, media_type=media) for path, role, media in entries]
    bundle = ArtifactBundle(entries=tuple(built))
    bundle.verify_bytes()
    return bundle
