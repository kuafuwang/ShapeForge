"""Small, read-only Git object reader used by the acceptance inventory.

The governance inventory must not spawn processes.  This module therefore reads
only ``.git/index``, refs, loose objects and ordinary v2 pack/idx files.  It is
intentionally scoped to local inspection, not a general Git implementation.
Untracked paths have no fabricated commit dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
from pathlib import Path, PurePosixPath
import struct
from typing import Iterator
import zlib

ROOT = Path(__file__).resolve().parents[2]


def _u32(data: bytes, offset: int = 0) -> int:
    return struct.unpack_from("!I", data, offset)[0]


def _u64(data: bytes, offset: int = 0) -> int:
    return struct.unpack_from("!Q", data, offset)[0]


def tracked_paths(root: Path = ROOT) -> frozenset[str]:
    """Read path names from a version-2/3 Git index without running git."""
    data = (root / ".git/index").read_bytes()
    if data[:4] != b"DIRC":
        raise ValueError("unsupported Git index signature")
    version, count = struct.unpack_from("!II", data, 4)
    if version not in (2, 3):
        raise ValueError(f"unsupported Git index version: {version}")
    offset = 12
    paths: set[str] = set()
    for _ in range(count):
        flags = struct.unpack_from("!H", data, offset + 60)[0]
        name_start = offset + 62
        if flags & 0x4000:  # extended flags occupy two more bytes in v3
            name_start += 2
        name_end = data.index(b"\0", name_start)
        paths.add(data[name_start:name_end].decode("utf-8", "surrogateescape"))
        entry_length = (name_end - offset + 1 + 7) & ~7
        offset += entry_length
    return frozenset(paths)


@dataclass(frozen=True)
class Commit:
    tree: str
    parents: tuple[str, ...]
    timestamp: int


class _Pack:
    def __init__(self, index_path: Path) -> None:
        self.index_path = index_path
        self.pack_path = index_path.with_suffix(".pack")
        self.offsets: dict[str, tuple[int, int]] = {}
        data = index_path.read_bytes()
        if data[:4] == b"\xfftOc":
            if _u32(data, 4) != 2:
                raise ValueError(f"unsupported pack index: {index_path}")
            count = _u32(data, 8 + 255 * 4)
            names = 8 + 256 * 4
            offsets = names + count * 20 + count * 4
            large = offsets + count * 4
            for number in range(count):
                oid = data[names + number * 20:names + (number + 1) * 20].hex()
                offset = _u32(data, offsets + number * 4)
                if offset & 0x80000000:
                    offset = _u64(data, large + (offset & 0x7FFFFFFF) * 8)
                self.offsets[oid] = (offset, number)
        else:
            count = _u32(data, 255 * 4)
            for number in range(count):
                entry = 256 * 4 + number * 24
                offset = _u32(data, entry)
                oid = data[entry + 4:entry + 24].hex()
                self.offsets[oid] = (offset, number)
        self.data = self.pack_path.read_bytes()
        if self.data[:4] != b"PACK":
            raise ValueError(f"unsupported pack signature: {self.pack_path}")

    @staticmethod
    def _object_header(data: bytes, offset: int) -> tuple[int, int, int]:
        byte = data[offset]
        offset += 1
        kind = (byte >> 4) & 7
        size = byte & 15
        shift = 4
        while byte & 0x80:
            byte = data[offset]
            offset += 1
            size |= (byte & 0x7F) << shift
            shift += 7
        return kind, size, offset

    @staticmethod
    def _delta_size(data: bytes, offset: int) -> tuple[int, int]:
        size = 0
        shift = 0
        while True:
            byte = data[offset]
            offset += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return size, offset

    @staticmethod
    def _apply_delta(base: bytes, delta: bytes) -> bytes:
        base_size, offset = _Pack._delta_size(delta, 0)
        result_size, offset = _Pack._delta_size(delta, offset)
        if base_size != len(base):
            raise ValueError("Git delta base size mismatch")
        result = bytearray()
        while offset < len(delta):
            command = delta[offset]
            offset += 1
            if command & 0x80:
                copy_offset = 0
                copy_size = 0
                for bit, shift in ((0x01, 0), (0x02, 8), (0x04, 16), (0x08, 24)):
                    if command & bit:
                        copy_offset |= delta[offset] << shift
                        offset += 1
                for bit, shift in ((0x10, 0), (0x20, 8), (0x40, 16)):
                    if command & bit:
                        copy_size |= delta[offset] << shift
                        offset += 1
                if copy_size == 0:
                    copy_size = 0x10000
                result.extend(base[copy_offset:copy_offset + copy_size])
            elif command:
                result.extend(delta[offset:offset + command])
                offset += command
            else:
                raise ValueError("invalid zero Git delta command")
        if len(result) != result_size:
            raise ValueError("Git delta result size mismatch")
        return bytes(result)


class GitObjects:
    """Read reachable commits and path tree ids from the local object database."""

    def __init__(self, root: Path = ROOT) -> None:
        self.root = root
        self.git = root / ".git"
        self.objects = self.git / "objects"
        self.packs = [_Pack(path) for path in sorted((self.objects / "pack").glob("*.idx"))]
        self._offset_objects: dict[tuple[int, int], tuple[str, bytes, str]] = {}
        self._objects: dict[str, tuple[str, bytes]] = {}
        self._commits: dict[str, Commit] = {}
        self._trees: dict[str, dict[str, tuple[str, str]]] = {}

    def _read_loose(self, oid: str) -> tuple[str, bytes] | None:
        path = self.objects / oid[:2] / oid[2:]
        if not path.is_file():
            return None
        raw = zlib.decompress(path.read_bytes())
        header, body = raw.split(b"\0", 1)
        kind, size = header.split(b" ", 1)
        if int(size) != len(body):
            raise ValueError(f"loose object size mismatch: {oid}")
        return kind.decode("ascii"), body

    def _at(self, pack_number: int, offset: int) -> tuple[str, bytes, str]:
        key = (pack_number, offset)
        if key in self._offset_objects:
            return self._offset_objects[key]
        pack = self.packs[pack_number]
        kind, expected_size, cursor = _Pack._object_header(pack.data, offset)
        if kind == 6:  # OFS_DELTA
            byte = pack.data[cursor]
            cursor += 1
            distance = byte & 0x7F
            while byte & 0x80:
                byte = pack.data[cursor]
                cursor += 1
                distance = ((distance + 1) << 7) | (byte & 0x7F)
            base_kind, base, _ = self._at(pack_number, offset - distance)
            delta = zlib.decompress(pack.data[cursor:])
            if len(delta) != expected_size:
                raise ValueError("packed delta size mismatch")
            body = _Pack._apply_delta(base, delta)
            kind_name = base_kind
        elif kind == 7:  # REF_DELTA
            base_oid = pack.data[cursor:cursor + 20].hex()
            cursor += 20
            base_kind, base = self.read(base_oid)
            delta = zlib.decompress(pack.data[cursor:])
            if len(delta) != expected_size:
                raise ValueError("packed delta size mismatch")
            body = _Pack._apply_delta(base, delta)
            kind_name = base_kind
        else:
            kind_name = {1: "commit", 2: "tree", 3: "blob", 4: "tag"}.get(kind, "")
            if not kind_name:
                raise ValueError(f"unsupported packed object type {kind}")
            body = zlib.decompress(pack.data[cursor:])
            base_oid = ""
        if kind not in (6, 7) and len(body) != expected_size:
            raise ValueError("packed object size mismatch")
        oid = hashlib.sha1(f"{kind_name} {len(body)}\0".encode() + body).hexdigest()
        result = (kind_name, body, oid)
        self._offset_objects[key] = result
        self._objects.setdefault(oid, (kind_name, body))
        return result

    def read(self, oid: str) -> tuple[str, bytes]:
        if oid in self._objects:
            return self._objects[oid]
        loose = self._read_loose(oid)
        if loose is not None:
            self._objects[oid] = loose
            return loose
        for number, pack in enumerate(self.packs):
            located = pack.offsets.get(oid)
            if located is not None:
                kind, body, _ = self._at(number, located[0])
                self._objects[oid] = (kind, body)
                return kind, body
        raise KeyError(f"Git object not found: {oid}")

    def commit(self, oid: str) -> Commit:
        if oid in self._commits:
            return self._commits[oid]
        kind, body = self.read(oid)
        if kind != "commit":
            raise ValueError(f"not a commit: {oid}")
        headers = body.split(b"\n\n", 1)[0].decode("utf-8", "replace").splitlines()
        tree = next(line.split()[1] for line in headers if line.startswith("tree "))
        parents = tuple(line.split()[1] for line in headers if line.startswith("parent "))
        committer = next(line for line in headers if line.startswith("committer "))
        timestamp = int(committer.rsplit(" ", 2)[1])
        result = Commit(tree, parents, timestamp)
        self._commits[oid] = result
        return result

    def head(self) -> str:
        reference = (self.git / "HEAD").read_text(encoding="ascii").strip()
        if reference.startswith("ref: "):
            name = reference[5:]
            path = self.git / name
            if path.is_file():
                return path.read_text(encoding="ascii").strip()
            for line in (self.git / "packed-refs").read_text(encoding="ascii").splitlines():
                if line.endswith(" " + name):
                    return line.split(" ", 1)[0]
            raise KeyError(f"unresolved HEAD ref: {name}")
        return reference

    def reachable_commits(self) -> list[str]:
        seen: set[str] = set()
        pending = [self.head()]
        ordered: list[str] = []
        while pending:
            oid = pending.pop()
            if oid in seen:
                continue
            seen.add(oid)
            ordered.append(oid)
            pending.extend(self.commit(oid).parents)
        return ordered

    def tree_entries(self, oid: str) -> dict[str, tuple[str, str]]:
        if oid in self._trees:
            return self._trees[oid]
        kind, body = self.read(oid)
        if kind != "tree":
            raise ValueError(f"not a tree: {oid}")
        entries: dict[str, tuple[str, str]] = {}
        offset = 0
        while offset < len(body):
            space = body.index(b" ", offset)
            nul = body.index(b"\0", space)
            mode = body[offset:space].decode("ascii")
            name = body[space + 1:nul].decode("utf-8", "surrogateescape")
            child = body[nul + 1:nul + 21].hex()
            entries[name] = (mode, child)
            offset = nul + 21
        self._trees[oid] = entries
        return entries

    def path_oid(self, commit_oid: str, path: str) -> str | None:
        current = self.commit(commit_oid).tree
        parts = PurePosixPath(path).parts
        for number, part in enumerate(parts):
            entry = self.tree_entries(current).get(part)
            if entry is None:
                return None
            mode, current = entry
            if number < len(parts) - 1 and not mode.startswith("04") and mode != "40000":
                return None
        return current

    def path_dates(self, path: str, commits: list[str] | None = None) -> tuple[str | None, str | None]:
        """Return first/last committer dates at which this exact path changed."""
        changed: list[int] = []
        for oid in commits if commits is not None else self.reachable_commits():
            commit = self.commit(oid)
            current = self.path_oid(oid, path)
            if current is None:
                continue
            if any(self.path_oid(parent, path) == current for parent in commit.parents):
                continue
            changed.append(commit.timestamp)
        if not changed:
            return None, None
        dates = [datetime.fromtimestamp(value, timezone.utc).date().isoformat() for value in (min(changed), max(changed))]
        return dates[0], dates[1]
