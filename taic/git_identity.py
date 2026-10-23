"""Git identity and repository state for review/audit records.

A review uses the same identity source as a commit: ``git var GIT_AUTHOR_IDENT``
(which honours the repository config and Git author environment), with a
``git config user.name/user.email`` fallback. No review form asks for a name.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_IDENT = re.compile(r"^(?P<name>.*?)\s*<(?P<email>[^>]*)>\s+\d+\s+[+-]\d{4}$")


@dataclass(frozen=True)
class GitAuthor:
    name: str
    email: str
    source: str  # git | env | unknown

    @property
    def display(self) -> str:
        if self.name and self.email:
            return f"{self.name} <{self.email}>"
        return self.name or self.email or "unknown"


def _run(args: list[str], root: Path) -> str:
    try:
        result = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def author(root: Path | None = None) -> GitAuthor:
    """The commit-style author, without mutating Git config or using the network."""
    base = Path(root or REPO_ROOT)
    ident = _run(["git", "var", "GIT_AUTHOR_IDENT"], base)
    match = _IDENT.match(ident)
    if match:
        return GitAuthor(match.group("name").strip(), match.group("email").strip(), "git")
    name = _run(["git", "config", "--get", "user.name"], base)
    email = _run(["git", "config", "--get", "user.email"], base)
    if name or email:
        return GitAuthor(name, email, "git")
    env_name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER") or ""
    env_email = os.environ.get("GIT_AUTHOR_EMAIL") or os.environ.get("EMAIL") or ""
    if env_name or env_email:
        return GitAuthor(env_name, env_email, "env")
    return GitAuthor("", "", "unknown")


def state(root: Path | None = None) -> tuple[str, bool | None]:
    """``(HEAD, dirty)`` at review time; ``("unknown", None)`` when Git is unavailable."""
    base = Path(root or REPO_ROOT)
    commit = _run(["git", "rev-parse", "HEAD"], base)
    if not commit:
        return "unknown", None
    status = _run(["git", "status", "--porcelain"], base)
    return commit, bool(status)
