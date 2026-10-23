"""Review actor identity follows Git's commit author, not a form field."""

from __future__ import annotations

import subprocess
from pathlib import Path

from taic import git_identity


class _Result:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def test_the_reviewer_uses_git_var_author_ident(monkeypatch):
    def fake_run(args, **_kwargs):
        if args[:3] == ["git", "var", "GIT_AUTHOR_IDENT"]:
            return _Result("Ada Lovelace <ada@example.test> 1700000000 +0000\n")
        if args[:2] == ["git", "rev-parse"]:
            return _Result("abc123\n")
        if args[:2] == ["git", "status"]:
            return _Result("")
        raise AssertionError(args)

    monkeypatch.setattr(git_identity.subprocess, "run", fake_run)
    who = git_identity.author()
    assert who.name == "Ada Lovelace"
    assert who.email == "ada@example.test"
    assert who.source == "git"
    assert who.display == "Ada Lovelace <ada@example.test>"
    assert git_identity.state() == ("abc123", False)


def test_identity_falls_back_to_git_config_then_environment(monkeypatch):
    def config_run(args, **_kwargs):
        if args[:3] == ["git", "var", "GIT_AUTHOR_IDENT"]:
            return _Result("", 128)
        if args[:4] == ["git", "config", "--get", "user.name"]:
            return _Result("Config Name\n")
        if args[:4] == ["git", "config", "--get", "user.email"]:
            return _Result("config@example.test\n")
        raise AssertionError(args)

    monkeypatch.setattr(git_identity.subprocess, "run", config_run)
    who = git_identity.author()
    assert who.display == "Config Name <config@example.test>"

    def env_run(args, **_kwargs):
        return _Result("", 128)

    monkeypatch.setattr(git_identity.subprocess, "run", env_run)
    monkeypatch.setenv("GIT_AUTHOR_NAME", "Env Name")
    monkeypatch.setenv("GIT_AUTHOR_EMAIL", "env@example.test")
    who = git_identity.author()
    assert who.display == "Env Name <env@example.test>"
    assert who.source == "env"
