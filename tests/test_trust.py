"""Per-target trust grants: scope, protocol, prompt, expiry, and the audit trail. No network, no GPU."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from taic import trust


def test_a_grant_is_target_and_task_scoped_and_written_to_the_target(tmp_path):
    entry = trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", effort="high",
        scopes=["overlay:tip"], granted_by="alice", reason="pilot", root=tmp_path,
    )
    assert (tmp_path / "apple-m4-max" / "trust.toml").is_file()
    assert trust.trust_path("apple-m4-max", tmp_path).read_text().startswith("version = 1")
    assert trust.load_grants("apple-m4-max", tmp_path)[0]["id"] == entry["id"]
    ok, why = trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", effort="high", scope="overlay:tip")
    assert ok and why == ""
    wrong_task, why = trust.check(entry, target_id="apple-m4-max", task="flash_attention", provider="deepseek", model="deepseek-flash", effort="high")
    assert not wrong_task and "task" in why
    wrong_scope, why = trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", effort="high", scope="propose:skill")
    assert not wrong_scope and "does not include" in why
    wrong_target, why = trust.check(entry, target_id="another-target", task="conv2d", provider="deepseek", model="deepseek-flash", effort="high")
    assert not wrong_target and "target" in why


def test_protocol_prompt_and_expiry_make_a_grant_stale(tmp_path):
    entry = trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path,
        protocol_version="3", prompt_sha1="prompt-a", scopes=["overlay:tip"],
    )
    assert trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", protocol_version="3", prompt_sha1="prompt-a")[0]
    ok, why = trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", protocol_version="4", prompt_sha1="prompt-a")
    assert not ok and "protocol" in why
    ok, why = trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", protocol_version="3", prompt_sha1="prompt-b")
    assert not ok and "prompt" in why
    expiring = trust.grant(
        "apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path,
        protocol_version="3", prompt_sha1="prompt-a", expires_at=time.time() + 60,
    )
    ok, why = trust.check(expiring, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", protocol_version="3", prompt_sha1="prompt-a", now=time.time() + 61)
    assert not ok and "expired" in why


def test_a_new_prompt_makes_the_existing_grant_stale_without_rewriting_it(tmp_path, monkeypatch):
    entry = trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path)
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda: entry["prompt_sha1"])
    assert trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash")[0]
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda: "different")
    assert not trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash")[0]


def test_revoke_keeps_an_append_only_audit_and_requires_the_exact_model(tmp_path):
    trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path, granted_by="alice")
    with pytest.raises(trust.TrustError):
        trust.revoke("apple-m4-max", task="conv2d", provider="deepseek", model="other", root=tmp_path)
    revoked = trust.revoke("apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", by="bob", reason="bad advice", root=tmp_path)
    assert revoked["status"] == "revoked"
    assert not trust.is_trusted("apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", root=tmp_path)[0]
    audit = trust.read_audit("apple-m4-max", tmp_path)
    assert [row["event"] for row in audit] == ["grant", "revoked"]
    assert audit[-1]["reason"] == "bad advice"
    again = trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path, granted_by="alice")
    assert again["status"] == "active"
    assert len(trust.load_grants("apple-m4-max", tmp_path)) == 1


def test_bad_scopes_and_target_ids_are_refused(tmp_path):
    with pytest.raises(trust.TrustError):
        trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", scopes=["everything"], root=tmp_path)
    with pytest.raises(trust.TrustError):
        trust.grant("../escape", "conv2d", "deepseek", "deepseek-flash", root=tmp_path)
    with pytest.raises(trust.TrustError):
        trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", expires_at=time.time() - 1, root=tmp_path)
