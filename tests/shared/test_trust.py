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
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda target_id=None: entry["prompt_sha1"])
    assert trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash")[0]
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda target_id=None: "different")
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


def test_a_grant_without_a_prompt_fingerprint_is_stale(tmp_path):
    """An old or hand-written row must not become a grant for every prompt surface."""
    from tcenv.schema import PROTOCOL_VERSION

    root = tmp_path / "targets"
    path = trust.trust_path("apple-m4-max", root)
    path.parent.mkdir(parents=True)
    path.write_text(
        "version = 1\n\n[[grants]]\n"
        'id = "hand-written"\ntarget_id = "apple-m4-max"\ntask = "conv2d"\n'
        'provider = "deepseek"\nmodel = "deepseek-flash"\neffort = ""\n'
        f'protocol_version = "{PROTOCOL_VERSION}"\nstatus = "active"\nscopes = ["overlay:tip"]\n'
    )
    entry = trust.load_grants("apple-m4-max", root)[0]
    ok, why = trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash", scope="overlay:tip", prompt_sha1="anything")
    assert not ok and "prompt" in why


def test_free_text_cannot_open_a_second_grant_table(tmp_path):
    """A task/reason is written through json.dumps, so newlines and quotes cannot forge a table."""
    import tomllib

    entry = trust.grant(
        "apple-m4-max", 'relu"\n[[grants]]\ntarget_id = "evil', "deepseek", "deepseek-flash",
        reason='x"\n[[grants]]\nstatus = "active"', root=tmp_path,
    )
    rows = trust.load_grants("apple-m4-max", tmp_path)
    assert len(rows) == 1 and rows[0]["id"] == entry["id"]
    assert tomllib.loads(trust.trust_path("apple-m4-max", tmp_path).read_text())["grants"][0]["reason"] == entry["reason"]


def test_the_fingerprint_names_the_prompt_surface():
    labels = {label for label, _ in trust.prompt_fingerprint_parts()}
    assert "kernel_agent.render_contract" in labels
    assert "harness.TEMPLATE" in labels
    assert "backends/metal/skills/msl_reference.md" in labels
    assert "taic/kernel_agent.py" not in labels and "taic/harness.py" not in labels
    assert trust.current_prompt_sha1() == trust.current_prompt_sha1()


def test_the_fingerprint_ignores_unrelated_code_in_the_prompt_modules(monkeypatch):
    """A billing or subprocess edit in kernel_agent.py must not invalidate a grant any more."""
    from taic import kernel_agent

    before = trust.current_prompt_sha1()

    def different_cost(*_a, **_k):
        return {}, False

    monkeypatch.setattr(kernel_agent, "spend_from_stream", different_cost)
    assert trust.current_prompt_sha1() == before


def test_the_fingerprint_tracks_prompt_functions(monkeypatch):
    from taic import kernel_agent

    before = trust.current_prompt_sha1()

    def changed_contract(brief, **_kw):
        return "changed TASK.md"

    monkeypatch.setattr(kernel_agent, "render_contract", changed_contract)
    assert trust.current_prompt_sha1() != before


def test_the_fingerprint_tracks_template_constants(monkeypatch):
    from taic import harness as harness_module

    before = trust.current_prompt_sha1()
    monkeypatch.setattr(harness_module, "TEMPLATE", "changed HARNESS.md")
    assert trust.current_prompt_sha1() != before


def test_backend_skills_come_from_the_target_profile_in_path_order(tmp_path, monkeypatch):
    """A second backend's skill docs must be covered by its target profile, not by editing trust.py."""
    from types import SimpleNamespace

    import tcenv.target

    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "b.md").write_text("b")
    (skills / "a.md").write_text("a")
    profile = SimpleNamespace(skills=skills, experience=tmp_path / "experience.md", device_doc=tmp_path / "device.md")
    (tmp_path / "experience.md").write_text("e")
    (tmp_path / "device.md").write_text("d")
    monkeypatch.setattr(tcenv.target, "load_target", lambda target_id=None: profile)
    assert [Path(item).name for item in trust._backend_skill_files()] == ["a.md", "b.md"]


def test_the_fingerprint_is_per_target(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tcenv.target

    profiles = {}
    for target, text in (("apple-m4-max", "default experience"), ("other", "other experience")):
        root = tmp_path / target
        root.mkdir()
        (root / "experience.md").write_text(text)
        (root / "device.md").write_text("device " + target)
        skills = root / "skills"
        skills.mkdir()
        (skills / "x.md").write_text("x")
        profiles[target] = SimpleNamespace(skills=skills, experience=root / "experience.md", device_doc=root / "device.md")
    monkeypatch.setattr(tcenv.target, "load_target", lambda target_id=None: profiles[target_id or "apple-m4-max"])
    assert trust.current_prompt_sha1("apple-m4-max") != trust.current_prompt_sha1("other")
    assert trust.current_prompt_sha1() == trust.current_prompt_sha1("apple-m4-max")


def test_a_grant_records_the_prompt_of_its_own_target(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import tcenv.target

    profiles = {}
    for target in ("apple-m4-max", "other"):
        root = tmp_path / target
        root.mkdir()
        (root / "experience.md").write_text(target)
        (root / "device.md").write_text(target)
        skills = root / "skills"
        skills.mkdir()
        (skills / "x.md").write_text(target)
        profiles[target] = SimpleNamespace(skills=skills, experience=root / "experience.md", device_doc=root / "device.md")
    monkeypatch.setattr(tcenv.target, "load_target", lambda target_id=None: profiles[target_id or "apple-m4-max"])
    entry = trust.grant("other", "conv2d", "deepseek", "deepseek-flash", root=tmp_path / "targets")
    assert entry["prompt_sha1"] == trust.current_prompt_sha1("other")
    assert entry["prompt_sha1"] != trust.current_prompt_sha1("apple-m4-max")
    ok, _why = trust.check(entry, target_id="other", task="conv2d", provider="deepseek", model="deepseek-flash", effort="")
    assert ok
