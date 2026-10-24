"""Per-target trust grants: scope, protocol, prompt, expiry, and the audit trail. No network, no GPU."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from taic import trust


def test_a_grant_is_target_and_task_scoped_and_written_to_the_target(tmp_path):
    """@brief 验证信任授予按 target 与 task 限定、落盘到该 target 的 ``trust.toml``，且校验时逐项拦截不匹配。

    正常情形通过且原因为空；task 不符、scope 不在范围（提示 ``does not include``）、
    target 不符三种情形都失败且原因提到对应字段。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    """
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
    """@brief 验证协议版本、prompt sha1 与过期时间任一不符时旧授权即失效，且原因指明哪一项。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    """
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
    """@brief 验证 prompt 指纹变化会令已有授权失效，但无需重写授权文件。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    @param monkeypatch pytest 的替换夹具，用于把 ``current_prompt_sha1`` 换成固定返回值。
    """
    entry = trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", root=tmp_path)
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda target_id=None: entry["prompt_sha1"])
    assert trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash")[0]
    monkeypatch.setattr(trust, "current_prompt_sha1", lambda target_id=None: "different")
    assert not trust.check(entry, target_id="apple-m4-max", task="conv2d", provider="deepseek", model="deepseek-flash")[0]


def test_revoke_keeps_an_append_only_audit_and_requires_the_exact_model(tmp_path):
    """@brief 验证撤销必须精确匹配 model，且审计日志只追加：先 grant 后 revoked，并记下撤销原因。

    另外确认重新 grant 后状态回到 active，且同一 target/task/model 只保留一条授权。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    @raises trust.TrustError model 与已授权的不一致时由 ``revoke`` 抛出。
    """
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
    """@brief 验证非法 scope、含路径穿越的 target id、以及已过去的过期时间都被拒绝。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    @raises trust.TrustError 上述三种非法授权请求均由 ``grant`` 抛出。
    """
    with pytest.raises(trust.TrustError):
        trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", scopes=["everything"], root=tmp_path)
    with pytest.raises(trust.TrustError):
        trust.grant("../escape", "conv2d", "deepseek", "deepseek-flash", root=tmp_path)
    with pytest.raises(trust.TrustError):
        trust.grant("apple-m4-max", "conv2d", "deepseek", "deepseek-flash", expires_at=time.time() - 1, root=tmp_path)


def test_a_grant_without_a_prompt_fingerprint_is_stale(tmp_path):
    """@brief 验证缺 prompt 指纹的旧行或手写行失效，不会变成对所有 prompt 面的永久授权。

    直接手写一份没有 ``prompt_sha1`` 的 ``trust.toml`` 行，校验时失败且原因提到 prompt。

    @param tmp_path pytest 提供的临时目录，用作手写授权文件所在的 target 根目录。
    """
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
    """@brief 验证自由文本不能伪造出第二张授权表：含换行的 task 与 reason 仍只产生一行，且内容原样往返。

    task 与 reason 都是经 ``json.dumps`` 写出的，所以引号与换行无法拼出新表头。

    @param tmp_path pytest 提供的临时目录，用作 target 根目录。
    """
    import tomllib

    entry = trust.grant(
        "apple-m4-max", 'relu"\n[[grants]]\ntarget_id = "evil', "deepseek", "deepseek-flash",
        reason='x"\n[[grants]]\nstatus = "active"', root=tmp_path,
    )
    rows = trust.load_grants("apple-m4-max", tmp_path)
    assert len(rows) == 1 and rows[0]["id"] == entry["id"]
    assert tomllib.loads(trust.trust_path("apple-m4-max", tmp_path).read_text())["grants"][0]["reason"] == entry["reason"]


def test_the_fingerprint_names_the_prompt_surface():
    """@brief 验证 prompt 指纹由具名片段组成：含契约渲染、HARNESS 模板与 MSL 参考文档，而非整个模块文件。

    同时确认同一状态下 ``current_prompt_sha1`` 结果稳定。
    """
    labels = {label for label, _ in trust.prompt_fingerprint_parts()}
    assert "kernel_agent.render_contract" in labels
    assert "harness.TEMPLATE" in labels
    assert "backends/metal/skills/msl_reference.md" in labels
    assert "taic/kernel_agent.py" not in labels and "taic/harness.py" not in labels
    assert trust.current_prompt_sha1() == trust.current_prompt_sha1()


def test_the_fingerprint_ignores_unrelated_code_in_the_prompt_modules(monkeypatch):
    """@brief 验证 prompt 模块里与提示无关的代码改动（计费或子进程相关）不再使已有授权失效。

    @param monkeypatch pytest 的替换夹具，用于把 ``kernel_agent.spend_from_stream`` 换成桩。
    """
    from taic import kernel_agent

    before = trust.current_prompt_sha1()

    def different_cost(*_a, **_k):
        return {}, False

    monkeypatch.setattr(kernel_agent, "spend_from_stream", different_cost)
    assert trust.current_prompt_sha1() == before


def test_the_fingerprint_tracks_prompt_functions(monkeypatch):
    """@brief 验证改动 prompt 函数（``kernel_agent.render_contract``）会让指纹变化。

    @param monkeypatch pytest 的替换夹具，用于插入返回固定文本的契约渲染桩。
    """
    from taic import kernel_agent

    before = trust.current_prompt_sha1()

    def changed_contract(brief, **_kw):
        return "changed TASK.md"

    monkeypatch.setattr(kernel_agent, "render_contract", changed_contract)
    assert trust.current_prompt_sha1() != before


def test_the_fingerprint_tracks_template_constants(monkeypatch):
    """@brief 验证改动 HARNESS 模板常量会让指纹变化。

    @param monkeypatch pytest 的替换夹具，用于替换 ``harness.TEMPLATE``。
    """
    from taic import harness as harness_module

    before = trust.current_prompt_sha1()
    monkeypatch.setattr(harness_module, "TEMPLATE", "changed HARNESS.md")
    assert trust.current_prompt_sha1() != before


def test_backend_skills_come_from_the_target_profile_in_path_order(tmp_path, monkeypatch):
    """@brief 验证其他 backend 的 skill 文档由 target profile 提供，并按路径顺序排好，无需改 trust.py。

    @param tmp_path pytest 提供的临时目录，用来放 skills 目录与 experience/device 文档。
    @param monkeypatch pytest 的替换夹具，用于让 ``load_target`` 返回构造好的 profile。
    """
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
    """@brief 验证指纹按 target 区分：不同 target 的 experience/device/skill 给出不同指纹，默认取 apple-m4-max。

    @param tmp_path pytest 提供的临时目录，用来为每个 target 建立文档与 skills 目录。
    @param monkeypatch pytest 的替换夹具，用于让 ``load_target`` 按 target id 返回对应 profile。
    """
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
    """@brief 验证授权记录的是它所属 target 自己的 prompt 指纹，而不是默认 target 的指纹。

    @param tmp_path pytest 提供的临时目录，用来为每个 target 建立文档与 skills 目录。
    @param monkeypatch pytest 的替换夹具，用于让 ``load_target`` 按 target id 返回对应 profile。
    """
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
