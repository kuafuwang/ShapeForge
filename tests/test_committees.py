"""Two committees hand off at every round boundary. Scripted model, no GPU, no paid call."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from taic.committees import Committee, Member, handoff_for, load_committees
from taic.llm import LLM, LLMConfig, LLMError, SearchConfig
from taic.orchestrator import Orchestrator, RunStartError

from .fakes import FakeClient
from .test_orchestrator import agent_from_script, code, fake_eval, llm_script, run

SECRET = "sk-committee-secret-0123456789"
ENDPOINT = "http://committee-endpoint.invalid"
FEEDBACK = "Evaluation feedback of this kernel:"
HANDOFF = "the previous turn's conversation is not included"


def _cfg(model: str = "fake", **kw) -> LLMConfig:
    base = dict(base_url=ENDPOINT, api_key=SECRET, model=model, reasoning_effort="off", use_tools=False, truncation_retry=False)
    base.update(kw)
    return LLMConfig(**base)


def _search(**kw) -> SearchConfig:
    fields = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False,
                  budget_usd=5.0, branches=1, max_rounds=2, max_repair_tries=1, eval_reps=3, patience=4, kernel_agent="pi")
    fields.update(kw)
    return SearchConfig(**fields)


def _pair(cfg: LLMConfig, first: str, second: str, *, second_cfg: LLMConfig | None = None) -> list[Committee]:
    left = Committee("first", [Member("deepseek", first, replace(cfg, model=first))])
    right_cfg = second_cfg if second_cfg is not None else replace(cfg, model=second)
    right = Committee("second", [Member("deepseek", second, right_cfg)])
    return [left, right]


def _orch(tmp_path, script, committees, cfg, *, search=None, client=None, log=None):
    client = client if client is not None else FakeClient(script)
    orch = Orchestrator(
        "relu", "fp16", "small", cfg, search or _search(),
        run_dir=tmp_path / "run", evaluate_fn=fake_eval([]), client=client, log=log or (lambda _s: None), committees=committees,
        agent_runner=agent_from_script(script) if script is not None else None,
    )
    return orch, client


def _kinds(client) -> list[tuple[str, str]]:
    out = []
    for call in client.calls:
        text = call["messages"][-1]["content"]
        if "Translate this Triton kernel" in text:
            kind = "init"
        elif "Diagnose the dominant bottleneck" in text:
            kind = "plan"
        elif "Implement exactly this optimization plan" in text:
            kind = "patch"
        elif "evaluation of your kernel failed" in text:
            kind = "repair"
        else:
            kind = "other"
        out.append((kind, call["model"]))
    return out


def test_different_models_hand_off_a_fresh_plan(tmp_path):
    cfg = _cfg("configured")
    committees = _pair(cfg, "model-a", "model-b")
    logs: list[str] = []
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=1.1")})
    orch, client = _orch(tmp_path, script, committees, cfg, log=logs.append)
    run(orch)
    events = [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines() if '"event": "agent"' in line]
    assert [(e["committee"], e["model"]) for e in events] == [("first", "model-a"), ("first", "model-a"), ("second", "model-b")]
    assert client.calls == []
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["model"] == "model-b" and state["committee"] == "second"
    text = "\n".join(logs)
    assert SECRET not in text and ENDPOINT not in text
    assert not (tmp_path / "run" / "NOTES.md").exists()


def test_same_model_still_hands_off_without_a_second_client(tmp_path, monkeypatch):
    made = []
    real = LLM

    def spy(*args, **kwargs):
        obj = real(*args, **kwargs)
        made.append(obj)
        return obj

    monkeypatch.setattr("taic.orchestrator.LLM", spy)
    cfg = _cfg("shared-model")
    committees = _pair(cfg, "shared-model", "shared-model")
    script = llm_script(code("speed=1.0"), patches={"A": code("speed=1.1")})
    orch, client = _orch(tmp_path, script, committees, cfg)
    run(orch)
    assert len(made) == 1 and made[0].client is client and orch.llm.client is client
    events = [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines() if '"event": "agent"' in line]
    assert [e["committee"] for e in events] == ["first", "first", "second"]
    assert all(e["model"] == "shared-model" for e in events)
    assert orch.ledger.cfg is cfg  # same model id does not replace the price config


def test_two_members_are_refused_before_any_model_call(tmp_path):
    cfg = _cfg()
    member = Member("deepseek", "model-a", replace(cfg, model="model-a"))
    extra = Member("deepseek", "model-b", replace(cfg, model="model-b"))
    committees = [
        Committee("first", [member, extra]),
        Committee("second", [Member("deepseek", "model-a", replace(cfg, model="model-a"))]),
    ]
    client = FakeClient(lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("model was called")))
    with pytest.raises(RunStartError, match="2 members") as raised:
        Orchestrator("relu", "fp16", "small", cfg, _search(), run_dir=tmp_path / "run", client=client, log=lambda _s: None, committees=committees)
    assert "not reduced to its first member" in str(raised.value)
    assert client.calls == []
    assert not (tmp_path / "run").exists()


def test_run_config_lists_both_committees_without_secrets(tmp_path):
    cfg = _cfg("configured")
    committees = _pair(cfg, "model-a", "model-b")
    client = FakeClient([])
    orch, _ = _orch(tmp_path, None, committees, cfg, client=client)
    text = (orch.run_dir / "run_config.json").read_text()
    assert SECRET not in text and ENDPOINT not in text and "api_key" not in text and "base_url" not in text
    doc = json.loads(text)
    assert doc["committees"] == [
        {"name": "first", "members": [{"provider": "deepseek", "model": "model-a"}]},
        {"name": "second", "members": [{"provider": "deepseek", "model": "model-b"}]},
    ]
    state = json.loads((orch.run_dir / "state.json").read_text())
    assert state["committee"] == "first" and state["model"] == "model-a"
    assert client.calls == []


def test_round_1_repair_stays_on_the_first_committee(tmp_path):
    cfg = _cfg("configured")
    committees = _pair(cfg, "model-a", "model-b")
    script = llm_script(code("speed=1.0"), patches={"A": code("status=compile_error")}, fixes=[code("speed=1.2")])
    orch, client = _orch(tmp_path, script, committees, cfg, search=_search(max_rounds=2))
    res = run(orch)
    assert res.best_label == "init"
    events = [json.loads(line) for line in (tmp_path / "run" / "transcript.jsonl").read_text().splitlines() if '"event": "agent"' in line]
    assert [(e["label"], e["committee"]) for e in events] == [("init", "first"), ("r1.p0", "first"), ("r2.p0", "second")]
    assert "boom" in orch._session_briefs[-1].failure
    assert client is orch.llm.client


def test_a_different_endpoint_opens_another_client_on_the_same_ledger(tmp_path, monkeypatch):
    opened = []

    class Dummy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            opened.append(kwargs["base_url"])

        async def close(self):
            opened.append("close")

    monkeypatch.setattr("openai.AsyncOpenAI", Dummy)
    cfg = _cfg("model-a")
    other = replace(cfg, base_url="http://other-endpoint.invalid", api_key="sk-other-secret-0123456789", model="model-b", price_out=4.0)
    committees = _pair(cfg, "model-a", "model-b", second_cfg=other)
    orch = Orchestrator(
        "relu", "fp16", "small", cfg, _search(max_rounds=0),
        run_dir=tmp_path / "run", evaluate_fn=fake_eval([]), log=lambda _s: None, committees=committees,
    )
    assert opened == [ENDPOINT]
    orch._activate_round(2)
    assert opened == [ENDPOINT, "http://other-endpoint.invalid"]
    assert orch.llm.ledger is orch._llms[0].ledger
    assert orch.llm.transcript is orch._llms[0].transcript
    assert orch.ledger.cfg.model == "model-b" and orch.ledger.cfg.price_out == 4.0
    assert orch._committee_name == "second"
    text = (orch.run_dir / "run_config.json").read_text()
    assert ENDPOINT not in text and "http://other-endpoint.invalid" not in text
    assert SECRET not in text and "sk-other-secret-0123456789" not in text
    import asyncio
    asyncio.run(orch._close_clients())
    assert opened.count("close") == 2


def test_the_model_picker_rewrites_both_committee_members():
    from observe.settings import _replace_in_members

    text = (
        '[llm]\nprovider = "deepseek"\nmodel = "deepseek-flash"  # keep\n\n'
        '[[committees]]\nname = "first"\n\n[[committees.members]]\nprovider = "deepseek"\nmodel = "deepseek-flash"\n\n'
        '[[committees]]\nname = "second"\n\n[[committees.members]]\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
    )
    out = _replace_in_members(text, "other", "big-model")
    assert out.count('provider = "other"') == 2 and out.count('model = "big-model"') == 2
    assert 'model = "deepseek-flash"  # keep' in out


def test_a_listed_pair_with_two_members_is_refused_before_credentials(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("credentials file was read")

    monkeypatch.setattr("taic.committees._providers", boom)
    cfg = _cfg("deepseek-flash")
    raw = {"llm": {"provider": "deepseek", "credentials_file": "/does/not/matter"}, "committees": [
        {"name": "first", "members": [{"provider": "deepseek", "model": "deepseek-flash"}, {"provider": "other", "model": "big"}]},
        {"name": "second", "members": [{"provider": "deepseek", "model": "deepseek-flash"}]},
    ]}
    with pytest.raises(LLMError, match="2 members"):
        load_committees(raw, active=cfg)


def test_cmd_run_and_run_one_load_the_two_members(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    monkeypatch.setenv("OPENAI_BASE_URL", ENDPOINT)
    toml = tmp_path / "taic.toml"
    toml.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        'use_tools = false\nreasoning_effort = "off"\n'
        "[search]\nbudget_usd = 5\nbranches = 1\nmax_rounds = 0\neval_reps = 3\n"
        "stop_at_roofline = false\nabi_guard = false\ntarget_note = false\nrich_history = false\nstatus_lines = false\n"
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n'
    )
    from experiments.common import run_one
    from taic import cli

    script = llm_script(code("speed=1.0"))
    client = FakeClient(script)
    res = run_one("relu", "fp16", "small", tmp_path / "ran", config_path=str(toml), client=client, evaluate_fn=fake_eval([]), log=lambda _s: None, preset="base", agent_runner=agent_from_script(script))
    assert res.best_label == "init"
    doc = json.loads((tmp_path / "ran" / "run_config.json").read_text())
    assert [m["model"] for c in doc["committees"] for m in c["members"]] == ["model-a", "model-b"]
    assert SECRET not in (tmp_path / "ran" / "run_config.json").read_text()
    agents = [json.loads(line) for line in (tmp_path / "ran" / "transcript.jsonl").read_text().splitlines() if '"event": "agent"' in line]
    assert [(e["label"], e["model"]) for e in agents] == [("init", "model-a")]

    captured = {}

    class Stub:
        def __init__(self, *args, **kwargs):
            captured["committees"] = kwargs.get("committees")

        async def run(self):
            from types import SimpleNamespace
            return SimpleNamespace(task="relu", dtype="fp16", model="model-a", best_speedup=1.0, confirmed_speedup=None, spent_usd=0.0, stop_reason="max_rounds", run_dir=str(tmp_path))

    monkeypatch.setattr("taic.orchestrator.Orchestrator", Stub)
    code_ok = cli.main(["run", "relu", "--size", "small", "--config", str(toml), "--run-dir", str(tmp_path / "stub")])
    assert code_ok == 0
    assert [(c.name, c.members[0].model) for c in captured["committees"]] == [("first", "model-a"), ("second", "model-b")]
    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[llm]\ncredentials_file = "/no/such/models.json"\nprovider = "deepseek"\nmodel = "deepseek-flash"\n'
        '[[committees]]\nname = "first"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-a"\n'
        '[[committees.members]]\nprovider = "other"\nmodel = "big"\n'
        '[[committees]]\nname = "second"\n[[committees.members]]\nprovider = "deepseek"\nmodel = "model-b"\n'
    )
    captured.clear()
    code_bad = cli.main(["run", "relu", "--size", "small", "--config", str(bad), "--run-dir", str(tmp_path / "bad-run")])
    assert code_bad == 5 and captured == {}


def test_a_member_can_carry_its_own_reasoning_effort():
    active = _cfg("same-model")
    raw = {
        "llm": {"provider": "deepseek"},
        "committees": [
            {"name": "first", "members": [{"provider": "deepseek", "model": "same-model", "reasoning_effort": "high"}]},
            {"name": "second", "members": [{"provider": "deepseek", "model": "same-model", "reasoning_effort": "low"}]},
        ],
    }
    committees = load_committees(raw, active=active)
    assert [c.members[0].cfg.reasoning_effort for c in committees] == ["high", "low"]

