"""B3：``msl-bundle`` 题面与受信面。

Two render surfaces carry the instructions the agent follows: the system prompt
(``prompts.system_prompt_v2``) and ``TASK.md`` (``kernel_agent.render_contract``).  A bundle cannot
keep the single-file launch sentences on either of them, and the single-file text must stay
byte-identical apart from the one corrected zero-fill sentence.  The second half of the file proves
that every sentence the bundle adds is actually inside the hashed prompt surface.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from taic import kernel_agent, prompts, trust
from taic.kernel_agent import SessionBrief, render_contract
from taic.llm import LLMConfig, SearchConfig
from taic.orchestrator import Orchestrator
from taic.task_binding import bind_task
from taic.v2_evaluation import V2LocalEvaluator
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tests.support.fakes import FakeClient, make_response
from tests.support.shape_portfolio_metal import write_p1_task

OP = "shape_portfolio_pointwise"

#: The corrected zero-fill sentence (G14) and the sentence it replaced in both engines.
NEW_ZERO_SENTENCE = (
    "Every workspace you declare with `zero` is filled with zeros inside the timed command buffer, on every iteration,"
)
OLD_ZERO_SENTENCE = "Every workspace is filled with zeros inside the timed command buffer, on every iteration, before your stages."
CHINESE_NEW_ZERO = (
    "声明 `zero` 的工作区在每次计时迭代里由评测填零，这次填零算进你的时间；"
    "没声明 `zero` 的工作区保留你的 Host Tiling 计划写进去的内容。"
)
CHINESE_OLD_ZERO = "工作区在每次计时迭代里由评测填零，这次填零算进你的时间。"

#: The four English sentences a bundle must not carry (they belong to msl-single).
STALE_RULES = (
    "its own grid, threads, optional dynamic threadgroup bytes and optional workspace buffers",
    "With the block, each\n   stage's grid and threads come from the block.",
    "Return the\n   complete source text of everything you define.",
    "workspace tensors are not function parameters",
)


def _binding(tmp_path: Path, *, track_id: str) -> object:
    root = tmp_path / "tasks"
    write_p1_task(root)
    return replace(bind_task(OP, "fp16", "threshold", tasks_root=root), track_id=track_id)


def _orchestrator(tmp_path: Path, *, track_id: str):
    binding = _binding(tmp_path, track_id=track_id)

    class _Never(V2LocalEvaluator):
        def _evaluate(self, request):  # pragma: no cover
            raise AssertionError("prompt tests never evaluate")

    return Orchestrator(
        OP, "fp16", "threshold",
        LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False, truncation_retry=False),
        SearchConfig(budget_usd=1.0, branches=1, max_rounds=0, correctness_only=True, eval_reps=3),
        run_dir=tmp_path / "run", evaluator=_Never(binding),
        client=FakeClient(lambda messages, **kw: make_response("x")),
        log=lambda s: None, track_id=track_id, binding=binding,
    )


@pytest.fixture(scope="module")
def bundle_texts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundle")
    orch = _orchestrator(tmp, track_id="msl-bundle")
    system = prompts.system_prompt_v2(orch.binding, use_tools=False, target_id=orch.target_id)
    brief = orch._session_brief(seedless=True, label="init", work_dir=tmp / "w")
    return orch, system, brief, render_contract(brief)


@pytest.fixture(scope="module")
def single_texts(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("single")
    orch = _orchestrator(tmp, track_id="msl-single")
    system = prompts.system_prompt_v2(orch.binding, use_tools=False, target_id=orch.target_id)
    brief = orch._session_brief(seedless=True, label="init", work_dir=tmp / "w")
    return orch, system, brief, render_contract(brief)


# ----------------------------------------------------------------- ①–⑥ the bundle prompt content


def test_the_bundle_prompt_carries_the_geometry_contract(bundle_texts):
    _orch, system, _brief, _task_md = bundle_texts
    assert "msl-bundle" in system
    for fragment in (
        "`schema_version`", "`source`: the string `candidate`", "`stages`: a non-empty ordered list",
        "`kernel`, `grid`, `threads`", "`threadgroup_bytes`", "`when`", "`derived_scalars`",
        '"schema_version": 1', '"source": "candidate"',
        "Multi-stage means several objects in `stages`",
        "dynamic* threadgroup",
        "dispatch is taken from the geometry entry",
        "A workspace tensor is bound by name as a kernel parameter",
        "must not be `device const`",
        "only the workspaces a stage declares",
    ):
        assert fragment in system, f"the bundle prompt is missing {fragment!r}"


def test_the_bundle_prompt_forbids_the_two_mistakes_that_get_refused(bundle_texts):
    _orch, system, _brief, _task_md = bundle_texts
    assert "dynamic threadgroup" in system and "must be 0" in system
    for stale in STALE_RULES:
        assert stale not in system, f"a single-file sentence survived in the bundle prompt: {stale!r}"
    # …and the replacements are present, so the deletion is not just an empty area.
    assert "candidate geometry entry" in system
    assert "submit the envelope" in system


def test_the_bundle_task_md_stops_sending_the_author_to_the_launch_block(bundle_texts):
    _orch, _system, brief, task_md = bundle_texts
    assert brief.track_id == "msl-bundle"
    assert "外层不会替你选 grid/threads" not in task_md
    assert "dispatch 由 `host/execution_geometry.json` 决定" in task_md
    assert "只承担 `math fast|safe` 与 workspace↔ABI 交叉检查" in task_md
    # The packaging path is actionable: the command names the real submission file.
    assert "scripts/msl_bundle_pack.py" in task_md
    assert "--track msl-bundle" in task_md


def test_the_bundle_signature_skeleton_keeps_the_workspace_slot(bundle_texts):
    orch, system, brief, task_md = bundle_texts
    # ``scratch`` is the fixture task's fp32 workspace: it must appear as a writable buffer parameter.
    assert "device float* scratch [[buffer(2)]]" in system
    assert "device float* scratch [[buffer(2)]]" in brief.abi_stub
    assert "device float* scratch [[buffer(2)]]" in task_md
    assert orch.binding.track_id == "msl-bundle"


def test_the_bundle_prompt_drops_the_wrong_workspace_sentence(bundle_texts):
    _orch, system, _brief, _task_md = bundle_texts
    assert "workspace tensors are not function parameters" not in system


def test_varlen_kda_real_tier_bundle_prompt_and_task_md_teach_host_tiling_c(tmp_path):
    binding = replace(bind_task("varlen_kda", "fp16", "v6_h64_fixed"), track_id="msl-bundle")
    stub = binding.signature_stub()
    assert "device const int* cu_seqlens [[buffer(8)]]" in stub
    assert "device uint* work_items [[buffer(14)]]" in stub
    assert "device int* item_counts [[buffer(15)]]" in stub
    assert "device int* handoff_flags [[buffer(16)]]" in stub
    assert "device float* handoff_states [[buffer(17)]]" in stub
    system = prompts.system_prompt_v2(binding, use_tools=False, target_id="apple-m4-max")
    for fragment in (
        "host/tiling.c",
        "Host Tiling (`host/tiling.c` vs `host/tiling.json`, mutually exclusive)",
        "tcenv.cli host-probe",
        "each stage uses contiguous slots `0..n-1`",
        "## Host Program (restricted C99 -> native64 worker, optional)",
        "Generated `host_contract.h` (full ABI)",
    ):
        assert fragment in system, f"varlen_kda bundle system prompt missing {fragment!r}"
    presentation = binding.presentation()
    brief = SessionBrief(
        repo=REPO_ROOT,
        task_name="varlen_kda",
        dtype="fp16",
        size="v6_h64_fixed",
        kernel_name="varlen_kda_kernel",
        triton_source="",
        description=presentation.statement,
        contract_text="{}",
        abi_stub=stub,
        rtol=0.01,
        atol_rel=0.005,
        v2=True,
        reference_source=presentation.reference_source,
        seedless=False,
        seed_source="kernel void varlen_kda_kernel() {}\n",
        scorecard="STATUS: ok | speedup=1.20x",
        failure="prev round compile error",
        notes_summary="round 1 notes",
        peer_tips="- [verified] tip 1",
        label="r2.p0",
        work_dir=str(tmp_path / "work"),
        track_id="msl-bundle",
    )
    task_md = render_contract(brief)
    assert "host/tiling.c" in task_md
    assert "tcenv.cli host-probe varlen_kda --source <你的 src 目录>/host/tiling.c --dtype fp16 --size v6_h64_fixed" in task_md
    assert "scripts/msl_bundle_pack.py" in task_md
    assert "每个 stage 的 `[[buffer(0..n-1)]]` 槽位必须从 0 连续编号" in task_md
    # Round 2+ dynamic feedback blocks survive even though varlen_kda's static statement + host_contract.h > 12KB.
    assert "外层成绩单：" in task_md and "STATUS: ok | speedup=1.20x" in task_md
    assert "上一轮外层失败反馈：" in task_md and "prev round compile error" in task_md
    assert "上一轮 NOTES" in task_md and "round 1 notes" in task_md



# ------------------------------------------------------------------ ② + ⑧ the single-file surface


def test_the_single_prompt_has_none_of_the_bundle_section(single_texts):
    _orch, system, brief, task_md = single_texts
    assert brief.track_id == "msl-single"
    assert "msl-bundle" not in system
    assert "host/execution_geometry.json" not in system
    assert "scripts/msl_bundle_pack.py" not in task_md
    # The single-file launch instruction is still the right one there.
    assert "外层不会替你选 grid/threads" in task_md
    # …and the single-file skeleton drops the workspace slot, as before.
    assert "device float* scratch" not in system


def test_the_single_prompt_is_byte_identical_apart_from_the_zero_fill_fix(single_texts):
    _orch, system, _brief, task_md = single_texts
    assert NEW_ZERO_SENTENCE in system
    assert OLD_ZERO_SENTENCE not in system
    assert CHINESE_NEW_ZERO in task_md
    assert CHINESE_OLD_ZERO not in task_md
    # The four bundle-conflict sentences are untouched for a single-file delivery.
    assert "With the block, each\n   stage's grid and threads come from the block." in system
    assert "workspace tensors are not function parameters" in system


def test_no_surface_carries_both_sides_of_a_conflict(bundle_texts):
    _orch, system, _brief, task_md = bundle_texts
    for surface in (system, task_md):
        assert not (
            "grid and threads come from the block" in surface and "candidate geometry entry" in surface
        )
        assert not ("scripts/msl_bundle_pack.py" in surface and "exactly one Metal code block" in surface)


def test_the_signature_stub_takes_no_track_argument():
    from taic.task_binding import TaskBinding

    assert list(inspect.signature(TaskBinding.signature_stub).parameters) == ["self"]


# --------------------------------------------------------------------------- 受信面（trust）


def _blob() -> bytes:
    return b"\0".join(data for _label, data in trust.prompt_fingerprint_parts())


def test_every_new_model_visible_sentence_is_inside_the_fingerprint():
    blob = _blob()
    for sentence in (
        "candidate geometry entry",
        "submit the envelope",
        "dynamic* threadgroup",
        "A workspace tensor is bound by name as a kernel parameter",
        "Multi-stage means several objects in `stages`",
        NEW_ZERO_SENTENCE,
        "scripts/msl_bundle_pack.py",
        "--track msl-bundle",
        "device float* scale_tab [[buffer(2)]]",
        "Host Tiling (`host/tiling.c` vs `host/tiling.json`, mutually exclusive)",
        "tcenv.cli host-probe",
        "each stage uses contiguous slots `0..n-1`",
    ):
        assert sentence.encode("utf-8") in blob, (
            f"{sentence!r} is not in the hashed prompt surface; it must live in a registered function body"
        )


def test_the_trusted_labels_include_the_three_new_symbols():
    labels = {label for label, _ in trust.prompt_fingerprint_parts()}
    assert {
        "task_binding.TaskBinding.signature_stub",
        "prompts.system_prompt_v2",
        "kernel_agent._check_command",
    } <= labels


def test_only_the_method_source_is_hashed_for_the_dotted_symbol():
    from taic.task_binding import TaskBinding

    data = trust._symbol_bytes("taic.task_binding", "TaskBinding.signature_stub")
    assert data == inspect.getsource(TaskBinding.signature_stub).encode("utf-8")
    assert b"def bind_task" not in data  # not the whole module


def test_editing_the_signature_stub_source_changes_the_fingerprint(monkeypatch):
    baseline = trust.current_prompt_sha1()
    real = inspect.getsource

    def fake(obj, *args, **kwargs):
        text = real(obj, *args, **kwargs)
        if getattr(obj, "__qualname__", "").endswith("TaskBinding.signature_stub"):
            return text.replace("def signature_stub", "def signature_stub ", 1)
        return text

    monkeypatch.setattr(inspect, "getsource", fake)
    assert trust.current_prompt_sha1() != baseline
    monkeypatch.undo()
    assert trust.current_prompt_sha1() == baseline


@pytest.mark.parametrize("qualname", ["_rules_for", "system_prompt_v2"])
def test_editing_a_registered_prompt_function_changes_the_fingerprint(monkeypatch, qualname):
    baseline = trust.current_prompt_sha1()
    real = inspect.getsource

    def fake(obj, *args, **kwargs):
        text = real(obj, *args, **kwargs)
        if getattr(obj, "__qualname__", "") == qualname:
            # A one-character edit anywhere in the body is enough: the whole body is hashed.
            return text.replace("{", "{ ", 1)
        return text

    monkeypatch.setattr(inspect, "getsource", fake)
    assert trust.current_prompt_sha1() != baseline
    monkeypatch.undo()
    assert trust.current_prompt_sha1() == baseline


def test_editing_an_unrelated_symbol_does_not_change_the_fingerprint(monkeypatch):
    from taic import task_binding

    baseline = trust.current_prompt_sha1()
    real = inspect.getsource

    def fake(obj, *args, **kwargs):
        text = real(obj, *args, **kwargs)
        if getattr(obj, "__qualname__", "").endswith("bind_task"):
            return text.replace("def bind_task", "def bind_task ", 1)
        return text

    monkeypatch.setattr(inspect, "getsource", fake)
    assert trust.current_prompt_sha1() == baseline
    monkeypatch.undo()
    assert task_binding.bind_task is not None


def test_a_misspelled_dotted_label_raises_instead_of_hashing_nothing():
    with pytest.raises(ValueError, match="does not resolve"):
        trust._symbol_bytes("taic.task_binding", "TaskBinding.no_such_method")
