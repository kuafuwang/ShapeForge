"""``search.hw_feedback``: off = byte-identical prompts and feedback (golden hashes of the hw_feedback-off path), on = a short honest block
in the seed feedback of the planner and the patch authors plus Skill 2c. Scripted LLM and evaluator: no network, no GPU (the AIR compile of the block is
the Metal compiler front end only and is skipped gracefully without the toolchain)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from taic import cli, hwfeatures as hw, hwfeedback as hf, prompts, resume as rs
from taic.llm import SearchConfig, load_config
from tasks import get_task
from tcenv.evaluate import EvalReport

from ...support import hw_golden as G
from ...support.fakes import FakeClient, make_response
from ...support.orchestrator import code, fake_eval, llm_script, make_orch, run
from ...support.resume import mark_dead, ns, script_for

REPO = Path(__file__).resolve().parents[3]
HEADER = "Hardware-feature analysis of this kernel"
SKILL_TITLE = "# Skill 2c: reading the hardware-feature analysis in the feedback"
CAVEAT = "Missing items are hints from static analysis, not requirements: the measured time decides."


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """@brief 把硬件特征分析的磁盘缓存与内存缓存都指向临时目录，避免用例之间互相污染。

    @param tmp_path pytest 提供的临时目录，用作 ``hf.CACHE_DIR``。
    @param monkeypatch pytest 的替换夹具，负责在用例结束后恢复原值。
    """
    monkeypatch.setattr(hf, "CACHE_DIR", tmp_path / "hwcache")
    hf._CACHE.clear()


def block(name: str, mutate=None, source: str | None = None, task: str | None = None) -> str | None:
    """@brief 用金样本 fixture 构造一份硬件特性反馈块，可选先对报告做变换。

    @param name 金样本名（``G.fixture`` 的键）。
    @param mutate 可选回调，接收并原地修改报告字典。
    @param source 可选，覆盖使用的 kernel 源码。
    @param task 可选，覆盖使用的 task 名（如带变体后缀的 gemm）。
    @return ``hf.hw_feedback`` 生成的文本块；不适用时为 None。
    """
    idx, src, rep = G.fixture(name)
    if mutate:
        mutate(rep)
    t = get_task(task or idx["task"], idx["dtype"], idx["size"])
    return hf.hw_feedback(t, src if source is None else source, EvalReport.from_json(rep), backend_id="metal")


def test_a_non_metal_backend_does_not_run_the_metal_analysis():
    """@brief 验证 backend_id 不是 metal 时直接返回 None，不跑 Metal 专用的硬件特性分析。"""
    idx, src, rep = G.fixture("fa_flash_best")
    t = get_task(idx["task"], idx["dtype"], idx["size"])
    assert hf.hw_feedback(t, src, EvalReport.from_json(rep), backend_id="other") is None


def lines_of(text: str) -> list[str]:
    """@brief 把文本按行拆开，便于统计行数与定位条目。

    @param text 输入文本。
    @return 保持原顺序的行列表。
    """
    return text.splitlines()


# ------------------------------------------------------------------------------------------------ switch off: byte identity


def _diff(got: dict, want: dict) -> list[str]:
    """@brief 列出两层字典之间的差异键，用于判断是否与金样本字节级一致。

    @param got 本次计算结果。
    @param want 金样本内容。
    @return 发生变化的键列表；空列表表示一致。
    """
    return G.changed_keys(got, want)


def test_switch_off_is_byte_identical_to_the_behaviour_recorded_before_the_feature():
    """@brief 验证关闭开关时生成的提示词与反馈与功能上线前记录的金样本完全一致。

    先确认金样本各段（系统提示词、用户消息、反馈、出站请求、编排器配置）长度足够，再比对差异键为空。
    """
    golden = json.loads(G.GOLDEN.read_text())
    assert len(golden["system_prompt"]) >= 100 and len(golden["user_messages"]) >= 100 and len(golden["feedback"]) >= 12 and len(golden["outbound"]) >= 4 and len(golden["orchestrator"]) == 4
    assert _diff(G.compute(), golden) == []


def test_the_golden_file_is_exactly_what_the_generator_writes():
    """@brief 验证金样本文件就是生成器的输出：``_meta`` 三个字段非空，正文逐字节等于重新生成的结果。"""
    disk = json.loads(G.GOLDEN.read_text())
    meta = disk.pop("_meta")
    for key in ("recorded_at", "git_commit", "reason"):
        assert meta.get(key)
    assert G.GOLDEN.read_text() == G.dumps(G.compute(), meta)


def test_the_request_whitelist_is_exactly_what_params_sends():
    """@brief 验证请求参数白名单与真实 params 发出的键集合完全一致。"""
    assert G.param_keys() == set(G.REQUEST_KEYS)


def test_fixture_bytes_match_the_index_digest():
    """@brief 验证每个 fixture 文件的字节与 ``index.json`` 里记录的 sha256 一致。"""
    index = json.loads((G.FIXTURES / "index.json").read_text())
    assert index
    for name, entry in index.items():
        assert entry["sha256"] == G.fixture_digest(name)


def test_switch_off_with_the_argument_spelled_out_is_the_same_thing():
    """@brief 验证把开关显式写成 False（系统提示与搜索配置各一处）与默认关闭的结果完全一致。"""
    golden = json.loads(G.GOLDEN.read_text())
    assert _diff(G.compute(extra_system={"hw_feedback": False}, extra_search={"hw_feedback": False}), golden) == []


def test_the_config_files_are_untouched_and_the_default_is_off():
    """@brief 验证 ``SearchConfig`` 默认关闭该开关，且仓库的两个配置里都不出现这个键（配置未被改动）。"""
    assert SearchConfig().hw_feedback is False
    assert "hw_feedback" not in (REPO / "taic.toml").read_text() and "hw_feedback" not in (REPO / "taic.paper.toml").read_text()


def test_load_config_turns_it_on_only_when_asked(monkeypatch):
    """@brief 验证 ``load_config`` 只在显式要求时打开开关，默认与 paper 预设都保持关闭。

    @param monkeypatch pytest 的环境变量夹具，用于注入假的 API key 与 base url。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://x")
    assert load_config(None, preset="base")[1].hw_feedback is False
    assert load_config(None, preset="base", hw_feedback=True)[1].hw_feedback is True
    assert load_config(None, preset="paper")[1].hw_feedback is False


# ------------------------------------------------------------------------------------------------ Skill 2c


def test_skill_2c_is_in_the_prompt_only_with_the_switch():
    """@brief 验证开关打开时只在系统提示词里插入完整的 Skill 2c 段落，其余内容字节不变。

    两个 task × use_tools × profiler_evidence 的组合都检查：关闭时无 Skill 2c，打开时多了该段且
    去掉该段后与关闭版相同；段落位置在 Skill 2 与 Skill 3 之间，开启 profiler 后位于 Skill 2b 之后。
    """
    skill = (REPO / "backends" / "metal" / "skills" / "hardware_feedback.md").read_text()
    section = "\n\n" + SKILL_TITLE + "\n" + skill  # the part starts with a newline and "\n".join adds another one
    for name in ("relu", "flash_attention"):
        task = get_task(name, "fp16", "paper")
        for tools in (False, True):
            for pe in (False, True):
                off = prompts.system_prompt(task, use_tools=tools, profiler_evidence=pe)
                on = prompts.system_prompt(task, use_tools=tools, profiler_evidence=pe, hw_feedback=True)
                assert "Skill 2c" not in off and section in on
                assert on.replace(section, "", 1) == off  # the switch only inserts the section
                assert on.index("# Skill 2") < on.index(SKILL_TITLE) < on.index("# Skill 3")
                if pe:
                    assert on.index("# Skill 2b") < on.index(SKILL_TITLE)


def test_the_skill_is_short_and_says_what_it_must():
    """@brief 验证技能文档简短（不超过 40 行）、措辞包含 static/hint/measured time decides 等要点，且不含代码块。"""
    text = (REPO / "backends" / "metal" / "skills" / "hardware_feedback.md").read_text()
    assert len(text.splitlines()) <= 40
    low = text.lower()
    for must in ("static", "hint", "not a measurement", "measured time decides", "90%", "roofline", "register pressure"):
        assert must in low, must
    assert "```" not in text  # no code: nothing for the skill compile test to chew on


# ------------------------------------------------------------------------------------------------ the block


def test_block_of_a_good_attention_kernel_lists_what_it_uses_and_is_short():
    """@brief 验证优质 attention kernel 的反馈块：标题与 caveat 齐全、长度受限、列出已用特性与性能数据。

    要求列出 simdgroup、online softmax、寄存器预取、16B 加载与 shuffle，给出 66% 峰值与 1.53x 下界，
    并且因为特性齐备而不出现缺失列表。
    """
    b = block("fa_flash_best")
    assert b.startswith(HEADER) and b.rstrip().endswith(CAVEAT)
    L = lines_of(b)
    assert len(L) <= 12 and len(b.split()) <= 170
    assert "simdgroup_matrix 8x8 (half in, float acc, 10 fragments per SIMD group)" in b
    for w in ("online softmax", "register prefetch", "operand fragments in registers", "device 16 B loads", "simd_shuffle_xor"):
        assert w in b, w
    assert "66% of the matrix peak" in b and "1.53x the compute roofline lower bound" in b
    assert "not found" not in b and "No expected feature is missing" in b  # nothing is pushed on a kernel that has it all


def test_block_of_an_attention_kernel_without_pipelining_gives_hints_not_orders():
    """@brief 验证缺 pipelining 的 attention kernel 得到的是提示而不是命令：不使用 must／required 之类措辞。"""
    b = block("fa_pro_best")
    assert "Expected for this kind of kernel but not found (hints):" in b
    assert "- pipelining:" in b and "- operand in registers:" in b and "online softmax" in b.split("Uses:")[1].split("\n")[0]
    assert len(lines_of(b)) <= 12 and b.rstrip().endswith(CAVEAT)
    assert not any(w in b.lower() for w in (" must ", "you have to", "required"))  # a hint, not an instruction


def test_a_bound_the_baseline_already_beats_is_not_called_the_ceiling():
    """@brief 验证下界已被 baseline 超过时不再把该下界当目标：块里写明它不成立，并点出更快的 baseline 用时。"""
    def beat(rep):
        rep["roofline"]["lower_bound_us"] = 931.8
        rep["baseline_us"] = {"mlx": 994.0, "torch_mps": 777.0}

    b = block("gemm_paper_best", mutate=beat)
    assert "does not hold" in b and "777" in b and "Winograd" not in b
    assert "cannot help beyond the roofline" not in b and b.rstrip().endswith(CAVEAT)


def test_a_kernel_at_the_roofline_gets_no_missing_list():
    """@brief 验证已达 roofline 的 kernel 不给缺失特性列表，只说明特性无法再帮忙，并给出已达的比例。"""
    for name in ("gemm_paper_best", "relu_best"):
        b = block(name)
        assert "hardware features cannot help beyond the roofline, so none is listed as missing" in b, name
        assert "not found" not in b and "- " not in "\n".join(l for l in lines_of(b) if l.startswith("- ")) and b.rstrip().endswith(CAVEAT)
    assert "98% of its compute roofline" in block("gemm_paper_best")


def test_saturated_elementwise_kernel_is_left_alone_even_when_it_is_scalar():
    """@brief 验证带宽已饱和（不低于 90%）的 elementwise kernel 即使全是标量访存也不给建议，只如实指出标量读写。"""
    naive = get_task("relu", "fp16", "paper").naive_msl  # scalar loads, and a roofline fraction >= 90%
    b = block("relu_best", source=naive)
    assert "hardware features cannot help beyond the roofline" in b and "not found" not in b
    assert "device scalar loads / scalar stores" in b


def test_unsaturated_scalar_elementwise_kernel_gets_the_wide_access_hint_with_the_caveat_about_scalar_loads():
    """@brief 验证带宽未饱和的标量 elementwise kernel 会得到向量化访问建议，并附上标量加载已能到 85-90% 带宽的说明。"""
    def slow(rep):
        rep["roofline"]["time_over_lower_bound"] = 1.6
        rep["roofline"]["achieved_gbps"] = 280.0
        rep["roofline"]["bandwidth_fraction"] = 0.62

    b = block("relu_best", mutate=slow, source=get_task("relu", "fp16", "paper").naive_msl)
    assert "- vector access:" in b and "scalar loads already reach about 85-90% of the bandwidth" in b
    assert "cannot help beyond the roofline" not in b


def test_scalar_gemm_is_told_that_the_matrix_product_runs_on_scalar_fma_and_by_how_much():
    """@brief 验证标量 gemm 被明说矩阵乘跑在标量 FMA 上并带上 roofline 倍数，且不列只在矩阵单元下才成立的条目。

    要求条目数在 1 到 3 之间、总行数受限、不出现浮点累加与寄存器分块这类依赖矩阵单元的项。
    """
    def slow(rep):
        rep["roofline"].update(time_over_lower_bound=3.0, achieved_tflops=5.5, compute_fraction=0.37, bound="compute")

    b = block("gemm_paper_best", mutate=slow, source=get_task("gemm", "fp16", "paper").naive_msl)
    assert "- matrix unit: no simdgroup_matrix: the matrix product runs on scalar FMA (the roofline shows 3.0x the lower bound)" in b
    items = [l for l in lines_of(b) if l.startswith("- ") and not l.startswith("- (+")]
    assert 1 <= len(items) <= 3 and len(lines_of(b)) <= 12 and b.rstrip().endswith(CAVEAT)
    assert "- float accumulate:" not in b and "- register blocking:" not in b  # items that only make sense around a matrix unit are not listed when there is none


def test_too_many_fragments_and_the_register_pressure_limit_are_warned_about():
    """@brief 验证片段过多与线程组上限同时被警告：给出片段数、参照 GEMM 的变慢倍数与线程上限，并限定结论强度。

    块里要明确这只是警告、并不证明存在寄存器压力。
    """
    from experiments.contract_sensitivity import gemm_source

    def slow(rep):
        rep["roofline"].update(time_over_lower_bound=20.0, achieved_tflops=0.9, compute_fraction=0.06, bound="compute")
        rep["pipeline"].update(max_total_threads_per_threadgroup=128, launch_threads_per_threadgroup=128)

    b = block("gemm_paper_best", mutate=slow, source=gemm_source("128x128x32t128", "fp16"), task="gemm@128x128x32t128")
    assert "Warning: 64 accumulator fragments per SIMD group; reference GEMMs with 32 or more fragments ran 12 to 26 times slower" in b
    assert "Do not add fragments" in b and "maxTotalThreadsPerThreadgroup=128" in b
    assert "does not prove register pressure" in b
    assert len(lines_of(b)) <= 12


def test_failed_candidates_and_unknown_tasks_get_no_block():
    """@brief 验证编译失败、没有特性期望的语义测试任务、以及状态非 ok 的候选都不产出反馈块。"""
    assert block("compile_error") is None
    idx, src, rep = G.fixture("relu_best")
    assert hf.hw_feedback(get_task("sem_gemm_neg", "fp16", "paper"), src, EvalReport.from_json(rep), backend_id="metal") is None  # no expectations for the semantic tests
    bad = dict(rep, status="incorrect")
    assert hf.hw_feedback(get_task("relu", "fp16", "paper"), src, EvalReport.from_json(bad), backend_id="metal") is None


def test_any_failure_returns_none_logs_a_warning_and_never_raises(monkeypatch, caplog):
    """@brief 验证探测器抛错时返回 None、写一条跳过警告，且不向上抛异常。

    @param monkeypatch pytest 的替换夹具，用于把 build_report 换成必抛错的桩。
    @param caplog pytest 的日志捕获夹具，用于确认 WARNING 级日志带跳过原因。
    """
    seen = []

    def boom(*a, **k):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(hw, "build_report", boom)
    idx, src, rep = G.fixture("fa_pro_best")
    t = get_task("flash_attention", "fp16", "paper")
    with caplog.at_level("WARNING", logger="taic.hwfeedback"):
        assert hf.hw_feedback(t, src, EvalReport.from_json(rep), log=seen.append, backend_id="metal") is None
    assert any("hw_feedback skipped: RuntimeError: detector exploded" in m for m in seen) and "hw_feedback skipped" in caplog.text


def test_a_slow_analysis_is_abandoned_after_the_time_limit(monkeypatch):
    """@brief 验证分析超过时间上限时被放弃并返回 None，整体耗时远小于分析本身的睡眠时长。

    @param monkeypatch pytest 的替换夹具，用于把时间上限调小并把计算换成长时间睡眠。
    """
    monkeypatch.setattr(hf, "TIME_LIMIT_S", 0.3)
    monkeypatch.setattr(hf, "_compute", lambda *a: time.sleep(5))
    idx, src, rep = G.fixture("fa_pro_best")
    t0 = time.time()
    assert hf.hw_feedback(get_task("flash_attention", "fp16", "paper"), src, EvalReport.from_json(rep), backend_id="metal") is None
    assert time.time() - t0 < 2.0


def test_the_result_is_cached_per_source_and_report(monkeypatch):
    """@brief 验证同一份源码与报告只计算一次：三次请求只触发一次底层计算。

    @param monkeypatch pytest 的替换夹具，用于统计底层计算函数的调用次数。
    """
    calls = []
    real = hf._compute
    monkeypatch.setattr(hf, "_compute", lambda *a: calls.append(1) or real(*a))
    for _ in range(3):
        assert block("fa_pro_best")
    assert len(calls) == 1


def test_without_the_metal_toolchain_the_block_is_still_made_from_the_source(monkeypatch):
    """@brief 验证没有 Metal 工具链时仍能仅凭源码产出反馈块（编译器前端部分优雅降级）。

    @param monkeypatch pytest 的替换夹具，用于把工具链探测换成返回 None。
    """
    monkeypatch.setattr(hw, "metal_toolchain", lambda: None)
    b = block("fa_pro_best")
    assert b and "simdgroup_matrix 8x8" in b


def test_the_core_does_not_import_the_experiments_package():
    """@brief 验证核心模块不依赖 experiments 包：导入分析与反馈模块后 sys.modules 里没有它。"""
    r = subprocess.run([sys.executable, "-c", "import sys, taic.hwfeatures, taic.hwfeedback; assert not [m for m in sys.modules if m == 'experiments' or m.startswith('experiments.')]"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ------------------------------------------------------------------------------------------------ in the search loop


def _requests(orch, needle: str) -> list[str]:
    """@brief 从编排器记下的 LLM 调用里取出发给模型且含指定片段的最后一条用户消息。

    @param orch 编排器实例。
    @param needle 用于筛选用例的文本片段。
    @return 命中的消息文本列表。
    """
    return [c["messages"][-1]["content"] for c in orch.llm.client.calls if needle in c["messages"][-1]["content"]]


def _fa_orch(tmp_path, plan=None, **search_kw):
    """@brief 用脚本化假客户端造一个 flash_attention 编排器，按需覆盖搜索开关与计划。

    @param tmp_path pytest 提供的临时目录，run 目录建在其下。
    @param plan 可选，脚本化返回的计划 JSON 文本。
    @param search_kw 要覆盖的 SearchConfig 字段（例如 ``hw_feedback=True``）。
    @return 配置好的 ``Orchestrator`` 实例。
    """
    from taic.llm import LLMConfig
    from taic.orchestrator import Orchestrator

    extra = {"plan": plan} if plan else {}
    script = llm_script(code("speed=0.9", "flash_attention_kernel"), patches={"A": code("speed=1.0", "flash_attention_kernel"), "B": code("speed=1.3", "flash_attention_kernel"), "C": code("speed=0.6", "flash_attention_kernel"),
                                                                              "MMA": code("speed=1.1", "flash_attention_kernel"), "pad": code("speed=1.2", "flash_attention_kernel"), "unrelated": code("speed=0.7", "flash_attention_kernel")}, **extra)
    cfg = LLMConfig(base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False, truncation_retry=False)
    off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
    search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=2, max_repair_tries=2, eval_reps=3, kernel_agent="pi", **{**off, **search_kw})
    return Orchestrator("flash_attention", "fp16", "small", cfg, search, run_dir=tmp_path / "run", evaluate_fn=G._scripted_eval("fa_pro_best"), client=FakeClient(script), log=lambda s: None)


def test_switch_on_puts_the_block_where_the_profiler_evidence_goes_and_nowhere_else(tmp_path):
    """@brief 验证开关打开后反馈块只出现在非 seedless 会话的 evidence 里（每次一次），seedless 会话不含。

    同时校验系统提示词含 Skill 2c、等于标准打开的提示词，且 run_config 记录了 True。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    """
    orch = _fa_orch(tmp_path, hw_feedback=True)
    run(orch)
    seeded = [brief.evidence for brief in orch._session_briefs if not brief.seedless]
    assert seeded and all(HEADER in text and text.count(HEADER) == 1 for text in seeded)
    assert all(HEADER not in (brief.evidence or "") for brief in orch._session_briefs if brief.seedless)
    assert SKILL_TITLE in orch.system and orch.system == prompts.system_prompt(orch.task, use_tools=False, hw_feedback=True)
    assert json.loads((tmp_path / "run" / "run_config.json").read_text())["search"]["hw_feedback"] is True


def test_switch_off_never_mentions_the_block_and_records_false(tmp_path):
    """@brief 验证开关关闭时任何会话的 evidence 都不含块、系统提示词无 Skill 2c，且 run_config 记 False。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    """
    orch = _fa_orch(tmp_path)
    run(orch)
    assert not any(HEADER in (brief.evidence or "") for brief in orch._session_briefs)
    assert SKILL_TITLE not in orch.system
    assert json.loads((tmp_path / "run" / "run_config.json").read_text())["search"]["hw_feedback"] is False


def test_the_feedback_of_a_failed_candidate_stays_as_it_is(tmp_path):
    """@brief 验证失败候选的 failure 文本保持原样，不会被注入硬件特性块。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    """
    calls = []
    orch = make_orch(tmp_path, llm_script(code("status=compile_error"), fixes=[code("speed=1.2")]), calls, max_rounds=1, hw_feedback=True)
    run(orch)
    failed = [brief for brief in orch._session_briefs if brief.failure]
    assert failed and all(HEADER not in brief.failure for brief in failed)


def test_a_failing_analysis_does_not_touch_the_search(tmp_path, monkeypatch):
    """@brief 验证分析抛错不影响搜索：仍然跑满两轮且产出 best，evidence 里也没有块。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    @param monkeypatch pytest 的替换夹具，用于把底层计算换成必抛错的桩。
    """
    monkeypatch.setattr(hf, "_compute", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    orch = _fa_orch(tmp_path, hw_feedback=True)
    res = run(orch)
    assert res.best_label and res.rounds == 2
    assert not any(HEADER in (brief.evidence or "") for brief in orch._session_briefs)


# ------------------------------------------------------------------------------------------------ resume


def test_a_resumed_run_keeps_the_switch_the_prompt_and_the_block(tmp_path):
    """@brief 验证中断后 resume 仍保留开关、系统提示词与反馈块，且设置来源标为 run_config.json。

    另确认系统提示词未被判为“被改过”，恢复运行后会话 evidence 里重新出现块。

    @param tmp_path pytest 提供的临时目录，用来放被中断的 run 目录。
    @raises SystemExit 脚本在第 2 轮主动结束进程时由 ``run`` 抛出。
    """
    calls = []
    orch = make_orch(tmp_path, script_for({"kill_at": 2}), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20, hw_feedback=True)
    with pytest.raises(SystemExit):
        run(orch)
    assert json.loads((tmp_path / "run" / "run_config.json").read_text())["search"]["hw_feedback"] is True
    mark_dead(tmp_path / "run")
    client = FakeClient(script_for())
    o2, snap, report, info, nb = cli.prepare_resume(ns(tmp_path / "run"), client=client, evaluate_fn=fake_eval([]), log=lambda s: None)
    assert o2.search.hw_feedback is True and SKILL_TITLE in o2.system and o2.system == snap.system
    assert report["settings"]["hw_feedback"] is True and report["provenance"]["search.hw_feedback"] == "run_config.json"
    assert "identical to what the current prompt files produce" in report["system_prompt"]  # not flagged as an edited prompt
    cli.run_prepared(ns(tmp_path / "run"), o2, snap, info, nb)
    assert any(HEADER in (brief.evidence or "") for brief in o2._session_briefs)


def test_a_resumed_run_without_the_switch_stays_without_it(tmp_path):
    """@brief 验证原本关闭开关的运行 resume 后仍然关闭：无 Skill 2c、设置为 False、无反馈块。

    @param tmp_path pytest 提供的临时目录，用来放被中断的 run 目录。
    @raises SystemExit 脚本在第 2 轮主动结束进程时由 ``run`` 抛出。
    """
    calls = []
    orch = make_orch(tmp_path, script_for({"kill_at": 2}), calls, budget=5.0, max_rounds=3, max_repair=2, patience=20)
    with pytest.raises(SystemExit):
        run(orch)
    mark_dead(tmp_path / "run")
    client = FakeClient(script_for())
    o2, snap, report, info, nb = cli.prepare_resume(ns(tmp_path / "run"), client=client, evaluate_fn=fake_eval([]), log=lambda s: None)
    assert o2.search.hw_feedback is False and SKILL_TITLE not in o2.system and report["settings"]["hw_feedback"] is False
    cli.run_prepared(ns(tmp_path / "run"), o2, snap, info, nb)
    assert not any(HEADER in (brief.evidence or "") for brief in o2._session_briefs)


def test_settings_of_a_run_without_run_config_are_read_from_its_recorded_prompt(tmp_path):
    """@brief 验证缺少 run_config 时从 transcript 记录的系统提示词推断开关：含 Skill 2c 则视为打开。

    @param tmp_path pytest 提供的临时目录，用来放 state.json 与模拟的 transcript 事件。
    """
    (tmp_path / "state.json").write_text("{}")
    for text, want in (("... # Skill 2c: reading the hardware-feature analysis ...", True), ("... # Skill 2b: ...", False)):
        ev = [{"event": "chat", "label": "init", "messages": [{"role": "system", "content": text}, {"role": "user", "content": "x"}]}]
        assert rs.infer_settings(tmp_path, ev)["search"]["hw_feedback"] is want


# ------------------------------------------------------------------------------------------------ command line and driver


def test_command_line_flags(monkeypatch):
    """@brief 验证 run 与 resume 子命令默认关闭开关，加了 ``--hw-feedback`` 后才打开。

    @param monkeypatch pytest 的替换夹具，用于替换两个命令处理函数并把参数捕获下来。
    """
    got = {}
    monkeypatch.setattr(cli, "cmd_run", lambda a: got.update(run=vars(a)) or 0)
    monkeypatch.setattr(cli, "cmd_resume", lambda a: got.update(resume=vars(a)) or 0)
    cli.main(["run", "relu"])
    cli.main(["resume", "somewhere"])
    assert got["run"]["hw_feedback"] is False and got["resume"]["hw_feedback"] is False
    cli.main(["run", "relu", "--hw-feedback"])
    cli.main(["resume", "somewhere", "--hw-feedback"])
    assert got["run"]["hw_feedback"] is True and got["resume"]["hw_feedback"] is True


def test_the_resume_flag_enables_it_for_a_run_that_did_not_record_it():
    """@brief 验证 resume 时命令行标志能给未记录该开关的运行打开它，并把来源标为 command line。"""
    info = {"model": None}
    settings = {"search": {"patience": 4}, "llm": {"model": "fake"}, "provenance": {}}
    a = ns(Path("."), dry_run=True, hw_feedback=True)
    cfg, search, prov = cli._resume_config(a, settings, info)
    assert search.hw_feedback is True and prov["search.hw_feedback"] == "command line"
    cfg, search, prov = cli._resume_config(ns(Path("."), dry_run=True), settings, info)
    assert search.hw_feedback is False


def test_the_ab_run_is_fa_flash_with_only_the_switch_changed():
    """@brief 验证 A/B 试跑的两份配置除 ``search.hw_feedback`` 外完全相同（同 task、模型与预算）。"""
    from experiments import p1_runs

    a, b = dict(p1_runs.RUNS["fa_flash"]), dict(p1_runs.RUNS["fa_flash_hw"])
    assert b.pop("search") == {"hw_feedback": True} and "search" not in a
    assert a == b  # same task, model and budget


# ------------------------------------------------------------------------------------------------ the A/B analysis script


def test_the_ab_analysis_reads_the_block_and_counts_plans_that_talk_about_it(tmp_path):
    """@brief 验证 A/B 分析脚本能读出反馈块，并在报告里统计谈到所列特性的计划。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    """
    from experiments import hw_feedback_ab as ab

    plan = ('```json\n{"bottleneck":"x","analysis":"y","plans":[{"title":"MMA","change":"compute QK^T with simdgroup_matrix 8x8 tiles"},'
            '{"title":"pad","change":"pad the threadgroup rows to avoid bank conflicts"},{"title":"unrelated","change":"change the loop order"}]}\n```')
    orch = _fa_orch(tmp_path, plan=plan, hw_feedback=True)
    run(orch)
    r = ab.analyze(tmp_path / "run")
    assert r["task"] == "flash_attention"
    seeded = [brief for brief in orch._session_briefs if not brief.seedless]
    assert seeded and HEADER in seeded[0].evidence
    assert r["rounds"] == []
    assert "plans mentioning a listed feature" in ab.markdown([r])


def test_the_ab_analysis_computes_the_list_for_a_control_run_that_had_no_block(tmp_path):
    """@brief 验证开关关闭的对照 run 里 ``feedback_in_run`` 为 False，且能从块文本解析出所列特性名。

    @param tmp_path pytest 提供的临时目录，用作 run 目录的父目录。
    """
    from experiments import hw_feedback_ab as ab

    run(_fa_orch(tmp_path))  # switch off: no block in any request
    r = ab.analyze(tmp_path / "run")
    assert r["feedback_in_run"] is False
    assert r["rounds"] == []
    assert ab.listed_features("- matrix unit: x\n- (+2 lower-priority item(s) omitted)\n- tg padding: y") == ["matrix_unit", "tg_padding"]
