"""Hashes of everything the LLM is sent, computed with the *current* code, for the byte-identity regression of ``search.hw_feedback = false``.

``compute()`` calls the prompt and feedback functions with ``hw_feedback`` off (the arguments from before that switch, or ``hw_feedback=False``).
``tests/support/golden/metal/hw_off.json`` is that output. Regenerate it on purpose when a prompt change other than the switch should become the new off baseline.
A request hash covers the body ``chat.completions.create`` is given, not only ``messages``: model, max_tokens, reasoning_effort, extra_body, and tools.
``REQUEST_KEYS`` is the whitelist of keys ``LLM._params`` may produce; the test fails when the two sets diverge.
Feedback entries store the text. System prompts stay hashes. Orchestrator entries are keyed by task and ``plan_mix``, and each entry lists its session contracts in order, so round 1 and round 2 cannot trade places unnoticed.
``_meta`` (when, which commit, why) is written by ``--record`` and is not part of ``compute()``.

    uv run python -m tests.support.hw_golden --record --reason "why the off baseline should change"
    # more than FORCE_AFTER keys also needs --force
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .v1_tasks import V1_TASK, v1_kernel_name

GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "metal"
GOLDEN = GOLDEN_DIR / "hw_off.json"
FIXTURES = GOLDEN_DIR / "fixtures"
# Keys ``LLM._params`` is allowed to put on the wire. ``messages`` is added by ``chat`` and is hashed with them.
REQUEST_KEYS = frozenset({"model", "max_tokens", "reasoning_effort", "extra_body", "tools"})
FORCE_AFTER = 40


def sha(text: str) -> str:
    """@brief 算文本的 sha1 十六进制摘要，作为金标文件里的指纹。

    @param text 待哈希的文本。
    @return 40 位十六进制摘要。
    """
    return hashlib.sha1(text.encode()).hexdigest()


def dumps(data: dict, meta: dict) -> str:
    """@brief 产出 ``--record`` 唯一允许写、格式测试唯一接受的金标文本。

    ``_meta`` 放在最前，缩进 0、键排序，保证同一份数据只有一个字节形态。

    @param data 各区段的指纹字典。
    @param meta 记录来源（时间、提交、理由）。
    @return 序列化后的 JSON 文本。
    """
    return json.dumps({"_meta": meta, **data}, indent=0, sort_keys=True)


def fixture_digest(name: str) -> str:
    """@brief 算金属源码与报告合起来的 sha256，使 fixture 的改动不能冒充提示词改动。

    @param name fixture 名，对应 ``<name>.metal`` 与 ``<name>.report.json``。
    @return 小写十六进制摘要。
    """
    metal = (FIXTURES / f"{name}.metal").read_bytes()
    report = (FIXTURES / f"{name}.report.json").read_bytes()
    return hashlib.sha256(metal + b"\0" + report).hexdigest()


def fixture(name: str) -> tuple[dict, str, dict]:
    """@brief 取一份已记录候选的索引条目、内核源码与报告 JSON。

    @param name fixture 名。
    @return ``(索引条目, 内核源码, 报告字典)``。
    """
    idx = json.loads((FIXTURES / "index.json").read_text())[name]
    return idx, (FIXTURES / f"{name}.metal").read_text(), json.loads((FIXTURES / f"{name}.report.json").read_text())


def param_keys() -> set[str]:
    """@brief 穷举 ``LLM._params`` 在 effort、工具与 max_tokens 各组合下会发出的键。

    @return 所有出现过的键名集合，用来和 ``REQUEST_KEYS`` 白名单比对。
    """
    from taic.llm import CostLedger, LLM, LLMConfig
    from taic.tools import LOOKUP_TOOL, QUICK_CHECK_TOOL

    tools = [LOOKUP_TOOL, QUICK_CHECK_TOOL]
    keys: set[str] = set()
    for effort in (None, "off", "low", "high"):
        for extra in ({}, {"top_p": 1}):
            cfg = LLMConfig(base_url="http://x", api_key="k", model="m", reasoning_effort="high", extra_body=dict(extra))
            llm = LLM(cfg, CostLedger(1.0), client=object())
            for tool_arg in (None, tools):
                for limit in (None, 1024):
                    keys |= set(llm._params(tool_arg, limit, effort))
    return keys


def task_grid() -> list[tuple[str, str, str]]:
    """@brief 列出所有参与的题目、dtype 与规模的组合。

    @return ``(题目名, dtype, 规模)`` 三元组列表；题目含普通题、保留的语义题与一个带变体的题。
    """
    from tasks import list_tasks
    from tasks.semantic import FIXED_NAMES

    # ``gemm@…`` 随 A5 的 v1 gemm 装配一起退役，A6 删除了这里的名字条目：网格里只留 v1 真能装配的
    # 题（``list_tasks``）与仍保留的 ``flash_attention@…`` 变体。迁移题由 v2 入口取样。
    names = list(list_tasks(True)) + list(FIXED_NAMES) + ["flash_attention@64x32t256"]
    return [(n, dt, "paper") for n in names for dt in ("fp16", "bf16")]


def system_prompts(extra: dict | None = None) -> dict[str, str]:
    """@brief 对网格里每个存在的组合，按是否带工具与 profiler 证据算系统提示词的指纹。

    @param extra 额外传给 ``system_prompt`` 的关键字参数。
    @return 键为 ``题目|dtype|规模|tools=<0/1>|pe=<0/1>``、值为 sha1 的字典。
    """
    from tasks import get_task

    from taic import prompts

    out = {}
    for name, dt, size in task_grid():
        try:
            task = get_task(name, dt, size)
        except Exception:  # noqa: BLE001 - a combination that does not exist is simply not recorded
            continue
        for tools in (False, True):
            for pe in (False, True):
                out[f"{name}|{dt}|{size}|tools={int(tools)}|pe={int(pe)}"] = sha(prompts.system_prompt(task, use_tools=tools, profiler_evidence=pe, **(extra or {})))
    return out


def user_messages() -> dict[str, str]:
    """@brief 对每个题目算初始、修复、规划与补丁四条用户消息的指纹。

    @return 键为消息类型与题目、值为 sha1 的字典；只在 fp16 上取样。
    """
    from tasks import get_task

    from taic import prompts

    out = {}
    for name, dt, size in [(n, d, s) for n, d, s in task_grid() if d == "fp16"]:
        try:
            task = get_task(name, dt, size)
        except Exception:  # noqa: BLE001
            continue
        out[f"initial|{name}"] = sha(prompts.initial_user(task))
        out[f"repair|{name}"] = sha(prompts.repair_user("FEEDBACK TEXT", 1, 3, source_fence="metal"))
        for mixed in (False, True):
            out[f"plan|{name}|mixed={int(mixed)}"] = sha(prompts.plan_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", ["h1", "h2"], 3, mixed=mixed, source_fence="metal"))
        out[f"patch|{name}"] = sha(prompts.patch_user(task, "KERNEL SOURCE\n", "FEEDBACK TEXT", {"title": "T", "change": "C"}, source_fence="metal"))
    return out


def schema4_build(name: str, report: dict) -> dict:
    """Rebuild one fixture's schema-4 ``build`` record from its own recorded facts.

    The fixtures were recorded on a real Mac and only kept the report; the assembled text and
    the live pipeline are gone, so nothing here claims a native executable.  ``ok`` is a
    ``full`` native build with ``native_absent_reason=portable_artifact_set`` (a re-read report,
    not a fresh outcome); a recorded compile failure is a ``partial`` whose assembled digest is
    honestly unknown (``null``), never a copy of the candidate.

    @param name fixture 名（``<name>.metal`` 是候选源码）。
    @param report 旧报告字典（提供 status/compile 与任务字段）。
    @return ``build`` 对象。
    """
    import hashlib

    from tcenv.backends.metal.source_builder import (
        METAL_BUILDER_ID, METAL_PROFILE_NAME, METAL_TOOLCHAIN_ID,
    )
    from tcenv.build_projection import project_partial, project_received
    from tcenv.remote import contracts
    from tcenv.remote.artifact_builder import (
        BuildArtifact, BuildError, BuildEvidence, BuildProvenance, BuildProfile,
    )
    from tcenv.remote.artifact_bundle import BundleEntry
    from tasks import get_task

    source = (FIXTURES / f"{name}.metal").read_bytes()
    entry = BundleEntry(
        path="kernel.metal", role="SOURCE", media_type="text/plain; charset=utf-8",
        size_bytes=len(source), sha256=hashlib.sha256(source).hexdigest(), data=source,
    )
    compile_doc = report.get("compile") if isinstance(report.get("compile"), dict) else {}
    profile = BuildProfile(
        build_profile=METAL_PROFILE_NAME,
        builder_id=METAL_BUILDER_ID,
        builder_version="1",
        build_host="control_host",
        toolchain_id=METAL_TOOLCHAIN_ID,
        toolchain_version=str(compile_doc.get("language_version") or "unknown"),
        sdk_version="macos-recorded",
        compile_profile={"math": str(compile_doc.get("math_mode") or "fast")},
    )
    specialization = get_task(report["task"], report["dtype"], report["size"]).specialization
    identity = {
        "opspec_digest": specialization.opspec_digest,
        "lowering_profile_digest": specialization.lowering_profile_digest,
    }
    if report["status"] == "ok":
        provenance = BuildProvenance(
            profile=profile, source_sha256=entry.sha256,
            artifact_set_digest=None,
        )
        # A received/prebuilt set is the only full shape that may omit a native executable for a
        # native build, so the fixture is recorded as a verified, loaded artifact set.
        from tcenv.remote.artifact_bundle import ArtifactBundle
        bundle = ArtifactBundle(entries=(entry,))
        provenance = BuildProvenance(
            profile=profile, source_sha256=entry.sha256,
            artifact_set_digest=bundle.artifact_set_digest,
        )
        evidence = BuildEvidence(
            mode="prebuilt", status="reused", provenance=provenance,
            artifacts=(BuildArtifact.from_entry(entry),),
            duration_ms=max(1, int(round(float(compile_doc.get("compile_s") or 0.0) * 1000))),
            log=None, loaded=(), placement="received",
            producer={
                "schema_id": contracts.NPU_BUILD_PRODUCER_RECORD_SCHEMA_ID,
                "status": "ok", "mode": "prebuilt",
                "duration_ms": max(1, int(round(float(compile_doc.get("compile_s") or 0.0) * 1000))),
                "log": None,
                "evidence_sha256": hashlib.sha256(b"recorded-metal-fixture").hexdigest(),
            },
        )
        return project_received(evidence, build_kind="native", identity=identity)
    # A recorded compile failure: the diagnostic text stays in ``compile``.
    provenance = BuildProvenance(
        profile=profile, source_sha256=entry.sha256, artifact_set_digest=None,
    )
    error = BuildError(
        "compile", str(compile_doc.get("errors") or "recorded compile failure"),
        provenance=provenance, assembled_sha256=None, duration_ms=1,
    )
    return project_partial(error, build_kind="native", identity=identity)


def rebuild_fixture(name: str, report: dict) -> dict:
    """Return one recorded fixture as a schema-4 report (idempotent).

    Fields the old record never wrote are filled with honest defaults (unknown device, empty
    timeline), never invented measurements.

    @param name fixture 名。
    @param report 旧报告字典。
    @return schema 4 报告字典。
    """
    from tcenv.schema import PROTOCOL_VERSION

    out = dict(report)
    out["schema_version"] = 4
    out.setdefault("target_id", "apple-m4-max")
    out.setdefault("backend", "metal")
    out.setdefault("backend_version", "1")
    out.setdefault("device_name", "")
    out.setdefault("gpu_family", "")
    out.setdefault("device_warnings", [])
    out.setdefault("timeline", [])
    out.setdefault("cache_hit", False)
    out.setdefault("protocol_version", PROTOCOL_VERSION)
    out["evaluation_fingerprint"] = None
    out["baseline_implementation_digest"] = None
    out["baseline_environment_digest"] = None
    out["build"] = schema4_build(name, out)
    return out


def feedback_texts() -> dict[str, str]:
    """@brief 把每份 fixture 报告渲染成反馈文本，并另存一份截断到 6000 字符的版本。

    @return 键为 ``format_feedback[|_cut]|<fixture>``、值为文本本身的字典（文本而不是哈希）。
    """
    from tcenv.evaluate import EvalReport
    from tcenv.report import format_feedback

    out = {}
    idx = json.loads((FIXTURES / "index.json").read_text())
    for name in sorted(idx):
        rep = EvalReport.from_json(json.loads((FIXTURES / f"{name}.report.json").read_text()))
        text = format_feedback(rep)
        out[f"format_feedback|{name}"] = text
        out[f"format_feedback_cut|{name}"] = text[:6000]
    return out


def outbound_requests() -> dict[str, str]:
    """@brief 覆盖完整请求体：消息固定，变化的是工具 schema、三档 effort 与 max_tokens。

    @return 键为组合描述、值为整个请求体 sha1 的字典。
    """
    from taic.llm import CostLedger, LLM, LLMConfig
    from taic.tools import LOOKUP_TOOL, QUICK_CHECK_TOOL

    tools = [LOOKUP_TOOL, QUICK_CHECK_TOOL]
    messages = [{"role": "user", "content": "PING"}]
    specs = (
        ("effort=off|tools=0|max=default|model=fake", "fake", "off", None, None),
        ("effort=high|tools=0|max=default|model=fake", "fake", "high", None, None),
        ("effort=low|tools=0|max=default|model=fake", "fake", "low", None, None),
        ("effort=high|tools=1|max=default|model=fake", "fake", "high", tools, None),
        ("effort=off|tools=1|max=1024|model=other", "other", "off", tools, 1024),
    )
    out = {}
    for name, model, effort, tool_arg, limit in specs:
        cfg = LLMConfig(base_url="http://x", api_key="k", model=model, reasoning_effort="high")
        body = {"messages": messages, **LLM(cfg, CostLedger(1.0), client=object())._params(tool_arg, limit, effort)}
        out[name] = sha(json.dumps(body, sort_keys=True))
    return out


def _scripted_eval(report_name: str):
    """@brief 造一个返回已记录报告的评测器，speedup 取自源码里的 ``// speed=`` 标记。

    @param report_name 作为模板的 fixture 名。
    @return 可直接当 ``evaluate_fn`` 用的函数；源码里没有标记时沿用报告里的 speedup。
    """
    import re

    from tcenv.evaluate import EvalReport

    base = json.loads((FIXTURES / f"{report_name}.report.json").read_text())

    def ev(task, dtype, size, source, *, mode="full", seed=None, reps=15, **kw):
        r = EvalReport.from_json(base)
        # 报告的身份必须跟请求一致：编排器会拿 task/dtype/size/mode 与请求核对，不一致就判
        # result_mismatch 并停掉整场搜索。模板是 paper 规模的 full 报告，这里改成这次请求的。
        r.task, r.dtype, r.size, r.mode = task, dtype, size, mode
        m = re.search(r"// speed=([\d.]+)", source)
        if m:
            sp = float(m.group(1))
            r.speedup, r.speedups, r.candidate_us = sp, {k: sp for k in (r.speedups or {"mlx": 1.0})}, 100.0 / sp
        return r

    return ev


def _orch_key(task: str, plan_mix: bool) -> str:
    """@brief 拼 orchestrator 区段的键，区分题目与是否开启混合规划。

    @param task 题目名。
    @param plan_mix 是否开启 plan_mix。
    @return 形如 ``orchestrator|<题目>|plan_mix=<0/1>`` 的键。
    """
    return f"orchestrator|{task}|plan_mix={int(plan_mix)}"


def orchestrator_messages(extra_search: dict | None = None) -> dict[str, object]:
    """@brief 记录一次脚本化搜索按顺序交给 hmz 的 TASK 契约指纹。

    这条路径不发 chat 请求，所以每个条目哈希的是契约文本；渲染时换成固定的合成路径，
    使指纹与 checkout 位置无关。

    @param extra_search 覆盖 ``SearchConfig`` 的额外开关。
    @return 键为 ``_orch_key``、值为条目数、合并 sha 与逐条 sha 的字典。
    """
    from taic.llm import LLMConfig, SearchConfig
    from taic.orchestrator import Orchestrator

    from .fakes import FakeClient
    from .orchestrator import KERNEL, code, llm_script

    out: dict[str, object] = {}
    import tempfile

    # The golden records the *v1 prompt mechanism* on preserved v1 tasks of two families:
    # an elementwise one and a compute one.  B1 migrated rope/gemv, so the elementwise specimen
    # is lion now (the global default v1 task is conv2d, which would duplicate the compute case).
    # The recorded feedback fixtures keep their bytes; only the task object that supplies the
    # contract words to the brief changes.
    # B2 removed lion/conv2d as well, so the cases now use the remaining static v1 tasks.
    # B3 之后 v1 只剩 sem_attention_odd_scale 与 sem_int 动态族。
    cases = (
        ("sem_attention_odd_scale", v1_kernel_name("sem_attention_odd_scale"), "relu_best", False),
        ("sem_attention_odd_scale", v1_kernel_name("sem_attention_odd_scale"), "relu_best", True),
        ("sem_int_5", v1_kernel_name("sem_int_5"), "fa_flash_best", False),
        ("sem_int_3", v1_kernel_name("sem_int_3"), "gemm_paper_best", True),
    )
    for task, kernel, rep_name, plan_mix in cases:
        script = llm_script(code("speed=0.9", kernel), patches={"A": code("speed=1.0", kernel)})
        llm_cfg = LLMConfig(
            base_url="http://x", api_key="sk-test-0123456789abcdef", model="fake", reasoning_effort="off", use_tools=False,
        )
        # This hash is the switches-off prompt. The production default is on; the off path is what the golden file records.
        off = dict(stop_at_roofline=False, abi_guard=False, target_note=False, rich_history=False, status_lines=False, plan_ledger=False)
        search = SearchConfig(budget_usd=5.0, branches=1, max_rounds=1, max_repair_tries=0, eval_reps=3, plan_mix=plan_mix, kernel_agent="pi", **{**off, **(extra_search or {})})
        with tempfile.TemporaryDirectory() as td:
            client = FakeClient(script)
            previous_catalog = os.environ.get("TAIC_CATALOG_ROOT")
            os.environ["TAIC_CATALOG_ROOT"] = str(Path(td) / "catalog")
            try:
                orch = Orchestrator(task, "fp16", "small", llm_cfg, search, run_dir=Path(td) / "run", evaluate_fn=_scripted_eval(rep_name), client=client, log=lambda s: None)
                asyncio.run(orch.run())
            finally:
                if previous_catalog is None:
                    os.environ.pop("TAIC_CATALOG_ROOT", None)
                else:
                    os.environ["TAIC_CATALOG_ROOT"] = previous_catalog
        from taic.kernel_agent import render_contract

        each = []
        for index, brief in enumerate(orch._session_briefs):
            # TASK has a byte budget, so rendering against a checkout-dependent absolute
            # path changes where optional sections are cut. Render against stable synthetic
            # paths before hashing; this keeps the golden independent of the checkout root.
            stable = replace(brief, repo=Path("/repo"), work_dir=f"/work/{brief.label}")
            text = render_contract(stable)
            each.append({"label": f"{index}", "sha": sha(text)})
        joined = "\n".join(f"{row['label']} {row['sha']}" for row in each)
        out[_orch_key(task, plan_mix)] = {"requests": len(each), "sha": sha(joined), "each": each}
    return out


def compute(extra_system: dict | None = None, extra_search: dict | None = None) -> dict:
    """@brief 用当前代码算出关掉 ``hw_feedback`` 时的全部指纹。

    @param extra_system 传给系统提示词渲染的额外参数。
    @param extra_search 覆盖 ``SearchConfig`` 的额外开关。
    @return 含系统提示词、用户消息、反馈、出站请求与 orchestrator 五个区段的数据。
    """
    return {
        "system_prompt": system_prompts(extra_system),
        "user_messages": user_messages(),
        "feedback": feedback_texts(),
        "outbound": outbound_requests(),
        "orchestrator": orchestrator_messages(extra_search),
    }


def changed_keys(got: dict, want: dict) -> list[str]:
    """@brief 列出记录值有差异的 ``区段:键`` 对。

    ``_meta`` 是来源信息而不是提示词，不参与比较。

    @param got 本次算出的数据。
    @param want 金标文件里的数据。
    @return 差异键列表，可直接打印。
    """
    lines = []
    sections = (set(want) | set(got)) - {"_meta"}
    for sec in sorted(sections):
        left = got.get(sec) or {}
        right = want.get(sec) or {}
        if not isinstance(left, dict) or not isinstance(right, dict):
            if left != right:
                lines.append(sec)
            continue
        for key in sorted(set(left) | set(right)):
            if left.get(key) != right.get(key):
                lines.append(f"{sec}:{key}")
    return lines


def _git_commit() -> str:
    """@brief 读当前 HEAD 提交号，写进金标文件的来源信息。

    @return 提交 sha；仓库不可用或 git 失败时返回空串。
    """
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=GOLDEN_DIR.parents[3], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _reason(argv: list[str]) -> str:
    """@brief 从命令行里取记录理由，两种写法都认。

    ``--reason "x"`` 与 ``--reason=x`` 都可以；值缺失时返回空串。

    @param argv 命令行参数列表。
    @return 理由文本。
    """
    for index, item in enumerate(argv):
        if item == "--reason":
            return argv[index + 1] if index + 1 < len(argv) else ""
        if item.startswith("--reason="):
            return item.split("=", 1)[1]
    return ""


if __name__ == "__main__":
    if "--record" in sys.argv:
        reason = _reason(sys.argv)
        if not reason.strip():
            sys.exit("refusing to record without --reason \"why this baseline should change\"")
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        data = compute()
        previous = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        diff = changed_keys(data, previous)
        print(f"{len(diff)} keys differ")
        for line in diff:
            print(line)
        if len(diff) > FORCE_AFTER and "--force" not in sys.argv:
            sys.exit(f"refusing to record {len(diff)} changes without --force")
        meta = {"recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"), "git_commit": _git_commit(), "reason": reason}
        GOLDEN.write_text(dumps(data, meta))
        print({k: len(v) for k, v in data.items()}, "->", GOLDEN)
    else:
        print(__doc__)
