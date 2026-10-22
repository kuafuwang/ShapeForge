"""Write one kernel by asking the pi coding agent, through hmz, to edit a file.

The search does not call the chat API to write kernels. One hmz session reads ``TASK.md``
and writes one submission file whose name is ``submission_name``. ``kernel.metal`` is the
seed, placed before the session, and the search does not read it. A missing submission
means this session returned no kernel. A number the agent prints is not the search result.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError
from .parse import extract_metal

from .target import skill_path

FLOW = Path(__file__).resolve().parent / "flows" / "kernel_once"
SKILL = skill_path("msl_reference")
EXPERIENCE = skill_path("experience")
_KERNEL = re.compile(r"\bkernel\s+void\s+(\w+)\b")
_USAGE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
_NAME_PART = re.compile(r"[^A-Za-z0-9._-]+")
# 一场会话的墙钟。hmz 的回合预算先到；这个秒数只在进程不退出时杀掉它。
def process_timeout_s(turn_minutes: int) -> float:
    """Kill hmz one minute after its own flow budget, so a longer turn still gets the wrap."""
    return (int(turn_minutes) + WRAP_MINUTES + FLOW_SLACK_MINUTES + 1) * 60


_PEER_HEADER = "本任务前序避坑提示（队友交接，仅限本运行参考；凡未附探针文件名与实测数字的硬件/算法上限断言，一律按未验证处理，鼓励你写探针翻案）："
WRAP_OUTPUT_TOKENS = 4000
WRAP_MINUTES = 3
FLOW_SLACK_MINUTES = 2
PROCESS_TIMEOUT_S = process_timeout_s(25)
TASK_MAX_BYTES = 12_000
SCRATCH_MAX_BYTES = 50_000_000


class AgentError(LLMError):
    """hmz or pi could not take this turn."""


@dataclass
class Prices:
    """USD per million tokens. Read from the credentials file. Never logged."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def as_dict(self) -> dict:
        """The rates recorded beside a session charge, so a later reader does not need today's price file."""
        return {"input": self.input, "output": self.output, "cache_read": self.cache_read, "cache_write": self.cache_write}


@dataclass
class SessionBrief:
    """The contract for one coding-agent session. The orchestrator fills it; this module only renders it."""

    repo: Path
    task_name: str
    dtype: str
    size: str
    kernel_name: str
    triton_source: str
    description: str
    contract_text: str
    abi_stub: str
    rtol: float
    atol_rel: float
    seed_source: str | None = None
    seed_report_json: str = ""
    scorecard: str = ""
    history: str = ""
    failure: str = ""
    notes_summary: str = ""
    bound_line: str = ""
    evidence: str = ""
    reps: int = 15
    require_contract_launch: bool = False
    min_improvement: float = 1.005
    noise_z: float = 2.0
    seedless: bool = False
    peer_tips: str = ""
    label: str = "init"
    work_dir: str = ""
    guidance_dir: str = ""
    target_id: str = "apple-m4-max"
    harness_overlay: str = ""


@dataclass
class AgentResult:
    """What one hmz session left on disk, plus the dollars we computed from its token counts."""

    source: str | None
    cost_usd: float | None
    cost_incomplete: bool
    status: str  # ok, timeout, failed
    error: str = ""
    spent: dict = field(default_factory=dict)
    pruned: dict = field(default_factory=dict)
    notes: str = ""
    feedback: str = ""
    self_checks: int = 0
    self_check_timeouts: int = 0
    self_check_gpu_s: float = 0.0


def submission_name(task: str, dtype: str, size: str, label: str) -> str:
    """The file this session must create. The search does not create it.

    The name is ``{task}-{dtype}-{size}-{label}.metal``. ``label`` is the session
    (``init``, ``r3.p0``), so two sessions in one run do not share a name. A character
    outside letters, digits, ``.``, ``_`` and ``-`` becomes ``-``.
    """
    parts = []
    for raw in (task, dtype, size, label):
        piece = _NAME_PART.sub("-", str(raw).strip()).strip("-.")
        parts.append(piece or "x")
    return "-".join(parts) + ".metal"


def kernel_text(text: str, kernel_name: str) -> str | None:
    """The Metal source in ``text``: a fenced block, or the whole file when it declares the kernel."""
    fenced = extract_metal(text, kernel_name)
    if fenced:
        return fenced
    body = (text or "").strip("\n")
    if "taic-launch" in body and _KERNEL.search(body):
        return body + "\n"
    if _KERNEL.search(body) and re.search(r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b", body):
        return body + "\n"
    return None


def agent_spec(provider: str, model: str, effort: str | None = None) -> str:
    """The ``-a assistant=`` value. pi names models as ``provider/id``, and an effort as ``:effort``."""
    spec = f"pi/{provider}/{model}"
    if effort:
        spec = f"{spec}:{effort}"
    return spec


def load_prices(path: Path, provider: str, model: str) -> Prices:
    """The model's prices. Raises ``AgentError`` when the provider, the model, or an output price is missing.

    The file also holds API keys. This function does not return them and callers must not log the document.
    """
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AgentError(f"cannot read model prices from {path}: {exc}") from exc
    providers = doc.get("providers") if isinstance(doc, dict) else None
    prov = providers.get(provider) if isinstance(providers, dict) else None
    if not isinstance(prov, dict) or not prov.get("apiKey"):
        raise AgentError(f"no credentials for provider {provider}")
    found = None
    for item in prov.get("models") or []:
        if isinstance(item, dict) and item.get("id") == model:
            found = item.get("cost") if isinstance(item.get("cost"), dict) else None
            break
    if not found or not isinstance(found.get("output"), (int, float)) or found.get("output") <= 0:
        raise AgentError(f"no output price for {provider}/{model}")
    output = float(found["output"])
    return Prices(
        input=float(found.get("input", output)),
        output=output,
        cache_read=float(found.get("cacheRead", found.get("input", output))),
        cache_write=float(found.get("cacheWrite", found.get("input", 0)) or 0),
    )


def output_tokens_for(usd: float, price_out_per_million: float) -> int:
    """How many output tokens ``usd`` buys at ``price_out_per_million``. At least 1."""
    if price_out_per_million <= 0:
        raise AgentError("output price must be positive")
    return max(1, int(float(usd) / float(price_out_per_million) * 1_000_000))


def dollars_for(spent: dict, prices: Prices) -> float:
    """USD for one session's token counts. ``reasoning`` is not priced: pi already counts it inside output."""
    rates = {
        "input": prices.input,
        "output": prices.output,
        "cache_read": prices.cache_read,
        "cache_write": prices.cache_write,
    }
    total = 0.0
    for kind, tokens in spent.items():
        rate = rates.get(kind)
        if rate is None or not isinstance(tokens, (int, float)):
            continue
        total += float(tokens) / 1_000_000 * rate
    return total


def parse_spent(value) -> dict[str, float]:
    """Token counts from one hmz ``result`` line.

    ``hmz exec --json`` writes ``spent`` with ``default=str``. ``Usage`` is not a dict, so the
    field is the string ``Usage(input=1, output=2)``. A dict is accepted too.
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if isinstance(value, str) and value.startswith("Usage("):
        return {name: float(num) for name, num in _USAGE.findall(value)}
    return {}


def spend_from_stream(text: str) -> tuple[dict[str, float], bool]:
    """Sum ``spent`` on every ``kind=result`` line. The bool is whether any result line was present.

    Each result line is one turn, not a running total, so the lines are added.
    """
    totals: dict[str, float] = {}
    saw = False
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("kind") != "result":
            continue
        saw = True
        for kind, tokens in parse_spent(obj.get("spent")).items():
            totals[kind] = totals.get(kind, 0.0) + tokens
    return totals, saw


def _shell(value: str) -> str:
    return shlex.quote(str(value))


def _self_check_dir(brief: SessionBrief) -> Path:
    """Hang log inside the session work directory, which the coding agent's sandbox can write.

    An earlier version put it in ``<run>/session_checks`` so deleting the work copy would not reset
    the hang count. The sandbox denied ``mkdir`` there and every timed self-check crashed after
    paying for the benchmark (17 agent reports). The work directory is already the agent's trust
    boundary; keep the log writable and archive it with the session instead.
    """
    work = _work_path(brief)
    return work / ".selfcheck"


def _work_path(brief: SessionBrief) -> Path:
    if brief.work_dir:
        return Path(brief.work_dir).resolve()
    return Path.cwd()


def _check_command(brief: SessionBrief, mode: str) -> str:
    """Eval command with absolute paths. ``uv run --directory`` changes the working directory."""
    name = submission_name(brief.task_name, brief.dtype, brief.size, brief.label)
    work = _work_path(brief)
    repo = Path(brief.repo).resolve()
    parts = [
        "uv", "run", "--directory", _shell(str(repo)),
        "python", "-m", "tcenv.cli", "eval", _shell(brief.task_name), _shell(str(work / name)),
        "--dtype", _shell(brief.dtype),
        "--size", _shell(brief.size),
        "--quick" if mode == "quick" else "--timed",
        "--target", _shell(brief.target_id),
    ]
    if mode == "timed":
        parts.extend(["--self-check-dir", _shell(str(_self_check_dir(brief)))])
        parts.append("--reps")
        parts.append(str(int(brief.reps)))
        parts.append("--min-improvement")
        parts.append(str(brief.min_improvement))
        parts.append("--noise-z")
        parts.append(str(brief.noise_z))
        if brief.seed_source and not brief.seedless:
            parts.extend(["--seed-report", _shell(str(work / "report.json"))])
        if brief.require_contract_launch:
            parts.append("--require-contract-launch")
    elif brief.require_contract_launch:
        parts.append("--require-contract-launch")
    return " ".join(parts)


def _feedback_check_command(brief: SessionBrief) -> str:
    work = _work_path(brief)
    repo = Path(brief.repo).resolve()
    return " ".join([
        "uv", "run", "--directory", _shell(str(repo)),
        "python", "-m", "taic.agent_feedback", "check", _shell(str(work / "FEEDBACK.md")),
        "--work", _shell(str(work)),
    ])


def _clip_bytes(text: str, limit: int) -> str:
    """Cut UTF-8 text at a byte budget without splitting a character."""
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[: max(0, limit)].decode("utf-8", "ignore")


def _cap_task(text: str, limit: int) -> str:
    """Trim one dynamic block at a byte budget. The full text stays on disk; the marker says so."""
    body = (text or "").strip()
    if len(body.encode("utf-8")) <= limit:
        return body
    return _clip_bytes(body, max(0, limit)).rstrip() + "\n…（本段已截断；全文见本目录文件）"


def render_contract(brief: SessionBrief, *, max_bytes: int = TASK_MAX_BYTES) -> str:
    """The KDA-shaped task contract written to ``TASK.md``, with a hard dynamic-block budget.

    The contract, signature, commands and evidence rules are required. Scorecard, failure,
    NOTES, tips, history and extra evidence are added in priority order while they fit; an
    omitted block names the file that still holds it.
    """
    quick = _check_command(brief, "quick")
    timed = _check_command(brief, "timed")
    submit = submission_name(brief.task_name, brief.dtype, brief.size, brief.label)
    overlay_text = ("\n" + _cap_task(brief.harness_overlay, 1800)) if brief.harness_overlay else ""
    lines = [
        f"# 目标",
        f"把 Triton 算子 `{brief.task_name}` 写成 Metal，dtype `{brief.dtype}`，规模 `{brief.size}`。",
        brief.description.strip(),
        "",
        "Triton 源码是规格，按它的语义写，不按名字或注释：",
        "```python",
        brief.triton_source.rstrip(),
        "```",
        "",
        "# 正确性",
        f"容差 |out - ref| <= {brief.rtol} * |ref| + {brief.atol_rel} * scale。",
        "NaN、Inf 必须和参考一致，符号也要一致。输入保持 `device const`，不许改。",
        "工作区在每次计时迭代里由评测填零，这次填零算进你的时间。",
        "",
        "# 约束",
        brief.contract_text.strip(),
        "",
        "签名骨架：",
        "```metal",
        brief.abi_stub.rstrip(),
        "```",
        "",
        "没有 `/* taic-launch */` 时，启动配置用契约里的。有的话，写成 "
        "`launch NAME grid=16,1,1 threads=128,1,1`。`grid=(16, 1, 1)` 和 `grid=16, 1, 1` 是同一个网格。"
        "除法必须整除。不是契约常量的名字，编译前就会被拒绝。`math fast` 或 `math safe` 只能有一行。",
        "最多 32 段。不许纹理。不许把任何旧核原文贴回来。",
        "着色器校验在你退出之后由外层做。插桩会把线程组内存报大，边界写保守些。",
        f"本场要交的文件是 `{submit}`。外层只读这个文件。开场时它不存在。",
        f"你写了它，外层就把里面的内容当作你交回的核。你不写它，外层记这场没有交回内核。",
        "`kernel.metal` 在有种子时已经放在本目录，只供阅读。外层不读它。改它不算交卷。",
        f"交卷文件始终放目前最好的一版。自测合格且更快，就用 `cp` 复制成 `{submit}`。还没有合格版时放最近一版，不抄种子。",
        "",
        "`experience.md` 是这个项目已经测过的东西：哪些启动契约与 tile 划算、哪些算子已经贴着带宽屋顶、"
        "哪个库换了算法。它是线索不是规则：你自己测出来的数比它高。",
        "",
        "# 前提要自己测",
        "成绩单、上一轮笔记和参考资料里关于这台机器的断言（某种操作数类型的矩阵乘加多快、某条指令存不存在、某个上限是多少）都是假设，不是事实。成绩单印了按操作数类型实测的矩阵峰值；再往细的速率没有人替你测过。若一句断言决定某个算法能不能做（例如「这种数据类型进矩阵单元就慢一倍」），先写一个几十行的小探针把它测出来，再决定放弃。只凭一句没测过的断言放弃的方向，不算试过；放弃时把这句断言原样写进 NOTES 并标「未验证」，下一轮可以重开。探针的写法在技能 `taic-tools` 里。",
        "",
        "# 验证命令",
        "只对答案，不计时：",
        "```",
        quick,
        "```",
        "",
        "# 评测命令",
        "对答案并计时。协议和外层相同。自测数字只供你判断，不进种子，也不进对外的加速比。",
        "```",
        timed,
        "```",
        "上面两条命令里的交卷文件和 report.json 是绝对路径。不要改成相对路径。",
        "命令会把一行结果追加到本目录的 `self_checks.jsonl`。出现过 timeout 或 crash 之后，再跑 `--timed` 会被拒绝。",
        "交卷前请在 `--timed` 后面加 `--validate` 跑最后一次：那一遍会开插桩校验（抓越界和竞争），外层也会做同一件事，早发现能省一场会话。",
        "这个目录最多记 20 次 `--timed`，累计 120 秒。超过之后 `--timed` 会拒绝，你仍可交卷，外层照常测。",
        "在本目录运行。GPU 锁是机器上那一把，不要改锁路径，也不要自己另写一把。",
        "一个不返回的核会冻住显示器。",
        "",
        "# 晋升标准",
        f"外层换种子要同时快过 {brief.min_improvement} 倍，并且超过 {brief.noise_z} 个标准误。",
        "这个门槛要用你这一版自己的 cv，所以这里不算一个目标微秒数。`--timed` 每次会印「够换种子」或「不够」。那一行注明自测不算数。",
    ]
    if brief.seedless:
        lines += [
            "",
            f"现在没有种子，也没有 `kernel.metal`。先把一颗核写进 `{submit}`。外层评测合格，它就成为种子。",
        ]
    lines += [
        "",
        "# 这个系统怎么工作",
        "你在一场接力里：两个委员会轮流，一轮一场新会话，对话不带过去。外层判卷是机械的。你自报的数字不进种子，也不进对外的加速比。",
        "固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档和队友留下的前提。你可以读整个仓库，只能写本目录。",
        f"本场的主产出始终是 `{submit}`。`NOTES.md` 和 `FEEDBACK.md` 当场追加，不要留到收尾。每试完一版，在 `NOTES.md` 记一行。每否决一个前提、每踩一个坑，就在 `FEEDBACK.md` 对应小节追加一行。",
        "给系统的建议只给人看，有人在看板复核后改仓库。给下一棒的避坑提示只在本运行内往下传，不跨题目。放弃某个方向所依据的断言必须写进「被否决的前提」节，不要夹在避坑里。",
        "**没有内容就不要写。** 你的输出按 token 计费：某一节没话说就整节省略，不要写「无」「（空）」「none」这类占位符——它们是纯成本，看板还要逐条清理。整份 `FEEDBACK.md` 都没有内容时，不要创建这个文件。",
        "文件开头一行 `kind: prompt|skill|tool|rule|decision`，可以写多个，用逗号分开；没有系统层面的建议或求助时，这一行也不要写。",
        "有内容时按这五个标题写，只写有话的那几节：`# 需要帮助`、`# 对系统的建议`、`# 卡住的原因`、`# 给下一棒的避坑提示`、`# 被否决的前提`。"
        "缺权限、sudo、许可证、凭据：当场写 `# 需要帮助`，能绕开也写。建议附 ```harness-proposal 块（见 HARNESS.md）。",
        "被否决的前提只写这两种：",
        "`[已实测: 文件名 | 一句命题 | 数字和单位]`",
        "`[未验证 | 一句命题 | 为什么没测]`",
        "探针格只写一个文件名。参数写进命题格或数字格。引用同运行别的会话，写 `r9.p0/sim_w63.py` 这种以 `agent_work` 为根的路径。",
        "「已实测」要写清测的是哪条命题。「会不会算错」和「误差多大」是两条。",
        "写完跑一次：",
        "```",
        _feedback_check_command(brief),
        "```",
        "比较两版快慢或做消融之前，先读技能 `taic-tools` 的「测量」节。动手前先读本目录的 `device.md` 和 `experience.md`。再查仓库 `docs/`、本运行 `agent_work/*/NOTES.md`。",
    ]
    lines += [
        "",
        "# 证据",
        "在本目录维护 `NOTES.md`：每一版改了什么、自测微秒和 cv、留下还是放弃、理由。",
        "涉及硬件上限的断言（速率、指令、容量）必须附探针文件名和实测数字；没有实测的写「未验证」。被否决的方向连同否决它的数字一起记，方便下一位复核。",
        "这是交给下一位的记录。下一位看不到这次对话。",
        "",
        f"交卷文件 `{submit}` 里一直放目前最好的一版。外层只读这个文件。",
    ]
    out = "\n".join(lines) + "\n"
    blocks: list[tuple[str, str]] = []
    if not brief.seedless:
        context = ["", "# 当前现场", f"种子在 `kernel.metal`。交卷写 `{submit}`，不要改种子文件。"]
        if brief.scorecard:
            context += ["", "外层成绩单：", _cap_task(brief.scorecard, 1000)]
        if brief.bound_line:
            context += ["", _cap_task(brief.bound_line, 300)]
        blocks.append(("当前现场", "\n".join(context) + "\n"))
        if brief.failure:
            blocks.append(("上一轮外层失败反馈", "\n\n上一轮外层失败反馈：\n" + _cap_task(brief.failure, 1000)))
        if overlay_text:
            blocks.append(("本运行 harness overlay", overlay_text))
        if brief.notes_summary:
            blocks.append((
                "上一轮 NOTES",
                "\n\n上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：\n" + _cap_task(brief.notes_summary, 1800),
            ))
        if brief.peer_tips:
            blocks.append(("本任务前序避坑提示", "\n\n" + _PEER_HEADER + "\n" + _cap_task(brief.peer_tips, 1500)))
        if brief.history:
            blocks.append(("最近的历史", "\n\n最近的历史：\n" + _cap_task(brief.history, 400)))
        if brief.evidence:
            blocks.append(("现场证据", "\n\n" + _cap_task(brief.evidence, 600)))
        blocks.append((
            "自测对照",
            "\n\n" + _cap_task(
                "合同里的外层成绩和你自己跑出来的数字对不上时，先把评测命令里的交卷文件换成 "
                + _shell(str(_work_path(brief) / "kernel.metal"))
                + " 跑一次 `--timed`，再改。交卷前的自测用上面的命令，文件名是交卷文件。",
                400,
            ),
        ))
    elif brief.peer_tips:
        blocks.append(("本任务前序避坑提示", "\n\n" + _PEER_HEADER + "\n" + _cap_task(brief.peer_tips, 1500)))

    omitted: list[str] = []
    # Reserve room for the omission note and for an active harness overlay. Without the
    # overlay reserve, a full scorecard can silently consume the whole budget and the
    # reviewed tip never reaches the next session.
    limit = max(0, max_bytes - 420 - len(overlay_text.encode("utf-8")))
    for name, text in blocks:
        room = limit - len(out.encode("utf-8"))
        if room <= 0:
            omitted.append(name)
            continue
        if name == "当前现场":
            # Keep the seed line and as much of the scorecard as the remaining room allows,
            # instead of dropping the whole block and leaving the agent without its scorecard.
            # Also leave the active overlay its reserved bytes even when the current-site
            # block would otherwise fit exactly.
            reserve = len(overlay_text.encode("utf-8")) if overlay_text else 0
            cap = max(0, room - reserve - 60)
            if cap >= 80:
                trimmed = _cap_task(text, cap)
                if trimmed:
                    out += trimmed
                    continue
            omitted.append(name)
            continue
        if len(text.encode("utf-8")) <= room:
            out += text
            continue
        omitted.append(name)
    if omitted:
        note = (
            "\n\n# 本轮省略（上下文预算）\n"
            f"以下段落因 {max_bytes} 字节预算未附：" + "、".join(omitted) + "。\n"
            "全文在本目录的 NOTES.md、FEEDBACK.md、feedback.jsonl 和 state.json；需要时读取，不要假设它们为空。\n"
        )
        room = max_bytes - len(out.encode("utf-8"))
        if room > 0:
            out += _clip_bytes(note, room)
    return out


def _self_check_summary(work: Path) -> tuple[int, int, float]:
    path = work / "self_checks.jsonl"
    if not path.is_file():
        return 0, 0, 0.0
    count = timeouts = 0
    gpu_s = 0.0
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        count += 1
        if row.get("status") in ("timeout", "crash"):
            timeouts += 1
        elapsed = row.get("elapsed_s")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            gpu_s += float(elapsed)
    return count, timeouts, gpu_s


def _guidance_source(brief: SessionBrief, name: str, fallback: Path) -> Path:
    """The frozen copy for this run when it exists, otherwise the file in the repository."""
    if brief.guidance_dir:
        snap = Path(brief.guidance_dir) / name
        if snap.is_file():
            return snap
    return fallback


def stage_guidance(work: Path, brief: SessionBrief) -> None:
    """Copy the device note, experience, and the language reference. A frozen snapshot wins when the run has one."""
    work.mkdir(parents=True, exist_ok=True)
    mapping = (
        ("device.md", skill_path("device", brief.target_id)),
        ("msl_reference.md", skill_path("msl_reference", brief.target_id)),
        ("experience.md", skill_path("experience", brief.target_id)),
    )
    snap = Path(brief.guidance_dir) if brief.guidance_dir else None
    for name, src in mapping:
        frozen = snap / name if snap is not None else None
        chosen = frozen if frozen is not None and frozen.is_file() else src
        if chosen.is_file():
            shutil.copyfile(chosen, work / name)



_WORK_KEEP = frozenset({"NOTES.md", "FEEDBACK.md", "self_checks.jsonl", "TASK.md", "HARNESS.md", ".selfcheck"})


_EVIDENCE_TOKEN = re.compile(r"[A-Za-z0-9_./-]+\.(?:py|metal|md|json|sh|txt)")


def _evidence_paths(work: Path) -> set[str]:
    """Top-level entries named by the session's own feedback or notes.

    A measured premise points at a probe file. Archiving that file would turn ``verified`` into
    ``downgraded`` on the next parse; keep the file (or its whole small directory) in place.
    """
    text = ""
    for name in ("FEEDBACK.md", "NOTES.md"):
        path = work / name
        try:
            text += path.read_text(encoding="utf-8", errors="replace") + "\n"
        except OSError:
            continue
    keep: set[str] = set()
    for token in _EVIDENCE_TOKEN.findall(text):
        clean = token.strip()
        if clean.startswith("PWD/"):
            clean = clean[4:]
        if clean.startswith("runs/"):
            continue
        parts = Path(clean).parts
        if not parts:
            continue
        # A token such as r11.p0/bench2.py belongs to a sibling work directory.
        if len(parts) > 1 and re.fullmatch(r"r\d+\.p\d+", parts[0]):
            continue
        top = work / parts[0]
        if top.exists():
            keep.add(parts[0])
    return keep


def prune_work(work: Path, *, label: str, keep: set[str] | None = None, max_bytes: int = SCRATCH_MAX_BYTES) -> dict:
    """Archive the session's oversized scratch and remove it from ``agent_work``.

    The files the search reads (NOTES, FEEDBACK, self_checks, TASK.md, kernel.metal, the
    submission) are kept. Everything else -- probe variants, ``v/`` ablation sources, ad-hoc
    scripts -- is tarred under ``<run>/scratch/<label>.tar.gz`` and deleted. If the archive
    itself exceeds ``max_bytes``, it is dropped and a JSON manifest records what was removed:
    the point is a hard cap, not an unbounded second copy.
    """
    work = Path(work)
    if not work.is_dir():
        return {"archived": [], "kept": [], "bytes": 0, "reason": "no work dir"}
    keep = set(_WORK_KEEP) | {str(name) for name in (keep or ())}
    keep |= _evidence_paths(work)
    entries = [path for path in work.iterdir() if path.name not in keep]
    kept = sorted(path.name for path in work.iterdir() if path.name in keep)
    if not entries:
        return {"archived": [], "kept": kept, "bytes": 0}

    def size_of(path: Path) -> int:
        if path.is_file():
            return path.stat().st_size
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    total = sum(size_of(path) for path in entries)
    run_dir = work.parent.parent
    scratch = run_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label or work.name).strip("-") or "session"
    archive = scratch / f"{safe}.tar.gz"
    index = 1
    while archive.exists():
        archive = scratch / f"{safe}.{index}.tar.gz"
        index += 1
    archived: list[str] = []
    reason = ""
    try:
        with tarfile.open(archive, "w:gz") as tar:
            for path in entries:
                tar.add(path, arcname=path.name)
                archived.append(path.name)
    except (OSError, tarfile.TarError) as exc:
        reason = f"archive failed: {type(exc).__name__}"
        archived = []
        archive.unlink(missing_ok=True)
    if archive.is_file() and archive.stat().st_size > max_bytes:
        reason = f"archive over {max_bytes} bytes"
        archive.unlink(missing_ok=True)
    if reason:
        (scratch / f"{safe}.json").write_text(json.dumps({
            "label": label, "reason": reason, "bytes": total,
            "entries": [{"name": path.name, "bytes": size_of(path)} for path in entries],
        }, indent=1), encoding="utf-8")
    for path in entries:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return {
        "archived": archived,
        "archive": str(archive) if archive.is_file() else None,
        "kept": kept,
        "bytes": total,
        "reason": reason,
    }


def run_kernel_agent(
    brief: SessionBrief,
    *,
    work: Path,
    provider: str,
    model: str,
    effort: str | None,
    prices: Prices,
    turn_output_tokens: int,
    turn_minutes: int,
    hmz: str | None = None,
    timeout_s: float | None = None,
) -> AgentResult:
    """Run one hmz session. A timeout still returns the kernel file; it does not raise.

    Raises:
      AgentError: hmz is not on PATH, or the process cannot be started.
    """
    binary = hmz or shutil.which("hmz")
    if not binary or not Path(binary).is_file():
        raise AgentError("hmz is not on PATH; install it with `uv tool install hmz`")
    work.mkdir(parents=True, exist_ok=True)
    submit = submission_name(brief.task_name, brief.dtype, brief.size, brief.label)
    submit_path = work / submit
    submit_path.unlink(missing_ok=True)
    seed_path = work / "kernel.metal"
    if brief.seed_source and not brief.seedless:
        seed_path.write_text(brief.seed_source)
    else:
        seed_path.unlink(missing_ok=True)
    if brief.seed_report_json and not brief.seedless:
        (work / "report.json").write_text(brief.seed_report_json)
    stage_guidance(work, brief)
    from .harness import write_template
    write_template(work)
    (work / "TASK.md").write_text(render_contract(brief))
    if timeout_s is None:
        timeout_s = process_timeout_s(turn_minutes)
    flow_minutes = int(turn_minutes) + WRAP_MINUTES + FLOW_SLACK_MINUTES
    argv = [
        binary, "exec", "-f", str(FLOW),
        "-a", f"assistant={agent_spec(provider, model, effort)}",
        "-p", f"turn_output_tokens={int(turn_output_tokens)}",
        "-p", f"turn_minutes={int(turn_minutes)}",
        "-p", f"wrap_output_tokens={WRAP_OUTPUT_TOKENS}",
        "-p", f"wrap_minutes={WRAP_MINUTES}",
        "-p", f"submit_name={submit}",
        "-p", f"budget.duration={flow_minutes}m",
        "--json",
        "--", f"Read TASK.md in this directory and do it. Write the finished kernel to {submit}. The search does not read kernel.metal.",
    ]
    status = "ok"
    error = ""
    try:
        from tcenv.children import forget, kill_group, popen_group

        proc = popen_group(
            argv, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
    except OSError as exc:
        raise AgentError(f"could not start hmz: {exc}") from exc
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                kill_group(proc)
                stdout, stderr = "", ""
            status = "timeout"
            error = f"hmz did not finish within {timeout_s:.0f}s"
    finally:
        kill_group(proc)
        forget(proc)
    stream = (stdout or "") + "\n" + (stderr or "")
    spent, saw_result = spend_from_stream(stream)
    cost = dollars_for(spent, prices) if saw_result else None
    written = submit_path.read_text() if submit_path.is_file() else ""
    source = kernel_text(written, brief.kernel_name)
    notes = ""
    notes_path = work / "NOTES.md"
    if notes_path.is_file():
        notes = notes_path.read_text()
    from .agent_feedback import read_feedback_text

    feedback, _over = read_feedback_text(work / "FEEDBACK.md")
    protected = _self_check_summary(_self_check_dir(brief))
    beside = _self_check_summary(work)
    checks, timeouts, gpu_s = protected[0] + beside[0], protected[1] + beside[1], protected[2] + beside[2]
    if status != "timeout" and proc.returncode not in (0, None):
        status = "failed"
        error = (stderr or stdout or "").strip()[-400:]
    pruned = prune_work(
        work, label=brief.label,
        keep={submission_name(brief.task_name, brief.dtype, brief.size, brief.label)},
    )
    return AgentResult(
        source=source, cost_usd=cost, cost_incomplete=not saw_result, status=status, error=error,
        spent=spent, notes=notes, feedback=feedback, self_checks=checks, self_check_timeouts=timeouts, self_check_gpu_s=gpu_s,
        pruned=pruned,
    )


def agent_for_script(script, kernel_name: str):
    """A stand-in used by tests whose fake chat client is the only model. One call is one session.

    The first seedless call is the draft. Later calls ask for plan A, then B, then C, which is how the
    old scripted tests tell their answers apart. A real search does not use this.
    """
    step = {"n": 0}

    def run(brief: SessionBrief, tag: str):
        del tag
        if brief.seedless and brief.failure:
            last = "The evaluation of your kernel failed\n" + brief.failure
        elif brief.seedless:
            last = "Translate this Triton kernel"
        else:
            title = "ABC"[step["n"] % 3]
            step["n"] += 1
            last = f"Implement exactly this optimization plan **{title}**\n" + (brief.history or "")
        resp = script([{"role": "user", "content": last}])
        text = ""
        choices = getattr(resp, "choices", None)
        if choices:
            text = choices[0].message.content or ""
        elif isinstance(resp, str):
            text = resp
        source = kernel_text(text, kernel_name)
        usage = getattr(resp, "usage", None)
        if usage is None:
            cost = 0.01
        else:
            hit = float(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
            miss = float(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
            out = float(getattr(usage, "completion_tokens", 0) or 0)
            cost = hit * 0.006 / 1_000_000 + miss * 0.30 / 1_000_000 + out * 1.20 / 1_000_000
        return AgentResult(
            source=source, cost_usd=cost, cost_incomplete=False,
            status="ok" if source else "failed", error="" if source else "no kernel",
        )

    return run


def preflight(committees, credentials: Path, hmz: str | None = None) -> None:
    """Fail before any session when hmz, the flow, a credential, or an output price is missing."""
    binary = hmz or shutil.which("hmz")
    if not binary:
        raise AgentError("hmz is not on PATH; install it with `uv tool install hmz`")
    if not FLOW.is_dir():
        raise AgentError(f"kernel flow is missing: {FLOW}")
    for committee in committees:
        member = committee.members[0]
        effort = member.cfg.reasoning_effort
        spec = agent_spec(member.provider, member.model, effort)
        if not re.fullmatch(r"pi/[^/\s]+/[^:\s]+(?::[A-Za-z0-9._-]+)?", spec):
            raise AgentError(f"hmz would not accept -a assistant={spec}")
        load_prices(credentials, member.provider, member.model)
