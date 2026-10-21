"""Write one kernel by asking the pi coding agent, through hmz, to edit a file.

The search does not call the chat API to write kernels. One hmz session reads ``TASK.md``,
edits ``kernel.metal``, and may time itself with ``tcenv eval --timed``. The search times the
file the agent leaves behind and decides whether it stays. A number the agent prints is not
the search result.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError
from .parse import extract_metal

FLOW = Path(__file__).resolve().parent / "flows" / "kernel_once"
SKILL = Path(__file__).resolve().parent / "skills" / "msl_reference.md"
# 实测过的经验条目（KernelWiki 的经验层）。和 msl_reference 一起放进工作区，只读。
EXPERIENCE = Path(__file__).resolve().parent / "skills" / "wiki" / "experience.md"
_KERNEL = re.compile(r"\bkernel\s+void\s+(\w+)\b")
_USAGE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
# 一场会话的墙钟。hmz 的回合预算先到；这个秒数只在进程不退出时杀掉它。
PROCESS_TIMEOUT_S = 25 * 60 + 3 * 60 + 2 * 60 + 60
_PEER_HEADER = "本任务前序避坑提示（队友交接，仅限本运行参考；凡未附探针文件名与实测数字的硬件/算法上限断言，一律按未验证处理，鼓励你写探针翻案）："
WRAP_OUTPUT_TOKENS = 4000
WRAP_MINUTES = 3
FLOW_SLACK_MINUTES = 2


class AgentError(LLMError):
    """hmz or pi could not take this turn."""


@dataclass
class Prices:
    """USD per million tokens. Read from the credentials file. Never logged."""

    input: float
    output: float
    cache_read: float
    cache_write: float


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


@dataclass
class AgentResult:
    """What one hmz session left on disk, plus the dollars we computed from its token counts."""

    source: str | None
    cost_usd: float | None
    cost_incomplete: bool
    status: str  # ok, timeout, failed
    error: str = ""
    spent: dict = field(default_factory=dict)
    notes: str = ""
    feedback: str = ""
    self_checks: int = 0
    self_check_timeouts: int = 0
    self_check_gpu_s: float = 0.0


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
        cache_write=float(found.get("cacheWrite", 0) or 0),
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


def _check_command(brief: SessionBrief, kernel_path: Path, mode: str) -> str:
    parts = [
        f"uv run --directory {brief.repo} python -m tcenv.cli eval {brief.task_name} {kernel_path.name}",
        f"--dtype {brief.dtype}",
        f"--size {brief.size}",
        "--quick" if mode == "quick" else "--timed",
    ]
    if mode == "timed":
        parts.append(f"--reps {int(brief.reps)}")
        parts.append(f"--min-improvement {brief.min_improvement}")
        parts.append(f"--noise-z {brief.noise_z}")
        if brief.seed_source and not brief.seedless:
            parts.append("--seed-report report.json")
        if brief.require_contract_launch:
            parts.append("--require-contract-launch")
    elif brief.require_contract_launch:
        parts.append("--require-contract-launch")
    return " ".join(parts)


def render_contract(brief: SessionBrief, kernel_path: Path) -> str:
    """The KDA-shaped task contract written to ``TASK.md``."""
    quick = _check_command(brief, kernel_path, "quick")
    timed = _check_command(brief, kernel_path, "timed")
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
        "最后一版不合格也留在 kernel.metal 里。不要为了过检查贴回旧核。",
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
            "现在没有种子。先写出一颗外层评测合格的核。外层评测合格，它就成为种子。",
        ]
    lines += [
        "",
        "# 这个系统怎么工作",
        "你在一场接力里：两个委员会轮流，一轮一场新会话，对话不带过去。往下传的有 `kernel.metal`、`NOTES.md`、外层成绩单，以及本运行内的「队友避坑提示」。",
        "外层判卷是机械的：全量评测、三级贴核检查、确认计时绑源码指纹。你自报的数字不进种子，也不进发布加速比。",
        "固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档、提示词和队友留下的前提。",
        "你可以读整个仓库（`docs/`、评测源码），只能写本目录。",
        "本场的主产出始终是 `kernel.metal`。`FEEDBACK.md` 写短、最后写。给系统的建议只给人看，有人在看板复核后改仓库。给下一棒的避坑提示只在本运行内往下传，不跨题目。放弃某个方向所依据的断言必须写进「被否决的前提」节，不要夹在避坑里。",
        "文件开头一行 `kind: prompt|skill|tool|rule|decision|none`，可以写多个，用逗号分开。只留避坑、没有系统建议时写 `none`。",
        "然后按这五个标题写，缺的可以留空：`# 需要帮助`、`# 对系统的建议`、`# 卡住的原因`、`# 给下一棒的避坑提示`、`# 被否决的前提`。",
        "被否决的前提里，每一条写成 `[已实测: 探针文件 | 测的命题 | 数字]` 或 `[未验证]`。",
    ]
    if not brief.seedless:
        lines += ["", "# 当前现场", "种子 `kernel.metal` 已经放在本目录。"]
        if brief.scorecard:
            lines += ["", "外层成绩单：", brief.scorecard.strip()]
        if brief.bound_line:
            lines += ["", brief.bound_line.strip()]
        if brief.failure:
            lines += ["", "上一轮外层失败反馈：", brief.failure.strip()]
        if brief.history:
            lines += ["", "最近的历史：", brief.history.strip()]
        if brief.notes_summary:
            lines += ["", "上一轮 NOTES（硬件上限类断言未附实测数字的，按未验证处理）：", brief.notes_summary.strip()]
        if brief.peer_tips:
            lines += ["", _PEER_HEADER, brief.peer_tips.strip()]
        if brief.evidence:
            lines += ["", brief.evidence.strip()]
        lines += [
            "",
            "合同里的外层成绩和你自己跑出来的数字对不上时，先对种子原样跑一次 `--timed`，再改。",
        ]
    elif brief.peer_tips:
        lines += ["", _PEER_HEADER, brief.peer_tips.strip()]
    lines += [
        "",
        "# 证据",
        "在本目录维护 `NOTES.md`：每一版改了什么、自测微秒和 cv、留下还是放弃、理由。",
        "涉及硬件上限的断言（速率、指令、容量）必须附探针文件名和实测数字；没有实测的写「未验证」。被否决的方向连同否决它的数字一起记，方便下一位复核。",
        "这是交给下一位的记录。下一位看不到这次对话。",
        "",
        f"改完后把核留在 `{kernel_path.name}`。外层只读这个文件。",
    ]
    return "\n".join(lines) + "\n"


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
    timeout_s: float = PROCESS_TIMEOUT_S,
) -> AgentResult:
    """Run one hmz session. A timeout still returns the kernel file; it does not raise.

    Raises:
      AgentError: hmz is not on PATH, or the process cannot be started.
    """
    binary = hmz or shutil.which("hmz")
    if not binary or not Path(binary).is_file():
        raise AgentError("hmz is not on PATH; install it with `uv tool install hmz`")
    work.mkdir(parents=True, exist_ok=True)
    kernel_path = work / "kernel.metal"
    kernel_path.write_text(brief.seed_source or "")
    if brief.seed_report_json and not brief.seedless:
        (work / "report.json").write_text(brief.seed_report_json)
    if SKILL.exists():
        shutil.copyfile(SKILL, work / "msl_reference.md")
    if EXPERIENCE.exists():
        shutil.copyfile(EXPERIENCE, work / "experience.md")
    (work / "TASK.md").write_text(render_contract(brief, kernel_path))
    flow_minutes = int(turn_minutes) + WRAP_MINUTES + FLOW_SLACK_MINUTES
    argv = [
        binary, "exec", "-f", str(FLOW),
        "-a", f"assistant={agent_spec(provider, model, effort)}",
        "-p", f"turn_output_tokens={int(turn_output_tokens)}",
        "-p", f"turn_minutes={int(turn_minutes)}",
        "-p", f"wrap_output_tokens={WRAP_OUTPUT_TOKENS}",
        "-p", f"wrap_minutes={WRAP_MINUTES}",
        "-p", f"budget.duration={flow_minutes}m",
        "--json",
        "--", "Read TASK.md in this directory and do it. Leave the finished kernel in kernel.metal.",
    ]
    status = "ok"
    error = ""
    try:
        proc = subprocess.Popen(
            argv, cwd=work, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True, env=os.environ.copy(),
        )
    except OSError as exc:
        raise AgentError(f"could not start hmz: {exc}") from exc
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        stdout, stderr = proc.communicate()
        status = "timeout"
        error = f"hmz did not finish within {timeout_s:.0f}s"
    stream = (stdout or "") + "\n" + (stderr or "")
    spent, saw_result = spend_from_stream(stream)
    cost = dollars_for(spent, prices) if saw_result else None
    written = kernel_path.read_text() if kernel_path.exists() else ""
    source = kernel_text(written, brief.kernel_name)
    notes = ""
    notes_path = work / "NOTES.md"
    if notes_path.is_file():
        notes = notes_path.read_text()
    from .agent_feedback import read_feedback_text

    feedback, _over = read_feedback_text(work / "FEEDBACK.md")
    checks, timeouts, gpu_s = _self_check_summary(work)
    if status != "timeout" and proc.returncode not in (0, None):
        status = "failed"
        error = (stderr or stdout or "").strip()[-400:]
    return AgentResult(
        source=source, cost_usd=cost, cost_incomplete=not saw_result, status=status, error=error,
        spent=spent, notes=notes, feedback=feedback, self_checks=checks, self_check_timeouts=timeouts, self_check_gpu_s=gpu_s,
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
