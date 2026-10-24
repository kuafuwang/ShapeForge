"""Write one kernel by asking the pi coding agent, through hmz, to edit a file.

The search does not call the chat API to write kernels. One hmz session reads ``TASK.md``
and writes one submission file whose name is ``submission_name``. ``kernel.metal`` is the
seed, placed before the session, and the search does not read it. A missing submission
means this session returned no kernel. A number the agent prints is not the search result.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import tarfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMError
from .parse import extract_source

from .target import skill_path

FLOW = Path(__file__).resolve().parent / "flows" / "kernel_once"
SKILL = skill_path("msl_reference")
EXPERIENCE = skill_path("experience")
_KERNEL = re.compile(r"\bkernel\s+void\s+(\w+)\b")
_USAGE = re.compile(r"([A-Za-z_]+)=(-?\d+(?:\.\d+)?)")
_NAME_PART = re.compile(r"[^A-Za-z0-9._-]+")
_DIAG_SECRET = re.compile(r"sk-[A-Za-z0-9_\-]{12,}")
_DIAG_PREVIEW_CHARS = 240
# 一场会话的墙钟。hmz 的回合预算先到；这个秒数只在进程不退出时杀掉它。
def process_timeout_s(turn_minutes: int) -> float:
    """@brief 算出杀掉 hmz 进程的墙钟秒数。

    hmz 自己的流程预算（回合分钟 + 收尾分钟 + 流程余量）通常先到；这个秒数只用在进程
    不肯退出时，再多留一分钟，让更长的回合仍然能拿到收尾。

    @param turn_minutes 一场会话的回合预算，单位为分钟。
    @return 进程超时的秒数。
    """
    return (int(turn_minutes) + WRAP_MINUTES + FLOW_SLACK_MINUTES + 1) * 60


_PEER_HEADER = "本任务前序避坑提示（队友交接，仅限本运行参考；凡未附探针文件名与实测数字的硬件/算法上限断言，一律按未验证处理，鼓励你写探针翻案）："
WRAP_OUTPUT_TOKENS = 4000
WRAP_MINUTES = 3
FLOW_SLACK_MINUTES = 2
PROCESS_TIMEOUT_S = process_timeout_s(25)
TASK_MAX_BYTES = 12_000
SCRATCH_MAX_BYTES = 50_000_000


class AgentError(LLMError):
    """@brief hmz 或 pi 无法完成这一回合。"""


@dataclass
class Prices:
    """@brief 每百万 token 的美元单价。

    价格从凭据文件读取。日志里永远不打印这些费率。
    """

    input: float
    output: float
    cache_read: float
    cache_write: float

    def as_dict(self) -> dict:
        """@brief 取记在一场会话花费旁边的费率。

        记下来的是当时的单价，后来的读者因此不必依赖今天的价格文件。

        @return 四个费率的字典：``input``、``output``、``cache_read``、``cache_write``。
        """
        return {"input": self.input, "output": self.output, "cache_read": self.cache_read, "cache_write": self.cache_write}


@dataclass
class SessionBrief:
    """@brief 一场编码 agent 会话的契约。

    字段由编排器填；本模块只负责把它渲染成 `TASK.md` 的文本。
    """

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
    file_ext: str = ".metal"
    language_label: str = "Metal"
    source_fence: str = "metal"
    reference_skill: str = "msl_reference"

    @property
    def seed_name(self) -> str:
        """@brief 种子文件名。

        @return ``kernel`` 加本语言的文件后缀，默认 ``kernel.metal``。
        """
        return f"kernel{self.file_ext}"


@dataclass
class AgentResult:
    """@brief 一场 hmz 会话留在磁盘上的东西，以及按它的 token 数算出的美元。

    ``status`` 取 ``ok``、``timeout``、``failed`` 三者之一：只有超时算 GPU 不应答，
    worker 崩溃是 harness 侧失败，单独计数，不触发挂死熔断。
    """

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
    # A worker crash is not a GPU hang: it is a harness-side failure, and only timeouts trip the
    # hang breaker. The two are counted apart so ``max_hangs=1`` cannot end a run on a SchemaError.
    self_check_crashes: int = 0
    self_check_gpu_s: float = 0.0
    # Structured hmz/pi diagnostics. These explain an agent_failed without changing its meaning.
    diagnostics: dict = field(default_factory=dict)
    diagnostics_path: str = ""


def epic_spend_by_label(run_dir: Path, *, root: Path | None = None) -> dict[str, float] | None:
    """@brief 一场运行的 hmz epic 花费，按会话标签汇总。

    A killed parent can leave its hmz session running: the session keeps calling the provider and
    writes its ``usage`` to ``~/.hmz/epics/.../epic.jsonl``, but the transcript never gets an agent
    event. The workspace basename is the session label (``init``, ``r2.p0``), so the caller can
    compare these totals with the labels the transcript did record and carry only the difference.

    @param run_dir 运行目录。
    @param root epic 根目录，缺省 ``~/.hmz/epics``（测试可注入）。
    @return ``{label: cost}``；没有可匹配的 epic 时为 ``None``。
    """
    base = Path(root) if root is not None else Path.home() / ".hmz" / "epics"
    if not base.is_dir():
        return None
    work = (Path(run_dir) / "agent_work").resolve()
    totals: dict[str, float] = {}
    for path in base.rglob("epic.jsonl"):
        workspace = ""
        cost = 0.0
        seen_usage = False
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "began" and isinstance(event.get("workspace"), str):
                workspace = event["workspace"]
            elif event.get("event") == "usage":
                number = event.get("cost")
                if isinstance(number, (int, float)) and not isinstance(number, bool):
                    cost += float(number)
                    seen_usage = True
        if not workspace or not seen_usage:
            continue
        try:
            if not Path(workspace).resolve().is_relative_to(work):
                continue
        except (OSError, ValueError):
            continue
        label = Path(workspace).name
        totals[label] = totals.get(label, 0.0) + cost
    return totals or None


def epic_spend(run_dir: Path, *, root: Path | None = None) -> float | None:
    """@brief ``epic_spend_by_label`` 的合计；没有可匹配的 epic 时为 ``None``。"""
    totals = epic_spend_by_label(run_dir, root=root)
    return sum(totals.values()) if totals else None


def submission_name(task: str, dtype: str, size: str, label: str, file_ext: str = ".metal") -> str:
    """@brief 算出本场会话必须创建的交卷文件名。

    名称是 ``{task}-{dtype}-{size}-{label}{file_ext}``。``label`` 是会话标识
    （``init``、``r3.p0``），所以同一次运行里的两场会话不会撞名。字母、数字、``.``、
    ``_`` 和 ``-`` 之外的字符一律变成 ``-``。外层不负责创建这个文件。

    @param task 任务名。
    @param dtype 数据类型。
    @param size 规模。
    @param label 会话标签，如 ``init``、``r3.p0``。
    @param file_ext 文件后缀，缺省 ``.metal``；没有点时自动补一个点。
    @return 交卷文件名。
    """
    parts = []
    for raw in (task, dtype, size, label):
        piece = _NAME_PART.sub("-", str(raw).strip()).strip("-.")
        parts.append(piece or "x")
    return "-".join(parts) + (file_ext if str(file_ext).startswith(".") else "." + str(file_ext))


def kernel_text(text: str, kernel_name: str, *, source_fence: str = "metal") -> str | None:
    """@brief 取出 ``text`` 里的内核源码。

    优先用围栏块；没有围栏时，整份文件只要声明了内核就算数。``source_fence`` 不是
    ``metal`` 时只认那个语言的围栏标签。

    @param text agent 写出的原始文本。
    @param kernel_name 期望的内核函数名。
    @param source_fence 围栏语言标签，缺省 ``metal``。
    @return 源码文本；取不到时为 ``None``。
    """
    tags = None if source_fence == "metal" else {source_fence}
    fenced = extract_source(text, kernel_name, tags=tags)
    if fenced:
        return fenced
    body = (text or "").strip("\n")
    if "taic-launch" in body and _KERNEL.search(body):
        return body + "\n"
    if _KERNEL.search(body) and re.search(r"\bkernel\s+void\s+" + re.escape(kernel_name) + r"\b", body):
        return body + "\n"
    return None


def agent_spec(provider: str, model: str, effort: str | None = None) -> str:
    """@brief 拼出 ``-a assistant=`` 的值。

    pi 的模型名写成 ``provider/id``，推理强度写成 ``:effort`` 后缀。

    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度；为空时只写 ``provider/id``。
    @return 传给 hmz 的 assistant 规格字符串。
    """
    spec = f"pi/{provider}/{model}"
    if effort:
        spec = f"{spec}:{effort}"
    return spec


def load_prices(path: Path, provider: str, model: str) -> Prices:
    """@brief 读某个模型的单价。

    同一份文件里还存着 API key：本函数不返回它们，调用方也不得把整份文档写进日志。

    @param path 凭据与价格文件的路径。
    @param provider provider 名。
    @param model 模型名。
    @return 该模型的四项费率。
    @raises AgentError 文件读不出或不是合法 JSON、provider 没有凭据、模型缺失，
        或输出价格不是正数时。
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
    """@brief 算 ``usd`` 在 ``price_out_per_million`` 下能买多少输出 token。

    @param usd 预算的美元数。
    @param price_out_per_million 每百万输出 token 的美元单价。
    @return token 数，至少 1。
    @raises AgentError 输出单价不是正数时。
    """
    if price_out_per_million <= 0:
        raise AgentError("output price must be positive")
    return max(1, int(float(usd) / float(price_out_per_million) * 1_000_000))


def dollars_for(spent: dict, prices: Prices) -> float:
    """@brief 按 token 数算一场会话的美元花费。

    ``reasoning`` 不计价：pi 已经把它算进 output 里。费率表里没有的计数类型直接跳过。

    @param spent token 计数，键为 ``input``、``output``、``cache_read``、``cache_write``。
    @param prices 这场会话用的费率。
    @return 美元花费。
    """
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
    """@brief 从 hmz 的一行 ``result`` 里取 token 计数。

    ``hmz exec --json`` 用 ``default=str`` 写 ``spent``，而 ``Usage`` 不是 dict，所以这个字段是
    字符串 ``Usage(input=1, output=2)``；传进来是 dict 也接受。

    @param value ``spent`` 字段的原始值。
    @return 计数类型的 token 数；认不出来时为空字典。
    """
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if isinstance(value, str) and value.startswith("Usage("):
        return {name: float(num) for name, num in _USAGE.findall(value)}
    return {}


# hmz closes a turn either way. ``kind=failed`` carries the same ``spent`` record as ``result``
# (hmz/coganchor/agents/pi.py: _answered), so billing only the result lines under-counts whatever
# a failed turn already cost. The two kinds are exclusive per turn, so they cannot double count.
_SPENT_EVENTS = frozenset({"result", "failed"})


def spend_from_stream(text: str) -> tuple[dict[str, float], bool]:
    """@brief 把每个关回合的行上的 ``spent`` 累加起来。

    hmz 不管成败都关掉一回合，``kind=result`` 与 ``kind=failed`` 都带同样的 ``spent`` 记录，
    所以两种行都算。每行是一个回合而不是累计值，因此逐行相加。

    @param text hmz 的 stdout 与 stderr 拼起来的流。
    @return ``(各计数类型的总和, 是否见过至少一个 token 计数)``。后者决定 ``cost_incomplete``：
        它说的是有没有账单，不是这一回合有没有成功。
    """
    totals: dict[str, float] = {}
    saw = False
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("kind") not in _SPENT_EVENTS:
            continue
        parsed = parse_spent(obj.get("spent"))
        if not parsed:
            continue
        saw = True
        for kind, tokens in parsed.items():
            totals[kind] = totals.get(kind, 0.0) + tokens
    return totals, saw


def _diagnostic_text(value, limit: int) -> str:
    """Return a bounded diagnostic string with common API-key forms redacted."""
    return _DIAG_SECRET.sub("<redacted>", str(value or "").strip())[:limit]


def diagnostics_from_stream(text: str) -> dict:
    """Read the structured health of an ``hmz exec --json`` stream.

    hmz already exposes the backend session id and maps pi's final ``errorMessage`` to a
    ``kind=failed`` event. Keep those facts beside the token accounting instead of requiring a
    forensic search through the global pi session directory. The final response text is private to
    this function: ``run_kernel_agent`` may save a short preview only when no kernel was submitted.
    """
    sessions: list[str] = []
    result_turns = failed_turns = tool_calls = 0
    last_kind = ""
    last_error = ""
    response = ""
    for line in text.splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        session = obj.get("session")
        if isinstance(session, str) and session and session not in sessions:
            sessions.append(session)
        kind = obj.get("kind")
        if kind == "tool":
            tool_calls += 1
        if kind not in _SPENT_EVENTS:
            continue
        last_kind = str(kind)
        if kind == "failed":
            failed_turns += 1
            last_error = _diagnostic_text(obj.get("text") or "the turn failed", 400)
            response = ""
        else:
            result_turns += 1
            response = str(obj.get("text") or "")
    return {
        "schema_version": 1,
        "session_ids": sessions,
        "result_turns": result_turns,
        "failed_turns": failed_turns,
        "tool_calls": tool_calls,
        "last_terminal_kind": last_kind or None,
        "last_error": last_error,
        "response_chars": len(response),
        "response_sha256": hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest() if response else None,
        "_response_text": response,
    }


def _shell(value: str) -> str:
    """@brief 给要拼进 shell 命令的一段文本加引号。

    @param value 原始值。
    @return 可安全拼进 shell 命令的单引号形式。
    """
    return shlex.quote(str(value))


def _self_check_dir(brief: SessionBrief) -> Path:
    """@brief 取本场会话的自测挂死日志目录，放在 agent 沙箱写得进的会话工作目录里。

    早先的版本把这个目录放在 ``<run>/session_checks``，以免删掉工作副本会把挂死计数清零。
    结果沙箱拒绝在那里 ``mkdir``，每次计时的自测在付过基准费用之后都崩掉（17 份 agent
    报告）。工作目录本来就是 agent 的信任边界，所以让日志保持可写，并把它随会话一起归档。

    @param brief 本场会话的契约。
    @return ``<工作目录>/.selfcheck`` 路径。
    """
    work = _work_path(brief)
    return work / ".selfcheck"


def _work_path(brief: SessionBrief) -> Path:
    """@brief 取本场会话的工作目录。

    契约里给了 ``work_dir`` 就用它（解析成绝对路径），否则用当前目录。

    @param brief 本场会话的契约。
    @return 工作目录的绝对路径。
    """
    if brief.work_dir:
        return Path(brief.work_dir).resolve()
    return Path.cwd()


def _check_command(brief: SessionBrief, mode: str) -> str:
    """@brief 拼自测/评测命令，路径一律写成绝对路径。

    ``uv run --directory`` 会把工作目录换掉，所以 repo 与交卷文件都用绝对路径。

    @param brief 本场会话的契约。
    @param mode ``quick`` 只对答案，``timed`` 对答案并计时。
    @return 一条可直接粘进 shell 的命令字符串。
    """
    name = submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
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
    """@brief 拼出检查 ``FEEDBACK.md`` 格式的命令。

    @param brief 本场会话的契约。
    @return 在会话工作目录里跑 ``taic.agent_feedback check`` 的命令字符串。
    """
    work = _work_path(brief)
    repo = Path(brief.repo).resolve()
    return " ".join([
        "uv", "run", "--directory", _shell(str(repo)),
        "python", "-m", "taic.agent_feedback", "check", _shell(str(work / "FEEDBACK.md")),
        "--work", _shell(str(work)),
    ])


def _clip_bytes(text: str, limit: int) -> str:
    """@brief 按字节预算截断 UTF-8 文本，且不切碎字符。

    @param text 待截断的文本。
    @param limit 字节上限。
    @return 不超预算时为原文，否则为截断后的文本；边界上的多字节字符被丢掉而不是切一半。
    """
    raw = (text or "").encode("utf-8")
    if len(raw) <= limit:
        return text
    return raw[: max(0, limit)].decode("utf-8", "ignore")


def _cap_task(text: str, limit: int) -> str:
    """@brief 按字节预算裁掉一个动态段落。

    全文仍然留在磁盘上，裁剪处会写一行标记说明这件事。

    @param text 段落文本。
    @param limit 字节上限。
    @return 首尾去空白的段落；超预算时截断并追加一行截断标记。
    """
    body = (text or "").strip()
    if len(body.encode("utf-8")) <= limit:
        return body
    return _clip_bytes(body, max(0, limit)).rstrip() + "\n…（本段已截断；全文见本目录文件）"


def render_contract(brief: SessionBrief, *, max_bytes: int = TASK_MAX_BYTES) -> str:
    """@brief 渲染写进 ``TASK.md`` 的 KDA 形状任务契约，动态段有硬字节预算。

    契约、签名、命令与证据规则是必需段。成绩单、失败反馈、NOTES、队友提示、历史与额外证据
    按优先级在预算内依次加入；被省掉的段落会点名仍然保存它的文件名。

    @param brief 本场会话的契约。
    @param max_bytes 整份 TASK.md 的字节预算，缺省 ``TASK_MAX_BYTES``。
    @return TASK.md 的完整文本（末尾一个换行）。
    """
    quick = _check_command(brief, "quick")
    timed = _check_command(brief, "timed")
    submit = submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
    overlay_text = ("\n" + _cap_task(brief.harness_overlay, 1800)) if brief.harness_overlay else ""
    lines = [
        f"# 目标",
        f"把 Triton 算子 `{brief.task_name}` 写成 {brief.language_label}，dtype `{brief.dtype}`，规模 `{brief.size}`。",
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
        f"```{brief.source_fence}",
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
        f"`{brief.seed_name}` 在有种子时已经放在本目录，只供阅读。外层不读它。改它不算交卷。",
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
            f"现在没有种子，也没有 `{brief.seed_name}`。先把一颗核写进 `{submit}`。外层评测合格，它就成为种子。",
        ]
    lines += [
        "",
        "# 这个系统怎么工作",
        "你在一场接力里：两个委员会轮流，一轮一场新会话，对话不带过去。外层判卷是机械的。你自报的数字不进种子，也不进对外的加速比。",
        "固定的是合同签名、容差、围栏和预算。可质疑的是笔记、文档和队友留下的前提。你可以读整个仓库，只能写本目录。",
        f"本场的主产出始终是 `{submit}`；先写一版可交的核，再实验。`NOTES.md`/`FEEDBACK.md` 当场追加；每版记一行，每个坑或否决前提也记一行。",
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
        # An active reviewed tip is more important than the scorecard. Put it before the
        # optional current-site block so a nearly full contract cannot squeeze it out.
        if overlay_text:
            blocks.append(("本运行 harness overlay", overlay_text))
        context = ["", "# 当前现场", f"种子在 `{brief.seed_name}`。交卷写 `{submit}`，不要改种子文件。"]
        if brief.scorecard:
            context += ["", "外层成绩单：", _cap_task(brief.scorecard, 1000)]
        if brief.bound_line:
            context += ["", _cap_task(brief.bound_line, 300)]
        blocks.append(("当前现场", "\n".join(context) + "\n"))
        if brief.failure:
            blocks.append(("上一轮外层失败反馈", "\n\n上一轮外层失败反馈：\n" + _cap_task(brief.failure, 1000)))
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
                + _shell(str(_work_path(brief) / brief.seed_name))
                + " 跑一次 `--timed`，再改。交卷前的自测用上面的命令，文件名是交卷文件。",
                400,
            ),
        ))
    elif brief.peer_tips:
        blocks.append(("本任务前序避坑提示", "\n\n" + _PEER_HEADER + "\n" + _cap_task(brief.peer_tips, 1500)))

    omitted: list[str] = []
    # The active overlay is already first in ``blocks``. Reserve only a small note when
    # the overlay is present; reserving the full note as well would leave no room for a
    # nearly full contract plus the reviewed tip.
    limit = max(0, max_bytes - (200 if overlay_text else 420))
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
            cap = max(0, room - 120)
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
        full_note = (
            "\n\n# 本轮省略（上下文预算）\n"
            f"以下段落因 {max_bytes} 字节预算未附：" + "、".join(omitted) + "。\n"
            "全文在本目录的 NOTES.md、FEEDBACK.md、feedback.jsonl 和 state.json；需要时读取，不要假设它们为空。\n"
        )
        compact_note = (
            "\n\n# 本轮省略（上下文预算）\n"
            "全文在本目录的 NOTES.md、FEEDBACK.md、feedback.jsonl 和 state.json；需要时读取，不要假设它们为空。\n"
        )
        note = compact_note if overlay_text else full_note
        room = max_bytes - len(out.encode("utf-8"))
        if room > 0:
            out += _clip_bytes(note, room)
    return out


def _self_check_summary(work: Path) -> tuple[int, int, float, int]:
    """@brief 汇总自测日志。

    超时说明 GPU 不再应答；崩溃是 worker 侧的错误（SchemaError、OOM、被杀掉的子进程），
    不能触发挂死熔断。

    @param work 会话工作目录。
    @return ``(checks, timeouts, gpu_s, crashes)``，即自测次数、超时数、GPU 秒数与崩溃数；
        日志不存在时为全 0。
    """
    path = work / "self_checks.jsonl"
    if not path.is_file():
        return 0, 0, 0.0, 0
    count = timeouts = crashes = 0
    gpu_s = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        count += 1
        if row.get("status") == "timeout":
            timeouts += 1
        elif row.get("status") == "crash":
            crashes += 1
        elapsed = row.get("elapsed_s")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            gpu_s += float(elapsed)
    return count, timeouts, gpu_s, crashes


def _guidance_source(brief: SessionBrief, name: str, fallback: Path) -> Path:
    """@brief 取一份指导文件：本运行有冻结快照就用快照，否则用仓库里的文件。

    @param brief 本场会话的契约。
    @param name 文件名，如 ``device.md``。
    @param fallback 仓库里的原始路径。
    @return 实际要用的文件路径。
    """
    if brief.guidance_dir:
        snap = Path(brief.guidance_dir) / name
        if snap.is_file():
            return snap
    return fallback


def stage_guidance(work: Path, brief: SessionBrief) -> None:
    """@brief 把设备说明、经验文件和语言参考拷进会话工作目录。

    运行里有冻结快照时优先用快照。

    @param work 会话工作目录，不存在时创建。
    @param brief 本场会话的契约。
    """
    work.mkdir(parents=True, exist_ok=True)
    mapping = (
        ("device.md", skill_path("device", brief.target_id)),
        (f"{brief.reference_skill}.md", skill_path(brief.reference_skill, brief.target_id)),
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
    """@brief 取会话自己的反馈或笔记里点名的顶层条目。

    一条已实测的前提会指向探针文件。把这些文件归档掉，会让下一次解析把 ``verified`` 降级成
    ``downgraded``；所以那个文件（或它所在的整个小目录）留在原地。

    @param work 会话工作目录。
    @return 需要保留的顶层条目名集合。
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
    """@brief 把会话的超额草稿归档，并从 ``agent_work`` 里删掉。

    外层要读的文件（NOTES、FEEDBACK、self_checks、TASK.md、kernel.metal、交卷文件）保留。
    其余东西——探针变体、``v/`` 消融源码、随手写的脚本——打成
    ``<run>/scratch/<label>.tar.gz`` 后删除。若压缩包本身超过 ``max_bytes``，就丢掉压缩包，
    改由一份 JSON 清单记录删了什么：这里要的是硬上限，不是无界的第二份拷贝。

    @param work 会话工作目录。
    @param label 会话标签，用作归档文件名。
    @param keep 额外要保留的顶层条目名。
    @param max_bytes 压缩包的字节上限，缺省 ``SCRATCH_MAX_BYTES``。
    @return 归档结果字典：``archived``、``archive``、``kept``、``bytes``、``reason`` 等。
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
    """@brief 跑一场 hmz 会话。

    超时也照样返回内核文件，不抛异常：会话已经计费，成本必须带回给调用方。

    @param brief 本场会话的契约。
    @param work 会话工作目录，不存在时创建。
    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度；为空时不带后缀。
    @param prices 该模型的费率。
    @param turn_output_tokens 这一回合的输出 token 上限。
    @param turn_minutes 这一回合的分钟预算。
    @param hmz hmz 可执行文件路径；``None`` 时从 PATH 上找。
    @param timeout_s 进程墙钟超时秒数；``None`` 时按回合预算推算。
    @return 本场会话的结果。
    @raises AgentError hmz 不在 PATH 上，或进程起不来时。
    """
    binary = hmz or shutil.which("hmz")
    if not binary or not Path(binary).is_file():
        raise AgentError("hmz is not on PATH; install it with `uv tool install hmz`")
    work.mkdir(parents=True, exist_ok=True)
    submit = submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext)
    submit_path = work / submit
    submit_path.unlink(missing_ok=True)
    seed_path = work / brief.seed_name
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
        "-p", f"seed_name={brief.seed_name}",
        "-p", f"budget.duration={flow_minutes}m",
        "--json",
        "--", f"Read TASK.md in this directory and do it. Write the finished kernel to {submit}. The search does not read {brief.seed_name}.",
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
    # The session is already billed at this point. A non-UTF-8 byte in a comment must degrade to a
    # replacement character, not raise away the AgentResult that carries the cost.
    written = submit_path.read_text(encoding="utf-8", errors="replace") if submit_path.is_file() else ""
    source = kernel_text(written, brief.kernel_name, source_fence=brief.source_fence)
    notes = ""
    notes_path = work / "NOTES.md"
    if notes_path.is_file():
        notes = notes_path.read_text(encoding="utf-8", errors="replace")
    from .agent_feedback import read_feedback_text

    feedback, _over = read_feedback_text(work / "FEEDBACK.md")
    protected = _self_check_summary(_self_check_dir(brief))
    beside = _self_check_summary(work)
    checks = protected[0] + beside[0]
    timeouts = protected[1] + beside[1]
    gpu_s = protected[2] + beside[2]
    crashes = protected[3] + beside[3]
    if status != "timeout" and proc.returncode not in (0, None):
        status = "failed"
        error = (stderr or stdout or "").strip()[-400:]
    diagnostics = diagnostics_from_stream(stdout or "")
    response = str(diagnostics.pop("_response_text", "") or "")
    diagnostics["hmz_returncode"] = proc.returncode
    if status == "timeout":
        diagnostics["failure_class"] = "timeout"
    elif status == "failed":
        diagnostics["failure_class"] = "process_failed"
    elif source is not None:
        diagnostics["failure_class"] = ""
    elif diagnostics["last_terminal_kind"] == "failed":
        diagnostics["failure_class"] = "turn_failed"
    elif diagnostics["last_terminal_kind"] == "result":
        diagnostics["failure_class"] = "no_submission_after_result"
    else:
        diagnostics["failure_class"] = "no_submission"
    if not error and source is None and diagnostics["last_error"]:
        error = diagnostics["last_error"]
    diagnostics_path = ""
    if source is None and response:
        # A normal answer with no submission is the one case where a short response preview is
        # needed to diagnose an irrelevant/model-confused turn. Never persist the whole stream.
        run_root = work.parent.parent if work.parent.name == "agent_work" else work.parent
        folder = run_root / "agent_diagnostics"
        name = (_NAME_PART.sub("_", brief.label).strip("._") or "session") + f"-{time.time_ns()}.json"
        path = folder / name
        detail = dict(diagnostics)
        detail["label"] = brief.label
        detail["response_preview"] = _diagnostic_text(response, _DIAG_PREVIEW_CHARS)
        try:
            from .control import write_json_atomic

            folder.mkdir(parents=True, exist_ok=True)
            write_json_atomic(path, detail)
            diagnostics_path = path.relative_to(run_root).as_posix()
        except OSError as exc:
            diagnostics["diagnostics_write_error"] = f"{type(exc).__name__}: {exc}"[:180]
    try:
        pruned = prune_work(
            work, label=brief.label,
            keep={submission_name(brief.task_name, brief.dtype, brief.size, brief.label, brief.file_ext), brief.seed_name},
        )
    except OSError as exc:  # pruning is housekeeping; it must not discard the billed session
        pruned = {"error": f"{type(exc).__name__}: {exc}"}
    return AgentResult(
        source=source, cost_usd=cost, cost_incomplete=not saw_result, status=status, error=error,
        spent=spent, notes=notes, feedback=feedback, self_checks=checks, self_check_timeouts=timeouts,
        self_check_crashes=crashes, self_check_gpu_s=gpu_s, pruned=pruned,
        diagnostics=diagnostics, diagnostics_path=diagnostics_path,
    )


def agent_for_script(script, kernel_name: str):
    """@brief 给「假 chat 客户端就是唯一模型」的那些测试用的替身。

    一次调用是一场会话。第一次无种子的调用是草稿；之后的调用依次要 plan A、B、C，
    老的分脚本测试就是靠这个区分答案的。真实搜索不走这条路。

    @param script 假的 chat 脚本，接收消息列表并返回响应。
    @param kernel_name 期望的内核函数名。
    @return 一个 ``run(brief, tag)`` 可调用对象，返回 ``AgentResult``。
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
        source = kernel_text(text, kernel_name, source_fence=brief.source_fence)
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
    """@brief 在任何会话开始之前，先查 hmz、流程目录、凭据和输出价格。

    @param committees 本次搜索的委员会列表，每个只看第一成员。
    @param credentials 凭据与价格文件的路径。
    @param hmz hmz 可执行文件路径；``None`` 时从 PATH 上找。
    @raises AgentError hmz 不在 PATH 上、kernel 流程目录缺失、assistant 规格 hmz 不接受，
        或某个成员的凭据与输出价格读不出来时。
    """
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
