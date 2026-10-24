"""The dashboard's write paths: adding budget to one run, and resetting every run.

What a click can do (``POST /api/runs/<run>/budget``):

* a run that is still running and polls its control file (``state.json`` says ``control_supported``): write ``control.json`` (live top-up);
* a run that stopped for budget, or whose process died: start ``python -m taic.cli resume`` for it as a detached process;
* anything else: refused with the reason (a run that converged, hit its round limit or a GPU hang is not continued; there is no override).

``POST /api/budget/reset`` is the one-click reset: every search that can continue gets $15 added and its round limit and patience lifted.
A run that hung the GPU, failed in the model, or never produced a kernel is skipped. The experiment ``cap.json``, when there is one, is raised by the same amount so the add is not refused.

``POST /api/stop`` sends SIGTERM to every live search and experiment queue in this repo. It signals that process group, and also every process group of a descendant and of a process whose command names that run's directory. ``hmz`` is started in its own session, so a group signal to the search alone leaves ``pi`` running. The dashboard process is never a target.

This spends real money and can start GPU work, so every POST needs: a loopback ``Host``, a same-origin ``Origin`` (when the browser sends one), the
per-process random token that the served page carries (``X-Observe-Token``), a positive amount under ``OBSERVE_MAX_TOPUP_USD`` (default 25), a run
that resolves inside ``runs/`` (no ``..``, no symlink out) and at most one request in flight per run. ``OBSERVE_CONTROL_DRYRUN=1`` does all of the
checks, records the command that *would* run in ``budget_events.json`` and starts nothing. The reset amount is fixed at $15 and is not taken from the request body.
"""

from __future__ import annotations

import hmac
import json
import math
import os
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from observe import detail, query
from taic import control, resume

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKEN = secrets.token_urlsafe(32)  # one per server process; the page that this process serves carries it
TOKEN_HEADER = "x-observe-token"
LOOPBACK = {"127.0.0.1", "localhost", "[::1]", "::1"}
DEFAULT_MAX_TOPUP_USD = 25.0
_locks: dict[str, threading.Lock] = {}
_guard = threading.Lock()


class Refused(Exception):
    """@brief 表示一个未执行的请求；``status`` 是 HTTP 状态码，``message`` 直接展示给用户。

    原始英文说明：A request that is not carried out; ``status`` is the HTTP status and ``message`` is shown to the user.
    """

    def __init__(self, status: int, message: str) -> None:
        """@brief 记录拒绝原因与对应的 HTTP 状态。

        @param status HTTP 状态码。
        @param message 给人看的拒绝原因。
        """
        super().__init__(message)
        self.status, self.message = status, message


def max_topup_usd() -> float:
    """@brief 取单次加预算上限，来自 ``OBSERVE_MAX_TOPUP_USD``。

    @return 上限美元数；环境变量非法或非正时返回默认 25。
    """
    try:
        v = float(os.environ.get("OBSERVE_MAX_TOPUP_USD", DEFAULT_MAX_TOPUP_USD))
    except ValueError:
        return DEFAULT_MAX_TOPUP_USD
    return v if math.isfinite(v) and v > 0 else DEFAULT_MAX_TOPUP_USD


def dry_run() -> bool:
    """@brief 报告是否处于干跑模式（``OBSERVE_CONTROL_DRYRUN``）。

    @return 干跑时为 True，此时只记录将要执行的动作。
    """
    return os.environ.get("OBSERVE_CONTROL_DRYRUN", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------------------------
# who may ask
# ---------------------------------------------------------------------------------------------


def _host_name(netloc: str) -> str:
    """@brief 从 ``Host`` 的 netloc 中取主机名，正确处理 IPv6 方括号。

    @param netloc ``Host`` 头里的主机与端口部分。
    @return 小写主机名。
    """
    netloc = (netloc or "").strip().lower()
    if netloc.startswith("["):
        return netloc.split("]")[0] + "]"
    return netloc.rsplit(":", 1)[0] if netloc.count(":") == 1 else netloc


def host_is_loopback(headers) -> bool:
    """@brief 判断请求的 ``Host`` 是否为 loopback（127.0.0.1 / localhost / ::1）。

    @param headers 请求头。
    @return 是 loopback 时为 True。
    """
    return _host_name(headers.get("host", "")) in LOOPBACK


def check_origin(headers) -> None:
    """@brief 防 DNS rebinding 与跨站请求：Host 必须是 loopback，有 Origin 时必须与页面同源。

    原始英文说明：DNS rebinding and cross-site requests: the Host must be loopback and an Origin, when there is one, must be the page's own origin.

    @param headers 请求头。
    @raises Refused Host 非 loopback 或跨源时抛出（HTTP 403）。
    """
    if not host_is_loopback(headers):
        raise Refused(403, "refused: this server only answers on 127.0.0.1 / localhost (Host header check)")
    origin = headers.get("origin")
    if origin is not None:
        o = urlsplit(origin)
        if o.scheme != "http" or (o.netloc or "").lower() != (headers.get("host") or "").lower():
            raise Refused(403, "refused: cross-origin request")


def check_token(headers) -> None:
    """@brief 校验 ``X-Observe-Token`` 是否为本进程下发的随机 token。

    @param headers 请求头。
    @raises Refused token 缺失或不匹配时抛出（HTTP 403）。
    """
    given = headers.get(TOKEN_HEADER, "")
    if not given or not hmac.compare_digest(given.encode(), TOKEN.encode()):
        raise Refused(403, "refused: missing or wrong X-Observe-Token (reload the page)")


def page_token(headers) -> str:
    """@brief 取要埋进页面的 token；只有 loopback Host 才拿得到（被重绑的域名拿不到）。

    原始英文说明：The token to embed in the served page: only for a loopback Host (a rebound hostname never gets it).

    @param headers 请求头。
    @return token 字符串；Host 非 loopback 时为空串。
    """
    return TOKEN if host_is_loopback(headers) else ""


# ---------------------------------------------------------------------------------------------
# what is possible for a run
# ---------------------------------------------------------------------------------------------


def _run_dir(run_id: str) -> Path:
    """@brief 把 run id 解析为运行目录，不存在则拒绝。

    @param run_id 运行 id。
    @return 运行目录。
    @raises Refused 索引里没有该运行时抛出（HTTP 404）。
    """
    path = query.resolve_run(run_id)
    if path is None:
        raise Refused(404, "no such run")
    return path


def _warnings(run_dir: Path, info: dict) -> list[str]:
    """@brief 列出继续运行前需要提醒用户的信息（旧版本没有记录 patience 等）。

    @param run_dir 运行目录。
    @param info ``resume.inspect_run`` 返回的检查结果。
    @return 警告文本列表，没有时为空列表。
    """
    out: list[str] = []
    if info.get("resumable") and not (run_dir / control.CONFIG_FILE).exists():
        out.append("this run was started by an older version: its settings are inferred from its prompts; 'patience' is not recorded, the continued run uses taic.toml's value")
    return out


def describe(run_id: str) -> dict:
    """@brief ``GET /api/runs/<run>/control``：点“加预算”会做什么，以及做不到时的原因。

    原始英文说明：what a click on 'add budget' would do, and why not when it cannot.

    @param run_id 运行 id。
    @return 描述可用动作、预算、历史、建议金额与警告的字典。
    """
    run_dir = _run_dir(run_id)
    info = resume.inspect_run(run_dir)
    live = bool(info["can_topup_live"])
    can_resume = bool(info["resumable"])
    reason = "" if (live or can_resume) else (info["live_reason"] or info["reason"] or "this run cannot be continued")
    hist = info["history"]
    return {
        "run": run_id,
        "kind": info["kind"],
        "stop_reason": info["stop_reason"],
        "can_topup_live": live,
        "can_resume": can_resume,
        "can_add_budget": live or can_resume,
        "action": "topup_live" if live else "resume" if can_resume else None,
        "reason_if_not": reason,
        "live_reason": info["live_reason"],
        "budget": {"current_usd": info["budget_usd"], "initial_usd": hist["initial_budget_usd"], "spent_usd": info["spent_usd"], "needs_budget": info["needs_budget"]},
        "best_speedup": info["best_speedup"],
        "history": {"topped_up": hist["topped_up"], "events": hist["events"], "applied": hist["applied"], "pending": hist["pending"]},
        "suggested_amounts": info["suggested_amounts"],
        "max_topup_usd": max_topup_usd(),
        "dry_run": dry_run(),
        "warnings": _warnings(run_dir, info),
        "pid": info["pid"],
    }


# ---------------------------------------------------------------------------------------------
# asking for budget
# ---------------------------------------------------------------------------------------------


def _number(value, name: str) -> float:
    """@brief 把请求体里的值校验为正的有限数。

    @param value 待校验的值。
    @param name 字段名，用于错误信息。
    @return 浮点数值。
    @raises Refused 不是数字或不是正数时抛出（HTTP 400）。
    """
    if value is None or isinstance(value, bool):
        raise Refused(400, f"{name} must be a number")
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise Refused(400, f"{name} must be a number") from None
    if not math.isfinite(v) or v <= 0:
        raise Refused(400, f"{name} must be a positive number")
    return v


def _amount_and_total(body: dict, current: float | None, spent: float) -> tuple[float, float]:
    """@brief 从 ``amount_usd`` 或 ``new_budget_usd`` 算出（追加金额，新总额）。

    增加额受单次上限约束，新总额必须高于已花费。
    原始英文说明：(amount to add, new total budget) from ``amount_usd`` or ``new_budget_usd``; the increase is capped, the total must exceed what is spent.

    @param body 请求体。
    @param current 当前预算，未知时为 None。
    @param spent 已花费美元数。
    @return 二元组：追加金额与新的预算总额。
    @raises Refused 字段缺失/冲突、超过上限或新总额不够时抛出（HTTP 400 或 409）。
    """
    if not isinstance(body, dict):
        raise Refused(400, "the body must be a JSON object")
    has_amount, has_total = body.get("amount_usd") is not None, body.get("new_budget_usd") is not None
    if has_amount == has_total:
        raise Refused(400, "give exactly one of amount_usd (USD to add) or new_budget_usd (the new total)")
    if has_amount:
        amount = _number(body["amount_usd"], "amount_usd")
        if current is None:
            raise Refused(409, "the run does not record its budget: send new_budget_usd")
        total = current + amount
    else:
        total = _number(body["new_budget_usd"], "new_budget_usd")
        amount = total - (current if current is not None else spent)
    cap = max_topup_usd()
    if amount > cap + 1e-9:
        raise Refused(400, f"refused: adding ${amount:.2f} exceeds the cap of ${cap:.2f} per request (OBSERVE_MAX_TOPUP_USD)")
    if current is not None and total <= current + 1e-9:
        raise Refused(400, f"the new budget ${total:.2f} must exceed the current ${current:.2f}")
    if total <= spent + 1e-9:
        raise Refused(400, f"the new budget ${total:.2f} must exceed what is already spent (${spent:.4f})")
    return amount, total


def promote_harness(run_id: str, proposal_key: str, *, root: Path | None = None) -> dict:
    """@brief 把一条已评审的提案写进该硬件的经验文件；与 ``taic harness apply --confirm`` 同一步。

    原始英文说明：Write one reviewed proposal into that hardware's experience file. Same step as ``taic harness apply --confirm``.

    @param run_id 运行 id。
    @param proposal_key 提案键。
    @param root 经验文件根目录，None 表示用默认位置。
    @return ``apply_patch`` 的结果字典。
    @raises Refused 提案为空/未评审为有用/已固化/无目标硬件时抛出（400 或 409）。
    """
    from taic.harness import ProposalError, apply_patch, board

    key = str(proposal_key or "").strip()
    if not key:
        raise Refused(400, "缺少提案")
    run_dir = _run_dir(run_id)
    doc = board(run_dir)
    row = next((item for item in doc["proposals"] if item.get("proposal_key") == key and item.get("status") == "valid"), None)
    if row is None:
        raise Refused(404, "没有这条有效提案")
    if not any(item.get("proposal_key") == key and item.get("useful") is True for item in doc["reviews"]):
        raise Refused(400, "这条还没有被独立 review 为有用，不能固化")
    if any(item.get("event") == "promotion" and item.get("proposal_key") == key for item in doc["decisions"]):
        raise Refused(409, "这条已经固化到本地")
    scope = (row.get("proposal") or {}).get("scope") if isinstance(row.get("proposal"), dict) else {}
    target_id = str((scope or {}).get("target_id") or "").strip()
    if not target_id:
        raise Refused(400, "提案没有目标硬件")
    with _lock_for(run_id):
        try:
            return apply_patch(run_dir, target_id=target_id, proposal_key=key, root=root, applied_by="dashboard")
        except ProposalError as exc:
            raise Refused(400, str(exc)) from exc


def convert_harness(run_id: str, seq: int, digest: str, body: dict) -> dict:
    """@brief 把人确认过的一条自由文本建议转换成结构化提案。

    人的点击就是门禁。转换本身在本地确定性完成，不调 LLM；``mode=ai`` 只记录由 AI 填了字段，
    写入仍需人工确认。
    原始英文说明：Human-confirmed conversion of one free-text suggestion into a structured proposal.

    @param run_id 运行 id。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @param body 请求体，需含 ``confirm=True`` 及可选的转换字段。
    @return ``convert_feedback`` 的结果字典。
    @raises Refused 未确认、找不到该建议、任务缺失或 mode 非法时抛出（400/404）。
    """
    from taic.harness import ProposalError, convert_feedback, free_text_suggestions

    run_dir = _run_dir(run_id)
    if body.get("confirm") is not True:
        raise Refused(400, "conversion needs explicit human confirmation")
    suggestions = free_text_suggestions(run_dir)
    source = next((row for row in suggestions if int(row.get("seq") or 0) == int(seq) and str(row.get("digest") or "") == str(digest)), None)
    if source is None:
        raise Refused(404, "no unconverted free-text suggestion with that seq/digest")
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").is_file() else {}
    scope = run_cfg.get("target") if isinstance(run_cfg.get("target"), dict) else {}
    target_id = str(body.get("target_id") or scope.get("id") or "apple-m4-max")
    task = str(body.get("task") or run_cfg.get("task") or "")
    dtype = str(body.get("dtype") or run_cfg.get("dtype") or "")
    if not task:
        raise Refused(400, "run has no task; cannot convert")
    mode = str(body.get("mode") or "manual")
    if mode not in ("manual", "ai"):
        raise Refused(400, "mode must be manual or ai")
    try:
        return convert_feedback(
            run_dir, seq=seq, digest=digest, target_id=target_id, task=task, dtype=dtype,
            surface=str(body.get("surface") or ""), action=str(body.get("action") or ""),
            probe=str(body.get("probe") or ""), measurement=str(body.get("measurement") or ""),
            reproduction=str(body.get("reproduction") or ""), expected=str(body.get("expected") or ""),
            rationale=str(body.get("rationale") or ""), by=str(body.get("by") or "dashboard"),
            confirm=True, mode=mode, supersede=str(body.get("supersede") or ""),
        )
    except ProposalError as exc:
        raise Refused(400, str(exc)) from exc


def review_harness(run_id: str, proposal_key: str, body: dict) -> dict:
    """@brief 对一条结构化（通常是已转换的）提案执行一次显式独立评审。

    只有人点了按钮才会花一次模型调用，从不隐式评审自由文本；既有的评审预算与信任语义不变。
    原始英文说明：Run one explicit independent review for a structured (usually converted) proposal.

    @param run_id 运行 id。
    @param proposal_key 提案键。
    @param body 请求体，可含 ``provider`` / ``model`` 指定评审者。
    @return 含 ``review``、``plan`` 与 ``settle`` 的字典。
    @raises Refused 找不到提案、已评过、评审者不可用或冲突时抛出；评审失败时抛 502。
    """
    from taic import trust as trust_mod
    from taic.committees import load_committees
    from taic.kernel_agent import Prices
    from taic.llm import load_config, load_raw_config
    from taic import review as review_mod
    from taic import harness

    run_dir = _run_dir(run_id)
    proposals = [row for row in harness.board(run_dir)["proposals"] if row.get("status") == "valid" and row.get("proposal_key") == proposal_key]
    if not proposals:
        raise Refused(404, "no valid proposal with that key")
    proposal = proposals[0]
    if any(row.get("event") == "review" and row.get("proposal_key") == proposal_key for row in harness.board(run_dir)["reviews"]):
        raise Refused(409, "this proposal already has a review")
    run_cfg = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8")) if (run_dir / "run_config.json").is_file() else {}
    recorded = run_cfg.get("committees") if isinstance(run_cfg.get("committees"), list) else []
    provider, model = str(body.get("provider") or ""), str(body.get("model") or "")
    if not provider or not model:
        proposal_committee = harness.proposal_committee(proposal)
        peers = [c for c in recorded if isinstance(c, dict) and c.get("name") != proposal_committee]
        members = (peers[0].get("members") if peers else None) or []
        if not members or not isinstance(members[0], dict):
            raw = load_raw_config(None, "base")
            cfg, _ = load_config(None, preset="base")
            configured = load_committees(raw, active=cfg)
            peers = [c for c in configured if c.name != proposal_committee] or configured
            members = [{"provider": m.provider, "model": m.model} for c in peers for m in c.members]
        if members and isinstance(members[0], dict):
            independent = [
                member for member in members
                if isinstance(member, dict)
                and not harness.reviewer_conflicts_with_group(proposals, str(member.get("provider") or ""), str(member.get("model") or ""))
            ]
            members = independent or members
        if not members or not isinstance(members[0], dict):
            raise Refused(400, "cannot choose a reviewer; pass provider and model")
        provider, model = str(members[0].get("provider") or ""), str(members[0].get("model") or "")
    raw = load_raw_config(None, "base")
    cfg, _ = load_config(None, preset="base")
    committees = load_committees(raw, active=cfg)
    member = next((m for c in committees for m in c.members if m.provider == provider and m.model == model), None)
    if member is None:
        raise Refused(400, f"reviewer {provider}/{model} is not in the configured committees")
    if harness.reviewer_conflicts_with_group(proposals, provider, model):
        raise Refused(400, f"reviewer {provider}/{model} is an author of this claim; choose an independent model")
    prices = Prices(member.cfg.price_in, member.cfg.price_out, member.cfg.price_cache_read, member.cfg.price_cache_write)
    work = run_dir / "harness_reviews" / proposal_key
    bundle = review_mod.evidence_bundle(run_dir, proposal)
    prompt = review_mod.build_review_prompt(proposal.get("proposal") or {}, run_id=str(proposal.get("run_id") or run_dir.name), bundle=bundle)
    plan = review_mod.estimate_review_budget(prices, prompt)
    try:
        verdict = review_mod.run_review(
            work, prompt=prompt, provider=provider, model=model,
            effort=member.cfg.reasoning_effort, prices=prices,
        )
    except review_mod.ReviewError as exc:
        raise Refused(502, str(exc)) from exc
    review_mod.append_review(run_dir, proposal, verdict)
    settled = harness.settle(
        run_dir, target_id=str(proposal.get("target_id") or ""), task=str(proposal.get("task") or ""),
        protocol_version=str(run_cfg.get("protocol_version") or trust_mod.current_protocol_version()),
        enabled=True, shadow=True,
    )
    return {"review": verdict, "plan": plan, "settle": settled}


def review_feedback(run_id: str, seq: int, digest: str, body: dict) -> dict:
    """@brief 记录一条 claim 评审，返回更新后的 claim 前先刷新索引。

    原始英文说明：Record one claim review and refresh the index before returning the updated claim.

    @param run_id 运行 id。
    @param seq 反馈条目序号。
    @param digest 条目摘要。
    @param body 评审内容。
    @return 落盘的评审结果。
    @raises Refused 请求体或摘要非法时抛 400；claim 不存在时抛 404。
    """
    from observe.feedback import apply_review
    from observe.index import get_index

    if not isinstance(body, dict):
        raise Refused(400, "the body must be an object")
    if not isinstance(digest, str) or not digest.strip():
        raise Refused(400, "the claim digest is empty")
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        try:
            stored = apply_review(run_dir, int(seq), digest, body)
        except ValueError as exc:
            raise Refused(400, str(exc)) from exc
        index = get_index()
        index.refresh_feedback(run_id, run_dir)
        found = index.conn.execute(
            "SELECT 1 FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?",
            (run_id, int(seq), digest),
        ).fetchone()
    if found is None:
        raise Refused(404, "no such feedback claim")
    return stored


def review_feedback_batch(run_id: str, rows: list[dict], *, dry_run: bool = False, method: str | None = None, batch_id: str = "") -> dict:
    """@brief 预览或原子地写入一批 claim 评审。

    改文件前先逐项校验，并检查每项的 ``expected_status``；干跑只读只校验。
    原始英文说明：Preview or atomically write a batch of claim reviews.

    @param run_id 运行 id。
    @param rows 评审项列表。
    @param dry_run 为 True 时只校验不写。
    @param method 评审来源，取 ``human`` / ``agent`` / ``script`` / ``api``。
    @param batch_id 批次 id，空串表示未指定。
    @return 含 ``dry_run``、``items`` 与 ``batch_id`` 的字典。
    @raises Refused 批次为空、method 非法时抛 400；claim 不存在时抛 404。
    """
    from observe.feedback import apply_reviews, preview_reviews
    from observe.index import get_index

    if not isinstance(rows, list) or not rows:
        raise Refused(400, "the batch must be a non-empty JSON array")
    if method and method not in ("human", "agent", "script", "api"):
        raise Refused(400, "method must be human, agent, script or api")
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        planned = preview_reviews(run_dir, rows)
        index = get_index()
        for item in planned:
            found = index.conn.execute(
                "SELECT 1 FROM feedback_claims WHERE run_id = ? AND seq = ? AND digest = ?",
                (run_id, int(item["seq"]), str(item["digest"])),
            ).fetchone()
            if found is None:
                raise Refused(404, f"no such feedback claim: {run_id}#{item['seq']}#{item['digest']}")
        if dry_run:
            return {"dry_run": True, "items": planned}
        try:
            stored = apply_reviews(run_dir, rows, method=method, batch_id=batch_id)
        except ValueError as exc:
            raise Refused(400, str(exc)) from exc
        index.refresh_feedback(run_id, run_dir)
    return {"dry_run": False, "items": stored, "batch_id": str(batch_id or (stored[0].get("batch_id") if stored else ""))}


def _lock_for(run_id: str) -> threading.Lock:
    """@brief 取（必要时新建）某个运行的进程内互斥锁。

    @param run_id 运行 id。
    @return 该运行对应的锁对象。
    """
    with _guard:
        return _locks.setdefault(run_id, threading.Lock())


def _note_manifest_budget(run_dir: Path, total: float) -> None:
    """@brief 把抬高的预算记为实验清单上的一次延期；冻结的实验条件不变。

    原始英文说明：A raised budget is an extension on the experiment manifest. Frozen conditions stay put.

    @param run_dir 运行目录。
    @param total 新的预算总额。
    """
    from experiments.manifest import ManifestError, extend_budget

    current = run_dir.resolve()
    for _ in range(8):
        if current.parent == current:
            return
        current = current.parent
        if not (current / "experiment.json").is_file():
            continue
        try:
            extend_budget(current, run=run_dir.resolve().relative_to(current).as_posix(), new_cap=total)
        except (ManifestError, ValueError, OSError):
            return
        return


def _resume_command(run_dir: Path, total: float) -> list[str]:
    """@brief 拼出续跑命令 ``taic.cli resume``（优先用 ``uv run python``）。

    @param run_dir 运行目录。
    @param total 续跑使用的新预算总额。
    @return 命令参数列表。
    """
    uv = shutil.which("uv")
    base = [uv, "run", "python"] if uv else [sys.executable]
    return [*base, "-m", "taic.cli", "resume", str(run_dir), "--budget", f"{total:.6f}", "--who", "dashboard"]


def _next_log(run_dir: Path) -> Path:
    """@brief 取下一个可用的 ``resume_<n>.log`` 路径。

    @param run_dir 运行目录。
    @return 尚不存在的日志路径。
    """
    n = 1
    while (run_dir / f"resume_{n}.log").exists():
        n += 1
    return run_dir / f"resume_{n}.log"


def _enforce_experiment_cap(run_dir: Path, total: float) -> None:
    """@brief 拒绝会超出实验 ``cap.json`` 的加预算；没有 cap 文件就没有额外限制。

    原始英文说明：Refuse a top-up that would reserve more than the experiment's cap.json allows. No cap file means no extra limit.

    @param run_dir 运行目录。
    @param total 新的预算总额。
    @raises Refused 超出实验上限时抛出（HTTP 400）。
    """
    from experiments.common import cap_allows

    ok, why = cap_allows(run_dir, total)
    if not ok:
        raise Refused(400, why)


def add_budget(run_id: str, body: dict) -> dict:
    """@brief ``POST /api/runs/<run>/budget``（已完成 origin 与 token 校验），返回实际做了或干跑将要做的事。

    原始英文说明：Returns what was done (or, in dry-run mode, what would be).

    @param run_id 运行 id。
    @param body 请求体，需含 ``amount_usd`` 或 ``new_budget_usd``。
    @return 含 ``ok``、``dry_run``、``action``、旧/新预算与追加金额的结果字典。
    @raises Refused 运行状态不允许、预算金额非法或超出实验 cap 时抛出（400/409）。
    """
    run_dir = _run_dir(run_id)
    with _lock_for(run_id):
        info = resume.inspect_run(run_dir)
        spent = float(info["spent_usd"] or 0.0)
        current = float(info["budget_usd"]) if info["budget_usd"] is not None else None
        if info["kind"] == "resuming":
            raise Refused(409, info["reason"])
        if info["kind"] == "running":
            if not info["can_topup_live"]:
                raise Refused(409, info["live_reason"] or info["reason"])
            pending = info["history"]["pending"]
            if pending:
                raise Refused(409, "a previous top-up of this run has not been read by the run yet; wait a moment")
            amount, total = _amount_and_total(body, current, spent)
            _enforce_experiment_cap(run_dir, total)
            if dry_run():
                ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would write control.json", action="topup_live")
                return {"ok": True, "dry_run": True, "action": "topup_live", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "event": ev}
            doc = control.write_control(run_dir, total, who="dashboard", amount_usd=amount, old_budget_usd=current)
            _note_manifest_budget(run_dir, total)
            return {"ok": True, "dry_run": False, "action": "topup_live", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "control": doc}
        if not info["resumable"]:
            raise Refused(409, info["reason"] or "this run cannot be continued")
        amount, total = _amount_and_total(body, current, spent)
        _enforce_experiment_cap(run_dir, total)
        cmd = _resume_command(run_dir, total)
        if dry_run():
            ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would start: " + " ".join(cmd), action="resume", command=cmd)
            return {"ok": True, "dry_run": True, "action": "resume", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "command": cmd, "event": ev}
        control.record_event(run_dir, "requested", current, total, who="dashboard", note="resume requested", action="resume")
        _note_manifest_budget(run_dir, total)
        log = _next_log(run_dir)
        with open(log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        # claim the run for the child at once, so that a second click before it has started is refused (the child, or the python that ``uv run`` starts
        # for it, recognises this lock as its own: ``starting`` and its parent's pid)
        try:
            control.write_json_atomic(control.lock_path(run_dir), {"pid": proc.pid, "started_at": time.time(), "who": "dashboard", "starting": True})
        except Exception:  # noqa: BLE001 - the child takes the lock itself
            pass
        return {"ok": True, "dry_run": False, "action": "resume", "old_budget_usd": current, "new_budget_usd": total, "amount_usd": amount, "pid": proc.pid, "log": log.name, "command": cmd}


# 一键重置：每个能继续的搜索在现有预算上加这么多，并把轮数上限和耐心次数抬到下面两个数。次数上限实际上不再拦住搜索，钱花完仍会停。
RESET_ADD_USD = 15.0
RESET_MAX_ROUNDS = 100_000
RESET_PATIENCE = 100_000


def _reset_plan_for(run_id: str) -> dict:
    """@brief 计算对单个搜索做重置会怎样；不写任何文件。

    原始英文说明：What a reset would do for one search. Does not write.

    @param run_id 运行 id。
    @return 含 ``id``、``kind``、``action``、``reason`` 与新旧预算的计划项。
    """
    run_dir = _run_dir(run_id)
    info = resume.inspect_run(run_dir, allow_count_stops=True)
    current = float(info["budget_usd"]) if info["budget_usd"] is not None else None
    spent = float(info["spent_usd"] or 0.0)
    item = {"id": run_id, "kind": info["kind"], "stop_reason": info["stop_reason"], "old_budget_usd": current, "action": "skip", "reason": "", "new_budget_usd": None}
    if current is None:
        item["reason"] = "没有记下预算"
        return item
    # 已经超支时，在现有预算上加 15 仍可能低于已花费。取预算和已花费里较高的那个再加 15，这样一定多出 15 美元可用。
    item["new_budget_usd"] = max(current, spent) + RESET_ADD_USD
    if info["kind"] == "resuming":
        item["reason"] = info["reason"] or "正在被另一个进程续跑"
        return item
    if info["kind"] == "running":
        if not info["can_topup_live"]:
            item["reason"] = info["live_reason"] or "运行中的进程不读控制文件"
            return item
        if info["history"]["pending"]:
            item["reason"] = "上一笔加预算还没被这次搜索读到"
            return item
        item["action"] = "topup_live"
        return item
    if info["resumable"]:
        item["action"] = "resume"
        return item
    item["new_budget_usd"] = None
    item["reason"] = info["reason"] or "不能重置"
    return item


def reset_preview() -> dict:
    """@brief ``GET /api/budget/reset``：哪些搜索会被重置、哪些会被跳过；不写任何东西。

    原始英文说明：which searches would be reset, and which would be skipped.

    @return 含追加金额、轮数/耐心上限、``dry_run`` 以及 ``runs`` / ``skipped`` 的字典。
    """
    rows = [_reset_plan_for(row["id"]) for row in query.history()]
    return {
        "add_usd": RESET_ADD_USD,
        "max_rounds": RESET_MAX_ROUNDS,
        "patience": RESET_PATIENCE,
        "dry_run": dry_run(),
        "runs": [row for row in rows if row["action"] != "skip"],
        "skipped": [row for row in rows if row["action"] == "skip"],
    }


def _raise_experiment_caps(rows: list[dict]) -> list[dict]:
    """@brief 按本次重置将预留的额外美元抬高各实验的 cap，使加预算不被拒。

    已结束的运行按实际花费计入占据，而重置要按新预算计入，差额可能超过 15 美元。
    原始英文说明：Raise each experiment cap by the extra dollars this reset will reserve, so the add is not refused.

    @param rows ``reset_preview`` 里将要执行的行。
    @return 每项含实验目录、旧/新 cap、运行数与增加额的列表。
    """
    from experiments.common import find_cap, run_commitment, write_cap

    grouped: dict[Path, dict] = {}
    for row in rows:
        run_dir = _run_dir(row["id"])
        found = find_cap(run_dir)
        if found is None:
            continue
        exp_dir, cap = found
        this = run_commitment(run_dir)
        extra = max(0.0, max(this, float(row["new_budget_usd"])) - this)
        slot = grouped.setdefault(exp_dir, {"old": cap, "add": 0.0, "n": 0})
        slot["add"] += extra
        slot["n"] += 1
    out = []
    for exp_dir, slot in grouped.items():
        new = slot["old"] + slot["add"]
        if not dry_run() and slot["add"] > 0:
            write_cap(exp_dir, new)
        out.append({"experiment": str(exp_dir), "old_cap_usd": slot["old"], "new_cap_usd": new, "n_runs": slot["n"], "added_usd": slot["add"]})
    return out


def _reset_resume_command(run_dir: Path, total: float) -> list[str]:
    """@brief 在续跑命令上加上抬高轮数上限与耐心的参数。

    @param run_dir 运行目录。
    @param total 新预算总额。
    @return 重置用的命令参数列表。
    """
    cmd = _resume_command(run_dir, total)
    return [*cmd[:-2], "--max-rounds", str(RESET_MAX_ROUNDS), "--patience", str(RESET_PATIENCE), "--lift-count-limits", *cmd[-2:]]


def _apply_reset(item: dict) -> dict:
    """@brief 执行一个重置计划项，把拒绝转成带原因的跳过项。

    @param item ``reset_preview`` 给出的计划项。
    @return 执行结果字典。
    """
    try:
        return _apply_reset_unchecked(item)
    except Refused as e:
        return {**item, "action": "skip", "reason": e.message, "new_budget_usd": None}


def _apply_reset_unchecked(item: dict) -> dict:
    """@brief 真正执行一个重置计划项（可写控制文件或启动续跑进程）。

    @param item 计划项，含 ``id``、``action`` 与新旧预算。
    @return 含实际动作、事件与（必要时）pid、日志名和命令的字典。
    @raises Refused 运行不存在或超出实验 cap 时抛出。
    """
    run_dir = _run_dir(item["id"])
    current, total = item["old_budget_usd"], item["new_budget_usd"]
    note = "reset: add $15 and lift the round limit and patience"
    with _lock_for(item["id"]):
        if item["action"] == "topup_live":
            if not dry_run():
                _enforce_experiment_cap(run_dir, total)
            if dry_run():
                ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would write control.json; " + note, action="reset")
                return {**item, "dry_run": True, "event": ev}
            doc = control.write_control(
                run_dir, total, who="dashboard", amount_usd=total - current, old_budget_usd=current, note=note,
                max_rounds=RESET_MAX_ROUNDS, patience=RESET_PATIENCE,
            )
            _note_manifest_budget(run_dir, total)
            return {**item, "dry_run": False, "control": doc}
        cmd = _reset_resume_command(run_dir, total)
        if not dry_run():
            _enforce_experiment_cap(run_dir, total)
        if dry_run():
            ev = control.record_event(run_dir, "dryrun", current, total, who="dashboard", note="would start: " + " ".join(cmd), action="reset", command=cmd)
            return {**item, "dry_run": True, "command": cmd, "event": ev}
        control.record_event(run_dir, "requested", current, total, who="dashboard", note=note, action="reset")
        _note_manifest_budget(run_dir, total)
        log = _next_log(run_dir)
        with open(log, "ab") as out:
            proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            control.write_json_atomic(control.lock_path(run_dir), {"pid": proc.pid, "started_at": time.time(), "who": "dashboard", "starting": True})
        except Exception:  # noqa: BLE001 - the child takes the lock itself
            pass
        return {**item, "dry_run": False, "pid": proc.pid, "log": log.name, "command": cmd}


def reset_budgets() -> dict:
    """@brief ``POST /api/budget/reset``：给每个能继续的搜索加 15 美元，并抬高轮数上限与耐心。

    原始英文说明：Adds $15 to every search that can continue and lifts its round limit and patience.

    @return 含 ``ok``、金额、上限、``caps``、``runs`` 与 ``skipped`` 的结果字典。
    """
    plan = reset_preview()
    caps = _raise_experiment_caps(plan["runs"])
    results = [_apply_reset(row) if row["action"] != "skip" else row for row in plan["runs"] + plan["skipped"]]
    # preview split them; apply keeps skips unchanged and replaces the ones it carried out
    done = [row for row in results if row["action"] != "skip"]
    skipped = [row for row in results if row["action"] == "skip"]
    return {"ok": True, "add_usd": RESET_ADD_USD, "max_rounds": RESET_MAX_ROUNDS, "patience": RESET_PATIENCE, "dry_run": dry_run(), "caps": caps, "runs": done, "skipped": skipped}


def _parse_ps(text: str) -> list[tuple[int, str]]:
    """@brief 从 ``ps -ax -o pid=,command=`` 输出中挑出本仓库的搜索或实验进程。

    原始英文说明：``(pid, command)`` lines from ``ps -ax -o pid=,command=`` that are a search or an experiment script of this repo.

    @param text ``ps`` 的原始输出。
    @return (pid, command) 元组列表。
    """
    found = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_s, _, cmd = line.partition(" ")
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        if _is_search_command(cmd):
            found.append((pid, cmd.strip()))
    return found


def _is_search_command(cmd: str) -> bool:
    """@brief 判断这条命令是否属于本按钮可以停止的进程；看板自己不算。

    原始英文说明：A process this button may stop. The dashboard itself is not one of them.

    @param cmd 进程命令行。
    @return 属于搜索或实验时为 True。
    """
    if "-m observe" in cmd or cmd.rstrip().endswith("observe"):
        return False
    return "-m taic.cli" in cmd or "-m experiments." in cmd


def _ps_commands() -> list[tuple[int, str]]:
    """@brief 读取并解析当前可停止的进程命令；``ps`` 不可用时返回空列表。

    @return (pid, command) 元组列表。
    """
    try:
        text = subprocess.check_output(["ps", "-ax", "-o", "pid=,command="], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    return _parse_ps(text)


def _stop_groups() -> tuple[list[dict], list[dict]]:
    """@brief 算出要发信号的进程组，以及看到但被拒绝的进程。

    多个搜索共用一个组时只发一次信号。
    原始英文说明：Process groups to signal, and commands that were seen but refused. One group is signaled once even if several searches share it.

    @return 二元组：待发信号的组列表与被拒列表。
    """
    labels: dict[int, list[str]] = {}
    commands: dict[int, str] = {}
    for row in query.history():
        if row.get("status") != "running":
            continue
        directory = query.resolve_run(row["id"])
        state = detail._load(directory / "state.json") if directory is not None else None
        pid = (state or {}).get("pid")
        if not query.pid_alive(pid):
            continue
        labels.setdefault(int(pid), []).append(row["id"])
    for job in query.experiment_jobs():
        if job.get("status") != "running":
            continue
        directory = detail.resolve_run(job["id"])
        state = detail._load(directory / "state.json") if directory is not None else None
        pid = (state or {}).get("pid")
        if not query.pid_alive(pid):
            continue
        labels.setdefault(int(pid), []).append(f"{job['id']}（{job.get('script') or '队列'}）")
    for pid, cmd in _ps_commands():
        if query.pid_alive(pid):
            commands[pid] = cmd
            labels.setdefault(pid, [])
    own_pgid = os.getpgrp()
    groups: dict[int, dict] = {}
    refused: list[dict] = []
    for pid, names in labels.items():
        if pid <= 1 or pid in (os.getpid(), os.getppid()):
            refused.append({"pid": pid, "ids": names, "reason": "这是看板自己的进程"})
            continue
        try:
            pgid = os.getpgid(pid)
        except OSError:
            refused.append({"pid": pid, "ids": names, "reason": "进程已经不在"})
            continue
        if pgid == own_pgid:
            refused.append({"pid": pid, "ids": names, "reason": "和看板在同一个进程组"})
            continue
        cmd = commands.get(pid) or _command_of(pid)
        if not _is_search_command(cmd):
            refused.append({"pid": pid, "ids": names, "reason": "不是本仓库的搜索进程", "command": cmd})
            continue
        slot = groups.setdefault(pgid, {"pgid": pgid, "pids": [], "ids": [], "command": "", "roots": []})
        slot["pids"].append(pid)
        for name in names:
            if name not in slot["ids"]:
                slot["ids"].append(name)
            root = _root_of(name)
            if root and root not in slot["roots"]:
                slot["roots"].append(root)
        if cmd and not slot["command"]:
            slot["command"] = cmd
    _attach_related(list(groups.values()), own_pgid)
    return list(groups.values()), refused


def _root_of(name: str) -> str:
    """@brief 把历史行 id 解析为运行的绝对目录；实验行会在 id 后接 ``（脚本）``。

    原始英文说明：Absolute run directory for a history id. Experiment rows append ``（脚本）`` after the id.

    @param name 历史行 id。
    @return 绝对目录文本；解析不到时为空串。
    """
    run_id = name.split("（", 1)[0]
    directory = query.resolve_run(run_id)
    if directory is None:
        directory = detail.resolve_run(run_id)
    if directory is None:
        return ""
    try:
        return str(directory.resolve())
    except OSError:
        return str(directory)


def _process_table() -> list[dict]:
    """@brief 列出全部进程的 pid、父 pid、进程组与命令；``ps`` 读不了时为空。

    原始英文说明：Every process: pid, parent, process group, command. Empty when ``ps`` cannot be read.

    @return 进程行字典列表。
    """
    try:
        text = subprocess.check_output(["ps", "-ax", "-o", "pid=,ppid=,pgid=,command="], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in text.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) < 3:
            continue
        try:
            pid, ppid, pgid = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        rows.append({"pid": pid, "ppid": ppid, "pgid": pgid, "command": parts[3] if len(parts) > 3 else ""})
    return rows


def _descendants(root: int, by_parent: dict[int, list[int]]) -> set[int]:
    """@brief 广度收集某个 pid 的全部后代。

    @param root 起始 pid。
    @param by_parent 父 pid 到子 pid 列表的映射。
    @return 后代 pid 集合（不含 ``root`` 自己）。
    """
    found: set[int] = set()
    stack = [root]
    while stack:
        parent = stack.pop()
        for child in by_parent.get(parent, []):
            if child in found or child == root:
                continue
            found.add(child)
            stack.append(child)
    return found


def _attach_related(groups: list[dict], own_pgid: int) -> None:
    """@brief 把子进程以及命令里提到该运行目录的进程所在进程组补进 ``related_pgids``。

    ``hmz exec`` 在新会话里启动，不在搜索的进程组里；已被重新托管的子进程，命令行里仍带运行目录下的
    ``agent_work``。
    原始英文说明：Add the process groups of children and of processes that name this run.

    @param groups ``_stop_groups`` 累计的待停组列表（就地修改）。
    @param own_pgid 看板自身的进程组，需排除。
    """
    table = _process_table()
    by_parent: dict[int, list[int]] = {}
    for row in table:
        by_parent.setdefault(row["ppid"], []).append(row["pid"])
    for group in groups:
        pids = set(group["pids"])
        for pid in list(pids):
            pids |= _descendants(pid, by_parent)
        roots = [root for root in group.get("roots", []) if root]
        if roots:
            for row in table:
                if any(root in row["command"] for root in roots):
                    pids.add(row["pid"])
                    pids |= _descendants(row["pid"], by_parent)
        pgids = []
        seen = {int(group["pgid"])}
        by_pid = {row["pid"]: row for row in table}
        for pid in pids:
            row = by_pid.get(pid)
            pgid = row["pgid"] if row else None
            if pgid is None:
                try:
                    pgid = os.getpgid(pid)
                except OSError:
                    continue
            if pgid <= 1 or pgid == own_pgid or pgid in seen:
                continue
            seen.add(pgid)
            pgids.append(pgid)
        group["related_pgids"] = pgids


def _command_of(pid: int) -> str:
    """@brief 取单个进程的命令行；读不到时返回空串。

    @param pid 进程 id。
    @return 命令行文本。
    """
    try:
        text = subprocess.check_output(["ps", "-o", "command=", "-p", str(pid)], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""
    return text.strip()


def stop_preview() -> dict:
    """@brief ``GET /api/stop``：会停哪些存活搜索；不写文件也不发信号。

    原始英文说明：which live searches would be stopped. Writes nothing and signals nothing.

    @return 含 ``groups``、``refused`` 与 ``dry_run`` 的字典。
    """
    groups, refused = _stop_groups()
    return {"groups": groups, "refused": refused, "dry_run": dry_run()}


def _signal_group(pgid: int) -> str | None:
    """@brief 向一个进程组发 SIGTERM；送达时返回 ``None``。

    原始英文说明：SIGTERM one process group. ``None`` when the signal was delivered.

    @param pgid 进程组 id。
    @return 错误文本；成功或进程已不存在时返回 None。
    """
    try:
        os.kill(-int(pgid), signal.SIGTERM)
    except ProcessLookupError:
        return None
    except OSError as e:
        return str(e)
    return None


def stop_runs() -> dict:
    """@brief ``POST /api/stop``：给每个存活搜索、其进程组及其启动的进程所在组发 SIGTERM。

    ``hmz`` 与 ``pi`` 在自己的会话里，只给搜索组发信号会让它们继续调模型。
    原始英文说明：SIGTERM each live search, its process group, and the groups of processes it started.

    @return 含 ``ok``、``dry_run``、``stopped`` 与 ``refused`` 的字典。
    """
    groups, refused = _stop_groups()
    stopped = []
    for group in groups:
        search_pgid = int(group["pgid"])
        # 先停 hmz / pi 那一组，再停搜索自己。反过来的话，搜索先死，hmz 变成孤儿并继续调模型。
        pgids = [int(item) for item in group.get("related_pgids") or [] if int(item) != search_pgid]
        pgids.append(search_pgid)
        if dry_run():
            stopped.append({**group, "pgids": pgids, "signaled": False, "dry_run": True})
            continue
        sent = []
        for pgid in pgids:
            error = _signal_group(pgid)
            if error:
                reason = f"信号没有送到：{error}" if pgid == search_pgid else f"关联进程组的信号没有送到：{error}"
                refused.append({**group, "pgid": pgid, "reason": reason})
                continue
            sent.append(pgid)
        if sent:
            stopped.append({**group, "pgids": sent, "signaled": True, "dry_run": False})
    return {"ok": True, "dry_run": dry_run(), "stopped": stopped, "refused": refused}


def parse_body(raw: bytes):
    """@brief 解析请求体为 JSON 值。

    @param raw 原始请求体字节。
    @return 解析后的 JSON 值（空体为 None）。
    @raises Refused 不是合法 UTF-8 JSON 时抛出（HTTP 400）。
    """
    try:
        return json.loads(raw.decode("utf-8") or "null")
    except (UnicodeDecodeError, ValueError):
        raise Refused(400, "the body is not valid JSON") from None
