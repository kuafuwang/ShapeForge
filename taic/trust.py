"""Per-target model trust grants, their append-only audit, and the checks a proposal needs.

Trust is deliberately *not* a model's self-report. A person grants it for one
(target, task, provider, model, effort) tuple, for named scopes, with the prompt and
measurement protocol it was granted against. A later prompt or protocol change makes the
grant stale instead of silently carrying it over. The registry lives with the target:

    targets/<target_id>/trust.toml
    targets/<target_id>/trust_audit.jsonl

No credentials are stored here. The file is small enough to review with the device data.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

from tcenv.schema import PROTOCOL_VERSION

REPO_ROOT = Path(__file__).resolve().parent.parent
SCOPES = frozenset({"overlay:tip", "overlay:premise", "propose:skill", "propose:task_wording"})
STATUSES = frozenset({"active", "suspended", "revoked"})
_TARGET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TrustError(ValueError):
    """@brief 无法满足的信任授权或查询。"""


def target_root(target_id: str, root: Path | None = None) -> Path:
    """@brief 取目标目录。

    @param target_id 目标标识，必须是合法的目录名。
    @param root 存放 ``targets/`` 的根；测试与备份仓库可指定。
    @return 该目标的目录路径。
    @raises TrustError ``target_id`` 不是合法的目标名时。
    """
    if not _TARGET.fullmatch(str(target_id or "")):
        raise TrustError(f"invalid target id {target_id!r}")
    base = Path(root) if root is not None else REPO_ROOT / "targets"
    return base / target_id


def trust_path(target_id: str, root: Path | None = None) -> Path:
    """@brief 取一个目标的 ``trust.toml`` 路径。

    @param target_id 目标标识。
    @param root 存放 ``targets/`` 的根。
    @return 信任登记表路径。
    """
    return target_root(target_id, root) / "trust.toml"


def audit_path(target_id: str, root: Path | None = None) -> Path:
    """@brief 取一个目标的 ``trust_audit.jsonl`` 路径。

    @param target_id 目标标识。
    @param root 存放 ``targets/`` 的根。
    @return 信任审计日志路径。
    """
    return target_root(target_id, root) / "trust_audit.jsonl"


# The model-facing prompt surface, by *text* rather than by whole file. A code edit in
# ``kernel_agent.py``/``harness.py`` that never reaches the model (billing, subprocess handling,
# a test path) must not invalidate a grant; a change to a prompt function, a template constant or a
# skill document must.
_PROMPT_FUNCTIONS = (
    ("prompts.system_prompt", "taic.prompts", "system_prompt"),
    ("prompts.system_prompt_v2", "taic.prompts", "system_prompt_v2"),
    ("kernel_agent._check_command", "taic.kernel_agent", "_check_command"),
    # ``signature_stub`` is a *method*, so the symbol is a dotted attribute path resolved on the
    # class; registering it as a module function would raise ``AttributeError`` and take the whole
    # fingerprint down with it.
    ("task_binding.TaskBinding.signature_stub", "taic.task_binding", "TaskBinding.signature_stub"),
    ("prompts.initial_user", "taic.prompts", "initial_user"),
    ("prompts.repair_user", "taic.prompts", "repair_user"),
    ("prompts.planning_system", "taic.prompts", "planning_system"),
    ("prompts.plan_user", "taic.prompts", "plan_user"),
    ("prompts.patch_user", "taic.prompts", "patch_user"),
    ("prompts._rules_for", "taic.prompts", "_rules_for"),
    ("track_prompt.prompt_style_for_track", "taic.track_prompt", "prompt_style_for_track"),
    ("track_prompt.prompt_style_for_id", "taic.track_prompt", "prompt_style_for_id"),
    ("prompts._roofline_notes", "taic.prompts", "_roofline_notes"),
    ("kernel_agent.render_contract", "taic.kernel_agent", "render_contract"),
    ("kernel_agent._cap_task", "taic.kernel_agent", "_cap_task"),
    ("harness.write_template", "taic.harness", "write_template"),
)
_PROMPT_CONSTANTS = (
    ("prompts.HARNESS_RULES", "taic.prompts", "HARNESS_RULES"),
    ("prompts.MIXED_PLANS", "taic.prompts", "MIXED_PLANS"),
    ("prompts.WIKI_DOCS", "taic.prompts", "WIKI_DOCS"),
    ("harness.TEMPLATE", "taic.harness", "TEMPLATE"),
)
_GLOBAL_TEXT_FILES = (
    "taic/flows/kernel_once/skills/taic-tools/SKILL.md",
)


def _relative_or_absolute(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _backend_skill_files(target_id: str | None = None) -> tuple[str, ...]:
    """@brief 一个目标档的后端技能文档，按路径排序。

    ``prompts.system_prompt`` 注入的 ``backend.skill_*`` 落在 ``backends/<backend>/skills/*.md``；
    后端由目标档决定，所以第二个 backend 不需要改本模块。

    @param target_id 目标标识；``None`` 用默认目标。
    @return 相对仓库根（或目标不在仓库内时的绝对路径）的文件列表；目录不存在时为空。
    """
    from tcenv.target import load_target

    try:
        skills = load_target(target_id).skills
    except Exception:  # noqa: BLE001 - a missing profile is reported by the caller that needs it
        return ()
    if not skills.is_dir():
        return ()
    return tuple(_relative_or_absolute(path) for path in sorted(skills.glob("*.md")))


def _target_text_files(target_id: str | None) -> tuple[str, ...]:
    """@brief 一个目标档注入的文本：experience.md、device.md 与该后端的技能文档。

    Resolved from the target profile instead of a hardcoded ``apple-m4-max`` path, so a grant for
    another target is bound to that target's documents.

    @param target_id 目标标识；``None`` 用默认目标。
    @return 文本文件列表。
    """
    from .target import skill_path

    paths: list[Path] = []
    for name in ("experience", "device"):
        try:
            path = skill_path(name, target_id)
        except Exception:  # noqa: BLE001
            continue
        if path.is_file():
            paths.append(path)
    return tuple(_relative_or_absolute(path) for path in paths) + _backend_skill_files(target_id)


def _prompt_text_files(target_id: str | None = None) -> tuple[str, ...]:
    return _GLOBAL_TEXT_FILES + _target_text_files(target_id)


def _resolve_symbol(module_name: str, symbol: str):
    """@brief 解析一个模块符号，支持点分属性路径（``Class.method``）。

    ``TaskBinding.signature_stub`` is a method: ``getattr(module, "signature_stub")`` would raise
    ``AttributeError`` and take ``prompt_fingerprint_parts()`` down with it.  A dotted path walks the
    attributes instead, and a label that does not resolve is a hard error -- silently hashing nothing
    would weaken the fingerprint without anyone noticing.

    @param module_name 模块名。
    @param symbol 符号名或点分属性路径。
    @return 解析到的对象。
    @raises ValueError 路径无法解析。
    """
    import importlib

    value = importlib.import_module(module_name)
    for part in str(symbol).split("."):
        try:
            value = getattr(value, part)
        except AttributeError as exc:
            raise ValueError(f"prompt symbol {module_name}.{symbol} does not resolve at {part!r}") from exc
    return value


def _symbol_bytes(module_name: str, symbol: str) -> bytes:
    import inspect

    value = _resolve_symbol(module_name, symbol)
    if callable(value):
        try:
            return inspect.getsource(value).encode("utf-8")
        except (OSError, TypeError):
            # A frozen or generated function: fall back to the module file, which is conservative.
            import importlib

            try:
                return Path(inspect.getfile(importlib.import_module(module_name))).read_bytes()
            except (OSError, TypeError):
                return symbol.encode("utf-8")
    return repr(value).encode("utf-8")


def prompt_fingerprint_parts(target_id: str | None = None) -> list[tuple[str, bytes]]:
    """@brief 指纹覆盖的提示面：函数源码、模板常量、目标文档与技能文档。

    @param target_id 目标标识；``None`` 用默认目标。
    @return ``(标签, 字节)`` 列表；标签是 ``模块.符号`` 或文件路径。
    """
    parts = [(label, _symbol_bytes(module, symbol)) for label, module, symbol in _PROMPT_FUNCTIONS + _PROMPT_CONSTANTS]
    for relative in _prompt_text_files(target_id):
        parts.append((relative, (REPO_ROOT / relative).read_bytes()))
    return parts


def current_prompt_sha1(target_id: str | None = None) -> str:
    """@brief 提示/TASK 面的指纹，改动它会让旧授权变旧。

    Only the text the model is asked to follow is hashed: the prompt-rendering functions, the
    template constants and the skill documents. Unrelated code in the same modules no longer
    changes the fingerprint. A grant must not silently survive a change to that text.

    @param target_id 目标标识；``None`` 用默认目标。两个 target 的文档不同，指纹也不同。
    """
    digest = hashlib.sha256()
    for label, data in prompt_fingerprint_parts(target_id):
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(data)
        digest.update(b"\0")
    return digest.hexdigest()


def current_protocol_version() -> str:
    """@brief 当前测量协议版本号，写入授权并在校验时比对。

    @return ``tcenv.schema.PROTOCOL_VERSION`` 的字符串形式。
    """
    return str(PROTOCOL_VERSION)


def _toml_value(value) -> str:
    """@brief 把一个 Python 值渲染成 TOML 字面量。

    @param value 布尔、数值、列表或字符串。
    @return TOML 字面量文本。
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # JSON basic strings use the same escaping TOML basic strings accept for our values.
    return json.dumps(str(value), ensure_ascii=False)


def _dump(grants: list[dict]) -> str:
    """@brief 把授权列表序列化成 ``trust.toml`` 的文本。

    每个授权写成一个 ``[[grants]]``，字段顺序固定，便于人读和 diff。

    @param grants 授权字典列表。
    @return 完整的 TOML 文本（末尾一个换行）。
    """
    lines = ["version = 1", ""]
    for grant in grants:
        lines.append("[[grants]]")
        for key in (
            "id", "target_id", "task", "provider", "model", "effort", "prompt_sha1",
            "protocol_version", "status", "granted_by", "granted_at", "reason",
            "evidence_run", "expires_at", "scopes",
        ):
            if key not in grant:
                continue
            lines.append(f"{key} = {_toml_value(grant[key])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _finite(value) -> float | None:
    """@brief 把输入转成有限浮点数。

    @param value 待转换的值；bool 不算数值。
    @return 有限浮点数；不合法时（含 NaN/Inf）为 ``None``。
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _grant_id(target_id: str, task: str, provider: str, model: str, effort: str) -> str:
    """@brief 算出一次授权的稳定标识。

    同一 (target, task, provider, model, effort) 组合总是得到同一个 id，所以重新授权会替换旧条目。

    @param target_id 目标标识。
    @param task 任务名。
    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度。
    @return 16 字符的十六进制摘要。
    """
    payload = "\0".join((str(target_id), str(task), str(provider), str(model), str(effort)))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def load_grants(target_id: str, root: Path | None = None) -> list[dict]:
    """@brief 读一个目标的信任登记表。

    @param target_id 目标标识。
    @param root 存放 ``targets/`` 的根。
    @return 授权字典列表；文件缺失或没有 ``grants`` 时为空列表。
    @raises TrustError 文件读不出来或 TOML 不合法时。
    """
    path = trust_path(target_id, root)
    if not path.is_file():
        return []
    try:
        doc = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise TrustError(f"cannot read {path}: {exc}") from exc
    rows = doc.get("grants")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _write(target_id: str, grants: list[dict], root: Path | None) -> None:
    """@brief 原子地重写 ``trust.toml``。

    @param target_id 目标标识。
    @param grants 要写入的授权列表。
    @param root 存放 ``targets/`` 的根。
    """
    path = trust_path(target_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    try:
        tmp.write_text(_dump(grants), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


def _audit(target_id: str, event: dict, root: Path | None) -> None:
    """@brief 追加一条审计记录，并在独占 flock 下 fsync。

    审计日志有自己的锁；全局顺序总是先登记表、后审计。

    @param target_id 目标标识。
    @param event 要记录的事件。
    @param root 存放 ``targets/`` 的根。
    """
    path = audit_path(target_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    lock = path.with_name(".trust_audit.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            with path.open("a", encoding="utf-8") as out:
                out.write(line)
                out.flush()
                os.fsync(out.fileno())
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def read_audit(target_id: str, root: Path | None = None) -> list[dict]:
    """@brief 读一个目标的信任审计日志。

    损坏的行直接跳过，不让一行坏数据挡住整份审计。

    @param target_id 目标标识。
    @param root 存放 ``targets/`` 的根。
    @return 审计事件列表；文件不存在时为空列表。
    """
    path = audit_path(target_id, root)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


@contextmanager
def _registry_lock(target_id: str, root: Path | None):
    """@brief 跨进程串行化一个目标的 ``trust.toml`` 读-改-写。

    没有它，搜索进程自动暂停某个模型、同时有人在跑 ``taic trust grant/revoke`` 时，会丢掉一边：
    审计里记下了一次登记表已经不再持有的暂停，于是降级时会失效开放。

    @param target_id 目标标识。
    @param root 存放 ``targets/`` 的根。
    """
    path = trust_path(target_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(".trust_registry.lock")
    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def grant(
    target_id: str,
    task: str,
    provider: str,
    model: str,
    *,
    effort: str = "",
    scopes: list[str] | tuple[str, ...] = ("overlay:tip",),
    granted_by: str = "cli",
    reason: str = "",
    evidence_run: str = "",
    expires_at: float | None = None,
    prompt_sha1: str | None = None,
    protocol_version: str | None = None,
    root: Path | None = None,
) -> dict:
    """@brief 授予或替换一条 (target, task, model) 信任条目。

    已销的条目可以重新授予。写入后追加一条 ``grant`` 审计，并返回新条目。

    @param target_id 目标标识。
    @param task 任务名。
    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度。
    @param scopes 授予的作用域集合。
    @param granted_by 授予人。
    @param reason 授予理由。
    @param evidence_run 依据的运行标识。
    @param expires_at 过期时间戳；``None`` 表示不过期。
    @param prompt_sha1 授权时的提示指纹；``None`` 时取当前指纹。
    @param protocol_version 授权时的协议版本；``None`` 时取当前版本。
    @param root 存放 ``targets/`` 的根。
    @return 新写入的授权条目。
    @raises TrustError 必填字段缺失、作用域不合法，或 ``expires_at`` 不是未来的有限时间时。
    """
    target_id, task = str(target_id).strip(), str(task).strip()
    provider, model, effort = str(provider).strip(), str(model).strip(), str(effort or "").strip()
    if not target_id or not task or not provider or not model:
        raise TrustError("target, task, provider and model are required")
    clean_scopes = sorted(set(str(scope).strip() for scope in scopes if str(scope).strip()))
    if not clean_scopes or any(scope not in SCOPES for scope in clean_scopes):
        raise TrustError(f"scopes must be a non-empty subset of {sorted(SCOPES)}")
    expiry = _finite(expires_at)
    if expires_at is not None and (expiry is None or expiry <= time.time()):
        raise TrustError("expires_at must be a future finite timestamp")
    ident = _grant_id(target_id, task, provider, model, effort)
    now = time.time()
    entry = {
        "id": ident,
        "target_id": target_id,
        "task": task,
        "provider": provider,
        "model": model,
        "effort": effort,
        "prompt_sha1": str(prompt_sha1 if prompt_sha1 is not None else current_prompt_sha1(target_id)),
        "protocol_version": str(protocol_version if protocol_version is not None else current_protocol_version()),
        "status": "active",
        "granted_by": str(granted_by or "cli"),
        "granted_at": now,
        "reason": str(reason or ""),
        "evidence_run": str(evidence_run or ""),
        "expires_at": expiry if expiry is not None else 0.0,
        "scopes": clean_scopes,
    }
    replaced = None
    with _registry_lock(target_id, root):
        grants = load_grants(target_id, root)
        for index, row in enumerate(grants):
            if row.get("id") == ident:
                replaced = row
                grants[index] = entry
                break
        else:
            grants.append(entry)
        _write(target_id, grants, root)
        _audit(target_id, {"t": now, "event": "grant", "grant": entry, "previous": replaced}, root)
    return entry


def set_status(
    target_id: str,
    *,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    status: str,
    by: str = "cli",
    reason: str = "",
    root: Path | None = None,
) -> dict:
    """@brief 暂停或销一条授权。

    @param target_id 目标标识。
    @param task 任务名。
    @param provider provider 名。
    @param model 模型名。
    @param effort 推理强度。
    @param status 新状态，只能是 ``suspended`` 或 ``revoked``。
    @param by 操作人。
    @param reason 操作理由。
    @param root 存放 ``targets/`` 的根。
    @return 更新后的授权条目。
    @raises TrustError 状态不是暂停/销，或没有对应的授权条目时。
    """
    if status not in ("suspended", "revoked"):
        raise TrustError("set_status only suspends or revokes; grant creates an active entry")
    ident = _grant_id(target_id, task, provider, model, effort)
    with _registry_lock(target_id, root):
        grants = load_grants(target_id, root)
        for row in grants:
            if row.get("id") != ident:
                continue
            previous = dict(row)
            row["status"] = status
            _write(target_id, grants, root)
            _audit(
                target_id,
                {"t": time.time(), "event": status, "by": by, "reason": reason, "previous": previous, "grant": row},
                root,
            )
            return dict(row)
    raise TrustError(f"no trust entry for {target_id}/{task} {provider}/{model} effort={effort!r}")


def revoke(target_id: str, **kwargs) -> dict:
    """@brief 销一条授权（``set_status`` 的薄封装）。

    @param target_id 目标标识。
    @param kwargs 其余参数原样传给 ``set_status``（需含 task/provider/model）。
    @return 更新后的授权条目。
    @raises TrustError 没有对应的授权条目时。
    """
    kwargs["status"] = "revoked"
    return set_status(target_id, **kwargs)


def suspend(target_id: str, **kwargs) -> dict:
    """@brief 暂停一条授权（``set_status`` 的薄封装）。

    @param target_id 目标标识。
    @param kwargs 其余参数原样传给 ``set_status``（需含 task/provider/model）。
    @return 更新后的授权条目。
    @raises TrustError 没有对应的授权条目时。
    """
    kwargs["status"] = "suspended"
    return set_status(target_id, **kwargs)


def check(
    entry: dict,
    *,
    target_id: str,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    scope: str = "overlay:tip",
    protocol_version: str | None = None,
    prompt_sha1: str | None = None,
    now: float | None = None,
) -> tuple[bool, str]:
    """@brief 判断一条授权此刻是否允许该提案者在该作用域下动手。

    逐项检查状态、目标、任务、provider/model/effort、作用域、过期时间、协议版本与提示指纹。

    @param entry 授权条目。
    @param target_id 当前目标标识。
    @param task 当前任务名。
    @param provider 提案者的 provider。
    @param model 提案者的模型。
    @param effort 提案者的推理强度。
    @param scope 本次要用的作用域。
    @param protocol_version 当前协议版本；``None`` 时取当前值。
    @param prompt_sha1 当前提示指纹；``None`` 时取当前值。
    @param now 当前时间戳；``None`` 时取 ``time.time()``。
    @return ``(是否允许, 不允许的原因)``；允许时原因为空字符串。
    """
    if not isinstance(entry, dict):
        return False, "missing trust entry"
    if entry.get("status") != "active":
        return False, f"trust is {entry.get('status') or 'not active'}"
    if entry.get("target_id") != target_id:
        return False, "trust is for another target"
    if entry.get("task") != task:
        return False, "trust is for another task"
    if (entry.get("provider"), entry.get("model"), str(entry.get("effort") or "")) != (provider, model, str(effort or "")):
        return False, "trust is for another provider/model/effort"
    if scope not in (entry.get("scopes") or []):
        return False, f"trust does not include {scope}"
    expiry = _finite(entry.get("expires_at"))
    if expiry is not None and expiry > 0 and (now if now is not None else time.time()) >= expiry:
        return False, "trust has expired"
    if str(entry.get("protocol_version") or "") != str(protocol_version if protocol_version is not None else current_protocol_version()):
        return False, "trust was granted for another measurement protocol"
    expected_prompt = str(entry.get("prompt_sha1") or "")
    if not expected_prompt:
        # A row without the fingerprint cannot be checked against the current prompt surface.
        # Treat it as stale rather than as a grant for every prompt.
        return False, "trust entry has no prompt fingerprint"
    if expected_prompt != str(prompt_sha1 if prompt_sha1 is not None else current_prompt_sha1(target_id)):
        return False, "trust was granted for another prompt"
    return True, ""


def is_trusted(
    target_id: str,
    *,
    task: str,
    provider: str,
    model: str,
    effort: str = "",
    scope: str = "overlay:tip",
    root: Path | None = None,
    **kwargs,
) -> tuple[bool, str]:
    """@brief 在某个目标的所有授权里找一条当前可用的。

    @param target_id 目标标识。
    @param task 当前任务名。
    @param provider 提案者的 provider。
    @param model 提案者的模型。
    @param effort 提案者的推理强度。
    @param scope 本次要用的作用域。
    @param root 存放 ``targets/`` 的根。
    @param kwargs 其余参数原样传给 ``check``（如 ``protocol_version``/``prompt_sha1``/``now``）。
    @return ``(是否受信任, 原因)``；没有可用授权时原因为 ``no active grant``。
    """
    for entry in load_grants(target_id, root):
        ok, why = check(
            entry, target_id=target_id, task=task, provider=provider, model=model,
            effort=effort, scope=scope, **kwargs,
        )
        if ok:
            return True, ""
    return False, "no active grant"
