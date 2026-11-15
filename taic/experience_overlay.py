"""Per-run treatment-only experience overlay.

The frozen overlay stays out of ``targets/<id>/experience.md`` and out of
``current_prompt_sha1``. A treatment session reads it once, inside the
experience file ``prepare_session`` stages. A control session, and every
default search, stages the filtered experience file exactly as before.

The binding is an explicit object on the run and the session. An environment
variable is not a binding. The sidecar records the path and the hashes; it
does not repeat the overlay text and it is not a task contract.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .experience_prompt import filter_experience_for_prompt

REPO = Path(__file__).resolve().parents[1]
OVERLAY_MARK = "这份文字是 P2-S 的冻结叠加层，还没有写入 targets/apple-m4-max/experience.md。"
BINDING_SCHEMA = "taic.experience_overlay.binding.1"
SESSION_SCHEMA = "taic.experience_overlay.session.1"
_AUDIT_KEYS = (
    "schema_id",
    "overlay_path",
    "overlay_sha256",
    "rendered_experience_sha256",
    "rendered_experience_bytes",
    "arm",
    "run_id",
    "overlay_copies",
    "authoritative_task_rules",
    "installed_in_global_experience",
    "not_a_task_contract",
)
INJECTION_SCHEMA = "taic.experience_overlay.injection.1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SECOND_CONTRACT = re.compile(r"(?m)^## 任务合同\b|这是权威任务规则")
_EXPERIENCE_RELATIVE = "targets/apple-m4-max/experience.md"


class ExperienceOverlayError(ValueError):
    """@brief 这一轮不能注入经验叠加层。"""


@dataclass(frozen=True)
class ExperienceOverlayBinding:
    """@brief 一次 treatment run 对一份冻结叠加层的显式绑定。

    ``run_id`` 是这一轮目录相对仓库的路径；目录不在仓库里时用解析后的绝对路径。
    对照臂没有绑定。绑定不是环境变量，也不是第二份任务合同。
    """

    overlay_path: str
    overlay_sha256: str
    arm: str
    run_id: str

    def identity(self) -> dict[str, str]:
        """@brief 路径、字节摘要、臂和 run。不含叠加层正文。"""
        return {
            "schema_id": BINDING_SCHEMA,
            "overlay_path": self.overlay_path,
            "overlay_sha256": self.overlay_sha256,
            "arm": self.arm,
            "run_id": self.run_id,
        }


def run_identity(run_dir: Path, repo: Path | None = None) -> str:
    """@brief 这一轮目录的稳定身份。仓库内用相对路径。"""
    root = (repo or REPO).resolve()
    resolved = Path(run_dir).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return resolved.as_posix()


def injection_document() -> dict[str, object]:
    """@brief dry-run 对注入路径的声明。实弹搜索不在这份声明里。"""
    return {
        "schema_id": INJECTION_SCHEMA,
        "ready": True,
        "surface": "prepare_session experience.md",
        "global_experience_md": False,
        "changes_current_prompt_sha1": False,
        "control_copies": 0,
        "treatment_copies": 1,
        "authoritative_task_rules": False,
        "live_search": "未授权/未执行",
        "approval_required": ["allow-spend", "allow-gpu", "approved-by", "positive finite delta"],
    }


def assert_global_experience_clean(repo: Path | None = None) -> None:
    """@brief 全局经验文件里不能已经有这份叠加层。"""
    path = (repo or REPO) / _EXPERIENCE_RELATIVE
    if OVERLAY_MARK in path.read_text(encoding="utf-8"):
        raise ExperienceOverlayError("the global experience file already contains the overlay")


def resolve_overlay_file(path: str, repo: Path | None = None) -> Path:
    """@brief 把记录下来的路径解析到仓库内的文件。越出仓库就拒绝。"""
    root = (repo or REPO).resolve()
    text = str(path or "").strip()
    if not text or text.startswith("~"):
        raise ExperienceOverlayError("overlay path is empty")
    candidate = Path(text)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ExperienceOverlayError("overlay path escapes the repository") from exc
    if not resolved.is_file():
        raise ExperienceOverlayError(f"overlay file does not exist: {relative.as_posix()}")
    return resolved


def verify_overlay_bytes(path: str, sha256: str, repo: Path | None = None) -> bytes:
    """@brief 读叠加层并核对摘要、非空、标记恰好一份、不是第二份任务合同。"""
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ExperienceOverlayError("overlay sha256 is not 64 hex characters")
    assert_global_experience_clean(repo)
    file = resolve_overlay_file(path, repo)
    data = file.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ExperienceOverlayError("overlay sha256 does not match the file")
    text = data.decode("utf-8")
    if not text.strip():
        raise ExperienceOverlayError("the overlay is empty")
    if text.count(OVERLAY_MARK) != 1:
        raise ExperienceOverlayError("the overlay must contain its marker exactly once")
    if _SECOND_CONTRACT.search(text):
        raise ExperienceOverlayError("the overlay must not install a second task contract")
    return data


def render_treatment_experience(raw: str, overlay: str) -> str:
    """@brief 过滤后的经验里恰好拼进一份叠加层。对照文本保持原过滤结果。"""
    if OVERLAY_MARK in raw:
        raise ExperienceOverlayError("the base experience already contains the overlay")
    if not overlay.strip():
        raise ExperienceOverlayError("the overlay is empty")
    if overlay.count(OVERLAY_MARK) != 1:
        raise ExperienceOverlayError("the overlay must contain its marker exactly once")
    if _SECOND_CONTRACT.search(overlay):
        raise ExperienceOverlayError("the overlay must not install a second task contract")
    control = filter_experience_for_prompt(raw)
    if OVERLAY_MARK in control:
        raise ExperienceOverlayError("the control filter already contains the overlay")
    rendered = filter_experience_for_prompt(raw.rstrip() + "\n\n" + overlay)
    if rendered.count(OVERLAY_MARK) != 1:
        raise ExperienceOverlayError("the overlay did not survive filtering as exactly one copy")
    return rendered


def binding_for_run(
    overlay_path: str,
    overlay_sha256: str,
    run_dir: Path,
    *,
    repo: Path | None = None,
) -> ExperienceOverlayBinding:
    """@brief 给这一轮 treatment 目录做绑定，并立刻核对文件字节。"""
    verify_overlay_bytes(overlay_path, overlay_sha256, repo)
    relative = resolve_overlay_file(overlay_path, repo).relative_to((repo or REPO).resolve()).as_posix()
    return ExperienceOverlayBinding(
        overlay_path=relative,
        overlay_sha256=overlay_sha256,
        arm="treatment",
        run_id=run_identity(run_dir, repo),
    )


def binding_from_document(document: dict, *, repo: Path | None = None) -> ExperienceOverlayBinding:
    """@brief 从 run_config 或会话记录还原绑定。臂必须是 treatment。"""
    if not isinstance(document, dict):
        raise ExperienceOverlayError("overlay record is not an object")
    path = document.get("overlay_path")
    digest = document.get("overlay_sha256")
    arm = document.get("arm")
    run_id = document.get("run_id")
    if not isinstance(path, str) or not isinstance(digest, str) or not isinstance(run_id, str):
        raise ExperienceOverlayError("overlay record is missing path, sha256, or run_id")
    if arm != "treatment":
        raise ExperienceOverlayError("only the treatment arm can carry an overlay binding")
    verify_overlay_bytes(path, digest, repo)
    return ExperienceOverlayBinding(path, digest, "treatment", run_id)


def render_bound_experience(
    raw: str,
    binding: ExperienceOverlayBinding | None,
    *,
    repo: Path | None = None,
    run_id: str,
) -> tuple[str, dict | None]:
    """@brief 按绑定渲染会话里的 experience.md。没有绑定就是原来的过滤结果。

    返回的记录只含路径和摘要，不含叠加层正文。
    """
    if binding is None:
        return filter_experience_for_prompt(raw), None
    if binding.arm != "treatment":
        raise ExperienceOverlayError("only the treatment arm can carry an overlay binding")
    if binding.run_id != run_id:
        raise ExperienceOverlayError("overlay binding run_id does not match this run")
    overlay = verify_overlay_bytes(binding.overlay_path, binding.overlay_sha256, repo).decode("utf-8")
    rendered = render_treatment_experience(raw, overlay)
    encoded = rendered.encode("utf-8")
    record = {
        "schema_id": SESSION_SCHEMA,
        "overlay_path": binding.overlay_path,
        "overlay_sha256": binding.overlay_sha256,
        "rendered_experience_sha256": hashlib.sha256(encoded).hexdigest(),
        "rendered_experience_bytes": len(encoded),
        "arm": "treatment",
        "run_id": run_id,
        "overlay_copies": 1,
        "authoritative_task_rules": False,
        "installed_in_global_experience": False,
        "not_a_task_contract": True,
    }
    return rendered, record


def resolve_run_overlay(
    passed: ExperienceOverlayBinding | None,
    *,
    run_dir: Path,
    resuming: bool,
    repo: Path | None = None,
) -> ExperienceOverlayBinding | None:
    """@brief 新 run 用调用方传入的绑定；续跑只认这份 run_config 里的绑定。

    续跑时传入的绑定必须和记录一致。记录属于别的 run 就拒绝。没有记录的 run 继续没有绑定。
    """
    identity = run_identity(run_dir, repo)
    recorded: ExperienceOverlayBinding | None = None
    if resuming:
        path = Path(run_dir) / "run_config.json"
        document: dict = {}
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                document = loaded
        raw = document.get("experience_overlay")
        if raw is not None:
            recorded = binding_from_document(raw, repo=repo)
            if recorded.run_id != identity:
                raise ExperienceOverlayError("recorded overlay belongs to a different run")
    if passed is not None and not isinstance(passed, ExperienceOverlayBinding):
        raise ExperienceOverlayError("experience overlay must be an explicit binding")
    if passed is None:
        return recorded
    if passed.arm != "treatment":
        raise ExperienceOverlayError("only the treatment arm can carry an overlay binding")
    if passed.run_id != identity:
        raise ExperienceOverlayError("overlay binding run_id does not match this run")
    verify_overlay_bytes(passed.overlay_path, passed.overlay_sha256, repo)
    if recorded is not None and recorded.identity() != passed.identity():
        raise ExperienceOverlayError("resume overlay does not match the recorded binding")
    return passed


def freeze_agent_guidance(dest: Path, *, target_id: str = "apple-m4-max") -> None:
    """@brief 把设备、经验和参考技能的原字节抄进这一轮。不写入叠加层。"""
    from tcenv.backends.registry import load_backend
    from tcenv.target import load_target

    from .target import skill_path

    backend = load_backend(load_target(target_id).backend)
    dest.mkdir(parents=True, exist_ok=True)
    names = ("device.md", "experience.md", f"{backend.reference_skill}.md")
    keys = ("device", "experience", backend.reference_skill)
    for name, key in zip(names, keys, strict=True):
        src = skill_path(key, target_id)
        data = src.read_bytes() if src.is_file() else b""
        (dest / name).write_bytes(data)


def seedless_kernel_brief(
    binding,
    search,
    *,
    repo: Path,
    run_dir: Path,
    work_dir: Path,
    guidance_dir: Path,
    overlay: ExperienceOverlayBinding | None,
    target_id: str = "apple-m4-max",
):
    """@brief 开跑前第一场无种子会话的契约。任务书字段与编排器的 seedless brief 对齐。

    叠加层只放在 ``experience_overlay`` 上。任务说明、规则、种子和搜索开关不从这里改。
    """
    import json as _json
    from dataclasses import replace

    from tcenv.backends.registry import load_backend
    from tcenv.target import load_target

    from .kernel_agent import SessionBrief
    from .orchestrator import _resolve_run_track

    track, track_id = _resolve_run_track(binding, "msl-single", None)
    binding = replace(binding, track_id=track_id, track_version=int(track.track_version))
    v2 = bool(binding.is_v2)
    target = load_target(target_id)
    backend = load_backend(target.backend)
    presentation = binding.presentation()
    tolerances = binding.tolerances()
    dsl = bool(binding.track.is_dsl)
    samples = getattr(search, "init_samples", 1)
    multi_draft = isinstance(samples, int) and not isinstance(samples, bool) and samples > 1
    tips = ""
    if getattr(search, "peer_tips", False):
        from .agent_feedback import peer_tips_for

        tips = peer_tips_for(run_dir, now=None)
    guidance = str(guidance_dir.resolve()) if (guidance_dir / "experience.md").is_file() else ""
    return SessionBrief(
        repo=repo,
        task_name=binding.name,
        dtype=binding.dtype,
        size=binding.size,
        kernel_name=f"{binding.contract.op_name}_kernel",
        triton_source="",
        description=presentation.statement,
        contract_text=_json.dumps(presentation.abi_document, indent=2, sort_keys=True),
        abi_stub=binding.signature_stub(),
        rtol=tolerances["rtol"],
        atol_rel=tolerances["atol_rel"],
        v2=v2,
        # eval-v2 没有 --validate。这里与 Orchestrator._session_brief 的 v2 任务书对齐。
        selfcheck_allow_validate=not v2,
        reference_source=presentation.reference_source,
        seedless=True,
        peer_tips=tips,
        label="init",
        work_dir=str(Path(work_dir).resolve()),
        guidance_dir=guidance,
        target_id=target_id,
        file_ext=".taic" if dsl else target.file_ext,
        language_label="TileLang" if dsl else backend.display_name,
        source_fence="python" if dsl else backend.source_fence,
        reference_skill=backend.reference_skill,
        source_name_pattern=r"\bkernel\s+void\s+[A-Za-z_][A-Za-z0-9_]*\b",
        record_direction=multi_draft or bool(getattr(search, "task_ledger", False)),
        reps=int(search.eval_reps),
        require_contract_launch=bool(getattr(search, "require_contract_launch", False)),
        min_improvement=float(search.min_improvement),
        noise_z=float(search.noise_z),
        sweep_tool=bool(getattr(search, "sweep_tool", False)),
        track_id=binding.track_id,
        run_id=overlay.run_id if overlay is not None else "",
        experience_overlay=overlay,
    )


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ExperienceOverlayError(f"{path.name} is not an object")
    return loaded


def _run_dir_for_session(brief) -> Path | None:
    """@brief 从这场会话的冻结指导目录或工作目录找回 run 根。"""
    found: list[Path] = []
    guidance = str(getattr(brief, "guidance_dir", "") or "")
    if guidance:
        found.append(Path(guidance).resolve().parent)
    work = str(getattr(brief, "work_dir", "") or "")
    if work:
        parent = Path(work).resolve().parent
        if parent.name == "agent_work":
            found.append(parent.parent)
    unique = {path.as_posix() for path in found}
    if len(unique) > 1:
        raise ExperienceOverlayError("session paths do not agree on the run directory")
    return found[0] if found else None


def audit_record(document: dict, label: str) -> dict:
    """@brief 核对一份不含正文的审计记录。渲染后摘要必须是 64 位十六进制。"""
    if not isinstance(document, dict) or set(document) != set(_AUDIT_KEYS):
        raise ExperienceOverlayError(f"{label} overlay record fields do not match the audit record")
    rendered = document["rendered_experience_sha256"]
    if not isinstance(rendered, str) or not _SHA256.fullmatch(rendered):
        raise ExperienceOverlayError(f"{label} rendered_experience_sha256 is not 64 hex characters")
    size = document["rendered_experience_bytes"]
    if type(size) is not int or size < 0:
        raise ExperienceOverlayError(f"{label} rendered_experience_bytes is not a byte count")
    if document["schema_id"] != SESSION_SCHEMA or document["arm"] != "treatment":
        raise ExperienceOverlayError(f"{label} overlay record is not a treatment session record")
    if document["overlay_copies"] != 1 or document["authoritative_task_rules"] is not False:
        raise ExperienceOverlayError(f"{label} overlay record is not a single non-authoritative copy")
    if document["installed_in_global_experience"] is not False or document["not_a_task_contract"] is not True:
        raise ExperienceOverlayError(f"{label} overlay record claims a task contract")
    if not isinstance(document["overlay_path"], str) or not isinstance(document["run_id"], str):
        raise ExperienceOverlayError(f"{label} overlay record is missing path or run_id")
    if not isinstance(document["overlay_sha256"], str) or not _SHA256.fullmatch(document["overlay_sha256"]):
        raise ExperienceOverlayError(f"{label} overlay sha256 is not 64 hex characters")
    return {key: document[key] for key in _AUDIT_KEYS}


def _reject_render_mismatch(label: str, stored: dict, expected: dict, *, target: str) -> None:
    if stored == expected:
        return
    if (
        stored["rendered_experience_sha256"] != expected["rendered_experience_sha256"]
        or stored["rendered_experience_bytes"] != expected["rendered_experience_bytes"]
    ):
        raise ExperienceOverlayError(f"{label} rendered_experience_sha256 does not match the {target}")
    raise ExperienceOverlayError(f"{label} overlay record does not match the {target}")


def _overlay_records(file_doc: dict | None, config: dict | None) -> dict | None:
    """@brief 对齐 run 文件和 run_config。两边都在时，配置里的键不能缺。

    没有 ``run_config.json`` 时只看 run 文件。``--render-only`` 就是这种：处理臂
    已经有 ``experience_overlay.json``，但还没有编排器写出的配置。
    """
    config_doc = None if config is None else config.get("experience_overlay")
    if config_doc is not None and file_doc is None:
        raise ExperienceOverlayError("run_config records an overlay but the run file is missing")
    if file_doc is not None and config is not None and config_doc is None:
        raise ExperienceOverlayError("run file records an overlay but run_config omits experience_overlay")
    return config_doc


def binding_for_session(brief) -> ExperienceOverlayBinding | None:
    """@brief 这场会话要注入的绑定。来源是这一轮目录里的记录，或 brief 上的同一份绑定。

    没有记录就是对照或默认搜索。记录和 brief 不一致、run 对不上、或 run_config 和
    run 文件对不上，都拒绝。环境变量不是来源。
    """
    explicit = getattr(brief, "experience_overlay", None)
    if explicit is not None and not isinstance(explicit, ExperienceOverlayBinding):
        raise ExperienceOverlayError("experience overlay must be an explicit binding")
    run_dir = _run_dir_for_session(brief)
    recorded: ExperienceOverlayBinding | None = None
    if run_dir is not None:
        file_doc = _read_json(run_dir / "experience_overlay.json")
        config = _read_json(run_dir / "run_config.json")
        config_doc = _overlay_records(file_doc, config)
        if file_doc is not None and config_doc is not None:
            file_audit = audit_record(file_doc, "run file")
            config_audit = audit_record(config_doc, "run_config")
            _reject_render_mismatch("run_config", config_audit, file_audit, target="run file")
        if file_doc is not None:
            recorded = binding_from_document(file_doc)
            if recorded.run_id != run_identity(run_dir):
                raise ExperienceOverlayError("recorded overlay belongs to a different run")
    if explicit is not None and run_dir is not None and explicit.run_id != run_identity(run_dir):
        raise ExperienceOverlayError("overlay binding run_id does not match this run")
    if explicit is not None and recorded is not None and explicit.identity() != recorded.identity():
        raise ExperienceOverlayError("session overlay does not match the recorded binding")
    return explicit if explicit is not None else recorded


def render_session_experience(brief, raw: str) -> tuple[str, dict | None]:
    """@brief 渲染这场会话的 experience.md，并在写文件前核对审计记录。

    没有绑定时只做原来的过滤，不读取 ``brief.repo``。有绑定时，run 文件、
    ``run_config.json`` 和这次渲染出的摘要必须一致，包括 ``rendered_experience_sha256``。
    """
    overlay = binding_for_session(brief)
    if overlay is None:
        return filter_experience_for_prompt(raw), None
    repo = getattr(brief, "repo", None)
    text, record = render_bound_experience(
        raw, overlay, repo=Path(repo) if repo else None, run_id=overlay.run_id,
    )
    if record is None:
        raise ExperienceOverlayError("treatment render did not produce an audit record")
    fresh = audit_record(record, "session render")
    run_dir = _run_dir_for_session(brief)
    if run_dir is not None:
        file_doc = _read_json(run_dir / "experience_overlay.json")
        config = _read_json(run_dir / "run_config.json")
        config_doc = _overlay_records(file_doc, config)
        if file_doc is not None:
            _reject_render_mismatch("run file", audit_record(file_doc, "run file"), fresh, target="session render")
        if config_doc is not None:
            _reject_render_mismatch("run_config", audit_record(config_doc, "run_config"), fresh, target="session render")
    return text, record


def install_run_overlay(
    run_dir: Path,
    overlay_path: str,
    overlay_sha256: str,
    *,
    repo: Path | None = None,
) -> dict:
    """@brief 把 treatment 绑定写进这一轮。快照里的经验文件保持原字节。

    ``run_config.json`` 已经存在时写入同一个不含正文的记录。渲染后的会话文件仍由
    ``prepare_session`` 产生，而且只产生一份。
    """
    from . import control

    binding = binding_for_run(overlay_path, overlay_sha256, run_dir, repo=repo)
    snapshot = Path(run_dir) / "agent_guidance" / "experience.md"
    if snapshot.is_file():
        raw = snapshot.read_text(encoding="utf-8")
    else:
        raw = ((repo or REPO) / _EXPERIENCE_RELATIVE).read_text(encoding="utf-8")
    _rendered, record = render_bound_experience(raw, binding, repo=repo, run_id=binding.run_id)
    destination = Path(run_dir) / "experience_overlay.json"
    existing = _read_json(destination)
    if existing is not None and existing != record:
        raise ExperienceOverlayError("this run already has a different overlay record")
    destination.write_text(dumps(record), encoding="utf-8")
    config_path = Path(run_dir) / "run_config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ExperienceOverlayError("run_config.json is not an object")
        prior = config.get("experience_overlay")
        if prior is not None and prior != record:
            raise ExperienceOverlayError("run_config overlay does not match this binding")
        config["experience_overlay"] = record
        control.write_json_atomic(config_path, config)
    return record


def dumps(document: dict) -> str:
    """@brief 稳定的 JSON 文本。"""
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


__all__ = [
    "BINDING_SCHEMA",
    "ExperienceOverlayBinding",
    "ExperienceOverlayError",
    "INJECTION_SCHEMA",
    "OVERLAY_MARK",
    "SESSION_SCHEMA",
    "assert_global_experience_clean",
    "binding_for_run",
    "binding_for_session",
    "binding_from_document",
    "dumps",
    "freeze_agent_guidance",
    "injection_document",
    "install_run_overlay",
    "audit_record",
    "render_bound_experience",
    "render_session_experience",
    "render_treatment_experience",
    "resolve_overlay_file",
    "resolve_run_overlay",
    "run_identity",
    "seedless_kernel_brief",
    "verify_overlay_bytes",
]
