"""Task-scoped compact experience overlay for one measured gemm_silu shape.

P2-S keeps the full overlay. P2-U's paid pair stays ``no_gain`` and is not an
approval for this file. Selection is an exact ``task/dtype/size/target_id``
projection of the ``gemm-silu-reg`` catalog lesson. Keyword search is not a
selector. The text is injected only through the P2-T treatment session path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from experiments.experience_catalog import _close, _fmt_us, build_catalog
from taic.experience_overlay import OVERLAY_MARK, verify_overlay_bytes

REPO = Path(__file__).resolve().parents[1]
FIXTURES = "tests/support/golden/shared/acceptance-epochs/fixtures"
_P2O = "tests/support/golden/shared/acceptance-epochs/fixtures/p2o"
FRESH_REL = f"{_P2O}/evidence/reg-small-confirmation.json"
PROBE_REL = f"{_P2O}/evidence/scratch-probe.json"
DECISION_REL = f"{_P2O}/evidence/decision.json"
DOC_REL = "docs/架构/gemm-silu-scratch-headroom.md"
FULL_REL = f"{FIXTURES}/p2s/evidence/experience-overlay.md"
COMPACT_REL = f"{FIXTURES}/p2v/evidence/compact-overlay.md"
P2U_CONCLUSION = "tests/support/golden/shared/acceptance-epochs/fixtures/p2u/conclusion.json"
P2U_DOC = "docs/架构/experience-overlay-paid-result.md"
EXPERIENCE_REL = "targets/apple-m4-max/experience.md"
# P2-V 是在 P2-U 当次对照臂上做的反事实投影，必须使用同一份历史 guidance。
P2U_BASELINE_REL = (
    "runs/p2u-experience-overlay-paid-20261114/control/gemm_silu_small_rep1/"
    "agent_guidance/experience.md"
)
PROJECTION_ID = "gemm-silu-fp16-small"
LESSON_ID = "gemm-silu-reg"
SCOPE = {
    "task": "gemm_silu",
    "dtype": "fp16",
    "size": "small",
    "target_id": "apple-m4-max",
}
REG_US = "1.8252"
FLOOR_US = "8.7713"
MAX_OVERLAY_BYTES = 1100
P2U_EXPERIMENT_ID = "p2u-experience-overlay-paid-20261114"
P2U_APPROVED_BY = "kuafu"
P2U_DELTA = 0.01
CONTROL_BYTES = 13415
FULL_TREATMENT_BYTES = 25469
CONTROL_SHA = "e972450e333550739378afe9362e17470bc9f59fba089a849835e28402a8fea2"
FULL_TREATMENT_SHA = "8e5adbecd3f764904ceab72c2e35fc7534d08931fde0a24210292cdf23513304"
FULL_OVERLAY_SHA = "8e3b9f975cb3988e03af09a12e131bfdb8a699631c8780d544d3836f4bdb97cc"
_SCOPE_KEYS = ("overlay_kind", "task", "dtype", "size", "target_id")
_BANNED = (
    "conv2d",
    "bitdelta",
    "kda",
    "varlen",
    "split-k",
    "splitk",
    "paper",
    "1160",
    "1.6721",
    "1.0071",
    "simdgroup",
    "steel",
    "bm=",
    "bk=",
    "grid=",
)
_REQUIRED = (
    "一次性 SiLU 中间结果留在寄存器",
    "不省 FLOPs",
    "fp32 写读",
    "barrier",
    "producer/consumer",
    "dispatch",
    REG_US,
    FLOOR_US,
    "not_a_candidate=true",
    "不是发布 speedup",
    "不指定候选 tiling",
    "fresh identity",
    "publication_allowed=false",
    "overlay_kind: compact",
    OVERLAY_MARK,
)


class CompactProjectionError(ValueError):
    """@brief 这一份紧凑叠加层不能被选中、渲染或执行。"""


def _read(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def lookup_projection(*, task: str, dtype: str, size: str, target_id: str) -> dict[str, str]:
    """@brief 用四元组精确命中唯一投影。大小写和子串都不算命中。"""
    found = {"task": task, "dtype": dtype, "size": size, "target_id": target_id}
    if found != SCOPE:
        raise CompactProjectionError(
            f"no compact projection for {task}/{dtype}/{size}/{target_id}"
        )
    return dict(SCOPE)


def parse_compact_scope(text: str) -> dict[str, str] | None:
    """@brief 读叠加层里的范围行。没有 ``overlay_kind`` 就是完整叠加层。"""
    found: dict[str, str] = {}
    for line in text.splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        if key not in _SCOPE_KEYS:
            continue
        if key in found:
            raise CompactProjectionError("compact overlay repeats a scope field")
        found[key] = value.strip()
    if not found:
        return None
    if "overlay_kind" not in found:
        raise CompactProjectionError("overlay has scope fields without overlay_kind")
    if found["overlay_kind"] != "compact":
        raise CompactProjectionError("unknown overlay_kind")
    missing = [key for key in _SCOPE_KEYS if key not in found or not found[key]]
    if missing:
        raise CompactProjectionError(f"compact overlay is missing {', '.join(missing)}")
    return {key: found[key] for key in _SCOPE_KEYS}


def _require_measured_row(root: Path) -> dict[str, Any]:
    """@brief 从目录投影和已复制的 gemm_silu 证据读 small 这一行。对不上就拒绝写正文。"""
    catalog, _full = build_catalog(root)
    lesson = next((item for item in catalog["lessons"] if item["id"] == LESSON_ID), None)
    if lesson is None:
        raise CompactProjectionError("gemm-silu-reg is missing from the catalog")
    if lesson.get("task") != SCOPE["task"] or lesson.get("dtype") != SCOPE["dtype"]:
        raise CompactProjectionError("gemm-silu-reg catalog identity drifted")
    if set(lesson["sizes"]) != {"small", "paper"}:
        raise CompactProjectionError("gemm-silu-reg sizes drifted")
    row = lesson["sizes"]["small"]
    fresh = row["fresh"]
    if fresh.get("source") != FRESH_REL or lesson.get("probe_source") != PROBE_REL or lesson.get("source") != DECISION_REL:
        raise CompactProjectionError("gemm_silu small sources drifted")
    if row.get("lower_bound_evidence_class") != "probe_floor":
        raise CompactProjectionError("small floor is no longer a probe")
    if fresh.get("evidence_class") != "fresh_subject" or fresh.get("status") != "confirmed":
        raise CompactProjectionError("small REG fresh subject drifted")
    if fresh.get("identity") != "match" or int(fresh.get("search_seed")) != 7 or int(fresh.get("confirmation_seed")) == 7:
        raise CompactProjectionError("small REG fresh identity drifted")
    decision = _read(root, DECISION_REL)
    probe = _read(root, PROBE_REL)
    if decision.get("task") != SCOPE["task"] or decision.get("dtype") != SCOPE["dtype"] or decision.get("target_id") != SCOPE["target_id"]:
        raise CompactProjectionError("gemm_silu decision identity drifted")
    small = decision["sizes"]["small"]
    if small.get("global_would_win") is not False:
        raise CompactProjectionError("gemm_silu small no longer rejects GLOBAL")
    if "global_saves_no_flops" not in small.get("stop_reasons", []):
        raise CompactProjectionError("gemm_silu small no longer records that GLOBAL saves no FLOPs")
    if decision.get("implementation", {}).get("global_candidate_source_added") is not False:
        raise CompactProjectionError("a GLOBAL gemm_silu candidate was added")
    if decision.get("probe_not_a_candidate") is not True or probe.get("not_a_candidate") is not True:
        raise CompactProjectionError("gemm_silu probe lost not_a_candidate")
    arm = probe["sizes"]["small"]["arms"]["two_stage_floor"]
    if int(arm.get("fill_bytes_per_iteration")) != 0 or int(arm.get("stages")) != 2:
        raise CompactProjectionError("two-stage floor is no longer a no-fill two-stage probe")
    if arm.get("stage_kernels") != ["probe_touch_producer", "probe_touch_consumer"]:
        raise CompactProjectionError("two-stage floor kernels drifted")
    _close(float(fresh["median_us"]), float(small["reg_candidate_us"]), source="compact REG fresh")
    _close(float(row["two_stage_floor_us"]), float(small["two_stage_floor_us"]), source="compact floor decision")
    _close(float(arm["wall_us_median"]), float(small["two_stage_floor_us"]), source="compact floor probe")
    if _fmt_us(float(fresh["median_us"])) != REG_US or _fmt_us(float(arm["wall_us_median"])) != FLOOR_US:
        raise CompactProjectionError("formatted gemm_silu numbers drifted")
    rationale = str(decision.get("rationale") or "")
    for phrase in ("It saves no FLOPs", "fp32 intermediate write+read", "producer + barrier + consumer"):
        if phrase not in rationale:
            raise CompactProjectionError(f"gemm_silu rationale lost {phrase!r}")
    headroom = (root / DOC_REL).read_text(encoding="utf-8")
    for phrase in ("留在寄存器", "没有实现 GLOBAL 候选", REG_US, FLOOR_US, "producer → barrier → consumer", "无 fill"):
        if phrase not in headroom:
            raise CompactProjectionError(f"gemm_silu headroom lost {phrase!r}")
    omitted = [item["id"] for item in catalog["lessons"] if item["id"] != LESSON_ID]
    return {
        "reg_fresh_us": float(fresh["median_us"]),
        "two_stage_floor_us": float(arm["wall_us_median"]),
        "confirmation_seed": int(fresh["confirmation_seed"]),
        "identity": "match",
        "search_seed": 7,
        "evidence_class_floor": "probe_floor",
        "not_a_candidate": True,
        "global_candidate_source_added": False,
        "global_would_win": False,
        "omitted_lesson_ids": omitted,
        "omitted_size": "paper",
        "sources": [FRESH_REL, PROBE_REL, DECISION_REL, DOC_REL],
    }


def render_compact_text() -> str:
    """@brief 渲染正文。数字只来自已经核对过的两个格式化值。"""
    lines = [
        "## gemm_silu fp16 small",
        "",
        "overlay_kind: compact",
        f"task: {SCOPE['task']}",
        f"dtype: {SCOPE['dtype']}",
        f"size: {SCOPE['size']}",
        f"target_id: {SCOPE['target_id']}",
        "Confidence: verified",
        OVERLAY_MARK,
        (
            "观察: 一次性 SiLU 中间结果留在寄存器。GLOBAL scratch 不省 FLOPs，"
            "还增加 fp32 写读、barrier、producer/consumer 和 dispatch。没有写 GLOBAL 候选。"
            f"REG fresh {REG_US} µs，两阶段 no-fill floor {FLOOR_US} µs，not_a_candidate=true。"
            "这是已测结构边界，不是发布 speedup。候选自有算法与 tiling。正确性与 fresh identity 仍按原合同。"
        ),
        (
            "来源: `docs/架构/gemm-silu-scratch-headroom.md`；"
            + (REPO / _P2O / "compact-source.txt").read_text(encoding="utf-8").rstrip("\n")
        ),
        f"本机实测: Apple M4 Max，fp16 small，REG fresh {REG_US} µs，两阶段 no-fill floor {FLOOR_US} µs。",
        "边界: 只覆盖 gemm_silu、fp16、small。不指定候选 tiling。不写其他任务合同。publication_allowed=false。",
        "",
    ]
    text = "\n".join(lines)
    lowered = text.casefold()
    for banned in _BANNED:
        if banned.casefold() in lowered:
            raise CompactProjectionError(f"compact overlay contains {banned}")
    for phrase in _REQUIRED:
        if phrase not in text:
            raise CompactProjectionError(f"compact overlay lost {phrase!r}")
    if text.count(OVERLAY_MARK) != 1 or text.count("## ") != 1:
        raise CompactProjectionError("compact overlay marker or heading drifted")
    if parse_compact_scope(text) != {"overlay_kind": "compact", **SCOPE}:
        raise CompactProjectionError("compact overlay scope lines drifted")
    size = len(text.encode("utf-8"))
    if not 400 <= size <= MAX_OVERLAY_BYTES:
        raise CompactProjectionError(f"compact overlay is {size} bytes")
    return text


def project_compact_overlay(
    root: Path | None = None,
    *,
    task: str = SCOPE["task"],
    dtype: str = SCOPE["dtype"],
    size: str = SCOPE["size"],
    target_id: str = SCOPE["target_id"],
) -> str:
    """@brief 投影并返回紧凑正文。错的任务、dtype、形状或目标直接拒绝。"""
    lookup_projection(task=task, dtype=dtype, size=size, target_id=target_id)
    _require_measured_row(root or REPO)
    return render_compact_text()


def projection_document(root: Path | None = None) -> dict[str, Any]:
    """@brief 投影记录。省略的课和 paper 只留名字，不抄它们的数字。"""
    root = root or REPO
    measured = _require_measured_row(root)
    text = render_compact_text()
    encoded = text.encode("utf-8")
    return {
        "schema_id": "shapeforge.p2v-compact-projection.1",
        "projection_id": PROJECTION_ID,
        "lesson_id": LESSON_ID,
        "overlay_kind": "compact",
        **SCOPE,
        "selector": "exact task dtype size target_id",
        "reg_fresh_us": measured["reg_fresh_us"],
        "reg_fresh_us_text": REG_US,
        "two_stage_floor_us": measured["two_stage_floor_us"],
        "two_stage_floor_us_text": FLOOR_US,
        "confirmation_seed": measured["confirmation_seed"],
        "identity": measured["identity"],
        "search_seed": measured["search_seed"],
        "evidence_class_floor": measured["evidence_class_floor"],
        "not_a_candidate": True,
        "global_candidate_source_added": False,
        "global_would_win": False,
        "publication_allowed": False,
        "speedup_published": False,
        "candidate_tiling_specified": False,
        "omitted_lesson_ids": measured["omitted_lesson_ids"],
        "omitted_size": measured["omitted_size"],
        "omitted_fields": ["reg_speedup_vs_best", "paper", "hard_lower_bound_over_reg"],
        "sources": measured["sources"],
        "overlay_sha256": hashlib.sha256(encoded).hexdigest(),
        "overlay_bytes": len(encoded),
    }


def compact_path(root: Path | None = None) -> Path:
    return (root or REPO) / COMPACT_REL


def read_compact_overlay(root: Path | None = None) -> str:
    """@brief 读冻结文件，并要求它与现场投影逐字相同。"""
    root = root or REPO
    projected = project_compact_overlay(root)
    path = compact_path(root)
    if not path.is_file():
        raise CompactProjectionError("compact overlay file is missing")
    stored = path.read_text(encoding="utf-8")
    if stored != projected:
        raise CompactProjectionError("compact overlay file does not match the projection")
    return stored


def p2u_boundary(root: Path | None = None) -> dict[str, Any]:
    """@brief 读 P2-U 的归档结论。本实验不复制它的批准。"""
    root = root or REPO
    conclusion = _read(root, P2U_CONCLUSION)
    doc = (root / P2U_DOC).read_text(encoding="utf-8")
    if conclusion.get("verdict") != "no_gain" or conclusion.get("adopt_overlay") is not False:
        raise CompactProjectionError("P2-U conclusion is no longer the recorded no_gain")
    if conclusion.get("publication_allowed") is not False or conclusion.get("delta") != P2U_DELTA:
        raise CompactProjectionError("P2-U conclusion delta or publication drifted")
    if "verdict=no_gain" not in doc or "adopt_overlay=false" not in doc or "批准人是 kuafu" not in doc:
        raise CompactProjectionError("P2-U contract no longer records the unpaid-to-reuse approval")
    return {
        "experiment_id": P2U_EXPERIMENT_ID,
        "verdict": "no_gain",
        "adopt_overlay": False,
        "approved_by": P2U_APPROVED_BY,
        "delta": P2U_DELTA,
        "reused": False,
    }


def approval_document() -> dict[str, Any]:
    """@brief 空白批准。填上任何一项也不等于可以执行。"""
    return {
        "schema_id": "shapeforge.p2v-compact-search-approval.1",
        "experiment_id": "p2v-compact-experience-overlay",
        "approved_by": None,
        "delta": None,
        "tolerable_overrun": None,
        "executed": False,
        "execution_record": None,
        "paid_search": "未授权/未执行",
        "inherits_p2u_approval": False,
        "adopt_compact_overlay": False,
        "publication_allowed": False,
        "plan_4_started": False,
    }


def build_compact_plan(root: Path | None = None, *, git_head: str | None = None) -> dict[str, Any]:
    """@brief 新的未批准 dry-run。内层仍是 P2-T 的经验计划，叠加层换成紧凑文件。"""
    from experiments.search_yield_ab import build_experience_plan

    root = root or REPO
    text = read_compact_overlay(root)
    prior = p2u_boundary(root)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    inner = build_experience_plan(
        overlay_sha256=digest,
        overlay_path=COMPACT_REL,
        tasks=(SCOPE["task"],),
        dtype=SCOPE["dtype"],
        size=SCOPE["size"],
        pairs=2,
        budget_usd=3.0,
        max_total_usd=12.0,
        git_head=git_head,
    )
    if inner.get("approved_by") is not None or inner.get("delta") is not None or inner.get("executed") is not False:
        raise CompactProjectionError("the inner experience plan is no longer unapproved")
    full_sha = hashlib.sha256((root / FULL_REL).read_bytes()).hexdigest()
    if full_sha != FULL_OVERLAY_SHA:
        raise CompactProjectionError("the full P2-S overlay bytes changed")
    return {
        "schema_id": "shapeforge.p2v-compact-overlay-plan.1",
        "epoch": "p2v-compact-experience-overlay-1",
        "experiment_id": "p2v-compact-experience-overlay",
        "new_experiment": True,
        "not_plan_4": True,
        "plan_4_started": False,
        "inherits_p2u_approval": False,
        "p2u_experiment_id": prior["experiment_id"],
        "p2u_verdict": prior["verdict"],
        "p2u_adopt_overlay": False,
        "adopt_compact_overlay": False,
        "approved_by": None,
        "delta": None,
        "tolerable_overrun": None,
        "executed": False,
        "execution_record": None,
        "paid_search": "未授权/未执行",
        "publication_allowed": False,
        "npu_simulated_only": True,
        "overlay_kind": "compact",
        "full_overlay_path": FULL_REL,
        "full_overlay_sha256": full_sha,
        "compact_overlay_path": COMPACT_REL,
        "compact_overlay_sha256": digest,
        "experience_plan": inner,
    }


def assert_compact_scope_matches_plan(plan: dict[str, Any]) -> None:
    """@brief 紧凑文件只能进它声明的那一题。完整叠加层保持原样。"""
    if plan.get("factor") != "experience-overlay":
        return
    text = verify_overlay_bytes(str(plan.get("overlay_path") or ""), str(plan.get("overlay_sha256") or "")).decode("utf-8")
    scope = parse_compact_scope(text)
    if scope is None:
        return
    expected = lookup_projection(
        task=scope["task"],
        dtype=scope["dtype"],
        size=scope["size"],
        target_id=scope["target_id"],
    )
    cells = plan.get("cells")
    if not isinstance(cells, list) or not cells:
        raise CompactProjectionError("compact overlay plan has no cells")
    if plan.get("tasks") != [expected["task"]] or plan.get("dtype") != expected["dtype"] or plan.get("size") != expected["size"]:
        raise CompactProjectionError("compact overlay scope does not match the plan task")
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise CompactProjectionError(f"compact overlay plan cell {index} is not an object")
        if cell.get("task") != expected["task"] or cell.get("dtype") != expected["dtype"] or cell.get("size") != expected["size"]:
            raise CompactProjectionError(f"compact overlay scope does not match plan cell {index}")


def assert_compact_execution_allowed(plan: dict[str, Any]) -> None:
    """@brief 紧凑实验没有批准。P2-U 的批准人、δ 和提额都不能让它执行。"""
    if plan.get("factor") != "experience-overlay":
        return
    text = verify_overlay_bytes(str(plan.get("overlay_path") or ""), str(plan.get("overlay_sha256") or "")).decode("utf-8")
    if parse_compact_scope(text) is None:
        return
    raise CompactProjectionError(
        "compact overlay experiment is not approved; P2-U approval does not authorize execution"
    )


def execute_compact_plan(*_args: Any, runner: Any = None, **_kwargs: Any) -> None:
    """@brief 拒绝执行。传入的 runner 不会被调用。"""
    if runner is not None:
        raise CompactProjectionError(
            "compact overlay experiment is not approved; P2-U approval does not authorize execution"
        )
    raise CompactProjectionError(
        "compact overlay experiment is not approved; P2-U approval does not authorize execution"
    )


def measure_render(root: Path | None = None) -> dict[str, Any]:
    """@brief 用同一条过滤函数比较对照、完整处理和紧凑处理。"""
    from taic.experience_overlay import render_treatment_experience
    from taic.experience_prompt import filter_experience_for_prompt

    root = root or REPO
    raw = (root / P2U_BASELINE_REL).read_text(encoding="utf-8")
    if OVERLAY_MARK in raw:
        raise CompactProjectionError("the global experience file already contains the overlay")
    compact = read_compact_overlay(root)
    full = (root / FULL_REL).read_text(encoding="utf-8")
    control = filter_experience_for_prompt(raw)
    compact_rendered = render_treatment_experience(raw, compact)
    full_rendered = render_treatment_experience(raw, full)
    control_bytes = len(control.encode("utf-8"))
    compact_bytes = len(compact_rendered.encode("utf-8"))
    full_bytes = len(full_rendered.encode("utf-8"))
    if control_bytes != CONTROL_BYTES or full_bytes != FULL_TREATMENT_BYTES:
        raise CompactProjectionError("control or full treatment bytes drifted")
    control_sha = hashlib.sha256(control.encode("utf-8")).hexdigest()
    compact_sha = hashlib.sha256(compact_rendered.encode("utf-8")).hexdigest()
    full_sha = hashlib.sha256(full_rendered.encode("utf-8")).hexdigest()
    if control_sha != CONTROL_SHA or full_sha != FULL_TREATMENT_SHA:
        raise CompactProjectionError("control or full treatment sha drifted")
    if compact.count(OVERLAY_MARK) != 1 or compact_rendered.count(OVERLAY_MARK) != 1 or control.count(OVERLAY_MARK) != 0:
        raise CompactProjectionError("overlay copy count drifted")
    if "overlay_kind: compact" not in compact_rendered or "overlay_kind: compact" in full_rendered:
        raise CompactProjectionError("rendered kind marker drifted")
    full_added = full_bytes - control_bytes
    compact_added = compact_bytes - control_bytes
    return {
        "control_bytes": control_bytes,
        "control_sha256": control_sha,
        "compact_treatment_bytes": compact_bytes,
        "compact_treatment_sha256": compact_sha,
        "full_treatment_bytes": full_bytes,
        "full_treatment_sha256": full_sha,
        "compact_overlay_bytes": len(compact.encode("utf-8")),
        "compact_overlay_sha256": hashlib.sha256(compact.encode("utf-8")).hexdigest(),
        "full_overlay_bytes": len(full.encode("utf-8")),
        "full_overlay_sha256": hashlib.sha256(full.encode("utf-8")).hexdigest(),
        "compact_added_bytes": compact_added,
        "full_added_bytes": full_added,
        "added_bytes_removed": full_added - compact_added,
    }


def render_comparison(manifest: dict[str, Any]) -> dict[str, Any]:
    """@brief 从真实 ``prepare_session`` 清单里取出稳定字段。"""
    cells = manifest["cells"]
    controls = [cell for cell in cells if cell["arm"] == "control"]
    treatments = [cell for cell in cells if cell["arm"] == "treatment"]
    if [cell["overlay_copies"] for cell in cells] != [0, 1, 1, 0]:
        raise CompactProjectionError("render copy pattern drifted")
    if len({cell["experience_sha256"] for cell in controls}) != 1:
        raise CompactProjectionError("control render is not stable")
    if len({cell["experience_sha256"] for cell in treatments}) != 1:
        raise CompactProjectionError("treatment render is not stable")
    sidecar = treatments[0]["experience_overlay"]
    if sidecar.get("overlay_path") != COMPACT_REL or sidecar.get("overlay_copies") != 1:
        raise CompactProjectionError("treatment record is not the compact overlay")
    if sidecar.get("authoritative_task_rules") is not False or sidecar.get("not_a_task_contract") is not True:
        raise CompactProjectionError("compact overlay record became a task contract")
    if sidecar.get("rendered_experience_sha256") != treatments[0]["experience_sha256"]:
        raise CompactProjectionError("rendered sha does not match the session file")
    sizes = measure_render()
    if treatments[0]["experience_bytes"] != sizes["compact_treatment_bytes"]:
        raise CompactProjectionError("prepare_session bytes do not match the filter")
    if treatments[0]["experience_sha256"] != sizes["compact_treatment_sha256"]:
        raise CompactProjectionError("prepare_session sha does not match the filter")
    if controls[0]["experience_bytes"] != sizes["control_bytes"] or controls[0]["experience_sha256"] != sizes["control_sha256"]:
        raise CompactProjectionError("control prepare_session drifted")
    return {
        "schema_id": "shapeforge.p2v-compact-render.1",
        "overlay_kind": "compact",
        "produces_candidates": False,
        "calls_model": False,
        "is_search_result": False,
        "paid_search": "未授权/未执行",
        "publication_allowed": False,
        "system_prompt_contains_overlay": False,
        "system_prompt_sha256": manifest["system_prompt_sha256"],
        "overlay_copies": {"control": 0, "treatment": 1},
        "overlay_path": sidecar["overlay_path"],
        "overlay_sha256": sidecar["overlay_sha256"],
        "rendered_experience_sha256": sidecar["rendered_experience_sha256"],
        "rendered_experience_bytes": sidecar["rendered_experience_bytes"],
        "authoritative_task_rules": False,
        "not_a_task_contract": True,
        "installed_in_global_experience": False,
        **sizes,
    }


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """@brief dry-run 或 render-only。``--execute`` 一律返回 2。"""
    import argparse
    import sys

    from experiments.search_yield_ab import render_experience_plan, write_plan

    parser = argparse.ArgumentParser(description="Compact experience overlay dry-run. Execute stays refused.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-spend", action="store_true")
    parser.add_argument("--allow-gpu", action="store_true")
    parser.add_argument("--approved-by", default="")
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--tolerable-overrun", default="")
    args = parser.parse_args(argv)
    modes = [name for name, enabled in (("dry-run", args.dry_run), ("render-only", args.render_only), ("execute", args.execute)) if enabled]
    if len(modes) != 1:
        print("experience_compact: choose one of --dry-run, --render-only, --execute", file=sys.stderr)
        return 2
    if args.execute:
        try:
            execute_compact_plan(
                runner=None,
                approved_by=args.approved_by,
                delta=args.delta,
                tolerable_overrun=args.tolerable_overrun,
                allow_spend=args.allow_spend,
                allow_gpu=args.allow_gpu,
            )
        except CompactProjectionError as exc:
            print(f"experience_compact: {exc}", file=sys.stderr)
            return 2
    try:
        plan = build_compact_plan()
    except (CompactProjectionError, ValueError) as exc:
        print(f"experience_compact: {exc}", file=sys.stderr)
        return 2
    try:
        written = write_plan(Path(args.out), plan)
    except ValueError as exc:
        print(f"experience_compact: {exc}", file=sys.stderr)
        return 2
    if args.render_only:
        try:
            manifest = render_experience_plan(Path(args.out), plan["experience_plan"])
        except (CompactProjectionError, ValueError) as exc:
            print(f"experience_compact: {exc}", file=sys.stderr)
            return 2
        print(dumps({
            "plan": str(written),
            "render_only": True,
            "cells": len(manifest["cells"]),
            "produces_candidates": False,
            "calls_model": False,
            "is_search_result": False,
            "paid_search": "未授权/未执行",
            "executed": False,
        }))
        return 0
    print(dumps({
        "plan": str(written),
        "cells": len(plan["experience_plan"]["cells"]),
        "new_experiment": True,
        "inherits_p2u_approval": False,
        "approved_by": None,
        "delta": None,
        "tolerable_overrun": None,
        "executed": False,
        "paid_search": "未授权/未执行",
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
