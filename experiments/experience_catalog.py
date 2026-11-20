"""Reviewable search input built from closed evidence. Does not edit the live prompt.

The catalog reads archived fresh subjects and real reports. Scalar arms and labelled
probe bounds stay in the overlay as counterexample context: they are not independent
teachable lessons and they are not confirmed speedups. ``kda tail``'s decision
measurement, the ``kda single`` N=1 self-check and the worker-geometry harness fix stay
in the overlay as boundary context: their numbers are not adopted and they are not
independent lessons. Occupied-retry and search-best figures stay out of the overlay.
Nothing here writes ``experience.md``, calls a model, or starts a search.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from taic.experience_prompt import filter_experience_for_prompt

REPO = Path(__file__).resolve().parents[1]
_FIXTURES = "tests/support/golden/shared/acceptance-epochs/fixtures"
_PHRASES = json.loads((REPO / _FIXTURES / "p2r" / "phrases.json").read_text(encoding="utf-8"))
P2M_EPOCH = "tests/support/golden/shared/acceptance-epochs/fixtures/p2m/prompt-epoch.json"
P2M_RENDER = "tests/support/golden/shared/acceptance-epochs/fixtures/p2m/rendered-prompts.json"
EXPERIENCE = "targets/apple-m4-max/experience.md"
OVERLAY_MARK = "这份文字是 P2-S 的冻结叠加层，还没有写入 targets/apple-m4-max/experience.md。"
STALE_SEQ_SCRATCH = "unrunnable today"
CONTEXT_RULE = _PHRASES["context_rule"]
BOUNDARY_IDS = (
    "seq-tail-decision-measurement",
    "seq-single-self-check",
    "p2r-worker-geometry",
)
DOC_SCRATCH = "docs/架构/中间结果复用与Scratch Workspace.md"
DOC_CONV_SMALL = "docs/架构/conv2d-small-mma-headroom.md"
DOC_GEMM = "docs/架构/gemm-silu-scratch-headroom.md"
DOC_BITDELTA = "docs/架构/bitdelta-unpack-headroom.md"
DOC_SPLITK = "docs/架构/gemm-splitk-headroom.md"
DOC_SEQ = "docs/架构/seq-state-handoff-headroom.md"
DOC_WORKER = "docs/架构/bundle-scratch-worker-geometry.md"
DECISION_SIZE_ROWS = ("kda_paper", "kda_small", "kda_tail", "varlen_mixed")
KDA_FRESH_ARMS = ("seq_three_piece", "single_piece_three_stage", "single_piece_single_stage")
KDA_DEVICE_PATH = "device/host_piece_handoff.metal"
KDA_HOST_PATH = "host/tiling.c"
KDA_HOST_GEOMETRY_PATH = "host/execution_geometry.json"
VARLEN_DEVICE_PATH = "device/candidate.metal"


def _read(root: Path, relative: str) -> Any:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _identity_kind(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("kind") or "")
    return ""


def _fresh(subject: dict[str, Any], *, source: str) -> dict[str, Any]:
    """A complete confirmation subject. Search seed 7 and a fresh seed are both required."""
    if subject.get("status") != "confirmed":
        raise ValueError(f"{source} is not a confirmed fresh subject")
    if int(subject.get("search_seed")) != 7:
        raise ValueError(f"{source} did not record search seed 7")
    fresh_seed = int(subject["confirmation_seed"])
    if fresh_seed == 7:
        raise ValueError(f"{source} reused the search seed")
    if _identity_kind(subject.get("identity")) != "match":
        raise ValueError(f"{source} identity is not match")
    report = subject["confirmation_report"]
    median = float(report["benchmark"]["candidate"]["median_us"])
    if not (median > 0):
        raise ValueError(f"{source} has no positive fresh median")
    return {
        "evidence_class": "fresh_subject",
        "status": "confirmed",
        "search_seed": 7,
        "confirmation_seed": fresh_seed,
        "identity": "match",
        "median_us": median,
        "baseline_us": {key: float(value) for key, value in report["baseline_us"].items()},
        "candidate_sha256": subject["candidate_sha256"],
        "source": source,
    }


def _close(left: float, right: float, *, source: str) -> None:
    if abs(left - right) > 1e-6 and abs(left - right) > 1e-9 * max(abs(left), abs(right), 1.0):
        raise ValueError(f"{source} drifted: {left} != {right}")


def _fmt_us(value: float) -> str:
    return f"{value:.2f}" if value >= 100 else f"{value:.4f}"


def _fmt_ratio(value: float) -> str:
    return f"{value:.4f}"


def _fmt_pct(fraction: float) -> str:
    return f"{fraction * 100:.1f}%"


def _lessons(root: Path) -> list[dict[str, Any]]:
    lessons: list[dict[str, Any]] = []
    lessons.append(_conv2d_mma(root))
    lessons.append(_conv2d_scalar(root))
    lessons.append(_conv2d_small(root))
    lessons.extend(_gemm_silu(root))
    lessons.extend(_bitdelta(root))
    lessons.append(_splitk(root))
    lessons.extend(_seq(root))
    lessons.append(_worker_geometry(root))
    return lessons


def _conv2d_mma(root: Path) -> dict[str, Any]:
    relative = f"{_FIXTURES}/p2i/evidence/conv2d-mma-f4-confirmation.json"
    document = _read(root, relative)
    global_u = _fresh(document["subject"], source=relative + "#subject")
    recompute = _fresh(document["recompute_subject"], source=relative + "#recompute_subject")
    if global_u["candidate_sha256"] != document["candidate_sha256"]:
        raise ValueError("conv2d global-U subject does not match the fixture digest")
    if recompute["candidate_sha256"] != document["recompute_sha256"]:
        raise ValueError("conv2d recompute subject does not match its digest")
    exact = recompute["median_us"] / global_u["median_us"]
    rounded = round(recompute["median_us"], 2) / round(global_u["median_us"], 2)
    return {
        "id": "conv2d-paper-shared-u",
        "teach": True,
        "evidence_class": "fresh_subject_pair",
        "claim_kind": "same_algorithm_reuse",
        "publication_allowed": False,
        "task": "conv2d",
        "dtype": "fp16",
        "size": "paper",
        "protocol": "confirm_candidate, search seed 7, fresh seed derived from the public master key, reps 15, identity match",
        "scope": "Apple M4 Max, fp16 paper, this F(4x4) simdgroup MMA consumer only",
        "global_u": global_u,
        "recompute": recompute,
        "exact_ratio_recompute_over_global": exact,
        "rounded_pair_ratio": rounded,
        "source": relative,
    }


def _conv2d_scalar(root: Path) -> dict[str, Any]:
    relative = f"{_FIXTURES}/p2i/evidence/conv2d-ablation.json"
    document = _read(root, relative)
    arms = []
    for arm in document["arms"]:
        if arm["status"] != "confirmed" or int(arm["confirmation_seed"]) == 7:
            raise ValueError(f"scalar arm {arm['label']} is not a fresh confirmation")
        if _identity_kind(arm["identity"]) != "match":
            raise ValueError(f"scalar arm {arm['label']} identity drifted")
        median = float(arm["confirmation"]["median_us"])
        arms.append({"label": arm["label"], "median_us": median, "confirmation_seed": int(arm["confirmation_seed"])})
    if min(arm["median_us"] for arm in arms) <= 10000:
        raise ValueError("scalar ablation is no longer the slow consumer this catalog refuses to teach as a win")
    return {
        "id": "conv2d-paper-scalar-ablation",
        "teach": False,
        "evidence_class": "fresh_subject",
        "claim_kind": "not_a_speedup",
        "publication_allowed": False,
        "task": "conv2d",
        "dtype": "fp16",
        "size": "paper",
        "why_not_taught": "Three fresh subjects exist, and every one is far slower than the library baselines. Global U did not win on that scalar consumer.",
        "arms": arms,
        "source": relative,
    }


def _conv2d_small(root: Path) -> dict[str, Any]:
    fresh_path = f"{_FIXTURES}/p2j/evidence/small-confirmation.json"
    floor_path = f"{_FIXTURES}/p2j/evidence/launch-floor.json"
    stop_path = f"{_FIXTURES}/p2n/evidence/decision.json"
    fresh = _fresh(_read(root, fresh_path), source=fresh_path)
    floor = _read(root, floor_path)
    stop = _read(root, stop_path)
    if floor.get("not_a_candidate") is not True or floor.get("publication_allowed") is not False:
        raise ValueError("conv2d small launch floor is not labelled as a non-candidate")
    two_stage = float(floor["arms"]["two_stage_no_fill"]["wall_us_median"])
    if floor["arms"]["two_stage_no_fill"]["floor_inside_1p10_gate"] is not False:
        raise ValueError("two-stage floor was not outside the 1.10 gate")
    _close(two_stage, 6.128749810159206, source="conv2d small two-stage floor")
    if stop.get("decision") != "stop" or stop.get("publication_allowed") is not False:
        raise ValueError("P2-N stop decision drifted")
    _close(float(stop["earlier_fresh_us"]), fresh["median_us"], source="P2-N earlier fresh")
    best = min(fresh["baseline_us"].values())
    return {
        "id": "conv2d-small-single-stage",
        "teach": True,
        "evidence_class": "fresh_subject_plus_probe_floor",
        "claim_kind": "boundary",
        "publication_allowed": False,
        "task": "conv2d",
        "dtype": "fp16",
        "size": "small",
        "protocol": "single-stage fresh confirmation; two-stage number is an incomplete launch floor",
        "scope": "fp16 small only; do not copy the paper two-stage U structure",
        "fresh": fresh,
        "speedup_vs_best_baseline": best / fresh["median_us"],
        "two_stage_floor": {
            "evidence_class": "probe_floor",
            "not_a_candidate": True,
            "wall_us_median": two_stage,
            "source": floor_path,
        },
        "weight_cache_stop": {
            "evidence_class": "measured_candidate",
            "claim_kind": "not_a_speedup",
            "candidate_us": float(stop["candidate_us"]),
            "slowdown_vs_earlier_pct": float(stop["slowdown_vs_earlier_pct"]),
            "source": stop_path,
        },
        "source": fresh_path,
    }


def _gemm_silu(root: Path) -> list[dict[str, Any]]:
    decision_path = f"{_FIXTURES}/p2o/evidence/decision.json"
    probe_path = f"{_FIXTURES}/p2o/evidence/scratch-probe.json"
    occupied_path = f"{_FIXTURES}/p2o/evidence/reg-paper-confirmation-occupied.json"
    decision = _read(root, decision_path)
    probe = _read(root, probe_path)
    if decision.get("probe_not_a_candidate") is not True or probe.get("not_a_candidate") is not True:
        raise ValueError("gemm_silu probe is not labelled as a non-candidate")
    if decision["implementation"]["global_candidate_source_added"] is not False:
        raise ValueError("a GLOBAL gemm_silu candidate was added")
    sizes = {}
    for size in ("small", "paper"):
        row = decision["sizes"][size]
        fresh_path = f"{_FIXTURES}/p2o/evidence/reg-{size}-confirmation.json"
        fresh = _fresh(_read(root, fresh_path), source=fresh_path)
        _close(fresh["median_us"], float(row["reg_candidate_us"]), source=f"gemm_silu {size}")
        _close(fresh["confirmation_seed"], int(row["reg_confirmation_seed"]), source=f"gemm_silu {size} seed")
        if row["global_would_win"] is not False:
            raise ValueError(f"gemm_silu {size} no longer rejects GLOBAL")
        sizes[size] = {
            "fresh": fresh,
            "hard_lower_bound_us": float(row["hard_lower_bound_us"]),
            "hard_lower_bound_over_reg": float(row["hard_lower_bound_over_reg"]),
            "two_stage_floor_us": float(row["two_stage_floor_us"]),
            "lower_bound_evidence_class": "probe_floor" if size == "small" else "traffic_bound",
            "reg_speedup_vs_best": float(row["reg_speedup_vs_best"]),
        }
    occupied = _read(root, occupied_path)
    return [
        {
            "id": "gemm-silu-reg",
            "teach": True,
            "evidence_class": "fresh_subject",
            "claim_kind": "keep_register_epilogue",
            "publication_allowed": False,
            "task": "gemm_silu",
            "dtype": "fp16",
            "sizes": sizes,
            "global_candidate_source_added": False,
            "probe_source": probe_path,
            "source": decision_path,
            "scope": "fp16 small and paper; the activation is per-element and has no second consumer",
        },
        {
            "id": "gemm-silu-occupied-retry",
            "teach": False,
            "evidence_class": "occupied_retry",
            "claim_kind": "excluded",
            "publication_allowed": False,
            "candidate_us": float(occupied["observed"]["candidate_us"]),
            "reason": occupied["reason"],
            "source": occupied_path,
        },
    ]


def _bitdelta(root: Path) -> list[dict[str, Any]]:
    decision_path = f"{_FIXTURES}/p2p/evidence/decision.json"
    probe_path = f"{_FIXTURES}/p2p/evidence/unpack-probe.json"
    decision = _read(root, decision_path)
    probe = _read(root, probe_path)
    if probe.get("not_a_candidate") is not True or decision.get("shared_candidate_source_added", decision["implementation"]["shared_candidate_source_added"]) is not False:
        raise ValueError("bitdelta SHARED was implemented or the probe lost its label")
    sizes = {}
    for size in ("small", "paper"):
        row = decision["sizes"][size]
        fresh_path = f"{_FIXTURES}/p2p/evidence/reg-{size}-confirmation.json"
        fresh = _fresh(_read(root, fresh_path), source=fresh_path)
        _close(fresh["median_us"], float(row["reg_candidate_us"]), source=f"bitdelta {size}")
        sizes[size] = {
            "fresh": fresh,
            "best_legal_shared_ratio_over_reg": float(row["best_legal_shared_ratio_over_reg"]),
            "ratio_evidence_class": "paired_probe",
            "reg_over_roofline": float(row["reg_over_roofline"]),
        }
    # The paper recoverable fraction is the decision's own field when present; otherwise derive
    # nothing. The paired ratio is the taught comparison and is already a probe, not a speedup.
    paper = decision["sizes"]["paper"]
    recoverable = paper.get("recoverable_decode_headroom_fraction")
    report_path = f"{_FIXTURES}/p2p/P.md"
    review_path = f"{_FIXTURES}/p2p/review.json"
    headroom = (root / DOC_BITDELTA).read_text(encoding="utf-8")
    report = (root / report_path).read_text(encoding="utf-8")
    review = (root / review_path).read_text(encoding="utf-8")
    if "不能读成精确倍数或两者之间的排名" not in headroom:
        raise ValueError("bitdelta headroom no longer refuses an exact small ranking")
    if "不声明两者之间的排序" not in report:
        raise ValueError("bitdelta report no longer refuses a small ordering")
    if "only the order-of-magnitude gap is claimable" not in review:
        raise ValueError("bitdelta review no longer records the small ranking as unclaimable")
    sizes["small"]["exact_ratio_taught"] = False
    sizes["small"]["ranking_taught"] = False
    sizes["paper"]["exact_ratio_taught"] = True
    sizes["paper"]["ranking_taught"] = True
    return [
        {
            "id": "bitdelta-reg",
            "teach": True,
            "evidence_class": "fresh_subject_plus_paired_probe",
            "claim_kind": "keep_on_chip_unpack",
            "publication_allowed": False,
            "task": "bitdelta_gemm",
            "dtype": "fp16",
            "sizes": sizes,
            "recoverable_decode_headroom_fraction": None if recoverable is None else float(recoverable),
            "shared_candidate_source_added": False,
            "small_teaching": "about_one_order_of_magnitude",
            "ranking_boundary_sources": [DOC_BITDELTA, report_path, review_path],
            "probe_source": probe_path,
            "source": decision_path,
            "scope": "fp16 small and paper; paper's paired median may be cited, small is only about one order of magnitude",
        },
        {
            "id": "bitdelta-small-search-best",
            "teach": False,
            "evidence_class": "search_best",
            "claim_kind": "excluded",
            "publication_allowed": False,
            "note": (root / _FIXTURES / "p2p" / "search-note.txt").read_text(encoding="utf-8").rstrip("\n"),
            "source": "docs/架构/bitdelta-unpack-headroom.md",
        },
    ]


def _splitk(root: Path) -> dict[str, Any]:
    analysis_path = f"{_FIXTURES}/p2l/evidence/analysis.json"
    floor_path = f"{_FIXTURES}/p2l/evidence/launch-floor.json"
    analysis = _read(root, analysis_path)
    floor = _read(root, floor_path)
    if analysis.get("implementation_started") is not False or analysis.get("candidate_source_added") is not False:
        raise ValueError("Split-K implementation was started")
    if floor.get("not_a_candidate") is not True:
        raise ValueError("Split-K floor is not labelled as a non-candidate")
    two_stage = float(floor["arms"]["two_stage_no_fill"]["wall_us_median"])
    _close(two_stage, float(analysis["small"]["two_stage_no_fill_floor_us"]), source="split-k floor")
    paper = analysis["paper"]
    numerator = float(paper["split2_partial_write_read_us_at_peak"])
    denominator = float(paper["existing_candidate_us"])
    percent = float(paper["split2_traffic_percent_of_candidate"])
    _close(percent, numerator / denominator * 100.0, source="split-k peak-bandwidth time equivalent")
    return {
        "id": "gemm-splitk-stop",
        "teach": True,
        "evidence_class": "probe_floor_and_archived_anchor",
        "claim_kind": "do_not_implement",
        "publication_allowed": False,
        "task": "gemm",
        "dtype": "fp16",
        "small_two_stage_floor_us": two_stage,
        "small_floor_over_existing": float(analysis["small"]["floor_over_existing_candidate"]),
        "paper_output_threadgroups": int(paper["output_threadgroups"]),
        "paper_split2_traffic_percent": percent,
        "paper_split2_numerator_us": numerator,
        "paper_split2_denominator_us": denominator,
        "paper_split2_numerator_field": "split2_partial_write_read_us_at_peak",
        "paper_split2_denominator_field": "existing_candidate_us",
        "paper_split2_unit": "peak_bandwidth_time_equivalent",
        "paper_split2_is_byte_ratio": False,
        "existing_candidate_evidence_class": "archived_anchor",
        "source": analysis_path,
        "probe_source": floor_path,
        "scope": "fp16 small and paper GEMM; the floor is not a candidate time",
    }


def _indexed_build_files(report: dict[str, Any]) -> dict[str, str]:
    """Path to sha256 inside one confirmation build. One path must not carry two hashes."""
    found: dict[str, str] = {}

    def take(path: Any, digest: Any) -> None:
        if not isinstance(path, str) or not isinstance(digest, str):
            return
        previous = found.get(path)
        if previous is not None and previous != digest:
            raise ValueError(f"{path} has two build hashes")
        found[path] = digest

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value and "sha256" in value:
                take(value["path"], value["sha256"])
            if "entry_path" in value and "entry_sha256" in value:
                take(value["entry_path"], value["entry_sha256"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(report["build"])
    return found


def _one_hash(mapping: dict[str, str], keys: list[str], label: str) -> str:
    values = {mapping[key] for key in keys}
    if len(values) != 1:
        raise ValueError(f"{label} is not one hash: {sorted(values)}")
    return values.pop()


def _varlen_plan_diff() -> dict[str, str]:
    """The packed-varlen arm diff. The only text change is the plan kernel's pieces assignment."""
    from tests.support.seq_handoff import VARLEN_PIECE_LINE, varlen_source

    three = varlen_source("pieces_three")
    one = varlen_source("pieces_one")
    removed = VARLEN_PIECE_LINE
    added = "    uint pieces = 1u;"
    if three.replace(removed, added, 1) != one or three.count(removed) != 1 or one.count(added) != 1:
        raise ValueError("varlen arms differ by more than the plan-stage pieces assignment")
    plan_at = three.find("kernel void varlen_plan")
    line_at = three.find(removed)
    next_kernel = three.find("\nkernel void", plan_at + 1)
    if plan_at < 0 or line_at < 0 or not plan_at < line_at < (next_kernel if next_kernel > 0 else len(three)):
        raise ValueError("the pieces assignment is not inside kernel void varlen_plan")
    return {
        "kernel": "varlen_plan",
        "removed_line": removed,
        "added_line": added,
        "pieces_three_sha256": hashlib.sha256(three.encode("utf-8")).hexdigest(),
        "pieces_one_sha256": hashlib.sha256(one.encode("utf-8")).hexdigest(),
    }


def _seq_controls(confirmations: dict[str, Any]) -> dict[str, Any]:
    """Split KDA's unchanged device file from varlen's changed plan kernel.

    The hashes come from each fresh subject's build record. ``varlen_source`` is then required
    to reproduce the two varlen device hashes, so a rewritten sentence cannot outrun the bytes.
    """
    subjects = confirmations["subjects"]
    kda_keys = [key for key in subjects if key.startswith("kda:")]
    if len(kda_keys) != 6:
        raise ValueError(f"expected six KDA fresh subjects, found {kda_keys}")
    kda_files = {key: _indexed_build_files(subjects[key]["confirmation_report"]) for key in kda_keys}
    device = _one_hash(
        {key: files[KDA_DEVICE_PATH] for key, files in kda_files.items()},
        kda_keys,
        KDA_DEVICE_PATH,
    )
    three_piece = [key for key in kda_keys if key.endswith("/seq_three_piece")]
    single_piece = [key for key in kda_keys if "/single_piece_" in key]
    three_stage = [key for key in kda_keys if not key.endswith("/single_piece_single_stage")]
    one_stage = [key for key in kda_keys if key.endswith("/single_piece_single_stage")]
    host_by_arm = {
        "seq_three_piece": _one_hash(
            {key: files[KDA_HOST_PATH] for key, files in kda_files.items()}, three_piece, "three-piece host"
        ),
        "single_piece": _one_hash(
            {key: files[KDA_HOST_PATH] for key, files in kda_files.items()}, single_piece, "single-piece host"
        ),
    }
    geometry_by_stages = {
        "3": _one_hash(
            {key: files[KDA_HOST_GEOMETRY_PATH] for key, files in kda_files.items()},
            three_stage,
            "three-stage host geometry",
        ),
        "1": _one_hash(
            {key: files[KDA_HOST_GEOMETRY_PATH] for key, files in kda_files.items()},
            one_stage,
            "single-stage host geometry",
        ),
    }
    if host_by_arm["seq_three_piece"] == host_by_arm["single_piece"]:
        raise ValueError("KDA host program did not change with the piece count")
    if geometry_by_stages["3"] == geometry_by_stages["1"]:
        raise ValueError("KDA host geometry did not change with the stage count")
    diff = _varlen_plan_diff()
    varlen_keys = {
        "pieces_three": "varlen_kda:mixed/pieces_three",
        "pieces_one": "varlen_kda:mixed/pieces_one",
    }
    varlen_files = {
        arm: _indexed_build_files(subjects[key]["confirmation_report"]) for arm, key in varlen_keys.items()
    }
    for arm, files in varlen_files.items():
        if KDA_HOST_PATH in files or "host/tiling.c" in files:
            raise ValueError(f"varlen {arm} has a host program")
        artifact = files[VARLEN_DEVICE_PATH]
        source_sha = diff[f"{arm}_sha256"]
        if artifact != source_sha:
            raise ValueError(f"varlen {arm} artifact {artifact} does not match varlen_source {source_sha}")
    if diff["pieces_three_sha256"] == diff["pieces_one_sha256"]:
        raise ValueError("varlen device bytes were collapsed into one hash")
    return {
        "kda": {
            "device_path": KDA_DEVICE_PATH,
            "device_sha256": device,
            "device_bytes_identical": True,
            "subject_count": len(kda_keys),
            "host_program_path": KDA_HOST_PATH,
            "host_program_sha256_by_arm": host_by_arm,
            "host_geometry_path": KDA_HOST_GEOMETRY_PATH,
            "host_geometry_sha256_by_stage_count": geometry_by_stages,
        },
        "varlen": {
            "device_path": VARLEN_DEVICE_PATH,
            "device_bytes_identical": False,
            "host_program_present": False,
            "pieces_three_sha256": diff["pieces_three_sha256"],
            "pieces_one_sha256": diff["pieces_one_sha256"],
            "diff": {
                "kernel": diff["kernel"],
                "removed_line": diff["removed_line"],
                "added_line": diff["added_line"],
                "only_text_change": True,
            },
        },
    }


def require_seq_size_rows(sizes: dict[str, Any]) -> None:
    """Decision rows are an exact set. A missing size is an error, not a skip."""
    found = set(sizes)
    expected = set(DECISION_SIZE_ROWS)
    if "kda_single" in found:
        raise ValueError("kda_single is a self-check, not a decision size")
    if found != expected:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise ValueError(f"SEQ decision sizes drifted; missing={missing or '-'} extra={extra or '-'}")


def _check_decision_algebra(row: dict[str, Any], name: str) -> None:
    """A−B is the decision's handoff field. B−C is its stage floor. A−C is the total."""
    arm_a = float(row["seq_three_piece_us"])
    arm_b = float(row["single_piece_three_stage_us"])
    arm_c = float(row["single_stage_us"])
    _close(float(row["handoff_us"]), arm_a - arm_b, source=f"{name} handoff")
    _close(float(row["stage_floor_us"]), arm_b - arm_c, source=f"{name} stage floor")
    _close(float(row["seq_overhead_us"]), arm_a - arm_c, source=f"{name} total")
    _close(
        float(row["seq_overhead_fraction_of_single_stage"]),
        (arm_a - arm_c) / arm_c,
        source=f"{name} fraction",
    )


def _apart(fresh: float, measured: float, source: str) -> None:
    if abs(fresh - measured) <= 0.01:
        raise ValueError(f"{source} fresh wall collapsed onto the decision measurement")


def _kda_fresh_row(
    subjects: dict[str, Any],
    confirm_path: str,
    decision_row: dict[str, Any],
    size: str,
) -> dict[str, Any]:
    """Read A, B and C. A−C changes both the piece count and the stage count."""
    arms = {}
    for arm in KDA_FRESH_ARMS:
        key = f"kda:{size}/{arm}"
        arms[arm] = _fresh(subjects[key], source=f"{confirm_path}#{key}")
    seeds = [arms[arm]["confirmation_seed"] for arm in KDA_FRESH_ARMS]
    if len(set(seeds)) != 3:
        raise ValueError(f"kda {size} confirmation seeds are not distinct: {seeds}")
    fresh_a = arms["seq_three_piece"]["median_us"]
    fresh_b = arms["single_piece_three_stage"]["median_us"]
    fresh_c = arms["single_piece_single_stage"]["median_us"]
    measured_a = float(decision_row["seq_three_piece_us"])
    measured_b = float(decision_row["single_piece_three_stage_us"])
    measured_c = float(decision_row["single_stage_us"])
    _apart(fresh_a, measured_a, f"kda {size} A")
    _apart(fresh_b, measured_b, f"kda {size} B")
    _apart(fresh_c, measured_c, f"kda {size} C")
    total = (fresh_a - fresh_c) / fresh_c
    if total <= 0:
        raise ValueError(f"kda {size}: fresh A is not slower than fresh C")
    return {
        "name": f"kda_{size}",
        "task": "kda",
        "size": size,
        "fresh_a_us": fresh_a,
        "fresh_b_us": fresh_b,
        "fresh_c_us": fresh_c,
        "fresh_seq_us": fresh_a,
        "fresh_single_us": fresh_c,
        "fresh_a_seed": arms["seq_three_piece"]["confirmation_seed"],
        "fresh_b_seed": arms["single_piece_three_stage"]["confirmation_seed"],
        "fresh_c_seed": arms["single_piece_single_stage"]["confirmation_seed"],
        "fresh_ab_us": fresh_a - fresh_b,
        "fresh_bc_us": fresh_b - fresh_c,
        "fresh_ac_total_overhead_fraction": total,
        "fresh_overhead_fraction": total,
        "ac_is_isolated_handoff": False,
        "same_input_paired_run": False,
        "measurement_seq_us": measured_a,
        "measurement_b_us": measured_b,
        "measurement_single_us": measured_c,
        "measurement_handoff_us": float(decision_row["handoff_us"]),
        "measurement_stage_floor_us": float(decision_row["stage_floor_us"]),
        "measurement_overhead_fraction": float(decision_row["seq_overhead_fraction_of_single_stage"]),
        "stages_differ": bool(decision_row["stages_differ_between_arms"]),
        "samples_are_not_the_same_run": True,
    }


def _seq(root: Path) -> list[dict[str, Any]]:
    decision_path = f"{_FIXTURES}/p2q/evidence/decision.json"
    confirm_path = f"{_FIXTURES}/p2q/evidence/confirmations.json"
    measurements_path = f"{_FIXTURES}/p2q/evidence/seq-handoff-measurements.json"
    decision = _read(root, decision_path)
    confirmations = _read(root, confirm_path)
    measurements = _read(root, measurements_path)
    require_seq_size_rows(decision["sizes"])
    for name in DECISION_SIZE_ROWS:
        _check_decision_algebra(decision["sizes"][name], name)
    subjects = confirmations["subjects"]
    fresh_keys = [f"kda:{size}/{arm}" for size in ("small", "paper") for arm in KDA_FRESH_ARMS]
    fresh_keys.extend(("varlen_kda:mixed/pieces_three", "varlen_kda:mixed/pieces_one"))
    missing = [key for key in fresh_keys if key not in subjects]
    if missing or len(fresh_keys) != 8:
        raise ValueError(f"the eight fresh subjects drifted: missing={missing}")
    taught = [
        _kda_fresh_row(subjects, confirm_path, decision["sizes"]["kda_small"], "small"),
        _kda_fresh_row(subjects, confirm_path, decision["sizes"]["kda_paper"], "paper"),
    ]
    pieces_three = _fresh(
        subjects["varlen_kda:mixed/pieces_three"],
        source=f"{confirm_path}#varlen_kda:mixed/pieces_three",
    )
    pieces_one = _fresh(
        subjects["varlen_kda:mixed/pieces_one"],
        source=f"{confirm_path}#varlen_kda:mixed/pieces_one",
    )
    if pieces_three["confirmation_seed"] == pieces_one["confirmation_seed"]:
        raise ValueError("varlen confirmation seeds are not distinct")
    varlen_decision = decision["sizes"]["varlen_mixed"]
    if varlen_decision["stages_differ_between_arms"] is not False:
        raise ValueError("varlen stages now differ; do not describe them as the same dispatch")
    varlen_total = (pieces_three["median_us"] - pieces_one["median_us"]) / pieces_one["median_us"]
    if varlen_total <= 0:
        raise ValueError("varlen fresh pieces_three is not slower than pieces_one")
    _apart(pieces_three["median_us"], float(varlen_decision["seq_three_piece_us"]), "varlen A")
    _apart(pieces_one["median_us"], float(varlen_decision["single_stage_us"]), "varlen C")
    taught.append({
        "name": "varlen_mixed",
        "task": "varlen_kda",
        "size": "mixed",
        "fresh_seq_us": pieces_three["median_us"],
        "fresh_single_us": pieces_one["median_us"],
        "fresh_slow_seed": pieces_three["confirmation_seed"],
        "fresh_fast_seed": pieces_one["confirmation_seed"],
        "fresh_overhead_fraction": varlen_total,
        "same_input_paired_run": False,
        "measurement_seq_us": float(varlen_decision["seq_three_piece_us"]),
        "measurement_single_us": float(varlen_decision["single_stage_us"]),
        "measurement_overhead_fraction": float(varlen_decision["seq_overhead_fraction_of_single_stage"]),
        "measurement_handoff_us": float(varlen_decision["handoff_us"]),
        "stages_differ": False,
        "samples_are_not_the_same_run": True,
    })
    scratch = decision["scratch_attempt"]["attempts"]["two_buffer_pingpong"]
    if scratch["scratch_bytes"] != scratch["workspace_bytes_at_the_same_capacity"]:
        raise ValueError("SEQ ping-pong no longer matches the workspace capacity")
    tail = decision["sizes"]["kda_tail"]
    algebra = decision["state_algebra"]["kda:single"]
    if algebra["pieces_per_chain"] != [1] or int(algebra["boundaries_used"]) != 0:
        raise ValueError("kda single is no longer the one-piece self-check")
    single_arms = measurements["measurements"]["kda:single"]
    single_medians = {}
    for arm in KDA_FRESH_ARMS:
        if arm not in single_arms:
            raise ValueError(f"kda single self-check is missing {arm}")
        single_medians[arm] = float(single_arms[arm]["median_us"])
    single_ab = (
        single_medians["seq_three_piece"] - single_medians["single_piece_three_stage"]
    ) / single_medians["single_piece_three_stage"]
    q_text = (root / f"{_FIXTURES}/p2q/Q.md").read_text(encoding="utf-8")
    headroom = (root / DOC_SEQ).read_text(encoding="utf-8")
    if "N=1 只有 1 个 piece" not in q_text:
        raise ValueError("P2-Q no longer calls kda single an N=1 self-check")
    if "L2 `capacity` 档没有纳入" not in q_text or "本机 Apple M4 Max" not in q_text:
        raise ValueError("P2-Q remaining limits no longer name the SEQ scope")
    if "不能外推成" not in headroom or "只有 L1 的 `mixed`" not in headroom:
        raise ValueError("SEQ headroom no longer states the paper and L2 limits")
    rationale = " ".join(decision["route_rationale"])
    if STALE_SEQ_SCRATCH not in rationale:
        raise ValueError("the stale P2-Q blocker sentence disappeared; do not teach a rewritten history")
    controls = _seq_controls(confirmations)
    return [
        {
            "id": "seq-serial-handoff",
            "teach": True,
            "evidence_class": "fresh_subject",
            "claim_kind": "do_not_split_for_concurrency",
            "publication_allowed": False,
            "protocol": (
                "KDA keeps device/host_piece_handoff.metal identical and changes host/tiling.c "
                "plus host/execution_geometry.json. varlen_kda has no host program; "
                "device/candidate.metal differs by the varlen_plan pieces assignment. "
                "All eight confirmation subjects pass through _fresh: kda small and paper each have "
                "A seq_three_piece, B single_piece_three_stage and C single_piece_single_stage, "
                "and varlen mixed has two arms. A−C is cross-confirmation total overhead, "
                "not an isolated handoff cost and not a same-input paired run. "
                "A−B and B−C are recorded separately and are also cross-confirmation."
            ),
            "controls": controls,
            "confirmed_rows": taught,
            "fresh_subject_count": 8,
            "ping_pong_bytes_equal": True,
            "stale_text_not_taught": STALE_SEQ_SCRATCH,
            "source": confirm_path,
            "decision_source": decision_path,
            "diff_source": "tests/support/seq_handoff.py",
            "scope": (
                "Apple M4 Max, fp16, kda small and paper, varlen_kda L1 mixed only; "
                "L2 capacity is outside this evidence. paper's A−C percentage does not mean SEQ is cheap"
            ),
        },
        {
            "id": "seq-tail-decision-measurement",
            "teach": False,
            "evidence_class": "decision_measurement",
            "claim_kind": "boundary_not_an_independent_lesson",
            "publication_allowed": False,
            "numbers_adopted": False,
            "independent_lesson": False,
            "name": "kda_tail",
            "in_decision_sizes": True,
            "in_fresh_confirmation_set": False,
            "seq_overhead_fraction_of_single_stage": float(tail["seq_overhead_fraction_of_single_stage"]),
            "handoff_us": float(tail["handoff_us"]),
            "stage_floor_us": float(tail["stage_floor_us"]),
            "why_not_an_independent_lesson": (
                "kda_tail is a decision-measurement row. It is not one of the eight fresh confirmation subjects. "
                "The number is recorded and is not adopted."
            ),
            "source": decision_path,
        },
        {
            "id": "seq-single-self-check",
            "teach": False,
            "evidence_class": "raw_self_check",
            "claim_kind": "boundary_not_an_independent_lesson",
            "publication_allowed": False,
            "numbers_adopted": False,
            "independent_lesson": False,
            "name": "kda_single",
            "in_decision_sizes": False,
            "pieces_per_chain": [1],
            "boundaries_used": 0,
            "medians_us": single_medians,
            "ab_fraction": single_ab,
            "why_not_an_independent_lesson": (
                "kda single is absent from decision.sizes. state_algebra and the N=1 measurement "
                "are a raw self-check, not a decision row, and the number is not adopted."
            ),
            "source": measurements_path,
            "algebra_source": decision_path,
        },
    ]


def _worker_geometry(root: Path) -> dict[str, Any]:
    relative = f"{_FIXTURES}/p2r/evidence/identity-delta.json"
    delta = _read(root, relative)
    if delta.get("evaluation_changed") is not True or delta.get("evaluation_device_verdict") is not False:
        raise ValueError("identity delta is no longer a non-device comparison")
    return {
        "id": "p2r-worker-geometry",
        "teach": False,
        "evidence_class": "harness_fix",
        "claim_kind": "not_a_search_lesson",
        "publication_allowed": False,
        "evaluation_changed": True,
        "evaluation_device_verdict": False,
        "why_not_taught": "Boundary context only. The worker now keeps non-empty scratch in the resolved geometry. The number is not adopted and this is not an independent lesson.",
        "source": relative,
    }


def _experiment(lessons: list[dict[str, Any]]) -> dict[str, Any]:
    gemm = next(item for item in lessons if item["id"] == "gemm-silu-reg")
    small = gemm["sizes"]["small"]
    paper_seq = next(row for row in next(item for item in lessons if item["id"] == "seq-serial-handoff")["confirmed_rows"] if row["name"] == "kda_paper")
    conv_small = next(item for item in lessons if item["id"] == "conv2d-small-single-stage")
    return {
        "task": "gemm_silu",
        "dtype": "fp16",
        "size": "small",
        "why": "The fresh REG kernel is faster than the best registered baseline, the eval is a few microseconds, and the current prompt already teaches ScratchSpec syntax without this measured refusal.",
        "reg_speedup_vs_best": small["reg_speedup_vs_best"],
        "reg_fresh_us": small["fresh"]["median_us"],
        "two_stage_floor_us": small["two_stage_floor_us"],
        "floor_evidence_class": "probe_floor",
        "rejected": [
            {
                "task": "conv2d",
                "size": "paper",
                "why": "The positive SHARED result is real, but one confirmation is hundreds of microseconds to tens of milliseconds.",
            },
            {
                "task": "conv2d",
                "size": "small",
                "why": "The fresh single-stage kernel is not faster than the best baseline, and P2-N stopped further micro-optimization.",
                "speedup_vs_best_baseline": conv_small["speedup_vs_best_baseline"],
            },
            {
                "task": "kda",
                "size": "paper",
                "why": "The fresh single-stage wall is about 187 ms, so a paired search is not the cheap arm.",
                "single_stage_us": paper_seq["fresh_single_us"],
            },
            {
                "task": "bitdelta_gemm",
                "size": "small",
                "why": "The baseline gap is larger, but a short search is less likely to rediscover the packed unpack than a fused GEMM epilogue.",
            },
        ],
    }


def build_catalog(root: Path | None = None) -> tuple[dict[str, Any], str]:
    """Return the catalog document and the overlay markdown rendered from it."""
    root = root or REPO
    lessons = _lessons(root)
    for item in lessons:
        if item["teach"]:
            item["overlay_role"] = "teachable"
        elif item["id"] == "conv2d-paper-scalar-ablation":
            item["overlay_role"] = "counterexample_context"
        elif item["id"] in BOUNDARY_IDS:
            item["overlay_role"] = "boundary_context"
        else:
            item["overlay_role"] = "catalog_only"
    catalog = {
        "schema_id": "shapeforge.p2s-experience-catalog.1",
        "epoch": "p2-s-experience-search-prep-1",
        "publication_allowed": False,
        "paid_search": "未授权/未执行",
        "overlay_context_rule": CONTEXT_RULE,
        "teachable_ids": [item["id"] for item in lessons if item["teach"]],
        "experiment": _experiment(lessons),
        "lessons": lessons,
    }
    overlay = render_overlay(catalog)
    catalog["overlay_sha256"] = hashlib.sha256(overlay.encode("utf-8")).hexdigest()
    catalog["overlay_bytes"] = len(overlay.encode("utf-8"))
    return catalog, overlay


def render_overlay(catalog: dict[str, Any]) -> str:
    """Treatment text. ``catalog_only`` lessons are not rendered.

    Counterexample context is rendered inside the conv2d section. Boundary-context
    lessons are rendered inside the SEQ section and are not independent lessons.
    """
    by_id = {item["id"]: item for item in catalog["lessons"]}
    parts = [
        _conv2d_section(by_id["conv2d-paper-shared-u"], by_id["conv2d-paper-scalar-ablation"]),
        _small_section(by_id["conv2d-small-single-stage"]),
        _gemm_section(by_id["gemm-silu-reg"]),
        _bitdelta_section(by_id["bitdelta-reg"]),
        _splitk_section(by_id["gemm-splitk-stop"]),
        _seq_section(
            by_id["seq-serial-handoff"],
            by_id["seq-tail-decision-measurement"],
            by_id["seq-single-self-check"],
            by_id["p2r-worker-geometry"],
        ),
    ]
    return "\n".join(parts).rstrip() + "\n"


def _source_line(*paths: str) -> str:
    if not any(path.startswith(("docs/", "runs/", "tcenv/", "taic/")) for path in paths):
        raise ValueError(f"a verified section needs a docs, runs, tcenv or taic source: {paths}")
    return "来源: " + "、".join(f"`{path}`" for path in paths)


def _verified(title: str, keywords: str, observation: list[str], measured: str, sources: tuple[str, ...], boundary: str) -> str:
    return "\n".join([
        f"## {title}",
        "",
        f"Keywords: {keywords}",
        "Confidence: verified",
        *observation,
        _source_line(*sources),
        f"本机实测: {measured}",
        f"边界: {boundary}",
        "",
    ])


def _conv2d_section(lesson: dict[str, Any], scalar: dict[str, Any]) -> str:
    slow = ", ".join(f"{arm['label']} {_fmt_us(arm['median_us'])} µs" for arm in scalar["arms"])
    global_us = _fmt_us(lesson["global_u"]["median_us"])
    recompute_us = _fmt_us(lesson["recompute"]["median_us"])
    ratio = _fmt_ratio(lesson["exact_ratio_recompute_over_global"])
    return _verified(
        "conv2d paper：全局 U 只在这套 MMA 上划算",
        "conv2d scratch shared winograd paper mma",
        [
            f"观察: {OVERLAY_MARK} {CONTEXT_RULE} fp16/paper 上，同一套 F(4×4) simdgroup MMA，全局 U 的 fresh 墙钟是 {global_us} µs，每个 threadgroup 重算 U 是 {recompute_us} µs。重算除以全局 U 等于 {ratio} 倍。两份都是 fresh subject（confirmed，identity match，confirmation seed 不是搜索种子 7）。这是同算法复用，不是发布的加速比。",
            f"同一形状上更早的逐线程标量三臂也是 fresh subject，但都远慢于库基线：{slow}。那三臂不能当成「scratch 一定更快」。它们留在这里只作反例上下文，不是独立 teachable lesson，也不是 confirmed speedup。",
        ],
        f"Apple M4 Max，fp16 paper，全局 U {global_us} µs，重算 {recompute_us} µs，比 {ratio}。标量三臂是反例，不是独立经验。",
        (lesson["source"], scalar["source"], DOC_SCRATCH),
        "只覆盖 Apple M4 Max 的 fp16 paper 和这份 MMA。不要把 producer 的十几微秒说成整个 kernel 的时间。标量三臂的数字不作为独立经验。publication_allowed=false。",
    )


def _small_section(lesson: dict[str, Any]) -> str:
    fresh = lesson["fresh"]
    floor = lesson["two_stage_floor"]
    stop = lesson["weight_cache_stop"]
    fresh_us = _fmt_us(fresh["median_us"])
    floor_us = _fmt_us(floor["wall_us_median"])
    return _verified(
        "conv2d small：单阶段重算，不要照搬 paper 的两阶段",
        "conv2d small single-stage recompute floor",
        [
            f"观察: fp16/small 的单阶段 fresh 墙钟是 {fresh_us} µs，相对最快基线的比值是 {_fmt_ratio(lesson['speedup_vs_best_baseline'])}，没有快过最快基线。两阶段、无 fill 的启动下限是 {floor_us} µs，而且 `not_a_candidate=true`；这是 probe 下界，不是 candidate 时间，也不是 confirmed speedup。它已经落在 1.10 门外面。",
            f"随后把权重缓进 lane 私有数组的候选是正确的，但墙钟 {_fmt_us(stop['candidate_us'])} µs，比这份 fresh 慢 {stop['slowdown_vs_earlier_pct']:.1f}%。P2-N 因此停止继续做 small 的微优化。这个百分比是变慢，不是加速比。",
        ],
        f"Apple M4 Max，fp16 small，单阶段 fresh {fresh_us} µs，两阶段下限 {floor_us} µs，not_a_candidate=true。",
        (lesson["source"], floor["source"], stop["source"], DOC_CONV_SMALL),
        "只覆盖登记的 fp16 small。不要为了使用 ScratchSpec 再写一个两阶段 small。publication_allowed=false。",
    )


def _gemm_section(lesson: dict[str, Any]) -> str:
    small = lesson["sizes"]["small"]
    paper = lesson["sizes"]["paper"]
    small_us = _fmt_us(small["fresh"]["median_us"])
    paper_us = _fmt_us(paper["fresh"]["median_us"])
    return _verified(
        "gemm_silu：SiLU 留在片段里，不要写全局 scratch",
        "gemm_silu reg silu epilogue scratch",
        [
            f"观察: `C = silu(A @ B)` 的中间结果只属于一个输出元素。fp16/small 的 REG fresh 是 {small_us} µs，相对最快基线 {_fmt_ratio(small['reg_speedup_vs_best'])} 倍；fp16/paper 的 REG fresh 是 {paper_us} µs，相对最快基线 {_fmt_ratio(paper['reg_speedup_vs_best'])} 倍。两份都是 fresh subject。paper 只比最快基线快不到 1%，不要把它写成稳定的生产加速。",
            f"没有实现 GLOBAL 候选。small 的两阶段启动下限是 {_fmt_us(small['two_stage_floor_us'])} µs，约 {_fmt_ratio(small['hard_lower_bound_over_reg'])} 倍于 REG；这是 probe 下界，不是 confirmed speedup。paper 的保守流量下界比 REG 高 {_fmt_pct(paper['hard_lower_bound_over_reg'] - 1)}，依据是 stage-2 必须读回 fp32 再写输出，不是一份 GLOBAL 核的 fresh 时间。",
        ],
        f"Apple M4 Max，fp16，REG fresh small {small_us} µs、paper {paper_us} µs。GLOBAL 没有实现。",
        tuple(lesson["sizes"][size]["fresh"]["source"] for size in ("small", "paper")) + (lesson["source"], lesson["probe_source"], DOC_GEMM),
        "只覆盖这两个 fp16 形状。不要把 conv2d 的全局 U 套到这个逐元素 epilogue 上。publication_allowed=false。",
    )


def _bitdelta_section(lesson: dict[str, Any]) -> str:
    small = lesson["sizes"]["small"]
    paper = lesson["sizes"]["paper"]
    if small["exact_ratio_taught"] or small["ranking_taught"] or not paper["exact_ratio_taught"]:
        raise ValueError("bitdelta would teach the unstable small ratio or ranking")
    paper_ratio = _fmt_ratio(paper["best_legal_shared_ratio_over_reg"])
    small_ratio = _fmt_ratio(small["best_legal_shared_ratio_over_reg"])
    recoverable = lesson["recoverable_decode_headroom_fraction"]
    recoverable_text = "决策记录没有单列可回收比例。" if recoverable is None else f"paper 上 REG 高于硬件下限的可回收比例是 {_fmt_pct(recoverable)}，这是同一份报告里的比值，不是另一套算法的上限。"
    small_us = _fmt_us(small["fresh"]["median_us"])
    paper_us = _fmt_us(paper["fresh"]["median_us"])
    text = _verified(
        "bitdelta_gemm：在 tile 里解包，不要先展开成全局权重",
        "bitdelta unpack reg shared",
        [
            f"观察: fp16/small 的 REG fresh 是 {small_us} µs，fp16/paper 是 {paper_us} µs。两份都是 fresh subject。没有写 SHARED 候选。paper 的同轮配对中位比是 {paper_ratio} 倍，可以保留。small 的两个最省合法形态在归档、重放和评审重跑之间翻过，只能教大约一个数量级，不能教精确倍数或排序。这是 paired probe，不是 fresh candidate，也不是 confirmed speedup。{recoverable_text}",
        ],
        f"Apple M4 Max，fp16。REG fresh small {small_us} µs、paper {paper_us} µs。paper 配对中位比 {paper_ratio}。small 只记大约一个数量级。",
        tuple(lesson["sizes"][size]["fresh"]["source"] for size in ("small", "paper"))
        + (lesson["source"], lesson["probe_source"], *lesson["ranking_boundary_sources"]),
        "只覆盖这两个 fp16 形状。small 不教精确倍数，也不教两个最省形态的排序。展开省下的是位运算，换来的是更大的全局流量和一个 stage。publication_allowed=false。",
    )
    if small_ratio in text:
        raise ValueError("the unstable small bitdelta ratio was rendered")
    return text


def _splitk_section(lesson: dict[str, Any]) -> str:
    if lesson["paper_split2_is_byte_ratio"] is not False or lesson["paper_split2_unit"] != "peak_bandwidth_time_equivalent":
        raise ValueError("Split-K percent lost its time-equivalent unit")
    numer = _fmt_us(lesson["paper_split2_numerator_us"])
    denom = _fmt_us(lesson["paper_split2_denominator_us"])
    percent = f"{lesson['paper_split2_traffic_percent']:.2f}%"
    floor_us = _fmt_us(lesson["small_two_stage_floor_us"])
    return _verified(
        "GEMM Split-K：这两个形状不要做",
        "gemm split-k reduction scratch",
        [
            f"观察: fp16/small 的两阶段无 fill 下限是 {floor_us} µs，约 {_fmt_ratio(lesson['small_floor_over_existing'])} 倍于归档的既有 candidate 锚点。这个下限 `not_a_candidate=true`，不是 candidate 时间。fp16/paper 已有 {lesson['paper_output_threadgroups']} 个输出 threadgroup。{percent} 不是字节流量比。它是峰值带宽时间等价：`{lesson['paper_split2_numerator_field']}` / `{lesson['paper_split2_denominator_field']}` = {numer} µs / {denom} µs。没有新增 Split-K 源码。",
        ],
        f"Apple M4 Max，fp16。small 两阶段下限 {floor_us} µs。paper 的峰值带宽时间等价比是 {percent}，分子是 {lesson['paper_split2_numerator_field']}。",
        (lesson["source"], lesson["probe_source"], DOC_SPLITK),
        "既有 candidate 微秒数是归档锚点，不是本阶段新的 fresh subject。这个百分比的单位是峰值带宽时间，不是字节比。不要因为 conv2d 的 SHARED 成立就做 Split-K。publication_allowed=false。",
    )


def _seq_control_text(controls: dict[str, Any]) -> str:
    """Render the path-specific controls. Prefixes are slices of the recorded hashes."""
    kda = controls["kda"]
    varlen = controls["varlen"]
    host = kda["host_program_sha256_by_arm"]
    geometry = kda["host_geometry_sha256_by_stage_count"]
    diff = varlen["diff"]
    return (
        f"KDA 的六份 fresh subject 里，`{kda['device_path']}` 都是 {kda['device_sha256'][:8]}…。"
        f"切链改的是 Host Program `{kda['host_program_path']}`："
        f"三 piece 为 {host['seq_three_piece'][:8]}…，两个单 piece 臂为 {host['single_piece'][:8]}…。"
        f"stage 数改的是 `{kda['host_geometry_path']}`："
        f"三 stage 为 {geometry['3'][:8]}…，单 stage 为 {geometry['1'][:8]}…。"
        "varlen mixed 的两份 fresh subject 没有 Host Program。"
        f"`{varlen['device_path']}` 不是同一份字节："
        f"pieces_three 是 {varlen['pieces_three_sha256'][:8]}…，pieces_one 是 {varlen['pieces_one_sha256'][:8]}…。"
        f"`varlen_source` 的唯一文本差别在 `kernel void {diff['kernel']}` 内，"
        f"把 `{diff['removed_line'].strip()}` 换成 `{diff['added_line'].strip()}`。"
        "这不是 Host Program 改切链。"
    )


def _kda_trio(row: dict[str, Any]) -> str:
    """A−C is total overhead. It is not the isolated handoff, and the three seeds differ."""
    return (
        f"A seq_three_piece {_fmt_us(row['fresh_a_us'])} µs（confirmation seed {row['fresh_a_seed']}），"
        f"B single_piece_three_stage {_fmt_us(row['fresh_b_us'])} µs（confirmation seed {row['fresh_b_seed']}），"
        f"C single_piece_single_stage {_fmt_us(row['fresh_c_us'])} µs（confirmation seed {row['fresh_c_seed']}）。"
        f"A−B 是 {_fmt_us(row['fresh_ab_us'])} µs，B−C 是 {_fmt_us(row['fresh_bc_us'])} µs。"
        f"A−C 相对 C 是 {_fmt_pct(row['fresh_ac_total_overhead_fraction'])}。"
        "三份 confirmation seed 不同，这不是同一次输入的 paired run。"
        "A−C 同时改了 piece 数和 stage 数，是跨 confirmation 的总开销，不能解释为隔离的 handoff 成本。"
        "A−B 与 B−C 也是跨 confirmation 的臂差，不能和决策样本的 handoff、stage floor 加成一个数。"
    )


def _seq_section(
    lesson: dict[str, Any],
    tail: dict[str, Any],
    single: dict[str, Any],
    geometry: dict[str, Any],
) -> str:
    rows = {row["name"]: row for row in lesson["confirmed_rows"]}
    small = rows["kda_small"]
    paper = rows["kda_paper"]
    mixed = rows["varlen_mixed"]
    paper_pct = _fmt_pct(paper["fresh_ac_total_overhead_fraction"])
    if tail["numbers_adopted"] or single["numbers_adopted"] or tail["independent_lesson"] or single["independent_lesson"]:
        raise ValueError("SEQ boundary numbers were marked as an independent lesson")
    return _verified(
        "因果状态：串行 stage 下切链不会变并发",
        "kda seq handoff serial stage",
        [
            f"观察: {_seq_control_text(lesson['controls'])} 八个 fresh subject 都经过确认读取：kda small 与 paper 各有 A/B/C，varlen mixed 有两臂。",
            f"kda small：{_kda_trio(small)}",
            f"kda paper：{_kda_trio(paper)} paper 的 A−C {paper_pct} 只说明相对一次慢递推的总开销很小，不能外推成 SEQ 便宜。",
            f"varlen mixed 只覆盖 L1：pieces_three {_fmt_us(mixed['fresh_seq_us'])} µs（confirmation seed {mixed['fresh_slow_seed']}）对 pieces_one {_fmt_us(mixed['fresh_single_us'])} µs（confirmation seed {mixed['fresh_fast_seed']}），相对是 {_fmt_pct(mixed['fresh_overhead_fraction'])}。两份 seed 不同，dispatch 的 stage 数相同，device 字节不同。不含 L2 capacity。",
            f"决策测量是另一份样本：small A−C {_fmt_pct(small['measurement_overhead_fraction'])}，其中 handoff 字段 {_fmt_us(small['measurement_handoff_us'])} µs、stage floor {_fmt_us(small['measurement_stage_floor_us'])} µs；paper {_fmt_pct(paper['measurement_overhead_fraction'])}；varlen {_fmt_pct(mixed['measurement_overhead_fraction'])}。不要把决策样本和 fresh 加成一个数。",
            f"kda tail 在 decision.sizes 里，A−C 是 {_fmt_pct(tail['seq_overhead_fraction_of_single_stage'])}，handoff 字段 {_fmt_us(tail['handoff_us'])} µs，stage floor {_fmt_us(tail['stage_floor_us'])} µs。它不是八个 fresh subject 之一。数值不采用，不作为独立经验。",
            f"kda single 不在 decision.sizes。state_algebra 的 pieces_per_chain 是 1，boundaries_used 是 0。测量文件里的 N=1 自检，A−B 相对 B 是 {_fmt_pct(single['ab_fraction'])}。这不是决策行。数值不采用，不作为独立经验。",
            _PHRASES["seq_scratch_observation"],
        ],
        (
            f"Apple M4 Max，fp16。kda small fresh A/B/C {_fmt_us(small['fresh_a_us'])}/"
            f"{_fmt_us(small['fresh_b_us'])}/{_fmt_us(small['fresh_c_us'])} µs；"
            f"kda paper fresh A/B/C {_fmt_us(paper['fresh_a_us'])}/"
            f"{_fmt_us(paper['fresh_b_us'])}/{_fmt_us(paper['fresh_c_us'])} µs；"
            f"varlen L1 mixed {_fmt_us(mixed['fresh_seq_us'])} µs 对 {_fmt_us(mixed['fresh_single_us'])} µs。"
            "这些不是同一次输入的 paired run。"
        ),
        (
            lesson["source"],
            lesson["decision_source"],
            single["source"],
            lesson["diff_source"],
            geometry["source"],
            DOC_SEQ,
            DOC_WORKER,
        ),
        (
            f"只覆盖 Apple M4 Max、fp16、kda 的 small 与 paper，以及 varlen_kda 的 L1 mixed。不含 L2 capacity。"
            f"paper 上跨 confirmation 的 A−C {paper_pct} 不能外推成 SEQ 便宜。"
            "kda tail 的决策测量和 kda single 的 N=1 自检数值不采用、不作为独立经验。"
            + _PHRASES["seq_boundary"]
            + "不要为了并发写跨 threadgroup 自旋。publication_allowed=false。NPU 仍是 simulation-only。"
        ),
    )


def prompt_boundary(root: Path | None = None, *, overlay: str | None = None) -> dict[str, Any]:
    """Render the live prompt and the counterfactual treatment. The live files stay untouched."""
    root = root or REPO
    if overlay is None:
        _catalog, overlay = build_catalog(root)
    from dataclasses import replace

    from taic.prompts import system_prompt_v2
    from taic.task_binding import bind_task
    from taic.trust import current_prompt_sha1
    from tests.support.live_anchors import load_live_anchors

    raw = (root / EXPERIENCE).read_text(encoding="utf-8")
    if OVERLAY_MARK in raw:
        raise ValueError("the live experience file already contains the P2-S overlay")
    control = filter_experience_for_prompt(raw)
    treatment = filter_experience_for_prompt(raw.rstrip() + "\n\n" + overlay)
    if OVERLAY_MARK not in treatment or OVERLAY_MARK in control:
        raise ValueError("the treatment filter did not isolate the overlay")
    epoch = _read(root, P2M_EPOCH)
    recorded = _read(root, P2M_RENDER)
    anchors = load_live_anchors(root)
    live_sha = current_prompt_sha1()
    if live_sha != anchors["prompt_sha1"] or live_sha == epoch["previous_prompt_sha1"]:
        raise ValueError("the live prompt fingerprint does not match live_anchors")

    def _render(task: str, size: str) -> dict[str, Any]:
        binding = bind_task(task, "fp16", size)
        tracks = {}
        for track_id in ("msl-single", "msl-bundle"):
            text = system_prompt_v2(replace(binding, track_id=track_id), use_tools=True, target_id="apple-m4-max")
            if OVERLAY_MARK in text:
                raise ValueError("the system prompt contains the uninstalled overlay")
            encoded = text.encode("utf-8")
            tracks[track_id] = {
                "sha256": hashlib.sha256(encoded).hexdigest(),
                "bytes": len(encoded),
            }
        return {"task": task, "dtype": "fp16", "size": size, "tracks": tracks}

    historical = _render("gemm", "small")
    experiment = _render("gemm_silu", "small")
    historical_match = {
        track_id: historical["tracks"][track_id]["sha256"] == row["system_prompt_sha256"]
        and historical["tracks"][track_id]["bytes"] == row["bytes"]
        for track_id, row in recorded["tracks"].items()
    }
    # G9 后来合法改变了任务书，P2-M 的旧渲染不再要求与 live 逐字节相同。
    # 这里报告实际比较结果，而不是把已知历史漂移当成 overlay 注入失败。
    # current_prompt_sha1 仍必须与 live_anchors 一致，配对内部仍只有 overlay 差别。
    control_bytes = control.encode("utf-8")
    treatment_bytes = treatment.encode("utf-8")
    return {
        "schema_id": "shapeforge.p2s-prompt-boundary.1",
        "epoch": "p2-s-experience-search-prep-1",
        "prompt_changed": False,
        "previous_prompt_sha1": epoch["current_prompt_sha1"],
        "current_prompt_sha1": live_sha,
        "p2m_prompt_epoch_sha256": hashlib.sha256((root / P2M_EPOCH).read_bytes()).hexdigest(),
        "p2m_rendered_prompts_match": historical_match,
        "historical_anchor_task": historical,
        "experiment_task_system_prompt": experiment,
        "system_prompt_identical_across_arms": True,
        "session_visible_difference": "filtered experience.md only; the system prompt render does not contain the overlay",
        "control_filtered_experience_sha256": hashlib.sha256(control_bytes).hexdigest(),
        "treatment_filtered_experience_sha256": hashlib.sha256(treatment_bytes).hexdigest(),
        "control_filtered_bytes": len(control_bytes),
        "treatment_filtered_bytes": len(treatment_bytes),
        "experience_file_sha256": hashlib.sha256((root / EXPERIENCE).read_bytes()).hexdigest(),
        "old_live_anchors": [
            "tests/shared/test_scratch_prompt.py and P2-M/verify.py still require current_prompt_sha1() to equal the P2-M epoch",
            _PHRASES["old_live_anchor"],
        ],
        "why_the_overlay_is_not_installed": "Installing it into experience.md would move the fingerprint, destroy the control arm, and make the closed live anchors fail. The treatment is this counterfactual filtered text.",
        "publication_allowed": False,
        "paid_search": "未授权/未执行",
    }


def approval_document() -> dict[str, Any]:
    """Blanks only. Filling any of these would look like an approval this phase does not have."""
    return {
        "schema_id": "shapeforge.p2s-search-approval.1",
        "experiment_id": None,
        "delta": None,
        "delta_unit": None,
        "allowed_regression": None,
        "approved_by": None,
        "owner_reviewed_on": None,
        "execution_record": None,
        "executed": False,
        "paid_search": "未授权/未执行",
        "gpu_window": None,
        "max_additional_cells": None,
        "tolerable_overrun": None,
        "conclusion_rule_filled": False,
    }


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write_preparation(directory: Path, *, overlay: str | None = None, catalog: dict[str, Any] | None = None) -> None:
    """Write the catalog, overlay, prompt boundary and blank approval. No search is started."""
    if catalog is None or overlay is None:
        catalog, overlay = build_catalog()
    evidence = directory / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "catalog.json").write_text(dumps(catalog), encoding="utf-8")
    (evidence / "experience-overlay.md").write_text(overlay, encoding="utf-8")
    (evidence / "prompt-boundary.json").write_text(dumps(prompt_boundary(overlay=overlay)), encoding="utf-8")
    (evidence / "approval.json").write_text(dumps(approval_document()), encoding="utf-8")


__all__ = [
    "OVERLAY_MARK",
    "approval_document",
    "build_catalog",
    "dumps",
    "prompt_boundary",
    "render_overlay",
]
