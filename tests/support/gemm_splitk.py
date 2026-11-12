"""Read-only arithmetic and replay checks for the rejected GEMM Split-K route."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
P2L = REPO_ROOT / "tests/support/golden/shared/shape-portfolio-阶段验收/P2-L"
FLOOR = P2L / "evidence" / "launch-floor.json"
ANALYSIS = P2L / "evidence" / "analysis.json"
PEAK_GBPS = 526.6592090468064
PEAK_FP16_TFLOPS = 14.765878729267158
SMALL_CANDIDATE_US = 1.8763327234945097
SMALL_BASELINE_US = 2.2444435181820985
PAPER_CANDIDATE_US = 1160.094681836199
PAPER_BASELINE_US = 1155.3576250662445


def load_json(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def _traffic_us(nbytes: int) -> float:
    return float(nbytes) / (PEAK_GBPS * 1e9) * 1e6


def analysis_document(floor: dict) -> dict:
    small_partial = 2 * 256 * 256 * 4
    paper_partial = 2 * 2048 * 2048 * 4
    roofline = 2 * 2048**3 / (PEAK_FP16_TFLOPS * 1e12) * 1e6
    two_stage = float(floor["arms"]["two_stage_no_fill"]["wall_us_median"])
    return {
        "schema_id": "shapeforge.gemm-splitk-negative-analysis.1",
        "publication_allowed": False,
        "implementation_started": False,
        "candidate_source_added": False,
        "decision": "do_not_implement",
        "small": {
            "existing_candidate_us": SMALL_CANDIDATE_US,
            "mlx_baseline_us": SMALL_BASELINE_US,
            "two_stage_no_fill_floor_us": two_stage,
            "floor_over_existing_candidate": two_stage / SMALL_CANDIDATE_US,
            "split2_partial_write_read_bytes": 2 * small_partial,
            "split2_partial_write_read_us_at_peak": _traffic_us(2 * small_partial),
            "reason": "The incomplete two-stage floor alone is slower than both the existing candidate and MLX baseline, before any real GEMM or reduction work.",
        },
        "paper": {
            "existing_candidate_us": PAPER_CANDIDATE_US,
            "mlx_baseline_us": PAPER_BASELINE_US,
            "output_threadgroups": 32 * 32,
            "roofline_us": roofline,
            "split2_partial_write_read_bytes": 2 * paper_partial,
            "split2_partial_write_read_us_at_peak": _traffic_us(2 * paper_partial),
            "split2_traffic_percent_of_candidate": _traffic_us(2 * paper_partial) / PAPER_CANDIDATE_US * 100.0,
            "reason": "There are already 1024 output threadgroups and the kernel is at the measured compute roofline; Split-2 adds traffic without relieving output parallelism or compute.",
        },
    }


def assert_negative() -> None:
    floor = load_json(FLOOR)
    if floor["not_a_candidate"] is not True or floor["publication_allowed"] is not False:
        raise AssertionError("launch-floor probe is being treated as a candidate")
    if floor["two_stage_floor_exceeds_existing_candidate"] is not True or floor["two_stage_floor_exceeds_mlx_baseline"] is not True:
        raise AssertionError("small two-stage floor no longer rejects Split-K")
    expected = analysis_document(floor)
    archived = load_json(ANALYSIS)
    if archived != expected:
        raise AssertionError("analysis.json is not the arithmetic projection of launch-floor.json and the fixed P3 anchors")
    if archived["implementation_started"] is not False or archived["candidate_source_added"] is not False:
        raise AssertionError("the rejected route incorrectly gained an implementation")
    if archived["decision"] != "do_not_implement" or archived["publication_allowed"] is not False:
        raise AssertionError("the negative result was rewritten as a publication")
