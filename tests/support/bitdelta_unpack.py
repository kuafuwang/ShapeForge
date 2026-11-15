"""bitdelta_gemm REG candidate, the unpack probe and the SHARED decision.

Pure Python on purpose: the shared CPU test imports this module, so it must not import
``tcenv.backends.metal`` or touch a GPU.  The Metal record/replay code lives in
``tests/backends/metal/bitdelta_unpack_record.py``.

The question this module encodes: for the two registered ``bitdelta_gemm`` shapes, does
decoding the packed sign words *once* into a global expanded matrix (SHARED: producer ->
barrier -> consumer) beat decoding them *inside every output tile* (REG)?  ``decision_document``
recomputes the answer from the archived fresh subjects and the measured probe, and the
verifier recomputes it again byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METAL_FIXTURES = ROOT / "tests" / "support" / "golden" / "metal" / "fixtures"
TREE = ROOT / "tests" / "support" / "golden" / "metal" / "bitdelta_reg"
_FIXTURE = ROOT / "tests/support/golden/shared/acceptance-epochs/fixtures/p2p"
P2P = _FIXTURE
EVIDENCE = _FIXTURE / "evidence"

SMALL_SHA256 = "29fdc86fe1af731173b3311d6102953eabc77a86e0f0effadcbb9c03455c2697"
PAPER_SHA256 = "988317bdccaa052aa348a820f2c5490a4a13873adaf789082b3452ccb8a504f7"
SIZES = ("small", "paper")
SHAPES = {"small": (128, 256, 256), "paper": (512, 4096, 4096)}
DECLARED_CASES = ("random", "edge_bits", "large", "tiny", "outlier", "ieee")
PROBE_SCHEMA = "shapeforge.bitdelta-unpack-probe.2"
PROBE_SUPERSEDED_SCHEMA = "shapeforge.bitdelta-unpack-probe.1"
DECISION_SCHEMA = "shapeforge.bitdelta-unpack-decision.1"
GATE_RATIO = 1.10

SMALL_CONFIRMATION = EVIDENCE / "reg-small-confirmation.json"
PAPER_CONFIRMATION = EVIDENCE / "reg-paper-confirmation.json"
PROBE = EVIDENCE / "unpack-probe.json"
PROBE_R1 = EVIDENCE / "unpack-probe-r1-superseded.json"
PROBE_R2 = EVIDENCE / "unpack-probe-r2-superseded.json"
PROBE_R3 = EVIDENCE / "unpack-probe-r3-superseded.json"
PROBE_R4 = EVIDENCE / "unpack-probe-r4-superseded.json"
PROBE_REPLAY = EVIDENCE / "unpack-probe-replay.json"
PROBE_REPLAY_R4 = EVIDENCE / "unpack-probe-replay-r4-superseded.json"
REPLAY_SCHEMA = "shapeforge.bitdelta-unpack-probe-replay.1"
DECISION = EVIDENCE / "decision.json"
SUPERSEDED = (PROBE_R1, PROBE_R2, PROBE_R3, PROBE_R4, PROBE_REPLAY_R4)

#: The r1 probe's B staging wrote four ``uint4`` passes per thread, i.e. 64 k rows into the
#: BLOCK_K=32 row threadgroup region.  Both constants are re-derived from the geometry here so
#: the record cannot silently claim zero overrun for a kernel that has one.
R1_B_STAGING_PASSES = 4
R1_THREADS = 128
BLOCK_K = 32
BLOCK_N = 64
C_STRIDE = BLOCK_N + 8
A_TILE_H = 64 * (BLOCK_K + 8)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_json(path: Path) -> dict:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise AssertionError(f"{path} is not a JSON object")
    return document


def dump_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def b_staging(passes: int, threads: int = R1_THREADS) -> dict:
    """Audit a cooperative ``uint4`` B staging loop: rows written, halves, threadgroup overrun."""
    halves = BLOCK_K * BLOCK_N
    written = threads * passes * 8
    last_s = threads * passes - 1
    rows = (last_s >> 3) + 1
    declared = A_TILE_H + BLOCK_K * C_STRIDE
    last_half = A_TILE_H + (last_s >> 3) * C_STRIDE + ((last_s & 7) << 3) + 8
    overrun = max(0, last_half - declared)
    return {
        "passes": passes,
        "rows_written": rows,
        "rows_declared": BLOCK_K,
        "halves_written": written,
        "halves_declared": halves,
        "over_read_ratio": written / halves,
        "threadgroup_overrun_halves": overrun,
        "threadgroup_overrun_bytes": 2 * overrun,
    }


R1_STAGING = b_staging(R1_B_STAGING_PASSES)
FIXED_STAGING = b_staging(2)


_LAUNCH_LINE = re.compile(r"launch (\w+) grid=(\d+),(\d+),(\d+) threads=(\d+),(\d+),(\d+)")
_TILE_CONST = re.compile(r"(?:constexpr|constant constexpr) uint (TM|TN|BLOCK_M|BLOCK_N) = (\d+);")


def source_geometry(size: str) -> dict:
    """Derive the REG candidate's own tiling from its bytes (not from a m//64 formula).

    The small candidate is a 32x32-tile kernel with ``grid=8,4,1``: eight column blocks, four row
    blocks and 32 output threadgroups.  The paper candidate is 64x64 with ``grid=64,8,1``.  The
    SHARED probe consumer is 64x64 for both sizes, so the two routes have different block counts
    and cannot share one row/column derivation.
    """
    body = launch_body(size)
    launches = _LAUNCH_LINE.findall(body)
    if len(launches) != 1:
        raise AssertionError(f"{size}: expected exactly one launch line, found {launches}")
    kernel, gx, gy, gz, tx, ty, tz = launches[0]
    constants = {}
    for name, value in _TILE_CONST.findall(stored_source(size)):
        constants.setdefault(name, int(value))
    tile_m = constants.get("TM", constants.get("BLOCK_M"))
    tile_n = constants.get("TN", constants.get("BLOCK_N"))
    if tile_m is None or tile_n is None:
        raise AssertionError(f"{size}: the REG candidate declares no tile constants: {constants}")
    m, n, _k = SHAPES[size]
    grid_x, grid_y = int(gx), int(gy)
    if grid_x * tile_n != n or grid_y * tile_m != m:
        raise AssertionError(
            f"{size}: grid {grid_x}x{grid_y} with a {tile_m}x{tile_n} tile does not cover {m}x{n}"
        )
    if int(gz) != 1 or int(ty) != 1 or int(tz) != 1:
        raise AssertionError(f"{size}: unexpected 3D launch {launches[0]}")
    shift = _decode_lane_shift(stored_source(size))
    return {
        "kernel": kernel,
        "threads": int(tx),
        "tile_m": tile_m,
        "tile_n": tile_n,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "row_blocks": grid_y,
        "col_blocks": grid_x,
        "output_threadgroups": grid_x * grid_y,
        "lanes_per_word": 1 << shift,
    }


def _decode_lane_shift(source: str) -> int:
    """Shift of the ``lid >> n`` variable that indexes the packed words (1=2 lanes, 2=4 lanes)."""
    assignments = re.findall(r"const uint (\w+) = lid >> (\d+);", source)
    hits = [
        (name, int(shift))
        for name, shift in assignments
        if re.search(r"bits\[[^\]]*\b" + re.escape(name) + r"\b", source)
    ]
    if len(hits) != 1:
        raise AssertionError(f"expected one packed-word lane mapping, found {hits}")
    name, shift = hits[0]
    if shift not in (1, 2):
        raise AssertionError(f"unexpected lane shift {shift} for {name!r}")
    return shift


def shared_geometry(size: str) -> dict:
    """The probe's SHARED consumer tiling: 64x64 for both registered shapes."""
    m, n, _k = SHAPES[size]
    if m % 64 or n % 64:
        raise AssertionError(f"{size}: the 64x64 SHARED tiling does not cover {m}x{n}")
    return {
        "tile_m": 64,
        "tile_n": 64,
        "row_blocks": m // 64,
        "col_blocks": n // 64,
        "output_threadgroups": (m // 64) * (n // 64),
    }


def stored_source(size: str) -> str:
    return (TREE / "variants" / size / "kernel.metal").read_text(encoding="utf-8")


def candidate_sha256(size: str) -> str:
    return sha256(TREE / "variants" / size / "kernel.metal")


def launch_body(size: str) -> str:
    source = stored_source(size)
    if source.count("/* taic-launch") != 1 or source.count("*/") < 1:
        raise AssertionError(f"{size} REG candidate does not carry exactly one launch block")
    return source.split("/* taic-launch", 1)[1].split("*/", 1)[0]


def assert_source_provenance() -> None:
    """Both REG sources decode in-tile, stay single stage and declare no scratch.

    The paper source is additionally checked against the registered fixture it was derived
    from: its A staging, MMA loop and epilogue must still appear verbatim, so "only the B path
    changed" is a property of the bytes and not of a comment.
    """
    for size in SIZES:
        source = stored_source(size)
        expected = SMALL_SHA256 if size == "small" else PAPER_SHA256
        if candidate_sha256(size) != expected:
            raise AssertionError(f"{size} REG candidate SHA-256 drifted")
        body = launch_body(size)
        launches = [line for line in body.splitlines() if line.startswith("launch ")]
        if len(launches) != 1:
            raise AssertionError(f"{size} REG candidate must be one stage, found {launches}")
        if "scratch" in body:
            raise AssertionError(f"{size} REG candidate declares scratch; REG must stay on-chip")
        if size == "paper" and "math fast" not in body:
            raise AssertionError("the paper REG candidate lost its fast-math launch flag")
        if size == "small" and body.strip() != "launch bitdelta_gemm_kernel grid=8,4,1 threads=128,1,1":
            raise AssertionError("the small REG candidate's launch block drifted from the archived candidate")
        for forbidden in ("atomic", "threadgroup_barrier(mem_flags::mem_device", "while ("):
            if forbidden in source:
                raise AssertionError(f"{size} REG candidate contains {forbidden!r}")
        if "simdgroup_multiply_accumulate" not in source:
            raise AssertionError(f"{size} REG candidate lost the matrix unit")
        if "threadgroup_barrier(mem_flags::mem_threadgroup)" not in source:
            raise AssertionError(f"{size} REG candidate lost the in-tile barrier")
        if "* SCALE" not in source:
            raise AssertionError(f"{size} REG candidate lost the OpSpec SCALE attribute")
    small = stored_source("small")
    if "bits[kb * N + n0 + nn]" not in small or "const uint nn = lid >> 2;" not in small:
        raise AssertionError("the small REG candidate no longer decodes one word per four lanes")
    if "constexpr uint TM = 32;" not in small or "constexpr uint TN = 32;" not in small:
        raise AssertionError("the small REG candidate lost its 32x32 tile")
    paper = stored_source("paper")
    if "bits[(k0 >> 5) * N + n0 + n]" not in paper or "const uint n = lid >> 1;" not in paper:
        raise AssertionError("the paper REG candidate no longer decodes one word per lane pair")
    fixture = (METAL_FIXTURES / "gemm_paper_best.metal").read_text(encoding="utf-8")
    for fragment in (
        "inline void load_a(device const half* A, uint m0, uint k0, uint lid,",
        "inline void store_a(threadgroup half* As, uint lid, uint4 x0, uint4 x1) {",
        "simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);",
        "const ushort qid = (ushort)(lane >> 2);",
        "for (uint s = lid; s < BLOCK_M * (BLOCK_N / 8u); s += 128u) {",
    ):
        if fixture.count(fragment) != 1:
            raise AssertionError(f"the paper GEMM fixture drifted: {fragment!r}")
        if paper.count(fragment) != 1:
            raise AssertionError(f"the paper REG candidate no longer shares the fixture's {fragment!r}")
    if "wfull" in paper or "wfull" in small:
        raise AssertionError("a REG candidate references a global expanded matrix")


def assert_no_shared_candidate() -> None:
    files = sorted(path.name for path in TREE.rglob("*") if path.is_file())
    if files != ["kernel.metal", "kernel.metal"]:
        raise AssertionError(f"the REG tree grew an unexpected file: {files}")
    variants = sorted(path.name for path in (TREE / "variants").iterdir() if path.is_dir())
    if variants != sorted(SIZES):
        raise AssertionError(f"the REG tree changed its variants: {variants}")
    for size in SIZES:
        if "scratch" in launch_body(size):
            raise AssertionError(f"{size} declares a global scratch")


def _report(subject: dict) -> dict:
    report = subject.get("confirmation_report")
    if not isinstance(report, dict):
        raise AssertionError("the subject has no confirmation_report")
    return report


def assert_cases(report: dict) -> None:
    if report.get("status") != "ok":
        raise AssertionError(f"fresh report is not ok: {report.get('status')!r}")
    cases = (report.get("correctness") or {}).get("cases") or []
    if [item.get("case") for item in cases] != list(DECLARED_CASES):
        raise AssertionError("the fresh report did not run the six declared cases")
    if not all(item.get("ok") is True for item in cases):
        raise AssertionError("a declared case failed")
    for item in cases:
        if item.get("n_bad") != 0:
            raise AssertionError(f"{item.get('case')} reports {item.get('n_bad')} bad elements")
        if item.get("unwritten") != 0:
            raise AssertionError(f"{item.get('case')} left {item.get('unwritten')} unwritten elements")
        if float(item.get("scaled_err")) > float(item.get("allow")):
            raise AssertionError(f"{item.get('case')} exceeded its scaled-error allowance")
    safety = report.get("safety") or {}
    if safety.get("ok") is not True or safety.get("nondeterminism") != []:
        raise AssertionError("the fresh report is not safe/deterministic")


def assert_subject(subject: dict, size: str) -> None:
    expected_sha = SMALL_SHA256 if size == "small" else PAPER_SHA256
    if subject.get("candidate_sha256") != expected_sha:
        raise AssertionError(f"{size} subject candidate SHA-256 drifted")
    if subject.get("provider") != "candidate" or subject.get("fallback_id") is not None:
        raise AssertionError(f"{size} subject did not resolve to its REG candidate")
    if subject.get("status") != "confirmed":
        raise AssertionError(f"{size} subject is not confirmed")
    if (subject.get("identity") or {}).get("kind") != "match":
        raise AssertionError(f"{size} subject identity is not a match")
    if subject.get("search_seed") == subject.get("confirmation_seed"):
        raise AssertionError(f"{size} fresh seed equals the search seed")
    report = _report(subject)
    assert_cases(report)
    if report.get("dtype") != "fp16" or report.get("size") != size or report.get("task") != "bitdelta_gemm":
        raise AssertionError(f"{size} subject report names the wrong instance")
    if (report.get("pipeline") or {}).get("stages") != 1:
        raise AssertionError(f"{size} REG candidate is not a single-stage pipeline")
    if not report.get("candidate_us") or report["candidate_us"] <= 0:
        raise AssertionError(f"{size} has no positive candidate time")
    if (report.get("benchmark") or {}).get("noisy") is not False:
        raise AssertionError(f"{size} fresh timing is flagged noisy")
    if report.get("build", {}).get("candidate_sha256") != expected_sha:
        raise AssertionError(f"{size} build identity does not name the REG candidate")
    roofline = report.get("roofline") or {}
    for key in ("lower_bound_us", "time_over_lower_bound", "peak_gbps_measured", "peak_tflops_measured"):
        if not isinstance(roofline.get(key), (int, float)) or roofline[key] <= 0:
            raise AssertionError(f"{size} fresh report has no usable roofline {key}")


def stable_identity(subject: dict) -> dict:
    """Fields that must survive a replay; wall-clock microseconds are deliberately absent."""
    report = _report(subject)
    build = report.get("build") or {}
    expected = subject.get("expected") or {}
    return {
        "candidate_sha256": subject.get("candidate_sha256"),
        "confirmation_seed": subject.get("confirmation_seed"),
        "instance_key": subject.get("instance_key"),
        "usage": subject.get("usage"),
        "identity": (subject.get("identity") or {}).get("kind"),
        "expected": expected,
        "build": {
            "candidate_sha256": build.get("candidate_sha256"),
            "assembled_sha256": build.get("assembled_sha256"),
            "native_digest": build.get("native_digest"),
            "lowering_profile_digest": build.get("lowering_profile_digest"),
            "opspec_digest": build.get("opspec_digest"),
            "artifact_set_digest": (build.get("artifacts") or {}).get("artifact_set_digest"),
        },
        "evaluation": {
            key: report.get(key)
            for key in (
                "evaluation_fingerprint",
                "baseline_environment_digest",
                "baseline_implementation_digest",
                "target_id",
                "protocol_version",
            )
        },
        "pipeline": report.get("pipeline"),
        "eligible": report.get("eligible"),
    }


def max_scaled_error(report: dict) -> tuple[str, float]:
    cases = (report.get("correctness") or {}).get("cases") or []
    worst = max(cases, key=lambda item: float(item["scaled_err"]))
    return str(worst["case"]), float(worst["scaled_err"])


def _best_baseline(report: dict) -> tuple[str, float]:
    baseline = report.get("baseline_us") or {}
    name, value = min(baseline.items(), key=lambda item: float(item[1]))
    return str(name), float(value)


def assert_probe(document: dict) -> None:
    """The probe must be a non-publishable measurement with a checkable staging audit."""
    if document.get("schema_id") != PROBE_SCHEMA:
        raise AssertionError("unknown unpack-probe schema")
    if document.get("not_a_candidate") is not True or document.get("publication_allowed") is not False:
        raise AssertionError("the probe must stay non-publishable and not a candidate")
    if document.get("cases_not_run") != list(DECLARED_CASES):
        raise AssertionError("the probe must list the declared cases it did not run")
    supersedes = document.get("supersedes") or {}
    if supersedes.get("schema_id") != PROBE_SUPERSEDED_SCHEMA:
        raise AssertionError("the probe no longer names the superseded r1 schema")
    for size in SIZES:
        one = (document.get("sizes") or {}).get(size)
        if not isinstance(one, dict):
            raise AssertionError(f"the probe has no {size} entry")
        m, n, k = SHAPES[size]
        if one.get("symbols") != {"M": m, "N": n, "K": k, "KB": k // 32}:
            raise AssertionError(f"{size} probe shape drifted")
        packed = 4 * (k // 32) * n
        if one.get("packed_bytes") != packed:
            raise AssertionError(f"{size} packed byte count drifted")
        if one.get("expanded_fp16_bytes") != 2 * k * n or one.get("expanded_fp32_bytes") != 4 * k * n:
            raise AssertionError(f"{size} expanded byte counts drifted")
        if one.get("decode_elements") != k * n:
            raise AssertionError(f"{size} decode element count drifted")
        audit = one.get("b_staging_audit") or {}
        if audit.get("rows_written") != BLOCK_K or audit.get("rows_declared") != BLOCK_K:
            raise AssertionError(f"{size}: the B staging does not cover exactly one BLOCK_K tile")
        if audit.get("over_read_ratio") != 1.0 or audit.get("threadgroup_overrun_bytes") != 0:
            raise AssertionError(f"{size}: the B staging still over-reads or overruns threadgroup memory")
        if audit.get("r1_defect_over_read_ratio") != R1_STAGING["over_read_ratio"]:
            raise AssertionError(f"{size}: the recorded r1 over-read is not 2x")
        if audit.get("r1_defect_overrun_bytes") != R1_STAGING["threadgroup_overrun_bytes"]:
            raise AssertionError(f"{size}: the recorded r1 threadgroup overrun drifted")
        arms = one.get("arms") or {}
        expected_arms = {
            "floor",
            "expand_fp32",
            "packed",
            "shared_fp32",
            "fill_fp16",
            "expand_fp16",
            "shared_fp16",
            "shared_packed16",
        }
        if set(arms) != expected_arms:
            raise AssertionError(f"{size} probe arms drifted: {sorted(arms)}")
        for name, arm in arms.items():
            if arm["wall_us_median"] <= 0 or arm["gpu_us_median"] <= 0:
                raise AssertionError(f"{size}/{name} has no positive time")
            if len(arm["wall_samples_us"]) != 9:
                raise AssertionError(f"{size}/{name} does not carry nine wall samples")
        if arms["packed"]["stages"] != 1 or arms["packed"]["fill_bytes_per_iteration"] != 0:
            raise AssertionError(f"{size}: the REG arm must be one stage with no fill")
        if arms["shared_packed16"]["stages"] != 2 or arms["shared_packed16"]["fill_bytes_per_iteration"] != 0:
            raise AssertionError(f"{size}: the packed-fp32 SHARED arm must be two stages without a fill")
        if arms["shared_fp16"]["fill_bytes_per_iteration"] != 2 * k * n:
            raise AssertionError(f"{size}: the fp16 device_zero arm must bill its K*N fill")
        if arms["shared_fp32"]["fill_bytes_per_iteration"] != 0:
            raise AssertionError(f"{size}: the fp32 poison arm must not pay a fill")
        if arms["shared_packed16"]["scratch_bytes"] != 2 * k * n:
            raise AssertionError(f"{size}: the packed-fp32 scratch footprint drifted")
        for name in ("expand_fp32", "expand_fp16", "shared_packed16"):
            coverage = arms[name].get("coverage") or {}
            expected_elements = k * n
            if coverage.get("expected_elements") != expected_elements:
                raise AssertionError(f"{size}/{name}: coverage does not span every decoded element")
            if coverage.get("read_back_elements") != expected_elements:
                raise AssertionError(f"{size}/{name}: the scratch read-back is incomplete")
            if coverage.get("mismatched_elements") != 0 or coverage.get("ok") is not True:
                raise AssertionError(f"{size}/{name}: the decode does not match the CPU reference")
        agreement = one.get("output_agreement") or {}
        for key in (
            "packed_vs_shared_fp32_equal",
            "packed_vs_shared_fp16_equal",
            "packed_vs_shared_packed16_equal",
        ):
            if agreement.get(key) is not True:
                raise AssertionError(f"{size}: {key} is false; the shared arms disagree with REG")
        if agreement.get("elements") != m * n:
            raise AssertionError(f"{size}: the output comparison does not cover M*N")
        no_fill = one.get("no_fill") or {}
        if set(no_fill) != {"shared_fp16", "shared_packed16"}:
            raise AssertionError(f"{size}: the measured no-fill arms drifted")
        for name, arm in no_fill.items():
            if arm.get("policy_fill_removed") is not True or arm.get("not_a_route") is not True:
                raise AssertionError(f"{size}/{name}: the no-fill measurement is mislabelled")
            if arm.get("fill_bytes_per_iteration") != 0:
                raise AssertionError(f"{size}/{name}: the no-fill measurement still bills a fill")
            if arm["wall_us_median"] <= 0:
                raise AssertionError(f"{size}/{name}: the no-fill measurement has no time")
        interleaved = one.get("interleaved") or {}
        if interleaved.get("rounds") != 12:
            raise AssertionError(f"{size}: the paired A/B does not carry twelve rounds")
        ratios = interleaved.get("paired_ratio_over_reg") or {}
        if set(ratios) != {"shared_fp32", "shared_fp16", "shared_packed16"}:
            raise AssertionError(f"{size}: the paired A/B arms drifted")
        for name, ratio in ratios.items():
            if len(ratio.get("per_round") or []) != 12:
                raise AssertionError(f"{size}/{name}: the paired A/B lost a round")
            if ratio["min"] <= 0 or ratio["max"] < ratio["min"]:
                raise AssertionError(f"{size}/{name}: the paired ratio range is not ordered")
            median = float(ratio["median"])
            if not min(ratio["per_round"]) <= median <= max(ratio["per_round"]):
                raise AssertionError(f"{size}/{name}: the paired median is outside its samples")
        traffic = one.get("traffic") or {}
        reg = source_geometry(size)
        shared = shared_geometry(size)
        # The recorded accounting must be the independent source derivation, not a m//64 guess.
        if traffic.get("reg_tiling") != {
            key: reg[key]
            for key in ("tile_m", "tile_n", "row_blocks", "col_blocks", "output_threadgroups", "lanes_per_word")
        }:
            raise AssertionError(f"{size}: REG tiling is not the candidate source geometry: {traffic.get('reg_tiling')} vs {reg}")
        if traffic.get("shared_tiling") != shared:
            raise AssertionError(f"{size}: SHARED tiling drifted: {traffic.get('shared_tiling')} vs {shared}")
        x_bytes = 2 * m * k
        if traffic["reg"]["packed_read_bytes"] != reg["row_blocks"] * packed:
            raise AssertionError(f"{size}: REG distinct packed-read accounting drifted")
        if traffic["reg"]["packed_read_bytes_issued"] != reg["lanes_per_word"] * reg["row_blocks"] * packed:
            raise AssertionError(f"{size}: REG issued packed-read accounting drifted")
        if traffic["reg"]["x_read_bytes"] != reg["col_blocks"] * x_bytes:
            raise AssertionError(f"{size}: REG x-read accounting drifted")
        if traffic["reg"]["y_write_bytes"] != 2 * m * n:
            raise AssertionError(f"{size}: REG y-write accounting drifted")
        if traffic["reg"]["output_threadgroups"] != reg["output_threadgroups"]:
            raise AssertionError(f"{size}: REG output parallelism drifted")
        if traffic["shared_fp16"]["consumer_b_read_bytes"] != shared["row_blocks"] * 2 * k * n:
            raise AssertionError(f"{size}: the fp16 consumer B accounting drifted")
        if traffic["shared_fp32"]["consumer_b_read_bytes"] != shared["row_blocks"] * 4 * k * n:
            raise AssertionError(f"{size}: the fp32 consumer B accounting drifted")
        if traffic["shared_fp16"]["x_read_bytes"] != shared["col_blocks"] * x_bytes:
            raise AssertionError(f"{size}: the SHARED x-read accounting drifted")
        if traffic["decode"]["reg_repeats"] != reg["row_blocks"]:
            raise AssertionError(f"{size}: REG decode repeats must be the candidate row blocks")
        if traffic["decode"]["reg_decoded_elements"] != reg["row_blocks"] * k * n:
            raise AssertionError(f"{size}: the REG decode-repeat accounting drifted")
        if traffic["decode"]["shared_decoded_elements"] != k * n:
            raise AssertionError(f"{size}: the SHARED decode-once accounting drifted")
        if traffic["b_staging"]["over_read_ratio"] != 1.0:
            raise AssertionError(f"{size}: the traffic model assumes a single-pass B staging")
        # The recorded arm grids must agree with the two tilings.
        packed_grid = arms["packed"]["grids"][0]
        if (packed_grid[0], packed_grid[1]) != (reg["col_blocks"], reg["row_blocks"]):
            raise AssertionError(f"{size}: the packed arm grid {packed_grid} is not the REG tiling")
        for name in ("shared_fp16", "shared_fp32", "shared_packed16"):
            grid = arms[name]["grids"][-1]
            if (grid[0], grid[1]) != (shared["col_blocks"], shared["row_blocks"]):
                raise AssertionError(f"{size}: {name} grid {grid} is not the 64x64 SHARED tiling")


def assert_superseded() -> None:
    """The rejected r1 probe and the two superseded reruns are archived, not deleted."""
    r1 = load_json(PROBE_R1)
    if r1.get("schema_id") != PROBE_SUPERSEDED_SCHEMA:
        raise AssertionError("the r1 probe is not the superseded schema")
    if r1.get("not_a_candidate") is not True or r1.get("publication_allowed") is not False:
        raise AssertionError("the r1 probe was not a non-publishable probe")
    audit = (r1.get("sizes") or {}).get("paper", {}).get("b_staging_audit")
    if audit is not None:
        raise AssertionError("the r1 probe already carries a staging audit; it is not the r1 bytes")
    for path in (PROBE_R2, PROBE_R3, PROBE_R4):
        document = load_json(path)
        if document.get("schema_id") != PROBE_SCHEMA:
            raise AssertionError(f"{path.name} is not a superseded unpack probe")
        if path == PROBE_R2 and "shared_packed16" in (document["sizes"]["paper"]["arms"]):
            raise AssertionError("the r2 probe already has the packed-fp32 arm; it is not the r2 bytes")
        if path == PROBE_R3 and "interleaved" in (document["sizes"]["paper"]):
            raise AssertionError("the r3 probe already has the paired A/B; it is not the r3 bytes")
        if path == PROBE_R4:
            # r4 measured the right arms but billed the small REG candidate with the 64x64 SHARED
            # geometry: two row blocks and 131,072 decoded elements instead of four and 262,144.
            small = document["sizes"]["small"]
            if "reg_tiling" in small["traffic"]:
                raise AssertionError("the r4 probe already carries per-route geometry; it is not r4")
            if small["traffic"]["decode"]["reg_repeats"] != 2:
                raise AssertionError("the r4 probe does not carry the 2x small accounting it is kept for")
            if small["traffic"]["reg"]["packed_read_bytes_issued"] != 65536:
                raise AssertionError("the r4 probe does not carry the 65,536-byte small issued accounting")
    stale = load_json(PROBE_REPLAY_R4)
    if stale.get("schema_id") != REPLAY_SCHEMA:
        raise AssertionError("the superseded replay is not the replay schema")
    if "reg_tiling" in stale["probe"]["sizes"]["small"]["traffic"]:
        raise AssertionError("the superseded replay already carries per-route geometry")


def assert_replay(document: dict) -> None:
    """The archived replay corroborates the stable comparisons and names the noise-limited ones.

    It is deliberately not an input of ``decision_document``: the decision is recomputed from
    ``unpack-probe.json`` alone, so a second probe run can never move a recorded number.
    """
    if document.get("schema_id") != REPLAY_SCHEMA:
        raise AssertionError("unknown unpack-probe replay schema")
    if document.get("not_a_candidate") is not True or document.get("publication_allowed") is not False:
        raise AssertionError("the replay must stay non-publishable and not a candidate")
    if document.get("authority") != "unpack-probe.json remains the decision source; this file is corroboration only":
        raise AssertionError("the replay claims decision authority it must not have")
    probe = document.get("probe")
    if not isinstance(probe, dict):
        raise AssertionError("the replay carries no probe document")
    assert_probe(probe)
    archived = load_json(PROBE)
    if (
        probe["sizes"]["paper"]["arms"]["packed"]["wall_samples_us"]
        == archived["sizes"]["paper"]["arms"]["packed"]["wall_samples_us"]
    ):
        raise AssertionError("the replay is a copy of the archived probe, not a second run")
    for size in SIZES:
        if probe["sizes"][size]["traffic"]["decode"]["reg_repeats"] != source_geometry(size)["row_blocks"]:
            raise AssertionError(f"{size}: the replay still bills REG with the wrong row-block count")
        ratios = probe["sizes"][size]["interleaved"]["paired_ratio_over_reg"]
        if size == "paper":
            if not ratios["shared_packed16"]["median"] < ratios["shared_fp32"]["median"] < ratios["shared_fp16"]["median"]:
                raise AssertionError("the paper ordering did not reproduce in the replay")
            if ratios["shared_packed16"]["median"] <= 1.0:
                raise AssertionError("the replay measured a legal SHARED form faster than REG at paper")
        else:
            # The two cheapest forms are within a few percent at small and can swap places.
            cheapest = min(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            dearest = max(ratios["shared_packed16"]["median"], ratios["shared_fp32"]["median"])
            if cheapest <= 4.0:
                raise AssertionError("the replay lost the order-of-magnitude gap at small")
            if dearest / cheapest > 1.25:
                raise AssertionError("the two cheapest legal forms are no longer noise-comparable at small")
        if ratios["shared_fp16"]["median"] <= 1.0:
            raise AssertionError(f"{size}: the replay measured the fp16 fill route faster than REG")


def _extrapolation_note(subtraction_us: float, floor_us: float, measured_no_fill_us: float) -> str:
    """Why the isolated fill arm is not subtracted, stated only where the arithmetic proves it."""
    if subtraction_us < floor_us:
        return (
            f"Subtracting the isolated fill arm from the fp16 route would report about "
            f"{subtraction_us:.1f} us, below the {floor_us:.1f} us hardware floor for the same GEMM. "
            "That subtraction is invalid: the memory-bound fill overlaps with the compute-bound "
            "consumer across the rotating sets. The measured no-fill pipeline is the honest number."
        )
    return (
        f"Subtracting the isolated fill arm would report {subtraction_us:.1f} us, which is not below "
        f"this size's {floor_us:.1f} us floor, so the subtraction proves nothing here; the measured "
        f"no-fill pipeline ({measured_no_fill_us:.1f} us) is used instead of any subtraction."
    )


def decision_document(subjects: dict[str, dict], probe: dict) -> dict:
    """Recompute the REG-vs-SHARED decision from the archived measurements.

    Single source of truth for ``evidence/decision.json``: the recorder calls it and the
    verifier recomputes it byte for byte.
    """
    assert_probe(probe)
    sizes = {}
    for size in SIZES:
        subject = subjects[size]
        assert_subject(subject, size)
        report = _report(subject)
        m, n, k = SHAPES[size]
        roofline = report["roofline"]
        one = probe["sizes"][size]
        arms = one["arms"]
        ratios = one["interleaved"]["paired_ratio_over_reg"]
        reg_us = float(report["candidate_us"])
        best_name, best_us = _best_baseline(report)
        floor_us = float(roofline["lower_bound_us"])
        headroom_us = reg_us - floor_us
        worst_case, worst_err = max_scaled_error(report)
        best_shared = min(ratios, key=lambda name: float(ratios[name]["median"]))
        best_shared_ratio = float(ratios[best_shared]["median"])
        best_shared_us = reg_us * best_shared_ratio
        marginal_fill_us = float(arms["shared_fp16"]["wall_us_median"]) - float(
            one["no_fill"]["shared_fp16"]["wall_us_median"]
        )
        isolated_fill_us = float(arms["fill_fp16"]["wall_us_median"])
        reg_geom = source_geometry(size)
        shared_geom = shared_geometry(size)
        stop = []
        if best_shared_ratio >= 1.0:
            stop.append("best_legal_shared_measured_slower")
        if headroom_us / reg_us < 0.10:
            stop.append("recoverable_decode_headroom_single_digit")
        if marginal_fill_us > 0:
            stop.append("fp16_device_zero_fill_is_a_timed_cost")
        if best_shared_ratio >= 1.0 and float(ratios["shared_fp32"]["median"]) >= 1.0:
            stop.append("every_measured_legal_shared_form_slower")
        sizes[size] = {
            "symbols": {"M": m, "N": n, "K": k, "KB": k // 32},
            "reg_candidate_sha256": subject["candidate_sha256"],
            "reg_candidate_us": reg_us,
            "reg_candidate_gpu_us_median": float(report["benchmark"]["candidate_gpu_us_median"]),
            "reg_cv": float(report["benchmark"]["noise_cv"]),
            "reg_noisy": report["benchmark"]["noisy"],
            "reg_confirmation_seed": subject["confirmation_seed"],
            "reg_search_seed": subject["search_seed"],
            "reg_confirmation_attempts": subject.get("confirmation_attempts"),
            "reg_worst_case": worst_case,
            "reg_worst_scaled_error": worst_err,
            "best_baseline": best_name,
            "best_baseline_us": best_us,
            "reg_latency_ratio_vs_best": reg_us / best_us,
            "reg_speedup_vs_best": best_us / reg_us,
            "roofline_lower_bound_us": floor_us,
            "roofline_bound": roofline["bound"],
            "peak_tflops_measured": float(roofline["peak_tflops_measured"]),
            "peak_gbps_measured": float(roofline["peak_gbps_measured"]),
            "reg_over_roofline": float(roofline["time_over_lower_bound"]),
            "recoverable_decode_headroom_us": headroom_us,
            "recoverable_decode_headroom_fraction": headroom_us / reg_us,
            "reg_tiling": {
                "tile_m": reg_geom["tile_m"],
                "tile_n": reg_geom["tile_n"],
                "row_blocks": reg_geom["row_blocks"],
                "col_blocks": reg_geom["col_blocks"],
                "output_threadgroups": reg_geom["output_threadgroups"],
                "lanes_per_word": reg_geom["lanes_per_word"],
                "threads": reg_geom["threads"],
            },
            "shared_tiling": shared_geom,
            "reg_x_read_bytes": one["traffic"]["reg"]["x_read_bytes"],
            "shared_x_read_bytes": one["traffic"]["shared_fp16"]["x_read_bytes"],
            "packed_bytes": one["packed_bytes"],
            "expanded_fp16_bytes": one["expanded_fp16_bytes"],
            "expanded_fp32_bytes": one["expanded_fp32_bytes"],
            "expand_ratio_fp16": one["expand_ratio_fp16"],
            "decode_elements": one["decode_elements"],
            "reg_decode_repeats": one["traffic"]["decode"]["reg_repeats"],
            "reg_decoded_elements": one["traffic"]["decode"]["reg_decoded_elements"],
            "shared_decoded_elements": one["traffic"]["decode"]["shared_decoded_elements"],
            "reg_packed_read_bytes": one["traffic"]["reg"]["packed_read_bytes"],
            "reg_packed_read_bytes_issued": one["traffic"]["reg"]["packed_read_bytes_issued"],
            "shared_fp16_consumer_b_read_bytes": one["traffic"]["shared_fp16"]["consumer_b_read_bytes"],
            "shared_fp32_consumer_b_read_bytes": one["traffic"]["shared_fp32"]["consumer_b_read_bytes"],
            "probe_reg_us": float(arms["packed"]["wall_us_median"]),
            "probe_to_fresh_delta_fraction": float(arms["packed"]["wall_us_median"]) / reg_us - 1.0,
            "measured_legal_shared": {
                name: {
                    "probe_wall_us": float(arms[name]["wall_us_median"]),
                    "paired_ratio_over_reg": float(ratios[name]["median"]),
                    "paired_ratio_min": float(ratios[name]["min"]),
                    "paired_ratio_max": float(ratios[name]["max"]),
                    "paired_rounds_slower_than_reg": int(ratios[name]["rounds_over_one"]),
                    "fill_bytes_per_iteration": arms[name]["fill_bytes_per_iteration"],
                    "stages": arms[name]["stages"],
                }
                for name in ("shared_packed16", "shared_fp32", "shared_fp16")
            },
            "best_legal_shared": best_shared,
            "best_legal_shared_ratio_over_reg": best_shared_ratio,
            "best_legal_shared_estimate_us": best_shared_us,
            "best_legal_shared_estimate_over_best_baseline": best_shared_us / best_us,
            "registered_baseline_gate_ratio": GATE_RATIO,
            "best_legal_shared_would_pass_registered_baseline_gate": best_shared_us / best_us <= GATE_RATIO,
            "adoption_bar_ratio_over_reg": 1.0,
            "best_legal_shared_max_round_ratio": float(ratios[best_shared]["max"]),
            "best_legal_shared_min_round_ratio": float(ratios[best_shared]["min"]),
            "two_stage_floor_us": float(arms["floor"]["wall_us_median"]),
            "two_stage_floor_over_reg": float(arms["floor"]["wall_us_median"]) / reg_us,
            "fp16_fill_isolated_us": isolated_fill_us,
            "fp16_fill_marginal_us": marginal_fill_us,
            "fp16_fill_marginal_fraction_of_route": marginal_fill_us
            / float(arms["shared_fp16"]["wall_us_median"]),
            "isolated_fill_subtraction_us": float(arms["shared_fp16"]["wall_us_median"]) - isolated_fill_us,
            "measured_no_fill_fp16_us": float(one["no_fill"]["shared_fp16"]["wall_us_median"]),
            "measured_no_fill_packed16_us": float(one["no_fill"]["shared_packed16"]["wall_us_median"]),
            "subtraction_error_us": float(one["no_fill"]["shared_fp16"]["wall_us_median"])
            - (float(arms["shared_fp16"]["wall_us_median"]) - isolated_fill_us),
            "extrapolation_note": _extrapolation_note(
                float(arms["shared_fp16"]["wall_us_median"]) - isolated_fill_us,
                floor_us,
                float(one["no_fill"]["shared_fp16"]["wall_us_median"]),
            ),
            "shared_would_win": best_shared_ratio < 1.0,
            "meets_adoption_bar": best_shared_ratio < 1.0,
            "probe_estimates_are_not_candidates": True,
            "stop_reasons": stop,
        }
        if sizes[size]["shared_would_win"]:
            raise AssertionError(f"{size}: a legal SHARED form measured faster than REG; the decision is invalid")
        if sizes[size]["meets_adoption_bar"]:
            raise AssertionError(f"{size}: a legal SHARED form beats its own REG cost; the decision is invalid")
    return {
        "schema_id": DECISION_SCHEMA,
        "task": "bitdelta_gemm",
        "dtype": "fp16",
        "target_id": "apple-m4-max",
        "publication_allowed": False,
        "adopted_route": "REG",
        "decision": "reg_only",
        "implementation": {
            "reg_candidate_added": True,
            "shared_candidate_source_added": False,
            "same_algorithm_ab_run": False,
            "reg_tree": "tests/support/golden/metal/bitdelta_reg",
            "reg_small_sha256": SMALL_SHA256,
            "reg_paper_sha256": PAPER_SHA256,
        },
        "shared_forms_measured": {
            "shared_packed16": (
                "legal cheapest: uninitialized_poison fp32 scratch, texture producer, two exact fp16 "
                "bit patterns per texel, no fill; stage 2 copies 16 B per 8 halves"
            ),
            "shared_fp32": "legal: poison fp32 texture producer, stage 2 reads fp32 and converts to half",
            "shared_fp16": "legal: device_zero fp16 scratch, harness fill inside the timed window",
            "shared_fp16_no_fill": (
                "not a route: the same fp16 pipeline with the declared fill removed from the timed "
                "window, to measure the no-fill bound on identical kernels"
            ),
        },
        "decision_basis": (
            "every measured legal SHARED form is slower than the REG candidate in a paired same-run "
            "A/B; no net gain over the incumbent implementation"
        ),
        "adoption_bar": (
            "adopt only a variant that is not slower than the incumbent REG candidate "
            "(ratio over REG < 1.0). This is an engineering adoption bar, not the harness gate."
        ),
        "adoption_bar_ratio_over_reg": 1.0,
        "registered_baseline_gate_ratio": GATE_RATIO,
        "registered_baseline_gate": (
            "the repository's fresh performance gate is latency_ratio = candidate_us / baseline_us <= "
            "1.10 against the registered baseline (tcenv/portfolio_performance/performance.py, "
            "tcenv/portfolio_performance/performance.py::MAX_REGRESSION_RATIO). It is recorded here "
            "only for context: at paper the cheapest legal SHARED estimate would pass it "
            "(about 0.97x of the fastest baseline), so that gate is NOT the reason SHARED was "
            "rejected; the rejection rests on the measured comparison against REG."
        ),
        "sizes": sizes,
        "rationale": (
            "SHARED decodes every weight once instead of once per output row block and then pays an "
            "extra producer stage, a barrier, a global write of the expanded matrix and a stage-2 read "
            "of that matrix in every row block. On this target the paper shape is compute-bound: the "
            "hardware floor is about 1.16 ms for the same GEMM, and the whole REG kernel is only 4.8% "
            "above it, so even a perfect same-algorithm rewrite could recover only a single-digit "
            "percentage. Measured paired against the REG arm in the same run, "
            "the cheapest legal form (packed fp16 bit patterns in a poison fp32 scratch, no fill) is "
            "2.59% slower at paper and 7.94x slower at small; the fp32 poison form is 9.2% slower, and "
            "the fp16 device_zero form is 47% slower because the fill is a timed cost. The no-fill "
            "measurement on identical kernels is still slower than REG, so the fill is not the reason. "
            "Therefore no SHARED candidate was written and REG is kept."
        ),
        "probe_not_a_candidate": True,
        "npu": {"simulated_only": True, "publication_allowed": False},
        "replay": {
            "probe": "uv run python experiments/bitdelta_unpack_probe.py <output>",
            "cpu_test": "uv run pytest -q -n 0 tests/shared/test_bitdelta_unpack_decision.py",
            "gpu_test": "uv run pytest -q -n 0 -m gpu tests/backends/metal/test_bitdelta_unpack_gpu.py",
        },
    }


def _historical_replay_verifier() -> str:
    """One historical command from the fixture. It is not emitted into a new decision."""
    raw = (_FIXTURE / "historical-replay-verifier.txt").read_text(encoding="utf-8")
    if raw.endswith("\n"):
        raw = raw[:-1]
    if not raw or raw != raw.strip() or "\n" in raw:
        raise AssertionError("historical replay verifier fixture is not one command")
    return raw


def _without_archived_verifier(document: dict) -> dict:
    """Drop only the historical command. Any other verifier string fails."""
    replay = document.get("replay")
    if not isinstance(replay, dict) or "verifier" not in replay:
        return document
    if replay.get("verifier") != _historical_replay_verifier():
        raise AssertionError("archived replay.verifier is not the historical command")
    compared = dict(document)
    compared_replay = dict(replay)
    del compared_replay["verifier"]
    compared["replay"] = compared_replay
    return compared


def assert_decision(document: dict, subjects: dict[str, dict], probe: dict) -> None:
    expected = decision_document(subjects, probe)
    if "verifier" in expected["replay"]:
        raise AssertionError("a recomputed decision must not name a package verifier")
    if _without_archived_verifier(document) != expected:
        raise AssertionError("decision.json is not the recomputation of its measurement sources")
    if document["adopted_route"] != "REG" or document["decision"] != "reg_only":
        raise AssertionError("the decision did not keep REG")
    if document["implementation"]["shared_candidate_source_added"] is not False:
        raise AssertionError("a SHARED candidate source was added")
    if document["implementation"]["same_algorithm_ab_run"] is not False:
        raise AssertionError("the decision claims a candidate A/B it did not run")
    if document["publication_allowed"] is not False:
        raise AssertionError("bitdelta_gemm may not publish a speedup")
    for size, one in document["sizes"].items():
        if one["shared_would_win"] or one["meets_adoption_bar"]:
            raise AssertionError(f"{size}: a legal SHARED form would win")
        if one["best_legal_shared_ratio_over_reg"] < 1.0:
            raise AssertionError(f"{size}: the best legal SHARED ratio is below 1.0")
        if one["reg_decode_repeats"] < 2:
            raise AssertionError(f"{size}: REG does not actually repeat the decode")
        if one["shared_decoded_elements"] * one["reg_decode_repeats"] != one["reg_decoded_elements"]:
            raise AssertionError(f"{size}: the decode-repeat arithmetic drifted")
        # The decision basis is the REG comparison; the registered-baseline gate is only context.
        if one["best_legal_shared_would_pass_registered_baseline_gate"] and one["shared_would_win"]:
            raise AssertionError(f"{size}: the baseline gate and the REG comparison disagree")
        if one["measured_no_fill_fp16_us"] < one["roofline_lower_bound_us"] * 0.98:
            raise AssertionError(f"{size}: a measured arm is below the hardware floor")
