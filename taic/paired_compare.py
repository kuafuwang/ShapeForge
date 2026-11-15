"""Host-side paired timing of two candidates.

The measurement alternates both runners in one process on one input batch.
It records medians, per-round ratios, an interval, round count, cv and both
SHAs. It never changes ``relation_to_best`` or the search seed. Input order
stays on the search seed (v2 uses 7); only the timing order is shuffled.

A model nomination is not host evidence. The host selects at most K=1 pair
per session with the rules in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA = "taic.paired_comparison.1"
NOMINATION_SCHEMA = "taic.paired_nomination.1"
SELECTION_SCHEMA = "taic.paired_host_selection.1"
NOMINATION_NAME = "paired_comparisons.jsonl"
HOST_EVIDENCE_NAME = "host_paired_comparisons.jsonl"
HOST_K = 1
NO_DIFF_BAND = (0.98, 1.02)
PREFERRED_CONCLUSIONS = frozenset({"no_diff", "regressed"})
Runner = Callable[[int], float]


class PairedCompareError(ValueError):
    """@brief 配对比较的输入不成立。"""


def normalized_pair(sha_a: str, sha_b: str) -> tuple[str, str]:
    """@brief 把两个 SHA 规范成有序对，使提名顺序不影响身份。

    @param sha_a 第一颗候选的 SHA-256。
    @param sha_b 第二颗候选的 SHA-256。
    @return 小写十六进制、按字典序排列的二元组。
    """
    left = str(sha_a or "").strip().lower()
    right = str(sha_b or "").strip().lower()
    if len(left) != 64 or len(right) != 64:
        raise PairedCompareError("both candidate SHAs must be 64 hex characters")
    return (left, right) if left <= right else (right, left)


def backend_capability(backend_id: str) -> dict:
    """@brief 后端是否提供宿主配对比较。Metal 可用，其它后端明确不支持。

    @param backend_id 后端标识。
    @return ``supported`` 与原因。不启动设备。
    """
    name = str(backend_id or "").strip().lower()
    if name == "metal":
        return {"backend": "metal", "supported": True, "reason": ""}
    return {
        "backend": name,
        "supported": False,
        "reason": f"backend {name!r} does not support paired comparison",
    }


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def conclude_samples(
    samples_a: Sequence[float],
    samples_b: Sequence[float],
    *,
    sha_a: str,
    sha_b: str,
    band: tuple[float, float] = NO_DIFF_BAND,
    input_seed: int = 7,
) -> dict:
    """@brief 由同一轮次上的两组耗时给出三种结论之一。

    比值是 ``time_a / time_b``。区间是比值均值加减 1.96 倍标准误。
    整个区间落在 ``band`` 内是没差；整个区间落在 ``band`` 外侧是有差；
    区间跨过边界、样本不足或出现非正耗时是噪声太大。

    @param samples_a 候选 A 的逐轮耗时（秒），与 B 逐项配对。
    @param samples_b 候选 B 的逐轮耗时（秒）。
    @param sha_a 候选 A 的 SHA-256。
    @param sha_b 候选 B 的 SHA-256。
    @param band 判「没差」的闭区间。
    @param input_seed 这批评测输入用的种子。计时顺序另打乱，不换这颗种子。
    @return 配对比较记录。``changes_seed`` 恒为 false。
    """
    pair = normalized_pair(sha_a, sha_b)
    # Keep the caller's A/B labels for direction. The stored pair is canonical.
    label_a = str(sha_a).strip().lower()
    label_b = str(sha_b).strip().lower()
    ratios: list[float] = []
    kept_a: list[float] = []
    kept_b: list[float] = []
    for left, right in zip(samples_a, samples_b):
        time_a = _finite_positive(left)
        time_b = _finite_positive(right)
        if time_a is None or time_b is None:
            continue
        kept_a.append(time_a)
        kept_b.append(time_b)
        ratios.append(time_a / time_b)
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "sha_a": label_a,
        "sha_b": label_b,
        "sha_pair": list(pair),
        "median_a_s": None,
        "median_b_s": None,
        "ratios": ratios,
        "interval": None,
        "rounds": len(ratios),
        "cv": None,
        "conclusion": "too_noisy",
        "direction": None,
        "band": [float(band[0]), float(band[1])],
        "input_seed": int(input_seed),
        "order_shuffled": True,
        "changes_seed": False,
    }
    if len(ratios) < 2 or len(samples_a) != len(samples_b):
        return document
    mean = statistics.fmean(ratios)
    spread = statistics.pstdev(ratios)
    cv = spread / mean if mean > 0 else 1.0
    se = spread / math.sqrt(len(ratios))
    lo = mean - 1.96 * se
    hi = mean + 1.96 * se
    low_band, high_band = float(band[0]), float(band[1])
    if hi < low_band:
        conclusion, direction = "diff", "a_faster"
    elif lo > high_band:
        conclusion, direction = "diff", "b_faster"
    elif lo >= low_band and hi <= high_band:
        conclusion, direction = "no_diff", None
    else:
        conclusion, direction = "too_noisy", None
    document.update(
        {
            "median_a_s": statistics.median(kept_a),
            "median_b_s": statistics.median(kept_b),
            "interval": [lo, hi],
            "cv": cv,
            "conclusion": conclusion,
            "direction": direction,
        }
    )
    return document


def compare_callables(
    runner_a: Runner,
    runner_b: Runner,
    *,
    sha_a: str,
    sha_b: str,
    reps: int = 8,
    seed: int = 7,
    band: tuple[float, float] = NO_DIFF_BAND,
    warmup_reps: int = 0,
) -> dict:
    """@brief 在同一进程里交错调用两颗 runner，再下结论。

    调用 ``tcenv.bench.interleaved``，不修改它。同一轮里的两个样本按下标配对。

    @param runner_a ``run_k(k) -> 秒``。
    @param runner_b ``run_k(k) -> 秒``。
    @param sha_a 候选 A 的 SHA-256。
    @param sha_b 候选 B 的 SHA-256。
    @param reps 计入统计的轮数。
    @param seed 只打乱计时顺序的种子。
    @param band 没差区间。
    @param warmup_reps 丢弃的预热轮数。假 runner 测试用 0。
    @return :func:`conclude_samples` 的记录，并带上两颗 runner 自己的 cv。
    """
    from tcenv.bench import interleaved

    stats = interleaved(
        {"a": runner_a, "b": runner_b},
        reps=int(reps),
        seed=int(seed),
        warmup_reps=int(warmup_reps),
        kmin=1,
    )
    document = conclude_samples(
        stats["a"].samples_s,
        stats["b"].samples_s,
        sha_a=sha_a,
        sha_b=sha_b,
        band=band,
        input_seed=int(seed),
    )
    document["cv_a"] = float(stats["a"].cv)
    document["cv_b"] = float(stats["b"].cv)
    return document


def _sha256_file(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _checked_row(row: Mapping[str, Any], host_pairs: set[tuple[str, str]] | None = None) -> tuple[dict | None, dict | None]:
    """@brief 丢掉缺文件、身份不符或已有等价宿主证据的提名。

    模型行上的 ``has_host_evidence`` 不算证据。只有宿主自己的测量文件里
    已经有这对 SHA，才记成已有证据。

    @return ``(合格行, 省略记录)``，二者必有一个为 None。
    """
    if row.get("unreadable"):
        return None, {
            "omit_reason": "unreadable_nomination",
            "detail": str(row.get("detail") or ""),
            "line": row.get("line"),
        }
    try:
        pair = normalized_pair(str(row.get("sha_a") or ""), str(row.get("sha_b") or ""))
    except PairedCompareError as exc:
        return None, {"omit_reason": "identity_mismatch", "detail": str(exc), "line": row.get("line")}
    # 模型可写行上的 files_present / identity_ok 不作数，缺省也不算通过。
    # 两个路径都必须指向真实文件，SHA 由宿主按字节重算。
    path_a = str(row.get("path_a") or "").strip()
    path_b = str(row.get("path_b") or "").strip()
    if not path_a or not path_b:
        return None, {"omit_reason": "missing_file", "sha_pair": list(pair)}
    digest_a = _sha256_file(Path(path_a))
    digest_b = _sha256_file(Path(path_b))
    if digest_a is None or digest_b is None:
        return None, {"omit_reason": "missing_file", "sha_pair": list(pair)}
    declared_a = str(row.get("sha_a") or "").strip().lower()
    declared_b = str(row.get("sha_b") or "").strip().lower()
    if digest_a != declared_a or digest_b != declared_b:
        return None, {"omit_reason": "identity_mismatch", "sha_pair": list(pair)}
    if pair in set(host_pairs or ()):
        return None, {"omit_reason": "equivalent_host_evidence", "sha_pair": list(pair)}
    kept = dict(row)
    kept["sha_pair"] = list(pair)
    kept["conclusion"] = str(row.get("conclusion") or "")
    kept["cited_by_rejected_premise"] = bool(row.get("cited_by_rejected_premise"))
    return kept, None


def select_nominations(rows: Sequence[Mapping[str, Any]], *, k: int = HOST_K, host_pairs: set[tuple[str, str]] | None = None) -> dict:
    """@brief 按固定规则从本场提名里选至多 K 对。模型的排列顺序无效。

    先按规范化 SHA 对去重。缺文件、身份不符、已有等价宿主证据的提名不占名额。
    被否决前提引用且结论为 ``no_diff`` 或 ``regressed`` 的排在前面。
    其余按 SHA 对排序。K 固定为 1。``reviewed`` 在真正重跑之前是 0。

    @param rows 本场全部提名，含读不出来的行。
    @param k 宿主复核上限。本计划固定为 1。
    @param host_pairs 宿主测量文件里已经有的 SHA 对。模型旗标不算。
    @return 选择记录。选中但还没重跑时 ``reviewed`` 为 0。
    """
    if int(k) != HOST_K:
        raise PairedCompareError("host review K is fixed at 1 until a later measurement says otherwise")
    known = set(host_pairs or ())
    omitted: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            omitted.append({"omit_reason": "unreadable_nomination", "detail": "nomination is not an object"})
            continue
        kept, dropped = _checked_row(row, known)
        if dropped is not None:
            omitted.append(dropped)
            continue
        assert kept is not None
        pair = tuple(kept["sha_pair"])
        grouped.setdefault(pair, []).append(kept)

    def preferred(item: dict) -> bool:
        return bool(
            item.get("cited_by_rejected_premise")
            and str(item.get("conclusion") or "") in PREFERRED_CONCLUSIONS
        )

    unique: dict[tuple[str, str], dict] = {}
    for pair, group in grouped.items():
        group = sorted(group, key=lambda item: (0 if preferred(item) else 1, str(item.get("conclusion") or "")))
        unique[pair] = group[0]
        for _extra in group[1:]:
            omitted.append({"omit_reason": "duplicate_sha_pair", "sha_pair": list(pair)})

    def sort_key(item: dict) -> tuple:
        return (0 if preferred(item) else 1, tuple(item["sha_pair"]))

    ordered = sorted(unique.values(), key=sort_key)
    selected = ordered[: int(k)]
    uncovered = [{"omit_reason": "over_k", "sha_pair": list(item["sha_pair"]), "conclusion": item.get("conclusion")} for item in ordered[int(k) :]]
    uncovered.extend(omitted)
    return {
        "schema": SELECTION_SCHEMA,
        "k": HOST_K,
        "nominations": len(list(rows)),
        "unique_eligible": len(ordered),
        "reviewed": 0,
        "uncovered_count": len(uncovered),
        "selected": [
            {
                "sha_pair": list(item["sha_pair"]),
                "sha_a": str(item.get("sha_a") or ""),
                "sha_b": str(item.get("sha_b") or ""),
                "path_a": str(item.get("path_a") or ""),
                "path_b": str(item.get("path_b") or ""),
                "conclusion": item.get("conclusion") or "",
                "cited_by_rejected_premise": bool(item.get("cited_by_rejected_premise")),
            }
            for item in selected
        ],
        "uncovered": uncovered,
        "ran_gpu": False,
        "gpu_seconds": 0.0,
        "changes_seed": False,
    }


def run_selected(
    selection: Mapping[str, Any],
    rerun: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None = None,
    *,
    evidence_dir: Path | None = None,
) -> dict:
    """@brief 没有合格提名时不调用重跑。重跑必须返回测量文档，不能只交一个秒数。

    选中但没有重跑时 ``reviewed`` 仍是 0。真正测完的对数才计入 ``reviewed``，
    这些文档的 ``elapsed_s`` 相加成为 ``gpu_seconds``。测完的文档同时写入
    宿主证据文件。

    @param selection :func:`select_nominations` 的结果。
    @param rerun 接收选中列表，返回每对的测量文档。没有合格对时不得被调用。
    @param evidence_dir 宿主证据目录。给出时才落盘。
    @return 带测量文档、复核数和 GPU 秒数的选择记录。
    """
    document = dict(selection)
    document["reviewed"] = 0
    document["measurements"] = []
    document["ran_gpu"] = False
    document["gpu_seconds"] = 0.0
    if not document.get("selected"):
        return document
    if rerun is None:
        document["rerun_reason"] = "no in-process runner is bound; the selection is recorded and the GPU was not used"
        return document
    produced = rerun(document["selected"])
    if isinstance(produced, (int, float)) and not isinstance(produced, bool):
        document["rerun_reason"] = "the rerun returned a duration without a paired measurement"
        return document
    measurements = [item for item in list(produced or []) if isinstance(item, dict)]
    applied = [item for item in measurements if item.get("applied") and "elapsed_s" in item]
    document["measurements"] = measurements
    document["reviewed"] = len(applied)
    document["ran_gpu"] = bool(applied)
    document["gpu_seconds"] = float(sum(float(item.get("elapsed_s") or 0.0) for item in applied))
    if evidence_dir is not None:
        for item in applied:
            append_host_evidence(Path(evidence_dir), item)
    return document


def append_nomination(directory: Path, row: Mapping[str, Any]) -> Path:
    """@brief 把模型提名写进工作目录的独立文件，不写权威 ``self_checks.jsonl``。

    @param directory 会话的 ``.selfcheck`` 目录。
    @param row 提名正文，必须带两个 SHA。
    @return 提名文件路径。
    """
    pair = normalized_pair(str(row.get("sha_a") or ""), str(row.get("sha_b") or ""))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / NOMINATION_NAME
    payload = {
        "schema": NOMINATION_SCHEMA,
        "sha_a": str(row.get("sha_a") or "").strip().lower(),
        "sha_b": str(row.get("sha_b") or "").strip().lower(),
        "sha_pair": list(pair),
        "conclusion": str(row.get("conclusion") or ""),
        "cited_by_rejected_premise": bool(row.get("cited_by_rejected_premise")),
        "path_a": str(row.get("path_a") or ""),
        "path_b": str(row.get("path_b") or ""),
        "source": "model",
        "host_evidence": False,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def apply_premise_citations(rows: Sequence[Mapping[str, Any]], feedback_text: str) -> list[dict]:
    """@brief 只按「被否决的前提」正文判断引用。模型自己标的旗标不算。

    @param rows 提名。
    @param feedback_text 本场 FEEDBACK.md。没有前提节时，全部视为未被引用。
    @return 新的提名列表。``cited_by_rejected_premise`` 只反映正文里是否同时出现两个 SHA。
    """
    premise = ""
    current = None
    for line in (feedback_text or "").splitlines():
        stripped = line.strip().lstrip("#").strip()
        if line.strip().startswith("#"):
            current = stripped
            continue
        if current == "被否决的前提":
            premise += "\n" + line
    lowered = premise.lower()
    out = []
    for row in rows:
        item = dict(row)
        left = str(item.get("sha_a") or "").strip().lower()
        right = str(item.get("sha_b") or "").strip().lower()
        item["cited_by_rejected_premise"] = bool(left and right and left in lowered and right in lowered)
        out.append(item)
    return out


def host_evidence_path(directory: Path) -> Path:
    """@brief 宿主证据的权威路径。与自测权威日志同一把钥匙，但是另一个文件。

    钥匙来自 ``authority_log``：账户主目录下的缓存，不看 ``HOME``、
    ``/tmp`` 或模型工作目录。工作目录里的同名文件不是证据。

    @param directory 会话的 ``.selfcheck`` 目录，只用来计算钥匙。
    @return 宿主拥有的 ``host_paired_comparisons.jsonl``。
    """
    from tcenv.selfcheck import authority_log

    return authority_log(Path(directory)).parent / HOST_EVIDENCE_NAME


def append_host_evidence(directory: Path, document: Mapping[str, Any]) -> Path:
    """@brief 把宿主自己测到的配对结果追加到权威证据文件。

    不在模型可写的 ``.selfcheck`` 里留一份镜像。写不到权威目录就失败，
    不把工作目录里的行当成已经测过。

    @param directory ``.selfcheck`` 目录，只用来计算权威钥匙。
    @param document 一次真正计时得到的配对记录，必须带两个 SHA。
    @return 权威证据文件路径。
    """
    pair = normalized_pair(str(document.get("sha_a") or ""), str(document.get("sha_b") or ""))
    path = host_evidence_path(directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    payload = dict(document)
    payload["schema"] = SCHEMA
    payload["host_evidence"] = True
    payload["sha_pair"] = list(pair)
    payload["changes_seed"] = False
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def load_host_pairs(directory: Path) -> set[tuple[str, str]]:
    """@brief 从宿主权威证据文件读出已经测过的 SHA 对。

    模型提名里的旗标不参与。工作目录里预先放好的同名文件也不参与。
    读不出来的证据行不算已覆盖。

    @param directory ``.selfcheck`` 目录，只用来计算权威钥匙。
    @return 规范化 SHA 对。
    """
    path = host_evidence_path(directory)
    found: set[tuple[str, str]] = set()
    if not path.is_file():
        return found
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict) or not row.get("applied") or row.get("host_evidence") is not True:
            continue
        try:
            found.add(normalized_pair(str(row.get("sha_a") or ""), str(row.get("sha_b") or "")))
        except PairedCompareError:
            continue
    return found


def host_select(
    directory: Path,
    *,
    feedback_text: str = "",
    rerun: Callable[[Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]] | None = None,
) -> dict:
    """@brief 读本场提名和宿主证据，选出至多一对；有选择时才重跑并保存测量。

    @param directory ``.selfcheck`` 目录。
    @param feedback_text 本场反馈原文。
    @param rerun 宿主重跑。没有合格提名时不会被调用。
    @return 选择记录，含实际复核数、未覆盖项和测量文档。
    """
    rows = apply_premise_citations(load_nominations(directory), feedback_text)
    selection = select_nominations(rows, host_pairs=load_host_pairs(directory))
    return run_selected(selection, rerun, evidence_dir=directory)


def load_nominations(directory: Path) -> list[dict]:
    """@brief 读出本场全部提名。坏行保留为未覆盖项，不悄悄丢掉。

    @param directory ``.selfcheck`` 目录。
    @return 提名字典列表。读不出来的行带 ``unreadable``。
    """
    path = directory / NOMINATION_NAME
    if not path.is_file():
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            rows.append({"unreadable": True, "detail": str(exc), "line": index})
            continue
        if not isinstance(row, dict):
            rows.append({"unreadable": True, "detail": "nomination is not an object", "line": index})
            continue
        row.setdefault("line", index)
        rows.append(row)
    return rows


def quota_debit(elapsed_s: float) -> dict:
    """@brief 占用一次自测配额的记账行。这一行不含两个候选 SHA。

    两个 SHA 只出现在 ``paired_comparisons.jsonl``。记账行交给现有的
    ``tcenv.selfcheck.append``，从而计入次数和 GPU 秒，但不是宿主证据。

    @param elapsed_s 这次比较的墙钟秒数。
    @return 不含 SHA 的记账字典。
    """
    if isinstance(elapsed_s, bool) or not isinstance(elapsed_s, (int, float)) or not math.isfinite(elapsed_s) or elapsed_s < 0:
        raise PairedCompareError("elapsed_s must be a non-negative finite number")
    return _quota_row({"status": "ok", "elapsed_s": float(elapsed_s)})


def _quota_row(document: Mapping[str, Any]) -> dict:
    """@brief 按 timed 自测的状态和耗时记一行。未知耗时不写成数字。"""
    status = str(document.get("status") or "").strip()
    reason = str(document.get("reason") or document.get("message") or "")
    if status not in {"ok", "timeout", "crash", "rejected", "error"}:
        lowered = reason.lower()
        if document.get("stage") == "timeout" or "timeout" in lowered or "deadline" in lowered:
            status = "timeout"
        elif document.get("applied"):
            status = "ok"
        else:
            status = "error"
    row = {
        "status": status,
        "kind": "paired_compare_quota",
        "note": "quota debit only; the two SHAs stay in paired_comparisons.jsonl and are not host evidence",
    }
    elapsed = document.get("elapsed_s", None)
    if "elapsed_s" in document and isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and math.isfinite(elapsed) and elapsed >= 0:
        row["elapsed_s"] = float(elapsed)
    else:
        row["elapsed"] = "unknown"
    return row


def record_nomination_and_debit(directory: Path, row: Mapping[str, Any], *, elapsed_s: float) -> dict:
    """@brief 先检查配额，再写提名，并用一条不含 SHA 的记账行占用配额。

    @param directory ``.selfcheck`` 目录。
    @param row 提名。
    @param elapsed_s 计入配额的秒数。
    @return 写入结果。配额已满时不写提名。
    @raises PairedCompareError 配额拒绝时抛出，且不写文件。
    """
    from tcenv.selfcheck import append, budget_reason, earlier_hang

    blocked = earlier_hang(directory) or budget_reason(directory)
    if blocked:
        raise PairedCompareError(blocked)
    path = append_nomination(directory, row)
    debit = quota_debit(elapsed_s)
    append(directory, debit)
    return {"nomination": str(path), "debit_has_sha": "sha_a" in debit or "sha_b" in debit, "elapsed_s": elapsed_s}


def record_supported_attempt(directory: Path, row: Mapping[str, Any], document: Mapping[str, Any]) -> dict:
    """@brief 支持的后端已经执行过一次，就按 timed 规则记账。

    成功、失败、超时和崩溃都占一次次数。``timeout`` / ``crash`` 进入既有挂死记录。
    没有可信耗时时写成 ``elapsed=unknown``，不补一个 GPU 秒数。

    @param directory ``.selfcheck`` 目录。
    @param row 提名。
    @param document 这次执行返回的测量或失败文档。
    @return 写入结果。
    """
    from tcenv.selfcheck import append

    path = append_nomination(directory, row)
    debit = _quota_row(document)
    append(directory, debit)
    return {
        "nomination": str(path),
        "debit_has_sha": "sha_a" in debit or "sha_b" in debit,
        "status": debit["status"],
        "elapsed": debit.get("elapsed_s", debit.get("elapsed")),
    }


def run_host_pair(
    path_a, path_b, *, sha_a: str, sha_b: str, seed: int, task: str, dtype: str, size: str,
    track_id: str = "msl-single", reps: int = 4,
) -> dict:
    """@brief 由宿主进程比较两份已经读入的候选。自测桥走这里，不让会话自己计时。

    @return 配对测量文档。
    """
    from tcenv.backends.metal.paired_compare import compare_for_backend

    return compare_for_backend(
        "metal", path_a=path_a, path_b=path_b, sha_a=sha_a, sha_b=sha_b, seed=int(seed),
        reps=int(reps), task=task, dtype=dtype, size=size, track_id=track_id,
    )
