"""Search decisions that do not call a model, touch a file, or use the GPU.

The live search and a resume replay both call these, so a stored report is judged by the same rule that judged it live.
"""

from __future__ import annotations

import math
import statistics


def candidate_eligible(report_ok: bool, speedup: float | None, *, correctness_only: bool) -> bool:
    """@brief 判断一个已测量的候选能否参与种子比较。

    ``report_ok`` 是报告自己的 ok 位（安全且数值正确）。计时搜索还需要加速比；只查正确性的搜索不需要。

    @param report_ok 报告自身的 ok 标记。
    @param speedup 候选的加速比；未计时时可为 ``None``。
    @param correctness_only 本次搜索是否只查正确性、不比较快慢。
    @return 可参与比较时为 ``True``。
    """
    if not report_ok:
        return False
    if correctness_only:
        return True
    return speedup is not None


def comparison_speedup(eligible: bool, speedup: float | None) -> float:
    """@brief 取比较时使用的加速比。

    不合格的候选按 0 算；合格但缺时间的按 1 算，既不奖励也不惩罚。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比；未计时时为 ``None``。
    @return 用于比较的浮点加速比。
    """
    if not eligible:
        return 0.0
    return 1.0 if speedup is None else float(speedup)


def session_budget_floor(costs: list[float] | tuple[float, ...]) -> float | None:
    """@brief 估算一次 agent 会话的费用，作为「还能不能再开一场」的下限。

    取最近三场已完成会话的中位数，刻意保守：钱不够跑完一场正常会话时，应当在再发一次请求前就停下，
    而不是开一场注定跑不完的会话。

    @param costs 已记录的各会话费用列表（按时间先后）。
    @return 最近至多三场正费用的中位数；没有任何历史时为 ``None``。
    """
    values = []
    for value in costs[-3:]:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0:
            values.append(number)
    return statistics.median(values) if values else None


def required_gain(min_improvement: float, noise_z: float, se_a: float, se_b: float) -> float:
    """@brief 算出一次测量要比另一快多少倍才算真的更快。

    取 ``min_improvement`` 与「1 加 ``noise_z`` 倍合并标准误」中的较大者。

    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @param se_a 一方测量的标准误。
    @param se_b 另一方测量的标准误。
    @return 需要的加速比门槛。
    """
    # A candidate is never faster by less than 1x, whatever the configured floor says. This also
    # keeps a negative or sub-1 min_improvement from promoting a slower kernel.
    return max(1.0, min_improvement, 1.0 + noise_z * math.hypot(se_a, se_b))


def compare_to_best(eligible: bool, speedup: float, best_speedup: float | None, gain: float) -> str:
    """@brief 比较候选与当前种子。

    ``best_speedup`` 为 ``None`` 表示还没有种子。不合格的候选总是 ``keep``；没有种子时 ``gain`` 不起作用。
    相等既不替换也不算平手：候选必须严格更快。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比。
    @param best_speedup 当前种子的加速比；没有种子时为 ``None``。
    @param gain 需要超出的倍数门槛。
    @return 候选成为新种子时为 ``replace``，快但在噪声内时为 ``tie``，其余为 ``keep``。
    """
    if not eligible:
        return "keep"
    if best_speedup is None or speedup > best_speedup * gain:
        return "replace"
    if speedup > best_speedup:
        return "tie"
    return "keep"


def relation_to_best(
    eligible: bool,
    speedup: float,
    best_speedup: float | None,
    se: float,
    best_se: float,
    *,
    min_improvement: float,
    noise_z: float,
) -> str:
    """@brief ``compare_to_best`` 的变体：先用两个标准误算出 ``required_gain``。

    还没有种子时忽略标准误，合格候选直接接替种子。

    @param eligible 候选是否合格。
    @param speedup 候选的加速比。
    @param best_speedup 当前种子的加速比；没有种子时为 ``None``。
    @param se 候选的标准误。
    @param best_se 种子的标准误。
    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @return ``replace``/``tie``/``keep`` 之一。
    """
    if best_speedup is None:
        return compare_to_best(eligible, speedup, None, min_improvement)
    gain = required_gain(min_improvement, noise_z, se, best_se)
    return compare_to_best(eligible, speedup, best_speedup, gain)


def in_confirm_pool(best_speedup: float, other_speedup: float, gain_of_best_over_other: float) -> bool:
    """@brief 判断当前最优是否没比 ``other`` 快出噪声。

    为真时两者都要在确认阶段复测，而不是直接拿当前最优发布。

    @param best_speedup 当前最优的加速比。
    @param other_speedup 另一颗候选的加速比。
    @param gain_of_best_over_other 最优相对另一颗的倍数门槛。
    @return 两者都应进确认池时为 ``True``。
    """
    return best_speedup <= other_speedup * gain_of_best_over_other


# 换一批输入会让同一颗核差 1–2%（taic/llm.py 的 confirm_batches 说明，docs/诚实说明.md）。
# 两颗确认中位数要比，标准误里加上这档的下沿，避免把不到 1% 的输入差当成变快。
# 同一颗核各批极差超过上沿时，公布层不把中位数当成精确值。
CONFIRM_INPUT_FLOOR = 0.01
CONFIRM_SPREAD = 0.02


def confirmation_se(within_rep_se: float, batch_floor: float = CONFIRM_INPUT_FLOOR) -> float:
    """@brief 把 repetitions 内标准误与换输入批次的下沿合并成一个标准误。

    @param within_rep_se 同一批输入重复测量得出的标准误。
    @param batch_floor 换一批输入带来的下沿，缺省为 ``CONFIRM_INPUT_FLOOR``。
    @return 两者平方和开根（负值先截到 0）。
    """
    return math.hypot(max(0.0, float(within_rep_se)), max(0.0, float(batch_floor)))


def batches_agree(values, spread: float = CONFIRM_SPREAD) -> bool:
    """@brief 判断各批输入的结果是否一致。

    @param values 各批测量的时间（微秒）；不是数值的项被忽略。
    @param spread 允许的最快/最慢批极差比例。
    @return 不足两批，或最快与最慢两批落在 ``spread`` 以内时为 ``True``。
    """
    nums = [float(v) for v in values or [] if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(nums) < 2:
        return True
    lo, hi = min(nums), max(nums)
    return lo > 0 and hi / lo <= 1.0 + spread


def choose_confirmed(measured: list[dict], *, min_improvement: float, noise_z: float) -> dict | None:
    """@brief 决定确认阶段可以指名哪一颗完整复测过的内核。

    每项带 ``median``、``se``（repetitions 内）与 ``search_best``。除非挑战者在把换输入下沿算进去后
    仍请过 ``required_gain``，否则搜索最优不变。搜索最优没有干净的复测时，最快的挑战者必须同时对每个
    其他挑战者都过这道门槛才能胜出。噪声分不开的一组返回 ``None``。

    @param measured 完整复测过的条目列表。
    @param min_improvement 搜索要求的最小提速倍数。
    @param noise_z 噪声倍数。
    @return 胜出的条目；分不开或列表为空时为 ``None``。
    """
    if not measured:
        return None

    def beats(faster: dict, slower: dict) -> bool:
        gain = required_gain(min_improvement, noise_z, confirmation_se(faster["se"]), confirmation_se(slower["se"]))
        return faster["median"] > slower["median"] * gain

    best = next((item for item in measured if item.get("search_best")), None)
    if best is not None:
        challengers = [item for item in measured if item is not best and beats(item, best)]
        if not challengers:
            return best
        ordered = sorted(challengers, key=lambda item: item["median"], reverse=True)
        top = ordered[0]
        if len(ordered) == 1 or all(beats(top, other) for other in ordered[1:]):
            return top
        return best
    ordered = sorted(measured, key=lambda item: item["median"], reverse=True)
    top = ordered[0]
    if len(ordered) == 1 or all(beats(top, other) for other in ordered[1:]):
        return top
    return None


def next_stagnant(replaced: bool, stagnant: int) -> int:
    """@brief 推进「连续没有新种子」的计数器。

    换掉种子就清零；落在噪声内的平手不会清零。

    @param replaced 本轮是否换掉了种子。
    @param stagnant 当前连续无进展的轮数。
    @return 更新后的计数。
    """
    return 0 if replaced else stagnant + 1


def patience_exhausted(stagnant: int, patience: int) -> bool:
    """@brief 判断连续无新种子的轮数是否已到耐心上限。

    @param stagnant 当前连续无进展的轮数。
    @param patience 允许的耐心轮数。
    @return 到达或超过上限时为 ``True``。
    """
    return stagnant >= patience


def feature_signature(source: str, *, backend_id: str) -> tuple:
    """@brief 提取源码的七个静态特征。

    不用编译器也不用 GPU；只有 Metal 后端实现了这些特征。后端不支持该能力时返回空元组。

    @param source 内核源码。
    @param backend_id 后端标识，决定是否支持特征签名。
    @return 七元组：matrix_unit、acc_bucket、tg_staging、double_buffer、register_prefetch、vector_load、online_softmax。
    """
    try:
        from tcenv.backends.registry import load_backend

        if not load_backend(backend_id).capabilities.supports_feature_signature:
            return ()
    except Exception:
        return ()
    from .hwfeatures import analyze_source

    from tcenv.backends.metal.launch import declared_stage_kernels

    launched = declared_stage_kernels(source)
    ev = analyze_source(source, kernel_names=launched)
    matrix = ev.get("matrix") or {}
    tiles = matrix.get("acc_tiles") or 0
    if not isinstance(tiles, int):
        tiles = 0
    if tiles >= 32:
        bucket = "32+"
    elif tiles >= 16:
        bucket = "16"
    elif tiles >= 4:
        bucket = "4"
    elif tiles >= 1:
        bucket = "1"
    else:
        bucket = "0"
    group = ev.get("threadgroup") or {}
    memory = ev.get("memory") or {}
    load = (memory.get("device_load") or {}).get("max_bytes") or 0
    online = ev.get("online_softmax") or {}
    return (
        bool(matrix.get("present")),
        bucket,
        bool(group.get("staging")),
        group.get("double_buffer") == "explicit",
        bool(group.get("register_prefetch")),
        int(load) >= 8,
        bool(online.get("present")),
    )


def _hamming(left: tuple, right: tuple) -> int:
    """@brief 逐位比较两个签名，返回不同位的个数。

    @param left 左侧特征签名。
    @param right 右侧特征签名。
    @return 不同位的个数。
    """
    return sum(a != b for a, b in zip(left, right))


def select_diverse_parent(candidates: list, best, *, task_name: str = "", max_archive: int = 4, backend_id: str = "metal"):
    """@brief 在归档里挑一颗与 ``best`` 在特征空间距离最远的合格内核。

    ``task_name`` 仅由调用方记录，不参与比较。归档最多保留 ``max_archive`` 个签名，每个签名由其最快的
    内核代表，慢于种子 85% 的候选直接丢掉。

    @param candidates 本轮收集到的候选列表。
    @param best 当前种子候选。
    @param task_name 任务名，仅记录、不参与比较。
    @param max_archive 归档保留的签名数上限。
    @param backend_id 后端标识，决定特征签名是否可用。
    @return 选中的候选；``best`` 为 ``None`` 或归档为空时为 ``None``。
    """
    del task_name
    if best is None:
        return None
    grouped: dict[tuple, object] = {}
    for cand in candidates:
        if not getattr(cand, "eligible", False) or getattr(cand, "recheck_rejected", False):
            continue
        if cand.source == best.source:
            continue
        if cand.speedup < 0.85 * best.speedup:
            continue
        sig = feature_signature(cand.source, backend_id=backend_id)
        kept = grouped.get(sig)
        if kept is None or cand.speedup > kept.speedup:
            grouped[sig] = cand
    archive = sorted(grouped.values(), key=lambda cand: cand.speedup, reverse=True)[:max_archive]
    if not archive:
        return None
    seed_sig = feature_signature(best.source, backend_id=backend_id)
    return max(archive, key=lambda cand: (_hamming(feature_signature(cand.source, backend_id=backend_id), seed_sig), cand.speedup, cand.label))


# ---------------------------------------------------------------------------------------------
# Confirmation identity: the same comparison for the live gate and the disk predicate.
# ---------------------------------------------------------------------------------------------

# The build fields that establish "this is the same built artifact".  ``artifact_set_digest`` is
# compared only when the dialect/report carries it; ``native_digest`` likewise (a simulator build
# has ``native_absent_reason`` instead).
BUILD_IDENTITY_FIELDS = (
    "candidate_sha256",
    "assembled_sha256",
    "opspec_digest",
    "lowering_profile_digest",
    "artifact_set_digest",
    "native_digest",
)

#: Build fields only a build can produce: the run has no current expectation for them, so the two
#: periods are compared to each other (never to a value copied out of one of them).
OUTCOME_ONLY_BUILD_FIELDS = frozenset({"assembled_sha256", "artifact_set_digest", "native_digest"})

# The evaluation fields that establish "this number was measured under the same semantics".
EVALUATION_IDENTITY_FIELDS = (
    "evaluation_fingerprint",
    "baseline_implementation_digest",
    "baseline_environment_digest",
    "protocol_version",
    "target_id",
    "backend",
    "backend_version",
)

IDENTITY_KINDS = ("match", "build_identity_mismatch", "build_identity_unknown", "evaluation_identity_mismatch", "evaluation_identity_unknown")

#: The caller states the comparison scope explicitly (plan §7.2).  ``local`` has no remote context
#: at all; ``remote`` compares the persisted outbox context (projection/row/validated result) as a
#: third party next to the two reports.  ``None`` cannot stand in for either.
IDENTITY_SCOPES = ("local", "remote")

#: The device-group components that identify the *execution* environment of a remote measurement.
DEVICE_GROUP_FIELDS = ("profile_digest", "topology_digest", "resource_generation")


def build_identity_view(report: dict | None) -> dict:
    """@brief 从 schema-4 报告投影出可比较的 build 身份视图（纯函数，不补字段）。

    A value that the report does not carry stays ``None``; the comparator treats it as *unknown*,
    never as equal and never as a reason to look somewhere else.

    @param report schema-4 报告字典；可以为 ``None``。
    @return ``{candidate_sha256, assembled_sha256, opspec_digest, lowering_profile_digest,
             artifact_set_digest, native_digest, provenance, shape, build_kind}``。
    """
    build = (report or {}).get("build") if isinstance(report, dict) else None
    if not isinstance(build, dict):
        return {field: None for field in BUILD_IDENTITY_FIELDS} | {"provenance": None, "shape": None, "build_kind": None}
    view = {field: build.get(field) for field in BUILD_IDENTITY_FIELDS}
    provenance = build.get("provenance")
    view["provenance"] = dict(provenance) if isinstance(provenance, dict) else None
    view["shape"] = build.get("shape")
    view["build_kind"] = build.get("build_kind")
    return view


def evaluation_identity_view(report: dict | None) -> dict:
    """@brief 从 schema-4 报告投影出评测身份视图（纯函数，不补字段）。

    @param report schema-4 报告字典；可以为 ``None``。
    @return 各评测身份字段；报告缺失/字段缺失时为 ``None``。
    """
    if not isinstance(report, dict):
        return {field: None for field in EVALUATION_IDENTITY_FIELDS + ("device_name", "gpu_family")}
    view = {field: report.get(field) for field in EVALUATION_IDENTITY_FIELDS}
    # Empty strings mean "the report did not state it"; that is an absent optional field, not a
    # value a comparison can use.
    for field in ("device_name", "gpu_family"):
        value = report.get(field)
        view[field] = value if isinstance(value, str) and value else None
    return view


def _flatten_mapping(value, prefix: str = "") -> dict:
    """Flatten nested provenance mappings into dotted field paths (scalars only)."""
    from collections.abc import Mapping

    flat: dict = {}
    if not isinstance(value, Mapping):
        return flat
    for key in value:
        item = value[key]
        path = f"{prefix}{key}"
        if isinstance(item, Mapping):
            flat.update(_flatten_mapping(item, path + "."))
        else:
            flat[path] = item
    return flat


def _identity_difference(domain: str, field: str, kind: str, side: str) -> dict:
    return {"domain": domain, "field": field, "kind": kind, "side": side}


def _compare_value(
    domain: str, field: str, *, expected, search, confirmation, differences: list,
    optional: bool = False, two_period: bool = False,
) -> None:
    """Record one field: mismatch per offending side, else unknown for a missing side.

    Two modes:

    * ``two_period`` (outcome-only fields: ``assembled_sha256``, ``artifact_set_digest``,
      ``native_digest`` — things only a build produces): there is no current expectation, so the
      search and confirmation reports are compared to each other.  Both present and different is
      a mismatch on *both* sides; one missing is unknown; both missing is not compared.
    * otherwise ``expected`` is the current, recomputable value.  ``None`` makes the field unknown
      (unless it is ``optional`` and absent everywhere); a missing side is unknown; a side whose
      value differs from ``expected`` is a mismatch on that side.
    """
    if two_period:
        if search is None and confirmation is None:
            return
        if search is None or confirmation is None:
            differences.append(_identity_difference(domain, field, "unknown", "search" if search is None else "confirmation"))
        elif search != confirmation:
            differences.append(_identity_difference(domain, field, "mismatch", "search"))
            differences.append(_identity_difference(domain, field, "mismatch", "confirmation"))
        return
    if expected is None:
        if search is None and confirmation is None and optional:
            return
        differences.append(_identity_difference(domain, field, "unknown", "expected"))
        return
    for side, value in (("search", search), ("confirmation", confirmation)):
        if value is None:
            differences.append(_identity_difference(domain, field, "unknown", side))
        elif value != expected:
            differences.append(_identity_difference(domain, field, "mismatch", side))


def remote_context_view(context) -> dict | None:
    """@brief 把一份远端上下文（已校验行/result 或固定 schema 投影）投影成可比较的字段。

    A pure reader: it accepts the ``CompletedEvaluationContext`` the outbox returns (``row`` +
    ``result``) or the fixed-schema projection, and never fills a missing field.  ``None`` means
    there is no remote context at all, which the comparator treats as *unknown*.

    @param context 远端上下文对象/字典；``None`` 表示没有。
    @return ``{candidate_digest, evaluator_epoch, device_group, build_provenance}`` 或 ``None``。
    """
    if context is None:
        return None
    row = getattr(context, "row", None)
    source = row if isinstance(row, dict) else (context if isinstance(context, dict) else None)
    if not isinstance(source, dict):
        return None
    # The row calls it ``candidate_digest``; the projection calls it ``candidate``.
    candidate = source.get("candidate_digest")
    if candidate is None:
        candidate = source.get("candidate")
    device_group = source.get("device_group")
    build_provenance = source.get("build_provenance")
    return {
        "candidate_digest": dict(candidate) if isinstance(candidate, dict) else candidate,
        "evaluator_epoch": source.get("evaluator_epoch"),
        "device_group": dict(device_group) if isinstance(device_group, dict) else None,
        "build_provenance": dict(build_provenance) if isinstance(build_provenance, dict) else None,
    }


def _device_group_value(view: dict | None, field: str):
    group = (view or {}).get("device_group")
    return group.get(field) if isinstance(group, dict) else None


def compare_confirmation_identity(
    *, candidate_sha256: str, expected: dict, search: dict | None, confirmation: dict | None,
    scope: str, search_context=None, confirmation_context=None,
) -> dict:
    """@brief 比较搜索报告与确认报告的 build/evaluation 身份，并与冻结的 expected 对照。

    A pure function: it reads the three documents it is given and nothing else.  It never fills a
    missing field from the current process, never opens a file and never compares handles or host
    paths.  The primary status follows "mismatch before unknown, build before evaluation, field
    path lexicographic", while ``differences`` keeps every finding.

    ``scope`` is explicit (plan §7.2): a ``local`` comparison must not receive a remote context,
    and a ``remote`` one treats a missing projection/row/result as *unknown* — ``None`` never
    stands in for "this was a local run".

    @param candidate_sha256 当前候选源码的 sha256（胜者）。
    @param expected 冻结的 expected：``{"build": {...}, "evaluation": {...}}``。
    @param search 搜索期报告的投影（``None`` 表示没有）。
    @param confirmation 确认期报告的投影（``None`` 表示没有）。
    @param scope ``local`` 或 ``remote``；必须显式给出。
    @param search_context 远端搜索期上下文（已校验行/result 或投影）；本地恒为 ``None``。
    @param confirmation_context 远端本批确认上下文；本地恒为 ``None``。
    @return ``{"kind": IDENTITY_KINDS..., "differences": [...]}``。
    @raises ValueError ``scope`` 非法，或本地比较却带了远端上下文时。
    """
    if scope not in IDENTITY_SCOPES:
        raise ValueError(f"scope must be one of {IDENTITY_SCOPES}, not {scope!r}")
    if scope == "local" and (search_context is not None or confirmation_context is not None):
        raise ValueError("a local identity comparison has no remote context")
    search_build = build_identity_view(search)
    confirm_build = build_identity_view(confirmation)
    search_eval = evaluation_identity_view(search)
    confirm_eval = evaluation_identity_view(confirmation)
    expected_build = dict((expected or {}).get("build") or {})
    expected_eval = dict((expected or {}).get("evaluation") or {})
    # The candidate itself is always current: a report that measured a different source is a
    # mismatch on the report's side, not an unknown.
    expected_build["candidate_sha256"] = candidate_sha256
    differences: list[dict] = []
    for field in BUILD_IDENTITY_FIELDS:
        _compare_value(
            "build", field, expected=expected_build.get(field),
            search=search_build.get(field), confirmation=confirm_build.get(field), differences=differences,
            optional=field in OUTCOME_ONLY_BUILD_FIELDS,
            two_period=field in OUTCOME_ONLY_BUILD_FIELDS,
        )
    # ``provenance`` is compared field by field so the difference paths stay point-wise.
    expected_provenance = expected_build.get("provenance") or {}
    search_provenance = search_build.get("provenance") or {}
    confirm_provenance = confirm_build.get("provenance") or {}
    # A full build always carries a provenance (structural validation guarantees it); comparing it
    # field by field keeps the difference paths point-wise and avoids one giant opaque mismatch.
    # ``artifact_set_digest`` is a build outcome; everything else is an executor/registry fact the
    # caller can recompute today.  When the executor cannot recompute its profile (no probe in
    # this process), the whole provenance is compared between the two periods instead of being
    # filled from the search report.
    expected_flat = _flatten_mapping(expected_provenance)
    search_flat = _flatten_mapping(search_provenance)
    confirm_flat = _flatten_mapping(confirm_provenance)
    for field in sorted(set(expected_flat) | set(search_flat) | set(confirm_flat)):
        _compare_value(
            "build", f"provenance.{field}", expected=expected_flat.get(field),
            search=search_flat.get(field), confirmation=confirm_flat.get(field), differences=differences,
            optional=field in ("toolchain.sdk_version",),
            # ``artifact_set_digest`` is a build outcome; every other provenance field is an
            # executor/registry fact.  When the executor has no current profile the whole
            # provenance is *unknown* (§7.2: an incomplete expected is unknown, never a
            # two-period-only comparison of the two reports' self-reported values).
            two_period=field == "artifact_set_digest",
        )
    for field in EVALUATION_IDENTITY_FIELDS + ("device_name", "gpu_family"):
        _compare_value(
            "evaluation", field, expected=expected_eval.get(field),
            search=search_eval.get(field), confirmation=confirm_eval.get(field), differences=differences,
            optional=field in ("device_name", "gpu_family"),
        )
    if scope == "remote":
        # The persisted outbox context is the third party for the fields a report cannot prove:
        # the CandidateDigest the evaluator bound, the device group / topology it ran on, the
        # evaluator epoch, and the request-level build document.  A missing context leaves them
        # unknown (F-10); it is never treated as a local run.
        #
        # The first three groups exist only in the NPU/build dialects.  A generic (Metal/source)
        # remote row has no CandidateDigest, no device group and no request-level build document,
        # and its candidate identity is already compared through ``candidate_sha256``.  Those
        # fields are therefore ``optional`` in the outcome-only sense: absent on *all* three sides
        # means not-applicable, while any side that carries one (an NPU row always does) turns a
        # missing expected into ``unknown`` — never into "not applicable".
        search_remote = remote_context_view(search_context)
        confirm_remote = remote_context_view(confirmation_context)
        _compare_value(
            "build", "candidate_digest",
            expected=expected_build.get("candidate_digest"),
            search=(search_remote or {}).get("candidate_digest"),
            confirmation=(confirm_remote or {}).get("candidate_digest"),
            differences=differences, optional=True,
        )
        expected_group = expected_build.get("device_group") or expected_eval.get("device_group")
        for field in DEVICE_GROUP_FIELDS:
            _compare_value(
                "build", f"device_group.{field}",
                expected=_device_group_value({"device_group": expected_group}, field),
                search=_device_group_value(search_remote, field),
                confirmation=_device_group_value(confirm_remote, field),
                differences=differences, optional=True,
            )
        _compare_value(
            "build", "evaluator_epoch",
            expected=expected_eval.get("evaluator_epoch"),
            search=(search_remote or {}).get("evaluator_epoch"),
            confirmation=(confirm_remote or {}).get("evaluator_epoch"),
            differences=differences,
        )
        _compare_value(
            "build", "build_provenance",
            expected=expected_build.get("build_provenance"),
            search=(search_remote or {}).get("build_provenance"),
            confirmation=(confirm_remote or {}).get("build_provenance"),
            differences=differences, optional=True,
        )
    mismatches = [item for item in differences if item["kind"] == "mismatch"]
    unknowns = [item for item in differences if item["kind"] == "unknown"]
    if mismatches:
        first = min(mismatches, key=lambda item: (0 if item["domain"] == "build" else 1, item["field"]))
        kind = "build_identity_mismatch" if first["domain"] == "build" else "evaluation_identity_mismatch"
    elif unknowns:
        first = min(unknowns, key=lambda item: (0 if item["domain"] == "build" else 1, item["field"]))
        kind = "build_identity_unknown" if first["domain"] == "build" else "evaluation_identity_unknown"
    else:
        kind = "match"
    differences.sort(key=lambda item: (item["domain"], item["field"], item["kind"], item["side"]))
    return {"kind": kind, "differences": differences}


CONFIRMATION_IDENTITY_STATUSES = (
    "build_identity_mismatch", "build_identity_unknown",
    "evaluation_identity_mismatch", "evaluation_identity_unknown",
)

#: The identity fields a *frozen* expected and the report it was frozen for must agree on.  The
#: outcome-only build digests are absent from every expected on purpose (they only exist after a
#: build).  ``protocol_version``/``target_id`` are required in both projections: the top-level pair
#: gets its dedicated mismatch codes in ``confirmation_publication``, and the ``evaluation`` copy is
#: required and compared here so a document that drops the pair on both sides cannot "agree".
_COMPARED_BUILD_FIELDS = ("candidate_sha256", "opspec_digest", "lowering_profile_digest")
_REQUIRED_EVALUATION_FIELDS = (
    "evaluation_fingerprint", "baseline_implementation_digest", "baseline_environment_digest",
    "backend", "backend_version", "protocol_version", "target_id",
)
#: Optional on *both* sides: the comparator skips them only when neither report nor expected states
#: one (#7.2 ``optional``).  Stating one on a single side is ``unknown``, never equal.
_OPTIONAL_EVALUATION_FIELDS = ("device_name", "gpu_family")

#: Provenance fields that only a build produces: compared between the two periods, never required
#: from the expected (§7.2, ``OUTCOME_ONLY_BUILD_FIELDS``).
_OUTCOME_ONLY_PROVENANCE_FIELDS = ("artifact_set_digest",)

#: The ``BuildProvenance`` paths a publishable report and its frozen expected must both state
#: (``contracts.build_provenance_document``'s closed field set, minus the outcome-only artifact
#: digest and the open ``compile_profile`` object compared as a whole).  Iterating only the union
#: of the two documents would call "both sides deleted it" agreement.
_REQUIRED_PROVENANCE_PATHS = (
    "schema_id", "builder_id", "builder_version", "build_host", "build_profile",
    "toolchain.id", "toolchain.version", "toolchain.sdk_version", "source_sha256",
)


def _identity_finding(domain: str, field: str, want, got, findings: list) -> None:
    """Record one expected-vs-report finding with the comparator's optional rules.

    ``want`` is the frozen expected's value, ``got`` the report's.  A required field needs both
    sides; an optional one may be absent on both (the caller only passes it here in that case).
    One-sided presence is ``unknown`` (the comparator's side-missing rule), never equal.
    """
    if want is None or got is None:
        findings.append((domain, field, "unknown"))
    elif want != got:
        findings.append((domain, field, "mismatch"))


def expected_report_identity(expected: dict | None, report: dict | None) -> str | None:
    """@brief 冻结 expected 与它绑定的报告自身身份是否一致且完整（纯函数）。

    The publication predicate used to accept a frozen expected that merely repeated
    ``protocol_version``/``target_id``, so two bypasses existed: a drifted expected whose identity
    contradicted the report the batches were compared against, and an **incomplete** expected
    (``evaluation_fingerprint``/baseline digests missing or ``None``) that still published because
    the batch records said ``match``.  Plan §7.2 says both are refusals: an expected that cannot
    state the identity is ``*_identity_unknown`` (the comparator's rule), and one that states a
    different value is ``*_identity_mismatch``.  Missing fields are never filled from the current
    environment.

    The rules mirror :func:`compare_confirmation_identity` exactly:

    * required build identity: ``candidate_sha256``/``opspec_digest``/``lowering_profile_digest``;
    * every provenance path either side states, except the outcome-only ``artifact_set_digest``
      (that one is compared between the two periods and never required from an expected);
    * required evaluation identity: the two baseline digests, the evaluation fingerprint, backend,
      backend version and ``protocol_version``/``target_id`` (the ``evaluation`` projection must
      state them too, so ``None == None`` can never count as agreement);
      ``device_name``/``gpu_family`` are optional on both sides.

    @param expected 冻结的 expected（``build``/``evaluation``）。
    @param report 确认报告文档（schema 4）。
    @return ``build_identity_mismatch``/``build_identity_unknown``/``evaluation_identity_mismatch``/
        ``evaluation_identity_unknown`` 或 ``None``（一致且完整）。
    """
    if not isinstance(expected, dict) or not isinstance(report, dict):
        return None
    expected_build = expected.get("build") if isinstance(expected.get("build"), dict) else {}
    expected_eval = expected.get("evaluation") if isinstance(expected.get("evaluation"), dict) else {}
    report_build = build_identity_view(report)
    report_eval = evaluation_identity_view(report)
    findings: list[tuple[str, str, str]] = []
    for field in _COMPARED_BUILD_FIELDS:
        _identity_finding("build", field, expected_build.get(field), report_build.get(field), findings)
    expected_provenance = expected_build.get("provenance")
    report_provenance = report_build.get("provenance")
    if not isinstance(expected_provenance, dict) or not isinstance(report_provenance, dict):
        # §7.2: a publishable report and its frozen expected both carry a complete
        # ``BuildProvenance``.  Iterating only the union of the two documents treated "both sides
        # deleted the provenance" as "nothing to compare" and published; an absent document is
        # ``build_identity_unknown``, never agreement.
        findings.append(("build", "provenance", "unknown"))
    else:
        expected_flat = _flatten_mapping(expected_provenance)
        report_flat = _flatten_mapping(report_provenance)
        # ``compile_profile`` is a bounded open object: compare it whole, and require it on both
        # sides so an emptied one cannot pass as "no paths to compare".
        _identity_finding(
            "build", "provenance.compile_profile",
            expected_provenance.get("compile_profile"), report_provenance.get("compile_profile"), findings,
        )
        for path in _REQUIRED_PROVENANCE_PATHS:
            _identity_finding(
                "build", f"provenance.{path}", expected_flat.get(path), report_flat.get(path), findings,
            )
        required = set(_REQUIRED_PROVENANCE_PATHS)
        for path in sorted((set(expected_flat) | set(report_flat)) - required):
            if path in _OUTCOME_ONLY_PROVENANCE_FIELDS or path.startswith("compile_profile."):
                continue
            _identity_finding(
                "build", f"provenance.{path}", expected_flat.get(path), report_flat.get(path), findings,
            )
    for field in _REQUIRED_EVALUATION_FIELDS:
        _identity_finding("evaluation", field, expected_eval.get(field), report_eval.get(field), findings)
    for field in _OPTIONAL_EVALUATION_FIELDS:
        want, got = expected_eval.get(field), report_eval.get(field)
        if want is None and got is None:
            continue
        _identity_finding("evaluation", field, want, got, findings)
    mismatches = [item for item in findings if item[2] == "mismatch"]
    unknowns = [item for item in findings if item[2] == "unknown"]
    for kind, group in (("mismatch", mismatches), ("unknown", unknowns)):
        if not group:
            continue
        domain, field, _kind = min(group, key=lambda item: (0 if item[0] == "build" else 1, item[1]))
        return f"{domain}_identity_{kind}"
    return None


def expected_drift_reason(frozen: dict | None, current: dict | None) -> str | None:
    """@brief finalize 时重算的 expected 是否已漂移出批次比较用的那份（纯函数）。

    The confirmation batches compare against one frozen expected.  Recomputing it when the winner
    is finalized is allowed only as a *check*: if the executor/registry identity moved, the frozen
    document is no longer the current one and publishing it would freeze an identity the batches
    never proved.  Any drift is an identity mismatch (fail closed), never a quiet refresh.

    @param frozen 各批实际比较所用的 expected。
    @param current finalize 时重算的 expected。
    @return ``build_identity_mismatch``/``evaluation_identity_mismatch`` 或 ``None``（未漂移）。
    """
    if not isinstance(frozen, dict) or not isinstance(current, dict):
        return "evaluation_identity_mismatch" if frozen != current else None
    frozen_build = frozen.get("build") if isinstance(frozen.get("build"), dict) else {}
    current_build = current.get("build") if isinstance(current.get("build"), dict) else {}
    if frozen_build != current_build:
        return "build_identity_mismatch"
    frozen_eval = frozen.get("evaluation") if isinstance(frozen.get("evaluation"), dict) else {}
    current_eval = current.get("evaluation") if isinstance(current.get("evaluation"), dict) else {}
    if frozen_eval != current_eval:
        return "evaluation_identity_mismatch"
    if (
        frozen.get("protocol_version") != current.get("protocol_version")
        or frozen.get("target_id") != current.get("target_id")
    ):
        return "evaluation_identity_mismatch"
    return None

# The reason codes the publication predicate may return.  They are the existing codes plus the
# confirmation-identity codes above; no free text ever enters a reason.
PUBLICATION_REASONS = (
    "no_result_json", "not_confirmed", "sha_mismatch", "label_mismatch", "values_mismatch",
    "speedup_mismatch", "candidate_us_mismatch", "protocol_mismatch", "target_mismatch",
    "timing_uncertain", "batches_disagree", "cache_hit", "missing_confirmation_block",
    "confirmation_ineligible", "confirmed",
) + CONFIRMATION_IDENTITY_STATUSES


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) > 0


def _publication(status: str, reason: str) -> dict:
    return {"allowed": status == "confirmed", "status": status, "reason": reason}


def _batch_ineligible(batch: dict) -> str | None:
    """Return the closed-set ineligibility reason for one confirmation batch, if any.

    A batch that may be published must have measured a *native, live* build: full shape, native
    build kind, measured provenance, no cache hit.  ``None``/absent fields are ineligible, not
    unknown — a confirmation that cannot show what it measured cannot be published.
    """
    if batch.get("shape") != "full":
        return "build_identity_unknown"
    if batch.get("build_kind") != "native":
        return "confirmation_ineligible"
    if batch.get("measurement_kind") != "measured":
        return "confirmation_ineligible"
    if batch.get("cache_hit") is not False:
        return "confirmation_ineligible"
    return None


def confirmation_publication(
    *, result: dict | None, confirm_report: dict | None, batches: list[dict] | None,
    expected: dict | None, bindings: dict,
) -> dict:
    """@brief 唯一最终发布谓词：结果、确认报告、逐批身份与冻结 expected 全部相符才发布。

    A pure function.  The caller supplies the binding group it derived from disk or from the live
    confirmation (``confirmation_source_sha256``, the confirm report's status, ``best.metal``'s
    hash and the winner kernel's hash); this function never reads a file, never re-probes the
    host, and never repairs a missing field.  The ``expected`` document is the *frozen* one from
    ``confirm_report["benchmark"]["confirmation"]["expected"]``: historical conclusions are not
    rewritten by today's library versions or device facts, and the frozen copy must still describe
    the report it was frozen for (:func:`expected_report_identity`) — a drifted or incomplete
    expected is an identity mismatch/unknown, not a confirmation.

    @param result ``result.json`` 的字典；可以为 ``None``。
    @param confirm_report 确认报告字典；可以为 ``None``。
    @param batches 每个确认批次的记录（label/identity/资格摘要）；可以为 ``None``。
    @param expected 冻结 expected（``build``/``evaluation``/``protocol_version``/``target_id``）。
    @param bindings ``{confirmation_source_sha256, confirm_report_status, best_source_sha256,
                      winner_source_sha256}``。
    @return ``{"allowed": bool, "status": 对外结论, "reason": 闭集原因码}``。
    """
    if not result:
        return _publication("in_progress", "no_result_json")
    if "confirmation_source_sha256" not in result:
        return _publication("旧版未绑定源码", "not_confirmed")
    recorded = result.get("confirmation_source_sha256")
    winner = result.get("confirm_winner_label")
    if result.get("confirmed_status") != "ok" or bindings.get("confirm_report_status") != "ok":
        return _publication("未确认", "not_confirmed")
    if (
        not isinstance(recorded, str) or recorded == ""
        or recorded != bindings.get("best_source_sha256")
        or recorded != bindings.get("winner_source_sha256")
    ):
        return _publication("未确认", "sha_mismatch")
    if not isinstance(winner, str) or winner == "":
        return _publication("未确认", "label_mismatch")
    value = result.get("confirmed_speedup")
    if not _finite_positive(value):
        return _publication("未确认", "not_confirmed")
    values = result.get("confirmed_values")
    if not isinstance(values, list) or not values:
        return _publication("未确认", "values_mismatch")
    numbers = [float(item) for item in values if isinstance(item, (int, float)) and not isinstance(item, bool)]
    if len(numbers) != len(values) or not all(_finite_positive(item) for item in numbers):
        return _publication("未确认", "values_mismatch")
    median = statistics.median(numbers)
    if abs(float(value) - median) > 1e-9 * max(1.0, abs(median)):
        return _publication("未确认", "speedup_mismatch")
    if not isinstance(confirm_report, dict):
        return _publication("未确认", "missing_confirmation_block")
    report_cached = confirm_report.get("cache_hit")
    if report_cached is True:
        return _publication("不确定", "cache_hit")
    if report_cached is not False:
        # The confirm report must *state* it was not served from cache (§7.2).  A per-batch
        # ``cache_hit: false`` cannot fill this in: a missing/``null``/non-boolean value is an
        # incomplete record, not a quiet confirmation.
        return _publication("未确认", "not_confirmed")
    benchmark = confirm_report.get("benchmark")
    confirmation = benchmark.get("confirmation") if isinstance(benchmark, dict) else None
    if not isinstance(confirmation, dict):
        return _publication("未确认", "missing_confirmation_block")
    # §7.2: the report's own build record must describe the candidate it measured.  The frozen
    # ``confirmation.kernel_sha256`` and the recorded source hash are not enough: a report whose
    # ``build.candidate_sha256`` names another candidate was not a measurement of this kernel, and
    # a confirmation report is a *fresh, full, native* measurement (§7.2), so a partial build or a
    # simulator build cannot be published even when the batch records say ``native``.
    report_build = confirm_report.get("build")
    if not isinstance(report_build, dict):
        return _publication("未确认", "confirmation_ineligible")
    if report_build.get("candidate_sha256") != recorded:
        return _publication("未确认", "sha_mismatch")
    if report_build.get("shape") != "full" or report_build.get("build_kind") != "native":
        return _publication("未确认", "confirmation_ineligible")
    if confirm_report.get("status") != "ok":
        return _publication("未确认", "confirmation_ineligible")
    if confirmation.get("kernel_sha256") != recorded:
        return _publication("未确认", "sha_mismatch")
    if confirmation.get("winner_label") != winner:
        return _publication("未确认", "label_mismatch")
    if confirmation.get("confirmed_values") != result.get("confirmed_values"):
        return _publication("未确认", "values_mismatch")
    if confirmation.get("speedup") != result.get("confirmed_speedup") or confirm_report.get("speedup") != result.get("confirmed_speedup"):
        return _publication("未确认", "speedup_mismatch")
    if result.get("candidate_us") is not None and confirmation.get("candidate_us") != result.get("candidate_us"):
        return _publication("未确认", "candidate_us_mismatch")
    if not isinstance(expected, dict) or not expected:
        # A run without the frozen expected cannot prove it was compared against the same
        # semantics; it is not a free pass to the current environment.
        return _publication("未确认", "not_confirmed")
    # §7.2: ``protocol_version``/``target_id`` are required identity on **both** sides.  A plain
    # ``!=`` comparison passes when both are missing or ``None`` (``None == None``), so a report and
    # a frozen expected that simply omit them would "agree" by accident.  Missing/None on either
    # side is ``unknown`` (the comparator's rule); only two stated, different values is a mismatch.
    # Today's constants never fill the gap, and the ``evaluation`` projection is checked by
    # ``expected_report_identity`` below.
    for field, reason in (("protocol_version", "protocol_mismatch"), ("target_id", "target_mismatch")):
        want, got = expected.get(field), confirm_report.get(field)
        if not isinstance(want, str) or not want or not isinstance(got, str) or not got:
            return _publication("未确认", "evaluation_identity_unknown")
        if want != got:
            return _publication("未确认", reason)
    # §7.2: the run's own ``result.target_id`` is part of the identity too.  Comparing only the
    # frozen expected with the report let a result that names another device publish, because the
    # two documents still agreed with each other.  Missing is unknown, not "matches".
    result_target = result.get("target_id")
    if not isinstance(result_target, str) or not result_target:
        return _publication("未确认", "evaluation_identity_unknown")
    if result_target != expected.get("target_id"):
        return _publication("未确认", "target_mismatch")
    # §7.2: the frozen expected must still describe the report it was frozen for, and it must state
    # the required identity.  A drifted or incomplete copy is an identity mismatch/unknown — a
    # batch record claiming ``match`` against it is self-contradictory, and missing fields are never
    # filled from today's environment.
    incoherent = expected_report_identity(expected, confirm_report)
    if incoherent is not None:
        return _publication("未确认", incoherent)
    from tcenv.bench import timing_is_uncertain

    if timing_is_uncertain(confirm_report):
        return _publication("不确定", "timing_uncertain")
    if not batches_agree(values):
        return _publication("不确定", "batches_disagree")
    if not isinstance(batches, list) or not batches:
        return _publication("未确认", "not_confirmed")
    # §7.2: an incomplete batch record is not publishable.  Every confirmed value is one measured
    # batch, so the records must pair with them one to one — a run that measured two batches and
    # recorded one identity (or one that counted a batch twice) cannot show what it published.
    if len(batches) != len(values):
        return _publication("未确认", "not_confirmed")
    # ...and *all* of them must belong to the winner: the count alone would let another
    # candidate's batch stand in for a missing winner batch.
    if any(not isinstance(batch, dict) or batch.get("label") != winner for batch in batches):
        return _publication("未确认", "not_confirmed")
    # Every batch of the *winner* must be recorded identically in ``result.confirmations``: one
    # place cannot claim a match the other does not have.  A candidate that failed its own
    # confirmation stays in ``confirmations`` as a failure record and is not published anyway.
    frozen = result.get("confirmations")
    if not isinstance(frozen, list) or not frozen:
        return _publication("未确认", "not_confirmed")
    by_label = {
        entry.get("label"): entry for entry in frozen
        if isinstance(entry, dict) and isinstance(entry.get("label"), str)
    }
    # A duplicated winner record would make ``by_label`` keep the last one and hide the first:
    # one candidate has one frozen confirmation entry, so more than one is a damaged document.
    if [entry.get("label") for entry in frozen if isinstance(entry, dict)].count(winner) > 1:
        return _publication("未确认", "not_confirmed")
    for batch in batches:
        entry = by_label.get(batch.get("label"))
        if not isinstance(entry, dict) or entry.get("identity") != batch.get("identity"):
            return _publication("未确认", "not_confirmed")
        # The per-candidate frozen entry is evidence too, not just a place to copy ``identity``
        # from.  A top-level ``confirmed_status=ok`` cannot overrule an entry that says this same
        # winner failed identity comparison (or still has failed batches).  Otherwise a damaged
        # result with contradictory records would publish through the read-disk gate even though
        # the live gate had refused it.
        entry_status = entry.get("status")
        if entry_status in CONFIRMATION_IDENTITY_STATUSES:
            return _publication("未确认", entry_status)
        if entry_status != "ok":
            return _publication("未确认", "not_confirmed")
        failed_batches = entry.get("failed_batches")
        if type(failed_batches) is not int or failed_batches != 0:
            return _publication("未确认", "not_confirmed")
        timing = batch.get("timing_uncertain")
        if timing is True:
            return _publication("不确定", "timing_uncertain")
        if timing is not False:
            # The field is required: ``None``/absent/any non-boolean is an incomplete record, not a
            # quiet batch.  Only an explicit ``False`` says "the timing was stable" (plan §7.2).
            return _publication("未确认", "not_confirmed")
        kind = (batch.get("identity") or {}).get("kind") if isinstance(batch.get("identity"), dict) else None
        if kind in CONFIRMATION_IDENTITY_STATUSES:
            return _publication("未确认", kind)
        if kind != "match":
            return _publication("未确认", "not_confirmed")
        ineligible = _batch_ineligible(batch)
        if ineligible is not None:
            return _publication("未确认", ineligible)
    labels = [batch.get("label") for batch in batches]
    if winner not in labels:
        return _publication("未确认", "label_mismatch")
    return _publication("confirmed", "confirmed")
