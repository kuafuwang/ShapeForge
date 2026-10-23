"""Extra sentences the search can append to a planner or repair prompt. Every helper returns "" or None when it has nothing to say,
so a switch that is off never changes the text the model sees.

The roofline lower bound uses this algorithm's operation-count estimate and a measured peak. A baseline that is faster than that bound
shows the bound does not apply. The reason can be a different algorithm, the operation count, the peak calibration, or the timing method.
It is not evidence that one particular algorithm, such as Winograd, was used.
"""

from __future__ import annotations

import re
from typing import Any

# A baseline a few percent under the bound is measurement noise (elementwise kernels sit on the bandwidth roof).
# Conv2d's torch-mps baseline is about 17% under the direct-convolution bound, which is a different algorithm.
CLEAR_OF_BOUND = 0.95

_WS = re.compile(r"\s+")
_NUM = re.compile(r"\d+")
_TG = re.compile(r"(\d+)\s*B exceeds the device limit of (\d+)\s*B")


def _report(report: Any) -> dict:
    """@brief 把报告统一成字典。

    @param report 报告对象；可以是字典、带 ``to_json`` 方法的数据类，或 ``None``。
    @return 报告字典；无法转换时为空字典。
    """
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    return report.to_json() if hasattr(report, "to_json") else {}


def _roof_and_baselines(report: Any) -> tuple[dict, dict]:
    """@brief 取出报告里的 roofline 与基线时间两部分。

    @param report 报告对象。
    @return ``(roofline, baseline_us)``；缺失或类型不对的那部分为空字典。
    """
    rep = _report(report)
    roof = rep.get("roofline") or {}
    bases = rep.get("baseline_us") or {}
    return roof if isinstance(roof, dict) else {}, bases if isinstance(bases, dict) else {}


def baseline_faster_than_bound(report: Any) -> tuple[str, float, float] | None:
    """@brief 判断最快的基线是否已经比 roofline 下界快。

    @param report 评测报告。
    @return 基线比下界快时为 ``(基线名, 其微秒数, 下界)``；否则为 ``None``。
    """
    roof, bases = _roof_and_baselines(report)
    lb = roof.get("lower_bound_us")
    usable = {k: float(v) for k, v in bases.items() if isinstance(v, (int, float)) and v > 0}
    if not isinstance(lb, (int, float)) or lb <= 0 or not usable:
        return None
    name, us = min(usable.items(), key=lambda kv: kv[1])
    return (name, us, float(lb)) if us < float(lb) * CLEAR_OF_BOUND else None


def bound_beaten_line(report: Any) -> str | None:
    """@brief 硬件特性块里的一句话：下界在这里不成立。

    比 ``target_note`` 短，后者给规划器看。

    @param report 评测报告。
    @return 一句话说明；下界仍成立时为 ``None``。
    """
    beat = baseline_faster_than_bound(report)
    if beat is None:
        return None
    name, us, lb = beat
    return (
        f"The roofline lower bound ({lb:.0f} us) does not hold here: {name} is already at {us:.0f} us. "
        f"The bound's assumptions (operation count, measured peak, or timing) do not apply. The reachable target is {us:.0f} us, not the bound."
    )


def target_note(report: Any) -> str | None:
    """@brief 基线已经打过下界时，给规划器的纠正说明。

    @param report 评测报告。
    @return 纠正文本；下界仍成立时为 ``None``。
    """
    beat = baseline_faster_than_bound(report)
    if beat is None:
        return None
    name, us, lb = beat
    return (
        f"The roofline lower bound ({lb:.1f} us) assumes this algorithm's operation count and a measured peak, and it does not hold for this task: "
        f"the fastest baseline ({name}, {us:.1f} us) is already faster than that bound. "
        f"That can come from a different algorithm, a different operation-count estimate, the peak calibration, or the timing method. "
        f"It is not evidence for one particular algorithm. "
        f"The measured baseline time is {us:.1f} us. A kernel near the bound can still be slower than the baseline."
    )


def at_roofline(report: Any, required_gain: float) -> bool:
    """@brief 启发式判断：种子是否已经贴近估算下界。

    条件是：没有基线打过下界、计时安静（``noisy`` 为 False）、标定存在，且余量在 ``required_gain`` 之内。
    基线已快于下界、测量有噪声、或缺实测峰值时都不触发。触发也不代表内核已达物理最优。

    @param report 评测报告。
    @param required_gain 本轮要求的加速比门槛。
    @return 可以早停时为 ``True``。
    """
    if baseline_faster_than_bound(report) is not None:
        return False
    rep = _report(report)
    bench = rep.get("benchmark") or {}
    if not isinstance(bench, dict) or bench.get("noisy") is not False:
        return False
    roof, _ = _roof_and_baselines(report)
    if not isinstance(roof.get("peak_gbps_measured"), (int, float)) or roof.get("peak_gbps_measured") <= 0:
        return False
    gap = roof.get("time_over_lower_bound")
    if not isinstance(gap, (int, float)) or gap <= 0 or not isinstance(required_gain, (int, float)):
        return False
    # 种子已经快过参考算法下界的 0.95 倍，说明运算量假设不成立。有启动说明本身不停用这条早停。
    if float(gap) < CLEAR_OF_BOUND:
        return False
    for key in ("compute_fraction", "bandwidth_fraction"):
        frac = roof.get(key)
        if isinstance(frac, (int, float)) and not isinstance(frac, bool) and float(frac) > 1.0 / CLEAR_OF_BOUND:
            return False
    return float(gap) <= float(required_gain)


def abi_signature(message: str) -> str:
    """@brief 把失败信息里的数字擦掉，得到 ABI 失败的签名。

    这样同一个限额违规在多次修复尝试之间可以比较相等。

    @param message 失败信息。
    @return 数字被替换为 ``#``、空白被折叠后的字符串。
    """
    return _NUM.sub("#", _WS.sub(" ", message or "").strip())


def threadgroup_limit_note(message: str) -> str | None:
    """@brief 内核声明的静态 threadgroup 内存超过设备上限时，加进修复提示的一句话。

    @param message 失败信息。
    @return 提示句；信息里没有这类限额违规时为 ``None``。
    """
    m = _TG.search(message or "")
    if not m:
        return None
    declared, limit = m.group(1), m.group(2)
    return (
        f"Static threadgroup memory must be under {limit} B. This kernel declares {declared} B. "
        f"The next kernel must declare less than {limit} B."
    )


def failure_phrase(status: str, message: str) -> str:
    """@brief 为「这一支没交出合格内核」生成一行历史记录。

    只保留一句短话，信息截到 160 字符。

    @param status 报告状态，如 ``abi_error``/``incorrect``/``compile_error``/``unsafe``。
    @param message 失败信息。
    @return 形如 ``kind: message`` 的一行；信息为空时只有 ``kind``。
    """
    msg = _WS.sub(" ", message or "").strip()
    msg = msg[:160]
    kind = {"abi_error": "abi", "incorrect": "incorrect", "compile_error": "compile", "unsafe": "unsafe"}.get(status, status or "failed")
    return f"{kind}: {msg}" if msg else kind


def status_lines(*, rnd: int, max_rounds: int, patience_left: int, spent: float, budget: float | None, required_gain: float) -> str:
    """@brief 告诉模型什么才算赢、还剩多少搜索空间的两行文本。

    @param rnd 当前轮次。
    @param max_rounds 本轮运行允许的最大轮数。
    @param patience_left 还剩几轮耐心。
    @param spent 已花美元。
    @param budget 总预算；``None`` 或 0 时不显示上限。
    @param required_gain 本轮要求的加速比门槛。
    @return 两行状态文本。
    """
    pct = max(0.0, (float(required_gain) - 1.0) * 100.0)
    money = f"spent ${spent:.2f}" + (f" of ${budget:.2f}" if budget else "")
    return (
        f"A candidate replaces this kernel only when it is at least {pct:.1f}% faster: that is the noise floor of this comparison, and a smaller gain does not count.\n"
        f"Round {rnd}/{max_rounds}, {patience_left} round(s) of patience left, {money}."
    )
