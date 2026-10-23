"""Randomized differential testing against a float64/float32 CPU reference.

A candidate must match the reference within

    |out - ref| <= rtol * |ref| + atol_rel * scale        (element-wise)

where ``scale`` defaults to the RMS of the reference (or a task-provided error scale such as
sum|x| for reductions). Mirroring the paper's ``max(tol_dtype, k * baseline_err)`` idea, the
allowed *scaled* error is relaxed to ``k`` times the error of the strongest baseline when that
baseline is itself less accurate than the nominal tolerance (capped, so that a broken baseline
cannot make the test vacuous).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Callable

import numpy as np
import torch

from ...distributions import generator
from .runtime import CompiledKernel, MetalContext, MetalRuntimeError
from .sanitize import ArgSet, POISON_PATTERN
from ...dtypes import FLOAT_MAX, FLOAT_TINY_SUBNORMAL, get_dtype
from ...task import Task
from ...tensors import from_raw, round_to, to_raw

K_BASELINE = 2.0
ALLOW_CAP = 8.0


@dataclass
class CaseData:
    case: str
    seed: int
    rounded: dict[str, torch.Tensor]
    raw: dict[str, bytes]
    ref: dict[str, torch.Tensor]
    scale: dict[str, torch.Tensor]
    input_sha256: str = ""


IEEE_CASE = "ieee"
IEEE_SPECIALS = (float("nan"), float("inf"), float("-inf"), -0.0)


def inject_specials(t: torch.Tensor, seed: int, max_n: int | None = None) -> torch.Tensor:
    """@brief 复制张量并在随机位置植入 NaN、±Inf 与 -0.0。

    @param t 源张量。
    @param seed 随机种子。
    @param max_n 植入个数的上限；省略时按张量规模推导。
    @return 植入特殊值后的新张量。

    Copy of ``t`` with NaN, +Inf, -Inf and -0.0 planted at random positions (always including the first and the last element).

    About one row in ten is affected (at least 8 and at most 256 positions per tensor), so that most of a row-wise or
    matrix output stays finite and checks the values that must *not* be disturbed by a special elsewhere. ``max_n`` caps the
    count for operands where one special poisons a whole slice of the output (the weights of a convolution: one NaN weight
    makes every pixel of that output channel NaN).
    """
    flat = t.reshape(-1).clone()
    numel = flat.numel()
    last = t.shape[-1] if t.dim() else 1
    rows = max(1, numel // max(1, last))
    n = int(min(256, max(8, 0.1 * rows, numel / 65536), max(1, numel // 4)))
    if max_n is not None:
        n = max(2, min(n, max_n))
    perm = torch.randperm(numel, generator=generator(seed))
    perm = perm[(perm != 0) & (perm != numel - 1)]
    pos = torch.cat([torch.tensor([0, numel - 1]), perm[: n - 2]])[:n]  # first/last element always included, no duplicates
    kinds = torch.randperm(n, generator=generator(seed + 1)) % len(IEEE_SPECIALS)
    flat[pos] = torch.tensor([IEEE_SPECIALS[int(k)] for k in kinds], dtype=flat.dtype)
    return flat.reshape(t.shape)


def _finite(t: torch.Tensor) -> torch.Tensor:
    """@brief 把非有限元素换成 0。

    @param t 输入张量。
    @return 替换后的张量；整数张量原样返回。
    """
    return torch.where(torch.isfinite(t), t, torch.zeros_like(t)) if t.is_floating_point() else t


def prepare_case(task: Task, case: str, seed: int) -> CaseData:
    """@brief 生成一个算例的输入、参考输出与误差尺度。

    @param task 任务对象。
    @param case 算例名（分布名、``ieee`` 或 ``domain``）。
    @param seed 随机种子。
    @return CaseData 实例。
    @raises ValueError 当全域扫描用例的输入不符合要求，或参考输出超出数据类型量程。
    """
    from .domain import DOMAIN_CASE, domain_input, tensor_sha256

    # 全域扫描不走采样分布，也不走 RMS 缩放和量程守卫。65504 的 relu 参考值就是该类型的最大值，守卫会把它误判成 harness bug。
    if case == DOMAIN_CASE:
        inputs = task.contract.inputs
        if task.dtype != "fp16" or len(inputs) != 1:
            raise ValueError(f"domain sweep needs one fp16 input, got {task.name} {task.dtype}")
        arg = inputs[0]
        values = domain_input(arg.shape[0], seed)
        rounded = {arg.name: values}
        raw = {arg.name: to_raw(values, arg.dtype)}
        ref = {k: v.reshape(task.contract.arg(k).shape) for k, v in task.reference(rounded).items()}
        scale = {k: torch.zeros((), dtype=torch.float64) for k in ref}
        return CaseData(case, seed, rounded, raw, ref, scale, tensor_sha256(values))

    ieee = case == IEEE_CASE
    logical = task.gen_inputs(task.meta.get("ieee_base", "normal") if ieee else case, seed)
    rounded: dict[str, torch.Tensor] = {}
    raw: dict[str, bytes] = {}
    for j, a in enumerate(task.contract.inputs):
        t = logical[a.name].reshape(a.shape)
        if ieee and get_dtype(a.dtype).is_float and a.name in task.meta.get("ieee_inputs", (a.name,)):
            t = inject_specials(t, seed + 104_729 * (j + 1), task.meta.get("ieee_max", {}).get(a.name))
        rounded[a.name] = round_to(t, a.dtype)
        raw[a.name] = to_raw(rounded[a.name], a.dtype)
    ref = {k: v.reshape(task.contract.arg(k).shape) for k, v in task.reference(rounded).items()}
    for k, v in ref.items():
        d = get_dtype(task.contract.arg(k).dtype)
        top = float(_finite(v).abs().max())
        if d.is_float and top > 0.95 * FLOAT_MAX[d.name]:
            raise ValueError(
                f"harness bug: reference output '{k}' of case '{case}' exceeds the {d.name} range "
                f"(max |ref| = {top:.3g}); adjust the input distribution of task {task.name}"
            )
    # Error scales describe the magnitude of the *finite* part: NaN / Inf outputs are compared exactly (see scaled_error).
    s_in, s_ref = ({k: _finite(v) for k, v in x.items()} if ieee else x for x in (rounded, ref))
    if task.error_scale is not None:
        scale = task.error_scale(s_in, s_ref)
    else:
        scale = {k: torch.sqrt(torch.mean(v.to(torch.float64) ** 2)) for k, v in s_ref.items()}
    return CaseData(case, seed, rounded, raw, ref, scale)


def scaled_error(out: torch.Tensor, ref: torch.Tensor, scale, rtol: float, atol_rel: float, exact: bool, floor: float = 0.0) -> torch.Tensor:
    """@brief 计算按容差归一化后的逐元素误差。

    @param out 候选输出。
    @param ref 参考输出。
    @param scale 误差尺度。
    @param rtol 相对容差。
    @param atol_rel 相对尺度的绝对容差。
    @param exact 是否要求逐位一致。
    @param floor 输出格式本身的最小间隔（亚正规数半间距）。
    @return 归一化误差张量，容差内不超过 1。

    ``floor`` is an absolute slack for the output format itself (half the spacing of the smallest
    subnormals): the unrounded reference is not representable below that granularity.

    Special values must agree exactly: NaN where the reference is NaN (and only there), and an infinity of the same sign
    where the reference is infinite."""
    o = out.to(torch.float64)
    r = ref.to(torch.float64)
    if exact:
        return torch.where(o != r, torch.tensor(float("inf"), dtype=torch.float64), torch.tensor(0.0, dtype=torch.float64))
    scale = torch.as_tensor(scale, dtype=torch.float64)
    tol = rtol * r.abs() + atol_rel * scale + floor
    diff = (o - r).abs()
    same_inf = torch.isinf(o) & torch.isinf(r) & (torch.sign(o) == torch.sign(r))
    same_nan = torch.isnan(o) & torch.isnan(r)
    diff = torch.where(same_inf | same_nan, torch.zeros_like(diff), diff)
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, float("inf")), diff)
    return diff / tol.clamp(min=1e-300).nan_to_num(nan=1.0, posinf=1.0)


def _floor(dtype: str) -> float:
    """@brief 返回该数据类型在亚正规区的最小间隔余量。

    @param dtype 数据类型名。
    @return 绝对余量；未知类型为 0。
    """
    return 0.51 * FLOAT_TINY_SUBNORMAL.get(dtype, 0.0)


@dataclass
class CaseResult:
    case: str
    seed: int
    ok: bool
    scaled_err: float
    allow: float
    baseline_scaled_err: float | None
    n_bad: int
    n_elems: int
    nan_count: int
    unwritten: int
    worst: dict | None
    input_sha256: str = ""
    safety: list[str] = field(default_factory=list)
    message: str = ""

    def to_json(self) -> dict:
        """@brief 转成可写入报告的字典。

        @return 字段字典，无穷大误差写成 ``"inf"``。
        """
        d = asdict(self)
        for k in ("scaled_err", "allow"):
            if d[k] == float("inf"):
                d[k] = "inf"
        return d


def _unwritten_count(raw: bytes, nbytes_per_elem: int) -> int:
    """@brief 统计仍保持毒化图案、从未被写过的输出元素个数。

    @param raw 输出原始字节。
    @param nbytes_per_elem 单元素字节数。
    @return 未被写入的元素个数。
    """
    a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, nbytes_per_elem)
    return int((a == POISON_PATTERN).all(axis=1).sum())


def check_case(
    task: Task,
    data: CaseData,
    out_raw: dict[str, bytes],
    baseline_outs: dict[str, dict[str, torch.Tensor]] | None,
) -> CaseResult:
    """@brief 把候选输出与参考输出比较，给出单个算例的结论。

    @param task 任务对象，提供容差与精确比较开关。
    @param data 算例数据。
    @param out_raw 输出名到原始字节的映射。
    @param baseline_outs 基线名到输出字典的映射；可为 None。
    @return CaseResult 结论。
    """
    from ...dtypes import get_dtype

    worst_all = 0.0
    n_bad_total = 0
    n_total = 0
    nan_total = 0
    unwritten_total = 0
    worst_info = None
    per_out = []
    base_err = None
    if baseline_outs:
        errs = []
        for name, bo in baseline_outs.items():
            e = 0.0
            for a in task.contract.outputs:
                s = scaled_error(bo[a.name], data.ref[a.name], data.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype))
                e = max(e, float(s.max()))
            errs.append(e)
        base_err = min(errs)  # the most accurate baseline defines how well the op *can* be done
    allow = 1.0
    # 基线自己的误差可以放宽抽样题。全域扫描不放宽，否则一个同样出错的基线会把错值放过去。
    if data.case != "domain" and base_err is not None and not task.exact:
        allow = min(ALLOW_CAP, max(1.0, K_BASELINE * base_err))

    for a in task.contract.outputs:
        d = get_dtype(a.dtype)
        got = from_raw(out_raw[a.name], a.dtype, a.shape)
        s = scaled_error(got, data.ref[a.name], data.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype))
        n_total += s.numel()
        nan_total += int(torch.isnan(got.to(torch.float64)).sum()) if d.is_float else 0
        unwritten_total += _unwritten_count(out_raw[a.name], d.nbytes)
        bad = s > allow
        n_bad_total += int(bad.sum())
        m = float(s.max())
        if m > worst_all or worst_info is None:
            idx = int(torch.argmax(s))
            pos = tuple(int(i) for i in np.unravel_index(idx, tuple(a.shape))) if a.shape else ()
            worst_info = {
                "output": a.name,
                "index": list(pos),
                "got": float(got.reshape(-1)[idx]),
                "want": float(data.ref[a.name].reshape(-1)[idx]),
                "scaled_err": m if m != float("inf") else "inf",
            }
            if data.case == "domain" and task.contract.inputs:
                iv = data.rounded[task.contract.inputs[0].name].reshape(-1)[idx]
                bits = int(iv.view(torch.int16).item()) & 0xFFFF
                worst_info["input_bits"] = f"0x{bits:04x}"
        worst_all = max(worst_all, m)
        per_out.append((a.name, m))

    ok = n_bad_total == 0
    msg = ""
    if not ok:
        parts = [f"{n_bad_total}/{n_total} elements out of tolerance (worst scaled error {worst_all:.3g}, allowed {allow:.3g})"]
        if unwritten_total:
            parts.append(f"{unwritten_total} output elements still hold the NaN poison pattern, i.e. were never written")
        if nan_total and nan_total != unwritten_total:
            parts.append(f"{nan_total} NaN outputs")
        if worst_info:
            parts.append(f"worst: {worst_info['output']}{worst_info['index']} got {worst_info['got']:.6g} want {worst_info['want']:.6g}")
        msg = "; ".join(parts)
    return CaseResult(
        case=data.case,
        seed=data.seed,
        ok=ok,
        scaled_err=worst_all,
        allow=allow,
        baseline_scaled_err=base_err,
        n_bad=n_bad_total,
        n_elems=n_total,
        nan_count=nan_total,
        unwritten=unwritten_total,
        worst=worst_info,
        message=msg,
        input_sha256=data.input_sha256,
    )


def launch_once(ctx: MetalContext, kernel: CompiledKernel, task: Task, argset: ArgSet, *, poison: bool = True, fill_zero: bool = False) -> dict[str, bytes]:
    """@brief 执行一次候选核，并按需毒化或清零输出。

    @param ctx Metal 上下文。
    @param kernel 已编译的候选核。
    @param task 任务对象。
    @param argset 绑定参数集合。
    @param poison 是否用毒化图案填充输出。
    @param fill_zero 是否改用全零初值（确定性对照）。
    @return 输出名到字节串的映射。
    """
    if fill_zero:
        for a in task.contract.outputs:
            argset.buffers[a.name].mem[argset.buffers[a.name].guard : argset.buffers[a.name].guard + a.nbytes] = bytes(a.nbytes)
    elif poison:
        argset.poison_outputs()
    ctx.run(kernel, [argset.args()], task.contract.grid, task.contract.threads)
    return argset.read_outputs()


def run_correctness(
    ctx: MetalContext,
    kernel: CompiledKernel,
    task: Task,
    argset: ArgSet,
    seed: int,
    *,
    cases: tuple[str, ...] | None = None,
    baseline_fns: Callable[[CaseData], dict[str, dict[str, torch.Tensor]] | None] | None = None,
    determinism: bool = True,
    fail_fast: bool = True,
    launcher=None,
) -> tuple[list[CaseResult], list[str], list[str]]:
    """@brief 跑完一个算例列表，汇总正确性、安全与确定性结论。

    @param ctx Metal 上下文。
    @param kernel 已编译的候选核。
    @param task 任务对象。
    @param argset 绑定参数集合。
    @param seed 随机种子。
    @param cases 要跑的算例名；省略时用任务自带的算例。
    @param baseline_fns 生成基线输出的回调。
    @param determinism 是否做多次启动的一致性对照。
    @param fail_fast 首个失败算例后是否立即停止。
    @param launcher 自定义单次启动函数；省略时用 launch_once。
    @return (算例结果, 安全问题, 非确定性问题) 三元组。

    Returns (case results, safety problems, determinism problems).
    """
    results: list[CaseResult] = []
    safety: list[str] = []
    nondet: list[str] = []
    cases = cases or task.cases
    for i, case in enumerate(cases):
        try:
            from tcenv.events import update_progress

            update_progress(stage="correctness", case=case, cases_done=i, cases_total=len(cases))
        except Exception:
            pass
        data = prepare_case(task, case, seed + 7919 * (i + 1))
        argset.reset_guards_all()
        argset.load_inputs(data.raw)
        out1 = launch_once(ctx, kernel, task, argset) if launcher is None else launcher(argset, poison=True, fill_zero=False)
        s = argset.safety_violations()
        base = baseline_fns(data) if baseline_fns else None
        res = check_case(task, data, out1, base)
        if s:
            res.safety = s
            safety.extend(f"[{case}] {m}" for m in s)
        results.append(res)
        if determinism and res.ok and not s:  # on every case: a race or an output-dependence can hide behind one input distribution
            out2 = launch_once(ctx, kernel, task, argset, fill_zero=True) if launcher is None else launcher(argset, poison=False, fill_zero=True)
            if out2 != out1:
                nondet.append(
                    "outputs depend on the initial contents of the output buffers (a run with zero-filled outputs differs from a NaN-filled run); "
                    "the kernel must write every output element and must not read or accumulate into outputs"
                )
            else:
                out3 = launch_once(ctx, kernel, task, argset) if launcher is None else launcher(argset, poison=True, fill_zero=False)
                if out3 != out1:
                    nondet.append("results differ between identical launches (data race or unordered atomics)")
        if fail_fast and (not res.ok or s or nondet):
            break
    return results, safety, nondet
