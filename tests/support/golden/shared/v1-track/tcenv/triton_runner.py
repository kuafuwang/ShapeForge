"""Runs a task's *Triton specification* through a Metal Triton backend (triton-msl) and times it.

Executed by the python of an isolated venv (``.venvs/triton-msl``) that has triton + triton-msl; the main
environment never imports Triton:  ``python -m tcenv.triton_runner request.json``.

Why this exists: the paper's baseline is the conventional compiler's output for the same Triton source. Upstream Triton has no
Metal backend, so the closest thing is a community backend. The kernel text shown to the LLM is real Triton (the launch
configuration is in its header comment), so it can be executed here unchanged.

Protocol (same as the main environment, see tcenv.bench): candidate-vs-vendor numbers are measured in the *same process*,
interleaved, so the result is the ratio ``triton / mlx`` (and ``triton / torch_mps``), which the main evaluation scales by its own
live MLX measurement. Like the paper's "autotuned Triton", the number of warps is swept and the fastest *numerically correct*
configuration is kept.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

NUM_WARPS_SWEEP = (1, 2, 4, 8)


def parse_launch_header(src: str) -> tuple[dict[str, float | int], int | None]:
    """@brief 解析 Triton 规格开头注释块里的启动参数。

    ``key = value`` pairs of the leading comment block ('# Launch: ...'); returns (constexprs, num_warps).

    @param src Triton 源码。
    @return ``(constexpr 字典, num_warps 或 None)``。
    """
    consts: dict[str, float | int] = {}
    num_warps = None
    for line in src.splitlines():
        if not line.startswith("#"):
            if line.strip() and not line.startswith("#"):
                break
            continue
        body = line.lstrip("# ").strip()
        # drop the grid expression, it contains '=' free but also commas inside parentheses
        body = re.sub(r"grid\s*=\s*\([^)]*\)", "", body)
        for k, v in re.findall(r"([A-Za-z_]\w*)\s*=\s*([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)", body, flags=re.IGNORECASE):
            val: float | int = float(v) if re.search(r"[.eE]", v) else int(v)
            if k == "num_warps":
                num_warps = int(val)
            else:
                consts[k] = val
    return consts, num_warps


def spec_source(task) -> tuple[str, str]:
    """@brief 取任务用于编译的 Triton 源码及变体说明。

    (source, note). A task may carry a *baseline variant* of its specification: the same function with the same inputs and
    outputs, written in a form the backend accepts (what a Triton user would do to get a kernel through that compiler).

    @param task 目标任务。
    @return ``(源码, 变体说明)``；没有变体时说明为空串。
    """
    v = (task.meta or {}).get("triton_baseline_variant")
    return (v["source"], v.get("note", "variant")) if v else (task.triton_source, "")


def spec_fingerprint(task) -> str:
    """@brief 计算一次测量的规格指纹。

    Identifies what a measurement was taken on: the spec that was compiled, the problem size and launch contract it was checked against,
    and the correctness cases. A cached measurement whose fingerprint differs from the live task's is stale and is not used.

    @param task 目标任务。
    @return 16 位十六进制指纹。
    """
    import hashlib

    h = hashlib.sha256()
    for part in (spec_source(task)[0], repr(task.contract.constants), repr(task.contract.grid), repr(task.contract.threads), repr(task.cases),
                 task.dtype, task.size, repr(sorted((task.evaluation_domain.get("ieee_max") or {}).items()))):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def parse_grid(src: str, header: dict) -> tuple[int, ...] | None:
    """@brief 解析启动头里的 ``grid = (...)`` 表达式。

    The ``grid = (...)`` expression of the launch header, evaluated with the header's values; None when absent.

    @param src Triton 源码。
    @param header 启动头常量字典。
    @return grid 三元组；没有该表达式时为 None。
    """
    m = re.search(r"grid\s*=\s*\(([^)]*)\)", "\n".join(ln for ln in src.splitlines() if ln.startswith("#")))
    if not m:
        return None
    parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
    return tuple(int(eval(p, {"__builtins__": {}}, dict(header))) for p in parts)


def load_spec_module(task, workdir: Path):
    """@brief 把任务规格写成临时模块并导入。

    @param task 目标任务。
    @param workdir 写入规格文件的目录。
    @return 已执行的 Python 模块对象。
    """
    path = workdir / f"spec_{task.name}_{task.dtype}_{task.size}.py"
    path.write_text(spec_source(task)[0])
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def bind_constexprs(jit_fn, header: dict) -> dict:
    """@brief 把 jit 函数的参数名与启动头常量按名匹配。

    Match the jitted function's parameter names to header values, case-insensitively.

    @param jit_fn Triton jit 函数。
    @param header 启动头常量字典。
    @return 参数名到常量值的字典。
    """
    lower = {k.lower(): v for k, v in header.items()}
    names = list(getattr(jit_fn, "arg_names", None) or jit_fn.fn.__code__.co_varnames[: jit_fn.fn.__code__.co_argcount])
    out = {}
    for n in names:
        if n.endswith("_ptr"):
            continue
        if n in header:
            out[n] = header[n]
        elif n.lower() in lower:
            out[n] = lower[n.lower()]
    return out


def build_launcher(task, workdir: Path):
    """@brief 加载规格并构造 Triton 启动器。

    Load the spec with Triton and return (jit_fn, constexpr kwargs, make_impl(num_warps), header num_warps, header).

    Every kernel argument is passed *by name*: ``<x>_ptr`` binds to the contract buffer ``x`` (case-insensitive), constexprs come from
    the launch header, and runtime scalars (e.g. BitDelta's ``scale``) from the contract's scalar arguments.

    @param task 目标任务。
    @param workdir 存放规格文件的目录。
    @return ``(jit_fn, constexpr 字典, make_impl, 头部 num_warps, 头部常量)``。
    @raises RuntimeError 内核参数无法绑定到契约 buffer、常量或标量时抛出。
    """
    import torch

    from tcenv.dtypes import get_dtype
    from tcenv.task import BaselineImpl

    c = task.contract
    mod = load_spec_module(task, workdir)
    src, variant_note = spec_source(task)
    header, nw_spec = parse_launch_header(src)
    if hasattr(mod, "pipeline"):
        # A multi-kernel variant: the backend refuses the single fused kernel, so the spec is a short sequence of kernels
        # (``pipeline(inputs_by_contract_name, num_warps) -> output(s)``), timed as one unit.
        names = [a.name for a in c.inputs]

        def make_pipeline_impl(nw: int) -> BaselineImpl:
            return BaselineImpl(f"triton_msl_w{nw}", "torch", lambda *inputs: mod.pipeline(dict(zip(names, inputs)), nw))

        return None, {}, make_pipeline_impl, nw_spec, header
    jit_fn = getattr(mod, c.kernel_name)
    consts = bind_constexprs(jit_fn, header)
    g = list(parse_grid(src, header) or c.grid)
    while len(g) > 1 and g[-1] == 1:
        g.pop()
    grid = tuple(g)
    params = list(jit_fn.arg_names)
    ptr_map: dict[str, tuple[str, int]] = {}
    for p in params:
        if p.endswith("_ptr"):
            base = p[:-4].lower()
            for i, a in enumerate(c.inputs):
                if a.name.lower() == base:
                    ptr_map[p] = ("in", i)
            for i, a in enumerate(c.outputs):
                if a.name.lower() == base:
                    ptr_map[p] = ("out", i)
            if p not in ptr_map:
                raise RuntimeError(f"kernel parameter {p} has no matching contract buffer")
    scalars = {a.name.lower(): a.value for a in c.scalars}
    for p in params:
        if p not in ptr_map and p not in consts and p.lower() not in scalars:
            raise RuntimeError(f"kernel parameter {p} is bound neither by the launch header nor by a contract scalar")

    def make_impl(nw: int) -> BaselineImpl:
        def call(*inputs):
            outs = [torch.empty(a.shape, dtype=get_dtype(a.dtype).torch, device="mps") for a in c.outputs]
            kw = {}
            for p in params:
                if p in ptr_map:
                    kind, i = ptr_map[p]
                    kw[p] = inputs[i] if kind == "in" else outs[i]
                elif p in consts:
                    kw[p] = consts[p]
                else:
                    kw[p] = scalars[p.lower()]
            jit_fn[grid](**kw, num_warps=nw)
            return outs[0] if len(outs) == 1 else tuple(outs)

        return BaselineImpl(f"triton_msl_w{nw}", "torch", call)

    return jit_fn, consts, make_impl, nw_spec, header


def main() -> int:
    """@brief 隔离环境里的 triton-msl 计时入口：读请求、持锁测量并写出结果。

    @return 恒为 0；错误写进结果文件的 ``status`` 字段。
    """
    req = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(req["out"])
    result: dict = {"status": "error", "task": req["task"], "dtype": req["dtype"], "size": req["size"]}
    from tcenv.gpulock import gpu_lock, set_wait_hooks

    if req.get("heartbeat"):
        hb = os.open(req["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)
        set_wait_hooks(
            lambda: os.pwrite(hb, f"W{time.time():020.6f}\n".encode(), 0),
            lambda: os.pwrite(hb, f"E{time.time():020.6f}\n".encode(), 0),
        )
    try:
        # Exclusive for the whole run: launches go through Python, so even untimed CPU work elsewhere distorts this baseline.
        with gpu_lock(f"triton-msl measurement of {req['task']} {req['dtype']} {req['size']}") as waited:
            result["gpu_lock_wait_s"] = waited
            _run(req, result)
    except BaseException as e:  # noqa: BLE001 - report everything, including backend refusals
        result["status"] = "unsupported" if type(e).__name__ in ("MetalNonRecoverableError", "CompilationError", "NotImplementedError") else "error"
        result["error"] = f"{type(e).__name__}: {str(e)[:600]}"
        result["traceback_tail"] = "".join(traceback.format_exc().splitlines(True)[-6:])
    out_path.write_text(json.dumps(result, indent=1))
    return 0


def _run(req: dict, result: dict) -> None:
    """@brief 执行一次完整的 triton-msl 基线测量。

    先逐个 num_warps 配置做正确性筛选，再交错计时取最快的正确配置，与 MLX / torch_mps 基线对比，
    并把所有尝试与结果写进 ``result``。

    @param req 已解析的请求字典。
    @param result 输出字典，函数会就地填充。
    @raises RuntimeError triton_msl 无法导入或 MPS 不可用时抛出。
    """
    import torch

    import triton

    from tasks import get_task
    from tcenv.backends.metal.baselines import BaselineRunner
    from tcenv.bench import interleaved
    from tcenv.task import BaselineImpl
    from tcenv.backends.metal.testing import _floor, prepare_case, scaled_error

    task = get_task(req["task"], req["dtype"], req["size"])
    c = task.contract
    result["spec_fingerprint"] = spec_fingerprint(task)
    result["versions"] = {"triton": triton.__version__, "torch": torch.__version__, "python": sys.version.split()[0]}
    try:
        import triton_msl

        result["versions"]["triton_msl"] = getattr(triton_msl, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"triton_msl is not importable: {e}")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")

    workdir = Path(req["workdir"])
    workdir.mkdir(parents=True, exist_ok=True)
    jit_fn, consts, make_impl, nw_spec, header = build_launcher(task, workdir)
    result["header_constexprs"] = consts
    result["spec_variant"] = spec_source(task)[1] or None
    if req.get("dry"):  # everything up to (not including) the first compile/launch
        result["status"] = "dry_ok"
        result["params"] = list(jit_fn.arg_names) if jit_fn is not None else ["<pipeline>"]
        return

    # ---- correctness of each configuration on every adversarial case ------------------------------------------
    cases = [prepare_case(task, case, 90_000 + i) for i, case in enumerate(task.cases)]
    sweep = [nw_spec] if nw_spec else []
    sweep += [w for w in NUM_WARPS_SWEEP if w not in sweep]
    good: list[int] = []
    cfg_log = {}
    for nw in sweep:
        impl = make_impl(nw)
        worst = 0.0
        try:
            for data in cases:
                r = BaselineRunner(impl, task, data.rounded, rotate=1)
                outs = r.outputs()
                for a in c.outputs:
                    s = scaled_error(outs[a.name], data.ref[a.name], data.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype))
                    worst = max(worst, float(s.max()))
            cfg_log[nw] = {"worst_scaled_error": worst}
            if worst <= 8.0:
                good.append(nw)
        except BaseException as e:  # noqa: BLE001
            cfg_log[nw] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            if nw == sweep[0] and type(e).__name__ == "MetalNonRecoverableError":
                raise  # the spec itself is refused by the backend; no point in sweeping
    result["configs"] = cfg_log
    if not good:
        result["status"] = "incorrect" if any("worst_scaled_error" in v for v in cfg_log.values()) else "unsupported"
        result["error"] = "no num_warps configuration produced correct results: " + json.dumps(cfg_log)[:500]
        return

    # ---- timing: sweep correct configs quickly, then an interleaved run with the vendor baselines ---------------
    data0 = prepare_case(task, task.cases[0], 4242)
    hb = None
    if req.get("heartbeat"):
        hb = os.open(req["heartbeat"], os.O_WRONLY | os.O_CREAT, 0o600)

    def guarded(runner: BaselineRunner) -> BaselineRunner:
        inner = runner.run_k

        def run_k(k: int) -> float:
            if hb is not None:
                os.pwrite(hb, f"B{time.time():020.6f}\n".encode(), 0)
            try:
                return inner(k)
            finally:
                if hb is not None:
                    os.pwrite(hb, f"E{time.time():020.6f}\n".encode(), 0)

        runner.run_k = run_k  # type: ignore[method-assign]
        return runner

    quick = {}
    for nw in good:
        r = guarded(BaselineRunner(make_impl(nw), task, data0.rounded))
        st = interleaved({"t": r.run_k}, reps=5, warmup_reps=1)["t"]
        quick[nw] = st.median_s
    best_nw = min(quick, key=quick.get)
    result["sweep_median_us"] = {str(k): v * 1e6 for k, v in quick.items()}
    result["best_num_warps"] = best_nw

    runners = {"triton_msl": guarded(BaselineRunner(make_impl(best_nw), task, data0.rounded))}
    for impl in task.baselines:
        if impl.name not in ("mlx", "torch_mps"):  # compiled baselines need the main environment's torch/Python (Inductor), not this one
            continue
        try:
            runners[impl.name] = guarded(BaselineRunner(impl, task, data0.rounded))
        except Exception as e:  # noqa: BLE001
            result.setdefault("vendor_errors", []).append(f"{impl.name}: {type(e).__name__}: {e}")

    # Python-side cost of one launch: if it approaches the time per launch the baseline is launch-bound and CPU load distorts it.
    # Warm up first (the first launches after building new buffer sets include one-time costs) and use the median of single launches.
    tri_impl = make_impl(best_nw)
    sets = runners["triton_msl"].sets
    for i in range(8):
        tri_impl.fn(*sets[i % len(sets)])
    torch.mps.synchronize()
    singles = []
    t0 = time.perf_counter()
    for i in range(60):
        a0 = time.perf_counter()
        tri_impl.fn(*sets[i % len(sets)])
        singles.append((time.perf_counter() - a0) * 1e6)
    t1 = time.perf_counter()
    torch.mps.synchronize()
    t2 = time.perf_counter()
    result["enqueue_us_per_launch"] = sorted(singles)[len(singles) // 2]
    result["total_us_per_launch"] = (t2 - t0) / 60 * 1e6

    # Interleaved measurement, repeated: this backend dispatches through Python, so competing CPU work inflates its time much more than
    # MLX's or torch's (a measurement taken while the machine was busy came out 2-3x too slow). A run counts as consistent when the
    # interleaved time is within 30% of the best time seen while sweeping num_warps; the best attempt is kept (conservative for any
    # claim of beating this baseline) and all attempts are recorded.
    attempts = []
    sweep_best = min(quick.values())
    for attempt in range(int(req.get("max_attempts", 3))):
        load = os.getloadavg()[0]
        stats = interleaved({n: r.run_k for n, r in runners.items()}, reps=int(req.get("reps", 15)), seed=attempt + 1, kmin=runners["triton_msl"].rotate)
        tri = stats["triton_msl"].median_s
        attempts.append(
            {
                "load_avg_1m": load,
                "median_us": {n: s.median_us for n, s in stats.items()},
                "cv": {n: s.cv for n, s in stats.items()},
                "consistent": tri <= 1.3 * sweep_best,
            }
        )
        if attempts[-1]["consistent"]:
            break
    best = min(attempts, key=lambda a: a["median_us"]["triton_msl"])
    result["attempts"] = attempts
    result["consistent"] = best["consistent"]
    result["median_us"] = best["median_us"]
    result["cv"] = best["cv"]
    result["ratio_to"] = {n: best["median_us"]["triton_msl"] / v for n, v in best["median_us"].items() if n != "triton_msl"}
    if result["enqueue_us_per_launch"] > 0.5 * best["median_us"]["triton_msl"]:
        result["warning"] = "launch-bound: the Python launch cost is more than half of the time per launch, so this ratio mostly measures dispatch overhead"
    result["status"] = "ok"


if __name__ == "__main__":
    raise SystemExit(main())
