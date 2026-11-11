"""Static bottleneck probe for OpSpec v2 candidates. No GPU dispatch.

``python -m taic.v2_hw_inspect`` reads an ``msl-single`` source or an ``msl-bundle``
envelope and reports declared ``simdgroup_float8x8`` slots, static threadgroup
bytes, and hardware-feature hints. Optional AIR compilation calls ``xcrun metal``
and still does not launch a kernel. This module does not import ``tcenv.backends``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSPECT_SCHEMA_ID = "taic.v2.hw-inspect.1"
BASIS_NOTE = (
    "源码/AIR 静态分析，不代替真机计时或 xctrace spill 计数。"
    "estimated_resident_threadgroups 是按目标 max_threadgroup_memory 算出的静态上界，不是编译器寄存器限制。"
)
_SATURATED_GAP = 1.0 / 0.90
_COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_DECL = re.compile(r"\bsimdgroup_float8x8\s+([^;]+);")
_DECLARATOR = re.compile(r"^([A-Za-z_]\w*)\s*(?:\[\s*([^\]]+?)\s*\])?")
_KERNEL = re.compile(r"\bkernel\s+void\s+([A-Za-z_]\w*)\b")
_KERNEL_DECL = _KERNEL
_PASTE = re.compile(r"([A-Za-z0-9_])\s*##\s*([A-Za-z0-9_])")
_LAUNCH_LINE = re.compile(
    r"^[ \t]*launch[ \t]+([A-Za-z_]\w*)[ \t]+grid=(\S+)[ \t]+threads=(\S+)",
    re.M,
)


@dataclass(frozen=True)
class StageInspection:
    """@brief 一个 stage 的静态瓶颈诊断。"""

    stage_index: int
    kernel_name: str
    source_path: str
    grid: tuple[int, int, int] | None
    threads: tuple[int, int, int] | None
    fragment_slots: int
    largest_fragment_array: int
    static_threadgroup_bytes: int
    estimated_resident_threadgroups: int | None
    uses: str
    warnings: tuple[str, ...]

    def document(self) -> dict[str, Any]:
        """@brief 该 stage 的 JSON 投影。"""
        return {
            "stage_index": self.stage_index,
            "kernel_name": self.kernel_name,
            "source_path": self.source_path,
            "grid": None if self.grid is None else list(self.grid),
            "threads": None if self.threads is None else list(self.threads),
            "fragment_slots": self.fragment_slots,
            "largest_fragment_array": self.largest_fragment_array,
            "static_threadgroup_bytes": self.static_threadgroup_bytes,
            "estimated_resident_threadgroups": self.estimated_resident_threadgroups,
            "uses": self.uses,
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class V2InspectReport:
    """@brief 一颗 v2 候选的静态硬件诊断。"""

    schema_id: str
    task: str
    dtype: str
    size: str
    target_id: str
    delivery: str
    stages: tuple[StageInspection, ...]
    missing_hints: tuple[str, ...]
    performance_line: str
    air_compiled: bool
    basis_note: str

    def document(self) -> dict[str, Any]:
        """@brief 整份诊断的 JSON 投影。"""
        return {
            "schema_id": self.schema_id,
            "task": self.task,
            "dtype": self.dtype,
            "size": self.size,
            "target_id": self.target_id,
            "delivery": self.delivery,
            "stages": [stage.document() for stage in self.stages],
            "missing_hints": list(self.missing_hints),
            "performance_line": self.performance_line,
            "air_compiled": self.air_compiled,
            "basis_note": self.basis_note,
        }

    def format_text(self) -> str:
        """@brief 给人读的多行诊断。"""
        lines = [
            f"{self.task} {self.dtype}/{self.size} on {self.target_id} ({self.delivery})",
            self.performance_line,
            self.basis_note,
        ]
        if self.missing_hints:
            lines.append("Expected for this kind of kernel but not found (hints):")
            lines.extend(f"- {hint}" for hint in self.missing_hints)
        elif "roofline" in self.performance_line and "suppressed" in self.performance_line:
            lines.append("Missing-feature hints suppressed: the current report is at the roofline.")
        for stage in self.stages:
            grid = "unset" if stage.grid is None else "x".join(str(v) for v in stage.grid)
            threads = "unset" if stage.threads is None else "x".join(str(v) for v in stage.threads)
            resident = "n/a" if stage.estimated_resident_threadgroups is None else str(stage.estimated_resident_threadgroups)
            lines.append(
                f"stage {stage.stage_index} {stage.kernel_name} ({stage.source_path}) "
                f"grid={grid} threads={threads} fragments={stage.fragment_slots} "
                f"largest_array={stage.largest_fragment_array} "
                f"threadgroup={stage.static_threadgroup_bytes}B "
                f"static_resident_upper_bound={resident}"
            )
            if stage.uses:
                lines.append(f"  Uses: {stage.uses}")
            for warning in stage.warnings:
                lines.append(f"  {warning}")
        if self.air_compiled:
            lines.append("AIR: compiled (static evidence only; nothing was launched on the GPU).")
        else:
            lines.append("AIR: not compiled.")
        return "\n".join(lines)


def inspect_v2_candidate(
    task: str,
    candidate_text: str,
    *,
    dtype: str = "fp16",
    size: str = "paper",
    target_id: str = "apple-m4-max",
    report: Mapping[str, Any] | None = None,
    compile_air_evidence: bool = False,
    air_compiler: Callable[..., dict] | None = None,
) -> V2InspectReport:
    """@brief 诊断一颗 v2 候选。不发射 GPU 内核。

    @param task 题名。
    @param candidate_text ``msl-single`` 源码或 ``msl-bundle`` envelope。
    @param dtype 数据类型。
    @param size 档位。
    @param target_id 目标 id，用来读 threadgroup 上限。
    @param report 可选的评测报告。只有 schema 4 的 roofline 会进入诊断。
    @param compile_air_evidence 为真时编译到 AIR。
    @param air_compiler 替换 ``taic.hwfeatures.compile_air`` 的可调用对象。
    @return 静态诊断。
    """
    from taic.artifacts import document_state
    from taic.hwfeatures import HardwareFeatureReport, analyze_source, detect_features, judge, measured_from_report, perf_test
    from taic.hwfeedback import _hints, _uses
    from taic.task_binding import bind_task

    binding = bind_task(task, dtype, size, target_id=target_id)
    if binding.mode != "v2" or binding.contract is None:
        raise ValueError(f"task {task!r} did not assemble as an OpSpec v2 contract")
    instance = binding.contract.instance
    consts = _consts(instance)
    names = {name: int(value) for name, value in consts.items()}
    limit = _threadgroup_limit(target_id)
    pieces = _pieces(candidate_text, names)
    measured = measured_from_report(dict(report)) if isinstance(report, Mapping) and document_state(report) == "current" else None
    roof = _roof(measured)
    saturated = _saturated(roof)
    air_by_source, air_compiled = _compile_pieces(pieces, consts, compile_air_evidence, air_compiler)
    stages: list[StageInspection] = []
    hint_lines: list[str] = []
    seen_hints: set[str] = set()
    for index, piece in enumerate(pieces):
        evidence = analyze_source(
            piece.source,
            consts=consts,
            kernel_names=[piece.kernel] if piece.kernel else None,
        )
        pressure = _source_pressure(_reachable_fragment_source(piece.source, piece.kernel))
        declared = int((evidence.get("threadgroup") or {}).get("declared_bytes") or 0)
        resident = _resident(declared, limit)
        compiled = air_by_source.get(piece.source)
        compiled_ok = compiled if isinstance(compiled, dict) and compiled.get("ok") else None
        try:
            features = detect_features(evidence, compiled_ok)
        except (KeyError, TypeError, ValueError):
            # A compiler can report ok without the AIR fields detect_features reads.
            # Keep air_compiled from the compiler result and continue on source evidence.
            compiled_ok = None
            features = detect_features(evidence, None)
        performance = perf_test(task, measured, None)
        features, missing = judge(task, features, performance)
        feature_report = HardwareFeatureReport(
            label=piece.kernel or f"stage{index}",
            task=task,
            dtype=dtype,
            source=evidence,
            compiled=compiled if isinstance(compiled, dict) else None,
            measured=measured,
            performance_test=performance,
            features=features,
            missing=missing,
            summary="",
        )
        warnings = tuple(_fragment_warnings(pressure["largest_array"]) + _threadgroup_warnings(declared, limit, resident))
        stages.append(
            StageInspection(
                stage_index=index,
                kernel_name=piece.kernel or evidence.get("kernel") or f"stage{index}",
                source_path=piece.path,
                grid=piece.grid,
                threads=piece.threads,
                fragment_slots=int(pressure["fragment_slots"]),
                largest_fragment_array=int(pressure["largest_array"]),
                static_threadgroup_bytes=declared,
                estimated_resident_threadgroups=resident,
                uses=_uses(feature_report),
                warnings=warnings,
            )
        )
        if not saturated:
            for line in _hints(feature_report, measured or {}, roof):
                text = line[2:] if line.startswith("- ") else line
                if text not in seen_hints:
                    seen_hints.add(text)
                    hint_lines.append(text)
    return V2InspectReport(
        schema_id=INSPECT_SCHEMA_ID,
        task=task,
        dtype=dtype,
        size=size,
        target_id=target_id,
        delivery=pieces[0].delivery if pieces else "msl-single",
        stages=tuple(stages),
        missing_hints=tuple(hint_lines),
        performance_line=_performance_line(roof, saturated),
        air_compiled=air_compiled,
        basis_note=BASIS_NOTE,
    )


def main(argv: list[str] | None = None) -> int:
    """@brief 命令行入口。参数或文件错误返回 2，诊断成功返回 0。

    @param argv 参数；None 时用进程参数。
    @return 退出码。
    """
    parser = argparse.ArgumentParser(prog="python -m taic.v2_hw_inspect", description="Static v2 hardware inspection. Does not launch a GPU kernel.")
    parser.add_argument("task")
    parser.add_argument("candidate")
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--size", default="paper")
    parser.add_argument("--target", default="apple-m4-max")
    parser.add_argument("--report", default=None)
    parser.add_argument("--air", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    path = Path(args.candidate)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"v2_hw_inspect: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    report = None
    if args.report:
        try:
            loaded = json.loads(Path(args.report).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"v2_hw_inspect: cannot read report {args.report}: {exc}", file=sys.stderr)
            return 2
        if not isinstance(loaded, dict):
            print("v2_hw_inspect: report JSON must be an object", file=sys.stderr)
            return 2
        report = loaded
    try:
        inspected = inspect_v2_candidate(
            args.task,
            text,
            dtype=args.dtype,
            size=args.size,
            target_id=args.target,
            report=report,
            compile_air_evidence=bool(args.air),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary; the library still raises to callers
        print(f"v2_hw_inspect: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(inspected.document(), indent=2, sort_keys=True))
    else:
        print(inspected.format_text())
    return 0


@dataclass(frozen=True)
class _Piece:
    delivery: str
    path: str
    kernel: str
    source: str
    grid: tuple[int, int, int] | None
    threads: tuple[int, int, int] | None


def _consts(instance: Any) -> dict[str, float]:
    """@brief 从 v2 实例的符号和整数属性合成常量。"""
    consts: dict[str, float] = {}
    for name, value in dict(instance.symbols).items():
        consts[str(name)] = float(int(value))
    for name, value in dict(instance.resolved_attrs).items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        consts.setdefault(str(name), float(value))
    return consts


def _threadgroup_limit(target_id: str) -> int:
    """@brief 读目标档案里的静态 threadgroup 上限。"""
    from tcenv.target import load_target

    raw = dict(load_target(target_id).raw_limits)
    try:
        return int(raw["max_threadgroup_memory_bytes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"target {target_id} has no integer max_threadgroup_memory_bytes") from exc


def _pieces(candidate_text: str, names: Mapping[str, int]) -> tuple[_Piece, ...]:
    """@brief 把单文件或 envelope 拆成逐 stage 的源码与启动规模。"""
    from tcenv.track.envelope import is_candidate_envelope

    if is_candidate_envelope(candidate_text):
        return _bundle_pieces(candidate_text, names)
    return _single_pieces(candidate_text, names)


def _bundle_pieces(candidate_text: str, names: Mapping[str, int]) -> tuple[_Piece, ...]:
    """@brief 解开 msl-bundle，按 execution geometry 求值每个 stage。"""
    from tcenv.track.envelope import decode_envelope
    from tcenv.track.plane import execution_geometry_from_document

    artifact = decode_envelope(candidate_text)
    schedule = next(
        (entry for entry in artifact.entries if entry.role == "schedule" or entry.path.endswith("execution_geometry.json")),
        None,
    )
    if schedule is None:
        raise ValueError("msl-bundle envelope has no host/execution_geometry.json schedule")
    document = json.loads(schedule.data.decode("utf-8"))
    geometry = execution_geometry_from_document(document)
    devices = [entry for entry in artifact.entries if entry.role == "device_kernel"]
    pieces: list[_Piece] = []
    for stage in geometry.stages:
        entry = _entry_for_kernel(devices, stage.kernel)
        pieces.append(
            _Piece(
                delivery="msl-bundle",
                path=entry.path,
                kernel=stage.kernel,
                source=entry.data.decode("utf-8"),
                grid=_eval_triple(stage.grid, names),
                threads=_eval_triple(stage.threads, names),
            )
        )
    if not pieces:
        raise ValueError("msl-bundle execution geometry has no stages")
    return tuple(pieces)


def _single_pieces(candidate_text: str, names: Mapping[str, int]) -> tuple[_Piece, ...]:
    """@brief 用纯文本读 ``taic-launch``；没有启动块时按 kernel 声明各记一段。"""
    body = _launch_body(candidate_text)
    found: list[_Piece] = []
    if body is not None:
        for kernel, grid_text, thread_text in _LAUNCH_LINE.findall(body):
            found.append(
                _Piece(
                    delivery="msl-single",
                    path="<candidate>",
                    kernel=kernel,
                    source=candidate_text,
                    grid=_eval_triple(grid_text.split(","), names),
                    threads=_eval_triple(thread_text.split(","), names),
                )
            )
    if found:
        return tuple(found)
    kernels = list(dict.fromkeys(_KERNEL.findall(candidate_text))) or ["kernel"]
    return tuple(
        _Piece("msl-single", "<candidate>", kernel, candidate_text, None, None)
        for kernel in kernels
    )


def _launch_body(source: str) -> str | None:
    """@brief 取出唯一一个以 ``taic-launch`` 开头的块注释正文。"""
    bodies = re.findall(r"/\*(.*?)\*/", source, flags=re.S)
    launches = []
    for body in bodies:
        tokens = body.split()
        if tokens and tokens[0] == "taic-launch":
            launches.append(body)
    if len(launches) != 1:
        return None
    return launches[0]


def _entry_for_kernel(entries: list[Any], kernel: str) -> Any:
    """@brief 找到唯一一份声明了 ``kernel void <kernel>`` 的 device 源文件。

    名字必须是展开宏和 ``##`` 粘贴之后的完整标识符。前缀重叠不算匹配。
    没有声明，或不止一个 device 条目声明了同一个名字，都直接拒绝。
    """
    matches = []
    for entry in entries:
        text = entry.data.decode("utf-8", errors="replace")
        if kernel in _declared_kernel_names(text):
            matches.append(entry)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"no device entry declares `kernel void {kernel}`")
    listed = ", ".join(str(entry.path) for entry in matches)
    raise ValueError(f"`kernel void {kernel}` is declared by more than one device entry: {listed}")


def _prepared_msl(source: str) -> str:
    """@brief 去掉注释和字符串，展开函数式宏，并完成 ``##`` 粘贴。"""
    from taic.hwfeatures import expand_macros, normalize, strip_comments_and_strings

    return _paste_tokens(expand_macros(normalize(strip_comments_and_strings(source or ""))))


def _paste_tokens(text: str) -> str:
    """@brief 把 ``name##STAGE`` 收成一个标识符。参数替换之后才粘贴。"""
    previous = None
    while previous != text:
        previous = text
        text = _PASTE.sub(r"\1\2", text)
    return text


def _declared_kernel_names(source: str) -> list[str]:
    """@brief 按出现顺序列出 ``kernel void`` 的完整名字。"""
    return _KERNEL_DECL.findall(_prepared_msl(source))


def _reachable_fragment_source(source: str, kernel: str | None) -> str:
    """@brief 只留下当前 kernel 和它能调用到的 helper，供槽位统计使用。

    msl-single 与 bundle 都走这里。同文件里没有被这次启动调用到的 kernel
    不进入扫描。可达性与 ``analyze_source`` 一样：从 kernel 体出发，按名字调用闭包。
    """
    from taic.hwfeatures import parse_functions

    funcs = parse_functions(_prepared_msl(source))
    kernels = [item for item in funcs if item.is_kernel]
    by_name = {item.name: item for item in kernels}
    if kernel:
        chosen = [by_name[kernel]] if kernel in by_name else []
    else:
        chosen = kernels[:1]
    if not chosen:
        return ""
    helpers = {item.name: item for item in funcs if not item.is_kernel}
    bodies = [item.body for item in chosen]
    reach: set[str] = set()
    todo = list(bodies)
    while todo:
        text = todo.pop()
        for name, helper in helpers.items():
            if name in reach or not re.search(rf"\b{re.escape(name)}\s*\(", text):
                continue
            reach.add(name)
            todo.append(helper.body)
    parts = [item.params for item in chosen]
    parts.extend(bodies)
    for name in sorted(reach):
        parts.append(helpers[name].params)
        parts.append(helpers[name].body)
    return "\n".join(parts)


def _eval_triple(terms: Any, names: Mapping[str, int]) -> tuple[int, int, int] | None:
    """@brief 把几何三元组求成整数。求值失败时返回 None，诊断继续。"""
    from tcenv.track.expr import evaluate_term

    if not isinstance(terms, (list, tuple)) or len(terms) != 3:
        return None
    values: list[int] = []
    try:
        for index, term in enumerate(terms):
            values.append(int(evaluate_term(term, names, where=f"launch[{index}]")))
    except (TypeError, ValueError, ArithmeticError):
        return None
    except Exception:
        return None
    return tuple(values)  # type: ignore[return-value]


def _compile_pieces(
    pieces: tuple[_Piece, ...],
    consts: Mapping[str, float],
    enabled: bool,
    air_compiler: Callable[..., dict] | None,
) -> tuple[dict[str, dict], bool]:
    """@brief 按源码文本去重编译 AIR。工具链缺失时记失败并继续。"""
    if not enabled:
        return {}, False
    compiler = air_compiler
    if compiler is None:
        from taic import hwfeatures as hw

        compiler = hw.compile_air
    header = _constant_header(consts)
    compiled: dict[str, dict] = {}
    ok = True
    attempted = False
    for piece in pieces:
        if piece.source in compiled or not piece.source:
            continue
        attempted = True
        try:
            evidence = compiler(header + piece.source)
        except Exception as exc:  # noqa: BLE001 - a missing toolchain must not abort the source diagnosis
            evidence = {"ok": False, "skipped": True, "error": str(exc)[:300]}
        if not isinstance(evidence, dict):
            evidence = {"ok": False, "error": "AIR compiler did not return an object"}
        compiled[piece.source] = evidence
        ok = ok and bool(evidence.get("ok"))
    return compiled, bool(attempted and ok)


def _constant_header(consts: Mapping[str, float]) -> str:
    """@brief 拼一份只含整数常量的 MSL 头，供 AIR 编译使用。"""
    lines = ["#include <metal_stdlib>", "using namespace metal;"]
    for name in sorted(consts):
        lines.append(f"constant constexpr uint {name} = {int(consts[name])};")
    return "\n".join(lines) + "\n"


def _literal_size(value: str | None) -> int | None:
    """@brief 解析数组维度里的十进制字面量。"""
    if value is None:
        return None
    text = value.strip()
    if text.endswith(("u", "U", "l", "L")):
        text = text[:-1]
    return int(text) if text.isdigit() else None


def _source_pressure(source: str) -> dict[str, int]:
    """@brief 统计声明的 simdgroup_float8x8 槽位。规则与 pressure.source_pressure 相同。"""
    text = _COMMENT.sub("", source or "")
    slots = 0
    arrays: list[int] = []
    for match in _DECL.finditer(text):
        for declarator in match.group(1).split(","):
            item = _DECLARATOR.match(declarator.strip())
            if not item:
                continue
            size = _literal_size(item.group(2))
            slots += 1 if size is None else size
            if size is not None:
                arrays.append(size)
    return {"fragment_slots": slots, "largest_array": max(arrays, default=0)}


def _fragment_warnings(largest: int) -> list[str]:
    """@brief 按本机实测给出累加器槽位警告。"""
    if largest >= 32:
        return [
            f"Warning: largest simdgroup_float8x8 array is {largest} slots; reference GEMMs with 32 fragments "
            "ran 12 to 26 times slower from register spills. This is a declared-slot upper bound, not an xctrace spill count."
        ]
    if 17 <= largest < 32:
        return [
            f"Warning: largest simdgroup_float8x8 array is {largest} slots; conv2d measured 16 slots at 690 us and "
            "32 slots at 1069 us. This is a source upper bound, not compiler register usage."
        ]
    return []


def _resident(declared: int, limit: int) -> int | None:
    """@brief 用静态占用除目标上限，得到驻留线程组的静态上界。"""
    if declared <= 0 or limit <= 0:
        return None
    if declared > limit:
        return 0
    return limit // declared


def _threadgroup_warnings(declared: int, limit: int, resident: int | None) -> list[str]:
    """@brief 静态 threadgroup 越界或跨过一半上限时的警告。"""
    if declared > limit:
        return [
            f"Warning: static threadgroup memory {declared} B exceeds the target limit {limit} B. "
            "The resident figure is a static upper bound, not a compiler register limit."
        ]
    if declared > limit // 2:
        return [
            f"Warning: static threadgroup memory {declared} B is above {limit // 2} B, so the static resident "
            f"upper bound drops to {resident} threadgroup/core. flash_attention at 9216 B kept 3 resident "
            "threadgroups and at 17408 B dropped to 1, 12-14% slower. This is a static upper bound from "
            "max_threadgroup_memory, not a compiler register limit."
        ]
    return []


def _roof(measured: Mapping[str, Any] | None) -> dict[str, Any]:
    """@brief 从当前报告的扁平实测里取出 roofline 字段。"""
    if not measured:
        return {}
    roof = {}
    for key in ("bound", "time_over_lower_bound", "achieved_tflops", "compute_fraction", "achieved_gbps", "bandwidth_fraction", "peak_tflops_measured", "peak_gbps_measured"):
        if measured.get(key) is not None:
            roof[key] = measured[key]
    return roof


def _saturated(roof: Mapping[str, Any]) -> bool:
    """@brief 当前报告达到 roofline 的 90% 时不再列缺失特征。"""
    gap = roof.get("time_over_lower_bound")
    if isinstance(gap, bool) or not isinstance(gap, (int, float)) or gap <= 0:
        return False
    return float(gap) <= _SATURATED_GAP


def _performance_line(roof: Mapping[str, Any], saturated: bool) -> str:
    """@brief 只用当前 schema 的 roofline 写一行性能说明。"""
    if not roof:
        return "No current schema-4 roofline measurement; source diagnosis only."
    gap = roof.get("time_over_lower_bound")
    bound = roof.get("bound") or "?"
    if not isinstance(gap, (int, float)) or isinstance(gap, bool):
        return f"Current schema-4 report is {bound}-bound; no time-over-lower-bound was recorded."
    if saturated:
        percent = min(100.0, 100.0 / float(gap))
        return (
            f"Current schema-4 report is at {percent:.0f}% of its {bound} roofline "
            f"({float(gap):.2f}x the lower bound); missing-feature hints suppressed."
        )
    return f"Current schema-4 report is {float(gap):.2f}x its {bound} roofline lower bound."


if __name__ == "__main__":
    raise SystemExit(main())
