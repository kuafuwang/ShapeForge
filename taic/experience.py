"""Read-only hints mined from finished candidate pairs. Nothing here calls a model or launches a kernel."""

from __future__ import annotations

import json
from pathlib import Path

from .hwfeatures import analyze_source
from .policy import feature_signature

_FEATURES = ("matrix_unit", "acc_bucket", "tg_staging", "double_buffer", "register_prefetch", "vector_load", "online_softmax")


def _present(source: str, backend_id: str = "metal") -> set[str]:
    """@brief 取源码里出现的静态特征名集合。

    @param source 内核源码。
    @param backend_id 后端标识，决定特征签名是否可用。
    @return 出现的特征名集合；后端不支持时为空集。
    """
    sig = feature_signature(source, backend_id=backend_id)
    return {name for name, bit in zip(_FEATURES, sig) if bit and bit != "0"}


def _bound(report: dict) -> str | None:
    """@brief 读报告里的 roofline 瓶颈类别。

    @param report 评测报告字典。
    @return ``memory`` 或 ``compute``；字段缺失或不认识时返回 ``None``。
    """
    roof = report.get("roofline") if isinstance(report, dict) else None
    bound = roof.get("bound") if isinstance(roof, dict) else None
    return bound if bound in ("memory", "compute") else None


def _target_info(run: Path) -> tuple[str, str]:
    """@brief 读一次运行的后端标识与内核扩展名，读不到时退回 Metal。

    @param run 运行目录。
    @return ``(backend_id, file_ext)``；``run_config.json`` 缺失或损坏时返回 ``("metal", ".metal")``。
    """
    try:
        doc = json.loads((run / "run_config.json").read_text())
        target = doc.get("target") if isinstance(doc.get("target"), dict) else {}
        target_id = str(target.get("id") or doc.get("target_id") or "")
        from tcenv.backends.registry import load_backend
        from tcenv.target import load_target

        profile = load_target(target_id or None)
        backend = load_backend(profile.backend)
        ext = backend.file_ext or ".metal"
        if not str(ext).startswith("."):
            ext = "." + str(ext)
        return backend.id, ext
    except Exception:
        return "metal", ".metal"


def build_experience_index(roots: list[Path], *, exclude_runs: set[str] | None = None) -> dict:
    """@brief 挖出「后一颗 ok 内核比前一颗快」的成对样本，以及它新增的静态特征。

    @param roots 需要扫描的根目录；每个根可以是单个运行目录，也可以是包含多个运行的上级目录。
    @param exclude_runs 要排除的运行目录名集合，使重复性试验可以把自己排除在外。
    @return 含 ``entries``（按瓶颈类别与特征聚合）、``wins``（成对样本数）与 ``pairs``（原始成对记录）的字典。
    """
    blocked = exclude_runs or set()
    wins: list[dict] = []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        runs = [root] if (root / "candidates").is_dir() else [p for p in root.rglob("candidates") if p.is_dir()]
        seen_dirs = {p.parent if p.name == "candidates" else p for p in runs}
        for run in sorted(seen_dirs):
            if run.name in blocked or any(part in blocked for part in run.parts):
                continue
            backend_id, ext = _target_info(run)
            best_source = None
            best_speed = None
            folders = sorted((run / "candidates").iterdir()) if (run / "candidates").is_dir() else []
            for folder in folders:
                report_path = folder / "report.json"
                kernel_path = folder / f"kernel{ext}"
                if not kernel_path.is_file():
                    kernel_path = next(iter(sorted(folder.glob("kernel.*"))), folder / "kernel.metal")
                if not report_path.is_file() or not kernel_path.is_file():
                    continue
                try:
                    report = json.loads(report_path.read_text())
                except ValueError:
                    continue
                if not isinstance(report, dict) or report.get("status") != "ok":
                    continue
                speed = report.get("speedup")
                if not isinstance(speed, (int, float)) or isinstance(speed, bool):
                    continue
                source = kernel_path.read_text()
                if best_source is not None and best_speed is not None and speed > best_speed * 1.005:
                    added = sorted(_present(source, backend_id) - _present(best_source, backend_id))
                    wins.append({"task": report.get("task"), "bound": _bound(report), "added": added, "ratio": speed / best_speed})
                if best_speed is None or speed > best_speed:
                    best_source, best_speed = source, float(speed)
    return {"entries": _aggregate(wins), "wins": len(wins), "pairs": wins}


def load_index(path: Path) -> dict:
    """@brief 读一份经验索引文件。

    @param path 索引 JSON 的路径。
    @return 解析后的字典；文件缺失、损坏或不是对象时返回 ``{"entries": []}``。
    """
    try:
        doc = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {"entries": []}
    return doc if isinstance(doc, dict) else {"entries": []}


def _task_base(name: object) -> str:
    """@brief 去掉 ``@variant`` 后缀，得到任务基名。

    @param name 任务名。
    @return 基名；``name`` 不是非空字符串时为空字符串。
    """
    if not isinstance(name, str) or not name:
        return ""
    return name.split("@", 1)[0]


def _aggregate(wins: list[dict]) -> list[dict]:
    """@brief 把成对样本按（瓶颈类别, 特性）聚合。

    条目按样本数降序排列；``median_ratio`` 取各样本加速比的中位数。

    @param wins 成对样本列表，每项含 ``added``/``bound``/``ratio``。
    @return 聚合后的条目列表。
    """
    grouped: dict[tuple, list[float]] = {}
    for row in wins:
        for feature in row.get("added") or []:
            grouped.setdefault((row.get("bound"), feature), []).append(float(row["ratio"]))
    entries = []
    for (bound, feature), ratios in sorted(grouped.items(), key=lambda item: (-len(item[1]), str(item[0]))):
        ordered = sorted(ratios)
        entries.append({"bound": bound, "feature": feature, "n": len(ratios), "median_ratio": ordered[len(ordered) // 2]})
    return entries


def retrieve_experience_hints(index: dict, task_name: str, bound: str | None, *, exclude_task: bool = False, k: int = 3) -> str:
    """@brief 生成至多 ``k`` 行经验提示。

    ``exclude_task`` 会丢掉任务基名（忽略 ``@variant``）等于 ``task_name`` 的所有样本。
    没有 ``pairs`` 列表的索引无法证明提示来自别的任务，因此直接不给提示，而不是改用混合过的 ``entries``。
    重复性试验传入空索引路径，根本不会走到这里。

    @param index 经验索引字典。
    @param task_name 当前任务名。
    @param bound 当前瓶颈类别；``memory``/``compute`` 时优先取同类样本。
    @param exclude_task 是否排除同任务基名的样本。
    @param k 最多几行提示。
    @return 提示文本；没有可用样本时为空字符串。
    """
    if exclude_task:
        pairs = index.get("pairs")
        if not isinstance(pairs, list):
            return ""
        base = _task_base(task_name)
        kept = [row for row in pairs if isinstance(row, dict) and _task_base(row.get("task")) != base]
        rows = _aggregate(kept)
    else:
        rows = [row for row in (index.get("entries") or []) if isinstance(row, dict) and row.get("n")]
    if bound in ("memory", "compute"):
        scoped = [row for row in rows if row.get("bound") in (bound, None)]
        if scoped:
            rows = scoped
    lines = []
    for row in rows[:k]:
        lines.append(f"- {row.get('feature')} showed up on a faster kernel in {row.get('n')} pair(s), median {float(row.get('median_ratio') or 1):.3f}x (static, not a requirement)")
    if not lines:
        return ""
    return "Experience from other measured pairs (hints, the timing decides):\n" + "\n".join(lines)


def write_index(path: Path, index: dict) -> None:
    """@brief 把经验索引写到磁盘，必要目录一并创建。

    @param path 目标 JSON 路径。
    @param index 要写出的索引字典。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=1))


# analyze_source is imported so a caller can see that mining is source-level. Re-export keeps the dependency obvious.
__all__ = ["analyze_source", "build_experience_index", "load_index", "retrieve_experience_hints", "write_index"]
