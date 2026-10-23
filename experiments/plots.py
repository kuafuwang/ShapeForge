"""Plots for the suite results: per-kernel speedup/cost frontiers (paper Figure 4) and a summary bar chart."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .common import conclusion_of_run, read_confirm, read_result  # noqa: E402

DTYPE_STYLE = {"fp16": ("#1f77b4", "FP16"), "bf16": ("#d62728", "BF16"), "fp32": ("#2ca02c", "FP32")}
ORDER = ["conv2d", "gemm_silu", "gemm", "gemv", "sigmoid", "relu", "silu", "swiglu", "rmsnorm", "reduction_sum", "rope", "softmax"]


def frontier(res: dict) -> list[tuple[float, float]]:
    """@brief 取每次改进时的累积花费与当时最佳加速比，画出阶梯前沿。

    (cumulative USD, best speedup so far) at every candidate that improved the best, as in the paper's figure.

    @param res 一份运行的 ``result.json`` 内容。
    @return ``(累积花费, 当时最佳加速比)`` 列表；轨迹缺失时为空列表。
    """
    pts = [(t["cost_usd"], t["best_speedup"]) for t in res.get("trace", []) if t.get("improved") and t.get("best_speedup")]
    return pts


def _collect(suite_dir: Path) -> dict[str, dict[str, dict]]:
    """@brief 把套件目录下每个运行的 result 按任务和数据类型归档，并补上发布加速比。

    @param suite_dir 套件输出目录。
    @return ``{任务: {dtype: result}}`` 的嵌套字典。
    """
    out: dict[str, dict[str, dict]] = {}
    for d in sorted(Path(suite_dir).iterdir()):
        if d.is_dir() and (d / "result.json").exists():
            r = read_result(d)
            r["published_speedup"] = conclusion_of_run(d, r)["speedup"]
            out.setdefault(r["task"], {})[r["dtype"]] = r
    return out


def plot_frontiers(suite_dir: Path, dest: Path | None = None) -> Path:
    """@brief 画每个任务的前沿子图：加速比对累积 API 花费。

    @param suite_dir 套件输出目录。
    @param dest 图片路径；为 ``None`` 时写到套件目录下的 ``frontiers.png``。
    @return 实际写出的图片路径。
    """
    data = _collect(suite_dir)
    tasks = [t for t in ORDER if t in data] + [t for t in data if t not in ORDER]
    cols = 3
    rows = max(1, -(-len(tasks) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.9 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, task in zip(axes.ravel(), tasks):
        ax.set_visible(True)
        for dt, res in sorted(data[task].items()):
            color, label = DTYPE_STYLE.get(dt, ("gray", dt))
            pts = frontier(res)
            if pts:
                xs, ys = zip(*pts)
                ax.step(xs, ys, where="post", color=color, alpha=0.5)
                ax.plot(xs, ys, "o", color=color, label=label, markersize=4)
            if res.get("published_speedup"):
                ax.axhline(res["published_speedup"], color=color, lw=0.6, ls=":")
        ax.axhline(1.0, color="black", ls="--", lw=0.8)
        ax.set_title(task, fontsize=10)
        ax.set_xlabel("cumulative API cost (USD)", fontsize=8)
        ax.set_ylabel("speedup vs best baseline", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Best validated candidate vs cumulative API cost (dashed = parity with the best vendor baseline (MLX / torch-MPS / compiled variants); dotted = confirmed with a fresh seed)", fontsize=9)
    fig.tight_layout()
    dest = dest or Path(suite_dir) / "frontiers.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def plot_summary(suite_dir: Path, dest: Path | None = None) -> Path:
    """@brief 画按任务分组的发布加速比柱状图。

    @param suite_dir 套件输出目录。
    @param dest 图片路径；为 ``None`` 时写到套件目录下的 ``summary.png``。
    @return 实际写出的图片路径。
    """
    data = _collect(suite_dir)
    tasks = [t for t in ORDER if t in data] + [t for t in data if t not in ORDER]
    dtypes = sorted({dt for t in data.values() for dt in t})
    fig, ax = plt.subplots(figsize=(1.0 * len(tasks) + 3, 4.2))
    w = 0.8 / max(len(dtypes), 1)
    for j, dt in enumerate(dtypes):
        xs, ys = [], []
        for i, t in enumerate(tasks):
            r = data[t].get(dt)
            if r and r.get("published_speedup"):
                xs.append(i + j * w - 0.4 + w / 2)
                ys.append(r["published_speedup"])
        ax.bar(xs, ys, width=w, color=DTYPE_STYLE.get(dt, ("gray", dt))[0], label=DTYPE_STYLE.get(dt, ("gray", dt))[1])
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("confirmed speedup vs best vendor baseline")
    ax.legend()
    fig.tight_layout()
    dest = dest or Path(suite_dir) / "summary.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def plot_suite(suite_dir: Path) -> list[Path]:
    """@brief 一次画出前沿图和汇总柱状图。

    @param suite_dir 套件输出目录。
    @return 两张图片的路径列表。
    """
    return [plot_frontiers(suite_dir), plot_summary(suite_dir)]


if __name__ == "__main__":
    for p in plot_suite(Path(sys.argv[1])):
        print(p)
