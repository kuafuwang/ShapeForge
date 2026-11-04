"""Test-only varlen tiers for the shipped planners.

The shipped tasks declare a handful of tiers; several edge cases only appear on a tier whose
``total_tokens``/``num_seqs`` the task does not ship (an all-ones profile, or a total too small
for an all-prime segmentation).  These helpers write a standard task directory that keeps the
shipped ``reference.py`` and every ABI/semantic field unchanged and replaces only ``sizes`` with
the probe distributions, so the *shipped* planner runs through the public loader and the
``ReferenceModule`` funnel instead of a copy.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence


def write_probe_task(
    origin: str | Path,
    target: str | Path,
    tiers: Mapping[str, Sequence[int]],
) -> Path:
    """@brief 写一份只用探针档位的标准任务目录（reference.py 原样复制）。

    @param origin shipped 任务目录（``tasks/<op>``）。
    @param target 探针任务目录（通常是 ``tmp_path/<op>``）。
    @param tiers ``{档名: seq_lens}``；H/D 与用例域沿用 shipped 文档。
    @return ``target`` 路径。
    """
    origin = Path(origin)
    target = Path(target)
    document = json.loads((origin / "opspec.json").read_text(encoding="utf-8"))
    base = next(iter(document["sizes"].values()))
    hidden = {name: int(base[name]) for name in ("H", "D") if name in base}
    document["sizes"] = {
        str(name): {**hidden, "seq_lens": [int(value) for value in lengths], "usage": ["golden"]}
        for name, lengths in tiers.items()
    }
    # discovery 的默认档必须指向一个存在的档位，否则 OpSpec 校验会拒绝这份探针文档。
    discovery = dict(document.get("discovery") or {})
    discovery["default_instance"] = {
        "dtype": str((discovery.get("default_instance") or {}).get("dtype", "fp16")),
        "size": next(iter(tiers)),
    }
    document["discovery"] = discovery
    target.mkdir(parents=True, exist_ok=True)
    (target / "opspec.json").write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    shutil.copyfile(origin / "reference.py", target / "reference.py")
    return target


__all__ = ["write_probe_task"]
