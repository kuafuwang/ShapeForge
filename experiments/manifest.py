"""Experiment identity written before the first child search.

The file is ``experiment.json`` in the experiment root. Frozen conditions and each arm's switch delta are immutable.
A later invocation may append child directories, add an arm that was not declared yet, or append a budget extension.
Changing the frozen conditions, or an arm's delta, is a new experiment and is refused.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from taic.control import write_json_atomic

FILENAME = "experiment.json"
VERSION = 1
KINDS = ("suite", "repeatability", "search_ab", "contract_menu", "committee_ab", "single")


class ManifestError(ValueError):
    """@brief 目录已属于另一个实验、或清单与本次调用对不上时抛出的错误。

    The directory already belongs to a different experiment.
    """


def canonical(doc: dict) -> str:
    """@brief 把字典序列化成可比较的规范 JSON。

    @param doc 要序列化的字典。
    @return 键已排序、分隔符紧凑的 JSON 文本。
    """
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_hash(frozen: dict) -> str:
    """@brief 算冻结条件的哈希，实验组不参与。

    SHA-256 of the canonical frozen conditions. Arms are not part of the hash.

    @param frozen 冻结条件字典。
    @return 十六进制哈希串。
    """
    return hashlib.sha256(canonical(frozen).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    """@brief 取当前仓库的 HEAD 提交号。

    @return 提交号；仓库不可用或 git 失败时返回空串。
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def experiment_id_for(root: Path) -> str:
    """@brief 算出实验目录的 ID：相对 runs 目录的路径，或目录名。

    Path of ``root`` relative to the runs directory, or the directory name when it lives elsewhere.

    @param root 实验目录。
    @return 用斜线分隔的相对路径；在任一基目录之外时用目录名。
    """

    root = Path(root).resolve()
    bases = []
    env = os.environ.get("AI_LOWERLING_RUNS_DIR")
    if env:
        bases.append(Path(env))
    bases.append(Path(__file__).resolve().parent.parent / "runs")
    for base in bases:
        try:
            return root.relative_to(base.resolve()).as_posix() or "."
        except ValueError:
            continue
    return root.name


@contextmanager
def _exclusive(root: Path):
    """@brief 拿住清单文件的排他锁，保证同时只有一个写者。

    One writer for ``experiment.json``. A top-up and an attach must not replace each other's document.

    @param root 实验目录。
    """

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with open(root / ".experiment.lock", "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def load(root: Path) -> dict | None:
    """@brief 读实验目录下的 ``experiment.json``。

    @param root 实验目录。
    @return 清单字典；文件不存在时返回 ``None``。
    @raises ManifestError 文件存在但不是本版本的清单。
    """
    path = Path(root) / FILENAME
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != VERSION:
        raise ManifestError(f"{path} is not an experiment manifest version {VERSION}")
    return doc


def _arm(arm_id: str, delta: dict, planned_n: int) -> dict:
    """@brief 造清单里的一个实验组条目。

    @param arm_id 实验组名。
    @param delta 这一组相对冻结条件的覆盖项。
    @param planned_n 计划跑几次。
    @return 含 ``arm_id``/``delta``/``planned_n``/``runs`` 的字典。
    """
    return {
        "arm_id": str(arm_id),
        "delta": delta if isinstance(delta, dict) else {},
        "planned_n": int(planned_n),
        "runs": [],
    }


def _same_delta(left: dict, right: dict) -> bool:
    """@brief 比较两份覆盖项在规范形式下是否一致。

    @param left 已登记的覆盖项。
    @param right 本次调用传入的覆盖项。
    @return 规范 JSON 相同时为 ``True``。
    """
    return canonical(left or {}) == canonical(right or {})


def ensure(
    root: Path,
    *,
    kind: str,
    driver: str,
    frozen: dict,
    arms: list[dict],
) -> dict:
    """@brief 新建清单，或在校验冻结条件与实验组后接受本次调用。

    Create the manifest, or accept this call when the frozen conditions and existing deltas match.

    ``arms`` items are ``{"arm_id", "delta", "planned_n"}``. A new arm may be added. An existing arm's
    delta may not change. ``planned_n`` may rise. The file is written before this function returns,
    so a caller can start a child search only after ``ensure`` succeeds.

    @param root 实验目录。
    @param kind 实验类型，取 ``KINDS`` 里的值。
    @param driver 写清单的驱动脚本名。
    @param frozen 冻结条件字典。
    @param arms 要声明的实验组列表。
    @return 实验清单（新建的或已存在的）。
    @raises ManifestError 类型未知、已属于另一个实验或条件不一致时。
    """
    if kind not in KINDS:
        raise ManifestError(f"unknown experiment kind {kind!r}")
    root = Path(root)
    with _exclusive(root):
        return _ensure_locked(root, kind=kind, driver=driver, frozen=frozen, arms=arms)


def _ensure_locked(root: Path, *, kind: str, driver: str, frozen: dict, arms: list[dict]) -> dict:
    """@brief 在已持锁的前提下比对并更新清单。

    @param root 实验目录。
    @param kind 实验类型。
    @param driver 驱动脚本名。
    @param frozen 冻结条件。
    @param arms 本次要声明的实验组。
    @return 写好的实验清单。
    @raises ManifestError 类型/驱动不符、冻结条件不同或已有实验组的覆盖项被改动时。
    """
    existing = load(root) if (root / FILENAME).is_file() else None
    if existing is None:
        doc = {
            "schema_version": VERSION,
            "experiment_id": experiment_id_for(root),
            "kind": kind,
            "driver": driver,
            "created_at": time.time(),
            "git_commit": _git_commit(),
            "frozen": frozen,
            "config_hash": config_hash(frozen),
            "arms": [_arm(item["arm_id"], item.get("delta") or {}, item.get("planned_n") or 0) for item in arms],
            "budget_extensions": [],
        }
        write_json_atomic(root / FILENAME, doc)
        return doc
    if existing.get("kind") != kind or existing.get("driver") != driver:
        raise ManifestError(f"{root} is already a {existing.get('driver')} experiment")
    if existing.get("config_hash") != config_hash(frozen) or canonical(existing.get("frozen") or {}) != canonical(frozen):
        raise ManifestError(f"{root} already records different frozen conditions; start a new experiment directory")
    by_id = {arm["arm_id"]: arm for arm in existing.get("arms") or []}
    changed = False
    for item in arms:
        arm_id = str(item["arm_id"])
        delta = item.get("delta") or {}
        planned = int(item.get("planned_n") or 0)
        current = by_id.get(arm_id)
        if current is None:
            current = _arm(arm_id, delta, planned)
            existing.setdefault("arms", []).append(current)
            by_id[arm_id] = current
            changed = True
            continue
        if not _same_delta(current.get("delta") or {}, delta):
            raise ManifestError(f"arm {arm_id} already has a different switch delta; start a new experiment directory")
        if planned > int(current.get("planned_n") or 0):
            current["planned_n"] = planned
            changed = True
    if changed:
        write_json_atomic(root / FILENAME, existing)
    return existing


def attach(root: Path, arm_id: str, relative: str) -> dict:
    """@brief 把一个子运行目录登记到已有实验组。

    Record one child directory on an existing arm. Does not change frozen conditions or the arm delta.

    @param root 实验目录。
    @param arm_id 实验组名。
    @param relative 子目录相对实验目录的路径。
    @return 更新后的实验清单。
    @raises ManifestError 路径跳到实验目录外、没有清单或实验组不存在时。
    """
    root = Path(root)
    rel = Path(relative).as_posix()
    if rel.startswith("/") or ".." in Path(rel).parts:
        raise ManifestError(f"child path must stay inside the experiment: {relative}")
    with _exclusive(root):
        doc = load(root)
        if doc is None:
            raise ManifestError(f"{root} has no experiment manifest")
        for arm in doc.get("arms") or []:
            if arm.get("arm_id") != arm_id:
                continue
            runs = arm.setdefault("runs", [])
            changed = False
            if rel not in runs:
                runs.append(rel)
                changed = True
            if len(runs) > int(arm.get("planned_n") or 0):
                arm["planned_n"] = len(runs)
                changed = True
            if changed:
                write_json_atomic(root / FILENAME, doc)
            return doc
    raise ManifestError(f"arm {arm_id} is not in {root / FILENAME}")


def extend_budget(root: Path, *, run: str, new_cap: float, at: float | None = None) -> dict:
    """@brief 追记一次预算追加，冻结条件与实验组不变。

    Append a budget extension. The frozen conditions and the arm deltas stay as they were.

    @param root 实验目录。
    @param run 追加预算的子运行。
    @param new_cap 新的上限，单位美元。
    @param at 追加时间；为 ``None`` 时取当前时间。
    @return 更新后的实验清单。
    @raises ManifestError 目录里没有实验清单时。
    """
    root = Path(root)
    with _exclusive(root):
        doc = load(root)
        if doc is None:
            raise ManifestError(f"{root} has no experiment manifest")
        doc.setdefault("budget_extensions", []).append({
            "run": Path(run).as_posix(),
            "new_cap": float(new_cap),
            "at": time.time() if at is None else float(at),
        })
        write_json_atomic(root / FILENAME, doc)
        return doc


def backfill(root: Path, *, kind: str, driver: str, frozen: dict | None = None) -> dict:
    """@brief 为早于清单机制的实验目录补一份清单。

    Create a manifest for an experiment directory that predates manifests.

    Every child directory that holds a ``run_config.json`` (or a ``state.json``) becomes an
    arm and is attached under its relative path. Budget is deliberately not part of the arm
    delta: it changed between reruns, and a delta that embeds it makes a later ``ensure``
    refuse the directory. The frozen block is marked ``backfilled`` so a reader knows the
    conditions were reconstructed, not recorded at launch.

    @param root 实验目录。
    @param kind 实验类型。
    @param driver 驱动脚本名。
    @param frozen 冻结条件；为 ``None`` 时写成 ``{"backfilled": True}``。
    @return 补好（或已存在）的实验清单。
    @raises ManifestError 目录里没有任何可补的子运行目录时。
    """
    root = Path(root)
    existing = load(root)
    if existing is not None:
        return existing
    seen: set[str] = set()
    arms: list[dict] = []
    for name in ("run_config.json", "state.json"):
        for path in sorted(root.rglob(name)):
            child = path.parent
            if child == root:
                continue
            relative = child.relative_to(root).as_posix()
            if relative in seen:
                continue
            seen.add(relative)
            config = _read_json(child / "run_config.json")
            state = _read_json(child / "state.json")
            task = config.get("task") or state.get("task")
            if not task:
                continue
            delta = {"task": task}
            for key in ("dtype", "size", "model"):
                value = config.get(key) or state.get(key)
                if value:
                    delta[key] = value
            arms.append({"arm_id": relative, "delta": delta, "planned_n": 1})
    if not arms:
        raise ManifestError(f"{root} has no child run directories to backfill")
    doc = ensure(root, kind=kind, driver=driver, frozen=frozen or {"backfilled": True}, arms=arms)
    for arm in arms:
        attach(root, arm["arm_id"], arm["arm_id"])
    return load(root) or doc


def _read_json(path: Path) -> dict:
    """@brief 读一个 JSON 文件，读不出来时返回空字典。

    @param path JSON 文件路径。
    @return 字典内容；文件缺失、损坏或不是字典时为空字典。
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def main(argv: list[str] | None = None) -> int:
    """@brief ``python -m experiments.manifest --backfill`` 的入口，给旧目录补清单。

    @param argv 命令行参数；为 ``None`` 时取 ``sys.argv``。
    @return 进程退出码，正常结束时为 0。
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m experiments.manifest")
    parser.add_argument("--backfill", metavar="ROOT", required=True)
    parser.add_argument("--kind", default="suite", choices=sorted(KINDS))
    parser.add_argument("--driver", required=True)
    args = parser.parse_args(argv)
    doc = backfill(Path(args.backfill), kind=args.kind, driver=args.driver)
    print(json.dumps({"experiment_id": doc.get("experiment_id"), "arms": len(doc.get("arms") or []), "root": args.backfill}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
