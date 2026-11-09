#!/usr/bin/env python3
"""B4 (part 2): freeze each used sem_int seed into a fixed v2 task directory.

The dynamic family ``sem_int_<seed>`` is a *generator*, not a task: the registered
``tasks.sem_int`` document generator materialises an OpSpec per seed.  v2's standard entry is a
fixed task contract, so the right migration is to **generate once and freeze**: each used seed
becomes its own ``tasks/sem_int_<seed>/`` directory with the generated OpSpec, an inlined
reference evaluator and no baselines (the family is correctness-only).

The evaluator is *extracted from the generator module*, not rewritten: ``Stmt``, ``wrap``,
``block_of``, ``statements_of``, ``_rules``, ``_implemented_rules`` and ``evaluate`` are copied
verbatim into each ``reference.py`` together with that seed's program document.  That keeps the
int64-then-wrap-int32 semantics byte-identical and keeps the oracle independent of the harness.

Run:  uv run python docs/架构/跨硬件算子生成分工-阶段验收/B4/migrate_sem_int.py [--check]
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

B4 = Path(__file__).resolve().parent
ROOT = B4.parents[3]
TASKS = ROOT / "tasks"
V1_MARKERS = ("task.py", "track.json", "__init__.py")
EVALUATOR_NAMES = {
    "RULE_VERSION",
    "ARITHMETIC",
    "_INT32_MASK",
    "M32",
    "WRAP_KINDS",
    "PLAIN_KINDS",
    "ALL_KINDS",
    "Stmt",
    "wrap",
    "block_of",
    "statements_of",
    "_rules",
    "_implemented_rules",
    "evaluate",
}


def _a5() -> Any:
    """@brief 载入 A5 迁移模块，复用 OpSpec 映射规则。

    @return A5 migrate 模块。
    """
    spec = importlib.util.spec_from_file_location("_b4_a5_migrate", B4.parent / "A5" / "migrate.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _evaluator_source() -> str:
    """@brief 从 ``tasks/sem_int/program.py` 提取求值器所需的全部顶层定义。

    @return 这些定义的源码文本（含装饰器）。
    """
    path = TASKS / "sem_int" / "program.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    parts: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in EVALUATOR_NAMES:
            # ast.get_source_segment(padded=True) 不一定带上装饰器；Stmt 必须保留 @dataclass，
            # 否则它没有自动生成的 __init__（position arguments 会报 "takes no arguments"）。
            decorators = "\n".join("@" + (ast.get_source_segment(text, decorator) or "") for decorator in node.decorator_list)
            segment = ast.get_source_segment(text, node, padded=True) or ""
            parts.append((decorators + "\n" if decorators else "") + segment)
        elif isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if names and names[0] in EVALUATOR_NAMES:
                parts.append(ast.get_source_segment(text, node) or "")
    return "\n\n".join(part for part in parts if part)


def _reference_source(seed: str, program_document: dict, track: dict) -> str:
    """@brief 生成一个 seed 的 reference.py（内嵌求值器与该 seed 的程序文档）。

    @param seed 题名（``sem_int_<n>``）。
    @param program_document 该 seed 的 ``program`` 文档。
    @param track v1 的 track 登记（gen_inputs 参数与描述）。
    @return 源文本。
    """
    params = track["implementations"]["gen_inputs"]["params"]
    lo, hi = (int(value) for value in params["random"])
    small_lo, small_hi = (int(value) for value in params["small"])
    constants = [
        f"_SEED_MASK = {int(params['seed_mask'])}",
        f"_LO, _HI = {lo}, {hi}",
        f"_SMALL_LO, _SMALL_HI = {small_lo}, {small_hi}",
        f"_EDGE_VALUES = torch.tensor({[int(v) for v in params['edge_values']]}, dtype=torch.int64)",
    ]
    program_json = json.dumps(program_document, ensure_ascii=False)
    return "\n".join(
        [
            f'"""{seed}: independent int64-then-wrap-int32 reference and the frozen v1 input stream.',
            "",
            "The evaluator (Stmt/wrap/block_of/statements_of/_rules/_implemented_rules/evaluate) is",
            "extracted verbatim from the registered sem_int generator; the program document of this",
            "seed is embedded below.  A v2 reference may not import the harness, so nothing here",
            "imports tasks/tcenv.  The arithmetic is 32-bit two's complement with wraparound after",
            "exactly the kinds the rule table marks as wrapping.",
            '"""',
            "",
            "import torch",
            "import json",
            "from dataclasses import dataclass",
            "",
            _evaluator_source(),
            "",
            f"_PROGRAM = json.loads(r'''{program_json}''')",
            "",
            "\n".join(constants),
            "",
            "",
            "def gen_inputs(case, seed, instance):",
            '    """The v1 case/seed stream (random/small/edge, int64 values)."""',
            "    generator = torch.Generator().manual_seed(int(seed) & _SEED_MASK)",
            '    n = int(instance.symbols["N"])',
            '    if case == "random":',
            "        make = lambda: torch.randint(_LO, _HI, (n,), generator=generator, dtype=torch.int64)",
            '    elif case == "small":',
            "        make = lambda: torch.randint(_SMALL_LO, _SMALL_HI, (n,), generator=generator, dtype=torch.int64)",
            "    else:  # edge values",
            "        make = lambda: _EDGE_VALUES[torch.randint(0, len(_EDGE_VALUES), (n,), generator=generator)]",
            '    return {"x": make(), "y": make()}',
            "",
            "",
            "def reference(inputs, attrs):",
            '    """Evaluate this seed\'s program document (int64, then wrap to int32)."""',
            '    return {"out": evaluate(_PROGRAM, inputs["x"], inputs["y"], int(attrs["SCALAR_K"]))}',
            "",
        ]
    )


def main() -> None:
    """@brief 为每个使用中的 seed 生成固定 v2 目录并做装配冒烟。

    @return 无。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    from tasks.sem_int import document as opspec_document
    from tasks.sem_int import program

    frozen = json.loads((B4 / "evidence" / "sem_int-frozen.json").read_text(encoding="utf-8"))
    seeds = []
    for combo in frozen["combos"]:
        if combo["task"] not in seeds:
            seeds.append(combo["task"])
    for seed in seeds:
        seed_int = int(seed.rsplit("_", 1)[1])
        combo = next(combo for combo in frozen["combos"] if combo["task"] == seed)
        # 迁移参数来自冻结证据，而不是 tasks/sem_int/track.json（首次运行后它已退役）。
        track = {"implementations": combo["implementations"], "description": combo["description"], "baselines": combo["baselines"]}
        directory = TASKS / f"sem_int_{seed_int}"  # op_name 由 int(seed) 决定：sem_int_007 -> sem_int_7
        v2_document = _a5().v2_opspec(combo["opspec"], seed)
        # v2 的 OpSpec schema 是闭集，不认识 ``program``：语句表属于 reference/oracle 的
        # 求值数据（已内嵌在 reference.py 的 _PROGRAM 里），不是 v2 合同字段。
        v2_document.pop("program", None)
        files = {
            "opspec.json": json.dumps(v2_document, ensure_ascii=False, indent=2) + "\n",
            "reference.py": _reference_source(seed, combo["opspec"]["program"], track),
            "reference.json": json.dumps({"schema_version": 1, "reference_version": 1, "error_scale": {"kind": "none"}}, indent=2) + "\n",
            # sem_int 是 correctness-only：与 A5 的 sem_* 语义题一样，不写 baselines.json/py
            # （空 baseline 表会被 v2 loader 拒绝，缺省才是合法形式）。
            "display.json": json.dumps({"schema_version": 1, "description": combo["description"]}, ensure_ascii=False, indent=2) + "\n",
        }
        if args.check:
            for name in files:
                print(f"would write {directory / name}")
            continue
        directory.mkdir(exist_ok=True)
        candidates = B4 / "evidence" / "candidates"
        candidates.mkdir(parents=True, exist_ok=True)
        (candidates / f"{seed}-naive.msl").write_text(combo["naive_msl"], encoding="utf-8")
        for name, text in files.items():
            (directory / name).write_text(text, encoding="utf-8")
            print(f"wrote {directory / name}")
        from tcenv.taskcontract import load_task_contract

        contract = load_task_contract(directory, dtype="i32", size="small")
        print(f"loaded v2 contract {contract.op_name}: dtype={contract.dtype} size={contract.size}")
    if not args.check:
        # v1 装配标记随迁移退役；生成器 program.py/document.py 保留为离线作者工具。
        for name in V1_MARKERS:
            path = TASKS / "sem_int" / name
            if path.is_file():
                path.unlink()
                print(f"deleted {path}")
        cache = TASKS / "sem_int" / "__pycache__"
        if cache.is_dir():
            shutil.rmtree(cache)


if __name__ == "__main__":
    main()
