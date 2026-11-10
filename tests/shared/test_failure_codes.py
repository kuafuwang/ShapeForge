"""Every typed failure code in the tree has one runbook row, and the runbook has no orphan rows."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNBOOK = ROOT / "docs" / "失败码.md"
SEARCH_ROOTS = ("tcenv", "taic")
CODE_PATTERN = re.compile(r'^\s+code = "([a-z0-9_]+)"', re.MULTILINE)
ROW_PATTERN = re.compile(r"^\| `([a-z0-9_]+)` \|", re.MULTILINE)


def _codes_in_source() -> dict[str, str]:
    """@brief 扫出代码里所有稳定失败码，并记下它第一次出现的文件。

    @return ``code -> 相对路径``。
    """
    found: dict[str, str] = {}
    for top in SEARCH_ROOTS:
        for path in sorted((ROOT / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for code in CODE_PATTERN.findall(text):
                found.setdefault(code, str(path.relative_to(ROOT)))
    return found


def _rows_in_runbook() -> dict[str, list[str]]:
    """@brief 只读「任务合同与交付平面」那一节的表行，避免把「表怎么读」的示例行当数据。

    @return ``code -> 单元格列表``（含 code 本身）。
    """
    rows: dict[str, list[str]] = {}
    section = False
    for line in RUNBOOK.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line.strip() == "## 任务合同与交付平面"
            continue
        if not section:
            continue
        match = ROW_PATTERN.match(line)
        if match is not None:
            cells = [cell.strip().strip("`") for cell in line.strip().strip("|").split("|")]
            rows[match.group(1)] = cells
    return rows


def test_every_typed_failure_code_has_a_runbook_row():
    """@brief 代码里新增 ``code`` 却没写进 docs/失败码.md，这里失败。

    失败码是给人和 agent 看的接口；只加代码不加翻译，等于让下一个人去读 traceback。
    """
    documented = set(_rows_in_runbook())
    missing = {code: path for code, path in _codes_in_source().items() if code not in documented}
    assert not missing, (
        "docs/失败码.md 缺这些失败码：\n"
        + "\n".join(f"  {code}  ({path})" for code, path in sorted(missing.items()))
    )


def test_the_runbook_has_no_codes_the_code_no_longer_raises():
    """@brief 反向：表里留着一个已经删掉的 ``code``，这里失败。

    只查单向的话，删掉一类错误后 runbook 会永远留着一条让人照做的死路。
    """
    live = set(_codes_in_source())
    stale = sorted(code for code in _rows_in_runbook() if code not in live)
    assert not stale, f"docs/失败码.md 有代码里已不存在的失败码：{', '.join(stale)}"


def test_every_runbook_row_says_what_it_means_and_what_to_do_next():
    """@brief 每行的「含义 / 常见原因 / 下一步 / 权威源」都不许空，也不许写成「见代码」。

    这是 CAKE 那条 runbook 规则的可执行版本：日志旁边的那一行才是半夜救命的东西。
    """
    vague = {"", "-", "见代码", "见实现", "检查配置", "TBD"}
    problems: list[str] = []
    for code, cells in _rows_in_runbook().items():
        if len(cells) != 5:
            problems.append(f"{code}: 应为 5 列（code + 四列），实际 {len(cells)}")
            continue
        for name, cell in zip(("含义", "常见原因", "下一步", "权威源"), cells[1:]):
            if cell.strip() in vague:
                problems.append(f"{code}: {name} 太空（{cell!r}）")
        sources = [part.strip().strip("`") for part in cells[4].split("<br>")]
        for source in sources:
            if not (ROOT / source).is_file():
                problems.append(f"{code}: 权威源不存在（{source}）")
    assert not problems, "docs/失败码.md 有问题行：\n" + "\n".join(f"  {p}" for p in problems)
