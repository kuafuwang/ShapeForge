"""Entry docs and skills stay reachable: no dead links, no resurrected tombstones, no unwired skill trees.

Rules are numbered data (`RULES`) so a failure names the rule that broke, the way the CAKE skill gates do it.
Each rule reads only what a newcomer actually opens, so a hit is always worth fixing:

* **D-T01** `docs/README.md` 是被删掉的旧入口，只能出现在"它不存在/别重建"的句子里。
* **D-L01** 入口文档里的站内链接：文件要在，`#anchor` 要能在目标文件里找到对应标题。
* **D-P01** 入口文档里带反引号、首段是仓库顶层目录的路径必须存在（运行期产物除外）。
* **D-S01** 技能：权威源在 `docs/agent-skills.md` 里登记，别名树是相对符号链接，描述带触发条件。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: 新人第一小时真的会打开的文件；规则只扫这些，避免把历史文档的旧路径变成噪音。
ENTRY_DOCS = (
    "README.md",
    "README.zh-CN.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CONTRIBUTING.md",
    "tests/README.md",
    "docs/agent-skills.md",
)

#: 技能权威源；别名树（.claude/.codex/.cursor/.agents）只是指向它们的相对符号链接。
CANONICAL_SKILLS = (
    ".pi/skills/harness-feedback",
    ".pi/skills/shapeforge",
    "taic/flows/kernel_once/skills/taic-tools",
)
ALIAS_TREES = (".claude/skills", ".codex/skills", ".cursor/skills", ".agents/skills")

#: 只在测试失败后由 harness 写出来的路径，不是仓库里的静态文件。
GENERATED_PATHS = frozenset({"logs/pytest/triage/triage_summary.md"})

#: 已退役入口：只允许以墓碑形式出现。新增一条必须同时更新 `docs/agent-skills.md` 与本文档的说明。
TOMBSTONES = ("docs/README.md",)
TOMBSTONE_WORDS = ("不存在", "不要", "不得", "别", "does not exist", "do not", "never")

RULES = {
    "D-T01": "退役入口只能以墓碑形式出现",
    "D-L01": "入口文档的站内链接与 #anchor 必须指向真实文件与真实标题",
    "D-P01": "入口文档引用的仓库顶层路径必须存在",
    "D-S01": "技能权威源已登记、别名是符号链接、描述带触发条件",
}

CITED_PATH = re.compile(r"`([A-Za-z0-9_.\-/]+\.[A-Za-z0-9]+)`")
LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _entry_docs() -> list[Path]:
    """@brief 入口文档及其权威技能文件，按解析后的真实路径去重（别名树是符号链接）。

    @return 去重后的文件列表。
    """
    paths = [ROOT / name for name in ENTRY_DOCS]
    for name in CANONICAL_SKILLS:
        paths.append(ROOT / name / "SKILL.md")
    seen: dict[Path, None] = {}
    for path in paths:
        seen.setdefault(path.resolve(), None)
    return list(seen)


def _headings(path: Path) -> set[str]:
    """@brief 收集 Markdown 标题的 GitHub 风格 slug，用来验证 `#anchor`。

    @param path Markdown 文件。
    @return slug 集合。
    """
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match is None:
            continue
        title = match.group(2).lower()
        slug = re.sub(r"[^\w\u4e00-\u9fff \-]", "", title).replace(" ", "-")
        out.add(slug)
    return out


def _top_level_dirs() -> set[str]:
    """@brief 仓库顶层目录名（含隐藏的 agent 目录），用来判断一个引用是否"锚定在仓库根"。

    @return 目录名集合。
    """
    names = {path.name for path in ROOT.iterdir() if path.is_dir()}
    return names | {".github", ".pi", ".agents", ".claude", ".codex", ".cursor"}


def _violations_tombstone() -> list[str]:
    """@brief D-T01：退役入口的每次出现都必须说明它已不存在。

    @return 违规描述列表。
    """
    problems: list[str] = []
    for path in _entry_docs():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for name in TOMBSTONES:
                if name not in line:
                    continue
                low = line.lower()
                if not any(word in low for word in TOMBSTONE_WORDS):
                    problems.append(
                        f"{path.relative_to(ROOT)}:{number} 提到已退役的 {name}，"
                        f"却没说明它不存在/不要重建：{line.strip()[:90]}")
    return problems


def _violations_links() -> list[str]:
    """@brief D-L01：站内链接的文件与 #anchor 都要真实存在。

    @return 违规描述列表。
    """
    problems: list[str] = []
    for path in _entry_docs():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for target in LINK.findall(line):
                if target.startswith(("http://", "https://", "mailto:")):
                    continue
                file_part, _, anchor = target.partition("#")
                target_path = (path.parent / file_part).resolve() if file_part else path
                where = f"{path.relative_to(ROOT)}:{number}"
                if not target_path.exists():
                    problems.append(f"{where} 链接目标不存在：{target}")
                    continue
                if anchor and target_path.suffix == ".md" and anchor.lower() not in _headings(target_path):
                    problems.append(f"{where} 链接锚点不存在：{target}")
    return problems


def _violations_cited_paths() -> list[str]:
    """@brief D-P01：反引号里以仓库顶层目录开头的路径必须存在。

    只认首段是顶层目录的引用：`opspec.json`、`TASK.md` 这类是相对当前运行目录的文件名，
    按仓库根去校验只会制造假警报，而假警报会让整条规则失效。

    @return 违规描述列表。
    """
    tops = _top_level_dirs()
    problems: list[str] = []
    for path in _entry_docs():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for ref in CITED_PATH.findall(line):
                if ref in GENERATED_PATHS or ref.split("/")[0] not in tops:
                    continue
                if ref in TOMBSTONES and any(word in line.lower() for word in TOMBSTONE_WORDS):
                    continue  # D-T01 允许的暮碑句：这里说的是“它不存在”，不是在引用一份文件
                if not (ROOT / ref).exists():
                    problems.append(f"{path.relative_to(ROOT)}:{number} 引用的路径不存在：{ref}")
    return problems


def _violations_skills() -> list[str]:
    """@brief D-S01：技能权威源登记、别名指向、描述触发条件三件事一起查。

    @return 违规描述列表。
    """
    problems: list[str] = []
    index = (ROOT / "docs" / "agent-skills.md").read_text(encoding="utf-8")
    for name in CANONICAL_SKILLS:
        skill = ROOT / name / "SKILL.md"
        if not skill.is_file():
            problems.append(f"权威技能源缺失：{name}/SKILL.md")
            continue
        if name not in index:
            problems.append(f"docs/agent-skills.md 没有登记权威技能源：{name}")
        text = skill.read_text(encoding="utf-8")
        front = text.split("---")[1] if text.startswith("---") else ""
        if "description:" not in front:
            problems.append(f"{name}/SKILL.md 缺少 frontmatter description")
        elif "use when" not in front.lower():
            problems.append(f"{name}/SKILL.md 的 description 没写触发条件（Use when …），模型无法判断何时加载")
    for tree in ALIAS_TREES:
        alias_root = ROOT / tree
        if not alias_root.is_dir():
            problems.append(f"别名技能树缺失：{tree}")
            continue
        for entry in sorted(alias_root.iterdir()):
            if not entry.is_symlink():
                problems.append(f"{tree}/{entry.name} 不是符号链接；技能只能有一份权威源")
                continue
            if not entry.resolve().is_dir():
                problems.append(f"{tree}/{entry.name} 指向不存在的目标：{entry.readlink()}")
    return problems


def test_no_entry_doc_resurrects_a_retired_entrypoint():
    """@brief D-T01：`docs/README.md` 只能出现在"已删除、别重建"的句子里。"""
    problems = _violations_tombstone()
    assert not problems, f"[D-T01] {RULES['D-T01']}\n" + "\n".join(f"  {p}" for p in problems)


def test_every_internal_link_and_anchor_resolves():
    """@brief D-L01：入口文档与技能的站内链接不烂，锚点跟着标题改名一起更新。"""
    problems = _violations_links()
    assert not problems, f"[D-L01] {RULES['D-L01']}\n" + "\n".join(f"  {p}" for p in problems)


def test_cited_repository_paths_exist():
    """@brief D-P01：入口文档反引号里引用的仓库顶层路径都存在。"""
    problems = _violations_cited_paths()
    assert not problems, f"[D-P01] {RULES['D-P01']}\n" + "\n".join(f"  {p}" for p in problems)


def test_skills_stay_wired_to_one_canonical_source():
    """@brief D-S01：权威源被登记、别名树全是符号链接、描述写清触发条件。"""
    problems = _violations_skills()
    assert not problems, f"[D-S01] {RULES['D-S01']}\n" + "\n".join(f"  {p}" for p in problems)
