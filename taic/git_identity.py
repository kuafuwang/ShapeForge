"""Git identity and repository state for review/audit records.

A review uses the same identity source as a commit: ``git var GIT_AUTHOR_IDENT``
(which honours the repository config and Git author environment), with a
``git config user.name/user.email`` fallback. No review form asks for a name.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_IDENT = re.compile(r"^(?P<name>.*?)\s*<(?P<email>[^>]*)>\s+\d+\s+[+-]\d{4}$")


@dataclass(frozen=True)
class GitAuthor:
    """@brief 提交风格的作者身份，供复核、审计记录使用。

    ``source`` 记录身份来自哪里：``git``（仓库配置或 Git 环境变量）、``env``（退化到环境变量）或 ``unknown``。
    """

    name: str
    email: str
    source: str  # git | env | unknown

    @property
    def display(self) -> str:
        """@brief 供记录展示的身份字符串。

        @return 有名字也有邮箱时为 ``名字 <邮箱>``，否则退化为名字、邮箱，最后是 ``unknown``。
        """
        if self.name and self.email:
            return f"{self.name} <{self.email}>"
        return self.name or self.email or "unknown"


def _run(args: list[str], root: Path) -> str:
    """@brief 在 ``root`` 下跑一条 Git 命令，只取标准输出。

    任何失败（进程起不来、超时、非零退出）都返回空字符串，让调用方退化到下一层来源。

    @param args 命令与参数，第一个元素应为 ``git``。
    @param root 命令的工作目录。
    @return 去掉首尾空白的标准输出；失败时为空字符串。
    """
    try:
        result = subprocess.run(
            args,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def author(root: Path | None = None) -> GitAuthor:
    """@brief 取提交风格的作者身份；不改动 Git 配置，也不访问网络。

    顺序是 ``git var GIT_AUTHOR_IDENT``（尊重仓库配置与 Git 作者环境变量），
    其次 ``git config user.name`` / ``user.email``，再次环境变量，最后是 ``unknown``。

    @param root 仓库目录；缺省为本仓库根目录。
    @return 解析出的 ``GitAuthor``；全都取不到时返回名字与邮箱为空、来源为 ``unknown`` 的对象。
    """
    base = Path(root or REPO_ROOT)
    ident = _run(["git", "var", "GIT_AUTHOR_IDENT"], base)
    match = _IDENT.match(ident)
    if match:
        return GitAuthor(match.group("name").strip(), match.group("email").strip(), "git")
    name = _run(["git", "config", "--get", "user.name"], base)
    email = _run(["git", "config", "--get", "user.email"], base)
    if name or email:
        return GitAuthor(name, email, "git")
    env_name = os.environ.get("GIT_AUTHOR_NAME") or os.environ.get("USER") or ""
    env_email = os.environ.get("GIT_AUTHOR_EMAIL") or os.environ.get("EMAIL") or ""
    if env_name or env_email:
        return GitAuthor(env_name, env_email, "env")
    return GitAuthor("", "", "unknown")


def state(root: Path | None = None) -> tuple[str, bool | None]:
    """@brief 复核当时的仓库状态。

    @param root 仓库目录；缺省为本仓库根目录。
    @return ``(HEAD, dirty)``；Git 不可用时返回 ``("unknown", None)``。
    """
    base = Path(root or REPO_ROOT)
    commit = _run(["git", "rev-parse", "HEAD"], base)
    if not commit:
        return "unknown", None
    status = _run(["git", "status", "--porcelain"], base)
    return commit, bool(status)
