#!/usr/bin/env python3
"""Check that this checkout can actually run something, and print the legal next step when it cannot.

    uv run python scripts/doctor.py [--json] [--port 8765]

Every check prints PASS / WARN / FAIL and, when it is not PASS, the one action that fixes it. The script never
mutates the repository, never touches the GPU beyond asking Metal for the default device, and never reads
credential values — only whether the provider setup exists.

Exit status is 0 when no check failed, 1 when at least one failed, 2 when the arguments are wrong. WARN does not
fail the run: a machine without a Metal device can still read evidence and run the CPU suite.

Read the checks top to bottom; they are ordered by what a newcomer hits first (`CHECKS` at the bottom).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MIN_PYTHON = (3, 12)


def _run(argv: list[str]) -> tuple[int, str]:
    """@brief 跑一条只读命令，把退出码和合并后的输出交回。

    @param argv 命令与其参数。
    @return ``(returncode, stdout+stderr)``；命令不存在时返回 ``(127, ...)``。
    """
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return 127, f"{argv[0]}: not found"
    except subprocess.TimeoutExpired:
        return 124, f"{' '.join(argv)}: timed out"
    return done.returncode, (done.stdout + done.stderr).strip()


def _result(check_id: str, status: str, detail: str, next_step: str = "") -> dict:
    """@brief 组装一条检查结果。

    @param check_id 稳定的检查名，测试与日志靠它引用。
    @param status ``pass`` / ``warn`` / ``fail``。
    @param detail 一行说明，写实际看到的值。
    @param next_step 非 PASS 时唯一一条可执行的下一步；PASS 时为空。
    @return 结果字典。
    """
    return {"id": check_id, "status": status, "detail": detail, "next": next_step}


def check_repo_layout(root: Path) -> dict:
    """@brief 仓库根该有的东西是否都在；目录被搬走或只克隆了一半时最先炸这里。

    @param root 待检查的仓库根。
    @return 检查结果。
    """
    missing = [
        name
        for name in ("pyproject.toml", "AGENTS.md", "README.md", "tasks", "tcenv", "taic", "tests")
        if not (root / name).exists()
    ]
    if missing:
        return _result("repo.layout", "fail", f"{root}: missing {', '.join(missing)}",
                       f"run doctor from inside the repository root (currently {root})")
    return _result("repo.layout", "pass", str(root))


def check_python() -> dict:
    """@brief 解释器版本要满足 pyproject 的 ``requires-python``。

    @return 检查结果。
    """
    version = sys.version_info[:3]
    shown = ".".join(str(part) for part in version)
    if version[:2] < MIN_PYTHON:
        return _result("python.version", "fail", f"{shown} < {'.'.join(map(str, MIN_PYTHON))}",
                       f"use a Python >= {'.'.join(map(str, MIN_PYTHON))} interpreter, e.g. `uv run python scripts/doctor.py`")
    return _result("python.version", "pass", f"{shown} ({sys.executable})")


def check_uv() -> dict:
    """@brief ``uv`` 是本仓库唯一的依赖入口，缺了 README 的每条命令都跑不了。

    @return 检查结果。
    """
    path = shutil.which("uv")
    if path is None:
        return _result("uv.present", "fail", "uv not on PATH",
                       "install uv (https://docs.astral.sh/uv/) then re-run this check")
    return _result("uv.present", "pass", path)


def check_core_imports() -> dict:
    """@brief 核心包能否导入；导入失败通常是依赖没同步，不是代码错。

    @return 检查结果。
    """
    broken: list[str] = []
    for name in ("tcenv", "taic", "tasks", "observe"):
        if importlib.util.find_spec(name) is None:
            broken.append(name)
    if broken:
        return _result("deps.core", "fail", f"cannot import {', '.join(broken)}",
                       "uv sync")
    return _result("deps.core", "pass", "tcenv, taic, tasks, observe importable")


def check_metal_toolchain() -> dict:
    """@brief Metal 编译器在不在；生产后端是 Apple GPU，没有它连 naive 路径都过不去。

    @return 检查结果。
    """
    if sys.platform != "darwin":
        return _result("metal.toolchain", "warn", f"{sys.platform}: Metal toolchain is macOS only",
                       "reading evidence and the CPU suite still work; run GPU checks on an Apple Silicon machine")
    code, out = _run(["xcrun", "-f", "metal"])
    if code != 0:
        return _result("metal.toolchain", "fail", out or "xcrun could not find the metal compiler",
                       "xcode-select --install (or `sudo xcode-select -s /Applications/Xcode.app`)")
    return _result("metal.toolchain", "pass", out)


def check_metal_device() -> dict:
    """@brief 问 Metal 要默认设备；并行跑 GPU 任务的机器上这一条最常被人忽略。

    @return 检查结果。
    """
    try:
        import Metal  # type: ignore[import-not-found]  # pyobjc，只在 macOS 上有
    except Exception as exc:  # noqa: BLE001  pyobjc 缺失或不是 macOS 都落到这里
        return _result("metal.device", "warn", f"pyobjc Metal unavailable: {type(exc).__name__}",
                       "uv sync; on non-macOS this stays a warning and GPU tests are skipped")
    try:
        device = Metal.MTLCreateSystemDefaultDevice()
    except Exception as exc:  # noqa: BLE001  驱动异常不该让自检崩掉
        return _result("metal.device", "warn", f"MTLCreateSystemDefaultDevice raised {type(exc).__name__}",
                       "re-login, or check that no other process holds the GPU exclusively")
    if device is None:
        return _result("metal.device", "fail", "MTLCreateSystemDefaultDevice returned nil",
                       "this machine exposes no Metal device; use an Apple Silicon machine for GPU work")
    try:
        name = str(device.name())
    except Exception:  # noqa: BLE001  老系统上 name() 可能不可用
        name = "unnamed device"
    return _result("metal.device", "pass", f"{name} ({platform.machine()})")


def check_host_program_toolchain(root: Path) -> dict:
    """@brief V3 的受限 C99/Wasm Host Program 工具链：clang + wasm-ld + Wasm engine。

    缺任何一项都是 ``backend_capability_unavailable``，但只 WARN：``host/tiling.json`` 的 linear
    路径与整个 CPU 套件不受影响。真正需要 Host Program 的候选会在 build/quick 处得到同名的
    typed 拒答，而不是基础设施崩溃。

    @param root 仓库根（用来把同名包导入到当前进程）。
    @return 检查结果。
    """
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from tcenv.hostprogram.errors import HostCapabilityUnavailable
        from tcenv.hostprogram.toolchain import discover_toolchain

        toolchain = discover_toolchain()
    except HostCapabilityUnavailable as exc:
        return _result(
            "host.toolchain",
            "warn",
            f"backend_capability_unavailable: {exc}",
            "brew install llvm lld && uv sync   # then: uv run python scripts/doctor.py",
        )
    except Exception as exc:  # noqa: BLE001 - doctor must never crash on a broken optional plane
        return _result(
            "host.toolchain",
            "warn",
            f"backend_capability_unavailable: {type(exc).__name__}: {exc}",
            "uv sync && uv run python scripts/doctor.py",
        )
    return _result(
        "host.toolchain",
        "pass",
        f"{toolchain.clang_version}; {toolchain.wasm_ld}; {toolchain.engine} {toolchain.engine_version}",
    )


def check_tasks(root: Path) -> dict:
    """@brief 标准任务目录能否被发现；发现为空说明 ``tasks/`` 被搬动或改坏了。

    @param root 仓库根。
    @return 检查结果。
    """
    sys.path.insert(0, str(root))
    if not (root / "tasks").is_dir():
        return _result("tasks.discovery", "fail", f"{root / 'tasks'} is missing",
                       "restore tasks/ from git: `git checkout -- tasks`")
    try:
        from tcenv.taskcontract import discover_standard_tasks

        v2 = [locator.task_id for locator in discover_standard_tasks()]
    except Exception as exc:  # noqa: BLE001  v2 发现失败要给出可执行的下一步
        return _result("tasks.discovery", "fail", f"v2 discovery raised {type(exc).__name__}: {exc}",
                       "uv run python -m tcenv.cli tasks to see the failing contract")
    if not v2:
        return _result("tasks.discovery", "fail", "no v2 standard task directories found",
                       "uv run python -m tcenv.cli tasks to see what discovery rejected")
    from tasks import list_tasks

    legacy = list_tasks(include_phase2=True)
    return _result(
        "tasks.discovery",
        "pass",
        f"{len(v2)} v2 tasks ({', '.join(v2[:4])}, …); {len(legacy)} preserved v1 tasks (`tcenv.cli tasks --legacy`)",
    )


def check_runs_dir(root: Path) -> dict:
    """@brief 运行目录可写；只读挂载或权限错时，搜索会在跑到一半才失败。

    @param root 仓库根。
    @return 检查结果。
    """
    runs = Path(os.environ.get("AI_LOWERLING_RUNS_DIR") or (root / "runs"))
    target = runs if runs.exists() else runs.parent
    if not target.exists():
        return _result("runs.writable", "fail", f"{runs} and {target} are both missing",
                       f"mkdir -p {runs}")
    if not os.access(target, os.W_OK):
        return _result("runs.writable", "fail", f"{target} is not writable",
                       f"fix permissions: chmod u+w {target}")
    state = "exists" if runs.exists() else "creatable"
    return _result("runs.writable", "pass", f"{runs} ({state})")


def check_dashboard(port: int) -> dict:
    """@brief 看板是否在跑；它不是必需品，但它是读证据最快的一条路。

    @param port 看板端口，默认 8765。
    @return 检查结果。
    """
    with socket.socket() as sock:
        sock.settimeout(0.3)
        alive = sock.connect_ex(("127.0.0.1", port)) == 0
    if alive:
        return _result("dashboard.http", "pass", f"http://127.0.0.1:{port}/ is listening")
    return _result("dashboard.http", "warn", f"nothing listening on 127.0.0.1:{port}",
                   "uv run python -m observe   # optional: local dashboard over the index")


def check_credentials() -> dict:
    """@brief 只报告凭据来源是否就位，绝不打印凭据值。

    @return 检查结果。
    """
    if os.environ.get("OPENAI_API_KEY"):
        return _result("credentials.provider", "pass", "OPENAI_API_KEY is set in the environment")
    candidate = Path(os.path.expanduser("~/.pi/agent/models.json"))
    if candidate.is_file():
        return _result("credentials.provider", "pass", f"credentials file present: {candidate}")
    return _result("credentials.provider", "warn", "no provider credentials found",
                   "only needed for paid search: export OPENAI_API_KEY (or provide ~/.pi/agent/models.json)")


def run_checks(root: Path | None = None, port: int = 8765) -> list[dict]:
    """@brief 按新人碰到的顺序跑完全部检查。

    @param root 仓库根；默认本文件所在仓库。
    @param port 看板端口。
    @return 结果字典列表。
    """
    root = Path(root) if root is not None else REPO_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))  # 从任何 cwd 调用都能 import 本仓库的包
    return [
        check_repo_layout(root),
        check_python(),
        check_uv(),
        check_core_imports(),
        check_metal_toolchain(),
        check_metal_device(),
        check_host_program_toolchain(root),
        check_tasks(root),
        check_runs_dir(root),
        check_dashboard(port),
        check_credentials(),
    ]


def _print_human(results: list[dict]) -> str:
    """@brief 把结果排成表，并在结尾只列出非 PASS 项对应的下一步。

    @param results 检查结果。
    @return 汇总行。
    """
    width = max(len(item["id"]) for item in results)
    order = {"pass": 0, "warn": 1, "fail": 2}
    for item in sorted(results, key=lambda r: (order[r["status"]], r["id"])):
        label = item["status"].upper()
        print(f"[{label:4s}] {item['id']:{width}s}  {item['detail']}")
    failed = [item for item in results if item["status"] == "fail"]
    warned = [item for item in results if item["status"] == "warn"]
    summary = f"{len(results) - len(failed) - len(warned)} passed, {len(warned)} warnings, {len(failed)} failed"
    print(f"\n{summary}")
    actionable = failed + warned
    if actionable:
        print("\nNext:")
        for item in actionable:
            print(f"  - {item['next']}")
    return summary


def main(argv: list[str] | None = None) -> int:
    """@brief 解析参数、跑检查、按结果返回退出码。

    @param argv 命令行参数；None 表示用 ``sys.argv``。
    @return 0 全部通过（WARN 不算失败），1 有 FAIL。
    """
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="Check that this checkout can run: toolchain, GPU, tasks, runs dir, dashboard, credentials.",
        epilog="Exit code 0 = no failures (warnings allowed), 1 = at least one failed check.",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output, one JSON object")
    parser.add_argument("--port", type=int, default=int(os.environ.get("OBSERVE_PORT", "8765")),
                        help="dashboard port to probe (default: OBSERVE_PORT or 8765)")
    parser.add_argument("--root", default=None, help="repository root to inspect (default: this checkout)")
    args = parser.parse_args(argv)

    results = run_checks(root=Path(args.root) if args.root else None, port=args.port)
    if args.json:
        print(json.dumps({"results": results, "ok": not any(r["status"] == "fail" for r in results)},
                         ensure_ascii=False, indent=2))
    else:
        _print_human(results)
    return 0 if not any(r["status"] == "fail" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
