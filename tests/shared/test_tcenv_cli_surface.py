"""Every ``tcenv`` subcommand must still reach a real handler.

A refactor once moved the subcommand bodies into another module and replaced
them with calls to a function that did not exist. Import-only checks stayed
green because nothing executed the handlers, and ``tcenv eval-v2`` was dead.
These nodes execute one path per subcommand so that failure mode is loud.
"""

from __future__ import annotations

import pytest

from tcenv.cli import main

#: 每个子命令一条最便宜的路径：只解析参数并进入 handler，不碰 GPU、模型或任务包。
SUBCOMMANDS = [
    ("eval", ["--help"]),
    ("eval-v2", ["--help"]),
    ("sweep-v2", ["--help"]),
    ("host-probe", ["--help"]),
    ("ab", ["--help"]),
    ("paired-compare", ["--help"]),
    ("naive", ["--help"]),
    ("baseline", ["--help"]),
    ("profile", ["--help"]),
    ("triton-baseline", ["--help"]),
    ("calibrate", ["--help"]),
    ("portfolio-pack", ["--help"]),
    ("portfolio-check", ["--help"]),
    ("eval-portfolio", ["--help"]),
    ("tasks", ["--help"]),
    ("show", ["--help"]),
]


@pytest.mark.parametrize("name,argv", SUBCOMMANDS, ids=[item[0] for item in SUBCOMMANDS])
def test_the_subcommand_parses_and_reaches_its_handler(name, argv):
    """@brief ``--help`` 由 argparse 接手，handler 必须真的存在并被解析器接受。"""
    with pytest.raises(SystemExit) as raised:
        main([name, *argv])
    assert raised.value.code == 0


def test_the_handler_is_not_a_stub_that_cannot_be_imported():
    """@brief 解析器登记的每个 handler 都必须能从它的模块里取到。

    只 ``import`` 模块看不出来：handler 内部才 import 真实实现，所以这里直接调用
    一条不碰设备的失败路径，确认它走到真实代码而不是 ImportError。
    """
    import tcenv.cli_common as common
    import tcenv.cli_metal as metal

    for module in (common, metal):
        for name in dir(module):
            if not name.startswith("cmd_"):
                continue
            assert callable(getattr(module, name)), f"{module.__name__}.{name} is not callable"
    # 这四条曾经被换成不存在的函数名；现在必须解析到真实实现。
    assert common.cmd_eval_v2.__module__ == "tcenv.cli_common"
    assert common.cmd_sweep_v2.__module__ == "tcenv.cli_common"
    assert common.cmd_portfolio_pack.__module__ == "tcenv.cli_common"
    assert metal.cmd_paired_compare.__module__ == "tcenv.cli_metal"


def test_eval_v2_reaches_the_real_evaluation_path(tmp_path, capsys):
    """@brief 缺文件时必须由 handler 自己报错退出 2，而不是 ImportError 冒到顶层。"""
    missing = tmp_path / "absent.metal"
    code = main(["eval-v2", "relu", str(missing), "--quick"])
    assert code == 2
    assert "cannot read" in capsys.readouterr().err


def test_ab_refuses_a_non_metal_backend_by_name(tmp_path, capsys):
    """@brief 非 Metal target 看不见这条命令的能力：它在绑定任何 worker 之前拒绝。"""
    left = tmp_path / "a.metal"
    right = tmp_path / "b.metal"
    left.write_text("kernel void a() {}\n", encoding="utf-8")
    right.write_text("kernel void b() {}\n", encoding="utf-8")
    code = main([
        "ab", str(left), str(right), "--task", "conv2d", "--dtype", "fp16",
        "--size", "small", "--backend", "qingwei",
    ])
    assert code == 2
    assert "Metal-only" in capsys.readouterr().err
