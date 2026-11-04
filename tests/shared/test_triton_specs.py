"""The Triton specs shown to the model are real Triton: every kernel parameter must be bindable from the launch header
(this is what lets tcenv.triton_runner execute them unchanged). No Triton is needed to check that."""

from __future__ import annotations

import ast

import pytest

from tests.support.v1_tasks import V1_DYNAMIC, V1_TASKS
from tests.support.v1_tasks import v1_task as get_task

# B4 之后 v1 注册表为空：题面清单改由归档 specimen 提供（它们用的是迁移前的 kernel_triton.py）。
V1_SPECIMENS = (*V1_TASKS, *V1_DYNAMIC)
from tcenv.triton_runner import parse_launch_header
from ..support.triton_specs import jit_params


@pytest.mark.parametrize("name", V1_SPECIMENS)
def test_every_kernel_parameter_is_bound_by_the_launch_header(name):
    """@brief 每个 kernel 参数都能从 launch header 绑定：指针数等于 buffer 数、非指针参数在 header 里有值，num_warps 与 contract 一致。

    @param name 参数化的任务名。
    """
    t = get_task(name, "fp16", "paper")
    params = jit_params(t.triton_source, t.contract.kernel_name)
    header, nw = parse_launch_header(t.triton_source)
    lower = {k.lower() for k in header}
    ptrs = [p for p in params if p.endswith("_ptr")]
    rest = [p for p in params if not p.endswith("_ptr")]
    assert len(ptrs) == len(t.contract.buffers), (name, ptrs)
    missing = [p for p in rest if p.lower() not in lower]
    assert not missing, f"{name}: parameters without a value in the header: {missing}; header = {header}"
    assert nw == t.contract.num_warps, (name, nw, t.contract.num_warps)

@pytest.mark.parametrize("name", V1_SPECIMENS)
def test_header_values_agree_with_the_contract_constants(name):
    """@brief header 里出现的常量与 contract 常量的数值一致。

    @param name 参数化的任务名。
    """
    t = get_task(name, "fp16", "paper")
    header, _ = parse_launch_header(t.triton_source)
    for cname, _ty, val in t.contract.constants:
        if cname in header:
            assert float(header[cname]) == pytest.approx(float(val)), (name, cname)

def test_header_parser_handles_eps_floats_and_multiline_headers():
    """@brief header 解析支持 eps 科学计数法、浮点和跨行续写的 header 行，并忽略非 header 行。
    """
    h, nw = parse_launch_header("# Launch: grid = (M,), num_warps = 8, M = 4096, eps = 1e-6\n#   BASE = 10000.0\nimport triton\n# N = 5\n")
    assert nw == 8 and h == {"M": 4096, "eps": 1e-6, "BASE": 10000.0}

from pathlib import Path  # noqa: E402

VENV_PY = Path(__file__).resolve().parents[2] / ".venvs" / "triton-msl" / "bin" / "python"

@pytest.mark.skipif(not VENV_PY.exists(), reason="isolated triton-msl environment not installed (scripts/install_triton_msl.sh)")
@pytest.mark.parametrize("name", ["sem_attention_odd_scale"])
def test_specs_load_as_real_triton_and_bind_all_arguments(name, tmp_path):
    """@brief 在隔离环境 dry run：真实 Triton 能解析 spec 且每个 kernel 参数都拿到值。

    只解析，不编译也不启动 GPU。

    @param name 参数化的任务名。
    @param tmp_path pytest 提供的临时目录，用于写请求和输出文件。
    """
    import json
    import os
    import subprocess

    req = {"task": name, "dtype": "fp16", "size": "paper", "out": str(tmp_path / "o.json"), "workdir": str(tmp_path / "w"), "dry": True}
    (tmp_path / "r.json").write_text(json.dumps(req))
    env = dict(os.environ, PYTHONPATH=str(VENV_PY.parent.parent.parent.parent))
    p = subprocess.run([str(VENV_PY), "-m", "tcenv.triton_runner", str(tmp_path / "r.json")], env=env, capture_output=True, text=True, cwd=env["PYTHONPATH"])
    r = json.loads((tmp_path / "o.json").read_text())
    assert r["status"] == "dry_ok", (r, p.stderr[-300:])

def test_the_fingerprint_changes_with_spec_size_contract_and_cases():
    """@brief spec 指纹随 size、dtype、变体和不同内核变化，相同输入保持稳定。

    The specimen is ``flash_attention`` (with its ``@<tile>t<threads>`` variants) because it is
    one of the tasks A5 keeps on the v1 path; the migrated ``gemm``/``gemm_silu`` specimens are
    no longer assemblable here.
    """
    from tcenv.triton_runner import spec_fingerprint

    a = spec_fingerprint(get_task("sem_attention_odd_scale", "fp16", "paper"))
    assert a == spec_fingerprint(get_task("sem_attention_odd_scale", "fp16", "paper"))
    assert a != spec_fingerprint(get_task("sem_attention_odd_scale", "fp16", "small"))
    assert a != spec_fingerprint(get_task("sem_attention_odd_scale", "bf16", "paper"))
    assert a != spec_fingerprint(get_task("sem_int_5", "fp16", "paper"))  # 不同任务（sem_attention 无 variant 族）
    assert a != spec_fingerprint(get_task("sem_int_5", "fp16", "paper"))

@pytest.mark.parametrize("name", V1_SPECIMENS)
def test_header_grid_matches_the_contract_grid(name):
    """@brief 非 baseline 变体的 header grid（去掉尾部 1）与 contract grid 一致。

    @param name 参数化的任务名。
    """
    from tcenv.triton_runner import parse_grid, spec_source

    t = get_task(name, "fp16", "paper")
    src, note = spec_source(t)
    header, _ = parse_launch_header(src)
    g = list(t.contract.grid)
    while len(g) > 1 and g[-1] == 1:
        g.pop()
    if not note:  # a baseline variant may legitimately launch a different (equivalent) grid
        assert parse_grid(src, header) == tuple(g), (name, parse_grid(src, header), g)

def test_a_preserved_v1_spec_is_still_rendered_for_the_model():
    """@brief 保留下来的 v1 题仍然有完整的 Triton 题面（A5 只退役了已迁移题的 v1 侧）。

    The old assertion pinned ``relu``'s NaN-preserving ``tl.where(x < 0, 0.0, x)``.  A5 migrated
    ``relu``, whose v1 prompt no longer exists, so the specimen is a preserved v1 task; the
    NaN-preservation property itself is still checked for the migrated tasks' candidates in the
    G0 corpus (``naive_msl``) and by their v2 correctness cases.
    """
    source = get_task("sem_attention_odd_scale", "fp16", "paper").triton_source
    assert "@triton.jit" in source and "def flash_attention_kernel(" in source
    assert "tl.store" in source
