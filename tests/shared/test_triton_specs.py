"""The Triton specs shown to the model are real Triton: every kernel parameter must be bindable from the launch header
(this is what lets tcenv.triton_runner execute them unchanged). No Triton is needed to check that."""

from __future__ import annotations

import ast

import pytest

from tasks import get_task, list_tasks
from tcenv.triton_runner import parse_launch_header
from ..support.triton_specs import jit_params


@pytest.mark.parametrize("name", list(list_tasks()) + ["mamba2_chunk_scan"])
def test_every_kernel_parameter_is_bound_by_the_launch_header(name):
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

@pytest.mark.parametrize("name", list(list_tasks()) + ["mamba2_chunk_scan"])
def test_header_values_agree_with_the_contract_constants(name):
    t = get_task(name, "fp16", "paper")
    header, _ = parse_launch_header(t.triton_source)
    for cname, _ty, val in t.contract.constants:
        if cname in header:
            assert float(header[cname]) == pytest.approx(float(val)), (name, cname)

def test_header_parser_handles_eps_floats_and_multiline_headers():
    h, nw = parse_launch_header("# Launch: grid = (M,), num_warps = 8, M = 4096, eps = 1e-6\n#   BASE = 10000.0\nimport triton\n# N = 5\n")
    assert nw == 8 and h == {"M": 4096, "eps": 1e-6, "BASE": 10000.0}

from pathlib import Path  # noqa: E402

VENV_PY = Path(__file__).resolve().parents[2] / ".venvs" / "triton-msl" / "bin" / "python"

@pytest.mark.skipif(not VENV_PY.exists(), reason="isolated triton-msl environment not installed (scripts/install_triton_msl.sh)")
@pytest.mark.parametrize("name", ["gemm", "gemm_silu", "conv2d", "bitdelta_gemm", "rmsnorm", "flash_attention", "mamba2_chunk_scan"])
def test_specs_load_as_real_triton_and_bind_all_arguments(name, tmp_path):
    """Dry run in the isolated environment: real Triton parses the spec and every kernel argument gets a value (no compile, no GPU launch)."""
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
    from tcenv.triton_runner import spec_fingerprint

    a = spec_fingerprint(get_task("gemm", "fp16", "paper"))
    assert a == spec_fingerprint(get_task("gemm", "fp16", "paper"))
    assert a != spec_fingerprint(get_task("gemm", "fp16", "small"))
    assert a != spec_fingerprint(get_task("gemm", "bf16", "paper"))
    assert a != spec_fingerprint(get_task("gemm@128x64x32t256", "fp16", "paper"))
    assert a != spec_fingerprint(get_task("gemm_silu", "fp16", "paper"))

@pytest.mark.parametrize("name", list(list_tasks()) + ["flash_attention", "lion", "bitdelta_gemm", "mamba2_chunk_scan"])
def test_header_grid_matches_the_contract_grid(name):
    from tcenv.triton_runner import parse_grid, spec_source

    t = get_task(name, "fp16", "paper")
    src, note = spec_source(t)
    header, _ = parse_launch_header(src)
    g = list(t.contract.grid)
    while len(g) > 1 and g[-1] == 1:
        g.pop()
    if not note:  # a baseline variant may legitimately launch a different (equivalent) grid
        assert parse_grid(src, header) == tuple(g), (name, parse_grid(src, header), g)

def test_baseline_variants_declare_a_grid_and_keep_the_kernel_name():
    from tcenv.triton_runner import parse_grid, spec_source

    for name in ("gemm", "gemv", "flash_attention"):
        t = get_task(name, "fp16", "paper")
        src, note = spec_source(t)
        assert note and f"def {t.contract.kernel_name}(" in src
        header, nw = parse_launch_header(src)
        assert parse_grid(src, header) and nw

@pytest.mark.parametrize("name", ["gemm", "gemm_silu", "conv2d"])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_variants_are_valid_python_with_every_placeholder_substituted(name, dtype):
    """The specs the backend can actually compile (it refuses the fused gemm_silu epilogue and the gathered conv2d operand)."""
    from tcenv.triton_runner import spec_source

    t = get_task(name, dtype, "paper")
    src, note = spec_source(t)
    assert note and "{{" not in src, (name, note)
    tree = ast.parse(src)
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if name == "gemm":
        assert t.contract.kernel_name in funcs
        return
    assert "pipeline" in funcs  # a multi-kernel variant: pipeline(inputs_by_contract_name, num_warps)
    jits = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name != "pipeline"]
    assert len(jits) >= 2
    assert ("torch.float16" if dtype == "fp16" else "torch.bfloat16") in src
    for a in t.contract.inputs:  # the pipeline reads inputs by their contract names
        assert f'inp["{a.name}"]' in src, (name, a.name)

@pytest.mark.parametrize("name", ["gemm", "gemv", "flash_attention"])
def test_single_kernel_baseline_variants_bind_every_parameter(name):
    """The variant is what the compiler baseline runs: its kernel parameters must all be bound by its own header (or a contract scalar)."""
    from tcenv.triton_runner import spec_source

    t = get_task(name, "fp16", "paper")
    src, note = spec_source(t)
    assert note
    params = jit_params(src, t.contract.kernel_name)
    header, nw = parse_launch_header(src)
    lower = {k.lower() for k in header} | {a.name.lower() for a in t.contract.scalars}
    assert len([p for p in params if p.endswith("_ptr")]) == len(t.contract.buffers)
    assert not [p for p in params if not p.endswith("_ptr") and p.lower() not in lower], (name, params, header)
    assert nw == t.contract.num_warps

def test_relu_spec_propagates_nan():
    assert "tl.where(x < 0, 0.0, x)" in get_task("relu", "fp16", "paper").triton_source
