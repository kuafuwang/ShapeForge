"""``affine_ramp``：第一道声明 workspace 的标准题（msl-bundle 的正例题）。

The task is the B5-b vertical slice's subject, so it carries its own evidence rather than borrowing
a fixed fixture: discovery, contract loading, every (dtype, size) instance, the independent fp64
oracle on both sizes, the ABI workspace span, the baseline closure and the rounding口径 that the
reference kernel and the baseline share.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from tcenv.opspec.v2 import instantiate_v2
from tcenv.taskcontract import load_task_contract
from tcenv.taskcontract.baselines import load_baselines
from tcenv.taskcontract.locator import REPO_ROOT, discover_standard_tasks
from tcenv.track.abi import derive_tensor_memory_contract
from tcenv.target import load_target

TASK_DIR = REPO_ROOT / "tasks" / "affine_ramp"
SUPPORT = REPO_ROOT / "tests" / "support" / "msl_bundle_affine_ramp"
SIZES = ("paper", "small")
DTYPES = ("fp16", "bf16")


@pytest.fixture(scope="module")
def contract():
    return load_task_contract(TASK_DIR, dtype="fp16", size="paper")


def test_the_task_is_discovered_as_a_v2_standard_entry():
    locators = {item.op_name: item for item in discover_standard_tasks(REPO_ROOT / "tasks", code_root=REPO_ROOT)}
    assert "affine_ramp" in locators
    locator = locators["affine_ramp"]
    assert (locator.dtype, locator.size) == ("fp16", "paper")
    assert len(locator.deployment_pack_digest) == 64


def test_the_contract_declares_a_workspace_and_no_seg_in_symbols(contract):
    assert contract.op_name == "affine_ramp"
    spaces = {tensor.name: tensor.space for tensor in contract.instance.tensors}
    assert spaces == {"x": "global", "y": "global", "scale_tab": "workspace"}
    # SEG/BASE/STEP are integer attrs, not symbols: the oracle only ever sees attrs.
    assert set(contract.instance.symbols) == {"N", "NS"}
    assert contract.instance.resolved_attrs == {"SEG": 64, "BASE": 2049, "STEP": 2}


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_every_instance_declares_the_ramp_shape(contract, dtype, size):
    instance = instantiate_v2(contract.spec, dtype, size)
    seg = int(instance.resolved_attrs["SEG"])
    assert instance.symbols["NS"] == math.ceil(instance.symbols["N"] / seg)
    assert instance.symbols["N"] == int(contract.spec.sizes[size]["N"])
    assert seg > 0
    tolerances = instance.tolerances
    assert tolerances.rtol > 0 and tolerances.atol_rel > 0


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("size", SIZES)
def test_the_reference_matches_an_independent_hand_computation(contract, dtype, size):
    from tcenv.taskcontract.reference import load_reference

    instance = instantiate_v2(contract.spec, dtype, size)
    oracle = load_reference(TASK_DIR, op_name="affine_ramp", instance=instance)
    n = int(instance.symbols["N"])
    seg = int(instance.resolved_attrs["SEG"])
    base = int(instance.resolved_attrs["BASE"])
    step = int(instance.resolved_attrs["STEP"])
    x = torch.tensor([(index * 37 % 17 - 8) / 8 for index in range(n)], dtype=torch.float32)
    if dtype == "fp16":
        x = x.to(torch.float16).to(torch.float32)
    else:
        x = x.to(torch.bfloat16).to(torch.float32)
    out = oracle.execute(instance, {"x": x}, dict(instance.resolved_attrs))["y"]
    # The oracle sees only inputs/attrs: recompute the ramp here, in fp64, without its help.
    expected = torch.tensor(
        [float(base + step * (index // seg)) * float(x[index]) for index in range(n)], dtype=torch.float64
    )
    assert torch.allclose(out, expected, rtol=0, atol=1e-9)
    # The harness compares in float64: the oracle result is widened, never narrowed.
    assert out.dtype in (torch.float64, torch.float32)


def test_the_abi_workspace_span_is_four_bytes_per_entry(contract):
    target = load_target("apple-m4-max")
    abi = derive_tensor_memory_contract(contract.instance, target=target)
    workspace = [tensor for tensor in abi.tensors if tensor.space == "workspace"]
    assert len(workspace) == 1
    tensor = workspace[0]
    assert tensor.name == "scale_tab" and tensor.dtype == "fp32"
    span = math.prod(int(v) for v in tensor.shape) * 4
    assert span == 4 * int(contract.instance.symbols["NS"])
    assert span % 4 == 0  # the launch grammar requires a multiple of 4


def test_the_baseline_package_loads_and_closes(contract):
    loaded = load_baselines(TASK_DIR, op_name="affine_ramp")
    assert "torch_mps" in loaded.runners
    runner = loaded.runners["torch_mps"]
    assert callable(runner)
    from tasks.affine_ramp import baselines as module

    instance = contract.instance
    assert (module.SEG, module.BASE, module.STEP) == (
        int(instance.resolved_attrs["SEG"]),
        int(instance.resolved_attrs["BASE"]),
        int(instance.resolved_attrs["STEP"]),
    )


def test_the_baseline_is_not_the_candidate_workspace_reader(contract):
    """The baseline builds its own ramp: it receives only ``contract.inputs``."""
    from tasks.affine_ramp import baselines as module

    instance = contract.instance
    x = torch.tensor([(index * 37 % 17 - 8) / 8 for index in range(int(instance.symbols["N"]))], dtype=torch.float16)
    out = module.run(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    seg = int(instance.resolved_attrs["SEG"])
    base = int(instance.resolved_attrs["BASE"])
    step = int(instance.resolved_attrs["STEP"])
    assert float(out[0]) == pytest.approx(float(base) * float(x[0]), rel=1e-3)


def test_the_rounding_split_is_observable_on_this_task():
    """Pre-rounding the ramp to fp16 differs bitwise from rounding once at the output.

    ``BASE = 2049`` is above 2048, where fp16's spacing is 2, so an odd ramp entry is not
    representable and the two construction paths disagree.  This pins the *reference kernel /
    baseline口径* (evidence hygiene); it is not a claim about the harness verdict, which is decided
    by ``tolerances + scaled_error`` only.
    """
    n, seg, base, step = 64, 64, 2049, 2
    x = torch.tensor([(index * 37 % 17 - 8) / 8 for index in range(n)], dtype=torch.float32)
    tab = base + step * torch.arange(1, dtype=torch.float32)
    index = torch.arange(n) // seg
    good = (tab[index] * x).to(torch.float16)
    bad = (tab[index].to(torch.float16) * x).to(torch.float16)
    assert not torch.equal(good.view(torch.int16), bad.view(torch.int16))


def test_the_shipped_bundle_assets_declare_the_same_stage(contract):
    import json

    geometry = json.loads((SUPPORT / "host" / "execution_geometry.json").read_text())
    assert geometry["schema_version"] == 1 and geometry["source"] == "candidate"
    stage = geometry["stages"][0]
    assert stage["kernel"] == "affine_ramp"
    assert stage["threadgroup_bytes"] == 0
    assert stage["workspace"] == [{"name": "scale_tab", "zero": False}]
    tiling = json.loads((SUPPORT / "host" / "tiling.json").read_text())
    write = tiling["writes"][0]
    assert (write["start"], write["step"], write["count"]) == ("BASE", "STEP", "NS")
    assert write["dtype"] == "fp32" and write["kind"] == "linear"
    # ``known_good.metal`` is a pack file and stays raw MSL; it is the identity of the byte-exact
    # device entry the packed bundle reuses.
    known_good = TASK_DIR / "known_good.metal"
    assert known_good.is_file()
    assert contract.fingerprint_document()["known_good_sha256"]
    assert b"device float* scale_tab [[buffer(2)]]" in known_good.read_bytes()
