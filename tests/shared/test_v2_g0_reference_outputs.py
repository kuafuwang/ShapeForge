"""A1: v2 reference and sampler outputs against the frozen G0 old-path corpus.

The shipped tasks stay v1.  Each v2 fixture is a test-only mapping of a v1 OpSpec with the
same cases/sizes and a reference that performs the same fp64 math.  For RMSNorm the v1
adversarial sampler is ported into the standard ``gen_inputs`` hook, so the test checks both
the generated input bytes and the fp64 reference bytes against G0.  For sigmoid (which has no
custom sampler) the archived G0 inputs are fed to the v2 reference and its output bytes are
compared.  No GPU, model or candidate is involved.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import pytest
import torch

from tcenv.taskcontract import load_task_contract
from tcenv.tensors import from_raw, to_raw

from ..support.v2_tasks import (
    RMSNORM_G0_REFERENCE,
    SIGMOID_REFERENCE,
    rmsnorm_opspec,
    sigmoid_opspec,
    write_task,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "tests/support/golden/shared/跨硬件算子生成分工-G0/corpus"
INDEX = json.loads((CORPUS / "index.json").read_text(encoding="utf-8"))
MAGIC = b"TG0TENS1"


def _combo(task: str, dtype: str) -> dict:
    """@brief 取 G0 small 档组合。

    @param task 算子名。
    @param dtype 数据类型。
    @return 索引中的组合。
    """
    matches = [
        combo
        for combo in INDEX["combos"]
        if combo["task"] == task and combo["dtype"] == dtype and combo["size"] == "small"
    ]
    assert len(matches) == 1, (task, dtype, len(matches))
    return matches[0]


def _archive(record: dict) -> tuple[dict[str, tuple[bytes, dict]], bytes]:
    """@brief 读取一个 G0 ``.tg0z`` 档案。

    The archive stores the raw storage bytes; the header's ``dtype`` is the logical torch
    dtype (float32), so the caller supplies the task storage dtype when reconstructing inputs.

    @param record case record。
    @return ``(name → (raw, header), archive_sha256)`` 与参考输出原始字节。
    """
    path = CORPUS / record["archive"]
    raw = zlib.decompress(path.read_bytes())
    assert raw.startswith(MAGIC), path
    header_len = struct.unpack("<I", raw[8:12])[0]
    header = json.loads(raw[12 : 12 + header_len])
    payload = raw[12 + header_len :]
    tensors: dict[str, tuple[bytes, dict]] = {}
    reference = b""
    for entry in header:
        piece = payload[entry["offset"] : entry["offset"] + entry["nbytes"]]
        assert len(piece) == entry["nbytes"], (path, entry["name"])
        if entry["role"] == "input":
            tensors[entry["name"]] = (piece, entry)
        elif entry["role"] == "reference":
            reference = piece
    assert reference, path
    assert path.name == f"{record['seed']}.tg0z"
    return tensors, reference


def _inputs(raw: dict[str, tuple[bytes, dict]], dtype: str) -> dict[str, torch.Tensor]:
    """@brief 把档案输入字节还原成逻辑张量（float32 加宽）。

    @param raw 名称 → (原始字节, header)。
    @param dtype 任务存储类型 ``fp16``/``bf16``。
    @return 输入张量。
    """
    return {
        name: from_raw(piece, dtype, tuple(entry["shape"]))
        for name, (piece, entry) in raw.items()
    }


@pytest.mark.parametrize(
    ("task", "opspec", "reference"),
    [
        ("rmsnorm", rmsnorm_opspec(), RMSNORM_G0_REFERENCE),
        ("sigmoid", sigmoid_opspec(), SIGMOID_REFERENCE),
    ],
    ids=["rmsnorm", "sigmoid"],
)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_v2_reference_matches_g0_bytes(tmp_path, task, opspec, reference, dtype):
    """@brief 同一输入下 v2 reference 的输出与 G0 旧路径逐字节一致。"""
    task_dir = write_task(tmp_path, op_name=task, opspec=opspec, reference=reference)
    contract = load_task_contract(task_dir, dtype=dtype)
    combo = _combo(task, dtype)

    assert tuple(contract.instance.case_domain.cases) == tuple(combo["cases"])
    assert contract.instance.size == "small"
    assert contract.instance.tolerances.rtol == combo["tolerances"]["rtol"]
    assert contract.instance.tolerances.atol_rel == combo["tolerances"]["atol_rel"]
    assert contract.instance.tolerances.exact is combo["tolerances"]["exact"]
    assert contract.instance.work_accounting.flops == combo["work"]["flops"]
    assert contract.instance.work_accounting.min_bytes == combo["work"]["min_bytes"]

    checked = 0
    for record in combo["case_records"]:
        raw_inputs, raw_reference = _archive(record)
        inputs = _inputs(raw_inputs, dtype)
        outputs = contract.reference.execute(contract.instance, inputs)
        assert outputs["y"].dtype == torch.float64
        assert outputs["y"].reshape(-1).contiguous().numpy().tobytes() == raw_reference, (
            task,
            dtype,
            record["case"],
            record["seed"],
        )
        checked += 1
    assert checked == len(combo["case_records"]) == 18


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_v2_rmsnorm_sampler_reproduces_g0_input_bytes(tmp_path, dtype):
    """@brief 标准 ``gen_inputs`` 端口复现 G0 的输入字节（种子流/cases/尺度不变）。"""
    task_dir = write_task(tmp_path, op_name="rmsnorm", opspec=rmsnorm_opspec(), reference=RMSNORM_G0_REFERENCE)
    contract = load_task_contract(task_dir, dtype=dtype)
    combo = _combo("rmsnorm", dtype)
    checked = 0
    for record in combo["case_records"]:
        if record["case"] == "ieee":
            # IEEE injection belongs to the old harness; the sampler itself is the base normal
            # stream.  The output comparison above still exercises the archived special values.
            continue
        raw_inputs, _ = _archive(record)
        generated = contract.reference.generate(contract.instance, record["case"], record["seed"])
        for name, (piece, _entry) in raw_inputs.items():
            assert to_raw(generated[name], dtype) == piece, (dtype, record["case"], record["seed"], name)
        checked += 1
    assert checked == 15
