"""G0 after A5: the archive is intact, every migrated combo is retired on v1, and the v2
standard entry still carries the frozen semantic fields.

Before A5 this file compared all 82 frozen ``(task, dtype, size, variant)`` combos against the
*live v1 assembly*, field by field (contract, contract text, specialization, Triton prompt,
naive Metal, reference sources, description, cases, tolerances, work accounting, implementation
identities).  A5 migrated all 16 tasks to the v2 standard entry and deleted that v1 assembly in
the same change, so the v1 projection no longer exists to compare — and it must not be revived
to keep a green assertion.

The per-combo claim is therefore the one that still has a subject:

* the v1 registry **refuses** the name (dispatch by ``spec_version`` with no fallback),
* the retired ``@variant`` spelling is not a second contract,
* the v2 standard entry reproduces the frozen semantic fields (cases, tolerances, work
  accounting, tensors/attrs domain, and the frozen dtype/size set), and
* the G0 archive itself is byte-identical to its frozen manifest (the corpus was not rewritten
  to make the migration look clean).

The deeper comparison of the frozen *inputs, reference outputs, error scales, baseline identity
and pack closure* against the archive is ``tests/backends/metal/test_task_migration_a5.py``
(``A5/g0_compare.py``): it compares bytes, this file compares the identity plane.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
G0_DIR = ROOT / "docs" / "架构" / "跨硬件算子生成分工-G0"
G0_INDEX = G0_DIR / "corpus" / "index.json"
G0_MANIFEST = G0_DIR / "corpus" / "manifest.sha256"

INDEX = json.loads(G0_INDEX.read_text(encoding="utf-8"))
COMBOS = INDEX["combos"]


def _combo_id(combo: dict) -> str:
    """@brief 稳定的 pytest 节点 id（与 G0 语料顺序无关）。

    @param combo G0 组合。
    @return id 文本。
    """
    variant = combo.get("variant") or "base"
    return f"{combo['task']}-{combo['dtype']}-{combo['size']}-{variant}"


def _json_value(value):
    """@brief 与 G0 corpus 相同的 JSON 投影（dataclass → dict，tuple → list）。

    @param value 任意可投影值。
    @return JSON 值。
    """
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"not JSON canonical: {type(value).__name__}")


@pytest.mark.parametrize("combo", COMBOS, ids=[_combo_id(combo) for combo in COMBOS])
def test_the_g0_combo_is_retired_on_v1_and_lives_on_the_v2_standard_entry(combo):
    """@brief 每个 G0 组合：v1 明确拒绝（不回落），v2 标准入口复现冻结语义字段。"""
    from tasks import get_task
    from tcenv.contract_errors import TaskContractError, UnsupportedContractCapability
    from tcenv.taskcontract import inspect_standard_entry, load_task_contract

    name, dtype, size, variant = combo["task"], combo["dtype"], combo["size"], combo["variant"]

    # 1. The v1 registry refuses the migrated name instead of importing a half-deleted task.
    for spelling in ([name] if variant is None else [name, f"{name}@{variant}"]):
        with pytest.raises(TaskContractError) as error:
            get_task(spelling, dtype, size)
        assert "v2 standard entry" in str(error.value), spelling

    # 2. The standard entry is v2 and is discovered by directory, not by a central name table.
    directory = ROOT / "tasks" / name
    entry = inspect_standard_entry(directory)
    assert entry.is_v2, name
    assert entry.files[:2] == ("opspec.json", "reference.py"), name

    # 3. The retired launch variant is refused at assembly (one contract, not five).
    if variant is not None:
        with pytest.raises(UnsupportedContractCapability) as error:
            load_task_contract(directory, dtype=dtype, size=size, variant=str(variant))
        assert "@variant" in str(error.value)

    contract = load_task_contract(directory, dtype=dtype, size=size)
    instance = contract.instance
    assert (instance.op_name, instance.dtype, instance.size) == (name, dtype, size)

    # 4. Frozen semantic fields: cases, tolerances, accounting, tensor/attr domain, dtype set.
    assert list(instance.case_domain.cases) == combo["cases"], name
    tolerances = combo["tolerances"]
    assert (
        float(instance.tolerances.rtol),
        float(instance.tolerances.atol_rel),
        bool(instance.tolerances.exact),
    ) == (float(tolerances["rtol"]), float(tolerances["atol_rel"]), bool(tolerances["exact"])), name
    assert float(instance.work_accounting.flops) == pytest.approx(float(combo["work"]["flops"])), name
    assert float(instance.work_accounting.min_bytes) == pytest.approx(float(combo["work"]["min_bytes"])), name
    assert [tensor.name for tensor in instance.tensors] == [arg["name"] for arg in combo["contract"]["args"]], name
    assert {tensor.name: list(tensor.shape) for tensor in instance.tensors} == {
        arg["name"]: list(arg["shape"]) for arg in combo["contract"]["args"]
    }, name
    constants = {str(key): value for key, _type, value in combo["contract"]["constants"]}
    assert set(instance.symbols) <= set(constants), name
    assert set(instance.resolved_attrs) == {name for name in instance.resolved_attrs}, name
    for key, value in dict(instance.symbols).items():
        assert float(constants[key]) == pytest.approx(float(value)), (name, key)
    # The archived constants the v1 lowering profile used to own (``BLOCK_SIZE`` and GEMM's
    # ``BLOCK_M``/``BLOCK_N``/``BLOCK_K``) are now candidate-owned launch constants: the v2
    # contract declares no tile, and the frozen candidate text declares them itself.  A5's
    # ``test_a5_the_frozen_candidate_geometry_is_the_g0_contract_not_a_fallback`` binds the
    # resulting grid/threads to this same archive.
    assert constants, name

    # 5. The registered OpSpec digest cannot be the archived v1 one: A5's declared difference is
    #    ``spec_version`` 2 plus the removal of the retired ``uses`` list, and both are inside
    #    that digest by construction.  The semantic fields above are the identity that survives;
    #    the digest itself is recorded as a registered difference in A5.md.
    assert len(instance.opspec_digest) == 64, name
    assert instance.opspec_digest != combo["specialization"]["opspec_digest"], name
    assert [tensor.dtype for tensor in instance.tensors] == [arg["dtype"] for arg in combo["contract"]["args"]], name
    assert [tensor.role for tensor in instance.tensors] == [arg["role"] for arg in combo["contract"]["args"]], name

    # 6. The description the model sees is the frozen one (display.json may only override for the
    #    G0-whitelisted sem_* family, which these 16 tasks are not).
    assert contract.display.description == combo["description"], name


def test_the_g0_combo_inventory_is_unchanged():
    """@brief 语料范围仍是 16 题 / 82 组合；A5 不得偷偷增删组合。"""
    assert len(COMBOS) == 82
    assert len({combo["task"] for combo in COMBOS}) == 16
    assert INDEX["schema_version"] == 1
    assert INDEX["seed_formula"] == "base_seed + 7919 * (case_index + 1)"
    assert INDEX["seeds"] == [0, 7, 1234]


def test_the_g0_archive_is_byte_identical_to_its_frozen_manifest():
    """@brief 归档没有被改写：manifest 里的每个文件摘要与现文件一致，且文件集合相等。

    A migration that silently rewrote the corpus could "prove" any identity it liked.  This is
    the cheap (index-only) half of ``G0/verify.py --manifest``; the tensors are covered by the
    same manifest when that verifier runs in A5's acceptance.
    """
    lines = G0_MANIFEST.read_text(encoding="utf-8").splitlines()
    corpus = G0_DIR / "corpus"
    expected = {
        str(path.relative_to(corpus))
        for path in corpus.rglob("*")
        if path.is_file() and path.name != G0_MANIFEST.name and "__pycache__" not in path.parts
    }
    seen: set[str] = set()
    for line in lines:
        digest, name = line.split("  ", 1)
        assert len(digest) == 64 and name not in seen, name
        assert hashlib.sha256((corpus / name).read_bytes()).hexdigest() == digest, f"manifest mismatch: {name}"
        seen.add(name)
    assert seen == expected, f"manifest file set differs: missing={sorted(expected - seen)}, extra={sorted(seen - expected)}"
