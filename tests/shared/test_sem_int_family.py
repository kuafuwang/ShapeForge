"""R2-4: the dynamic integer family ``sem_int_<seed>`` (domain-level acceptance).

The family is *dynamic*: its names are ``sem_int_<seed>`` for a registered seed domain, so
there is no finite static list to compare.  This file establishes the acceptance domain and
compares it against the frozen R0 generator (``tasks.semantic::IntProgram``) as a read-only
oracle:

* registered dtype aliases ``{fp16, bf16, i32} -> i32`` and the registered seed domain
  (non-negative, at most 18 digits) — both are deliberate, documented narrowing vs the
  frozen implementation, which ignored the requested dtype and accepted any Python int;
* for a fixed corpus *and* a deterministic property corpus, the materialised OpSpec document
  (statements/constants/final), the derived Contract/describe/header/ABI/assemble, the
  rendered prompt, the known-good source, the case domain, tolerances, work accounting and
  ``meta`` agree with the frozen task, and inputs/reference bytes are identical;
* the program document is the authority: an edited constant changes the digest and the
  rendered code, and an invalid document is refused by the OpSpec validator.

CPU only.
"""

from __future__ import annotations

import copy
import dataclasses

import pytest
import torch

from tests.support.frozen_tasks import (
    SEEDS,
    FrozenSnapshot,
    assert_tensor_equal,
    extract_snapshot,
    fields_of,
    frozen_tasks,
    logical_cases,
    materialised_scale,
    old_build,
)
from tasks.sem_int import document as sem_int_document
from tasks.sem_int import program as sem_int_program
from tasks.sem_int.document import DTYPE_ALIASES, SEED_DIGITS_MAX, parse_seed

#: Fixed corpus: boundaries plus the shapes used by the experiments' semantic runs.
CORPUS = (0, 1, 2, 7, 42, 123, 1234, 99991, 424242, 10**17)
#: Deterministic property corpus over the low seeds (the generator is deterministic in the seed).
PROPERTY = tuple(range(0, 16))


@pytest.fixture(scope="session")
def frozen(tmp_path_factory) -> FrozenSnapshot:
    """@brief 导出 R0 登记提交的 ``tasks/``（共享 helper）。

    @return :class:`tests.support.frozen_tasks.FrozenSnapshot`。
    """
    return extract_snapshot(tmp_path_factory.mktemp("r0_sem_int"))


def _document(seed: int, size: str = "small") -> dict:
    """@brief 物化一份程序文档（不经 OpSpec 校验，供负例使用）。

    @param seed 种子。
    @param size 尺寸档位。
    @return OpSpec 文档。
    """
    return sem_int_document.opspec_document(seed, size)


def test_the_registered_dtype_aliases_are_explicit_and_narrowed(frozen: FrozenSnapshot):
    """@brief 登记别名 ``{fp16,bf16,i32}->i32``；其他 dtype 拒绝（有意收窄）。"""
    assert DTYPE_ALIASES == {"fp16": "i32", "bf16": "i32", "i32": "i32"}
    # B4 之后装配走固定 v2 目录（其 opspec 声明 dtypes=["i32"]），别名归一由迁移的逐字节证据
    # 覆盖；这里钉住的是**登记域本身**：哪些拼写被接受、映射到哪个 dtype。
    assert set(DTYPE_ALIASES.values()) == {"i32"}
    for bad in ("fp32", "f16", "int8", "u32", "i16"):
        assert bad not in DTYPE_ALIASES
    # The frozen task ignored the request and always materialised i32; accepting ``fp32`` was
    # an accident of that path, not a declared alias.  The narrowing is registered in R2.md.
    with frozen_tasks(frozen.root):
        legacy = old_build("sem_int_3", "fp32", "small")
    assert legacy.dtype == "i32"


def test_the_seed_domain_is_registered_and_bounded(frozen: FrozenSnapshot):
    """@brief 种子域＝非负、至多 18 位十进制整数；域外拒绝（含超长拼写）。"""
    assert parse_seed("sem_int_0") == 0
    limit = 10**SEED_DIGITS_MAX
    assert parse_seed(f"sem_int_{limit - 1}") == limit - 1
    # A legal alias spelling, and the 18-digit boundary with leading zeros: both must stay valid.
    assert parse_seed("sem_int_007") == 7
    assert parse_seed("sem_int_" + "0" * 17 + "7") == 7
    overflow_spellings = (
        # 19 digits, all but the last zero: the *value* is 1, the *spelling* is out of domain.
        "sem_int_" + "0" * 18 + "1",
        "sem_int_" + "0" * 40,
        f"sem_int_{limit}0",
    )
    # ASCII only: ``str.isdigit()`` is true for Unicode decimal digits, so a spelling like
    # ``sem_int_١`` would otherwise parse to 1 and address a task the domain never declared.
    unicode_digits = (
        "sem_int_\u0661",       # ARABIC-INDIC ONE
        "sem_int_\u06f1",       # EXTENDED ARABIC-INDIC ONE
        "sem_int_\uff11",       # FULLWIDTH DIGIT ONE
        "sem_int_1\u0661",      # mixed: an ASCII digit followed by a Unicode one
        "sem_int_\u00b2",       # superscript two: isdigit() is true, isdecimal() is not
    )
    for bad in (
        "sem_int_-1",
        "sem_int_1e3",
        "sem_int_",
        "sem_int_x",
        "sem_int_ 1",
        f"sem_int_{limit}",
        *overflow_spellings,
        *unicode_digits,
    ):
        with pytest.raises(ValueError):
            parse_seed(bad)
    # The domain is bounded by spelling length and by the ASCII alphabet, not only by value:
    # prove the value check alone would have accepted the overflow spellings.
    assert int("0" * 18 + "1") == 1 < limit
    assert "\u0661".isdigit() and int("\u0661") == 1
    # Frozen accepted negative seeds (random.Random accepts any int); that widening is gone.
    with frozen_tasks(frozen.root):
        assert old_build("sem_int_-5", "fp16", "small").dtype == "i32"


def test_the_materialised_document_matches_the_frozen_program(frozen: FrozenSnapshot):
    """@brief 域级对照：文档语句/常量/final 与冻结 ``IntProgram.generate`` 逐字段相同。"""
    for seed in CORPUS + PROPERTY:
        with frozen_tasks(frozen.root):
            import importlib

            semantic = importlib.import_module("tasks.semantic")
            expected = semantic.IntProgram.generate(seed)
        got = sem_int_program.statements_of(_document(seed)["program"])
        # ``Stmt`` is a different class on each side (frozen vs live module), so compare the
        # closed field values rather than class identity.
        assert [dataclasses.asdict(entry) for entry in got] == [
            dataclasses.asdict(entry) for entry in expected.stmts
        ], f"seed {seed}: statements drifted"
        assert _document(seed)["program"]["final"] == expected.final, f"seed {seed}: final drifted"
        # The program document is closed: launch facts stay in the profile, not here.
        assert set(_document(seed)["program"]) == {"kind", "block", "rules", "statements", "final"}


def test_the_program_document_is_the_authority_not_the_seed():
    """@brief 文档的一条语句/一条规则都是权威：改了就变摘要，也变渲染与求值。"""
    import copy

    from tcenv.opspec.model import OpSpecError, opspec_from_document

    base = _document(5)
    assert opspec_from_document(base, source="generated:test").digest

    def view(program_document: dict) -> tuple[str, str, "torch.Tensor"]:
        source = sem_int_program.triton_source(program_document, 4096, num_warps=2, scalar_k=sem_int_program.SCALAR_K)
        known_good = sem_int_program.msl(
            program_document, threads=64, num_warps=2, per_thread=sem_int_program.block_of(program_document) // 64
        )
        x = torch.tensor([1, -2, 3, -4] * 1024, dtype=torch.int64)
        y = torch.tensor([5, 6, -7, 8] * 1024, dtype=torch.int64)
        return source, known_good, sem_int_program.evaluate(program_document, x, y, sem_int_program.SCALAR_K)

    def edited(mutate) -> dict:
        document = copy.deepcopy(base["program"])
        mutate(document)
        return document

    reference_source, reference_known_good, reference_out = view(base["program"])

    def first(program_document: dict, kind: str) -> dict:
        return next(entry for entry in program_document["statements"] if entry["kind"] == kind)

    # Statement data is consumed.
    def bump_statement(program_document: dict) -> None:
        first(program_document, "bin")["const"] = int(first(program_document, "bin")["const"]) ^ 1

    # Rule data is consumed: the hash shift changes the prompt, the MSL and the reference.
    def bump_hash(program_document: dict) -> None:
        program_document["rules"]["hash_shift"] = 14

    def bump_index(program_document: dict) -> None:
        program_document["rules"]["index_mask"] = 6

    for mutate, label in ((bump_statement, "statement"), (bump_hash, "hash_shift"), (bump_index, "index_mask")):
        program_document = edited(mutate)
        assert opspec_from_document({**base, "program": program_document}, source="generated:test").digest != (
            opspec_from_document(base, source="generated:test").digest
        ), label
        source, known_good, out = view(program_document)
        assert (source, known_good) != (reference_source, reference_known_good), label
        assert not torch.equal(out, reference_out), label

    # The block size is semantic (a reduction spans one block) and drives all three views.
    blocked = edited(lambda program_document: program_document.update({"block": 128}))
    assert opspec_from_document({**base, "program": blocked}, source="generated:test").digest != (
        opspec_from_document(base, source="generated:test").digest
    )
    assert sem_int_program.triton_source(blocked, 4096, num_warps=2, scalar_k=1) != sem_int_program.triton_source(
        base["program"], 4096, num_warps=2, scalar_k=1
    )
    assert sem_int_program.msl(blocked, threads=64, num_warps=2, per_thread=2) != sem_int_program.msl(
        base["program"], threads=64, num_warps=2, per_thread=4
    )

    # Closed grammar + kind-specific numeric domains: the validator, not the generator, is the
    # authority.  Each entry is accepted first, then mutated into one illegal document.
    def rejected(mutate) -> None:
        document = {
            **base,
            "program": {**base["program"], "statements": [dict(s) for s in base["program"]["statements"]]},
        }
        binary = next(entry for entry in document["program"]["statements"] if entry["kind"] == "bin")
        mod = next(entry for entry in document["program"]["statements"] if entry["kind"] == "mod")
        shift = next(entry for entry in document["program"]["statements"] if entry["kind"] == "shl")
        mutate(document["program"], binary, mod, shift)
        with pytest.raises(OpSpecError):
            opspec_from_document(document, source="generated:test")

    rejected(lambda program_document, binary, mod, shift: binary.update({"kind": "frobnicate"}))
    rejected(lambda program_document, binary, mod, shift: binary.update({"rounds": 2}))
    rejected(lambda program_document, binary, mod, shift: binary.update({"b": "nope"}))
    rejected(lambda program_document, binary, mod, shift: binary.update({"op": "/"}))
    rejected(lambda program_document, binary, mod, shift: binary.update({"c": "xv"}))
    rejected(lambda program_document, binary, mod, shift: binary.update({"dst": "xv"}))
    rejected(lambda program_document, binary, mod, shift: program_document.update({"final": ["nope"]}))
    rejected(lambda program_document, binary, mod, shift: mod.update({"const": 0}))
    rejected(lambda program_document, binary, mod, shift: mod.update({"const": -7}))
    rejected(lambda program_document, binary, mod, shift: shift.update({"const": -1}))
    rejected(lambda program_document, binary, mod, shift: shift.update({"const": 32}))
    rejected(lambda program_document, binary, mod, shift: program_document["rules"].update({"hash_shift": 32}))
    rejected(lambda program_document, binary, mod, shift: program_document["rules"].update({"shl_range": [0, 40]}))
    rejected(lambda program_document, binary, mod, shift: program_document["rules"].update({"wrap_kinds": ["bin"]}))
    rejected(lambda program_document, binary, mod, shift: program_document.update({"block": 0}))


def test_every_registered_kind_has_a_declared_accept_and_reject_boundary():
    """@brief 表驱动文法不变量：12 个 kind 全部覆盖，不依赖生成器恰好抽到哪些语句。

    每个 kind 先构造一份最小合法文档（必须被接受），再逐项违反文法（必须被拒绝）：
    未定义操作数、`dst` 重复/遮蔽保留名、reduction 结果当 block 用、标量当作 final、
    标量槽位放 block、必需/禁止字段、数值域边界。这是对**文法**的检查，不是种子抽样。
    """
    from tcenv.opspec.model import OpSpecError, opspec_from_document

    base = _document(0, "small")
    fill = {"kind": "bin", "dst": "tz", "a": "xv", "b": "yv", "op": "^"}
    legal = {
        "bin": {"kind": "bin", "dst": "t1", "a": "xv", "b": "yv", "op": "^"},
        "sbin": {"kind": "sbin", "dst": "t1", "a": "xv", "b": "K", "op": "+"},
        "shl": {"kind": "shl", "dst": "t1", "a": "xv", "const": 5},
        "shr": {"kind": "shr", "dst": "t1", "a": "xv", "const": 5},
        "mod": {"kind": "mod", "dst": "t1", "a": "xv", "const": 7},
        "where": {"kind": "where", "dst": "t1", "a": "xv", "b": "yv", "c": "xv", "d": "yv", "op": "<"},
        "neg": {"kind": "neg", "dst": "t1", "a": "xv"},
        "hash": {"kind": "hash", "dst": "t1", "a": "xv", "const": 3},
        "loop": {"kind": "loop", "dst": "t1", "a": "xv", "b": "yv", "const": 3, "rounds": 2},
        "sum": {"kind": "sum", "dst": "s1", "a": "xv"},
        "max": {"kind": "max", "dst": "s1", "a": "xv"},
        "min": {"kind": "min", "dst": "s1", "a": "xv"},
    }
    assert set(legal) == set(sem_int_program.ALL_KINDS)

    def document(statements, final=("tz",), block=256):
        return {**base, "program": {**base["program"], "block": block, "statements": statements, "final": list(final)}}

    def rejected(doc, label: str) -> None:
        with pytest.raises(OpSpecError):
            opspec_from_document(doc, source=f"grammar:{label}")

    for kind, statement in legal.items():
        statements = [copy.deepcopy(statement), copy.deepcopy(fill)]
        opspec_from_document(document(statements), source=f"grammar:{kind}")  # the legal shape is accepted

        # Operands must be defined before they are read.
        undefined = copy.deepcopy(statements)
        undefined[0]["a"] = "never_defined"
        rejected(document(undefined), f"{kind}:undefined")

        # A reduction result is a scalar: it cannot feed a block operand or be the final value.
        if kind in ("sum", "max", "min"):
            scalar_as_block = copy.deepcopy(statements)
            scalar_as_block[1]["a"] = "s1"
            rejected(document(scalar_as_block), f"{kind}:scalar-as-block")
            rejected(document(statements, final=("s1",)), f"{kind}:scalar-final")

        # Unknown/duplicate destinations are refused.
        duplicate = copy.deepcopy(statements)
        duplicate[1]["dst"] = "xv"
        rejected(document(duplicate), f"{kind}:duplicate-dst")

    # Kind-specific field and numeric boundaries.
    for kind, field, value in (
        ("bin", "op", "/"),
        ("bin", "const", 1 << 31),
        ("sbin", "b", "yv"),
        ("where", "op", "+"),
        ("shl", "const", 32),
        ("shl", "const", -1),
        ("shr", "const", 32),
        ("mod", "const", 0),
        ("mod", "const", -7),
        ("mod", "const", 1 << 31),
        ("hash", "const", 1 << 31),
        ("loop", "rounds", 0),
        ("loop", "rounds", 65),
    ):
        statements = [copy.deepcopy(legal[kind]), copy.deepcopy(fill)]
        statements[0][field] = value
        rejected(document(statements), f"{kind}:{field}={value}")

    # Missing fields, forbidden fields, duplicate statements and a bad block size.
    missing = [copy.deepcopy(legal["where"]), copy.deepcopy(fill)]
    missing[0].pop("d")
    rejected(document(missing), "where:missing-d")
    extra = [copy.deepcopy(legal["neg"]), copy.deepcopy(fill)]
    extra[0]["const"] = 3
    rejected(document(extra), "neg:extra-const")
    rejected(document([copy.deepcopy(fill), copy.deepcopy(fill)]), "duplicate-statement")
    rejected(document([copy.deepcopy(fill)], block=0), "block:0")
    # A document without any statement is refused too.
    rejected(document([]), "empty-program")


def _view(view: str, program_document: dict):
    """@brief 单份视图的产物（题面文本 / known-good 文本 / 参考输出）。

    单独取一份视图，才能证明「这一份视图」自己拒绝了越域规则：一次算三份会让最先抛出的那份
    冒充另外两份的结果。

    @param view ``triton``/``msl``/``reference``。
    @param program_document ``program`` 文档。
    @return 文本或 ``torch.Tensor``。
    """
    block = sem_int_program.block_of(program_document)
    threads = 64
    if view == "triton":
        return sem_int_program.triton_source(program_document, 4096, num_warps=2, scalar_k=12345)
    if view == "msl":
        return sem_int_program.msl(program_document, threads=threads, num_warps=2, per_thread=block // threads)
    return sem_int_program.evaluate(program_document, _x(), _y(), 12345)


def _view_outputs(program_document: dict) -> dict:
    """@brief 三份视图各自的产物。

    @param program_document ``program`` 文档。
    @return ``{"triton": str, "msl": str, "reference": torch.Tensor}``。
    """
    return {name: _view(name, program_document) for name in _VIEWS}


def _x() -> torch.Tensor:
    """@brief 覆盖 int32 溢出与符号边的输入 x。

    @return int64 张量。
    """
    return torch.tensor([(i * 7919) % (1 << 21) - (1 << 20) for i in range(4096)], dtype=torch.int64)


def _y() -> torch.Tensor:
    """@brief 覆盖 int32 溢出与符号边的输入 y。

    @return int64 张量。
    """
    return torch.tensor([(i * 104729) % (1 << 21) - (1 << 20) for i in range(4096)], dtype=torch.int64)


def _rule_document(statements: list, **rules) -> dict:
    """@brief 用给定语句与规则覆写构造一份可校验的程序文档。

    @param statements 语句表。
    @param rules 要覆写的规则键。
    @return ``program`` 文档。
    """
    base = _document(0, "small")
    program_document = copy.deepcopy(base["program"])
    program_document["statements"] = statements
    program_document["final"] = ["tz"]
    program_document["rules"].update(rules)
    return {**base, "program": program_document, "formula": sem_int_document.formula_for(program_document["rules"])}


#: ``mod`` with a divisor the small mask admits; the value is the final block, so the difference
#: in mask is observable instead of being erased by a later wrap or a saturated comparison.
_MOD_STATEMENTS = [{"kind": "mod", "dst": "tz", "a": "xv", "const": 7}]
#: ``index`` is observable through the final line only, so the statement list stays minimal.
_INDEX_STATEMENTS = [{"kind": "bin", "dst": "tz", "a": "xv", "b": "yv", "op": "^"}]


_VIEWS = ("triton", "msl", "reference")


def _view(view: str, program_document: dict):
    """@brief 单份视图的产物（题面文本 / known-good 文本 / 参考输出）。

    单独取一份视图，才能证明「这一份视图」自己拒绝了越域规则：一次算三份会让最先抛出的那份
    冒充另外两份的结果。

    @param view ``triton``/``msl``/``reference``。
    @param program_document ``program`` 文档。
    @return 文本或 ``torch.Tensor``。
    """
    block = sem_int_program.block_of(program_document)
    threads = 64
    if view == "triton":
        return sem_int_program.triton_source(program_document, 4096, num_warps=2, scalar_k=12345)
    if view == "msl":
        return sem_int_program.msl(program_document, threads=threads, num_warps=2, per_thread=block // threads)
    return sem_int_program.evaluate(program_document, _x(), _y(), 12345)


def _view_outputs(program_document: dict) -> dict:
    """@brief 三份视图各自的产物。

    @param program_document ``program`` 文档。
    @return ``{"triton": str, "msl": str, "reference": torch.Tensor}``。
    """
    return {name: _view(name, program_document) for name in _VIEWS}


def _x() -> torch.Tensor:
    """@brief 覆盖 int32 溢出与符号边的输入 x。

    @return int64 张量。
    """
    return torch.tensor([(i * 7919) % (1 << 21) - (1 << 20) for i in range(4096)], dtype=torch.int64)


def _y() -> torch.Tensor:
    """@brief 覆盖 int32 溢出与符号边的输入 y。

    @return int64 张量。
    """
    return torch.tensor([(i * 104729) % (1 << 21) - (1 << 20) for i in range(4096)], dtype=torch.int64)


def _rule_document(statements: list, **rules) -> dict:
    """@brief 用给定语句与规则覆写构造一份可校验的程序文档。

    @param statements 语句表。
    @param rules 要覆写的规则键。
    @return ``program`` 文档。
    """
    base = _document(0, "small")
    program_document = copy.deepcopy(base["program"])
    program_document["statements"] = statements
    program_document["final"] = ["tz"]
    program_document["rules"].update(rules)
    return {**base, "program": program_document, "formula": sem_int_document.formula_for(program_document["rules"])}


#: ``mod`` with a divisor the small mask admits; the value is the final block, so the difference
#: in mask is observable instead of being erased by a later wrap or a saturated comparison.
_MOD_STATEMENTS = [{"kind": "mod", "dst": "tz", "a": "xv", "const": 7}]
#: ``index`` is observable through the final line only, so the statement list stays minimal.
_INDEX_STATEMENTS = [{"kind": "bin", "dst": "tz", "a": "xv", "b": "yv", "op": "^"}]


def test_every_view_reads_the_declared_mod_mask():
    """@brief ``mod_mask`` 由每份视图读取：改掩码则题面/known-good/参考三者都变。"""
    from tcenv.opspec.model import opspec_from_document

    small = _rule_document(copy.deepcopy(_MOD_STATEMENTS), mod_mask=7)
    full = _rule_document(copy.deepcopy(_MOD_STATEMENTS), mod_mask=0x7FFFFFFF)
    opspec_from_document(small, source="rules:mod_mask")
    opspec_from_document(full, source="rules:mod_mask")

    small_out, full_out = _view_outputs(small["program"]), _view_outputs(full["program"])
    assert "& 0x7)" in small_out["triton"] and "& 0x7FFFFFFF)" in full_out["triton"]
    assert "0x7FFFFFFF" not in small_out["triton"]
    assert "& 0x7u" in small_out["msl"] and "& 0x7FFFFFFFu" in full_out["msl"]
    assert "0x7FFFFFFF" not in small_out["msl"]
    assert not torch.equal(small_out["reference"], full_out["reference"])


def test_every_view_reads_the_declared_index_mask():
    """@brief ``index_mask`` 同样由每份视图读取，且题面公式来自同一份规则表。"""
    from tcenv.opspec.model import opspec_from_document

    seven = _rule_document(copy.deepcopy(_INDEX_STATEMENTS), index_mask=7)
    six = _rule_document(copy.deepcopy(_INDEX_STATEMENTS), index_mask=6)
    for document in (seven, six):
        opspec_from_document(document, source="rules:index_mask")

    seven_out, six_out = _view_outputs(seven["program"]), _view_outputs(six["program"])
    assert "(idx & 7)" in seven_out["triton"] and "(idx & 6)" in six_out["triton"]
    assert "(idx[e] & 7u)" in seven_out["msl"] and "(idx[e] & 6u)" in six_out["msl"]
    assert not torch.equal(seven_out["reference"], six_out["reference"])
    # One source: the formula is rendered from the same rules block, never from a constant.
    assert "& 7)" in seven["formula"] and "& 6)" in six["formula"]
    assert sem_int_document.formula_for(seven["program"]["rules"]) == seven["formula"]


def test_every_view_refuses_rules_it_cannot_implement(monkeypatch):
    """@brief 越域规则由**每份视图**拒绝；把任一 kind 搬出登记分区，三份视图都拒绝。

    这是 §11.2 的负例：一个视图静默忽略声明值是缺陷，因此逐视图断言，而不是只比较三份输出的
    元组（那样一份陈旧视图会被另两份掩盖）。分区被搬动时**参考视图也必须拒绝**：`loop` 每一轮
    的乘积都在内部回绕（登记算术），int64 求值若照常执行就会保留登记行为、无声忽略声明；
    既然三份视图都无法表达任意分区，就不存在“参考实现支持任意分区”的说法。
    """
    from tcenv.opspec.model import opspec_from_document

    base = _rule_document(copy.deepcopy(_MOD_STATEMENTS), mod_mask=7)
    opspec_from_document(base, source="rules:baseline")

    def outputs_or_error(program_document: dict) -> dict:
        got = {}
        for view in _VIEWS:
            try:
                got[view] = _view(view, program_document)
            except ValueError as exc:
                got[view] = exc
        return got

    def refused(document: dict) -> None:
        results = outputs_or_error(document["program"])
        for view in _VIEWS:
            assert isinstance(results[view], ValueError), f"{view} accepted an out-of-domain rule"

    # Out-of-domain numeric rules: every view refuses rather than rendering the old value.
    for rules in (
        {"mod_mask": -1},
        {"mod_mask": 1 << 31},
        {"hash_shift": 32},
        {"loop_shift": -1},
        {"index_mask": -2},
        {"shl_range": [0, 40]},
        {"version": sem_int_program.RULE_VERSION + 1},
        {"arithmetic": "float64"},
    ):
        refused(_rule_document(copy.deepcopy(_MOD_STATEMENTS), **rules))

    # A partition that does not cover every kind is refused too.
    partial = _rule_document(copy.deepcopy(_MOD_STATEMENTS))
    partial["program"]["rules"]["wrap_kinds"] = ["bin"]
    refused(partial)

    # The registered partition is what every view implements, so moving a kind is a change none of
    # them can express: each refuses it (the reference included — see the docstring).
    def moved(kind: str, statements: list) -> dict:
        document = _rule_document(copy.deepcopy(statements))
        rules = document["program"]["rules"]
        rules["wrap_kinds"] = [name for name in rules["wrap_kinds"] if name != kind]
        rules["plain_kinds"] = rules["plain_kinds"] + [kind]
        opspec_from_document(document, source=f"rules:{kind}-plain")  # the OpSpec gate accepts it
        return document

    # ``loop`` is the case that proves the reference cannot honour an arbitrary partition: its
    # per-round product wraps inside the loop body, so moving it to ``plain`` must be *refused*
    # rather than evaluated as if the declaration had not changed.
    statements_by_kind = {
        "loop": [
            {"kind": "loop", "dst": "t1", "a": "xv", "b": "yv", "const": 3, "rounds": 2},
            {"kind": "where", "dst": "tz", "a": "t1", "b": "yv", "c": "xv", "d": "yv", "op": ">"},
        ],
        "sbin": [
            {"kind": "sbin", "dst": "t1", "a": "xv", "b": "K", "op": "*"},
            {"kind": "where", "dst": "tz", "a": "t1", "b": "yv", "c": "xv", "d": "yv", "op": ">"},
        ],
        "shl": [
            {"kind": "shl", "dst": "t1", "a": "xv", "const": 21},
            {"kind": "where", "dst": "tz", "a": "t1", "b": "yv", "c": "xv", "d": "yv", "op": ">"},
        ],
    }
    for kind in ("loop", "sbin", "shl"):
        statements = statements_by_kind[kind]
        # The registered partition still builds and evaluates (positive control per kind).
        registered = _rule_document(copy.deepcopy(statements))
        opspec_from_document(registered, source=f"rules:{kind}-registered")
        assert not isinstance(_view_outputs(registered["program"])["reference"], ValueError)
        refused(moved(kind, statements))

    # The registered rules stay readable (a positive control for the gate above).
    for view in _VIEWS:
        got = _view_outputs(base["program"])[view]
        assert not isinstance(got, ValueError)


def test_the_formula_has_no_second_copy_of_the_index_mask(monkeypatch):
    """@brief 题面公式随规则表变化：模块里不存在写死 ``& 7`` 的第二份事实（§11.2-4）。"""
    from tcenv.opspec.model import opspec_from_document

    base = _document(3, "small")
    assert sem_int_document.formula_for(base["program"]["rules"]).count("& 7)") == 1
    assert "& 6)" in sem_int_document.formula_for({**base["program"]["rules"], "index_mask": 6})
    # No module-level spelling of the mask survives: the whole-module surface is checked, so a
    # re-introduced constant would be caught here rather than only in one rendered document.
    assert not hasattr(sem_int_document, "FORMULA")

    # Change the rule *table itself* and the generated document's formula follows.
    original = sem_int_program.rules_document

    def widened() -> dict:
        table = original()
        table["index_mask"] = 6
        return table

    monkeypatch.setattr(sem_int_program, "rules_document", widened)
    generated = sem_int_document.opspec_document(3, "small")
    assert generated["program"]["rules"]["index_mask"] == 6
    assert generated["formula"] == sem_int_document.formula_for(generated["program"]["rules"])
    assert "& 6)" in generated["formula"] and "& 7)" not in generated["formula"]
    # The rendered views read the same 6, so prose and code cannot drift.
    rendered = sem_int_program.triton_source(generated["program"], 4096, num_warps=2, scalar_k=1)
    assert "(idx & 6)" in rendered
    opspec_from_document(generated, source="rules:formula")
