"""CPU checks for omitted scratch declarations and partial-word coverage probes.

No device is opened.  The paint helpers stay on signaling-NaN words so a legal
quiet-NaN store is still coverage, while bytes that are not a full word are
probed too.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tcenv.backends.metal.launch import LaunchSpec, ScratchDecl
from tcenv.backends.metal.v2_runtime import (
    _paint_scratch,
    _poison_word,
    _stale_words,
    workspace_size_problems,
)


def _quiet(word: bytes) -> bytes:
    """Signaling NaN with the quiet bit set: a legal arithmetic store, not the probe."""
    return (int.from_bytes(word, "little") | 0x00400000).to_bytes(4, "little")


def _geometry(*scratch):
    return SimpleNamespace(workspace_uses=(), scratch=tuple(scratch))


def _abi():
    return SimpleNamespace(tensors=())


def _decl(name: str, nbytes: int) -> ScratchDecl:
    return ScratchDecl(name, "fp32", ("4",), "uninitialized_poison", nbytes)


def test_launch_omitting_scratch_is_refused_when_geometry_has_it():
    """@brief 几何有 scratch 而 launch 完全不列时必须拒绝。"""
    geometry = _geometry(SimpleNamespace(name="uwin", nbytes=16))
    omitted = LaunchSpec("safe", (), (), ())
    for spec in (omitted, None):
        problems = workspace_size_problems(_abi(), geometry, spec)
        assert any("resolved geometry uses scratch 'uwin'" in item and "does not declare" in item for item in problems)


def test_extra_scratch_and_nbytes_mismatch_are_refused():
    """@brief 多出来的 scratch、字节数对不上，都拒绝。"""
    geometry = _geometry(SimpleNamespace(name="uwin", nbytes=16))
    extra = LaunchSpec("safe", (), (), (_decl("uwin", 16), _decl("other", 16)))
    extra_problems = workspace_size_problems(_abi(), geometry, extra)
    assert any("launch declares scratch 'other'" in item for item in extra_problems)

    mismatch = LaunchSpec("safe", (), (), (_decl("uwin", 8),))
    mismatch_problems = workspace_size_problems(_abi(), geometry, mismatch)
    assert any("launch declares 8 bytes" in item and "16" in item for item in mismatch_problems)


def test_an_exact_scratch_declaration_matches():
    """@brief 名字和字节数都一致时没有 scratch 问题。"""
    geometry = _geometry(SimpleNamespace(name="uwin", nbytes=16))
    spec = LaunchSpec("safe", (), (), (_decl("uwin", 16),))
    assert workspace_size_problems(_abi(), geometry, spec) == []


def test_empty_scratch_keeps_the_old_empty_problem_list_and_digest():
    """@brief 两边都没有 scratch 时问题列表仍为空，launch 摘要也不长出 scratch 行。"""
    empty = _geometry()
    spec = LaunchSpec("safe", (), (), ())
    assert workspace_size_problems(_abi(), empty, spec) == []
    assert workspace_size_problems(_abi(), empty, None) == []
    legacy = SimpleNamespace(workspace_uses=())
    assert workspace_size_problems(_abi(), legacy, None) == []
    assert "scratch" not in spec.digest()


@pytest.mark.parametrize("length", [0, 1, 3, 4, 5, 7, 8])
def test_paint_covers_the_declared_length_and_a_legal_word_is_not_stale(length: int):
    """@brief 0/1/3/4/5/7/8 字节都涂满；完整字被静音或改写后不再算未覆盖。"""
    raw = bytes([0x11]) * length
    painted = _paint_scratch(raw, salt=2, keep_prefix=0)
    assert len(painted) == length
    if length == 0:
        assert _stale_words(painted, painted, start=0) == []
        return
    assert painted != raw
    assert _stale_words(painted, painted, start=0)
    covered = bytearray(painted)
    offset = 0
    while offset + 4 <= length:
        assert painted[offset : offset + 4] == _poison_word(offset // 4, 2)
        covered[offset : offset + 4] = _quiet(painted[offset : offset + 4])
        offset += 4
    while offset < length:
        covered[offset] ^= 0xFF
        offset += 1
    assert _stale_words(painted, bytes(covered), start=0) == []


def test_unaligned_keep_prefix_is_probed_without_touching_the_kept_byte():
    """@brief keep_prefix 不在 4 字节边界上时，中间那几字节也要被涂到。"""
    raw = bytes([0x11]) * 8
    painted = _paint_scratch(raw, salt=3, keep_prefix=1)
    assert painted[0] == raw[0]
    assert painted[1:4] != raw[1:4]
    assert painted[4:8] == _poison_word(1, 3)
    # A legal store of the aligned word leaves the unaligned prefix stale.
    partial = bytearray(painted)
    partial[4:8] = _quiet(painted[4:8])
    stale = _stale_words(painted, bytes(partial), start=1)
    assert 4 not in stale
    assert stale == [1, 2, 3]
    partial[1:4] = b"\x22\x22\x22"
    assert _stale_words(painted, bytes(partial), start=1) == []


def test_a_short_tail_left_equal_to_the_probe_is_stale_and_a_covered_tail_is_not():
    """@brief 5 字节里只改完整字、留下尾字节，算未覆盖；尾字节也被改掉才算覆盖。"""
    painted = _paint_scratch(b"\x00" * 5, salt=1, keep_prefix=0)
    assert painted[:4] == _poison_word(0, 1)
    retained = bytearray(painted)
    retained[:4] = _quiet(painted[:4])
    assert _stale_words(painted, bytes(retained), start=0) == [4]
    retained[4] ^= 0xFF
    assert _stale_words(painted, bytes(retained), start=0) == []
