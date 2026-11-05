"""Typed failures of the restricted C99 / Wasm Host Program plane (V3).

Two failure classes must never be confused:

* :class:`HostProgramError` — the *candidate* (its C source, its spans, its return code or its
  imports) is at fault.  A host run that fails here yields no workspace span and no resolved
  scalar: the caller must treat the whole launch as a candidate failure, never commit a partial
  write.
* :class:`HostCapabilityUnavailable` — this machine cannot build or run the Host Program at all
  (no ``clang``, no ``wasm-ld``, no Wasm engine).  That is an infrastructure capability gap, not a
  candidate defect, and the JSON linear ``host/tiling.json`` path keeps working.

Both codes are registered in ``docs/失败码.md`` and locked by
``tests/shared/test_failure_codes.py``.
"""

from __future__ import annotations


class HostProgramError(ValueError):
    """@brief 候选 Host Program 本身不合法（源码/编译/返回码/区间/import）。

    一个失败的 host run 绝不返回部分 span：调用方只能把它当作候选失败处理。
    """

    code = "host_program_invalid"


class HostCapabilityUnavailable(RuntimeError):
    """@brief 本机缺 clang / wasm-ld / Wasm engine，Host Program 无法构建或执行。

    这不是候选代码错误；``host/tiling.json`` 的 linear 路径不受影响。``scripts/doctor.py``
    用同一个判定给出可执行的下一步。
    """

    code = "backend_capability_unavailable"


__all__ = ["HostCapabilityUnavailable", "HostProgramError"]
