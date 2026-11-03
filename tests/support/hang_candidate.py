"""The opt-in *dangerous* GPU hang candidate, built for the preserved v1 ``lion`` ABI.

``tests/backends/metal/test_tcenv.py`` keeps one test behind ``TCENV_ALLOW_GPU_HANG_TESTS=1``:
it launches a non-terminating kernel to prove the supervisor kills it and the environment
recovers.  That test must not run by default, so the candidate is built and checked *without
executing it*: :func:`lion_hang_findings` inspects the source statically.

A6 retired the migrated ``relu`` assembly but kept this dangerous test.  Its body used to be a
relu-shaped ``x[j]``/``y[j]`` loop; ``lion``'s ABI is three inputs then two outputs
(``p``/``g``/``m``/``p_out``/``m_out``), so the retired body would have failed to compile --
the test would then see ``compile_error`` instead of the expected ``timeout`` and the recovery
assertion would be meaningless.  The body below writes **both** outputs.
"""

from __future__ import annotations

#: The canonical naive ``lion`` loop body (one copy; ``test_tcenv`` imports it).
LION_BODY = """if (j < N) {
            float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]);
            float c = BETA1 * mv + (1.0f - BETA1) * gv;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            p_out[j] = half(pv * (1.0f - LR * WD) - LR * u);
            m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv);
        }"""

#: Non-terminating body for the same ABI: both outputs are written *after* the spin, so a
#: supervisor that kills the launch leaves the recovery kernel as the only way to reach ``ok``.
HANG_BODY = """if (j < N) {
            float pv = float(p[j]), mv = float(m[j]);
            volatile uint spin = 0;
            while (spin < 0xFFFFFFFFu) { spin += (pv == pv) ? 0u : 1u; }
            p_out[j] = half(pv);
            m_out[j] = half(mv);
        }"""

#: Buffer slots of the preserved ``lion`` ABI, in declaration order.
LION_INPUTS = ("p", "g", "m")
LION_OUTPUTS = ("p_out", "m_out")
LION_SLOTS = LION_INPUTS + LION_OUTPUTS


def lion_hang_source(naive: str) -> str:
    """@brief 把 naive lion 的标准循环体换成死循环体（不执行）。

    @param naive ``get_task("lion", ...).naive_msl``。
    @return 死循环候选源码。
    @raises AssertionError naive 源码不是预期的 lion 形状（ABI 变了）。
    """
    assert LION_BODY in naive, "the lion naive body changed; port the hang candidate with it"
    return naive.replace(LION_BODY, HANG_BODY)


def lion_hang_findings(naive: str, source: str) -> list[str]:
    """@brief 静态检查死循环候选与 lion ABI 对齐（编译前，不启动 GPU）。

    @param naive ``get_task("lion", ...).naive_msl``。
    @param source :func:`lion_hang_source` 的结果。
    @return 问题列表；空表示候选形状正确。
    """
    findings: list[str] = []
    if HANG_BODY not in source:
        findings.append("the hang body is not in the candidate")
    for name in LION_SLOTS:
        if f"device " not in source or f"* {name} [[buffer(" not in source:
            findings.append(f"the ABI argument {name!r} is missing")
    for retired in ("x[j]", "y[j]", "x[", "y["):
        if retired in source:
            findings.append(f"the retired relu-style access {retired!r} is still present")
    for output in LION_OUTPUTS:
        if f"{output}[j] = half(" not in source:
            findings.append(f"the output {output!r} is never written")
    if "while (spin < 0xFFFFFFFFu)" not in source:
        findings.append("the spin loop is missing")
    # Every buffer slot referenced in the source must exist in the ABI.
    for slot in range(len(LION_SLOTS), len(LION_SLOTS) + 4):
        if f"[[buffer({slot})]]" in source:
            findings.append(f"buffer slot {slot} is outside the lion ABI")
    return findings
