"""Full-domain input for a unary fp16 elementwise kernel, and the evidence record that describes it.

The sweep is one correctness case. It is not a formal proof, and it is not part of ``task.cases``: search and
confirmation keep the six sampled cases. A passing record applies only to the launch contract, the toolchain and
the source hash written into it.
"""

from __future__ import annotations

import hashlib

import torch

from ...distributions import generator

DOMAIN_CASE = "domain"
PATTERN_COUNT = 1 << 16

NOT_COVERED = (
    "other sizes and launch contracts",
    "other toolchains or math modes",
    "tail paths (n is always a multiple of BLOCK_SIZE)",
    "every value at every lane",
)


def domain_input(n: int, seed: int) -> torch.Tensor:
    """@brief 生成包含全部 65536 个 fp16 位模式的输入张量。

    @param n 张量长度，必须是 65536 的正整数倍。
    @param seed 置换用的随机种子。
    @return 打乱后的 fp16 张量。
    @raises ValueError 当 ``n`` 不是 65536 的正整数倍。

    Every fp16 bit pattern, repeated to length ``n`` and placed at a random lane.

    ``n`` must be a positive multiple of 65536. One copy (the small size) is a permutation of the patterns.
    More copies (paper is 256) repeat each pattern that many times, still shuffled, so a value is not pinned to one lane.
    """
    if n <= 0 or n % PATTERN_COUNT:
        raise ValueError(f"domain sweep length must be a positive multiple of {PATTERN_COUNT}, got {n}")
    bits = torch.arange(-32768, 32768, dtype=torch.int16)
    pattern = bits.view(torch.float16)
    values = pattern.repeat(n // PATTERN_COUNT)
    return values[torch.randperm(n, generator=generator(seed))]


def tensor_sha256(t: torch.Tensor) -> str:
    """@brief 计算张量原始字节的 SHA-256。

    @param t 任意 torch 张量。
    @return 十六进制摘要。
    """
    raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def pattern_counts(t: torch.Tensor) -> torch.Tensor:
    """@brief 统计每个 fp16 位模式在张量里出现的次数。

    @param t 输入张量。
    @return 长度为 65536 的计数张量。

    How many times each of the 65536 fp16 patterns appears.
    """
    codes = t.detach().contiguous().view(torch.int16).to(torch.int32) + 32768
    return torch.bincount(codes, minlength=PATTERN_COUNT)


def sampling_coverage() -> dict:
    """@brief 生成“采样覆盖”记录，用于未做全域扫描的评测。

    @return 覆盖度字典。
    """
    return {"kind": "sampling", "scope": ""}


def domain_coverage(
    *,
    dtype: str,
    contract_sha256: str,
    source_sha256: str,
    task_fingerprint: str,
    toolchain: dict,
    layout: dict,
    input_sha256: str,
) -> dict:
    """@brief 生成全域扫描通过的证据记录。

    @param dtype 输入数据类型。
    @param contract_sha256 启动契约描述的摘要。
    @param source_sha256 候选源码的摘要。
    @param task_fingerprint 任务指纹。
    @param toolchain 工具链信息（设备、macOS、MSL 与 math 模式）。
    @param layout 扫描布局（种子、每个图案的份数、位置说明）。
    @param input_sha256 扫描输入张量的摘要。
    @return 覆盖度证据字典。
    """
    copies = layout.get("copies_per_pattern")
    scope = (
        f"All 65536 fp16 bit patterns of the single input, under this launch contract, on {toolchain.get('device') or 'this device'} "
        f"({toolchain.get('macos') or 'unknown macOS'}, MSL {toolchain.get('msl') or '?'}, math {toolchain.get('math_mode') or '?'}). "
        f"Each pattern appears {copies} time(s) at a random position. Not a proof about other contracts, toolchains, or tail paths."
    )
    return {
        "kind": "domain_sweep",
        "passed": True,
        "dtype": dtype,
        "domain": "all 65536 bit patterns of the single input",
        "contract_sha256": contract_sha256,
        "source_sha256": source_sha256,
        "task_fingerprint": task_fingerprint,
        "toolchain": toolchain,
        "layout": layout,
        "compare": {
            "nan": "any NaN equals any NaN",
            "zero_sign": "not checked",
            "tolerance": "rtol*|ref| + subnormal floor, no RMS term",
        },
        "input_sha256": input_sha256,
        "not_covered": list(NOT_COVERED),
        "scope": scope,
    }


def coverage_for_report(task, results, kernel_source: str, toolchain: dict) -> dict:
    """@brief 根据评测结果选择 domain_sweep 还是 sampling 记录。

    @param task 任务对象。
    @param results 各算例的结果列表。
    @param kernel_source 候选核源码。
    @param toolchain 工具链信息。
    @return 覆盖度字典。

    ``domain_sweep`` only when the task declared the sweep and every domain result passed. Otherwise sampling.
    """
    domain = [r for r in results if r.case == DOMAIN_CASE]
    if not domain or not (task.meta or {}).get("domain_case") or not all(r.ok for r in domain):
        return sampling_coverage()
    import hashlib

    from ...task import task_fingerprint

    n = task.contract.inputs[0].shape[0]
    first = domain[0]
    return domain_coverage(
        dtype=task.dtype,
        contract_sha256=hashlib.sha256(task.contract.describe().encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
        task_fingerprint=task_fingerprint(task, "metal"),
        toolchain=toolchain,
        layout={"seed": first.seed, "copies_per_pattern": n // PATTERN_COUNT, "positions": "random permutation"},
        input_sha256=first.input_sha256,
    )
