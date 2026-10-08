"""Full-domain input for a unary fp16 elementwise kernel, and the evidence record that describes it.

The sweep is one correctness case. It is not a formal proof, and it is not part of ``task.cases``: search and
confirmation keep the six sampled cases. A passing record applies only to the launch contract, the toolchain and
the source hash written into it.
"""

from __future__ import annotations

import hashlib

import torch

from .distributions import generator

DOMAIN_CASE = "domain"
PATTERN_COUNT = 1 << 16

NOT_COVERED = (
    "other sizes and launch contracts",
    "other toolchains or math modes",
    "tail paths (n is always a multiple of BLOCK_SIZE)",
    "every value at every lane",
)


def domain_input(n: int, seed: int) -> torch.Tensor:
    """Every fp16 bit pattern, repeated to length ``n`` and placed at a random lane.

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
    raw = t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def pattern_counts(t: torch.Tensor) -> torch.Tensor:
    """How many times each of the 65536 fp16 patterns appears."""
    codes = t.detach().contiguous().view(torch.int16).to(torch.int32) + 32768
    return torch.bincount(codes, minlength=PATTERN_COUNT)


def sampling_coverage() -> dict:
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
    """``domain_sweep`` only when the task declared the sweep and every domain result passed. Otherwise sampling."""
    domain = [r for r in results if r.case == DOMAIN_CASE]
    if not domain or not (task.meta or {}).get("domain_case") or not all(r.ok for r in domain):
        return sampling_coverage()
    import hashlib

    from .task import task_fingerprint

    n = task.contract.inputs[0].shape[0]
    first = domain[0]
    return domain_coverage(
        dtype=task.dtype,
        contract_sha256=hashlib.sha256(task.contract.describe().encode("utf-8")).hexdigest(),
        source_sha256=hashlib.sha256(kernel_source.encode("utf-8")).hexdigest(),
        task_fingerprint=task_fingerprint(task),
        toolchain=toolchain,
        layout={"seed": first.seed, "copies_per_pattern": n // PATTERN_COUNT, "positions": "random permutation"},
        input_sha256=first.input_sha256,
    )
