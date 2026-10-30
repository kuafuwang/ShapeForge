"""Generate the R0 canonical-identity vectors (no production code is imported).

Every vector is a *canonical document* per plan §3.1: object keys are non-empty
ASCII, every non-integer number has already been normalised to its shortest
round-trip decimal string, and the document contains no float.  The digest of a
vector is ``SHA256(RFC8785(document))``.

Run from the repository root:

    uv run python docs/架构/构建执行与证据语义升级-R0/gen_vectors.py

The script deliberately uses only the standard library so that the vectors and
their digests are independent of ``tcenv`` (see ``verify.py``, which recomputes
them a second, independent way).
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE / "vectors" / "documents"


def canonical_bytes(document: dict) -> bytes:
    """RFC 8785 bytes for the identity subset (ASCII keys, no floats)."""
    return json.dumps(
        document, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest(document: dict) -> str:
    return hashlib.sha256(canonical_bytes(document)).hexdigest()


def load(name: str) -> dict:
    return json.loads((DOCS / f"{name}.json").read_text(encoding="utf-8"))


def write(name: str, document: dict) -> None:
    (DOCS / f"{name}.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


OPSPEC = load("opspec")
PROFILE = load("lowering_profile")
OPSPEC_DIGEST = digest(OPSPEC)
PROFILE_DIGEST = digest(PROFILE)

SEMANTIC_INSTANCE = {
    "opspec_digest": OPSPEC_DIGEST,
    "dtype": "fp16",
    "size": "small",
    "shapes": {
        "symbols": {"M": 64, "H": 512},
        "tensors": {"x": [64, 512], "w": [512], "y": [64, 512]},
    },
    "resolved_attrs": {"EPS": "1e-06"},
    "spec_params": {"M": "64", "H": "512"},
    "tolerances": {"rtol": "0.002", "atol_rel": "0.001", "exact": False},
    "case_domain": {
        "cases": ["normal", "wide_range", "large", "tiny", "outlier", "ieee"],
        "inputs": {"ieee": {"special_values": "inf, nan, denormal"}},
    },
    # Fixture values: the accounting rules themselves are R1's decision (§4.2).
    "work_accounting": {"flops": "131072.0", "min_bytes": "133120.0"},
}

SPECIALIZATION = {
    "opspec_digest": OPSPEC_DIGEST,
    "lowering_profile_digest": PROFILE_DIGEST,
    "spec_params": {"M": "64", "H": "512"},
    "backend_params": {},
    "contract": {
        "kernel_name": "rmsnorm_kernel",
        "grid": [64, 1, 1],
        "threads": [256, 1, 1],
        "gpu": "Apple M4 Max (Apple9 GPU family)",
        "args": [
            {"name": "x", "index": 0, "role": "input", "dtype": "fp16", "shape": [64, 512], "value": None, "doc": "input tensor"},
            {"name": "w", "index": 1, "role": "input", "dtype": "fp16", "shape": [512], "value": None, "doc": "scale vector"},
            {"name": "y", "index": 2, "role": "output", "dtype": "fp16", "shape": [64, 512], "value": None, "doc": "output tensor"},
        ],
        "constants": [
            {"name": "M", "msl_type": "uint", "value": 64},
            {"name": "H", "msl_type": "uint", "value": 512},
            {"name": "EPS", "msl_type": "float", "value": "1e-06"},
        ],
    },
}

TASK = {
    "fingerprint_kind": "task",
    "fingerprint_version": 2,
    "task": "rmsnorm",
    "dtype": "fp16",
    "size": "small",
    "backend_id": "metal",
    "semantic_instance": SEMANTIC_INSTANCE,
    "specialization": SPECIALIZATION,
    "triton_source": "import triton\nimport triton.language as tl\n\n# fixture: tasks/rmsnorm/kernel_triton.py rendered for fp16/M=64/H=512\n",
    "description": "Row-wise RMSNorm: y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w; the sum of squares is accumulated in fp32 and the result rounded once.",
    "reference_source": "# fixture: tasks/rmsnorm reference rendered for fp16\n",
}

BASELINE_IMPLEMENTATION = {
    "fingerprint_kind": "baseline_implementation",
    "fingerprint_version": 1,
    "registry_version": "task-baselines-1",
    "backend_id": "metal",
    "baselines": [
        {
            "name": "mlx",
            "framework": "mlx",
            "kind": "task",
            "implementation": {"implementation_id": "rmsnorm.mlx", "implementation_version": 1, "params": {"dtype": "fp16"}},
        },
        {
            "name": "torch_mps",
            "framework": "torch",
            "kind": "task",
            "implementation": {"implementation_id": "rmsnorm.torch", "implementation_version": 1, "params": {"dtype": "fp16"}},
        },
    ],
}

BASELINE_ENVIRONMENT = {
    "fingerprint_kind": "baseline_environment",
    "fingerprint_version": 1,
    "backend_id": "metal",
    "target_id": "apple-m4-max",
    "baselines": [
        {"name": "mlx", "framework": "mlx", "kind": "task", "available": True, "library": {"name": "mlx", "version": "0.29.3"}},
        {"name": "torch_mps", "framework": "torch", "kind": "task", "available": True, "library": {"name": "torch", "version": "2.9.0"}},
    ],
}

TASK_PACK = {
    "schema_version": 1,
    "files": [
        {"path": "tasks/rmsnorm/kernel_triton.py", "size": 1450, "sha256": "11" * 32},
        {"path": "tasks/rmsnorm/naive.metal", "size": 1420, "sha256": "22" * 32},
        {"path": "tasks/rmsnorm/opspec.json", "size": 2100, "sha256": "33" * 32},
        {"path": "tasks/rmsnorm/task.py", "size": 5300, "sha256": "44" * 32},
        {"path": "tasks/rmsnorm/track.json", "size": 900, "sha256": "55" * 32},
    ],
}

TASK_DIGEST = digest(TASK)
PACK_DIGEST = digest(TASK_PACK)
BASELINE_IMPL_DIGEST = digest(BASELINE_IMPLEMENTATION)
BASELINE_ENV_DIGEST = digest(BASELINE_ENVIRONMENT)

EVALUATION = {
    "fingerprint_kind": "evaluation",
    "fingerprint_version": 1,
    "task_fingerprint": TASK_DIGEST,
    "task_pack_digest": PACK_DIGEST,
    "gen_inputs": {"implementation_id": "rmsnorm.gen_inputs", "implementation_version": 1, "params": {"cases": "declared-domain"}},
    "reference": {"implementation_id": "rmsnorm.reference", "implementation_version": 1, "params": {}},
    "error_scale": {"implementation_id": "rmsnorm.error_scale", "implementation_version": 1, "params": {"dtype": "fp16"}},
    "baseline_implementation_digest": BASELINE_IMPL_DIGEST,
    "baseline_environment_digest": BASELINE_ENV_DIGEST,
}

write("semantic_instance", SEMANTIC_INSTANCE)
write("specialization", SPECIALIZATION)
write("task", TASK)
write("baseline_implementation", BASELINE_IMPLEMENTATION)
write("baseline_environment", BASELINE_ENVIRONMENT)
write("task_pack", TASK_PACK)
write("evaluation", EVALUATION)

DIGESTS = {
    "opspec": {"document": "vectors/documents/opspec.json", "digest": OPSPEC_DIGEST},
    "lowering_profile": {"document": "vectors/documents/lowering_profile.json", "digest": PROFILE_DIGEST},
    "task_fingerprint": {"document": "vectors/documents/task.json", "digest": TASK_DIGEST},
    "baseline_implementation": {"document": "vectors/documents/baseline_implementation.json", "digest": BASELINE_IMPL_DIGEST},
    "baseline_environment": {"document": "vectors/documents/baseline_environment.json", "digest": BASELINE_ENV_DIGEST},
    "task_pack_manifest": {"document": "vectors/documents/task_pack.json", "digest": PACK_DIGEST},
    "evaluation_fingerprint": {"document": "vectors/documents/evaluation.json", "digest": digest(EVALUATION)},
}
(HERE / "vectors" / "digests.json").write_text(
    json.dumps(DIGESTS, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
)
for name, item in DIGESTS.items():
    print(f"{name:24s} {item['digest']}")
