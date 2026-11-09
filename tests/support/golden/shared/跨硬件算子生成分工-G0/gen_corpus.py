#!/usr/bin/env python3
"""Generate or replay the G0 old-path corpus using only the current v1 task path.

No GPU, model, or new IR/renderer is used.  ``prepare_case`` is the single source of
inputs, reference outputs and error scales.  Small cases preserve exact tensor bytes;
paper/plan cases preserve tensor metadata/digests plus the reproducible case seed.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import subprocess
import zlib
from pathlib import Path

import torch

from tasks import get_task
from tcenv.backends.metal.testing import prepare_case

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CORPUS = HERE / "corpus"
INDEX = CORPUS / "index.json"
MANIFEST = CORPUS / "manifest.sha256"
SEEDS = (0, 7, 1234)
BASE_TASKS = (
    "sigmoid", "relu", "silu", "swiglu", "rmsnorm", "softmax", "reduction_sum",
    "gemm", "gemm_silu",
)
SEMANTIC_TASKS = ("sem_gemm_neg", "sem_relu_gelu_comment", "sem_gelu_named_relu")
GENERATED_TASKS = ("gen_relu", "gen_sigmoid", "gen_silu", "gen_swiglu")
GEMM_VARIANTS = (None, "128x64x32t256", "64x64x32t256", "32x32x16t128", "128x128x32t512")
MAGIC = b"TG0TENS1"


def head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True).stdout.strip()


def raw_bytes(tensor) -> bytes:
    return tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def json_value(value):
    if dataclasses.is_dataclass(value):
        return json_value(dataclasses.asdict(value))
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"not JSON canonical: {type(value).__name__}")


def inventory() -> list[dict]:
    result = []
    for task in BASE_TASKS + SEMANTIC_TASKS + GENERATED_TASKS:
        variants = GEMM_VARIANTS if task == "gemm" else (None,)
        sizes = ["paper", "small"] + (["plan"] if task == "reduction_sum" else [])
        for variant in variants:
            full_name = task + ("@" + variant if variant else "")
            for size in sizes:
                for dtype in ("fp16", "bf16"):
                    result.append({"task": task, "name": full_name, "dtype": dtype, "size": size, "variant": variant})
    return result


def track_document(task: str) -> dict:
    if task.startswith("gen_"):
        path = REPO / "tasks" / "generated" / task[4:] / "track.json"
    else:
        path = REPO / "tasks" / task / "track.json"
    return json.loads(path.read_text(encoding="utf-8"))


def tensor_archive(data) -> tuple[bytes, dict]:
    header = []
    payload = bytearray()
    for role, tensors in (("input", data.rounded), ("reference", data.ref), ("scale", data.scale)):
        for name in sorted(tensors):
            tensor = tensors[name]
            raw = data.raw[name] if role == "input" else raw_bytes(tensor)
            offset = len(payload)
            payload.extend(raw)
            header.append({
                "role": role,
                "name": name,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
                "offset": offset,
                "nbytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return MAGIC + len(encoded).to_bytes(4, "little") + encoded + bytes(payload), {"entries": header}


def case_record(task, case: str, index: int, base_seed: int, raw: bool, relative: str | None):
    seed = base_seed + 7919 * (index + 1)
    data = prepare_case(task, case, seed)
    archive, metadata = tensor_archive(data)
    packed = zlib.compress(archive, level=9) if raw else None
    record = {
        "case": case,
        "seed": seed,
        "input_sha256": data.input_sha256,
        "tensors": metadata,
        "archive": relative if raw else None,
        "archive_sha256": hashlib.sha256(packed).hexdigest() if raw else None,
    }
    return record, packed


def build_combo(spec: dict, source_head: str) -> tuple[dict, dict[str, bytes]]:
    task = get_task(spec["name"], spec["dtype"], spec["size"])
    track = track_document(spec["task"])
    implementations = json_value(task.evaluation_implementations)
    task_pack = implementations.get("task_pack")
    assembly_defects = [] if task_pack else ["task_pack_not_registered"]
    combo = {
        "task": spec["task"],
        "name": spec["name"],
        "dtype": spec["dtype"],
        "size": spec["size"],
        "variant": spec["variant"],
        "source_head": source_head,
        "contract": json_value(task.contract),
        "contract_text": {
            "describe": task.contract.describe(),
            "header": task.contract.render_header(),
            "abi": task.contract.abi_stub(),
        },
        "specialization": json_value(task.specialization),
        "triton_source": task.triton_source,
        "naive_msl": task.naive_msl,
        "reference_sources": json_value(task.reference_sources),
        "description": task.description,
        "cases": list(task.cases),
        "tolerances": {"rtol": task.rtol, "atol_rel": task.atol_rel, "exact": task.exact},
        "work": {"flops": float(task.flops), "min_bytes": float(task.min_bytes)},
        "meta": json_value(task.meta),
        "baselines": implementations["baselines"],
        "evaluation_implementations": implementations,
        "assembly_defects": assembly_defects,
        "template_literals": track.get("template_literals"),
        "pack_files": (task_pack or track)["pack_files"],
        "task_pack_digest": (task_pack or {}).get("digest"),
        "case_records": [],
    }
    archives: dict[str, bytes] = {}
    raw = spec["size"] == "small"
    stem = spec["name"].replace("@", "__")
    for case_index, case in enumerate(task.cases):
        for base_seed in SEEDS:
            relative = f"tensors/{stem}/{spec['dtype']}/{case}/{base_seed + 7919 * (case_index + 1)}.tg0z" if raw else None
            record, archive = case_record(task, case, case_index, base_seed, raw, relative)
            combo["case_records"].append(record)
            if archive is not None:
                archives[relative] = archive
    return combo, archives


def generate() -> tuple[dict, dict[str, bytes]]:
    source_head = head()
    combos = []
    archives: dict[str, bytes] = {}
    specs = inventory()
    for index, spec in enumerate(specs, 1):
        print(f"[{index}/{len(specs)}] {spec['name']} {spec['dtype']} {spec['size']}", flush=True)
        combo, produced = build_combo(spec, source_head)
        combos.append(combo)
        archives.update(produced)
    document = {
        "schema_version": 1,
        "generator": "current-v1-get_task+prepare_case",
        "source_head": source_head,
        "seeds": list(SEEDS),
        "seed_formula": "base_seed + 7919 * (case_index + 1)",
        "small": "exact input/reference/scale tensor bytes in deterministic TG0 archives",
        "paper_and_plan": "shape/dtype/sha256 plus reproducible case seed; no raw tensor archive",
        "combos": combos,
    }
    return document, archives


def manifest_bytes(document: dict, archives: dict[str, bytes]) -> bytes:
    lines = [f"{hashlib.sha256(canonical(document)).hexdigest()}  index.json"]
    lines.extend(f"{hashlib.sha256(archives[name]).hexdigest()}  {name}" for name in sorted(archives))
    return ("\n".join(lines) + "\n").encode("utf-8")


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"


def write_corpus(document: dict, archives: dict[str, bytes]) -> None:
    verify_archive_records(document, archives)
    CORPUS.mkdir(parents=True, exist_ok=True)
    INDEX.write_bytes(canonical(document))
    for relative, content in archives.items():
        path = CORPUS / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    MANIFEST.write_bytes(manifest_bytes(document, archives))


def verify_archive_records(document: dict, archives: dict[str, bytes]) -> None:
    expected_paths = {
        record["archive"]
        for combo in document["combos"]
        for record in combo["case_records"]
        if record["archive"] is not None
    }
    if set(archives) != expected_paths:
        raise AssertionError("raw archive path set does not match case records")
    for combo in document["combos"]:
        for record in combo["case_records"]:
            relative = record["archive"]
            if relative is None:
                if record["archive_sha256"] is not None:
                    raise AssertionError("digest-only case has an archive digest")
                continue
            if f"/{combo['dtype']}/" not in relative:
                raise AssertionError(f"archive path loses dtype: {relative}")
            packed = archives[relative]
            if hashlib.sha256(packed).hexdigest() != record["archive_sha256"]:
                raise AssertionError(f"archive digest mismatch: {relative}")
            raw = zlib.decompress(packed)
            if not raw.startswith(MAGIC) or len(raw) < len(MAGIC) + 4:
                raise AssertionError(f"bad archive magic: {relative}")
            size = int.from_bytes(raw[len(MAGIC):len(MAGIC) + 4], "little")
            start = len(MAGIC) + 4
            header = json.loads(raw[start:start + size])
            if header != record["tensors"]["entries"]:
                raise AssertionError(f"archive metadata mismatch: {relative}")
            payload = raw[start + size:]
            for entry in header:
                offset, nbytes = entry["offset"], entry["nbytes"]
                content = payload[offset:offset + nbytes]
                if len(content) != nbytes or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise AssertionError(f"tensor bytes mismatch: {relative}:{entry['role']}/{entry['name']}")


def self_test(document: dict, archives: dict[str, bytes]) -> None:
    def dtype_loss(doc: dict, blobs: dict) -> None:
        for combo in doc["combos"]:
            if combo["size"] == "small":
                record = combo["case_records"][0]
                record["archive"] = record["archive"].replace(f"/{combo['dtype']}/", "/unknown/")
                return

    def metadata_drift(doc: dict, blobs: dict) -> None:
        for combo in doc["combos"]:
            if combo["size"] == "small":
                combo["case_records"][0]["tensors"]["entries"][0]["dtype"] = "fp32"
                return

    def payload_corruption(doc: dict, blobs: dict) -> None:
        key = next(iter(blobs))
        blobs[key] = b"not-zlib"

    def extra_archive(doc: dict, blobs: dict) -> None:
        blobs["tensors/unexpected.tg0z"] = b"extra"

    cases = [dtype_loss, metadata_drift, payload_corruption, extra_archive]
    for mutate in cases:
        doc, blobs = copy.deepcopy(document), dict(archives)
        mutate(doc, blobs)
        try:
            verify_archive_records(doc, blobs)
        except (AssertionError, ValueError, zlib.error, json.JSONDecodeError):
            continue
        raise AssertionError(f"negative fixture accepted: {mutate.__name__}")
    print(f"OK: {len(cases)} corpus negative fixtures rejected")


def replay() -> None:
    expected = json.loads(INDEX.read_text(encoding="utf-8"))
    expected_archives = {}
    for line in MANIFEST.read_text().splitlines()[1:]:
        digest, relative = line.split("  ", 1)
        path = CORPUS / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise AssertionError(f"archive manifest mismatch: {relative}")
        expected_archives[relative] = path.read_bytes()
    verify_archive_records(expected, expected_archives)
    if head() != expected["source_head"]:
        raise AssertionError("source HEAD moved; replay requires the recorded old-path checkout, not the new path")
    actual, actual_archives = generate()
    if canonical(actual) != canonical(expected):
        raise AssertionError("old-path corpus content drifted")
    if actual_archives != expected_archives:
        raise AssertionError("old-path tensor archive set or bytes drifted")
    print(f"OK: replayed {len(actual['combos'])} combos and {len(actual_archives)} raw tensor archives")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.replay:
        replay()
        return
    document, archives = generate()
    write_corpus(document, archives)
    if args.self_test:
        self_test(document, archives)
    cases = sum(len(combo["case_records"]) for combo in document["combos"])
    print(f"OK: {len(document['combos'])} combos, {cases} case/seed records, {len(archives)} raw archives")
    print(f"OK: {INDEX}")
    print(f"OK: {MANIFEST}")


if __name__ == "__main__":
    main()
