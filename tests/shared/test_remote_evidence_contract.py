"""Phase 2C1 transport-neutral digest, receipt, signature, and legacy goldens."""

from __future__ import annotations

import base64
import copy
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tcenv import evaluation as legacy
from tcenv.remote import contracts

from ..support.remote_contract import GO_VERIFIER, GOLDEN, load_evaluator_golden


@pytest.fixture(scope="module")
def golden() -> dict:
    return load_evaluator_golden()


def _source(golden: dict) -> bytes:
    return base64.b64decode(golden["source_base64"])


def test_request_digest_matches_frozen_golden(golden: dict):
    source = _source(golden)
    assert b"\r\n" in source and source.endswith(b"\r\n")
    document = contracts.request_digest_document(golden["request"], source_bytes=source)
    assert document == golden["request_digest_document"]
    assert contracts.canonical_digest_document(document).decode("utf-8") == golden["request_canonical_json"]
    assert contracts.request_digest(golden["request"], source_bytes=source) == golden["request_digest"]
    assert hashlib.sha256(source).hexdigest() == golden["request"]["candidate_sha256"]


def test_exact_source_bytes_are_not_normalized(golden: dict):
    request = copy.deepcopy(golden["request"])
    changed_source = _source(golden).replace(b"\r\n", b"\n")
    request["candidate_sha256"] = hashlib.sha256(changed_source).hexdigest()
    assert contracts.request_digest(request, source_bytes=changed_source) != golden["request_digest"]

    # The evidence layer binds bytes before any supported-text decoder runs.
    binary_source = b"\xff\x00\r\n"
    request["candidate_sha256"] = hashlib.sha256(binary_source).hexdigest()
    assert contracts.request_digest(request, source_bytes=binary_source)

    with pytest.raises(contracts.ContractError, match="implicit text encoding"):
        contracts.request_digest(golden["request"], source_bytes="not bytes")
    with_source = {**golden["request"], "source": "implicit text"}
    with pytest.raises(contracts.ContractError, match="implicit text encoding"):
        contracts.request_digest(with_source, source_bytes=_source(golden))


def test_request_digest_excludes_only_non_execution_envelope(golden: dict):
    envelope = {
        **golden["request"],
        "source": _source(golden),
        "request_id": "another-request-id",
        "attempt_id": "another-attempt-id",
        "authorization": "Bearer must-not-be-hashed",
        "credentials": {"private_key": "must-not-be-hashed"},
        "session_grant": "must-not-be-hashed",
        "tls_identity": "another-host",
        "rpc_deadline_ms": 999,
        "trace_id": "trace",
        "retry_count": 8,
        "stream_sequence": 9,
        "workdir": "/private/local/work",
        "out": "/private/local/out",
        "heartbeat": "/private/local/heartbeat",
        "progress_path": "/private/local/progress",
        "self_check_dir": "/private/local/selfcheck",
        "expected": {"site_id": "another-site"},
        "evaluator_epoch": "a" * 64,
        "api_contract_id": "another-api",
        "evidence_contract_id": "another-evidence",
        "capability_id": "another-capability",
        "required_features": ["another-feature"],
        "lease_fence": 99,
    }
    assert contracts.request_digest(envelope, source_bytes=_source(golden)) == golden["request_digest"]


def test_every_execution_field_and_explicit_default_affects_digest(golden: dict):
    mutations = {
        "kind": ("selfcheck", {"mode": "timed"}),
        "run_id": ("other-run", {}),
        "session_id": ("other-session", {}),
        "task": ("other-task", {}),
        "dtype": ("fp32", {}),
        "size": ("large", {}),
        "target_id": ("other-target", {}),
        "task_fingerprint": ("a" * 64, {}),
        "mode": ("full", {}),
        "seed": (None, {}),
        "reps": (16, {}),
        "queue_timeout_ms": (30001, {}),
        "candidate_timeout_ms": (420001, {}),
        "cases": (None, {}),
        "fast_bench": (True, {}),
        "require_contract_launch": (True, {}),
        "validate": (True, {}),
        "measure_serial": (True, {}),
        "cache_policy": ("disabled", {}),
    }
    for field, (value, extra) in mutations.items():
        changed = copy.deepcopy(golden["request"])
        changed[field] = value
        changed.update(extra)
        assert contracts.request_digest(changed, source_bytes=_source(golden)) != golden["request_digest"], field

    empty_cases = copy.deepcopy(golden["request"])
    empty_cases["cases"] = []
    null_cases = copy.deepcopy(golden["request"])
    null_cases["cases"] = None
    assert contracts.request_digest(empty_cases, source_bytes=_source(golden)) != contracts.request_digest(
        null_cases, source_bytes=_source(golden)
    )

    for field in golden["request"]:
        missing = copy.deepcopy(golden["request"])
        del missing[field]
        with pytest.raises(contracts.ContractError, match="missing explicit"):
            contracts.request_digest(missing, source_bytes=_source(golden))


def test_unknown_fields_and_seconds_are_rejected(golden: dict):
    for field in ("queue_timeout_s", "candidate_timeout_s", "new_execution_option"):
        changed = copy.deepcopy(golden["request"])
        changed[field] = 1
        with pytest.raises(contracts.ContractError, match="unknown request fields"):
            contracts.request_digest(changed, source_bytes=_source(golden))


@pytest.mark.parametrize("field", ["queue_timeout_ms", "candidate_timeout_ms"])
@pytest.mark.parametrize("value", [0, 3_600_001, 1.0, True])
def test_timeout_integer_bounds(golden: dict, field: str, value: object):
    request = copy.deepcopy(golden["request"])
    request[field] = value
    with pytest.raises(contracts.ContractError, match="integer bounds"):
        contracts.request_digest(request, source_bytes=_source(golden))


@pytest.mark.parametrize("value", [0, 10_001, 1.0, True])
def test_reps_integer_bounds(golden: dict, value: object):
    request = copy.deepcopy(golden["request"])
    request["reps"] = value
    with pytest.raises(contracts.ContractError, match="integer bounds"):
        contracts.request_digest(request, source_bytes=_source(golden))


def test_seconds_conversion_golden_and_invalid_values(golden: dict):
    for vector in golden["timeout_seconds_to_ms"]:
        assert contracts.seconds_to_timeout_ms(vector["seconds"]) == vector["milliseconds"]
    assert contracts.seconds_to_timeout_ms(0.0001) == 1
    assert contracts.seconds_to_timeout_ms("1e-3") == 1
    for value in (0, -1, "NaN", "Infinity", "3600.0001", True, None, " 1 ", "1.", ".5", "+1", "01"):
        with pytest.raises(contracts.ContractError):
            contracts.seconds_to_timeout_ms(value)
    for value in golden.get("timeout_seconds_rejected", []):
        with pytest.raises(contracts.ContractError):
            contracts.seconds_to_timeout_ms(value)


def test_seed_presence_decimal_and_uint64_bounds(golden: dict):
    maximum = copy.deepcopy(golden["request"])
    maximum["seed"] = str(2**64 - 1)
    assert contracts.request_digest(maximum, source_bytes=_source(golden)) != golden["request_digest"]
    for value in (7, "", "00", "01", "-1", "+1", str(2**64), "1" * 10000):
        changed = copy.deepcopy(golden["request"])
        changed["seed"] = value
        with pytest.raises(contracts.ContractError, match="seed"):
            contracts.request_digest(changed, source_bytes=_source(golden))


def test_attempt_scoped_receipt_golden(golden: dict):
    vector = golden["receipt"]
    document = contracts.receipt_id_document(
        request_id=vector["request_id"],
        attempt_id=vector["attempt_id"],
        request_digest=golden["request_digest"],
    )
    assert document == vector["document"]
    assert contracts.canonical_digest_document(document).decode("utf-8") == vector["canonical_json"]
    assert contracts.attempt_receipt_id(
        request_id=vector["request_id"],
        attempt_id=vector["attempt_id"],
        request_digest=golden["request_digest"],
    ) == vector["receipt_id"]
    other_attempt = contracts.attempt_receipt_id(
        request_id=vector["request_id"], attempt_id="attempt-0002", request_digest=golden["request_digest"]
    )
    assert other_attempt != vector["receipt_id"]


def test_result_signature_golden_blob_and_key_id_binding(golden: dict):
    vector = golden["signature"]
    unsigned = vector["unsigned_result"]
    key = bytes.fromhex(vector["key_hex"])
    document = contracts.result_signature_document(unsigned, key_id=vector["key_id"])
    assert document == vector["document"]
    assert contracts.canonical_digest_document(document).decode("utf-8") == vector["canonical_json"]
    assert contracts.sign_result(unsigned, key_id=vector["key_id"], key=key) == vector["hmac_sha256"]
    assert contracts.verify_result_signature(
        unsigned, key_id=vector["key_id"], key=key, signature=vector["hmac_sha256"]
    )

    for name in ("report", "evidence"):
        raw = base64.b64decode(vector[f"{name}_base64"])
        descriptor = unsigned[name]
        assert isinstance(descriptor["size_bytes"], int) and not isinstance(descriptor["size_bytes"], bool)
        assert contracts.validate_blob(descriptor, raw) == descriptor
        with pytest.raises(contracts.ContractError, match="size"):
            contracts.validate_blob(descriptor, raw + b"x")
        with pytest.raises(contracts.ContractError, match="digest"):
            contracts.validate_blob(descriptor, b"\0" * len(raw))

    assert not contracts.verify_result_signature(
        unsigned, key_id=vector["key_id"], key=bytes(reversed(key)), signature=vector["hmac_sha256"]
    )
    assert not contracts.verify_result_signature(
        unsigned, key_id="site-golden/revoked", key=key, signature=vector["hmac_sha256"]
    )
    assert not contracts.verify_result_signature(
        unsigned, key_id=vector["key_id"], key=key, signature="0" * 64
    )
    other_key_id_signature = contracts.sign_result(unsigned, key_id="site-golden/next", key=key)
    assert other_key_id_signature != vector["hmac_sha256"]

    tampered = copy.deepcopy(unsigned)
    tampered["report"]["sha256"] = "0" * 64
    assert not contracts.verify_result_signature(
        tampered, key_id=vector["key_id"], key=key, signature=vector["hmac_sha256"]
    )
    tampered = copy.deepcopy(unsigned)
    tampered["provenance"]["evaluator_epoch"] = "6" * 64
    assert not contracts.verify_result_signature(
        tampered, key_id=vector["key_id"], key=key, signature=vector["hmac_sha256"]
    )
    tampered = copy.deepcopy(unsigned)
    tampered["receipt"]["attempt_id"] = "attempt-0002"
    with pytest.raises(contracts.ContractError, match="receipt_id"):
        contracts.result_signature_document(tampered, key_id=vector["key_id"])


def test_external_provenance_requires_nonempty_build_and_pack_digests(golden: dict):
    unsigned = golden["signature"]["unsigned_result"]
    for field in ("worker_build", "task_pack_digest", "calibration_digest"):
        changed = copy.deepcopy(unsigned)
        changed["provenance"][field] = ""
        with pytest.raises(contracts.ContractError):
            contracts.result_signature_document(changed, key_id=golden["signature"]["key_id"])
    changed = copy.deepcopy(unsigned)
    changed["provenance"]["worker_build"] = "x" * 257
    with pytest.raises(contracts.ContractError, match="worker_build"):
        contracts.result_signature_document(changed, key_id=golden["signature"]["key_id"])


def test_canonical_document_rejects_ambiguous_json():
    for value in ({"value": 1.0}, {"value": 2**53}, {"é": 1}):
        with pytest.raises(contracts.ContractError):
            contracts.canonical_digest_document(value)
    with pytest.raises(contracts.ContractError, match="Unicode"):
        contracts.canonical_digest_document({"value": "\ud800"})


def test_legacy_digest_receipt_and_hmac_are_unchanged(golden: dict):
    vector = golden["legacy"]
    request = legacy.EvaluationRequest.from_dict(vector["request"])
    assert request.request_digest == vector["request_digest"]
    assert legacy.receipt_id_for(request) == vector["receipt_id"]
    result = legacy.EvaluationResult.from_dict(vector["result"])
    key = bytes.fromhex(vector["key_hex"])
    assert legacy.sign_result(result, key) == vector["signature"]
    result.signature = vector["signature"]
    assert legacy.verify_result_signature(result, key)
    assert request.source_bytes == _source(golden)
    assert golden["request_digest"] != vector["request_digest"]
    assert golden["receipt"]["receipt_id"] != vector["receipt_id"]


def test_independent_go_verifier_runs_when_available(golden: dict, tmp_path: Path):
    go = shutil.which("go")
    if go is None:
        pytest.skip("Go toolchain is not installed; cross-language verifier cannot run")
    env = os.environ.copy()
    env.update({
        "GOTOOLCHAIN": "local",
        "GOENV": "off",
        "GOCACHE": str(tmp_path / "go-cache"),
        "GOPATH": str(tmp_path / "go-path"),
        "GOMODCACHE": str(tmp_path / "go-mod-cache"),
    })
    completed = subprocess.run(
        [go, "run", ".", "-golden", str(GOLDEN)],
        cwd=GO_VERIFIER,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.startswith("ok request=")
    assert golden["request_digest"] in completed.stdout
    assert golden["receipt"]["receipt_id"] in completed.stdout
    assert golden["signature"]["hmac_sha256"] in completed.stdout
