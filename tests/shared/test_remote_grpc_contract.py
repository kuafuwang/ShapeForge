"""Phase 2C1 Protobuf schema, generated artifacts, and dependency boundaries."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
descriptor = pytest.importorskip("google.protobuf.descriptor", reason="Protobuf is an optional grpc dependency")
descriptor_pb2 = pytest.importorskip("google.protobuf.descriptor_pb2", reason="Protobuf is an optional grpc dependency")
pb2 = pytest.importorskip("tcenv.remote.v1.evaluator_pb2", reason="Protobuf is an optional grpc dependency")

ROOT = Path(__file__).resolve().parents[2]


def test_evaluator_service_has_exactly_the_six_contract_rpcs():
    service = pb2.DESCRIPTOR.services_by_name["Evaluator"]
    assert service.full_name == "tcenv.remote.v1.Evaluator"
    assert [method.name for method in service.methods] == [
        "GetCapabilities", "Submit", "GetJob", "Watch", "Cancel", "Retry",
    ]
    expected = {
        "GetCapabilities": ("CapabilitiesRequest", "Capabilities"),
        "Submit": ("SubmitRequest", "JobSnapshot"),
        "GetJob": ("JobRef", "JobSnapshot"),
        "Watch": ("WatchRequest", "JobEvent"),
        "Cancel": ("CancelRequest", "JobSnapshot"),
        "Retry": ("RetryRequest", "JobSnapshot"),
    }
    for method in service.methods:
        request_name, response_name = expected[method.name]
        assert method.input_type.name == request_name
        assert method.output_type.name == response_name
        assert not method.client_streaming
        assert method.server_streaming is (method.name == "Watch")


def test_request_source_integer_timeouts_and_explicit_presence():
    fields = pb2.EvaluationRequest.DESCRIPTOR.fields_by_name
    assert fields["source"].type == descriptor.FieldDescriptor.TYPE_BYTES
    assert fields["seed"].type == descriptor.FieldDescriptor.TYPE_STRING
    assert fields["seed"].has_presence
    assert fields["cases"].type == descriptor.FieldDescriptor.TYPE_MESSAGE
    assert fields["cases"].has_presence
    for name in ("queue_timeout_ms", "candidate_timeout_ms"):
        assert fields[name].type == descriptor.FieldDescriptor.TYPE_UINT32

    request = pb2.EvaluationRequest()
    assert not request.HasField("seed")
    assert not request.HasField("cases")
    request.seed = "0"
    assert request.HasField("seed")
    request.cases.SetInParent()
    assert request.HasField("cases") and list(request.cases.names) == []


def test_source_bytes_survive_protobuf_without_text_coercion():
    source = b"\xff\x00kernel\r\n  \x80"
    request = pb2.EvaluationRequest(source=source)
    decoded = pb2.EvaluationRequest.FromString(request.SerializeToString())
    assert decoded.source == source
    assert isinstance(decoded.source, bytes)


def test_identity_attempt_blob_and_signature_field_types():
    provenance = pb2.EvaluatorProvenance()
    assert not provenance.HasField("attempt_id")
    assert not provenance.HasField("request_digest")
    provenance.attempt_id = "attempt-1"
    provenance.request_digest = "a" * 64
    assert provenance.HasField("attempt_id") and provenance.HasField("request_digest")

    watch = pb2.WatchRequest()
    assert not watch.HasField("attempt_id")
    watch.attempt_id = "attempt-1"
    assert watch.HasField("attempt_id")

    event = pb2.JobEvent()
    assert not event.HasField("attempt_id")
    event.attempt_id = "attempt-1"
    assert event.HasField("attempt_id")

    result = pb2.EvaluationResult()
    assert not result.HasField("evidence")
    result.evidence.SetInParent()
    assert result.HasField("evidence")

    blob_fields = pb2.OpaqueBlob.DESCRIPTOR.fields_by_name
    assert blob_fields["size_bytes"].type == descriptor.FieldDescriptor.TYPE_UINT64
    assert blob_fields["sha256"].type == descriptor.FieldDescriptor.TYPE_STRING
    assert blob_fields["data"].type == descriptor.FieldDescriptor.TYPE_BYTES
    signature_fields = pb2.ResultSignature.DESCRIPTOR.fields_by_name
    assert signature_fields["algorithm"].type == descriptor.FieldDescriptor.TYPE_STRING
    assert signature_fields["key_id"].type == descriptor.FieldDescriptor.TYPE_STRING
    assert signature_fields["signature"].type == descriptor.FieldDescriptor.TYPE_BYTES
    for message, field in ((pb2.Attempt, "lease_fence"), (pb2.JobEvent, "sequence"), (pb2.Receipt, "session_sequence")):
        assert message.DESCRIPTOR.fields_by_name[field].type == descriptor.FieldDescriptor.TYPE_UINT64


def test_all_contract_envelopes_and_state_values_exist():
    names = {
        "EvaluationRequest", "EvaluatorProvenance", "EvaluationResult", "Receipt", "JobSnapshot",
        "Attempt", "Capabilities", "CapabilitiesRequest", "JobProgress", "JobEvent", "CancelRequest",
        "RetryRequest", "EvaluationErrorDetail", "ExpectedIdentities", "ResourceLimits",
    }
    assert names <= set(pb2.DESCRIPTOR.message_types_by_name)
    assert pb2.JOB_STATE_INTERRUPTED == 4
    assert pb2.JOB_STATE_COMPLETED == 5
    assert pb2.JOB_STATE_FAILED == 6
    assert pb2.JOB_STATE_CANCELLED == 7
    assert pb2.JOB_STATE_EXPIRED == 8
    assert pb2.ATTEMPT_STATE_INTERRUPTED == 3
    assert pb2.FAILURE_DOMAIN_INFRASTRUCTURE == 1
    for name in ("evaluator_epoch", "api_contract_id", "evidence_contract_id"):
        assert name in pb2.ExpectedIdentities.DESCRIPTOR.fields_by_name or name in pb2.ExpectedEvaluator.DESCRIPTOR.fields_by_name


def test_standard_google_rpc_status_carries_typed_evaluation_details():
    pytest.importorskip("google.rpc", reason="googleapis-common-protos is optional")
    from google.rpc import error_details_pb2, status_pb2

    status = status_pb2.Status(code=9, message="required feature is unavailable")
    status.details.add().Pack(error_details_pb2.ErrorInfo(
        reason="REQUIRED_FEATURE_UNAVAILABLE",
        domain="tcenv.remote.v1",
        metadata={"request_id": "request-1"},
    ))
    retry = error_details_pb2.RetryInfo()
    retry.retry_delay.seconds = 1
    status.details.add().Pack(retry)
    typed = pb2.EvaluationErrorDetail(
        request_id="request-1",
        reason="REQUIRED_FEATURE_UNAVAILABLE",
        missing_features=["watch-snapshot-reset"],
        failure_domain=pb2.FAILURE_DOMAIN_PROTOCOL,
        retryable=False,
    )
    status.details.add().Pack(typed)
    assert status.details[-1].type_url == "type.googleapis.com/tcenv.remote.v1.EvaluationErrorDetail"
    decoded = pb2.EvaluationErrorDetail()
    assert status.details[-1].Unpack(decoded)
    assert decoded == typed
    assert status.details[0].Is(error_details_pb2.ErrorInfo.DESCRIPTOR)
    assert status.details[1].Is(error_details_pb2.RetryInfo.DESCRIPTOR)


def test_no_unversioned_struct_any_or_json_tunnel_is_used():
    file_descriptors = [pb2.DESCRIPTOR, *pb2.DESCRIPTOR.dependencies]
    assert {item.name for item in pb2.DESCRIPTOR.dependencies} == {"google/protobuf/timestamp.proto"}
    for file in file_descriptors:
        for message in file.message_types_by_name.values():
            assert message.full_name not in {"google.protobuf.Struct", "google.protobuf.Any"}
            for field in message.fields:
                assert field.message_type is None or field.message_type.full_name not in {
                    "google.protobuf.Struct", "google.protobuf.Any",
                }


def test_checked_in_descriptor_set_matches_generated_message_schema():
    descriptor_set = descriptor_pb2.FileDescriptorSet()
    descriptor_set.ParseFromString(
        (ROOT / "proto/tcenv/remote/v1/evaluator.desc").read_bytes()
    )
    files = {file.name: file for file in descriptor_set.file}
    assert "google/protobuf/timestamp.proto" in files
    generated = files["tcenv/remote/v1/evaluator.proto"]
    assert generated.package == "tcenv.remote.v1"
    runtime = descriptor_pb2.FileDescriptorProto.FromString(pb2.DESCRIPTOR.serialized_pb)
    # The runtime pool omits default json_name values; the descriptor set records
    # them explicitly. Compare every other schema field after normalizing that
    # derivable serialization detail rather than comparing container bytes.
    for file in (generated, runtime):
        for message in file.message_type:
            for field in message.field:
                field.ClearField("json_name")
    assert generated == runtime


def test_v1_schema_changes_are_backward_compatible_with_the_frozen_baseline():
    """Reject field-number reuse, type changes, RPC renames and unreserved deletions."""
    baseline_path = ROOT / "tests/support/golden/remote/evaluator_v1_baseline.desc"
    assert baseline_path.is_file(), "the v1 compatibility baseline is required"
    baseline_set = descriptor_pb2.FileDescriptorSet()
    baseline_set.ParseFromString(baseline_path.read_bytes())
    current_set = descriptor_pb2.FileDescriptorSet()
    current_set.ParseFromString((ROOT / "proto/tcenv/remote/v1/evaluator.desc").read_bytes())
    baseline = next(file for file in baseline_set.file if file.name == "tcenv/remote/v1/evaluator.proto")
    current = next(file for file in current_set.file if file.name == "tcenv/remote/v1/evaluator.proto")
    assert baseline.package == current.package
    assert baseline.syntax == current.syntax

    baseline_service = baseline.service[0]
    current_service = current.service[0]
    assert baseline_service.name == current_service.name == "Evaluator"
    assert len(baseline_service.method) == len(current_service.method) == 6
    for old, new in zip(baseline_service.method, current_service.method, strict=True):
        assert old.name == new.name
        assert old.input_type == new.input_type
        assert old.output_type == new.output_type
        assert old.client_streaming == new.client_streaming
        assert old.server_streaming == new.server_streaming

    def reserved_numbers(message):
        return {number for span in message.reserved_range for number in range(span.start, span.end)}

    for old_message in baseline.message_type:
        new_message = next((item for item in current.message_type if item.name == old_message.name), None)
        assert new_message is not None, f"message deleted without a v2: {old_message.name}"
        new_fields = {field.number: field for field in new_message.field}
        reserved = reserved_numbers(new_message)
        for old_field in old_message.field:
            new_field = new_fields.get(old_field.number)
            if new_field is None:
                assert old_field.number in reserved or old_field.name in new_message.reserved_name, (
                    f"{old_message.name}.{old_field.name} was deleted without reservation"
                )
                continue
            for attribute in ("name", "number", "label", "type", "type_name", "default_value", "oneof_index", "proto3_optional"):
                assert getattr(old_field, attribute) == getattr(new_field, attribute), (
                    f"{old_message.name}.{old_field.name} changed {attribute}"
                )
        for old_enum in old_message.enum_type:
            new_enum = next((item for item in new_message.enum_type if item.name == old_enum.name), None)
            assert new_enum is not None
            old_values = {value.name: value for value in old_enum.value}
            new_values = {value.name: value for value in new_enum.value}
            for name, value in old_values.items():
                assert name in new_values and new_values[name].number == value.number

    for old_enum in baseline.enum_type:
        new_enum = next((item for item in current.enum_type if item.name == old_enum.name), None)
        assert new_enum is not None
        old_values = {value.name: value for value in old_enum.value}
        new_values = {value.name: value for value in new_enum.value}
        for name, value in old_values.items():
            assert name in new_values and new_values[name].number == value.number


def test_generated_grpc_stub_exposes_the_same_six_methods():
    pytest.importorskip("grpc", reason="grpcio is optional for local-only installs")
    module = importlib.import_module("tcenv.remote.v1.evaluator_pb2_grpc")
    servicer_methods = {
        name for name in ("GetCapabilities", "Submit", "GetJob", "Watch", "Cancel", "Retry")
        if callable(getattr(module.EvaluatorServicer, name, None))
    }
    assert servicer_methods == {"GetCapabilities", "Submit", "GetJob", "Watch", "Cancel", "Retry"}
    assert hasattr(module, "EvaluatorStub")
    assert hasattr(module, "add_EvaluatorServicer_to_server")


def test_local_evaluator_contract_and_messages_import_without_grpc():
    code = r'''
import builtins
import sys
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "grpc" or name.startswith("grpc."):
        raise AssertionError("local import attempted to import grpc")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
from tcenv.evaluation import LocalEvaluator
from tcenv.remote import contracts
from tcenv.remote.v1 import evaluator_pb2
import taic.remote_runtime
assert LocalEvaluator and contracts and evaluator_pb2 and taic.remote_runtime
assert "grpc" not in sys.modules
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, timeout=30, check=False
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_reproducible_codegen_has_no_drift():
    protoc = shutil.which("protoc")
    uv = shutil.which("uv")
    if not protoc or not uv:
        if os.environ.get("CI"):
            pytest.fail("CI requires protoc 29.3 and uv with the locked codegen group")
        pytest.skip("protoc 29.3 and uv are required for codegen drift check")
    completed = subprocess.run(
        [uv, "run", "--group", "codegen", "python", "scripts/generate_evaluator_proto.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "up to date" in completed.stdout
