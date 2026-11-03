"""Single-submission segmented envelope: identity, exact bytes and the agent chain (A3, CPU).

The envelope is transport, not a second candidate format.  These tests pin the parts the plan
requires: ``candidate_sha256`` is the candidate-entry manifest digest (never the envelope hash),
all file segments survive decode, a corrupted/missing/extra segment fails closed, and the
agent pruner/submitter/self-check request do not split or drop the segments.
"""

from __future__ import annotations

import json

import pytest

from tcenv.evaluation import EvaluationRequest
from tcenv.track import (
    CandidateArtifact,
    CandidateEntry,
    EnvelopeError,
    decode_envelope,
    encode_envelope,
)


def _candidate() -> CandidateArtifact:
    """@brief 一个两文件 + 几何的 msl-bundle 候选。

    @return 候选。
    """
    geometry = json.dumps(
        {
            "schema_version": 1,
            "source": "candidate",
            "stages": [{"kernel": "fill", "grid": [1, 1, 1], "threads": [256, 1, 1]}],
        },
        sort_keys=True,
    ).encode()
    return CandidateArtifact(
        track_id="msl-bundle",
        entries=(
            CandidateEntry(
                path="kernel.metal",
                role="device_kernel",
                media_type="text/plain",
                data=b"kernel void fill(device half *x) {\n  // body with a taic-end-file word\n}\n",
            ),
            CandidateEntry(path="host/tiling.metal", role="host_tiling", media_type="text/plain", data=b"// tiling\n"),
            CandidateEntry(path="host/execution_geometry.json", role="schedule", media_type="application/json", data=geometry),
        ),
    )


def test_an_envelope_round_trips_and_keeps_the_candidate_manifest_identity():
    """@brief 解码后字节一致；candidate_sha256 是 manifest digest，不是 envelope 哈希。"""
    import hashlib

    candidate = _candidate()
    text = encode_envelope(candidate)
    decoded = decode_envelope(text)
    assert decoded.track_id == candidate.track_id
    def by_path(item):
        return {
            entry.path: (entry.role, entry.media_type, entry.data) for entry in item.entries
        }

    assert by_path(decoded) == by_path(candidate)
    assert decoded.candidate_sha256 == candidate.candidate_sha256 == hashlib.sha256(candidate.manifest_bytes).hexdigest()
    assert decoded.candidate_sha256 != hashlib.sha256(text.encode()).hexdigest()
    # The encoder sorts entries, so the same candidate bytes give one envelope text.
    reordered = CandidateArtifact(track_id=candidate.track_id, entries=tuple(reversed(candidate.entries)))
    assert encode_envelope(reordered) == text
    for entry in candidate.entries:
        assert entry.data.decode() in text


def test_a_corrupted_missing_or_extra_segment_is_refused():
    """@brief 段长度/摘要/清单不一致都拒绝，不做“修一下继续”。"""
    candidate = _candidate()
    text = encode_envelope(candidate)
    with pytest.raises(EnvelopeError):
        decode_envelope(text.replace("kernel void fill", "kernel void evil", 1))
    # A missing segment: remove one whole length-delimited segment while keeping the manifest.
    import base64

    def segment(body: str, path: str) -> str:
        start = body.index("taic-file " + base64.b64encode(path.encode()).decode() + " ")
        end = body.index("\ntaic-end-file\n", start) + len("\ntaic-end-file\n")
        return body[start:end]

    broken = text.replace(segment(text, "host/tiling.metal"), "")
    assert broken != text
    with pytest.raises(EnvelopeError, match="missing declared entries"):
        decode_envelope(broken)
    # An extra segment whose path is not in the fixed manifest.
    extra = CandidateEntry(path="host/extra.metal", role="host_tiling", media_type="text/plain", data=b"x\n")
    spliced = text.replace(
        "taic-end-envelope\n",
        "taic-file "
        + __import__("base64").b64encode(extra.path.encode()).decode()
        + " host_tiling "
        + __import__("base64").b64encode(extra.media_type.encode()).decode()
        + " 2 "
        + extra.sha256
        + "\nx\n\ntaic-end-file\ntaic-end-envelope\n",
    )
    with pytest.raises(EnvelopeError, match="undeclared|missing"):
        decode_envelope(spliced)
    with pytest.raises(EnvelopeError, match="must start"):
        decode_envelope("kernel void k() {}")
    with pytest.raises(EnvelopeError, match="trailing"):
        decode_envelope(text + "junk")
    with pytest.raises(EnvelopeError, match="single-source"):
        encode_envelope(
            CandidateArtifact(
                track_id="msl-single",
                entries=(CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),),
            )
        )
    with pytest.raises(EnvelopeError, match="UTF-8"):
        encode_envelope(
            CandidateArtifact(
                track_id="msl-bundle",
                entries=(
                    CandidateEntry(path="k.metal", role="device_kernel", media_type="text/plain", data=b"k\n"),
                    CandidateEntry(path="host/t.metal", role="host_tiling", media_type="text/plain", data=b"\xff"),
                ),
            )
        )


def test_the_agent_extractor_and_request_keep_every_file_segment():
    """@brief 围栏/裸文本抽取与 EvaluationRequest 往返都不拆 envelope。"""
    from taic.kernel_agent import kernel_text
    from taic.parse import extract_source

    candidate = _candidate()
    envelope = encode_envelope(candidate)
    fenced = "here is the bundle\n```metal\n" + envelope + "\n```\n"
    extracted = extract_source(fenced, "fill")
    assert extracted is not None
    assert decode_envelope(extracted).candidate_sha256 == candidate.candidate_sha256
    assert decode_envelope(kernel_text(fenced, "fill")).candidate_sha256 == candidate.candidate_sha256
    bare = kernel_text(envelope, "fill")
    assert bare is not None and decode_envelope(bare).candidate_sha256 == candidate.candidate_sha256

    request = EvaluationRequest(
        task="rmsnorm",
        dtype="fp16",
        size="small",
        source=extracted,
        task_fingerprint="a" * 64,
        candidate_sha256=candidate.candidate_sha256,
    )
    wire = json.loads(json.dumps(request.to_dict()))
    assert wire["source"] == extracted
    rebuilt = EvaluationRequest.from_dict(wire)
    assert rebuilt.source == extracted
    assert decode_envelope(rebuilt.source).candidate_sha256 == candidate.candidate_sha256


def test_prune_work_keeps_the_bundle_submission_file(tmp_path):
    """@brief prune_work 只归档草稿，交卷文件（含 envelope 全部段）原样保留。"""
    from taic.kernel_agent import prune_work

    run = tmp_path / "run"
    work = run / "sessions" / "init" / "agent_work"
    work.mkdir(parents=True)
    submission = work / "rmsnorm-fp16-small-init.metal"
    submission.write_text(encode_envelope(_candidate()), encoding="utf-8")
    draft = work / "scratch-draft.metal"
    draft.write_text("// discard me\n", encoding="utf-8")
    result = prune_work(work, label="init", keep={submission.name})
    assert submission.name in result["kept"]
    assert submission.is_file()
    assert decode_envelope(submission.read_text(encoding="utf-8")).candidate_sha256 == _candidate().candidate_sha256
    assert not draft.exists()
    assert result["archived"]


def test_the_v1_self_check_refuses_a_segmented_source_before_writing_one_file():
    """@brief 单文件自检/编译器不得把 envelope 写成 kernel.metal 或截断段。"""
    from tcenv.evaluate import segmented_source_refusal

    text = encode_envelope(_candidate())
    refusal = segmented_source_refusal(text)
    assert refusal and "must not split" in refusal
    assert segmented_source_refusal("kernel void fill() {}\n") is None
    # A near-miss is not an envelope and stays on the ordinary single-file path.
    assert segmented_source_refusal(text.replace("taic-end-envelope", "")) is None


def test_evaluation_request_uses_the_manifest_identity_on_default_and_from_dict():
    """@brief envelope 提交的缺省/from_dict/显式身份都只能是 manifest digest，不能是 transport 哈希。"""
    import hashlib

    from tcenv.evaluation import EvaluationError, EvaluationRequest

    candidate = _candidate()
    text = encode_envelope(candidate)
    transport = hashlib.sha256(text.encode()).hexdigest()
    request = EvaluationRequest(task="rmsnorm", dtype="fp16", size="small", source=text, task_fingerprint="a" * 64)
    assert request.candidate_sha256 == candidate.candidate_sha256 != transport
    assert EvaluationRequest.from_dict(request.to_dict()).candidate_sha256 == candidate.candidate_sha256
    explicit = EvaluationRequest(
        task="rmsnorm", dtype="fp16", size="small", source=text, candidate_sha256=candidate.candidate_sha256,
        task_fingerprint="a" * 64,
    )
    assert explicit.candidate_sha256 == candidate.candidate_sha256
    with pytest.raises(EvaluationError, match="transport hash"):
        EvaluationRequest(
            task="rmsnorm", dtype="fp16", size="small", source=text, candidate_sha256=transport,
            task_fingerprint="a" * 64,
        )
    with pytest.raises(EvaluationError, match="transport hash"):
        EvaluationRequest.from_dict({**request.to_dict(), "candidate_sha256": transport})
    with pytest.raises(EvaluationError, match="envelope is invalid"):
        EvaluationRequest(
            task="rmsnorm", dtype="fp16", size="small", source=text.replace("// tiling", "// tampered"),
            task_fingerprint="a" * 64,
        )


def test_a_tampered_taic_file_header_sha_is_refused():
    """@brief 头部 sha 与 manifest/body 不一致即拒绝，不能只看 manifest/body 相符。"""
    text = encode_envelope(_candidate())
    lines = text.splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith("taic-file "))
    fields = lines[index].split(" ")
    fields[-1] = "0" * 64
    lines[index] = " ".join(fields)
    with pytest.raises(EnvelopeError, match="header sha256"):
        decode_envelope("\n".join(lines) + "\n")


def test_the_decoder_enforces_the_same_segment_limit_as_the_encoder():
    """@brief 解码端也拒绝超过 64 个 taic-file 段。"""
    candidate = CandidateArtifact(
        track_id="msl-bundle",
        entries=tuple(
            CandidateEntry(
                path=f"part{i}.txt",
                role="device_kernel" if i == 0 else "host_tiling",
                media_type="text/plain",
                data=f"line {i}\n".encode(),
            )
            for i in range(64)
        ),
    )
    text = encode_envelope(candidate)
    end = text.rfind("\ntaic-end-envelope\n")
    start = text.rfind("taic-file ", 0, end)
    segment = text[start:end]
    duplicated = text[:end] + "\n" + segment + text[end:]
    with pytest.raises(EnvelopeError, match="at most 64 entries"):
        decode_envelope(duplicated)
