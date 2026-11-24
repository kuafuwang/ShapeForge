"""Current evaluation request and local-provenance contracts; no legacy local evaluator."""

from pathlib import Path

import pytest

from tcenv.evaluation import (
    EvaluationRequest, EvaluatorProvenance, local_provenance, request_digest,
)
from tests.support.v2_specimen import V2_TASK, v2_pack_registered_task

def test_request_digest_ignores_request_id_but_tracks_source_bytes():
    """@brief request_id 是幂等键，不进入 body digest；源码变化必须改变 digest。"""
    first = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="a\n")
    second = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="a\n")
    changed = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="a\r\n")
    assert request_digest(first) == request_digest(second)
    assert request_digest(first) != request_digest(changed)
    assert first.request_id != second.request_id
    assert first.candidate_sha256 != changed.candidate_sha256

    local_path = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="a\n", workdir=Path("/tmp/a"))
    other_path = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="a\n", workdir=Path("/tmp/b"))
    assert request_digest(local_path) == request_digest(other_path)

def test_local_provenance_binds_the_registered_task_pack_digest():
    """本地测量上下文的 ``task_pack_digest`` 是登记的 pack 摘要，不是空串。"""
    request = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="body")
    provenance = local_provenance(request)
    registered = v2_pack_registered_task(V2_TASK, "fp16", "small").evaluation_implementations["task_pack"]["digest"]
    assert provenance.task_pack_digest == registered
    assert len(provenance.task_pack_digest) == 64

def test_rmsnorm_registers_its_v2_pack_and_computes_a_contract_fingerprint():
    """RMSNorm 迁移后仍必须提供完整 task-pack 与合同身份，不能只在 worker 路径偶然可用。"""
    from tcenv.taskcontract import load_task_contract
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.track import REPO_ROOT

    root = REPO_ROOT / "tasks" / "rmsnorm"
    contract = load_task_contract(root, dtype="fp16", size="small")
    closure = task_pack_closure(root, root=REPO_ROOT)
    from tests.support.pack_identity import assert_task_pack_identity

    assert "opspec.json" in closure.task_files and "reference.py" in closure.task_files
    # Contract pack (track digest) and deployment closure (task + neutral code) are different
    # domains; each is checked exactly, never with a 64-character shape fallback.
    assert_task_pack_identity(contract, closure, task_dir=root, repo_root=REPO_ROOT)
    assert isinstance(contract.fingerprint(), str) and len(contract.fingerprint()) == 64

def test_local_provenance_is_typed_and_stable_for_an_unchanged_request(tmp_path):
    """@brief 同一请求的本地 provenance 类型/epoch 稳定，且不伪装成远程站点。"""
    request = EvaluationRequest(
        task=V2_TASK,
        dtype="fp16",
        size="small",
        source="kernel void relu_kernel() {}",
        workdir=tmp_path,
    )
    first = local_provenance(request)
    second = local_provenance(request)
    assert isinstance(first, EvaluatorProvenance)
    assert first == second
    assert first.site_id == "local"
    assert first.epoch == second.epoch
    assert first.request_digest == request_digest(request)
    assert first.protocol_version
    assert first.publication_allowed is True

def test_kernel_source_alias_matches_source():
    """@brief 旧 API 的 kernel_source 拼写可作为只读兼容属性使用。"""
    request = EvaluationRequest(task=V2_TASK, dtype="fp16", size="small", source="body")
    assert request.kernel_source == "body"

def test_provenance_from_dict_rejects_missing_fields():
    """@brief provenance 缺字段必须是结构化错误，不能静默接受。"""
    import pytest

    from tcenv.evaluation import EvaluationError

    with pytest.raises(EvaluationError):
        EvaluatorProvenance.from_dict({})
