"""Phase 0b local evaluator adapter tests. No worker, GPU or model."""

from pathlib import Path

from tcenv.evaluate import EvalReport
from tcenv.evaluation import (
    EvaluationRequest,
    EvaluatorProvenance,
    LocalEvaluator,
    local_provenance,
    request_digest,
)


def _report() -> EvalReport:
    """Return a report object without touching a device."""
    return EvalReport(task="relu", dtype="fp16", size="small", mode="quick")


def test_local_evaluator_forwards_every_current_evaluate_argument(tmp_path):
    """@brief LocalEvaluator 必须逐项转发当前 tcenv.evaluate 的全部关键字参数。"""
    report = _report()
    captured: dict = {}

    def fake_evaluate(task, dtype, size, kernel_source, **kwargs):
        captured.update({"task": task, "dtype": dtype, "size": size, "kernel_source": kernel_source, **kwargs})
        return report

    request = EvaluationRequest(
        task="relu",
        dtype="fp16",
        size="small",
        source="kernel void relu_kernel() {}",
        mode="quick",
        seed=17,
        reps=5,
        timeout=31.0,
        cases=("case_a", "case_b"),
        workdir=tmp_path / "work",
        progress_path=tmp_path / "progress.json",
        fast_bench=True,
        require_contract_launch=True,
        self_check_dir=tmp_path / "selfcheck",
        validate=True,
        measure_serial=True,
        eval_cache=True,
        target_id="apple-m4-max",
    )
    result = LocalEvaluator(evaluate_fn=fake_evaluate).formal(request)

    assert result.report is report
    assert captured == {
        "task": "relu",
        "dtype": "fp16",
        "size": "small",
        "kernel_source": "kernel void relu_kernel() {}",
        "mode": "quick",
        "seed": 17,
        "reps": 5,
        "timeout": 31.0,
        "cases": ["case_a", "case_b"],
        "workdir": tmp_path / "work",
        "progress_path": tmp_path / "progress.json",
        "fast_bench": True,
        "require_contract_launch": True,
        "self_check_dir": tmp_path / "selfcheck",
        "validate": True,
        "measure_serial": True,
        "eval_cache": True,
        "eval_cache_refresh": False,
        "target_id": "apple-m4-max",
    }
    assert result.provenance.site_id == "local"
    assert result.provenance.target_id == "apple-m4-max"
    assert result.provenance.measurement_kind == "measured"
    assert len(result.provenance.request_digest) == 64
    assert len(result.provenance.epoch) == 64


def test_selfcheck_uses_the_same_local_semantics(tmp_path):
    """@brief 自测本地路径与 formal 共用现有 evaluate，但保留 selfcheck 目录。"""
    captured: dict = {}

    def fake_evaluate(task, dtype, size, kernel_source, **kwargs):
        captured.update(kwargs)
        return _report()

    request = EvaluationRequest(
        task="relu",
        dtype="fp16",
        size="small",
        source="kernel void relu_kernel() {}",
        kind="selfcheck",
        mode="timed",
        self_check_dir=tmp_path / "checks",
    )
    result = LocalEvaluator(evaluate_fn=fake_evaluate).selfcheck(request)
    assert captured["self_check_dir"] == tmp_path / "checks"
    assert captured["mode"] == "timed"
    assert result.report.task == "relu"


def test_request_digest_ignores_request_id_but_tracks_source_bytes():
    """@brief request_id 是幂等键，不进入 body digest；源码变化必须改变 digest。"""
    first = EvaluationRequest(task="relu", dtype="fp16", size="small", source="a\n")
    second = EvaluationRequest(task="relu", dtype="fp16", size="small", source="a\n")
    changed = EvaluationRequest(task="relu", dtype="fp16", size="small", source="a\r\n")
    assert request_digest(first) == request_digest(second)
    assert request_digest(first) != request_digest(changed)
    assert first.request_id != second.request_id
    assert first.candidate_sha256 != changed.candidate_sha256

    local_path = EvaluationRequest(task="relu", dtype="fp16", size="small", source="a\n", workdir=Path("/tmp/a"))
    other_path = EvaluationRequest(task="relu", dtype="fp16", size="small", source="a\n", workdir=Path("/tmp/b"))
    assert request_digest(local_path) == request_digest(other_path)


def test_local_adapter_refuses_unsupported_validate_mode():
    """@brief validate 模式不能被本地 adapter 静默降级成 quick。"""
    request = EvaluationRequest(task="relu", dtype="fp16", size="small", source="body", mode="validate")
    try:
        LocalEvaluator(evaluate_fn=lambda *args, **kwargs: _report()).formal(request)
    except ValueError as exc:
        assert "validate" in str(exc)
    else:
        raise AssertionError("validate must be refused rather than run as quick")


def test_local_provenance_binds_the_registered_task_pack_digest():
    """本地测量上下文的 ``task_pack_digest`` 是登记的 pack 摘要，不是空串。"""
    from tasks import get_task

    request = EvaluationRequest(task="relu", dtype="fp16", size="small", source="body")
    provenance = local_provenance(request)
    registered = get_task("relu", "fp16", "small").evaluation_implementations["task_pack"]["digest"]
    assert provenance.task_pack_digest == registered
    assert len(provenance.task_pack_digest) == 64


def test_local_provenance_is_typed_and_stable_for_an_unchanged_request(tmp_path):
    """@brief 同一请求的本地 provenance 类型/epoch 稳定，且不伪装成远程站点。"""
    request = EvaluationRequest(
        task="relu",
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


def test_local_evaluator_defaults_to_the_current_evaluate_symbol():
    """@brief 未注入 fake 时，adapter 绑定现有 evaluate，而不是另一个评测实现。"""
    from tcenv.evaluate import evaluate

    assert LocalEvaluator()._evaluate_fn is evaluate


def test_kernel_source_alias_matches_source():
    """@brief 旧 API 的 kernel_source 拼写可作为只读兼容属性使用。"""
    request = EvaluationRequest(task="relu", dtype="fp16", size="small", source="body")
    assert request.kernel_source == "body"


def test_provenance_from_dict_rejects_missing_fields():
    """@brief provenance 缺字段必须是结构化错误，不能静默接受。"""
    import pytest

    from tcenv.evaluation import EvaluationError

    with pytest.raises(EvaluationError):
        EvaluatorProvenance.from_dict({})
