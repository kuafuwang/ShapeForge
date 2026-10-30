"""Schema 4: the closed report shape, the one ``build`` record and its refusals.

No device, no model: these cases exercise the reader/permitter surface that R4 re-specifies
(§6.2/§6.4).  FakeNpu and FakeTsm are test-support builders; a ``full`` report from them is
structurally legal but ``build_kind != native``, so it can never be published.
"""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from tcenv.build_projection import project_full, project_partial, project_received
from tcenv.evaluate import EvalReport
from tcenv.remote import contracts
from tcenv.remote.artifact_builder import (
    BuildArtifact,
    BuildEvidence,
    BuildOutcome,
    BuildProvenance,
    SourceBundle,
)
from tcenv.remote.artifact_bundle import ArtifactBundle
from tcenv.schema import SchemaError, read_report_json
from tests.support.report_build import (
    attach_identity, native_build, native_compile_failure, simulated_build,
    simulated_compile_failure, task_identity,
)


def _task():
    from tasks import get_task

    return get_task("relu", "fp16", "small")


def _identity() -> dict:
    return task_identity("relu", "fp16", "small")


def _simulator_builder():
    from tests.support.npu_sim.fake_backend import FakeNpuBackend

    task = _task()
    backend = FakeNpuBackend()
    return backend.artifact_builder(
        kernel_contract=task.contract, specialization=task.specialization,
    ), task


def _outcome(source: bytes = b"kernel void demo() {}\n"):
    builder, task = _simulator_builder()
    return builder.build(SourceBundle(source), task_contract=task.contract), builder


def _base(status: str = "ok") -> dict:
    """A valid schema-4 report dict for the given status, built through the projectors."""
    from tests.support.report_build import report_for_request

    # The report stands for the local Metal executor in these shape tests, so the one native
    # structural fixture is requested explicitly.  Simulator cases replace the build themselves.
    return report_for_request(
        {"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "seed": 1,
         "target_id": "apple-m4-max"},
        status=status, build_kind="native",
    )


def _with(document: dict, **changes) -> dict:
    out = copy.deepcopy(document)
    if "status" in changes and changes["status"] != out.get("status"):
        # ``eligible`` is a derived display field; keep it equal to ``status == "ok"``.
        out["eligible"] = changes["status"] == "ok"
    out.update(changes)
    return out


def _build_with(document: dict, **changes) -> dict:
    out = copy.deepcopy(document["build"])
    out.update(changes)
    return out


def _without_build_key(document: dict, key: str) -> dict:
    out = copy.deepcopy(document)
    del out["build"][key]
    return out


# -- accepted shapes ---------------------------------------------------------
def test_a_local_native_full_and_a_compile_error_partial_load():
    """结构 native full（带 native document/digest）与结构 compile_error partial 都能读回。

    这不是一次真实 Metal 编译：fixture 是独立的 control-host/native 结构记录；FakeNpu 的
    simulator 结果只用于 simulator 形状与拒绝面。
    """
    # The native positive is an independent structural fixture: a typed control-host outcome
    # with Metal-native provenance, not a FakeNpu outcome relabeled as ``native``.
    native_full = native_build(_task(), b"kernel void demo() {}\n")
    assert native_full["build_kind"] == "native" and native_full["placement"] == "control_host"
    assert native_full["provenance"]["builder_id"] == "metal-native-structural-fixture"
    assert native_full["native_digest"] and native_full["native"]["backend_id"] == "metal"
    report = _with(_base("ok"), build=native_full)
    loaded = EvalReport.from_json(report)
    assert loaded.build["shape"] == "full" and loaded.build["build_kind"] == "native"

    failure = native_compile_failure(_task(), b"kernel void demo() {}\n")
    assert failure["shape"] == "partial" and failure["build_kind"] == "native"
    assert failure["error_class"] == "compile"
    partial = _with(_base("compile_error"), build=failure)
    assert EvalReport.from_json(partial).build["shape"] == "partial"


def test_a_post_build_failure_keeps_the_full_build():
    """构建后失败（incorrect/unsafe/...) 保留 full，不能变成 partial 或 null。"""
    for status in ("abi_error", "unsafe", "incorrect", "runtime_error"):
        report = _with(_base("ok"), status=status)
        loaded = EvalReport.from_json(report)
        assert loaded.build["shape"] == "full"


def test_a_received_prebuilt_full_and_a_simulated_full_with_a_fake_handle_load():
    """received prebuilt full（portable_artifact_set）与带假 handle 的模拟 full 都合法。"""
    # received prebuilt: a verified/loaded artifact set with the producer link, no native doc.
    outcome, builder = _outcome()
    producer = outcome.evidence.producer_record()
    from tcenv.remote.artifact_builder import BuildEvidence

    received = BuildEvidence(
        mode="prebuilt", status="reused", provenance=outcome.provenance,
        artifacts=outcome.evidence.artifacts, duration_ms=1, log=None, loaded=(
            (outcome.bundle.entries[0].path, outcome.bundle.entries[0].sha256),
        ), placement="received", producer=producer,
    )
    received_doc = project_received(received, build_kind="simulator", identity=_identity())
    assert received_doc["native_digest"] is None
    assert received_doc["native_absent_reason"] == "not_native_build"
    assert EvalReport.from_json(_with(_base("ok"), build=received_doc)).build["placement"] == "received"

    # simulator full with a fake live handle: legal structure, still not publishable.
    simulated = simulated_build(_task(), b"kernel void demo() {}\n")
    assert simulated["build_kind"] == "simulator" and simulated["native_digest"]
    assert EvalReport.from_json(_with(_base("ok"), build=simulated)).build["shape"] == "full"


@pytest.mark.parametrize("status", ["timeout", "crash", "refused_hang", "unchanged", "duplicate", "external_duplicate"])
def test_statuses_that_never_have_a_build_load_with_null(status):
    """这些状态只能是 build: null；带任何 build 都被拒绝。"""
    loaded = EvalReport.from_json(_with(_base("ok"), status=status, build=None))
    assert loaded.build is None
    with pytest.raises(SchemaError, match="needs human inspection"):
        EvalReport.from_json(_with(_base("ok"), status=status))


def test_rejected_may_be_null_or_full_but_not_partial():
    """rejected 在编译前是 null，构建后是 full；partial 不属于它。"""
    assert EvalReport.from_json(_with(_base("ok"), status="rejected", build=None)).build is None
    assert EvalReport.from_json(_with(_base("ok"), status="rejected")).build["shape"] == "full"
    partial = simulated_compile_failure(_task(), b"// COMPILE_FAIL\nkernel void demo() {}\n")
    with pytest.raises(SchemaError, match="cannot carry"):
        EvalReport.from_json(_with(_base("compile_error"), status="rejected", build=partial))


# -- refused documents -------------------------------------------------------
def test_the_retired_top_level_simulated_key_is_refused():
    """旧的顶层 ``simulated`` 键是表外字段，一律拒绝（模拟性只由 provenance/build_kind 承载）。"""
    with pytest.raises(SchemaError, match="fields this process does not know"):
        EvalReport.from_json(_with(_base("ok"), simulated=True))
    with pytest.raises(SchemaError, match="fields this process does not know"):
        EvalReport.from_json(_with(_base("ok"), simulated=False))


def test_pending_or_missing_status_is_not_a_persisted_state():
    """pending（构造默认值）或缺 status 都不是持久状态。"""
    with pytest.raises(SchemaError, match="pending"):
        EvalReport.from_json(_with(_base("ok"), status="pending"))
    document = _base("ok")
    del document["status"]
    with pytest.raises(SchemaError, match="missing"):
        EvalReport.from_json(document)


def test_a_missing_build_key_or_common_field_is_refused():
    document = _base("ok")
    del document["build"]
    with pytest.raises(SchemaError, match="missing \\['build'\\]"):
        EvalReport.from_json(document)
    for key in ("shape", "build_kind", "candidate_sha256", "assembled_sha256",
                "opspec_digest", "lowering_profile_digest", "provenance"):
        with pytest.raises(SchemaError, match="build is invalid|build is missing"):
            EvalReport.from_json(_without_build_key(_base("ok"), key))


def test_a_field_outside_the_build_table_is_refused():
    with pytest.raises(SchemaError, match="does not know"):
        EvalReport.from_json(_with(_base("ok"), build=_build_with(_base("ok"), extra_build_field=1)))


def test_partial_full_fields_and_full_partial_fields_are_refused():
    full_only = {"placement": "control_host", "artifacts": {}, "evidence": {}, "native_digest": None}
    with pytest.raises(SchemaError, match="does not know"):
        EvalReport.from_json(_with(
            _base("compile_error"),
            build=_build_with(_base("compile_error"), **full_only),
        ))
    with pytest.raises(SchemaError, match="does not know"):
        EvalReport.from_json(_with(
            _base("ok"), build=_build_with(_base("ok"), error_class="compile", duration_ms=1, log=None),
        ))


def test_source_only_can_never_carry_a_native_executable():
    """source_only 从不编译：build_kind 与 native document 的组合非法。"""
    simulated = simulated_build(_task(), b"kernel void demo() {}\n")
    with pytest.raises(SchemaError, match="source_only"):
        EvalReport.from_json(_with(_base("ok"), build=_build_with(
            _base("ok"), **{**simulated, "build_kind": "source_only"},
        )))
    # The projector refuses to create that combination in the first place.
    with pytest.raises(SchemaError, match="source_only"):
        project_full(_outcome()[0], build_kind="source_only", identity=_identity())


def test_source_only_full_without_native_is_legal():
    """source_only 的 full 是合法的（无 native、not_native_build），只是不可确认。"""
    outcome, _builder = _outcome()
    # A source_only build never compiles: the artifact set is the submitted source, so the
    # bundle carries no backend-native executable at all.
    from tcenv.remote.artifact_bundle import ArtifactBundle

    bundle = ArtifactBundle(entries=outcome.bundle.entries)
    provenance = outcome.provenance.__class__(
        profile=outcome.provenance.profile, source_sha256=outcome.provenance.source_sha256,
        artifact_set_digest=bundle.artifact_set_digest,
    )
    evidence = BuildEvidence(
        mode="prebuilt", status="ok", provenance=provenance,
        artifacts=(BuildArtifact.from_entry(bundle.entries[0]),), duration_ms=0, log=None,
        placement="control_host",
    )
    source_only_outcome = BuildOutcome(bundle=bundle, provenance=provenance, evidence=evidence)
    document = project_full(source_only_outcome, build_kind="source_only", identity=_identity())
    assert document["native_digest"] is None
    assert document["native_absent_reason"] == "not_native_build"
    assert EvalReport.from_json(_with(_base("ok"), build=document)).build["shape"] == "full"


def test_the_evaluation_identity_keys_are_required_and_hex_or_null():
    for key in ("evaluation_fingerprint", "baseline_implementation_digest", "baseline_environment_digest"):
        document = _base("ok")
        del document[key]
        with pytest.raises(SchemaError, match="missing"):
            EvalReport.from_json(document)
        for bad in ("x" * 64, "A" * 64, "0" * 63, 1, True):
            with pytest.raises(SchemaError, match="hex"):
                EvalReport.from_json(_with(_base("ok"), **{key: bad}))
        assert EvalReport.from_json(_with(_base("ok"), **{key: "a" * 64})).to_json()[key] == "a" * 64


@pytest.mark.parametrize("status,shape", [
    ("ok", "partial"), ("compile_error", "full"), ("abi_error", "partial"),
    ("unsafe", None), ("incorrect", "partial"), ("runtime_error", None),
])
def test_status_and_shape_outside_the_closed_table_is_refused(status, shape):
    full = _base("ok")["build"]
    partial = _base("compile_error")["build"]
    build = None if shape is None else (full if shape == "full" else partial)
    with pytest.raises(SchemaError, match="build is invalid"):
        EvalReport.from_json(_with(_base("ok"), status=status, build=build))


def test_native_absent_reason_must_match_the_build_kind_and_exclude_native():
    simulated = simulated_build(_task(), b"kernel void demo() {}\n")
    with pytest.raises(SchemaError, match="mutually exclusive"):
        EvalReport.from_json(_with(_base("ok"), build=_build_with(
            _base("ok"), **{**simulated, "native_absent_reason": "not_native_build"},
        )))
    no_native = copy.deepcopy(simulated)
    del no_native["native"]
    no_native["native_digest"] = None
    no_native["native_absent_reason"] = "portable_artifact_set"
    with pytest.raises(SchemaError, match="does not match build_kind"):
        EvalReport.from_json(_with(_base("ok"), build=no_native))
    with pytest.raises(SchemaError, match="must state native_absent_reason"):
        missing_reason = copy.deepcopy(no_native)
        del missing_reason["native_absent_reason"]
        EvalReport.from_json(_with(_base("ok"), build=missing_reason))


def test_a_native_digest_that_does_not_bind_the_document_is_refused():
    simulated = simulated_build(_task(), b"kernel void demo() {}\n")
    forged = copy.deepcopy(simulated)
    forged["native_digest"] = "f" * 64
    with pytest.raises(SchemaError, match="native_digest"):
        EvalReport.from_json(_with(_base("ok"), build=forged))


def test_a_received_prebuilt_full_cannot_claim_a_native_executable():
    outcome, _builder = _outcome()
    received = BuildEvidence(
        mode="prebuilt", status="reused", provenance=outcome.provenance,
        artifacts=outcome.evidence.artifacts, duration_ms=1, loaded=(),
        placement="received", producer=outcome.evidence.producer_record(),
    )
    document = project_received(received, build_kind="native", identity=_identity())
    with_native = copy.deepcopy(document)
    with_native["native"] = simulated_build(_task(), b"kernel demo")["native"]
    with_native["native_digest"] = simulated_build(_task(), b"kernel demo")["native_digest"]
    del with_native["native_absent_reason"]
    with pytest.raises(SchemaError, match="received prebuilt"):
        EvalReport.from_json(_with(_base("ok"), build=with_native))


def test_the_candidate_digest_must_bind_the_provenance_and_the_source_entry():
    document = _base("ok")
    with pytest.raises(SchemaError, match="source_sha256"):
        EvalReport.from_json(_with(document, build=_build_with(document, candidate_sha256="a" * 64)))
    other = copy.deepcopy(document)
    other["build"]["artifacts"]["entries"][0]["sha256"] = "b" * 64
    # The manifest digest no longer matches the entries, which is checked before the SOURCE link.
    with pytest.raises(SchemaError, match="does not bind the listed entries"):
        EvalReport.from_json(other)


def test_a_native_outcome_must_hold_one_live_handle():
    """native BuildOutcome 缺 live handle 或两处指向不同对象时，构造即失败。"""
    from tcenv.remote.artifact_bundle import ArtifactBundle, NativeExecutable

    outcome, _builder = _outcome()
    native = outcome.bundle.native
    entry = outcome.bundle.entries[0]

    def dead(handle):
        return NativeExecutable(
            backend_id=native.backend_id, kind=native.kind,
            entry_path=native.entry_path, entry_sha256=native.entry_sha256,
            descriptor=native.descriptor, handle=handle,
        )

    # No live handle at all: two records that are equal only because the handle is excluded.
    with pytest.raises(ValueError, match="non-empty live executable handle"):
        BuildOutcome(
            bundle=ArtifactBundle(entries=(entry,), native=dead(None)),
            provenance=outcome.provenance,
            evidence=BuildEvidence(
                mode="prebuilt", status="ok", provenance=outcome.provenance,
                artifacts=outcome.evidence.artifacts, native=dead(None),
                placement="control_host",
            ),
        )
    # Two different live objects behind the same descriptor are not one build.
    with pytest.raises(ValueError, match="not the artifact set's live executable"):
        BuildOutcome(
            bundle=ArtifactBundle(entries=(entry,), native=native),
            provenance=outcome.provenance,
            evidence=BuildEvidence(
                mode="prebuilt", status="ok", provenance=outcome.provenance,
                artifacts=outcome.evidence.artifacts, native=dead(object()),
                placement="control_host",
            ),
        )


# -- strict JSON + duplicate keys -------------------------------------------
def test_a_duplicate_key_is_refused_before_the_report_is_built():
    document = _base("ok")
    text = json.dumps(document)[:-1] + ', "build": null}'
    with pytest.raises(SchemaError, match="repeats the key 'build'"):
        read_report_json(text)
    # json.loads alone would silently keep the last value; the strict loader is the difference.
    assert json.loads(text)["build"] is None


# -- structure: the retired key and the strict readers ----------------------
def _production_trees():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for name in ("tcenv", "taic", "experiments"):
        yield from (root / name).rglob("*.py")


def _report_simulated_access(path) -> list[str]:
    """Return ``file:line`` for every ``report['simulated']`` read or write."""
    import ast

    offenders = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "report":
            key = node.slice
            if isinstance(key, ast.Constant) and key.value == "simulated":
                offenders.append(f"{path}:{node.lineno}")
        if (
            isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get" and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "report"
            and node.args and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "simulated"
        ):
            offenders.append(f"{path}:{node.lineno}")
    return offenders


def test_no_production_module_reads_or_writes_the_retired_report_key():
    """生产树里报告级的 ``simulated`` 读写为零：模拟性由 provenance 与 build_kind 承载。"""
    offenders = [hit for path in _production_trees() for hit in _report_simulated_access(path)]
    assert offenders == []


def test_no_reader_feeds_plain_json_loads_into_the_report_validator():
    """报告的读盘点必须先过拒绝重复键的 loader，不能把 ``json.loads`` 的 dict 直接交给 from_json。"""
    import ast

    offenders = []
    for path in _production_trees():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "from_json" or not node.args:
                continue
            argument = node.args[0]
            if (
                isinstance(argument, ast.Call) and isinstance(argument.func, ast.Attribute)
                and argument.func.attr == "loads"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == []


def test_the_outbox_retained_result_reader_never_parses_json_loosely():
    """§6.4：``_read_completed_result`` 必须用 ``loads_strict``，不能先 ``json.loads`` 再间接校验。

    The AST registry below only knows which files call ``EvalReport.from_json``; a reader that first
    parses the envelope with ``json.loads`` and later hands ``payload["report"]`` to ``from_json`` has
    already lost the duplicate-key information, so the registry alone cannot see the bypass.  This
    pins the one place that reads a retained report envelope.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    module = ast.parse((root / "taic/evaluation_outbox.py").read_text(encoding="utf-8"))
    reader = next(
        node for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "_read_completed_result"
    )
    calls = [
        node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for node in ast.walk(reader) if isinstance(node, ast.Call)
    ]
    assert "loads_strict" in calls
    assert "loads" not in calls  # a bare ``json.loads`` in this reader is the bypass


def test_every_registered_report_reader_uses_the_strict_loader():
    """§6.4 登记的读盘点都导入了严格 loader（重复键在构造报告前被拒）。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    readers = {
        root / "tcenv/evaluate.py": "read_report_json",
        root / "taic/resume.py": "read_report_json",
        root / "taic/selfcheck_bridge.py": "loads_strict",
        root / "taic/hwfeedback.py": "read_report_json",
        root / "experiments/hw_feedback_ab.py": "loads_strict",
        root / "tcenv/remote/job_types.py": "loads_strict",
        root / "taic/experience.py": "read_report_json",
        # 保留 result 的只读恢复（R5 §25）：整份 result 信封先经拒绝重复键的 loader，再交给
        # ``EvalReport.from_json``；``json.loads`` 会先丢掉重复键信息。
        root / "taic/evaluation_outbox.py": "loads_strict",
    }
    missing = [str(path) for path, token in readers.items() if token not in path.read_text(encoding="utf-8")]
    assert missing == []


def test_every_from_json_call_site_is_a_registered_reader():
    """§6.4：生产树中每个 ``EvalReport.from_json`` 调用点都在登记表里，新增读盘点必须先登记。"""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    registered = {
        "tcenv/evaluate.py",          # worker 回读 + 缓存读
        "tcenv/remote/job_store.py",  # evaluator 侧签名前
        "taic/orchestrator.py",       # host 侧 result 解析
        "taic/selfcheck_bridge.py",   # self-check main
        "taic/resume.py",             # _read_candidates
        "taic/hwfeedback.py",         # CLI 单报告
        "experiments/hw_feedback_ab.py",
        "taic/experience.py",         # 经验挖掘（只读 4）
        # 发布读盘边界（R5 §23）：确认报告必须过同一份 schema-4 闭集，非法文档按损坏/不可判拒绝。
        "experiments/common.py",
        # 保留 result 的只读恢复（R5 §24）：两种 dialect 的 retained report 都必须是非损坏的 schema 4。
        "taic/evaluation_outbox.py",
    }
    found = set()
    for path in _production_trees():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_json":
                found.add(str(path.relative_to(root)))
    unregistered = sorted(found - registered)
    assert unregistered == []
    # Every registered reader still exists (a renamed/moved reader must update this list).
    missing = sorted(name for name in registered if not (root / name).is_file())
    assert missing == []


# -- the helper defaults ------------------------------------------------------
def test_the_default_report_for_request_is_a_simulator_build():
    """`report_for_request` 的默认调用者是模拟服务/模拟 selfcheck：默认必须是 simulator。

    这是不依赖 socket 的 helper 级断言：vendor_sim 的 worker_a/worker_b 与模拟 selfcheck 都直接调用
    它而没有传 build_kind，所以默认值就是它们的语义。
    """
    from tests.support.report_build import report_for_request

    report = report_for_request(
        {"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "seed": 1,
         "target_id": "apple-m4-max"},
        source=b"kernel void demo() {}\n",
    )
    assert report["build"]["build_kind"] == "simulator"
    assert report["build"]["provenance"]["builder_id"] == "fake-npu-builder"
    assert report["build"]["native"]["backend_id"] == "sim-npu"
    assert "simulated" not in report

    native = report_for_request(
        {"task": "relu", "dtype": "fp16", "size": "small", "mode": "quick", "seed": 1,
         "target_id": "apple-m4-max"},
        source=b"kernel void demo() {}\n", build_kind="native",
    )
    assert native["build"]["build_kind"] == "native"
    assert native["build"]["provenance"]["builder_id"] == "metal-native-structural-fixture"
    assert native["build"]["native"]["backend_id"] == "metal"


def test_a_simulator_helper_report_cannot_claim_a_native_build():
    """模拟路由（默认）永远不是 native：不能借默认参数伪装成本地 Metal 结构正例。"""
    from tests.support.report_build import native_build, simulated_build

    simulator = simulated_build(_task(), b"kernel void demo() {}\n")
    native = native_build(_task(), b"kernel void demo() {}\n")
    assert simulator["build_kind"] == "simulator" != native["build_kind"] == "native"
    assert simulator["provenance"]["builder_id"] != native["provenance"]["builder_id"]
