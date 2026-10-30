"""The production build path is singular.

The evaluator does not compile through a backend hook: it obtains the backend's
registered ``ArtifactBuilder`` and calls ``ArtifactBuilder.build``.  Metal's only
implementation of that builder is ``MetalSourceBuilder``, and the runtime's
``MetalContext.compile`` is an implementation detail of that builder rather than
a second production entry point.  These checks are structural (they read the
production tree) plus real-Mac checks that run the real ``tcenv.evaluate`` and
compare its compile report with the legacy runtime call.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TCENV = ROOT / "tcenv"
#: The runtime primitive itself and the builder that owns it are the only files
#: allowed to call ``compile``/``compile_all`` on a context.
BUILD_PRIMITIVE_FILES = {
    TCENV / "backends" / "metal" / "runtime.py",
    TCENV / "backends" / "metal" / "source_builder.py",
}


def _production_python_files():
    return sorted(
        path for path in TCENV.rglob("*.py")
        if path not in BUILD_PRIMITIVE_FILES and "__pycache__" not in path.parts
    )


def test_production_does_not_call_a_backend_compile_hook_or_the_runtime_primitive():
    offenders: list[str] = []
    pattern = re.compile(
        r"\b(?:backend|ctx|context|metal_context|MetalContext)\s*\.\s*"
        r"(?:compile|compile_all)\s*\("
    )
    for path in _production_python_files():
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{path.relative_to(ROOT)}:{line}: {match.group(0)}")
    assert offenders == [], "\n".join(offenders)


def test_the_backend_protocol_has_no_compile_entry_point():
    from tcenv.backends.base import Backend

    source = inspect.getsource(Backend)
    assert not re.search(r"def\s+compile\s*\(", source)
    assert not re.search(r"def\s+compile_all\s*\(", source)
    assert re.search(r"def\s+artifact_builder\s*\(", source)

    from tcenv.backends.metal.backend import MetalBackend

    assert not hasattr(MetalBackend, "compile")
    assert not hasattr(MetalBackend, "compile_all")
    assert callable(MetalBackend.artifact_builder)


def test_the_evaluator_and_profiler_call_the_unified_builder():
    evaluate_source = (TCENV / "evaluate.py").read_text(encoding="utf-8")
    assert "backend.artifact_builder(" in evaluate_source
    assert "builder.build(" in evaluate_source
    assert "backend.compile(" not in evaluate_source
    assert "backend.compile_all(" not in evaluate_source

    xcworker_source = (TCENV / "xcworker.py").read_text(encoding="utf-8")
    assert "backend.artifact_builder(" in xcworker_source
    assert "ctx.compile(" not in xcworker_source
    assert "ctx.compile_all(" not in xcworker_source

    # The builder is the only Metal class that constructs a pipeline from source.
    builder_source = (TCENV / "backends" / "metal" / "source_builder.py").read_text(
        encoding="utf-8",
    )
    assert "context.compile(" in builder_source
    tree = ast.parse(builder_source)
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "MetalSourceBuilder"
        for node in tree.body
    )


@pytest.mark.gpu
def test_the_real_evaluator_builds_through_the_registered_builder(monkeypatch):
    from tasks import get_task
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.evaluate import _evaluate_in_process
    from tcenv.backends.metal.runtime import get_context

    task = get_task("relu", "fp16", "small")
    source = task.naive_msl
    calls: list[dict] = []
    real = MetalBackend.artifact_builder

    def spy(self, **kwargs):
        builder = real(self, **kwargs)
        original = builder.build

        def build(source_bundle, **build_kwargs):
            calls.append({
                "kwargs": kwargs,
                "sha256": source_bundle.sha256,
                "bytes": source_bundle.source,
                "task_contract": build_kwargs.get("task_contract"),
            })
            return original(source_bundle, **build_kwargs)

        builder.build = build
        return builder

    monkeypatch.setattr(MetalBackend, "artifact_builder", spy)
    report = _evaluate_in_process(
        "relu", "fp16", "small", source,
        mode="quick", seed=7, reps=2, measure_serial=False,
        target_id="apple-m4-max",
    )
    assert report.status == "ok", report.message
    assert len(calls) == 1
    assert calls[0]["kwargs"]["kernel_contract"] is task.contract
    assert calls[0]["task_contract"] is task.contract
    assert calls[0]["kwargs"]["kernel_names"] is None
    # candidate_sha256 is still the exact candidate text, not the assembled
    # contract header.
    import hashlib

    assert calls[0]["sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert calls[0]["bytes"] == source.encode("utf-8")

    legacy = get_context().compile(
        task.contract.assemble(source)[0], task.contract.kernel_name, math="fast",
    )
    assert report.compile["ok"] is True
    assert report.compile["language_version"] == legacy.language_version
    assert report.compile["math_mode"] == "fast"
    assert report.compile["launch"] is None
    assert report.compile["compile_s"] > 0
    assert report.compile["pipeline_s"] > 0


@pytest.mark.gpu
def test_an_invalid_kernel_reaches_compile_error_through_the_builder(monkeypatch):
    from tasks import get_task
    from tcenv.backends.metal.backend import MetalBackend
    from tcenv.backends.metal.runtime import MetalCompileError, get_context
    from tcenv.evaluate import _evaluate_in_process

    task = get_task("relu", "fp16", "small")
    broken = task.naive_msl + "\nthis is not valid metal;\n"
    calls: list[str] = []
    real = MetalBackend.artifact_builder

    def spy(self, **kwargs):
        builder = real(self, **kwargs)
        original = builder.build

        def build(source_bundle, **build_kwargs):
            calls.append(source_bundle.sha256)
            return original(source_bundle, **build_kwargs)

        builder.build = build
        return builder

    monkeypatch.setattr(MetalBackend, "artifact_builder", spy)
    report = _evaluate_in_process(
        "relu", "fp16", "small", broken,
        mode="quick", seed=7, reps=2, measure_serial=False,
        target_id="apple-m4-max",
    )
    assert report.status == "compile_error", report.message
    assert len(calls) == 1
    assert report.stage == "compile"

    # The legacy runtime call produces the same diagnostic; the report keeps the
    # pre-refactor text shape (rewritten only, without an exception-class prefix)
    # and the same classification.
    assembled, header_lines = task.contract.assemble(broken)
    with pytest.raises(MetalCompileError) as legacy:
        get_context().compile(assembled, task.contract.kernel_name, math="fast")
    expected = task.contract.rewrite_compiler_lines(str(legacy.value), header_lines)
    assert report.compile["ok"] is False
    assert report.compile["errors"] == expected
    assert "kernel.metal:" in expected
    assert not expected.startswith("MetalCompileError:")
