"""The production build path is singular: AST/call-graph, no backend import, no GPU.

The evaluator does not compile through a backend hook.  It obtains the backend's registered
``ArtifactBuilder`` from ``backend.artifact_builder`` and calls ``ArtifactBuilder.build``; the
only implementation point that reaches the Metal runtime primitive is
``MetalSourceBuilder._compile_assembled``.  Probes, calibration and the runtime's own zero-fill
kernel go through the same factory with *registered diagnostic profiles* (raw-source assemble,
no task header), and the runtime never compiles on its own.

These checks parse the production tree instead of searching it with regexes: an alias, a
``getattr`` lookup or a helper that delegates would slip past a text search, and regexes cannot
tell which function a call sits in.  The real-Mac cases (a live pipeline, a real compile error)
live in ``tests/backends/metal/test_build_path_gpu.py`` — this file is CPU-only and imports no
backend module.

CPU only.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TCENV = ROOT / "tcenv"

#: Files allowed to name the runtime primitive ``compile``/``compile_all`` at all, and the exact
#: functions inside them that may: the primitive definitions themselves, ``compile_all``'s call
#: into its own primitive, and the builder's single ``_compile_assembled`` implementation point.
PRIMITIVE_FILES = {
    "tcenv/backends/metal/runtime.py": {"compile", "compile_all"},
    "tcenv/backends/metal/source_builder.py": {"_compile_assembled"},
}
#: Names that must not be reached as a "convenience compile" anywhere in production.
FORBIDDEN_NAMES = {"compile_source", "compile_all", "compile"}


def _production_files() -> list[Path]:
    """@brief 生产树里的 Python 文件（不含测试与缓存）。

    @return 路径列表。
    """
    return sorted(
        path
        for path in TCENV.rglob("*.py")
        if "__pycache__" not in path.parts and "tests" not in path.parts
    )


def _enclosing_function(tree: ast.AST) -> dict[int, str]:
    """@brief 每个节点所在的函数/方法名（模块级为 ``<module>``）。

    @param tree 语法树。
    @return 行号 → 名字；用法是 ``node.lineno`` 查表。
    """
    owner: dict[int, str] = {}

    def visit(node: ast.AST, name: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(child):
                    owner[getattr(inner, "lineno", -1)] = child.name
                visit(child, child.name)
            else:
                owner.setdefault(getattr(child, "lineno", -1), name)
                visit(child, name)

    visit(tree, "<module>")
    return owner


#: Receivers whose ``.compile`` is regular-expression compilation, not a pipeline build.
_REGEX_RECEIVERS = {"re", "_RE", "regex"}


def _attribute_calls(tree: ast.AST) -> list[tuple[int, str, str]]:
    """@brief 收集 ``x.attr(...)`` 形式的调用（行号, 属性名, 接收者名）。

    ``re.compile(...)`` is regular-expression compilation and is not a build path; it is the only
    receiver excluded by name (the receiver has to be the ``re`` module).

    @param tree 语法树。
    @return 调用列表。
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            receiver = node.func.value
            name = receiver.id if isinstance(receiver, ast.Name) else ast.unparse(receiver)
            found.append((node.lineno, node.func.attr, name))
    return found


def test_only_the_registered_implementation_points_call_the_runtime_primitive():
    """@brief 除 ``_compile_assembled`` 与 runtime primitive 自身外，生产树没有 source→pipeline 调用。

    Aliases and delegation are covered because the check is on the parsed call graph (``x.attr``),
    not on a variable name: ``ctx.compile``, ``getattr(ctx, "compile")`` and a renamed receiver all
    appear here.
    """
    offenders: list[str] = []
    for path in _production_files():
        relative = path.relative_to(ROOT).as_posix()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        allowed = PRIMITIVE_FILES.get(relative, set())
        owner = _enclosing_function(tree)
        for lineno, attribute, receiver in _attribute_calls(tree):
            if attribute not in FORBIDDEN_NAMES:
                continue
            if attribute == "compile" and receiver in _REGEX_RECEIVERS:
                continue
            function = owner.get(lineno, "<module>")
            # ``allowed`` names the enclosing *functions* that may use the primitive: the
            # definitions themselves (``compile``/``compile_all``) and the builder's single
            # ``_compile_assembled`` point.
            if function in allowed:
                continue
            offenders.append(f"{relative}:{lineno}: {function}() calls .{attribute}(...)")
    assert offenders == [], "\n".join(offenders)


def test_no_convenience_compile_entry_point_exists():
    """@brief ``compile_source`` 这类便捷入口被删除，而不是“没人调用”。"""
    offenders: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                "compile_source",
                "compile_probe_source",
            ):
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: def {node.name}")
    assert offenders == [], "\n".join(offenders)


def test_getattr_cannot_smuggle_the_primitive_in():
    """@brief ``getattr(x, "compile")`` / ``x.__getattribute__("compile")`` 同样是旁路。"""
    offenders: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = ast.unparse(node.func)
            if name not in ("getattr", "hasattr", "operator.attrgetter"):
                continue
            if not node.args:
                continue
            literal = node.args[1] if len(node.args) > 1 else None
            if isinstance(literal, ast.Constant) and literal.value in FORBIDDEN_NAMES:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: {name}(..., {literal.value!r})")
    assert offenders == [], "\n".join(offenders)


def test_the_backend_protocol_has_no_compile_hook_but_has_the_factory():
    """@brief 协议里只有 ``artifact_builder``；``MetalBackend`` 不再暴露 compile/compile_all。

    读源码而不是 import：``tests/shared`` 不得 import ``tcenv.backends``（``test_structure.py``）。
    """
    base = ast.parse((TCENV / "backends" / "base.py").read_text(encoding="utf-8"))
    protocol_methods = {
        node.name
        for node in ast.walk(base)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "artifact_builder" in protocol_methods
    assert not {"compile", "compile_all", "compile_source"} & protocol_methods

    metal = ast.parse((TCENV / "backends" / "metal" / "backend.py").read_text(encoding="utf-8"))
    metal_methods = {
        node.name
        for node in ast.walk(metal)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert {"artifact_builder", "prepare_context"} <= metal_methods
    assert not {"compile", "compile_all", "compile_source"} & metal_methods


#: Call sites that must obtain a builder from the backend factory and then call its ``build``.
FACTORY_CALL_SITES = (
    "tcenv/evaluate.py",
    "tcenv/xcworker.py",
    "tcenv/backends/metal/probes.py",
    "tcenv/backends/metal/profile.py",
)


def _calls_by_attr(tree: ast.AST, attr: str) -> list[ast.Call]:
    """@brief 收集所有 ``<任意接收者>.attr(...)`` 调用。

    @param tree 语法树。
    @param attr 属性名。
    @return 调用节点列表。
    """
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
    ]


def test_every_production_build_call_site_goes_through_the_factory():
    """@brief 每个生产构建点都**解析出** ``artifact_builder(...)`` 与 ``.build(...)``，且顺序正确。

    这是 AST 检查，不是子串搜索：注释、文档字符串或一个叫 ``artifact_builder`` 的变量都不算数。
    直接 ``MetalSourceBuilder(...)`` 也算失败（那是绕过工厂的第二条路径），别名后再调用同样被发现。
    """
    offenders: list[str] = []
    for relative in FACTORY_CALL_SITES:
        path = ROOT / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        factories = _calls_by_attr(tree, "artifact_builder")
        builds = _calls_by_attr(tree, "build")
        if not factories:
            offenders.append(f"{relative}: no resolved backend.artifact_builder(...) call")
            continue
        if not builds:
            offenders.append(f"{relative}: no resolved builder.build(...) call")
            continue
        first_factory = min(node.lineno for node in factories)
        later_builds = [node.lineno for node in builds if node.lineno > first_factory]
        if not later_builds:
            offenders.append(f"{relative}: every .build(...) happens before the factory is asked")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "MetalSourceBuilder"
            ):
                offenders.append(f"{relative}:{node.lineno}: constructs MetalSourceBuilder directly")
        # A candidate build must hand the factory its Specialization (§5.1/§5.3-2).
        if relative in ("tcenv/evaluate.py", "tcenv/xcworker.py"):
            keywords = {
                keyword.arg
                for node in factories
                for keyword in node.keywords
            }
            if "specialization" not in keywords:
                offenders.append(f"{relative}: artifact_builder(...) does not pass a specialization")
    assert offenders == [], "\n".join(offenders)


def _alias_of_forbidden(tree: ast.AST) -> dict[str, str]:
    """@brief 记录 ``name = <expr>.compile`` / ``getattr(x, "compile")`` 这类别名。

    @param tree 语法树。
    @return 别名 → 被别名的属性名。
    """
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Attribute) and value.attr in FORBIDDEN_NAMES:
            aliases[target.id] = value.attr
        elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "getattr":
            if len(value.args) > 1 and isinstance(value.args[1], ast.Constant) and value.args[1].value in FORBIDDEN_NAMES:
                aliases[target.id] = str(value.args[1].value)
    return aliases


def _alias_offences(relative: str, text: str) -> list[str]:
    """@brief 在一段源码里找出“先别名、再调用”的编译旁路。

    @param relative 诊断用的文件路径。
    @param text 源码文本。
    @return 违规描述列表。
    """
    tree = ast.parse(text)
    aliases = _alias_of_forbidden(tree)
    if not aliases:
        return []
    allowed = PRIMITIVE_FILES.get(relative, set())
    owner = _enclosing_function(tree)
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in aliases:
            continue
        function = owner.get(node.lineno, "<module>")
        if function in allowed:
            continue
        offenders.append(
            f"{relative}:{node.lineno}: {function}() calls the alias {node.func.id!r} "
            f"of .{aliases[node.func.id]}"
        )
    return offenders


def test_a_compile_alias_is_a_bypass_and_is_reported():
    """@brief ``compile = ctx.compile; compile(...)`` 与 ``getattr`` 别名同样算旁路。

    The scan follows the assignment, so renaming the receiver or binding it first does not hide a
    second compile call.  The allow-listed functions are exempt for the same reason they are in the
    attribute test (the primitive's own definitions and the builder's single implementation point).
    """
    # Positive controls for the checker itself: the shapes a substring search would miss.
    for snippet in (
        "def f(ctx, s):\n    compile = ctx.compile\n    return compile(s, 'k')\n",
        "def f(ctx, s):\n    run = getattr(ctx, 'compile')\n    return run(s, 'k')\n",
        "def f(ctx, s):\n    c2 = ctx.compile_all\n    return c2(s, ['a', 'b'])\n",
    ):
        assert _alias_offences("tcenv/example.py", snippet), snippet

    offenders: list[str] = []
    for path in _production_files():
        relative = path.relative_to(ROOT).as_posix()
        offenders.extend(_alias_offences(relative, path.read_text(encoding="utf-8")))
    assert offenders == [], "\n".join(offenders)


def test_the_only_library_only_probe_is_the_toolchain_language_version():
    """@brief 只建临时 library、不建 pipeline 的例外只允许 ``MetalContext.language_version``。

    A temporary ``newLibraryWithSource`` is a toolchain capability probe; anywhere else it would be
    a second source→module path.  The whitelist is by exact enclosing method name: the probe itself
    and the runtime primitive ``compile`` (whose library is the candidate's own pipeline).
    """
    offenders: list[str] = []
    for path in _production_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        owner = _enclosing_function(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if not node.func.attr.startswith("newLibraryWithSource"):
                continue
            function = owner.get(node.lineno, "<module>")
            if function not in {"language_version", "compile"}:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}:{node.lineno}: {function}()")
    assert offenders == [], "\n".join(offenders)


def test_the_internal_fill_pipeline_cannot_be_smuggled_in_without_an_injection():
    """@brief runtime 的清零核没有回退：未注入就是运行时错误，不能返回裸 pipeline。"""
    import ast as _ast

    runtime = _ast.parse((TCENV / "backends" / "metal" / "runtime.py").read_text(encoding="utf-8"))
    fill = next(
        node
        for node in _ast.walk(runtime)
        if isinstance(node, _ast.FunctionDef) and node.name == "_fill_kernel"
    )
    raises = [
        node for node in _ast.walk(fill)
        if isinstance(node, _ast.Raise) and node.exc is not None
    ]
    assert raises, "the un-injected path must raise, not build a pipeline"
    text = _ast.unparse(raises[0])
    assert "MetalRuntimeError" in text
