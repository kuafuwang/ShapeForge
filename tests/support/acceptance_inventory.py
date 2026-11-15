"""Statically inventory closed acceptance packages and their live coupling.

Only file reads, regular expressions and Python AST are used.  In particular,
this module never imports or executes a closed ``verify.py`` and never starts a
subprocess.  Git metadata is read directly by :mod:`acceptance_git`.
"""

from __future__ import annotations

import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import re
import warnings

from .acceptance_git import GitObjects, tracked_paths
from .acceptance_manifest import (
    GOLDEN_SHARED,
    OLD_DOCS_PREFIX,
    manifest_paths,
    read_manifest,
    resolve_manifest_target,
)

ROOT = Path(__file__).resolve().parents[2]
SHARED = ROOT / GOLDEN_SHARED
NPU_PACKAGE = Path("docs/架构/清微NPU数据模拟")
TEST_FILE = r"tests/(?:shared|backends)/[^\s\"'<>`]+?\.py"
NODE = re.compile(
    TEST_FILE + r"::[A-Za-z_][A-Za-z0-9_]*(?:\[[^\]\n]+\])?"
)
FILE_SELECTOR = re.compile(TEST_FILE + r"(?![\w:])")
SHA256 = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
TEXT_SUFFIXES = {".py", ".json", ".md", ".sha256", ".toml", ".yaml", ".yml", ".txt"}

#: ``nodes.json`` also contains historical failure lists, prose and duplicated
#: evidence indexes.  Only these fields are actual fixed-node registrations;
#: every other node-shaped string remains a reference, not a rename/delete pin.
NODE_REGISTRY_FIELDS = {
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0": ("positive", "negative"),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation": (
        "adapter_positive", "adapter_negative", "evidence_positive", "evidence_negative",
    ),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-isolation-profile": ("node_ids",),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT0-tilelang-probe": ("node_ids",),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT1a": ("new", "negative"),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT1b": ("positive",),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT2": ("cpu_nodes", "gpu_nodes"),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT4": ("cpu_nodes", "fake_agent_gpu_nodes"),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT6": (
        "mt6_cpu", "mt6_builder_cpu", "mt6_confirmation_cpu", "mt6_resume_cpu", "real_metal_gpu",
    ),
    "tests/support/golden/shared/多DSL-Track-阶段验收/MT7": ("observe_and_structure",),
}


def _relative(path: Path, root: Path = ROOT) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


@lru_cache(maxsize=None)
def _parse_python(path: Path) -> ast.AST:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(_read_text(path))


def _disk_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        candidate for candidate in path.rglob("*")
        if candidate.is_file() and "__pycache__" not in candidate.parts
    )


def _package_paths(root: Path = ROOT) -> list[Path]:
    """Discover packages by their static artifacts, never by importing them."""
    paths: set[Path] = {path.parent for path in manifest_paths(root)}
    shared = root / GOLDEN_SHARED
    for verifier in shared.glob("**/verify.py"):
        paths.add(verifier.parent)
    npu_verify = root / NPU_PACKAGE / "verify.py"
    if npu_verify.is_file():
        paths.add(npu_verify.parent)
    # B1--B4 intentionally have no manifest or verifier.
    cross = shared / "跨硬件算子生成分工-阶段验收"
    for boundary in cross.glob("B[1-4]/boundary.json"):
        paths.add(boundary.parent)
    # The V6 re-record is evidence-only and deliberately has no verifier.
    paths.update(path for path in shared.glob("varlen-v6-chunked重录-*") if path.is_dir())
    # G0/corpus has its own manifest but is a data shard of G0, not another package.
    paths.discard(shared / "跨硬件算子生成分工-G0/corpus")
    return sorted(paths, key=lambda path: _relative(path, root))


def _verifier_for(package: Path, root: Path = ROOT) -> Path | None:
    if package.is_file():
        return package
    candidate = package / "verify.py"
    return candidate if candidate.is_file() else None


def _kind(package: Path, root: Path = ROOT) -> str:
    relative = _relative(package, root)
    if relative.startswith(GOLDEN_SHARED.as_posix() + "/v1-track/"):
        return "archived_code"
    return "verifier" if _verifier_for(package, root) is not None else "evidence_only"


@lru_cache(maxsize=None)
def _aliases(package: Path, root: Path = ROOT) -> set[str]:
    current = _relative(package, root)
    aliases = {current}
    prefix = GOLDEN_SHARED.as_posix() + "/"
    if current.startswith(prefix):
        aliases.add(OLD_DOCS_PREFIX + current[len(prefix):])
    if package.name not in {"v1-track", "verify.py"}:
        # Phase names such as P2-I and directory names such as B-evidence-...
        # are how live tests construct paths from a common acceptance root.
        aliases.add(package.name)
    if package.parent == root / GOLDEN_SHARED:
        short = package.name.rsplit("-", 1)[-1]
        if re.fullmatch(r"[A-Z]+[0-9]+[a-z]?", short):
            aliases.add(short)
    return aliases


@lru_cache(maxsize=None)
def _alias_pattern(alias: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w.-])" + re.escape(alias) + r"(?![\w-])")


def _references(text: str, package: Path, root: Path = ROOT) -> bool:
    # A stage name is a path component, not a prefix: P2 must not match P2-I,
    # and MT0 must not match MT0-isolation.  A following ``.md`` is allowed so
    # the phase's companion document remains an inbound reference.
    return any(
        alias in text and _alias_pattern(alias).search(text) is not None
        for alias in _aliases(package, root)
    )


def _top_level_calls(tree: ast.AST) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            calls.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            calls.add(node.func.attr)
    return calls


def _string_values(node: ast.AST, constants: dict[str, set[str]]) -> set[str]:
    """Expand simple static strings, tuples and f-strings without executing AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Name):
        return constants.get(node.id, set())
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "Path" and node.args:
            return _string_values(node.args[0], constants)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "resolve":
            return _string_values(node.func.value, constants)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath":
            base = _string_values(node.func.value, constants)
            for argument in node.args:
                right = _string_values(argument, constants)
                base = {a.rstrip("/") + "/" + b.lstrip("/") for a in base for b in right}
            return base
        return set()
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        return {str(Path(value).parent) for value in _string_values(node.value, constants)}
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            index = node.slice.value
            return {
                str(Path(value).parents[index])
                for value in _string_values(node.value.value, constants)
            }
        return set()
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        values: set[str] = set()
        for element in node.elts:
            values.update(_string_values(element, constants))
        return values
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Div)):
        left, right = _string_values(node.left, constants), _string_values(node.right, constants)
        if isinstance(node.op, ast.Add):
            return {a + b for a in left for b in right}
        return {a.rstrip("/") + "/" + b.lstrip("/") for a in left for b in right}
    if isinstance(node, ast.JoinedStr):
        values = {""}
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                pieces = {part.value}
            elif isinstance(part, ast.FormattedValue):
                pieces = _string_values(part.value, constants)
            else:
                return set()
            if not pieces or len(values) * len(pieces) > 256:
                return set()
            values = {prefix + piece for prefix in values for piece in pieces}
        return values
    return set()


def _static_constants(tree: ast.AST, root: Path | None = None, source_path: Path | None = None) -> dict[str, set[str]]:
    constants: dict[str, set[str]] = {"ROOT": {str(root)}, "REPO_ROOT": {str(root)}} if root is not None else {}
    if source_path is not None:
        constants["__file__"] = {str(source_path)}
    assignments = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))]
    for _ in range(3):
        for node in assignments:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if node.value is None:
                continue
            values = _string_values(node.value, constants)
            if values:
                for target in targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = values
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)) and isinstance(node.target, ast.Name):
            values = _string_values(node.iter, constants)
            if values:
                constants[node.target.id] = values
        elif isinstance(node, ast.For) and isinstance(node.target, (ast.Tuple, ast.List)) and isinstance(node.iter, (ast.Tuple, ast.List)):
            for row in node.iter.elts:
                if not isinstance(row, (ast.Tuple, ast.List)) or len(row.elts) != len(node.target.elts):
                    continue
                for target, value in zip(node.target.elts, row.elts):
                    if isinstance(target, ast.Name):
                        values = _string_values(value, constants)
                        constants.setdefault(target.id, set()).update(values)
    return constants


def _python_node_strings(tree: ast.AST) -> set[str]:
    """Extract literal and statically interpolated pytest node ids."""
    constants = _static_constants(tree)
    strings: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.add(node.value)
        elif isinstance(node, ast.JoinedStr):
            strings.update(_string_values(node, constants))
    # Handle the small one-argument helpers such as
    # ``_gpu_node_ids(BACKEND_GEOMETRY_NODES)`` without evaluating code.
    functions = {
        node.name: node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and len(node.args.args) == 1
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or len(node.args) != 1:
            continue
        function = functions.get(node.func.id)
        if function is None:
            continue
        values = _string_values(node.args[0], constants)
        if not values:
            continue
        local = dict(constants)
        local[function.args.args[0].arg] = values
        for comprehension in (node for node in ast.walk(function) if isinstance(node, ast.comprehension)):
            if isinstance(comprehension.target, ast.Name):
                expanded = _string_values(comprehension.iter, local)
                if expanded:
                    local[comprehension.target.id] = expanded
        for candidate in ast.walk(function):
            if isinstance(candidate, ast.JoinedStr):
                strings.update(_string_values(candidate, local))
    return {node for text in strings for node in NODE.findall(text)}


def _gpu_decorator(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "gpu"
    if isinstance(node, ast.Attribute):
        return node.attr == "gpu"
    return isinstance(node, ast.Call) and (
        isinstance(node.func, ast.Name) and node.func.id == "gpu"
        or isinstance(node.func, ast.Attribute) and node.func.attr == "gpu"
    )


@lru_cache(maxsize=None)
def _gpu_test_functions(filename: str) -> frozenset[str]:
    path = Path(filename)
    if not path.is_file():
        return frozenset()
    try:
        tree = _parse_python(path)
    except SyntaxError:
        return frozenset()
    module_gpu = any(
        isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(isinstance(target, ast.Name) and target.id == "pytestmark" for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        ))
        and node.value is not None
        and any(_gpu_decorator(candidate) for candidate in ast.walk(node.value))
        for node in tree.body
    )
    marked = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_gpu_decorator(decorator) for decorator in node.decorator_list)
    }
    if module_gpu:
        marked.update(
            node.name for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        )
    return frozenset(marked)


def _node_is_gpu(node: str, root: Path) -> bool:
    filename, _, remainder = node.partition("::")
    function = remainder.rsplit("::", 1)[-1].split("[", 1)[0]
    return bool(function) and function in _gpu_test_functions(str(root / filename))


def _depends(verifier: Path | None, package: Path, root: Path = ROOT) -> list[str]:
    if verifier is None:
        return []
    source = _read_text(verifier)
    try:
        tree = _parse_python(verifier)
        calls = _top_level_calls(tree)
    except SyntaxError:
        tree = ast.parse("")
        calls = set()
    depends: list[str] = []
    if "current_prompt_sha1" in calls:
        depends.append("prompt")
    if calls & {"code_import_closure", "task_pack_closure"}:
        depends.append("closure")
    if any(token in source for token in (
        "start-worktree", "acceptance-baseline", "start-state.json", "rev-parse",
        "BASE_HEAD", "base_head", "parent_head", "git_head", "git/head.txt",
    )):
        depends.append("worktree_head")
    if any(token in source for token in ("tracks.json", "load_track_registry", "registry_digest")):
        depends.append("track_registry")
    nodes = set(NODE.findall(source)) | _python_node_strings(tree)
    nodes_json = _read_text(package / "nodes.json") if package.is_dir() else ""
    nodes.update(NODE.findall(nodes_json))
    selectors = set(FILE_SELECTOR.findall(source)) | set(FILE_SELECTOR.findall(nodes_json))
    if (
        any(_node_is_gpu(node, root) for node in nodes)
        or any(_gpu_test_functions(str(root / selector)) for selector in selectors)
    ):
        depends.append("gpu_nodes")
    return depends


def _verifier_definitions(verifier: Path | None) -> tuple[set[str], set[str]]:
    """Return callable definitions and all public module-level symbols."""
    if verifier is None:
        return set(), set()
    try:
        tree = _parse_python(verifier)
    except SyntaxError:
        return set(), set()
    callables = {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    symbols = set(callables)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            symbols.update(target.id for target in targets if isinstance(target, ast.Name))
    return callables, symbols


def _is_verifier_path(value: str, verifier: Path, root: Path) -> bool:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    try:
        relative = candidate.relative_to(root / "docs/架构")
    except ValueError:
        pass
    else:
        candidate = root / GOLDEN_SHARED / relative
    return candidate == verifier.resolve()


def _loads_verifier(tree: ast.AST, verifier: Path, root: Path, source_path: Path) -> bool:
    """Bind dynamic loads to their static path argument, not nearby prose."""
    constants = _static_constants(tree, root, source_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        arguments: list[ast.AST] = []
        if name == "spec_from_file_location" and len(node.args) >= 2:
            arguments = [node.args[1]]
        elif name == "run_path" and node.args:
            arguments = [node.args[0]]
        elif name in {"_module", "_load_module"} and len(node.args) >= 2:
            arguments = [node.args[1]]
        for argument in arguments:
            if any(
                _is_verifier_path(value, verifier, root)
                for value in _string_values(argument, constants)
            ):
                return True
    return False


def _import_usage(live_file: Path, package: Path, verifier: Path | None, root: Path = ROOT) -> dict[str, object] | None:
    """Recognize dynamic verifier loading without executing either module."""
    if verifier is None:
        return None
    source = _read_text(live_file)
    if (
        not source
        or "verify.py" not in source
        or not any(marker in source for marker in ("exec_module", "run_path"))
        or not _references(source, package, root)
    ):
        return None
    try:
        tree = _parse_python(live_file)
    except SyntaxError:
        return None
    if not (_top_level_calls(tree) & {"exec_module", "run_path"}):
        return None
    if not _loads_verifier(tree, verifier, root, live_file):
        return None
    callables, symbols = _verifier_definitions(verifier)
    used = {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in symbols
    }
    used.update(
        constant.value for constant in ast.walk(tree)
        if isinstance(constant, ast.Constant) and isinstance(constant.value, str)
        and constant.value in symbols
    )
    return {
        "path": _relative(live_file, root),
        "functions": sorted(used & callables),
        "symbols": sorted(used),
    }


def _live_sources(root: Path = ROOT) -> dict[str, list[Path]]:
    tests = [root / "tests/conftest.py"]
    tests += list((root / "tests/shared").rglob("*.py"))
    tests += list((root / "tests/backends").rglob("*.py"))
    tests += [
        path for path in (root / "tests/support").rglob("*.py")
        if (root / "tests/support/golden") not in path.parents
    ]
    code: list[Path] = []
    for name in ("taic", "tcenv", "experiments", "tasks", "scripts", "backends", "observe"):
        directory = root / name
        if directory.is_dir():
            code.extend(directory.rglob("*.py"))
    docs = list((root / "docs").rglob("*.md"))
    docs += [path for path in root.glob("*.md") if path.name != "AGENTS.md"]
    return {
        "active_tests": sorted(set(tests)),
        "active_code": sorted(set(code)),
        "docs": sorted(set(docs)),
        "AGENTS.md": [root / "AGENTS.md"],
    }


def _package_texts(package: Path, root: Path = ROOT) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in _disk_files(package):
        if path.suffix not in TEXT_SUFFIXES and path.name != "manifest.sha256":
            continue
        if "corpus" in path.relative_to(package).parts and path.suffix != ".sha256":
            continue
        text = _read_text(path)
        if text:
            result[_relative(path, root)] = text
    return result


def _is_live_node(node: str, root: Path) -> bool:
    """Exclude historical prose about nodes that no longer exist."""
    filename, _, remainder = node.partition("::")
    function = remainder.rsplit("::", 1)[-1].split("[", 1)[0]
    path = root / filename
    if not function or not path.is_file():
        return False
    try:
        tree = _parse_python(path)
    except SyntaxError:
        return False
    return any(
        isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == function
        for candidate in ast.walk(tree)
    )


def _registration_texts(package: Path, verifier: Path | None, root: Path = ROOT) -> dict[str, str]:
    """Read only the verifier and its explicit ``nodes.json`` registration.

    Reviews, acceptance prose and whole-suite failure lists may mention node ids,
    but renaming one of those nodes does not invalidate the verifier's node
    registry.  Those references are reported separately from the pinned set.
    """
    if verifier is None:
        return {}
    texts = {_relative(verifier, root): _read_text(verifier)}
    nodes = package / "nodes.json"
    if package.is_dir() and nodes.is_file():
        relative = _relative(package, root)
        fields = NODE_REGISTRY_FIELDS[relative]  # fail closed for a new schema
        document = json.loads(nodes.read_text(encoding="utf-8"))
        registration = {field: document[field] for field in fields}
        texts[_relative(nodes, root)] = json.dumps(registration, ensure_ascii=False)
    return texts


def _literal_pinned_nodes(package: Path, root: Path = ROOT, texts: dict[str, str] | None = None) -> set[str]:
    texts = _package_texts(package, root) if texts is None else texts
    return {node for text in texts.values() for node in NODE.findall(text)}


def _all_pinned_nodes(package: Path, root: Path = ROOT, texts: dict[str, str] | None = None) -> set[str]:
    texts = _package_texts(package, root) if texts is None else texts
    nodes = _literal_pinned_nodes(package, root, texts)
    for name, text in texts.items():
        if name.endswith(".py"):
            try:
                nodes.update(_python_node_strings(_parse_python(root / name)))
            except SyntaxError:
                pass
    return nodes


def _history(package: Path, tracked: frozenset[str], git: GitObjects, commits: list[str], root: Path) -> dict[str, object]:
    relative = _relative(package, root)
    prefix = relative + "/"
    owned = sorted(path for path in tracked if path == relative or path.startswith(prefix))
    on_disk = _disk_files(package)
    first = last = None
    if owned:
        first, last = git.path_dates(relative, commits)
    return {
        "tracked_files": len(owned),
        "tracked_bytes": sum((root / path).stat().st_size for path in owned if (root / path).is_file()),
        "files": len(on_disk),
        "bytes": sum(path.stat().st_size for path in on_disk),
        "first_commit_date": first,
        "last_commit_date": last,
    }


def _looks_like_path(value: str) -> bool:
    return "/" in value or value.endswith((".py", ".md", ".json", ".sha256", ".toml", ".metal", ".c", ".txt"))


def _pin_rows(
    package: Path,
    packages: list[Path],
    texts: dict[str, str],
    root: Path,
) -> dict[Path, dict[str, set[str]]]:
    """Collect only digest records explicitly associated with a target path."""
    rows: dict[Path, dict[str, set[str]]] = {}
    resolved_packages = [(other, other.resolve()) for other in packages]

    def register(source: Path, target_name: str, digest: str | None) -> None:
        if digest is not None and not SHA256.fullmatch(digest) or not _looks_like_path(target_name):
            return
        target, _ = resolve_manifest_target(source, target_name, root)
        if not target.is_relative_to(root):
            return
        candidates = [
            other for other, resolved in resolved_packages if other != package
            and (target == resolved or target.is_relative_to(resolved))
        ]
        if not candidates:
            return
        other = max(candidates, key=lambda candidate: len(candidate.parts))
        rows.setdefault(other, {}).setdefault(_relative(source, root), set()).add(_relative(target, root))

    def walk_json(value: object, source: Path, context: str = "") -> None:
        if isinstance(value, dict):
            history_counts = value.get("history_manifests")
            history_hashes = value.get("history_manifest_sha256")
            if isinstance(history_counts, dict) or isinstance(history_hashes, dict):
                phases = set(history_counts if isinstance(history_counts, dict) else {})
                phases.update(history_hashes if isinstance(history_hashes, dict) else {})
                for phase in phases:
                    if not isinstance(phase, str):
                        continue
                    digest = history_hashes.get(phase) if isinstance(history_hashes, dict) else None
                    register(source, str(source.parent.parent / phase / "manifest.sha256"), digest)
            local_context = context
            for key in ("package", "old_package"):
                candidate = value.get(key)
                if isinstance(candidate, str) and _looks_like_path(candidate):
                    local_context = candidate
            for key, item in value.items():
                if not isinstance(key, str):
                    continue
                if isinstance(item, str) and SHA256.fullmatch(item):
                    if _looks_like_path(key):
                        target = key
                        if local_context and not key.startswith(("docs/", "tests/", "tcenv/", "taic/", "backends/", "targets/", "experiments/")):
                            target = local_context.rstrip("/") + "/" + key
                        register(source, target, item)
                    for path_key in ("path", "relative", "record", "manifest", "file", "document", "verifier"):
                        target = value.get(path_key)
                        if isinstance(target, str):
                            register(source, target, item)
                    if key.endswith("_sha256"):
                        prefix = key[:-len("_sha256")]
                        for path_key in (prefix + "_path", prefix, prefix + "_relative"):
                            target = value.get(path_key)
                            if isinstance(target, str):
                                register(source, target, item)
                elif isinstance(item, (dict, list)):
                    child_context = local_context
                    if key in {"files", "documents", "old_package_unchanged", "observations"}:
                        child_context = local_context
                    walk_json(item, source, child_context)
        elif isinstance(value, list):
            for item in value:
                walk_json(item, source, context)

    for name, text in texts.items():
        source = root / name
        if name.endswith(".sha256"):
            try:
                entries = list(read_manifest(source, root))
            except (ValueError, OSError):
                entries = []
            for entry in entries:
                register(source, entry.name, entry.expected_sha256)
        elif name.endswith(".json"):
            try:
                walk_json(json.loads(text), source)
            except json.JSONDecodeError:
                pass
        elif name.endswith(".py"):
            try:
                tree = _parse_python(source)
            except SyntaxError:
                continue
            constants = _static_constants(tree)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.endswith("/manifest.sha256"):
                        register(source, node.value, None)
                elif isinstance(node, ast.JoinedStr):
                    for target in _string_values(node, constants):
                        if target.endswith("/manifest.sha256"):
                            register(source, target, None)
                if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict):
                    try:
                        walk_json(ast.literal_eval(node.value), source)
                    except (ValueError, TypeError, SyntaxError):
                        continue
    return rows


def build_inventory(root: Path = ROOT) -> dict[str, object]:
    """Return a JSON-serializable static inventory of every package."""
    packages = _package_paths(root)
    kinds = {package: _kind(package, root) for package in packages}
    verifiers = {package: _verifier_for(package, root) for package in packages}
    package_texts = {package: _package_texts(package, root) for package in packages}
    live_sources = _live_sources(root)
    live_texts = {
        category: {path: _read_text(path) for path in paths}
        for category, paths in live_sources.items()
    }
    tracked = tracked_paths(root)
    git = GitObjects(root)
    commits = git.reachable_commits()
    records: list[dict[str, object]] = []
    for package in packages:
        relative = _relative(package, root)
        verifier = verifiers[package]
        manifest = package / "manifest.sha256" if package.is_dir() else package.parent / "manifest.sha256"
        has_manifest = manifest.is_file()
        manifest_lists_verifier = False
        if has_manifest and verifier is not None:
            manifest_lists_verifier = any(
                entry.target.resolve() == verifier.resolve()
                for entry in read_manifest(manifest, root)
            )
        inbound: dict[str, object] = {}
        for category, sources in live_texts.items():
            inbound[category] = sorted(
                _relative(path, root) for path, text in sources.items()
                if path.is_file() and _references(text, package, root)
            )
        closed_inbound: list[dict[str, object]] = []
        for other in packages:
            if other == package:
                continue
            files = sorted(
                name for name, text in package_texts[other].items()
                if _references(text, package, root)
            )
            if files:
                closed_inbound.append({"package": _relative(other, root), "files": files})
        inbound["closed_packages"] = closed_inbound
        imported_by_live: list[dict[str, object]] = []
        for category in ("active_tests", "active_code"):
            for path in live_sources[category]:
                usage = _import_usage(path, package, verifier, root)
                if usage is not None:
                    imported_by_live.append(usage)
        pin_rows = _pin_rows(package, packages, package_texts[package], root)
        pins = [
            {
                "package": _relative(other, root),
                "files": sorted(sources),
                "pinned_paths": sorted({path for paths in sources.values() for path in paths}),
            }
            for other, sources in sorted(pin_rows.items(), key=lambda item: _relative(item[0], root))
        ]
        registration_texts = _registration_texts(package, verifier, root)
        registered_nodes = _all_pinned_nodes(package, root, registration_texts)
        registered_literal = _literal_pinned_nodes(package, root, registration_texts)
        referenced_nodes = _all_pinned_nodes(package, root, package_texts[package])
        live_nodes = sorted(node for node in registered_nodes if _is_live_node(node, root))
        referenced_live_nodes = sorted(node for node in referenced_nodes if _is_live_node(node, root))
        record = {
            "path": relative,
            "kind": kinds[package],
            "has_manifest": has_manifest,
            "manifest_lists_verifier": manifest_lists_verifier,
            "depends": _depends(verifier, package, root),
            "pins_other_packages": pins,
            "inbound": inbound,
            "imported_by_live": sorted(imported_by_live, key=lambda item: item["path"]),
            "pinned_live_nodes": live_nodes,
            "literal_pinned_live_nodes": sorted(set(live_nodes) & registered_literal),
            "derived_live_nodes": sorted(set(live_nodes) - registered_literal),
            "referenced_live_nodes": referenced_live_nodes,
            "inactive_node_references": sorted(referenced_nodes - set(referenced_live_nodes)),
        }
        record.update(_history(package, tracked, git, commits, root))
        records.append(record)
    return {
        "schema_version": 1,
        "packages": records,
        "summary": {
            "packages": len(records),
            "verifiers": sum(record["kind"] == "verifier" for record in records),
            "evidence_only": sum(record["kind"] == "evidence_only" for record in records),
            "archived_code": sum(record["kind"] == "archived_code" for record in records),
            "manifests": len(manifest_paths(root)),
            "pinned_live_nodes": len({node for record in records for node in record["pinned_live_nodes"]}),
            "referenced_live_nodes": len({
                node for record in records for node in record["referenced_live_nodes"]
            }),
            "literal_pinned_live_nodes": len({
                node for record in records for node in record["literal_pinned_live_nodes"]
            }),
            "derived_only_pinned_live_nodes": len({
                node for record in records for node in record["pinned_live_nodes"]
            } - {
                node for record in records for node in record["literal_pinned_live_nodes"]
            }),
            "inactive_node_references": len({
                node for record in records for node in record["inactive_node_references"]
            }),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit the complete inventory as JSON")
    args = parser.parse_args()
    document = build_inventory(ROOT)
    if args.json:
        print(json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for record in document["packages"]:
        print(
            f"{record['path']}: {record['kind']}; depends={','.join(record['depends']) or '-'}; "
            f"live-imports={len(record['imported_by_live'])}; nodes={len(record['pinned_live_nodes'])}; "
            f"files={record['files']}; bytes={record['bytes']}"
        )
    print(json.dumps(document["summary"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
