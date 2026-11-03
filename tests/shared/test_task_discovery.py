"""v2 task discovery, locator fingerprint and staged-pack rebuild (A3, CPU)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tcenv.taskcontract import (
    AssemblyError,
    BaselinePackage,
    BaselineRegistration,
    BaselineRegistry,
    PackClosureError,
    TaskDiscoveryError,
    discover_standard_tasks,
    task_pack_closure,
)
from tcenv.taskcontract.pack import code_import_closure
from tcenv.track import REPO_ROOT

from ..support.v2_tasks import rmsnorm_opspec, write_task


def _opspec(op_name="rmsnorm"):
    return rmsnorm_opspec(op_name)


def test_discovery_finds_v2_directories_and_skips_v1(tmp_path):
    """@brief 自动发现只认 v2 标准目录；v1 目录不进 v2 路径。"""
    root = tmp_path / "tasks"
    write_task(root)
    legacy = root / "legacy"
    legacy.mkdir()
    (legacy / "opspec.json").write_text('{"spec_version": 1}', encoding="utf-8")
    (legacy / "track.json").write_text("{}", encoding="utf-8")
    found = discover_standard_tasks(root, code_root=REPO_ROOT)
    assert [locator.task_id for locator in found] == ["rmsnorm"]
    assert found[0].op_name == "rmsnorm"
    assert found[0].dtype == "fp16" and found[0].size == "small"
    assert found[0].identity == found[0].identity
    assert found[0].task_dir == root / "rmsnorm"


def test_discovery_refuses_a_second_task_for_one_op_with_a_different_pack(tmp_path):
    """@brief 同一 op 的两个不同 pack 是两条权威，拒绝。"""
    root = tmp_path / "tasks"
    write_task(root / "a", op_name="same", opspec=_opspec("same"))
    write_task(
        root / "b",
        op_name="same",
        opspec=_opspec("same"),
        reference="import torch\n\ndef reference(inputs, attrs):\n    return {'y': inputs['x']}\n",
    )
    with pytest.raises(TaskDiscoveryError, match="two tasks with different packs"):
        discover_standard_tasks(root, code_root=REPO_ROOT)


def test_a_staged_pack_rebuilds_on_a_worker_without_the_control_host_path(tmp_path):
    """@brief stage → 另一个目录重建：locator 指纹一致，篡改即拒绝。"""
    root = tmp_path / "control" / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    staged = locator.stage()
    assert staged.locator == locator.document()
    rebuilt = staged.rebuild(tmp_path / "worker", code_root=REPO_ROOT)
    assert rebuilt.document() == locator.document()
    assert rebuilt.identity == locator.identity
    assert rebuilt.task_dir == tmp_path / "worker" / "rmsnorm"
    assert rebuilt.load().pack_files == locator.load().pack_files

    tampered_files = tuple(
        (name, data + b"\n# drift\n") if name == "reference.py" else (name, data) for name, data in staged.files
    )
    from tcenv.taskcontract import StagedTaskPack

    with pytest.raises(TaskDiscoveryError):
        StagedTaskPack(locator=staged.locator, files=tampered_files).rebuild(
            tmp_path / "worker2", code_root=REPO_ROOT
        )
    with pytest.raises(TaskDiscoveryError, match="non-empty"):
        staged.rebuild(tmp_path / "worker", code_root=REPO_ROOT)


def test_the_pack_closure_is_static_deterministic_and_hardware_neutral(tmp_path):
    """@brief pack 闭包由 import 关系派生；排除 backend/taic/tests，知识文件不进入摘要。"""
    files = code_import_closure(REPO_ROOT)
    assert "tcenv/taskcontract/reference.py" in files
    assert "tcenv/opspec/v2.py" in files
    assert "tcenv/opspec/schema/opspec.v2.schema.json" in files
    assert "tcenv/taskcontract/schema/baselines.v1.schema.json" in files
    assert not any("backends" in name or name.startswith(("taic", "tests", "tasks")) for name in files)
    assert files == code_import_closure(REPO_ROOT)

    task_dir = write_task(tmp_path / "rmsnorm")
    first = task_pack_closure(task_dir, root=REPO_ROOT)
    assert first.task_files == ("opspec.json", "reference.py")
    (task_dir / "guidance_catalog.md").write_text("wiki\n", encoding="utf-8")
    (task_dir / "skills").mkdir()
    (task_dir / "skills" / "author.md").write_text("hint\n", encoding="utf-8")
    second = task_pack_closure(task_dir, root=REPO_ROOT)
    assert second.task_files == first.task_files and second.digest == first.digest
    (task_dir / "reference.py").write_text((task_dir / "reference.py").read_text() + "\n# drift\n", encoding="utf-8")
    assert task_pack_closure(task_dir, root=REPO_ROOT).digest != first.digest


def test_the_baseline_registry_has_one_authority_per_operator_and_name():
    """@brief 不同算子可复用 baseline 名；同一 (op,name) 的身份/runner 源不同即拒绝。"""
    registry = BaselineRegistry()
    registration = BaselineRegistration(
        name="vendor", framework="torch", implementation_id="vendor.baseline", implementation_version=1, params={"dtype": "fp16"}
    )
    package = BaselinePackage(registrations=(registration,), runners={"vendor": lambda *args: args}, source_sha256="a" * 64)
    registry.register("rmsnorm", package)
    registry.register("softmax", package)
    assert registry.names("rmsnorm") == ("vendor",)
    assert registry.get("rmsnorm", "vendor") == registration.document()
    registry.register("rmsnorm", package)  # identical registration is idempotent
    other_runner = BaselinePackage(
        registrations=(registration,), runners={"vendor": lambda *args: args}, source_sha256="b" * 64
    )
    with pytest.raises(AssemblyError, match="different identity/runner source"):
        registry.register("rmsnorm", other_runner)
    other_identity = BaselinePackage(
        registrations=(
            BaselineRegistration(
                name="vendor", framework="torch", implementation_id="vendor.v2", implementation_version=2, params={}
            ),
        ),
        runners={"vendor": lambda *args: args},
        source_sha256="a" * 64,
    )
    with pytest.raises(AssemblyError, match="different identity"):
        registry.register("rmsnorm", other_identity)
    assert len(registry.document()["baselines"]) == 2


def test_staged_pack_refuses_absolute_escape_paths_before_writing_anything(tmp_path):
    """@brief staged 文件名/task_id 只能是规范化相对 POSIX 路径；外部路径绝不产生。"""
    from tcenv.taskcontract import StagedTaskPack, TaskLocator

    root = tmp_path / "control" / "tasks"
    write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    staged = locator.stage()
    outside = tmp_path / "outside.txt"
    with pytest.raises(TaskDiscoveryError, match="absolute"):
        StagedTaskPack(locator=staged.locator, files=staged.files + ((str(outside), b"escape"),))
    assert not outside.exists()
    with pytest.raises(TaskDiscoveryError, match="absolute"):
        StagedTaskPack(locator={**staged.locator, "task_id": str(tmp_path / "escape")}, files=staged.files)
    with pytest.raises(TaskDiscoveryError, match="normalized|backslash"):
        StagedTaskPack(locator=staged.locator, files=staged.files + (("../escape.txt", b"escape"),))
    with pytest.raises(TaskDiscoveryError, match="normalized|backslash"):
        StagedTaskPack(locator=staged.locator, files=staged.files + (("host\\escape.txt", b"escape"),))
    with pytest.raises(TaskDiscoveryError, match="absolute"):
        TaskLocator(
            task_id="/absolute",
            op_name=locator.op_name,
            dtype=locator.dtype,
            size=locator.size,
            pack_digest=locator.pack_digest,
            deployment_pack_digest=locator.deployment_pack_digest,
            task_dir=locator.task_dir,
        )
    assert not (tmp_path / "escape").exists()


def test_load_refuses_a_task_file_or_code_closure_that_drifted_after_discovery(tmp_path):
    """@brief locator.load 必须重算并比对完整身份，不能拿旧 digest 评测新字节。"""
    root = tmp_path / "tasks"
    task_dir = write_task(root)
    locator = discover_standard_tasks(root, code_root=REPO_ROOT)[0]
    with open(task_dir / "reference.py", "a", encoding="utf-8") as handle:
        handle.write("\n# drifted after discovery\n")
    with pytest.raises(TaskDiscoveryError, match="drifted since discovery"):
        locator.load()

    fake_code = tmp_path / "fake-code"
    (fake_code / "tcenv" / "taskcontract").mkdir(parents=True)
    (fake_code / "tcenv" / "opspec").mkdir(parents=True)
    (fake_code / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
    (fake_code / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
    code_locator = discover_standard_tasks(root, code_root=fake_code)[0]
    (fake_code / "tcenv" / "opspec" / "__init__.py").write_text("# code drift\n", encoding="utf-8")
    with pytest.raises(TaskDiscoveryError, match="drifted since discovery"):
        code_locator.load()


def test_the_pack_closure_runs_in_a_worker_without_the_control_host_repo(tmp_path):
    """@brief 只部署闭包文件到独立根即可发现任务；缺 schema 资源则失败，证明资源在闭包内。"""
    worker = tmp_path / "worker"
    write_task(worker / "tasks")
    for relative in code_import_closure(REPO_ROOT):
        source = REPO_ROOT / relative
        target = worker / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    probe = (
        "import json; from tcenv.taskcontract import discover_standard_tasks; "
        "print(json.dumps([item.identity for item in discover_standard_tasks('tasks')]))"
    )
    env = {**os.environ, "PYTHONPATH": str(worker)}
    first = subprocess.run([sys.executable, "-c", probe], cwd=worker, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    identities = json.loads(first.stdout.strip().splitlines()[-1])
    assert len(identities) == 1 and len(identities[0]) == 64

    # A symlinked schema is followed by the loader but must not be silently omitted from the pack.
    schema = worker / "tcenv/opspec/schema/opspec.v2.schema.json"
    external = tmp_path / "outside-schema.json"
    external.write_text('{"type": "object"}\n', encoding="utf-8")
    schema.unlink()
    schema.symlink_to(external)
    symlinked = subprocess.run([sys.executable, "-c", probe], cwd=worker, env=env, capture_output=True, text=True)
    assert symlinked.returncode != 0
    assert "symlink" in symlinked.stderr

    # The schema is not named by an import; remove it from the deployed closure and discovery must
    # fail at load time.  This is the exact failure the hand-written/test-only closure missed.
    schema.unlink()
    second = subprocess.run([sys.executable, "-c", probe], cwd=worker, env=env, capture_output=True, text=True)
    assert second.returncode != 0
    assert "opspec.v2.schema.json" in second.stderr or "No such file" in second.stderr


def test_a_runtime_schema_change_moves_the_deployment_pack_digest(tmp_path):
    """@brief schema 资源也在同一闭包摘要中；改 schema 必须改 deployment digest。"""
    task_dir = write_task(tmp_path / "task")
    fake = tmp_path / "code"
    (fake / "tcenv" / "taskcontract").mkdir(parents=True)
    (fake / "tcenv" / "opspec" / "schema").mkdir(parents=True)
    (fake / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
    schema = fake / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json"
    schema.write_text('{"type": "object"}\n', encoding="utf-8")
    first = task_pack_closure(task_dir, root=fake)
    assert "tcenv/opspec/schema/opspec.v2.schema.json" in first.code_files
    schema.write_text('{"type": "object", "required": []}\n', encoding="utf-8")
    assert task_pack_closure(task_dir, root=fake).digest != first.digest


def test_a_symlinked_runtime_schema_is_refused_by_closure_discovery_and_load(tmp_path):
    """@brief 运行时 schema 的 symlink 必须在闭包/发现/load 三层都拒绝，不能静默省略。"""
    tasks = tmp_path / "tasks"
    write_task(tasks)
    fake = tmp_path / "code"
    (fake / "tcenv" / "taskcontract").mkdir(parents=True)
    (fake / "tcenv" / "opspec" / "schema").mkdir(parents=True)
    (fake / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
    schema = fake / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json"
    schema.write_text('{"type": "object"}\n', encoding="utf-8")
    assert "tcenv/opspec/schema/opspec.v2.schema.json" in code_import_closure(fake)
    locator = discover_standard_tasks(tasks, code_root=fake)[0]

    external = tmp_path / "external-schema.json"
    external.write_text('{"type": "object"}\n', encoding="utf-8")
    schema.unlink()
    schema.symlink_to(external)
    with pytest.raises(PackClosureError, match="symlink"):
        code_import_closure(fake)
    with pytest.raises(PackClosureError, match="symlink"):
        task_pack_closure(tasks / "rmsnorm", root=fake)
    with pytest.raises(TaskDiscoveryError, match="task pack closure failed"):
        discover_standard_tasks(tasks, code_root=fake)
    with pytest.raises(TaskDiscoveryError, match="task pack closure failed"):
        locator.load()

    # The loader follows the link; changing the external schema after discovery must still be a
    # refusal, never ACCEPTED_DRIFT.
    external.write_text('{"type": "object", "changed": true}\n', encoding="utf-8")
    with pytest.raises(TaskDiscoveryError):
        locator.load()


def test_a_symlinked_schema_directory_is_refused(tmp_path):
    """@brief schema 目录本身是 symlink 时也不得静默省略。"""
    fake = tmp_path / "code"
    (fake / "tcenv" / "taskcontract").mkdir(parents=True)
    (fake / "tcenv" / "opspec").mkdir(parents=True)
    (fake / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside-schema-dir"
    outside.mkdir()
    (outside / "opspec.v2.schema.json").write_text('{"type": "object"}\n', encoding="utf-8")
    (fake / "tcenv" / "opspec" / "schema").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PackClosureError, match="schema directory"):
        code_import_closure(fake)


def test_a_nested_symlinked_schema_directory_is_refused(tmp_path):
    """@brief schema 下嵌套目录 symlink 不能被 rglob 静默跳过；closure/discovery/load 都拒绝。"""
    tasks = tmp_path / "tasks"
    write_task(tasks)
    fake = tmp_path / "code"
    (fake / "tcenv" / "taskcontract").mkdir(parents=True)
    (fake / "tcenv" / "opspec" / "schema").mkdir(parents=True)
    (fake / "tcenv" / "taskcontract" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "tcenv" / "opspec" / "__init__.py").write_text("", encoding="utf-8")
    (fake / "tcenv" / "opspec" / "schema" / "opspec.v2.schema.json").write_text('{"type": "object"}\n', encoding="utf-8")
    locator = discover_standard_tasks(tasks, code_root=fake)[0]

    outside = tmp_path / "outside-schema-tree"
    outside.mkdir()
    extra = outside / "extra.json"
    extra.write_text('{"extra": 1}\n', encoding="utf-8")
    (fake / "tcenv" / "opspec" / "schema" / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PackClosureError, match="symlinked directory"):
        code_import_closure(fake)
    with pytest.raises(PackClosureError, match="symlinked directory"):
        task_pack_closure(tasks / "rmsnorm", root=fake)
    with pytest.raises(TaskDiscoveryError, match="task pack closure failed"):
        discover_standard_tasks(tasks, code_root=fake)
    with pytest.raises(TaskDiscoveryError, match="task pack closure failed"):
        locator.load()
    extra.write_text('{"extra": 2}\n', encoding="utf-8")
    with pytest.raises(TaskDiscoveryError):
        locator.load()
