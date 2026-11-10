"""CPU tests for the shape-portfolio search driver.  Runners and evaluation are injected."""

from __future__ import annotations

import ast
import inspect
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import taic.portfolio_prompt as portfolio_prompt
from taic.kernel_agent import prepare_session, render_contract
from taic.kernel_agent import SessionBrief
from taic.portfolio_prompt import render_assemble_task
from taic.portfolio_search import run_portfolio_search
from taic.trust import prompt_fingerprint_parts
from taic.cli import main
from tcenv.portfolio_assembly import benchmark_sizes, domain_for_sizes, encode_parent, parent_entries_from_tree
from tcenv.target import load_target
from tcenv.track.envelope import encode_envelope
from tcenv.track.plane import CandidateArtifact
from tests.support.kda_fixed_bundle import bundle_entries
from tests.support.portfolio_assembly import PAPER_GUARD, SMALL_GUARD, gemm_variant_source
from taic.task_binding import bind_task

_TARGET = "apple-m4-max"
_REFUSE = {"kind": "refuse", "reasons": ["outside_candidate_coverage"]}
_STATE_KEYS = (
    "schema_id", "task", "dtype", "target_id", "track", "sizes", "run_seed", "shards", "assemble",
    "evaluation_status", "performance", "confirmation", "spent_usd", "stop_reason", "publication_allowed",
)


@lru_cache(maxsize=None)
def _domain(task: str):
    """@brief 绑定一次题目并造出全部 benchmark 档的域。

    @param task 题名。
    @return ``(contract, domain)``。
    """
    contract = bind_task(task, "fp16", "paper", target_id=_TARGET).locator.load()
    sizes = benchmark_sizes(contract.spec, "fp16")
    return contract, domain_for_sizes(contract, load_target(_TARGET), sizes)


def _result(size: str, speedup: float = 1.5) -> dict:
    """@brief 一份分片 ``result.json``。

    @param size 档名。
    @param speedup 搜索期加速比。
    @return 文档。
    """
    return {
        "best_label": size,
        "best_speedup": speedup,
        "confirmed_speedup": speedup,
        "spent_usd": 0.1,
    }


def _write_gemm_shard(directory: Path, contract, size: str, *, best: bool = True) -> None:
    """@brief 在分片目录里写下结果，可选写下 ``best.metal``。

    @param directory 分片目录。
    @param contract gemm 合同。
    @param size 档名。
    @param best 是否写核。
    """
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "result.json").write_text(json.dumps(_result(size)) + "\n", encoding="utf-8")
    if best:
        (directory / "best.metal").write_text(gemm_variant_source(contract, size), encoding="utf-8")


def _coverage(root: Path, variants: list[dict]) -> None:
    """@brief 写 ``coverage.json``。

    @param root ``src`` 目录。
    @param variants variant 条目。
    """
    document = {
        "schema_id": "tcenv.portfolio-assembly.coverage.1",
        "planner": {"id": "assemble", "version": 1},
        "variants": variants,
        "fallback": _REFUSE,
    }
    (root / "coverage.json").write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def _good_variants() -> list[dict]:
    """@brief gemm 两档各接各的 guard。

    @return variant 条目。
    """
    return [
        {"variant_key": "small", "guard": SMALL_GUARD},
        {"variant_key": "paper", "guard": PAPER_GUARD},
    ]


def _pack(work: Path, domain) -> str:
    """@brief 用会话目录里的 ``src`` 打出父包文本。

    @param work 会话目录。
    @param domain 组装域。
    @return envelope 文本。
    """
    return encode_parent(parent_entries_from_tree(work / "src", domain))


def _view(source, *, cost_incomplete: bool = False, cost: float = 0.01):
    """@brief 假会话的返回值。

    @param source 交卷文本；没有交卷时为 None。
    @param cost_incomplete 花费是否不完整。
    @param cost 美元。
    @return 简单对象。
    """
    return SimpleNamespace(
        source=source, cost_usd=None if cost_incomplete else cost, cost_incomplete=cost_incomplete,
        spent={"output": 1}, provider="fake", model="fake",
    )


def _confirm_ok(**kwargs):
    """@brief 每个点都返回 confirmed，避免缺省实现去跑真机。

    @param kwargs 与 ``confirm_subject`` 相同的关键字。
    @return 确认文档。
    """
    return {
        "status": "confirmed",
        "reason": "injected",
        "coverage_kind": "tune_confirmation",
        "instance_key": kwargs["request"].instance_key,
    }


def _evaluation(status: str, problems: tuple[str, ...] = ()):
    """@brief 假评测结果。

    @param status 状态。
    @param problems 问题。
    @return 简单对象。
    """
    document = {"status": status, "problems": list(problems), "publication_allowed": False}
    return SimpleNamespace(
        status=status, problems=problems, performance={"publishable": status == "ok"}, document=lambda: document,
    )


def _run(tmp_path: Path, **kwargs):
    """@brief 用临时目录跑驱动。缺省预算很小，避免和真实预设搅在一起。

    @param tmp_path 测试目录。
    @param kwargs 传给 ``run_portfolio_search`` 的其余参数。
    @return 退出码。
    """
    kwargs.setdefault("budget", 1.0)
    kwargs.setdefault("assemble_budget", 1.0)
    kwargs.setdefault("seed", 11)
    return run_portfolio_search("gemm", "fp16", run_dir=tmp_path / "run", **kwargs)


def _state(tmp_path: Path) -> dict:
    """@brief 读状态文件。

    @param tmp_path 测试目录。
    @return 文档。
    """
    return json.loads((tmp_path / "run" / "portfolio_search.json").read_text(encoding="utf-8"))


def test_a_quiet_portfolio_is_packed_after_one_assemble(tmp_path):
    """@brief 两个分片、一次组装、评测 ok：父包在，状态键齐全，没有 portfolio_run.json。"""
    contract, domain = _domain("gemm")

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *args, **kwargs: _evaluation("ok"), confirm=_confirm_ok,
    )
    assert code == 0
    run = tmp_path / "run"
    assert (run / "portfolio" / "parent.envelope").is_file()
    assert not (run / "portfolio_run.json").exists()
    document = _state(tmp_path)
    assert set(_STATE_KEYS) <= set(document)
    assert document["publication_allowed"] is False
    assert document["evaluation_status"] == "ok"
    assert document["stop_reason"] is None
    assert document["schema_id"] == "taic.portfolio-search.result.1"


def test_one_benchmark_size_is_refused(tmp_path):
    """@brief 只给一个带 benchmark 的 size 时启动被拒。"""
    calls = []
    code = _run(tmp_path, sizes=("small",), shard_runner=lambda size, shard_dir: calls.append(size))
    assert code == 5
    assert calls == []


def test_bundle_track_on_gemm_is_refused_for_missing_workspace(capsys, tmp_path):
    """@brief gemm 没有 workspace，不能用 msl-bundle。"""
    code = _run(tmp_path, track="msl-bundle")
    assert code == 5
    assert "workspace" in capsys.readouterr().err


def test_nominal_cost_above_the_cap_does_not_start_shards(tmp_path):
    """@brief 名义总花费超过上限时拒绝启动，分片函数没被调用。"""
    calls = []
    code = _run(
        tmp_path, budget=10.0, assemble_budget=10.0, max_total_usd=1.0,
        shard_runner=lambda size, shard_dir: calls.append(size),
    )
    assert code == 5
    assert calls == []


def test_a_shard_without_a_kernel_stops_before_assemble(tmp_path):
    """@brief 分片没有 best.metal 时停在 shard_failed，不开组装会话。"""
    contract, _domain_gemm = _domain("gemm")
    sessions = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size, best=size != "paper")

    code = _run(tmp_path, shard_runner=shard_runner, session_runner=lambda **kwargs: sessions.append(kwargs))
    assert code == 1
    assert _state(tmp_path)["stop_reason"].startswith("shard_failed:")
    assert sessions == []


def test_an_existing_result_is_not_searched_again(tmp_path):
    """@brief 已经有 result.json 的分片不再调用 shard_runner。"""
    contract, domain = _domain("gemm")
    run = tmp_path / "run"
    _write_gemm_shard(run / "shards" / "paper", contract, "paper")
    calls = []

    def shard_runner(size, shard_dir):
        calls.append(size)
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("ok"), confirm=_confirm_ok,
    )
    assert code == 0
    assert calls == ["small"]


def test_an_overlapping_first_submission_is_rejected_and_the_retry_sees_why(tmp_path):
    """@brief 第一次 guard 重叠时不评测；第二次的 TASK.md 带上原因，并通过。"""
    contract, domain = _domain("gemm")
    texts = []
    evaluations = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        texts.append(task_text)
        if attempt == 1:
            _coverage(work / "src", [
                {"variant_key": "small", "guard": ["ge", "M", 1]},
                {"variant_key": "paper", "guard": ["ge", "M", 1]},
            ])
        else:
            _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def evaluate(*args, **kwargs):
        evaluations.append(kwargs["mode"])
        return _evaluation("ok")

    code = _run(tmp_path, shard_runner=shard_runner, session_runner=session_runner, evaluate=evaluate, confirm=_confirm_ok)
    assert code == 0
    assert evaluations == ["timed"]
    assert "两个以上变体同时命中" in texts[1]


def test_a_changed_variant_byte_is_rejected_by_staged_problems(tmp_path):
    """@brief 会话改了变体里的一个字节，staged_problems 拒绝，不评测。"""
    contract, domain = _domain("gemm")
    evaluations = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        path = work / "src" / "variants" / "small" / "kernel.metal"
        original = bytearray(path.read_bytes())
        original[-1] ^= 1
        path.write_bytes(bytes(original))
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, assemble_attempts=1, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: evaluations.append(1),
    )
    assert code == 1
    assert evaluations == []
    assert _state(tmp_path)["stop_reason"] == "assemble_failed"
    assert "不一致" in _state(tmp_path)["assemble"][0]["rejection"]


def test_an_extra_variant_is_rejected(tmp_path):
    """@brief 会话多加一个没摆过的变体时被拒。"""
    contract, domain = _domain("gemm")
    evaluations = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        extra = work / "src" / "variants" / "extra"
        extra.mkdir()
        (extra / "kernel.metal").write_text((work / "src" / "variants" / "small" / "kernel.metal").read_text(), encoding="utf-8")
        _coverage(work / "src", _good_variants() + [{"variant_key": "extra", "guard": ["eq", "M", 3]}])
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, assemble_attempts=1, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: evaluations.append(1),
    )
    assert code == 1
    assert evaluations == []
    assert "没有摆过" in _state(tmp_path)["assemble"][0]["rejection"]


def test_two_rejected_evaluations_stop_as_assemble_failed(tmp_path):
    """@brief 评测两次都是 rejected 时 stop_reason 是 assemble_failed。"""
    contract, domain = _domain("gemm")

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("rejected", ("slow",)),
    )
    assert code == 1
    assert _state(tmp_path)["stop_reason"] == "assemble_failed"


def test_timing_uncertain_reevaluates_once_without_a_new_session(tmp_path):
    """@brief timing_uncertain 不再开会话，只把同一份父包再评一次。"""
    contract, domain = _domain("gemm")
    sessions = []
    evaluations = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        sessions.append(attempt)
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def evaluate(*args, **kwargs):
        evaluations.append(1)
        return _evaluation("timing_uncertain")

    code = _run(tmp_path, assemble_attempts=2, shard_runner=shard_runner, session_runner=session_runner, evaluate=evaluate)
    assert code == 1
    assert sessions == [1]
    assert evaluations == [1, 1]
    assert _state(tmp_path)["stop_reason"] == "timing_uncertain"
    assert (tmp_path / "run" / "portfolio" / "parent.envelope").is_file()
    assert (tmp_path / "run" / "portfolio" / "evaluation-1.json").is_file()
    assert (tmp_path / "run" / "portfolio" / "evaluation-2.json").is_file()


def test_a_hang_stops_the_circuit_and_opens_no_later_session(tmp_path):
    """@brief 评测返回 hang 时熔断，之后没有新会话。"""
    contract, domain = _domain("gemm")
    sessions = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        sessions.append(attempt)
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, assemble_attempts=2, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("hang"),
    )
    assert code == 1
    assert sessions == [1]
    assert _state(tmp_path)["stop_reason"] == "gpu_hang_circuit_breaker"


def test_incomplete_session_cost_stops(tmp_path):
    """@brief 会话花费不完整时停下，原因是 cost_incomplete。"""
    contract, _domain_gemm = _domain("gemm")
    evaluations = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=lambda **kwargs: _view(None, cost_incomplete=True),
        evaluate=lambda *a, **k: evaluations.append(1),
    )
    assert code == 1
    assert evaluations == []
    assert _state(tmp_path)["stop_reason"] == "cost_incomplete"


def test_a_bundle_shard_is_unpacked_into_host_files(tmp_path):
    """@brief bundle 分片的 best.metal 解开后，目录里有 tiling.c 和 execution_geometry.json。"""
    envelope = encode_envelope(CandidateArtifact(track_id="msl-bundle", entries=tuple(bundle_entries())))
    seen = []

    def shard_runner(size, shard_dir):
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "result.json").write_text(json.dumps(_result(size)) + "\n", encoding="utf-8")
        (shard_dir / "best.metal").write_text(envelope, encoding="utf-8")

    def session_runner(*, attempt, work, task_text, submission):
        for size in ("small", "tail", "paper"):
            host = work / "src" / "variants" / size / "host"
            seen.append((
                (host / "tiling.c").is_file(),
                (host / "execution_geometry.json").is_file(),
            ))
        return _view(None, cost_incomplete=True)

    code = run_portfolio_search(
        "kda", "fp16", run_dir=tmp_path / "run", track="msl-bundle", budget=1.0, assemble_budget=1.0,
        seed=11, shard_runner=shard_runner, session_runner=session_runner,
    )
    assert code == 1
    assert seen
    assert all(flags == (True, True) for flags in seen)


def test_render_assemble_task_names_the_shape_table_and_stays_small(tmp_path):
    """@brief 组装说明含形状表、两条命令、交卷路径和「不许改」，例子不用这道题的真实取值。"""
    _contract, domain = _domain("gemm")
    shapes = []
    for instance in domain.instances:
        shapes.append({
            "size": instance.size,
            "usage": list(instance.size_usage),
            "symbols": {name: int(value) for name, value in instance.symbols.items()},
        })
    submission = tmp_path / "gemm-fp16-portfolio-assemble1.metal"
    text = render_assemble_task(
        repo=Path("/repo"), task="gemm", dtype="fp16", target_id=_TARGET, sizes=domain.sizes,
        shapes=shapes,
        shards=[{"size": "paper", "best_speedup": 1.2, "confirmed_speedup": None, "track": "msl-single"}],
        src=tmp_path / "src", submission=submission, rejection="上次拒绝" * 40,
    )
    assert "不许改" in text
    assert "portfolio-pack" in text and "portfolio-check" in text
    assert "--sizes paper,small" in text
    assert str(submission) in text
    assert "NOTES.md" in text
    assert '["and", ["ge", "T", 64], ["lt", "T", 128]]' in text
    assert '["eq", "M", 256]' not in text
    assert "master seed" not in text.lower()
    assert "未见" not in text
    assert len(text.encode("utf-8")) <= 8000
    assert "M=" in text and "搜索期" in text


def test_render_assemble_task_has_a_hard_byte_limit_for_wide_shape_tables():
    """@brief 16 档、每档 40 个符号也保留必需说明，且返回值不超过 8000 字节。"""
    sizes = tuple(f"s{index}" for index in range(16))
    shapes = [
        {
            "size": size,
            "usage": ["golden", "benchmark"],
            "symbols": {f"S{column}": (row + 1) * 1000 + column for column in range(40)},
        }
        for row, size in enumerate(sizes)
    ]
    shards = [
        {"size": size, "best_speedup": 1.25, "confirmed_speedup": None, "track": "msl-single"}
        for size in sizes
    ]
    repo = Path("/r")
    src = Path("/s")
    submission = Path("/o.metal")
    rejection_prefix = "RATIO_PREFIX_fp16/paper/benchmark:effective_ratio>1.10\n"
    text = render_assemble_task(
        repo=repo,
        task="wide",
        dtype="fp16",
        target_id=_TARGET,
        sizes=sizes,
        shapes=shapes,
        shards=shards,
        src=src,
        submission=submission,
        rejection=rejection_prefix + "x" * 2000,
    )
    assert len(text.encode("utf-8")) <= 8000
    assert rejection_prefix in text
    assert "# 上一次被拒" in text
    assert "| size | usage | symbols |" in text
    assert "| size | best_speedup | confirmed_speedup | track |" in text
    for size in sizes:
        assert text.count(f"| {size} |") == 2
    size_text = ",".join(sizes)
    pack = (
        f"uv run --directory {repo} python -m tcenv.cli portfolio-pack wide "
        f"--src {src} --out {submission} --dtype fp16 --sizes {size_text} --target {_TARGET}"
    )
    check = (
        f"uv run --directory {repo} python -m tcenv.cli portfolio-check wide "
        f"{submission} --dtype fp16 --sizes {size_text} --target {_TARGET}"
    )
    assert f"`{pack}`" in text
    assert f"`{check}`" in text
    assert str(submission) in text
    assert "不许改" in text
    assert "NOTES.md" in text
    assert '["and", ["ge", "T", 64], ["lt", "T", 128]]' in text
    assert "搜索期" in text


def test_symbol_cells_shrink_before_a_short_rejection_is_touched():
    """@brief 8×52 只超几十字节时应缩符号格，而不是丢掉完整 ratio 原因。"""
    sizes = tuple(f"s{index}" for index in range(8))
    shapes = [
        {
            "size": size,
            "usage": ["golden", "benchmark"],
            "symbols": {f"S{column}": 10_000_000 + row * 100 + column for column in range(52)},
        }
        for row, size in enumerate(sizes)
    ]
    shards = [
        {"size": size, "best_speedup": 1.25, "confirmed_speedup": None, "track": "msl-single"}
        for size in sizes
    ]
    reason = "RATIO_REASON_fp16/paper/benchmark:effective_ratio>1.10"
    text = render_assemble_task(
        repo=Path("/r"),
        task="wide",
        dtype="fp16",
        target_id=_TARGET,
        sizes=sizes,
        shapes=shapes,
        shards=shards,
        src=Path("/s"),
        submission=Path("/o.metal"),
        rejection=reason,
    )
    assert len(text.encode("utf-8")) <= 8000
    assert reason in text
    assert "# 上一次被拒" in text
    assert "…" in text


def test_assemble_prompt_source_is_the_fingerprinted_template():
    """@brief 模板正文与 guard 字面量必须位于被指纹登记的函数源码里。"""
    source = inspect.getsource(render_assemble_task)
    assert "不许改" in source
    assert '["and", ["ge", "T", 64], ["lt", "T", 128]]' in source
    parts = dict(prompt_fingerprint_parts())
    assert parts["portfolio_prompt.render_assemble_task"] == source.encode("utf-8")


def test_module_level_body_cannot_bypass_the_fingerprinted_prompt(monkeypatch):
    """@brief 劫持旧的模块级正文 helper 不得改变指纹外的组装题面。"""
    monkeypatch.setattr(portfolio_prompt, "_body", lambda **kwargs: "FINGERPRINT_ESCAPE", raising=False)
    text = render_assemble_task(
        repo=Path("/r"),
        task="gemm",
        dtype="fp16",
        target_id=_TARGET,
        sizes=("small",),
        shapes=[{"size": "small", "usage": ["benchmark"], "symbols": {"M": 256}}],
        shards=[{"size": "small", "best_speedup": 1.0, "confirmed_speedup": None, "track": "msl-single"}],
        src=Path("/s"),
        submission=Path("/o.metal"),
    )
    assert "不许改" in text
    assert "FINGERPRINT_ESCAPE" not in text


def test_prepare_session_without_task_text_matches_render_contract(tmp_path):
    """@brief 不传 task_text 时，TASK.md 仍等于 render_contract。"""
    brief = SessionBrief(
        repo=tmp_path, task_name="gemm", dtype="fp16", size="small", kernel_name="gemm",
        triton_source="def gemm():\n    pass\n", description="desc", contract_text="contract",
        abi_stub="kernel void gemm()", rtol=1e-2, atol_rel=1e-2, seedless=True,
    )
    prepare_session(brief, tmp_path / "work")
    assert (tmp_path / "work" / "TASK.md").read_text(encoding="utf-8") == render_contract(brief)


def test_portfolio_search_does_not_import_backends_at_module_level():
    """@brief 驱动模块的顶层不导入 tcenv.backends。"""
    path = Path(__file__).resolve().parents[2] / "taic" / "portfolio_search.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            if name == "tcenv.backends" or name.startswith("tcenv.backends."):
                offenders.append(f"{node.lineno}:{name}")
    assert offenders == []


def test_injected_confirmation_writes_one_file_per_point(tmp_path):
    """@brief 每个点都 confirmed 时，整体是 confirmed，文件数等于点数。"""
    contract, domain = _domain("gemm")

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("ok"), confirm=_confirm_ok,
    )
    assert code == 0
    document = _state(tmp_path)
    assert document["confirmation"]["status"] == "confirmed"
    assert document["publication_allowed"] is False
    files = list((tmp_path / "run" / "confirmations").glob("*.json"))
    assert len(files) == len(domain.plan.instances)
    assert {row["instance_key"] for row in document["confirmation"]["points"]} == {
        point.instance_key for point in domain.plan.instances
    }


def test_one_rejected_point_rejects_the_confirmation(tmp_path):
    """@brief 有一个点不是 confirmed 时，整体记 rejected。"""
    contract, domain = _domain("gemm")

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def confirm(**kwargs):
        key = kwargs["request"].instance_key
        if key == "fp16/paper/golden":
            return {"status": "rejected", "reason": "fresh mismatch", "coverage_kind": "sampled_unseen", "instance_key": key}
        return _confirm_ok(**kwargs)

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("ok"), confirm=confirm,
    )
    assert code == 1
    document = _state(tmp_path)
    assert document["confirmation"]["status"] == "rejected"
    assert "fresh mismatch" in document["confirmation"]["reasons"]
    assert document["publication_allowed"] is False


def test_confirmation_is_not_called_when_evaluation_is_not_ok(tmp_path):
    """@brief 评测不是 ok 时，确认函数一次都不调用。"""
    contract, domain = _domain("gemm")
    calls = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def confirm(**kwargs):
        calls.append(kwargs["request"].instance_key)
        return _confirm_ok(**kwargs)

    code = _run(
        tmp_path, assemble_attempts=1, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("rejected", ("slow",)), confirm=confirm,
    )
    assert code == 1
    assert calls == []
    assert _state(tmp_path)["confirmation"] is None


def test_a_contract_error_stops_the_later_points(tmp_path):
    """@brief 某个点抛 TaskContractError 时记下消息，不再确认后面的点。"""
    from tcenv.contract_errors import TaskContractError

    contract, domain = _domain("gemm")
    calls = []

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def confirm(**kwargs):
        calls.append(kwargs["request"].instance_key)
        raise TaskContractError("stop here")

    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("ok"), confirm=confirm,
    )
    assert code == 1
    assert len(calls) == 1
    document = _state(tmp_path)["confirmation"]
    assert document["status"] == "rejected"
    assert document["reasons"] == ["stop here"]


def test_real_freeze_selects_benchmark_tune_and_golden_unseen(tmp_path, monkeypatch):
    """@brief 真实的 freeze 和 ValidationPlan 把 benchmark 当 tune、golden 当未见，fresh 种子不是 7。"""
    from taic.orchestrator import confirmation_master_seed

    contract, domain = _domain("gemm")
    seen = {}

    def shard_runner(size, shard_dir):
        _write_gemm_shard(shard_dir, contract, size)

    def session_runner(*, attempt, work, task_text, submission):
        _coverage(work / "src", _good_variants())
        return _view(_pack(work, domain))

    def confirm_subject(**kwargs):
        assert "track_id" not in kwargs
        assert "track_version" not in kwargs
        validation = kwargs["validation"]
        seen["tune"] = tuple(validation.tune_shards)
        seen["unseen"] = tuple(validation.unseen_instance_keys)
        seen["seeds"] = dict(validation.fresh_seeds)
        return _confirm_ok(**kwargs)

    monkeypatch.setattr("tcenv.backends.metal.portfolio_confirmation.confirm_subject", confirm_subject)
    code = _run(
        tmp_path, shard_runner=shard_runner, session_runner=session_runner,
        evaluate=lambda *a, **k: _evaluation("ok"),
    )
    assert code == 0
    assert set(seen["tune"]) == {"fp16/paper/benchmark", "fp16/small/benchmark"}
    assert set(seen["unseen"]) == {"fp16/paper/golden", "fp16/small/golden"}
    assert set(seen["seeds"]) == set(seen["unseen"])
    assert all(seed != 7 for seed in seen["seeds"].values())
    secret = confirmation_master_seed(11).hex()
    text = (tmp_path / "run" / "portfolio_search.json").read_text(encoding="utf-8")
    assert secret not in text
    assert secret.upper() not in text
    assert _state(tmp_path)["publication_allowed"] is False
    assert not (tmp_path / "run" / "portfolio_run.json").exists()


def test_dry_run_exits_zero_without_creating_shards(tmp_path, capsys):
    """@brief dry-run 退出 0，并且不创建 shards，gemm 顺序仍是 paper、small。"""
    code = main([
        "portfolio-run", "gemm", "--dtype", "fp16", "--run-dir", str(tmp_path / "run"),
        "--dry-run", "--budget", "1", "--assemble-budget", "1",
    ])
    assert code == 0
    assert "sizes: paper,small" in capsys.readouterr().out
    assert not (tmp_path / "run" / "shards").exists()


def test_invalid_preset_is_a_start_diagnostic_not_a_traceback(tmp_path, capsys):
    """@brief 无效 preset 转成退出码 5，dry-run 不建分片也不泄漏 traceback。"""
    code = main([
        "portfolio-run", "gemm", "--dtype", "fp16", "--run-dir", str(tmp_path / "run"),
        "--dry-run", "--budget", "1", "--assemble-budget", "1",
        "--preset", "definitely-not-a-preset",
    ])
    captured = capsys.readouterr()
    assert code == 5
    assert "definitely-not-a-preset" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "run" / "shards").exists()


def test_varlen_explicit_sizes_dry_run_uses_the_selected_declared_size(tmp_path, capsys):
    """@brief varlen 没有 paper 档时，显式两档仍能完成 dry-run 且不建 shards。"""
    code = main([
        "portfolio-run", "varlen_kda", "--dtype", "fp16",
        "--sizes", "v6_h64_fixed,v6_h96_fixed", "--run-dir", str(tmp_path / "run"),
        "--dry-run", "--budget", "1", "--assemble-budget", "1",
    ])
    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert "sizes: v6_h64_fixed,v6_h96_fixed" in captured.out
    assert not (tmp_path / "run" / "shards").exists()
