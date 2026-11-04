"""Shared task assembly: track binding + OpSpec → ``Task`` (plan §4.2/§4.4).

Every task module keeps only its representation/oracle glue: the registered
implementations the track names (``gen_inputs``/``reference``/``error_scale``/baselines).
The semantic facts come from ``opspec.json`` and the contract from the registered
LoweringProfile, so a task module never constructs a ``Contract`` and never writes a
grid/threads/num_warps literal.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from tcenv.lowering import Specialization, derive_specialization, lower_variant
from tcenv.opspec.model import SemanticInstance, instantiate, load_opspec
from tcenv.task import BaselineImpl, Task
from tcenv.track import REPO_ROOT, TrackBinding, load_track, pack_digest

TASKS_ROOT = REPO_ROOT / "tasks"


@dataclass(frozen=True)
class GeneratedText:
    """@brief 生成族的题面与 known-good 文本（由 track 登记的生成器产生）。

    A generated track (e.g. the ``sem_int`` family) has no static template: its prompt and
    known-good source are rendered from the OpSpec program by a registered implementation.
    The text enters the task exactly where a rendered template would, so
    ``task_fingerprint``/``reference_for`` keep their single source.
    """

    spec_text: str
    known_good: str


def derive(task_dir: str, dtype: str, size: str) -> tuple[TrackBinding, SemanticInstance, Specialization]:
    """@brief 读取 track/OpSpec 并完成两阶段派生。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @return ``(track, instance, specialization)``。
    """
    track = load_track(TASKS_ROOT / task_dir / "track.json")
    instance = instantiate(load_opspec(track.opspec_path), dtype, size)
    specialization = derive_specialization(
        instance, backend_id=track.backend_id, profile_name=track.lowering_profile
    )
    return track, instance, specialization


def derive_variant(
    task_dir: str, dtype: str, size: str, variant: str
) -> tuple[TrackBinding, SemanticInstance, Specialization]:
    """@brief 读取 track/OpSpec 并由登记的 variant factory 派生启动变体。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param variant variant 文法文本（由 track 登记的 factory 解析）。
    @return ``(track, instance, specialization)``。
    @raises ValueError track 没有登记 variant factory。
    """
    track = load_track(TASKS_ROOT / task_dir / "track.json")
    factory_id = track.variant_factory_id
    if factory_id is None:
        raise ValueError(f"{task_dir}: no variant factory is registered for this task")
    instance = instantiate(load_opspec(track.opspec_path), dtype, size)
    specialization = lower_variant(
        instance, backend_id=track.backend_id, factory_id=factory_id, variant=variant
    )
    return track, instance, specialization


def _registration(track: TrackBinding, role: str) -> dict[str, Any]:
    """@brief 取某个角色的登记身份（缺登记是装配错误）。

    @param track track 绑定。
    @param role ``gen_inputs``/``reference``/``error_scale``。
    @return 登记文档。
    @raises ValueError 该角色未登记。
    """
    entry = track.implementations[role]
    if entry is None:
        raise ValueError(f"track {track.path}: {role} is not registered")
    return dict(entry)


def require_attr(instance: SemanticInstance, name: str) -> float:
    """@brief 取一个已物化的浮点 attr（登记缺失时立刻失败）。

    @param instance 语义实例。
    @param name attr 名。
    @return 浮点值。
    """
    value = instance.resolved_attrs[name]
    return float(value)


def evaluation_meta(
    instance: SemanticInstance,
    track: TrackBinding,
    specialization: Specialization,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """@brief 由 OpSpec + track 物化的只读 ``Task.meta`` 视图（H-17）。

    ``domain_case``/``ieee_*`` 来自语义实例的 ``cases.inputs``，``triton_baseline_variant``
    来自 track 的已登记模板。``extras`` 只接受纯诊断键（如 ``baseline_note``），不得携带
    语义或启动事实。

    @param instance 语义实例。
    @param track track 绑定。
    @param specialization 本次 build 的派生结果。
    @param extras 额外诊断键。
    @return ``Task.meta`` 视图。
    """
    meta = instance.evaluation_meta()
    variant = track.baseline_variant
    if variant is not None:
        meta["triton_baseline_variant"] = {
            "source": track.render(variant, instance=instance, specialization=specialization),
            "note": track.baseline_variant_note,
        }
    for key, value in (extras or {}).items():
        if value not in (None, ""):
            meta[str(key)] = value
    return meta


def build_task(
    task_dir: str,
    dtype: str,
    size: str,
    *,
    implementations: Mapping[str, Callable],
    baseline_fns: Mapping[str, Callable],
    derived: tuple[TrackBinding, SemanticInstance, Specialization] | None = None,
    meta_extras: Mapping[str, Any] | None = None,
    task_name: str | None = None,
    generated: GeneratedText | None = None,
) -> Task:
    """@brief 用 track 登记与实现闭包装配一个 ``Task``。

    @param task_dir ``tasks/`` 下的目录名。
    @param dtype 元素类型。
    @param size 尺寸档位。
    @param implementations ``gen_inputs``/``reference``/``error_scale`` 的实现闭包。
    @param baseline_fns baseline 名 → 实现闭包（必须覆盖 track 登记的全部 baseline）。
    @param derived 本次 build 已经算出的 ``(track, instance, specialization)``；给出时绝不再派生第二次。
    @param meta_extras 额外的纯诊断 ``Task.meta`` 键（不得是语义/启动事实）。
    @param task_name 覆盖 ``Task.name``（variant 题名带 ``@<variant>`` 后缀）。
    @param generated 生成族已渲染的题面/known-good；模板块必须为 ``None``。
    @return 装配好的 ``Task``。
    @raises ValueError track 登记与实现不符，或模板/生成器不匹配。
    """
    track, instance, specialization = derived if derived is not None else derive(task_dir, dtype, size)
    gen_inputs = implementations["gen_inputs"]
    reference = implementations["reference"]
    error_scale = implementations.get("error_scale")

    baselines: list[BaselineImpl] = []
    for entry in track.baselines:
        name = str(entry["name"])
        fn = baseline_fns.get(name)
        if fn is None:
            raise ValueError(f"{task_dir}: baseline {name!r} is registered but has no implementation")
        baselines.append(
            BaselineImpl(
                name,
                str(entry["framework"]),
                fn,
                implementation_id=str(entry["implementation_id"]),
                implementation_version=int(entry["implementation_version"]),
                params=dict(entry["params"]),
            )
        )
    unused = sorted(set(baseline_fns) - {str(entry["name"]) for entry in track.baselines})
    if unused:
        raise ValueError(f"{task_dir}: implementations for unregistered baselines {unused}")

    generated_blocks = (track.spec_text_generator, track.known_good_generator)
    if any(block is not None for block in generated_blocks):
        if generated is None:
            raise ValueError(f"{task_dir}: track declares a generated spec text/known-good; pass generated=...")
        triton_source, known_good = generated.spec_text, generated.known_good
    else:
        if generated is not None:
            raise ValueError(f"{task_dir}: track uses templates; generated text is a second source")
        triton_source, known_good = track.render_spec_text(instance, specialization), track.render_known_good(
            instance, specialization
        )

    registrations = {
        # The pack identifies *which files* implement these registered identities, so the
        # evaluation fingerprint covers the implementation bytes, not a closure ``repr``.
        "task_pack": {
            "pack_files": list(track.pack_files),
            "digest": pack_digest(track.pack_files),
        },
        "gen_inputs": _registration(track, "gen_inputs"),
        "reference": _registration(track, "reference"),
        "error_scale": track.implementations["error_scale"],
        "baselines": [dict(entry) for entry in track.baselines],
    }
    if track.spec_text_generator is not None:
        registrations["spec_text"] = dict(track.spec_text_generator)
    if track.known_good_generator is not None:
        registrations["known_good"] = dict(track.known_good_generator)

    return Task(
        name=task_name or instance.op_name,
        dtype=dtype,
        size=size,
        contract=specialization.contract,
        triton_source=triton_source,
        description=track.description,
        cases=tuple(instance.case_domain.cases),
        gen_inputs=gen_inputs,
        reference=reference,
        error_scale=error_scale,
        baselines=tuple(baselines),
        flops=instance.work_accounting.flops,
        min_bytes=instance.work_accounting.min_bytes,
        rtol=instance.tolerances.rtol,
        atol_rel=instance.tolerances.atol_rel,
        exact=instance.tolerances.exact,
        naive_msl=known_good,
        meta=evaluation_meta(instance, track, specialization, meta_extras),
        semantic_instance=instance,
        specialization=specialization,
        evaluation_implementations=registrations,
    )


__all__ = ["TASKS_ROOT", "GeneratedText", "build_task", "derive", "derive_variant", "evaluation_meta", "require_attr"]
