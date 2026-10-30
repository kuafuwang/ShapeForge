# 按行 RMSNorm。语义只写在 opspec.json；这里只做表示/oracle 装配。
# 题面模板、description、dtype 拼写、known-good 与登记实现身份都在 track.json；
# 契约（核名、grid、threads、常量、gpu）由登记的 Metal LoweringProfile 经 lower 派生。
import torch
import torch.nn.functional as F

from tcenv.distributions import sample
from tcenv.lowering import derive_specialization
from tcenv.opspec.model import instantiate, load_opspec
from tcenv.task import BaselineImpl, Task
from tcenv.track import REPO_ROOT, TrackBinding, load_track, pack_digest

TASK_DIR = "rmsnorm"
TRACK_NAME = "track.json"


def _track() -> TrackBinding:
    """@brief 读取本题的 track 绑定。

    @return 通过校验的 :class:`TrackBinding`。
    """
    return load_track(REPO_ROOT / "tasks" / TASK_DIR / TRACK_NAME)


def _gen_inputs(params: dict, shape: tuple[int, int], dtype: str):
    """@brief 按登记参数构造输入生成闭包（参数来自 track 登记，不是第二份字面量）。

    @param params track 里 ``implementations.gen_inputs.params``。
    @param shape ``(M, H)``。
    @param dtype 元素类型。
    @return ``(case, seed) -> {"x", "w"}``。
    """
    mags = dict(params["mags"])
    default_mag = float(params["default_mag"])
    large = float(params["large"])
    weight_offset = float(params["weight_offset"])
    weight_scale = float(params["weight_scale"])
    weight_seed_offset = int(params["weight_seed_offset"])

    def gen_inputs(case, seed):
        x = sample(case, shape, seed, dtype, mag=mags.get(case, default_mag), large=large)
        w = weight_offset + weight_scale * sample("normal", (shape[1],), seed + weight_seed_offset, dtype)
        return {"x": x, "w": w}

    return gen_inputs


def _reference(params: dict, eps: float):
    """@brief 按登记参数构造参考实现闭包；EPS 来自语义实例。

    @param params track 里 ``implementations.reference.params``。
    @param eps 已物化的 ``EPS`` attr。
    @return ``inp -> {"y"}``。
    """
    if params["accumulate"] != "float64":
        raise ValueError(f"rmsnorm reference only implements float64 accumulation, got {params['accumulate']!r}")

    def reference(inp):
        x, w = inp["x"].double(), inp["w"].double()
        return {"y": x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps) * w}

    return reference


def _mlx_baseline(eps: float):
    """@brief mlx 侧 RMSNorm 基线，走厂商融合实现。

    @param eps 已物化的 ``EPS`` attr。
    @return ``(x, w) -> mx.fast.rms_norm(x, w, eps)``。
    """

    def _mlx(x, w):
        import mlx.core as mx

        return mx.fast.rms_norm(x, w, eps)

    return _mlx


def _torch_baseline(eps: float):
    """@brief torch 侧 RMSNorm 基线。

    @param eps 已物化的 ``EPS`` attr。
    @return ``(x, w) -> F.rms_norm(x, (H,), w, eps)``。
    """

    def _torch(x, w):
        return F.rms_norm(x, (x.shape[-1],), w, eps)

    return _torch


def build(dtype: str, size: str) -> Task:
    """@brief 装配按行 RMSNorm 题。

    ``instantiate`` 产生语义实例，``lower`` 由登记的 ``metal-rmsnorm`` profile 派生契约；
    题面、known-good、description 与登记实现身份来自 track 绑定。

    @param dtype 元素类型，``fp16`` 或 ``bf16``。
    @param size 尺寸档位，``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    @raises tcenv.opspec.model.OpSpecError dtype/size 不在声明域内。
    @raises tcenv.lowering.LoweringError 未登记 profile 或派生非法。
    """
    track = _track()
    spec = load_opspec(track.opspec_path)
    instance = instantiate(spec, dtype, size)
    specialization = derive_specialization(
        instance, backend_id=track.backend_id, profile_name=track.lowering_profile
    )
    eps = float(instance.resolved_attrs["EPS"])
    implementations = track.implementations
    gen_params = dict(implementations["gen_inputs"]["params"])
    ref_params = dict(implementations["reference"]["params"])
    shape = (int(instance.symbols["M"]), int(instance.symbols["H"]))

    baselines = []
    for entry in track.baselines:
        params = dict(entry["params"])
        if params.get("eps_attr") != "EPS":
            raise ValueError(f"baseline {entry['name']!r} must take EPS from the semantic instance")
        fn = _mlx_baseline(eps) if entry["framework"] == "mlx" else _torch_baseline(eps)
        baselines.append(
            BaselineImpl(
                str(entry["name"]),
                str(entry["framework"]),
                fn,
                implementation_id=str(entry["implementation_id"]),
                implementation_version=int(entry["implementation_version"]),
                params=params,
            )
        )

    return Task(
        name="rmsnorm",
        dtype=dtype,
        size=size,
        contract=specialization.contract,
        triton_source=track.render_spec_text(instance, specialization),
        description=track.description,
        cases=tuple(instance.case_domain.cases),
        gen_inputs=_gen_inputs(gen_params, shape, dtype),
        reference=_reference(ref_params, eps),
        baselines=tuple(baselines),
        flops=instance.work_accounting.flops,
        min_bytes=instance.work_accounting.min_bytes,
        rtol=instance.tolerances.rtol,
        atol_rel=instance.tolerances.atol_rel,
        naive_msl=track.render_known_good(instance, specialization),
        semantic_instance=instance,
        specialization=specialization,
        evaluation_implementations={
            "task_pack": {
                "pack_files": list(track.pack_files),
                "digest": pack_digest(track.pack_files),
            },
            "gen_inputs": dict(implementations["gen_inputs"]),
            "reference": dict(implementations["reference"]),
            "error_scale": implementations["error_scale"],
            "baselines": [dict(entry) for entry in track.baselines],
        },
    )
