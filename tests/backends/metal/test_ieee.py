"""IEEE-754 special values (NaN, +-Inf, -0) in the correctness tests, and the determinism check on every case."""

from __future__ import annotations

from pathlib import Path

import math

import pytest
import torch

from tasks import get_task, list_tasks
from tcenv.backends.metal.testing import IEEE_CASE, CaseData, check_case, inject_specials, prepare_case, scaled_error

ALL_TASKS = list(list_tasks(include_phase2=True))
INF, NAN = float("inf"), float("nan")


# --------------------------------------------------------------------------- injection


@pytest.mark.parametrize("shape", [(4096,), (64, 128), (3, 5, 7), (1 << 16,)])
def test_inject_specials_plants_all_kinds_and_leaves_the_input_alone(shape):
    """@brief 验证注入特殊值时涵盖 NaN/±Inf/-0、不改原张量、且数量有上限并可重放。

    同时检查首尾元素必定是特殊值，注入个数在 8 到 257 之间。

    @param shape 输入张量形状，由 parametrize 提供。
    """
    t = torch.randn(shape)
    before = t.clone()
    u = inject_specials(t, 17)
    assert torch.equal(t, before) and u.shape == t.shape
    flat = u.reshape(-1)
    assert int(torch.isnan(u).sum()) >= 1 and int((u == INF).sum()) >= 1 and int((u == -INF).sum()) >= 1
    assert int(((u == 0) & torch.signbit(u)).sum()) >= 1  # -0.0
    assert not torch.isfinite(flat[0]) or flat[0] == 0  # first and last element are always special
    assert not torch.isfinite(flat[-1]) or flat[-1] == 0
    changed = int((~torch.isfinite(u)).sum()) + int(((u == 0) & torch.signbit(u)).sum())
    assert 8 <= changed <= 256 + 1  # bounded, and most of the tensor stays untouched
    again = inject_specials(t, 17)  # reproducible from the seed (NaN != NaN, so compare with NaN replaced)
    assert torch.equal(torch.nan_to_num(again, nan=7.0), torch.nan_to_num(u, nan=7.0))


# --------------------------------------------------------------------------- comparison semantics


def err(out, ref, scale=1.0):
    """@brief 用固定容差计算输出与参考之间的缩放误差，便于断言。

    @param out 被测输出值序列。
    @param ref 参考值序列。
    @param scale 误差缩放系数，默认 1.0。
    @return 每个元素一个误差值的一维张量。
    """
    return scaled_error(torch.tensor(out), torch.tensor(ref), scale, 1e-3, 1e-3, False)


def test_matching_special_values_are_exact_matches():
    """@brief 验证输出与参考的特殊值完全相同（NaN/Inf/-Inf）时误差为零。"""
    assert err([NAN, INF, -INF, 1.0], [NAN, INF, -INF, 1.0]).max() == 0


@pytest.mark.parametrize(
    "out,ref",
    [
        (1.0, NAN),  # a NaN that was dropped
        (NAN, 1.0),  # a NaN that appeared
        (INF, -INF),  # wrong sign
        (65504.0, INF),  # overflow handled as a finite number
        (INF, 1.0),
        (NAN, INF),
        (INF, NAN),
    ],
)
def test_mismatching_special_values_fail(out, ref):
    """@brief 验证特殊值不匹配（丢失或误现 NaN、符号错、溢出成有限值等）一律判为超容差。

    @param out 被测输出值。
    @param ref 参考值，由 parametrize 提供各类不匹配组合。
    """
    e = err([out], [ref])
    assert math.isinf(float(e[0])) or float(e[0]) > 1, e


def test_finite_tolerance_is_unchanged():
    """@brief 验证有限值仍按原容差判定：千分之五以内通过，超出则失败。"""
    assert err([1.0005], [1.0], scale=1.0)[0] < 1
    assert err([1.5], [1.0], scale=1.0)[0] > 1


# --------------------------------------------------------------------------- the case on every task (CPU only)


@pytest.mark.parametrize("name", ALL_TASKS)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_every_task_has_a_meaningful_ieee_case(name, dtype):
    """@brief 验证每个任务每种 dtype 都有真能检出问题的 ieee 用例，且可由种子复现。

    参考输出既要有特殊值，也要保留足够的有限值供数值比较；误差缩放必须有限；
    同种子可复现，与不同种子的原始数据不同。

    @param name 任务名，由 parametrize 提供。
    @param dtype 数据类型，fp16 或 bf16。
    """
    task = get_task(name, dtype, "small")
    assert IEEE_CASE in task.cases
    d = prepare_case(task, IEEE_CASE, 99)
    assert any(int((~torch.isfinite(v)).sum()) for v in d.ref.values()), "the reference sees no special value at all"
    assert any(int(torch.isfinite(v).sum()) for v in d.ref.values()), "the whole reference output is non-finite: nothing is left to check numerically"
    for k, s in d.scale.items():
        assert bool(torch.isfinite(torch.as_tensor(s)).all()), f"non-finite error scale for {k}"
    n_special = sum(int(((~torch.isfinite(v.float())) | ((v.float() == 0) & torch.signbit(v.float()))).sum()) for v in d.rounded.values() if v.is_floating_point())
    assert n_special >= 8
    again = prepare_case(task, IEEE_CASE, 99)
    assert d.raw == again.raw  # reproducible from the seed
    assert d.raw != prepare_case(task, IEEE_CASE, 100).raw


# --------------------------------------------------------------------------- end to end (GPU)

gpu = pytest.mark.gpu

# ``lion`` is the preserved v1 elementwise task with an ``if (j < N)`` body.  A5 retired the
# migrated relu assembly, so the harness-level checks below mutate lion's body instead; the
# property under test is the IEEE / determinism detection, not the operator.
LION_OK = """if (j < N) {
            float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]);
            float c = BETA1 * mv + (1.0f - BETA1) * gv;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            p_out[j] = half(pv * (1.0f - LR * WD) - LR * u);
            m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv);
        }"""
LION_DROPS_NAN = """if (j < N) {
            float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]);
            bool special = isnan(pv) || isnan(gv) || isnan(mv) || isinf(pv) || isinf(gv) || isinf(mv);
            if (special) { p_out[j] = half(0.0f); m_out[j] = half(0.0f); }
            else {
                float c = BETA1 * mv + (1.0f - BETA1) * gv;
                float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
                p_out[j] = half(pv * (1.0f - LR * WD) - LR * u);
                m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv);
            }
        }"""
# correct on every case when the output starts out NaN-filled, but writes 0 instead of the result for NaN inputs when it does not
LION_NAN_DEPENDS_ON_OUTPUT = """if (j < N) {
            float pv = float(p[j]), gv = float(g[j]), mv = float(m[j]);
            float c = BETA1 * mv + (1.0f - BETA1) * gv;
            float u = c > 0.0f ? 1.0f : (c < 0.0f ? -1.0f : 0.0f);
            float r = pv * (1.0f - LR * WD) - LR * u;
            p_out[j] = half((isnan(pv) && !isnan(float(p_out[j]))) ? 0.0f : r);
            m_out[j] = half(BETA2 * mv + (1.0f - BETA2) * gv);
        }"""




def _evaluate_lion(source: str):
    """@brief 在 v2 标准入口上评测一个 lion 候选（small/fp16）。

    B2 之后 lion 不再有 v1 装配，这两个 IEEE 行为测试改走真实 v2 ``evaluate_task``。
    注意必须显式给出完整 case 表：``quick`` 模式默认只跑 normal/wide_range/outlier，
    不带 ieee 的话这两个测试会假性通过。
    """
    from tcenv.backends.metal.track_adapter import MetalTrackBuilder, MetalTrackEvaluator, msl_single_geometry_resolver
    from tcenv.taskcontract import load_task_contract
    from tcenv.taskcontract.locator import TaskLocator
    from tcenv.taskcontract.pack import task_pack_closure
    from tcenv.taskcontract.runner import evaluate_task
    from tcenv.track import REPO_ROOT

    directory = Path(__file__).resolve().parents[3] / "tasks" / "lion"
    contract = load_task_contract(directory, dtype="fp16", size="small")
    locator = TaskLocator(
        task_id="lion", op_name=contract.op_name, dtype=contract.dtype, size=contract.size,
        pack_digest=contract.pack_digest,
        deployment_pack_digest=task_pack_closure(directory, root=REPO_ROOT).digest,
        task_dir=directory, code_root=REPO_ROOT,
    )
    candidate = (
        "/* taic-launch\nmath fast\nlaunch lion_kernel grid=64,1,1 threads=128,1,1\n*/\n"
        "constant uint BLOCK_SIZE = 1024;\n" + source
    )
    return evaluate_task(
        locator,
        track_id="msl-single",
        candidate_source=candidate,
        builder=MetalTrackBuilder(),
        evaluator=MetalTrackEvaluator(
            mode="quick", seed=5, reps=1,
            cases=("normal", "wide_range", "large", "tiny", "outlier", "ieee"),
        ),
        geometry_resolver=msl_single_geometry_resolver(contract),
        track_registry_path=Path(__file__).resolve().parents[3] / "backends" / "metal" / "tracks.json",
    )

def lion_with(body: str) -> str:
    """@brief 把 lion naive 内核里的一段标准循环体替换成给定实现。

    @param body 替换后的内核循环体源码。
    @return 替换好的完整 MSL 源码。
    """
    from tests.support.v1_specimen import specimen_task

    src = specimen_task("lion", "fp16", "small").naive_msl  # B2：lion 已迁，用归档候选文本
    assert LION_OK in src
    return src.replace(LION_OK, body)


@gpu
def test_a_kernel_that_drops_nan_is_rejected_by_the_ieee_case_only():
    """@brief 验证把 NaN 当 0 输出的内核只会被 ieee 用例揪出，不会误判其他用例。

    需要 GPU。
    """
    report = _evaluate_lion(lion_with(LION_DROPS_NAN)).report
    assert report["status"] == "incorrect"
    assert [c["case"] for c in report["correctness"]["cases"] if not c["ok"]] == ["ieee"]
    # The report names the wrong value and the non-finite reference (the first injected special
    # in lion's ``p`` may be any of nan/-inf/inf, so all three spellings are accepted).
    assert "got 0 want" in report["message"]
    assert any(spelling in report["message"] for spelling in ("want nan", "want -inf", "want inf"))


@gpu
def test_determinism_is_checked_on_every_case_not_only_the_first():
    """@brief 验证确定性检查会跑遍所有用例，最后一个 ieee 用例也能发现依赖输出初值的内核。

    需要 GPU。
    """
    report = _evaluate_lion(lion_with(LION_NAN_DEPENDS_ON_OUTPUT)).report
    assert report["status"] == "unsafe" and "initial" in report["message"]
    # all the earlier cases passed, the problem only shows on the last one
    cases = report["correctness"]["cases"]
    assert cases[-1]["case"] == "ieee" and not cases[-1]["ok"]
    assert all(c["ok"] for c in cases[:-1])


@gpu
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_naive_kernels_of_all_tasks_pass_the_full_case_list_including_ieee(dtype):
    """@brief 验证所有任务的 naive 内核在该 dtype 下都能跑完包含 ieee 的完整用例表。

    需要 GPU。

    @param dtype 数据类型，fp16 或 bf16。
    """
    from tcenv.evaluate import evaluate

    for name in ALL_TASKS:
        t = get_task(name, dtype, "small")
        rep = evaluate(name, dtype, "small", t.naive_msl, mode="quick", seed=3)
        assert rep.status == "ok", f"{name} {dtype}: {rep.message}"
        assert IEEE_CASE in [c["case"] for c in rep.correctness["cases"]]


@gpu
@pytest.mark.parametrize("name", ALL_TASKS)
def test_at_least_one_vendor_baseline_reproduces_the_reference_on_special_values(name):
    """@brief 验证每个任务至少有一个厂商标杆能在特殊值上复现参考结果。

    ieee 用例只有在库实现能达到 IEEE 语义时才是公平要求，因此取所有标杆里的最小
    最差误差来判定。需要 GPU。

    @param name 任务名，由 parametrize 提供。
    """
    from tcenv.backends.metal.baselines import BaselineRunner
    from tcenv.backends.metal.testing import _floor

    task = get_task(name, "fp16", "small")
    d = prepare_case(task, IEEE_CASE, 1234)
    worst: dict[str, float] = {}
    for impl in task.baselines:
        outs = BaselineRunner(impl, task, d.rounded, rotate=1).outputs()
        worst[impl.name] = max(
            float(scaled_error(outs[a.name], d.ref[a.name], d.scale[a.name], task.rtol, task.atol_rel, task.exact, _floor(a.dtype)).max())
            for a in task.contract.outputs
        )
    assert min(worst.values()) <= 1.0, worst


def test_building_a_task_for_one_dtype_does_not_change_another_dtypes_tolerances():
    """@brief 回归用例：先构建一种 dtype 再构建另一种，不得改掉前者的容差。

    Lion 任务曾把 atol_rel 存在模块级全局变量里，导致最后构建的 dtype 覆盖了前面的值。
    B2 之后 lion 在 v2 标准入口上，这个回归直接走 v2 的合同与 error_scale 钩子。
    """
    from tcenv.backends.metal.v2_runtime import prepare_v2_case
    from tcenv.taskcontract import load_task_contract

    root = Path(__file__).resolve().parents[3]

    def scale(dtype):
        contract = load_task_contract(root / "tasks" / "lion", dtype=dtype, size="small")
        case = prepare_v2_case(contract, "tiny", 3)
        return float(case.scale["p_out"].max())

    first = scale("fp16")
    scale("bf16")
    assert scale("fp16") == first


def test_inject_specials_respects_a_cap_but_keeps_first_and_last():
    """@brief 验证注入特殊值受 max_n 上限约束，同时仍保留首尾两个元素为特殊值。"""
    import torch

    from tcenv.backends.metal.testing import inject_specials

    t = torch.ones(128, 3, 3, 64)
    out = inject_specials(t, 5, max_n=4)
    special = ~torch.isfinite(out) | (out == 0)  # NaN, +-Inf, -0.0; the ones are never 0
    assert int(special.sum()) == 4
    assert bool(special.reshape(-1)[0]) and bool(special.reshape(-1)[-1])


def test_conv2d_ieee_reference_keeps_most_outputs_finite_and_matches_conv2d_on_finite_inputs():
    """@brief 验证 conv2d 的 ieee 参考保大多数输出有限，且有限输入下与 F.conv2d 一致。

    CPU conv2d 可能把一个 Inf 在 Winograd 分块里扩散成 NaN，参考实现必须避免，
    否则该用例什么都检不出来。
    """
    import torch
    import torch.nn.functional as F

    from tcenv.backends.metal.v2_runtime import prepare_v2_case
    from tcenv.taskcontract import load_task_contract

    contract = load_task_contract(Path(__file__).resolve().parents[3] / "tasks" / "conv2d", dtype="fp16", size="small")
    d = prepare_v2_case(contract, "ieee", 11)
    y = d.ref["y"]
    assert torch.isfinite(y).double().mean() > 0.75
    assert torch.isinf(y).any() and torch.isnan(y).any()
    # finite inputs take the F.conv2d path; the tap-wise path must agree with it
    n = prepare_v2_case(contract, "normal", 12)
    ref = contract.reference.execute(contract.instance, n.rounded)["y"]
    tap = torch.zeros_like(ref, dtype=torch.float32)
    x, w = n.rounded["x"].float(), n.rounded["w"].float()
    OH, OW = ref.shape[1], ref.shape[2]
    for kh in range(3):
        for kw in range(3):
            tap += x[:, kh:kh + OH, kw:kw + OW, :] @ w[:, kh, kw, :].T
    assert torch.allclose(tap.double(), ref, rtol=1e-4, atol=1e-3 * float(ref.abs().max()))


def test_a_non_finite_error_scale_fails_instead_of_passing():
    """@brief 验证误差缩放为 NaN 或 Inf 时误差判为无限，而不是被当成通过。"""
    for scale in (float("nan"), float("inf")):
        errors = scaled_error(torch.tensor([0.0]), torch.tensor([0.0]), scale, 1e-3, 1e-3, False)
        assert torch.isinf(errors).all(), scale


def test_the_absolute_floor_is_half_the_smallest_subnormal():
    """@brief 验证每种 dtype 的绝对误差下限介于最小次正规数的一半与自身之间，整数类型为零。"""
    from tcenv.backends.metal.testing import _floor
    from tcenv.dtypes import FLOAT_TINY_SUBNORMAL

    for dtype, tiny in FLOAT_TINY_SUBNORMAL.items():
        assert 0.5 <= _floor(dtype) / tiny < 1.0, dtype
    assert _floor("i32") == 0.0


def _tolerance_task(*, exact: bool = False):
    """@brief 造一个只有单个 fp16 输出的最小假任务，用于剥离任务本身的容差对比。

    @param exact 是否要求精确相等。
    @return 带 contract/rtol/atol_rel/exact 的 SimpleNamespace。
    """
    from types import SimpleNamespace

    arg = SimpleNamespace(name="y", dtype="fp16", shape=(1,))
    return SimpleNamespace(contract=SimpleNamespace(outputs=[arg], inputs=[], arg=lambda _name: arg), rtol=1e-3, atol_rel=1e-3, exact=exact)


def _tolerance_data(*, case: str = "normal"):
    """@brief 造一份单输出、参考值为 1.0 的 CaseData。

    @param case 用例名。
    @return 构造好的 CaseData。
    """
    return CaseData(case=case, seed=1, rounded={}, raw={}, ref={"y": torch.tensor([1.0], dtype=torch.float64)}, scale={"y": torch.tensor(1.0, dtype=torch.float64)})


def _tolerance_raw(value: float) -> dict[str, bytes]:
    """@brief 把单个 fp16 输出值包装成原始字节字典，模拟内核回传的原始输出。

    @param value 要编码的浮点值。
    @return 键为输出名 y、值为 fp16 字节的字典。
    """
    return {"y": torch.tensor([value], dtype=torch.float16).numpy().tobytes()}


def test_the_baseline_can_widen_the_tolerance_but_never_past_the_cap():
    """@brief 验证基准可适度放宽容差但受 ALLOW_CAP 封顶，且只在 normal 与非 exact 用例生效。"""
    task, data, candidate = _tolerance_task(), _tolerance_data(), _tolerance_raw(1.004)
    alone = check_case(task, data, candidate, None)
    assert alone.allow == 1.0 and not alone.ok
    perfect = {"mlx": {"y": torch.tensor([1.0])}}
    assert check_case(task, data, candidate, perfect).allow == 1.0
    sloppy = {"mlx": {"y": torch.tensor([1.01])}}
    allowed = check_case(task, data, candidate, sloppy)
    assert allowed.allow == 8.0 and allowed.ok  # capped at ALLOW_CAP, not unbounded
    broken = {"mlx": {"y": torch.tensor([1.1])}}
    assert check_case(task, data, _tolerance_raw(1.05), broken).ok is False
    assert check_case(task, _tolerance_data(case="domain"), candidate, sloppy).allow == 1.0
    assert check_case(_tolerance_task(exact=True), data, candidate, sloppy).allow == 1.0

