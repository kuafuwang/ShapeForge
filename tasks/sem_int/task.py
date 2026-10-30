# 动态整数语义族 sem_int_<seed>：程序由登记的生成器确定性物化成 OpSpec JSON（含完整语句表与
# 数值规则），task 装配只读该 JSON；契约与启动事实来自登记的 metal-sem-int profile。
# dtype 是显式别名表：{fp16, bf16, i32} -> i32，其他 dtype 拒绝（§4.4 的有意接受域收窄）。
from __future__ import annotations

from typing import Callable

import torch

from tcenv.lowering import derive_specialization
from tcenv.opspec.model import instantiate, opspec_from_document
from tcenv.task import Task
from tcenv.track import REPO_ROOT, load_track

from tasks._assembly import GeneratedText, build_task, require_attr

from . import document as opspec_document
from . import program

INT_PREFIX = "sem_int_"
TASK_DIR = "sem_int"
PROFILE_NAME = "metal-sem-int"
TRACK_PATH = REPO_ROOT / "tasks" / TASK_DIR / "track.json"

#: 显式 dtype 别名：整数题按 i32 物化，请求 dtype 只决定“允许哪些拼写”。
DTYPE_ALIASES = {"fp16": "i32", "bf16": "i32", "i32": "i32"}
#: 登记的种子域：非负、至多 18 位的十进制整数（上限保证 ``op_name``/任务名长度有界）。
#: 冻结实现接受任意 Python 整数（含负数），这里有意收窄并在 R2 覆盖清单登记。
SEED_DIGITS_MAX = 18
#: 登记生成器身份与参数（种子是生成器输入，不是语义合同；语义在物化出的文档里）。
GENERATOR_ID = "sem_int.int_program"
GENERATOR_VERSION = 1
GENERATOR_PARAMS = {"n_stmts": 105, "program_kind": "int32-block-program"}


def parse_seed(name: str) -> int:
    """@brief 从题目名里取种子并校验登记的种子域。

    @param name 形如 ``sem_int_<seed>`` 的题目名。
    @return ``0 <= seed < 10**18`` 的种子。
    @raises ValueError 名字不以 ``sem_int_`` 开头、后缀不是十进制整数或超出种子域。
    """
    if not name.startswith(INT_PREFIX):
        raise ValueError(f"not an integer program task: {name!r}")
    suffix = name[len(INT_PREFIX) :]
    # ASCII only, explicitly: ``str.isdigit`` is true for Unicode decimal digits
    # (``"١".isdigit()``), so it would accept spellings this registered domain never declared.
    if not suffix or any(character not in "0123456789" for character in suffix):
        raise ValueError(
            f"{name!r}: the seed must be a non-negative ASCII decimal integer (registered seed domain)"
        )
    if len(suffix) > SEED_DIGITS_MAX:
        # The domain is bounded by *spelling* length, not only by value: a longer suffix with
        # leading zeros parses to the same int, so a value check alone lets it through.
        raise ValueError(
            f"{name!r}: the seed spelling must be at most {SEED_DIGITS_MAX} ASCII decimal digits "
            f"(registered seed domain), got {len(suffix)}"
        )
    seed = int(suffix)
    if seed >= 10**SEED_DIGITS_MAX:
        raise ValueError(f"{name!r}: the seed must be below 10**{SEED_DIGITS_MAX} (registered seed domain)")
    return seed


def _gen_inputs(params: dict, n: int) -> Callable[[str, int], dict[str, torch.Tensor]]:
    """@brief 按 track 登记参数构造输入生成闭包（与冻结实现逐位一致）。

    @param params ``implementations.gen_inputs.params``。
    @param n 元素个数。
    @return ``(case, seed) -> {"x", "y"}``。
    """
    seed_mask = int(params["seed_mask"])
    lo, hi = (int(value) for value in params["random"])
    small_lo, small_hi = (int(value) for value in params["small"])
    edge_values = torch.tensor([int(value) for value in params["edge_values"]], dtype=torch.int64)

    def gen_inputs(case: str, seed: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed & seed_mask)
        if case == "random":
            make = lambda: torch.randint(lo, hi, (n,), generator=generator, dtype=torch.int64)
        elif case == "small":
            make = lambda: torch.randint(small_lo, small_hi, (n,), generator=generator, dtype=torch.int64)
        else:  # edge values
            make = lambda: edge_values[torch.randint(0, len(edge_values), (n,), generator=generator)]
        return {"x": make(), "y": make()}

    return gen_inputs


def build(name: str, dtype: str, size: str) -> Task:
    """@brief 装配一道整数语义题：生成器 → OpSpec 文档 → 实例 → 登记 profile → Task。

    @param name ``sem_int_<seed>``。
    @param dtype 请求 dtype；只接受 ``DTYPE_ALIASES`` 里的拼写。
    @param size ``paper`` 或 ``small``。
    @return 装配好的 ``Task``。
    @raises ValueError dtype 不在别名表、size 非法或名字里的种子不是整数。
    """
    alias = DTYPE_ALIASES.get(dtype)
    if alias is None:
        raise ValueError(
            f"{name}: dtype {dtype!r} is not an accepted spelling of this integer family; "
            f"registered aliases are {sorted(DTYPE_ALIASES)}"
        )
    seed = parse_seed(name)
    text = opspec_document.opspec_document(seed, size)
    source = f"generated:{GENERATOR_ID}:v{GENERATOR_VERSION}:seed={seed}"
    spec = opspec_from_document(text, source=source)
    instance = instantiate(spec, alias, size)
    track = load_track(TRACK_PATH)
    specialization = derive_specialization(instance, backend_id=track.backend_id, profile_name=PROFILE_NAME)
    document = dict(spec.document["program"])
    n = int(instance.symbols["N"])
    scalar_k = int(require_attr(instance, "SCALAR_K"))
    contract = specialization.contract
    # Cross-check the semantic block size against the registered launch facts: the document's
    # block is a reduction domain *and* the profile's BLOCK constant, so the two must agree.
    profile_constants = {name: value for name, _msl, value in contract.constants}
    block = program.block_of(document)
    if int(profile_constants["BLOCK"]) != block:
        raise ValueError(
            f"{name}: the program document block {block} disagrees with the registered profile's "
            f"BLOCK {profile_constants['BLOCK']}"
        )
    threads = int(specialization.contract.threads[0])
    num_warps = int(specialization.backend_params["NUM_WARPS"])
    if threads != num_warps * 32 or block % threads:
        raise ValueError(f"{name}: block {block} / threads {threads} / warps {num_warps} is not a valid launch")
    triton_source = program.triton_source(document, n, num_warps=num_warps, scalar_k=scalar_k)
    known_good = program.msl(
        document, threads=threads, num_warps=num_warps, per_thread=block // threads
    )

    def reference(inp: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {"out": program.evaluate(document, inp["x"], inp["y"], scalar_k)}

    return build_task(
        TASK_DIR,
        alias,
        size,
        implementations={
            "gen_inputs": _gen_inputs(dict(track.implementations["gen_inputs"]["params"]), n),
            "reference": reference,
            "error_scale": None,
        },
        baseline_fns={},
        derived=(track, instance, specialization),
        meta_extras={"program_lines": len(triton_source.splitlines())},
        task_name=name,
        generated=GeneratedText(spec_text=triton_source, known_good=known_good),
    )


__all__ = [
    "DTYPE_ALIASES",
    "GENERATOR_ID",
    "GENERATOR_PARAMS",
    "GENERATOR_VERSION",
    "INT_PREFIX",
    "PROFILE_NAME",
    "SEED_DIGITS_MAX",
    "build",
    "parse_seed",
]
