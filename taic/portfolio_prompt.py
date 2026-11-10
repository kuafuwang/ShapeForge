"""TASK.md for the session that writes a shape-portfolio coverage.json.

The text is part of the prompt fingerprint.  It names shapes and search-time
scores.  It does not name confirmation seeds.
"""

from __future__ import annotations

from pathlib import Path


def render_assemble_task(
    *,
    repo: Path,
    task: str,
    dtype: str,
    target_id: str,
    sizes: tuple[str, ...] | list[str],
    shapes: list[dict],
    shards: list[dict],
    src: Path,
    submission: Path,
    rejection: str = "",
) -> str:
    """@brief 生成组装会话的 ``TASK.md``。

    例子里的符号是 ``T``，不用这道题的真实取值。上一次拒绝原因截到 1500 字节。
    全文不超过 8000 字节。模板、字面量和截断都在这个函数里，因为信任指纹只读取本函数源码。

    @param repo 仓库根，写进 ``uv run --directory``。
    @param task 题名。
    @param dtype 数据类型。
    @param target_id 目标档案。
    @param sizes 这次的 size，顺序写进命令。
    @param shapes 每个 size 的 ``symbols`` 与 ``usage``。
    @param shards 每个分片的搜索期成绩。
    @param src 已摆好变体的 ``src`` 目录。
    @param submission 交卷文件的绝对路径。
    @param rejection 上一次被拒的原因；空字符串表示没有。
    @return TASK.md 全文，末尾一个换行。
    """
    max_bytes = 8000
    rejection_bytes = 1500
    example_guard = '["and", ["ge", "T", 64], ["lt", "T", 128]]'

    def _clip(text: str, limit: int) -> str:
        """按 UTF-8 字节截断，不留下半个码点。"""
        raw = text.encode("utf-8")
        if len(raw) <= limit:
            return text
        return raw[:limit].decode("utf-8", errors="ignore")

    def _symbol_cell(text: str, limit: int | None) -> str:
        """超宽形状表按固定上限缩单元格，省略号明确标记被省略的符号。"""
        if limit is None or len(text.encode("utf-8")) <= limit:
            return text
        if limit <= 0:
            return ""
        marker = "…"
        marker_size = len(marker.encode("utf-8"))
        if limit <= marker_size:
            return _clip(text, limit)
        return _clip(text, limit - marker_size) + marker

    def _body(*, symbol_cell_bytes: int | None, rejection_text: str) -> str:
        """拼正文；调用方先完整渲染，再按确定档位收缩符号单元格。"""
        shape_lines = ["| size | usage | symbols |", "| --- | --- | --- |"]
        for row in shapes:
            symbols = ", ".join(f"{name}={value}" for name, value in row["symbols"].items())
            symbols = _symbol_cell(symbols, symbol_cell_bytes)
            usage = ",".join(str(item) for item in row["usage"])
            shape_lines.append(f"| {row['size']} | {usage} | {symbols} |")
        score_lines = [
            "| size | best_speedup | confirmed_speedup | track |",
            "| --- | --- | --- | --- |",
        ]
        for row in shards:
            score_lines.append(
                f"| {row['size']} | {row.get('best_speedup')} | "
                f"{row.get('confirmed_speedup')} | {row.get('track')} |"
            )
        if rejection_text:
            rejected = (
                "\n# 上一次被拒\n" + rejection_text.rstrip()
                + "\n\n这些数字是搜索期成绩，不是确认成绩。\n"
            )
        else:
            rejected = "\n这些数字是搜索期成绩，不是确认成绩。\n"
        return (
            f"# 目标\n"
            f"写 `src/coverage.json`，把这几颗核拼成一个组合。题是 `{task}`，dtype `{dtype}`，目标 `{target_id}`。\n"
            f"不要改核。`src/variants/` 下的文件不许改、不许增删。\n\n"
            f"# 形状表\n"
            + "\n".join(shape_lines)
            + "\n\n# 搜索期成绩\n"
            "下面是各分片在搜索期的成绩。那是搜索期数字，只供你选 guard，不能当成已确认的速度。\n"
            + "\n".join(score_lines)
            + "\n\n# 规则\n"
            "每个点恰好由一个变体接住，或者明确留给 fallback。一颗核可以接多个 size，也可以不用。\n"
            "变体目录名就是 size 名，`coverage.json` 里的 `variant_key` 用这些名字。\n"
            "guard 可用的运算：`eq ne lt le gt ge and or not rem`。可用的名字就是形状表里的符号名。\n"
            f"举例用的是别的题的符号，不要照抄到这道题上：`{example_guard}`。\n\n"
            "# coverage.json\n"
            "顶层只有这四个键：\n"
            "```json\n"
            "{\n"
            '  "schema_id": "tcenv.portfolio-assembly.coverage.1",\n'
            '  "planner": {"id": "你起的名字", "version": 1},\n'
            '  "variants": [{"variant_key": "size 名", "guard": ["ge", "T", 1]}],\n'
            '  "fallback": {"kind": "refuse", "reasons": ["outside_candidate_coverage"]}\n'
            "}\n"
            "```\n"
            "`variants[i]` 只写 `variant_key`、`guard`，以及可选的 `bucket_key`。没写 `bucket_key` 时用 `variant_key`。\n"
            "`variant_key` 集合必须和 `src/variants/` 下的目录名完全相等。\n\n"
            "# fallback\n"
            "两种写法。`reasons` 只能是 `outside_candidate_coverage`、`resource_unavailable`、`variant_unavailable`。\n"
            '- `{"kind": "refuse", "reasons": ["outside_candidate_coverage"]}`：不许有 `src/fallback/` 目录。\n'
            "- `registered_implementation`：必须有 `src/fallback/`，并带 `implementation_id`（字符串）和 "
            "`implementation_version`（正整数）。实现型 fallback 的文件放 `src/fallback/`，内容必须是 "
            "`src/variants/` 下某一颗核的原样拷贝。\n\n"
            "# 命令\n"
            "先打包，再检查。`--sizes` 就是这次的列表，不要改。\n"
            f"`{pack}`\n"
            f"`{check}`\n\n"
            "# 交卷\n"
            f"交卷文件的绝对路径是 `{submission}`。先过 `portfolio-check` 再交。\n"
            "维护 `NOTES.md`。\n"
            f"{rejected}"
        )

    size_text = ",".join(str(size) for size in sizes)
    pack = (
        f"uv run --directory {repo} python -m tcenv.cli portfolio-pack {task} "
        f"--src {src} --out {submission} --dtype {dtype} --sizes {size_text} --target {target_id}"
    )
    check = (
        f"uv run --directory {repo} python -m tcenv.cli portfolio-check {task} "
        f"{submission} --dtype {dtype} --sizes {size_text} --target {target_id}"
    )
    clipped_rejection = _clip(rejection, rejection_bytes)
    text = _body(symbol_cell_bytes=None, rejection_text=clipped_rejection)
    if len(text.encode("utf-8")) <= max_bytes:
        return text

    # 先保留完整（最多 1500 字节的）拒绝原因，只收缩 symbols 单元格。
    # size 行、成绩行、表头、命令和交卷路径在这一阶梯里始终完整。
    for cell_bytes in (256, 192, 128, 96, 64, 48, 32, 24, 16, 8, 0):
        text = _body(symbol_cell_bytes=cell_bytes, rejection_text=clipped_rejection)
        if len(text.encode("utf-8")) <= max_bytes:
            return text

    # 空 symbols 单元格仍装不下时，才从 1500 字节裁剪稿的末尾缩短拒绝原因。
    # 二分寻找能放下的最大 UTF-8 字节前缀；非空前缀会保留「上一次被拒」标题。
    skeleton = _body(symbol_cell_bytes=0, rejection_text="")
    if len(skeleton.encode("utf-8")) > max_bytes:
        raise ValueError("portfolio assembly commands and score table exceed the 8000-byte TASK.md limit")
    raw_rejection = clipped_rejection.encode("utf-8")
    lower, upper = 0, len(raw_rejection)
    best = skeleton
    while lower <= upper:
        candidate_bytes = (lower + upper) // 2
        candidate = _clip(clipped_rejection, candidate_bytes)
        text = _body(symbol_cell_bytes=0, rejection_text=candidate)
        if len(text.encode("utf-8")) <= max_bytes:
            best = text
            lower = candidate_bytes + 1
        else:
            upper = candidate_bytes - 1
    return best


__all__ = ["render_assemble_task"]
