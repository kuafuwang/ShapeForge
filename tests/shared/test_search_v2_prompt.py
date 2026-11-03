"""v2 的题面：OpSpec/ABI/只读 reference，没有 Triton 段，自检命令是 eval-v2。"""

from __future__ import annotations

from taic.kernel_agent import SessionBrief, render_contract
from taic.prompts import system_prompt_v2
from taic.task_binding import bind_task


def _brief() -> SessionBrief:
    binding = bind_task("gemm", "fp16", "small")
    presentation = binding.presentation()
    return SessionBrief(
        repo=binding.locator.code_root,
        task_name="gemm", dtype="fp16", size="small",
        kernel_name="gemm_kernel", triton_source="",
        description=presentation.statement,
        contract_text="{}",
        abi_stub=binding.signature_stub(),
        rtol=0.002, atol_rel=0.001,
        v2=True, reference_source=presentation.reference_source,
        work_dir="/tmp/work", label="init",
    )


def test_the_v2_system_prompt_has_no_triton_source_and_requires_a_candidate_launch():
    binding = bind_task("gemm", "fp16", "small")
    text = system_prompt_v2(binding, use_tools=False, target_id=binding.target_id)
    assert "There is NO Triton source" in text
    assert "/* taic-launch */" in text
    assert "TensorMemoryContract" in text
    assert "def reference(" in text  # 只读 reference 是题面的一部分
    assert "Triton 源码是规格" not in text


def test_the_v2_task_md_uses_eval_v2_and_never_offers_a_contract_launch():
    text = render_contract(_brief())
    assert "tcenv.cli eval-v2 gemm" in text
    assert "tcenv.cli eval gemm" not in text
    assert "必须自带且只能带一个 `/* taic-launch */` 块" in text
    assert "Triton 源码是规格" not in text
    assert "def reference(" in text


def test_a_v1_brief_keeps_the_v1_command_and_launch_rule():
    brief = _brief()
    brief.v2 = False
    brief.triton_source = "def kernel(): pass"
    text = render_contract(brief)
    assert "tcenv.cli eval gemm" in text and "eval-v2" not in text
    assert "没有 `/* taic-launch */` 时，启动配置用契约里的" in text
    assert "Triton 源码是规格" in text
