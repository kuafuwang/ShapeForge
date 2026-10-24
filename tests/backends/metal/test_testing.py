"""Correctness launch safety checks that do not require a GPU."""

from __future__ import annotations

from types import SimpleNamespace

from tcenv.backends.metal import testing


def test_safety_is_checked_after_every_determinism_launch(monkeypatch):
    """@brief 第二次确定性启动才越界时也必须返回 safety，而不是空列表。"""
    task = SimpleNamespace(cases=["case"])
    monkeypatch.setattr(testing, "prepare_case", lambda _task, _case, _seed: SimpleNamespace(raw=b"x"))
    monkeypatch.setattr(testing, "check_case", lambda _task, _data, _out, _base: SimpleNamespace(ok=True, safety=[]))

    class Args:
        def __init__(self):
            self.launches = 0

        def reset_guards_all(self):
            pass

        def load_inputs(self, _raw):
            pass

        def safety_violations(self):
            return ["guard write"] if self.launches == 2 else []

    args = Args()

    def launch(_args, **_kwargs):
        args.launches += 1
        return [1.0]

    results, safety, nondet = testing.run_correctness(
        None, None, task, args, seed=1, cases=["case"], baseline_fns=None,
        determinism=True, fail_fast=True, launcher=launch,
    )
    assert args.launches == 2
    assert safety == ["[case] guard write"]
    assert nondet == []
    assert results[0].safety == ["guard write"]
