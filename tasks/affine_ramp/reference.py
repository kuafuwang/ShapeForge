"""affine_ramp: the independent oracle of the workspace/Host-Tiling standard task.

``y[n] = scale_tab[n // SEG] * x[n]`` where ``scale_tab[s] = BASE + STEP * s`` is the fp32 ramp the
candidate's Host Tiling plan materialises into the declared workspace.

Two things this module deliberately does not do:

* it never reads a symbol or a workspace.  ``reference(inputs, attrs)`` sees only ``inputs`` and
  ``attrs``; the width ``N`` comes from ``x`` itself and the ramp shape from ``attrs``.
* it does not round to the output type.  Rounding is a *tolerance* fact of the harness, not
  something the oracle may bake in; it stays in fp64 and the comparison decides.

``import torch`` is required: the loader executes this file as a plain module and injects no names.
"""

import torch


def reference(inputs, attrs):
    """@brief fp64 oracle of the ramp-scaled elementwise product.

    @param inputs 合同输入（``x`` 已按声明 dtype 舍入并加宽成 float32）。
    @param attrs 已解析的整数 attr（``SEG``、``BASE``、``STEP``）。
    @return ``{"y": ...}``（fp64）。
    """
    # x arrives as float32 (the harness rounds each float tensor to its declared dtype and widens
    # back), so x.dtype says nothing about the declared output type.  Go to float64 immediately.
    x = inputs["x"].to(torch.float64)
    seg = int(attrs["SEG"])
    base = int(attrs["BASE"])
    step = int(attrs["STEP"])
    n = int(x.shape[0])
    ns = (n + seg - 1) // seg
    tab = base + step * torch.arange(ns, dtype=torch.float64, device=x.device)
    index = torch.arange(n, device=x.device) // seg
    return {"y": tab[index] * x}
