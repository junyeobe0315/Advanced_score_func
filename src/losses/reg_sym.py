from __future__ import annotations

from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def reg_sym_estimator(score_fn: TensorFn, x: torch.Tensor, sigma: torch.Tensor, K: int = 1) -> torch.Tensor:
    x_req = x.requires_grad_(True)
    total = 0.0

    for _ in range(int(K)):
        v = torch.randn_like(x_req)

        # VJP: J^T v = grad <s, v>
        s = score_fn(x_req, sigma)
        vjp = torch.autograd.grad((s * v).sum(), x_req, create_graph=True, retain_graph=True)[0]

        # JVP: J v
        _, jvp = torch.autograd.functional.jvp(
            lambda inp: score_fn(inp, sigma),
            (x_req,),
            (v,),
            create_graph=True,
            strict=False,
        )

        diff = jvp - vjp
        diff_norm2 = (_flatten_per_sample(diff) ** 2).sum(dim=1)
        total = total + diff_norm2.mean()

    return total / float(K)
