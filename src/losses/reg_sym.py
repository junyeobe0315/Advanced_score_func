from __future__ import annotations

from typing import Callable

import torch


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Flatten every non-batch dimension for per-sample norm computation.

    Args:
        x: Tensor with batch dimension first.

    Returns:
        Tensor of shape ``[B, D_flat]``.

    How it works:
        Uses ``reshape`` to preserve batch size while collapsing all remaining
        dimensions into a single feature axis.
    """
    return x.reshape(x.shape[0], -1)


def reg_sym_estimator(score_fn: TensorFn, x: torch.Tensor, sigma: torch.Tensor, K: int = 1) -> torch.Tensor:
    """Estimate Jacobian asymmetry energy ``||J - J^T||_F^2`` via probing.

    Args:
        score_fn: Callable returning score tensor from ``(x, sigma)``.
        x: Noisy input batch.
        sigma: Per-sample noise levels with shape ``[B]``.
        K: Number of Gaussian probe vectors.

    Returns:
        Scalar estimate of asymmetry regularizer.

    How it works:
        For each random probe ``v``:
        1) Compute ``J^T v`` by VJP with ``autograd.grad``.
        2) Compute ``J v`` by JVP with ``torch.autograd.functional.jvp``.
        3) Use ``||Jv - J^Tv||^2`` as a probe estimate.
        4) Average over samples and probes.
    """
    # Keep x differentiable because both VJP and JVP need Jacobian access.
    x_req = x.requires_grad_(True)
    total = 0.0

    for _ in range(int(K)):
        # Gaussian probe vector for Hutchinson-style Frobenius estimation.
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

        # Antisymmetric component action on probe vector.
        diff = jvp - vjp
        diff_norm2 = (_flatten_per_sample(diff) ** 2).sum(dim=1)
        total = total + diff_norm2.mean()

    return total / float(K)
