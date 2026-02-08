from __future__ import annotations

import torch
import torch.nn as nn


class PotentialNet(nn.Module):
    """Abstract interface for scalar potential networks.

    Notes:
        A potential network predicts scalar energy ``phi(x, sigma)``. Score is
        obtained via gradient w.r.t input: ``s(x,sigma) = grad_x phi(x,sigma)``.
    """

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Return scalar potential values.

        Args:
            x: Input tensor ``[B,...]``.
            sigma: Per-sample noise tensor ``[B]``.

        Returns:
            Potential tensor ``[B,1]``.
        """
        raise NotImplementedError


def score_from_potential(
    model: nn.Module,
    x: torch.Tensor,
    sigma: torch.Tensor,
    *,
    create_graph: bool,
) -> torch.Tensor:
    """Differentiate potential model output to obtain score field.

    Args:
        model: Potential network module.
        x: Input tensor.
        sigma: Per-sample sigma tensor.
        create_graph: Keep graph for higher-order derivatives when True.

    Returns:
        Score tensor with same shape as ``x``.

    How it works:
        Enables gradient on input, sums scalar potentials, and calls
        ``torch.autograd.grad`` to compute spatial derivative.
    """
    x_req = x.requires_grad_(True)
    phi = model(x_req, sigma)
    if phi.ndim == 1:
        phi = phi[:, None]
    grad = torch.autograd.grad(phi.sum(), x_req, create_graph=create_graph)[0]
    return grad
