from __future__ import annotations

import torch

from src.models import score_fn_from_model
from src.models.potential_mlp_toy import PotentialMLPToy


def test_struct_score_has_symmetric_jacobian_in_toy() -> None:
    """Score from potential model should have symmetric Jacobian in 2D."""
    model = PotentialMLPToy(dim=2, hidden_dim=32, depth=2, sigma_embed_dim=16)
    score_fn = score_fn_from_model(model, variant="struct", create_graph=True)

    x = torch.randn(8, 2)
    sigma = torch.exp(torch.empty(8).uniform_(-1.0, 1.0))

    asym_vals = []
    for i in range(x.shape[0]):
        xi = x[i : i + 1].requires_grad_(True)
        si = sigma[i : i + 1]
        yi = score_fn(xi, si)

        grads = []
        for d in range(2):
            g = torch.autograd.grad(yi[0, d], xi, retain_graph=True, create_graph=False)[0][0]
            grads.append(g)
        jac = torch.stack(grads, dim=0)
        asym = jac - jac.t()
        asym_vals.append((asym ** 2).sum())

    mean_asym = torch.stack(asym_vals).mean().item()
    assert mean_asym < 1e-4
