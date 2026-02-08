from __future__ import annotations

import torch

from src.losses.reg_sym import reg_sym_estimator


def test_reg_sym_finite_diff_matches_jvp_on_linear_field() -> None:
    """Finite-difference fallback should match JVP/VJP on simple linear fields."""
    A = torch.tensor([[2.0, -1.0], [0.5, 3.0]])

    def score_fn(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Linear field with constant Jacobian matrix A."""
        del sigma
        return x @ A.t()

    x = torch.randn(512, 2, requires_grad=True)
    sigma = torch.ones(512)

    torch.manual_seed(0)
    val_jvp = reg_sym_estimator(score_fn, x, sigma, K=4, method="jvp_vjp")
    torch.manual_seed(0)
    val_fd = reg_sym_estimator(score_fn, x, sigma, K=4, method="finite_diff", eps_fd=1.0e-4)

    assert abs(val_jvp.item() - val_fd.item()) < 1.0e-2
