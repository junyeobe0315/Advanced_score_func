from __future__ import annotations

from typing import Callable

import torch


ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    """Flatten non-batch dimensions for per-sample norm evaluation."""
    return x.reshape(x.shape[0], -1)


def _jvp_forward_mode(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Compute Jacobian-vector product ``Jv`` via autograd JVP."""
    _, jvp = torch.autograd.functional.jvp(
        lambda inp: score_fn(inp, sigma),
        (x,),
        (v,),
        create_graph=True,
        strict=False,
    )
    return jvp


def _jvp_finite_difference(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    v: torch.Tensor,
    eps_fd: float,
) -> torch.Tensor:
    """Approximate Jacobian-vector product ``Jv`` using finite difference.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Input batch with gradient enabled.
        sigma: Per-sample sigma tensor.
        v: Probe directions matching ``x`` shape.
        eps_fd: Finite-difference epsilon.

    Returns:
        Approximate ``Jv`` tensor.
    """
    s_plus = score_fn(x + eps_fd * v, sigma)
    s_base = score_fn(x, sigma)
    return (s_plus - s_base) / eps_fd


def jacobian_asymmetry_estimator(
    score_fn: ScoreFn,
    x: torch.Tensor,
    sigma: torch.Tensor,
    num_probes: int = 1,
    method: str = "jvp_vjp",
    eps_fd: float = 1.0e-3,
) -> torch.Tensor:
    """Estimate ``||J - J^T||_F^2`` with Hutchinson probing.

    Args:
        score_fn: Score function ``s(x, sigma)``.
        x: Batched noisy inputs.
        sigma: Per-sample sigma values ``[B]``.
        num_probes: Number of Gaussian probe vectors.
        method: ``"jvp_vjp"``, ``"finite_diff"``, or ``"auto"``.
        eps_fd: Finite-difference epsilon for fallback mode.

    Returns:
        Scalar tensor estimating Jacobian asymmetry energy.

    How it works:
        Let ``A = J - J^T``. Hutchinson identity gives
        ``||A||_F^2 = E_v ||Av||^2``. For each probe ``v`` we compute:
        - ``J^T v`` via VJP: ``grad_x <s(x), v>``
        - ``J v`` via JVP or finite difference fallback
        and average ``||Jv - J^Tv||^2`` over probes.
    """
    probes = max(int(num_probes), 1)
    x_req = x.requires_grad_(True)

    total = torch.zeros((), device=x_req.device, dtype=x_req.dtype)
    for _ in range(probes):
        # Gaussian probe used by Hutchinson estimator.
        v = torch.randn_like(x_req)

        s = score_fn(x_req, sigma)
        jtv = torch.autograd.grad((s * v).sum(), x_req, create_graph=True, retain_graph=True)[0]

        method_now = method
        if method_now not in {"jvp_vjp", "finite_diff", "auto"}:
            raise ValueError(f"unknown jacobian estimator method: {method_now}")

        if method_now == "finite_diff":
            jv = _jvp_finite_difference(score_fn, x_req, sigma, v, eps_fd=eps_fd)
        else:
            try:
                jv = _jvp_forward_mode(score_fn, x_req, sigma, v)
            except RuntimeError:
                if method_now == "jvp_vjp":
                    raise
                jv = _jvp_finite_difference(score_fn, x_req, sigma, v, eps_fd=eps_fd)

        diff = jv - jtv
        total = total + (_flatten_per_sample(diff) ** 2).sum(dim=1).mean()

    return total / float(probes)


def low_noise_gate(sigma: torch.Tensor, sigma0: float) -> torch.Tensor:
    """Return a binary low-noise gate mask ``1[sigma <= sigma0]``.

    Args:
        sigma: Per-sample sigma tensor.
        sigma0: Low-noise threshold.

    Returns:
        Float mask tensor with same shape as ``sigma``.
    """
    return (sigma <= float(sigma0)).to(dtype=sigma.dtype)
